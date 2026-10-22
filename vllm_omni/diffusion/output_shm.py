# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared-memory storage for multiview video output, written view by view.

The output rank allocates one named shared-memory segment that holds a
per-view readiness word followed by the BF16 video, each camera view stored
contiguously. Decode writes every view into the segment; after the D2H copy
and the [-1, 1] clamp of a view, its readiness word is published under a
process-shared file lock. The wire handle carries the shape, dtype, frames per
view and the segment name, never tensor storage. A reader borrows the mapping
and waits only for the view it is about to consume. The values are not
converted or reduced.
"""

from __future__ import annotations

import contextlib
import contextvars
import fcntl
import math
import threading
import time
from collections.abc import Iterator, Sequence
from multiprocessing import shared_memory
from typing import Any, TypeGuard

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.shm_utils import require_shared_memory

logger = init_logger(__name__)

_OUTPUT_OWNER = contextvars.ContextVar("diffusion_output_owner", default=False)
_FAILED_TRANSFERS: list[Any] = []
_WIRE_TYPE = "cosmos3_view_shm_v1"
# One int64 readiness word per view, padded to a page so the video data starts
# page aligned.
_STATE_BYTES = 8
_HEADER_ALIGNMENT = 4096
# Bound on how long a reader waits for the producer to publish one view.
_VIEW_WAIT_SECONDS = 300
_VIEW_POLL_SECONDS = 0.001


def _header_bytes(views: int) -> int:
    return (views * _STATE_BYTES + _HEADER_ALIGNMENT - 1) // _HEADER_ALIGNMENT * _HEADER_ALIGNMENT


@contextlib.contextmanager
def output_owner(enabled: bool) -> Iterator[None]:
    """Use the executor's actual reply decision, including DP and CFG=1."""
    token = _OUTPUT_OWNER.set(enabled)
    try:
        yield
    finally:
        _OUTPUT_OWNER.reset(token)


def is_output_owner() -> bool:
    return _OUTPUT_OWNER.get()


class _Mapping:
    def __init__(self, shm: shared_memory.SharedMemory, array: np.ndarray) -> None:
        self.shm = shm
        self.__array_interface__ = array.__array_interface__

    def __del__(self) -> None:
        with contextlib.suppress(BufferError):
            self.shm.close()


class SharedVideo:
    """BF16 ``[B, C, V*F, H, W]`` video in shared memory with per-view readiness.

    Producer: :meth:`allocate`, then :meth:`copy_view` or a write into
    :meth:`view` followed by :meth:`record_view` for every view, then
    :meth:`finalize`. Consumer: :meth:`from_handle`, then :meth:`wait_view`.
    """

    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        shape: Sequence[int],
        dtype: torch.dtype,
        frames: Sequence[int],
        *,
        owner: bool = False,
    ) -> None:
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.frames = tuple(frames)
        self.shm = shm
        self.owner = owner
        self.exported = False
        self._registered = False
        self._stream: Any = None
        self._events: dict[int, Any] = {}
        self._thread: threading.Thread | None = None
        self._offset = _header_bytes(len(frames))
        array = np.ndarray((math.prod(shape),), dtype=np.uint16, buffer=shm.buf, offset=self._offset)
        # The tensor's base owns the mapping even after this wrapper dies.
        self.tensor = torch.from_numpy(np.asarray(_Mapping(shm, array))).view(dtype)
        self._state = np.ndarray((len(frames),), dtype=np.int64, buffer=shm.buf)

    @classmethod
    def allocate(
        cls, shape: Sequence[int], dtype: torch.dtype, frames: Sequence[int], *, cuda: bool = False
    ) -> SharedVideo:
        """Create the segment; with ``cuda``, register it for asynchronous D2H copies.

        Registration uses cudaHostRegister, so it is CUDA only.
        """
        if dtype != torch.bfloat16:
            raise ValueError("Shared multiview video storage requires BF16.")
        if (
            len(shape) != 5
            or any(type(d) is not int or d <= 0 for d in shape)
            or not frames
            or any(type(f) is not int or f <= 0 for f in frames)
            or sum(frames) != shape[2]
        ):
            raise ValueError("Invalid multiview SHM dimensions")
        if _FAILED_TRANSFERS:
            raise RuntimeError("A CUDA SHM transfer failed cleanup; restart the worker")
        required = _header_bytes(len(frames)) + math.prod(shape) * 2
        require_shared_memory(required)
        shm = shared_memory.SharedMemory(create=True, size=required)
        obj = None
        try:
            obj = cls(shm, shape, dtype, frames, owner=True)
            obj._state.fill(0)
            if cuda:
                if torch.version.cuda is None:
                    raise ValueError("Direct CUDA SHM is unavailable on this platform")
                rt = torch.cuda.cudart()
                error = rt.cudaHostRegister(obj.tensor.data_ptr(), obj.tensor.numel() * 2, 0)
                if int(error) != 0:
                    raise RuntimeError(f"cudaHostRegister failed: {rt.cudaGetErrorString(error)}")
                obj._registered = True
                obj._stream = torch.cuda.Stream()
            return obj
        except BaseException:
            if obj is not None:
                obj.unlink()
            else:
                shm.close()
                with contextlib.suppress(FileNotFoundError):
                    shm.unlink()
            raise

    @classmethod
    def from_handle(cls, handle: dict[str, Any]) -> SharedVideo:
        if handle.get("__type__") != _WIRE_TYPE or handle.get("dtype") != "torch.bfloat16":
            raise ValueError("Invalid multiview SHM handle")
        shape, frames = handle["shape"], handle["frames"]
        if len(shape) != 5 or any(type(x) is not int or x <= 0 for x in shape + frames):
            raise ValueError("Invalid multiview SHM shape")
        if not frames or sum(frames) != shape[2]:
            raise ValueError("Invalid multiview SHM frame counts")
        shm = shared_memory.SharedMemory(name=handle["name"])
        try:
            if _header_bytes(len(frames)) + math.prod(shape) * 2 != shm.size:
                raise ValueError("Invalid multiview SHM size")
            obj = cls(shm, shape, torch.bfloat16, frames)
            shm.unlink()
            return obj
        except BaseException:
            shm.close()
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()
            raise

    def handle(self) -> dict[str, Any]:
        self.exported = True
        return {
            "__type__": _WIRE_TYPE,
            "name": self.shm.name,
            "shape": list(self.shape),
            "frames": list(self.frames),
            "dtype": str(self.dtype),
        }

    def _status(self, index: int, value: int | None = None) -> int:
        # flock also orders publication across independent processes. No CUDA
        # event is exported to another process or queried before synchronization.
        # SharedMemory exposes its file descriptor only as a private attribute.
        fd = self.shm._fd  # type: ignore[attr-defined]
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            if value is not None:
                self._state[index] = value
            return int(self._state[index])
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)

    def view(self, index: int) -> torch.Tensor:
        """Storage of one view, ``[B, C, F, H, W]``, without waiting for readiness.

        Each view is physically contiguous so CUDA can copy straight into it; a
        narrow of ``[B, C, V*F, H, W]`` would need a hidden CPU staging buffer.
        """
        b, c, _, h, w = self.shape
        frame_elements = b * c * h * w
        start = sum(self.frames[:index]) * frame_elements
        return self.tensor.narrow(0, start, self.frames[index] * frame_elements).reshape(b, c, self.frames[index], h, w)

    def copy_view(self, index: int, source: torch.Tensor) -> None:
        """Copy one decoded view in, asynchronously from CUDA when registered."""
        if not 0 <= index < len(self.frames) or index in self._events:
            raise ValueError("View index is invalid or already copied")
        target = self.view(index)
        if tuple(source.shape) != tuple(target.shape) or source.dtype != self.dtype:
            raise ValueError("Decoded view disagrees with SHM metadata")
        if self._stream is None:
            target.copy_(source)
            self._events[index] = None
        else:
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                target.copy_(source, non_blocking=True)
                source.record_stream(self._stream)
                event = torch.cuda.Event()
                event.record(self._stream)
            self._events[index] = event

    def record_view(self, index: int, stream: Any) -> None:
        """Record a write into :meth:`view` made on another CUDA stream.

        The SHM stream joins ``stream`` before publishing readiness or
        unregistering host memory. Views may be recorded in any order.
        """
        if not 0 <= index < len(self.frames) or index in self._events:
            raise ValueError("View index is invalid or already copied")
        if self._stream is None:
            raise RuntimeError("External CUDA writes require registered SHM")
        self._stream.wait_stream(stream)
        event = torch.cuda.Event()
        event.record(self._stream)
        self._events[index] = event

    def finalize(self, low: float, high: float) -> None:
        """Clamp every view to ``[low, high]`` and publish it once its copy completes.

        With registered storage this runs on a background thread, so the caller
        does not wait for the D2H copies.
        """
        if len(self._events) != len(self.frames) or self._thread is not None:
            raise RuntimeError("Incomplete or already finalized SHM output")

        @torch.inference_mode()
        def finish() -> None:
            try:
                for index in range(len(self.frames)):
                    event = self._events[index]
                    if event is not None:
                        event.synchronize()
                    self.view(index).clamp_(low, high)
                    self._status(index, 1)
            except BaseException:
                logger.exception("Publishing shared multiview video failed; readers will see an error.")
                for index in range(len(self.frames)):
                    self._status(index, -1)
            finally:
                self._unregister()

        if self._stream is None:
            finish()
        else:
            self._thread = threading.Thread(target=finish, name="MultiviewSHMCompletion", daemon=True)
            self._thread.start()

    def _unregister(self) -> None:
        if self._registered:
            assert self._stream is not None
            try:
                self._stream.synchronize()
                with torch.accelerator.device_index(self._stream.device.index):
                    rt = torch.cuda.cudart()
                    error = rt.cudaHostUnregister(self.tensor.data_ptr())
                if int(error) != 0:
                    raise RuntimeError("cudaHostUnregister failed")
                self._registered = False
            except BaseException:
                # Never unmap storage whose device use is uncertain.
                logger.exception("Unregistering shared multiview video failed; later outputs use CPU tensors.")
                _FAILED_TRANSFERS.append(self)
                for index in range(len(self.frames)):
                    self._status(index, -1)

    def wait_view(self, index: int) -> torch.Tensor:
        """Wait until view ``index`` is published and return it as ``[B, C, F, H, W]``."""
        if not 0 <= index < len(self.frames):
            raise ValueError("View does not match SHM metadata")
        deadline = time.monotonic() + _VIEW_WAIT_SECONDS
        while True:
            state = self._status(index)
            if state < 0:
                raise RuntimeError("Multiview SHM producer failed")
            if state == 1:
                return self.view(index)
            if time.monotonic() >= deadline:
                raise TimeoutError("Multiview SHM producer did not complete the view")
            time.sleep(_VIEW_POLL_SECONDS)

    def to_tensor(self) -> torch.Tensor:
        """Wait for every view and return a ``[B, C, V*F, H, W]`` tensor copy."""
        return torch.cat([self.wait_view(index) for index in range(len(self.frames))], dim=2)

    def unlink(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.shm.unlink()

    def __del__(self) -> None:
        if self._registered and self._thread is None:
            self._unregister()
        if self.owner and not self.exported:
            self.unlink()
        # tensor's _Mapping closes only once all borrowed tensors are gone.


def is_shared_video_handle(value: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict) and value.get("__type__") == _WIRE_TYPE


def direct_video_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether this rank writes its video output straight into shared memory.

    Only the rank that replies with the output does. Registration and the
    asynchronous copies use cudaHostRegister and CUDA streams, so the device
    must be CUDA, and the storage holds BF16 bits only.
    """
    return is_output_owner() and device.type == "cuda" and torch.version.cuda is not None and dtype == torch.bfloat16


def allocate_video_output(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    frames: Sequence[int],
    *,
    shared: bool,
    cuda: bool,
    pin_memory: bool = False,
) -> torch.Tensor | SharedVideo:
    """Host storage for a camera-major video: shared memory if ``shared``, else a CPU tensor.

    If the shared segment cannot be created or registered, a CPU tensor is
    returned instead. Insufficient /dev/shm capacity is raised, not hidden.
    """
    if shared:
        try:
            return SharedVideo.allocate(shape, dtype, frames, cuda=cuda)
        except (RuntimeError, OSError) as exc:
            # Log each distinct failure once; exception objects never compare equal.
            logger.warning_once("Shared multiview video storage is unavailable, using a CPU tensor: %s", str(exc))
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
