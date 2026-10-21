# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owner-allocated host storage for opt-in multiview output.

The wire contains shape/dtype/view lengths and a shared-memory name, never a
Tensor storage. A per-view completion word is published under a process-shared
file lock after D2H and the original CPU clamp. Readers borrow the mapping and
wait only for the view they are about to consume. No floating-point reduction
or conversion takes place in this transport.
"""

from __future__ import annotations

import contextlib
import contextvars
import fcntl
import math
import threading
import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import torch

_OUTPUT_OWNER = contextvars.ContextVar("diffusion_output_owner", default=False)
_FAILED_TRANSFERS: list[Any] = []
_WIRE_TYPE = "cosmos3_view_shm_v1"


@contextlib.contextmanager
def output_owner(enabled: bool):
    """Use the executor's actual reply decision, including DP and CFG=1."""
    token = _OUTPUT_OWNER.set(enabled)
    try:
        yield
    finally:
        _OUTPUT_OWNER.reset(token)


def is_output_owner() -> bool:
    return _OUTPUT_OWNER.get()


class _Mapping:
    def __init__(self, shm, array):
        self.shm = shm
        self.__array_interface__ = array.__array_interface__

    def __del__(self):
        with contextlib.suppress(BufferError):
            self.shm.close()


class SharedVideo:
    """A borrowed CPU tensor with per-view readiness and an owning mapping."""

    def __init__(self, shm, shape, dtype, frames, *, owner=False):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.frames = tuple(frames)
        self.shm = shm
        self.owner = owner
        self.exported = False
        self._registered = False
        self._stream = None
        self._events = {}
        self._thread = None
        self._offset = ((len(frames) * 8 + 4095) // 4096) * 4096
        array = np.ndarray((math.prod(shape),), dtype=np.uint16, buffer=shm.buf, offset=self._offset)
        # The tensor's base owns the mapping even after this wrapper dies.
        self.tensor = torch.from_numpy(np.asarray(_Mapping(shm, array))).view(dtype)
        self._state = np.ndarray((len(frames),), dtype=np.int64, buffer=shm.buf)

    @classmethod
    def allocate(cls, shape, dtype, frames, *, cuda=False):
        if dtype != torch.bfloat16:
            raise ValueError("Direct multiview SHM requires BF16; other dtypes use the original path")
        if len(shape) != 5 or not frames or any(f <= 0 for f in frames) or sum(frames) != shape[2]:
            raise ValueError("Invalid multiview SHM dimensions")
        if _FAILED_TRANSFERS:
            raise RuntimeError("A CUDA SHM transfer failed cleanup; restart the worker")
        offset = ((len(frames) * 8 + 4095) // 4096) * 4096
        shm = shared_memory.SharedMemory(create=True, size=offset + math.prod(shape) * 2)
        obj = cls(shm, shape, dtype, frames, owner=True)
        obj._state.fill(0)
        try:
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
            obj.unlink()
            raise

    @classmethod
    def from_handle(cls, handle):
        if handle.get("__type__") != _WIRE_TYPE or handle.get("dtype") != "torch.bfloat16":
            raise ValueError("Invalid multiview SHM handle")
        shape, frames = handle["shape"], handle["frames"]
        if len(shape) != 5 or any(type(x) is not int or x <= 0 for x in shape + frames):
            raise ValueError("Invalid multiview SHM shape")
        if not frames or sum(frames) != shape[2]:
            raise ValueError("Invalid multiview SHM frame counts")
        shm = shared_memory.SharedMemory(name=handle["name"])
        try:
            offset = ((len(frames) * 8 + 4095) // 4096) * 4096
            if offset + math.prod(shape) * 2 != shm.size:
                raise ValueError("Invalid multiview SHM size")
            obj = cls(shm, shape, torch.bfloat16, frames)
            shm.unlink()
            return obj
        except BaseException:
            shm.close()
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()
            raise

    def handle(self):
        self.exported = True
        return {
            "__type__": _WIRE_TYPE,
            "name": self.shm.name,
            "shape": list(self.shape),
            "frames": list(self.frames),
            "dtype": str(self.dtype),
        }

    def _status(self, index, value=None):
        # flock also orders publication across independent processes. No CUDA
        # event is exported to another process or queried before synchronization.
        fcntl.flock(self.shm._fd, fcntl.LOCK_EX)
        try:
            if value is not None:
                self._state[index] = value
            return int(self._state[index])
        finally:
            fcntl.flock(self.shm._fd, fcntl.LOCK_UN)

    def _view(self, index):
        # Each camera is physically contiguous so CUDA can copy straight into
        # SHM. A B,C,V*F,H,W narrow would require a hidden CPU staging buffer.
        b, c, _, h, w = self.shape
        frame_elements = b * c * h * w
        start = sum(self.frames[:index]) * frame_elements
        return self.tensor.narrow(0, start, self.frames[index] * frame_elements).reshape(b, c, self.frames[index], h, w)

    def copy_view(self, index, source):
        if not 0 <= index < len(self.frames) or index in self._events:
            raise ValueError("View index is invalid or already copied")
        target = self._view(index)
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

    def record_view(self, index, stream):
        """Record a direct write made on an existing transport stream.

        The SHM stream joins the producer before publishing readiness or
        unregistering host memory. Owner scheduling may visit views out of order.
        """
        if not 0 <= index < len(self.frames) or index in self._events:
            raise ValueError("View index is invalid or already copied")
        if self._stream is None:
            raise RuntimeError("External CUDA writes require registered SHM")
        self._stream.wait_stream(stream)
        event = torch.cuda.Event()
        event.record(self._stream)
        self._events[index] = event

    def clamp_(self, low, high):
        """Publish views after the pipeline's original CPU clamp, asynchronously."""
        if len(self._events) != len(self.frames) or self._thread is not None:
            raise RuntimeError("Incomplete or already finalized SHM output")

        @torch.inference_mode()
        def finish():
            try:
                for index in range(len(self.frames)):
                    event = self._events[index]
                    if event is not None:
                        event.synchronize()
                    self._view(index).clamp_(low, high)
                    self._status(index, 1)
            except BaseException:
                for index in range(len(self.frames)):
                    self._status(index, -1)
            finally:
                self._unregister()

        if self._stream is None:
            finish()
        else:
            self._thread = threading.Thread(target=finish, name="MultiviewSHMCompletion", daemon=True)
            self._thread.start()
        return self

    def _unregister(self):
        if self._registered:
            try:
                self._stream.synchronize()
                with torch.cuda.device(self._stream.device):
                    rt = torch.cuda.cudart()
                    error = rt.cudaHostUnregister(self.tensor.data_ptr())
                if int(error) != 0:
                    raise RuntimeError("cudaHostUnregister failed")
                self._registered = False
            except BaseException:
                # Never unmap storage whose device use is uncertain.
                _FAILED_TRANSFERS.append(self)
                for index in range(len(self.frames)):
                    self._status(index, -1)

    def narrow(self, dim, start, count):
        if dim != 2:
            raise ValueError("SharedVideo is consumed by complete views on dimension 2")
        offset = 0
        for index, frames in enumerate(self.frames):
            if (start, count) == (offset, frames):
                deadline = time.monotonic() + 300
                while True:
                    state = self._status(index)
                    if state < 0:
                        raise RuntimeError("Multiview SHM producer failed")
                    if state == 1:
                        return self._view(index)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Multiview SHM producer did not complete the view")
                    time.sleep(0.001)
            offset += frames
        raise ValueError("View does not match SHM metadata")

    def unlink(self):
        with contextlib.suppress(FileNotFoundError):
            self.shm.unlink()

    def __del__(self):
        if self._registered and self._thread is None:
            self._unregister()
        if self.owner and not self.exported:
            self.unlink()
        # tensor's _Mapping closes only once all borrowed tensors are gone.


def is_shared_video_handle(value):
    return isinstance(value, dict) and value.get("__type__") == _WIRE_TYPE
