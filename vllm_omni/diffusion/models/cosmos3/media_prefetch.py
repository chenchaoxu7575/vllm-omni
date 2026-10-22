# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded CPU preparation of camera media on VAE owners.

Worker threads decode and resize camera media ahead of the VAE; all GPU work
stays on the calling thread.
"""

from __future__ import annotations

import math
import os
import socket
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v3 as iio
import numpy as np
import torch

from vllm_omni.diffusion.utils.video_decode import frame_to_uint8_rgb

from .transfer import IMAGE_EXTENSIONS, media_to_uint8_cthw, resize_center_crop_uint8_cthw

if TYPE_CHECKING:
    from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator

# Threads reserved per rank for the FFmpeg backend and the producer, apart from
# the consumer's ATen pool. FFmpeg with one codec thread still runs demux,
# filter, encode, mux and pipe-management threads (six in imageio-ffmpeg 7).
_FFMPEG_BACKEND_THREADS = 8
# Upper bound on the decoded plus resized bytes of one resize block.
_RESIZE_BLOCK_BYTES = 32 << 20


def prepare_workers(
    local_ranks: int | None = None, affinity_peers: Sequence[tuple[str, Iterable[int]]] | None = None
) -> int:
    """Return the resize worker count that fits this rank's CPU budget.

    The budget covers nested ATen pools, the decoder, the producer and the GPU
    launch thread. ``local_ranks`` defaults to ``LOCAL_WORLD_SIZE``, then the
    more conservative ``WORLD_SIZE`` (for example on multi-node launchers).
    Unknown topology or missing CPU affinity information disables workers.
    """
    if not hasattr(os, "sched_getaffinity"):
        return 0
    if local_ranks is None:
        try:
            ranks = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "0")))
        except ValueError:
            return 0
        if not ranks and torch.distributed.is_initialized():
            ranks = torch.distributed.get_world_size()
    else:
        ranks = local_ranks
    if ranks < 1:
        return 0
    try:
        cpus = len(os.sched_getaffinity(0))
    except OSError:
        return 0
    budget = cpus // ranks
    if affinity_peers is not None and hasattr(os, "sched_getaffinity"):
        local = [set(mask) for host, mask in affinity_peers if host == socket.gethostname()]
        if local and len(local) <= ranks:
            # Count only ranks that can run on each CPU. Unobserved local
            # DP/TP ranks conservatively share every CPU; no WORLD collective.
            shared = Counter(cpu for mask in local for cpu in mask)
            unseen = ranks - len(local)
            own = os.sched_getaffinity(0)
            if all(shared[cpu] + unseen > 0 for cpu in own):
                budget = int(sum(Fraction(1, shared[cpu] + unseen) for cpu in own))
    intra = torch.get_num_threads()
    # Each resize worker may also use a full ATen intra-op pool.
    return max(0, (budget - intra - _FFMPEG_BACKEND_THREADS) // intra)


def prepare_workers_for_groups(sp: GroupCoordinator, cfg: GroupCoordinator) -> int:
    """Budget workers across overlapping CPU affinity masks of the SP x CFG grid."""
    dist = torch.distributed
    if not dist.is_initialized() or not hasattr(os, "sched_getaffinity"):
        return 0
    try:
        mask = tuple(sorted(os.sched_getaffinity(0)))
    except OSError:
        mask = ()
    own = (socket.gethostname(), mask)
    reports: list[Any] = [None] * sp.world_size
    dist.all_gather_object(reports, own, group=sp.cpu_group)
    groups: list[Any] = [None] * cfg.world_size
    dist.all_gather_object(groups, reports, group=cfg.cpu_group)
    peers = [item for group in groups for item in group]
    if any(not mask for _, mask in peers):
        return 0
    return prepare_workers(affinity_peers=peers)


class ViewPrefetch(AbstractContextManager):
    """Prepare at most one future item, in consumption order; join on every exit."""

    def __init__(self, keys: Iterable[str], prepare: Callable[[str], Any]) -> None:
        self.keys = iter(keys)
        self.prepare = prepare
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vae-prefetch")
        self.future: tuple[str, Future] | None = None

    def _submit(self) -> None:
        key = next(self.keys, None)
        self.future = None if key is None else (key, self.pool.submit(self.prepare, key))

    def __enter__(self) -> ViewPrefetch:
        self._submit()
        return self

    def take(self, key: str) -> Any:
        if self.future is None or key != self.future[0]:
            raise ValueError("VAE prefetch consumption differs from owner order.")
        result = self.future[1].result()
        self._submit()
        return result

    def __exit__(self, *exc: object) -> None:
        if self.future is not None:
            self.future[1].cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)


def _is_legacy_ffmpeg_reader(reader: Any) -> bool:
    """Whether imageio opened the file with its legacy FFmpeg plugin.

    imageio exposes the legacy plugin's format only through the private
    ``_format`` attribute. Other readers return False and use the generic
    media conversion instead.
    """
    return reader.__class__.__name__ == "LegacyPlugin" and reader._format.name == "FFMPEG"


class FramePreparation(AbstractContextManager):
    """Decode frames in display order while a pool resizes bounded frame blocks.

    Inter-frame codec dependencies stay in the imageio/FFmpeg decoder. There is
    no seek, frame resampling, codec substitution or GPU decoding.
    """

    def __init__(self, workers: int, block_bytes: int = _RESIZE_BLOCK_BYTES) -> None:
        self.workers = workers
        self.block_bytes = block_bytes
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vae-frames")

    def __exit__(self, *exc: object) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

    def pixels(
        self,
        view: Mapping[str, Any],
        *,
        field: str,
        height: int,
        width: int,
        num_frames: int,
        keep_first: bool,
        require_complete: bool,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one camera's normalized ``[1, 3, num_frames, height, width]`` pixels."""
        output = torch.empty((3, num_frames, height, width), dtype=dtype, device="cpu")
        value = view.get(f"{field}_path", view.get(field))
        if value is None:
            if field != "vision":
                raise ValueError(f"Cosmos3 multiview camera {view['camera_key']!r} is missing {field} input.")
            output.copy_(torch.tensor(128, dtype=dtype, device="cpu").div_(127.5).sub_(1.0))
            return output.unsqueeze(0)
        limit = 1 if keep_first else num_frames
        if isinstance(value, str | Path) and Path(value).suffix.lower() not in IMAGE_EXTENSIONS:
            count = self._video(value, output, limit, height, width)
        else:
            # Images, tensors and frame lists use the generic media conversion.
            frames = media_to_uint8_cthw(value, height=height, width=width, max_frames=limit)
            count = min(frames.shape[1], num_frames)
            output[:, :count].copy_(frames[:, :count]).div_(127.5).sub_(1.0)
        if require_complete and count < num_frames:
            raise ValueError(
                f"Known camera {view['camera_key']!r} requires a complete RGB video of {num_frames} frames."
            )
        if count < 1:
            raise ValueError("Cosmos3 multiview view media decoded to zero frames.")
        if count < num_frames:
            output[:, count:].copy_(output[:, count - 1 : count])
        return output.unsqueeze(0)

    def _video(self, path: str | Path, output: torch.Tensor, limit: int, height: int, width: int) -> int:
        pending: deque[Future] = deque()
        block: list[np.ndarray] = []
        count = 0
        batch = None
        with iio.imopen(path, "r") as reader:
            if not _is_legacy_ffmpeg_reader(reader):
                frames = media_to_uint8_cthw(path, height=height, width=width, max_frames=limit)
                count = frames.shape[1]
                output[:, :count].copy_(frames).div_(127.5).sub_(1.0)
                return count
            # Cap both codec and RGB conversion threads; resizing runs in the pool.
            iterator = reader.iter(input_params=["-threads", "1"], output_params=["-threads", "1"])
            for frame in iterator:
                array = frame_to_uint8_rgb(frame)
                if batch is None:
                    h, w = array.shape[:2]
                    scale = max(width / w, height / h)
                    rh, rw = math.ceil(h * scale), math.ceil(w * scale)
                    batch = max(1, self.block_bytes // (3 * 4 * (h * w + rh * rw)))
                block.append(array)
                count += 1
                if len(block) == batch or count == limit:
                    start = count - len(block)
                    pending.append(self.pool.submit(self._block, block, output, start, height, width))
                    block = []
                    # Keep one bounded queued block so a single resize
                    # worker can overlap with decoding the following block.
                    if len(pending) > self.workers:
                        pending.popleft().result()
                if count == limit:
                    break
            if block:
                pending.append(self.pool.submit(self._block, block, output, count - len(block), height, width))
            for future in pending:
                future.result()
        return count

    @staticmethod
    def _block(arrays: list[np.ndarray], output: torch.Tensor, start: int, height: int, width: int) -> None:
        frames = torch.stack([torch.from_numpy(a).permute(2, 0, 1) for a in arrays], dim=1).contiguous()
        resized = resize_center_crop_uint8_cthw(frames, height, width)
        # Cast to the output dtype, then divide and subtract as separate rounded
        # steps, matching uint8_cthw_to_normalized_5d.
        output[:, start : start + len(arrays)].copy_(resized).div_(127.5).sub_(1.0)
