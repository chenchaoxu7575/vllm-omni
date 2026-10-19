# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded owner-local CPU preparation; GPU work stays on the calling thread."""

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path

import torch

from vllm_omni.diffusion.utils.video_decode import _frame_to_uint8_rgb

from .transfer import IMAGE_EXTENSIONS, media_to_uint8_cthw, resize_center_crop_uint8_cthw


def prepare_workers(requested=0, local_ranks=None):
    """Budget nested ATen pools, decoder, producer and the GPU launch thread.

    WORLD_SIZE is a conservative fallback when a launcher does not provide
    LOCAL_WORLD_SIZE (e.g. multi-node deployments). Unknown topology disables
    the pipeline. Explicit worker requests are capped by the same budget.
    """
    if local_ranks is None:
        local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "0")))
        if not local_ranks and torch.distributed.is_initialized():
            local_ranks = torch.distributed.get_world_size()
    if local_ranks < 1:
        return 0
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    budget = cpus // local_ranks
    intra = torch.get_num_threads()
    # FFmpeg's single-threaded codec still has demux/filter/encode/mux and
    # pipe-management threads (six in imageio-ffmpeg 7). Reserve eight for
    # that backend and the producer, separately from the consumer's ATen
    # pool. This is backend overhead, not a fixed worker/rank/CPU count.
    # Each block worker may also use a full ATen intra-op pool.
    backend_overhead = 8
    limit = max(0, (budget - intra - backend_overhead) // intra)
    return min(requested, limit) if requested else limit


class ViewPrefetch(AbstractContextManager):
    """One future view at most, in original owner order; join on every exit."""

    def __init__(self, keys, prepare):
        self.keys = iter(keys)
        self.prepare = prepare
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vae-prefetch")
        self.future = None

    def _submit(self):
        key = next(self.keys, None)
        self.future = None if key is None else (key, self.pool.submit(self.prepare, key))

    def __enter__(self):
        self._submit()
        return self

    def take(self, key):
        expected, future = self.future
        if key != expected:
            raise ValueError("VAE prefetch consumption differs from owner order.")
        result = future.result()
        self._submit()
        return result

    def __exit__(self, *exc):
        if self.future is not None:
            self.future[1].cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        return False


class FramePreparation(AbstractContextManager):
    """Decode in display order while bounded frame blocks resize in a pool.

    Inter-frame codec dependencies stay in the existing imageio/FFmpeg
    decoder. No seek, frame resampling, codec substitution or GPU decoding.
    """

    def __init__(self, workers, block_bytes=32 << 20):
        self.workers = workers
        self.block_bytes = block_bytes
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vae-frames")

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=True)
        return False

    def pixels(self, view, *, field, height, width, num_frames, keep_first, require_complete, dtype):
        output = torch.empty((3, num_frames, height, width), dtype=dtype, device="cpu")
        value = view.get(f"{field}_path", view.get(field))
        if value is None:
            if field != "vision":
                raise ValueError(f"Cosmos3 multiview camera {view['camera_key']!r} is missing {field} input.")
            output.copy_(torch.tensor(128, dtype=dtype, device="cpu").div_(127.5).sub_(1.0))
            return output.unsqueeze(0)
        limit = 1 if keep_first else num_frames
        if isinstance(value, (str, Path)) and Path(value).suffix.lower() not in IMAGE_EXTENSIONS:
            count = self._video(value, output, limit, height, width)
        else:
            # Preserve the original payload conversion for image/tensor/list.
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

    def _video(self, path, output, limit, height, width):
        import imageio.v3 as iio

        pending = deque()
        block = []
        count = 0
        batch = None
        # Same legacy FFmpeg plugin selected for these files by the original
        # imageio path; cap both codec and RGB conversion threads. Other
        # plugins are handled by the unchanged media conversion path below.
        with iio.imopen(path, "r") as reader:
            if reader.__class__.__name__ != "LegacyPlugin" or reader._format.name != "FFMPEG":
                frames = media_to_uint8_cthw(path, height=height, width=width, max_frames=limit)
                count = frames.shape[1]
                output[:, :count].copy_(frames).div_(127.5).sub_(1.0)
                return count
            iterator = reader.iter(input_params=["-threads", "1"], output_params=["-threads", "1"])
            for frame in iterator:
                array = _frame_to_uint8_rgb(frame)
                if batch is None:
                    h, w = array.shape[:2]
                    scale = max(width / w, height / h)
                    import math

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
    def _block(arrays, output, start, height, width):
        frames = torch.stack([torch.from_numpy(a).permute(2, 0, 1) for a in arrays], dim=1).contiguous()
        resized = resize_center_crop_uint8_cthw(frames, height, width)
        # Preserve uint8 -> BF16, division rounding, then subtraction rounding.
        output[:, start : start + len(arrays)].copy_(resized).div_(127.5).sub_(1.0)
