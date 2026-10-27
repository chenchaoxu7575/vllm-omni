# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""How the Cosmos3 multiview pipeline hands its video to its postprocessor.

The decoded video travels from the worker to the engine through shared-memory
IPC, which widens BF16 to FP32 because NumPy has no BF16 type: twice the
shared memory, and a conversion pass on each side. The pipeline hands over a
uint16 view of the BF16 storage instead, which IPC moves unchanged, and the
postprocessor views it as BF16 again.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

# Frames converted at a time, bounding the temporaries to a few frames.
_CHUNK_FRAMES = 16


def video_for_transport(video: torch.Tensor) -> tuple[torch.Tensor, dict[str, str]]:
    """Return the tensor to put in the output payload and the internal metadata that restores it.

    BF16 becomes a uint16 view of the same storage, the raw-bits form the
    upstream registered-SHM transport also uses. Other dtypes are unchanged.
    """
    if video.dtype != torch.bfloat16:
        return video, {}
    return video.view(torch.uint16), {"video_dtype": "bfloat16"}


def video_from_transport(video: torch.Tensor, internal: Mapping[str, Any]) -> torch.Tensor:
    """Undo ``video_for_transport``; a video without its marker is returned as is."""
    if internal.get("video_dtype") != "bfloat16":
        return video
    return video.view(torch.bfloat16)


def write_video_numpy(video: torch.Tensor, out: np.ndarray) -> None:
    """Write ``VideoProcessor.postprocess_video(video, output_type="np")`` into ``out``.

    ``video`` is [1, 3, F, H, W] in [-1, 1]; ``out`` is float32 [1, F, H, W, 3].
    As in ``VideoProcessor``, the multiply-add and the clamp round in the dtype
    of ``video`` before the float32 conversion, so the result is bit-identical
    without converting the whole clip at once.
    """
    target = torch.from_numpy(out)
    for start in range(0, video.shape[2], _CHUNK_FRAMES):
        chunk = video[:, :, start : start + _CHUNK_FRAMES] * 0.5
        chunk.add_(0.5).clamp_(0, 1)
        target[:, start : start + chunk.shape[2]].copy_(chunk.permute(0, 2, 3, 4, 1))
