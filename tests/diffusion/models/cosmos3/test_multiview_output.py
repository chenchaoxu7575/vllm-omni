# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The multiview video transport form and postprocessing, against the original path."""

import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from diffusers.video_processor import VideoProcessor

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm, unpack_diffusion_output_shm
from vllm_omni.diffusion.models.cosmos3.multiview_output import video_for_transport, video_from_transport

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _video(num_frames: int, dtype: torch.dtype, height: int = 6, width: int = 10) -> torch.Tensor:
    generator = torch.Generator().manual_seed(num_frames)
    # Values past [-1, 1] exercise the clamp.
    return (torch.randn(1, 3, num_frames, height, width, generator=generator) * 0.7).to(dtype)


def _postprocess(monkeypatch, check_video_safety=None):
    guardrails: Any = types.ModuleType("vllm_omni.diffusion.models.cosmos3.guardrails")
    guardrails.is_guardrails_enabled = lambda od_config, sampling_params=None: check_video_safety is not None
    guardrails.check_video_safety = check_video_safety
    monkeypatch.setitem(sys.modules, guardrails.__name__, guardrails)
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import get_cosmos3_multiview_post_process_func

    return get_cosmos3_multiview_post_process_func(SimpleNamespace(model_config={}))


def _original_postprocess(video, frames_per_view, output_type, check_video_safety=None):
    """PR 6917's path: IPC delivers the BF16 video, then VideoProcessor converts each camera."""
    processor = VideoProcessor(vae_scale_factor=16)
    views = []
    for start in range(0, video.shape[2], frames_per_view):
        view = video[:, :, start : start + frames_per_view]
        if check_video_safety is not None:
            view = check_video_safety(view)
        views.append(processor.postprocess_video(view, output_type=output_type))
    if output_type == "np":
        return np.concatenate(views, axis=1)
    if output_type == "pt":
        return torch.cat(views, dim=1)
    return [[frame for view in views for frame in view[0]]]


def test_transport_keeps_bf16_bits_through_shared_memory() -> None:
    video = _video(34, torch.bfloat16, height=80, width=128)
    # Values that a conversion to FP16 would change.
    video[0, 0, 0, 0, :3] = torch.tensor([1e-30, -3e-8, 7e4])
    sent, internal = video_for_transport(video)
    assert sent.data_ptr() == video.data_ptr()

    output = DiffusionOutput(output={"payload": {"video": sent}, "metadata": {"internal": internal}})
    pack_diffusion_output_shm(output)
    handle = output.output["payload"]["video"]
    # Sent as 2-byte raw bits, not widened to FP32.
    assert handle["numpy_dtype"] == "uint16" and handle["nbytes"] == video.numel() * 2
    unpack_diffusion_output_shm(output)

    received = video_from_transport(output.output["payload"]["video"], output.output["metadata"]["internal"])
    assert received.dtype == torch.bfloat16
    assert torch.equal(received.view(torch.uint16), video.view(torch.uint16))


def test_transport_leaves_other_dtypes_unchanged() -> None:
    video = _video(3, torch.float32)
    sent, internal = video_for_transport(video)
    assert sent is video and internal == {}
    assert video_from_transport(video, {}) is video
    assert video_from_transport(video, {"private": True}) is video


def _scaled_guardrail(view: torch.Tensor) -> torch.Tensor:
    # Like the real guardrail: returns float32 in [-1, 1] with changed values.
    return view.float() * 0.75


@pytest.mark.parametrize("output_type", ["np", "pt", "pil"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("num_views", "frames_per_view"), [(1, 1), (1, 17), (3, 16), (3, 33)])
@pytest.mark.parametrize("check_video_safety", [None, _scaled_guardrail])
def test_postprocess_matches_original(
    monkeypatch, output_type, dtype, num_views, frames_per_view, check_video_safety
) -> None:
    video = _video(num_views * frames_per_view, dtype)
    sent, internal = video_for_transport(video)
    lidar = torch.rand(1, 3, 2, 4, 5)
    output = {
        "payload": {"video": sent, "lidar": lidar},
        "metadata": {
            "internal": internal,
            "multiview": {"cameras": [f"camera_{i}" for i in range(num_views)], "frames_per_view": frames_per_view},
        },
    }

    result = _postprocess(monkeypatch, check_video_safety)(output, output_type=output_type)

    expected = _original_postprocess(video, frames_per_view, output_type, check_video_safety)
    actual = result["payload"]["video"]
    if output_type == "np":
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    elif output_type == "pt":
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    else:
        assert len(actual[0]) == len(expected[0])
        assert all(np.array_equal(np.asarray(a), np.asarray(e)) for a, e in zip(actual[0], expected[0]))
    assert torch.equal(result["payload"]["lidar"], lidar)
    assert "internal" not in result["metadata"]


def test_latent_postprocess_returns_the_restored_video(monkeypatch) -> None:
    video = _video(4, torch.bfloat16)
    sent, internal = video_for_transport(video)
    output = {"payload": {"video": sent}, "metadata": {"internal": internal}}

    result = _postprocess(monkeypatch)(output, output_type="latent")

    assert result["payload"]["video"].dtype == torch.bfloat16
    assert torch.equal(result["payload"]["video"], video)
    # The caller's output is not modified.
    assert output["payload"]["video"] is sent
