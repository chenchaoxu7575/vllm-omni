# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 multiview BF16 video keeps its bits through shared-memory IPC."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm, unpack_diffusion_output_shm
from vllm_omni.diffusion.shm_utils import SharedMemoryCapacityError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def payload(video, enabled=True):
    return DiffusionOutput(
        output={
            "payload": {"video": video},
            "metadata": {"internal": {"video_bf16_transport": enabled}},
        }
    )


@pytest.mark.parametrize("strided", [False, True])
def test_bf16_all_bit_patterns_round_trip(strided):
    # Includes signed zero, subnormals, infinities and all NaN payloads.
    source = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).repeat(16).reshape(1024, 1024)
    if strided:
        source = source.t()
    output = payload(source)
    pack_diffusion_output_shm(output)
    handle = output.output["payload"]["video"]
    assert handle["nbytes"] == source.numel() * 2
    assert handle["torch_dtype_view"] and handle["borrow_on_unpack"]
    unpack_diffusion_output_shm(output)
    result = output.output["payload"]["video"]
    assert torch.equal(result.contiguous().view(torch.int16), source.contiguous().view(torch.int16))


@pytest.mark.parametrize("dtype,enabled", [(torch.float32, True), (torch.bfloat16, False)])
def test_ipc_fallback_preserves_values(dtype, enabled):
    source = torch.arange(600000, dtype=torch.float32).remainder(257).to(dtype)
    output = payload(source, enabled)
    pack_diffusion_output_shm(output)
    handle = output.output["payload"]["video"]
    assert handle["nbytes"] == source.numel() * 4
    unpack_diffusion_output_shm(output)
    assert torch.equal(output.output["payload"]["video"].view(torch.uint8), source.view(torch.uint8))


def test_bf16_capacity_error_is_not_widened_or_mutated(monkeypatch):
    source = torch.zeros(600000, dtype=torch.bfloat16)
    output = payload(source)
    monkeypatch.setattr("os.statvfs", lambda _: SimpleNamespace(f_bavail=1, f_frsize=4096))
    with pytest.raises(SharedMemoryCapacityError, match="required 1200000 bytes, available 4096 bytes"):
        pack_diffusion_output_shm(output)
    assert output.output["payload"]["video"] is source
