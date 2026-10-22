# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""/dev/shm capacity is checked before shared-memory output is created."""

from multiprocessing import shared_memory
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm
from vllm_omni.diffusion.shm_utils import SharedMemoryCapacityError, require_shared_memory

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_capacity_exact_fit_and_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only blocks available to unprivileged users count.
    monkeypatch.setattr("os.statvfs", lambda _: SimpleNamespace(f_bavail=2, f_bfree=99, f_frsize=4096))
    require_shared_memory(8192)
    with pytest.raises(SharedMemoryCapacityError, match="required 8193 bytes, available 8192 bytes"):
        require_shared_memory(8193)


def test_pack_fails_before_creating_shared_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.statvfs", lambda _: SimpleNamespace(f_bavail=1, f_frsize=4096))
    created = []
    monkeypatch.setattr(shared_memory, "SharedMemory", lambda **kwargs: created.append(kwargs))
    source = torch.zeros(600000)
    output = DiffusionOutput(output=source)

    with pytest.raises(SharedMemoryCapacityError, match="required 2400000 bytes, available 4096 bytes"):
        pack_diffusion_output_shm(output)
    assert created == []
    assert output.output is source
