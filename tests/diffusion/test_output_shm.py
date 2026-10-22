# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import gc
from multiprocessing import shared_memory

import pytest
import torch

from vllm_omni.diffusion.output_shm import SharedVideo, is_output_owner, output_owner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_output_owner_context_restores_after_error():
    assert not is_output_owner()
    with output_owner(True):
        assert is_output_owner()
        with pytest.raises(RuntimeError), output_owner(False):
            assert not is_output_owner()
            raise RuntimeError("RPC failed")
        assert is_output_owner()
    assert not is_output_owner()


@pytest.mark.parametrize("frames", [(17,), (1, 17, 33)])
def test_view_storage_is_contiguous_and_byte_exact(frames):
    source = [torch.randn(1, 3, f, 7, 5, dtype=torch.bfloat16).transpose(3, 4) for f in frames]
    owner = SharedVideo.allocate((1, 3, sum(frames), 5, 7), torch.bfloat16, frames)
    for i, view in enumerate(source):
        owner.copy_view(i, view)
    owner.clamp_(-1, 1)
    handle = owner.handle()
    reader = SharedVideo.from_handle(handle)
    start = 0
    for f, view in zip(frames, source):
        borrowed = reader.narrow(2, start, f)
        assert borrowed.is_contiguous()
        assert torch.equal(borrowed.view(torch.int16), view.clamp(-1, 1).view(torch.int16))
        start += f
    del owner, reader
    gc.collect()
    assert torch.equal(borrowed.view(torch.int16), source[-1].clamp(-1, 1).view(torch.int16))
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle["name"])


def test_abandoned_lease_unlinks_and_error_wakes_consumer():
    owner = SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,))
    name = owner.shm.name
    owner._status(0, -1)
    with pytest.raises(RuntimeError, match="producer failed"):
        owner.narrow(2, 0, 2)
    del owner
    gc.collect()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_invalid_dimensions_and_dtype_rejected_before_allocation():
    with pytest.raises(ValueError):
        SharedVideo.allocate((1, 3, 2, 5, 7), torch.float32, (2,))
    with pytest.raises(ValueError):
        SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (1,))


def test_copy_shape_error_does_not_publish_view():
    owner = SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,))
    with pytest.raises(ValueError):
        owner.copy_view(0, torch.zeros(1, 3, 1, 5, 7, dtype=torch.bfloat16))
    assert owner._status(0) == 0
    with pytest.raises(RuntimeError):
        owner.clamp_(-1, 1)


def test_completion_thread_preserves_model_inference_mode():
    with torch.inference_mode():
        owner = SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,))
        owner.copy_view(0, torch.zeros(1, 3, 2, 5, 7, dtype=torch.bfloat16))
    assert owner.tensor.is_inference()
    # Events are None on CPU; this selects the production background path.
    owner._stream = object()
    owner.clamp_(-1, 1)
    owner._thread.join(5)
    assert not owner._thread.is_alive()
    assert owner._status(0) == 1


def test_capacity_precheck_prevents_allocation(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.shm_utils import SharedMemoryCapacityError

    monkeypatch.setattr("os.statvfs", lambda _: SimpleNamespace(f_bavail=1, f_frsize=4096))
    calls = []
    monkeypatch.setattr(shared_memory, "SharedMemory", lambda **kw: calls.append(kw))
    # 4096 header + 420 bytes BF16 pixels.
    with pytest.raises(SharedMemoryCapacityError, match="required 4516 bytes, available 4096 bytes"):
        SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,))
    assert calls == []


def test_capacity_exact_fit_and_free_blocks(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.shm_utils import SharedMemoryCapacityError, require_shared_memory

    monkeypatch.setattr("os.statvfs", lambda _: SimpleNamespace(f_bavail=2, f_bfree=99, f_frsize=4096))
    require_shared_memory(8192)
    with pytest.raises(SharedMemoryCapacityError, match="required 8193 bytes, available 8192 bytes"):
        require_shared_memory(8193)


@pytest.mark.parametrize(
    "device,cuda,dtype,owner,expected",
    [
        ("cuda", "13.0", torch.bfloat16, True, True),
        ("cuda", None, torch.bfloat16, True, False),
        ("cpu", "13.0", torch.bfloat16, True, False),
        ("xpu", "13.0", torch.bfloat16, True, False),
        ("npu", "13.0", torch.bfloat16, True, False),
        ("cuda", "13.0", torch.float32, True, False),
        ("cuda", "13.0", torch.bfloat16, False, False),
    ],
)
def test_direct_storage_automatic_fallback(monkeypatch, device, cuda, dtype, owner, expected):
    from types import SimpleNamespace

    from vllm_omni.diffusion.output_shm import direct_video_supported

    monkeypatch.setattr(torch.version, "cuda", cuda)
    with output_owner(owner):
        assert direct_video_supported(SimpleNamespace(type=device), dtype) == expected
