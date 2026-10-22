# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared-memory multiview video output: storage, publication, fallbacks and IPC."""

from __future__ import annotations

import contextlib
import gc
from multiprocessing import shared_memory
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.ipc import discard_direct_output_shm, pack_diffusion_output_shm, unpack_diffusion_output_shm
from vllm_omni.diffusion.models.cosmos3.vae_owner import GridCoord, decode_and_gather
from vllm_omni.diffusion.output_shm import (
    SharedVideo,
    allocate_video_output,
    direct_video_supported,
    is_output_owner,
    output_owner,
)
from vllm_omni.diffusion.shm_utils import SharedMemoryCapacityError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch):
    """Registered storage without a GPU: host registration and streams are recorded."""
    calls: list = []

    class Stream:
        device = SimpleNamespace(index=0)

        def wait_stream(self, other: object) -> None:
            calls.append("wait_stream")

        def synchronize(self) -> None:
            calls.append("stream_synchronize")

    class Event:
        fail = False

        def record(self, stream: object = None) -> None:
            calls.append("event_record")

        def synchronize(self) -> None:
            if Event.fail:
                raise RuntimeError("device copy failed")
            calls.append("event_synchronize")

    class Runtime:
        def cudaHostRegister(self, ptr: int, nbytes: int, flags: int) -> int:
            calls.append(("register", nbytes))
            return 0

        def cudaHostUnregister(self, ptr: int) -> int:
            calls.append("unregister")
            return 0

        def cudaGetErrorString(self, error: int) -> str:
            return "error"

    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch.cuda, "cudart", lambda: Runtime())
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.accelerator, "device_index", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    yield SimpleNamespace(calls=calls, Event=Event)
    # Release registered storage while the fakes are still installed.
    gc.collect()


def join_completion(video: SharedVideo) -> None:
    assert video._thread is not None
    video._thread.join(5)
    assert not video._thread.is_alive()


def test_output_owner_context_restores_after_error() -> None:
    assert not is_output_owner()
    with output_owner(True):
        assert is_output_owner()
        with pytest.raises(RuntimeError), output_owner(False):
            assert not is_output_owner()
            raise RuntimeError("RPC failed")
        assert is_output_owner()
    assert not is_output_owner()


@pytest.mark.parametrize("frames", [(17,), (1, 17, 33)])
def test_views_round_trip_byte_exact(frames: tuple[int, ...]) -> None:
    source = [torch.randn(1, 3, f, 7, 5, dtype=torch.bfloat16).transpose(3, 4) for f in frames]
    owner = SharedVideo.allocate((1, 3, sum(frames), 5, 7), torch.bfloat16, frames)
    for i, view in enumerate(source):
        owner.copy_view(i, view)
    owner.finalize(-1, 1)
    handle = owner.handle()
    reader = SharedVideo.from_handle(handle)
    for i, view in enumerate(source):
        borrowed = reader.wait_view(i)
        assert borrowed.is_contiguous()
        assert torch.equal(borrowed.view(torch.int16), view.clamp(-1, 1).view(torch.int16))
    whole = reader.to_tensor()
    assert torch.equal(whole, torch.cat([view.clamp(-1, 1) for view in source], dim=2))
    del owner, reader
    gc.collect()
    # Borrowed storage outlives both wrappers; the name is already unlinked.
    assert torch.equal(borrowed.view(torch.int16), source[-1].clamp(-1, 1).view(torch.int16))
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle["name"])


def test_registered_views_publish_on_completion_thread(fake_cuda) -> None:
    with torch.inference_mode():
        owner = SharedVideo.allocate((1, 3, 4, 5, 7), torch.bfloat16, (2, 2), cuda=True)
        for i in range(2):
            owner.copy_view(i, torch.full((1, 3, 2, 5, 7), 2.0 * i - 3, dtype=torch.bfloat16))
    assert owner.tensor.is_inference()
    owner.finalize(-1, 1)
    join_completion(owner)
    reader = SharedVideo.from_handle(owner.handle())
    assert reader.wait_view(0).eq(-1).all() and reader.wait_view(1).eq(-1).all()
    assert ("register", 3 * 4 * 5 * 7 * 2) in fake_cuda.calls
    assert fake_cuda.calls.count("event_synchronize") == 2 and fake_cuda.calls.count("unregister") == 1


def test_producer_failure_wakes_reader(fake_cuda) -> None:
    owner = SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,), cuda=True)
    owner.copy_view(0, torch.zeros(1, 3, 2, 5, 7, dtype=torch.bfloat16))
    fake_cuda.Event.fail = True
    owner.finalize(-1, 1)
    join_completion(owner)
    reader = SharedVideo.from_handle(owner.handle())
    with pytest.raises(RuntimeError, match="producer failed"):
        reader.wait_view(0)
    assert "unregister" in fake_cuda.calls


def test_abandoned_storage_is_unlinked() -> None:
    owner = SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,))
    name = owner.shm.name
    del owner
    gc.collect()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_invalid_dimensions_and_dtype_rejected_before_allocation() -> None:
    with pytest.raises(ValueError):
        SharedVideo.allocate((1, 3, 2, 5, 7), torch.float32, (2,))
    with pytest.raises(ValueError):
        SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (1,))


def test_incomplete_output_cannot_be_published() -> None:
    owner = SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,))
    with pytest.raises(ValueError):
        owner.copy_view(0, torch.zeros(1, 3, 1, 5, 7, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="Incomplete"):
        owner.finalize(-1, 1)


def test_capacity_precheck_prevents_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.statvfs", lambda _: SimpleNamespace(f_bavail=1, f_frsize=4096))
    calls: list = []
    monkeypatch.setattr(shared_memory, "SharedMemory", lambda **kw: calls.append(kw))
    # One page of readiness words plus 420 bytes of BF16 pixels.
    with pytest.raises(SharedMemoryCapacityError, match="required 4516 bytes, available 4096 bytes"):
        SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,))
    assert calls == []


def test_failed_construction_does_not_leak_the_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    names = []
    create = shared_memory.SharedMemory

    def tracking(**kwargs: Any) -> shared_memory.SharedMemory:
        segment = create(**kwargs)
        names.append(segment.name)
        return segment

    def broken_mapping(array: object) -> torch.Tensor:
        raise RuntimeError("mapping failed")

    monkeypatch.setattr(shared_memory, "SharedMemory", tracking)
    # Fail inside the constructor, after the segment exists.
    monkeypatch.setattr(torch, "from_numpy", broken_mapping)
    with pytest.raises(RuntimeError, match="mapping failed"):
        SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,))
    monkeypatch.undo()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=names[0])


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
def test_direct_storage_gate(monkeypatch: pytest.MonkeyPatch, device, cuda, dtype, owner, expected) -> None:
    monkeypatch.setattr(torch.version, "cuda", cuda)
    with output_owner(owner):
        assert direct_video_supported(SimpleNamespace(type=device), dtype) == expected


def test_allocate_video_output_uses_shared_storage_when_requested() -> None:
    shared = allocate_video_output((1, 3, 4, 2, 3), torch.bfloat16, [2, 2], shared=True, cuda=False)
    assert isinstance(shared, SharedVideo) and shared.frames == (2, 2)
    plain = allocate_video_output((1, 3, 4, 2, 3), torch.bfloat16, [2, 2], shared=False, cuda=False)
    assert isinstance(plain, torch.Tensor) and plain.shape == (1, 3, 4, 2, 3)


@pytest.mark.parametrize("failure", [RuntimeError, OSError])
def test_allocate_video_output_falls_back_on_storage_errors(monkeypatch: pytest.MonkeyPatch, failure) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise failure("storage creation or CUDA registration failed")

    monkeypatch.setattr(SharedVideo, "allocate", fail)
    video = allocate_video_output((1, 3, 4, 2, 3), torch.bfloat16, [2, 2], shared=True, cuda=True)
    assert isinstance(video, torch.Tensor) and video.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "failure", [SharedMemoryCapacityError("required 100 bytes, available 0 bytes"), ValueError("bad shape")]
)
def test_allocate_video_output_raises_other_errors(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(SharedVideo, "allocate", fail)
    with pytest.raises(type(failure)):
        allocate_video_output((1, 3, 4, 2, 3), torch.bfloat16, [2, 2], shared=True, cuda=True)


class CopyingExchange:
    """Output-rank exchange stand-in: the gathered tensor lands in its destination."""

    def __init__(self) -> None:
        self.rank = 0
        self.coord = GridCoord(0, 2, 0, 2)
        self.stream = object()
        self.gathered: list[str] = []

    def gather_to_output(self, tensor: torch.Tensor | None, owner: int, destination: torch.Tensor | None) -> None:
        assert tensor is not None and destination is not None
        destination.copy_(tensor)


@pytest.mark.parametrize("views", [1, 4, 7])
def test_owner_decode_writes_views_into_shared_storage(fake_cuda, views: int) -> None:
    keys = [str(i) for i in range(views)] + ["lidar"]
    owners = {key: index % 4 for index, key in enumerate(keys)}
    order = sorted(keys, key=lambda key: owners[key] == 0)
    clip = (1, 3, 3, 2, 5)
    specs = {key: (clip, torch.bfloat16) for key in keys[:-1]}
    specs["lidar"] = ((1, 3, 3, 2, 2), torch.float32)
    video = SharedVideo.allocate((1, 3, 3 * views, 2, 5), torch.bfloat16, [3] * views, cuda=True)
    lidar = torch.empty(specs["lidar"][0])
    destinations = {str(i): video.view(i) for i in range(views)} | {"lidar": lidar}
    exchange = CopyingExchange()
    recorded = []

    def decode(key: str) -> torch.Tensor:
        shape, dtype = specs[key]
        return torch.full(shape, 0.25 * (keys.index(key) + 1), dtype=dtype)

    def after_gather(key: str) -> None:
        recorded.append(key)
        if key != "lidar":
            video.record_view(int(key), exchange.stream)

    # Every item is decoded locally here; the stand-in copies it as a receive would.
    decode_and_gather(
        order=order,
        owners=dict.fromkeys(keys, 0),
        specs=specs,
        exchange=exchange,
        decode=decode,
        destinations=destinations,
        device=torch.device("cpu"),
        after_gather=after_gather,
    )
    assert recorded == order
    video.finalize(-1, 1)
    join_completion(video)
    reader = SharedVideo.from_handle(video.handle())
    for i in range(views):
        expected = min(0.25 * (i + 1), 1.0)
        assert reader.wait_view(i).eq(expected).all()
    assert lidar.eq(0.25 * len(keys)).all()


def test_ipc_hands_over_shared_video_and_discards_on_failure() -> None:
    video = SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,))
    video.copy_view(0, torch.full((1, 3, 2, 5, 7), 0.5, dtype=torch.bfloat16))
    video.finalize(-1, 1)
    output = DiffusionOutput(output={"payload": {"video": video}})
    pack_diffusion_output_shm(output)
    handle = output.output["payload"]["video"]
    assert isinstance(handle, dict) and handle["frames"] == [2]
    unpack_diffusion_output_shm(output)
    received = output.output["payload"]["video"]
    assert isinstance(received, SharedVideo) and received.wait_view(0).eq(0.5).all()

    abandoned = SharedVideo.allocate((1, 3, 2, 5, 7), torch.bfloat16, (2,))
    packed = DiffusionOutput(output={"payload": {"video": abandoned}})
    pack_diffusion_output_shm(packed)
    discard_direct_output_shm(packed)
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=packed.output["payload"]["video"]["name"])
