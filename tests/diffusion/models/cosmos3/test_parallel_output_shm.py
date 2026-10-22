# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise production owner decode and CPU fallbacks without CUDA or models."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.vae_owner import VaeWork, plan_vae_work
from vllm_omni.diffusion.output_shm import SharedVideo, output_owner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def method(name, extra=None):
    path = Path(__file__).parents[4] / "vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3_multiview.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Cosmos3MultiviewPipeline"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = dict(torch=torch, as_bool=lambda x, default: bool(x), **(extra or {}))
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), env)
    return env[name]


def group(size, rank=0):
    return SimpleNamespace(
        world_size=size, rank_in_group=rank, ranks=list(range(size)), cpu_group=object(), device_group=object()
    )


@pytest.mark.parametrize("cp,cfg", [(1, 1), (1, 2), (2, 1), (1, 4), (2, 2), (4, 1), (1, 8), (2, 4), (4, 2), (8, 1)])
@pytest.mark.parametrize(
    "joint,count,frames,h,w", [(joint, count, 3, 2, 5) for joint in [False, True] for count in [1, 4, 7, 11]]
)
@pytest.mark.parametrize(
    "dtype,enabled,owner",
    [
        (torch.bfloat16, True, True),
        (torch.float32, True, True),
        (torch.bfloat16, False, True),
        (torch.bfloat16, True, False),
    ],
)
def test_owner_decode_direct_and_fallback(monkeypatch, cp, cfg, joint, count, frames, h, w, dtype, enabled, owner):
    import vllm_omni.diffusion.distributed.parallel_state as ps

    sp, cg = group(cp), group(cfg)
    monkeypatch.setattr(ps, "get_sp_group", lambda: sp)
    monkeypatch.setattr(ps, "get_cfg_group", lambda: cg)
    latents = torch.arange(count * frames * h * w, dtype=torch.float32).reshape(1, 1, count * frames, h, w)
    clips = [latents.narrow(2, i * frames, frames).repeat(1, 3, 1, 1, 1).to(dtype) for i in range(count)]
    lidar = torch.arange(3 * frames * h * w, dtype=torch.float32).reshape(1, 3, frames, h, w)
    work = [VaeWork(str(i), "camera", frames, h, w) for i in range(count)]
    if joint:
        work.append(VaeWork("lidar", "lidar", frames, h, w))
    plan = plan_vae_work(work, cp, cfg)
    transfers = []

    class Exchange:
        def __init__(self, device, sp, cfg, metadata):
            self.rank = 0
            self.stream = object()
            self.order = iter(metadata[1])

        def transfer(self, value, source, *, gather, destination):
            key = next(self.order)
            reference = lidar if key == "lidar" else clips[int(key)]
            if source == 0:
                assert torch.equal(value, reference)
            destination.copy_(reference)
            transfers.append((key, destination))

        def finish(self, *, cpu):
            assert cpu

    original_allocate = SharedVideo.allocate
    monkeypatch.setattr(
        SharedVideo,
        "allocate",
        classmethod(lambda cls, shape, dtype, frames, cuda: original_allocate(shape, dtype, frames)),
    )
    monkeypatch.setattr(SharedVideo, "record_view", lambda self, i, stream: self._events.__setitem__(i, None))
    monkeypatch.setattr(torch.version, "cuda", "test" if enabled else None)
    original_empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *a, **kw: original_empty(
            *a, **{k: ("cpu" if k == "device" else v) for k, v in kw.items() if k != "pin_memory"}
        ),
    )
    decoder = SimpleNamespace(config={"range_projection": {"native_height": h, "semantic_width": w}})

    class LidarDecoder:
        config = decoder.config

        def __call__(self, x):
            return lidar

    obj = SimpleNamespace(
        device=torch.device("cuda"),
        vae=SimpleNamespace(dtype=dtype),
        vae_scale_factor_temporal=1,
        vae_scale_factor_spatial=1,
        lidar_decoder=LidarDecoder(),
        _decode_latents=lambda x: x.repeat(1, 3, 1, 1, 1).to(dtype),
    )
    with output_owner(owner):
        video, got_lidar = method("_owner_decode_outputs", {"VaeOwnerExchange": Exchange})(
            obj, [latents, lidar], count, frames, plan, {"return_output": True} if joint else None
        )
    direct = dtype == torch.bfloat16 and enabled and owner
    assert isinstance(video, SharedVideo) == direct
    video.clamp_(-1, 1)
    for i, clip in enumerate(clips):
        assert torch.equal(video.narrow(2, i * frames, frames).view(torch.uint8), clip.clamp(-1, 1).view(torch.uint8))
    if direct:
        for key, destination in transfers:
            if key != "lidar":
                assert destination.data_ptr() == video._view(int(key)).data_ptr()
    assert (got_lidar is None) == (not joint)
    if joint:
        assert torch.equal(got_lidar, lidar)


def test_shm_out_of_order_and_duplicate():
    video = SharedVideo.allocate((1, 3, 6, 2, 3), torch.bfloat16, [2, 2, 2])
    for i in [2, 0, 1]:
        video.copy_view(i, torch.full((1, 3, 2, 2, 3), i / 2, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        video.copy_view(0, video._view(0))
    video.clamp_(-1, 1)
    assert all(video._status(i) == 1 for i in range(3))


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError, OSError, ValueError, MemoryError])
def test_storage_failure_falls_back_except_capacity(monkeypatch, distributed, failure):
    import vllm_omni.diffusion.distributed.parallel_state as ps
    from vllm_omni.diffusion.shm_utils import SharedMemoryCapacityError

    monkeypatch.setattr(torch.version, "cuda", "test")
    monkeypatch.setattr(ps, "get_sp_group", lambda: group(1))
    monkeypatch.setattr(ps, "get_cfg_group", lambda: group(2))
    allocations = []

    def fail(*args, **kwargs):
        allocations.append(True)
        if failure is MemoryError:
            raise SharedMemoryCapacityError("required 100 bytes, available 0 bytes")
        raise failure("storage creation or CUDA registration failed")

    monkeypatch.setattr(SharedVideo, "allocate", fail)
    original_empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: original_empty(
            *args, **{k: ("cpu" if k == "device" else v) for k, v in kwargs.items() if k != "pin_memory"}
        ),
    )
    latents = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 2, 3)
    expected = latents.repeat(1, 3, 1, 1, 1).to(torch.bfloat16)
    obj = SimpleNamespace(
        device=torch.device("cuda"),
        vae=SimpleNamespace(dtype=torch.bfloat16),
        vae_scale_factor_temporal=1,
        vae_scale_factor_spatial=1,
        _decode_latents=lambda x: x.repeat(1, 3, 1, 1, 1).to(torch.bfloat16),
    )

    class Exchange:
        def __init__(self, device, sp, cfg, metadata):
            self.rank = 0

        def transfer(self, value, source, *, gather, destination):
            destination.copy_(value)

        def finish(self, *, cpu):
            assert cpu

    def run():
        if distributed:
            result, lidar = method("_owner_decode_outputs", {"VaeOwnerExchange": Exchange})(
                obj, [latents], 2, 2, ({"0": 0, "1": 0}, ["0", "1"]), None
            )
            assert lidar is None
            return result
        return method("_decode_multiview_latents")(obj, latents, num_views=2, latent_frames_per_view=2)

    with output_owner(True):
        if failure is MemoryError:
            with pytest.raises(SharedMemoryCapacityError, match="required 100 bytes, available 0 bytes"):
                run()
        else:
            actual = run()
            assert isinstance(actual, torch.Tensor)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert len(allocations) == 1
