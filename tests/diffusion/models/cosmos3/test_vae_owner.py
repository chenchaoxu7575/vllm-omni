# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owner placement must depend on request geometry, not a particular rig."""

import itertools

import pytest

from vllm_omni.diffusion.models.cosmos3.vae_owner import VaeWork, plan_vae_work

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "sp,cfg", [(1, 1), (1, 2), (2, 1), (1, 4), (2, 2), (4, 1), (1, 8), (2, 4), (4, 2), (8, 1), (3, 2)]
)
@pytest.mark.parametrize("joint,mixed", list(itertools.product([False, True], repeat=2)))
def test_owner_plan_is_geometry_driven_and_rank_independent(sp, cfg, joint, mixed):
    size = sp * cfg
    for count in {1, 4, 7, 11, max(1, size - 1), size, size + 3}:
        items = [
            VaeWork(str(i), "camera", 5 + (i % 3 if mixed else 0), 4 + (i % 2 if mixed else 0), 8) for i in range(count)
        ]
        if joint:
            items.append(VaeWork("lidar", "lidar", 3, 2, 8))
        owners, order = plan_vae_work(items, sp, cfg)
        assert len(order) == len(items) == len(owners)
        assert set(order) == {x.key for x in items}
        assert all(0 <= r < size for r in owners.values())
        for _rank in range(size):
            assert plan_vae_work(list(reversed(items)), sp, cfg) == (owners, order)
        loads = [sum(x.cost for x in items if owners[x.key] == r) for r in range(size)]
        assert max(loads) <= sum(x.cost for x in items) / size + max(x.cost for x in items)


def test_owner_ties_prefer_shortest_output_route():
    items = [VaeWork("heavy", "camera", 100, 10, 10), VaeWork("small", "camera", 1, 1, 1)]
    owners, order = plan_vae_work(items, 2, 2)
    assert order == ["heavy", "small"]
    assert owners == {"heavy": 0, "small": 1}


def test_owner_rejects_ambiguous_work():
    item = VaeWork("camera", "camera", 1, 1, 1)
    with pytest.raises(ValueError):
        plan_vae_work([item, item], 2, 2)
    with pytest.raises(ValueError):
        plan_vae_work([item], 0, 2)
    with pytest.raises(ValueError):
        plan_vae_work([VaeWork("camera", "camera", 0, 1, 1)], 2, 2)


def test_replica_topology_rejects_unknown_or_overlapping_axes():
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.cosmos3.vae_owner import replica_geometry_supported

    def group(ranks):
        return SimpleNamespace(
            ranks=ranks, world_size=len(ranks), rank_in_group=0, cpu_group=object(), device_group=object()
        )

    assert not replica_geometry_supported(None, group([0]))
    assert not replica_geometry_supported(group([0]), group([0]))
    assert not replica_geometry_supported(group([0, 1]), group([0, 1]))
    assert not replica_geometry_supported(group([0, 0]), group([0, 4]))
    assert not replica_geometry_supported(group([0, 1]), group([4, 5]))
    assert replica_geometry_supported(group([0, 1, 2, 3]), group([0, 4]))
    assert replica_geometry_supported(group([0]), group([0, 4]))
    assert replica_geometry_supported(group([0, 1]), group([0]))


@pytest.mark.parametrize(
    "reason",
    [
        "cpu",
        "rocm",
        "xpu",
        "npu",
        "fp32",
        "vae_fp32",
        "one_item",
        "uninitialized",
        "distributed_vae",
        "no_groups",
        "one_rank",
        "unknown",
        "non_nccl",
    ],
)
def test_owner_automatic_fallback(monkeypatch, reason):
    from types import SimpleNamespace

    import torch

    import vllm_omni.diffusion.distributed.parallel_state as ps
    import vllm_omni.platforms as platforms
    from vllm_omni.diffusion.models.cosmos3.vae_owner import vae_replica_groups

    def group(ranks):
        return SimpleNamespace(
            ranks=ranks, world_size=len(ranks), rank_in_group=0, cpu_group=object(), device_group=object()
        )

    sp, cfg = group([0, 1, 2, 3]), group([0, 4])
    if reason == "one_rank":
        sp, cfg = group([0]), group([0])
    if reason == "unknown":
        sp = None
    monkeypatch.setattr(platforms, "current_omni_platform", SimpleNamespace(is_cuda=lambda: reason != "rocm"))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: reason != "uninitialized")
    monkeypatch.setattr(torch.distributed, "get_backend", lambda _: "gloo" if reason == "non_nccl" else "nccl")

    def get_sp():
        if reason == "no_groups":
            raise AssertionError("not initialized")
        return sp

    monkeypatch.setattr(ps, "get_sp_group", get_sp)
    monkeypatch.setattr(ps, "get_cfg_group", lambda: cfg)
    device = SimpleNamespace(type=reason if reason in {"cpu", "xpu", "npu"} else "cuda")
    dtype = torch.float32 if reason == "fp32" else torch.bfloat16
    vae = SimpleNamespace(
        dtype=torch.float32 if reason == "vae_fp32" else torch.bfloat16,
        is_distributed_enabled=lambda: reason == "distributed_vae",
    )
    assert vae_replica_groups(device, dtype, vae, 1 if reason == "one_item" else 12) is None
