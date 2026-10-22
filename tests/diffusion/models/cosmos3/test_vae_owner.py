# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owner placement of Cosmos3 multiview VAE work."""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.models.cosmos3.vae_owner import VaeWork, plan_vae_work

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# SP x CFG grids up to eight ranks, including single-axis grids.
GRIDS = [(1, 2), (2, 1), (1, 4), (2, 2), (4, 1), (1, 8), (2, 4), (4, 2), (8, 1), (3, 2)]


def request_work(views: int, *, height: int, width: int, fields: int, lidar: bool) -> list[VaeWork]:
    items = [VaeWork(str(i), "camera", 93, height, width, encodes=fields) for i in range(views)]
    if lidar:
        items.append(VaeWork("lidar", "lidar", 31, 128, 1808, encodes=1.0))
    return items


@pytest.mark.parametrize("sp,cfg", GRIDS)
@pytest.mark.parametrize("views,lidar", [(1, True), (4, False), (7, True), (11, True), (11, False)])
def test_plan_is_deterministic_and_balanced(sp: int, cfg: int, views: int, lidar: bool) -> None:
    items = request_work(views, height=480, width=832, fields=2, lidar=lidar)
    plan = plan_vae_work(items, sp, cfg)

    assert sorted(plan.order) == sorted(x.key for x in items)
    assert set(plan.owners) == set(plan.order)
    assert all(0 <= rank < sp * cfg for rank in plan.owners.values())
    # Every rank derives the same plan regardless of item order.
    assert plan_vae_work(list(reversed(items)), sp, cfg) == plan
    loads = [sum(x.cost for x in items if plan.owners[x.key] == rank) for rank in range(sp * cfg)]
    assert max(loads) <= sum(x.cost for x in items) / (sp * cfg) + max(x.cost for x in items)


def test_ties_prefer_shortest_output_route() -> None:
    items = [VaeWork("heavy", "camera", 100, 10, 10), VaeWork("small", "camera", 1, 1, 1)]
    plan = plan_vae_work(items, 2, 2)
    assert plan.order == ["heavy", "small"]
    assert plan.owners == {"heavy": 0, "small": 1}


@pytest.mark.parametrize(
    "height,width,fields",
    [(480, 832, 2), (480, 832, 1), (480, 832, 0), (720, 1280, 2), (720, 1280, 1)],
)
@pytest.mark.parametrize("sp,cfg", [(4, 2), (2, 4), (8, 1)])
def test_lidar_goes_first_only_when_heavier_than_a_camera(
    height: int, width: int, fields: int, sp: int, cfg: int
) -> None:
    items = request_work(11, height=height, width=width, fields=fields, lidar=True)
    camera, lidar = items[0], items[-1]
    plan = plan_vae_work(items, sp, cfg)

    if lidar.cost > camera.cost:
        assert plan.order[0] == "lidar"
        assert plan.owners["lidar"] == 0
    else:
        assert plan.order[: len(items) - 1] == sorted(x.key for x in items[:-1])
        assert plan.order[-1] == "lidar"


def test_plan_rejects_ambiguous_work() -> None:
    item = VaeWork("camera", "camera", 1, 1, 1)
    with pytest.raises(ValueError):
        plan_vae_work([item, item], 2, 2)
    with pytest.raises(ValueError):
        plan_vae_work([item], 0, 2)
    with pytest.raises(ValueError):
        plan_vae_work([VaeWork("camera", "camera", 0, 1, 1)], 2, 2)
