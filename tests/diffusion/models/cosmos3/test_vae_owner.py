# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owner placement must depend on request geometry, not a particular rig."""

import itertools

import pytest

from vllm_omni.diffusion.models.cosmos3.vae_owner import VaeWork, plan_vae_work

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("sp,cfg", [(1, 1), (1, 2), (2, 1), (2, 2), (4, 2), (3, 2)])
@pytest.mark.parametrize("joint,mixed", list(itertools.product([False, True], repeat=2)))
def test_owner_plan_is_geometry_driven_and_rank_independent(sp, cfg, joint, mixed):
    size = sp * cfg
    for count in {1, max(1, size - 1), size, size + 3}:
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
