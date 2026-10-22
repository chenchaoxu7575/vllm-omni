# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-sensor VAE owner placement on the sequence-parallel x CFG-parallel grid.

Each camera view, and the optional LiDAR stream, is assigned to one owner rank
that runs all of that sensor's VAE encodes and its decode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# Relative VAE cost per input pixel-frame, normalized to one camera decode, as
# (one encode pass, one decode). The camera encode weight includes CPU media
# preparation. The weights only order items and compare per-rank loads; they
# never affect numerics or timeouts.
_RELATIVE_COST: dict[str, tuple[float, float]] = {
    "camera": (0.897, 1.0),
    "lidar": (6.73, 6.79),
}


@dataclass(frozen=True)
class VaeWork:
    """VAE work of one sensor. ``encodes`` counts full-clip encode passes."""

    key: str
    kind: Literal["camera", "lidar"]
    frames: int
    height: int
    width: int
    encodes: float = 1.0
    decode: bool = True

    @property
    def cost(self) -> float:
        encode, decode = _RELATIVE_COST[self.kind]
        return self.frames * self.height * self.width * (encode * self.encodes + decode * self.decode)


@dataclass(frozen=True)
class VaePlan:
    """Owner grid rank per work key, and the transport order shared by all ranks."""

    owners: dict[str, int]
    order: list[str]


def _hops_to_output(rank: int, sp_size: int) -> int:
    """Gather hops from ``rank`` to grid rank 0: one per non-zero grid axis."""
    return int(rank % sp_size != 0) + int(rank // sp_size != 0)


def plan_vae_work(items: Sequence[VaeWork], sp_size: int, cfg_size: int) -> VaePlan:
    """Place work by longest processing time first.

    Items are visited by decreasing cost, ties by key. Each goes to the least
    loaded rank; ties prefer fewer gather hops, then the lower rank. Every rank
    derives the same plan from the same request, and ``order`` is also the
    transport order on ranks that own no work.
    """
    if sp_size < 1 or cfg_size < 1 or len({x.key for x in items}) != len(items):
        raise ValueError("Invalid VAE grid geometry or duplicate work keys.")
    if any(min(x.frames, x.height, x.width) < 1 or x.encodes < 0 for x in items):
        raise ValueError("VAE work requires positive geometry and nonnegative cost.")
    ordered = sorted(items, key=lambda x: (-x.cost, x.key))
    loads = [0.0] * (sp_size * cfg_size)
    owners: dict[str, int] = {}
    for item in ordered:
        owner = min(range(len(loads)), key=lambda r: (loads[r], _hops_to_output(r, sp_size), r))
        owners[item.key] = owner
        loads[owner] += item.cost
    return VaePlan(owners, [x.key for x in ordered])
