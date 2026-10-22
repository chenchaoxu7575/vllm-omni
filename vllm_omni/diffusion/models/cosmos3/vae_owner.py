# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-sensor VAE owners on the sequence-parallel x CFG-parallel grid.

Each camera view, and the optional LiDAR stream, is assigned to one owner rank
that runs all of that sensor's VAE encodes and its decode. Encoded latents are
broadcast from the owner to every grid rank; decoded outputs are gathered on
grid rank 0, which assembles the request output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_sp_group,
    model_parallel_is_initialized,
)
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator

logger = init_logger(__name__)

# Relative VAE cost per input pixel-frame, normalized to one camera decode, as
# (one encode pass, one decode). The camera encode weight includes CPU media
# preparation. The weights only order items and compare per-rank loads; they
# never affect numerics or timeouts.
_RELATIVE_COST: dict[str, tuple[float, float]] = {
    "camera": (0.897, 1.0),
    "lidar": (6.73, 6.79),
}

# (field, latent shape, dtype) for each encode of one sensor.
FieldSpec = tuple[str, tuple[int, ...], torch.dtype]


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


@dataclass(frozen=True)
class GridCoord:
    """Position of this rank on the grid; grid rank = ``cfg_rank * sp_size + sp_rank``."""

    sp_rank: int
    sp_size: int
    cfg_rank: int
    cfg_size: int

    @property
    def rank(self) -> int:
        return self.cfg_rank * self.sp_size + self.sp_rank


@dataclass(frozen=True)
class CommStep:
    """One communication step; ``peer`` is a rank index within the ``axis`` group."""

    op: Literal["broadcast", "send", "recv"]
    axis: Literal["sp", "cfg"]
    peer: int


def broadcast_steps(owner: int, coord: GridCoord) -> list[CommStep]:
    """Owner to its CFG row along SP, then every SP column along CFG."""
    owner_cfg, owner_sp = divmod(owner, coord.sp_size)
    steps = []
    if coord.cfg_rank == owner_cfg and coord.sp_size > 1:
        steps.append(CommStep("broadcast", "sp", owner_sp))
    if coord.cfg_size > 1:
        steps.append(CommStep("broadcast", "cfg", owner_cfg))
    return steps


def gather_steps(owner: int, coord: GridCoord) -> list[CommStep]:
    """Owner to SP rank 0 of its CFG row, then along CFG to grid rank 0."""
    owner_cfg, owner_sp = divmod(owner, coord.sp_size)
    steps = []
    if coord.cfg_rank == owner_cfg and owner_sp != 0:
        if coord.sp_rank == owner_sp:
            steps.append(CommStep("send", "sp", 0))
        elif coord.sp_rank == 0:
            steps.append(CommStep("recv", "sp", owner_sp))
    if coord.sp_rank == 0 and owner_cfg != 0:
        if coord.cfg_rank == owner_cfg:
            steps.append(CommStep("send", "cfg", 0))
        elif coord.cfg_rank == 0:
            steps.append(CommStep("recv", "cfg", owner_cfg))
    return steps


def gather_needs_buffer(owner: int, coord: GridCoord) -> bool:
    """Whether this rank receives the owner's output, to relay it or to output it."""
    return any(step.op == "recv" for step in gather_steps(owner, coord))


@dataclass(frozen=True)
class VaeGrid:
    """The SP and CFG groups that share VAE work."""

    sp: GroupCoordinator
    cfg: GroupCoordinator

    @property
    def coord(self) -> GridCoord:
        return GridCoord(self.sp.rank_in_group, self.sp.world_size, self.cfg.rank_in_group, self.cfg.world_size)


def _grid_axes_valid(sp: GroupCoordinator | None, cfg: GroupCoordinator | None) -> bool:
    """Require two known, independent grid axes that meet only at this rank."""
    for group in (sp, cfg):
        if group is None:
            return False
        size = getattr(group, "world_size", 0)
        ranks = getattr(group, "ranks", ())
        rank = getattr(group, "rank_in_group", -1)
        if size < 1 or len(ranks) != size or len(set(ranks)) != size or not 0 <= rank < size:
            return False
        if getattr(group, "cpu_group", None) is None or getattr(group, "device_group", None) is None:
            return False
    assert sp is not None and cfg is not None
    return (
        sp.world_size * cfg.world_size > 1
        and sp.ranks[sp.rank_in_group] == cfg.ranks[cfg.rank_in_group]
        and len(set(sp.ranks) & set(cfg.ranks)) == 1
    )


def vae_grid(device: torch.device, dtype: torch.dtype, vae: Any, work_count: int) -> VaeGrid | None:
    """Return the grid that shares VAE work, or None to run the VAE serially.

    Owner scheduling is limited to the configuration it was validated on:

    - CUDA device and platform, with NCCL device groups: the exchange relies on
      NCCL ``Work.wait()`` only ordering the communication stream, not
      blocking the host.
    - BF16 model and VAE: the validated precision. Owner and serial paths were
      compared byte for byte only in BF16.
    - A non-distributed VAE: a distributed VAE already splits each decode.
    - At least two work items and a grid larger than one rank.

    Any other configuration keeps the serial path on every rank.
    """
    result = _grid_or_reason(device, dtype, vae, work_count)
    if isinstance(result, str):
        logger.info_once("Cosmos3 multiview VAE runs serially on every rank: %s.", result)
        return None
    logger.info_once(
        "Cosmos3 multiview VAE work is distributed over the SP=%d x CFG=%d grid.",
        result.sp.world_size,
        result.cfg.world_size,
    )
    return result


def _grid_or_reason(device: torch.device, dtype: torch.dtype, vae: Any, work_count: int) -> VaeGrid | str:
    if device.type != "cuda" or not current_omni_platform.is_cuda():
        return "requires a CUDA device"
    if dtype != torch.bfloat16 or getattr(vae, "dtype", None) != torch.bfloat16:
        return "requires a BF16 model and VAE"
    if work_count < 2:
        return "a single VAE work item"
    if not dist.is_initialized():
        return "torch.distributed is not initialized"
    distributed = getattr(vae, "is_distributed_enabled", None)
    if callable(distributed) and distributed():
        return "the VAE is already distributed"
    if not model_parallel_is_initialized():
        return "model parallel groups are not initialized"
    sp, cfg = get_sp_group(), get_cfg_group()
    if not _grid_axes_valid(sp, cfg):
        return "no independent SP x CFG grid with more than one rank"
    if any(dist.get_backend(g.device_group) != "nccl" for g in (sp, cfg) if g.world_size > 1):
        return "requires NCCL device groups"
    return VaeGrid(sp, cfg)


class VaeOwnerExchange:
    """Move VAE inputs and outputs between owners on the existing SP and CFG groups.

    The WORLD group is never used, so data- and tensor-parallel ranks stay
    isolated. Every rank visits the same items in the same order. Each
    ``Work.wait()`` only makes the communication stream wait; the compute stream
    waits once in :meth:`finish`. ``record_stream`` keeps the allocator from
    reusing transported tensors while communication may still access them.
    """

    def __init__(self, device: torch.device, grid: VaeGrid, metadata: object) -> None:
        sp, cfg = grid.sp, grid.cfg
        self.grid = grid
        self.coord = grid.coord
        self.rank = self.coord.rank
        # Confirm that every grid rank planned the same work before transport.
        reports: list[Any] = [None] * sp.world_size
        dist.all_gather_object(reports, metadata, group=sp.cpu_group)
        groups: list[Any] = [None] * cfg.world_size
        dist.all_gather_object(groups, reports, group=cfg.cpu_group)
        if any(x != metadata for group in groups for x in group):
            raise ValueError("VAE owner work metadata differs between grid ranks.")
        # Communication stream; transported tensors are written on it.
        self.stream = current_omni_platform.Stream(device=device)
        self.stream.wait_stream(current_omni_platform.current_stream(device))
        self._tensors: list[torch.Tensor] = []

    def broadcast_from_owner(self, tensor: torch.Tensor, owner: int) -> None:
        """Broadcast ``tensor`` in place from grid rank ``owner`` to every grid rank."""
        self._run(broadcast_steps(owner, self.coord), tensor)

    def gather_to_output(self, tensor: torch.Tensor | None, owner: int, destination: torch.Tensor | None) -> None:
        """Send the owner's ``tensor`` to grid rank 0 and copy it into ``destination`` there.

        ``tensor`` is the owner's output or a receive buffer on relaying ranks
        and is None on ranks outside the route.
        """
        self._run(gather_steps(owner, self.coord), tensor, destination)

    def _run(self, steps: list[CommStep], tensor: torch.Tensor | None, destination: torch.Tensor | None = None) -> None:
        ready = current_omni_platform.Event()
        ready.record(current_omni_platform.current_stream())
        with current_omni_platform.stream(self.stream):
            self.stream.wait_event(ready)
            for step in steps:
                self._issue(step, tensor).wait()
            if destination is not None:
                assert tensor is not None
                _copy_to_host(destination, tensor)
            if tensor is not None:
                tensor.record_stream(self.stream)
                self._tensors.append(tensor)

    def _issue(self, step: CommStep, tensor: torch.Tensor | None) -> Any:
        group = self.grid.sp if step.axis == "sp" else self.grid.cfg
        peer = group.ranks[step.peer]
        if step.op == "broadcast":
            return dist.broadcast(tensor, src=peer, group=group.device_group, async_op=True)
        if step.op == "send":
            return dist.isend(tensor, dst=peer, group=group.device_group)
        return dist.irecv(tensor, src=peer, group=group.device_group)

    def finish(self, *, cpu: bool = False) -> None:
        """Make outputs usable: by the host when ``cpu``, else by the compute stream."""
        if cpu:
            self.stream.synchronize()
        else:
            current_omni_platform.current_stream().wait_stream(self.stream)
            for tensor in self._tensors:
                tensor.record_stream(current_omni_platform.current_stream())


def _copy_to_host(destination: torch.Tensor, tensor: torch.Tensor) -> None:
    if destination.is_contiguous():
        destination.copy_(tensor, non_blocking=True)
        return
    # A camera slice of [B, C, V*F, H, W] is not contiguous. Copy contiguous
    # [F, H, W] spans directly into pinned host storage: a strided D2H copy may
    # stage through a temporary and reorder on the host, which is not asynchronous.
    for batch in range(destination.shape[0]):
        for channel in range(destination.shape[1]):
            destination[batch, channel].copy_(tensor[batch, channel], non_blocking=True)


def receive_buffers(
    plan: VaePlan, specs: Mapping[str, Sequence[FieldSpec]], rank: int, device: torch.device
) -> dict[tuple[str, str], torch.Tensor]:
    """Allocate a broadcast receive buffer for every field this rank does not own."""
    return {
        (key, field): torch.empty(shape, dtype=dtype, device=device)
        for key in plan.order
        for field, shape, dtype in specs[key]
        if plan.owners[key] != rank
    }


def broadcast_owned_work(
    *,
    plan: VaePlan,
    specs: Mapping[str, Sequence[FieldSpec]],
    exchange: VaeOwnerExchange,
    buffers: Mapping[tuple[str, str], torch.Tensor],
    prepare: Callable[[str], Any],
    encode: Callable[[str, str, Any], torch.Tensor],
) -> dict[tuple[str, str], torch.Tensor]:
    """Encode owned fields and broadcast every field in plan order.

    ``prepare(key)`` runs once per owned item before its fields;
    ``encode(key, field, prepared)`` returns one field's latents. The caller
    calls ``exchange.finish()`` before using the returned tensors.
    """
    values = {}
    for key in plan.order:
        owned = plan.owners[key] == exchange.rank
        prepared = prepare(key) if owned else None
        for field, shape, dtype in specs[key]:
            if owned:
                value = encode(key, field, prepared)
                if tuple(value.shape) != shape or value.dtype != dtype:
                    raise ValueError(f"VAE output geometry differs from planned geometry for {key}/{field}.")
                value = value.contiguous()
            else:
                value = buffers[key, field]
            exchange.broadcast_from_owner(value, plan.owners[key])
            values[key, field] = value
    return values


def decode_and_gather(
    *,
    order: Sequence[str],
    owners: Mapping[str, int],
    specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
    exchange: VaeOwnerExchange,
    decode: Callable[[str], torch.Tensor],
    destinations: Mapping[str, torch.Tensor] | None,
    device: torch.device,
    after_gather: Callable[[str], None] | None = None,
) -> None:
    """Decode owned items and gather every item to grid rank 0 in ``order``.

    ``destinations`` maps each key to its host output on grid rank 0 and is
    None elsewhere. Receive buffers are allocated before any decode.
    ``after_gather(key)`` runs once the copy of ``key`` has been queued. The
    caller calls ``exchange.finish(cpu=True)`` before reading the destinations.
    """
    buffers = {
        key: torch.empty(specs[key][0], dtype=specs[key][1], device=device)
        for key in order
        if gather_needs_buffer(owners[key], exchange.coord)
    }
    for key in order:
        value = buffers.get(key)
        if exchange.rank == owners[key]:
            value = decode(key).contiguous()
            if tuple(value.shape) != specs[key][0] or value.dtype != specs[key][1]:
                raise ValueError(f"Decoded VAE geometry differs from planned geometry for {key}.")
        exchange.gather_to_output(value, owners[key], destinations[key] if destinations is not None else None)
        if after_gather is not None:
            after_gather(key)
