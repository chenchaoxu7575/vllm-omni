# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owner placement, routing and exchange loops of Cosmos3 multiview VAE scheduling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import vae_owner
from vllm_omni.diffusion.models.cosmos3.vae_owner import (
    GridCoord,
    VaeGrid,
    VaeOwnerExchange,
    VaePlan,
    VaeWork,
    broadcast_owned_work,
    broadcast_steps,
    decode_and_gather,
    gather_needs_buffer,
    gather_steps,
    plan_vae_work,
    receive_buffers,
    vae_grid,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# SP x CFG grids up to eight ranks, including single-axis grids.
GRIDS = [(1, 2), (2, 1), (1, 4), (2, 2), (4, 1), (1, 8), (2, 4), (4, 2), (8, 1), (3, 2)]


def coords(sp_size: int, cfg_size: int) -> list[GridCoord]:
    return [GridCoord(rank % sp_size, sp_size, rank // sp_size, cfg_size) for rank in range(sp_size * cfg_size)]


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


def axis_groups(sp: int, cfg: int, axis: str) -> list[list[int]]:
    """Grid ranks of every SP group (a CFG row) or CFG group (an SP column), in group order."""
    if axis == "sp":
        return [[row * sp + i for i in range(sp)] for row in range(cfg)]
    return [[i * sp + column for i in range(cfg)] for column in range(sp)]


@pytest.mark.parametrize("sp,cfg", GRIDS)
def test_broadcast_reaches_every_rank(sp: int, cfg: int) -> None:
    grid = coords(sp, cfg)
    for owner in range(sp * cfg):
        steps = {coord.rank: broadcast_steps(owner, coord) for coord in grid}
        assert all(
            [s.axis for s in rank_steps] in ([], ["sp"], ["cfg"], ["sp", "cfg"]) for rank_steps in steps.values()
        )
        holders = {owner}
        for axis in ("sp", "cfg"):
            for members in axis_groups(sp, cfg, axis):
                calls = [next((s for s in steps[rank] if s.axis == axis), None) for rank in members]
                if calls[0] is None:
                    assert all(call is None for call in calls)
                    continue
                # One collective: every member joins with the same source.
                assert all(call == calls[0] for call in calls)
                assert members[calls[0].peer] in holders
                holders.update(members)
        assert holders == set(range(sp * cfg)), (sp, cfg, owner)


@pytest.mark.parametrize("sp,cfg", GRIDS)
def test_gather_routes_to_output_rank(sp: int, cfg: int) -> None:
    grid = coords(sp, cfg)
    for owner in range(sp * cfg):
        holders = {owner}
        steps = {coord.rank: gather_steps(owner, coord) for coord in grid}
        assert all(
            [s.axis for s in rank_steps] in ([], ["sp"], ["cfg"], ["sp", "cfg"]) for rank_steps in steps.values()
        )
        for axis in ("sp", "cfg"):
            sends = [(c, s) for c in grid for s in steps[c.rank] if s.axis == axis and s.op == "send"]
            recvs = [(c, s) for c in grid for s in steps[c.rank] if s.axis == axis and s.op == "recv"]
            assert len(sends) == len(recvs) <= 1
            for (sender, send), (receiver, recv) in zip(sends, recvs):
                assert sender.rank in holders
                if axis == "sp":
                    assert receiver.rank == sender.cfg_rank * sp + send.peer
                    assert sender.rank == receiver.cfg_rank * sp + recv.peer
                else:
                    assert receiver.rank == send.peer * sp + sender.sp_rank
                    assert sender.rank == recv.peer * sp + receiver.sp_rank
                holders.add(receiver.rank)
        assert 0 in holders
        for coord in grid:
            # Receivers are exactly grid rank 0 and SP rank 0 of the owner's CFG row.
            receiver = coord.rank == 0 or (coord.cfg_rank == owner // sp and coord.sp_rank == 0)
            assert gather_needs_buffer(owner, coord) == (receiver and coord.rank != owner)


class RecordingExchange:
    """Exchange stand-in that records calls and moves no data."""

    def __init__(self, rank: int, sp: int = 2, cfg: int = 2) -> None:
        self.rank = rank
        self.coord = GridCoord(rank % sp, sp, rank // sp, cfg)
        self.calls: list[tuple] = []

    def broadcast_from_owner(self, tensor: torch.Tensor, owner: int) -> None:
        self.calls.append(("broadcast", owner, tensor))

    def gather_to_output(self, tensor: torch.Tensor | None, owner: int, destination: torch.Tensor | None) -> None:
        self.calls.append(("gather", owner, tensor, destination))
        if destination is not None:
            assert tensor is not None
            destination.copy_(tensor)


def test_broadcast_owned_work_encodes_owned_fields_in_plan_order() -> None:
    plan = VaePlan({"0": 1, "1": 0, "lidar": 1}, ["lidar", "0", "1"])
    specs = {
        "0": [("vision", (1, 2), torch.float32), ("control", (1, 2), torch.float32)],
        "1": [("vision", (1, 2), torch.float32), ("control", (1, 2), torch.float32)],
        "lidar": [("control", (1, 3), torch.bfloat16)],
    }
    exchange = RecordingExchange(rank=1)
    buffers = receive_buffers(plan, specs, exchange.rank, torch.device("cpu"))
    assert set(buffers) == {("1", "vision"), ("1", "control")}
    events: list[tuple] = []

    def encode(key: str, field: str) -> torch.Tensor:
        events.append((key, field))
        shape, dtype = next((s, d) for f, s, d in specs[key] if f == field)
        return torch.zeros(shape, dtype=dtype)

    values = broadcast_owned_work(plan=plan, specs=specs, exchange=exchange, buffers=buffers, encode=encode)

    assert events == [("lidar", "control"), ("0", "vision"), ("0", "control")]
    assert [(op, owner) for op, owner, _ in exchange.calls] == [("broadcast", 1)] * 3 + [("broadcast", 0)] * 2
    assert values["1", "vision"] is buffers["1", "vision"]
    assert [tensor for *_, tensor in exchange.calls] == [values[k, f] for k in plan.order for f, _, _ in specs[k]]


def test_broadcast_owned_work_rejects_unplanned_geometry() -> None:
    plan = VaePlan({"0": 0}, ["0"])
    specs = {"0": [("control", (1, 2), torch.float32)]}
    with pytest.raises(ValueError, match="planned geometry"):
        broadcast_owned_work(
            plan=plan,
            specs=specs,
            exchange=RecordingExchange(rank=0),
            buffers={},
            encode=lambda key, field: torch.zeros(1, 3),
        )


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_decode_and_gather_decodes_owned_items_and_fills_output(rank: int) -> None:
    owners = {"0": 3, "1": 2, "lidar": 1, "2": 0}
    order = ["0", "1", "lidar", "2"]
    specs = {key: ((1, 3, 2, 2, 2), torch.float32) for key in order}
    exchange = RecordingExchange(rank=rank)
    outputs = {key: torch.full(specs[key][0], -1.0) for key in order} if rank == 0 else None
    decoded = []

    def decode(key: str) -> torch.Tensor:
        decoded.append(key)
        return torch.full(specs[key][0], float(order.index(key)))

    decode_and_gather(
        order=order,
        owners=owners,
        specs=specs,
        exchange=exchange,
        decode=decode,
        destinations=outputs,
        device=torch.device("cpu"),
    )

    assert decoded == [key for key in order if owners[key] == rank]
    assert [(call[0], call[1]) for call in exchange.calls] == [("gather", owners[key]) for key in order]
    for key, (_, _, tensor, destination) in zip(order, exchange.calls):
        receives = gather_needs_buffer(owners[key], exchange.coord)
        assert (tensor is not None) == (receives or owners[key] == rank)
        assert (destination is not None) == (rank == 0)
    if outputs is not None:
        # Only the output rank's own item holds real data in this stand-in.
        assert torch.equal(outputs["2"], torch.full((1, 3, 2, 2, 2), 3.0))


events: list[tuple] = []


class FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name

    def wait_stream(self, other: FakeStream) -> None:
        events.append(("wait_stream", self.name, other.name))

    def wait_event(self, event: object) -> None:
        events.append(("wait_event", self.name))

    def synchronize(self) -> None:
        events.append(("synchronize", self.name))


class FakeTensor:
    def __init__(self, name: str) -> None:
        self.name = name

    def record_stream(self, stream: FakeStream) -> None:
        events.append(("record_stream", self.name, stream.name))


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch):
    events.clear()
    current = FakeStream("compute")
    active = [current]

    class Event:
        def record(self, stream: FakeStream) -> None:
            events.append(("record_event", stream.name))

    class StreamContext:
        def __init__(self, stream: FakeStream) -> None:
            self.stream = stream

        def __enter__(self) -> None:
            active.append(self.stream)

        def __exit__(self, *exc: object) -> None:
            active.pop()

    platform = SimpleNamespace(
        Stream=lambda device: FakeStream("comm"),
        Event=Event,
        stream=StreamContext,
        current_stream=lambda device=None: active[-1],
    )

    def collective(op: str):
        def run(tensor, *, group, src=None, dst=None, async_op=False):
            events.append((op, group, src if dst is None else dst, tensor.name, active[-1].name))
            return SimpleNamespace(wait=lambda: events.append(("wait", active[-1].name)))

        return run

    def all_gather_object(output: list, obj: object, group: str) -> None:
        events.append(("all_gather_object", group))
        output[:] = [obj] * len(output)

    fake_dist = SimpleNamespace(
        broadcast=collective("broadcast"),
        isend=collective("isend"),
        irecv=collective("irecv"),
        all_gather_object=all_gather_object,
    )
    monkeypatch.setattr(vae_owner, "current_omni_platform", platform)
    monkeypatch.setattr(vae_owner, "dist", fake_dist)
    return events


def make_grid(sp_rank: int, sp_size: int, cfg_rank: int, cfg_size: int) -> VaeGrid:
    sp_ranks = [cfg_rank * sp_size + i for i in range(sp_size)]
    cfg_ranks = [i * sp_size + sp_rank for i in range(cfg_size)]
    sp = SimpleNamespace(world_size=sp_size, rank_in_group=sp_rank, ranks=sp_ranks, cpu_group="sp.cpu")
    sp.device_group = "sp.dev"
    cfg = SimpleNamespace(world_size=cfg_size, rank_in_group=cfg_rank, ranks=cfg_ranks, cpu_group="cfg.cpu")
    cfg.device_group = "cfg.dev"
    return VaeGrid(sp, cfg)  # type: ignore[arg-type]


def test_exchange_issues_steps_on_the_communication_stream(fake_runtime) -> None:
    # Grid rank 5 of a 2 (SP) x 4 (CFG) grid: SP rank 1, CFG rank 2.
    exchange = VaeOwnerExchange(torch.device("cpu"), make_grid(1, 2, 2, 4), ("plan",))
    exchange.broadcast_from_owner(FakeTensor("latent"), owner=4)
    exchange.gather_to_output(FakeTensor("video"), owner=5, destination=None)
    exchange.finish()

    assert fake_runtime == [
        ("all_gather_object", "sp.cpu"),
        ("all_gather_object", "cfg.cpu"),
        ("wait_stream", "comm", "compute"),
        # Owner 4 is SP rank 0 of CFG row 2: SP broadcast in that row, then CFG.
        ("record_event", "compute"),
        ("wait_event", "comm"),
        ("broadcast", "sp.dev", 4, "latent", "comm"),
        ("wait", "comm"),
        ("broadcast", "cfg.dev", 5, "latent", "comm"),
        ("wait", "comm"),
        ("record_stream", "latent", "comm"),
        # This rank owns the video: send to SP rank 0 of its row (grid rank 4).
        ("record_event", "compute"),
        ("wait_event", "comm"),
        ("isend", "sp.dev", 4, "video", "comm"),
        ("wait", "comm"),
        ("record_stream", "video", "comm"),
        ("wait_stream", "compute", "comm"),
        ("record_stream", "latent", "compute"),
        ("record_stream", "video", "compute"),
    ]


def test_exchange_rejects_disagreeing_metadata(fake_runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    def all_gather_object(output: list, obj: object, group: str) -> None:
        output[:] = [obj] * (len(output) - 1) + [("other",)]

    monkeypatch.setattr(vae_owner.dist, "all_gather_object", all_gather_object)
    with pytest.raises(ValueError, match="metadata differs"):
        VaeOwnerExchange(torch.device("cpu"), make_grid(0, 2, 0, 2), ("plan",))


def test_copy_to_host_handles_camera_slices() -> None:
    video = torch.zeros(1, 3, 6, 2, 2)
    view = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 2, 2)
    vae_owner._copy_to_host(video.narrow(2, 2, 2), view)
    assert torch.equal(video.narrow(2, 2, 2), view)
    assert video.narrow(2, 0, 2).eq(0).all() and video.narrow(2, 4, 2).eq(0).all()


@pytest.fixture
def owner_ready(monkeypatch: pytest.MonkeyPatch):
    """A CUDA BF16 environment with a 4 x 2 grid; each case breaks one condition."""

    def group(ranks: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            ranks=ranks, world_size=len(ranks), rank_in_group=0, cpu_group=object(), device_group=object()
        )

    state = SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        dtype=torch.bfloat16,
        vae=SimpleNamespace(dtype=torch.bfloat16, is_distributed_enabled=lambda: False),
        work_count=12,
        sp=group([0, 1, 2, 3]),
        cfg=group([0, 4]),
        is_cuda=True,
        dist_initialized=True,
        parallel_initialized=True,
        backend="nccl",
    )
    monkeypatch.setattr(vae_owner, "current_omni_platform", SimpleNamespace(is_cuda=lambda: state.is_cuda))
    monkeypatch.setattr(vae_owner.dist, "is_initialized", lambda: state.dist_initialized)
    monkeypatch.setattr(vae_owner.dist, "get_backend", lambda group: state.backend)
    monkeypatch.setattr(vae_owner, "model_parallel_is_initialized", lambda: state.parallel_initialized)
    monkeypatch.setattr(vae_owner, "get_sp_group", lambda: state.sp)
    monkeypatch.setattr(vae_owner, "get_cfg_group", lambda: state.cfg)
    state.group = group
    return state


def test_vae_grid_uses_sp_and_cfg_groups(owner_ready) -> None:
    grid = vae_grid(owner_ready.device, owner_ready.dtype, owner_ready.vae, owner_ready.work_count)
    assert grid is not None and grid.sp is owner_ready.sp and grid.cfg is owner_ready.cfg


@pytest.mark.parametrize(
    "change",
    [
        {"device": SimpleNamespace(type="cpu")},
        {"is_cuda": False},
        {"dtype": torch.float32},
        {"vae": SimpleNamespace(dtype=torch.float32, is_distributed_enabled=lambda: False)},
        {"vae": SimpleNamespace(dtype=torch.bfloat16, is_distributed_enabled=lambda: True)},
        {"work_count": 1},
        {"dist_initialized": False},
        {"parallel_initialized": False},
        {"backend": "gloo"},
    ],
)
def test_vae_grid_falls_back_to_serial(owner_ready, change: dict) -> None:
    for name, value in change.items():
        setattr(owner_ready, name, value)
    assert vae_grid(owner_ready.device, owner_ready.dtype, owner_ready.vae, owner_ready.work_count) is None


@pytest.mark.parametrize(
    "sp_ranks,cfg_ranks",
    [([0], [0]), ([0, 1], [0, 1]), ([0, 0], [0, 4]), ([0, 1], [4, 5])],
)
def test_vae_grid_rejects_invalid_axes(owner_ready, sp_ranks: list[int], cfg_ranks: list[int]) -> None:
    owner_ready.sp, owner_ready.cfg = owner_ready.group(sp_ranks), owner_ready.group(cfg_ranks)
    assert vae_grid(owner_ready.device, owner_ready.dtype, owner_ready.vae, owner_ready.work_count) is None


@pytest.mark.parametrize("sp_ranks,cfg_ranks", [([0], [0, 4]), ([0, 1], [0])])
def test_vae_grid_accepts_single_axis_grids(owner_ready, sp_ranks: list[int], cfg_ranks: list[int]) -> None:
    owner_ready.sp, owner_ready.cfg = owner_ready.group(sp_ranks), owner_ready.group(cfg_ranks)
    assert vae_grid(owner_ready.device, owner_ready.dtype, owner_ready.vae, owner_ready.work_count) is not None
