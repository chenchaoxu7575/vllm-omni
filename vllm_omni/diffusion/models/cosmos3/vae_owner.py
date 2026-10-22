# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Geometry-based replica scheduling and asynchronous VAE transport."""

from dataclasses import dataclass

# Seconds per input pixel-frame. Calibration: H100 round3 VAE phase measurements
# (201x480x832 RGB; 67x128x1808 LiDAR), including camera CPU preparation.
# These are scheduling estimates, never timing predictions or correctness gates.
VAE_COST_PER_PIXEL = {
    "camera": (6.28 / (201 * 480 * 832), 3.50 / (201 * 480 * 832)),
    "lidar": (4.55 / (67 * 128 * 1808), 4.59 / (67 * 128 * 1808)),
}


@dataclass(frozen=True)
class VaeWork:
    key: str
    kind: str
    frames: int
    height: int
    width: int
    encode_fraction: float = 1.0
    decode: bool = True

    @property
    def cost(self):
        encode, decode = VAE_COST_PER_PIXEL[self.kind]
        return self.frames * self.height * self.width * (encode * self.encode_fraction + decode * self.decode)


def plan_vae_work(items, sp_size, cfg_size):
    """LPT placement; ties minimize output hops, then replica rank.

    Every rank uses the same ordered request metadata. Each sensor remains on
    one owner for all its encodes and its decode. The returned order is also
    the transport order, including on ranks with no local work.
    """
    if sp_size < 1 or cfg_size < 1 or len({x.key for x in items}) != len(items):
        raise ValueError("Invalid VAE replica geometry or duplicate work keys.")
    if any(min(x.frames, x.height, x.width) < 1 or x.encode_fraction < 0 for x in items):
        raise ValueError("VAE work requires positive geometry and nonnegative cost.")
    ordered = sorted(items, key=lambda x: (-x.cost, x.key))
    loads = [0.0] * (sp_size * cfg_size)
    owners = {}
    for item in ordered:
        owner = min(range(len(loads)), key=lambda r: (loads[r], int(r % sp_size != 0) + int(r // sp_size != 0), r))
        owners[item.key] = owner
        loads[owner] += item.cost
    return owners, [x.key for x in ordered]


class VaeOwnerExchange:
    """Use existing SP/CFG groups, never WORLD (DP/TP stay isolated).

    All ranks visit the identical item list. A broadcast travels SP then CFG;
    a gather travels to the owner's SP0 then CFG0. Work.wait only inserts a
    dependency on the communication stream. The compute stream waits once,
    after all local producers have been submitted. Keep tensors and Work
    handles alive until completion; record_stream protects allocator reuse.
    """

    def __init__(self, device, sp, cfg, metadata):
        import torch
        import torch.distributed as dist

        self.sp, self.cfg = sp, cfg
        self.rank = cfg.rank_in_group * sp.world_size + sp.rank_in_group
        reports = [None] * sp.world_size
        dist.all_gather_object(reports, metadata, group=sp.cpu_group)
        groups = [None] * cfg.world_size
        dist.all_gather_object(groups, reports, group=cfg.cpu_group)
        if any(x != metadata for group in groups for x in group):
            raise ValueError("VAE owner work metadata differs between replica ranks.")
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        self.pending = []
        self.buffers = []

    def transfer(self, tensor, owner, *, gather=False, destination=None):
        import torch
        import torch.distributed as dist

        sp, cfg = self.sp, self.cfg
        owner_cfg, owner_sp = divmod(owner, sp.world_size)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)

            def launch(work):
                self.pending.append(work)
                work.wait()

            if not gather:
                if cfg.rank_in_group == owner_cfg and sp.world_size > 1:
                    launch(dist.broadcast(tensor, src=sp.ranks[owner_sp], group=sp.device_group, async_op=True))
                if cfg.world_size > 1:
                    launch(dist.broadcast(tensor, src=cfg.ranks[owner_cfg], group=cfg.device_group, async_op=True))
            else:
                if cfg.rank_in_group == owner_cfg and owner_sp != 0:
                    if sp.rank_in_group == owner_sp:
                        launch(dist.isend(tensor, dst=sp.ranks[0], group=sp.device_group))
                    elif sp.rank_in_group == 0:
                        launch(dist.irecv(tensor, src=sp.ranks[owner_sp], group=sp.device_group))
                if sp.rank_in_group == 0 and owner_cfg != 0:
                    if cfg.rank_in_group == owner_cfg:
                        launch(dist.isend(tensor, dst=cfg.ranks[0], group=cfg.device_group))
                    elif cfg.rank_in_group == 0:
                        launch(dist.irecv(tensor, src=cfg.ranks[owner_cfg], group=cfg.device_group))
                if self.rank == 0:
                    if destination.is_contiguous():
                        destination.copy_(tensor, non_blocking=True)
                    else:
                        # A camera slice of [B,C,V*F,H,W] is not contiguous.
                        # Copy contiguous pixel spans directly into pinned host
                        # storage: a strided D2H copy may need a temporary and
                        # host-side reordering, preventing asynchronous delivery.
                        for batch in range(destination.shape[0]):
                            for channel in range(destination.shape[1]):
                                destination[batch, channel].copy_(tensor[batch, channel], non_blocking=True)
            if tensor is not None:
                tensor.record_stream(self.stream)
                self.buffers.append(tensor)

    def finish(self, *, cpu=False):
        import torch

        if cpu:
            self.stream.synchronize()  # CPU output is consumed immediately.
        else:
            torch.cuda.current_stream().wait_stream(self.stream)
            for tensor in self.buffers:
                tensor.record_stream(torch.cuda.current_stream())


def replica_geometry_supported(sp, cfg):
    """Require known, independent SP/CFG replica groups with valid rank maps."""
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
    # Both axes must refer to this same world rank; they share no other rank.
    return (
        sp.world_size * cfg.world_size > 1
        and sp.ranks[sp.rank_in_group] == cfg.ranks[cfg.rank_in_group]
        and len(set(sp.ranks) & set(cfg.ranks)) == 1
    )


def vae_replica_groups(device, dtype, vae, work_count):
    """Automatic owner scheduling; return None to use the original serial path.

    Only independent NVIDIA CUDA BF16 replicas share work. Unknown/uninitialized
    group topology, one replica, one work item and distributed VAE implementations
    use the pipeline's existing preparation, encode and decode behavior.
    """
    import torch
    import torch.distributed as dist

    from vllm_omni.platforms import current_omni_platform

    if (
        device.type != "cuda"
        or not current_omni_platform.is_cuda()
        or dtype != torch.bfloat16
        or getattr(vae, "dtype", None) != torch.bfloat16
        or work_count < 2
        or not dist.is_initialized()
    ):
        return None
    distributed = getattr(vae, "is_distributed_enabled", None)
    if callable(distributed) and distributed():
        return None
    from vllm_omni.diffusion.distributed.parallel_state import get_cfg_group, get_sp_group

    try:
        sp, cfg = get_sp_group(), get_cfg_group()
    except AssertionError:
        # Existing group accessors assert when model parallelism is not set up.
        return None
    if not replica_geometry_supported(sp, cfg):
        return None
    if any(dist.get_backend(g.device_group) != "nccl" for g in (sp, cfg) if g.world_size > 1):
        return None
    return sp, cfg
