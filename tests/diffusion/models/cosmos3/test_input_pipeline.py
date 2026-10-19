# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import threading

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.input_pipeline import FramePreparation, ViewPrefetch, prepare_workers
from vllm_omni.diffusion.models.cosmos3.transfer import media_to_uint8_cthw

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_thread_budget(monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda _: set(range(224)))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    assert prepare_workers(local_ranks=8) == 2
    assert prepare_workers(100, local_ranks=8) == 2
    assert prepare_workers(1, local_ranks=8) == 1
    monkeypatch.setattr("os.sched_getaffinity", lambda _: set(range(8)))
    assert prepare_workers(local_ranks=8) == 0


def test_prefetch_bounded_and_ordered():
    calls = []
    event = threading.Event()

    def prepare(key):
        calls.append(key)
        event.set()
        return key

    with ViewPrefetch(["a", "b", "c"], prepare) as pre:
        assert event.wait(5)
        assert calls == ["a"]
        assert pre.take("a") == "a"
        assert pre.take("b") == "b"
        assert pre.take("c") == "c"
    assert calls == ["a", "b", "c"]
    with ViewPrefetch([], prepare):
        pass


def test_prefetch_propagates_exception_and_joins():
    def prepare(_):
        raise ValueError("decode error")

    with pytest.raises(ValueError, match="decode error"):
        with ViewPrefetch(["a"], prepare) as pre:
            pre.take("a")
    assert not any(t.name.startswith("vae-prefetch") for t in threading.enumerate())


@pytest.mark.parametrize("frames,h,w", [(1, 19, 31), (7, 41, 17), (13, 25, 38)])
@pytest.mark.parametrize("first", [False, True])
def test_payload_rounding_and_padding(frames, h, w, first):
    source = torch.arange(3 * frames * h * w).remainder(256).to(torch.uint8).reshape(3, frames, h, w)
    want = media_to_uint8_cthw(source, height=23, width=37, max_frames=1 if first else 11)
    want = want.to(torch.bfloat16).div(127.5).sub(1.0)
    if want.shape[1] < 11:
        want = torch.cat([want, want[:, -1:].expand(-1, 11 - want.shape[1], -1, -1)], dim=1)
    with FramePreparation(2, 4096) as fp:
        got = fp.pixels(
            {"control": source},
            field="control",
            height=23,
            width=37,
            num_frames=11,
            keep_first=first,
            require_complete=False,
            dtype=torch.bfloat16,
        )
    assert torch.equal(got[0].contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))
