#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""π0.5 LeRobot parity (in-process): the bit-for-bit correctness oracle.

Verifies that vllm-omni's ``Pi05ForActionPrediction`` produces matching action
chunks with LeRobot's ``PI05Policy`` when fed the same weights, the same
pre-processed inputs, and the same initial noise. π0.5 is a flow-matching model
(Euler-integrated ODE from t=1 → t=0 with a fixed ``num_steps``), so the output
is deterministic once the noise is fixed and ``torch.allclose`` on the final
action chunk is a valid oracle.

Three things this must cover that the π0 parity test does not:

1. **The discretized-state prompt.** π0.5's state reaches the model as language
   tokens, so a prompt mismatch shows up as an action mismatch — the tokens
   themselves are compared explicitly (``test_pi05_prompt_parity``).
2. **AdaRMS conditioning.** The action expert's per-layer modulation is the main
   new numerical path.
3. **Relative actions**, both ``use_relative_actions=False`` and ``True``.

Run in a SEPARATE ``lerobot[pi]`` venv (avoids dep conflict with the vllm-omni
env), with the vllm-omni ``pi05`` package importable::

    python -m pytest tests/diffusion/models/pi05/test_pi05_parity.py -v -s

Skipped automatically when LeRobot is not installed (e.g. the vllm-omni env).
CPU/float32 with fixed defaults; the only override is ``PI05_PARITY_MODEL_PATH``
(a local π0.5 checkpoint dir in LeRobot format, to skip the HF download).
"""

from __future__ import annotations

import copy
import importlib.util
import os

import pytest
import torch

# local_model: needs real weights + a lerobot venv, so it runs locally rather
# than in the ready-CI. Additionally gated on lerobot being importable.
_HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion]


# ─── Config (fixed; matches LeRobot defaults for a π0.5 checkpoint) ────
DEVICE = "cpu"
DTYPE_STR = "float32"
ATOL = 1e-4
NUM_STEPS = 10
BATCH_SIZE = 2
ACTION_DIM = 32
STATE_DIM = 32
ACTION_HORIZON = 50
MAX_TOKEN_LEN = 200  # π0 uses 48
NUM_STATE_BINS = 256

MODEL_PATH = os.environ.get("PI05_PARITY_MODEL_PATH", "lerobot/pi05_base")

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _resolve_checkpoint_dir() -> str:
    if os.path.isdir(MODEL_PATH):
        return MODEL_PATH
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_PATH, repo_type="model")


# ─── Dummy dataset stats ──────────────────────────────────────────────
def _dummy_dataset_stats() -> dict:
    """Identity-ish stats. π0.5 normalizes with QUANTILES by default, so q01/q99
    carry the load here where π0's mean/std would.

    Using dummy stats means this test needs no checkpoint with real
    ``norm_stats`` — it verifies numerical equivalence, not calibration.
    """
    return {
        "observation.state": {
            "mean": torch.zeros(STATE_DIM),
            "std": torch.ones(STATE_DIM),
            "q01": -torch.ones(STATE_DIM),
            "q99": torch.ones(STATE_DIM),
        },
        "action": {
            "mean": torch.zeros(ACTION_DIM),
            "std": torch.ones(ACTION_DIM),
            "q01": -torch.ones(ACTION_DIM),
            "q99": torch.ones(ACTION_DIM),
        },
        "images": {
            cam: {
                "mean": torch.zeros(3, 224, 224),
                "std": torch.ones(3, 224, 224),
                "q01": torch.zeros(3, 224, 224),
                "q99": torch.ones(3, 224, 224),
            }
            for cam in CAMERAS
        },
    }


def _create_dummy_batch(batch_size: int = BATCH_SIZE, num_views: int = 3, device: str = DEVICE) -> dict:
    """Reproducible dummy inputs — identical across both implementations."""
    g = torch.Generator(device="cpu").manual_seed(0)
    prompt = "Pick up the red block and place it in the bin"
    batch = {
        "observation.state": torch.randn(batch_size, STATE_DIM, generator=g, dtype=torch.float32).to(device),
        "action": torch.randn(batch_size, ACTION_HORIZON, ACTION_DIM, generator=g, dtype=torch.float32).to(device),
        "task": [prompt for _ in range(batch_size)],
    }
    for cam in CAMERAS[:num_views]:
        batch[f"observation.images.{cam}"] = torch.rand(batch_size, 3, 224, 224, generator=g, dtype=torch.float32).to(
            device
        )
    return batch


# ─── Instantiation ────────────────────────────────────────────────────
def _instantiate_lerobot(use_relative_actions: bool = False):
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    policy = PI05Policy.from_pretrained(MODEL_PATH, strict=True)
    policy.to(DEVICE)
    policy.config.device = DEVICE
    policy.config.use_relative_actions = use_relative_actions
    policy.eval()

    pre, post = make_pi05_pre_post_processors(config=policy.config, dataset_stats=_dummy_dataset_stats())
    return policy, pre, post


def _instantiate_vllm_omni(use_relative_actions: bool = False, action_feature_names=None):
    """Build the vllm-omni π0.5 model in isolation (no pipeline, no engine)."""
    from vllm_omni.diffusion.models.pi05 import Pi05Config, Pi05ForActionPrediction

    cfg = Pi05Config(
        max_action_dim=ACTION_DIM,
        max_state_dim=STATE_DIM,
        chunk_size=ACTION_HORIZON,
        num_inference_steps=NUM_STEPS,
        tokenizer_max_length=MAX_TOKEN_LEN,
        state_num_bins=NUM_STATE_BINS,
        dtype=DTYPE_STR,
        use_relative_actions=use_relative_actions,
        relative_exclude_joints=["gripper"] if action_feature_names else [],
        action_feature_names=action_feature_names,
    )
    model = Pi05ForActionPrediction(cfg)
    model.to(DEVICE).eval()
    _load_lerobot_weights(model)
    return model, cfg


def _load_lerobot_weights(model):
    import safetensors.torch

    path = os.path.join(_resolve_checkpoint_dir(), "model.safetensors")
    state = safetensors.torch.load_file(path)
    filled = model.load_weights(list(state.items()))
    assert filled, "no weights were loaded — the remap rules are broken"


def _extract_lerobot_model_inputs(lerobot_policy, processed_batch):
    """Mimic what ``PI05Policy.predict_action_chunk`` feeds into
    ``self.model.sample_actions``. Using these *exact* tensors for vllm-omni
    means any divergence must come from the core model, not preprocessing.

    Note π0.5 has no ``prepare_state`` step: the state is already inside
    ``lang_tokens``.
    """
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    images, img_masks = lerobot_policy._preprocess_images(processed_batch)
    return images, img_masks, processed_batch[OBS_LANGUAGE_TOKENS], processed_batch[OBS_LANGUAGE_ATTENTION_MASK]


def _make_fixed_noise(batch_size: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(42)
    return torch.randn(batch_size, ACTION_HORIZON, ACTION_DIM, generator=g, dtype=torch.float32).to(device)


# ─── Main parity test ─────────────────────────────────────────────────
@pytest.mark.skipif(not _HAS_LEROBOT, reason="lerobot not installed (run in a lerobot venv).")
@pytest.mark.parametrize("num_views", [1, 2, 3])
def test_pi05_vllm_omni_vs_lerobot(num_views):
    lerobot_policy, lerobot_pre, _ = _instantiate_lerobot()
    omni_model, _ = _instantiate_vllm_omni()

    raw_batch = _create_dummy_batch(num_views=num_views)
    processed = lerobot_pre(copy.deepcopy(raw_batch))
    images, img_masks, lang_tokens, lang_masks = _extract_lerobot_model_inputs(lerobot_policy, processed)
    noise = _make_fixed_noise(raw_batch["observation.state"].shape[0], DEVICE)

    # The input contract: 256 tokens per view + a constant 200 text tokens.
    assert lang_tokens.shape[1] == MAX_TOKEN_LEN, "π0.5 must pad text to exactly 200 tokens"

    with torch.no_grad():
        lerobot_actions = lerobot_policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, noise=noise, num_steps=NUM_STEPS
        )
        omni_actions = omni_model.sample_actions(
            images=images,
            image_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            noise=noise,
            num_steps=NUM_STEPS,
        )

    diff = (lerobot_actions.float() - omni_actions.float()).abs()
    print(f"[parity] views={num_views} |Δ| max={diff.max().item():.2e} mean={diff.mean().item():.2e}")
    if not torch.allclose(lerobot_actions.float(), omni_actions.float(), atol=ATOL):
        _diagnose_divergence(lerobot_policy.model, omni_model, images, img_masks, lang_tokens, lang_masks, noise)
    assert torch.allclose(lerobot_actions.float(), omni_actions.float(), atol=ATOL), (
        f"actions differ beyond atol={ATOL}; max_diff={diff.max().item():.2e}"
    )


@pytest.mark.skipif(not _HAS_LEROBOT, reason="lerobot not installed (run in a lerobot venv).")
def test_pi05_prompt_parity():
    """π0.5's state is carried as prompt tokens, so the tokenized prompt is
    itself part of the contract. A bin-index mismatch here (e.g. from running
    the discretizer before the normalizer) would otherwise only surface as a
    small, plausible-looking action difference.
    """
    from transformers import AutoTokenizer

    from vllm_omni.diffusion.models.pi05.processor_pi05 import build_pi05_prompt, tokenize_prompt

    lerobot_policy, lerobot_pre, _ = _instantiate_lerobot()
    raw_batch = _create_dummy_batch(batch_size=1)
    processed = lerobot_pre(copy.deepcopy(raw_batch))

    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

    lerobot_tokens = processed[OBS_LANGUAGE_TOKENS][0]

    stats = _dummy_dataset_stats()["observation.state"]
    prompt = build_pi05_prompt(
        task=raw_batch["task"][0],
        state=raw_batch["observation.state"][0],
        max_state_dim=STATE_DIM,
        state_norm_stats={"q01": stats["q01"], "q99": stats["q99"]},
        state_num_bins=NUM_STATE_BINS,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", padding_side="right")
    ids, _ = tokenize_prompt(tokenizer, prompt, MAX_TOKEN_LEN)

    assert len(ids) == MAX_TOKEN_LEN
    assert torch.equal(torch.tensor(ids), lerobot_tokens.cpu()), "π0.5 prompt tokens diverge from LeRobot"


@pytest.mark.skipif(not _HAS_LEROBOT, reason="lerobot not installed (run in a lerobot venv).")
def test_pi05_relative_actions_parity():
    """With ``use_relative_actions=True`` the checkpoint's norm_stats live in
    relative space, so the post-pipeline must add the raw state back. Compares
    the full pre → model → post path against LeRobot's."""
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    from vllm_omni.diffusion.models.pi05.processor_pi05 import Pi05RelativeActions

    action_names = [f"joint_{i}" for i in range(ACTION_DIM - 1)] + ["gripper"]
    lerobot_policy, lerobot_pre, lerobot_post = _instantiate_lerobot(use_relative_actions=True)
    lerobot_policy.config.action_feature_names = action_names
    omni_model, cfg = _instantiate_vllm_omni(use_relative_actions=True, action_feature_names=action_names)

    raw_batch = _create_dummy_batch()
    processed = lerobot_pre(copy.deepcopy(raw_batch))
    images, img_masks = lerobot_policy._preprocess_images(processed)
    lang_tokens = processed[OBS_LANGUAGE_TOKENS]
    lang_masks = processed[OBS_LANGUAGE_ATTENTION_MASK]
    noise = _make_fixed_noise(raw_batch["observation.state"].shape[0], DEVICE)

    with torch.no_grad():
        lerobot_actions = lerobot_post(
            lerobot_policy.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, noise=noise, num_steps=NUM_STEPS
            )
        )
        omni_raw = omni_model.sample_actions(
            images=images,
            image_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            noise=noise,
            num_steps=NUM_STEPS,
        )

    rel = Pi05RelativeActions(
        enabled=True,
        exclude_joints=cfg.relative_exclude_joints,
        action_names=action_names,
        max_action_dim=ACTION_DIM,
    )
    omni_actions = omni_model._unnormalize_actions(omni_raw)
    omni_actions = rel.to_absolute(omni_actions, raw_batch["observation.state"][0])

    lerobot_actions = torch.as_tensor(lerobot_actions).float()
    assert torch.allclose(lerobot_actions, omni_actions.float(), atol=ATOL), (
        "relative-action path diverges from LeRobot; "
        f"max_diff={(lerobot_actions - omni_actions.float()).abs().max().item():.2e}"
    )


# ─── Per-stage divergence diagnostics ─────────────────────────────────
@torch.no_grad()
def _diagnose_divergence(lerobot_flow_model, omni_model, images, img_masks, lang_tokens, lang_masks, noise):
    """Localize a numerical mismatch to a specific stage:
    1. Prefix embeddings (SigLIP / embed_tokens / projector) — image vs lang.
    2. Prefix KV cache layer 0 — PaliGemma LM attention.
    3. The AdaRMS timestep conditioning vector.
    4. A single denoise_step velocity at t=1.0 — action expert forward.
    """
    from vllm_omni.diffusion.models.pi05.modeling_pi05 import (
        make_att_2d_masks,
        prepare_attention_masks_4d,
    )

    lr_embs, lr_pad, lr_att = lerobot_flow_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    sg_embs, sg_pad, sg_att = omni_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    print(f"[diag] prefix_embs max|Δ| = {(lr_embs.float() - sg_embs.float()).abs().max().item():.2e}")
    print(f"[diag] prefix_pad_masks equal: {torch.equal(lr_pad, sg_pad)}")

    img_len = 256 * len(images)
    print(f"[diag]   image slice max|Δ| = {(lr_embs[:, :img_len] - sg_embs[:, :img_len]).abs().max().item():.2e}")
    print(f"[diag]   lang  slice max|Δ| = {(lr_embs[:, img_len:] - sg_embs[:, img_len:]).abs().max().item():.2e}")

    prefix_att_4d = prepare_attention_masks_4d(make_att_2d_masks(sg_pad, sg_att))
    prefix_pos = torch.cumsum(sg_pad, dim=1) - 1
    _, sg_kv = omni_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_4d,
        position_ids=prefix_pos,
        past_key_values=None,
        inputs_embeds=[sg_embs, None],
        use_cache=True,
    )

    t = torch.ones(noise.shape[0], dtype=torch.float32)
    sg_cond = omni_model.embed_timestep(t)
    print(f"[diag] adarms cond: shape={tuple(sg_cond.shape)} mean={sg_cond.mean().item():+.4f}")

    sg_vt = omni_model.denoise_step(sg_pad, sg_kv, noise, t)
    print(f"[diag] denoise_step(t=1) v_t mean={sg_vt.mean().item():+.6f}")


if __name__ == "__main__":
    test_pi05_vllm_omni_vs_lerobot(3)
