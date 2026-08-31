# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Preprocessing for the π0.5 VLA model.

Converts a raw robot observation (multi-camera images + language instruction +
proprioceptive state) into the tensors that ``Pi05ForActionPrediction.sample_actions``
consumes. Shaped like ``processor_pi0`` — a set of stateless helpers the
diffusion pipeline calls directly (the DreamZero contract: the pipeline owns its
preprocessing).

**The defining π0.5 difference:** state is not projected by a ``state_proj``
layer. It is normalized to ``[-1, 1]``, discretized into ``state_num_bins``
bins, and serialized into the language prompt::

    "Task: <instruction>, State: <b0> <b1> ... <bN>;\nAction: "

so ``sample_actions`` receives no state tensor at all.

Functional spec — LeRobot ``make_pi05_pre_post_processors``. Input side, 7 steps,
order is load-bearing:

===  =========================================  ==========================================
  #  LeRobot step                               here
===  =========================================  ==========================================
  1  ``rename_observations``                    ``_extract_images`` via ``image_key_map``
  2  ``add_batch_dim``                          tensors built with a leading batch dim of 1
  3  ``RelativeActionsProcessorStep``           :class:`Pi05RelativeActions` (input side)
  4  ``normalize``                              :func:`normalize_state`
  5  ``Pi05PrepareStateTokenizerProcessorStep`` :func:`discretize_state` + :func:`build_pi05_prompt`
  6  ``TokenizerProcessorStep``                 :func:`tokenize_prompt`
  7  ``to_device``                              ``device=`` on every tensor built below
===  =========================================  ==========================================

Output side, 3 steps: ``unnormalize`` (model-side ``_unnormalize_actions``) →
:meth:`Pi05RelativeActions.to_absolute` → ``to_cpu``.

Two ordering constraints that fail *silently* if broken:

* **Step 4 must precede step 5.** The discretizer bins over ``[-1, 1]`` and
  assumes the state is already normalized into that range. Reversed, every bin
  index is wrong and nothing raises.
* **Step 3 uses the raw state, before step 4.** Relative actions live in the raw
  action space (``relative = action - state``); the output side adds the same
  raw state back after unnormalization.

Reference:
  - OpenPI: openpi/src/openpi/shared/image_tools.py (resize_with_pad)
  - OpenPI: openpi/src/openpi/models/pi0_config.py (PaliGemmaTokenizer.tokenize)
  - LeRobot: lerobot/src/lerobot/policies/pi05/processor_pi05.py
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vllm_omni.diffusion.models.pi05.config import resolve_excluded_action_indices

logger = logging.getLogger(__name__)

# Defaults straight from the π0.5 reference configs.
PI05_IMAGE_SIZE = 224  # openpi/models/model.py IMAGE_RESOLUTION
PI05_NUM_IMAGE_TOKENS = 256  # SigLIP So400m/14 on 224×224 → (224/14)**2
PI05_IMAGE_TOKEN_INDEX = 257152  # openpi/models_pytorch/gemma_pytorch.py
PI05_MAX_CAMERAS = 3  # openpi/models/model.py (3 camera slots)
PI05_MAX_TOKEN_LEN = 200  # openpi/models/pi0_config.py — π0 uses 48
PI05_NUM_BINS = 256  # openpi PaliGemmaTokenizer.tokenize()


# ──────────────────────────────────────────────────────────────────────
# Image preprocessing (identical to π0 — SigLIP is unchanged in π0.5)
# ──────────────────────────────────────────────────────────────────────
def resize_with_pad(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize ``(B, C, H, W)`` images to the target shape, preserving aspect
    ratio with -1 padding on the short side.

    Matches openpi ``image_tools.resize_with_pad_torch`` — the clamp to
    [-1, 1] is what lets the padded region blend with SigLIP-normalized
    pixels without adding signal at the boundary.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4-D (B,C,H,W), got {images.ndim}-D")
    _, _, cur_h, cur_w = images.shape
    ratio = max(cur_w / target_width, cur_h / target_height)
    rh, rw = int(cur_h / ratio), int(cur_w / ratio)
    align_corners = False if mode == "bilinear" else None
    resized = F.interpolate(images, size=(rh, rw), mode=mode, align_corners=align_corners)
    resized = resized.clamp(-1.0, 1.0)
    ph, rem_h = divmod(target_height - rh, 2)
    pw, rem_w = divmod(target_width - rw, 2)
    return F.pad(resized, (pw, pw + rem_w, ph, ph + rem_h), value=-1.0)


def pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL → ``(1, C, H, W)`` float32 in ``[-1, 1]`` (SigLIP normalization)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class Pi05ImageProcessor:
    """Minimal image preprocessor: image → normalized + padded ``[-1,1]`` tensor."""

    def __init__(self, image_size: int = PI05_IMAGE_SIZE):
        self.image_size = image_size

    def preprocess_single(self, image: Any) -> torch.Tensor:
        """Accept a PIL image, an HWC uint8/float ndarray, or a CHW tensor and
        return a ``(1, 3, image_size, image_size)`` float tensor in ``[-1, 1]``."""
        t = self._to_tensor(image)
        if t.shape[2] != self.image_size or t.shape[3] != self.image_size:
            t = resize_with_pad(t, self.image_size, self.image_size)
        return t

    def _to_tensor(self, image: Any) -> torch.Tensor:
        if isinstance(image, Image.Image):
            return pil_image_to_tensor(image)
        if isinstance(image, np.ndarray):
            arr = image
            # HWC → normalize uint8/[0,255] to [-1,1]; assume already [-1,1] if float.
            if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                if np.issubdtype(arr.dtype, np.integer) or arr.max() > 1.0:
                    arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
                else:
                    arr = arr.astype(np.float32)
                return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return torch.as_tensor(arr, dtype=torch.float32)
        if isinstance(image, torch.Tensor):
            t = image
            if t.ndim == 3:  # (C, H, W) → (1, C, H, W)
                t = t.unsqueeze(0)
            return t.to(dtype=torch.float32)
        raise TypeError(f"Unsupported image type for π0.5 preprocessing: {type(image)}")

    def make_empty_image(self) -> torch.Tensor:
        """Fill tensor for an unused camera slot — pure -1, matches OpenPI/LeRobot."""
        return torch.full((1, 3, self.image_size, self.image_size), -1.0)


# ──────────────────────────────────────────────────────────────────────
# State: pad → normalize → discretize → prompt  (π0.5's defining path)
# ──────────────────────────────────────────────────────────────────────
def pad_or_truncate_state(raw_state: Any, max_state_dim: int) -> np.ndarray:
    """Zero-pad / truncate the raw state to ``(max_state_dim,)`` float32."""
    if raw_state is None:
        return np.zeros((max_state_dim,), dtype=np.float32)
    if isinstance(raw_state, torch.Tensor):
        raw_state = raw_state.detach().cpu().numpy()
    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if state.shape[0] < max_state_dim:
        state = np.pad(state, (0, max_state_dim - state.shape[0]))
    elif state.shape[0] > max_state_dim:
        state = state[:max_state_dim]
    return state.astype(np.float32)


def _stat_vector(value: Any, max_state_dim: int, fill: float) -> np.ndarray:
    if value is None:
        return np.full((max_state_dim,), fill, dtype=np.float32)
    return pad_or_truncate_state(value, max_state_dim)


def _infer_norm_mode(stats: dict[str, Any]) -> str | None:
    """Infer the normalization mode from which stat keys are present.

    LeRobot's π0.5 defaults ``STATE``/``ACTION`` to ``NormalizationMode.QUANTILES``
    (π0 uses ``MEAN_STD``), so a π0.5 checkpoint typically carries ``q01``/``q99``
    rather than ``mean``/``std``.
    """
    mode = stats.get("mode")
    if mode is not None:
        return str(mode).lower()
    if "mean" in stats and "std" in stats:
        return "mean_std"
    if "min" in stats and "max" in stats:
        return "min_max"
    if "q01" in stats and "q99" in stats:
        return "quantile"
    if "low" in stats and "high" in stats:
        return "quantile"
    return None


def normalize_state(
    raw_state: Any,
    *,
    max_state_dim: int,
    state_norm_stats: dict[str, Any] | None,
) -> np.ndarray:
    """Normalize the state to the ``[-1, 1]`` scale ready for discretization.

    **Deliberately does not clip.** LeRobot's ``NormalizeProcessor`` applies the
    affine map and nothing else (``normalize_processor.py``: ``2.0 * (tensor -
    q01) / denom - 1.0``), and returns the tensor unchanged when stats are
    missing. A state outside the quantile range therefore leaves this function
    outside ``[-1, 1]`` and lands in :func:`discretize_state`'s ``-1`` bin —
    which is what the checkpoint was trained with. Clamping here would silently
    rewrite those dimensions to bin ``0`` and change the prompt the model sees;
    LeRobot parity catches it.

    With no stats the state passes through untouched: a client that already
    normalizes its own state is a supported deployment mode.
    """
    state = pad_or_truncate_state(raw_state, max_state_dim)
    if not state_norm_stats:
        return state

    stats = state_norm_stats
    mode = _infer_norm_mode(stats)

    if mode == "mean_std":
        mean = _stat_vector(stats.get("mean"), max_state_dim, 0.0)
        std = _stat_vector(stats.get("std"), max_state_dim, 1.0)
        std = np.where(np.abs(std) < 1e-6, 1.0, std)
        return (state - mean) / std

    if mode == "min_max":
        vmin = _stat_vector(stats.get("min"), max_state_dim, -1.0)
        vmax = _stat_vector(stats.get("max"), max_state_dim, 1.0)
        denom = np.where(np.abs(vmax - vmin) < 1e-6, 1.0, vmax - vmin)
        return 2.0 * (state - vmin) / denom - 1.0

    if mode == "quantile":
        low_key = "q01" if "q01" in stats else "low"
        high_key = "q99" if "q99" in stats else "high"
        low = _stat_vector(stats.get(low_key), max_state_dim, -1.0)
        high = _stat_vector(stats.get(high_key), max_state_dim, 1.0)
        denom = np.where(np.abs(high - low) < 1e-6, 1.0, high - low)
        return 2.0 * (state - low) / denom - 1.0

    raise ValueError(
        f"Unsupported π0.5 state_norm_stats mode: {mode!r}. Expected one of mean_std / min_max / quantile."
    )


def discretize_state(state: np.ndarray, *, num_bins: int = PI05_NUM_BINS) -> np.ndarray:
    """Discretize a ``[-1, 1]`` state into ``num_bins`` integer bins.

    Byte-for-byte LeRobot's ``processor_pi05.py``::

        np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

    A state below ``-1`` lands in bin ``-1``, and that negative bin is part of
    the contract: the checkpoint was trained with ``" -1"`` in the state prompt
    for those dimensions. Clipping it to ``0`` changes the tokens the model
    sees, so this must not clip — parity caught exactly that (7 of 32 state
    dims differed, every one of them ``-1`` vs ``0``).

    ``bins`` stays float64, matching LeRobot's default ``linspace`` dtype, so
    boundary values fall on the same side.

    **Assumes the state is already normalized** — see the module docstring.
    """
    bins = np.linspace(-1.0, 1.0, num_bins + 1)[:-1]
    return (np.digitize(np.asarray(state, dtype=np.float32), bins=bins) - 1).astype(np.int64)


def build_pi05_prompt(
    *,
    task: str,
    state: Any,
    max_state_dim: int,
    state_norm_stats: dict[str, Any] | None,
    state_num_bins: int = PI05_NUM_BINS,
) -> str:
    """Build the π0.5 prompt: instruction + serialized discretized state.

    Matches LeRobot's ``Pi05PrepareStateTokenizerProcessorStep``, including the
    task cleanup (``strip``, ``_`` → space, newline → space) and the exact
    template. The template already ends in a newline, so — unlike π0 — there is
    no separate newline-appending step.
    """
    cleaned_task = (task or "").strip().replace("_", " ").replace("\n", " ")
    normed = normalize_state(state, max_state_dim=max_state_dim, state_norm_stats=state_norm_stats)
    bins = discretize_state(normed, num_bins=state_num_bins)
    state_str = " ".join(str(int(x)) for x in bins.tolist())
    return f"Task: {cleaned_task}, State: {state_str};\nAction: "


def tokenize_prompt(tokenizer, text: str, max_token_len: int = PI05_MAX_TOKEN_LEN):
    """Return ``(input_ids, attention_mask)`` lists, length exactly ``max_token_len``.

    ``padding="max_length"`` is what makes the prefix a constant shape: the text
    segment is always ``max_token_len`` tokens regardless of the instruction, so
    only ``attention_mask.sum()`` varies per request.
    """
    enc = tokenizer(
        text,
        padding="max_length",
        max_length=max_token_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors=None,
    )
    return list(enc["input_ids"]), list(enc["attention_mask"])


# ──────────────────────────────────────────────────────────────────────
# Relative actions
# ──────────────────────────────────────────────────────────────────────
class Pi05RelativeActions:
    """The relative/absolute action transform, as a single paired object.

    LeRobot builds one ``RelativeActionsProcessorStep`` and hands *the same
    instance* to ``AbsoluteActionsProcessorStep``; the two directions must
    agree on ``enabled`` and on which dimensions are excluded, so they are one
    object here too.

    **Deviation from LeRobot, on purpose.** LeRobot's step keeps the reference
    state on ``self`` between the pre- and post-pass. That is safe for a
    single-threaded training loop and unsafe for a server: two in-flight
    requests would share one reference state and silently corrupt each other's
    actions. Here the state is passed explicitly to :meth:`to_absolute`, so the
    object stays immutable after construction and is safe to share across
    requests.

    Transform (LeRobot / OpenPI): ``relative = action - state`` on the way in,
    ``absolute = relative + state`` on the way out, applied only to the
    dimensions *not* named in ``exclude_joints``. Gripper open/close is an
    absolute command, which is why it is excluded by default.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        exclude_joints: list[str] | None = None,
        action_names: list[str] | None = None,
        max_action_dim: int = 32,
    ):
        self.enabled = bool(enabled)
        self.exclude_joints = list(exclude_joints or [])
        self.action_names = list(action_names) if action_names else None
        self.max_action_dim = int(max_action_dim)

        # Boolean mask over action dims: True = this dim is relative to state.
        mask = np.ones((self.max_action_dim,), dtype=bool)
        if self.enabled:
            for idx in resolve_excluded_action_indices(self.exclude_joints, self.action_names):
                if 0 <= idx < self.max_action_dim:
                    mask[idx] = False
            # Padding beyond the real action dimensions carries no signal;
            # leaving it "relative" would add state noise into dead channels.
            if self.action_names:
                mask[len(self.action_names) :] = False
        else:
            mask[:] = False
        self.relative_mask = mask

    @property
    def num_relative_dims(self) -> int:
        return int(self.relative_mask.sum())

    def _state_row(self, state: Any, device, dtype) -> torch.Tensor:
        """Raw state → ``(B, max_action_dim)`` aligned with the action dims.

        Accepts one state for the whole batch (``(D,)``, the serving path,
        where the pipeline runs B=1) or one state per sample (``(B, D)``).
        LeRobot caches the batched state and shifts each sample by its own row,
        so the per-sample form has to be honoured: ``pad_or_truncate_state``
        flattens, which would silently reduce ``(B, D)`` to sample 0's state and
        apply it to every sample — wrong answers, no error.
        """
        if isinstance(state, torch.Tensor):
            arr = state.detach().cpu().numpy()
        else:
            arr = state
        arr = np.asarray(arr if arr is not None else 0.0, dtype=np.float32)
        if arr.ndim >= 2:
            if arr.ndim > 2:
                raise ValueError(f"Expected state shaped (D,) or (B, D), got {tuple(arr.shape)}")
            rows = np.stack([pad_or_truncate_state(row, self.max_action_dim) for row in arr])
            return torch.as_tensor(rows, device=device, dtype=dtype)
        padded = pad_or_truncate_state(state, self.max_action_dim)
        return torch.as_tensor(padded, device=device, dtype=dtype)[None, :]

    def to_relative(self, actions: torch.Tensor, state: Any) -> torch.Tensor:
        """``absolute → relative``. Input side (step 3).

        Not used on the inference path — there are no input actions to convert
        at serving time — but it is what the transform *means*, and the parity
        test exercises it as the inverse of :meth:`to_absolute`.
        """
        if not self.enabled:
            return actions
        return self._shift(actions, state, sign=-1.0)

    def to_absolute(self, actions: torch.Tensor, state: Any) -> torch.Tensor:
        """``relative → absolute``. Output side (step 2 of the post-pipeline).

        ``state`` must be the **raw** state — the same one the model was given
        before normalization — because relative actions live in raw action space.
        """
        if not self.enabled:
            return actions
        return self._shift(actions, state, sign=+1.0)

    def _shift(self, actions: torch.Tensor, state: Any, *, sign: float) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions shaped (B, horizon, action_dim), got {tuple(actions.shape)}")
        action_dim = actions.shape[-1]
        if action_dim != self.max_action_dim:
            raise ValueError(
                f"Action dim {action_dim} does not match max_action_dim={self.max_action_dim}; "
                "the relative mask would be misaligned."
            )
        state_row = self._state_row(state, actions.device, actions.dtype)  # (1, D)
        mask = torch.as_tensor(self.relative_mask, device=actions.device)
        delta = torch.where(mask, state_row, torch.zeros_like(state_row))
        # Broadcast over the action horizon: every step of the chunk is
        # expressed relative to the same current state.
        return actions + sign * delta[:, None, :]


# ──────────────────────────────────────────────────────────────────────
# Model input assembly
# ──────────────────────────────────────────────────────────────────────
def _extract_images(robot_obs: dict, config) -> dict[str, Any]:
    """Pull a ``{feature_key: image}`` map out of a raw robot obs.

    This is functional step 1 (``rename_observations``): keys are translated
    through ``config.image_key_map`` so serving wire names map onto the
    checkpoint's ``input_features`` identities.
    """
    images = robot_obs.get("images")
    if not isinstance(images, dict):
        images = {k: v for k, v in robot_obs.items() if _is_image_like(v)}
    key_map = getattr(config, "image_key_map", None) or {}
    return {key_map.get(k, k): v for k, v in images.items() if _is_image_like(v)}


def _is_image_like(value: Any) -> bool:
    if isinstance(value, (Image.Image, torch.Tensor)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim >= 3
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value).ndim >= 3
        except Exception:  # noqa: BLE001
            return False
    return False


def build_model_inputs(robot_obs: dict, config, tokenizer, device: torch.device):
    """Convert a raw robot observation into ``sample_actions`` inputs.

    Returns ``(images, image_masks, lang_tokens, lang_masks)`` — note there is
    **no state tensor**: π0.5 carries the state inside ``lang_tokens``.

    Camera ordering follows ``config.image_feature_keys`` exactly (the ordered
    identities from the checkpoint's ``input_features``); for each, use the
    supplied image (mask ``True``) or a ``-1``-filled empty image (mask ``False``).
    """
    image_size = int(config.image_resolution[0])
    img_proc = Pi05ImageProcessor(image_size=image_size)
    max_cameras = max(1, int(getattr(config, "max_cameras", PI05_MAX_CAMERAS)))

    feature_keys = config.image_feature_keys or []
    obs_images = _extract_images(robot_obs, config)
    if not feature_keys:
        # No declared camera order — fall back to whatever the obs provides,
        # capped at max_cameras (preserves insertion order).
        feature_keys = list(obs_images.keys())[:max_cameras]

    images: list[torch.Tensor] = []
    image_masks: list[torch.Tensor] = []
    for key in feature_keys[:max_cameras]:
        img = obs_images.get(key)
        if img is not None:
            tensor = img_proc.preprocess_single(img).to(device=device)
            mask = True
        else:
            tensor = img_proc.make_empty_image().to(device=device)
            mask = False
        images.append(tensor)
        image_masks.append(torch.tensor([mask], dtype=torch.bool, device=device))

    # Pad up to max_cameras with empty slots if the checkpoint declares fewer
    # cameras than the model attends to.
    while len(images) < max_cameras:
        images.append(img_proc.make_empty_image().to(device=device))
        image_masks.append(torch.tensor([False], dtype=torch.bool, device=device))

    # Steps 4 + 5 + 6: normalize → discretize → prompt → tokenize.
    prompt = build_pi05_prompt(
        task=robot_obs.get("prompt", "") or "",
        state=robot_obs.get("state"),
        max_state_dim=config.max_state_dim,
        state_norm_stats=getattr(config, "state_norm_stats", None),
        state_num_bins=getattr(config, "state_num_bins", PI05_NUM_BINS),
    )
    ids, attn = tokenize_prompt(tokenizer, prompt, config.tokenizer_max_length)
    lang_tokens = torch.tensor([ids], dtype=torch.long, device=device)
    lang_masks = torch.tensor([attn], dtype=torch.bool, device=device)

    return images, image_masks, lang_tokens, lang_masks


def prefix_token_budget(config, num_real_cameras: int) -> dict[str, int]:
    """Report the prefix token layout for a request — the A4/C1 input contract.

    ``1..3 views × 256 image tokens + a constant 200 text tokens``, so the
    tensor shape is fixed at ``256 * max_cameras + tokenizer_max_length`` and
    only ``valid_prefix_len`` varies per request. Exposed for tests and for
    latency accounting.
    """
    max_cameras = max(1, int(getattr(config, "max_cameras", PI05_MAX_CAMERAS)))
    text_len = int(config.tokenizer_max_length)
    return {
        "image_tokens": PI05_NUM_IMAGE_TOKENS * max_cameras,
        "valid_image_tokens": PI05_NUM_IMAGE_TOKENS * min(num_real_cameras, max_cameras),
        "text_tokens": text_len,
        "total_prefix_len": PI05_NUM_IMAGE_TOKENS * max_cameras + text_len,
    }
