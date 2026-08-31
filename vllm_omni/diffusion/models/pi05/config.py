# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config surface for the π0.5 VLA model in vllm-omni.

Deliberately shaped exactly like ``pi0/config.py``: a small dataclass that
consumes the raw LeRobot ``config.json`` (the field surface of
``lerobot.policies.pi05.PI05Config``) and keeps only the runtime-relevant
fields. Transformer dimensions are derived from ``paligemma_variant`` /
``action_expert_variant`` inside the model via ``get_gemma_config``.

What π0.5 adds on top of the π0 config surface:

* ``tokenizer_max_length = 200`` (π0 uses 48).
* ``state_num_bins`` — state is discretized into language tokens instead of
  going through a ``state_proj`` layer.
* ``use_relative_actions`` / ``relative_exclude_joints`` /
  ``action_feature_names`` — the relative-action contract. See
  ``processor_pi05.Pi05RelativeActions``.
* Quantile normalization stats. LeRobot's π0.5 defaults ``STATE`` and
  ``ACTION`` to ``NormalizationMode.QUANTILES`` where π0 uses ``MEAN_STD``.

**Checkpoint boundary rule.** A capability that the checkpoint *declares* but
this implementation does not *consume* must raise, not be silently dropped.
π0.5 checkpoints can declare MEM (short-horizon observation memory) and RTC
(real-time chunking); neither is supported here, and serving such a checkpoint
anyway would produce plausible-looking wrong actions. See
``_reject_unsupported_capabilities``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# LeRobot / OpenPI observation key conventions.
ACTION = "action"
OBS_STR = "observation"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGES = OBS_STR + ".images"

# π0.5 discretizes normalized state into this many bins before serializing it
# into the prompt. Ref: openpi ``PaliGemmaTokenizer.tokenize()``.
DEFAULT_STATE_NUM_BINS = 256


class UnsupportedCheckpointCapabilityError(ValueError):
    """A checkpoint declares a capability this implementation does not consume.

    Raised at load time rather than silently ignored: every one of these
    capabilities changes what a *correct* action chunk looks like, and none of
    them is visible in the weights alone.
    """


def resolve_excluded_action_indices(
    exclude_joints: list[str] | None,
    action_names: list[str] | None,
) -> list[int]:
    """Map ``relative_exclude_joints`` names onto action-vector indices.

    Matching is exact name first, then substring (a checkpoint may name the
    gripper dimension ``gripper_position`` while the config just says
    ``gripper``). Single source of truth for both the config-time validation
    and the runtime mask in ``processor_pi05.Pi05RelativeActions``.

    Returns an empty list when there is nothing to exclude. Raises when a name
    cannot be resolved — an unresolvable exclusion would otherwise silently
    become "make this dimension relative too".
    """
    if not exclude_joints:
        return []
    if not action_names:
        raise UnsupportedCheckpointCapabilityError(
            f"Cannot resolve relative_exclude_joints={exclude_joints!r} without action_feature_names."
        )

    indices: list[int] = []
    unresolved: list[str] = []
    for name in exclude_joints:
        exact = [i for i, candidate in enumerate(action_names) if candidate == name]
        hits = exact or [i for i, candidate in enumerate(action_names) if name in candidate]
        if not hits:
            unresolved.append(name)
        indices.extend(hits)

    if unresolved:
        raise UnsupportedCheckpointCapabilityError(
            f"relative_exclude_joints entries {unresolved!r} match no entry in action_feature_names={action_names!r}."
        )
    return sorted(set(indices))


@dataclass
class Pi05Config:
    """π0.5 VLA config (dataclass, not an HF ``PretrainedConfig``)."""

    # Backbone variants — mapped to Gemma dimensions by ``get_gemma_config``.
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # Action chunk shape.
    chunk_size: int = 50
    max_action_dim: int = 32
    max_state_dim: int = 32

    # Flow-matching denoising schedule.
    num_inference_steps: int = 10
    # Sinusoidal timestep embedding periods (must match OpenPI/LeRobot).
    min_period: float = 4e-3
    max_period: float = 4.0

    # Image preprocessing. π0.5/SigLIP only support square inputs.
    image_resolution: tuple[int, int] = (224, 224)
    # π0.5 pads text to 200 tokens (π0 uses 48). The prompt now also carries the
    # serialized state, which is why it is so much longer.
    tokenizer_max_length: int = 200
    # Number of camera slots the model attends to (real + padded).
    max_cameras: int = 3

    # π0.5-specific: number of bins the normalized state is discretized into.
    state_num_bins: int = DEFAULT_STATE_NUM_BINS

    # Weight dtype the checkpoint was saved in.
    dtype: str = "float32"

    # ── Relative actions ──────────────────────────────────────────────
    # When true, the model was trained on actions expressed relative to the
    # current state, and ``norm_stats`` holds *relative-space* statistics.
    # Serving such a checkpoint without the transform silently yields wrong
    # actions — the weights look identical either way.
    use_relative_actions: bool = False
    # Joint names kept absolute (gripper open/close is an absolute quantity).
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Ordered action feature names. LeRobot populates this at training time
    # from dataset metadata; it must therefore be present in the checkpoint's
    # config.json for ``relative_exclude_joints`` to be resolvable at serving
    # time (there is no dataset to fall back on).
    action_feature_names: list[str] | None = None

    # Per-dataset normalization stats (schema matches LeRobot's
    # ``NormalizerProcessorStep``). ``None`` means identity / pass-through.
    # π0.5 defaults to quantile mode; see ``_build_norm_buffers``.
    norm_stats: dict | None = None
    # Convenience view of ``norm_stats["state"]`` used by the prompt builder.
    state_norm_stats: dict | None = None

    # Ordered list of image feature keys, i.e. the **camera order** that must
    # be reproduced for LeRobot parity.
    image_feature_keys: list[str] | None = None
    # Optional map from raw OpenPI obs keys → ``image_feature_keys`` entries.
    image_key_map: dict[str, str] = field(default_factory=dict)

    # Stored for reference / camera-order derivation; not used at inference.
    input_features: dict[str, Any] = field(default_factory=dict)
    output_features: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce list → tuple (JSON has no tuples) and validate squareness.
        res = self.image_resolution
        if not isinstance(res, (tuple, list)) or len(res) != 2 or res[0] != res[1]:
            raise ValueError(f"π0.5 expects a square image_resolution (H == W); got {res!r}.")
        self.image_resolution = (int(res[0]), int(res[1]))

        if self.state_num_bins < 2:
            raise ValueError(f"state_num_bins must be >= 2, got {self.state_num_bins}.")

        # Derive the camera order from input_features if not given explicitly.
        if self.image_feature_keys is None and self.input_features:
            self.image_feature_keys = [key for key in self.input_features if key.startswith(OBS_IMAGES + ".")]

        # ``state_norm_stats`` is just a view onto norm_stats["state"].
        if self.state_norm_stats is None and isinstance(self.norm_stats, dict):
            self.state_norm_stats = self.norm_stats.get("state")

        self._validate_relative_actions()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_relative_actions(self) -> None:
        """``relative_exclude_joints`` is only meaningful if we can resolve the
        names to action indices, which needs ``action_feature_names``.

        Failing loudly here is the whole point: an unresolvable exclusion list
        would otherwise degrade to "make every dimension relative", which is a
        wrong-but-plausible action chunk (the gripper would be driven by a
        delta instead of an absolute command).
        """
        if not self.use_relative_actions:
            return
        if not self.relative_exclude_joints:
            return
        if not self.action_feature_names:
            raise UnsupportedCheckpointCapabilityError(
                "config declares use_relative_actions=True with "
                f"relative_exclude_joints={self.relative_exclude_joints!r}, but "
                "action_feature_names is missing, so those joint names cannot be "
                "resolved to action indices. LeRobot fills action_feature_names "
                "from dataset metadata at training time; a servable checkpoint "
                "must carry it in config.json. Either add action_feature_names, "
                "or set relative_exclude_joints=[] to make every dimension relative."
            )
        # Resolve eagerly so an unresolvable name fails at load, not mid-request.
        resolve_excluded_action_indices(self.relative_exclude_joints, self.action_feature_names)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path) -> Pi05Config:
        """Build from a checkpoint directory's ``config.json``."""
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            return cls()
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_model_config(raw)

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any] | None) -> Pi05Config:
        """Build from a config dict (LeRobot ``config.json`` or deploy yaml).

        Unlike a plain ``{k: v for k in allowed}`` filter, this first rejects
        the keys that declare capabilities we do not implement.
        """
        if not model_config:
            return cls()

        raw = dict(model_config)
        _reject_unsupported_capabilities(raw)

        if "image_resolution" in raw:
            raw["image_resolution"] = tuple(raw["image_resolution"])

        allowed = {item.name for item in dataclass_fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in allowed}

        dropped = sorted(set(raw) - set(filtered))
        if dropped:
            # Not an error: a LeRobot config.json legitimately carries training-only
            # keys (optimizer_*, scheduler_*, ...). Anything in it that *would*
            # change inference behaviour is handled above instead.
            logger.debug("π0.5 config: ignoring %d non-runtime key(s): %s", len(dropped), dropped)
        return cls(**filtered)


# ----------------------------------------------------------------------
# Checkpoint-boundary rule
# ----------------------------------------------------------------------
# Each entry: config key → (predicate on the raw value, human explanation).
# A key only trips when its value is actually *enabled*; a checkpoint that
# carries the field at its default is servable.
_UNSUPPORTED: dict[str, tuple[Any, str]] = {
    "use_visual_memory": (
        lambda v: bool(v),
        "MEM visual memory feeds historical image frames into SigLIP. It needs a "
        "per-session observation history, which this stateless serving path does "
        "not keep.",
    ),
    "use_proprioceptive_memory": (
        lambda v: bool(v),
        "MEM proprioceptive memory replaces the discretized state prompt tokens "
        "with a continuous projected state token (it flips "
        "include_state_in_prompt to False), so the prompt this implementation "
        "builds would not match what the checkpoint was trained on.",
    ),
    "rtc_config": (
        lambda v: v is not None,
        "Real-Time Chunking requires prefix guidance inside the denoising loop "
        "and per-request carry-over of the previous action chunk. Not implemented.",
    ),
    "n_obs_steps": (
        lambda v: v is not None and int(v) > 1,
        "n_obs_steps > 1 means the policy consumes an observation history; this "
        "path is first-order Markov (one observation in, one action chunk out).",
    ),
}


def _reject_unsupported_capabilities(raw: dict[str, Any]) -> None:
    """Raise if the checkpoint enables something we do not consume.

    Scope note: this validates that we correctly consume *the checkpoint we were
    handed*. Choosing which checkpoint to load belongs to the caller.
    """
    problems: list[str] = []
    for key, (is_enabled, why) in _UNSUPPORTED.items():
        if key not in raw:
            continue
        value = raw[key]
        try:
            tripped = is_enabled(value)
        except (TypeError, ValueError):
            tripped = True
        if tripped:
            problems.append(f"  - {key}={value!r}: {why}")

    if problems:
        raise UnsupportedCheckpointCapabilityError(
            "This checkpoint declares capabilities the vllm-omni π0.5 "
            "implementation does not support:\n"
            + "\n".join(problems)
            + "\nServing it anyway would produce plausible-looking but wrong actions."
        )

    # `rtc_training_max_delay > 0` is deliberately NOT an error. It is a
    # property of how the checkpoint was *trained* (it had clean action
    # prefixes sampled during training), not a request to run RTC at
    # inference. Such a checkpoint is still correct to serve without RTC.
    delay = raw.get("rtc_training_max_delay")
    if delay:
        logger.info(
            "π0.5 config: checkpoint was trained with rtc_training_max_delay=%s. "
            "Real-Time Chunking is not implemented here; serving proceeds without it.",
            delay,
        )
