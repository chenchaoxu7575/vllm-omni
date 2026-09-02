# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
from collections.abc import Mapping
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

# LeRobot fields that affect training, export, or Hub metadata but not the
# frozen inference graph. Keep this list explicit: accepting an arbitrary key
# would turn checkpoint typos into silent defaulting.
_LEROBOT_TRAINING_ONLY_KEYS = {
    "device",
    "use_amp",
    "push_to_hub",
    "repo_id",
    "private",
    "tags",
    "license",
    "pretrained_path",
    "pretrained_revision",
    "time_sampling_beta_alpha",
    "time_sampling_beta_beta",
    "time_sampling_scale",
    "time_sampling_offset",
    "normalization_mapping",
    "gradient_checkpointing",
    "compile_model",
    "compile_mode",
    "freeze_vision_encoder",
    "train_expert_only",
    "optimizer_lr",
    "optimizer_betas",
    "optimizer_eps",
    "optimizer_weight_decay",
    "optimizer_grad_clip_norm",
    "scheduler_warmup_steps",
    "scheduler_decay_steps",
    "scheduler_decay_lr",
}


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
    # The stateless OpenPI endpoint returns one complete predicted chunk.
    n_action_steps: int = 50
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

    # Checkpoint feature schemas. The input schema determines camera order; the
    # output schema determines the unpadded action width returned on the wire.
    input_features: dict[str, Any] = field(default_factory=dict)
    output_features: dict[str, Any] = field(default_factory=dict)
    # OpenPI handshake metadata from the deploy config. Construction validates
    # it against the resolved checkpoint contract.
    policy_server_config: dict[str, Any] = field(default_factory=dict)

    # Derived from output_features[ACTION].shape; never accepted as a second
    # source of truth in config.json.
    action_dim: int = field(init=False)

    def __post_init__(self) -> None:
        # Coerce list → tuple (JSON has no tuples) and validate squareness.
        res = self.image_resolution
        if not isinstance(res, (tuple, list)) or len(res) != 2 or res[0] != res[1]:
            raise ValueError(f"π0.5 expects a square image_resolution (H == W); got {res!r}.")
        self.image_resolution = (int(res[0]), int(res[1]))
        if self.image_resolution[0] < 1:
            raise ValueError(f"image_resolution must be positive, got {self.image_resolution!r}.")

        for name in (
            "chunk_size",
            "n_action_steps",
            "max_action_dim",
            "max_state_dim",
            "num_inference_steps",
            "tokenizer_max_length",
            "max_cameras",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.n_action_steps != self.chunk_size:
            raise UnsupportedCheckpointCapabilityError(
                "The stateless OpenPI serving path returns one complete action chunk and "
                f"does not support n_action_steps={self.n_action_steps} with "
                f"chunk_size={self.chunk_size}."
            )

        if isinstance(self.state_num_bins, bool) or not isinstance(self.state_num_bins, int) or self.state_num_bins < 2:
            raise ValueError(f"state_num_bins must be >= 2, got {self.state_num_bins}.")
        if self.min_period <= 0 or self.max_period <= self.min_period:
            raise ValueError(f"Expected 0 < min_period < max_period, got {self.min_period!r} and {self.max_period!r}.")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"dtype must be 'float32' or 'bfloat16', got {self.dtype!r}.")

        # Derive the camera order from input_features if not given explicitly.
        if self.image_feature_keys is None and self.input_features:
            self.image_feature_keys = [key for key in self.input_features if key.startswith(OBS_IMAGES + ".")]
        if self.image_feature_keys:
            if len(set(self.image_feature_keys)) != len(self.image_feature_keys):
                raise ValueError(f"image_feature_keys contains duplicates: {self.image_feature_keys!r}.")
            if len(self.image_feature_keys) > self.max_cameras:
                raise ValueError(
                    f"Checkpoint declares {len(self.image_feature_keys)} image features but "
                    f"max_cameras={self.max_cameras}."
                )
        self._validate_input_features()

        # ``state_norm_stats`` is just a view onto norm_stats["state"].
        if self.state_norm_stats is None and isinstance(self.norm_stats, dict):
            self.state_norm_stats = self.norm_stats.get("state")

        self._derive_action_dim()
        self._validate_policy_server_config()
        self._validate_relative_actions()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_input_features(self) -> None:
        """Validate a declared LeRobot observation schema."""
        if not self.input_features:
            return
        if not isinstance(self.input_features, Mapping):
            raise ValueError(f"input_features must be a mapping, got {type(self.input_features).__name__}.")

        state_feature = self.input_features.get(OBS_STATE)
        if state_feature is not None:
            self._validate_feature_shape(
                name=OBS_STATE,
                feature=state_feature,
                expected_type="STATE",
                expected_rank=1,
            )
            state_dim = int(state_feature["shape"][0])
            if state_dim > self.max_state_dim:
                raise ValueError(f"Checkpoint state_dim={state_dim} exceeds max_state_dim={self.max_state_dim}.")

        for key in self.image_feature_keys or []:
            feature = self.input_features.get(key)
            if feature is None:
                raise ValueError(f"image_feature_keys contains {key!r}, but input_features does not declare it.")
            self._validate_feature_shape(name=key, feature=feature, expected_type="VISUAL", expected_rank=3)
            shape = tuple(feature["shape"])
            expected_shape = (3, *self.image_resolution)
            if shape != expected_shape:
                raise ValueError(f"input_features[{key!r}].shape must be {expected_shape}, got {shape}.")

    @staticmethod
    def _validate_feature_shape(
        *,
        name: str,
        feature: Any,
        expected_type: str,
        expected_rank: int,
    ) -> None:
        if not isinstance(feature, Mapping):
            raise ValueError(f"input_features[{name!r}] must be a mapping.")
        if str(feature.get("type", "")).upper() != expected_type:
            raise ValueError(f"input_features[{name!r}].type must be {expected_type!r}.")
        shape = feature.get("shape")
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != expected_rank
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape)
        ):
            raise ValueError(
                f"input_features[{name!r}].shape must contain {expected_rank} positive integers, got {shape!r}."
            )

    def _derive_action_dim(self) -> None:
        """Resolve the real action width from the checkpoint output schema."""
        self.action_dim = self.max_action_dim
        if not self.output_features:
            return
        if not isinstance(self.output_features, Mapping):
            raise ValueError(f"output_features must be a mapping, got {type(self.output_features).__name__}.")

        unknown = sorted(set(self.output_features) - {ACTION})
        if unknown:
            raise UnsupportedCheckpointCapabilityError(
                f"π0.5 serving supports only the {ACTION!r} output feature; got {unknown!r}."
            )
        feature = self.output_features.get(ACTION)
        if not isinstance(feature, Mapping):
            raise ValueError("output_features must declare an 'action' mapping.")
        feature_type = str(feature.get("type", "")).upper()
        if feature_type != "ACTION":
            raise ValueError(f"output_features['action'].type must be 'ACTION', got {feature.get('type')!r}.")
        shape = feature.get("shape")
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 1
            or isinstance(shape[0], bool)
            or not isinstance(shape[0], int)
            or shape[0] < 1
        ):
            raise ValueError(f"output_features['action'].shape must be [action_dim], got {shape!r}.")
        if shape[0] > self.max_action_dim:
            raise ValueError(f"Checkpoint action_dim={shape[0]} exceeds max_action_dim={self.max_action_dim}.")
        self.action_dim = int(shape[0])

    def _validate_policy_server_config(self) -> None:
        """Keep OpenPI handshake metadata aligned with the model contract."""
        if not self.policy_server_config:
            return
        if not isinstance(self.policy_server_config, Mapping):
            raise ValueError("policy_server_config must be a mapping.")

        expected = {
            "action_horizon": self.chunk_size,
            "action_dim": self.action_dim,
            "max_action_dim": self.max_action_dim,
            "max_cameras": self.max_cameras,
        }
        for key, value in expected.items():
            declared = self.policy_server_config.get(key)
            if declared is not None and declared != value:
                raise ValueError(
                    f"policy_server_config.{key}={declared!r} does not match the resolved π0.5 value {value!r}."
                )
        declared_resolution = self.policy_server_config.get("image_resolution")
        if declared_resolution is not None and tuple(declared_resolution) != self.image_resolution:
            raise ValueError(
                "policy_server_config.image_resolution="
                f"{declared_resolution!r} does not match image_resolution={self.image_resolution!r}."
            )

    def _validate_checkpoint_feature_schema(self, config_path: Path) -> None:
        """Require the feature schema that defines checkpoint I/O semantics."""
        if not isinstance(self.input_features, Mapping) or OBS_STATE not in self.input_features:
            raise ValueError(f"{config_path} must declare input_features[{OBS_STATE!r}].")
        if not self.image_feature_keys:
            raise ValueError(f"{config_path} must declare at least one {OBS_IMAGES!r} input feature.")
        if not isinstance(self.output_features, Mapping) or ACTION not in self.output_features:
            raise ValueError(f"{config_path} must declare output_features[{ACTION!r}].")

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
        """Build from a checkpoint directory's ``config.json``.

        Normalization stats are *not* in ``config.json``. LeRobot keeps them in
        the processor sidecar, so they are loaded separately and backfilled
        here — see :func:`load_lerobot_norm_stats`.
        """
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"π0.5 checkpoint is missing required config: {config_path}.")
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        config = cls.from_model_config(raw)
        config._validate_checkpoint_feature_schema(config_path)

        if config.norm_stats is None:
            stats = load_lerobot_norm_stats(checkpoint_dir)
            if stats:
                config.norm_stats = stats
                config.state_norm_stats = stats.get("state")
        return config

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any] | None) -> Pi05Config:
        """Build from a config dict (LeRobot ``config.json`` or deploy yaml).

        Only an explicit LeRobot training/metadata allowlist may be ignored.
        Unknown keys and enabled unsupported capabilities fail closed.
        """
        if not model_config:
            return cls()

        raw = dict(model_config)
        _reject_unsupported_capabilities(raw)

        model_type = raw.pop("type", "pi05")
        if model_type != "pi05":
            raise ValueError(f"Expected a π0.5 checkpoint (type='pi05'), got type={model_type!r}.")

        if "image_resolution" in raw:
            raw["image_resolution"] = tuple(raw["image_resolution"])

        allowed = {item.name for item in dataclass_fields(cls) if item.init}
        known_capability_keys = set(_UNSUPPORTED) | {"memory_frames", "rtc_training_max_delay"}
        unknown = sorted(set(raw) - allowed - _LEROBOT_TRAINING_ONLY_KEYS - known_capability_keys)
        if unknown:
            raise ValueError(
                "Unknown π0.5 config key(s): "
                f"{unknown}. Only explicit runtime fields and reviewed LeRobot training metadata are accepted."
            )

        filtered = {key: value for key, value in raw.items() if key in allowed}
        ignored = sorted(set(raw) & _LEROBOT_TRAINING_ONLY_KEYS)
        if ignored:
            logger.debug("π0.5 config: ignoring LeRobot training/metadata keys: %s", ignored)
        return cls(**filtered)


# ----------------------------------------------------------------------
# LeRobot normalization-stats sidecar
# ----------------------------------------------------------------------
# LeRobot keeps normalization stats out of ``config.json``. It serializes the
# processor pipeline as ``policy_preprocessor.json`` (structure only) plus one
# safetensors file per *stateful* step, named in that step's ``state_file``
# entry. ``NormalizerProcessorStep.state_dict`` flattens the stats to
# ``"<feature_name>.<stat_name>"`` keys.
_PREPROCESSOR_JSON = "policy_preprocessor.json"

# LeRobot ``NormalizationMode`` → (our mode name, the two stat tensors it uses).
# ``IDENTITY`` maps to None: the feature is deliberately left untransformed.
#
# The mode has to come from ``norm_map``. A LeRobot state_dict carries *all* of
# mean/std/min/max/q01/q99 — ``compute_stats`` emits the full set regardless of
# norm_map — so inferring it from which keys are present would read a QUANTILES
# checkpoint as ``mean_std`` and apply a wrong affine map.
_NORM_MODE_FROM_LEROBOT: dict[str, tuple[str, str, str] | None] = {
    "IDENTITY": None,
    "MEAN_STD": ("mean_std", "mean", "std"),
    "MIN_MAX": ("min_max", "min", "max"),
    "QUANTILES": ("quantile", "q01", "q99"),
}
# The two features we consume, as (LeRobot FeatureType, our norm_stats key).
# Feature renaming happens in an earlier processor step, so these names are
# canonical by the time the normalizer runs. VISUAL is absent by design: π0.5
# normalizes images in the image processor, not from these stats.
_NORM_STATS_FEATURES = {OBS_STATE: ("STATE", "state"), ACTION: ("ACTION", "action")}


def load_lerobot_norm_stats(checkpoint_dir: str | Path) -> dict[str, dict[str, Any]] | None:
    """Load normalization stats from a LeRobot checkpoint's processor sidecar.

    Returns a ``norm_stats``-shaped dict (``{"state": {...}, "action": {...}}``)
    carrying an **explicit** ``mode``, or ``None`` when the checkpoint ships no
    stats. ``lerobot/pi05_base`` is the latter case: its normalizer step has no
    ``state_file``, i.e. normalization is identity and the client is expected to
    send an already-normalized state.

    Raises on a mode we cannot reproduce rather than serving the checkpoint with
    the wrong transform, which fails silently — a wrongly normalized state still
    yields a plausible-looking action chunk.
    """
    checkpoint_dir = Path(checkpoint_dir)
    preprocessor_path = checkpoint_dir / _PREPROCESSOR_JSON
    if not preprocessor_path.exists():
        return None
    with open(preprocessor_path, encoding="utf-8") as f:
        steps = json.load(f).get("steps", [])

    step = next((s for s in steps if s.get("registry_name") == "normalizer_processor"), {})
    state_file = step.get("state_file")
    if not state_file:
        logger.info(
            "π0.5 config: %s declares no normalizer state file — the checkpoint ships no "
            "normalization stats and the state passes through unchanged.",
            _PREPROCESSOR_JSON,
        )
        return None
    state_path = checkpoint_dir / state_file
    if not state_path.exists():
        raise FileNotFoundError(
            f"π0.5 checkpoint preprocessor {preprocessor_path} declares normalizer state "
            f"{state_file!r}, but {state_path} does not exist."
        )

    norm_map = {str(k).upper(): str(v).upper() for k, v in ((step.get("config") or {}).get("norm_map") or {}).items()}

    import safetensors.torch

    flat = safetensors.torch.load_file(str(state_path))

    stats: dict[str, dict[str, Any]] = {}
    for feature_name, (feature_type, stats_key) in _NORM_STATS_FEATURES.items():
        lerobot_mode = norm_map.get(feature_type, "IDENTITY")
        if lerobot_mode not in _NORM_MODE_FROM_LEROBOT:
            raise ValueError(
                f"π0.5 checkpoint declares normalization mode {lerobot_mode!r} for {feature_type}, "
                f"which this implementation cannot reproduce. Expected one of "
                f"{sorted(_NORM_MODE_FROM_LEROBOT)}."
            )
        selected = _NORM_MODE_FROM_LEROBOT[lerobot_mode]
        if selected is None:
            continue

        mode, *stat_names = selected
        missing = [name for name in stat_names if f"{feature_name}.{name}" not in flat]
        if missing:
            raise ValueError(
                f"π0.5 checkpoint declares {lerobot_mode} normalization for {feature_type} but its "
                f"normalizer state is missing {missing} for feature {feature_name!r}."
            )
        stats[stats_key] = {"mode": mode} | {name: flat[f"{feature_name}.{name}"].tolist() for name in stat_names}

    if not stats:
        return None
    logger.info(
        "π0.5 config: loaded normalization stats from %s — %s.",
        state_file,
        ", ".join(f"{key}={value['mode']}" for key, value in sorted(stats.items())),
    )
    return stats


# ----------------------------------------------------------------------
# Checkpoint-boundary rule
# ----------------------------------------------------------------------
# Each entry: config key → (predicate on the raw value, human explanation).
# A key only trips when its value is actually *enabled*; a checkpoint that
# carries the field at its default is servable.
_UNSUPPORTED: dict[str, tuple[Any, str]] = {
    "use_peft": (
        lambda v: bool(v),
        "PEFT checkpoints require adapter-aware loading; this loader accepts only a complete merged checkpoint.",
    ),
    "empty_cameras": (
        lambda v: v is not None and int(v) != 0,
        "LeRobot's empty_cameras feature mutates the checkpoint input schema. Declare the final camera "
        "features explicitly and use max_cameras for tail padding instead.",
    ),
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
