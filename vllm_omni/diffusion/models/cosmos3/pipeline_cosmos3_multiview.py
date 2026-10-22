# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3-Nano-Transfer-Auto pipeline.

Camera-only and joint V1.2 camera/LiDAR inference with independent sensor
geometries and per-camera captions. Only camera targets are decoded.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_extras.cosmos3 import (
    COSMOS3_MULTIVIEW_UNSUPPORTED_NEGATIVE_FIELDS,
    COSMOS3_TRANSFER_HINT_KEYS,
    normalize_multiview_aspect_ratio,
    path_media_kind,
    reject_multiview_negative_fields,
    validate_multiview_request,
)
from vllm_omni.model_extras.cosmos3_lidar import load_lidar_frames, required_lidar_sweeps

from .action import find_closest_target_size
from .lidar_vae import Cosmos3LidarDecoder, Cosmos3LidarEncoder
from .multiview_config import _validated_multiview_deployment_config
from .multiview_flex_attention import (
    DEFAULT_MAX_UND_TOKENS,
    MaskItem,
    MultiviewLayout,
    expand_multiview_condition_frame_indexes,
    validate_multiview_backend,
)
from .multiview_packing import pack_state, patch_grid, unpack_state
from .multiview_parallel import validate_multiview_parallel_config
from .multiview_prompts import (
    COSMOS3_AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT,
    COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
    control_emphasis,
    format_rig_view_captions,
)
from .pipeline_cosmos3 import (
    COSMOS3_TRANSFER_SYSTEM_PROMPT,
    Cosmos3OmniDiffusersPipeline,
    _ceil_video_num_frames,
    get_cosmos3_ir_op_priority_func,
)
from .transfer import (
    as_bool,
    media_hw,
    media_to_uint8_cthw,
)
from .transformer_cosmos3 import _tf_config_get
from .utils import VIDEO_RES_SIZE_INFO
from .vae_owner import (
    FieldSpec,
    VaeGrid,
    VaeOwnerExchange,
    VaePlan,
    VaeWork,
    broadcast_owned_work,
    decode_and_gather,
    plan_vae_work,
    receive_buffers,
    vae_grid,
)

logger = init_logger(__name__)

# Select the runtime kernel strategy (maskless | triton); both obey
# the same attention rules.
COSMOS3_MULTIVIEW_BACKEND_ENV = "VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND"

# Per-camera frame count when the request supplies none.
COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES = 201
# Rates and frame counts outside these bounds are allowed with a warning.
COSMOS3_MULTIVIEW_RECOMMENDED_FPS_RANGE = (10.0, 30.0)
COSMOS3_MULTIVIEW_RECOMMENDED_NUM_FRAMES_RANGE = (24, 400)
# Measured LiDAR sweeps conditioned when lidar.condition_path omits a count,
# as in the reference inference.
COSMOS3_MULTIVIEW_DEFAULT_LIDAR_CONDITION_SWEEPS = 1
# The tokenizer appends eos and vision_start after truncating. Derive the
# request ceiling from the sparse attention's single fixed UND capacity so the
# two cannot drift and accidentally trigger shape-specific recompilation.
COSMOS3_MULTIVIEW_PROMPT_FRAMING_TOKENS = 2
COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH = DEFAULT_MAX_UND_TOKENS - COSMOS3_MULTIVIEW_PROMPT_FRAMING_TOKENS


def _media_kind(value: Any) -> str:
    if isinstance(value, str | Path):
        return path_media_kind(value)
    if isinstance(value, PIL.Image.Image):
        return "image"
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.ndim == 5:
            return "image" if tensor.shape[2] == 1 else "video"
        if tensor.ndim == 4:
            temporal_dim = 1 if tensor.shape[0] in (3, 4) else 0
            return "image" if tensor.shape[temporal_dim] == 1 else "video"
        if tensor.ndim == 3:
            return "image"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return "image" if len(value) == 1 else "video"
    raise TypeError(f"Unsupported Cosmos3 multiview media payload type: {type(value).__name__}.")


def _normalize_local_condition_indexes(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, int):
        values = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise TypeError("Cosmos3 multiview condition_frame_indexes_vision must be an int, list, or CSV string.")
    return sorted({int(index) for index in values})


def _pad_multiview_view_video(
    frames: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Pad one camera to ``num_frames`` by replicating its last frame.

    The reference pads a view into a mid-gray canvas and then replicates the
    last decoded frame over the tail, so gray only ever survives for a view
    with no media at all.  Admission requires a control clip for every camera
    and vision for all cameras or none, so that case cannot reach here; an
    empty clip is a decode failure and is reported rather than silently
    generating a gray camera.
    """
    if frames.ndim != 4 or tuple(frames.shape[:1]) != (3,) or tuple(frames.shape[2:]) != (height, width):
        raise ValueError(
            "Cosmos3 multiview view frames must have shape [3, T, H, W] at the output resolution, "
            f"got {tuple(frames.shape)}."
        )
    fill = min(int(frames.shape[1]), num_frames)
    if fill <= 0:
        raise ValueError("Cosmos3 multiview view media decoded to zero frames.")
    video = frames[:, :fill]
    if fill < num_frames:
        video = torch.cat([video, video[:, -1:].expand(-1, num_frames - fill, -1, -1)], dim=1)
    return video.contiguous()


def _resolve_multiview_resolution(sp: Any, multiview: Mapping[str, Any], *, default: str = "480") -> str:
    """Resolve the variant-owned resolution without inheriting the image default."""
    resolution = multiview.get("resolution")
    if resolution is None:
        extra = sp.extra_args if isinstance(sp.extra_args, Mapping) else {}
        resolution = extra.get("resolution")
    if resolution is None:
        resolution = default
    resolution = str(resolution)
    if resolution not in ("480", "720"):
        raise ValueError(f"Cosmos3 multiview supports resolutions '480' and '720', got {resolution!r}.")
    return resolution


def _resolve_multiview_geometry(
    sp: Any,
    multiview: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
    *,
    default_resolution: str = "480",
) -> tuple[str, str, int, int]:
    """Select one bucket for every camera from an override or the first WSM."""
    resolution = _resolve_multiview_resolution(sp, multiview, default=default_resolution)
    extra = sp.extra_args if isinstance(sp.extra_args, Mapping) else {}
    requested_ratio = multiview.get("aspect_ratio")
    if requested_ratio is None:
        requested_ratio = extra.get("aspect_ratio")
    aspect_ratio = normalize_multiview_aspect_ratio(requested_ratio)
    sizes = VIDEO_RES_SIZE_INFO[resolution]
    if aspect_ratio == "auto":
        view = views[0]
        control = view.get("control_path", view.get("control"))
        if control is None:
            control = view.get("vision_path", view.get("vision"))
        if control is None:
            # Ordinary T2V has no source canvas; use the reference video bucket.
            return _resolve_multiview_geometry(
                sp, {**multiview, "aspect_ratio": "16,9"}, views, default_resolution=default_resolution
            )
        source = str(control) if isinstance(control, str | Path) else "decoded WSM input"
        try:
            source_hw = media_hw(control)
            if source_hw is None or min(source_hw) <= 0:
                raise ValueError("input has no readable positive spatial dimensions")
        except Exception as exc:
            raise ValueError(
                f"Cannot detect Cosmos3 multiview aspect ratio from camera {view['camera_key']!r} "
                f"WSM input {source!r}: {exc}"
            ) from exc
        width, height = find_closest_target_size(*source_hw, resolution)
        aspect_ratio = next(ratio for ratio, size in sizes.items() if size == (width, height))
    else:
        width, height = sizes[aspect_ratio]
    for key, expected in (("width", width), ("height", height)):
        requested = getattr(sp, key, None)
        if requested is not None and int(requested) != expected:
            raise ValueError(
                f"Cosmos3 multiview resolution={resolution!r} requires {key}={expected}, got {requested} "
                f"for aspect_ratio={aspect_ratio!r}."
            )
    return resolution, aspect_ratio, width, height


def _resolve_multiview_frame_rate(value: Any, default_fps: float) -> float:
    """Validate the request frame rate.

    Any finite positive rate is accepted: fps only feeds the fps-modulated
    temporal mRoPE, the prompt metadata, and the mask timestamps. Rates outside
    the recommended range are allowed with a warning.
    """
    if isinstance(value, bool):
        raise TypeError("Cosmos3 multiview FPS must be a number, not a boolean.")
    try:
        frame_rate = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Cosmos3 multiview FPS must be a number, got {value!r}.") from exc
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"Cosmos3 multiview FPS must be finite and positive, got {value!r}.")
    low, high = COSMOS3_MULTIVIEW_RECOMMENDED_FPS_RANGE
    if not low <= frame_rate <= high:
        logger.warning(
            "Cosmos3 multiview FPS %s is outside the recommended range [%s, %s]; the checkpoint default "
            "is %s FPS, so quality may be degraded.",
            frame_rate,
            low,
            high,
            default_fps,
        )
    return frame_rate


def _resolve_multiview_num_frames(value: Any, temporal_compression_factor: int) -> int:
    """Resolve the per-camera frame count, rounding up to the VAE grid.

    ``None`` and the ``OmniDiffusionSamplingParams`` legacy image default of one
    frame select the variant default. Other lengths are rounded up to the Wan
    VAE's ``4k+1`` grid instead of being rejected, so 200 becomes 201.
    """
    if value is None or (not isinstance(value, bool) and value == 1):
        return COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES
    if isinstance(value, bool):
        raise TypeError("Cosmos3 multiview num_frames must be an integer, not a boolean.")
    try:
        num_frames = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Cosmos3 multiview num_frames must be an integer, got {value!r}.") from exc
    if num_frames <= 1:
        raise ValueError(f"Cosmos3 multiview num_frames must be greater than 1, got {value!r}.")
    rounded = _ceil_video_num_frames(num_frames, temporal_compression_factor)
    if rounded != num_frames:
        logger.info(
            "Rounded Cosmos3 multiview num_frames from %d to %d for temporal compression factor %d.",
            num_frames,
            rounded,
            temporal_compression_factor,
        )
    low, high = COSMOS3_MULTIVIEW_RECOMMENDED_NUM_FRAMES_RANGE
    if not low <= rounded <= high:
        logger.warning(
            "Cosmos3 multiview num_frames %d is outside the recommended range [%d, %d]; quality may be degraded.",
            rounded,
            low,
            high,
        )
    return rounded


def _multiview_system_prompt(*, transfer: bool, joint: bool) -> str:
    """Select the AV task prompt used by reference inference."""
    if joint:
        return COSMOS3_AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT
    if transfer:
        return COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT
    return COSMOS3_TRANSFER_SYSTEM_PROMPT


def _multiview_request_captions(request: Any) -> list[str]:
    """Return the per-camera captions a request carries, ignoring malformed fields.

    Structural validation happens in the pipeline; this only collects text the
    guardrail must see before the request is admitted.
    """
    extra = getattr(request.sampling_params, "extra_args", None)
    multiview = extra.get("multiview") if isinstance(extra, Mapping) else None
    views = multiview.get("views") if isinstance(multiview, Mapping) else None
    if not isinstance(views, Sequence) or isinstance(views, str | bytes):
        return []
    captions = []
    for view in views:
        caption = view.get("prompt") if isinstance(view, Mapping) else None
        if isinstance(caption, str) and caption.strip():
            captions.append(caption)
    return captions


def get_cosmos3_multiview_pre_process_func(od_config: OmniDiffusionConfig):
    """Build the request preprocessor for Cosmos3-Nano-Transfer-Auto.

    Camera and LiDAR media are decoded by the pipeline, so the only request-time
    work is the Cosmos3 text guardrail over the shared prompt and every
    per-camera caption. The multiview postprocessor applies the video
    guardrail to each camera.
    """
    from .guardrails import check_text_safety, ensure_initialized, is_guardrails_enabled

    # Eager-load guardrail models at pipeline build time when the server-level
    # gate is on. Per-request overrides only decide whether the loaded models
    # are *invoked* — they cannot turn checks on without a server-side preload.
    if is_guardrails_enabled(od_config):
        ensure_initialized(od_config)

    def pre_process_func(request: Any) -> Any:
        if not is_guardrails_enabled(od_config, request.sampling_params):
            return request
        prompt = request.prompt
        check_text_safety(prompt if isinstance(prompt, str) else str(prompt.get("prompt", "")))
        for caption in _multiview_request_captions(request):
            check_text_safety(caption)
        return request

    return pre_process_func


def get_cosmos3_multiview_post_process_func(od_config: OmniDiffusionConfig):
    """Process camera-major video one view at a time and preserve numeric LiDAR."""
    from .guardrails import check_video_safety, is_guardrails_enabled

    video_processor = VideoProcessor(vae_scale_factor=16)

    def post_process_func(output: dict[str, Any], output_type: str = "np", sampling_params=None):
        if output_type == "latent":
            return output

        payload = output.get("payload")
        if not isinstance(payload, dict) or "video" not in payload:
            raise ValueError("Cosmos3 multiview postprocess requires a video payload.")
        metadata = output.get("metadata") or {}
        public_metadata = {key: value for key, value in metadata.items() if key != "internal"}
        multiview = public_metadata.get("multiview")
        if not isinstance(multiview, Mapping):
            raise ValueError("Cosmos3 multiview output requires multiview metadata.")
        cameras = multiview.get("cameras")
        frames_per_view = multiview.get("frames_per_view")
        if (
            not isinstance(cameras, list | tuple)
            or not cameras
            or not isinstance(frames_per_view, int)
            or isinstance(frames_per_view, bool)
            or frames_per_view <= 0
        ):
            raise ValueError("Cosmos3 multiview output requires cameras and a positive integer frames_per_view.")
        video = payload["video"]
        if video.ndim != 5 or video.shape[:2] != (1, 3) or video.shape[2] != len(cameras) * frames_per_view:
            raise ValueError("Cosmos3 multiview output must have shape [1, 3, V*F, H, W] matching its metadata.")
        if output_type not in ("np", "pt", "pil"):
            raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

        guardrails_enabled = is_guardrails_enabled(od_config, sampling_params)
        processed_video = None
        for view_index in range(len(cameras)):
            frame_start = view_index * frames_per_view
            # Move only this camera to the host, even for an in-process CUDA
            # caller. Guardrail conversions then stay bounded to one CPU clip.
            view = video.narrow(2, frame_start, frames_per_view).detach().cpu()
            if guardrails_enabled:
                view = check_video_safety(view)
            processed_view = video_processor.postprocess_video(view, output_type=output_type)
            if processed_video is None:
                if output_type == "pil":
                    processed_video = [[]]
                else:
                    output_shape = (1, video.shape[2], *processed_view.shape[2:])
                    if output_type == "np":
                        processed_video = np.empty(output_shape, dtype=processed_view.dtype)
                    else:
                        processed_video = torch.empty(output_shape, dtype=processed_view.dtype, device="cpu")
            if output_type == "pil":
                processed_video[0].extend(processed_view[0])
            else:
                processed_video[:, frame_start : frame_start + frames_per_view] = processed_view
            # Retaining converted clips and concatenating them would duplicate
            # the complete processed multiview output.
            del view, processed_view

        assert processed_video is not None
        processed_payload: dict[str, Any] = {"video": processed_video}
        if (lidar := payload.get("lidar")) is not None:
            # Numeric range images must never pass through RGB processing.
            processed_payload["lidar"] = lidar.detach().to(device="cpu", dtype=torch.float32).contiguous()
        return {"payload": processed_payload, "metadata": public_metadata}

    return post_process_func


@dataclass(frozen=True)
class _EncodeRequest:
    """Request inputs consumed by the VAE encoders."""

    views: Sequence[Mapping[str, Any]]
    height: int
    width: int
    num_frames: int
    frame_rate: float
    has_vision: bool
    condition_indexes: Sequence[int]
    condition_video_as_image: bool
    completion: bool
    selected_hints: Sequence[str]
    lidar: Mapping[str, Any] | None


@dataclass
class _EncodedInputs:
    """Request latents from the VAE encoders; identical on every rank."""

    vision: torch.Tensor | None
    control: torch.Tensor | None
    lidar_control: torch.Tensor | None
    lidar_condition: torch.Tensor | None
    # Owner placement, reused for decode; None on the serial path.
    plan: VaePlan | None = None


@dataclass
class _DecodedOutputs:
    video: torch.Tensor
    lidar: torch.Tensor | None


@dataclass
class _OwnerWork:
    plan: VaePlan
    items: list[VaeWork]
    specs: dict[str, list[FieldSpec]]
    lidar_sweeps: int
    lidar_shape: tuple[int, ...] | None


class Cosmos3MultiviewPipeline(Cosmos3OmniDiffusersPipeline):
    """Joint camera/LiDAR and camera-only generation with checkpoint-owned view layouts."""

    # The generic engine warmup has no per-camera WSM inputs and uses image
    # geometry that is invalid for this multiview pipeline. Compile the
    # model on its first real request instead of weakening request validation.
    dummy_run_num_frames: ClassVar[int] = 0
    _encoder_modules: ClassVar[list[str]] = []

    def predict_noise(self, **kwargs):
        # Resolve branch metadata on the host before entering the transformer.
        lengths_by_text = kwargs.pop("_multiview_caption_lengths", None)
        if lengths_by_text is not None:
            kwargs["caption_lengths"] = lengths_by_text[kwargs["text_ids"].data_ptr()]
        # The shared transfer loop passes these to every branch. Captions are
        # already compacted, hints carry no per-control weight, and the LiDAR
        # condition prefix is applied by _apply_transfer_condition.
        for key in ("text_mask", "control_weights", "lidar_condition_latents"):
            kwargs.pop(key, None)
        return super().predict_noise(**kwargs)

    def combine_multi_branch_cfg_noise(self, predictions, true_cfg_scale, cfg_normalize=False):
        if not isinstance(true_cfg_scale, dict) or true_cfg_scale.get("mode") != "cosmos3_transfer":
            return super().combine_multi_branch_cfg_noise(predictions, true_cfg_scale, cfg_normalize)
        combined = super().combine_multi_branch_cfg_noise(predictions, true_cfg_scale, cfg_normalize=False)
        if cfg_normalize and true_cfg_scale.get("branch_mode") != "control_only":
            reference = predictions[0]
            if true_cfg_scale.get("branch_mode") == "control_and_text":
                reference = predictions[1] + true_cfg_scale["control_guidance"] * (predictions[0] - predictions[1])
            ratio = (reference.norm(dim=1, keepdim=True) / (combined.norm(dim=1, keepdim=True) + 1e-8)).clamp(0, 1)
            combined = combined * ratio
        return combined

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        multiview_config = _validated_multiview_deployment_config(od_config.tf_model_config)
        resolved_backend = self._resolve_attention_backend()
        validate_multiview_parallel_config(
            od_config.parallel_config,
            num_attention_heads=int(_tf_config_get(od_config.tf_model_config, "num_attention_heads", 32)),
            num_key_value_heads=int(_tf_config_get(od_config.tf_model_config, "num_key_value_heads", 8)),
            intermediate_size=int(_tf_config_get(od_config.tf_model_config, "intermediate_size", 12288)),
        )
        if od_config.enable_session_state_manager:
            raise ValueError("Cosmos3-Nano-Transfer-Auto does not support enable_session_state_manager.")
        super().__init__(od_config=od_config, prefix=prefix)
        if self.device.type != "cuda":
            raise ValueError("Cosmos3-Nano-Transfer-Auto requires CUDA for multiview attention.")

        self.multiview_config = multiview_config
        self._encoder_modules = []
        self._vae_modules = ["vae"]
        self.lidar_encoder = None
        self.lidar_decoder = None
        if multiview_config.get("lidar") is not None:
            self.lidar_encoder = Cosmos3LidarEncoder.from_pretrained(
                od_config.model, multiview_config["lidar"], self.device, od_config=od_config
            )
            self._encoder_modules = ["lidar_encoder"]
            self.lidar_decoder = Cosmos3LidarDecoder.from_pretrained(
                od_config.model, multiview_config["lidar"], self.device, od_config=od_config
            )
            self._vae_modules.append("lidar_decoder")
        self.multiview_cameras = tuple(multiview_config["cameras"])
        self.multiview_cross_view_past_window_seconds = multiview_config["cross_view_past_window_seconds"]
        self.multiview_backend = resolved_backend

    def setup_compile(self) -> None:
        """Compile multiview GEN layers with static shapes."""
        from vllm_omni.diffusion.compile import regionally_compile

        if self.od_config.diffusion_compile_granularity != "regional":
            logger.warning(
                "Cosmos3 multiview uses regional compilation; diffusion_compile_granularity=%r is ignored.",
                self.od_config.diffusion_compile_granularity,
            )

        regionally_compile(self.transformer, dynamic=False)

    @staticmethod
    def _resolve_attention_backend() -> str:
        """Pick the kernel strategy from the worker's vLLM-bundled FlashAttention.

        Exact maskless passes on FA4/FA3/FA2 varlen are the default, including
        on Blackwell. Triton FlexAttention remains the fallback when no bundled
        FlashAttention is importable. Both implement the same visibility rules.
        """
        override = os.environ.get(COSMOS3_MULTIVIEW_BACKEND_ENV)
        if override is not None:
            try:
                return validate_multiview_backend(override)
            except ValueError as exc:
                raise ValueError(f"{exc} (from {COSMOS3_MULTIVIEW_BACKEND_ENV}={override!r})") from exc
        try:
            from vllm_omni.diffusion.attention.backends.utils.fa import resolve_vllm_flash_attn_version

            resolve_vllm_flash_attn_version()
        except (ImportError, RuntimeError):
            return "triton"  # Non-CUDA hosts or no bundled FlashAttention.
        return "maskless"

    def _parse_multiview_request(self, sp: Any) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
        extra = sp.extra_args if isinstance(sp.extra_args, Mapping) else {}
        config = self.multiview_config
        reject_multiview_negative_fields(
            {field: getattr(sp, field, None) for field in COSMOS3_MULTIVIEW_UNSUPPORTED_NEGATIVE_FIELDS},
            "sampling_params",
        )
        if extra.get("lidar") is not None and config.get("lidar") is None:
            raise ValueError("Joint camera+LiDAR requests require a complete joint checkpoint.")
        return validate_multiview_request(
            extra,
            self.multiview_cameras,
            media_kind=_media_kind,
        )

    @staticmethod
    def _view_value(view: Mapping[str, Any], field: str) -> Any:
        return view.get(f"{field}_path", view.get(field))

    def _prepare_camera_major_pixels(
        self,
        views: Sequence[Mapping[str, Any]],
        *,
        field: str,
        height: int,
        width: int,
        num_frames: int,
        keep_first: bool,
        require_complete: bool = False,
    ) -> torch.Tensor:
        # Write each view into the final buffer. In particular, normalize an
        # image condition before repeating it across the requested frames.
        camera_major = torch.empty((3, len(views) * num_frames, height, width), dtype=self.dtype, device="cpu")
        for view_index, view in enumerate(views):
            output = camera_major[:, view_index * num_frames : (view_index + 1) * num_frames]
            value = self._view_value(view, field)
            if value is None:
                if field == "vision":
                    gray = torch.tensor(128, dtype=self.dtype, device="cpu").div_(127.5).sub_(1.0)
                    output.copy_(gray)
                    continue
                raise ValueError(f"Cosmos3 multiview camera {view['camera_key']!r} is missing {field} input.")
            frames = media_to_uint8_cthw(
                value,
                height=height,
                width=width,
                max_frames=1 if keep_first else num_frames,
            )
            if require_complete and frames.shape[1] < num_frames:
                raise ValueError(
                    f"Known camera {view['camera_key']!r} requires a complete RGB video of {num_frames} frames."
                )
            frame_count = min(int(frames.shape[1]), num_frames)
            frames = _pad_multiview_view_video(frames, num_frames=frame_count, height=height, width=width)
            # Cast to the model dtype on copy, then divide and subtract in place
            # as separate rounded steps, matching uint8_cthw_to_normalized_5d.
            normalized = output[:, :frame_count]
            normalized.copy_(frames).div_(127.5).sub_(1.0)
            if frame_count < num_frames:
                output[:, frame_count:].copy_(normalized[:, -1:])
        return camera_major.unsqueeze(0)

    def _encode_multiview_video(
        self,
        camera_major_video: torch.Tensor,
        *,
        num_views: int,
        frames_per_view: int,
    ) -> torch.Tensor:
        expected_frames = num_views * frames_per_view
        if camera_major_video.ndim != 5 or camera_major_video.shape[2] != expected_frames:
            raise ValueError(
                "Cosmos3 multiview pixel video must be camera-major [1, 3, V*F, H, W]: "
                f"shape={tuple(camera_major_video.shape)}, V={num_views}, F={frames_per_view}."
            )
        per_view = [
            self._encode_video_tensor(camera_major_video[:, :, view * frames_per_view : (view + 1) * frames_per_view])
            for view in range(num_views)
        ]
        latent_frames = {int(latent.shape[2]) for latent in per_view}
        if len(latent_frames) != 1:
            raise ValueError(f"Cosmos3 multiview per-camera VAE encodes have unequal lengths: {latent_frames}.")
        return torch.cat(per_view, dim=2)

    def _decode_multiview_latents(
        self,
        camera_major_latents: torch.Tensor,
        *,
        num_views: int,
        latent_frames_per_view: int,
    ) -> torch.Tensor:
        if (
            num_views <= 0
            or latent_frames_per_view <= 0
            or camera_major_latents.ndim != 5
            or camera_major_latents.shape[2] != num_views * latent_frames_per_view
        ):
            raise ValueError(
                "Cosmos3 multiview latents must be camera-major before decode: "
                f"shape={tuple(camera_major_latents.shape)}, V={num_views}, F={latent_frames_per_view}."
            )
        decoded_output = None
        for view in range(num_views):
            decoded_view = self._decode_latents(
                camera_major_latents.narrow(2, view * latent_frames_per_view, latent_frames_per_view)
            )
            if decoded_view.ndim != 5 or decoded_view.shape[2] <= 0:
                raise ValueError("Cosmos3 multiview VAE decode must return non-empty [B, C, F, H, W] clips.")
            if decoded_output is None:
                view_shape = decoded_view.shape
                frames_per_view = view_shape[2]
                output_shape = (*view_shape[:2], num_views * frames_per_view, *view_shape[3:])
                decoded_output = torch.empty(output_shape, dtype=decoded_view.dtype, device="cpu")
            elif decoded_view.shape != view_shape or decoded_view.dtype != decoded_output.dtype:
                raise ValueError("Cosmos3 multiview VAE decoded camera clips must have matching shapes and dtypes.")
            # Copy each camera before decoding the next: retaining the GPU clips
            # and concatenating them would require two full multiview videos.
            decoded_output.narrow(2, view * frames_per_view, frames_per_view).copy_(decoded_view)
            del decoded_view

        assert decoded_output is not None
        return decoded_output

    def _mask_transfer_noise(
        self, noise: torch.Tensor, velocity_mask: torch.Tensor, shared_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        # Broadcast the compact camera temporal mask on a view of the packed
        # prediction. A LiDAR condition is always a prefix, so no mask tensor.
        streams = unpack_state(noise, shared_kwargs["packed_shapes"])
        streams[0].mul_(velocity_mask)
        if condition_frames := shared_kwargs.get("lidar_condition_frames", 0):
            streams[1][:, :, :condition_frames].zero_()
        return noise

    def _apply_transfer_condition(
        self,
        latents: torch.Tensor,
        velocity_mask: torch.Tensor,
        condition_latents: torch.Tensor,
        shared_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        # UniPC and Euler return a new sample, separate from solver history.
        # Restore the conditions there without repacking either stream.
        streams = unpack_state(latents, shared_kwargs["packed_shapes"])
        streams[0].mul_(velocity_mask).add_((1.0 - velocity_mask) * condition_latents)
        if condition_frames := shared_kwargs.get("lidar_condition_frames", 0):
            streams[1][:, :, :condition_frames].copy_(shared_kwargs["lidar_condition_latents"])
        return latents

    def _prepare_multiview_latents(
        self,
        *,
        encoded_target: torch.Tensor | None,
        condition_indexes: Sequence[int],
        num_views: int,
        num_frames: int,
        height: int,
        width: int,
        generator: torch.Generator,
        injected_latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_frames_per_view = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            1,
            self.transformer.latent_channel_size,
            num_views * latent_frames_per_view,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        # The denoising state stays in sampling_dtype; predict_noise casts only the
        # transformer input to the model dtype.
        if injected_latents is None:
            noise = randn_tensor(shape, generator=generator, device=self.device, dtype=self.sampling_dtype)
        else:
            noise = injected_latents.to(device=self.device, dtype=self.sampling_dtype)
            if tuple(noise.shape) != shape:
                raise ValueError(
                    "Cosmos3 multiview injected latents have the wrong shape: "
                    f"expected={shape}, got={tuple(noise.shape)}."
                )
        condition_mask = torch.zeros(1, 1, shape[2], 1, 1, device=self.device, dtype=self.sampling_dtype)
        condition_latents = torch.zeros_like(noise)
        if condition_indexes:
            if encoded_target is None:
                raise ValueError("Cosmos3 multiview condition indexes require per-camera vision inputs.")
            encoded = encoded_target
            if tuple(encoded.shape) != shape:
                raise ValueError(
                    f"Cosmos3 multiview target VAE latent shape mismatch: expected={shape}, got={tuple(encoded.shape)}."
                )
            for index in condition_indexes:
                condition_mask[:, :, index] = 1
                condition_latents[:, :, index : index + 1] = encoded[:, :, index : index + 1]
        latents = condition_mask * condition_latents + (1.0 - condition_mask) * noise
        return latents, 1.0 - condition_mask, condition_latents

    def _camera_pixels(self, request: _EncodeRequest, views: Sequence[Mapping[str, Any]], field: str) -> torch.Tensor:
        """Normalized camera-major pixels of ``views`` for one input field."""
        vision = field == "vision"
        return self._prepare_camera_major_pixels(
            views,
            field=field,
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            keep_first=vision and request.condition_video_as_image,
            require_complete=vision and request.completion,
        )

    def _encode_inputs(self, *, request: _EncodeRequest, grid: VaeGrid | None) -> _EncodedInputs:
        """VAE-encode the camera vision/control inputs and the LiDAR inputs.

        Without a grid every rank encodes every input. With a grid each sensor
        is encoded on its owner and broadcast. Every rank returns the same
        latents. Vision is encoded only when frames are conditioned on it.
        """
        if grid is not None:
            return self._owner_encode_inputs(request=request, grid=grid)
        num_views, num_frames = len(request.views), request.num_frames
        target_pixels = self._camera_pixels(request, request.views, "vision") if request.has_vision else None
        control_pixels = self._camera_pixels(request, request.views, "control") if request.selected_hints else None
        vision = None
        if request.condition_indexes and target_pixels is not None:
            vision = self._encode_multiview_video(target_pixels, num_views=num_views, frames_per_view=num_frames)
        control = (
            self._encode_multiview_video(control_pixels, num_views=num_views, frames_per_view=num_frames)
            if control_pixels is not None
            else None
        )
        del target_pixels, control_pixels
        lidar_control = lidar_condition = None
        if request.lidar is not None:
            sweeps = required_lidar_sweeps(num_frames, request.frame_rate, self.multiview_config["lidar"]["fps"])
            lidar_frames = load_lidar_frames(request.lidar["control_path"], num_sweeps=sweeps)
            lidar_control = self.lidar_encoder(lidar_frames).to(device=self.device, dtype=self.dtype)
            del lidar_frames
            if request.lidar.get("condition_path") is not None:
                lidar_condition = self._encode_lidar_condition(request.lidar, sweeps, lidar_control.shape)
        return _EncodedInputs(
            vision=vision, control=control, lidar_control=lidar_control, lidar_condition=lidar_condition
        )

    def _plan_owner_work(self, *, request: _EncodeRequest, grid: VaeGrid, fields: Sequence[str]) -> _OwnerWork:
        """Describe every sensor's VAE work and latent geometry, then place it on the grid."""
        num_views, num_frames = len(request.views), request.num_frames
        height, width = request.height, request.width
        camera_shape = (
            1,
            self.transformer.latent_channel_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        # Media of any source size or duration is normalized to the request
        # geometry first, so cost and latent shapes use that geometry.
        items = [VaeWork(str(i), "camera", num_frames, height, width, encodes=len(fields)) for i in range(num_views)]
        specs: dict[str, list[FieldSpec]] = {
            str(i): [(field, camera_shape, self.sampling_dtype) for field in fields] for i in range(num_views)
        }
        sweeps = 0
        lidar_shape = None
        if request.lidar is not None:
            config = self.lidar_encoder.config
            sweeps = required_lidar_sweeps(num_frames, request.frame_rate, config["fps"])
            h, w = config["network_config"]["resolution"]
            sh, sw = config["spatial_compression"]
            lidar_shape = (1, config["latent_channels"], sweeps, h // sh, w // sw)
            specs["lidar"] = [("control", lidar_shape, self.dtype)]
            count = 0
            if request.lidar.get("condition_path") is not None:
                count = request.lidar.get("num_conditional_sweeps", COSMOS3_MULTIVIEW_DEFAULT_LIDAR_CONDITION_SWEEPS)
                if not 0 < count < sweeps:
                    raise ValueError("LiDAR conditional sweeps must leave at least one generated sweep.")
                specs["lidar"].append(("vision", (*lidar_shape[:2], count, *lidar_shape[3:]), self.sampling_dtype))
            decode = request.lidar.get("return_output", False)
            items.append(VaeWork("lidar", "lidar", sweeps, h, w, encodes=1 + count / sweeps, decode=decode))
        plan = plan_vae_work(items, grid.sp.world_size, grid.cfg.world_size)
        return _OwnerWork(plan=plan, items=items, specs=specs, lidar_sweeps=sweeps, lidar_shape=lidar_shape)

    def _encode_owned_item(
        self,
        key: str,
        field: str,
        *,
        request: _EncodeRequest,
        work: _OwnerWork,
    ) -> torch.Tensor:
        """Encode one field of an item owned by this rank."""
        if key == "lidar":
            assert request.lidar is not None and work.lidar_shape is not None
            if field == "control":
                frames = load_lidar_frames(request.lidar["control_path"], num_sweeps=work.lidar_sweeps)
                return self.lidar_encoder(frames).to(device=self.device, dtype=self.dtype)
            return self._encode_lidar_condition(request.lidar, work.lidar_sweeps, work.lidar_shape)
        index = int(key)
        pixels = self._camera_pixels(request, request.views[index : index + 1], field)
        return self._encode_video_tensor(pixels)

    def _owner_encode_inputs(self, *, request: _EncodeRequest, grid: VaeGrid) -> _EncodedInputs:
        views = request.views
        fields = (["vision"] if request.has_vision and request.condition_indexes else []) + (
            ["control"] if request.selected_hints else []
        )
        work = self._plan_owner_work(request=request, grid=grid, fields=fields)
        plan = work.plan
        metadata = (tuple(view["camera_key"] for view in views), work.items, plan.owners, plan.order, work.specs)
        exchange = VaeOwnerExchange(self.device, grid, metadata)
        buffers = receive_buffers(plan, work.specs, exchange.rank, self.device)

        def encode(key: str, field: str) -> torch.Tensor:
            return self._encode_owned_item(key, field, request=request, work=work)

        values = broadcast_owned_work(plan=plan, specs=work.specs, exchange=exchange, buffers=buffers, encode=encode)
        exchange.finish()
        cameras = {field: torch.cat([values[str(i), field] for i in range(len(views))], dim=2) for field in fields}
        lidar_control = values.get(("lidar", "control"))
        if lidar_control is not None:
            # The LiDAR encoder returns channels-last storage. Restore it after
            # contiguous transport: patchify/projection otherwise selects a
            # different GEMM layout and changes rounding.
            lidar_control = lidar_control.contiguous(memory_format=torch.channels_last_3d)
        if request.has_vision and not request.condition_indexes:
            # Vision is not encoded without conditioned frames; still validate
            # its media on every rank, as the serial path does.
            self._camera_pixels(request, views, "vision")
        return _EncodedInputs(
            vision=cameras.get("vision"),
            control=cameras.get("control"),
            lidar_control=lidar_control,
            lidar_condition=values.get(("lidar", "vision")),
            plan=plan,
        )

    def _decode_outputs(
        self,
        *,
        grid: VaeGrid | None,
        plan: VaePlan | None,
        final_targets: Sequence[torch.Tensor],
        num_views: int,
        latent_frames_per_view: int,
        lidar_request: Mapping[str, Any] | None,
    ) -> _DecodedOutputs | None:
        """Decode the camera video, and LiDAR when requested, clamped to [-1, 1].

        Without a grid every rank decodes everything. With a grid each item is
        decoded on its owner and gathered on grid rank 0; other ranks return None.
        """
        return_lidar = lidar_request is not None and lidar_request.get("return_output", False)
        if grid is None:
            video = self._decode_multiview_latents(
                final_targets[0],
                num_views=num_views,
                latent_frames_per_view=latent_frames_per_view,
            ).clamp_(-1, 1)
            lidar = self.lidar_decoder(final_targets[1]) if return_lidar else None
            return _DecodedOutputs(video=video, lidar=lidar)
        assert plan is not None
        owner_video, lidar = self._owner_decode_outputs(
            grid=grid,
            plan=plan,
            final_targets=final_targets,
            num_views=num_views,
            latent_frames_per_view=latent_frames_per_view,
            return_lidar=return_lidar,
        )
        if owner_video is None:
            return None
        owner_video.clamp_(-1, 1)
        return _DecodedOutputs(video=owner_video, lidar=lidar)

    def _owner_decode_outputs(
        self,
        *,
        grid: VaeGrid,
        plan: VaePlan,
        final_targets: Sequence[torch.Tensor],
        num_views: int,
        latent_frames_per_view: int,
        return_lidar: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        latents = final_targets[0]
        frames = latent_frames_per_view
        shape = (
            latents.shape[0],
            3,
            (frames - 1) * self.vae_scale_factor_temporal + 1,
            latents.shape[3] * self.vae_scale_factor_spatial,
            latents.shape[4] * self.vae_scale_factor_spatial,
        )
        specs = {str(i): (shape, self.vae.dtype) for i in range(num_views)}
        if return_lidar:
            projection = self.lidar_decoder.config["range_projection"]
            h, w = projection["native_height"], projection["semantic_width"]
            specs["lidar"] = ((final_targets[1].shape[0], 3, final_targets[1].shape[2], h, w), torch.float32)
        # Queue every receive before the output rank decodes its own items;
        # otherwise its first decode delays every remote transfer.
        order = sorted((key for key in plan.order if key in specs), key=lambda key: plan.owners[key] == 0)
        exchange = VaeOwnerExchange(self.device, grid, (plan.owners, order, specs))
        video = lidar = None
        destinations = None
        if exchange.rank == 0:
            video = torch.empty(
                (*shape[:2], num_views * shape[2], *shape[3:]), dtype=self.vae.dtype, device="cpu", pin_memory=True
            )
            destinations = {str(i): video.narrow(2, i * shape[2], shape[2]) for i in range(num_views)}
            if "lidar" in specs:
                lidar = torch.empty(specs["lidar"][0], dtype=torch.float32, device="cpu", pin_memory=True)
                destinations["lidar"] = lidar

        def decode(key: str) -> torch.Tensor:
            if key == "lidar":
                return self.lidar_decoder(final_targets[1])
            return self._decode_latents(latents.narrow(2, int(key) * frames, frames))

        decode_and_gather(
            order=order,
            owners=plan.owners,
            specs=specs,
            exchange=exchange,
            decode=decode,
            destinations=destinations,
            device=self.device,
        )
        exchange.finish(cpu=True)
        return video, lidar

    def _encode_lidar_condition(
        self, lidar_request: Mapping[str, Any], num_sweeps: int, target_shape: Sequence[int]
    ) -> torch.Tensor:
        """Encode the measured sweeps that condition the start of the generated LiDAR.

        The reference encodes the prefix inside an otherwise empty (zero) target
        clip. The streaming tokenizer is frame-causal within a chunk, but how much
        history a chunk keeps depends on that chunk's length, so encoding the bare
        prefix would change the latents of a trailing partial chunk. Zero-padding
        the prefix to the target's chunk boundary reproduces the full clip's chunk
        lengths for every chunk holding prefix sweeps; the causal padding cannot
        affect the prefix latents.
        """
        count = lidar_request.get("num_conditional_sweeps", COSMOS3_MULTIVIEW_DEFAULT_LIDAR_CONDITION_SWEEPS)
        if not 0 < count < num_sweeps:
            raise ValueError(
                f"Cosmos3 lidar.num_conditional_sweeps={count} must leave at least one of the request's "
                f"{num_sweeps} LiDAR sweeps to generate."
            )
        frames = load_lidar_frames(lidar_request["condition_path"], num_sweeps=count)
        chunk = self.lidar_encoder.config["streaming_chunk_frames"]
        padded = min(math.ceil(count / chunk) * chunk, num_sweeps)
        if padded > count:
            frames = torch.cat([frames, frames.new_zeros(frames.shape[0], padded - count, *frames.shape[2:])], dim=1)
        latents = self.lidar_encoder(frames)[:, :, :count].to(device=self.device, dtype=self.sampling_dtype)
        expected = (*target_shape[:2], count, *target_shape[3:])
        if tuple(latents.shape) != expected:
            raise ValueError(f"Cosmos3 LiDAR condition latents must have shape {expected}, got {tuple(latents.shape)}.")
        return latents

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("Cosmos3MultiviewPipeline supports exactly one prompt per request.")
        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            request_per_view_negative_prompt = None
        elif isinstance(prompt_data, Mapping):
            reject_multiview_negative_fields(prompt_data, "prompt")
            request_per_view_negative_prompt = prompt_data.get("per_view_negative_prompt")
        else:
            raise TypeError(f"Unsupported Cosmos3 multiview prompt type: {type(prompt_data).__name__}.")

        sp = req.sampling_params
        multiview, views = self._parse_multiview_request(sp)
        num_views = len(views)
        deployment = self.multiview_config
        defaults = deployment["inference_defaults"]
        lidar_request = self._get_sp_param(sp, "lidar", None)
        selected_hints = [key for key in COSMOS3_TRANSFER_HINT_KEYS if self._get_sp_param(sp, key, None) is not None]
        requested_num_frames = multiview.get("num_frames")
        if requested_num_frames is None:
            requested_num_frames = sp.num_frames
        num_frames = _resolve_multiview_num_frames(requested_num_frames, self.vae_scale_factor_temporal)
        resolution, aspect_ratio, width, height = _resolve_multiview_geometry(
            sp, multiview, views, default_resolution=defaults["resolution"]
        )
        frame_rate_value = self._get_sp_param(sp, "resolved_frame_rate", None)
        if frame_rate_value is None:
            frame_rate_value = self._get_sp_param(sp, "frame_rate", None)
        if frame_rate_value is None:
            frame_rate_value = self._get_sp_param(sp, "fps", None)
        if frame_rate_value is None:
            frame_rate_value = defaults["fps"]
        frame_rate = _resolve_multiview_frame_rate(frame_rate_value, defaults["fps"])

        condition_video_as_image = as_bool(multiview.get("condition_video_as_image"), False)
        known_views = [index for index, view in enumerate(views) if self._view_value(view, "vision") is not None]
        has_vision = bool(known_views)
        completion = has_vision and len(known_views) < num_views
        vision_kind = _media_kind(self._view_value(views[known_views[0]], "vision")) if has_vision else None
        latent_frames_per_view = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_t = num_views * latent_frames_per_view
        raw_indexes = multiview.get("condition_frame_indexes_vision")
        if raw_indexes is None:
            if not has_vision:
                local_indexes = []
            elif condition_video_as_image or vision_kind == "image":
                local_indexes = [0]
            else:
                local_indexes = [0, 1]
        else:
            local_indexes = _normalize_local_condition_indexes(raw_indexes)
        if completion:
            if raw_indexes is not None and local_indexes:
                raise ValueError("View completion conditions all frames of known views; omit condition frame indexes.")
            condition_indexes = [
                view * latent_frames_per_view + frame for view in known_views for frame in range(latent_frames_per_view)
            ]
        else:
            if any(index < 0 or index >= latent_frames_per_view for index in local_indexes):
                raise ValueError("Camera condition frame index is outside the generated latent clip.")
            condition_indexes = expand_multiview_condition_frame_indexes(local_indexes, num_views, latent_t)

        grid = vae_grid(self.device, self.dtype, self.vae, num_views + (lidar_request is not None))
        request = _EncodeRequest(
            views=views,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            has_vision=has_vision,
            condition_indexes=condition_indexes,
            condition_video_as_image=condition_video_as_image,
            completion=completion,
            selected_hints=selected_hints,
            lidar=lidar_request,
        )
        encoded = self._encode_inputs(request=request, grid=grid)

        generator = sp.generator
        seed = self._resolve_seed(sp, generator)
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        injected_latents = sp.latents if isinstance(sp.latents, torch.Tensor) else None
        latents, velocity_mask, condition_latents = self._prepare_multiview_latents(
            encoded_target=encoded.vision,
            condition_indexes=condition_indexes,
            num_views=num_views,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator,
            injected_latents=injected_latents,
        )
        control_latents = encoded.control
        lidar_control_latents = encoded.lidar_control
        lidar_condition_latents = encoded.lidar_condition
        vae_plan = encoded.plan
        del encoded
        if control_latents is not None and control_latents.shape != latents.shape:
            raise ValueError(
                "Cosmos3 multiview WSM and target latent shapes must match: "
                f"control={tuple(control_latents.shape)}, target={tuple(latents.shape)}."
            )
        actual_latent_t = int(latents.shape[2])
        max_sequence_length = int(
            self._get_sp_param(sp, "max_sequence_length", COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH)
            or COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH
        )
        if max_sequence_length > COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "Cosmos3 multiview max_sequence_length cannot exceed the variant ceiling the sparse "
                f"attention is sized for: requested={max_sequence_length}, "
                f"ceiling={COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH}."
            )
        patch_h, patch_w, _, _ = self.transformer._pad_to_patch_size(latents.shape[3], latents.shape[4])
        camera_shape = (actual_latent_t, patch_h, patch_w)
        camera_rate = self.vae_scale_factor_temporal / frame_rate
        items = (
            [MaskItem(camera_shape, num_views, is_control=True, seconds_per_frame=camera_rate)]
            if control_latents is not None
            else []
        )
        items.append(MaskItem(camera_shape, num_views, seconds_per_frame=camera_rate))
        targets = [latents]
        lidar_condition_frames = 0
        if lidar_request is not None:
            assert lidar_control_latents is not None
            lidar_config = deployment["lidar"]
            # Continue the request RNG after camera noise; reseeding would
            # reuse its initial stream and discard caller-supplied RNG state.
            lidar_noise = randn_tensor(
                lidar_control_latents.shape,
                generator=generator,
                device=self.device,
                dtype=self.sampling_dtype,
            )
            if lidar_request.get("condition_path") is not None:
                assert lidar_condition_latents is not None
                lidar_condition_frames = int(lidar_condition_latents.shape[2])
                # As in the reference, the sample starts clean on the measured prefix.
                lidar_noise[:, :, :lidar_condition_frames] = lidar_condition_latents
            targets.append(lidar_noise)
            lt, lh, lw = lidar_noise.shape[2:]
            del lidar_noise
            # LiDAR has its own patch size (1x1 on Phase-2 checkpoints), so
            # its token grid and mRoPE positions do not follow the camera's.
            lhp, lwp = patch_grid(lh, lw, self.transformer.lidar_patch_hw)
            for is_control in (True, False):
                items.append(
                    MaskItem(
                        (lt, lhp, lwp),
                        1,
                        view_offset=num_views,
                        is_control=is_control,
                        is_lidar=True,
                        seconds_per_frame=lidar_config["temporal_compression_factor"] / lidar_config["fps"],
                    )
                )
        layout = MultiviewLayout(
            cross_view_past_window_seconds=self.multiview_cross_view_past_window_seconds,
            backend=self.multiview_backend,
            items=tuple(items),
            max_und_tokens=DEFAULT_MAX_UND_TOKENS * num_views,
        )

        # Match training caption dropout unless an explicit per-camera negative is supplied.
        per_view_negative_prompt = request_per_view_negative_prompt
        if per_view_negative_prompt is None:
            per_view_negative_prompt = self._get_sp_param(sp, "per_view_negative_prompt", None)
        if per_view_negative_prompt is not None and not isinstance(per_view_negative_prompt, str):
            raise ValueError("Cosmos3 per_view_negative_prompt must be a string.")
        emphasis = as_bool(
            self._get_sp_param(sp, "emphasize_control_in_prompt", defaults["emphasize_control_in_prompt"]),
            defaults["emphasize_control_in_prompt"],
        )
        suffix = (
            control_emphasis(selected_hints[0], joint=lidar_request is not None)
            if selected_hints and emphasis
            else None
        )
        # Per-camera captions carry the sampled-rig and current-camera headers
        # and whole-second durations, exactly as in training and the reference.
        captions = format_rig_view_captions([view["prompt"] for view in views], [view["camera_key"] for view in views])
        system_prompt = _multiview_system_prompt(transfer=bool(selected_hints), joint=lidar_request is not None)
        branches = []
        for caption in captions:
            branches.append(
                self._format_and_tokenize_prompts(
                    caption,
                    per_view_negative_prompt or "",
                    num_frames,
                    frame_rate,
                    height,
                    width,
                    max_sequence_length,
                    sp,
                    use_system_prompt=True,
                    system_prompt=system_prompt,
                    prompt_suffix=suffix,
                    use_duration_template=True,
                    use_resolution_template=True,
                    negative_metadata_mode="same" if per_view_negative_prompt else "none",
                    aspect_ratio_override=aspect_ratio,
                    truncate_duration=True,
                )
            )

        # Compact each camera's real caption tokens into one sequence. The host-side
        # lengths give each segment its own causal UND pass and position origin.
        def combine_branch(ids_index: int, mask_index: int) -> tuple[torch.Tensor, tuple[int, ...]]:
            ids = [branch[ids_index][branch[mask_index].bool()].reshape(1, -1) for branch in branches]
            return torch.cat(ids, dim=1), tuple(tokens.shape[1] for tokens in ids)

        cond_ids, cond_lengths = combine_branch(0, 1)
        uncond_ids, uncond_lengths = combine_branch(2, 3)

        guidance_scale = self._resolve_guidance_scale(sp, defaults["guidance"])
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("Cosmos3 multiview guidance must be finite and non-negative.")
        num_inference_steps = int(sp.num_inference_steps or defaults["num_steps"])
        flow_shift = float(self._get_sp_param(sp, "flow_shift", defaults["shift"]))
        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps
        self._set_flow_shift(flow_shift)
        # As in reference rectified-flow inference, the schedule depends only on
        # steps and shift. Requests and older exports may carry an EDM-style
        # sigma_max (e.g. 80); it is accepted for compatibility and ignored.
        self._set_timesteps(num_inference_steps, device=self.device, shift=flow_shift)

        rig_view_embedding = deployment["rig_view_embedding"]
        # Physical rig IDs, not request positions: subsets and reordered views
        # keep each camera's trained row. LiDAR uses the table's final row.
        rig_view_ids = torch.tensor(
            [rig_view_embedding["camera_ids"][view["camera_key"]] for view in views],
            dtype=torch.long,
            device=self.device,
        )
        shared_kwargs = {
            "_multiview_caption_lengths": {
                cond_ids.data_ptr(): cond_lengths,
                uncond_ids.data_ptr(): uncond_lengths,
            },
            "noisy_frame_mask": velocity_mask,
            "packed_shapes": tuple(tuple(tensor.shape[1:]) for tensor in targets),
            "lidar_control_latents": lidar_control_latents,
            "multiview_layout": layout,
            "rig_view_ids": rig_view_ids,
            "lidar_condition_frames": lidar_condition_frames,
            "lidar_condition_latents": lidar_condition_latents,
        }
        # Transfer ownership of the initial state to the denoising loop. The
        # caller must not retain either source tensors or the packed sample.
        initial_state = [pack_state(targets)]
        del targets, latents
        packed = self.diffuse_transfer(
            latents=initial_state.pop(),
            timesteps=self.scheduler.timesteps,
            cond_ids=cond_ids,
            # The multiview transformer reads caption lengths, not a text mask.
            cond_mask=None,
            uncond_ids=uncond_ids,
            uncond_mask=None,
            guidance_scale=guidance_scale,
            control_guidance=float(self._get_sp_param(sp, "control_guidance", defaults["control_guidance"])),
            control_guidance_interval=self._get_sp_param(
                sp, "control_guidance_interval", defaults["control_guidance_interval"]
            ),
            guidance_interval=self._get_sp_param(sp, "guidance_interval", defaults["guidance_interval"]),
            control_latents=[control_latents] if control_latents is not None else [],
            shared_kwargs=shared_kwargs,
            velocity_mask=velocity_mask,
            condition_latents=condition_latents,
            generator=generator,
            normalize_cfg=as_bool(
                self._get_sp_param(sp, "normalize_cfg", defaults["normalize_cfg"]), defaults["normalize_cfg"]
            ),
            open_guidance_interval=True,
            text_cfg_below_one=True,
        )
        final_targets = unpack_state(packed, shared_kwargs["packed_shapes"])
        del condition_latents, control_latents, lidar_control_latents, lidar_condition_latents, shared_kwargs
        decoded = self._decode_outputs(
            grid=grid,
            plan=vae_plan,
            final_targets=final_targets,
            num_views=num_views,
            latent_frames_per_view=latent_frames_per_view,
            lidar_request=lidar_request,
        )
        stage_durations = self.stage_durations if hasattr(self, "stage_durations") else None
        if decoded is None:
            # Only grid rank 0 holds the gathered output.
            return DiffusionOutput(output=None, stage_durations=stage_durations)
        payload = {"video": decoded.video}
        lidar_metadata = {}
        if lidar_request is not None and lidar_request.get("return_output", False):
            lidar_output = decoded.lidar
            assert lidar_output is not None
            payload["lidar"] = lidar_output
            lidar_config = self.lidar_decoder.config
            lidar_metadata = {
                "lidar": {
                    "fps": lidar_config["fps"],
                    "num_frames": lidar_output.shape[2],
                    "shape": list(lidar_output.shape),
                    "dtype": "float32",
                    "channels": ["range", "intensity", "validity"],
                    "units": ["metres", "unit", "binary" if lidar_config["apply_validity_mask"] else "probability"],
                    "validity_threshold": lidar_config["range_projection"].get("validity_threshold", 0.5),
                    "apply_validity_mask": lidar_config["apply_validity_mask"],
                    "start_time_seconds": 0.0,
                    "range_projection": dict(lidar_config["range_projection"]),
                }
            }
        return DiffusionOutput(
            output={
                "payload": payload,
                "metadata": {
                    **lidar_metadata,
                    "multiview": {
                        "cameras": [view["camera_key"] for view in views],
                        "frames_per_view": num_frames,
                        "fps": frame_rate,
                        "resolution": resolution,
                        "aspect_ratio": aspect_ratio,
                        "width": width,
                        "height": height,
                    },
                },
            },
            stage_durations=stage_durations,
        )


__all__ = [
    "Cosmos3MultiviewPipeline",
    "get_cosmos3_ir_op_priority_func",
    "get_cosmos3_multiview_pre_process_func",
    "get_cosmos3_multiview_post_process_func",
]
