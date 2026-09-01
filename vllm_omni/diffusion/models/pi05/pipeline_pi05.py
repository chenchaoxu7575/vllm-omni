# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 VLA pipeline for vllm-omni.

Entry point for ``DiffusionEngine.step() → pipeline.forward(req)``. Mirrors the
DreamZero contract and the π0 pipeline: the pipeline owns ALL preprocessing. It
reads the raw robot observation from
``req.sampling_params.extra_args["robot_obs"]`` (delivered by the OpenPI
realtime serving layer), builds model inputs, runs flow-matching denoising, and
returns ``DiffusionOutput(output={"actions": ndarray})``.

π0.5 is stateless across calls (no KV reuse, first-order Markov), so
``session_id`` / ``reset`` from the OpenPI protocol are accepted but ignored.

The post-processing order is load-bearing and matches LeRobot::

    unnormalize → absolute actions → to_cpu

``AbsoluteActionsProcessorStep`` must run *after* unnormalization, because a
relative-action checkpoint's ``norm_stats`` are computed in relative space.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.pi05.config import Pi05Config
from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from vllm_omni.diffusion.models.pi05.processor_pi05 import (
    Pi05RelativeActions,
    build_model_inputs,
)
from vllm_omni.diffusion.models.pi05_pipeline_config import PI05_PIPELINE as PI05_PIPELINE
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

# π0.5 pins the PaliGemma tokenizer (LeRobot hardcodes it too).
DEFAULT_PI05_TOKENIZER = "google/paligemma-3b-pt-224"


def _pi05_post_process(x):
    """Module-level identity post-process (picklable across the orchestrator's
    multiprocess boundary — a local closure is not)."""
    return x


def get_pi05_post_process_func(od_config: OmniDiffusionConfig):
    """π0.5 returns actions directly; post-processing is identity."""
    del od_config
    return _pi05_post_process


class Pi05Pipeline(nn.Module):
    """π0.5 VLA pipeline: raw robot obs → continuous action chunk.

    Registered as ``"Pi05Pipeline"`` in the diffusion registry.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.model_dir = self._resolve_model_dir(od_config.model)
        self.config = self._build_config(od_config)

        custom_args = od_config.custom_pipeline_args or {}
        self.tokenizer_source = str(custom_args.get("tokenizer", self._resolve_tokenizer_source()))

        self._torch_dtype = self._resolve_dtype(od_config)
        self._device = self._resolve_device(od_config)

        self.tokenizer = self._load_tokenizer()
        self.model = self._initialize_model()

        # One object serving both pipeline directions, matching LeRobot's
        # "same instance" pairing of Relative/AbsoluteActionsProcessorStep.
        self.relative_actions = Pi05RelativeActions(
            enabled=self.config.use_relative_actions,
            exclude_joints=self.config.relative_exclude_joints,
            action_names=self.config.action_feature_names,
            max_action_dim=self.config.max_action_dim,
        )
        if self.relative_actions.enabled:
            logger.info(
                "Pi05Pipeline: relative actions enabled — %d of %d action dims are "
                "state-relative (excluded joints: %s).",
                self.relative_actions.num_relative_dims,
                self.config.max_action_dim,
                self.config.relative_exclude_joints,
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_model_dir(model: str | None) -> str | None:
        """Return a local directory for ``model``; download an HF repo id if needed."""
        if not model:
            return None
        if os.path.isdir(model):
            return model
        # Via repo_utils' shared HfApi rather than huggingface_hub directly, so
        # the download carries vLLM's user agent like every other repo access.
        from vllm.transformers_utils.repo_utils import hf_api

        return hf_api().snapshot_download(
            repo_id=model,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )

    def _build_config(self, od_config: OmniDiffusionConfig) -> Pi05Config:
        """Build Pi05Config from deploy-yaml model_config, falling back to the
        checkpoint's config.json (raw LeRobot format).

        The deploy yaml is authoritative but must not silently drop
        checkpoint-derived fields it omits, so camera order, normalization stats
        and the relative-action contract are backfilled from the checkpoint.
        """
        if od_config.model_config:
            config = Pi05Config.from_model_config(dict(od_config.model_config))
            if self.model_dir:
                ckpt = Pi05Config.from_pretrained(self.model_dir)
                if not config.image_feature_keys:
                    config.image_feature_keys = ckpt.image_feature_keys
                    if not config.input_features:
                        config.input_features = ckpt.input_features
                if config.norm_stats is None:
                    config.norm_stats = ckpt.norm_stats
                    config.state_norm_stats = ckpt.state_norm_stats
                # A relative-action checkpoint served without the transform is
                # silently wrong, so the checkpoint's setting wins if the yaml
                # is silent about it.
                if not config.use_relative_actions and ckpt.use_relative_actions:
                    logger.warning(
                        "Pi05Pipeline: checkpoint declares use_relative_actions=True but the "
                        "deploy yaml did not; honouring the checkpoint. Its norm_stats are in "
                        "relative action space."
                    )
                    config.use_relative_actions = True
                    config.relative_exclude_joints = ckpt.relative_exclude_joints
                    if config.action_feature_names is None:
                        config.action_feature_names = ckpt.action_feature_names
                    config._validate_relative_actions()
            return config
        if self.model_dir:
            return Pi05Config.from_pretrained(self.model_dir)
        return Pi05Config()

    def _resolve_tokenizer_source(self) -> str:
        """Prefer the checkpoint dir if it ships tokenizer files; else PaliGemma."""
        if self.model_dir and os.path.isdir(self.model_dir):
            if os.path.exists(os.path.join(self.model_dir, "tokenizer_config.json")):
                return self.model_dir
        return DEFAULT_PI05_TOKENIZER

    @staticmethod
    def _resolve_dtype(od_config: OmniDiffusionConfig) -> torch.dtype:
        dt = od_config.dtype
        if isinstance(dt, torch.dtype):
            return dt
        return getattr(torch, str(dt).split(".")[-1], torch.float32)

    @staticmethod
    def _resolve_device(od_config: OmniDiffusionConfig) -> torch.device:
        from vllm_omni.diffusion.distributed.utils import get_local_device

        try:
            return get_local_device()
        except Exception:  # noqa: BLE001
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        # padding_side="right" is part of the π0.5 spec: the prefix is a fixed
        # 200-token block whose live tokens must start at index 0.
        return AutoTokenizer.from_pretrained(self.tokenizer_source, padding_side="right")

    def has_real_checkpoint(self) -> bool:
        return bool(self.model_dir) and os.path.exists(os.path.join(self.model_dir, "model.safetensors"))

    def _initialize_model(self) -> Pi05ForActionPrediction:
        model = Pi05ForActionPrediction(self.config)
        if self.has_real_checkpoint():
            self._load_checkpoint(model)
        else:
            logger.info("Pi05Pipeline: no model.safetensors under %s; using random init.", self.model_dir)
        model.to(device=self._device, dtype=self._torch_dtype)
        model.eval()
        return model

    def _load_checkpoint(self, model: Pi05ForActionPrediction) -> None:
        import safetensors.torch

        path = os.path.join(self.model_dir, "model.safetensors")
        logger.info("Pi05Pipeline: loading π0.5 weights from %s", path)
        state = safetensors.torch.load_file(path)
        model.load_weights(list(state.items()))

    # ------------------------------------------------------------------
    # Framework weight-loading hook
    # ------------------------------------------------------------------
    def load_weights(self, weights=()):  # noqa: D401
        """No-op for the diffusion loader: π0.5 self-loads its checkpoint."""
        for _ in weights:
            pass
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        extra_args = getattr(req.sampling_params, "extra_args", None) or {}
        robot_obs = extra_args.get("robot_obs")

        if robot_obs is None:
            # Dummy warmup path (no obs): return zeros so engine warmup/capture
            # doesn't crash. Mirrors DreamZero's dummy-run handling.
            first_prompt = req.prompts[0] if req.prompts else ""
            prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
            num_steps = getattr(req.sampling_params, "num_inference_steps", None)
            if prompt == "dummy run" or num_steps == 1:
                logger.info("Pi05Pipeline: dummy warmup request without robot_obs — returning zeros.")
                return DiffusionOutput(
                    output={
                        "actions": np.zeros(
                            (self.config.chunk_size, self.config.max_action_dim),
                            dtype=np.float32,
                        )
                    },
                )
            return DiffusionOutput(
                error="Pi05Pipeline.forward requires sampling_params.extra_args['robot_obs'].",
            )

        # Input steps 1-7. Note: no state tensor comes back — π0.5 serializes
        # the (normalized, discretized) state into lang_tokens.
        images, image_masks, lang_tokens, lang_masks = build_model_inputs(
            robot_obs, self.config, self.tokenizer, self._device
        )

        noise = extra_args.get("noise")
        if noise is not None and not isinstance(noise, torch.Tensor):
            noise = torch.as_tensor(noise, dtype=torch.float32, device=self._device)
        elif isinstance(noise, torch.Tensor):
            noise = noise.to(device=self._device, dtype=torch.float32)

        num_steps = extra_args.get("num_inference_steps")

        actions = self.model.sample_actions(
            images=images,
            image_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            noise=noise,
            num_steps=num_steps,
        )

        # Output step 1: unnormalize.
        actions = self.model._unnormalize_actions(actions)
        # Output step 2: relative → absolute, against the RAW state (the same
        # state the prompt encoded, before normalization).
        if self.relative_actions.enabled:
            actions = self.relative_actions.to_absolute(actions, robot_obs.get("state"))

        # Output step 3: to_cpu. (B=1, horizon, action_dim) → (horizon, action_dim).
        actions_np = actions.squeeze(0).float().cpu().numpy()

        return DiffusionOutput(output={"actions": actions_np})
