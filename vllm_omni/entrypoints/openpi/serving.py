# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for robot policy inference via `/v1/realtime/robot/openpi`.

Flow: raw obs → engine request → actions.
The loaded policy model owns dataset transforms inside its pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from vllm.logger import init_logger

logger = init_logger(__name__)

ActionOutput = np.ndarray | dict[str, np.ndarray]

# Inference parameters a client may override per observation. The π0 / π0.5
# pipelines already read these from ``sampling_params.extra_args``; without this
# forwarding they can only ever take their deploy-config defaults.
#
# Whitelisted rather than blanket-forwarded: an observation is mostly camera
# frames and robot state, and copying every key would turn any stray or
# misspelled one into an engine parameter.
_FORWARDED_INFERENCE_PARAMS = (
    # Flow-matching denoising steps. Fewer steps trades accuracy for latency,
    # which a robot may want to vary with its control budget.
    "num_inference_steps",
    # Initial noise for the denoising ODE. Pinning it makes a request
    # reproducible, which is what the parity tests compare against.
    "noise",
)


def _to_builtin_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: _to_builtin_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin_container(item) for item in value]
    return value


def _extract_inference_params(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the whitelisted per-request inference params out of an observation.

    Raises on a value the pipeline cannot honour instead of letting it through.
    ``num_inference_steps=0`` would return the initial noise unchanged — a
    well-shaped, finite, and completely wrong action chunk.
    """
    params: dict[str, Any] = {}
    for key in _FORWARDED_INFERENCE_PARAMS:
        if key not in obs:
            continue
        value = obs[key]
        if value is None:
            continue
        if key == "num_inference_steps":
            # bool is an int subclass, and msgpack round-trips numpy scalars.
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"num_inference_steps must be a positive integer, got {value!r}.")
            value = int(value)
        params[key] = value
    return params


@dataclass(frozen=True)
class PolicyServerConfig:
    """OpenPI policy server handshake config.

    Values are model-specific and must be provided by the loaded policy model.
    """

    values: dict[str, Any]

    @classmethod
    def from_model_config(cls, model_config: Any) -> PolicyServerConfig:
        if isinstance(model_config, Mapping):
            raw_config = model_config.get("policy_server_config")
        else:
            raw_config = getattr(model_config, "policy_server_config", None)

        if raw_config is None:
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        if isinstance(raw_config, cls):
            return raw_config
        if not isinstance(raw_config, Mapping):
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        return cls(_to_builtin_container(raw_config))

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin_container(self.values)


class ServingRealtimeRobotOpenPI:
    """Robot policy serving layer for OpenPI protocol.

    Model-specific transform/state lives in the diffusion pipeline.
    """

    def __init__(
        self,
        engine_client: Any,
        model_name: str | None = None,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name
        self.policy_server_config = self._get_policy_server_config(engine_client)
        self._request_counter = count()

    @classmethod
    def create_policy_server(
        cls,
        engine_client: Any,
        model_name: str | None = None,
    ) -> ServingRealtimeRobotOpenPI | None:
        try:
            return cls(engine_client=engine_client, model_name=model_name)
        except ValueError as exc:
            if "policy_server_config" not in str(exc):
                raise
            logger.info("Robot OpenPI serving disabled for model %s", model_name)
            return None

    @staticmethod
    def _get_policy_server_config(engine_client: Any) -> PolicyServerConfig:
        model_config = None
        get_od_config = getattr(engine_client, "get_diffusion_od_config", None)
        if callable(get_od_config):
            od_config = get_od_config()
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            for stage_config in getattr(engine_client, "stage_configs", []) or []:
                if getattr(stage_config, "stage_type", None) != "diffusion":
                    continue
                engine_args = getattr(stage_config, "engine_args", None)
                model_config = getattr(engine_args, "model_config", None)
                if model_config is not None:
                    break

        if model_config is None:
            od_config = getattr(engine_client, "od_config", None)
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            model_config = getattr(engine_client, "model_config", None)
        return PolicyServerConfig.from_model_config(model_config)

    def reset(self, obs: dict) -> None:
        """Compatibility hook; per-connection state lives in RobotRealtimeConnection."""

    async def infer(self, obs: dict, *, session_id: str, reset: bool) -> ActionOutput:
        """raw obs → engine → actions."""
        # Build request, run inference through AsyncOmni
        request = self._build_request(obs, session_id=session_id, reset=reset)
        result = None
        # OpenPI policy serving is one request -> one action reply. AsyncOmni
        # exposes an async iterator, so consume it to completion and use the
        # final output, matching other non-streaming OpenAI serving paths.
        async for output in self.engine_client.generate(
            prompt=request.prompt,
            request_id=request.request_id,
            sampling_params_list=[request.sampling_params],
        ):
            result = output
        if result is None:
            raise RuntimeError("Robot OpenPI request produced no output.")

        return self._extract_actions(result)

    def _next_request_id(self, session_id: str) -> str:
        return f"robot-{session_id}-{next(self._request_counter)}"

    def _build_request(self, obs: dict, *, session_id: str, reset: bool) -> Any:
        """Build engine request from raw robot obs.

        Returns an `OmniDiffusionRequest` payload consumed by
        `AsyncOmni.generate()` and routed to the diffusion stage.
        """
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        extra_args = {
            "reset": reset,
            "session_id": session_id,
            "robot_obs": obs,
        }
        extra_args.update(_extract_inference_params(obs))

        prompt = obs.get("prompt", "")
        sampling_params = OmniDiffusionSamplingParams(extra_args=extra_args)
        return OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=self._next_request_id(session_id),
        )

    def _extract_actions(self, result: Any) -> ActionOutput:
        """Extract actions from engine result."""
        multimodal_output = getattr(result, "multimodal_output", None)
        if not isinstance(multimodal_output, Mapping):
            raise RuntimeError("Missing multimodal_output in robot policy result")

        actions = multimodal_output.get("actions")
        if actions is None:
            raise RuntimeError("Missing multimodal_output['actions'] in robot policy result")
        if isinstance(actions, Mapping):
            return {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
        return np.asarray(actions, dtype=np.float32)
