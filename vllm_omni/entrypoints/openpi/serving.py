# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Serving layer for robot policy inference via `/v1/realtime/robot/openpi`.

Flow: raw obs → engine request → actions.
The loaded policy model owns dataset transforms inside its pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from typing import Any, TypeAlias

import numpy as np
from omegaconf import OmegaConf
from vllm.logger import init_logger

logger = init_logger(__name__)

# Annotated so mypy reads it as a type alias rather than a module variable.
ActionOutput: TypeAlias = np.ndarray | dict[str, np.ndarray]

# Inference parameters live in a dedicated single-stage wire namespace so
# arbitrary robot feature names cannot be confused with engine controls.
_INFERENCE_PARAM_NAMES = (
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


def _extract_inference_params(
    obs: Mapping[str, Any],
    policy_server_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the strict per-request ``sampling_params`` namespace.

    Unknown names, invalid values, and legacy top-level controls fail before the
    request reaches the engine. This prevents a typo from silently falling back
    to a deploy-config default.
    """
    legacy = sorted(key for key in _INFERENCE_PARAM_NAMES if key in obs)
    if legacy:
        raise ValueError(f"Inference controls {legacy!r} must be nested under observation['sampling_params'].")

    if "sampling_params" not in obs:
        return {}
    raw_params = obs["sampling_params"]
    if not isinstance(raw_params, Mapping):
        raise ValueError(f"sampling_params must be a mapping, got {type(raw_params).__name__}.")

    unknown = sorted(set(raw_params) - set(_INFERENCE_PARAM_NAMES))
    if unknown:
        # Naming the whole namespace beats guessing at the intent: it is two
        # keys long, so the supported set is shorter than a fuzzy suggestion
        # would be and it stays useful for a key that resembles neither.
        raise ValueError(f"Unknown sampling_params key(s): {unknown}. Supported: {sorted(_INFERENCE_PARAM_NAMES)}.")

    params: dict[str, Any] = {}
    for key in _INFERENCE_PARAM_NAMES:
        if key not in raw_params:
            continue
        value = raw_params[key]
        if key == "num_inference_steps":
            # bool is an int subclass, and msgpack round-trips numpy scalars.
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"num_inference_steps must be a positive integer, got {value!r}.")
            value = int(value)
        elif key == "noise":
            try:
                array = np.asarray(value)
            except Exception as exc:
                raise ValueError("noise must be a numeric rank-3 array.") from exc
            if array.ndim != 3 or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
                raise ValueError(
                    f"noise must be a real numeric rank-3 array, got shape={array.shape}, dtype={array.dtype}."
                )
            if not np.isfinite(array).all():
                raise ValueError("noise must contain only finite values.")
            expected_horizon = (policy_server_config or {}).get("action_horizon")
            expected_dim = (policy_server_config or {}).get(
                "max_action_dim", (policy_server_config or {}).get("action_dim")
            )
            if expected_horizon is not None and expected_dim is not None:
                expected_shape = (1, int(expected_horizon), int(expected_dim))
                if array.shape != expected_shape:
                    raise ValueError(f"noise must have shape {expected_shape}, got {array.shape}.")
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

        inference_params = _extract_inference_params(obs, self.policy_server_config.values)
        robot_obs = {key: value for key, value in obs.items() if key != "sampling_params"}
        extra_args = {
            "reset": reset,
            "session_id": session_id,
            "robot_obs": robot_obs,
        }
        if "noise" in inference_params:
            extra_args["noise"] = inference_params["noise"]

        prompt = obs.get("prompt", "")
        sampling_params = OmniDiffusionSamplingParams(
            num_inference_steps=inference_params.get("num_inference_steps"),
            extra_args=extra_args,
        )
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
