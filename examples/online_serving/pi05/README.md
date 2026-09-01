# π0.5 VLA — OpenPI realtime serving

[π0.5](https://www.physicalintelligence.company/blog/pi05) is a Vision-Language-Action
model from Physical Intelligence: multi-camera images + a language instruction +
robot proprioceptive state → a continuous action chunk via flow-matching denoising
(it does **not** emit text tokens). This example serves π0.5 over the OpenPI realtime
websocket protocol at `/v1/realtime/robot/openpi`.

The wire protocol is identical to π0's. What differs is server-side: π0.5 pads text
to 200 tokens instead of 48, has no `state_proj`, and instead **discretizes the state
into 256 bins and serializes it into the prompt**, so the model sees only images and
language tokens. See `recipes/lerobot/Pi05.md` for the full π0 / π0.5 comparison.

## Install extras

The core `pip install -e .` does not include the OpenPI client used here:

- `openpi-client` (from the [openpi](https://github.com/Physical-Intelligence/openpi)
  repo: `pip install -e packages/openpi-client`), `websockets`, `msgpack`, `msgpack-numpy`

## Weights

π0.5 uses the LeRobot `lerobot/pi05_base` checkpoint (HF, ~14.5 GB, float32). Either
let the server download it (`MODEL=lerobot/pi05_base`) or point at a local copy
(`MODEL=/path/to/pi05_base`). The PaliGemma tokenizer is pulled from
`google/paligemma-3b-pt-224`, which is gated — accept its license on HF and log in
(`hf auth login`) first.

## Run the server

```bash
vllm serve lerobot/pi05_base --omni --port 8000 \
    --served-model-name pi05 \
    --deploy-config vllm_omni/deploy/pi05.yaml \
    --enforce-eager --disable-log-stats
```

The deploy config (`vllm_omni/deploy/pi05.yaml`) declares a single diffusion stage
(`Pi05Pipeline`), float32, `max_num_seqs: 1`, and the `policy_server_config` handshake
metadata (3 cameras, `joint_position`, action horizon 50, action dim 32).

### Commonly adjusted `model_config` keys

These live under `stages[0].model_config` in `vllm_omni/deploy/pi05.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `chunk_size` | `50` | Action-chunk length (timesteps) the model predicts per inference. |
| `num_inference_steps` | `10` | Flow-matching Euler denoising steps. |
| `max_action_dim` | `32` | Action dimensionality (state/action are padded to this). |
| `max_state_dim` | `32` | Proprioceptive-state dimensionality (zero-padded to this). |
| `image_resolution` | `[224, 224]` | Per-camera input size (square; SigLIP). |
| `tokenizer_max_length` | `200` | Max PaliGemma prompt tokens (π0 uses 48; the prompt also carries the state). |
| `state_num_bins` | `256` | Bins the normalized state is discretized into before it enters the prompt. |
| `max_cameras` | `3` | Camera slots the model attends to (real + `-1`-padded). |
| `image_feature_keys` | 3 `observation.images.*` keys | Camera order the model attends to. |
| `image_key_map` | `{}` | Map raw obs camera keys → `image_feature_keys` (empty = verbatim). |
| `use_relative_actions` | `false` | Predict actions relative to the current state. Leave false unless the checkpoint was trained that way. |
| `relative_exclude_joints` | `["gripper"]` | Action names left absolute when `use_relative_actions` is on. |

The `policy_server_config` block below them is the OpenPI handshake metadata
advertised to the client; keep its `action_horizon` / `action_dim` /
`image_resolution` in sync with the `model_config` values above. The e2e test
`tests/e2e/online_serving/test_pi05_expansion.py::test_pi05_openpi_online` connects to a
live server and asserts the advertised metadata matches these `pi05.yaml` values.

## Run the client

```bash
python examples/online_serving/pi05/openpi_client.py --host 127.0.0.1 --port 8000 \
    --prompt "pick up the red block and place it in the bin"
```

It connects, prints the server metadata, sends robot observations, and prints the
returned `[action_horizon, action_dim] = [50, 32]` action chunks. Replace the blank
cameras / zero state in `_make_dummy_obs` with real frames (HWC uint8) and
proprioceptive state to drive a robot.

### Observation format

The client sends a flat dict per inference:

```python
{
    "observation.images.base_0_rgb":       np.uint8[H, W, 3],
    "observation.images.left_wrist_0_rgb": np.uint8[H, W, 3],
    "observation.images.right_wrist_0_rgb":np.uint8[H, W, 3],
    "state":   np.float32[state_dim],   # zero-padded to max_state_dim=32 server-side,
                                        # then normalized, binned and put in the prompt
    "prompt":  "pick up the red block",
    "session_id": "<uuid>",             # accepted but ignored (π0.5 is stateless)
}
```

Camera keys must match the server's `image_feature_keys` (the checkpoint's
`input_features` order). If your robot uses different camera names, set
`model_config.image_key_map` in `pi05.yaml` to map raw obs keys → feature keys.

## Correctness

π0.5's action chunks match the LeRobot `PI05Policy` reference under `lerobot/pi05_base`,
fixed noise and 10 denoising steps (`torch.allclose(atol=1e-4)`; 1/2/3 cameras, the
prompt tokens including the discretized state, and relative actions — see
`tests/diffusion/models/pi05/test_pi05_parity.py`). Unit coverage for the processor,
config validation and weight remapping lives in
`tests/diffusion/models/pi05/test_pi05_units.py`. An OpenPI websocket online-serving e2e
lives in `tests/e2e/online_serving/test_pi05_expansion.py::test_pi05_openpi_online`.

## Limitations

- **Normalization stats**: loaded from the checkpoint automatically. LeRobot keeps
  them out of `config.json` — `policy_preprocessor.json` holds the structure and a
  companion `policy_preprocessor_step_*.safetensors` holds the numbers — and both are
  read here, with the mode (`mean_std` / `min_max` / π0.5's default `quantile`) taken
  from the sidecar's `norm_map`. A `norm_stats` block in `config.json` or in the deploy
  yaml overrides the sidecar. `lerobot/pi05_base` declares no state file at all, i.e.
  identity normalization: the state passes through and the client is expected to send
  an already-normalized state. A normalization mode this implementation cannot
  reproduce raises at load time rather than being served with the wrong transform.
- **MEM and RTC are rejected, not silently ignored**: a checkpoint declaring
  short-horizon observation memory (`use_visual_memory`, `use_proprioceptive_memory`)
  or real-time chunking (`rtc_config`) fails at load time, because this stateless
  serving path keeps no per-session history. See `recipes/lerobot/Pi05.md`.
- **Functional path only**: no CUDA Graph capture, no Triton kernels, no prefix or
  encoder caching; `max_num_seqs: 1`. Those are tracked separately in
  [#6867](https://github.com/vllm-project/vllm-omni/issues/6867).
