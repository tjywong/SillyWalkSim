"""Turn a trained real-robot checkpoint (Onshape-Beam-Real or -Backward) into policy.npz for ../real_robot/drive_real.py.

The robot script then needs only numpy (no PyTorch, no mjlab). This script needs PyTorch, so run it on the cluster
(from the mujoco-hpc folder; send the checkpoint and the params folder next to it):

  python3 runIt.py --with runs/train-.../logs/rsl_rl/onshape_beam_real/<date>/model_1499.pt \\
    --with runs/train-.../logs/rsl_rl/onshape_beam_real/<date>/params \\
    09_onshape_robot/mjlab_task/export_policy.py model_1499.pt

or anywhere PyTorch is installed:  python export_policy.py <checkpoint.pt> [out.npz]

It reads the control rate, the beam-speed scale and the driving direction from params/env.yaml (saved by training
next to the checkpoint), so the robot script always uses the numbers the policy was trained with.
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

if len(sys.argv) not in (2, 3):
  sys.exit(__doc__)
ckpt = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) == 3 else ckpt.with_name("policy.npz")

state = torch.load(ckpt, map_location="cpu", weights_only=True)["actor_state_dict"]
weights = {k: v.numpy().astype(np.float64) for k, v in state.items()}
layers = sorted({int(m.group(1)) for k in weights if (m := re.match(r"mlp\.(\d+)\.weight", k))})
obs_size = weights[f"mlp.{layers[0]}.weight"].shape[1]
if obs_size != 16:
  sys.exit(f"This checkpoint's policy takes {obs_size} inputs, not 16: it is not a real-robot policy. Train "
           "Onshape-Beam-Real (real_task.py); the simulation-only tasks use readings the robot cannot measure.")

# settings the policy was trained with
yaml_path = next((p for p in (ckpt.parent / "params" / "env.yaml", Path("params/env.yaml")) if p.exists()), None)
if yaml_path is None:
  sys.exit("params/env.yaml not found next to the checkpoint: send the params folder too (see the top of this file).")
env_yaml = yaml_path.read_text()
decimation = int(re.search(r"^decimation: (\d+)", env_yaml, re.M).group(1))
timestep = float(re.search(r"timestep: ([0-9.e-]+)", env_yaml).group(1))
actions = env_yaml[env_yaml.index("\nactions:"):]
scale = float(re.search(r"\n    scale: ([0-9.e-]+)", actions).group(1))
m = re.search(r"forward_speed:.*?direction: (-?\d+)", env_yaml, re.S)
direction = int(m.group(1)) if m else 1

np.savez(
  out,
  obs_mean=weights["obs_normalizer._mean"].reshape(-1),
  obs_std=weights["obs_normalizer._std"].reshape(-1),
  obs_eps=np.array(0.01),            # rsl_rl's EmpiricalNormalization: (x - mean) / (std + 0.01)
  n_layers=np.array(len(layers)),
  **{f"W{i}": weights[f"mlp.{l}.weight"] for i, l in enumerate(layers)},
  **{f"b{i}": weights[f"mlp.{l}.bias"] for i, l in enumerate(layers)},
  max_beam_speed=np.array(scale),
  control_dt=np.array(decimation * timestep),
  direction=np.array(direction),
)
print(f"wrote {out}: {len(layers)} layers, {obs_size} inputs, acts every {decimation * timestep * 1000:.0f} ms, "
      f"beam speed up to {scale} rad/s, direction {'forward (+x)' if direction > 0 else 'reverse (-x)'}")
