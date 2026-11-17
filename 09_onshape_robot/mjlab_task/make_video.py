"""Record a video of a trained policy. Made for the cluster, which has no screen.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G \
      --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
      --with runs/<your run>/logs/rsl_rl/drive_straight/<date folder>/model_499.pt \
      09_onshape_robot/mjlab_task/make_video.py model_499.pt

For the beam robot, send beam_task.py and the beams folder instead, and name the task after the checkpoint:

    ... --with 09_onshape_robot/mjlab_task/beam_task.py --with 09_onshape_robot/beams \
      --with runs/<your run>/logs/rsl_rl/beam_drive_straight/<date folder>/model_999.pt \
      09_onshape_robot/mjlab_task/make_video.py model_999.pt Beam-Drive-Straight

runIt.py brings the .mp4 home to ./runs/ and opens it. (mjlab's own play.py also records a video, but then opens a
web viewer that never exits on a machine without a screen.)
"""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import importlib

for _module in ("drive_task", "beam_task", "real_task"):   # register whichever task files were sent along
  try:
    importlib.import_module(_module)
  except ModuleNotFoundError as e:
    if e.name != _module:
      raise

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

SECONDS = 10         # of driving

if len(sys.argv) not in (2, 3):
  sys.exit("usage: make_video.py model_499.pt [Drive-Straight | Beam-Drive-Straight | Onshape-Beam-Drive-Straight | Onshape-Beam-Drive-Backward | Onshape-Beam-Real[-V2] | Onshape-Beam-Real[-V2]-Backward]")
checkpoint = Path(sys.argv[1])
TASK = sys.argv[2] if len(sys.argv) == 3 else "Drive-Straight"

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
PER_SECOND = round(1 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation))   # policy steps per second (50, or 20)
STEPS = SECONDS * PER_SECOND
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 640, 480
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(env, video_folder=".", step_trigger=lambda step: step == 0, video_length=STEPS, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = env.unwrapped.scene["robot"]
print("  second   x (m)   y (m)   heading (deg)   speed (m/s)")
obs = env.get_observations()
with torch.no_grad():
  for step in range(STEPS):
    obs = env.step(policy(obs))[0]
    if step % PER_SECOND == PER_SECOND - 1:   # once a second
      x, y = robot.data.root_link_pos_w[0, :2].tolist()
      w, qx, qy, qz = robot.data.root_link_quat_w[0].tolist()
      yaw = torch.atan2(torch.tensor(2 * (w * qz + qx * qy)), torch.tensor(1 - 2 * (qy * qy + qz * qz))).item()
      vx = robot.data.root_link_lin_vel_w[0, 0].item()
      print(f"  {(step + 1) / PER_SECOND:5.0f}  {x:7.2f} {y:7.2f}  {yaw * 57.296:11.1f}  {vx:12.2f}")
env.close()
print("wrote the video (an .mp4 next to this script)")
