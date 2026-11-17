"""mjlab task: teach the two-beam robot (../beams/robot.xml) to drive in a straight line down +x.

The robot has two motors, each spinning a straight beam instead of a wheel. A beam only pushes while one of its
ends is on the floor, and two beams turning in step flip the robot over (try it: python3 drive.py beams/scene.xml),
so going forward needs a gait: keep the beams out of step, don't flip, and correct the heading as it drifts.
The policy learns that gait from scratch.

Each robot starts at the origin, pointing a little off to one side, with its beams at random angles. It is
rewarded for going down +x at TARGET_SPEED while staying on the line y = 0, facing +x and staying upright.
The policy sees its own velocities and tilt, its heading error and sideways offset, the angle and speed of each
beam, and its last command; it outputs one speed per motor.

Importing this file registers the task "Beam-Drive-Straight" (see train.py / play.py next to it).
Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  bad_orientation,
  base_ang_vel,
  base_lin_vel,
  joint_vel_rel,
  last_action,
  projected_gravity,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
# the beam robot: next to this file when run by runIt.py, one folder up in the repo
_HERE = Path(__file__).resolve().parent
ROBOT_XML = next(p for p in (_HERE / "beams" / "robot.xml", _HERE.parent / "beams" / "robot.xml") if p.exists())
LEFT, RIGHT = "left_beam", "right_beam"         # joint names in robot.xml
TARGET_SPEED = 0.15         # m/s; turning the beams a quarter-turn out of step at 10 rad/s by hand gives about this
MAX_BEAM_SPEED = 15.0       # rad/s; a policy output of 1.0 means this beam speed
BEAM_KD = 0.05              # how hard a motor pushes toward the beam speed it was told (N*m per rad/s)
BEAM_TORQUE_LIMIT = 0.15    # N*m; about a small hobby gearmotor
START_YAW = 0.3             # rad; the robot starts pointing up to this far off +x
MAX_TILT_DEG = 70.0         # the episode ends if the chassis tips further than this (it has flipped)
MAX_OFF_LINE = 1.0          # m; the episode ends if the robot wanders this far from y = 0
EPISODE_S = 10.0
# -------------------------------------------------------------------------------------------------------

_ROBOT = SceneEntityCfg("robot")
_BEAMS = SceneEntityCfg("robot", joint_names=(LEFT, RIGHT))


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
  for actuator in list(spec.actuators):     # mjlab adds its own motors (below)
    spec.delete(actuator)
  chassis = spec.body("chassis")
  chassis.add_freejoint()                   # let the robot move around
  for g in chassis.geoms:                   # only the beams should grip; the chassis slides
    g.friction[0] = 0.01
  return spec


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    # high enough that a beam pointing straight down clears the floor; the robot drops onto its beams
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.065), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target, so the motor only chases the speed we ask for
        BuiltinPdActuatorCfg(
          target_names_expr=(LEFT, RIGHT), stiffness=0.0, damping=BEAM_KD, effort_limit=BEAM_TORQUE_LIMIT
        ),
      ),
    ),
  )


# ---- observations and rewards: how the robot is doing relative to "straight down +x" -------------------


def _yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from this env's start line (its y axis), in metres."""
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]


def heading_and_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw), off-line distance]. Shape [num_envs, 3]."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw), _off_line(env)], dim=-1)


def beam_angles(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _BEAMS) -> torch.Tensor:
  """[sin, cos] of each beam's angle. The beams spin round and round, so the raw angle grows forever;
  sin and cos give the same information without growing. Shape [num_envs, 4]."""
  q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)


def forward_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for going along +x at TARGET_SPEED: 0 when standing still, 1 at the target, less beyond it, negative backwards.

  It slopes everywhere (a narrow bump around the target would give a robot that starts out still nothing to learn from).
  """
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  return torch.clamp(1.0 - torch.square((vx - TARGET_SPEED) / TARGET_SPEED), min=-1.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_off_line(env) / 0.1)        # -1 at 10 cm off the line


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_yaw(env))


def stay_upright(env: ManagerBasedRlEnv) -> torch.Tensor:
  """0 when level, down to -1 when on its side: how much of 'down' points sideways or forward in the robot's frame."""
  g = env.scene["robot"].data.projected_gravity_b
  return -(torch.square(g[:, 0]) + torch.square(g[:, 1]))


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


# ---- environment ----------------------------------------------------------------------------------------


def beam_drive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "gravity": ObservationTermCfg(func=projected_gravity),       # which way is down: tells it how tipped it is
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "beam_angle": ObservationTermCfg(func=beam_angles, params={"asset_cfg": _BEAMS}),   # where each beam is in its turn
    "beam_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "beams": JointVelocityActionCfg(
      entity_name="robot", actuator_names=(LEFT, RIGHT), scale=MAX_BEAM_SPEED, use_default_offset=True
    ),
  }

  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
    ),
    # start each beam at a random angle, so the policy learns to sort out the timing between them itself
    "reset_beams": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={"position_range": (-math.pi, math.pi), "velocity_range": (0.0, 0.0), "asset_cfg": _BEAMS},
    ),
  }

  rewards = {
    "forward_speed": RewardTermCfg(func=forward_speed, weight=1.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=0.5),
    "face_forward": RewardTermCfg(func=face_forward, weight=0.5),
    "stay_upright": RewardTermCfg(func=stay_upright, weight=0.5),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "flipped": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(MAX_TILT_DEG)}),
    "off_line": TerminationTermCfg(func=too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=3.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis",
      distance=0.6,
      elevation=-20.0,
      azimuth=140.0,
    ),
    # 2 ms physics steps (the beam tips are small and fast), 10 per policy step: the policy acts 50 times a second
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    decimation=10,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def beam_drive_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(128, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="beam_drive_straight",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )


register_mjlab_task(
  task_id="Beam-Drive-Straight",
  env_cfg=beam_drive_env_cfg(),
  play_env_cfg=beam_drive_env_cfg(play=True),
  rl_cfg=beam_drive_ppo_runner_cfg(),
)
