"""mjlab task: teach a two-beam robot to drive in a straight line down +x (or backward, down -x).

Two robots, three tasks:
  Beam-Drive-Straight           the hand-made stand-in, ../beams/robot.xml
  Onshape-Beam-Drive-Straight   your Onshape export, ../onshape/robot.xml (a LEGO chassis with a Technic beam on each
                                motor; its joints are still called left_wheel / right_wheel)
  Onshape-Beam-Drive-Backward   the same Onshape robot driving in reverse: still facing +x, travelling down -x

The robot has two motors, each spinning a straight beam instead of a wheel. A beam only pushes while one of its
ends is on the floor, and two beams turning in step flip the robot over (try it: python3 drive.py beams/scene.xml),
so going forward needs a gait: keep the beams out of step, don't flip, and correct the heading as it drifts.
The policy learns that gait from scratch.

Each robot starts at the origin, pointing a little off to one side, with its beams at random angles. It is
rewarded for going down +x at TARGET_SPEED while staying on the line y = 0, facing +x and staying upright.
The policy sees its own velocities and tilt, its heading error and sideways offset, the angle and speed of each
beam, and its last command; it outputs one speed per motor.

Importing this file registers all three tasks (see train.py / play.py next to it).
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
# The tasks: task name -> (folder holding robot.xml, left motor joint, right motor joint, run name, chassis mass,
# direction). Chassis mass is in kg, None to keep what robot.xml says; the chassis keeps its shape of inertia, scaled
# to the new mass. Direction +1 drives forward (down +x), -1 drives in reverse (down -x, still facing +x).
# The folder is next to this file when run by runIt.py (--with 09_onshape_robot/<folder>), one folder up in the repo.
ROBOTS = {
  "Beam-Drive-Straight": ("beams", "left_beam", "right_beam", "beam_drive_straight", None, +1),
  # the Onshape export names the joints after wheels, and swapped: "left_wheel" turns the beam on the right (y < 0)
  "Onshape-Beam-Drive-Straight": ("onshape", "left_wheel", "right_wheel", "onshape_beam_drive_straight", 0.3, +1),   # Onshape says 0.097
  "Onshape-Beam-Drive-Backward": ("onshape", "left_wheel", "right_wheel", "onshape_beam_drive_backward", 0.3, -1),
}
TARGET_SPEED = 0.15         # m/s; turning the beams a quarter-turn out of step at 10 rad/s by hand gives about this
MAX_BEAM_SPEED = 15.0       # rad/s; a policy output of 1.0 means this beam speed
BEAM_KD = 0.05              # how hard a motor pushes toward the beam speed it was told (N*m per rad/s)
BEAM_TORQUE_LIMIT = 0.15    # N*m; about a small hobby gearmotor
START_YAW = 0.3             # rad; the robot starts pointing up to this far off +x
MAX_TILT_DEG = 70.0         # the episode ends if the chassis tips further than this (it has flipped)
MAX_OFF_LINE = 1.0          # m; the episode ends if the robot wanders this far from y = 0
EPISODE_S = 10.0
# The floor is a wood tabletop and the robot is plastic (LEGO ABS): about 0.3 to 0.4 sliding friction, the same for
# the beams and the chassis. MuJoCo normally uses the larger of the two surfaces' friction (mjlab's floor says 1.0),
# so the robot's parts are given contact priority to make this number the one used.
TABLE_FRICTION = 0.35
# Motor drag and inertia, the same for both robots. (onshape-to-robot writes frictionloss 0.1 N*m, a placeholder that
# would eat two thirds of BEAM_TORQUE_LIMIT, so it is replaced with these small-gearmotor values.)
JOINT_DAMPING, JOINT_ARMATURE, JOINT_FRICTIONLOSS = 0.0005, 0.0005, 0.0
# -------------------------------------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent


def _robot_xml(folder: str) -> Path:
  for p in (_HERE / folder / "robot.xml", _HERE.parent / folder / "robot.xml"):
    if p.exists():
      return p
  raise FileNotFoundError(f"{folder}/robot.xml not found: send it with --with 09_onshape_robot/{folder}")


def _set_mass(spec: mujoco.MjSpec, body: mujoco.MjsBody, mass: float) -> None:
  """Give `body` this mass, scaling its inertia by the same factor (same shape, heavier material)."""
  if not body.explicitinertial:             # mass comes from the geoms: write it out as an explicit inertial first
    m = spec.compile()
    b = m.body(body.name).id
    body.explicitinertial = True
    body.mass, body.ipos, body.iquat, body.inertia = m.body_mass[b], m.body_ipos[b], m.body_iquat[b], m.body_inertia[b]
  scale = mass / body.mass
  body.mass = mass
  body.inertia = [x * scale for x in body.inertia]
  body.fullinertia = [x * scale for x in body.fullinertia]   # Onshape exports give this one (NaN-free) instead


def _get_spec(folder: str, joints: tuple[str, str], chassis_mass: float | None = None) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_robot_xml(folder)))
  for actuator in list(spec.actuators):     # mjlab adds its own motors (below)
    spec.delete(actuator)
  chassis = spec.body("chassis")
  if chassis_mass is not None:
    _set_mass(spec, chassis, chassis_mass)
  chassis.add_freejoint()                   # let the robot move around
  for g in spec.geoms:                      # every part of the robot: plastic on wood (see TABLE_FRICTION)
    g.friction[0] = TABLE_FRICTION
    g.priority = 1                          # its friction wins over the floor's
  for name in joints:
    j = spec.joint(name)
    j.armature, j.frictionloss = JOINT_ARMATURE, JOINT_FRICTIONLOSS
    try:
      j.damping = JOINT_DAMPING
    except TypeError:                       # newer MuJoCo keeps damping as 3 numbers; the first is plain damping
      j.damping = [JOINT_DAMPING, 0.0, 0.0]
  return spec


def _robot_cfg(folder: str, joints: tuple[str, str], chassis_mass: float | None) -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: _get_spec(folder, joints, chassis_mass),
    # high enough that a beam pointing straight down clears the floor; the robot drops onto its beams
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.065), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target, so the motor only chases the speed we ask for
        BuiltinPdActuatorCfg(
          target_names_expr=joints, stiffness=0.0, damping=BEAM_KD, effort_limit=BEAM_TORQUE_LIMIT
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


def beam_angles(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """[sin, cos] of each beam's angle. The beams spin round and round, so the raw angle grows forever;
  sin and cos give the same information without growing. Shape [num_envs, 4]."""
  q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)


def forward_speed(env: ManagerBasedRlEnv, direction: int = 1) -> torch.Tensor:
  """Reward for going along +x (direction +1) or -x (direction -1) at TARGET_SPEED: 0 when standing still, 1 at the
  target, less beyond it, negative the wrong way.

  It slopes everywhere (a narrow bump around the target would give a robot that starts out still nothing to learn from).
  """
  vx = direction * env.scene["robot"].data.root_link_lin_vel_w[:, 0]
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


def beam_drive_env_cfg(task: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  folder, left, right, _, chassis_mass, direction = ROBOTS[task]
  joints = (left, right)
  beams = SceneEntityCfg("robot", joint_names=joints)
  actor_terms = {
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "gravity": ObservationTermCfg(func=projected_gravity),       # which way is down: tells it how tipped it is
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "beam_angle": ObservationTermCfg(func=beam_angles, params={"asset_cfg": beams}),   # where each beam is in its turn
    "beam_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "beams": JointVelocityActionCfg(
      entity_name="robot", actuator_names=joints, scale=MAX_BEAM_SPEED, use_default_offset=True
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
      params={"position_range": (-math.pi, math.pi), "velocity_range": (0.0, 0.0), "asset_cfg": beams},
    ),
  }

  rewards = {
    "forward_speed": RewardTermCfg(func=forward_speed, weight=1.0, params={"direction": direction}),
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
      entities={"robot": _robot_cfg(folder, joints, chassis_mass)},
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


def beam_drive_ppo_runner_cfg(task: str) -> RslRlOnPolicyRunnerCfg:
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
    experiment_name=ROBOTS[task][3],     # checkpoints go to logs/rsl_rl/<this>/
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )


for _task in ROBOTS:
  register_mjlab_task(
    task_id=_task,
    env_cfg=beam_drive_env_cfg(_task),
    play_env_cfg=beam_drive_env_cfg(_task, play=True),
    rl_cfg=beam_drive_ppo_runner_cfg(_task),
  )
