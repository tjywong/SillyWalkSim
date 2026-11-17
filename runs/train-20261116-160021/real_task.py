"""mjlab task: the Onshape beam robot, trained so the policy can drive the REAL robot (a LEGO Double Motor over
Bluetooth, see ../real_robot/drive_real.py).

This is a separate file on purpose: beam_task.py and its tasks (Beam-Drive-Straight, Onshape-Beam-Drive-Straight,
Onshape-Beam-Drive-Backward) are not changed and keep working. This file borrows the robot, the rewards and the
constants from beam_task.py, so send both:  --with 09_onshape_robot/mjlab_task/beam_task.py
--with 09_onshape_robot/mjlab_task/real_task.py --with 09_onshape_robot/onshape

What is different from Onshape-Beam-Drive-Straight, and why:

  * The policy only sees what the Double Motor can measure: its gyroscope (how fast it turns), its accelerometer
    (which way is down), its heading (yaw, zeroed at the start), each beam's angle and speed (the motor encoders) and
    its own last command. It does NOT see its velocity or how far it is from the line (nothing on the robot measures
    those). The critic, which only exists during training, still sees everything, which helps it learn.
  * Beam angles are given as sin/cos of TWICE the angle: a Technic beam held at its middle looks the same after half a
    turn, so this way the policy cannot tell (and does not need to know) which end of the beam is which.
  * It acts 20 times a second (every 50 ms) instead of 50, which Bluetooth can keep up with.
  * Its commands reach the motors 30 to 80 ms late, and its readings are up to one step (50 ms) old: Bluetooth delay.
  * Its readings are noisy, like real sensors.
  * Every robot in training is a little different: floor friction 0.25 to 0.5, chassis mass 0.7x to 1.3x (and its
    centre of mass shifted by up to 5 mm), motor strength and torque 0.6x to 1.4x. The real robot is somewhere in
    that range, so the policy has to work for all of them.

Tasks: "Onshape-Beam-Real" (forward, down +x) and "Onshape-Beam-Real-Backward" (in reverse, down -x), and their
second version "Onshape-Beam-Real-V2" and "Onshape-Beam-Real-V2-Backward" (see V2 below: made after the first real
runs flipped and crept left). The first version is kept exactly as it was.
"""
from __future__ import annotations

import math

import torch

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  is_alive,
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
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

try:
  from mjlab.actuator import IdealPdActuatorCfg
except ImportError:  # older mjlab keeps it in its own module
  from mjlab.actuator.pd_actuator import IdealPdActuatorCfg

import beam_task as bt  # the robot, rewards and constants (unchanged)

# ---- constants you may want to change ----------------------------------------------------------------
MAX_BEAM_SPEED = 12.0        # rad/s for a policy output of 1.0. Must not be more than the Double Motor's top speed
                             # (drive_real.py --calibrate measures it); if it is, lower this and train again.
PHYSICS_DT = 0.002
CONTROL_DT = 0.05            # s between policy decisions (20 per second)
CMD_DELAY_S = (0.03, 0.08)   # command latency range (Bluetooth + motor controller)
OBS_DELAY_STEPS = (0, 1)     # readings are 0 to 1 policy steps old
FRICTION_RANGE = (0.25, 0.5)        # plastic on wood, give or take (all robot parts)
MASS_SCALE_RANGE = (0.7, 1.3)       # chassis mass, times the 0.3 kg in beam_task.py
COM_SHIFT_M = 0.005                 # chassis centre of mass moves up to this far in each direction
MOTOR_SCALE_RANGE = (0.6, 1.4)      # motor strength (damping gain) and torque limit
# sensor noise, uniform +/- this much (in the units the policy sees)
NOISE = {"gyro": 0.15, "gravity": 0.05, "heading": 0.03, "beam_angle": 0.05, "beam_vel": 1.0}

# The first version (Onshape-Beam-Real, -Backward): the settings above, unchanged.
V1 = {
  "max_beam_speed": MAX_BEAM_SPEED,
  "cmd_delay_s": CMD_DELAY_S,
  "motor_scale_range": MOTOR_SCALE_RANGE,
  "noise": NOISE,
  "tilt_limit_deg": bt.MAX_TILT_DEG,
  # reward weights; 0 = not used
  "weights": {"forward_speed": 1.0, "stay_on_line": 0.5, "face_forward": 0.5, "stay_upright": 0.5,
              "action_rate": -0.05, "rocking": 0.0, "sideways_speed": 0.0},
}

# The second version (Onshape-Beam-Real-V2, -V2-Backward). The first real runs flipped (the real robot rocked nose
# up/down 2 to 4 times more than in simulation) and crept left (the simulation does too: the robot cannot measure
# how far it has drifted sideways, and nothing in training stopped a slow sideways creep).
V2 = {
  **V1,
  "max_beam_speed": 8.0,                      # gentler gait: each beam stroke kicks less (was 12)
  "cmd_delay_s": (0.03, 0.15),                # allow for slower Bluetooth (was 0.03 to 0.08)
  "motor_scale_range": (0.5, 2.0),            # the real motors may push harder than modelled (was 0.6 to 1.4)
  "noise": {**NOISE, "beam_vel": 4.0},        # real beam-speed readings are spiky (was 1.0)
  "tilt_limit_deg": 60.0,                     # training episodes end at 60 degrees: keep a margin (was 70; 45 was
                                              # too strict: episodes ended before the policy could learn anything)
  # Reward weights, measured rather than guessed: for a gait that really drives (version 1's, run against the
  # simulated Double Motor) these give about +0.33 per second more than sitting still, the same margin version 1
  # learned with. (The first V2 tries got this wrong: -3.08 per second, so sitting still or flipping at once
  # scored best, and nothing was learned.)
  "weights": {
    **V1["weights"],
    "forward_speed": 2.0,                     # (was 1.0) so driving stays worth more than the extra penalties
    "stay_upright": 1.0,                      # (was 0.5)
    "rocking": 0.005,                         # new: penalty on rocking speed (roll and pitch rates)
    "stay_on_line": 1.0,                      # (was 0.5)
    "sideways_speed": 0.05,                   # new, small: the gait sways ~9 cm/s with every stroke; only slow drift
                                              # should be discouraged, and stay_on_line does most of that
    "face_forward": 1.0,                      # (was 0.5)
    "alive": 0.5,                             # new: per second not flipped, so flipping at once never pays
  },
}

REAL_TASKS = {  # task name -> (direction: +1 forward / -1 reverse, run name for logs/rsl_rl/<run name>/, settings)
  "Onshape-Beam-Real": (+1, "onshape_beam_real", V1),
  "Onshape-Beam-Real-Backward": (-1, "onshape_beam_real_backward", V1),
  "Onshape-Beam-Real-V2": (+1, "onshape_beam_real_v2", V2),
  "Onshape-Beam-Real-V2-Backward": (-1, "onshape_beam_real_v2_backward", V2),
}
# -------------------------------------------------------------------------------------------------------

FOLDER, LEFT, RIGHT, _, CHASSIS_MASS, _ = bt.ROBOTS["Onshape-Beam-Drive-Straight"]
JOINTS = (LEFT, RIGHT)
_BEAMS = SceneEntityCfg("robot", joint_names=JOINTS)
_CHASSIS = SceneEntityCfg("robot", body_names=("chassis",))


def heading(env) -> torch.Tensor:
  """[sin(yaw), cos(yaw)]: the heading the Double Motor's IMU gives (zeroed when the run starts)."""
  yaw = bt._yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def beam_angles_2x(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """[sin(2q), cos(2q)] for each beam: the same after half a turn, like the beam itself. Shape [num_envs, 4]."""
  q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.sin(2 * q), torch.cos(2 * q)], dim=-1)


def rocking(env) -> torch.Tensor:
  """Minus the squared roll and pitch rates (rad/s): rocking side to side and nose up/down."""
  w = env.scene["robot"].data.root_link_ang_vel_b
  return -(torch.square(w[:, 0]) + torch.square(w[:, 1]))


def sideways_speed(env) -> torch.Tensor:
  """Minus the squared sideways speed, -1 at 5 cm/s: catches a slow creep long before it adds up to an offset."""
  return -torch.square(env.scene["robot"].data.root_link_lin_vel_w[:, 1] / 0.05)


def _noise(settings: dict, name: str) -> UniformNoiseCfg:
  return UniformNoiseCfg(n_min=-settings["noise"][name], n_max=settings["noise"][name])


def _robot_cfg(settings: dict) -> EntityCfg:
  delay = settings["cmd_delay_s"]
  lag = (round(delay[0] / PHYSICS_DT), round(delay[1] / PHYSICS_DT))
  return EntityCfg(
    spec_fn=lambda: bt._get_spec(FOLDER, JOINTS, CHASSIS_MASS),
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.065), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # the same motor as beam_task.py (pushes BEAM_KD * speed error, capped at BEAM_TORQUE_LIMIT), but computed
        # in mjlab so its commands can arrive late
        IdealPdActuatorCfg(
          target_names_expr=JOINTS,
          stiffness=0.0,
          damping=bt.BEAM_KD,
          effort_limit=bt.BEAM_TORQUE_LIMIT,
          delay_min_lag=lag[0],
          delay_max_lag=lag[1],
        ),
      ),
    ),
  )


def real_env_cfg(task: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  direction, _, settings = REAL_TASKS[task]
  w = settings["weights"]
  delay = {"delay_min_lag": OBS_DELAY_STEPS[0], "delay_max_lag": OBS_DELAY_STEPS[1]}

  # What the real robot can measure, in this order (drive_real.py builds exactly this list).
  actor_terms = {
    "gyro": ObservationTermCfg(func=base_ang_vel, noise=_noise(settings, "gyro"), **delay),                # rad/s, body frame
    "gravity": ObservationTermCfg(func=projected_gravity, noise=_noise(settings, "gravity"), **delay),    # unit vector
    "heading": ObservationTermCfg(func=heading, noise=_noise(settings, "heading"), **delay),              # sin, cos of yaw
    "beam_angle": ObservationTermCfg(
      func=beam_angles_2x, params={"asset_cfg": _BEAMS}, noise=_noise(settings, "beam_angle"), **delay
    ),
    "beam_vel": ObservationTermCfg(func=joint_vel_rel, noise=_noise(settings, "beam_vel"), **delay),      # rad/s
    "last_action": ObservationTermCfg(func=last_action),
  }
  # The critic is only used while training, so it may also see what the robot cannot measure.
  critic_terms = {
    **{k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor_terms.items()},
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "heading_offset": ObservationTermCfg(func=bt.heading_and_offset),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic_terms),
  }

  actions = {
    "beams": JointVelocityActionCfg(entity_name="robot", actuator_names=JOINTS, scale=settings["max_beam_speed"],
                                    use_default_offset=True),
  }

  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform, mode="reset",
      params={"pose_range": {"yaw": (-bt.START_YAW, bt.START_YAW)}, "velocity_range": {}},
    ),
    "reset_beams": EventTermCfg(
      func=reset_joints_by_offset, mode="reset",
      params={"position_range": (-math.pi, math.pi), "velocity_range": (0.0, 0.0), "asset_cfg": _BEAMS},
    ),
    # every training robot is a little different (drawn once, when training starts)
    "friction": EventTermCfg(
      func=dr.geom_friction, mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)), "ranges": FRICTION_RANGE,
              "operation": "abs", "shared_random": True},
    ),
    "chassis_mass": EventTermCfg(
      func=dr.pseudo_inertia, mode="startup",
      params={"asset_cfg": _CHASSIS,
              "alpha_range": (math.log(MASS_SCALE_RANGE[0]) / 2, math.log(MASS_SCALE_RANGE[1]) / 2),
              "t_range": (-COM_SHIFT_M, COM_SHIFT_M)},
    ),
    "motor_strength": EventTermCfg(
      func=dr.pd_gains, mode="startup",
      params={"asset_cfg": _BEAMS, "kp_range": (1.0, 1.0), "kd_range": settings["motor_scale_range"]},
    ),
    "motor_torque": EventTermCfg(
      func=dr.effort_limits, mode="startup",
      params={"asset_cfg": _BEAMS, "effort_limit_range": settings["motor_scale_range"]},
    ),
  }
  if play:  # a movie shows the robot as designed, not a random variation
    for k in ("friction", "chassis_mass", "motor_strength", "motor_torque"):
      events.pop(k)

  rewards = {
    "forward_speed": RewardTermCfg(func=bt.forward_speed, weight=w["forward_speed"], params={"direction": direction}),
    "stay_on_line": RewardTermCfg(func=bt.stay_on_line, weight=w["stay_on_line"]),
    "face_forward": RewardTermCfg(func=bt.face_forward, weight=w["face_forward"]),
    "stay_upright": RewardTermCfg(func=bt.stay_upright, weight=w["stay_upright"]),
    # smoother commands than in simulation only: real motors and Bluetooth do not like jumps
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=w["action_rate"]),
  }
  for name, func in (("rocking", rocking), ("sideways_speed", sideways_speed), ("alive", is_alive)):   # V2 only
    if w.get(name):
      rewards[name] = RewardTermCfg(func=func, weight=w[name])

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "flipped": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(settings["tilt_limit_deg"])}),
    "off_line": TerminationTermCfg(func=bt.too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": _robot_cfg(settings)},
                   num_envs=1, env_spacing=3.0),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="chassis",
                        distance=0.6, elevation=-20.0, azimuth=140.0),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=PHYSICS_DT)),
    decimation=round(CONTROL_DT / PHYSICS_DT),
    episode_length_s=1e10 if play else bt.EPISODE_S,
  )


def real_ppo_runner_cfg(task: str) -> RslRlOnPolicyRunnerCfg:
  cfg = bt.beam_drive_ppo_runner_cfg("Onshape-Beam-Drive-Straight")
  cfg.experiment_name = REAL_TASKS[task][1]
  cfg.max_iterations = 1500   # randomization makes it harder; give it a little longer
  return cfg


for _task in REAL_TASKS:
  register_mjlab_task(
    task_id=_task,
    env_cfg=real_env_cfg(_task),
    play_env_cfg=real_env_cfg(_task, play=True),
    rl_cfg=real_ppo_runner_cfg(_task),
  )
