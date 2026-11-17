"""Drive the real two-beam robot (a LEGO Education Double Motor, over Bluetooth) with a policy trained in simulation.

The policy must come from the real-robot task (Onshape-Beam-Real or Onshape-Beam-Real-Backward in
../mjlab_task/real_task.py), turned into policy.npz by ../mjlab_task/export_policy.py. See README.md next to this file.

  python drive_real.py --calibrate              once per robot build: learn how the hub is mounted (writes calibration.json)
  python drive_real.py --check                  print what the policy would see, live, with the motors off
  python drive_real.py policy.npz               drive until Ctrl-C (or --seconds 20 to stop on its own)
  python drive_real.py policy.npz --sim         the same program against a simulated Double Motor (needs mujoco)

Needs:  python -m pip install "legoeducation>=1.1" numpy      (and mujoco for --sim)
Stop at any time with Ctrl-C: the motors stop. They also stop when the robot tips past STOP_TILT_DEG.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "calibration.json"

# ---- facts about the simulated robot (../onshape/robot.xml), which the real robot must match -----------------
# The two beam joints, in the order the policy uses. The Onshape export swapped the names: "left_wheel" is the beam
# on the robot's RIGHT side (y < 0) and "right_wheel" the one on its LEFT side.
JOINTS = ("left_wheel", "right_wheel")
SIDE = {"left_wheel": "right", "right_wheel": "left"}
# Which way a POSITIVE joint angle turns, seen from the robot's LEFT side (standing beside it, looking across it).
PLUS_TURN_SEEN_FROM_LEFT = {"left_wheel": "counterclockwise", "right_wheel": "clockwise"}
# The joint angle (radians) at which each beam points straight up and down. Only the angle modulo 180 degrees matters
# (the policy sees sin and cos of twice the angle, and the beam looks the same after half a turn).
Q_VERTICAL = {"left_wheel": math.radians(102.0), "right_wheel": math.radians(86.5)}

# Which Double Motor to connect to: the colour and number of its Connection Card. Override with --card-color and
# --card-serial, or use --card-color any to take the first Double Motor found.
CARD_COLOR = "purple"
CARD_SERIAL = "5164"

STOP_TILT_DEG = 60.0         # stop the motors if the robot tips further than this
NOTIFY_MS = 20               # how often the Double Motor sends its readings
CAL_SPEED_PERCENT = 40       # motor speed used to measure the motors during --calibrate


# ===================================================================================================================
# The hub: the real Double Motor, or a simulated one. Both give raw readings in the Double Motor's own units.
# ===================================================================================================================
class RealHub:
  def __init__(self, card_color: str = CARD_COLOR, card_serial: str = CARD_SERIAL):
    try:
      import legoeducation as le
    except ImportError:
      sys.exit('legoeducation is not installed: python -m pip install "legoeducation>=1.1" numpy')
    self.le = le
    self.dm = le.DoubleMotor()
    filters, label = {}, "the first Double Motor found"
    if card_color and card_color.lower() != "any":
      code = getattr(le, f"LEGO_COLOR_{card_color.upper()}", None)
      if code is None:
        sys.exit(f"Unknown card colour {card_color!r}. Try: red, yellow, blue, teal, green, purple, white, magenta,"
                 " orange, azure, or any.")
      filters["card_color"] = code
      label = f"the Double Motor with the {card_color.lower()} Connection Card"
    if card_serial and card_serial.lower() != "any":
      filters["card_serial"] = str(card_serial)          # a string keeps leading zeros (e.g. "0049")
      label += f" number {card_serial}"
    print(f"Connecting to {label} over Bluetooth (turn it on)...")
    self.dm.connect(device_notification_delay=NOTIFY_MS, **filters)
    print("connected")

  def read(self) -> dict:
    """Raw readings: accelerometer and gyroscope (3 each, Double Motor units), and for each hub motor (0 = MOTOR_LEFT,
    1 = MOTOR_RIGHT) its absolute angle (0..359 deg) and its total turned angle (deg)."""
    imu, m = self.dm.imu_device, self.dm.motor
    return {
      "accel": np.array([imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ], float),
      "gyro": np.array([imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ], float),
      "abs_deg": np.array([m[0].absolutePosition, m[1].absolutePosition], float),
      "pos_deg": np.array([m[0].position, m[1].position], float),
    }

  def run(self, percents) -> None:
    """Turn hub motor i at percents[i] (-100..100; positive = the Double Motor's clockwise)."""
    le = self.le
    self.dm.begin_batch()
    for i, p in enumerate(percents):
      p = int(round(max(-100, min(100, p))))
      direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if p >= 0 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
      self.dm.motor_run(direction=direction, motor=(le.MOTOR_LEFT, le.MOTOR_RIGHT)[i], speed=abs(p), blocking=False)
    self.dm.end_batch(blocking=False)

  def stop(self) -> None:
    for m in (self.le.MOTOR_LEFT, self.le.MOTOR_RIGHT):
      try:
        self.dm.motor_stop(motor=m)
      except Exception:
        pass

  def now(self) -> float:
    return time.monotonic()

  def sleep(self, s: float) -> None:
    time.sleep(max(0.0, s))

  def close(self) -> None:
    self.stop()
    try:
      self.dm.disconnect()
    except Exception:
      pass


class SimHub:
  """A simulated Double Motor on the Onshape robot in plain MuJoCo, for testing this program without hardware.

  It is deliberately awkward, so the unit and direction conversions are really tested: the hub is mounted rotated,
  its accelerometer reports the downward direction, its motors are crossed over (MOTOR_LEFT drives the beam on the
  robot's right), one motor turns the opposite way, its absolute-angle zero is somewhere random, and its commands
  arrive 50 ms late. --sim writes the calibration that matches it (what --calibrate would find on such a robot).
  """
  PHYS_DT = 0.002
  RAD_S_PER_PERCENT = 0.18          # this pretend motor: 18 rad/s at 100 %
  BEAM_KD, BEAM_TORQUE_LIMIT = 0.05, 0.15
  CMD_DELAY_S = 0.05

  def __init__(self, seed: int = 0):
    import mujoco
    self.mj = mujoco
    rng = self.rng = np.random.default_rng(seed)
    spec = mujoco.MjSpec.from_file(str(HERE.parent / "onshape" / "robot.xml"))
    for a in list(spec.actuators):
      spec.delete(a)
    chassis = spec.body("chassis")
    scale = 0.3 / chassis.mass                       # the 0.3 kg chassis, as in training
    chassis.mass *= scale
    chassis.fullinertia = [x * scale for x in chassis.fullinertia]
    chassis.add_freejoint()
    for g in spec.geoms:
      g.friction[0], g.priority = 0.35, 1
    for j in JOINTS:
      jt = spec.joint(j)
      jt.armature, jt.frictionloss = 0.0005, 0.0
      try:
        jt.damping = 0.0005
      except TypeError:
        jt.damping = [0.0005, 0.0, 0.0]
      a = spec.add_actuator(name=j, target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
      a.gainprm[0] = 1.0                              # ctrl = torque (N*m), computed below
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01))
    self.m = spec.compile()
    self.m.opt.timestep = self.PHYS_DT
    self.d = mujoco.MjData(self.m)
    self.jq = [self.m.jnt_qposadr[self.m.joint(j).id] for j in JOINTS]
    self.jv = [self.m.jnt_dofadr[self.m.joint(j).id] for j in JOINTS]
    self.act = [self.m.actuator(j).id for j in JOINTS]

    # the awkward hub: hub axes = R_HB^T * body axes
    self.R_HB = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])  # rows: body x, y, z in hub axes
    self.ACCEL_COUNTS_PER_G, self.ACCEL_SIGN = 1000.0, -1.0                          # reports "down"
    self.GYRO_COUNTS_PER_RAD_S = 1800.0 / math.pi                                     # tenths of a degree per second
    self.HUB_MOTOR_OF = {"left_wheel": 0, "right_wheel": 1}                          # MOTOR_LEFT on the right side
    self.CMD_SIGN = {"left_wheel": +1, "right_wheel": -1}                            # +% -> +q / -q
    self.ENC_SIGN = {"left_wheel": -1, "right_wheel": +1}                            # +q -> encoder down / up
    self.ABS_AT_VERTICAL = {j: float(rng.uniform(0, 360)) for j in JOINTS}

    # start like a robot put down on the table: a little off +x, beams anywhere, then let it settle
    yaw = rng.uniform(-0.3, 0.3)
    self.d.qpos[2] = 0.065
    self.d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    for a in self.jq:
      self.d.qpos[a] = rng.uniform(-math.pi, math.pi)
    self.t = 0.0
    self.pending = []            # (time it arrives, target speeds rad/s per joint)
    self.target = np.zeros(2)
    for _ in range(50):          # let it settle (up to 5 s): a robot balanced on a beam end slowly tips over
      self.sleep(0.1)
      if np.linalg.norm(self.d.qvel) < 1e-3:
        break
    self.t = 0.0

  def calibration(self) -> dict:
    """What --calibrate finds on this pretend robot."""
    return {
      "R_hub_to_body": self.R_HB.tolist(),
      "accel_up_sign": self.ACCEL_SIGN,
      "gyro_rad_s_per_count": 1.0 / self.GYRO_COUNTS_PER_RAD_S,
      "motors": {j: {"hub_motor": self.HUB_MOTOR_OF[j], "cmd_sign": self.CMD_SIGN[j], "enc_sign": self.ENC_SIGN[j],
                     "abs_deg_at_vertical": self.ABS_AT_VERTICAL[j], "rad_s_per_percent": self.RAD_S_PER_PERCENT}
                 for j in JOINTS},
      "note": "made by drive_real.py --sim for its simulated Double Motor; not for a real robot",
    }

  def _body_rot(self):
    return self.d.xmat[self.m.body("chassis").id].reshape(3, 3)       # body -> world

  def read(self) -> dict:
    R = self._body_rot()
    up_body = R.T @ np.array([0, 0, 1.0])
    w_body = self.d.qvel[3:6]                                           # freejoint angular velocity: body frame
    rng = self.rng
    accel = self.ACCEL_SIGN * self.ACCEL_COUNTS_PER_G * (self.R_HB.T @ up_body) + rng.normal(0, 10, 3)
    gyro = self.GYRO_COUNTS_PER_RAD_S * (self.R_HB.T @ w_body) + rng.normal(0, 5, 3) + 3.0   # + a little bias
    abs_deg, pos_deg = np.zeros(2), np.zeros(2)
    for k, j in enumerate(JOINTS):
      i = self.HUB_MOTOR_OF[j]
      enc = self.ENC_SIGN[j] * math.degrees(self.d.qpos[self.jq[k]] - Q_VERTICAL[j])
      abs_deg[i] = round((enc + self.ABS_AT_VERTICAL[j]) % 360.0)
      pos_deg[i] = round(self.ENC_SIGN[j] * math.degrees(self.d.qpos[self.jq[k]]))
    return {"accel": np.round(accel), "gyro": np.round(gyro), "abs_deg": abs_deg, "pos_deg": pos_deg}

  def run(self, percents) -> None:
    tgt = np.zeros(2)
    for k, j in enumerate(JOINTS):
      p = max(-100, min(100, round(percents[self.HUB_MOTOR_OF[j]])))
      tgt[k] = self.CMD_SIGN[j] * p * self.RAD_S_PER_PERCENT
    self.pending.append((self.t + self.CMD_DELAY_S, tgt))

  def stop(self) -> None:
    self.pending.append((self.t + self.CMD_DELAY_S, np.zeros(2)))

  def now(self) -> float:
    return self.t

  def sleep(self, s: float) -> None:
    end = self.t + max(s, self.PHYS_DT)        # always move on at least one physics step
    while self.t < end - 1e-9:
      while self.pending and self.pending[0][0] <= self.t:
        self.target = self.pending.pop(0)[1]
      for k in range(2):
        tau = self.BEAM_KD * (self.target[k] - self.d.qvel[self.jv[k]])
        self.d.ctrl[self.act[k]] = max(-self.BEAM_TORQUE_LIMIT, min(self.BEAM_TORQUE_LIMIT, tau))
      self.mj.mj_step(self.m, self.d)
      self.t += self.PHYS_DT

  def close(self) -> None:
    pass

  def where(self):
    """True position and heading (only the simulation knows these)."""
    w, x, y, z = self.d.qpos[3:7]
    return self.d.qpos[0], self.d.qpos[1], math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# ===================================================================================================================
# Raw readings -> what the policy sees (the same 16 numbers, in the same order, as real_task.py)
# ===================================================================================================================
class Senses:
  def __init__(self, hub, cal: dict):
    self.hub, self.cal = hub, cal
    self.R = np.array(cal["R_hub_to_body"])
    self.motor = cal["motors"]
    self.gyro_bias = np.zeros(3)
    self.Rwb = None            # estimated orientation (body -> world), heading zeroed at the start
    self.yaw = 0.0
    self.t_last = None
    self.prev_pos = None

  def measure_gyro_bias(self, seconds: float = 1.0, tries: int = 5) -> None:
    """Average the gyroscope while the robot is still: its reading when nothing turns. A robot that is still
    settling (tipping off a beam end) would spoil this, so it checks the robot really is still, and waits if not."""
    for attempt in range(tries):
      samples = []
      end = self.hub.now() + seconds
      while self.hub.now() < end:
        samples.append(self.hub.read()["gyro"])
        self.hub.sleep(NOTIFY_MS / 1000)
      samples = np.array(samples) * self.cal["gyro_rad_s_per_count"]
      first, last = samples[: len(samples) // 2].mean(0), samples[len(samples) // 2:].mean(0)
      if np.linalg.norm(first - last) < 0.01 and samples.std(0).max() < 0.05:      # rad/s: steady and quiet
        self.gyro_bias = samples.mean(0) / self.cal["gyro_rad_s_per_count"]
        return
      print("  the robot is still moving: keep it still (lay it flat if it rocks on a beam)...")
    sys.exit("The robot never kept still, so the gyroscope could not be zeroed.")

  def gyro_body(self, raw) -> np.ndarray:
    return self.R @ ((raw["gyro"] - self.gyro_bias) * self.cal["gyro_rad_s_per_count"])

  def accel_down_body(self, raw) -> np.ndarray:
    """'Down' from the accelerometer alone (shaken by the gait; the orientation filter smooths it)."""
    up_hub = self.cal["accel_up_sign"] * raw["accel"]
    n = np.linalg.norm(up_hub)
    return -(self.R @ up_hub) / n if n > 0 else np.array([0, 0, -1.0])

  ACCEL_TRUST = 1.0          # 1/s: how fast the accelerometer pulls the tilt estimate back (gyro does the rest)

  def update_orientation(self, raw) -> None:
    """Track the full orientation: turn it by the gyroscope, nudge its tilt toward the accelerometer (a Mahony
    filter). Heading and 'down' then mean exactly what they mean in training (yaw of the body's orientation, and
    gravity seen from the body), even while the robot rocks."""
    t = self.hub.now()
    up_meas = -self.accel_down_body(raw)
    if self.Rwb is None:                                # start: level from the accelerometer, heading 0
      x = np.array([1.0, 0, 0]) - up_meas[0] * up_meas
      x /= np.linalg.norm(x)
      self.Rwb = np.array([x, np.cross(up_meas, x), up_meas])   # rows: world x, y, z in body axes
    else:
      dt = t - self.t_last
      up_est = self.Rwb[2]                              # world "up" in body axes
      w = self.gyro_body(raw) + self.ACCEL_TRUST * np.cross(up_meas, up_est)
      angle = np.linalg.norm(w) * dt
      if angle > 0:
        k = w / np.linalg.norm(w)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        step = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K   # body turns by w*dt
        self.Rwb = self.Rwb @ step
        u, _, vt = np.linalg.svd(self.Rwb)              # keep it a rotation
        self.Rwb = u @ vt
    self.yaw = math.atan2(self.Rwb[1, 0], self.Rwb[0, 0])
    self.t_last = t

  def gravity_body(self) -> np.ndarray:
    return -self.Rwb[2]

  def beams(self, raw):
    q, pos = np.zeros(2), np.zeros(2)
    for k, j in enumerate(JOINTS):
      c = self.motor[j]
      i = c["hub_motor"]
      q[k] = c["enc_sign"] * math.radians(raw["abs_deg"][i] - c["abs_deg_at_vertical"]) + Q_VERTICAL[j]
      pos[k] = c["enc_sign"] * math.radians(raw["pos_deg"][i])
    return q, pos

  def observation(self, raw, last_action, dt: float) -> np.ndarray:
    q, pos = self.beams(raw)
    qd = np.zeros(2) if self.prev_pos is None or dt <= 0 else (pos - self.prev_pos) / dt
    self.prev_pos = pos
    return np.concatenate([
      self.gyro_body(raw),                                  # gyro (3)
      self.gravity_body(),                                  # gravity (3)
      [math.sin(self.yaw), math.cos(self.yaw)],             # heading (2)
      np.sin(2 * q), np.cos(2 * q),                         # beam_angle (4)
      qd,                                                   # beam_vel (2)
      last_action,                                          # last_action (2)
    ])

  def percents(self, beam_speeds) -> list:
    """Beam speeds (rad/s, per joint) -> Double Motor speed % per hub motor."""
    out = [0.0, 0.0]
    for k, j in enumerate(JOINTS):
      c = self.motor[j]
      out[c["hub_motor"]] = max(-100.0, min(100.0, c["cmd_sign"] * beam_speeds[k] / c["rad_s_per_percent"]))
    return out


# ===================================================================================================================
# The policy (a small neural network, run with numpy)
# ===================================================================================================================
class Policy:
  def __init__(self, path: Path):
    p = np.load(path)
    self.mean, self.std, self.eps = p["obs_mean"], p["obs_std"], float(p["obs_eps"])
    self.layers = [(p[f"W{i}"], p[f"b{i}"]) for i in range(int(p["n_layers"]))]
    self.max_beam_speed = float(p["max_beam_speed"])
    self.control_dt = float(p["control_dt"])
    self.direction = int(p["direction"])

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    x = (obs - self.mean) / (self.std + self.eps)
    for i, (W, b) in enumerate(self.layers):
      x = W @ x + b
      if i < len(self.layers) - 1:
        x = np.where(x > 0, x, np.expm1(x))   # ELU
    return x


# ===================================================================================================================
# Modes
# ===================================================================================================================
def drive(hub, cal: dict, policy: Policy, seconds: float, log_path: Path | None, sim: bool) -> None:
  senses = Senses(hub, cal)
  slowest = min(c["rad_s_per_percent"] * 100 for c in cal["motors"].values())
  if slowest < policy.max_beam_speed:
    print(f"WARNING: your motors reach {slowest:.1f} rad/s but the policy asks for up to {policy.max_beam_speed} rad/s."
          f" Set MAX_BEAM_SPEED in real_task.py to about {0.9 * slowest:.0f} and train again.")
  print("Keep the robot still on the table: measuring the gyroscope...")
  senses.measure_gyro_bias()
  where = "toward -x (in reverse)" if policy.direction < 0 else "forward"
  how_long = "until Ctrl-C" if math.isinf(seconds) else f"for {seconds:.0f} s (Ctrl-C stops it sooner)"
  print(f"Driving {where} {how_long}, deciding every {policy.control_dt * 1000:.0f} ms.")

  rows, last_action = [], np.zeros(2)
  t0 = hub.now()
  next_t, prev_t, next_print = t0, t0, 1.0
  try:
    while hub.now() - t0 < seconds:
      raw = hub.read()
      senses.update_orientation(raw)
      if hub.now() >= next_t - 1e-4:
        now = hub.now()
        obs = senses.observation(raw, last_action, now - prev_t)
        prev_t = now
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -obs[5]))))
        if tilt > STOP_TILT_DEG:
          print(f"tipped {tilt:.0f} degrees: stopping")
          break
        action = policy(obs)
        speeds = action * policy.max_beam_speed
        pct = senses.percents(speeds)
        hub.run(pct)
        last_action = action
        rows.append([round(now - t0, 3), *np.round(obs, 4), *np.round(action, 4), *np.round(pct, 1)])
        next_t += policy.control_dt
        if now - t0 >= next_print:
          extra = ""
          if sim:
            x, y, h = hub.where()
            extra = f"   (true position x={x:+.2f} m, y={y:+.2f} m, heading {h:+.0f} deg)"
          print(f"t={now - t0:4.1f}s  heading {math.degrees(senses.yaw):+5.0f} deg, tilt {tilt:3.0f} deg, "
                f"motors {pct[0]:+4.0f}% {pct[1]:+4.0f}%{extra}")
          next_print += 1.0
      hub.sleep(min(NOTIFY_MS / 1000 / 2, max(0.0, next_t - hub.now())))
  except KeyboardInterrupt:
    print("stopped by Ctrl-C")
  finally:
    hub.stop()
  if log_path and rows:
    names = ["t"] + [f"obs_{n}" for n in ("gyro_x", "gyro_y", "gyro_z", "grav_x", "grav_y", "grav_z", "sin_yaw",
                                          "cos_yaw", "sin2q_L", "sin2q_R", "cos2q_L", "cos2q_R", "qd_L", "qd_R",
                                          "last_L", "last_R")] + ["action_L", "action_R", "pct_motor0", "pct_motor1"]
    with open(log_path, "w", newline="") as f:
      csv.writer(f).writerows([names, *rows])
    print(f"log: {log_path}  (columns _L/_R are the joints {JOINTS[0]} / {JOINTS[1]})")


def check(hub, cal: dict) -> None:
  """Print what the policy would see. Tilt, turn and spin the beams by hand to check the signs (see README.md)."""
  senses = Senses(hub, cal)
  print("Keep the robot still: measuring the gyroscope...")
  senses.measure_gyro_bias()
  print("Now move it by hand. Ctrl-C to stop.")
  prev, last = hub.now(), hub.now()
  try:
    while True:
      raw = hub.read()
      senses.update_orientation(raw)
      if hub.now() - last >= 0.25:
        obs = senses.observation(raw, np.zeros(2), hub.now() - prev)
        prev = last = hub.now()
        q, _ = senses.beams(raw)
        from_vertical = lambda k: (math.degrees(q[k] - Q_VERTICAL[JOINTS[k]]) + 90) % 180 - 90   # -90..90
        g = obs[3:6]
        print(f"gravity {g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f} | turning {obs[2]:+5.2f} rad/s | heading "
              f"{math.degrees(senses.yaw):+5.0f} deg | beams {from_vertical(0):4.0f} {from_vertical(1):4.0f} deg from vertical"
              f" | beam speed {obs[12]:+5.1f} {obs[13]:+5.1f} rad/s")
      hub.sleep(NOTIFY_MS / 1000)
  except KeyboardInterrupt:
    pass


def calibrate(hub: RealHub) -> None:
  def ask(prompt: str, options: str) -> str:
    while True:
      a = input(prompt).strip().lower()[:1]
      if a in options:
        return a

  def average(seconds: float, key: str) -> np.ndarray:
    samples, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
      samples.append(hub.read()[key])
      time.sleep(NOTIFY_MS / 1000)
    return np.mean(samples, axis=0)

  print("\n1/5  Put the robot flat on the table and keep it still, then press Enter.")
  input()
  a_level, bias = average(2.0, "accel"), average(1.0, "gyro")
  print(f"     accelerometer {np.round(a_level)} (size {np.linalg.norm(a_level):.0f} = 1 g)")

  print("2/5  Lift the FRONT of the robot (the end it should drive toward) about 30-45 degrees, hold it still,"
        " and press Enter.")
  input()
  a_tilt = average(1.5, "accel")

  print("3/5  Put it back down. Press Enter, then turn the robot ONE full turn COUNTERCLOCKWISE (seen from above) by"
        " hand, back to where it started, and press Enter again.")
  input()
  turned, t_prev = np.zeros(3), time.monotonic()
  import threading
  done = threading.Event()
  threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
  while not done.is_set():
    time.sleep(NOTIFY_MS / 1000)
    t = time.monotonic()
    turned += (hub.read()["gyro"] - bias) * (t - t_prev)
    t_prev = t
  line = a_level / np.linalg.norm(a_level)
  along = turned @ line
  off_axis = math.degrees(math.acos(min(1.0, abs(along) / max(np.linalg.norm(turned), 1e-9))))
  if off_axis > 25:
    print(f"     WARNING: the turn was {off_axis:.0f} degrees off vertical. Turn it flat on the table and redo.")
  accel_up_sign = 1.0 if along > 0 else -1.0       # turning counterclockwise from above = turning about "up"
  gyro_rad_s_per_count = 2 * math.pi / abs(along)
  z = accel_up_sign * line
  up_tilt = accel_up_sign * a_tilt / np.linalg.norm(a_tilt)
  x = up_tilt - (up_tilt @ z) * z                   # lifting the front tips "up" toward the front
  if np.linalg.norm(x) < 0.2:
    sys.exit("The tilt in step 2 was too small. Run --calibrate again and lift the front higher.")
  x /= np.linalg.norm(x)
  y = np.cross(z, x)
  R = np.array([x, y, z])                           # rows: body x, y, z in hub axes
  print(f"     gyroscope: {gyro_rad_s_per_count * 180 / math.pi:.4f} deg/s per count;"
        f" accelerometer reports {'up' if accel_up_sign > 0 else 'down'}")

  print("4/5  Motors. Hold the robot in the air so the beams can turn freely.")
  motors = {}
  for i, name in enumerate(("MOTOR_LEFT", "MOTOR_RIGHT")):
    input(f"     Press Enter to turn {name} for 2 seconds...")
    start = hub.read()["pos_deg"][i]
    t0 = time.monotonic()
    pct = [0, 0]
    pct[i] = CAL_SPEED_PERCENT
    hub.run(pct)
    time.sleep(2.0)
    turned_deg = hub.read()["pos_deg"][i] - start
    dt = time.monotonic() - t0
    hub.stop()
    side = {"l": "left", "r": "right"}[ask("     Which beam turned: the one on the robot's Left or Right side? [l/r] ", "lr")]
    cw = ask("     Standing at the robot's LEFT side, did that beam turn clockwise? [y/n] ", "yn") == "y"
    joint = next(j for j in JOINTS if SIDE[j] == side)
    plus_cw = PLUS_TURN_SEEN_FROM_LEFT[joint] == "clockwise"
    cmd_sign = 1 if cw == plus_cw else -1
    enc_sign = (1 if turned_deg > 0 else -1) * cmd_sign
    rate = math.radians(abs(turned_deg)) / dt / CAL_SPEED_PERCENT
    print(f"     {name} -> beam on the {side} ({joint}), {rate * 100:.1f} rad/s at 100 % (measured at {CAL_SPEED_PERCENT} %)")
    motors[joint] = {"hub_motor": i, "cmd_sign": cmd_sign, "enc_sign": enc_sign, "rad_s_per_percent": rate}
  if len(motors) != 2:
    sys.exit("Both motors seemed to drive the same beam. Run --calibrate again.")

  input("     Press Enter to run both motors at 100 % for 2 seconds (top speed)...")
  start = hub.read()["pos_deg"]
  t0 = time.monotonic()
  hub.run([100, 100])
  time.sleep(2.0)
  top = np.radians(np.abs(hub.read()["pos_deg"] - start)) / (time.monotonic() - t0)
  hub.stop()
  print(f"     top speed {top[0]:.1f} and {top[1]:.1f} rad/s (real_task.py MAX_BEAM_SPEED must be below this)")

  print("5/5  Turn both beams by hand so each points straight up and down (vertical), then press Enter.")
  input()
  abs_now = hub.read()["abs_deg"]
  for j, c in motors.items():
    c["abs_deg_at_vertical"] = float(abs_now[c["hub_motor"]])

  cal = {"R_hub_to_body": R.tolist(), "accel_up_sign": accel_up_sign, "gyro_rad_s_per_count": gyro_rad_s_per_count,
         "motors": motors, "top_speed_rad_s": top.tolist()}
  CALIBRATION.write_text(json.dumps(cal, indent=2))
  print(f"\nwrote {CALIBRATION}. Next: python drive_real.py --check")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("policy", nargs="?", help="policy.npz from export_policy.py")
  ap.add_argument("--calibrate", action="store_true", help="learn how the hub and motors are mounted")
  ap.add_argument("--check", action="store_true", help="print the policy's inputs live, motors off")
  ap.add_argument("--sim", action="store_true", help="use a simulated Double Motor (needs mujoco)")
  ap.add_argument("--seconds", type=float, default=None,
                  help="stop after this long (default: run until Ctrl-C; 10 s with --sim)")
  ap.add_argument("--seed", type=int, default=0, help="--sim: starting pose and hub setup")
  ap.add_argument("--card-color", default=CARD_COLOR, help=f"Connection Card colour (default {CARD_COLOR}; 'any' = no filter)")
  ap.add_argument("--card-serial", default=CARD_SERIAL, help=f"Connection Card number (default {CARD_SERIAL}; 'any' = no filter)")
  args = ap.parse_args()

  if args.sim:
    hub = SimHub(args.seed)
    cal = hub.calibration()
  else:
    hub = RealHub(args.card_color, args.card_serial)
    if args.calibrate:
      try:
        calibrate(hub)
      finally:
        hub.close()
      return
    if not CALIBRATION.exists():
      hub.close()
      sys.exit(f"{CALIBRATION.name} not found: run  python drive_real.py --calibrate  first.")
    cal = json.loads(CALIBRATION.read_text())

  try:
    if args.check:
      check(hub, cal)
    elif args.policy:
      log = HERE / f"run-{time.strftime('%Y%m%d-%H%M%S')}{'-sim' if args.sim else ''}.csv"
      seconds = args.seconds if args.seconds is not None else (10.0 if args.sim else math.inf)
      drive(hub, cal, Policy(Path(args.policy)), seconds, log, args.sim)
    else:
      ap.print_help()
  finally:
    hub.close()


if __name__ == "__main__":
  main()
