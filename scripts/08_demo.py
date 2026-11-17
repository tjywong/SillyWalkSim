"""Two paddles hit a ball back and forth between them; every bounce peaks at APEX metres.
Rendered to a video (the cluster has no screen).

    python 08_demo.py

Each paddle can slide up and down and tilt. How the paddle that is about to hit aims:
  1. Predict where and how fast the falling ball will meet the paddle's face.
  2. Pick the velocity the ball should leave with: upward fast enough to peak at APEX, sideways fast enough
     to come down on the other paddle.
  3. Tilt the paddle so its face points along (leave velocity - arrive velocity). The contact has no
     friction, so the face can only push along that direction.
  4. Pick how fast to swing up, using a small bounce model (in the direction the face points):
        leave speed = a * (arrive speed) + b * (paddle speed) + c
  5. After each bounce, compare the real peak with APEX and refit a, b, c, so the aim keeps improving.
"""
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import imageio
import mujoco
import numpy as np

SECONDS = 10.0         # how long to simulate
OUT = "bounce.mp4"     # the video file to write
APEX = 1.2             # m; height every bounce should peak at (the ball's centre)
PADDLE_X = (0.5, 1.5)  # m; where the two paddles stand

# The world, described in MuJoCo's XML format.
XML = f"""
<mujoco>
  <compiler angle="radian"/>   <!-- joint ranges in radians (MuJoCo's default is degrees) -->
  <!-- A small timestep keeps every hit the same; at the default 2 ms, peaks scatter by about +-7 cm. -->
  <option timestep="0.0005" integrator="implicitfast"/>

  <default>
    <default class="paddle">
      <!-- condim="1": no friction between ball and paddle, so a hit pushes only straight out of the face. -->
      <geom type="box" size=".25 .25 .02" rgba=".8 .8 .8 1" mass="5" condim="1"/>
    </default>
  </default>

  <worldbody>
    <light pos="1 -1 4"/>
    <camera name="side" pos="1 -3 1.5" xyaxes="1 0 0 0 .3 1"/>

    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>

    <body name="ball" pos="{PADDLE_X[0]} 1 1.5">
      <freejoint/>
      <!-- solref = "-stiffness -damping" of the contact. Less damping = bouncier. -->
      <geom type="sphere" size=".1" rgba="1 .3 .2 1" solref="-20000 -100" condim="1"/>
    </body>

    <!-- The paddles: flat plates that slide up and down and tilt toward each other. -->
    <body name="paddle0" pos="{PADDLE_X[0]} 1 .5">
      <joint name="lift0" type="slide" axis="0 0 1" range="-.3 .3"/>
      <joint name="tilt0" type="hinge" axis="0 1 0" range="-.6 .6"/>
      <geom class="paddle"/>
    </body>
    <body name="paddle1" pos="{PADDLE_X[1]} 1 .5">
      <joint name="lift1" type="slide" axis="0 0 1" range="-.3 .3"/>
      <joint name="tilt1" type="hinge" axis="0 1 0" range="-.6 .6"/>
      <geom class="paddle"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Per paddle: lift height (m, relative to z = .5) and lift speed (m/s) together track a moving target;
         tilt holds an angle (rad, positive = face turned toward +x). -->
    <position name="lift_pos0" joint="lift0" kp="40000"/>
    <velocity name="lift_vel0" joint="lift0" kv="2000"/>
    <position name="tilt0"     joint="tilt0" kp="20000" kv="100"/>
    <position name="lift_pos1" joint="lift1" kp="40000"/>
    <velocity name="lift_vel1" joint="lift1" kv="2000"/>
    <position name="tilt1"     joint="tilt1" kp="20000" kv="100"/>
  </actuator>
</mujoco>
"""

# ---- the paddle's swing ------------------------------------------------------------------------------
G = 9.81
REST = -0.25           # m; paddle height while waiting (relative to its body pos)
HIT = -0.10            # m; paddle height when it meets the ball
COAST = 0.03           # s; keep moving at the hitting speed this long, while the ball is still squashed on it
DECEL = 4 * G          # m/s^2; how hard the paddle brakes after the hit (more than g, so the ball flies off)
RETURN = 0.4           # s; time to glide back down to REST
REACH = 0.1 + 0.02     # m; ball radius + half the paddle's thickness: ball centre to paddle centre at contact
HIT_Z = 0.5 + HIT      # height of a paddle's centre at the moment of the hit


def swing(tau, v):
    """Paddle (height, speed) tau seconds after the swing starts, for a hit at upward speed v."""
    a = v * v / (2 * (HIT - REST))            # constant push that reaches speed v exactly at HIT
    t_up, t_brake = v / a, v / DECEL
    if tau < t_up:                            # speed up
        return REST + 0.5 * a * tau**2, a * tau
    tau -= t_up
    if tau < COAST:                           # hit the ball at a steady speed
        return HIT + v * tau, v
    top = HIT + v * COAST
    tau -= COAST
    if tau < t_brake:                         # brake; the ball flies on
        return top + v * tau - 0.5 * DECEL * tau**2, v - DECEL * tau
    top += v * t_brake - 0.5 * DECEL * t_brake**2
    tau -= t_brake
    if tau < RETURN:                          # smooth glide back down
        x = tau / RETURN
        return top + (REST - top) * (3 * x**2 - 2 * x**3), (REST - top) * (6 * x - 6 * x**2) / RETURN
    return None                               # swing finished


# ---- the bounce model, along the face's normal: leave = a*arrive + b*paddle + c, refit after every bounce --
GUESS = np.array([0.37, 1.44, 0.07])        # a good first guess for the ball above; it gets refined
TRUST_GUESS = 10.0                          # how many bounces' worth of weight the guess keeps
model_abc = GUESS.copy()
seen = []                                   # one row per bounce: (arrive speed, paddle speed, leave speed)


def refit():
    """Least-squares fit of a, b, c to every bounce so far, pulled toward GUESS.
    (Bounces all look alike, so without that pull a few of them can give a wild fit.)"""
    X = np.array([[vin, vp, 1.0] for vin, vp, _ in seen])
    y = np.array([vout for _, _, vout in seen])
    model_abc[:] = np.linalg.solve(X.T @ X + TRUST_GUESS * np.eye(3), X.T @ y + TRUST_GUESS * GUESS)


def plan_hit(i, pos, vel):
    """Plan paddle i's hit on a ball now at pos=(x, z) moving at vel=(vx, vz).
    Returns (seconds until contact, tilt angle, paddle speed, face normal, arrive velocity, contact height)."""
    p = np.array([PADDLE_X[i], HIT_Z])
    other_x = PADDLE_X[1 - i]
    n = np.array([0.0, 1.0])
    for _ in range(5):   # contact point, arrive velocity and face direction depend on each other: iterate
        # Contact: ball centre is REACH from the face, i.e. n . (ball(t) - p) = REACH. A quadratic in t.
        A = -0.5 * G * n[1]
        B = n @ vel
        C = n @ (pos - p) - REACH
        disc = B * B - 4 * A * C
        if disc < 0:
            return None
        t = max((-B - math.sqrt(disc)) / (2 * A), (-B + math.sqrt(disc)) / (2 * A))
        if t < 0:
            return None
        ball_at = pos + vel * t + np.array([0.0, -0.5 * G * t * t])
        v_in = vel + np.array([0.0, -G * t])
        # Leave velocity: up to APEX, and across to land centred on the other paddle (whose face will
        # mirror this one, so the ball should arrive REACH out from that paddle's centre).
        up = math.sqrt(2 * G * max(APEX - ball_at[1], 0.05))
        flight = 2 * up / G
        v_out = np.array([(other_x - REACH * n[0] - ball_at[0]) / flight, up])
        dv = v_out - v_in
        n = dv / np.linalg.norm(dv)
    a, b, c = model_abc
    arrive, leave = -(v_in @ n), v_out @ n
    speed = min(max((leave - a * arrive - c) / (b * n[1]), 0.2), 3.5)
    return t, math.atan2(n[0], n[1]), speed, n, (arrive, speed * n[1]), ball_at[1]


# ---- run it --------------------------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_string(XML)   # build the physics model
data = mujoco.MjData(model)                   # the state: positions, velocities, time
renderer = mujoco.Renderer(model, 480, 640)   # draws pictures of the state
ball = data.body("ball")
ball_vel = data.joint(0)                      # the ball's freejoint: qvel = (vx, vy, vz, wx, wy, wz)

receiver = 0                 # which paddle hits next
swing_start = [None, None]   # per paddle: time its swing started, or None while it waits
swing_v = [0.0, 0.0]         # per paddle: upward speed planned for its hit
tilt = [0.0, 0.0]      # tilt each paddle is holding
last_hit = None        # (face normal, (arrive, paddle) speeds, contact height) until its peak is measured
prev_vz = 0.0
peaks = []

FPS = 30
frames = []
while data.time < SECONDS:
    # Advance the physics by one video frame (many small steps), then take a picture.
    while data.time < len(frames) / FPS:
        x, z = ball.xpos[0], ball.xpos[2]
        vx, vz = ball_vel.qvel[0], ball_vel.qvel[2]

        # At each peak: record how this bounce went and refit the model.
        if prev_vz > 0 >= vz:
            peaks.append((x, z))
            if last_hit is not None:
                n, (arrive, paddle), z_hit = last_hit
                leave_up = math.sqrt(2 * G * max(z - z_hit, 0.0))   # launch speed upward that reaches this peak
                seen.append((arrive, paddle, n @ np.array([vx, leave_up])))
                refit()
                last_hit = None
        prev_vz = vz

        # Until the swing starts: keep re-planning the hit, hold the planned tilt, start just in time.
        r = receiver
        if swing_start[r] is None:
            plan = plan_hit(r, np.array([x, z]), np.array([vx, vz]))
            if plan is not None:
                t_hit, tilt[r], v, n, speeds, z_hit = plan
                if vz < 0 and t_hit <= 2 * (HIT - REST) / v:
                    swing_start[r], swing_v[r] = data.time + t_hit - 2 * (HIT - REST) / v, v
                    last_hit = (n, speeds, z_hit)
        elif data.time > swing_start[r] + 2 * (HIT - REST) / swing_v[r] + COAST:
            receiver = 1 - r                   # ball launched: the other paddle starts aiming now

        for i in (0, 1):
            height, speed = REST, 0.0
            if swing_start[i] is not None and data.time >= swing_start[i]:
                target = swing(data.time - swing_start[i], swing_v[i])
                if target is None:             # swing done: wait at REST for the next one
                    swing_start[i] = None
                else:
                    height, speed = target
            data.actuator(f"lift_pos{i}").ctrl[0] = height
            data.actuator(f"lift_vel{i}").ctrl[0] = speed
            data.actuator(f"tilt{i}").ctrl[0] = tilt[i]

        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="side")
    frames.append(renderer.render())

renderer.close()
imageio.mimwrite(OUT, frames, fps=FPS)
print(f"wrote {OUT}: {len(frames)} frames, {SECONDS:g} s of simulation")
print("bounce peaks (m):", " ".join(f"{z:.3f}" for _, z in peaks))
print("peak x (m):      ", " ".join(f"{x:.3f}" for x, _ in peaks))
