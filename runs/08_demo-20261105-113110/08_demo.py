"""A ball drops onto a paddle, which hits it back up so every bounce peaks at APEX metres.
Rendered to a video (the cluster has no screen).

    python 08_demo.py

How the paddle aims each hit:
  1. While the ball falls, predict when and how fast it will reach the hitting height.
  2. Pick the paddle speed that should launch it just fast enough to reach APEX:
        launch speed = a * (incoming speed) + b * (paddle speed) + c
  3. Swing the paddle up so it is moving at exactly that speed when the ball arrives.
  4. After each bounce, compare the real peak with APEX and refit a, b, c, so the aim keeps improving.
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

# The world, described in MuJoCo's XML format.
XML = """
<mujoco>
  <!-- A small timestep keeps every hit the same; at the default 2 ms, peaks scatter by about +-7 cm. -->
  <option timestep="0.0005" integrator="implicitfast"/>

  <worldbody>
    <light pos="0 -1 4"/>
    <camera name="side" pos="0 -3 1.5" xyaxes="1 0 0 0 .3 1"/>

    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>

    <body name="ball" pos="1 1 1.5">
      <freejoint/>
      <!-- solref = "-stiffness -damping" of the contact. Less damping = bouncier. -->
      <geom type="box" size=".1 .1 .1" rgba="0 0 0 1" solref="-20000 -100"/>
    </body>

    <!-- The paddle: a flat box under the ball that can only slide up and down. -->
    <body name="paddle" pos="1 1 .5">
      <joint name="lift" type="slide" axis="0 0 1" range="-.3 .3"/>
      <geom type="box" size=".25 .25 .02" rgba=".8 .8 .8 1" mass="5"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Together these track a moving target: ctrl[0] = height (m, relative to z = .5), ctrl[1] = speed (m/s). -->
    <position joint="lift" kp="40000"/>
    <velocity joint="lift" kv="2000"/>
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
BALL_HALF = 0.1        # m; half the ball's height
PADDLE_TOP = 0.5 + HIT + 0.02 + BALL_HALF   # height of the ball's centre at the moment of the hit


def swing(tau, v):
    """Paddle (height, speed) tau seconds after the swing starts, for a hit at speed v."""
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


def swing_length(v):
    return v / (v * v / (2 * (HIT - REST))) + COAST + v / DECEL + RETURN


# ---- the bounce model: launch speed = a*incoming + b*paddle + c, refit after every bounce --------------
GUESS = np.array([0.35, 1.4, 0.05])        # a good first guess for the ball above; it gets refined
TRUST_GUESS = 10.0                          # how many bounces' worth of weight the guess keeps
model_abc = GUESS.copy()
seen = []                                   # one row per bounce: (incoming speed, paddle speed, launch speed)


def refit():
    """Least-squares fit of a, b, c to every bounce so far, pulled toward GUESS.
    (Bounces all look alike, so without that pull a few of them can give a wild fit.)"""
    X = np.array([[vin, vp, 1.0] for vin, vp, _ in seen])
    y = np.array([vout for _, _, vout in seen])
    model_abc[:] = np.linalg.solve(X.T @ X + TRUST_GUESS * np.eye(3), X.T @ y + TRUST_GUESS * GUESS)


# ---- run it --------------------------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_string(XML)   # build the physics model
data = mujoco.MjData(model)                   # the state: positions, velocities, time
renderer = mujoco.Renderer(model, 480, 640)   # draws pictures of the state
ball = data.body("ball")

swing_start = None     # time the current swing started, or None while the paddle waits
swing_v = 0.0          # paddle speed planned for this hit
last_hit = None        # (incoming speed, paddle speed) of the latest hit, until its peak is measured
prev_vz = 0.0
peaks = []

FPS = 30
frames = []
while data.time < SECONDS:
    # Advance the physics by one video frame (many small steps), then take a picture.
    while data.time < len(frames) / FPS:
        z, vz = ball.xpos[2], data.qvel[2]

        # At each peak: record how this bounce went and refit the model.
        if prev_vz > 0 >= vz:
            peaks.append(z)
            if last_hit is not None:
                seen.append((*last_hit, math.sqrt(2 * G * max(z - PADDLE_TOP, 0.0))))
                refit()
                last_hit = None
        prev_vz = vz

        # While the ball falls: plan the hit, and start the swing just in time.
        if swing_start is None and vz < 0 and z > PADDLE_TOP:
            t_hit = (vz + math.sqrt(vz * vz + 2 * G * (z - PADDLE_TOP))) / G   # time until the ball reaches PADDLE_TOP
            v_in = G * t_hit - vz                                                # its speed then (downwards)
            v_out = math.sqrt(2 * G * (APEX - PADDLE_TOP))                       # launch speed that peaks at APEX
            a, b, c = model_abc
            v = min(max((v_out - a * v_in - c) / b, 0.2), 3.0)
            t_up = 2 * (HIT - REST) / v
            if t_hit <= t_up:
                swing_start, swing_v = data.time + t_hit - t_up, v
                last_hit = (v_in, v)

        target = (REST, 0.0)
        if swing_start is not None and data.time >= swing_start:
            target = swing(data.time - swing_start, swing_v) or (REST, 0.0)
            if data.time - swing_start > swing_length(swing_v):
                swing_start = None
        data.ctrl[0], data.ctrl[1] = target

        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="side")
    frames.append(renderer.render())

renderer.close()
imageio.mimwrite(OUT, frames, fps=FPS)
print(f"wrote {OUT}: {len(frames)} frames, {SECONDS:g} s of simulation")
print("bounce peaks (m):", " ".join(f"{p:.3f}" for p in peaks))
