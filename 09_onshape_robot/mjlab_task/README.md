# Train the robot to drive straight with mjlab

`drive_task.py` turns your Onshape robot (`../onshape/robot.xml`) into an [mjlab](https://github.com/mujocolab/mjlab)
reinforcement-learning task. 4096 copies of the robot start slightly off-angle and learn to drive down +x while staying
on the line y = 0. The policy outputs one speed per wheel.

mjlab trains only on an NVIDIA GPU, so training runs on the cluster. (On a Mac it can only play back.)

| File | What it is |
|---|---|
| `drive_task.py` | The task: robot, rewards, resets. **Tune this one.** |
| `train.py`, `play.py` | mjlab's own train/play commands, with our task registered first |
| `make_video.py` | Records a movie of a trained policy on the cluster (step 4) |
| `beam_task.py` | The two-beam robots (stand-in and Onshape), forward and reverse, simulation only |
| `real_task.py` | The Onshape beam robot trained for the real Double Motor (see `../real_robot/`) |
| `export_policy.py` | Turns a `real_task.py` checkpoint into `policy.npz` for `../real_robot/drive_real.py` |
| `train.sbatch` | A Slurm job that trains on a GPU, for running by hand on the cluster |

How it works in plain words, with the rewards and the words used (policy, iteration, checkpoint): [../docs/how-drive-straight-works.md](../docs/how-drive-straight-works.md).

Everything runs from your laptop with `runIt.py`, in the `mujoco-hpc` folder (venv active, VPN on if off campus).

## 1. Install

Nothing extra: `python3 setup.py` (the same setup as the bouncing-ball demo) already installed mjlab on the cluster.
If you set up before mjlab was added, rerun `python3 setup.py --reinstall`.

## 2. Check the task loads (short run)

```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
  09_onshape_robot/mjlab_task/train.py Drive-Straight --env.scene.num-envs 64
```
Under "job output" you should see training iterations with a mean reward. If it stops with an error instead, the
message says what to fix in `drive_task.py` (see Tuning).

## 3. Train

Same command with more robots and time (`runIt.py` waits until the job ends, so keep the terminal open). A few minutes
is enough to see it learn; add `--agent.max-iterations 2000` (about 13 minutes, use `--time 00:45:00`) for a better driver:
```
python3 runIt.py --gpu --time 00:15:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
  09_onshape_robot/mjlab_task/train.py Drive-Straight --env.scene.num-envs 4096
```
The logs and checkpoints (`model_<n>.pt`, saved every 50 iterations) come back to
`runs/train-<time>/logs/rsl_rl/drive_straight/<date>/` on your laptop. The last one is `model_499.pt` for the default 500
iterations, `model_1999.pt` for 2000.

## 4. Watch the result: a movie made on the cluster

`make_video.py` loads a checkpoint, drives one robot for 10 seconds on a GPU node, records an `.mp4`, and prints where the
robot is each second (x forward, y sideways, heading; straight driving keeps y and heading near 0). `runIt.py` brings the
movie home and opens it. Send it your checkpoint with a third `--with` (find the file with `ls runs/train-*/logs/rsl_rl/drive_straight/*/`):
```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
  --with runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/drive_straight/XXXX-XX-XX_XX-XX-XX/model_499.pt \
  09_onshape_robot/mjlab_task/make_video.py model_499.pt
```
Do not type the `X`s. They stand for the folder names in your own `runs/` folder: type `runs/train-` and press Tab to
complete each name. Use the checkpoint name you have (`model_1999.pt` if you trained 2000 iterations) in both places.

Optional, watching live on your Mac instead: mjlab's `play.py` needs **Python 3.11 to 3.13** (with 3.14 pip only finds a
very old mjlab and it fails with `No module named 'mjlab.actuator'`). Make an environment with Python 3.13
(`python3.13 -m venv ~/mjlab-play`, activate it, `python -m pip install mjlab`) and run `python play.py Drive-Straight
--checkpoint-file <your model file>` from this folder. You do not need this for the movie.

## The two-beam robot (`Beam-Drive-Straight`)

`beam_task.py` is the same idea for a different robot: [`../beams/robot.xml`](../beams/robot.xml), the same size as the
Onshape chassis, but each motor spins a straight 10 cm **beam** (fixed at its middle) instead of a wheel. A beam
only pushes while one of its ends is on the floor, and two beams turning in step flip the robot over. So unlike the
wheel robot, "both motors forward" does not drive straight: the policy has to learn a gait (keep the beams out of
step, don't flip, steer). To help it, it also sees each beam's angle and which way is down.

Driving the beams by hand shows what to expect (both at 10 rad/s, 5 s, plain MuJoCo):

| Beams | Result |
|---|---|
| in step | flips over and goes backward |
| a quarter-turn apart | forward at about 0.15 m/s, tilts up to 40 degrees, drifts 26 degrees off heading |

Run it like the wheel task, sending `beam_task.py` and the `beams` folder instead of `drive_task.py` and `onshape`.
It trains for 1000 iterations by default (about 15 to 25 minutes on a GPU node; use `--time 00:45:00`):
```
python3 runIt.py --gpu --time 00:45:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/beam_task.py --with 09_onshape_robot/beams \
  09_onshape_robot/mjlab_task/train.py Beam-Drive-Straight --env.scene.num-envs 4096
```
Checkpoints come back to `runs/train-<time>/logs/rsl_rl/beam_drive_straight/<date>/`. For the movie, add the task name
after the checkpoint:
```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/beam_task.py --with 09_onshape_robot/beams \
  --with runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/beam_drive_straight/XXXX-XX-XX_XX-XX-XX/model_999.pt \
  09_onshape_robot/mjlab_task/make_video.py model_999.pt Beam-Drive-Straight
```
Measured (a test run on a laptop CPU with 512 robots, 400 iterations, about 15 minutes, mjlab 1.6.1): the policy finds
a forward gait by iteration 50 (mean reward 4.8, from -0.2) and reaches 7.1 by 400. With the beams kept out of step
the whole time, in 10 s the robot went **1.38 m** (0.14 m/s, target 0.15) and stayed **within 2 cm** of the line, with
heading wobbling between -17 and +2 degrees and ending at -1. Training on the cluster with 4096 robots and 1000 iterations
should do at least as well.

### Your Onshape beam robot (`Onshape-Beam-Drive-Straight`)

The same task, on your own Onshape export in [`../onshape/robot.xml`](../onshape/robot.xml): a LEGO chassis with a
9.6 cm Technic beam on each motor, 5.2 cm apart. Onshape gives the chassis 0.097 kg; the task sets it to **0.3 kg**
(0.32 kg in all), scaling its inertia to match. Change that number, the last entry for this robot in `ROBOTS` at the
top of `beam_task.py`, or set it to `None` to use the CAD mass. Send the `onshape` folder instead of `beams`:
```
python3 runIt.py --gpu --time 00:45:00 --mem 16G --cpus 4 --with 09_onshape_robot/mjlab_task/beam_task.py --with 09_onshape_robot/onshape 09_onshape_robot/mjlab_task/train.py Onshape-Beam-Drive-Straight --env.scene.num-envs 4096
```
Checkpoints come back to `runs/train-<time>/logs/rsl_rl/onshape_beam_drive_straight/<date>/`, and the movie is the
command above with `--with 09_onshape_robot/onshape`, that folder in the checkpoint path, and
`Onshape-Beam-Drive-Straight` as the last word.

**Driving in reverse (`Onshape-Beam-Drive-Backward`):** the same robot and settings, but it is rewarded for going down
**-x** at `TARGET_SPEED` while still facing +x (rear end first). Train and film it with the same commands, using the
task name `Onshape-Beam-Drive-Backward`; its checkpoints go to `logs/rsl_rl/onshape_beam_drive_backward/`. The
direction is the last entry for each task in `ROBOTS` (`+1` forward, `-1` reverse).

Things to know about the export: its joints are still called `left_wheel` and `right_wheel` (and swapped: `left_wheel`
turns the beam on the right), its two motor shafts point opposite ways (the same speed turns the beams in opposite
directions; the policy learns around that), and its chassis sticks out about 6 cm in front of the shafts. The
`frictionloss 0.1` that onshape-to-robot writes on the joints is a placeholder that would use up most of the motor's
torque, so `beam_task.py` replaces the joint drag with the same small-gearmotor values as the stand-in
(`JOINT_DAMPING`, `JOINT_ARMATURE`, `JOINT_FRICTIONLOSS`). If you re-export from Onshape, keep the assembly's part names
(`chassis`, and the two joints), or change them in `ROBOTS` at the top of `beam_task.py`.

### For the real robot (`Onshape-Beam-Real`, `Onshape-Beam-Real-Backward`)

`real_task.py` trains the Onshape beam robot so the policy can drive the **real** robot (a LEGO Double Motor over
Bluetooth). It is a separate file and leaves every task above unchanged. The policy sees only what the Double Motor
measures, acts 20 times a second, and is trained with Bluetooth delay, sensor noise, and a different friction, mass
and motor strength for every robot. `Onshape-Beam-Real-V2` (and `-V2-Backward`) is a second version made after the
first real runs flipped and crept left: gentler, more upright, less sideways creep (table in the real-robot README).
Send `beam_task.py` with it. Training, exporting to `policy.npz`
(`export_policy.py`), calibrating and driving are all in [`../real_robot/README.md`](../real_robot/README.md).

**Surface (every robot and task, and `drive.py`):** a flat wood tabletop under a plastic (LEGO ABS) robot, sliding
friction `TABLE_FRICTION = 0.35` for every part (wheels, beams, chassis, skid). It is set in `beam_task.py`,
`drive_task.py` and `drive.py`, and the floors in the `scene.xml` files say the same. MuJoCo normally takes the larger
of two touching surfaces' friction, and mjlab's floor is 1.0, so the robot's parts get contact priority to make 0.35
the value actually used. (Results measured before this change, including the tables and numbers in this README, used
1.0 everywhere: the chassis was meant to slide at 0.01, but the floor's 1.0 won.)

Tuning is in `beam_task.py`: `TARGET_SPEED` (0.15 m/s), `MAX_BEAM_SPEED`, `BEAM_TORQUE_LIMIT`, `MAX_TILT_DEG`, and the
reward weights. To change the robot itself (beam length, motor spacing, masses), edit `../beams/robot.xml`, and look at it
first with `python3 runIt.py 09_onshape_robot/drive.py beams/scene.xml --with 09_onshape_robot/beams`.

## Tuning (all in `drive_task.py`)

- **Wheels barely turn or the robot crawls:** raise `WHEEL_KD` or `WHEEL_TORQUE_LIMIT`. The motor pushes with
  `WHEEL_KD * (asked speed - actual speed)` N*m, capped at the limit.
- **Robot jerks and flips:** lower `WHEEL_TORQUE_LIMIT`, or make the `action_rate` reward weight more negative.
- **It drives straight but slowly:** `TARGET_SPEED` (0.5 m/s) is far more than this robot can do (driving both wheels at
  8 rad/s by hand in `drive.py` moves it about 0.03 m/s), so the reward simply keeps paying for more speed. Lower it to
  about 0.15 if you want it to hold a steady speed, or raise `MAX_WHEEL_SPEED` and the wheel torque to go faster.
- **It wiggles along the line:** raise the `stay_on_line` and `face_forward` weights, lower `START_YAW`.
- **Make it harder:** raise `START_YAW`, or add friction and mass randomization (mjlab's `mjlab.envs.mdp.dr`) so the
  policy copes with the jerkiness of your rod wheels and not one exact friction value.

## What to expect

A run of 4096 robots takes about 3 to 4 minutes for 500 iterations and about 13 minutes for 2000 on a Pax GPU node.
`Mean reward` starts near -0.1 and reaches about +2 after 500 iterations and about +2.6 after 2000.

Results from the movies (10 seconds, starting pointing 0.4 radians off-axis):

| Robot and training | Distance forward | Sideways (y) | Heading |
|---|---|---|---|
| **Sample robot** (the one in `onshape/` until you export your own), 300 iterations | 4.7 m (0.5 m/s, the target speed) | within 0.1 m | within 2 degrees |
| Author's own Onshape robot with rod wheels, 500 iterations | 1.15 m | down to -0.5 m, ends -0.13 m | wanders between -43 and +38 degrees |
| Author's own Onshape robot with rod wheels, 2000 iterations | 1.05 m | stays within about 0.25 m | stays within about 27 degrees |

With the sample robot the mean reward reaches about +9.8 within 150 iterations (about a minute) and the robot drives
straight at the target speed. Your own robot will do better or worse depending on its wheels: the author's rod wheels are
jerky and slippery, so it crawls at about 0.1 m/s (three times faster than the hand-driven plan in `drive.py`) and still
wobbles in heading. For a robot like that, more training helps, and so do the Tuning ideas above (a gentler
`WHEEL_TORQUE_LIMIT`, heavier steering penalties, a lower `TARGET_SPEED`).

Tested on Pax (GPU node `pax050`, mjlab 1.6.0, torch 2.11 for CUDA 12.8, warp 1.14). The cluster's GPU driver is CUDA
12.9, which is why `03_setup_env.sh` installs CUDA 12 builds of PyTorch and Warp. If you see "NVIDIA driver is too old",
that did not happen: run `python3 setup.py clean` to start from scratch.
