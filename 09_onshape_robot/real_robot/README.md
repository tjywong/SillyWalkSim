# Drive the real robot with the trained policy

`drive_real.py` runs a policy trained in simulation on the real two-beam robot: a LEGO Education **Double Motor** with
a Technic beam on each shaft, talking to your laptop over Bluetooth through the `legoeducation` library. The laptop
reads the Double Motor's sensors, runs the policy 20 times a second, and sends back a speed for each motor.

Only a policy from the **real-robot task** works here: `Onshape-Beam-Real` (forward) or `Onshape-Beam-Real-Backward`
(in reverse), in [`../mjlab_task/real_task.py`](../mjlab_task/real_task.py). The simulation-only tasks
(`Onshape-Beam-Drive-Straight` and the rest) are still there and unchanged, but their policies use readings the robot
cannot measure (its own speed and its distance from the line), so they cannot drive it.

| File | What it is |
|---|---|
| `drive_real.py` | Calibrate, check the sensors, drive, or test against a simulated Double Motor (`--sim`) |
| `calibration.json` | Written by `--calibrate`: how the hub and motors are mounted on your robot |
| `run-<time>.csv` | Written by each drive: every reading, decision and motor command, for when something goes wrong |

## 1. Train the real-robot policy (cluster)

From the `mujoco-hpc` folder, VPN on. It needs `beam_task.py` too (the robot and the rewards come from there):
```
python3 runIt.py --gpu --time 01:00:00 --mem 16G --cpus 4 --with 09_onshape_robot/mjlab_task/beam_task.py --with 09_onshape_robot/mjlab_task/real_task.py --with 09_onshape_robot/onshape 09_onshape_robot/mjlab_task/train.py Onshape-Beam-Real --env.scene.num-envs 4096
```
1500 iterations. Use `Onshape-Beam-Real-Backward` to drive in reverse. The checkpoints come back to
`runs/train-<time>/logs/rsl_rl/onshape_beam_real/<date>/`. Optionally watch it first with `make_video.py` (send the
same three `--with`s plus the checkpoint, task name `Onshape-Beam-Real`).

**Version 2 (`Onshape-Beam-Real-V2`, `Onshape-Beam-Real-V2-Backward`)** is the same with the changes made after the
first real runs flipped and crept left (the settings are `V2` in `real_task.py`; version 1 is unchanged):

| Setting | Version 1 | Version 2 | Why |
|---|---|---|---|
| Training episode ends at a tilt of | 70 deg | **60 deg** | keep a margin before a real flip (45 was too strict) |
| `stay_upright` reward weight | 0.5 | **2.0** | |
| Rocking-speed penalty (roll and pitch rate) | none | **0.02** | the real robot rocked nose up/down 2 to 4 times more |
| Top beam speed (`max_beam_speed`) | 12 rad/s | **8 rad/s** | gentler strokes |
| Beam-speed sensor noise | 1 rad/s | **4 rad/s** | the real readings are spiky |
| Command delay | 30 to 80 ms | **30 to 150 ms** | slower Bluetooth |
| Motor strength range | 0.6x to 1.4x | **0.5x to 2.0x** | real motors may push harder |
| `stay_on_line` reward weight | 0.5 | **1.5** | it crept left |
| Sideways-speed penalty | none | **0.5** | stops a slow creep |
| `face_forward` reward weight | 0.5 | **1.0** | hold heading more firmly |
| Alive bonus (per second not flipped) | none | **1.0** | without it, the extra penalties made flipping at once score better than driving, and the first V2 run learned nothing |

Train it with the same command and task name `Onshape-Beam-Real-V2`; its checkpoints go to
`logs/rsl_rl/onshape_beam_real_v2/`. Export and drive it exactly like version 1 (the export reads its 8 rad/s and
the rest from the run's `params` folder).

## 2. Turn the checkpoint into `policy.npz` (cluster)

The robot script has no PyTorch; this step converts the network to plain numbers. Send the checkpoint and the
`params` folder next to it (it holds the settings the policy was trained with):
```
python3 runIt.py --with runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/onshape_beam_real/XXXX-XX-XX_XX-XX-XX/model_1499.pt --with runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/onshape_beam_real/XXXX-XX-XX_XX-XX-XX/params 09_onshape_robot/mjlab_task/export_policy.py model_1499.pt
```
`policy.npz` comes back in `runs/export_policy-<time>/`. (Tab-complete the `X`s, as for the movie.)

## 3. Set up the laptop

A Python 3.11 to 3.13 environment with:
```
python -m pip install "legoeducation>=1.1" numpy mujoco
```
(`mujoco` is only for `--sim`.) Test the whole program first against a simulated Double Motor; nothing connects:
```
python drive_real.py path/to/policy.npz --sim
```
It prints, every second, what the robot thinks (heading, tilt, motor %) and where it really is. If the simulated robot
drives forward and stays near y = 0, the policy and the program are fine, and any trouble on the real robot is the
robot or its calibration.

## 4. Build the robot like the simulation

The policy learned one robot; the closer yours is, the better it drives:
- The Double Motor's two shafts point sideways, each with a Technic beam (about 9.6 cm) **held at its middle**.
- The beams are about 5 cm apart, as in `../onshape/robot.xml`.
- About **0.3 kg** for the chassis (training allowed 0.21 to 0.39 kg). Weigh it.
- A smooth wooden table (training allowed friction 0.25 to 0.5).

## 5. Calibrate (once per build)

`drive_real.py` connects only to the Double Motor with the **purple Connection Card number 5164** (`CARD_COLOR` and
`CARD_SERIAL` at the top of the script), so it never grabs someone else's robot nearby. For a different card, add
`--card-color red --card-serial 0049` (any mode), or `--card-color any --card-serial any` for the first one found.

Turn the Double Motor on, then:
```
python drive_real.py --calibrate
```
It walks you through five steps (about two minutes):
1. Robot flat and still: finds "down".
2. Front lifted: finds "forward". The front is the end it should drive toward.
3. One full turn counterclockwise by hand: scales the gyroscope.
4. Each motor turns for 2 s, robot held in the air; you say which beam turned and which way. Then both run at full
   speed to measure the top speed.
5. Both beams turned by hand to vertical: their zero.

If it says the top speed is below `MAX_BEAM_SPEED` in `real_task.py` (12 rad/s), lower that number and train again.

## 6. Check the sensors

```
python drive_real.py --check
```
Motors stay off. Move the robot by hand and look at the numbers:
- Flat on the table: `gravity` about `+0.00 +0.00 -1.00`.
- Nose down: the first number goes **positive**. Left side down: the second goes **positive** too.
- Turn it counterclockwise (from above): `heading` goes up.
- Turn a beam by hand: its angle changes, and reads about **0 whenever it is vertical** (either end down). The beams
  are listed in the policy's order: first the beam on the robot's **right**, then the one on its left.

If any of these are wrong, calibrate again.

## 7. Drive

Robot on the table, pointing where you want it to go (its heading starts at 0 there):
```
python drive_real.py path/to/policy.npz              (runs until Ctrl-C; --seconds 20 stops on its own)
```
It zeroes the gyroscope (keep the robot still), then drives. **Ctrl-C stops the motors at any time**, and they stop
on their own if it tips past 60 degrees. For the very first try, hold the robot just above the table and check the
beams turn steadily rather than jerking back and forth.

## When it does not drive well

Look at `run-<time>.csv` and compare with what the simulation does (`--sim` writes the same file):
- **Motors stuck at +/-100 %:** the policy wants more speed than the motors give. Lower `MAX_BEAM_SPEED`, retrain.
- **Jerky, back and forth:** Bluetooth is slower than training assumed. Raise `CMD_DELAY_S` in `real_task.py` (for
  example to `(0.05, 0.15)`), retrain.
- **Falls over where the simulation does not:** check the mass and the beam spacing, or widen `MASS_SCALE_RANGE` and
  `MOTOR_SCALE_RANGE`, retrain.
- **Drives but curves away:** the heading comes from the gyroscope and drifts a few degrees over 10 s. Recalibrate
  (step 3 sets the gyroscope's scale), and keep runs short.

What it cannot do: the robot has no way to measure how far it has drifted sideways, so it holds its **heading** but not
the line. For that, add an outside sensor (a camera over the table, or a colour sensor following a taped line).
