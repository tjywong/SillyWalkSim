"""Train a task. Same as mjlab's own `train`, but it registers our tasks first.

    python train.py Drive-Straight --env.scene.num-envs 4096        # two-wheel robot
    python train.py Beam-Drive-Straight --env.scene.num-envs 4096   # two-beam robot
"""
import importlib
import sys

# Register every task file sent along (runIt.py copies only the ones you name with --with):
#   drive_task.py -> "Drive-Straight" (two wheels),  beam_task.py -> "Beam-Drive-Straight" (two beams)
for _module in ("drive_task", "beam_task"):
  try:
    importlib.import_module(_module)
  except ModuleNotFoundError as e:
    if e.name != _module:
      raise

from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
