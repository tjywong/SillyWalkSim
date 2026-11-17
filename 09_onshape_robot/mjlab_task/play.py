"""Watch a trained policy (or a dummy one). Same as mjlab's `play`, with our tasks registered.
Works the same for Beam-Drive-Straight (logs/rsl_rl/beam_drive_straight/...).

    python play.py Drive-Straight --agent zero          # wheels do nothing: checks the robot loads
    python play.py Drive-Straight --agent random        # random wheel speeds
    python play.py Drive-Straight --checkpoint-file logs/rsl_rl/drive_straight/<run>/model_499.pt
"""
import importlib
import sys

# Register every task file sent along (runIt.py copies only the ones you name with --with):
#   drive_task.py -> "Drive-Straight" (two wheels),  beam_task.py -> "Beam-Drive-Straight", "Onshape-Beam-Drive-Straight", "-Backward" (two beams)
#   real_task.py (needs beam_task.py too) -> "Onshape-Beam-Real", "-Backward", "Onshape-Beam-Real-V2", "-V2-Backward" (for the real robot)
for _module in ("drive_task", "beam_task", "real_task"):
  try:
    importlib.import_module(_module)
  except ModuleNotFoundError as e:
    if e.name != _module:
      raise

from mjlab.scripts.play import main

if __name__ == "__main__":
  sys.exit(main())
