#!/usr/bin/env bash
# Step 3 (runs ON the cluster login node): survey what is available.
# Read-only; installs nothing.

# limited SECS CMD...: run CMD but give up after SECS seconds instead of hanging. The checks that read the shared
# disk (module lists, quota) can get stuck for many minutes when the cluster storage is slow, and none of them is needed.
# Output goes to a temp file, so a stuck command left behind cannot hold this script (or the SSH session) open.
limited() {
  local secs=$1; shift
  local out; out=$(mktemp)
  ( "$@" >"$out" 2>&1 ) </dev/null &
  local pid=$! i=0
  disown "$pid" 2>/dev/null   # no "Terminated" message if we stop it
  while kill -0 "$pid" 2>/dev/null && [ "$i" -lt "$secs" ]; do sleep 1; i=$((i+1)); done
  if kill -0 "$pid" 2>/dev/null; then
    pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; rm -f "$out"; return 124
  fi
  cat "$out"; rm -f "$out"
}
SLOW="(skipped: took over 20 s, the cluster's shared disk is slow right now; not needed for setup)"

echo "== Who/where =="
echo "user: $(whoami)   host: $(hostname)   date: $(date)"

echo; echo "== Slurm (the job scheduler) =="
if command -v sbatch >/dev/null; then sbatch --version; else echo "MISSING: sbatch not found"; fi

echo; echo "== Partitions (queues). Look for 'gpu' if you want GPUs =="
sinfo -s 2>/dev/null || echo "sinfo failed"

echo; echo "== Your account / allowed partitions =="
sacctmgr -nP show assoc user="$(whoami)" format=account,partition,qos 2>/dev/null | head || true

echo; echo "== Python on the login node =="
for p in python3 python; do command -v $p >/dev/null && { echo "$p -> $(command -v $p): $($p --version 2>&1)"; }; done

echo; echo "== Modules (python / anaconda / cuda / mesa) =="
if command -v module >/dev/null 2>&1 || type module >/dev/null 2>&1; then
  if mods=$(limited 20 module avail); then
    echo "$mods" | grep -iE 'python|anaconda|miniconda|cuda|mesa|osmesa|ffmpeg' | head -30
  else echo "$SLOW"; fi
else
  echo "no 'module' command (try: source /etc/profile.d/modules.sh)"
fi

echo; echo "== Is MuJoCo (or related) already installed on the cluster? =="
if ! mods=$(limited 20 module -t avail); then echo "$SLOW"
elif found=$(echo "$mods" | grep -iE 'mujoco|mjx|gymnasium|brax|isaac|robosuite' | head -10) && [ -n "$found" ]; then
  echo "preinstalled modules:"; echo "$found"
else echo "no MuJoCo module on this cluster (expected). 03_setup_env.sh installs it in your own venv."; fi

echo; echo "== Disk =="
if d=$(limited 20 df -h "$HOME"); then echo "$d" | tail -1; else echo "df $SLOW"; fi
if q=$(limited 20 quota -s); then echo "$q" | head -5; else echo "quota $SLOW"; fi

echo; echo "== Internet from login node (needed for pip install) =="
if curl -sI --max-time 8 https://pypi.org >/dev/null; then echo "ok: pypi reachable"; else echo "FAIL: cannot reach pypi"; fi

echo; echo "Next, on the cluster:  bash 03_setup_env.sh"
