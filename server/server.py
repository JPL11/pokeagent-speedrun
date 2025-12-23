#!/usr/bin/env python3
"""
Unified launcher for:
  - server.app (game server)
  - server.frame_server (frame streaming)
  - train_rl_vlm_screen_aware.py (training)

Notes:
  - The emulator is created INSIDE server.app via EmeraldEmulator, so we do NOT
    start mGBA separately here.
  - This script lives in server/server.py and uses the repo's .venv Python
    directly, so you don't need to manually activate the virtualenv.
"""

import subprocess
import time
from pathlib import Path


REPO = Path("/home/jpli/pokeagent-speedrun").resolve()
VENV_PYTHON = REPO / ".venv" / "bin" / "python"

SERVER_LOG = REPO / "server.log"
FRAME_LOG = REPO / "frame_server.log"
TRAINING_LOG = REPO / "training.log"


def start_process(cmd, log_path: Path):
    """
    Start a subprocess with stdout/stderr redirected to the given log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w")
    # Do NOT use shell=True so pkill -f still works on the command strings
    return subprocess.Popen(cmd, stdout=f, stderr=f, cwd=str(REPO))


def kill_existing():
    """
    Best‑effort kill of any previous runs so we don't double‑start things.
    """
    patterns = [
        "train_rl_vlm_screen_aware.py",
        "python -m server.app",
        "python -m server.frame_server",
    ]
    for pat in patterns:
        try:
            subprocess.run(["pkill", "-f", pat], check=False)
        except Exception:
            pass
    time.sleep(2)


def main():
    # Ensure venv Python exists; we will use it for all Python processes.
    if not VENV_PYTHON.exists():
        raise SystemExit(f"Virtualenv Python not found at {VENV_PYTHON}")

    print("Killing any existing server / frame_server / training processes...")
    kill_existing()

    # 1) Start main game server (server.app on port 8000)
    server_cmd = [str(VENV_PYTHON), "-m", "server.app", "--port", "8000"]
    server_proc = start_process(server_cmd, SERVER_LOG)
    print(f"server.app started (pid {server_proc.pid}) on http://localhost:8000")

    # 2) Start frame server (for streaming frames on port 8001)
    frame_cmd = [str(VENV_PYTHON), "-m", "server.frame_server", "--port", "8001"]
    frame_proc = start_process(frame_cmd, FRAME_LOG)
    print(f"frame_server started (pid {frame_proc.pid}) on http://localhost:8001/stream")

    # Small delay so the servers are fully up before training connects
    time.sleep(3)

    # 3) Start training (2M steps, VLM every 50 steps)
    training_cmd = [
        str(VENV_PYTHON),
        "train_rl_vlm_screen_aware.py",
        "--timesteps",
        "2000000",
        "--vlm-frequency",
        "50",
    ]
    training_proc = start_process(training_cmd, TRAINING_LOG)
    print(f"Training started (pid {training_proc.pid})")

    print(
        "\nTail these logs in separate terminals if something looks off:\n"
        "  server.log          # game server status\n"
        "  frame_server.log    # frame streaming status\n"
        "  training.log        # RL + VLM training loop\n"
    )


if __name__ == "__main__":
    main()


