#!/usr/bin/env python3
import subprocess
import time
import os
from pathlib import Path


REPO = Path("/home/jpli/pokeagent-speedrun").resolve()
MGBA = REPO / "mGBA-0.10.5-ubuntu64-focal" / "mGBA"
# You mentioned the ROM is in the Emerald-GBAdvance directory as rom.gba
ROM = REPO / "Emerald-GBAdvance" / "rom.gba"
LUA = REPO / "pokemon_env" / "mgba" / "PokemonInterface.lua"

SERVER_LOG = REPO / "server.log"
FRAME_LOG = REPO / "frame_server.log"
TRAINING_LOG = REPO / "training.log"
MGBA_LOG = REPO / "mgba.log"


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
        str(MGBA),
    ]
    for pat in patterns:
        try:
            subprocess.run(["pkill", "-f", pat], check=False)
        except Exception:
            pass
    time.sleep(2)


def main():
    # Require the venv so we don't accidentally use system Python
    if os.environ.get("VIRTUAL_ENV") is None:
        raise SystemExit("Activate .venv first:  source .venv/bin/activate")

    if not MGBA.exists():
        raise SystemExit(f"mGBA executable not found at {MGBA}")
    if not ROM.exists():
        raise SystemExit(f"ROM not found at {ROM}")
    if not LUA.exists():
        raise SystemExit(f"Lua bridge not found at {LUA}")

    print("Killing any existing server / frame_server / training / mGBA processes...")
    kill_existing()

    # 1) Start mGBA with Lua bridge (non‑headless first, as you requested)
    mgba_cmd = [str(MGBA), str(ROM), "--lua", str(LUA)]
    mgba_proc = start_process(mgba_cmd, MGBA_LOG)
    print(f"mGBA started (pid {mgba_proc.pid}). You should see the emulator window.")

    # Give Lua bridge a moment to attach and start exposing RAM
    time.sleep(5)

    # 2) Start main game server (server.app on port 8000)
    server_cmd = ["python", "-m", "server.app", "--port", "8000"]
    server_proc = start_process(server_cmd, SERVER_LOG)
    print(f"server.app started (pid {server_proc.pid}) on http://localhost:8000")

    # 3) Start frame server (for streaming frames on port 8001)
    frame_cmd = ["python", "-m", "server.frame_server", "--port", "8001"]
    frame_proc = start_process(frame_cmd, FRAME_LOG)
    print(f"frame_server started (pid {frame_proc.pid}) on http://localhost:8001/stream")

    # Small delay so the servers are fully up before training connects
    time.sleep(3)

    # 4) Start training (2M steps, VLM every 50 steps)
    training_cmd = [
        "python",
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
        "  mgba.log            # emulator + Lua bridge output\n"
    )


if __name__ == "__main__":
    main()


