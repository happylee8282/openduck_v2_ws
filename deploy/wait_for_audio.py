"""Wait for the configured speaker before the gait process can initialize motors."""
import argparse
import json
import os
import subprocess
import sys
import time

import pygame


def wait_for_audio(timeout):
    if os.environ.get("SDL_AUDIODRIVER") != "pulseaudio":
        raise RuntimeError("SDL_AUDIODRIVER must be pulseaudio")
    sink = os.environ.get("PULSE_SINK")
    if not sink:
        raise RuntimeError("PULSE_SINK must name the speaker output")

    deadline = time.monotonic() + timeout
    last_error = "audio output is not ready"
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["/usr/bin/pw-dump"],
                capture_output=True, text=True, check=True, timeout=3,
            )
            nodes = json.loads(result.stdout)
            ready = any(
                node.get("info", {}).get("props", {}).get("node.name") == sink
                and node.get("info", {}).get("props", {}).get("media.class") == "Audio/Sink"
                for node in nodes
            )
            if not ready:
                raise RuntimeError("Speaker output not found: " + sink)
            pygame.mixer.init()
            print("Audio ready: " + sink, flush=True)
            return
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, pygame.error) as exc:
            last_error = str(exc)
        finally:
            pygame.mixer.quit()
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise RuntimeError("Audio readiness timed out: " + last_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    try:
        wait_for_audio(args.timeout)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(1)