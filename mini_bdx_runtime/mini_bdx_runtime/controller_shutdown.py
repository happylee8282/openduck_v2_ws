"""Controller shutdown timing and ordered cleanup; no hardware is imported here."""
import math
import subprocess


class HoldToShutdown:
    def __init__(self, hold_seconds=7.0, max_sample_age=0.5):
        if not math.isfinite(hold_seconds) or hold_seconds <= 0:
            raise ValueError("Shutdown hold duration must be positive and finite")
        self.hold_seconds = hold_seconds
        self.max_sample_age = max_sample_age
        self.held_since = None
        self.last_sample = None
        self.armed = False
        self.fired = False

    def update(self, pressed, sampled_at, now):
        if self.fired:
            return False
        age = now - sampled_at
        gap = None if self.last_sample is None else sampled_at - self.last_sample
        if (sampled_at <= 0 or not math.isfinite(age)
                or age < 0 or age > self.max_sample_age
                or (gap is not None and (gap < 0 or gap > self.max_sample_age))):
            self.held_since = None
            self.last_sample = None
            self.armed = False
            return False
        self.last_sample = sampled_at
        if not pressed:
            self.armed = True
            self.held_since = None
            return False
        # Require a released sample after startup, disconnect, or stale input.
        if not self.armed:
            return False
        if self.held_since is None:
            self.held_since = sampled_at
        if sampled_at - self.held_since >= self.hold_seconds:
            self.fired = True
            return True
        return False


def stop_and_poweroff(hwi, antennas=None, controller=None, runner=None):
    """Attempt cleanup before requesting shutdown; caller must stay stopped."""
    actions = [("motor torque", hwi.turn_off)]
    if antennas is not None:
        actions.append(("antenna PWM", antennas.stop))
    if controller is not None and hasattr(controller, "close"):
        actions.append(("controller", controller.close))
    for name, action in actions:
        try:
            action()
        except Exception as exc:
            print(f"[SHUTDOWN] Could not stop {name}: {exc}", flush=True)
    if runner is None:
        runner = subprocess.run
    try:
        result = runner(
            ["sudo", "-n", "/usr/bin/systemctl", "poweroff"],
            check=False, timeout=10,
        )
        if result.returncode != 0:
            print(f"[SHUTDOWN] Poweroff failed ({result.returncode}); staying stopped.", flush=True)
            return False
        print("[SHUTDOWN] Poweroff requested; staying stopped.", flush=True)
        return True
    except Exception as exc:
        print(f"[SHUTDOWN] Poweroff failed: {exc}; staying stopped.", flush=True)
        return False