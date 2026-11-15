"""Shared v2 loader. Legacy checkpoints require retraining."""

from open_duck_v2.runtime import export_policy
from open_duck_v2.runtime import load_rsl_rl_actor as load_rsl_rl_actor


def export_torchscript(
    checkpoint_path, output_path, device="cpu", obs_dim=101, action_dim=14
):
    if obs_dim != 101 or action_dim != 14:
        raise ValueError("Expected the v2 101/14 contract")
    return export_policy(checkpoint_path, output_path, device)
