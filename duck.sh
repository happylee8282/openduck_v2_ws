#!/usr/bin/env bash
# Isaac Lab entry point. The existing Pi runtime uses run_duck.sh.
set -euo pipefail
DUCK_WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$DUCK_WORKSPACE_ROOT/open-duck-isaac-lab/run.sh" "$@"
