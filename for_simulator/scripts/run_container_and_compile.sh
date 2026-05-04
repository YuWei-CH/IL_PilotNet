#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash for_simulator/scripts/run_container_and_compile.sh [workspace_path]

Default workspace_path:
  ~/gem_simulation_ws

What it does:
  1. cd ~/gem_simulation_ws
  2. cd src/POLARIS_GEM_Simulator
  3. bash run_docker_container.sh
EOF
  return 0 2>/dev/null || exit 0
fi

WORKSPACE_PATH="${1:-$HOME/gem_simulation_ws}"
REPO_PATH="$WORKSPACE_PATH/src/POLARIS_GEM_Simulator"

if [[ ! -d "$WORKSPACE_PATH" ]]; then
  echo "Workspace path not found: $WORKSPACE_PATH" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "$REPO_PATH" ]]; then
  echo "Repository path not found: $REPO_PATH" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "$REPO_PATH/run_docker_container.sh" ]]; then
  echo "Missing script: $REPO_PATH/run_docker_container.sh" >&2
  return 1 2>/dev/null || exit 1
fi

cd "$WORKSPACE_PATH"
cd src/POLARIS_GEM_Simulator
bash run_docker_container.sh
