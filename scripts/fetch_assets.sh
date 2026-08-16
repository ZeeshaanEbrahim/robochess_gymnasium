#!/usr/bin/env bash
#
# Populate the vendor mesh directories under assets/ from a local
# mujoco_menagerie checkout.
#
# The meshes are ~34 MB and are gitignored; the model XML files that sit
# alongside them are tracked, because they carry local modifications (the 2F-85
# tool mount on the UR5e wrist, matched <option> settings, and so on). Never
# copy the vendor XML over those.
#
# Safe to re-run: copies are unconditional overwrites of the mesh files only, so
# a second run just refreshes them in place. Nothing is appended or duplicated.
#
# Usage:
#   scripts/fetch_assets.sh [MENAGERIE_DIR]
#
# MENAGERIE_DIR defaults to $MUJOCO_MENAGERIE_DIR, then to
# ~/reference_assets/mujoco_menagerie.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MENAGERIE="${1:-${MUJOCO_MENAGERIE_DIR:-$HOME/reference_assets/mujoco_menagerie}}"

if [[ ! -d "$MENAGERIE" ]]; then
  echo "error: mujoco_menagerie not found at: $MENAGERIE" >&2
  echo "Clone it, or pass the path: scripts/fetch_assets.sh /path/to/mujoco_menagerie" >&2
  exit 1
fi

# src_subdir : dest_subdir : file glob
MODELS=(
  "universal_robots_ur5e:ur5e:*.obj"
  "robotiq_2f85:robotiq_2f85:*.stl"
)

status=0
for entry in "${MODELS[@]}"; do
  IFS=: read -r src_name dest_name glob <<<"$entry"
  src="$MENAGERIE/$src_name/assets"
  dest="$REPO_ROOT/assets/$dest_name/assets"

  if [[ ! -d "$src" ]]; then
    echo "error: missing source directory: $src" >&2
    status=1
    continue
  fi

  mkdir -p "$dest"

  # shellcheck disable=SC2086
  if ! compgen -G "$src/$glob" >/dev/null; then
    echo "error: no $glob files in $src" >&2
    status=1
    continue
  fi

  cp -f "$src"/$glob "$dest"/
  echo "$dest_name: $(find "$dest" -maxdepth 1 -name "$glob" | wc -l) files -> assets/$dest_name/assets/"
done

if [[ $status -ne 0 ]]; then
  echo "fetch_assets: completed with errors" >&2
  exit $status
fi

echo "fetch_assets: done. Verify with:"
echo "  python -c \"import mujoco; mujoco.MjModel.from_xml_path('assets/scene.xml'); print('ok')\""
