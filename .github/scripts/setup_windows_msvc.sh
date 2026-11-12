#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Prepare genuine x64 MSVC compiler tools for Linux-hosted Wine execution.
# The target headers and libraries remain owned by setup_windows_cross.sh; this
# downloads only the compiler package and the pinned wrappers that launch it.
set -euo pipefail

: "${MSVC_WINE_ROOT:?}"

verify_installation() {
  local root="$1"
  local -a versions
  mapfile -t versions < <(find "$root/VC/Tools/MSVC" -mindepth 1 -maxdepth 1 -type d -print)
  if [[ ${#versions[@]} -ne 1 ]]; then
    echo "Expected exactly one MSVC toolset under $root; found ${#versions[@]}." >&2
    return 1
  fi
  local bin_dir="${versions[0]}/bin/Hostx64/x64"
  local path
  for path in \
    "$bin_dir/cl.exe" \
    "$bin_dir/dumpbin.exe" \
    "$bin_dir/lib.exe" \
    "$bin_dir/link.exe" \
    "$bin_dir/ml64.exe" \
    "$bin_dir/1033/clui.dll" \
    "$bin_dir/1033/linkui.dll" \
    "$root/wrappers/cl" \
    "$root/wrappers/lib" \
    "$root/wrappers/link" \
    "$root/wrappers/ml64" \
    "$root/wrappers/wine-msvc.sh"; do
    test -s "$path"
  done
}

if [[ -e "$MSVC_WINE_ROOT" ]]; then
  verify_installation "$MSVC_WINE_ROOT"
  exit 0
fi

mkdir -p "$(dirname "$MSVC_WINE_ROOT")"
scratch="$(mktemp -d "$(dirname "$MSVC_WINE_ROOT")/.setup-msvc.XXXXXXXX")"
trap 'rm -rf "$scratch"' EXIT

fetch() {
  local url="$1" digest="$2" destination="$3"
  curl --fail --location --retry 3 --retry-all-errors \
    --silent --show-error "$url" --output "$destination"
  printf '%s  %s\n' "$digest" "$destination" | sha256sum --check
}

msvc_wine_commit=5dd115047b7ddcbb53cbc8b5b14163a20f4334fe
fetch \
  "https://github.com/mstorsjo/msvc-wine/archive/${msvc_wine_commit}.tar.gz" \
  9620a6ab2b4a267114fe99fd7c953264f945c768ee3f0e6a0d5318cb106115c5 \
  "$scratch/msvc-wine.tar.gz"
tar -xf "$scratch/msvc-wine.tar.gz" -C "$scratch"
msvc_wine_source="$scratch/msvc-wine-${msvc_wine_commit}"

readarray -t manifest < <(python3 - <<'PY'
import json
from pathlib import Path

document = json.loads(Path('.github/scripts/windows-sdk-channel.json').read_text())
payload, = document['channelItems'][0]['payloads']
print(payload['url'])
print(payload['sha256'])
PY
)
fetch "${manifest[0]}" "${manifest[1]}" "$scratch/VisualStudio.vsman"

python3 "$msvc_wine_source/vsdownload.py" \
  --accept-license \
  --manifest "$scratch/VisualStudio.vsman" \
  --with-default no \
  --architecture x64 \
  --dest "$scratch/install" \
  Microsoft.VC.14.44.17.14.Tools.HostX64.TargetX64

mkdir -p "$scratch/install/wrappers"
for wrapper in cl lib link ml64 wine-msvc.sh; do
  install -m 0755 "$msvc_wine_source/wrappers/$wrapper" \
    "$scratch/install/wrappers/$wrapper"
done
install -m 0644 "$msvc_wine_source/LICENSE.txt" \
  "$scratch/install/wrappers/LICENSE.txt"

# The telemetry helper is not part of compilation and has caused Wine failures
# in upstream users. It must not be discoverable beside cl.exe.
find "$scratch/install/VC/Tools/MSVC" -type f -iname vctip.exe -delete

verify_installation "$scratch/install"
mv "$scratch/install" "$MSVC_WINE_ROOT"
