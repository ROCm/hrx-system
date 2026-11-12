#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Prepare genuine x64 MSVC compiler tools for Linux-hosted Wine execution.
# The target headers and libraries remain owned by setup_windows_cross.sh; this
# downloads only the compiler package and the pinned wrappers that launch it.
set -euo pipefail

: "${MSVC_WINE_ROOT:?}"
: "${WINEPREFIX:?}"
if [[ "${WINEARCH:-}" != win64 ]]; then
  echo "Linux-hosted x64 MSVC requires WINEARCH=win64." >&2
  exit 1
fi

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
    "$bin_dir/pgodb140.dll" \
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

fetch() {
  local url="$1" digest="$2" destination="$3"
  curl --fail --location --retry 3 --retry-all-errors \
    --silent --show-error "$url" --output "$destination"
  printf '%s  %s\n' "$digest" "$destination" | sha256sum --check
}

install_compiler() {
  mkdir -p "$(dirname "$MSVC_WINE_ROOT")"
  scratch="$(mktemp -d "$(dirname "$MSVC_WINE_ROOT")/.setup-msvc.XXXXXXXX")"
  trap 'rm -rf "$scratch"' EXIT

  local msvc_wine_commit=5dd115047b7ddcbb53cbc8b5b14163a20f4334fe
  fetch \
    "https://github.com/mstorsjo/msvc-wine/archive/${msvc_wine_commit}.tar.gz" \
    9620a6ab2b4a267114fe99fd7c953264f945c768ee3f0e6a0d5318cb106115c5 \
    "$scratch/msvc-wine.tar.gz"
  tar -xf "$scratch/msvc-wine.tar.gz" -C "$scratch"
  local msvc_wine_source="$scratch/msvc-wine-${msvc_wine_commit}"

  local -a manifest
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

  # link.exe imports pgodb140.dll even for ordinary non-PGO links. Microsoft
  # distributes that dependency separately from the host compiler tools.
  python3 "$msvc_wine_source/vsdownload.py" \
    --accept-license \
    --manifest "$scratch/VisualStudio.vsman" \
    --with-default no \
    --architecture x64 \
    --dest "$scratch/install" \
    Microsoft.VC.14.44.17.14.Tools.HostX64.TargetX64 \
    Microsoft.VC.14.44.17.14.Premium.Tools.HostX64.TargetX64

  mkdir -p "$scratch/install/wrappers"
  local wrapper
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
}

initialize_wine_prefix() {
  local wine_loader wine_server
  wine_loader="$(command -v wine64 || command -v wine || true)"
  wine_server="$(command -v wineserver || true)"
  if [[ -z "$wine_loader" && -x /usr/lib/wine/wine64 ]]; then
    wine_loader=/usr/lib/wine/wine64
  fi
  if [[ -z "$wine_loader" || -z "$wine_server" ]]; then
    echo "Linux-hosted MSVC requires a 64-bit Wine loader and wineserver." >&2
    return 1
  fi
  mkdir -p "$(dirname "$WINEPREFIX")"
  local prefix_log
  prefix_log="$(mktemp "$(dirname "$WINEPREFIX")/.setup-wine.XXXXXXXX.log")"
  local server_status=0
  env WINEPREFIX="$WINEPREFIX" "$wine_server" -k >>"$prefix_log" 2>&1 || server_status=$?
  if [[ "$server_status" -ne 0 && "$server_status" -ne 1 ]]; then
    echo "Could not stop the prior MSVC Wine server (status $server_status):" >&2
    cat "$prefix_log" >&2
    rm -f "$prefix_log"
    return 1
  fi
  if ! env WINEPREFIX="$WINEPREFIX" "$wine_server" -w >>"$prefix_log" 2>&1; then
    echo "The prior MSVC Wine server did not finish shutting down:" >&2
    cat "$prefix_log" >&2
    rm -f "$prefix_log"
    return 1
  fi
  if [[ ! -s "$WINEPREFIX/.update-timestamp" || \
        ! -s "$WINEPREFIX/system.reg" || ! -s "$WINEPREFIX/user.reg" ]]; then
    if ! env DISPLAY= WINEARCH=win64 WINEDEBUG=-all \
      WINEDLLOVERRIDES='mscoree=d;mshtml=d;winemenubuilder.exe=d' \
      WINEPREFIX="$WINEPREFIX" \
      "$wine_loader" wineboot --init >"$prefix_log" 2>&1; then
      echo "Could not initialize the MSVC Wine prefix at $WINEPREFIX:" >&2
      cat "$prefix_log" >&2
      rm -f "$prefix_log"
      return 1
    fi
    if ! env DISPLAY= WINEDEBUG=-all WINEPREFIX="$WINEPREFIX" \
      "$wine_loader" cmd.exe /c exit >>"$prefix_log" 2>&1; then
      echo "The initialized MSVC Wine prefix could not run a client:" >&2
      cat "$prefix_log" >&2
      rm -f "$prefix_log"
      return 1
    fi
    if ! env WINEPREFIX="$WINEPREFIX" "$wine_server" -w >>"$prefix_log" 2>&1; then
      echo "The initialized MSVC Wine prefix did not finish shutting down:" >&2
      cat "$prefix_log" >&2
      rm -f "$prefix_log"
      return 1
    fi
  fi
  if ! env WINEPREFIX="$WINEPREFIX" "$wine_server" -p >>"$prefix_log" 2>&1; then
    echo "Could not retain the MSVC Wine server:" >&2
    cat "$prefix_log" >&2
    rm -f "$prefix_log"
    return 1
  fi
  if ! env DISPLAY= WINEDEBUG=-all WINEPREFIX="$WINEPREFIX" \
    "$wine_loader" cmd.exe /c exit >>"$prefix_log" 2>&1; then
    echo "The retained MSVC Wine server could not run a client:" >&2
    cat "$prefix_log" >&2
    rm -f "$prefix_log"
    return 1
  fi
  rm -f "$prefix_log"
}

if [[ -e "$MSVC_WINE_ROOT" ]]; then
  verify_installation "$MSVC_WINE_ROOT"
else
  install_compiler
fi
initialize_wine_prefix
