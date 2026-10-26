#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Prepare the target SDK/runtime inputs for the normal Linux clang-cl toolchain.
# The workflow caches the extracted inputs, never the download/unpack scratch.
set -euo pipefail

: "${WINDOWS_CROSS_ROOT:?}"
mkdir -p "$WINDOWS_CROSS_ROOT"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

fetch() {
  local url="$1" digest="$2" destination="$3"
  curl --fail --location --silent --show-error "$url" --output "$destination"
  printf '%s  %s\n' "$digest" "$destination" | sha256sum --check
}

fetch \
  https://github.com/Jake-Shadle/xwin/releases/download/0.10.0/xwin-0.10.0-x86_64-unknown-linux-musl.tar.gz \
  d870eb4b2f390878af6da1ccd3cf321d22fcb72720984853b4be732ae597fc88 \
  "$scratch/xwin.tar.gz"
tar -xf "$scratch/xwin.tar.gz" -C "$scratch"

# xwin verifies SDK/CRT packages against this manifest but does not verify the
# manifest itself. Seed its cache with the exact independently pinned bytes.
readarray -t manifest < <(python3 - <<'PY'
import json
from pathlib import Path
document = json.loads(Path('.github/scripts/windows-sdk-channel.json').read_text())
payload, = document['channelItems'][0]['payloads']
print(payload['url'])
print(payload['sha256'])
PY
)
mkdir -p "$scratch/cache/dl"
catalog="$scratch/cache/dl/pkg_manifest_${manifest[1]}.vsman"
fetch "${manifest[0]}" "${manifest[1]}" "$catalog"
"$scratch/xwin-0.10.0-x86_64-unknown-linux-musl/xwin" \
  --accept-license --manifest .github/scripts/windows-sdk-channel.json \
  --cache-dir "$scratch/cache" --sdk-version 10.0.26100 \
  --crt-version 14.44.17.14 --arch x86_64 --variant desktop splat \
  --use-winsysroot-style --preserve-ms-arch-notation --include-debug-libs \
  --output "$WINDOWS_CROSS_ROOT/sdk"

# The unpacked SDK is the retained input. Release the acquisition cache before
# fetching compiler-rt so peak storage does not include two full download sets.
mv "$catalog" "$scratch/catalog.json"
catalog="$scratch/catalog.json"
rm -rf "$scratch/cache" "$scratch/xwin.tar.gz" "$scratch/xwin-0.10.0-x86_64-unknown-linux-musl"

# LLVM's 22.1.8 Ubuntu package supplies the Linux compiler. The matching Windows
# installer supplies only the target compiler-rt libraries and ASAN DLL here.
fetch \
  https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.8/LLVM-22.1.8-win64.exe \
  16e5709785fef73c854646241c4a92c5cd574318d1b33c63330dd7721903e55c \
  "$scratch/llvm.exe"
runtime_names=(
  clang_rt.builtins-x86_64.lib
  clang_rt.asan_dynamic-x86_64.dll
  clang_rt.asan_dynamic-x86_64.lib
  clang_rt.asan_dynamic_runtime_thunk-x86_64.lib
  clang_rt.asan_static_runtime_thunk-x86_64.lib
)
7z e -y "-o$WINDOWS_CROSS_ROOT/compiler-rt" "$scratch/llvm.exe" \
  "${runtime_names[@]/#/lib/clang/22/lib/windows/}"
for name in "${runtime_names[@]}"; do
  test -s "$WINDOWS_CROSS_ROOT/compiler-rt/$name"
done
rm "$scratch/llvm.exe"

# Microsoft's STL ASAN library is an optional package in the same catalog.
# xwin's versioned directory uses the package-family spelling, while the VSIX
# members use the actual VC tools version.
companion="$(python3 build_tools/ci/windows_stl_asan.py \
  --tools-dir "$scratch/14.44.35207" --catalog "$catalog" \
  --output-dir "$scratch/stl-asan")"
cp "$companion/stl_asan.lib" \
  "$WINDOWS_CROSS_ROOT/sdk/VC/Tools/MSVC/14.44.17.14/lib/x64/stl_asan.lib"
