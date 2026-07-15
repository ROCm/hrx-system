#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Regenerates the checked-in ELF compatibility fixtures from the adjacent C
# source. Normal builds only embed these files and do not require Clang, LLD, or
# llvm-strip. The tools are maintenance dependencies used when the executable
# library ABI or fixture source changes.

set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
TESTDATA_DIR="${ROOT_DIR}/runtime/src/iree/hal/local/elf/testdata"
SOURCE_FILE="${TESTDATA_DIR}/elementwise_mul_library.c"
CLANG="${CLANG:-clang}"
LLVM_STRIP="${LLVM_STRIP:-llvm-strip}"
RESOURCE_DIR="$("${CLANG}" -print-resource-dir)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

# executable_library.h includes <assert.h> for the C static_assert spelling.
# Cross-compiling a freestanding payload intentionally has no target libc
# sysroot, so provide only that language macro and use Clang's builtin headers
# for stddef.h and stdint.h.
cat >"${TEMP_DIR}/assert.h" <<'EOF'
#ifndef IREE_ELF_TEST_ASSERT_H_
#define IREE_ELF_TEST_ASSERT_H_
#define static_assert _Static_assert
#endif  // IREE_ELF_TEST_ASSERT_H_
EOF

COMMON_FLAGS=(
  -std=c11
  -O2
  -ffreestanding
  -nostdinc
  -isystem "${TEMP_DIR}"
  -isystem "${RESOURCE_DIR}/include"
  -I"${ROOT_DIR}/runtime/src"
  -fPIC
  -fvisibility=hidden
  -fno-ident
  -fno-stack-protector
  -fno-unwind-tables
  -fno-asynchronous-unwind-tables
  -fno-sanitize=all
  -ffunction-sections
  -fdata-sections
  -shared
  -nostdlib
  -fuse-ld=lld
  "-Wl,--build-id=none"
  "-Wl,--gc-sections"
  "-Wl,--hash-style=sysv"
  "-Wl,-z,defs"
)

compile_fixture() {
  local output_name="$1"
  local target_triple="$2"
  shift 2

  local temporary_output="${TEMP_DIR}/${output_name}"
  echo "Generating ${output_name}"
  "${CLANG}" \
    --target="${target_triple}" \
    "${COMMON_FLAGS[@]}" \
    "$@" \
    "${SOURCE_FILE}" \
    -o "${temporary_output}"
  "${LLVM_STRIP}" --strip-all "${temporary_output}"
  chmod 0644 "${temporary_output}"
  mv "${temporary_output}" "${TESTDATA_DIR}/${output_name}"
}

compile_fixture elementwise_mul_arm_32.so armv7a-unknown-linux-gnueabihf \
  -march=armv7-a -mfloat-abi=hard -mfpu=vfpv3-d16
compile_fixture elementwise_mul_arm_64.so aarch64-unknown-linux-gnu
compile_fixture elementwise_mul_riscv_32.so riscv32-unknown-linux-gnu \
  -march=rv32imaf -mabi=ilp32f
compile_fixture elementwise_mul_riscv_64.so riscv64-unknown-linux-gnu \
  -march=rv64imafdc -mabi=lp64d
compile_fixture elementwise_mul_x86_32.so i686-unknown-linux-gnu
compile_fixture elementwise_mul_x86_64.so x86_64-unknown-linux-gnu
