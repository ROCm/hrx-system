// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check binary with all build-enabled product providers linked in.
// Test builds may additionally enable the synthetic test provider.

#include "loom/target/arch/cmd/check/provider.h"
#include "loom/target/configured/compiler_provider_set.h"
#include "loom/tooling/export/cxx/loom_check.h"
#include "loom/tooling/input/configured.h"
#include "loom/tooling/input/loomc_configured.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/provider.h"

#ifndef LOOM_CHECK_HAVE_TEST_PROVIDER
#define LOOM_CHECK_HAVE_TEST_PROVIDER 0
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#if LOOM_CHECK_HAVE_TEST_PROVIDER
#include "loom/tools/loom-check/test_provider.h"
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#ifndef LOOM_CHECK_HAVE_EMIT_AMDGPU
#define LOOM_CHECK_HAVE_EMIT_AMDGPU 0
#endif  // LOOM_CHECK_HAVE_EMIT_AMDGPU
#ifndef LOOM_CHECK_HAVE_TARGET_SPIRV
#define LOOM_CHECK_HAVE_TARGET_SPIRV 0
#endif  // LOOM_CHECK_HAVE_TARGET_SPIRV
#ifndef LOOM_CHECK_HAVE_EMIT_SPIRV
#define LOOM_CHECK_HAVE_EMIT_SPIRV 0
#endif  // LOOM_CHECK_HAVE_EMIT_SPIRV
#ifndef LOOM_CHECK_HAVE_TARGET_WASM
#define LOOM_CHECK_HAVE_TARGET_WASM 0
#endif  // LOOM_CHECK_HAVE_TARGET_WASM
#ifndef LOOM_CHECK_HAVE_TARGET_VM
#define LOOM_CHECK_HAVE_TARGET_VM 0
#endif  // LOOM_CHECK_HAVE_TARGET_VM
#ifndef LOOM_CHECK_HAVE_EMIT_WASM
#define LOOM_CHECK_HAVE_EMIT_WASM 0
#endif  // LOOM_CHECK_HAVE_EMIT_WASM
#ifndef LOOM_CHECK_HAVE_TARGET_X86
#define LOOM_CHECK_HAVE_TARGET_X86 0
#endif  // LOOM_CHECK_HAVE_TARGET_X86
#ifndef LOOM_CHECK_HAVE_TARGET_XDNA
#define LOOM_CHECK_HAVE_TARGET_XDNA 0
#endif  // LOOM_CHECK_HAVE_TARGET_XDNA

#if LOOM_CHECK_HAVE_EMIT_AMDGPU
#include "loom/target/arch/amdgpu/check/provider.h"
#endif  // LOOM_CHECK_HAVE_EMIT_AMDGPU
#if LOOM_CHECK_HAVE_TARGET_SPIRV
#include "loom/target/arch/spirv/check/provider.h"
#endif  // LOOM_CHECK_HAVE_TARGET_SPIRV
#if LOOM_CHECK_HAVE_EMIT_SPIRV
#include "loom/tooling/target/spirv/check/provider.h"
#endif  // LOOM_CHECK_HAVE_EMIT_SPIRV
#if LOOM_CHECK_HAVE_TARGET_WASM
#include "loom/target/arch/wasm/check/provider.h"
#endif  // LOOM_CHECK_HAVE_TARGET_WASM
#if LOOM_CHECK_HAVE_TARGET_VM
#include "loom/target/arch/vm/check/provider.h"
#include "loom/tooling/target/vm/check/provider.h"
#endif  // LOOM_CHECK_HAVE_TARGET_VM
#if LOOM_CHECK_HAVE_EMIT_WASM
#include "loom/tooling/target/wasm/check/provider.h"
#endif  // LOOM_CHECK_HAVE_EMIT_WASM
#if LOOM_CHECK_HAVE_TARGET_X86
#include "loom/target/arch/x86/check/provider.h"
#endif  // LOOM_CHECK_HAVE_TARGET_X86
#if LOOM_CHECK_HAVE_TARGET_XDNA
#include "loom/target/arch/amd/xdna/aie2p/check/provider.h"
#if LOOM_CHECK_HAVE_TEST_PROVIDER
#include "loom/target/arch/amd/xdna/aie2p/check/configuration.h"
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#endif  // LOOM_CHECK_HAVE_TARGET_XDNA

static const loom_check_provider_t* const kLoomCheckProviders[] = {
    &loom_cxx_export_check_provider,
#if LOOM_CHECK_HAVE_TEST_PROVIDER
    &loom_check_test_provider,
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
    &loom_cmd_check_provider,
#if LOOM_CHECK_HAVE_EMIT_AMDGPU
    &loom_amdgpu_check_provider,
#endif  // LOOM_CHECK_HAVE_EMIT_AMDGPU
#if LOOM_CHECK_HAVE_TARGET_SPIRV
    &loom_spirv_check_provider,
#endif  // LOOM_CHECK_HAVE_TARGET_SPIRV
#if LOOM_CHECK_HAVE_EMIT_SPIRV
    &loom_spirv_emit_check_provider,
#endif  // LOOM_CHECK_HAVE_EMIT_SPIRV
#if LOOM_CHECK_HAVE_TARGET_WASM
    &loom_wasm_check_provider,
#endif  // LOOM_CHECK_HAVE_TARGET_WASM
#if LOOM_CHECK_HAVE_TARGET_VM
    // VM target records and descriptors.
    &loom_vm_check_provider,
    // VM artifact emission and disassembly.
    &loom_vm_emit_check_provider,
#endif  // LOOM_CHECK_HAVE_TARGET_VM
#if LOOM_CHECK_HAVE_EMIT_WASM
    &loom_wasm_emit_check_provider,
#endif  // LOOM_CHECK_HAVE_EMIT_WASM
#if LOOM_CHECK_HAVE_TARGET_X86
    &loom_x86_check_provider,
#endif  // LOOM_CHECK_HAVE_TARGET_X86
#if LOOM_CHECK_HAVE_TARGET_XDNA
    &loom_aie2p_check_provider,
#if LOOM_CHECK_HAVE_TEST_PROVIDER
    &loom_aie2p_configuration_check_provider,
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#endif  // LOOM_CHECK_HAVE_TARGET_XDNA
};

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);
  const loom_check_compile_provider_t compile_provider = {
      .import = loom_configured_input_loomc_importer(),
  };
  const loom_check_provider_set_t provider_set = {
      .providers = kLoomCheckProviders,
      .provider_count = IREE_ARRAYSIZE(kLoomCheckProviders),
      .target_provider_set = loom_configured_emitter_provider_set(),
      .compile_provider = &compile_provider,
  };
  const int exit_code = loom_check_provider_main(
      argc, argv, &provider_set, loom_configured_input_providers());
  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
