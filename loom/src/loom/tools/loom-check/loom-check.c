// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check binary with all build-enabled product providers linked in.
// Test builds may additionally enable the synthetic test provider.

#include <stdio.h>

#include "loom/ops/test/registry.h"
#include "loom/target/arch/cmd/check/provider.h"
#include "loom/target/provider.h"
#include "loom/tooling/export/cxx/loom_check.h"
#include "loom/tooling/input/loomc_configured.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/provider.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/target/configured.h"
#include "loomc/target/configured_target.h"

#ifndef LOOM_CHECK_HAVE_TEST_PROVIDER
#define LOOM_CHECK_HAVE_TEST_PROVIDER 0
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#if LOOM_CHECK_HAVE_TEST_PROVIDER
#include "loom/target/test/provider.h"
#include "loom/tools/loom-check/test_provider.h"
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#ifndef LOOM_CHECK_HAVE_ARCH_AIE2P
#define LOOM_CHECK_HAVE_ARCH_AIE2P 0
#endif  // LOOM_CHECK_HAVE_ARCH_AIE2P
#ifndef LOOM_CHECK_HAVE_ARCH_AMDGPU
#define LOOM_CHECK_HAVE_ARCH_AMDGPU 0
#endif  // LOOM_CHECK_HAVE_ARCH_AMDGPU
#ifndef LOOM_CHECK_HAVE_ARCH_SPIRV
#define LOOM_CHECK_HAVE_ARCH_SPIRV 0
#endif  // LOOM_CHECK_HAVE_ARCH_SPIRV
#ifndef LOOM_CHECK_HAVE_ARTIFACT_AIE2P_ELF
#define LOOM_CHECK_HAVE_ARTIFACT_AIE2P_ELF 0
#endif  // LOOM_CHECK_HAVE_ARTIFACT_AIE2P_ELF
#ifndef LOOM_CHECK_HAVE_ARTIFACT_AMDGPU_ELF
#define LOOM_CHECK_HAVE_ARTIFACT_AMDGPU_ELF 0
#endif  // LOOM_CHECK_HAVE_ARTIFACT_AMDGPU_ELF
#ifndef LOOM_CHECK_HAVE_ARTIFACT_SPIRV_BINARY
#define LOOM_CHECK_HAVE_ARTIFACT_SPIRV_BINARY 0
#endif  // LOOM_CHECK_HAVE_ARTIFACT_SPIRV_BINARY
#ifndef LOOM_CHECK_HAVE_ARTIFACT_VM_BINARY
#define LOOM_CHECK_HAVE_ARTIFACT_VM_BINARY 0
#endif  // LOOM_CHECK_HAVE_ARTIFACT_VM_BINARY
#ifndef LOOM_CHECK_HAVE_ARTIFACT_WASM_BINARY
#define LOOM_CHECK_HAVE_ARTIFACT_WASM_BINARY 0
#endif  // LOOM_CHECK_HAVE_ARTIFACT_WASM_BINARY
#ifndef LOOM_CHECK_HAVE_ARTIFACT_X86_ELF
#define LOOM_CHECK_HAVE_ARTIFACT_X86_ELF 0
#endif  // LOOM_CHECK_HAVE_ARTIFACT_X86_ELF
#if LOOM_CHECK_HAVE_ARCH_AIE2P
#if LOOM_CHECK_HAVE_TEST_PROVIDER
#include "loom/target/arch/amd/xdna/aie2p/check/configuration.h"
#include "loom/target/arch/amd/xdna/aie2p/check/provider.h"
#include "loom/target/arch/amd/xdna/aie2p/check/test_provider.h"
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#endif  // LOOM_CHECK_HAVE_ARCH_AIE2P
#if LOOM_CHECK_HAVE_ARCH_AMDGPU
#include "loom/target/arch/amdgpu/check/provider.h"
#if LOOM_CHECK_HAVE_TEST_PROVIDER
#include "loom/target/arch/amdgpu/check/test_provider.h"
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#endif  // LOOM_CHECK_HAVE_ARCH_AMDGPU
#if LOOM_CHECK_HAVE_ARCH_SPIRV
#include "loom/target/arch/spirv/check/provider.h"
#endif  // LOOM_CHECK_HAVE_ARCH_SPIRV
#if LOOM_CHECK_HAVE_ARTIFACT_AIE2P_ELF
#include "loom/target/emit/native/aie2p/check_provider.h"
#endif  // LOOM_CHECK_HAVE_ARTIFACT_AIE2P_ELF
#if LOOM_CHECK_HAVE_ARTIFACT_AMDGPU_ELF
#include "loom/target/emit/native/amdgpu/loom_check_provider.h"
#if LOOM_CHECK_HAVE_TEST_PROVIDER
#include "loom/target/emit/native/amdgpu/loom_check_test_provider.h"
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#endif  // LOOM_CHECK_HAVE_ARTIFACT_AMDGPU_ELF
#if LOOM_CHECK_HAVE_ARTIFACT_SPIRV_BINARY
#include "loom/tooling/target/spirv/loom_check_provider.h"
#endif  // LOOM_CHECK_HAVE_ARTIFACT_SPIRV_BINARY
#if LOOM_CHECK_HAVE_ARTIFACT_VM_BINARY
#include "loom/tooling/target/vm/check/provider.h"
#endif  // LOOM_CHECK_HAVE_ARTIFACT_VM_BINARY
#if LOOM_CHECK_HAVE_ARTIFACT_WASM_BINARY
#include "loom/tooling/target/wasm/loom_check_provider.h"
#endif  // LOOM_CHECK_HAVE_ARTIFACT_WASM_BINARY
#if LOOM_CHECK_HAVE_ARTIFACT_X86_ELF
#if LOOM_CHECK_HAVE_TEST_PROVIDER
#include "loom/target/emit/native/x86/loom_check_provider.h"
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#endif  // LOOM_CHECK_HAVE_ARTIFACT_X86_ELF
static const loom_check_provider_t* const kLoomCheckProviders[] = {
    &loom_cxx_export_check_provider,
#if LOOM_CHECK_HAVE_TEST_PROVIDER
    &loom_check_test_provider,
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
    &loom_cmd_check_provider,
#if LOOM_CHECK_HAVE_ARCH_AIE2P
#if LOOM_CHECK_HAVE_TEST_PROVIDER
    &loom_aie2p_arch_check_provider,
    &loom_aie2p_check_test_provider,
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#endif  // LOOM_CHECK_HAVE_ARCH_AIE2P
#if LOOM_CHECK_HAVE_ARCH_AMDGPU
    &loom_amdgpu_arch_check_provider,
#if LOOM_CHECK_HAVE_TEST_PROVIDER
    &loom_amdgpu_arch_check_test_provider,
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#endif  // LOOM_CHECK_HAVE_ARCH_AMDGPU
#if LOOM_CHECK_HAVE_ARCH_SPIRV
    &loom_spirv_arch_check_provider,
#endif  // LOOM_CHECK_HAVE_ARCH_SPIRV
#if LOOM_CHECK_HAVE_ARTIFACT_AIE2P_ELF
    &loom_aie2p_artifact_check_provider,
#endif  // LOOM_CHECK_HAVE_ARTIFACT_AIE2P_ELF
#if LOOM_CHECK_HAVE_ARTIFACT_AMDGPU_ELF
    &loom_amdgpu_native_check_provider,
#if LOOM_CHECK_HAVE_TEST_PROVIDER
    &loom_amdgpu_native_check_test_provider,
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#endif  // LOOM_CHECK_HAVE_ARTIFACT_AMDGPU_ELF
#if LOOM_CHECK_HAVE_ARTIFACT_SPIRV_BINARY
    &loom_spirv_emit_check_provider,
#endif  // LOOM_CHECK_HAVE_ARTIFACT_SPIRV_BINARY
#if LOOM_CHECK_HAVE_ARTIFACT_VM_BINARY
    &loom_vm_emit_check_provider,
#endif  // LOOM_CHECK_HAVE_ARTIFACT_VM_BINARY
#if LOOM_CHECK_HAVE_ARTIFACT_WASM_BINARY
    &loom_wasm_emit_check_provider,
#endif  // LOOM_CHECK_HAVE_ARTIFACT_WASM_BINARY
#if LOOM_CHECK_HAVE_ARTIFACT_X86_ELF
#if LOOM_CHECK_HAVE_TEST_PROVIDER
    &loom_x86_native_check_provider,
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
#endif  // LOOM_CHECK_HAVE_ARTIFACT_X86_ELF
};

// Developer tools accept test.* syntax without synthetic compiler behavior.
// The test binary's full provider registers that dialect along with its passes
// and descriptors, so it replaces this syntax-only contribution.
#if !LOOM_CHECK_HAVE_TEST_PROVIDER
static const loom_target_provider_t kLoomCheckSyntaxProvider = {
    .register_context = loom_test_dialect_register,
};
#endif  // !LOOM_CHECK_HAVE_TEST_PROVIDER

static const loom_target_provider_t* const kLoomCheckTargetExtensions[] = {
#if LOOM_CHECK_HAVE_TEST_PROVIDER
    &loom_test_target_provider,
#if LOOM_CHECK_HAVE_ARCH_AIE2P
    &loom_aie2p_configuration_check_target_provider,
#endif  // LOOM_CHECK_HAVE_ARCH_AIE2P
#else
    &kLoomCheckSyntaxProvider,
#endif  // LOOM_CHECK_HAVE_TEST_PROVIDER
};

static const loom_target_provider_set_t kLoomCheckTargetExtensionSet = {
    .providers = kLoomCheckTargetExtensions,
    .provider_count = IREE_ARRAYSIZE(kLoomCheckTargetExtensions),
};

static loomc_status_t loom_check_create_target_environment(
    loomc_allocator_t allocator, loomc_target_environment_t** out_environment) {
  loomc_target_environment_t* inputs[3] = {NULL, NULL, NULL};
  // Target support includes command IR and architecture-only builds. Compiler
  // support adds artifact emitters; public composition deduplicates the target
  // providers shared by both packages.
  loomc_status_t status =
      loomc_target_environment_create_configured_target(allocator, &inputs[0]);
  if (loomc_status_is_ok(status)) {
    status = loomc_target_environment_create_configured(allocator, &inputs[1]);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_target_environment_create_from_provider_set(
        &kLoomCheckTargetExtensionSet, allocator, &inputs[2]);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_target_environment_create(IREE_ARRAYSIZE(inputs), inputs,
                                             allocator, out_environment);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    loomc_target_environment_release(inputs[i]);
  }
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);
  loomc_target_environment_t* target_environment = NULL;
  iree_status_t status =
      iree_status_from_loomc(loom_check_create_target_environment(
          loomc_allocator_system(), &target_environment));
  const loom_check_compile_provider_t compile_provider = {
      .import = loom_configured_input_loomc_importer(),
  };
  const loom_check_provider_set_t provider_set = {
      .providers = kLoomCheckProviders,
      .provider_count = IREE_ARRAYSIZE(kLoomCheckProviders),
      .target_environment = target_environment,
      .compile_provider = &compile_provider,
  };
  int exit_code = 1;
  if (iree_status_is_ok(status)) {
    exit_code = loom_check_provider_main(argc, argv, &provider_set,
                                         loom_configured_input_loomc_formats());
  } else {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }
  loomc_target_environment_release(target_environment);
  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
