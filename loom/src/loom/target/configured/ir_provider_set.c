// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/configured/ir_provider_set.h"

#include "loom/target/arch/cmd/provider.h"

#ifndef LOOM_CONFIG_TARGET_HAVE_VM
#define LOOM_CONFIG_TARGET_HAVE_VM 0
#endif  // LOOM_CONFIG_TARGET_HAVE_VM
#ifndef LOOM_CONFIG_TARGET_HAVE_AMDGPU
#define LOOM_CONFIG_TARGET_HAVE_AMDGPU 0
#endif  // LOOM_CONFIG_TARGET_HAVE_AMDGPU
#ifndef LOOM_CONFIG_TARGET_HAVE_XDNA
#define LOOM_CONFIG_TARGET_HAVE_XDNA 0
#endif  // LOOM_CONFIG_TARGET_HAVE_XDNA
#ifndef LOOM_CONFIG_TARGET_HAVE_SPIRV
#define LOOM_CONFIG_TARGET_HAVE_SPIRV 0
#endif  // LOOM_CONFIG_TARGET_HAVE_SPIRV
#ifndef LOOM_CONFIG_TARGET_HAVE_WASM
#define LOOM_CONFIG_TARGET_HAVE_WASM 0
#endif  // LOOM_CONFIG_TARGET_HAVE_WASM
#ifndef LOOM_CONFIG_TARGET_HAVE_X86
#define LOOM_CONFIG_TARGET_HAVE_X86 0
#endif  // LOOM_CONFIG_TARGET_HAVE_X86

#if LOOM_CONFIG_TARGET_HAVE_VM
#include "loom/target/arch/vm/ir_provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_VM
#if LOOM_CONFIG_TARGET_HAVE_AMDGPU
#include "loom/target/arch/amdgpu/ir_provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_AMDGPU
#if LOOM_CONFIG_TARGET_HAVE_XDNA
#include "loom/target/arch/amd/xdna/aie2p/ir_provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_XDNA
#if LOOM_CONFIG_TARGET_HAVE_SPIRV
#include "loom/target/arch/spirv/ir_provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_SPIRV
#if LOOM_CONFIG_TARGET_HAVE_WASM
#include "loom/target/arch/wasm/ir_provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_WASM
#if LOOM_CONFIG_TARGET_HAVE_X86
#include "loom/target/arch/x86/ir_provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_X86

static const loom_target_provider_t* const kConfiguredIrProviders[] = {
    &loom_cmd_target_provider,
#if LOOM_CONFIG_TARGET_HAVE_VM
    &loom_vm_ir_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_VM
#if LOOM_CONFIG_TARGET_HAVE_AMDGPU
    &loom_amdgpu_ir_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_AMDGPU
#if LOOM_CONFIG_TARGET_HAVE_XDNA
    &loom_aie2p_ir_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_XDNA
#if LOOM_CONFIG_TARGET_HAVE_SPIRV
    &loom_spirv_ir_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_SPIRV
#if LOOM_CONFIG_TARGET_HAVE_WASM
    &loom_wasm_ir_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_WASM
#if LOOM_CONFIG_TARGET_HAVE_X86
    &loom_x86_ir_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_X86
};

static const loom_target_provider_set_t kConfiguredIrProviderSet = {
    .providers = kConfiguredIrProviders,
    .provider_count = IREE_ARRAYSIZE(kConfiguredIrProviders),
};

const loom_target_provider_set_t* loom_configured_ir_provider_set(void) {
  return &kConfiguredIrProviderSet;
}
