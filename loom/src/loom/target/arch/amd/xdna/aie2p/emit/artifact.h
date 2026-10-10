// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Complete AIE2P XDNA artifact compilation.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_ARTIFACT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_ARTIFACT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/function_version.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_aie2p_xdna_artifact_request_t {
  // Prepared AIE2P target-low IR. Complete workers participate in shared
  // allocation repair during native compilation.
  loom_module_t* module;

  // Concrete compiler function versions participating in compilation.
  const loom_function_version_list_t* function_versions;

  // Low descriptor registry containing AIE2P core, array and configuration
  // sets.
  const loom_low_descriptor_registry_t* low_descriptor_registry;

  // Optional caller-owned structured compile report for this compilation. The
  // report may retain target backing storage until report deinitialization.
  loom_target_compile_report_t* compile_report;

  // Diagnostic emitter receiving target diagnostics.
  iree_diagnostic_emitter_t diagnostic_emitter;

  // Invocation-local scratch arena.
  iree_arena_allocator_t* scratch_arena;

  // Host allocator owning the returned byte sequence.
  iree_allocator_t allocator;
} loom_aie2p_xdna_artifact_request_t;

// Compiles one complete Loom-owned XDNA ELF byte sequence. Structured rejection
// returns OK with |out_emitted| false and no contents.
iree_status_t loom_aie2p_xdna_compile_artifact(
    const loom_aie2p_xdna_artifact_request_t* request, bool* out_emitted,
    iree_byte_sequence_t** out_contents);

// Canonical emitter producing complete XDNA ELF artifacts.
extern const loom_target_emitter_t loom_aie2p_xdna_artifact_emitter;

// Canonical XDNA emission for in-process target environments. Device identity
// comes from the prepared array target facts; no emission-time target override
// or intermediate tile artifacts are required.
extern const loom_target_provider_t loom_aie2p_xdna_artifact_provider;

// Complete AIE2P compiler provider set containing the target architecture and
// canonical XDNA artifact emitter.
extern const loom_target_provider_set_t loom_aie2p_compiler_provider_set;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_ARTIFACT_H_
