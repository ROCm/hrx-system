// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL kernel-library emission for prepared target-low Loom modules.
//
// This emits prepared low.kernel.def entries into native HSACO bytes.
// Source-to-low lowering, HAL ABI/resource materialization, and target-low
// preparation are owned by the caller's compile pipeline.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_HAL_KERNEL_LIBRARY_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_HAL_KERNEL_LIBRARY_H_

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/native/amdgpu/runtime_globals.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/artifact_manifest_collect.h"
#include "loom/target/reporting/report.h"
#include "loom/target/types.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_hal_kernel_library_options_t {
  // Optional concrete compiler function versions participating in this
  // emission. The list and its version objects are borrowed for the call.
  const loom_function_version_list_t* function_versions;
  // Optional AMDGPU runtime support globals emitted into the HSACO.
  loom_amdgpu_runtime_global_flags_t runtime_globals;
  // Optional caller-owned code-object data symbols emitted into the HSACO.
  const loom_amdgpu_hsaco_data_symbol_t* data_symbols;
  // Number of entries in |data_symbols|.
  iree_host_size_t data_symbol_count;
  // Diagnostic sink used for verification, materialization, scheduling, and
  // allocation diagnostics. A NULL callback still counts diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Source resolver used to render carets for op-backed diagnostics.
  loom_source_resolver_t source_resolver;
  // Maximum diagnostics to emit before the active subsystem stops walking.
  // Zero uses a conservative default.
  uint32_t max_errors;
  // Optional caller-owned structured compile report to populate.
  loom_target_compile_report_t* report;
  // True to retain target-owned textual assembly listings for debug artifacts.
  bool capture_target_listing;
  // Emitted artifact name recorded inside an artifact manifest.
  iree_string_view_t artifact_name;
  // Artifact manifest sidecar identifier.
  iree_string_view_t artifact_manifest_identifier;
  // Optional artifact manifest collection request.
  loom_target_artifact_manifest_collect_options_t artifact_manifest;
} loom_amdgpu_hal_kernel_library_options_t;

// Allocator-owned AMDGPU HSACO kernel-library artifact.
typedef struct loom_amdgpu_hal_kernel_library_t {
  // Allocator-owned AMDGPU target key used to emit the artifact.
  iree_string_view_t target_key;
  // Durable target bundle resolved from the emitted entries.
  loom_target_bundle_storage_t target_bundle_storage;
  // Owned reference to immutable HSACO ELF image contents.
  iree_byte_sequence_t* hsaco_data;
  // Textual listing format for |target_listing_data|.
  iree_string_view_t target_listing_format;
  // Owned reference to immutable textual target listing contents.
  iree_byte_sequence_t* target_listing_data;
  // Artifact manifest sidecar produced beside |hsaco_data|.
  loom_target_emit_sidecar_artifact_t artifact_manifest;
} loom_amdgpu_hal_kernel_library_t;

// Releases storage owned by |library|. Safe to call on a zero-initialized
// library object.
void loom_amdgpu_hal_kernel_library_deinitialize(
    loom_amdgpu_hal_kernel_library_t* library, iree_allocator_t allocator);

// Emits |module| into an allocator-owned AMDGPU HAL kernel library.
//
// |module| must already contain the prepared target-low entries intended for
// the artifact. Target records are resolved through the linked descriptor
// registry without materializing companion target records in the IR.
// |out_emitted| is false when target preflight or diagnostics rejected the
// module; status remains reserved for infrastructure failures. The caller owns
// |out_library| when |out_emitted| is true and must release it with
// loom_amdgpu_hal_kernel_library_deinitialize.
iree_status_t loom_amdgpu_emit_hal_kernel_library(
    loom_module_t* module,
    const loom_amdgpu_hal_kernel_library_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_amdgpu_hal_kernel_library_t* out_library);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_HAL_KERNEL_LIBRARY_H_
