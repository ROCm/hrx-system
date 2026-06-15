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
#include "loom/error/diagnostic.h"
#include "loom/ir/ir.h"
#include "loom/target/artifact_manifest_collect.h"
#include "loom/target/compile_report.h"
#include "loom/target/provider.h"
#include "loom/target/types.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_hal_kernel_library_options_t {
  // Optional AMDHSA processor name such as `gfx1100` overriding the selected
  // target record's processor. This preserves the target record's
  // descriptor-set family while letting JIT runners specialize to the concrete
  // HAL device ISA.
  iree_string_view_t processor;
  // Optional runtime/device target selection applied to compatible module
  // target records before entry selection, verification, scheduling, and
  // allocation.
  loom_target_selection_t target_selection;
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
  // Allocator-owned target id used by the AMDGPU loader.
  iree_string_view_t executable_format;
  // Allocator-owned HSACO ELF image bytes.
  uint8_t* hsaco_data;
  // Number of bytes in |hsaco_data|.
  iree_host_size_t hsaco_data_length;
  // Textual listing format for |target_listing_data|.
  iree_string_view_t target_listing_format;
  // Allocator-owned textual target listing.
  char* target_listing_data;
  // Number of bytes in |target_listing_data|, excluding the trailing NUL.
  iree_host_size_t target_listing_data_length;
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
