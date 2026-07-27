// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_SPIRV_EMIT_H_
#define LOOMC_TARGET_SPIRV_EMIT_H_

#include "loomc/emit.h"
#include "loomc/target/spirv/base.h"

/// @file
/// SPIR-V emission option space.
///
/// Link the SPIR-V target binding package to make the generic
/// `loomc_emit_module` operation capable of producing
/// `LOOMC_ARTIFACT_FORMAT_SPIRV` artifacts. This header owns SPIR-V-specific
/// emission descriptors that may be attached to `loomc_emit_options_t::next`.
///
/// @par Example
/// Emit a prepared SPIR-V target-low module:
///
/// @code{.c}
/// loomc_emit_options_t emit_options = {
///     .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
///     .structure_size = sizeof(loomc_emit_options_t),
///     .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
///     .identifier = loomc_make_cstring_view("kernel.spv"),
/// };
///
/// loomc_result_t* emit_result = NULL;
/// loomc_status_t status = loomc_emit_module(
///     target_environment, workspace, module, &emit_options,
///     loomc_allocator_system(), &emit_result);
/// if (!loomc_status_is_ok(status)) return status;
/// if (!loomc_result_succeeded(emit_result)) {
///   // Inspect typed diagnostics from emit_result.
/// }
/// loomc_result_release(emit_result);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// SPIR-V binary emission options.
///
/// SPIR-V emission uses the generic artifact format, artifact identifier, and
/// target selection settings on `loomc_emit_options_t` and shared invocation
/// descriptors. This target-specific descriptor carries no additional stable
/// fields today.
typedef struct loomc_spirv_emit_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;
} loomc_spirv_emit_options_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_SPIRV_EMIT_H_
