// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_KERNEL_H_
#define LOOMC_TARGET_KERNEL_H_

#include "loomc/compile.h"
#include "loomc/emit.h"
#include "loomc/product.h"

/// @file
/// Immutable executable kernel products.
///
/// A kernel product compiles every request root into one shared target-native
/// artifact. Roots requested as host-launchable logical kernels additionally
/// project into one shared launch-configuration artifact. The product records
/// exact artifact and function ordinals for both facets; loading code never
/// reparses artifacts or joins entries by name.
///
/// Root goals identify the semantic value being requested. A physical entry
/// is ready for a parent that already owns launch geometry. A host-launchable
/// root accepts authored workload arguments and therefore carries the coupled
/// launch policy needed to derive physical geometry.

#ifdef __cplusplus
extern "C" {
#endif

/// Produces only the physical executable entry for a request root.
#define LOOMC_KERNEL_ROOT_GOAL_EXECUTABLE_ENTRY LOOMC_REQUEST_ROOT_GOAL_DEFAULT

/// Produces a physical entry and its coupled host launch configuration.
#define LOOMC_KERNEL_ROOT_GOAL_HOST_LAUNCHABLE 1u

/// Invalid product-local artifact ordinal.
#define LOOMC_KERNEL_ARTIFACT_ORDINAL_INVALID UINT32_MAX

/// Invalid artifact-local function ordinal.
#define LOOMC_KERNEL_FUNCTION_ORDINAL_INVALID UINT32_MAX

/// Product projection for one request root.
///
/// Several roots may name the same artifacts and artifact-local functions.
/// Executable fields are always valid. Launch-config fields are both invalid
/// for executable-entry roots and both valid for host-launchable roots.
typedef struct loomc_kernel_product_root_t {
  /// Product-local target-native executable artifact ordinal.
  uint32_t executable_artifact_ordinal;

  /// Artifact-local target-native function ordinal.
  uint32_t executable_function_ordinal;

  /// Product-local launch-config artifact ordinal, or INVALID.
  uint32_t launch_config_artifact_ordinal;

  /// Artifact-local launch-config function ordinal, or INVALID.
  uint32_t launch_config_function_ordinal;
} loomc_kernel_product_root_t;

/// Compiles one immutable kernel request into executable artifacts.
///
/// Request root order becomes product export order. Duplicate source roots are
/// preserved, including roots that request different goals. All roots share
/// one native emission, and all host-launchable roots share one launch-config
/// emission. An executable-only request performs no launch capture or VM
/// emission.
///
/// `compile_options` controls ordinary compiler specialization, configuration,
/// and diagnostic artifacts. `LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG` is
/// rejected because launch production is selected by root goals. `emit_options`
/// selects the target-native emitter and its options.
///
/// @param compiler Prepared compiler whose context owns the target environment.
/// @param workspace Invocation-local scratch workspace.
/// @param pass_program Prepared target compilation pipeline.
/// @param request Immutable request requiring the kernel product descriptor.
/// @param compile_options Compile invocation options, or NULL for defaults.
/// @param emit_options Target-native emission options, or NULL for defaults.
/// @param allocator Host allocator used for product and result storage.
/// @param out_product Receives a retained product only when the result
/// succeeds.
/// @param out_result Receives a retained operation result.
/// @return OK when the operation ran to a result. Non-OK statuses represent
/// structural API misuse or infrastructure failure.
///
/// @ownership
/// The caller releases `out_product` with `loomc_product_release` and
/// `out_result` with `loomc_result_release`.
///
/// @lifetime
/// The returned product owns immutable artifact references and root rows. It
/// retains no request, mutable module, workspace, IR, or target facts.
LOOMC_API_EXPORT loomc_status_t loomc_kernel_product_build_request(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, const loomc_request_t* request,
    const loomc_compile_options_t* compile_options,
    const loomc_emit_options_t* emit_options, loomc_allocator_t allocator,
    loomc_product_t** out_product, loomc_result_t** out_result);

/// Returns the process-local kernel-product descriptor.
///
/// @return Process-lifetime descriptor for kernel products.
LOOMC_API_EXPORT const loomc_product_descriptor_t*
loomc_kernel_product_descriptor(void);

/// Returns one root projection by product-local ordinal.
///
/// @param product Kernel product to query.
/// @param ordinal Product-local root ordinal.
/// @param out_root Receives the borrowed root record when found.
/// @return True when `ordinal` was valid and `out_root` was populated.
LOOMC_API_EXPORT bool loomc_kernel_product_root_at(
    const loomc_product_t* product, loomc_host_size_t ordinal,
    loomc_kernel_product_root_t* out_root);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_KERNEL_H_
