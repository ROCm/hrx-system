// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Body-blind kernel launch-configuration materialization.

#ifndef LOOM_LINK_KERNEL_CONFIG_MATERIALIZER_H_
#define LOOM_LINK_KERNEL_CONFIG_MATERIALIZER_H_

#include "loom/link/plan_materializer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Ordinary IR projected from one selected bytecode kernel definition.
typedef struct loom_link_kernel_config_materialization_t {
  // Standalone module owned by the caller.
  loom_module_t* module;
  // Private kernel declaration retaining the source contract.
  loom_symbol_ref_t kernel_declaration;
  // Private pure function from workload arguments to xyz workgroup counts.
  loom_symbol_ref_t configuration_function;
} loom_link_kernel_config_materialization_t;

// Materializes the contract and launch configuration of one selected bytecode
// kernel without reading its implementation root region.
//
// |kernel_symbol_ordinal| is an index-wide identity from |plan| and must be
// selected by that plan. The source must be a bytecode kernel definition with
// a launch-configuration region. The returned module contains a private
// kernel.decl, a private pure inline func.def, and a target.decl when the
// source contract is targeted. Implementation IR and its root-region payload
// are neither decoded nor validated.
//
// The selected configuration must be structurally self-contained apart from
// its target contract. References to other source symbols fail explicitly.
iree_status_t loom_link_plan_materialize_bytecode_kernel_config(
    const loom_link_plan_t* plan, iree_host_size_t kernel_symbol_ordinal,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name,
    loom_link_kernel_config_materialization_t* out_materialization);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_KERNEL_CONFIG_MATERIALIZER_H_
