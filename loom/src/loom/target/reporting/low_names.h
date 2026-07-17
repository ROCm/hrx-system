// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_REPORTING_LOW_NAMES_H_
#define LOOM_TARGET_REPORTING_LOW_NAMES_H_

#include "loom/analysis/liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns whether |descriptor| has the exact semantic |tag|.
bool loom_target_compile_report_descriptor_semantic_tag_is(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_string_view_t tag);

// Returns the semantic tag of |descriptor|, or empty when unavailable.
iree_string_view_t loom_target_compile_report_descriptor_semantic_tag(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns a module string or |fallback| when |string_id| is unavailable.
iree_string_view_t loom_target_compile_report_module_string(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t fallback);

// Returns the diagnostic name of |value_id|.
iree_string_view_t loom_target_compile_report_value_name(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns the diagnostic label of |block|.
iree_string_view_t loom_target_compile_report_block_name(
    const loom_module_t* module, const loom_block_t* block);

// Returns the target register-class name for |value_class|.
iree_string_view_t loom_target_compile_report_value_class_name(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_liveness_value_class_t value_class);

// Returns the operation mnemonic of |op|.
iree_string_view_t loom_target_compile_report_op_name(
    const loom_module_t* module, const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TARGET_REPORTING_LOW_NAMES_H_
