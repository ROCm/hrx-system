// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Directional function-like symbol contract comparison and merge.

#ifndef LOOM_LINK_FUNC_CONTRACT_H_
#define LOOM_LINK_FUNC_CONTRACT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/remap.h"

#ifdef __cplusplus
extern "C" {
#endif

// Semantic reason a source contract cannot be applied to a selected contract.
typedef enum loom_link_func_contract_mismatch_kind_e {
  // The contracts are compatible.
  LOOM_LINK_FUNC_CONTRACT_MISMATCH_NONE = 0,
  // One named contract field is incompatible.
  LOOM_LINK_FUNC_CONTRACT_MISMATCH_FIELD = 1,
  // One named signature sequence has a different element count.
  LOOM_LINK_FUNC_CONTRACT_MISMATCH_COUNT = 2,
  // One named signature sequence has an incompatible element type.
  LOOM_LINK_FUNC_CONTRACT_MISMATCH_TYPE = 3,
} loom_link_func_contract_mismatch_kind_t;

// One allocation-free semantic contract mismatch.
typedef struct loom_link_func_contract_mismatch_t {
  // Mismatch category, or NONE when the contracts are compatible.
  loom_link_func_contract_mismatch_kind_t kind;
  // Stable contract field name identifying the mismatch.
  iree_string_view_t field_name;
  // Category-specific mismatch detail.
  union {
    // Element counts for a COUNT mismatch.
    struct {
      // Number of elements required by the source contract.
      iree_host_size_t source;
      // Number of elements exposed by the selected contract.
      iree_host_size_t selected;
    } counts;
    // Zero-based element ordinal for a TYPE mismatch.
    iree_host_size_t type_ordinal;
  } detail;
} loom_link_func_contract_mismatch_t;

// Returns true when |mismatch| records an incompatibility.
static inline bool loom_link_func_contract_mismatch_present(
    const loom_link_func_contract_mismatch_t* mismatch) {
  return mismatch->kind != LOOM_LINK_FUNC_CONTRACT_MISMATCH_NONE;
}

// One function-like contract projected into a module's identity domain.
//
// The view borrows every array and operation descriptor. Signature values,
// types, attributes, strings, and symbol references belong to |module|.
typedef struct loom_link_func_contract_t {
  // Module owning every projected identity in this contract.
  const loom_module_t* module;
  // Structural symbol definition implemented by the source operation.
  const loom_symbol_definition_descriptor_t* symbol_definition;
  // Function-like operation descriptor.
  const loom_func_like_vtable_t* function;
  // Kernel workload arguments preceding ordinary function arguments.
  loom_value_slice_t workload_arguments;
  // Ordinary function argument values.
  loom_value_slice_t arguments;
  // Function result values.
  loom_value_slice_t results;
  // Tied-result records in result order.
  const loom_tied_result_t* tied_results;
  // Number of entries in |tied_results|.
  uint16_t tied_result_count;
  // Complete function-like operation attribute array.
  loom_attribute_t* attributes;
} loom_link_func_contract_t;

// Projects one verified function-like operation into a borrowed contract view.
//
// |op| must implement the function-like symbol interface. The returned view
// borrows |module| and |op| and requires no teardown.
loom_link_func_contract_t loom_link_func_contract_from_op(
    const loom_module_t* module, loom_op_t* op);

// Checks whether |source| can be applied to |selected| without writing either
// contract.
//
// The comparison is directional: absent source fields impose no constraint,
// while a present source field must be representable by the selected contract.
// |remap| must project |source->module| into |selected->module|. Callers may
// reuse it across comparisons between the same pair of modules so signature
// correspondence and remapped identity storage scale with the modules instead
// of the number of candidate pairs. Remapping may intern projected identities
// in |selected->module|.
// Semantic incompatibility is returned through |out_mismatch| with OK status;
// statuses are reserved for allocation and remap failures.
iree_status_t loom_link_func_contract_check(
    const loom_link_func_contract_t* source,
    const loom_link_func_contract_t* selected, loom_ir_remap_t* remap,
    loom_link_func_contract_mismatch_t* out_mismatch);

enum loom_link_func_contract_merge_flag_bits_e {
  // Merge output visibility, import, and export linkage fields.
  LOOM_LINK_FUNC_CONTRACT_MERGE_FLAG_OUTPUT = 1u << 0,
};
typedef uint32_t loom_link_func_contract_merge_flags_t;

// Applies |source| to |selected| and fills absent selected fields.
//
// The comparison rules are identical to loom_link_func_contract_check.
// Compatible source fields absent from |selected| are remapped into
// |selected->module| and written through |selected->attributes|. Semantic
// incompatibility is returned through |out_mismatch| with OK status.
iree_status_t loom_link_func_contract_merge(
    const loom_link_func_contract_t* source,
    const loom_link_func_contract_t* selected, loom_ir_remap_t* remap,
    loom_link_func_contract_merge_flags_t flags,
    loom_link_func_contract_mismatch_t* out_mismatch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_FUNC_CONTRACT_H_
