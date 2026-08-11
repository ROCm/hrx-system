// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/test/registry.h"

#include <stdint.h>

#include "loom/ops/test/ops.h"
#include "loom/ops/test/types.h"

iree_status_t loom_test_dialect_register(loom_context_t* context) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
  iree_host_size_t semantics_count = 0;
  const loom_op_semantics_t* semantics =
      loom_test_dialect_op_semantics(&semantics_count);
  if (count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "test dialect has %" PRIhsz
                            " ops, exceeding the uint16_t registry cap",
                            count);
  }
  if (semantics_count != count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test dialect semantics count %" PRIhsz
                            " does not match vtable count %" PRIhsz,
                            semantics_count, count);
  }
  IREE_RETURN_IF_ERROR(loom_context_register_dialect(context, LOOM_DIALECT_TEST,
                                                     vtables, (uint16_t)count));
  IREE_RETURN_IF_ERROR(loom_context_register_dialect_semantics(
      context, LOOM_DIALECT_TEST, semantics, (uint16_t)count));
  iree_host_size_t condition_refinement_count = 0;
  const loom_condition_refinement_descriptor_t* condition_refinements =
      loom_test_dialect_condition_refinements(&condition_refinement_count);
  IREE_RETURN_IF_ERROR(loom_context_register_condition_refinements(
      context, LOOM_DIALECT_TEST, condition_refinements,
      condition_refinement_count));
  iree_host_size_t parameterized_attr_count = 0;
  const loom_parameterized_attr_descriptor_t* parameterized_attrs =
      loom_test_dialect_parameterized_attrs(&parameterized_attr_count);
  if (parameterized_attr_count > 0) {
    IREE_RETURN_IF_ERROR(loom_context_register_parameterized_attrs(
        context, LOOM_DIALECT_TEST, parameterized_attrs,
        parameterized_attr_count));
  }
  return loom_type_registry_register_types(context,
                                           loom_test_type_registry_entries,
                                           loom_test_type_registry_entry_count);
}
