// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/op_registry.h"

#include <stdint.h>

#include "loom/ops/encoding/families.h"
#include "loom/ops/op_registry_tables.h"

static iree_status_t loom_op_registry_register_dialect(
    loom_context_t* context,
    const loom_op_registry_dialect_registration_t* registration) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = registration->vtables_fn(&count);
  iree_host_size_t semantics_count = 0;
  const loom_op_semantics_t* semantics =
      registration->semantics_fn(&semantics_count);
  if (count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dialect %u has %" PRIhsz
                            " ops, exceeding the uint16_t registry cap",
                            (unsigned)registration->dialect_id, count);
  }
  if (semantics_count != count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "dialect %u semantics count %" PRIhsz
                            " does not match vtable count %" PRIhsz,
                            (unsigned)registration->dialect_id, semantics_count,
                            count);
  }
  IREE_RETURN_IF_ERROR(loom_context_register_dialect(
      context, registration->dialect_id, vtables, (uint16_t)count));
  return loom_context_register_dialect_semantics(
      context, registration->dialect_id, semantics, (uint16_t)count);
}

iree_status_t loom_op_registry_register_all_dialects(loom_context_t* context) {
  for (iree_host_size_t i = 0; i < loom_op_registry_dialect_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_op_registry_register_dialect(
        context, &loom_op_registry_dialects[i]));
  }
  return loom_context_register_builtin_encoding_vtables(context);
}
