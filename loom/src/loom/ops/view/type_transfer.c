// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/roles.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/type_propagation.h"

iree_status_t loom_view_subview_type_transfer(
    loom_type_transfer_context_t* context, const loom_module_t* module,
    loom_op_t* op) {
  const loom_value_slice_t physical_encoding =
      loom_view_subview_physical_encoding(op);
  const loom_value_id_t source = loom_view_subview_source(op);
  const loom_value_id_t result = loom_view_subview_result(op);
  const loom_type_t source_type =
      loom_type_transfer_value_type(context, source);
  const loom_type_t result_type =
      loom_type_transfer_value_type(context, result);

  if (physical_encoding.count == 0) {
    if (loom_type_has_static_encoding(result_type)) {
      IREE_RETURN_IF_ERROR(loom_type_transfer_seed_encoding_attachment(
          context, source, result_type.encoding_id,
          result_type.encoding_flags));
    }
    if (loom_type_has_static_encoding(source_type)) {
      IREE_RETURN_IF_ERROR(loom_type_transfer_seed_encoding_attachment(
          context, result, source_type.encoding_id,
          source_type.encoding_flags));
    }
    return iree_ok_status();
  }

  const loom_encoding_role_t source_role =
      loom_encoding_type_role(module, source_type);
  if (source_role == LOOM_ENCODING_ROLE_UNKNOWN) {
    return iree_ok_status();
  }
  return loom_type_transfer_seed_static_structure_from_type(
      context, physical_encoding.values[0],
      loom_type_encoding_with_role(source_role));
}
