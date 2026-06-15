// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"

static iree_status_t loom_target_emit(iree_diagnostic_emitter_t emitter,
                                      const loom_op_t* op,
                                      const loom_error_def_t* error,
                                      const loom_diagnostic_param_t* params,
                                      iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_target_emit_attr_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_target_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                          IREE_ARRAYSIZE(params));
}

static iree_string_view_t loom_target_attr_name(const loom_module_t* module,
                                                const loom_op_t* op,
                                                uint8_t attr_index) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  return loom_attr_descriptor_name(&vtable->attr_descriptors[attr_index]);
}

static iree_status_t loom_target_verify_u32_projection(
    const loom_module_t* module, const loom_op_t* op,
    const loom_target_projection_t* projection,
    iree_diagnostic_emitter_t emitter) {
  loom_attribute_t attr = loom_op_attrs(op)[projection->attr_index];
  if (loom_attr_is_absent(attr)) return iree_ok_status();
  const int64_t value = loom_attr_as_i64(attr);
  if (value >= 0 && value <= UINT32_MAX) return iree_ok_status();
  return loom_target_emit_attr_constraint(
      emitter, op, loom_target_attr_name(module, op, projection->attr_index),
      value, IREE_SV("an unsigned 32-bit integer"));
}

static iree_status_t loom_target_verify_u64_projection(
    const loom_module_t* module, const loom_op_t* op,
    const loom_target_projection_t* projection,
    iree_diagnostic_emitter_t emitter) {
  loom_attribute_t attr = loom_op_attrs(op)[projection->attr_index];
  if (loom_attr_is_absent(attr)) return iree_ok_status();
  const int64_t value = loom_attr_as_i64(attr);
  if (value >= 0) return iree_ok_status();
  return loom_target_emit_attr_constraint(
      emitter, op, loom_target_attr_name(module, op, projection->attr_index),
      value, IREE_SV("a non-negative integer"));
}

static iree_status_t loom_target_verify_projection(
    const loom_module_t* module, const loom_op_t* op,
    const loom_target_projection_t* projection,
    iree_diagnostic_emitter_t emitter) {
  switch (projection->value_kind) {
    case LOOM_TARGET_PROJECTION_VALUE_ENUM_U8:
    case LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW:
      return iree_ok_status();
    case LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32:
      return loom_target_verify_u32_projection(module, op, projection, emitter);
    case LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64:
      return loom_target_verify_u64_projection(module, op, projection, emitter);
  }
  return iree_ok_status();
}

iree_status_t loom_target_record_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_target_like_t target = loom_target_like_cast(module, op);
  const loom_target_like_descriptor_t* descriptor =
      loom_target_like_descriptor(target);
  const uint32_t selector =
      (uint32_t)loom_attr_as_enum(loom_target_like_selector(target));
  if (selector >= descriptor->bundle_table->count ||
      descriptor->bundle_table->values[selector] == NULL) {
    iree_string_view_t selector_attr_name =
        loom_target_attr_name(module, op, target.vtable->selector_attr_index);
    return loom_target_emit_attr_constraint(
        emitter, op, selector_attr_name, selector,
        IREE_SV("a linked target row selector"));
  }
  for (uint8_t i = 0; i < descriptor->projection_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_target_verify_projection(
        module, op, &descriptor->projections[i], emitter));
  }
  return iree_ok_status();
}
