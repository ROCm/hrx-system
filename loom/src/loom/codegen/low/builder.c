// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/builder.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/descriptor_traits.h"
#include "loom/ir/module.h"
#include "loom/target/registers.h"

static iree_status_t loom_low_validate_register_type_parts(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id,
    uint32_t unit_count) {
  if (unit_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "register type unit count must be non-zero");
  }
  if (reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "target-low register class ID %" PRIu16
                            " is not present in the selected descriptor set",
                            reg_class_id);
  }
  return iree_ok_status();
}

iree_status_t loom_low_build_register_type(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t* out_type) {
  *out_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_validate_register_type_parts(
      descriptor_set, reg_class_id, unit_count));
  *out_type = loom_low_register_type(descriptor_set->stable_id, reg_class_id,
                                     unit_count);
  return iree_ok_status();
}

iree_status_t loom_low_build_typed_register_type(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    uint16_t reg_class_id, uint32_t unit_count, loom_type_t value_type,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_validate_register_type_parts(
      descriptor_set, reg_class_id, unit_count));
  return loom_module_intern_register_type(
      module, descriptor_set->stable_id,
      loom_low_register_type_pack_payload1(reg_class_id, unit_count),
      value_type, out_type);
}

iree_status_t loom_low_build_descriptor_implicit_resource_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_type_t* out_type) {
  *out_type = loom_type_none();
  const loom_low_operand_t* operand =
      loom_low_descriptor_implicit_resource_operand(descriptor_set, descriptor);
  IREE_ASSERT(operand != NULL,
              "low descriptor must provide an implicit resource operand");
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const loom_low_reg_class_alt_t* alternative =
        &descriptor_set
             ->reg_class_alts[operand->reg_class_alt_start + (uint32_t)i];
    if (iree_any_bit_set(alternative->flags,
                         LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    return loom_low_build_register_type(descriptor_set,
                                        alternative->reg_class_id,
                                        operand->unit_count, out_type);
  }
  IREE_ASSERT_UNREACHABLE(
      "low descriptor implicit resource has no register-class form");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_low_build_resolved_descriptor_br(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_block_t* dest,
    const loom_value_id_t* args, iree_host_size_t arg_count,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  if (arg_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low.br argument count exceeds uint16_t range");
  }
  IREE_ASSERT_EQ(descriptor->carrier, LOOM_LOW_DESCRIPTOR_CARRIER_BRANCH);
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set, descriptor);
  IREE_ASSERT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op_with_successors(
      builder, LOOM_OP_LOW_BR, (uint16_t)arg_count, /*result_count=*/0,
      /*successor_count=*/1, /*region_count=*/0, /*tied_result_count=*/0,
      /*attribute_count=*/1, location, out_op));
  loom_op_successors(*out_op)[0] = dest;
  if (arg_count != 0) {
    memcpy(loom_op_operands(*out_op), args,
           arg_count * sizeof(loom_value_id_t));
  }
  loom_op_attrs(*out_op)[loom_low_br_descriptor_ATTR_INDEX] =
      loom_attr_scoped_enum(descriptor_ordinal);
  (*out_op)->traits =
      loom_low_descriptor_effective_traits(descriptor_set, descriptor);
  return loom_builder_finalize_op(builder, *out_op);
}

iree_status_t loom_low_build_resolved_descriptor_switch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_value_id_t selector,
    loom_block_t* default_dest, loom_block_t* const* target_dests,
    iree_host_size_t target_count, loom_location_id_t location,
    loom_op_t** out_op) {
  *out_op = NULL;
  if (target_count == 0 || target_count >= UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low.switch target count must be in the range [1, 65534]");
  }
  IREE_ASSERT_EQ(descriptor->carrier, LOOM_LOW_DESCRIPTOR_CARRIER_SWITCH);
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set, descriptor);
  IREE_ASSERT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);

  IREE_RETURN_IF_ERROR(loom_builder_allocate_op_with_successors(
      builder, LOOM_OP_LOW_SWITCH, /*operand_count=*/1, /*result_count=*/0,
      (uint16_t)(target_count + 1), /*region_count=*/0,
      /*tied_result_count=*/0, /*attribute_count=*/1, location, out_op));
  loom_op_operands(*out_op)[0] = selector;
  loom_op_successors(*out_op)[0] = default_dest;
  memcpy(loom_op_successors(*out_op) + 1, target_dests,
         target_count * sizeof(*target_dests));
  loom_op_attrs(*out_op)[loom_low_switch_descriptor_ATTR_INDEX] =
      loom_attr_scoped_enum(descriptor_ordinal);
  (*out_op)->traits =
      loom_low_descriptor_effective_traits(descriptor_set, descriptor);
  return loom_builder_finalize_op(builder, *out_op);
}

iree_status_t loom_low_build_resolved_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    const loom_type_t* result_types, iree_host_size_t result_count,
    const loom_tied_result_t* tied_results, iree_host_size_t tied_result_count,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set, descriptor);
  IREE_ASSERT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  if (operand_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low.op operand count exceeds uint16_t range");
  }
  if (result_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low.op result count exceeds uint16_t range");
  }
  if (tied_result_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low.op tied result count exceeds uint16_t range");
  }

  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, LOOM_OP_LOW_OP, (uint16_t)operand_count, (uint16_t)result_count,
      /*region_count=*/0, (uint16_t)tied_result_count, /*attribute_count=*/3,
      location, out_op));
  (*out_op)->traits =
      loom_low_descriptor_effective_traits(descriptor_set, descriptor);
  if (operand_count > 0) {
    memcpy(loom_op_operands(*out_op), operands,
           operand_count * sizeof(loom_value_id_t));
  }
  loom_op_attrs(*out_op)[loom_low_op_descriptor_ATTR_INDEX] =
      loom_attr_scoped_enum(descriptor_ordinal);
  if (attrs.count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
        builder->module, attrs,
        &loom_op_attrs(*out_op)[loom_low_op_attrs_ATTR_INDEX]));
  }
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    loom_value_id_t result_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_builder_define_value(builder, result_types[i], &result_id));
    loom_op_results(*out_op)[i] = result_id;
  }
  if (tied_result_count > 0) {
    memcpy(loom_op_tied_results(*out_op), tied_results,
           tied_result_count * sizeof(loom_tied_result_t));
  }
  return loom_builder_finalize_op(builder, *out_op);
}

iree_status_t loom_low_build_resolved_descriptor_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_named_attr_slice_t attrs,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set, descriptor);
  IREE_ASSERT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);

  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, LOOM_OP_LOW_CONST, /*operand_count=*/0, /*result_count=*/1,
      /*region_count=*/0, /*tied_result_count=*/0, /*attribute_count=*/2,
      location, out_op));
  (*out_op)->traits =
      loom_low_descriptor_effective_traits(descriptor_set, descriptor);
  loom_op_attrs(*out_op)[loom_low_const_descriptor_ATTR_INDEX] =
      loom_attr_scoped_enum(descriptor_ordinal);
  if (attrs.count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
        builder->module, attrs,
        &loom_op_attrs(*out_op)[loom_low_const_attrs_ATTR_INDEX]));
  }
  loom_value_id_t result_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_define_value(builder, result_type, &result_id));
  loom_op_results(*out_op)[0] = result_id;
  return loom_builder_finalize_op(builder, *out_op);
}
