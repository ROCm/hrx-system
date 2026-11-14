// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_abi.h"

#include <inttypes.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/bda_abi.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/arch/spirv/scalar_types.h"
#include "loom/target/artifact_diagnostics.h"
#include "loom/target/emit/spirv/binary_format.h"
#include "loom/target/emit/spirv/module_instructions.h"
#include "loom/target/registers.h"

typedef struct loom_spirv_module_abi_slot_type_info_t {
  // SPIR-V type ID used by packet emission after ABI materialization.
  uint32_t value_type_id;
  // SPIR-V type ID stored in the descriptor-backed slot field.
  uint32_t field_type_id;
  // StorageBuffer pointer-to-descriptor-block type ID for OpVariable.
  uint32_t variable_type_id;
  // StorageBuffer pointer-to-field type ID for OpAccessChain results.
  uint32_t field_pointer_type_id;
  // Natural byte alignment attached to loads/stores for this slot field.
  uint32_t field_alignment;
} loom_spirv_module_abi_slot_type_info_t;

static loom_spirv_binary_writer_t* loom_spirv_module_abi_section(
    loom_spirv_module_abi_context_t* context,
    loom_spirv_module_section_t section) {
  return loom_spirv_module_builder_section(context->builder, section);
}

static uint32_t loom_spirv_module_abi_allocate_id(
    loom_spirv_module_abi_context_t* context) {
  return loom_spirv_module_builder_allocate_id(context->builder);
}

static iree_string_view_t loom_spirv_module_abi_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, string_id);
}

static iree_status_t loom_spirv_module_abi_op_name(
    loom_spirv_module_abi_context_t* context, const loom_op_t* source_op,
    uint32_t id, iree_string_view_t name) {
  if (id == 0 || iree_string_view_is_empty(name)) {
    return iree_ok_status();
  }
  const uint32_t prefix_operands[] = {id};
  return loom_spirv_module_builder_emit_string(
      context->builder, LOOM_SPIRV_MODULE_SECTION_DEBUG, LOOM_SPIRV_OP_NAME,
      prefix_operands, IREE_ARRAYSIZE(prefix_operands), name, NULL, 0,
      context->diagnostic_emitter, source_op, context->accepted);
}

static iree_status_t loom_spirv_module_abi_value_name(
    loom_spirv_module_abi_context_t* context, loom_value_id_t value_id,
    uint32_t id) {
  const loom_value_t* value = loom_module_value(context->module, value_id);
  return loom_spirv_module_abi_op_name(
      context,
      loom_value_is_block_arg(value)
          ? loom_value_def_block(value)->parent_region->parent_op
          : loom_value_def_op(value),
      id,
      loom_spirv_module_abi_string_or_empty(context->module, value->name_id));
}

static iree_status_t loom_spirv_module_abi_define_value(
    loom_spirv_module_abi_context_t* context, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t value_ref, bool emit_name) {
  loom_spirv_module_value_table_define(context->value_table, value_id,
                                       value_ref);
  if (emit_name) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_module_abi_value_name(context, value_id, value_ref.id));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_abi_lookup_value(
    loom_spirv_module_abi_context_t* context, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t* out_value_ref) {
  *out_value_ref =
      loom_spirv_module_value_table_lookup(context->value_table, value_id);
  return iree_ok_status();
}

static bool loom_spirv_module_abi_uses_raw_bda(
    const loom_spirv_function_plan_t* function_plan) {
  return loom_spirv_function_plan_target_bundle(function_plan)
             ->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL;
}

static bool loom_spirv_module_abi_type_is_named_opaque(
    const loom_spirv_module_abi_context_t* context, loom_type_t type,
    iree_string_view_t expected_name) {
  if (loom_type_kind(type) != LOOM_TYPE_DIALECT ||
      loom_type_dialect_param_count(type) != 0) {
    return false;
  }
  const iree_string_view_t actual_name = loom_spirv_module_abi_string_or_empty(
      context->module, loom_type_dialect_name_id(type));
  return iree_string_view_equal(actual_name, expected_name);
}

static loom_type_t loom_spirv_module_abi_type_attr(
    const loom_spirv_module_abi_context_t* context, loom_type_id_t type_id) {
  if (type_id >= context->module->types.count) {
    return loom_type_none();
  }
  return loom_type_table_get(&context->module->types, type_id);
}

static loom_spirv_value_type_t loom_spirv_module_abi_low_register_value_type(
    loom_type_t type) {
  IREE_ASSERT(loom_low_type_is_register(type));
  IREE_ASSERT_EQ(loom_low_register_type_descriptor_set_stable_id(type),
                 SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID);

  loom_spirv_value_type_t value_type = {0};
  const bool resolved =
      loom_spirv_value_type_from_low_register_type(type, &value_type);
  IREE_ASSERT(resolved);
  return value_type;
}

static loom_spirv_value_type_t loom_spirv_module_abi_low_resource_value_type(
    loom_spirv_module_abi_context_t* context, const loom_op_t* op) {
  loom_type_t result_type =
      loom_module_value_type(context->module, loom_low_resource_result(op));
  return loom_spirv_module_abi_low_register_value_type(result_type);
}

static uint16_t loom_spirv_module_abi_bda_resource_binding_ordinal(
    loom_spirv_module_abi_context_t* context, const loom_op_t* op) {
  IREE_ASSERT_EQ(loom_low_resource_import_kind(op),
                 LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING);

  const int64_t index = loom_low_resource_index(op);
  IREE_ASSERT_GE(index, 0);
  IREE_ASSERT_LT(index, UINT16_MAX);

  const loom_type_t source_type = loom_spirv_module_abi_type_attr(
      context, loom_low_resource_source_type(op));
  IREE_ASSERT(loom_spirv_module_abi_type_is_named_opaque(
      context, source_type, IREE_SV("hal.buffer")));

  const loom_spirv_value_type_t value_type =
      loom_spirv_module_abi_low_resource_value_type(context, op);
  IREE_ASSERT_EQ(value_type.value_class,
                 LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS);

  return (uint16_t)index;
}

static iree_status_t loom_spirv_module_abi_module_processed(
    loom_spirv_module_abi_context_t* context, iree_string_view_t value) {
  return loom_spirv_module_builder_emit_string(
      context->builder, LOOM_SPIRV_MODULE_SECTION_MODULE_PROCESSED,
      LOOM_SPIRV_OP_MODULE_PROCESSED, NULL, 0, value, NULL, 0,
      context->diagnostic_emitter, context->function_op, context->accepted);
}

static iree_status_t loom_spirv_module_abi_bda_metadata_field(
    loom_spirv_module_abi_context_t* context, iree_string_builder_t* text,
    iree_string_view_t entry_name, const char* field_prefix, uint32_t value) {
  iree_string_builder_reset(text);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      text, "iree.vulkan.bda.v1[%.*s].%s%" PRIu32, (int)entry_name.size,
      entry_name.data, field_prefix, value));
  return loom_spirv_module_abi_module_processed(context,
                                                iree_string_builder_view(text));
}

static uint8_t loom_spirv_module_abi_logical_parameter_byte_length(
    loom_type_t logical_type, const loom_spirv_module_abi_slot_t* slot) {
  IREE_ASSERT(loom_type_is_scalar(logical_type));
  const loom_scalar_type_t element_type = loom_type_element_type(logical_type);
  if (element_type == LOOM_SCALAR_TYPE_INDEX ||
      element_type == LOOM_SCALAR_TYPE_OFFSET) {
    return (uint8_t)(slot->constant_word_count * sizeof(uint32_t));
  }
  const int32_t bit_width = loom_scalar_type_bitwidth(element_type);
  IREE_ASSERT_GT(bit_width, 0);
  const uint8_t byte_length = (uint8_t)iree_max(1, bit_width / 8);
  IREE_ASSERT_LE(byte_length, slot->constant_word_count * sizeof(uint32_t));
  return byte_length;
}

static iree_status_t loom_spirv_module_abi_emit_bda_parameters(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan, iree_string_view_t entry_name,
    iree_string_builder_t* text) {
  const loom_func_type_data_t* signature =
      context->function_plan->logical_signature;
  if (signature == NULL) {
    return iree_ok_status();
  }

  iree_string_builder_reset(text);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      text, "iree.vulkan.bda.parameters.v1[%.*s]=", (int)entry_name.size,
      entry_name.data));
  uint16_t binding_ordinal = 0;
  iree_host_size_t direct_argument_ordinal = 0;
  for (uint16_t i = 0; i < signature->arg_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(text, ";"));
    }
    if (loom_type_is_buffer(signature->types[i])) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          text, "b%" PRIu16, binding_ordinal++));
      continue;
    }
    IREE_ASSERT_LT(direct_argument_ordinal, plan->arg_count);
    const loom_spirv_module_abi_slot_t* slot =
        &plan->args[direct_argument_ordinal++];
    IREE_ASSERT_LE(slot->constant_word_offset,
                   LOOM_SPIRV_BDA_MAX_REFLECTED_CONSTANT_WORD_OFFSET);
    const uint8_t byte_length =
        loom_spirv_module_abi_logical_parameter_byte_length(signature->types[i],
                                                            slot);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        text, "c%" PRIu16 ",%" PRIu8,
        (uint16_t)(slot->constant_word_offset * sizeof(uint32_t)),
        byte_length));
  }
  IREE_ASSERT_EQ(binding_ordinal, plan->bda_binding_count);
  IREE_ASSERT_EQ(direct_argument_ordinal, plan->arg_count);
  return loom_spirv_module_abi_module_processed(context,
                                                iree_string_builder_view(text));
}

static void loom_spirv_module_abi_share_bda_root(
    loom_spirv_module_abi_context_t* context,
    loom_spirv_module_abi_plan_t* plan) {
  loom_spirv_module_shared_bda_root_t* root = context->shared_bda_root;
  if (root->root_variable_id == 0) {
    root->root_variable_id = loom_spirv_module_abi_allocate_id(context);
  }
  root->constant_word_count =
      iree_max(root->constant_word_count, plan->bda_constant_word_count);
  plan->bda_root.variable_id = root->root_variable_id;
}

static iree_status_t loom_spirv_module_abi_emit_scalar_value_type(
    loom_spirv_module_abi_context_t* context,
    loom_spirv_scalar_type_t scalar_type,
    const loom_spirv_scalar_type_descriptor_t** out_descriptor,
    uint32_t* out_type_id) {
  const loom_spirv_scalar_type_descriptor_t* descriptor =
      loom_spirv_scalar_type_descriptor(scalar_type);
  IREE_ASSERT(descriptor != NULL);
  loom_spirv_module_builder_require_feature_bits(
      context->builder, descriptor->required_feature_bits);
  *out_descriptor = descriptor;
  return loom_spirv_emit_type_scalar(context->type_context, scalar_type,
                                     out_type_id);
}

static iree_status_t loom_spirv_module_abi_slot_type_info(
    loom_spirv_module_abi_context_t* context,
    loom_spirv_value_type_t value_type,
    loom_spirv_module_abi_slot_type_info_t* out_type_info) {
  *out_type_info = (loom_spirv_module_abi_slot_type_info_t){0};
  switch (value_type.value_class) {
    case LOOM_SPIRV_VALUE_CLASS_SCALAR: {
      const loom_spirv_scalar_type_descriptor_t* descriptor = NULL;
      IREE_RETURN_IF_ERROR(loom_spirv_module_abi_emit_scalar_value_type(
          context, value_type.scalar_type, &descriptor,
          &out_type_info->value_type_id));
      out_type_info->field_type_id = out_type_info->value_type_id;
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_struct(
              context->type_context, out_type_info->field_type_id,
              &out_type_info->variable_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_field(
              context->type_context, out_type_info->field_type_id,
              &out_type_info->field_pointer_type_id));
      out_type_info->field_alignment = iree_max(1u, descriptor->bit_width / 8);
      return iree_ok_status();
    }
    case LOOM_SPIRV_VALUE_CLASS_BOOL: {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_type_bool(
          context->type_context, &out_type_info->value_type_id));
      IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(
          context->type_context, &out_type_info->field_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_struct(
              context->type_context, out_type_info->field_type_id,
              &out_type_info->variable_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_field(
              context->type_context, out_type_info->field_type_id,
              &out_type_info->field_pointer_type_id));
      out_type_info->field_alignment = 4;
      return iree_ok_status();
    }
    case LOOM_SPIRV_VALUE_CLASS_OFFSET64:
    case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS: {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(
          context->type_context, &out_type_info->value_type_id));
      out_type_info->field_type_id = out_type_info->value_type_id;
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_struct(
              context->type_context, out_type_info->field_type_id,
              &out_type_info->variable_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_field(
              context->type_context, out_type_info->field_type_id,
              &out_type_info->field_pointer_type_id));
      out_type_info->field_alignment = 8;
      return iree_ok_status();
    }
    case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP:
    case LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION:
    case LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION_ARRAY:
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY:
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
    case LOOM_SPIRV_VALUE_CLASS_VECTOR:
    case LOOM_SPIRV_VALUE_CLASS_BOOL_VECTOR:
    case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
      break;
  }
  IREE_CHECK_UNREACHABLE("verified SPIR-V shader-entry ABI value class");
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_abi_declare_slot_variable(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_slot_t* slot) {
  loom_spirv_module_abi_slot_type_info_t type_info = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_slot_type_info(
      context, slot->value_type, &type_info));

  const uint32_t descriptor_set_operands[] = {
      slot->variable_id,
      LOOM_SPIRV_DECORATION_DESCRIPTOR_SET,
      0,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_abi_section(context,
                                    LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
      LOOM_SPIRV_OP_DECORATE, descriptor_set_operands,
      IREE_ARRAYSIZE(descriptor_set_operands)));
  const uint32_t binding_operands[] = {
      slot->variable_id,
      LOOM_SPIRV_DECORATION_BINDING,
      slot->binding,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_abi_section(context,
                                    LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
      LOOM_SPIRV_OP_DECORATE, binding_operands,
      IREE_ARRAYSIZE(binding_operands)));

  const uint32_t variable_operands[] = {
      type_info.variable_type_id,
      slot->variable_id,
      LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_module_abi_section(context,
                                    LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_VARIABLE, variable_operands,
      IREE_ARRAYSIZE(variable_operands));
}

static uint8_t loom_spirv_module_abi_slot_constant_word_count(
    loom_spirv_value_type_t value_type) {
  switch (value_type.value_class) {
    case LOOM_SPIRV_VALUE_CLASS_BOOL:
      return 1;
    case LOOM_SPIRV_VALUE_CLASS_SCALAR: {
      const loom_spirv_scalar_type_descriptor_t* descriptor =
          loom_spirv_scalar_type_descriptor(value_type.scalar_type);
      IREE_ASSERT(descriptor != NULL);
      if (descriptor->bit_width <= 32) {
        return 1;
      }
      if (descriptor->bit_width == 64) {
        return 2;
      }
      break;
    }
    case LOOM_SPIRV_VALUE_CLASS_OFFSET64:
      return 2;
    case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS:
    case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP:
    case LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION:
    case LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION_ARRAY:
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY:
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
    case LOOM_SPIRV_VALUE_CLASS_VECTOR:
    case LOOM_SPIRV_VALUE_CLASS_BOOL_VECTOR:
    case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
      break;
  }
  IREE_CHECK_UNREACHABLE("verified SPIR-V raw-BDA direct ABI value class");
  return 0;
}

static iree_status_t loom_spirv_module_abi_build_shader_entry_plan(
    loom_spirv_module_abi_context_t* context, const loom_block_t* entry_block,
    loom_spirv_module_abi_plan_t* plan) {
  const iree_host_size_t arg_count = entry_block->arg_count;
  const iree_host_size_t result_count = context->function_op->result_count;
  const iree_host_size_t descriptor_slot_count = arg_count + result_count;

  plan->kind = LOOM_SPIRV_MODULE_ABI_PLAN_SHADER_ENTRY;
  plan->arg_count = arg_count;
  plan->result_count = result_count;
  if (arg_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(context->scratch_arena, arg_count,
                                  sizeof(*plan->args), (void**)&plan->args));
  }
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->scratch_arena, result_count, sizeof(*plan->results),
        (void**)&plan->results));
  }
  if (descriptor_slot_count != 0) {
    uint32_t zero_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_i32_constant(context->type_context, 0, &zero_id));
  }

  uint32_t binding = 0;
  for (iree_host_size_t i = 0; i < arg_count; ++i) {
    loom_spirv_module_abi_slot_t* slot = &plan->args[i];
    slot->value_id = loom_block_arg_id(entry_block, (uint16_t)i);
    slot->binding = binding++;
    slot->variable_id = loom_spirv_module_abi_allocate_id(context);
    slot->value_type = loom_spirv_module_abi_low_register_value_type(
        loom_module_value_type(context->module, slot->value_id));
    IREE_RETURN_IF_ERROR(
        loom_spirv_module_abi_declare_slot_variable(context, slot));
  }
  const loom_value_id_t* results = loom_op_const_results(context->function_op);
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    loom_spirv_module_abi_slot_t* slot = &plan->results[i];
    slot->value_id = results[i];
    slot->binding = binding++;
    slot->variable_id = loom_spirv_module_abi_allocate_id(context);
    slot->value_type = loom_spirv_module_abi_low_register_value_type(
        loom_module_value_type(context->module, slot->value_id));
    IREE_ASSERT(slot->value_type.value_class == LOOM_SPIRV_VALUE_CLASS_SCALAR ||
                slot->value_type.value_class == LOOM_SPIRV_VALUE_CLASS_BOOL ||
                slot->value_type.value_class ==
                    LOOM_SPIRV_VALUE_CLASS_OFFSET64);
    IREE_RETURN_IF_ERROR(
        loom_spirv_module_abi_declare_slot_variable(context, slot));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_abi_build_raw_bda_plan(
    loom_spirv_module_abi_context_t* context, const loom_block_t* entry_block,
    loom_spirv_module_abi_plan_t* plan) {
  IREE_ASSERT_EQ(context->function_op->result_count, 0);

  if (entry_block->arg_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->scratch_arena, entry_block->arg_count, sizeof(*plan->args),
        (void**)&plan->args));
  }
  uint16_t constant_word_count = 0;
  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    loom_spirv_module_abi_slot_t* slot = &plan->args[i];
    slot->value_id = loom_block_arg_id(entry_block, i);
    slot->value_type = loom_spirv_module_abi_low_register_value_type(
        loom_module_value_type(context->module, slot->value_id));
    switch (slot->value_type.value_class) {
      case LOOM_SPIRV_VALUE_CLASS_SCALAR:
      case LOOM_SPIRV_VALUE_CLASS_BOOL:
      case LOOM_SPIRV_VALUE_CLASS_OFFSET64: {
        slot->constant_word_count =
            loom_spirv_module_abi_slot_constant_word_count(slot->value_type);
        break;
      }
      case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS:
      case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
      case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP:
      case LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION:
      case LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION_ARRAY:
      case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY:
      case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
      case LOOM_SPIRV_VALUE_CLASS_VECTOR:
      case LOOM_SPIRV_VALUE_CLASS_BOOL_VECTOR:
      case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
        IREE_CHECK_UNREACHABLE("verified SPIR-V raw-BDA direct ABI value");
        break;
    }
    IREE_ASSERT_LE(constant_word_count,
                   (uint16_t)(UINT16_MAX - slot->constant_word_count));
    slot->constant_word_offset = constant_word_count;
    constant_word_count =
        (uint16_t)(constant_word_count + slot->constant_word_count);
  }

  const loom_func_type_data_t* signature =
      context->function_plan->logical_signature;
  uint16_t binding_count = 0;
  if (signature != NULL) {
    for (uint16_t i = 0; i < signature->arg_count; ++i) {
      binding_count += loom_type_is_buffer(signature->types[i]);
    }
  } else {
    const loom_op_t* op = entry_block->first_op;
    while (op != NULL &&
           (loom_low_resource_isa(op) || loom_low_live_in_isa(op))) {
      if (loom_low_resource_isa(op)) {
        const uint16_t binding_ordinal =
            loom_spirv_module_abi_bda_resource_binding_ordinal(context, op);
        binding_count =
            iree_max(binding_count, (uint16_t)(binding_ordinal + 1));
      }
      op = op->next_op;
    }
  }

  plan->kind = LOOM_SPIRV_MODULE_ABI_PLAN_HAL_KERNEL_RAW_BDA;
  plan->arg_count = entry_block->arg_count;
  plan->bda_binding_count = binding_count;
  plan->bda_constant_word_count = constant_word_count;
  loom_spirv_module_abi_share_bda_root(context, plan);
  return iree_ok_status();
}

iree_status_t loom_spirv_module_abi_build_plan(
    loom_spirv_module_abi_context_t* context, const loom_block_t* entry_block,
    loom_spirv_module_abi_plan_t* plan) {
  if (loom_spirv_module_abi_uses_raw_bda(context->function_plan)) {
    return loom_spirv_module_abi_build_raw_bda_plan(context, entry_block, plan);
  }
  return loom_spirv_module_abi_build_shader_entry_plan(context, entry_block,
                                                       plan);
}

static iree_status_t loom_spirv_module_abi_slot_field_pointer(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_slot_t* slot, uint32_t field_pointer_type_id,
    uint32_t* out_field_pointer_id) {
  uint32_t zero_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_i32_constant(context->type_context, 0, &zero_id));
  const uint32_t result_id = loom_spirv_module_abi_allocate_id(context);
  const uint32_t operands[] = {
      field_pointer_type_id,
      result_id,
      slot->variable_id,
      zero_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_abi_section(context,
                                    LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_ACCESS_CHAIN, operands, IREE_ARRAYSIZE(operands)));
  *out_field_pointer_id = result_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_abi_load_slot_field(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_slot_t* slot,
    const loom_spirv_module_abi_slot_type_info_t* type_info,
    loom_spirv_module_value_ref_t* out_field_ref) {
  uint32_t field_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_slot_field_pointer(
      context, slot, type_info->field_pointer_type_id, &field_pointer_id));
  const uint32_t result_id = loom_spirv_module_abi_allocate_id(context);
  const uint32_t operands[] = {
      type_info->field_type_id,   result_id,
      field_pointer_id,           LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      type_info->field_alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_abi_section(context,
                                    LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LOAD, operands, IREE_ARRAYSIZE(operands)));
  *out_field_ref = (loom_spirv_module_value_ref_t){
      .id = result_id,
      .type_id = type_info->field_type_id,
      .value_type = {0},
  };
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_abi_materialize_shader_entry_arg(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_slot_t* slot) {
  loom_spirv_module_abi_slot_type_info_t type_info = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_slot_type_info(
      context, slot->value_type, &type_info));
  loom_spirv_module_value_ref_t field_ref = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_load_slot_field(
      context, slot, &type_info, &field_ref));
  if (slot->value_type.value_class == LOOM_SPIRV_VALUE_CLASS_BOOL) {
    uint32_t zero_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_i32_constant(context->type_context, 0, &zero_id));
    const uint32_t result_id = loom_spirv_module_abi_allocate_id(context);
    const uint32_t operands[] = {
        type_info.value_type_id,
        result_id,
        field_ref.id,
        zero_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_module_abi_section(context,
                                      LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_I_NOT_EQUAL, operands, IREE_ARRAYSIZE(operands)));
    return loom_spirv_module_abi_define_value(
        context, slot->value_id,
        (loom_spirv_module_value_ref_t){
            .id = result_id,
            .type_id = type_info.value_type_id,
            .value_type = slot->value_type,
        },
        true);
  }
  field_ref.value_type = slot->value_type;
  return loom_spirv_module_abi_define_value(context, slot->value_id, field_ref,
                                            true);
}

static iree_status_t loom_spirv_module_abi_materialize_shader_entry_args(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan) {
  for (iree_host_size_t i = 0; i < plan->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_module_abi_materialize_shader_entry_arg(
        context, &plan->args[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_abi_materialize_bda_constant_word(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan, uint16_t word_offset,
    uint32_t* out_word_id) {
  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(context->type_context, &u32_type_id));
  uint32_t u32_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_push_constant_u32(
      context->type_context, &u32_pointer_type_id));
  uint32_t constants_member_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_constant(
      context->type_context, LOOM_SPIRV_BDA_ROOT_CONSTANT_MEMBER_INDEX,
      &constants_member_id));
  uint32_t word_index_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_constant(
      context->type_context, word_offset, &word_index_id));
  const uint32_t index_ids[] = {
      constants_member_id,
      word_index_id,
  };
  uint32_t word_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_access_chain(
      context->builder, u32_pointer_type_id, plan->bda_root.variable_id,
      index_ids, IREE_ARRAYSIZE(index_ids), &word_pointer_id));
  return loom_spirv_module_emit_load_aligned(context->builder, u32_type_id,
                                             word_pointer_id, 4, out_word_id);
}

static iree_status_t loom_spirv_module_abi_materialize_bda_u64_bits(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan,
    const loom_spirv_module_abi_slot_t* slot, uint32_t* out_u64_id) {
  uint32_t low_word_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_materialize_bda_constant_word(
      context, plan, slot->constant_word_offset, &low_word_id));
  uint32_t high_word_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_materialize_bda_constant_word(
      context, plan, (uint16_t)(slot->constant_word_offset + 1),
      &high_word_id));

  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u64(context->type_context, &u64_type_id));
  uint32_t low_u64_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      context->builder, LOOM_SPIRV_OP_U_CONVERT, u64_type_id, low_word_id,
      &low_u64_id));
  uint32_t high_u64_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      context->builder, LOOM_SPIRV_OP_U_CONVERT, u64_type_id, high_word_id,
      &high_u64_id));
  uint32_t shift_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_u64_constant(context->type_context, 32, &shift_id));
  uint32_t shifted_high_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
      context->builder, LOOM_SPIRV_OP_SHIFT_LEFT_LOGICAL, u64_type_id,
      high_u64_id, shift_id, &shifted_high_id));
  return loom_spirv_module_emit_binary_result(
      context->builder, LOOM_SPIRV_OP_BITWISE_OR, u64_type_id, shifted_high_id,
      low_u64_id, out_u64_id);
}

static iree_status_t loom_spirv_module_abi_materialize_bda_scalar_arg(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan,
    const loom_spirv_module_abi_slot_t* slot) {
  const loom_spirv_scalar_type_descriptor_t* descriptor = NULL;
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_emit_scalar_value_type(
      context, slot->value_type.scalar_type, &descriptor, &result_type_id));

  uint32_t result_id = 0;
  if (descriptor->bit_width < 32) {
    uint32_t word_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_module_abi_materialize_bda_constant_word(
        context, plan, slot->constant_word_offset, &word_id));
    // The low bits of each ABI word hold the scalar payload. Bitcasting the
    // whole word maps those bits to component zero without requiring a narrow
    // integer capability for floating-point arguments.
    uint32_t vector_type_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(
        context->type_context, result_type_id, 32 / descriptor->bit_width,
        &vector_type_id));
    uint32_t vector_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
        context->builder, LOOM_SPIRV_OP_BITCAST, vector_type_id, word_id,
        &vector_id));
    result_id = loom_spirv_module_abi_allocate_id(context);
    const uint32_t operands[] = {result_type_id, result_id, vector_id, 0};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_module_abi_section(context,
                                      LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_COMPOSITE_EXTRACT, operands, IREE_ARRAYSIZE(operands)));
  } else {
    uint32_t bits_type_id = 0;
    uint32_t bits_id = 0;
    if (descriptor->bit_width == 32) {
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_u32(context->type_context, &bits_type_id));
      IREE_RETURN_IF_ERROR(loom_spirv_module_abi_materialize_bda_constant_word(
          context, plan, slot->constant_word_offset, &bits_id));
    } else if (descriptor->bit_width == 64) {
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_u64(context->type_context, &bits_type_id));
      IREE_RETURN_IF_ERROR(loom_spirv_module_abi_materialize_bda_u64_bits(
          context, plan, slot, &bits_id));
    } else {
      IREE_CHECK_UNREACHABLE("verified SPIR-V raw-BDA scalar bit width");
    }
    IREE_RETURN_IF_ERROR(loom_spirv_module_emit_bitcast_if_needed(
        context->builder, result_type_id, bits_type_id, bits_id, &result_id));
  }
  return loom_spirv_module_abi_define_value(context, slot->value_id,
                                            (loom_spirv_module_value_ref_t){
                                                .id = result_id,
                                                .type_id = result_type_id,
                                                .value_type = slot->value_type,
                                            },
                                            true);
}

static iree_status_t loom_spirv_module_abi_materialize_bda_bool_arg(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan,
    const loom_spirv_module_abi_slot_t* slot) {
  uint32_t word_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_materialize_bda_constant_word(
      context, plan, slot->constant_word_offset, &word_id));
  uint32_t zero_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_i32_constant(context->type_context, 0, &zero_id));
  uint32_t bool_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_bool(context->type_context, &bool_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
      context->builder, LOOM_SPIRV_OP_I_NOT_EQUAL, bool_type_id, word_id,
      zero_id, &result_id));
  return loom_spirv_module_abi_define_value(context, slot->value_id,
                                            (loom_spirv_module_value_ref_t){
                                                .id = result_id,
                                                .type_id = bool_type_id,
                                                .value_type = slot->value_type,
                                            },
                                            true);
}

static iree_status_t loom_spirv_module_abi_materialize_bda_u64_arg(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan,
    const loom_spirv_module_abi_slot_t* slot) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u64(context->type_context, &u64_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_materialize_bda_u64_bits(
      context, plan, slot, &result_id));
  return loom_spirv_module_abi_define_value(
      context, slot->value_id,
      (loom_spirv_module_value_ref_t){
          .id = result_id,
          .type_id = u64_type_id,
          .value_type =
              {
                  .value_class = LOOM_SPIRV_VALUE_CLASS_OFFSET64,
                  .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U64,
              },
      },
      true);
}

static iree_status_t loom_spirv_module_abi_materialize_bda_arg(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan,
    const loom_spirv_module_abi_slot_t* slot) {
  switch (slot->value_type.value_class) {
    case LOOM_SPIRV_VALUE_CLASS_SCALAR:
      return loom_spirv_module_abi_materialize_bda_scalar_arg(context, plan,
                                                              slot);
    case LOOM_SPIRV_VALUE_CLASS_BOOL:
      return loom_spirv_module_abi_materialize_bda_bool_arg(context, plan,
                                                            slot);
    case LOOM_SPIRV_VALUE_CLASS_OFFSET64:
      return loom_spirv_module_abi_materialize_bda_u64_arg(context, plan, slot);
    case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS:
    case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP:
    case LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION:
    case LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION_ARRAY:
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY:
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
    case LOOM_SPIRV_VALUE_CLASS_VECTOR:
    case LOOM_SPIRV_VALUE_CLASS_BOOL_VECTOR:
    case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
      break;
  }
  IREE_CHECK_UNREACHABLE("verified SPIR-V raw-BDA direct ABI value class");
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_abi_materialize_bda_args(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan) {
  for (iree_host_size_t i = 0; i < plan->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_module_abi_materialize_bda_arg(
        context, plan, &plan->args[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_module_abi_materialize_entry_args(
    loom_spirv_module_abi_context_t* context,
    loom_spirv_module_abi_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_SPIRV_MODULE_ABI_PLAN_SHADER_ENTRY:
      return loom_spirv_module_abi_materialize_shader_entry_args(context, plan);
    case LOOM_SPIRV_MODULE_ABI_PLAN_HAL_KERNEL_RAW_BDA:
      return loom_spirv_module_abi_materialize_bda_args(context, plan);
  }
  IREE_CHECK_UNREACHABLE("known SPIR-V module ABI plan kind");
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_abi_materialize_bda_root(
    loom_spirv_module_abi_context_t* context,
    loom_spirv_module_abi_plan_t* plan) {
  if (plan->bda_root.binding_table_pointer_id != 0) {
    return iree_ok_status();
  }

  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u64(context->type_context, &u64_type_id));
  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(context->type_context, &u32_type_id));
  uint32_t root_u64_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_push_constant_u64(
      context->type_context, &root_u64_pointer_type_id));
  uint32_t root_u32_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_push_constant_u32(
      context->type_context, &root_u32_pointer_type_id));
  uint32_t address_table_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_ptr_physical_storage_buffer_bda_address_table(
          context->type_context, &address_table_pointer_type_id));

  uint32_t binding_table_address_index_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_constant(
      context->type_context, 0, &binding_table_address_index_id));
  uint32_t binding_base_index_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_constant(context->type_context, 2,
                                                    &binding_base_index_id));

  uint32_t binding_table_address_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_access_chain(
      context->builder, root_u64_pointer_type_id, plan->bda_root.variable_id,
      &binding_table_address_index_id, 1, &binding_table_address_pointer_id));
  uint32_t binding_table_address_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_load_aligned(
      context->builder, u64_type_id, binding_table_address_pointer_id, 8,
      &binding_table_address_id));

  const uint32_t convert_operands[] = {
      address_table_pointer_type_id,
      loom_spirv_module_abi_allocate_id(context),
      binding_table_address_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_abi_section(context,
                                    LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_CONVERT_U_TO_PTR, convert_operands,
      IREE_ARRAYSIZE(convert_operands)));

  uint32_t binding_base_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_access_chain(
      context->builder, root_u32_pointer_type_id, plan->bda_root.variable_id,
      &binding_base_index_id, 1, &binding_base_pointer_id));
  uint32_t binding_base_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_load_aligned(
      context->builder, u32_type_id, binding_base_pointer_id, 4,
      &binding_base_id));

  plan->bda_root.binding_table_pointer_id = convert_operands[1];
  plan->bda_root.binding_base_id = binding_base_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_abi_bda_binding_index(
    loom_spirv_module_abi_context_t* context,
    loom_spirv_module_abi_plan_t* plan, uint16_t binding_ordinal,
    uint32_t* out_binding_index_id) {
  IREE_RETURN_IF_ERROR(
      loom_spirv_module_abi_materialize_bda_root(context, plan));
  if (binding_ordinal == 0) {
    *out_binding_index_id = plan->bda_root.binding_base_id;
    return iree_ok_status();
  }

  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(context->type_context, &u32_type_id));
  uint32_t binding_ordinal_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      context->type_context, binding_ordinal, &binding_ordinal_id));
  const uint32_t result_id = loom_spirv_module_abi_allocate_id(context);
  const uint32_t operands[] = {
      u32_type_id,
      result_id,
      plan->bda_root.binding_base_id,
      binding_ordinal_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_abi_section(context,
                                    LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_I_ADD, operands, IREE_ARRAYSIZE(operands)));
  *out_binding_index_id = result_id;
  return iree_ok_status();
}

iree_status_t loom_spirv_module_abi_materialize_resource(
    loom_spirv_module_abi_context_t* context,
    loom_spirv_module_abi_plan_t* plan, const loom_op_t* op) {
  const uint16_t binding_ordinal =
      loom_spirv_module_abi_bda_resource_binding_ordinal(context, op);

  uint32_t binding_index_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_bda_binding_index(
      context, plan, binding_ordinal, &binding_index_id));

  uint32_t zero_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_i32_constant(context->type_context, 0, &zero_id));
  uint32_t address_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_physical_storage_buffer_u64(
      context->type_context, &address_pointer_type_id));
  const uint32_t index_ids[] = {zero_id, binding_index_id};
  uint32_t address_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_access_chain(
      context->builder, address_pointer_type_id,
      plan->bda_root.binding_table_pointer_id, index_ids,
      IREE_ARRAYSIZE(index_ids), &address_pointer_id));

  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u64(context->type_context, &u64_type_id));
  uint32_t address_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_load_aligned(
      context->builder, u64_type_id, address_pointer_id, 8, &address_id));

  return loom_spirv_module_abi_define_value(
      context, loom_low_resource_result(op),
      (loom_spirv_module_value_ref_t){
          .id = address_id,
          .type_id = u64_type_id,
          .value_type =
              {
                  .value_class = LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS,
              },
      },
      true);
}

static iree_status_t loom_spirv_module_abi_store_slot_value(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_slot_t* slot,
    loom_spirv_module_value_ref_t value) {
  loom_spirv_module_abi_slot_type_info_t type_info = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_slot_type_info(
      context, slot->value_type, &type_info));
  IREE_ASSERT(loom_spirv_value_type_equal(value.value_type, slot->value_type));
  IREE_ASSERT_EQ(value.type_id, type_info.value_type_id);
  uint32_t store_value_id = value.id;
  if (slot->value_type.value_class == LOOM_SPIRV_VALUE_CLASS_BOOL) {
    uint32_t one_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_i32_constant(context->type_context, 1, &one_id));
    uint32_t zero_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_i32_constant(context->type_context, 0, &zero_id));
    store_value_id = loom_spirv_module_abi_allocate_id(context);
    const uint32_t select_operands[] = {
        type_info.field_type_id, store_value_id, value.id, one_id, zero_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_module_abi_section(context,
                                      LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_SELECT, select_operands,
        IREE_ARRAYSIZE(select_operands)));
  }
  uint32_t field_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_slot_field_pointer(
      context, slot, type_info.field_pointer_type_id, &field_pointer_id));
  const uint32_t operands[] = {
      field_pointer_id,
      store_value_id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      type_info.field_alignment,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_module_abi_section(context,
                                    LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_STORE, operands, IREE_ARRAYSIZE(operands));
}

iree_status_t loom_spirv_module_abi_store_return_values(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan, const loom_op_t* op) {
  IREE_ASSERT_EQ(op->operand_count, plan->result_count);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (iree_host_size_t i = 0; i < plan->result_count; ++i) {
    loom_spirv_module_value_ref_t value = {0};
    IREE_RETURN_IF_ERROR(
        loom_spirv_module_abi_lookup_value(context, operands[i], &value));
    IREE_RETURN_IF_ERROR(loom_spirv_module_abi_store_slot_value(
        context, &plan->results[i], value));
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_module_abi_emit_entry_metadata(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan, iree_string_view_t entry_name) {
  if (plan->kind != LOOM_SPIRV_MODULE_ABI_PLAN_HAL_KERNEL_RAW_BDA) {
    return iree_ok_status();
  }
  if (entry_name.size > (UINT16_MAX - 1u) * 4u - 1u) {
    *context->accepted = false;
    return loom_target_artifact_emit_limit(
        context->diagnostic_emitter, context->function_op, IREE_SV("SPIR-V"),
        IREE_SV("instruction string bytes"), entry_name.size,
        (UINT16_MAX - 1u) * 4u - 1u);
  }
  // One scratch buffer holds the entry name, numeric fields, and the compact
  // logical parameter table. Each parameter token needs at most 15 bytes.
  // The signature count is u16 and the entry name is bounded above.
  const uint16_t argument_count =
      context->function_plan->logical_signature
          ? context->function_plan->logical_signature->arg_count
          : 0;
  const iree_host_size_t text_capacity =
      argument_count * 15u + entry_name.size + 80u;
  char* text_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      context->scratch_arena, text_capacity, (void**)&text_storage));
  iree_string_builder_t text;
  iree_string_builder_initialize_with_storage(text_storage, text_capacity,
                                              &text);
  const uint32_t subgroup_size =
      loom_spirv_function_plan_target_bundle(context->function_plan)
          ->snapshot->subgroup_size;
  if (subgroup_size != 0) {
    IREE_RETURN_IF_ERROR(loom_spirv_module_abi_bda_metadata_field(
        context, &text, entry_name, "subgroup_size=", subgroup_size));
  }
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_bda_metadata_field(
      context, &text, entry_name, "root=0,", LOOM_SPIRV_BDA_ROOT_BYTE_LENGTH));
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_bda_metadata_field(
      context, &text, entry_name,
      "constant_offset=", LOOM_SPIRV_BDA_ROOT_CONSTANT_BYTE_OFFSET));
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_bda_metadata_field(
      context, &text, entry_name,
      "constant_length=", (uint32_t)plan->bda_constant_word_count * 4u));
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_bda_metadata_field(
      context, &text, entry_name, "bindings=", plan->bda_binding_count));
  return loom_spirv_module_abi_emit_bda_parameters(context, plan, entry_name,
                                                   &text);
}

iree_status_t loom_spirv_module_abi_emit_shared_bda_root(
    loom_spirv_module_abi_context_t* context) {
  const loom_spirv_module_shared_bda_root_t* root = context->shared_bda_root;
  if (root->root_variable_id == 0) {
    return iree_ok_status();
  }

  uint32_t root_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_push_constant_bda_root(
      context->type_context, root->constant_word_count, &root_pointer_type_id));
  const uint32_t variable_operands[] = {
      root_pointer_type_id,
      root->root_variable_id,
      LOOM_SPIRV_STORAGE_CLASS_PUSH_CONSTANT,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_module_abi_section(context,
                                    LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_VARIABLE, variable_operands,
      IREE_ARRAYSIZE(variable_operands));
}
