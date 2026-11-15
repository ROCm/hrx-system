// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_storage.h"

#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/isa.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/artifact_diagnostics.h"
#include "loom/target/registers.h"

// Type IDs are private to this allocation so finalizing its optional
// ArrayStride decoration cannot change another function's implicit array.
struct loom_spirv_workgroup_array_t {
  // Address operation retaining the root handle and diagnostic location.
  const loom_op_t* address_op;
  // Array pointer consumed by function instructions.
  loom_spirv_module_value_ref_t address;
  // Allocation-private array type receiving the selected layout decoration.
  uint32_t array_type_id;
};

static loom_spirv_feature_bits_t loom_spirv_module_storage_available_features(
    const loom_spirv_module_storage_context_t* context) {
  return loom_spirv_function_plan_target_bundle(context->function)
      ->config->contract_feature_bits;
}

iree_status_t loom_spirv_module_storage_reserve(
    loom_spirv_module_storage_t* storage,
    const loom_spirv_module_storage_context_t* context, const loom_op_t* op) {
  const loom_module_t* module = context->values->value_domain->module;
  if (!context->function->is_entry ||
      loom_type_storage_space(loom_module_value_type(
          module, loom_low_storage_reserve_storage(op))) !=
          LOOM_STORAGE_SPACE_WORKGROUP ||
      !iree_any_bit_set(
          loom_spirv_module_storage_available_features(context),
          LOOM_SPIRV_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_KHR)) {
    return iree_ok_status();
  }
  return loom_low_storage_layout_builder_append(
      module, op, context->types->scratch_arena, &storage->layout);
}

static uint32_t loom_spirv_module_storage_element_byte_count(
    loom_spirv_scalar_type_t scalar_type) {
  const loom_spirv_scalar_type_descriptor_t* scalar_descriptor =
      loom_spirv_scalar_type_descriptor(scalar_type);
  IREE_ASSERT(scalar_descriptor != NULL);
  IREE_ASSERT_NE(scalar_descriptor->bit_width, 0u);
  IREE_ASSERT_EQ(scalar_descriptor->bit_width % 8, 0u);
  return scalar_descriptor->bit_width / 8;
}

static iree_status_t loom_spirv_module_storage_emit_type(
    const loom_op_t* reserve_op, loom_spirv_scalar_type_t scalar_type,
    uint32_t storage_class, bool deferred_layout,
    loom_spirv_type_context_t* type_context, uint32_t* out_array_type_id,
    uint32_t* out_array_pointer_type_id) {
  const uint32_t element_byte_count =
      loom_spirv_module_storage_element_byte_count(scalar_type);
  const int64_t byte_length = loom_low_storage_reserve_byte_length(reserve_op);
  const int64_t byte_alignment =
      loom_low_storage_reserve_byte_alignment(reserve_op);
  IREE_ASSERT_GT(byte_length, 0);
  IREE_ASSERT_EQ(byte_length % (int64_t)element_byte_count, 0);
  IREE_ASSERT_GE(byte_alignment, (int64_t)element_byte_count);
  const uint64_t element_count = (uint64_t)byte_length / element_byte_count;
  if (element_count == 0 || element_count > UINT32_MAX) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V logical storage element count");
  }

  uint32_t element_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_scalar(type_context, scalar_type, &element_type_id));
  uint32_t length_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      type_context, (uint32_t)element_count, &length_id));
  if (deferred_layout) {
    *out_array_type_id =
        loom_spirv_module_builder_allocate_id(type_context->builder);
    const uint32_t operands[] = {*out_array_type_id, element_type_id,
                                 length_id};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_module_builder_section(
            type_context->builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
        LOOM_SPIRV_OP_TYPE_ARRAY, operands, IREE_ARRAYSIZE(operands)));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_type_array(type_context, element_type_id, length_id,
                                   /*array_stride=*/0, out_array_type_id));
  }
  return loom_spirv_emit_type_pointer(
      type_context, storage_class, *out_array_type_id,
      /*pointer_array_stride=*/0, out_array_pointer_type_id);
}

static iree_status_t loom_spirv_module_storage_emit_variable(
    loom_spirv_module_storage_t* storage,
    const loom_spirv_module_storage_context_t* context,
    const loom_op_t* address_op, const loom_op_t* reserve_op,
    loom_spirv_value_type_t value_type,
    loom_spirv_module_value_ref_t* out_value_ref) {
  loom_spirv_type_context_t* types = context->types;
  loom_spirv_module_builder_t* builder = types->builder;
  const bool workgroup =
      value_type.value_class == LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY;
  const uint32_t storage_class = workgroup ? LOOM_SPIRV_STORAGE_CLASS_WORKGROUP
                                           : LOOM_SPIRV_STORAGE_CLASS_FUNCTION;
  const bool deferred_layout =
      workgroup && context->function->is_entry &&
      iree_any_bit_set(loom_spirv_module_storage_available_features(context),
                       LOOM_SPIRV_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_KHR);
  uint32_t array_type_id = 0;
  uint32_t array_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_storage_emit_type(
      reserve_op, value_type.scalar_type, storage_class, deferred_layout, types,
      &array_type_id, &array_pointer_type_id));
  const uint32_t address_id = loom_spirv_module_builder_allocate_id(builder);
  *out_value_ref = (loom_spirv_module_value_ref_t){
      .id = address_id,
      .type_id = array_pointer_type_id,
      .value_type = value_type,
  };
  if (deferred_layout) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        types->scratch_arena, storage->array_count, storage->array_count + 1,
        sizeof(*storage->arrays), &storage->array_capacity,
        (void**)&storage->arrays));
    storage->arrays[storage->array_count++] = (loom_spirv_workgroup_array_t){
        .address_op = address_op,
        .address = *out_value_ref,
        .array_type_id = array_type_id,
    };
    storage->layout_features |=
        LOOM_SPIRV_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_KHR;
    const uint32_t element_bytes =
        loom_spirv_module_storage_element_byte_count(value_type.scalar_type);
    if (element_bytes == 1) {
      storage->layout_features |=
          LOOM_SPIRV_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_8BIT_ACCESS_KHR;
    } else if (element_bytes == 2) {
      storage->layout_features |=
          LOOM_SPIRV_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_16BIT_ACCESS_KHR;
    }
    return iree_ok_status();
  }

  const uint32_t operands[] = {array_pointer_type_id, address_id,
                               storage_class};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      workgroup ? loom_spirv_module_builder_section(
                      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION)
                : context->function_prefix,
      LOOM_SPIRV_OP_VARIABLE, operands, IREE_ARRAYSIZE(operands)));
  if (workgroup) {
    *context->uses_implicit_workgroup = true;
    IREE_RETURN_IF_ERROR(loom_spirv_entry_interface_declare(
        context->entry_interface, context->function_id, address_id,
        types->scratch_arena));
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_module_storage_emit_address(
    loom_spirv_module_storage_t* storage_state,
    const loom_spirv_module_storage_context_t* context, const loom_op_t* op,
    loom_spirv_module_value_ref_t* out_value_ref) {
  loom_spirv_module_value_table_t* value_table = context->values;

  *out_value_ref = (loom_spirv_module_value_ref_t){0};
  IREE_ASSERT_EQ(loom_low_storage_address_offset(op), 0);
  const loom_local_value_domain_t* value_domain = value_table->value_domain;
  const loom_value_id_t storage = loom_low_storage_address_storage(op);
  const loom_value_t* storage_value =
      loom_module_value(value_domain->module, storage);
  IREE_ASSERT(storage_value != NULL);
  IREE_ASSERT(!loom_value_is_block_arg(storage_value));
  const loom_op_t* reserve_op = loom_value_def_op(storage_value);
  IREE_ASSERT(reserve_op != NULL);
  IREE_ASSERT(loom_low_storage_reserve_isa(reserve_op));

  loom_spirv_value_type_t value_type = {0};
  const loom_type_t result_type = loom_module_value_type(
      value_domain->module, loom_low_storage_address_result(op));
  IREE_ASSERT(loom_low_type_is_register(result_type));
  IREE_ASSERT_EQ(loom_low_register_type_descriptor_set_stable_id(result_type),
                 SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID);
  const bool resolved = loom_spirv_value_type_from_reg_class_id(
      loom_low_register_type_class_id(result_type), &value_type);
  IREE_ASSERT(resolved);
  IREE_ASSERT(
      value_type.value_class == LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY ||
      value_type.value_class == LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION_ARRAY);
  if (loom_spirv_module_value_table_exists(value_table, storage)) {
    const loom_spirv_module_value_ref_t value_ref =
        loom_spirv_module_value_table_lookup(value_table, storage);
    IREE_ASSERT(loom_spirv_value_type_equal(value_ref.value_type, value_type));
    *out_value_ref = value_ref;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_spirv_module_storage_emit_variable(
      storage_state, context, op, reserve_op, value_type, out_value_ref));
  loom_spirv_module_value_table_define(value_table, storage, *out_value_ref);
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_storage_emit_implicit(
    const loom_spirv_module_storage_t* storage,
    const loom_spirv_module_storage_context_t* context) {
  for (iree_host_size_t i = 0; i < storage->array_count; ++i) {
    const loom_spirv_module_value_ref_t address = storage->arrays[i].address;
    const uint32_t operands[] = {address.type_id, address.id,
                                 LOOM_SPIRV_STORAGE_CLASS_WORKGROUP};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_module_builder_section(
            context->types->builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
        LOOM_SPIRV_OP_VARIABLE, operands, IREE_ARRAYSIZE(operands)));
    IREE_RETURN_IF_ERROR(loom_spirv_entry_interface_declare(
        context->entry_interface, context->function_id, address.id,
        context->types->scratch_arena));
  }
  *context->uses_implicit_workgroup |= storage->array_count != 0;
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_storage_emit_explicit(
    const loom_spirv_module_storage_t* storage,
    const loom_spirv_module_storage_context_t* context) {
  loom_spirv_type_context_t* types = context->types;
  loom_spirv_module_builder_t* builder = types->builder;
  // OpTypeStruct has one result ID and a 16-bit instruction word count.
  if (storage->array_count > UINT16_MAX - 2) {
    *context->accepted = false;
    return loom_target_artifact_emit_limit(
        context->diagnostic_emitter, context->function->function_op,
        IREE_SV("SPIR-V"), IREE_SV("Workgroup block members"),
        storage->array_count, UINT16_MAX - 2);
  }

  loom_low_storage_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_builder_finish(
      &storage->layout, types->scratch_arena, &layout));
  loom_spirv_binary_writer_t* declarations = loom_spirv_module_builder_section(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION);
  loom_spirv_binary_writer_t* annotations = loom_spirv_module_builder_section(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION);
  const uint32_t block_type_id = loom_spirv_module_builder_allocate_id(builder);
  uint32_t inline_members[16];
  uint32_t* members = inline_members;
  if (storage->array_count + 1 > IREE_ARRAYSIZE(inline_members)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        types->scratch_arena, storage->array_count + 1, sizeof(*members),
        (void**)&members));
  }
  members[0] = block_type_id;
  for (iree_host_size_t i = 0; i < storage->array_count; ++i) {
    const loom_spirv_workgroup_array_t* array = &storage->arrays[i];
    loom_low_storage_layout_reference_t reference;
    loom_low_storage_layout_lookup_reference(
        &layout.index, layout.records,
        loom_low_storage_address_storage(array->address_op), &reference);
    if (reference.reservation.byte_offset > UINT32_MAX) {
      *context->accepted = false;
      return loom_target_artifact_emit_limit(
          context->diagnostic_emitter, array->address_op, IREE_SV("SPIR-V"),
          IREE_SV("Workgroup member byte offset"),
          reference.reservation.byte_offset, UINT32_MAX);
    }
    const uint32_t stride[] = {array->array_type_id,
                               LOOM_SPIRV_DECORATION_ARRAY_STRIDE,
                               loom_spirv_module_storage_element_byte_count(
                                   array->address.value_type.scalar_type)};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        annotations, LOOM_SPIRV_OP_DECORATE, stride, IREE_ARRAYSIZE(stride)));
    const uint32_t offset[] = {block_type_id, (uint32_t)i,
                               LOOM_SPIRV_DECORATION_OFFSET,
                               (uint32_t)reference.reservation.byte_offset};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        annotations, LOOM_SPIRV_OP_MEMBER_DECORATE, offset,
        IREE_ARRAYSIZE(offset)));
    members[i + 1] = array->array_type_id;
  }
  // Offsets belong to this entry, so a member-type-only cache cannot own it.
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      declarations, LOOM_SPIRV_OP_TYPE_STRUCT, members,
      storage->array_count + 1));
  const uint32_t block[] = {block_type_id, LOOM_SPIRV_DECORATION_BLOCK};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      annotations, LOOM_SPIRV_OP_DECORATE, block, IREE_ARRAYSIZE(block)));
  uint32_t pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
      types, LOOM_SPIRV_STORAGE_CLASS_WORKGROUP, block_type_id,
      /*pointer_array_stride=*/0, &pointer_type_id));
  const uint32_t variable_id = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t variable[] = {pointer_type_id, variable_id,
                               LOOM_SPIRV_STORAGE_CLASS_WORKGROUP};
  IREE_RETURN_IF_ERROR(
      loom_spirv_binary_write_instruction(declarations, LOOM_SPIRV_OP_VARIABLE,
                                          variable, IREE_ARRAYSIZE(variable)));
  IREE_RETURN_IF_ERROR(loom_spirv_entry_interface_declare(
      context->entry_interface, context->function_id, variable_id,
      types->scratch_arena));
  // The reserved pointer IDs are defined after all private OpVariables and
  // before any executable body instruction, including conditional first uses.
  for (iree_host_size_t i = 0; i < storage->array_count; ++i) {
    const loom_spirv_module_value_ref_t address = storage->arrays[i].address;
    uint32_t member_index_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_u32_constant(types, (uint32_t)i, &member_index_id));
    const uint32_t operands[] = {address.type_id, address.id, variable_id,
                                 member_index_id};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        context->function_prefix, LOOM_SPIRV_OP_ACCESS_CHAIN, operands,
        IREE_ARRAYSIZE(operands)));
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_module_storage_finish(
    const loom_spirv_module_storage_t* storage,
    const loom_spirv_module_storage_context_t* context,
    loom_spirv_feature_bits_t* required_features) {
  const bool explicit_layout =
      context->function->is_entry && !*context->uses_implicit_workgroup &&
      iree_all_bits_set(loom_spirv_module_storage_available_features(context),
                        storage->layout_features);
  if (!explicit_layout &&
      iree_any_bit_set(
          *required_features,
          LOOM_SPIRV_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_KHR)) {
    *context->accepted = false;
    return loom_target_artifact_emit_requirement(
        context->diagnostic_emitter, context->function->function_op,
        IREE_SV("SPIR-V"),
        *context->uses_implicit_workgroup || !context->function->is_entry
            ? IREE_SV(
                  "aligned Workgroup access requires storage-owning helpers "
                  "to be inlined by the normal compilation pipeline")
            : IREE_SV("explicit Workgroup layout must support every storage "
                      "element width used by the entry"));
  }
  if (!storage->array_count) {
    return iree_ok_status();
  }
  if (!explicit_layout) {
    return loom_spirv_module_storage_emit_implicit(storage, context);
  }
  IREE_RETURN_IF_ERROR(
      loom_spirv_module_storage_emit_explicit(storage, context));
  *required_features |= storage->layout_features;
  loom_spirv_module_builder_require_feature_bits(context->types->builder,
                                                 storage->layout_features);
  return iree_ok_status();
}
