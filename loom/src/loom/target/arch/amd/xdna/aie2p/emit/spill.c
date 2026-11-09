// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/spill.h"

#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/spill_tables.inl"
#include "loom/target/arch/amd/xdna/aie2p/emit/stack_frame.h"
#include "loom/target/registers.h"

typedef struct loom_aie2p_spill_transfer_t {
  // Shape and identity established by the shared immutable action plan.
  loom_low_spill_materialization_traffic_t traffic;
  // Selected native component sequence for each chunk of this transfer.
  const loom_aie2p_spill_recipe_t* recipe;
  // Negative displacement of the complete reservation below SP.
  int64_t byte_offset;
} loom_aie2p_spill_transfer_t;

typedef struct loom_aie2p_spill_plan_t {
  // Target descriptor set retained by the allocation snapshot.
  const loom_low_descriptor_set_t* descriptors;
  // Interned name of the native displacement field.
  loom_string_id_t immediate_name;
  // Address carrier for transfers beyond the SP-relative encoding range.
  loom_type_t pointer_type;
  // Signed displacement carrier for explicit stack addresses.
  loom_type_t modifier_type;
  // Selected transfers in the shared plan's execution order.
  loom_aie2p_spill_transfer_t* transfers;
} loom_aie2p_spill_plan_t;

// Far stack addresses use the ordinary pointer/modifier instruction family.
// Every setup result has one recipe-local consumer and is reported to the
// shared allocator together with the final memory transfer.
static iree_status_t loom_aie2p_spill_emit_address(
    const loom_aie2p_spill_plan_t* plan, loom_builder_t* builder,
    int64_t byte_offset, loom_location_id_t location,
    loom_low_spill_materialized_value_t* values) {
  const loom_low_spill_materialized_value_flags_t flags =
      LOOM_LOW_SPILL_MATERIALIZED_VALUE_FLAG_PRIVATE_TO_CONSUMER;
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, plan->descriptors,
      &plan->descriptors->descriptors[AIE2P_CORE_DESCRIPTOR_REF_STACK_ADDRESS],
      0, NULL, 0, loom_make_named_attr_slice(NULL, 0), &plan->pointer_type, 1,
      NULL, 0, location, &op));
  values[0] = (loom_low_spill_materialized_value_t){
      .value_id = loom_low_op_results(op).values[0], .flags = flags};
  const loom_named_attr_t immediate = {.name_id = plan->immediate_name,
                                       .value = loom_attr_i64(byte_offset)};
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, plan->descriptors,
      &plan->descriptors
           ->descriptors[AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_MODIFIER_MOVXM],
      loom_make_named_attr_slice(&immediate, 1), plan->modifier_type, location,
      &op));
  values[1] = (loom_low_spill_materialized_value_t){
      .value_id = loom_low_const_result(op), .flags = flags};
  const loom_value_id_t operands[] = {values[0].value_id, values[1].value_id};
  const loom_tied_result_t tie = {.result_index = 0, .operand_index = 0};
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, plan->descriptors,
      &plan->descriptors
           ->descriptors[AIE2P_CORE_DESCRIPTOR_REF_ADDRESS_ADD_A_REGISTER],
      0, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &plan->pointer_type, 1, &tie, 1,
      location, &op));
  values[2] = (loom_low_spill_materialized_value_t){
      .value_id = loom_low_op_results(op).values[0], .flags = flags};
  return iree_ok_status();
}

static iree_status_t loom_aie2p_spill_emit(
    void* user_data, iree_host_size_t instruction_index,
    loom_builder_t* builder, loom_value_id_t storage_value_id,
    loom_value_id_t source_value_id, loom_location_id_t location,
    iree_arena_allocator_t* arena,
    loom_low_spill_materialized_action_t* out_action) {
  const loom_aie2p_spill_plan_t* plan = user_data;
  const loom_aie2p_spill_transfer_t* transfer =
      &plan->transfers[instruction_index];
  const loom_low_spill_materialization_traffic_t* traffic = &transfer->traffic;
  const loom_aie2p_spill_recipe_t* recipe = transfer->recipe;
  const bool store =
      traffic->kind == LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_STORE;
  const uint32_t unit_count =
      loom_low_register_type_unit_count(traffic->value_type);
  const uint32_t chunk_count = unit_count / recipe->unit_count;
  const bool bridge = recipe->reg_class_id !=
                      loom_low_register_type_class_id(traffic->value_type);
  const loom_type_t logical_chunk_type =
      chunk_count == 1 ? traffic->value_type
                       : loom_low_register_carrier_type_with_unit_count(
                             traffic->value_type, recipe->unit_count);
  loom_type_t chunk_type = logical_chunk_type;
  if (bridge) {
    chunk_type = loom_low_register_type(
        plan->descriptors->stable_id, recipe->reg_class_id, recipe->unit_count);
  }
  // The shared reservation inventory owns storage. Native transfers consume
  // its projected depth, independent of the frame's final tile address.
  (void)storage_value_id;
  loom_low_spill_materialized_value_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, chunk_count * (4u * recipe->piece_count + 2u) + 1u,
      sizeof(*values), (void**)&values));
  loom_value_id_t* chunks = NULL;
  if (!store) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, chunk_count, sizeof(*chunks), (void**)&chunks));
  }
  iree_host_size_t value_count = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t chunk = 0; chunk < chunk_count && iree_status_is_ok(status);
       ++chunk) {
    const uint32_t unit_offset = chunk * recipe->unit_count;
    loom_value_id_t source = source_value_id;
    if (store && chunk_count != 1) {
      loom_op_t* op = NULL;
      status = loom_low_slice_build(builder, source, unit_offset,
                                    logical_chunk_type, location, &op);
      if (iree_status_is_ok(status)) {
        source = loom_low_slice_result(op);
        values[value_count++] = (loom_low_spill_materialized_value_t){
            .value_id = source,
        };
      }
    }
    if (iree_status_is_ok(status) && store && bridge) {
      loom_op_t* op = NULL;
      status = loom_low_copy_build(builder, source, /*detached=*/false,
                                   chunk_type, location, &op);
      if (iree_status_is_ok(status)) {
        source = loom_low_copy_result(op);
        values[value_count++] = (loom_low_spill_materialized_value_t){
            .value_id = source,
        };
      }
    }
    loom_value_id_t result = LOOM_VALUE_ID_INVALID;
    for (uint8_t part = 0;
         part < recipe->piece_count && iree_status_is_ok(status); ++part) {
      const loom_aie2p_spill_piece_t* piece =
          &kSpillPieces[recipe->piece_start + part];
      if (traffic->defined_parts != 0 &&
          (traffic->defined_parts & piece->register_parts) == 0) {
        continue;
      }
      const bool continuation = !store && result != LOOM_VALUE_ID_INVALID;
      const uint16_t* descriptor_ordinals =
          continuation ? piece->continuation_descriptors
                       : piece->descriptors[store];
      const int64_t byte_offset =
          transfer->byte_offset + piece->byte_offset +
          (int64_t)(traffic->unit_offset + unit_offset) *
              piece->unit_byte_stride +
          (int64_t)traffic->slot_unit_count * piece->plane_byte_stride;
      const loom_low_descriptor_t* descriptor =
          &plan->descriptors->descriptors[descriptor_ordinals[0]];
      const loom_low_descriptor_layout_t* descriptor_layout =
          &plan->descriptors->descriptor_layouts[descriptor->layout_index];
      const bool uses_far_address =
          byte_offset <
          plan->descriptors->immediates[descriptor_layout->immediate_start]
              .signed_min;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      if (uses_far_address) {
        status = loom_aie2p_spill_emit_address(plan, builder, byte_offset,
                                               location, values + value_count);
        if (iree_status_is_ok(status)) {
          address = values[value_count + 2].value_id;
          value_count += 3;
          descriptor = &plan->descriptors->descriptors[descriptor_ordinals[1]];
        }
      }
      if (iree_status_is_ok(status)) {
        loom_value_id_t operands[2];
        iree_host_size_t operand_count = 0;
        if (store) {
          operands[operand_count++] = source;
        }
        if (uses_far_address) {
          operands[operand_count++] = address;
        }
        const loom_tied_result_t tie = {
            .result_index = 0, .operand_index = (uint16_t)operand_count};
        if (continuation) {
          operands[operand_count++] = result;
        }
        const loom_named_attr_t immediate = {
            .name_id = plan->immediate_name,
            .value = loom_attr_i64(uses_far_address ? 0 : byte_offset),
        };
        loom_op_t* op = NULL;
        status = loom_low_build_resolved_descriptor_op(
            builder, plan->descriptors, descriptor, 0, operands, operand_count,
            loom_make_named_attr_slice(&immediate, 1),
            store ? NULL : &chunk_type, store ? 0 : 1,
            continuation ? &tie : NULL, continuation ? 1 : 0, location, &op);
        if (iree_status_is_ok(status) && !store) {
          result = loom_low_op_results(op).values[0];
          values[value_count++] = (loom_low_spill_materialized_value_t){
              .value_id = result,
          };
        }
      }
    }
    if (iree_status_is_ok(status) && !store) {
      if (bridge) {
        loom_op_t* copy = NULL;
        status = loom_low_copy_build(builder, result, /*detached=*/false,
                                     logical_chunk_type, location, &copy);
        if (iree_status_is_ok(status)) {
          result = loom_low_copy_result(copy);
          values[value_count++] = (loom_low_spill_materialized_value_t){
              .value_id = result,
          };
        }
      }
      chunks[chunk] = result;
    }
  }
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (iree_status_is_ok(status) && !store) {
    result = chunks[0];
    if (chunk_count != 1) {
      loom_op_t* concat = NULL;
      status = loom_low_concat_build(builder, chunks, chunk_count,
                                     traffic->value_type, location, &concat);
      if (iree_status_is_ok(status)) {
        result = loom_low_concat_result(concat);
        values[value_count++] = (loom_low_spill_materialized_value_t){
            .value_id = result,
        };
      }
    }
  }
  *out_action = (loom_low_spill_materialized_action_t){
      .value_id = result,
      .emitted_values = values,
      .emitted_value_count = value_count,
  };
  return status;
}

iree_status_t loom_aie2p_spill_plan(
    void* user_data, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t spill_plan_count,
    const loom_low_spill_materialization_plan_t* materialization_plan,
    iree_arena_allocator_t* arena,
    loom_low_spill_materialization_traffic_plan_t* out_plan) {
  loom_aie2p_stack_layout_t* layout = user_data;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_stack_layout_extend(&table->storage_layout, layout));
  uint64_t* depths = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, spill_plan_count, sizeof(*depths), (void**)&depths));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < spill_plan_count && iree_status_is_ok(status); ++i) {
    if (!loom_low_spill_materialization_record_has_traffic(
            &materialization_plan->records[i])) {
      continue;
    }
    const uint32_t representative =
        materialization_plan->slot_representatives
            ? materialization_plan->slot_representatives[i]
            : i;
    if (representative != i) {
      depths[i] = depths[representative];
    } else {
      status = loom_aie2p_stack_layout_append(
          LOOM_STORAGE_SPACE_STACK, spill_plans[i].byte_size,
          spill_plans[i].byte_alignment, layout, &depths[i]);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  loom_aie2p_spill_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*plan), (void**)&plan));
  *plan =
      (loom_aie2p_spill_plan_t){.descriptors = table->target.descriptor_set};
  IREE_RETURN_IF_ERROR(loom_module_intern_string(table->module, IREE_SV("imm"),
                                                 &plan->immediate_name));
  plan->pointer_type = loom_low_register_type(
      plan->descriptors->stable_id, AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1);
  plan->modifier_type = loom_low_register_type(
      plan->descriptors->stable_id, AIE2P_CORE_REG_CLASS_ID_AIE2P_EM, 1);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, materialization_plan->instruction_count, sizeof(*plan->transfers),
      (void**)&plan->transfers));
  for (iree_host_size_t i = 0; i < materialization_plan->instruction_count;
       ++i) {
    const uint32_t action =
        loom_low_spill_materialization_instruction_action_index(
            &materialization_plan->instructions[i]);
    loom_aie2p_spill_transfer_t* transfer = &plan->transfers[i];
    const bool has_traffic = loom_low_spill_materialization_plan_query_traffic(
        table->module, spill_plans, materialization_plan, action,
        &transfer->traffic);
    IREE_ASSERT(has_traffic);
    const uint32_t spill_index =
        materialization_plan->actions[action].spill_plan_index;
    transfer->byte_offset = -(int64_t)depths[spill_index];
    const uint16_t class_id =
        loom_low_register_type_class_id(transfer->traffic.value_type);
    const uint32_t unit_count =
        loom_low_register_type_unit_count(transfer->traffic.value_type);
    const uint8_t* recipes = kSpillClassRecipes[class_id];
    const bool wide = recipes[1] && (unit_count % 2 == 0) &&
                      (transfer->traffic.unit_offset % 2 == 0);
    IREE_ASSERT_NE(recipes[wide], 0);
    transfer->recipe = &kSpillRecipes[recipes[wide]];
  }
  *out_plan = (loom_low_spill_materialization_traffic_plan_t){
      .user_data = plan, .emit = loom_aie2p_spill_emit};
  return iree_ok_status();
}
