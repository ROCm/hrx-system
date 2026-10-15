// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/packet_legalization.h"

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/combining.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/transforms/vector/packet_materialization.h"
#include "loom/transforms/vector/packet_plan.h"

// Dynamic windows also materialize one constant/add pair for every packet
// after the first. The wider bound covers a 64-lane predicate snapshot feeding
// signed and unsigned 64-bit consumers while keeping malformed or enormous
// aggregate types from expanding without limit.
#define LOOM_VECTOR_PACKET_DYNAMIC_WINDOW_OP_LIMIT 384u

static bool loom_vector_packet_dynamic_window_expansion_is_bounded(
    const loom_vector_packetization_t* packetization,
    loom_value_id_t dynamic_offset,
    const loom_vector_packet_shape_t* result_shape, uint32_t consumer_count,
    uint32_t consumer_branch_count,
    const loom_vector_packet_consumer_materializer_t* materializer) {
  iree_host_size_t operation_count = 0;
  iree_host_size_t consumer_operation_count = 0;
  if (consumer_count != 0 &&
      (materializer == NULL || materializer->operation_count == 0)) {
    return false;
  }
  if (!iree_host_size_checked_mul(
          consumer_count,
          materializer == NULL ? 0 : materializer->operation_count,
          &consumer_operation_count)) {
    return false;
  }

  const loom_target_legalization_context_t* context = packetization->context;
  const loom_value_facts_t base_offset_facts =
      context->fact_table == NULL
          ? loom_value_facts_unknown()
          : loom_value_fact_table_lookup(context->fact_table, dynamic_offset);
  for (uint32_t chunk_index = 0; chunk_index < result_shape->chunk_count;
       ++chunk_index) {
    const uint32_t packet_offset = chunk_index * result_shape->chunk_lane_count;
    const uint32_t packet_lane_count =
        iree_min(result_shape->lane_count - packet_offset,
                 result_shape->chunk_lane_count);
    loom_value_facts_t packet_offset_facts = base_offset_facts;
    if (packet_offset != 0) {
      const loom_value_facts_t static_offset_facts =
          loom_value_facts_exact_i64(packet_offset);
      loom_value_facts_addi(&base_offset_facts, &static_offset_facts,
                            &packet_offset_facts);
    }
    iree_host_size_t packet_operation_count = 0;
    if (!loom_vector_packet_dynamic_materialization_operation_count(
            packetization, packet_offset_facts, packet_lane_count,
            &packet_operation_count) ||
        !iree_host_size_checked_add(operation_count, packet_operation_count,
                                    &operation_count) ||
        !iree_host_size_checked_add(operation_count, consumer_operation_count,
                                    &operation_count) ||
        (packet_offset != 0 &&
         !iree_host_size_checked_add(operation_count, 2u, &operation_count))) {
      return false;
    }
  }

  if (result_shape->chunk_count > 1) {
    const uint32_t result_group_count =
        consumer_branch_count == 0 ? 1u : consumer_branch_count;
    if (!iree_host_size_checked_add(operation_count, result_group_count,
                                    &operation_count)) {
      return false;
    }
  }
  return operation_count <= LOOM_VECTOR_PACKET_DYNAMIC_WINDOW_OP_LIMIT;
}

typedef struct loom_vector_packet_consumer_branch_t {
  // Index of this branch's first operation in the flattened consumer array.
  uint32_t consumer_start;
  // Number of unary operations in this branch.
  uint32_t consumer_count;
} loom_vector_packet_consumer_branch_t;

static bool loom_vector_packet_dynamic_window_consumer_is_eligible(
    const loom_target_legalization_context_t* context,
    const loom_op_t* slice_op, loom_value_id_t source, loom_op_t* consumer,
    const loom_vector_packet_consumer_materializer_t* materializer) {
  if (consumer == NULL || consumer->parent_block != slice_op->parent_block ||
      consumer->operand_count != 1 || consumer->result_count != 1 ||
      consumer->region_count != 0 || consumer->successor_count != 0 ||
      consumer->tied_result_count != 0 ||
      loom_op_const_operands(consumer)[0] != source || materializer == NULL ||
      materializer->matches == NULL || materializer->supports_packet == NULL ||
      materializer->materialize == NULL ||
      !materializer->matches(materializer->user_data, context->module,
                             consumer) ||
      !iree_all_bits_set(loom_op_effective_traits(context->module, consumer),
                         LOOM_TRAIT_PURE | LOOM_TRAIT_DECOMPOSABLE)) {
    return false;
  }
  const loom_value_t* result_value =
      loom_module_value(context->module, loom_op_const_results(consumer)[0]);
  uint32_t result_lane_count = 0;
  return result_value != NULL && !loom_value_has_attribute_uses(result_value) &&
         loom_vector_packet_static_lane_count(result_value->type,
                                              &result_lane_count) &&
         result_lane_count != 0;
}

// Selects explicitly materializable pure unary branches exclusively consuming
// a dynamic window. Replaying every branch with each producer packet keeps
// transient carriers such as predicate masks from accumulating before their
// compact payloads are formed. Each branch may continue through an exclusive
// supported unary chain; shaped lane changes and other operations terminate
// the branch.
static uint32_t loom_vector_packet_select_dynamic_window_consumers(
    const loom_target_legalization_context_t* context,
    const loom_op_t* slice_op,
    const loom_vector_packet_consumer_materializer_t* materializer,
    loom_op_t** out_consumers, uint32_t consumer_capacity,
    loom_vector_packet_consumer_branch_t* out_branches,
    uint32_t branch_capacity, uint32_t* out_consumer_count) {
  *out_consumer_count = 0;
  const loom_value_id_t root = loom_vector_slice_result(slice_op);
  const loom_value_t* root_value = loom_module_value(context->module, root);
  if (root_value == NULL || root_value->use_count == 0 ||
      root_value->use_count > branch_capacity ||
      loom_value_has_attribute_uses(root_value) ||
      loom_module_value_has_type_uses(context->module, root)) {
    return 0;
  }

  const loom_use_t* root_uses = loom_value_uses(root_value);
  uint32_t branch_count = 0;
  for (uint32_t use_index = 0; use_index < root_value->use_count; ++use_index) {
    loom_value_id_t source = root;
    loom_op_t* consumer = loom_use_user_op(root_uses[use_index]);
    if (!loom_vector_packet_dynamic_window_consumer_is_eligible(
            context, slice_op, source, consumer, materializer) ||
        *out_consumer_count == consumer_capacity) {
      *out_consumer_count = 0;
      return 0;
    }

    loom_vector_packet_consumer_branch_t* branch =
        &out_branches[branch_count++];
    branch->consumer_start = *out_consumer_count;
    do {
      out_consumers[(*out_consumer_count)++] = consumer;
      ++branch->consumer_count;
      source = loom_op_const_results(consumer)[0];
      const loom_value_t* value = loom_module_value(context->module, source);
      if (value == NULL || value->use_count != 1 ||
          loom_value_has_attribute_uses(value) ||
          loom_module_value_has_type_uses(context->module, source) ||
          *out_consumer_count == consumer_capacity) {
        break;
      }
      consumer = loom_use_user_op(loom_value_uses(value)[0]);
    } while (loom_vector_packet_dynamic_window_consumer_is_eligible(
        context, slice_op, source, consumer, materializer));
  }
  return branch_count;
}

static bool loom_vector_packet_dynamic_window_consumers_support_packet(
    const loom_target_legalization_context_t* context, loom_type_t source_type,
    loom_op_t* const* consumers, uint32_t consumer_count,
    const loom_vector_packet_consumer_materializer_t* materializer,
    uint32_t lane_count) {
  if (lane_count == 0 ||
      !loom_target_legalization_has_source_vector_carrier(
          context, loom_vector_packet_type(source_type, lane_count))) {
    return false;
  }
  for (uint32_t i = 0; i < consumer_count; ++i) {
    if (!materializer->supports_packet(materializer->user_data, context,
                                       consumers[i], lane_count)) {
      return false;
    }
  }
  return true;
}

// Selects the widest packet carrier shared by the dynamic source and every
// explicitly fused consumer. Both the regular packet and any irregular tail
// must have complete target representations.
static bool loom_vector_packet_dynamic_window_constrain_chunk_lane_count(
    const loom_target_legalization_context_t* context, loom_type_t source_type,
    uint32_t result_lane_count, loom_op_t* const* consumers,
    uint32_t consumer_count,
    const loom_vector_packet_consumer_materializer_t* materializer,
    uint32_t* inout_chunk_lane_count) {
  if (consumer_count == 0) {
    return true;
  }
  for (uint32_t candidate_lane_count = *inout_chunk_lane_count;
       candidate_lane_count > 0; --candidate_lane_count) {
    const uint32_t tail_lane_count = result_lane_count % candidate_lane_count;
    if (!loom_vector_packet_dynamic_window_consumers_support_packet(
            context, source_type, consumers, consumer_count, materializer,
            candidate_lane_count) ||
        (tail_lane_count != 0 &&
         !loom_vector_packet_dynamic_window_consumers_support_packet(
             context, source_type, consumers, consumer_count, materializer,
             tail_lane_count))) {
      continue;
    }
    *inout_chunk_lane_count = candidate_lane_count;
    return true;
  }
  return false;
}

static iree_status_t loom_vector_packet_materialize_dynamic_window_consumers(
    loom_target_legalization_context_t* context, loom_op_t* const* consumers,
    uint32_t consumer_count, uint32_t lane_count,
    const loom_vector_packet_consumer_materializer_t* materializer,
    loom_value_id_t source, loom_value_id_t* out_result) {
  loom_value_id_t result = source;
  for (uint32_t i = 0; i < consumer_count; ++i) {
    IREE_RETURN_IF_ERROR(
        materializer->materialize(materializer->user_data, context,
                                  consumers[i], lane_count, result, &result));
  }
  *out_result = result;
  return iree_ok_status();
}

static bool loom_vector_packet_dynamic_snapshot_expansion_is_bounded(
    const loom_vector_packet_shape_t* source_shape,
    const loom_vector_packet_shape_t* result_shape, uint32_t consumer_count,
    uint32_t consumer_branch_count,
    const loom_vector_packet_consumer_materializer_t* materializer) {
  const uint32_t group_count =
      consumer_branch_count == 0 ? 1u : consumer_branch_count;
  uint64_t operation_count = (uint64_t)group_count * 4u;
  const uint32_t materializer_operation_count =
      materializer == NULL ? 0 : materializer->operation_count;
  operation_count +=
      (uint64_t)source_shape->chunk_count *
      (1u + (uint64_t)consumer_count * materializer_operation_count +
       group_count);
  operation_count += (uint64_t)group_count * result_shape->chunk_count;
  if (result_shape->chunk_count > 1) {
    operation_count += (uint64_t)(result_shape->chunk_count - 1u) * 2u;
    operation_count += group_count;
  }
  return operation_count <= LOOM_VECTOR_PACKET_DYNAMIC_WINDOW_OP_LIMIT;
}

// Reifies an immutable packet tuple into private storage while each packet is
// available, then reads runtime-selected packets directly from that storage.
// Unary consumer branches are applied before the stores so narrow transient
// carriers do not remain live across the complete tuple.
static iree_status_t loom_vector_packet_legalize_dynamic_snapshot(
    loom_vector_packetization_t* packetization, loom_op_t* slice_op,
    loom_value_id_t dynamic_offset,
    const loom_vector_packet_shape_t* source_shape,
    const loom_vector_packet_shape_t* result_shape, loom_op_t* const* consumers,
    const loom_vector_packet_consumer_branch_t* consumer_branches,
    uint32_t consumer_count, uint32_t consumer_branch_count,
    const loom_vector_packet_consumer_materializer_t* consumer_materializer,
    bool* out_rewritten) {
  *out_rewritten = false;
  loom_target_legalization_context_t* context = packetization->context;
  const loom_value_id_t source = loom_vector_slice_source(slice_op);
  loom_vector_packetized_value_t* snapshot =
      loom_vector_packet_find(packetization, source);
  if (snapshot == NULL ||
      snapshot->source_mode != LOOM_VECTOR_PACKET_SOURCE_MODE_SNAPSHOT ||
      snapshot->source_partition.snapshot_count == 0 ||
      !loom_vector_packet_dynamic_snapshot_expansion_is_bounded(
          source_shape, result_shape, consumer_count, consumer_branch_count,
          consumer_materializer)) {
    return iree_ok_status();
  }

  const uint32_t group_count =
      consumer_branch_count == 0 ? 1u : consumer_branch_count;
  loom_type_t staging_types[LOOM_VECTOR_PACKET_DYNAMIC_WINDOW_OP_LIMIT];
  loom_value_id_t staging_views[LOOM_VECTOR_PACKET_DYNAMIC_WINDOW_OP_LIMIT];
  for (uint32_t group_index = 0; group_index < group_count; ++group_index) {
    loom_type_t element_source_type = snapshot->source_type;
    if (consumer_branch_count != 0) {
      const loom_vector_packet_consumer_branch_t branch =
          consumer_branches[group_index];
      const loom_op_t* terminal =
          consumers[branch.consumer_start + branch.consumer_count - 1u];
      element_source_type = loom_module_value_type(
          context->module, loom_op_const_results(terminal)[0]);
    }
    const int32_t element_bit_count = loom_vector_packet_element_bit_count(
        packetization->policy, loom_type_element_type(element_source_type));
    if (element_bit_count <= 0 || element_bit_count % 8 != 0) {
      return iree_ok_status();
    }
    staging_types[group_index] =
        loom_vector_packet_type(element_source_type, source_shape->lane_count);
  }

  const loom_value_t* snapshot_value =
      loom_module_value(context->module, snapshot->source);
  loom_op_t* snapshot_op = loom_value_def_op(snapshot_value);
  if (snapshot_op == NULL ||
      snapshot_op->parent_block != slice_op->parent_block ||
      slice_op->parent_block->first_op == NULL) {
    return iree_ok_status();
  }
  const loom_value_fact_vector_partition_t partition =
      snapshot->source_partition;
  for (uint32_t packet_index = 0; packet_index < partition.snapshot_count;
       ++packet_index) {
    const loom_value_t* packet_value = loom_module_value(
        context->module, partition.snapshot_components[packet_index].value);
    if (packet_value == NULL) {
      return iree_ok_status();
    }
    if (loom_value_is_block_arg(packet_value)) {
      if (loom_value_def_block(packet_value) != slice_op->parent_block) {
        return iree_ok_status();
      }
    } else {
      const loom_op_t* packet_op = loom_value_def_op(packet_value);
      if (packet_op == NULL ||
          packet_op->parent_block != slice_op->parent_block) {
        return iree_ok_status();
      }
    }
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_before(builder, slice_op->parent_block->first_op);
  for (uint32_t group_index = 0; group_index < group_count; ++group_index) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_build_staging_view(
        packetization, staging_types[group_index], source_shape, snapshot_op,
        &staging_views[group_index]));
  }

  for (uint32_t packet_index = 0; packet_index < partition.snapshot_count;
       ++packet_index) {
    const loom_value_id_t source_packet =
        partition.snapshot_components[packet_index].value;
    const loom_value_t* packet_value =
        loom_module_value(context->module, source_packet);
    const loom_op_t* packet_op = loom_value_is_block_arg(packet_value)
                                     ? NULL
                                     : loom_value_def_op(packet_value);
    if (packet_op != NULL) {
      loom_builder_set_after(builder, (loom_op_t*)packet_op);
    } else {
      loom_builder_set_before(builder, slice_op);
    }
    uint32_t packet_lane_count = 0;
    const bool packet_shape_valid = loom_vector_packet_static_lane_count(
        packet_value->type, &packet_lane_count);
    IREE_ASSERT_TRUE(packet_shape_valid);
    (void)packet_shape_valid;
    for (uint32_t packet_lane_offset = 0;
         packet_lane_offset < packet_lane_count;
         packet_lane_offset += source_shape->chunk_lane_count) {
      const uint32_t lane_count =
          iree_min(packet_lane_count - packet_lane_offset,
                   source_shape->chunk_lane_count);
      const uint32_t source_lane_offset =
          packet_index * partition.snapshot_lane_count + packet_lane_offset;
      loom_value_id_t source_chunk = source_packet;
      if (packet_lane_offset != 0 || lane_count != packet_lane_count) {
        const int64_t static_lane_offset = packet_lane_offset;
        loom_op_t* slice_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_slice_build(
            builder, source_packet, /*offsets=*/NULL, /*offsets_count=*/0,
            &static_lane_offset, /*static_offsets_count=*/1,
            loom_vector_packet_type(snapshot->source_type, lane_count),
            snapshot_op->location, &slice_op));
        source_chunk = loom_vector_slice_result(slice_op);
      }
      const loom_vector_packet_slice_t packet_slice = {
          .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
          .static_lane_offset = source_lane_offset,
          .lane_count = lane_count,
      };
      for (uint32_t group_index = 0; group_index < group_count; ++group_index) {
        loom_value_id_t packet = source_chunk;
        if (consumer_branch_count != 0) {
          const loom_vector_packet_consumer_branch_t branch =
              consumer_branches[group_index];
          IREE_RETURN_IF_ERROR(
              loom_vector_packet_materialize_dynamic_window_consumers(
                  context, consumers + branch.consumer_start,
                  branch.consumer_count, lane_count, consumer_materializer,
                  source_chunk, &packet));
        }
        IREE_RETURN_IF_ERROR(loom_vector_packet_staging_store(
            packetization, staging_views[group_index], &packet_slice, packet,
            snapshot_op));
      }
    }
  }

  loom_builder_set_before(builder, slice_op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  iree_host_size_t result_packet_count = 0;
  const bool packet_count_valid = iree_host_size_checked_mul(
      group_count, result_shape->chunk_count, &result_packet_count);
  IREE_ASSERT_TRUE(packet_count_valid);
  (void)packet_count_valid;
  loom_value_id_t* result_packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, result_packet_count, sizeof(*result_packets),
      (void**)&result_packets));
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  for (uint32_t chunk_index = 0; chunk_index < result_shape->chunk_count;
       ++chunk_index) {
    const uint32_t packet_offset = chunk_index * result_shape->chunk_lane_count;
    loom_value_id_t packet_offset_value = dynamic_offset;
    if (packet_offset != 0) {
      loom_op_t* constant_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          builder, loom_attr_i64(packet_offset), index_type, slice_op->location,
          &constant_op));
      loom_op_t* add_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          builder, dynamic_offset, loom_index_constant_result(constant_op),
          index_type, slice_op->location, &add_op));
      packet_offset_value = loom_index_add_result(add_op);
    }
    const loom_vector_packet_slice_t packet_slice = {
        .dynamic_lane_offset = packet_offset_value,
        .lane_count = iree_min(result_shape->lane_count - packet_offset,
                               result_shape->chunk_lane_count),
    };
    for (uint32_t group_index = 0; group_index < group_count; ++group_index) {
      IREE_RETURN_IF_ERROR(loom_vector_packet_staging_load(
          packetization, staging_views[group_index], staging_types[group_index],
          &packet_slice, snapshot_op,
          &result_packets[group_index * result_shape->chunk_count +
                          chunk_index]));
    }
  }

  if (consumer_branch_count == 0) {
    loom_value_id_t replacement = result_packets[0];
    if (result_shape->chunk_count > 1) {
      loom_op_t* concat_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_concat_build(
          builder, /*axis=*/0, result_packets, result_shape->chunk_count,
          loom_module_value_type(context->module,
                                 loom_vector_slice_result(slice_op)),
          slice_op->location, &concat_op));
      replacement = loom_vector_concat_result(concat_op);
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        rewriter, slice_op, &replacement, 1, value_checkpoint));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, slice_op, &replacement, 1));
  } else {
    for (uint32_t branch_index = 0; branch_index < consumer_branch_count;
         ++branch_index) {
      const loom_vector_packet_consumer_branch_t branch =
          consumer_branches[branch_index];
      loom_op_t* terminal =
          consumers[branch.consumer_start + branch.consumer_count - 1u];
      loom_value_id_t* branch_packets =
          result_packets + branch_index * result_shape->chunk_count;
      loom_value_id_t replacement = branch_packets[0];
      if (result_shape->chunk_count > 1) {
        loom_op_t* concat_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_concat_build(
            builder, /*axis=*/0, branch_packets, result_shape->chunk_count,
            loom_module_value_type(context->module,
                                   loom_op_const_results(terminal)[0]),
            terminal->location, &concat_op));
        replacement = loom_vector_concat_result(concat_op);
      }
      IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
          rewriter, terminal, &replacement, 1, value_checkpoint));
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
          rewriter, terminal, &replacement, 1));
      for (uint32_t i = branch.consumer_count; i > 1; --i) {
        bool erased = false;
        IREE_RETURN_IF_ERROR(loom_rewriter_erase_if_dead(
            rewriter, consumers[branch.consumer_start + i - 2u], &erased));
        IREE_ASSERT_TRUE(erased);
      }
    }
    bool erased = false;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_erase_if_dead(rewriter, slice_op, &erased));
    IREE_ASSERT_TRUE(erased);
  }
  IREE_RETURN_IF_ERROR(
      loom_vector_packet_erase_dead_sources(packetization, NULL));
  loom_builder_restore(builder, saved_ip);
  *out_rewritten = true;
  return iree_ok_status();
}

typedef struct loom_vector_packet_dynamic_window_t {
  // Authored aggregate selected at a runtime origin.
  loom_value_id_t source;
  // Runtime leading-axis origin.
  loom_value_id_t dynamic_offset;
  // Static source vector type.
  loom_type_t source_type;
  // Static result vector type.
  loom_type_t result_type;
  // Number of lanes in |source_type|.
  uint32_t source_lane_count;
  // Number of lanes in |result_type|.
  uint32_t result_lane_count;
  // Widest policy packet admitted for |source_type|.
  uint32_t maximum_chunk_lane_count;
} loom_vector_packet_dynamic_window_t;

// Describes the dynamic-window family implemented by packet replay. Keeping
// this shared between planning and materialization prevents either phase from
// claiming a producer chain that the other cannot lower.
static bool loom_vector_packet_dynamic_window_describe(
    const loom_target_legalization_context_t* context,
    const loom_target_vector_packet_policy_t* policy, const loom_op_t* op,
    loom_vector_packet_dynamic_window_t* out_window) {
  *out_window = (loom_vector_packet_dynamic_window_t){0};
  if (policy == NULL ||
      !iree_any_bit_set(
          policy->flags,
          LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_DYNAMIC_WINDOW_REPLAY) ||
      !loom_vector_slice_isa(op)) {
    return false;
  }

  const loom_value_slice_t offsets = loom_vector_slice_offsets(op);
  const loom_attribute_t static_offsets = loom_vector_slice_static_offsets(op);
  const loom_value_id_t source = loom_vector_slice_source(op);
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_slice_result(op));
  uint32_t source_lane_count = 0;
  uint32_t result_lane_count = 0;
  uint32_t maximum_chunk_lane_count = 0;
  if (offsets.count != 1 || static_offsets.kind != LOOM_ATTR_I64_ARRAY ||
      static_offsets.count != 1 || static_offsets.i64_array[0] != INT64_MIN ||
      !loom_type_element_type_equals(source_type, result_type) ||
      !loom_vector_packet_static_lane_count(source_type, &source_lane_count) ||
      !loom_vector_packet_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count >= source_lane_count ||
      !loom_vector_packet_maximum_chunk_lane_count(policy, source_type,
                                                   &maximum_chunk_lane_count)) {
    return false;
  }

  *out_window = (loom_vector_packet_dynamic_window_t){
      .source = source,
      .dynamic_offset = offsets.values[0],
      .source_type = source_type,
      .result_type = result_type,
      .source_lane_count = source_lane_count,
      .result_lane_count = result_lane_count,
      .maximum_chunk_lane_count = maximum_chunk_lane_count,
  };
  return true;
}

iree_status_t loom_vector_packet_dynamic_window_owns_consumer_chain(
    loom_target_legalization_context_t* context,
    const loom_target_vector_packet_policy_t* policy, const loom_op_t* op,
    bool* out_owns_consumer_chain) {
  *out_owns_consumer_chain = false;
  loom_vector_packet_dynamic_window_t window = {0};
  if (!loom_vector_packet_dynamic_window_describe(context, policy, op,
                                                  &window)) {
    return iree_ok_status();
  }

  loom_target_contract_query_result_t query_result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(
      loom_target_legalization_query_contract(context, op, &query_result));
  *out_owns_consumer_chain =
      query_result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL &&
      query_result.outcome != LOOM_TARGET_CONTRACT_QUERY_INVALID_IR;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_legalize_dynamic_slice(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy,
    const loom_vector_packet_consumer_materializer_t* consumer_materializer,
    bool* out_rewritten) {
  *out_rewritten = false;
  loom_vector_packet_dynamic_window_t window = {0};
  if (!loom_vector_packet_dynamic_window_describe(context, policy, op,
                                                  &window)) {
    return iree_ok_status();
  }

  const loom_value_id_t source = window.source;
  const loom_type_t source_type = window.source_type;
  const loom_type_t result_type = window.result_type;
  const uint32_t source_lane_count = window.source_lane_count;
  const uint32_t result_lane_count = window.result_lane_count;
  uint32_t chunk_lane_count = window.maximum_chunk_lane_count;

  loom_op_t* consumers[LOOM_VECTOR_PACKET_DYNAMIC_WINDOW_OP_LIMIT] = {0};
  loom_vector_packet_consumer_branch_t
      consumer_branches[LOOM_VECTOR_PACKET_DYNAMIC_WINDOW_OP_LIMIT] = {0};
  uint32_t consumer_count = 0;
  const uint32_t consumer_branch_count =
      loom_vector_packet_select_dynamic_window_consumers(
          context, op, consumer_materializer, consumers,
          IREE_ARRAYSIZE(consumers), consumer_branches,
          IREE_ARRAYSIZE(consumer_branches), &consumer_count);
  if (!loom_vector_packet_dynamic_window_constrain_chunk_lane_count(
          context, source_type, result_lane_count, consumers, consumer_count,
          consumer_materializer, &chunk_lane_count)) {
    return iree_ok_status();
  }

  loom_vector_packet_shape_t source_shape = {
      .lane_count = source_lane_count,
      .chunk_lane_count = chunk_lane_count,
      .chunk_count = (source_lane_count - 1u) / chunk_lane_count + 1u,
  };
  loom_vector_packetization_t packetization = {0};
  loom_vector_packetization_initialize(context, policy, &packetization);
  bool producer_selected = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_select_value_shape(
      &packetization, source, &source_shape, &producer_selected));
  if (!producer_selected) {
    return iree_ok_status();
  }
  const uint32_t selected_tail_lane_count =
      result_lane_count % source_shape.chunk_lane_count;
  if (consumer_count != 0 &&
      (!loom_vector_packet_dynamic_window_consumers_support_packet(
           context, source_type, consumers, consumer_count,
           consumer_materializer, source_shape.chunk_lane_count) ||
       (selected_tail_lane_count != 0 &&
        !loom_vector_packet_dynamic_window_consumers_support_packet(
            context, source_type, consumers, consumer_count,
            consumer_materializer, selected_tail_lane_count)))) {
    return iree_ok_status();
  }
  const loom_vector_packet_shape_t result_shape = {
      .lane_count = result_lane_count,
      .chunk_lane_count = source_shape.chunk_lane_count,
      .chunk_count =
          (result_lane_count - 1u) / source_shape.chunk_lane_count + 1u,
  };
  if (!loom_vector_packet_dynamic_window_expansion_is_bounded(
          &packetization, window.dynamic_offset, &result_shape, consumer_count,
          consumer_branch_count, consumer_materializer)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_vector_packet_classify_memory_loads(&packetization, op,
                                               /*allow_streaming=*/true));
  if (!loom_vector_packet_can_materialize(&packetization, &source_shape)) {
    return iree_ok_status();
  }

  bool source_reads_preserved = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_preserve_memory_loads(
      &packetization, &source_shape, &source_reads_preserved));
  if (source_reads_preserved) {
    *out_rewritten = true;
    return iree_ok_status();
  }
  if (loom_vector_packet_has_snapshots(&packetization)) {
    return loom_vector_packet_legalize_dynamic_snapshot(
        &packetization, op, window.dynamic_offset, &source_shape, &result_shape,
        consumers, consumer_branches, consumer_count, consumer_branch_count,
        consumer_materializer, out_rewritten);
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const uint32_t result_packet_group_count =
      consumer_branch_count == 0 ? 1u : consumer_branch_count;
  iree_host_size_t result_packet_count = 0;
  if (!iree_host_size_checked_mul(result_packet_group_count,
                                  result_shape.chunk_count,
                                  &result_packet_count)) {
    return iree_ok_status();
  }
  loom_value_id_t* result_packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, result_packet_count, sizeof(*result_packets),
      (void**)&result_packets));
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  for (uint32_t chunk_index = 0; chunk_index < result_shape.chunk_count;
       ++chunk_index) {
    const uint32_t packet_offset = chunk_index * result_shape.chunk_lane_count;
    loom_value_id_t dynamic_offset = window.dynamic_offset;
    if (packet_offset != 0) {
      loom_op_t* packet_offset_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          builder, loom_attr_i64(packet_offset), index_type, op->location,
          &packet_offset_op));
      loom_op_t* adjusted_offset_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          builder, dynamic_offset, loom_index_constant_result(packet_offset_op),
          index_type, op->location, &adjusted_offset_op));
      dynamic_offset = loom_index_add_result(adjusted_offset_op);
    }
    const loom_vector_packet_slice_t packet_slice = {
        .dynamic_lane_offset = dynamic_offset,
        .lane_count = iree_min(result_lane_count - packet_offset,
                               result_shape.chunk_lane_count),
    };
    loom_vector_packetized_value_t* packetized_source = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_packet_materialize(
        &packetization, source, &packet_slice, &packetized_source));
    if (consumer_branch_count == 0) {
      result_packets[chunk_index] = packetized_source->packet;
      continue;
    }
    for (uint32_t branch_index = 0; branch_index < consumer_branch_count;
         ++branch_index) {
      const loom_vector_packet_consumer_branch_t branch =
          consumer_branches[branch_index];
      IREE_RETURN_IF_ERROR(
          loom_vector_packet_materialize_dynamic_window_consumers(
              context, consumers + branch.consumer_start, branch.consumer_count,
              packet_slice.lane_count, consumer_materializer,
              packetized_source->packet,
              &result_packets[branch_index * result_shape.chunk_count +
                              chunk_index]));
    }
  }

  if (consumer_branch_count == 0) {
    loom_value_id_t replacement = result_packets[0];
    if (result_shape.chunk_count > 1) {
      loom_op_t* concat_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_concat_build(
          builder, /*axis=*/0, result_packets, result_shape.chunk_count,
          result_type, op->location, &concat_op));
      replacement = loom_vector_concat_result(concat_op);
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        rewriter, op, &replacement, 1, value_checkpoint));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, op, &replacement, 1));
  } else {
    for (uint32_t branch_index = 0; branch_index < consumer_branch_count;
         ++branch_index) {
      const loom_vector_packet_consumer_branch_t branch =
          consumer_branches[branch_index];
      loom_op_t* replacement_op =
          consumers[branch.consumer_start + branch.consumer_count - 1u];
      loom_value_id_t* branch_packets =
          result_packets + branch_index * result_shape.chunk_count;
      loom_value_id_t replacement = branch_packets[0];
      if (result_shape.chunk_count > 1) {
        const loom_type_t replacement_type = loom_module_value_type(
            context->module, loom_op_const_results(replacement_op)[0]);
        loom_op_t* concat_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_concat_build(
            builder, /*axis=*/0, branch_packets, result_shape.chunk_count,
            replacement_type, replacement_op->location, &concat_op));
        replacement = loom_vector_concat_result(concat_op);
      }
      IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
          rewriter, replacement_op, &replacement, 1, value_checkpoint));
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
          rewriter, replacement_op, &replacement, 1));
      for (uint32_t i = branch.consumer_count; i > 1; --i) {
        bool erased = false;
        IREE_RETURN_IF_ERROR(loom_rewriter_erase_if_dead(
            rewriter, consumers[branch.consumer_start + i - 2u], &erased));
        IREE_ASSERT_TRUE(erased);
      }
    }
    bool erased = false;
    IREE_RETURN_IF_ERROR(loom_rewriter_erase_if_dead(rewriter, op, &erased));
    IREE_ASSERT_TRUE(erased);
  }
  IREE_RETURN_IF_ERROR(
      loom_vector_packet_erase_dead_sources(&packetization, NULL));
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_legalize_table_lookup(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten) {
  *out_rewritten = false;

  const loom_value_id_t indices = loom_vector_table_lookup_indices(op);
  const loom_type_t index_type =
      loom_module_value_type(context->module, indices);
  const loom_type_t result_type = loom_module_value_type(
      context->module, loom_vector_table_lookup_result(op));
  loom_vector_packet_shape_t shape = {0};
  if (!loom_vector_packet_table_lookup_shape(policy, index_type, result_type,
                                             &shape)) {
    return iree_ok_status();
  }

  loom_vector_packetization_t packetization = {0};
  loom_vector_packetization_initialize(context, policy, &packetization);
  packetization.selection_flags =
      LOOM_VECTOR_PACKET_SELECTION_FLAG_CAPTURE_EXTERNAL;
  packetization.captured_value_location = op->location;
  bool producer_selected = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_select_value_shape(
      &packetization, indices, &shape, &producer_selected));
  if (producer_selected) {
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_classify_memory_loads(&packetization, op,
                                                 /*allow_streaming=*/true));
    if (!loom_vector_packet_can_materialize(&packetization, &shape)) {
      return iree_ok_status();
    }
    bool source_reads_preserved = false;
    IREE_RETURN_IF_ERROR(loom_vector_packet_preserve_memory_loads(
        &packetization, &shape, &source_reads_preserved));
    if (source_reads_preserved) {
      *out_rewritten = true;
      return iree_ok_status();
    }
    if (loom_vector_packet_shared_snapshot_requires_staging(&packetization,
                                                            &shape, op)) {
      IREE_RETURN_IF_ERROR(
          loom_vector_packet_canonicalize_snapshots(&packetization, &shape));
      *out_rewritten = true;
      return iree_ok_status();
    }
    if (!loom_vector_packet_static_expansion_is_bounded(&packetization,
                                                        &shape)) {
      return iree_ok_status();
    }
  } else if (!loom_vector_packet_static_operation_count_is_bounded(
                 /*operations_per_chunk=*/2, &shape)) {
    return iree_ok_status();
  }

  loom_value_id_t* result_packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, shape.chunk_count, sizeof(*result_packets),
      (void**)&result_packets));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t table = loom_vector_table_lookup_table(op);
  for (uint32_t chunk_index = 0; chunk_index < shape.chunk_count;
       ++chunk_index) {
    const uint32_t lane_offset = chunk_index * shape.chunk_lane_count;
    const loom_vector_packet_slice_t slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = lane_offset,
        .lane_count =
            iree_min(shape.lane_count - lane_offset, shape.chunk_lane_count),
    };
    loom_value_id_t packet_indices = LOOM_VALUE_ID_INVALID;
    if (producer_selected) {
      loom_vector_packetized_value_t* packetized_indices = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_packet_materialize(
          &packetization, indices, &slice, &packetized_indices));
      packet_indices = packetized_indices->packet;
    } else {
      const int64_t static_offset = lane_offset;
      const loom_type_t packet_index_type =
          loom_vector_packet_type(index_type, slice.lane_count);
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          builder, indices, /*offsets=*/NULL, /*offsets_count=*/0,
          &static_offset, /*static_offsets_count=*/1, packet_index_type,
          op->location, &slice_op));
      packet_indices = loom_vector_slice_result(slice_op);
    }

    const loom_type_t packet_result_type =
        loom_vector_packet_type(result_type, slice.lane_count);
    loom_op_t* packet_lookup_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_table_lookup_build(
        builder, table, packet_indices, packet_result_type, op->location,
        &packet_lookup_op));
    result_packets[chunk_index] =
        loom_vector_table_lookup_result(packet_lookup_op);
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_concat_build(
      builder, /*axis=*/0, result_packets, shape.chunk_count, result_type,
      op->location, &concat_op));
  const loom_value_id_t replacement = loom_vector_concat_result(concat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  if (producer_selected) {
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_erase_dead_sources(&packetization, NULL));
  }
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_legalize_load(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten) {
  *out_rewritten = false;

  loom_vector_memory_footprint_t footprint = {0};
  if (!loom_vector_memory_footprint_describe(
          loom_vector_packet_fact_context(context), context->module, op,
          &footprint) ||
      footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE) {
    return iree_ok_status();
  }
  loom_vector_packet_shape_t shape = {0};
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_packet_shape_from_type(context, policy,
                                          footprint.vector_type, &shape)) {
    return iree_ok_status();
  }
  bool packet_supported = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_memory_shape_constrain(
      context, op, policy, footprint.vector_type, &shape, &packet_supported));
  if (!packet_supported ||
      !loom_vector_memory_cache_policy_from_op(context->module, op,
                                               &cache_policy) ||
      !loom_vector_packet_memory_can_build_chunk_origins(context, &footprint,
                                                         &shape)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_packet_rewrite_memory_load_components(
      context, op, policy, &footprint, &shape, cache_policy, out_rewritten));
  return iree_ok_status();
}

// Stores an exact retained partition without rebuilding and re-slicing its
// logical aggregate. This is both the terminal consumer for mixed native and
// scalar load tails and the ownership boundary that lets dead structural glue
// disappear before target contract verification.
static iree_status_t loom_vector_packet_legalize_partition_store(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy,
    const loom_vector_memory_footprint_t* footprint,
    loom_vector_memory_cache_policy_t cache_policy,
    loom_value_fact_vector_partition_t partition, bool* out_rewritten) {
  *out_rewritten = false;
  uint32_t store_lane_count = 0;
  if (partition.count == 0 ||
      !loom_vector_packet_static_lane_count(footprint->vector_type,
                                            &store_lane_count) ||
      partition.lane_count != store_lane_count) {
    return iree_ok_status();
  }

  iree_host_size_t operation_count = partition.count;
  if (footprint->static_indices.count != 0 &&
      footprint->static_indices
              .i64_array[footprint->static_indices.count - 1u] == INT64_MIN &&
      partition.count > 1) {
    iree_host_size_t origin_operation_count = 0;
    if (!iree_host_size_checked_mul(partition.count - 1u, 2u,
                                    &origin_operation_count) ||
        !iree_host_size_checked_add(operation_count, origin_operation_count,
                                    &operation_count)) {
      return iree_ok_status();
    }
  }
  if (operation_count > LOOM_VECTOR_PACKET_STATIC_OP_LIMIT) {
    return iree_ok_status();
  }

  const loom_type_t element_type =
      loom_type_scalar(loom_type_element_type(footprint->vector_type));
  uint32_t lane_offset = 0;
  for (uint32_t i = 0; i < partition.count; ++i) {
    const loom_value_fact_vector_partition_component_t component =
        partition.components[i];
    const loom_type_t component_type =
        loom_module_value_type(context->module, component.value);
    if (loom_type_is_scalar(component_type)) {
      if (component.lane_count != 1 ||
          !loom_type_equal(component_type, element_type)) {
        return iree_ok_status();
      }
    } else {
      uint32_t component_lane_count = 0;
      if (!loom_vector_packet_static_lane_count(component_type,
                                                &component_lane_count) ||
          component_lane_count != component.lane_count ||
          !loom_type_element_type_equals(component_type,
                                         footprint->vector_type)) {
        return iree_ok_status();
      }
      uint32_t selected_lane_count = 0;
      IREE_RETURN_IF_ERROR(loom_vector_packet_memory_chunk_lane_count_select(
          context, op, policy, footprint->vector_type, component_lane_count,
          &selected_lane_count));
      if (selected_lane_count != component_lane_count) {
        return iree_ok_status();
      }
    }
    if (lane_offset > store_lane_count ||
        component.lane_count > store_lane_count - lane_offset) {
      return iree_ok_status();
    }
    lane_offset += component.lane_count;
  }
  if (lane_offset != store_lane_count) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  lane_offset = 0;
  for (uint32_t i = 0; i < partition.count; ++i) {
    const loom_value_fact_vector_partition_component_t component =
        partition.components[i];
    const loom_type_t component_type =
        loom_module_value_type(context->module, component.value);
    const loom_value_id_t* dynamic_indices = NULL;
    iree_host_size_t dynamic_index_count = 0;
    const int64_t* static_indices = NULL;
    iree_host_size_t static_index_count = 0;
    bool origin_built = false;
    IREE_RETURN_IF_ERROR(loom_vector_packet_memory_build_chunk_origin(
        context, footprint, op, lane_offset, &dynamic_indices,
        &dynamic_index_count, &static_indices, &static_index_count,
        &origin_built));
    IREE_ASSERT_TRUE(origin_built);
    if (loom_type_is_scalar(component_type)) {
      loom_op_t* store_op = NULL;
      IREE_RETURN_IF_ERROR(loom_view_store_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_store_memory_flags(op), component.value, footprint->view,
          dynamic_indices, dynamic_index_count, static_indices,
          static_index_count, cache_policy.cache_scope,
          cache_policy.cache_temporal, op->location, &store_op));
    } else {
      loom_op_t* store_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_store_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_store_memory_flags(op), component.value, footprint->view,
          dynamic_indices, dynamic_index_count, static_indices,
          static_index_count, cache_policy.cache_scope,
          cache_policy.cache_temporal, op->location, &store_op));
    }
    lane_offset += component.lane_count;
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  *out_rewritten = true;
  return iree_ok_status();
}

// Returns true when the exact partition contains a native vector component and
// can remove structural work that the regular snapshot view cannot. Pure
// scalar partitions stay on the ordinary packet path so targets can regroup
// their elements into native memory carriers.
static bool loom_vector_packet_partition_has_distinct_vector_components(
    const loom_module_t* module, loom_value_fact_vector_partition_t partition) {
  if (partition.count == 0) {
    return false;
  }
  bool has_vector_component = false;
  bool differs_from_snapshot = partition.count != partition.snapshot_count;
  for (uint32_t i = 0; i < partition.count; ++i) {
    has_vector_component |= loom_type_is_vector(
        loom_module_value_type(module, partition.components[i].value));
    if (!differs_from_snapshot &&
        (partition.components[i].value !=
             partition.snapshot_components[i].value ||
         partition.components[i].lane_count !=
             partition.snapshot_components[i].lane_count)) {
      differs_from_snapshot = true;
    }
  }
  return has_vector_component && differs_from_snapshot;
}

// Returns true when |root_op| is the only semantic consumer of |value_id|.
// Exact-only partitions have no regular snapshot to enter the packet graph,
// so direct consumption is limited to this ownership shape.
static bool loom_vector_packet_value_is_exclusive_to_op(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_op_t* root_op) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (!value || !loom_value_has_single_use(value) ||
      loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(module, value_id)) {
    return false;
  }
  return loom_use_user_op(*loom_value_single_use(value)) == root_op;
}

// Captured values are sliced from an already materialized target carrier.
// Values without a complete carrier mapping must be legalized before the
// terminal store is reconsidered.
static bool loom_vector_packet_captured_values_have_carriers(
    const loom_vector_packetization_t* packetization) {
  loom_target_legalization_context_t* context = packetization->context;
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    const loom_vector_packetized_value_t* value = &packetization->values[i];
    if (value->source_mode != LOOM_VECTOR_PACKET_SOURCE_MODE_CAPTURED) {
      continue;
    }
    if (!loom_target_legalization_has_source_vector_carrier(
            context, value->source_type)) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_vector_packet_legalize_store(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten) {
  *out_rewritten = false;

  loom_vector_memory_footprint_t store_footprint = {0};
  if (!loom_vector_memory_footprint_describe(
          loom_vector_packet_fact_context(context), context->module, op,
          &store_footprint) ||
      store_footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE) {
    return iree_ok_status();
  }
  loom_vector_packet_shape_t shape = {0};
  if (!loom_vector_packet_shape_from_type(
          context, policy, store_footprint.vector_type, &shape)) {
    return iree_ok_status();
  }
  bool packet_supported = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_memory_shape_constrain(
      context, op, policy, store_footprint.vector_type, &shape,
      &packet_supported));
  if (!packet_supported) {
    return iree_ok_status();
  }

  loom_vector_memory_cache_policy_t store_cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, op,
                                               &store_cache_policy) ||
      !loom_vector_packet_memory_can_build_chunk_origins(
          context, &store_footprint, &shape)) {
    return iree_ok_status();
  }
  const loom_value_fact_vector_partition_t source_partition =
      loom_value_fact_table_query_vector_partition(
          context->rewriter->fact_table, store_footprint.value);
  const bool has_distinct_exact_components =
      loom_vector_packet_partition_has_distinct_vector_components(
          context->module, source_partition);

  loom_vector_packetization_t packetization = {0};
  loom_vector_packetization_initialize(context, policy, &packetization);
  packetization.selection_flags =
      LOOM_VECTOR_PACKET_SELECTION_FLAG_CAPTURE_NON_DECOMPOSABLE;
  packetization.captured_value_location = op->location;
  bool producer_selected = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_select_value_shape(
      &packetization, store_footprint.value, &shape, &producer_selected));
  if (producer_selected) {
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_classify_memory_loads(&packetization, op,
                                                 /*allow_streaming=*/true));
  }

  if (!loom_vector_packet_captured_values_have_carriers(&packetization) ||
      (producer_selected &&
       !loom_vector_packet_can_materialize(&packetization, &shape))) {
    return iree_ok_status();
  }
  if (!producer_selected) {
    if (has_distinct_exact_components && source_partition.snapshot_count == 0 &&
        loom_vector_packet_value_is_exclusive_to_op(
            context->module, store_footprint.value, op)) {
      IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_partition_store(
          context, op, policy, &store_footprint, store_cache_policy,
          source_partition, out_rewritten));
      if (*out_rewritten) {
        return iree_ok_status();
      }
    }
    const loom_value_t* source_value =
        loom_module_value(context->module, store_footprint.value);
    if (!loom_value_is_block_arg(source_value)) {
      const loom_op_t* source_op = loom_value_def_op(source_value);
      loom_target_contract_query_result_t source_result =
          loom_target_contract_query_result_empty();
      IREE_RETURN_IF_ERROR(loom_target_legalization_query_contract(
          context, source_op, &source_result));
      if (source_result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
        return iree_ok_status();
      }
    }
    if (!loom_vector_packet_static_operation_count_is_bounded(
            /*operations_per_chunk=*/2, &shape)) {
      return iree_ok_status();
    }
    // A legal producer already owns its target snapshot. Producer
    // decomposition is only needed to stream oversized computations; native
    // memory packets can consume slices of a captured value without replaying
    // its reads.
    loom_rewriter_t* rewriter = context->rewriter;
    loom_builder_set_before(&rewriter->builder, op);
    IREE_RETURN_IF_ERROR(loom_vector_packet_store_captured_value(
        &packetization, &store_footprint, &shape, store_cache_policy, op));
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
    *out_rewritten = true;
    return iree_ok_status();
  }
  bool source_reads_preserved = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_preserve_memory_loads(
      &packetization, &shape, &source_reads_preserved));
  if (source_reads_preserved) {
    *out_rewritten = true;
    return iree_ok_status();
  }
  if (loom_vector_packet_shared_snapshot_requires_staging(&packetization,
                                                          &shape, op)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_canonicalize_snapshots(&packetization, &shape));
    *out_rewritten = true;
    return iree_ok_status();
  }

  const bool static_expansion_is_bounded =
      loom_vector_packet_static_expansion_is_bounded(&packetization, &shape);
  if (static_expansion_is_bounded && has_distinct_exact_components) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_partition_store(
        context, op, policy, &store_footprint, store_cache_policy,
        source_partition, out_rewritten));
    if (*out_rewritten) {
      IREE_RETURN_IF_ERROR(
          loom_vector_packet_erase_dead_sources(&packetization, NULL));
      return iree_ok_status();
    }
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const bool can_interleave =
      loom_vector_packet_store_can_interleave(&packetization, &store_footprint);
  const bool has_snapshots = loom_vector_packet_has_snapshots(&packetization);
  const bool has_captured_values =
      loom_vector_packet_has_captured_values(&packetization);
  const bool requires_static_slices =
      loom_vector_packet_requires_static_slices(&packetization);
  const bool static_store_required = !can_interleave || has_snapshots ||
                                     has_captured_values ||
                                     requires_static_slices;
  if (static_store_required && static_expansion_is_bounded) {
    if (can_interleave && (has_snapshots || has_captured_values)) {
      IREE_RETURN_IF_ERROR(loom_vector_packet_static_interleaved_store(
          &packetization, &store_footprint, &shape, store_cache_policy, op));
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_packet_static_store(
          &packetization, &store_footprint, &shape, store_cache_policy, op));
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_erase_dead_sources(&packetization, NULL));
    *out_rewritten = true;
    return iree_ok_status();
  }
  if (has_snapshots) {
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_canonicalize_snapshots(&packetization, &shape));
    *out_rewritten = true;
    return iree_ok_status();
  }
  if (has_captured_values || requires_static_slices) {
    return iree_ok_status();
  }
  if (!can_interleave) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_staged_store(
        &packetization, &store_footprint, &shape, store_cache_policy, op));
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_erase_dead_sources(&packetization, NULL));
    *out_rewritten = true;
    return iree_ok_status();
  }

  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const uint32_t loop_lane_count =
      (shape.lane_count / shape.chunk_lane_count) * shape.chunk_lane_count;
  loom_op_t* lower_bound_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(0), index_type, op->location, &lower_bound_op));
  loom_op_t* upper_bound_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(loop_lane_count),
                                index_type, op->location, &upper_bound_op));
  loom_op_t* step_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(shape.chunk_lane_count),
                                index_type, op->location, &step_op));
  loom_op_t* loop_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      builder, /*build_flags=*/0, loom_index_constant_result(lower_bound_op),
      loom_index_constant_result(upper_bound_op),
      loom_index_constant_result(step_op), /*iter_args=*/NULL,
      /*iter_args_count=*/0, /*result_types=*/NULL, /*tied_results=*/NULL,
      /*tied_result_count=*/0,
      /*pipeline_depth=*/LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, op->location, &loop_op));

  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(builder, loop_op, loom_scf_for_body(loop_op));
  const loom_vector_packet_slice_t loop_slice = {
      .dynamic_lane_offset =
          loom_region_entry_arg_id(loom_scf_for_body(loop_op), 0),
      .lane_count = shape.chunk_lane_count,
  };
  loom_vector_packetized_value_t* packetized_value = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_packet_materialize(
      &packetization, store_footprint.value, &loop_slice, &packetized_value));
  IREE_RETURN_IF_ERROR(loom_vector_packet_store(
      &packetization, &store_footprint, &loop_slice, packetized_value->packet,
      store_cache_policy, op));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, NULL, 0, op->location, &yield_op));
  loom_builder_restore(builder, saved_ip);

  if (loop_lane_count < shape.lane_count) {
    const loom_vector_packet_slice_t tail_slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = loop_lane_count,
        .lane_count = shape.lane_count - loop_lane_count,
    };
    IREE_RETURN_IF_ERROR(loom_vector_packet_materialize(
        &packetization, store_footprint.value, &tail_slice, &packetized_value));
    IREE_RETURN_IF_ERROR(loom_vector_packet_store(
        &packetization, &store_footprint, &tail_slice, packetized_value->packet,
        store_cache_policy, op));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_vector_packet_erase_dead_sources(&packetization, NULL));
  *out_rewritten = true;
  return iree_ok_status();
}

// Rewrites selected producer loads at their authored observation points. The
// greedy driver then rebuilds facts and lets the reduction consume the exact
// native component partition on a later visit.
static iree_status_t loom_vector_packet_preserve_reduce_loads(
    loom_target_legalization_context_t* context, loom_op_t* reduce_op,
    const loom_target_vector_packet_policy_t* policy, loom_value_id_t input,
    loom_vector_packet_shape_t shape, bool* out_preserved) {
  *out_preserved = false;
  loom_vector_packetization_t packetization = {0};
  loom_vector_packetization_initialize(context, policy, &packetization);
  packetization.selection_flags =
      LOOM_VECTOR_PACKET_SELECTION_FLAG_CAPTURE_EXTERNAL;
  packetization.captured_value_location = reduce_op->location;
  bool producer_selected = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_select_value_shape(
      &packetization, input, &shape, &producer_selected));
  if (!producer_selected) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_packet_classify_memory_loads(
      &packetization, reduce_op, /*allow_streaming=*/false));
  if (!loom_vector_packet_can_materialize(&packetization, &shape)) {
    return iree_ok_status();
  }
  return loom_vector_packet_preserve_memory_load_components(
      &packetization, &shape, out_preserved);
}

iree_status_t loom_vector_packet_legalize_reduce(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy,
    loom_vector_packet_reduce_result_t* out_result) {
  *out_result = LOOM_VECTOR_PACKET_REDUCE_RESULT_NONE;

  const loom_value_id_t input = loom_vector_reduce_input(op);
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  const loom_value_fact_vector_partition_t input_partition =
      loom_value_fact_table_query_vector_partition(context->fact_table, input);
  loom_vector_packet_partition_reduce_plan_t partition_plan = {0};
  IREE_RETURN_IF_ERROR(loom_vector_packet_partition_reduce_plan_select(
      context, op, policy, input_partition, &partition_plan));
  if (partition_plan.mode != LOOM_VECTOR_PACKET_PARTITION_REDUCE_MODE_NONE) {
    loom_rewriter_t* rewriter = context->rewriter;
    loom_builder_set_before(&rewriter->builder, op);
    const loom_value_id_t value_checkpoint =
        loom_rewriter_value_checkpoint(rewriter);
    const loom_value_t* input_value = loom_module_value(context->module, input);
    loom_op_t* input_op =
        input_value != NULL && !loom_value_is_block_arg(input_value)
            ? loom_value_def_op(input_value)
            : NULL;
    loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_packet_partition_reduce(
        context, &partition_plan, input_partition, op, &accumulator));
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        rewriter, op, &accumulator, 1, value_checkpoint));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, op, &accumulator, 1));
    if (input_op != NULL) {
      bool erased = false;
      IREE_RETURN_IF_ERROR(
          loom_rewriter_erase_if_dead(rewriter, input_op, &erased));
    }
    *out_result = LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
    return iree_ok_status();
  }
  const bool authored_reduction_is_legal =
      context->contract_query_result != NULL &&
      context->contract_query_result->outcome ==
          LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  if (authored_reduction_is_legal) {
    *out_result = LOOM_VECTOR_PACKET_REDUCE_RESULT_RETAIN_NATIVE;
  }
  loom_vector_packet_shape_t shape = {0};
  const bool requires_packet_shape =
      loom_vector_packet_shape_from_type(context, policy, input_type, &shape);
  if (!requires_packet_shape) {
    return iree_ok_status();
  }
  bool packet_reduction_supported = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_reduce_shape_constrain(
      context, op, policy, input_type, &shape, &packet_reduction_supported));
  loom_vector_packet_shape_t combining_shape = shape;
  bool packet_combining_supported = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_reduce_combine_shape_constrain(
      context, op, policy, input_type, &combining_shape,
      &packet_combining_supported));
  if (packet_reduction_supported || packet_combining_supported) {
    const loom_vector_packet_shape_t component_shape =
        packet_combining_supported ? combining_shape : shape;
    bool source_reads_preserved = false;
    IREE_RETURN_IF_ERROR(loom_vector_packet_preserve_reduce_loads(
        context, op, policy, input, component_shape, &source_reads_preserved));
    if (source_reads_preserved) {
      *out_result = LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
      return iree_ok_status();
    }
  }
  if (authored_reduction_is_legal) {
    return iree_ok_status();
  }
  if (!packet_reduction_supported) {
    // Keep the reduction and its decomposable producer graph under one plan.
    // A scalar-width packet lets the same graph materializer rebuild source
    // lanes without first constructing a native vector that the reduction
    // would immediately extract again.
    shape.chunk_lane_count = 1;
    shape.chunk_count = shape.lane_count;
  }

  loom_vector_packetization_t packetization = {0};
  loom_vector_packetization_initialize(context, policy, &packetization);
  packetization.selection_flags =
      LOOM_VECTOR_PACKET_SELECTION_FLAG_CAPTURE_EXTERNAL;
  packetization.captured_value_location = op->location;
  bool producer_selected = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_select_value_shape(
      &packetization, input, &shape, &producer_selected));
  if (!producer_selected) {
    if (authored_reduction_is_legal) {
      return iree_ok_status();
    }
    iree_host_size_t operations_per_chunk = 0;
    if (!iree_host_size_checked_mul(shape.chunk_lane_count, 2u,
                                    &operations_per_chunk) ||
        !loom_vector_packet_static_operation_count_is_bounded(
            operations_per_chunk, &shape)) {
      return iree_ok_status();
    }
    *out_result = LOOM_VECTOR_PACKET_REDUCE_RESULT_CAPTURE_INPUT;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_vector_packet_classify_memory_loads(&packetization, op,
                                               /*allow_streaming=*/true));
  if (!loom_vector_packet_can_materialize(&packetization, &shape)) {
    return iree_ok_status();
  }
  bool source_reads_preserved = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_preserve_memory_loads(
      &packetization, &shape, &source_reads_preserved));
  if (source_reads_preserved) {
    *out_result = LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
    return iree_ok_status();
  }
  if (loom_vector_packet_shared_snapshot_requires_staging(&packetization,
                                                          &shape, op)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_canonicalize_snapshots(&packetization, &shape));
    *out_result = LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const bool has_snapshots = loom_vector_packet_has_snapshots(&packetization);
  const bool has_captured_values =
      loom_vector_packet_has_captured_values(&packetization);
  const bool requires_static_slices =
      loom_vector_packet_requires_static_slices(&packetization);
  if ((!packet_reduction_supported || has_snapshots || has_captured_values ||
       requires_static_slices) &&
      loom_vector_packet_static_expansion_is_bounded(&packetization, &shape)) {
    loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_packet_static_reduce(
        &packetization, input, &shape, op,
        /*fold_packet_inputs=*/packet_reduction_supported, &accumulator));
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        rewriter, op, &accumulator, 1, value_checkpoint));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, op, &accumulator, 1));
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_erase_dead_sources(&packetization, NULL));
    *out_result = LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
    return iree_ok_status();
  }
  if (has_snapshots) {
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_canonicalize_snapshots(&packetization, &shape));
    *out_result = LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
    return iree_ok_status();
  }
  if (requires_static_slices) {
    return iree_ok_status();
  }
  const loom_combining_kind_t kind = loom_vector_reduce_kind(op);
  const uint8_t fastmath_flags = loom_vector_reduce_fastmath(op);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_reduce_result(op));
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const uint32_t loop_lane_count =
      (shape.lane_count / shape.chunk_lane_count) * shape.chunk_lane_count;
  loom_op_t* lower_bound_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(0), index_type, op->location, &lower_bound_op));
  loom_op_t* upper_bound_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(loop_lane_count),
                                index_type, op->location, &upper_bound_op));
  loom_op_t* step_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(shape.chunk_lane_count),
                                index_type, op->location, &step_op));
  const loom_value_id_t initial_accumulator = loom_vector_reduce_init(op);
  loom_op_t* loop_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      builder, /*build_flags=*/0, loom_index_constant_result(lower_bound_op),
      loom_index_constant_result(upper_bound_op),
      loom_index_constant_result(step_op), &initial_accumulator, 1,
      /*result_types=*/NULL,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*pipeline_depth=*/LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, op->location, &loop_op));

  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(builder, loop_op, loom_scf_for_body(loop_op));
  const loom_vector_packet_slice_t loop_slice = {
      .dynamic_lane_offset =
          loom_region_entry_arg_id(loom_scf_for_body(loop_op), 0),
      .lane_count = shape.chunk_lane_count,
  };
  loom_vector_packetized_value_t* packetized_input = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_packet_materialize(
      &packetization, input, &loop_slice, &packetized_input));
  loom_op_t* packet_reduce_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
      builder, kind, fastmath_flags, packetized_input->packet,
      loom_region_entry_arg_id(loom_scf_for_body(loop_op), 1), result_type,
      op->location, &packet_reduce_op));
  const loom_value_id_t next_accumulator =
      loom_vector_reduce_result(packet_reduce_op);
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &next_accumulator, 1,
                                            op->location, &yield_op));
  loom_builder_restore(builder, saved_ip);

  loom_value_id_t accumulator = loom_scf_for_results(loop_op).values[0];
  if (loop_lane_count < shape.lane_count) {
    const loom_vector_packet_slice_t tail_slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = loop_lane_count,
        .lane_count = shape.lane_count - loop_lane_count,
    };
    IREE_RETURN_IF_ERROR(loom_vector_packet_materialize(
        &packetization, input, &tail_slice, &packetized_input));
    loom_op_t* tail_reduce_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
        builder, kind, fastmath_flags, packetized_input->packet, accumulator,
        result_type, op->location, &tail_reduce_op));
    accumulator = loom_vector_reduce_result(tail_reduce_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &accumulator, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &accumulator, 1));
  IREE_RETURN_IF_ERROR(
      loom_vector_packet_erase_dead_sources(&packetization, NULL));
  *out_result = LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
  return iree_ok_status();
}
