// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/loop_like.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/value_replacement.h"
#include "loom/rewrite/remap.h"

typedef struct loom_loop_like_value_range_t {
  // First replacement ordinal.
  uint16_t offset;
  // Number of replacement values.
  uint16_t count;
} loom_loop_like_value_range_t;

static loom_loop_like_value_range_t loom_loop_like_source_range(
    const uint16_t* source_offsets, uint16_t source_ordinal) {
  const uint16_t begin = source_offsets[source_ordinal];
  const uint16_t end = source_offsets[source_ordinal + 1];
  return (loom_loop_like_value_range_t){
      .offset = begin,
      .count = (uint16_t)(end - begin),
  };
}

static uint16_t loom_loop_like_iter_args_operand_offset(loom_loop_like_t loop) {
  if (!loop.vtable->segmented_operands) {
    return loop.vtable->iter_args_operand_field_index;
  }
  const uint16_t* segment_counts =
      loom_op_const_operand_segment_counts(loop.op);
  uint32_t offset = 0;
  for (uint8_t i = 0; i < loop.vtable->iter_args_operand_field_index; ++i) {
    offset += segment_counts[i];
  }
  return (uint16_t)offset;
}

static iree_status_t loom_loop_like_validate_replacement_domain(
    iree_string_view_t name, uint16_t source_count, uint16_t target_count,
    const uint16_t* source_offsets) {
  if (source_count == 0) {
    if (target_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "an empty source %.*s domain cannot acquire replacement values",
          (int)name.size, name.data);
    }
    return iree_ok_status();
  }
  if (!source_offsets || source_offsets[0] != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source %.*s offsets must begin at replacement ordinal zero",
        (int)name.size, name.data);
  }
  for (uint16_t i = 0; i < source_count; ++i) {
    if (source_offsets[i] > source_offsets[i + 1]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source %.*s offsets must be monotonically increasing",
          (int)name.size, name.data);
    }
  }
  if (source_offsets[source_count] != target_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source %.*s offsets end at %u but the replacement has %u values",
        (int)name.size, name.data, (unsigned)source_offsets[source_count],
        (unsigned)target_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_like_validate_replacement_state(
    loom_loop_like_t source, const loom_loop_like_replacement_state_t* state) {
  if (!state) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replacement state is NULL");
  }
  const uint16_t source_header_count = loom_loop_like_iter_args(source).count;
  const uint16_t source_result_count = source.op->result_count;
  const uint16_t target_header_count = state->initial_values.count;
  if (target_header_count != 0 && !state->initial_values.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replacement header values are NULL");
  }
  if (state->result_count != 0 && !state->result_types) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replacement result types are NULL");
  }
  if (loom_loop_like_condition_region(source) && target_header_count != 0 &&
      !state->header_types) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replacement header types are NULL");
  }
  IREE_RETURN_IF_ERROR(loom_loop_like_validate_replacement_domain(
      IREE_SV("header"), source_header_count, target_header_count,
      state->source_header_offsets));
  IREE_RETURN_IF_ERROR(loom_loop_like_validate_replacement_domain(
      IREE_SV("result"), source_result_count, state->result_count,
      state->source_result_offsets));

  if (!loom_loop_like_condition_region(source)) {
    if (source_header_count != source_result_count ||
        target_header_count != state->result_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "counted loop header and result domains must have equal arity");
    }
    if (source_header_count != 0) {
      for (uint16_t i = 0; i <= source_header_count; ++i) {
        if (state->source_header_offsets[i] !=
            state->source_result_offsets[i]) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "counted loop header and result mappings must match");
        }
      }
    }
  }
  return iree_ok_status();
}

static loom_loop_like_value_range_t loom_loop_like_target_operand_range(
    uint16_t source_operand, uint16_t source_state_operand_offset,
    uint16_t source_state_count, uint16_t target_state_count,
    const uint16_t* source_header_offsets) {
  if (source_operand >= source_state_operand_offset &&
      source_operand < source_state_operand_offset + source_state_count) {
    const uint16_t source_ordinal =
        (uint16_t)(source_operand - source_state_operand_offset);
    loom_loop_like_value_range_t range =
        loom_loop_like_source_range(source_header_offsets, source_ordinal);
    range.offset = (uint16_t)(source_state_operand_offset + range.offset);
    return range;
  }
  if (source_operand < source_state_operand_offset) {
    return (loom_loop_like_value_range_t){
        .offset = source_operand,
        .count = 1,
    };
  }
  return (loom_loop_like_value_range_t){
      .offset =
          (uint16_t)(source_operand + target_state_count - source_state_count),
      .count = 1,
  };
}

static iree_status_t loom_loop_like_plan_tied_results(
    loom_loop_like_t source, const loom_loop_like_replacement_state_t* state,
    uint16_t source_state_operand_offset, iree_arena_allocator_t* scratch_arena,
    loom_tied_result_t** out_tied_results, uint16_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  const uint16_t source_header_count = loom_loop_like_iter_args(source).count;
  const uint16_t target_header_count = state->initial_values.count;
  const loom_tied_result_t* source_ties = loom_op_tied_results(source.op);
  uint32_t target_tie_count = 0;
  for (uint16_t i = 0; i < source.op->tied_result_count; ++i) {
    IREE_ASSERT_LT(source_ties[i].result_index, source.op->result_count);
    IREE_ASSERT_LT(source_ties[i].operand_index, source.op->operand_count);
    const loom_loop_like_value_range_t result_range =
        loom_loop_like_source_range(state->source_result_offsets,
                                    source_ties[i].result_index);
    const loom_loop_like_value_range_t operand_range =
        loom_loop_like_target_operand_range(
            source_ties[i].operand_index, source_state_operand_offset,
            source_header_count, target_header_count,
            state->source_header_offsets);
    if (result_range.count != operand_range.count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot preserve source result %u tie: replacement result and "
          "operand ranges have %u and %u values",
          (unsigned)source_ties[i].result_index, (unsigned)result_range.count,
          (unsigned)operand_range.count);
    }
    target_tie_count += result_range.count;
  }
  if (target_tie_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "replacement loop requires %u tied results",
                            (unsigned)target_tie_count);
  }
  if (target_tie_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* target_ties = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, target_tie_count,
                                sizeof(*target_ties), (void**)&target_ties));
  uint16_t target_tie_ordinal = 0;
  for (uint16_t i = 0; i < source.op->tied_result_count; ++i) {
    const loom_loop_like_value_range_t result_range =
        loom_loop_like_source_range(state->source_result_offsets,
                                    source_ties[i].result_index);
    const loom_loop_like_value_range_t operand_range =
        loom_loop_like_target_operand_range(
            source_ties[i].operand_index, source_state_operand_offset,
            source_header_count, target_header_count,
            state->source_header_offsets);
    for (uint16_t j = 0; j < result_range.count; ++j) {
      target_ties[target_tie_ordinal++] = (loom_tied_result_t){
          .result_index = (uint16_t)(result_range.offset + j),
          .operand_index = (uint16_t)(operand_range.offset + j),
          .has_type_change = source_ties[i].has_type_change,
      };
    }
  }
  IREE_ASSERT_EQ(target_tie_ordinal, target_tie_count);
  *out_tied_results = target_ties;
  *out_tied_result_count = (uint16_t)target_tie_count;
  return iree_ok_status();
}

static iree_status_t loom_loop_like_copy_block_presentation(
    loom_module_t* module, const loom_block_t* source, loom_block_t* target) {
  target->label_id = source->label_id;
  target->flags = source->flags;
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_block_comments(module, source, &comment_count);
  return comment_count == 0 ? iree_ok_status()
                            : loom_module_attach_block_comments(
                                  module, target, comments, comment_count);
}

static iree_status_t loom_loop_like_create_region(
    loom_builder_t* builder, loom_op_t* target, uint8_t region_index,
    const loom_region_t* source_region, loom_block_t** out_entry) {
  IREE_RETURN_IF_ERROR(
      loom_builder_create_region(builder, target, region_index, out_entry));
  loom_region_t* target_region = loom_op_regions(target)[region_index];
  target_region->flags = source_region->flags;
  target_region->source_flags = source_region->source_flags;
  return loom_loop_like_copy_block_presentation(
      builder->module, loom_region_const_entry_block(source_region),
      *out_entry);
}

static iree_status_t loom_loop_like_define_fresh_state_arguments(
    loom_builder_t* builder, loom_block_t* block,
    const loom_type_t* state_types, uint16_t state_count,
    loom_value_slice_t* out_state) {
  const uint16_t state_offset = block->arg_count;
  for (uint16_t i = 0; i < state_count; ++i) {
    loom_value_id_t argument = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, block, state_types[i], &argument));
  }
  *out_state = (loom_value_slice_t){
      .values = state_count == 0 ? NULL : block->arg_ids + state_offset,
      .count = state_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_loop_like_define_reserved_state_arguments(
    loom_builder_t* builder, loom_block_t* block,
    const loom_type_t* state_types, uint16_t state_count,
    loom_value_slice_t* out_state) {
  const uint16_t state_offset = block->arg_count;
  for (uint16_t i = 0; i < state_count; ++i) {
    loom_value_id_t argument = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_builder_define_value(builder, state_types[i], &argument));
    IREE_RETURN_IF_ERROR(loom_block_add_arg(builder->module, block, argument));
  }
  *out_state = (loom_value_slice_t){
      .values = state_count == 0 ? NULL : block->arg_ids + state_offset,
      .count = state_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_loop_like_copy_one_to_one_names(
    loom_module_t* module, const uint16_t* source_offsets,
    loom_value_slice_t source_values, loom_value_slice_t target_values) {
  for (uint16_t i = 0; i < source_values.count; ++i) {
    const loom_loop_like_value_range_t range =
        loom_loop_like_source_range(source_offsets, i);
    if (range.count == 1) {
      IREE_RETURN_IF_ERROR(loom_module_copy_value_name(
          module, source_values.values[i], target_values.values[range.offset]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_like_copy_names(
    loom_builder_t* builder, loom_loop_like_t source,
    const loom_loop_like_replacement_state_t* state,
    const loom_loop_like_replacement_t* replacement) {
  IREE_RETURN_IF_ERROR(loom_loop_like_copy_one_to_one_names(
      builder->module, state->source_result_offsets,
      (loom_value_slice_t){
          .values = loom_op_results(source.op),
          .count = source.op->result_count,
      },
      replacement->results));
  const loom_block_t* source_body =
      loom_region_const_entry_block(loom_loop_like_body(source));
  const uint16_t source_iv_count =
      loom_loop_like_iv(source) == LOOM_VALUE_ID_INVALID ? 0 : 1;
  if (source_iv_count != 0) {
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(
        builder->module, loom_block_arg_id(source_body, 0),
        loom_block_arg_id(replacement->body_entry, 0)));
  }
  IREE_RETURN_IF_ERROR(loom_loop_like_copy_one_to_one_names(
      builder->module, state->source_result_offsets,
      (loom_value_slice_t){
          .values = source_body->arg_count == source_iv_count
                        ? NULL
                        : source_body->arg_ids + source_iv_count,
          .count = (uint16_t)(source_body->arg_count - source_iv_count),
      },
      replacement->body_state));
  const loom_region_t* source_condition =
      loom_loop_like_condition_region(source);
  if (source_condition) {
    const loom_block_t* source_condition_entry =
        loom_region_const_entry_block(source_condition);
    IREE_RETURN_IF_ERROR(loom_loop_like_copy_one_to_one_names(
        builder->module, state->source_header_offsets,
        (loom_value_slice_t){
            .values = source_condition_entry->arg_ids,
            .count = source_condition_entry->arg_count,
        },
        replacement->condition_state));
  }
  return iree_ok_status();
}

iree_status_t loom_loop_like_build_replacement(
    loom_builder_t* builder, loom_loop_like_t source,
    const loom_loop_like_replacement_state_t* state,
    iree_arena_allocator_t* scratch_arena,
    loom_loop_like_replacement_t* out_replacement) {
  *out_replacement = (loom_loop_like_replacement_t){0};
  if (!builder || !builder->module || !builder->arena || !scratch_arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replacement requires a builder and scratch arena");
  }
  if (!loom_loop_like_isa(source)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source operation is not LoopLike");
  }
  IREE_RETURN_IF_ERROR(
      loom_loop_like_validate_replacement_state(source, state));

  const loom_op_vtable_t* vtable = loom_op_vtable(builder->module, source.op);
  IREE_ASSERT(vtable && vtable->loop_like == source.vtable);
  const loom_value_slice_t source_header = loom_loop_like_iter_args(source);
  const uint16_t source_state_operand_offset =
      loom_loop_like_iter_args_operand_offset(source);
  const uint16_t target_header_count = state->initial_values.count;
  const uint16_t target_result_count = state->result_count;
  const uint32_t target_operand_count_32 = (uint32_t)source.op->operand_count -
                                           source_header.count +
                                           target_header_count;
  if (target_operand_count_32 > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "replacement loop requires %u operands",
                            (unsigned)target_operand_count_32);
  }
  const uint16_t target_operand_count = (uint16_t)target_operand_count_32;

  uint16_t* segment_counts = NULL;
  const uint8_t segment_count = loom_op_vtable_operand_segment_count(vtable);
  if (segment_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, segment_count,
                                                   sizeof(*segment_counts),
                                                   (void**)&segment_counts));
    memcpy(segment_counts, loom_op_const_operand_segment_counts(source.op),
           segment_count * sizeof(*segment_counts));
    segment_counts[source.vtable->iter_args_operand_field_index] =
        target_header_count;
  }

  loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_loop_like_plan_tied_results(
      source, state, source_state_operand_offset, scratch_arena, &tied_results,
      &tied_result_count));

  loom_op_t* target = NULL;
  if (segment_count != 0) {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op_with_successors(
        builder, source.op->kind, target_operand_count, segment_counts,
        segment_count, target_result_count, source.op->successor_count,
        source.op->region_count, tied_result_count, source.op->attribute_count,
        source.op->location, &target));
  } else {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op_with_successors(
        builder, source.op->kind, target_operand_count, target_result_count,
        source.op->successor_count, source.op->region_count, tied_result_count,
        source.op->attribute_count, source.op->location, &target));
  }
  target->instance_flags = source.op->instance_flags;
  target->traits = source.op->traits;
  target->intrinsic_read_spaces = source.op->intrinsic_read_spaces;
  target->intrinsic_write_spaces = source.op->intrinsic_write_spaces;
  target->flags |= source.op->flags & LOOM_OP_SOURCE_PRESENTATION_FLAG_MASK;

  loom_value_id_t* target_operands = loom_op_operands(target);
  const loom_value_id_t* source_operands = loom_op_const_operands(source.op);
  if (source_state_operand_offset != 0) {
    memcpy(target_operands, source_operands,
           source_state_operand_offset * sizeof(*target_operands));
  }
  if (target_header_count != 0) {
    memcpy(target_operands + source_state_operand_offset,
           state->initial_values.values,
           target_header_count * sizeof(*target_operands));
  }
  const uint16_t source_suffix_offset =
      (uint16_t)(source_state_operand_offset + source_header.count);
  const uint16_t source_suffix_count =
      (uint16_t)(source.op->operand_count - source_suffix_offset);
  if (source_suffix_count != 0) {
    memcpy(target_operands + source_state_operand_offset + target_header_count,
           source_operands + source_suffix_offset,
           source_suffix_count * sizeof(*target_operands));
  }
  if (source.op->successor_count != 0) {
    memcpy(loom_op_successors(target), loom_op_const_successors(source.op),
           source.op->successor_count * sizeof(loom_block_t*));
  }

  const uint8_t body_region_index = source.vtable->body_region_index;
  const uint8_t condition_region_index = source.vtable->condition_region_index;
  IREE_ASSERT_EQ(source.op->region_count,
                 condition_region_index == LOOM_REGION_INDEX_NONE ? 1 : 2);
  for (uint8_t i = 0; i < source.op->region_count; ++i) {
    loom_block_t* entry = NULL;
    IREE_RETURN_IF_ERROR(loom_loop_like_create_region(
        builder, target, i, loom_op_regions(source.op)[i], &entry));
    if (i == body_region_index) {
      out_replacement->body_entry = entry;
    } else {
      IREE_ASSERT_EQ(i, condition_region_index);
      out_replacement->condition_entry = entry;
    }
  }

  if (loom_loop_like_iv(source) != LOOM_VALUE_ID_INVALID) {
    const loom_type_t iv_type =
        loom_module_value_type(builder->module, loom_loop_like_iv(source));
    loom_value_id_t target_iv = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, out_replacement->body_entry, iv_type, &target_iv));
  }
  IREE_RETURN_IF_ERROR(loom_loop_like_define_fresh_state_arguments(
      builder, out_replacement->body_entry, state->result_types,
      target_result_count, &out_replacement->body_state));
  if (out_replacement->condition_entry) {
    IREE_RETURN_IF_ERROR(loom_loop_like_define_reserved_state_arguments(
        builder, out_replacement->condition_entry, state->header_types,
        target_header_count, &out_replacement->condition_state));
  }

  for (uint16_t i = 0; i < target_result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_result(
        builder, state->result_types[i], &loom_op_results(target)[i]));
  }

  out_replacement->results = (loom_value_slice_t){
      .values = loom_op_results(target),
      .count = target_result_count,
  };
  if (target_result_count != 0) {
    IREE_RETURN_IF_ERROR(loom_module_assign_value_types(
        builder->module, out_replacement->results.values,
        out_replacement->body_state.values, target_result_count));
  }
  out_replacement->loop = (loom_loop_like_t){
      .op = target,
      .vtable = source.vtable,
  };
  if (tied_result_count != 0) {
    memcpy(loom_op_tied_results(target), tied_results,
           tied_result_count * sizeof(*tied_results));
  }
  if (source.op->attribute_count != 0) {
    memcpy(loom_op_attrs(target), loom_op_const_attrs(source.op),
           source.op->attribute_count * sizeof(loom_attribute_t));
  }
  IREE_RETURN_IF_ERROR(
      loom_loop_like_copy_names(builder, source, state, out_replacement));
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, target));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(builder->module, source.op, &comment_count);
  if (comment_count != 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_op_comments(
        builder->module, target, comments, comment_count));
  }
  return iree_ok_status();
}
