// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/accumulator.h"

#include <inttypes.h>
#include <string.h>

#include "loom/format/text/parser/context.h"

//===----------------------------------------------------------------------===//
// Op accumulator
//===----------------------------------------------------------------------===//

void loom_parsed_op_initialize(loom_parsed_op_t* parsed) {
  memset(parsed, 0, sizeof(*parsed));
  parsed->operand_ids = parsed->inline_operand_ids;
  parsed->operand_capacity = LOOM_PARSED_OP_INLINE_OPERANDS;
  parsed->successors = parsed->inline_successors;
  parsed->successor_label_tokens = parsed->inline_successor_label_tokens;
  parsed->successor_capacity = LOOM_PARSED_OP_INLINE_SUCCESSORS;
  parsed->result_ids = parsed->inline_result_ids;
  parsed->result_name_tokens = parsed->inline_result_name_tokens;
  parsed->result_capacity = LOOM_PARSED_OP_INLINE_RESULTS;
  parsed->attributes = parsed->inline_attributes;
  parsed->attribute_capacity = LOOM_PARSED_OP_INLINE_ATTRS;
  parsed->regions = parsed->inline_regions;
  parsed->region_capacity = LOOM_PARSED_OP_INLINE_REGIONS;
  parsed->tied_results = parsed->inline_tied_results;
  parsed->tied_result_capacity = LOOM_PARSED_OP_INLINE_TIED;
  parsed->field_spans = parsed->inline_field_spans;
  parsed->field_span_capacity = LOOM_PARSED_OP_INLINE_FIELD_SPANS;
  parsed->operand_segment_counts = parsed->inline_operand_segment_counts;
  parsed->operand_segment_capacity = LOOM_PARSED_OP_INLINE_OPERAND_SEGMENTS;
}

void loom_parsed_op_reset(loom_parsed_op_t* parsed) {
  parsed->operand_count = 0;
  parsed->successor_count = 0;
  parsed->result_count = 0;
  parsed->tied_result_count = 0;
  parsed->field_span_count = 0;
  parsed->attribute_count = 0;
  parsed->region_count = 0;
  parsed->operand_segment_count = 0;
  parsed->instance_flags = 0;
}

iree_status_t loom_parser_acquire_parsed_op(loom_parser_t* parser,
                                            loom_parsed_op_t** out_parsed) {
  loom_parsed_op_t* parsed = parser->parsed_op_free_list;
  if (parsed) {
    parser->parsed_op_free_list = parsed->next_free;
    parsed->next_free = NULL;
    loom_parsed_op_reset(parsed);
    *out_parsed = parsed;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate(&parser->parser_arena,
                                           sizeof(*parsed), (void**)&parsed));
  loom_parsed_op_initialize(parsed);
  *out_parsed = parsed;
  return iree_ok_status();
}

void loom_parser_release_parsed_op(loom_parser_t* parser,
                                   loom_parsed_op_t* parsed) {
  if (!parsed) {
    return;
  }
  parsed->next_free = parser->parsed_op_free_list;
  parser->parsed_op_free_list = parsed;
}

static iree_status_t loom_parser_grow_bounded_array(
    iree_arena_allocator_t* arena, iree_host_size_t existing_count,
    iree_host_size_t minimum_capacity, iree_host_size_t element_size,
    iree_host_size_t maximum_capacity, const char* storage_name,
    iree_host_size_t* inout_capacity, void** inout_ptr) {
  if (minimum_capacity > maximum_capacity ||
      existing_count > maximum_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "%s count exceeds storage limit (%" PRIhsz ")",
                            storage_name, maximum_capacity);
  }

  iree_host_size_t doubled_capacity = 0;
  if (!iree_host_size_checked_mul(*inout_capacity, 2, &doubled_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "%s capacity overflow",
                            storage_name);
  }
  iree_host_size_t new_capacity =
      doubled_capacity > minimum_capacity ? doubled_capacity : minimum_capacity;
  if (new_capacity > maximum_capacity) {
    new_capacity = maximum_capacity;
  }

  void* new_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, new_capacity, element_size, &new_ptr));
  if (*inout_ptr && existing_count > 0) {
    memcpy(new_ptr, *inout_ptr, existing_count * element_size);
  }
  *inout_ptr = new_ptr;
  *inout_capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_add_operand(loom_parsed_op_t* parsed,
                                         iree_arena_allocator_t* arena,
                                         loom_value_id_t value_id) {
  if (parsed->operand_count >= parsed->operand_capacity) {
    iree_host_size_t capacity = parsed->operand_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->operand_count,
        (iree_host_size_t)parsed->operand_count + 1, sizeof(loom_value_id_t),
        UINT16_MAX, "parsed op operand", &capacity,
        (void**)&parsed->operand_ids));
    parsed->operand_capacity = (uint16_t)capacity;
  }
  parsed->operand_ids[parsed->operand_count++] = value_id;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_set_operand(loom_parsed_op_t* parsed,
                                         iree_arena_allocator_t* arena,
                                         uint16_t index,
                                         loom_value_id_t value_id) {
  iree_host_size_t required_capacity = (iree_host_size_t)index + 1;
  if (required_capacity > parsed->operand_capacity) {
    iree_host_size_t capacity = parsed->operand_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->operand_count, required_capacity,
        sizeof(loom_value_id_t), UINT16_MAX, "parsed op operand", &capacity,
        (void**)&parsed->operand_ids));
    parsed->operand_capacity = (uint16_t)capacity;
  }
  while (parsed->operand_count <= index) {
    parsed->operand_ids[parsed->operand_count++] = LOOM_VALUE_ID_INVALID;
  }
  parsed->operand_ids[index] = value_id;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_prepare_operand_segments(
    loom_parsed_op_t* parsed, iree_arena_allocator_t* arena,
    uint8_t segment_count) {
  if (segment_count > parsed->operand_segment_capacity) {
    iree_host_size_t capacity = parsed->operand_segment_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->operand_segment_count, segment_count, sizeof(uint16_t),
        UINT8_MAX, "parsed op operand segment", &capacity,
        (void**)&parsed->operand_segment_counts));
    parsed->operand_segment_capacity = (uint8_t)capacity;
  }
  parsed->operand_segment_count = segment_count;
  memset(parsed->operand_segment_counts, 0,
         (iree_host_size_t)segment_count * sizeof(uint16_t));
  return iree_ok_status();
}

iree_status_t loom_parsed_op_add_segmented_operand(
    loom_parsed_op_t* parsed, iree_arena_allocator_t* arena,
    uint8_t segment_index, loom_value_id_t value_id,
    uint16_t* out_operand_index) {
  if (segment_index >= parsed->operand_segment_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operand segment index %u is out of range for %u segment(s)",
        segment_index, parsed->operand_segment_count);
  }
  if (parsed->operand_segment_counts[segment_index] == UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "parsed operand segment %u exceeds storage limit",
                            segment_index);
  }
  *out_operand_index = parsed->operand_count;
  IREE_RETURN_IF_ERROR(loom_parsed_op_add_operand(parsed, arena, value_id));
  ++parsed->operand_segment_counts[segment_index];
  return iree_ok_status();
}

iree_status_t loom_parsed_op_set_successor(loom_parsed_op_t* parsed,
                                           iree_arena_allocator_t* arena,
                                           uint8_t index, loom_block_t* block,
                                           loom_token_t label_token) {
  if (index == UINT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "successor field index exceeds storage limit");
  }
  iree_host_size_t required_capacity = (iree_host_size_t)index + 1;
  if (required_capacity > parsed->successor_capacity) {
    iree_host_size_t capacity = parsed->successor_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->successor_count, required_capacity,
        sizeof(loom_block_t*), UINT8_MAX, "parsed op successor", &capacity,
        (void**)&parsed->successors));
    iree_host_size_t token_capacity = parsed->successor_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->successor_count, capacity, sizeof(loom_token_t),
        UINT8_MAX, "parsed op successor label", &token_capacity,
        (void**)&parsed->successor_label_tokens));
    IREE_ASSERT(token_capacity == capacity);
    parsed->successor_capacity = (uint8_t)capacity;
  }
  while (parsed->successor_count <= index) {
    parsed->successors[parsed->successor_count] = NULL;
    parsed->successor_label_tokens[parsed->successor_count] = loom_token_none();
    ++parsed->successor_count;
  }
  parsed->successors[index] = block;
  parsed->successor_label_tokens[index] = label_token;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_add_result(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        loom_value_id_t value_id,
                                        loom_token_t name_token) {
  if (parsed->result_count == UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "parsed op result count exceeds storage limit "
                            "(%u)",
                            (unsigned)UINT16_MAX);
  }
  IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
      parsed, arena, LOOM_LOCATION_FIELD_RESULT, parsed->result_count,
      name_token, name_token.line, name_token.end_column));
  if (parsed->result_count >= parsed->result_capacity) {
    iree_host_size_t capacity = parsed->result_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->result_count, (iree_host_size_t)parsed->result_count + 1,
        sizeof(loom_value_id_t), UINT16_MAX, "parsed op result", &capacity,
        (void**)&parsed->result_ids));
    iree_host_size_t name_token_capacity = parsed->result_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->result_count, capacity, sizeof(loom_token_t), UINT16_MAX,
        "parsed op result name", &name_token_capacity,
        (void**)&parsed->result_name_tokens));
    IREE_ASSERT(name_token_capacity == capacity);
    parsed->result_capacity = (uint16_t)capacity;
  }
  parsed->result_ids[parsed->result_count] = value_id;
  parsed->result_name_tokens[parsed->result_count] = name_token;
  ++parsed->result_count;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_set_attribute(loom_parsed_op_t* parsed,
                                           iree_arena_allocator_t* arena,
                                           uint8_t field_index,
                                           loom_attribute_t attr) {
  if (field_index == UINT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "parsed op attribute index exceeds storage limit "
                            "(%u)",
                            (unsigned)(UINT8_MAX - 1));
  }
  if (field_index >= parsed->attribute_capacity) {
    iree_host_size_t capacity = parsed->attribute_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->attribute_count, (iree_host_size_t)field_index + 1,
        sizeof(loom_attribute_t), UINT8_MAX, "parsed op attribute", &capacity,
        (void**)&parsed->attributes));
    // Zero newly exposed slots so unset attributes read as zero.
    memset(&parsed->attributes[parsed->attribute_count], 0,
           (capacity - parsed->attribute_count) * sizeof(loom_attribute_t));
    parsed->attribute_capacity = (uint8_t)capacity;
  }
  if (field_index > parsed->attribute_count) {
    memset(&parsed->attributes[parsed->attribute_count], 0,
           (field_index - parsed->attribute_count) * sizeof(loom_attribute_t));
  }
  parsed->attributes[field_index] = attr;
  if (field_index >= parsed->attribute_count) {
    parsed->attribute_count = field_index + 1;
  }
  return iree_ok_status();
}

iree_status_t loom_parsed_op_add_region(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        loom_region_t* region) {
  if (parsed->region_count >= parsed->region_capacity) {
    iree_host_size_t capacity = parsed->region_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->region_count, (iree_host_size_t)parsed->region_count + 1,
        sizeof(loom_region_t*), UINT8_MAX, "parsed op region", &capacity,
        (void**)&parsed->regions));
    parsed->region_capacity = (uint8_t)capacity;
  }
  parsed->regions[parsed->region_count++] = region;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_set_region(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        uint8_t index, loom_region_t* region) {
  if (index == UINT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "parsed op region index exceeds storage limit "
                            "(%u)",
                            (unsigned)(UINT8_MAX - 1));
  }
  iree_host_size_t required_capacity = (iree_host_size_t)index + 1;
  if (required_capacity > parsed->region_capacity) {
    iree_host_size_t capacity = parsed->region_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->region_count, required_capacity, sizeof(loom_region_t*),
        UINT8_MAX, "parsed op region", &capacity, (void**)&parsed->regions));
    parsed->region_capacity = (uint8_t)capacity;
  }
  while (parsed->region_count <= index) {
    parsed->regions[parsed->region_count++] = NULL;
  }
  parsed->regions[index] = region;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_add_tied_result(loom_parsed_op_t* parsed,
                                             iree_arena_allocator_t* arena,
                                             loom_tied_result_t tied) {
  if (parsed->tied_result_count >= parsed->tied_result_capacity) {
    iree_host_size_t capacity = parsed->tied_result_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->tied_result_count,
        (iree_host_size_t)parsed->tied_result_count + 1,
        sizeof(loom_tied_result_t), UINT16_MAX, "parsed op tied result",
        &capacity, (void**)&parsed->tied_results));
    parsed->tied_result_capacity = (uint16_t)capacity;
  }
  parsed->tied_results[parsed->tied_result_count++] = tied;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_add_field_span(
    loom_parsed_op_t* parsed, iree_arena_allocator_t* arena,
    loom_location_field_kind_t kind, uint16_t index, loom_token_t start_token,
    uint32_t end_line, uint32_t end_column) {
  if (start_token.kind == LOOM_TOKEN_NONE ||
      start_token.kind == LOOM_TOKEN_EOF) {
    return iree_ok_status();
  }
  if (start_token.line > UINT16_MAX || start_token.column > UINT16_MAX ||
      end_line > UINT16_MAX || end_column > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "parsed op field span source location exceeds storage limit");
  }
  if (parsed->field_span_count >= parsed->field_span_capacity) {
    iree_host_size_t capacity = parsed->field_span_capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        arena, parsed->field_span_count,
        (iree_host_size_t)parsed->field_span_count + 1,
        sizeof(loom_location_field_span_t), UINT16_MAX, "parsed op field span",
        &capacity, (void**)&parsed->field_spans));
    parsed->field_span_capacity = (uint16_t)capacity;
  }
  parsed->field_spans[parsed->field_span_count++] =
      (loom_location_field_span_t){
          .kind = kind,
          .index = index,
          .start_line = (uint16_t)start_token.line,
          .start_col = (uint16_t)start_token.column,
          .end_line = (uint16_t)end_line,
          .end_col = (uint16_t)end_column,
      };
  return iree_ok_status();
}

static iree_status_t loom_parser_add_pending_arg(
    loom_parser_t* parser, loom_parser_pending_block_args_t* pending_block_args,
    loom_value_id_t value_id, loom_token_t name_token) {
  if (pending_block_args->count >= pending_block_args->capacity) {
    iree_host_size_t capacity = pending_block_args->capacity;
    IREE_RETURN_IF_ERROR(loom_parser_grow_bounded_array(
        &parser->parser_arena, pending_block_args->count,
        (iree_host_size_t)pending_block_args->count + 1,
        sizeof(loom_parser_pending_block_arg_t), UINT16_MAX,
        "pending block argument", &capacity,
        (void**)&pending_block_args->entries));
    pending_block_args->capacity = (uint16_t)capacity;
  }
  pending_block_args->entries[pending_block_args->count++] =
      (loom_parser_pending_block_arg_t){
          .value_id = value_id,
          .name_token = name_token,
      };
  return iree_ok_status();
}

iree_status_t loom_parser_add_pending_block_arg(loom_parser_t* parser,
                                                loom_value_id_t value_id,
                                                loom_token_t name_token) {
  return loom_parser_add_pending_arg(parser, &parser->pending_block_args,
                                     value_id, name_token);
}

iree_status_t loom_parser_add_pending_func_arg(loom_parser_t* parser,
                                               loom_value_id_t value_id,
                                               loom_token_t name_token) {
  return loom_parser_add_pending_arg(parser, &parser->pending_func_args,
                                     value_id, name_token);
}

// Clears the active pending-arg list while retaining arena-backed storage for
// reuse by later REGION elements.
void loom_parser_pending_block_args_clear(
    loom_parser_pending_block_args_t* pending_block_args) {
  pending_block_args->count = 0;
}

void loom_parser_pending_block_args_truncate(
    loom_parser_pending_block_args_t* pending_block_args, uint16_t count) {
  if (pending_block_args->count > count) {
    pending_block_args->count = count;
  }
}
