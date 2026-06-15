// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_ACCUMULATOR_H_
#define LOOM_FORMAT_TEXT_PARSER_ACCUMULATOR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/text/tokenizer.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_parser_t loom_parser_t;
typedef struct loom_parsed_op_t loom_parsed_op_t;

// Pending block argument prepared by FUNC_ARGS, BINDING_LIST, or an implicit
// region operand such as a loop IV.
typedef struct loom_parser_pending_block_arg_t {
  loom_value_id_t value_id;
  // Name to define in the child region scope when REGION consumes this arg.
  loom_token_t name_token;
} loom_parser_pending_block_arg_t;

// Growable scratch list of values that should become entry-block arguments for
// the next parsed REGION element.
typedef struct loom_parser_pending_block_args_t {
  loom_parser_pending_block_arg_t* entries;
  uint16_t count;
  uint16_t capacity;
} loom_parser_pending_block_args_t;

// A successor edge whose textual target label is resolved after the enclosing
// region has parsed all of its block labels.
typedef struct loom_parser_pending_successor_ref_t {
  loom_region_t* region;
  loom_op_t* op;
  loom_token_t label_token;
  uint8_t successor_index;
} loom_parser_pending_successor_ref_t;

// Growable scratch list of unresolved CFG successor labels.
typedef struct loom_parser_pending_successor_refs_t {
  loom_parser_pending_successor_ref_t* entries;
  iree_host_size_t count;
  iree_host_size_t capacity;
} loom_parser_pending_successor_refs_t;

#define LOOM_PARSED_OP_INLINE_OPERANDS 16
#define LOOM_PARSED_OP_INLINE_SUCCESSORS 4
#define LOOM_PARSED_OP_INLINE_RESULTS 8
#define LOOM_PARSED_OP_INLINE_ATTRS 8
#define LOOM_PARSED_OP_INLINE_REGIONS 4
#define LOOM_PARSED_OP_INLINE_TIED 4
#define LOOM_PARSED_OP_INLINE_FIELD_SPANS 16
#define LOOM_PARSED_OP_INLINE_OPERAND_SEGMENTS 8

// Accumulates parsed fields during format walk. Pointers start aimed at the
// inline arrays and redirect to parser_arena spill storage on growth.
struct loom_parsed_op_t {
  loom_parsed_op_t* next_free;

  loom_value_id_t* operand_ids;
  loom_block_t** successors;
  loom_token_t* successor_label_tokens;
  loom_value_id_t* result_ids;
  loom_token_t* result_name_tokens;
  loom_attribute_t* attributes;
  loom_region_t** regions;
  loom_tied_result_t* tied_results;
  loom_location_field_span_t* field_spans;
  uint16_t* operand_segment_counts;

  uint16_t operand_count;
  uint16_t operand_capacity;
  uint8_t successor_count;
  uint8_t successor_capacity;
  uint16_t result_count;
  uint16_t result_capacity;
  uint16_t tied_result_count;
  uint16_t tied_result_capacity;
  uint16_t field_span_count;
  uint16_t field_span_capacity;
  uint8_t attribute_count;
  uint8_t attribute_capacity;
  uint8_t region_count;
  uint8_t region_capacity;
  uint8_t operand_segment_count;
  uint8_t operand_segment_capacity;
  uint8_t instance_flags;

  loom_value_id_t inline_operand_ids[LOOM_PARSED_OP_INLINE_OPERANDS];
  loom_block_t* inline_successors[LOOM_PARSED_OP_INLINE_SUCCESSORS];
  loom_token_t inline_successor_label_tokens[LOOM_PARSED_OP_INLINE_SUCCESSORS];
  loom_value_id_t inline_result_ids[LOOM_PARSED_OP_INLINE_RESULTS];
  loom_token_t inline_result_name_tokens[LOOM_PARSED_OP_INLINE_RESULTS];
  loom_attribute_t inline_attributes[LOOM_PARSED_OP_INLINE_ATTRS];
  loom_region_t* inline_regions[LOOM_PARSED_OP_INLINE_REGIONS];
  loom_tied_result_t inline_tied_results[LOOM_PARSED_OP_INLINE_TIED];
  loom_location_field_span_t
      inline_field_spans[LOOM_PARSED_OP_INLINE_FIELD_SPANS];
  uint16_t
      inline_operand_segment_counts[LOOM_PARSED_OP_INLINE_OPERAND_SEGMENTS];
};

void loom_parsed_op_initialize(loom_parsed_op_t* parsed);
void loom_parsed_op_reset(loom_parsed_op_t* parsed);
iree_status_t loom_parser_acquire_parsed_op(loom_parser_t* parser,
                                            loom_parsed_op_t** out_parsed);
void loom_parser_release_parsed_op(loom_parser_t* parser,
                                   loom_parsed_op_t* parsed);
iree_status_t loom_parsed_op_add_operand(loom_parsed_op_t* parsed,
                                         iree_arena_allocator_t* arena,
                                         loom_value_id_t value_id);
iree_status_t loom_parsed_op_set_operand(loom_parsed_op_t* parsed,
                                         iree_arena_allocator_t* arena,
                                         uint16_t index,
                                         loom_value_id_t value_id);
iree_status_t loom_parsed_op_prepare_operand_segments(
    loom_parsed_op_t* parsed, iree_arena_allocator_t* arena,
    uint8_t segment_count);
iree_status_t loom_parsed_op_add_segmented_operand(
    loom_parsed_op_t* parsed, iree_arena_allocator_t* arena,
    uint8_t segment_index, loom_value_id_t value_id,
    uint16_t* out_operand_index);
iree_status_t loom_parsed_op_set_successor(loom_parsed_op_t* parsed,
                                           iree_arena_allocator_t* arena,
                                           uint8_t index, loom_block_t* block,
                                           loom_token_t label_token);
iree_status_t loom_parsed_op_add_result(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        loom_value_id_t value_id,
                                        loom_token_t name_token);
iree_status_t loom_parsed_op_set_attribute(loom_parsed_op_t* parsed,
                                           iree_arena_allocator_t* arena,
                                           uint8_t index,
                                           loom_attribute_t attr);
iree_status_t loom_parsed_op_add_region(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        loom_region_t* region);
iree_status_t loom_parsed_op_set_region(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        uint8_t index, loom_region_t* region);
iree_status_t loom_parsed_op_add_tied_result(loom_parsed_op_t* parsed,
                                             iree_arena_allocator_t* arena,
                                             loom_tied_result_t tied);
iree_status_t loom_parsed_op_add_field_span(
    loom_parsed_op_t* parsed, iree_arena_allocator_t* arena,
    loom_location_field_kind_t kind, uint16_t index, loom_token_t start_token,
    uint32_t end_line, uint32_t end_column);
iree_status_t loom_parse_format_add_field_span(loom_parser_t* parser,
                                               loom_parsed_op_t* parsed,
                                               loom_location_field_kind_t kind,
                                               uint16_t index,
                                               loom_token_t start_token);
iree_status_t loom_parser_add_pending_block_arg(loom_parser_t* parser,
                                                loom_value_id_t value_id,
                                                loom_token_t name_token);
iree_status_t loom_parser_add_pending_func_arg(loom_parser_t* parser,
                                               loom_value_id_t value_id,
                                               loom_token_t name_token);
void loom_parser_pending_block_args_clear(
    loom_parser_pending_block_args_t* pending);
void loom_parser_pending_block_args_truncate(
    loom_parser_pending_block_args_t* pending, uint16_t count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_ACCUMULATOR_H_
