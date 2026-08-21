// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/provider_import_sink.h"

#include <stdlib.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/module/ops.h"

// Borrowed semantic view used only while canonical imports are constructed.
typedef struct loom_link_provider_import_view_t {
  // Target-module provider string ID.
  loom_string_id_t provider_id;
  // Stable target-module provider string view.
  iree_string_view_t provider;
  // Target symbol anchors.
  loom_symbol_ref_array_t anchors;
  // Normalized source comments.
  iree_string_view_list_t comments;
  // Existing import operation replaced at finish, or NULL for projected rows.
  loom_op_t* existing_op;
  // True when the source import had authored leading vertical separation.
  bool leading_blank_line;
} loom_link_provider_import_view_t;

static int loom_link_provider_import_view_compare(const void* lhs_ptr,
                                                  const void* rhs_ptr) {
  const loom_link_provider_import_view_t* lhs =
      (const loom_link_provider_import_view_t*)lhs_ptr;
  const loom_link_provider_import_view_t* rhs =
      (const loom_link_provider_import_view_t*)rhs_ptr;
  int comparison = iree_string_view_compare(lhs->provider, rhs->provider);
  if (comparison != 0) {
    return comparison;
  }
  const iree_host_size_t shared_comment_count =
      iree_min(lhs->comments.count, rhs->comments.count);
  for (iree_host_size_t i = 0; i < shared_comment_count; ++i) {
    comparison = iree_string_view_compare(lhs->comments.values[i],
                                          rhs->comments.values[i]);
    if (comparison != 0) {
      return comparison;
    }
  }
  if (lhs->comments.count < rhs->comments.count) {
    return -1;
  }
  if (lhs->comments.count > rhs->comments.count) {
    return 1;
  }
  if (lhs->leading_blank_line != rhs->leading_blank_line) {
    return lhs->leading_blank_line ? 1 : -1;
  }
  return 0;
}

static iree_status_t loom_link_provider_import_copy_comments(
    iree_arena_allocator_t* arena, iree_string_view_list_t comments,
    const iree_string_view_t** out_comments) {
  *out_comments = NULL;
  if (comments.count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t payload_size = 0;
  for (iree_host_size_t i = 0; i < comments.count; ++i) {
    if (!iree_host_size_checked_add(payload_size, comments.values[i].size,
                                    &payload_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "provider import comment payload overflow");
    }
  }
  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_mul_add(payload_size, comments.count,
                                      sizeof(*comments.values),
                                      &storage_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "provider import comment storage overflow");
  }

  iree_string_view_t* copied_comments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, storage_size, (void**)&copied_comments));
  char* copied_payload = (char*)(copied_comments + comments.count);
  for (iree_host_size_t i = 0; i < comments.count; ++i) {
    const iree_string_view_t comment = comments.values[i];
    if (comment.size == 0) {
      copied_comments[i] = iree_string_view_empty();
      continue;
    }
    memcpy(copied_payload, comment.data, comment.size);
    copied_comments[i] = iree_make_string_view(copied_payload, comment.size);
    copied_payload += comment.size;
  }
  *out_comments = copied_comments;
  return iree_ok_status();
}

iree_status_t loom_link_provider_import_sink_initialize(
    loom_module_t* target_module, iree_arena_allocator_t* arena,
    iree_host_size_t row_capacity, iree_host_size_t anchor_capacity,
    loom_link_provider_import_sink_t* out_sink) {
  *out_sink = (loom_link_provider_import_sink_t){
      .target_module = target_module,
      .arena = arena,
  };
  if (row_capacity > UINT32_MAX || anchor_capacity > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "projected provider imports exceed uint32 capacity");
  }
  if (row_capacity == 0 && anchor_capacity != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "provider import anchors require a row capacity");
  }
  if (row_capacity > anchor_capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "provider import row capacity exceeds nonempty anchor capacity");
  }

  loom_link_provider_import_sink_t sink = {
      .target_module = target_module,
      .arena = arena,
      .rows = {.capacity = row_capacity},
      .anchors = {.capacity = anchor_capacity},
  };
  if (row_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, row_capacity,
                                                   sizeof(*sink.rows.values),
                                                   (void**)&sink.rows.values));
  }
  if (anchor_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, anchor_capacity, sizeof(*sink.anchors.values),
        (void**)&sink.anchors.values));
  }
  *out_sink = sink;
  return iree_ok_status();
}

iree_status_t loom_link_provider_import_sink_append(
    loom_link_provider_import_sink_t* sink, iree_string_view_t provider,
    loom_symbol_ref_array_t target_anchors, iree_string_view_list_t comments,
    bool leading_blank_line) {
  IREE_ASSERT(sink->rows.count < sink->rows.capacity);
  IREE_ASSERT(target_anchors.count > 0);
  IREE_ASSERT(target_anchors.count <=
              sink->anchors.capacity - sink->anchors.count);
  IREE_ASSERT(comments.count <= UINT16_MAX);

  loom_string_id_t provider_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(sink->target_module, provider, &provider_id));
  const iree_string_view_t* copied_comments = NULL;
  IREE_RETURN_IF_ERROR(loom_link_provider_import_copy_comments(
      sink->arena, comments, &copied_comments));

  const uint32_t first_anchor = (uint32_t)sink->anchors.count;
  memcpy(sink->anchors.values + sink->anchors.count, target_anchors.values,
         target_anchors.count * sizeof(*sink->anchors.values));
  sink->anchors.count += target_anchors.count;
  sink->rows.values[sink->rows.count++] =
      (loom_link_provider_import_sink_row_t){
          .provider_id = provider_id,
          .anchors =
              {
                  .first = first_anchor,
                  .count = (uint32_t)target_anchors.count,
              },
          .comments =
              {
                  .values = copied_comments,
                  .count = (uint16_t)comments.count,
              },
          .leading_blank_line = leading_blank_line,
      };
  return iree_ok_status();
}

static iree_host_size_t loom_link_provider_import_count_existing(
    const loom_module_t* module) {
  iree_host_size_t count = 0;
  const loom_op_t* op = NULL;
  loom_block_for_each_op(loom_region_const_entry_block(module->body), op) {
    count += loom_module_import_isa(op);
  }
  return count;
}

static void loom_link_provider_import_fill_views(
    loom_link_provider_import_sink_t* sink,
    loom_link_provider_import_view_t* views) {
  iree_host_size_t view_ordinal = 0;
  for (iree_host_size_t i = 0; i < sink->rows.count; ++i) {
    const loom_link_provider_import_sink_row_t* row = &sink->rows.values[i];
    views[view_ordinal++] = (loom_link_provider_import_view_t){
        .provider_id = row->provider_id,
        .provider = sink->target_module->strings.entries[row->provider_id],
        .anchors = loom_make_symbol_ref_array(
            sink->anchors.values + row->anchors.first, row->anchors.count),
        .comments =
            {
                .count = row->comments.count,
                .values = row->comments.values,
            },
        .leading_blank_line = row->leading_blank_line,
    };
  }

  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(sink->target_module), op) {
    if (!loom_module_import_isa(op)) {
      continue;
    }
    const loom_string_id_t provider_id = loom_module_import_provider(op);
    iree_host_size_t comment_count = 0;
    const iree_string_view_t* comments =
        loom_module_op_comments(sink->target_module, op, &comment_count);
    views[view_ordinal++] = (loom_link_provider_import_view_t){
        .provider_id = provider_id,
        .provider = sink->target_module->strings.entries[provider_id],
        .anchors = loom_module_import_symbols(op),
        .comments = {.count = comment_count, .values = comments},
        .existing_op = op,
        .leading_blank_line =
            iree_any_bit_set(op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE),
    };
  }
}

static iree_status_t loom_link_provider_import_group_comment_count(
    const loom_link_provider_import_view_t* views, iree_host_size_t view_count,
    iree_host_size_t* out_comment_count) {
  *out_comment_count = 0;
  for (iree_host_size_t i = 0; i < view_count; ++i) {
    if (!iree_host_size_checked_add(*out_comment_count, views[i].comments.count,
                                    out_comment_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "merged provider import comment count overflow");
    }
  }
  if (*out_comment_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "merged provider import has more than %u comment lines",
        (unsigned)UINT16_MAX);
  }
  return iree_ok_status();
}

static iree_status_t loom_link_provider_import_build_group(
    loom_link_provider_import_sink_t* sink,
    const loom_link_provider_import_view_t* views, iree_host_size_t view_count,
    uint64_t* anchor_bits, loom_symbol_ref_t* unique_anchors) {
  iree_host_size_t unique_anchor_count = 0;
  bool leading_blank_line = false;
  for (iree_host_size_t i = 0; i < view_count; ++i) {
    const loom_link_provider_import_view_t* view = &views[i];
    leading_blank_line |= view->leading_blank_line;
    for (iree_host_size_t j = 0; j < view->anchors.count; ++j) {
      const loom_symbol_ref_t anchor = view->anchors.values[j];
      const uint64_t mask = UINT64_C(1) << (anchor.symbol_id % 64u);
      uint64_t* word = &anchor_bits[anchor.symbol_id / 64u];
      if (iree_any_bit_set(*word, mask)) {
        continue;
      }
      *word |= mask;
      unique_anchors[unique_anchor_count++] = anchor;
    }
  }
  IREE_ASSERT(unique_anchor_count <= UINT16_MAX);
  const loom_symbol_ref_t duplicate_ref = loom_module_canonicalize_symbol_set(
      sink->target_module, unique_anchors, (uint16_t)unique_anchor_count);
  IREE_ASSERT(!loom_symbol_ref_is_valid(duplicate_ref));

  iree_host_size_t comment_count = 0;
  IREE_RETURN_IF_ERROR(loom_link_provider_import_group_comment_count(
      views, view_count, &comment_count));
  const iree_arena_checkpoint_t comment_checkpoint =
      iree_arena_checkpoint_save(sink->arena);
  iree_string_view_t* comments = NULL;
  iree_status_t status = iree_ok_status();
  if (comment_count > 0) {
    status = iree_arena_allocate_array(sink->arena, comment_count,
                                       sizeof(*comments), (void**)&comments);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t comment_ordinal = 0;
    for (iree_host_size_t i = 0; i < view_count; ++i) {
      if (views[i].comments.count > 0) {
        memcpy(comments + comment_ordinal, views[i].comments.values,
               views[i].comments.count * sizeof(*comments));
      }
      comment_ordinal += views[i].comments.count;
    }

    loom_builder_t builder;
    loom_builder_initialize(sink->target_module, &sink->target_module->arena,
                            loom_module_block(sink->target_module), &builder);
    loom_op_t* import_op = NULL;
    status = loom_module_import_build(
        &builder, views[0].provider_id,
        loom_make_symbol_ref_array(unique_anchors, unique_anchor_count),
        LOOM_LOCATION_NONE, &import_op);
    if (iree_status_is_ok(status) && leading_blank_line) {
      import_op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
    }
    if (iree_status_is_ok(status)) {
      status = loom_module_attach_op_comments(sink->target_module, import_op,
                                              comments, comment_count);
    }
  }
  for (iree_host_size_t i = 0; i < unique_anchor_count; ++i) {
    const loom_symbol_ref_t anchor = unique_anchors[i];
    anchor_bits[anchor.symbol_id / 64u] &=
        ~(UINT64_C(1) << (anchor.symbol_id % 64u));
  }
  iree_arena_checkpoint_restore(&comment_checkpoint);
  return status;
}

iree_status_t loom_link_provider_import_sink_finish(
    loom_link_provider_import_sink_t* sink) {
  const iree_host_size_t existing_count =
      loom_link_provider_import_count_existing(sink->target_module);
  iree_host_size_t view_count = 0;
  if (!iree_host_size_checked_add(sink->rows.count, existing_count,
                                  &view_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "provider import view count overflow");
  }
  if (view_count == 0) {
    return iree_ok_status();
  }

  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(sink->arena);
  loom_link_provider_import_view_t* views = NULL;
  iree_status_t status = iree_arena_allocate_array(
      sink->arena, view_count, sizeof(*views), (void**)&views);
  if (iree_status_is_ok(status)) {
    loom_link_provider_import_fill_views(sink, views);
    qsort(views, view_count, sizeof(*views),
          loom_link_provider_import_view_compare);
  }

  const iree_host_size_t target_symbol_count =
      sink->target_module->symbols.count;
  loom_symbol_ref_t* unique_anchors = NULL;
  if (iree_status_is_ok(status) && target_symbol_count > 0) {
    status = iree_arena_allocate_array(sink->arena, target_symbol_count,
                                       sizeof(*unique_anchors),
                                       (void**)&unique_anchors);
  }
  const iree_host_size_t anchor_word_count = (target_symbol_count + 63u) / 64u;
  uint64_t* anchor_bits = NULL;
  if (iree_status_is_ok(status) && anchor_word_count > 0) {
    status =
        iree_arena_allocate_array(sink->arena, anchor_word_count,
                                  sizeof(*anchor_bits), (void**)&anchor_bits);
    if (iree_status_is_ok(status)) {
      memset(anchor_bits, 0, anchor_word_count * sizeof(*anchor_bits));
    }
  }

  for (iree_host_size_t first = 0;
       first < view_count && iree_status_is_ok(status);) {
    iree_host_size_t end = first + 1;
    while (end < view_count &&
           iree_string_view_equal(views[first].provider, views[end].provider)) {
      ++end;
    }
    status = loom_link_provider_import_build_group(
        sink, views + first, end - first, anchor_bits, unique_anchors);
    first = end;
  }

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < view_count; ++i) {
      if (views[i].existing_op) {
        status = loom_op_erase(sink->target_module, views[i].existing_op);
        if (!iree_status_is_ok(status)) {
          break;
        }
      }
    }
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}
