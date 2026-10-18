// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func/location_capture.h"

#include <string.h>

#include "loom/ops/func/ops.h"

// One record per reached location serves both the traversal and identity map.
// Records have stable arena addresses even as the bucket table grows.
typedef struct loom_location_capture_node_t {
  // Original module-local identity.
  loom_location_id_t location;
  // Next source child to visit.
  uint32_t next_child;
  // Completed postorder identity, or UINT32_MAX while visiting.
  uint32_t index;
  // First traversal parent; NULL for the root.
  struct loom_location_capture_node_t* parent;
  // Next record in the identity hash bucket.
  struct loom_location_capture_node_t* hash_next;
  // Next completed record in postorder.
  struct loom_location_capture_node_t* next;
  // Semantic node whose payload is owned by the output module.
  loom_attribute_t attribute;
} loom_location_capture_node_t;

typedef struct loom_location_capture_t {
  // Owns the source graph and constructed semantic attributes.
  loom_module_t* module;
  // Borrows admitted original source snapshots for this capture.
  loom_source_resolver_t resolver;
  // Owns temporary traversal records and constructor arrays.
  iree_arena_allocator_t* arena;
  // Location identity map, with power-of-two bucket count.
  loom_location_capture_node_t** buckets;
  // Number of allocated hash buckets.
  uint32_t bucket_count;
  // Number of distinct reached locations, bounded by the attribute array ABI.
  uint32_t node_count;
  // Head of the completed postorder list.
  loom_location_capture_node_t* first;
  // Tail of the completed postorder list.
  loom_location_capture_node_t* last;
} loom_location_capture_t;

static uint32_t loom_location_capture_hash(loom_location_id_t location) {
  uint32_t hash = location;
  hash ^= hash >> 16;
  hash *= 0x7feb352du;
  hash ^= hash >> 15;
  hash *= 0x846ca68bu;
  return hash ^ (hash >> 16);
}

static loom_location_capture_node_t* loom_location_capture_find(
    const loom_location_capture_t* capture, loom_location_id_t location) {
  if (!capture->bucket_count) {
    return NULL;
  }
  uint32_t bucket =
      loom_location_capture_hash(location) & (capture->bucket_count - 1);
  for (loom_location_capture_node_t* node = capture->buckets[bucket]; node;
       node = node->hash_next) {
    if (node->location == location) {
      return node;
    }
  }
  return NULL;
}

static iree_status_t loom_location_capture_reach(
    loom_location_capture_t* capture, loom_location_id_t location,
    loom_location_capture_node_t* parent,
    loom_location_capture_node_t** out_node) {
  *out_node = loom_location_capture_find(capture, location);
  if (*out_node) {
    return iree_ok_status();
  }
  if (capture->node_count == UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "captured location exceeds 65535 semantic nodes");
  }
  if (capture->node_count >= capture->bucket_count / 2) {
    uint32_t count = capture->bucket_count ? capture->bucket_count * 2 : 16;
    loom_location_capture_node_t** buckets = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        capture->arena, count, sizeof(*buckets), (void**)&buckets));
    memset(buckets, 0, count * sizeof(*buckets));
    for (uint32_t i = 0; i < capture->bucket_count; ++i) {
      loom_location_capture_node_t* node = capture->buckets[i];
      while (node) {
        loom_location_capture_node_t* next = node->hash_next;
        uint32_t bucket =
            loom_location_capture_hash(node->location) & (count - 1);
        node->hash_next = buckets[bucket];
        buckets[bucket] = node;
        node = next;
      }
    }
    capture->buckets = buckets;
    capture->bucket_count = count;
  }
  loom_location_capture_node_t* node = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(capture->arena, sizeof(*node), (void**)&node));
  uint32_t bucket =
      loom_location_capture_hash(location) & (capture->bucket_count - 1);
  *node = (loom_location_capture_node_t){
      .location = location,
      .index = UINT32_MAX,
      .parent = parent,
      .hash_next = capture->buckets[bucket],
  };
  capture->buckets[bucket] = node;
  ++capture->node_count;
  *out_node = node;
  return iree_ok_status();
}

// The resolver supplies exact byte extents in the original snapshot. Extend
// them to complete lines without rescanning the source preceding this range.
static iree_const_byte_span_t loom_location_capture_lines(
    const loom_source_range_t* range) {
  iree_host_size_t start = range->start;
  while (start && range->source.data[start - 1] != '\n') {
    --start;
  }
  iree_host_size_t end = range->end;
  // An exclusive end at the next line's start already contains the final
  // complete source line. A point range still captures its containing line.
  if (end == start || !end || range->source.data[end - 1] != '\n') {
    while (end < range->source.size && range->source.data[end] != '\n') {
      ++end;
    }
    if (end < range->source.size) {
      ++end;
    }
  }
  return iree_make_const_byte_span(
      range->source.data ? range->source.data + start : NULL, end - start);
}

static iree_status_t loom_location_capture_file(
    loom_location_capture_t* capture, loom_location_id_t location,
    const loom_location_entry_t* entry, loom_attribute_t* out_attribute) {
  loom_module_t* module = capture->module;
  loom_string_id_t source;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, module->sources.entries[entry->file.source_id], &source));
  const int64_t coordinates[] = {entry->file.start_line, entry->file.start_col,
                                 entry->file.end_line, entry->file.end_col};
  loom_attribute_t* fields = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(capture->arena, entry->file.field_span_count,
                                sizeof(*fields), (void**)&fields));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < entry->file.field_span_count && iree_status_is_ok(status); ++i) {
    const loom_location_field_span_t field =
        loom_location_file_field_span_at(entry, (uint16_t)i);
    const int64_t range[] = {field.start_line, field.start_col, field.end_line,
                             field.end_col};
    status = loom_func_location_field_attr_make(
        module, field.kind, field.index,
        (loom_i64_array_t){.values = range, .count = 4}, &fields[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  loom_func_location_file_attr_build_flags_t flags = 0;
  bool synthetic = iree_any_bit_set(entry->flags, LOOM_LOCATION_FLAG_SYNTHETIC);
  if (synthetic) {
    flags |= LOOM_FUNC_LOCATION_FILE_ATTR_BUILD_FLAG_HAS_SYNTHETIC;
  }
  if (entry->file.field_span_count) {
    flags |= LOOM_FUNC_LOCATION_FILE_ATTR_BUILD_FLAG_HAS_FIELDS;
  }
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  loom_source_range_t range;
  if (entry->file.start_line && entry->file.start_col && entry->file.end_line &&
      entry->file.end_col &&
      loom_source_resolve(capture->resolver, module, location, &range) &&
      range.provenance == LOOM_SOURCE_PROVENANCE_EXACT_SOURCE) {
    flags |= LOOM_FUNC_LOCATION_FILE_ATTR_BUILD_FLAG_HAS_TEXT;
    text = loom_location_capture_lines(&range);
  }
  return loom_func_location_file_attr_make(
      module, flags, source,
      (loom_i64_array_t){.values = coordinates, .count = 4}, synthetic,
      (loom_parameterized_attr_array_t){fields, entry->file.field_span_count},
      text, out_attribute);
}

static iree_status_t loom_location_capture_attribute(
    loom_location_capture_t* capture, loom_location_capture_node_t* node,
    const loom_location_entry_t* entry) {
  loom_module_t* module = capture->module;
  bool synthetic = iree_any_bit_set(entry->flags, LOOM_LOCATION_FLAG_SYNTHETIC);
  switch (entry->kind) {
    case LOOM_LOCATION_NONE:
      return loom_func_location_unknown_attr_make(module, &node->attribute);
    case LOOM_LOCATION_FILE:
      return loom_location_capture_file(capture, node->location, entry,
                                        &node->attribute);
    case LOOM_LOCATION_FUSED: {
      int64_t* children = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(capture->arena, entry->fused.count,
                                    sizeof(*children), (void**)&children));
      for (uint32_t i = 0; i < entry->fused.count; ++i) {
        children[i] =
            loom_location_capture_find(capture, entry->fused.children[i])
                ->index;
      }
      return loom_func_location_fused_attr_make(
          module,
          synthetic ? LOOM_FUNC_LOCATION_FUSED_ATTR_BUILD_FLAG_HAS_SYNTHETIC
                    : 0,
          (loom_i64_array_t){.values = children, .count = entry->fused.count},
          synthetic, &node->attribute);
    }
    case LOOM_LOCATION_OPAQUE: {
      loom_string_id_t source;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          module, module->sources.entries[entry->opaque.source_id], &source));
      return loom_func_location_opaque_attr_make(
          module,
          synthetic ? LOOM_FUNC_LOCATION_OPAQUE_ATTR_BUILD_FLAG_HAS_SYNTHETIC
                    : 0,
          source,
          iree_make_const_byte_span(entry->opaque.data,
                                    entry->opaque.data_length),
          synthetic, &node->attribute);
    }
    case LOOM_LOCATION_TAGGED: {
      loom_func_location_tagged_attr_build_flags_t flags =
          synthetic ? LOOM_FUNC_LOCATION_TAGGED_ATTR_BUILD_FLAG_HAS_SYNTHETIC
                    : 0;
      int64_t child = 0;
      if (entry->tagged.child != LOOM_LOCATION_UNKNOWN) {
        flags |= LOOM_FUNC_LOCATION_TAGGED_ATTR_BUILD_FLAG_HAS_CHILD;
        child = loom_location_capture_find(capture, entry->tagged.child)->index;
      }
      return loom_func_location_tagged_attr_make(
          module, flags, entry->tagged.tag,
          iree_make_const_byte_span(entry->tagged.data,
                                    entry->tagged.data_length),
          child, synthetic, &node->attribute);
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown location kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_func_location_capture(
    loom_module_t* module, loom_location_id_t location,
    loom_source_resolver_t resolver, iree_arena_allocator_t* scratch_arena,
    loom_parameterized_attr_array_t* out_nodes) {
  *out_nodes = (loom_parameterized_attr_array_t){0};
  loom_location_capture_t capture = {
      .module = module, .resolver = resolver, .arena = scratch_arena};
  loom_location_capture_node_t* node = NULL;
  IREE_RETURN_IF_ERROR(
      loom_location_capture_reach(&capture, location, NULL, &node));
  const loom_location_entry_t unknown = {0};
  uint32_t count = 0;
  iree_status_t status = iree_ok_status();
  while (node && iree_status_is_ok(status)) {
    const loom_location_entry_t* entry =
        node->location == LOOM_LOCATION_UNKNOWN
            ? &unknown
            : loom_location_table_const_entry(&module->locations,
                                              node->location);
    uint32_t child_count = 0;
    if (entry->kind == LOOM_LOCATION_FUSED) {
      child_count = entry->fused.count;
    } else if (entry->kind == LOOM_LOCATION_TAGGED &&
               entry->tagged.child != LOOM_LOCATION_UNKNOWN) {
      child_count = 1;
    }
    if (node->next_child < child_count) {
      loom_location_id_t child = entry->kind == LOOM_LOCATION_FUSED
                                     ? entry->fused.children[node->next_child]
                                     : entry->tagged.child;
      ++node->next_child;
      loom_location_capture_node_t* reached = NULL;
      status = loom_location_capture_reach(&capture, child, node, &reached);
      if (iree_status_is_ok(status) && reached->index == UINT32_MAX) {
        node = reached;
      }
    } else {
      status = loom_location_capture_attribute(&capture, node, entry);
      if (iree_status_is_ok(status)) {
        node->index = count++;
        if (capture.last) {
          capture.last->next = node;
        } else {
          capture.first = node;
        }
        capture.last = node;
        node = node->parent;
      }
    }
  }
  loom_attribute_t* nodes = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&module->arena, count, sizeof(*nodes),
                                       (void**)&nodes);
  }
  if (iree_status_is_ok(status)) {
    for (node = capture.first; node; node = node->next) {
      nodes[node->index] = node->attribute;
    }
    *out_nodes = (loom_parameterized_attr_array_t){nodes, count};
  }
  return status;
}
