// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/type_import.h"

#include "iree/base/internal/math.h"
#include "loom/ir/module.h"
#include "loom/ir/type_identity.h"

// A frame becomes a memo leaf after its children have been canonicalized. Its
// embedded branch splits the prior memo subtree from that leaf, so memoization
// needs no allocation beyond the traversal frames themselves.
typedef struct loom_type_import_frame_t {
  // Borrowed input identity, immutable until the importing call completes.
  loom_type_t source;
  // Unfinished owner of this child, or NULL at the root.
  struct loom_type_import_frame_t* parent;
  // First child span: arguments, dialect parameters or a register value type.
  struct {
    // Borrowed contiguous child types.
    const loom_type_t* types;
    // Number of child types.
    uint32_t count;
  } first;
  // Second child span for separately supplied function results.
  struct {
    // Borrowed contiguous result types.
    const loom_type_t* types;
    // Number of result types.
    uint32_t count;
  } second;
  // Canonical immediate children, in representation order.
  loom_type_id_t* children;
  // Number of children already canonicalized.
  uint32_t completed;
  // Published canonical result, valid after all children complete.
  loom_type_id_t result;
  // Memo edges: low bit tags a completed frame leaf; even edges are branches.
  uintptr_t edges[2];
  // Differing bit in the 192-bit by-value identity, decreasing along paths.
  uint32_t bit;
} loom_type_import_frame_t;

static uint64_t loom_type_import_word(loom_type_t type, uint32_t word) {
  if (word > 0) {
    return type.dims[word - 1];
  }
  return (uint64_t)type.header | ((uint64_t)type.encoding_id << 32) |
         ((uint64_t)type.encoding_flags << 48);
}

static uint32_t loom_type_import_side(loom_type_t type, uint32_t bit) {
  return (uint32_t)((loom_type_import_word(type, bit / 64) >> (bit % 64)) & 1);
}

static loom_type_import_frame_t* loom_type_import_frame(uintptr_t edge) {
  return (loom_type_import_frame_t*)(edge & ~(uintptr_t)1);
}

static loom_type_id_t loom_type_import_find(uintptr_t root, loom_type_t type) {
  if (!root) {
    return LOOM_TYPE_ID_INVALID;
  }
  while (!(root & 1)) {
    const loom_type_import_frame_t* frame = loom_type_import_frame(root);
    root = frame->edges[loom_type_import_side(type, frame->bit)];
  }
  const loom_type_import_frame_t* frame = loom_type_import_frame(root);
  for (uint32_t word = 0; word < 3; ++word) {
    if (loom_type_import_word(type, word) !=
        loom_type_import_word(frame->source, word)) {
      return LOOM_TYPE_ID_INVALID;
    }
  }
  return frame->result;
}

static void loom_type_import_insert(uintptr_t* root,
                                    loom_type_import_frame_t* frame) {
  const uintptr_t leaf = (uintptr_t)frame | 1;
  if (!*root) {
    *root = leaf;
    return;
  }
  uintptr_t existing = *root;
  while (!(existing & 1)) {
    const loom_type_import_frame_t* branch = loom_type_import_frame(existing);
    existing = branch->edges[loom_type_import_side(frame->source, branch->bit)];
  }
  const loom_type_t other = loom_type_import_frame(existing)->source;
  uint32_t word = 2;
  uint64_t difference = loom_type_import_word(frame->source, word) ^
                        loom_type_import_word(other, word);
  while (!difference && word > 0) {
    --word;
    difference = loom_type_import_word(frame->source, word) ^
                 loom_type_import_word(other, word);
  }
  // Acyclic inputs are memoized on completion, before a later occurrence can
  // create a second frame for the same immutable source identity.
  IREE_ASSERT(difference != 0);
  frame->bit = word * 64 + 63 - iree_math_count_leading_zeros_u64(difference);
  uintptr_t* edge = root;
  while (!(*edge & 1)) {
    loom_type_import_frame_t* branch = loom_type_import_frame(*edge);
    if (branch->bit < frame->bit) {
      break;
    }
    edge = &branch->edges[loom_type_import_side(frame->source, branch->bit)];
  }
  const uint32_t side = loom_type_import_side(frame->source, frame->bit);
  frame->edges[side] = leaf;
  frame->edges[side ^ 1] = *edge;
  *edge = (uintptr_t)frame;
}

static bool loom_type_import_is_composite(loom_type_t type) {
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_FUNCTION:
    case LOOM_TYPE_DIALECT:
      return true;
    case LOOM_TYPE_REGISTER:
      return loom_type_register_has_value_type(type);
    case LOOM_TYPE_VIEW:
      return loom_type_has_reference_element(type);
    default:
      return false;
  }
}

static iree_status_t loom_type_import_initialize_frame(
    loom_type_t type, loom_type_import_frame_t* frame) {
  *frame = (loom_type_import_frame_t){.source = type};
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "function type has a NULL argument/result payload");
      }
      frame->first.types = data->types;
      frame->first.count = (uint32_t)data->arg_count + data->result_count;
      break;
    }
    case LOOM_TYPE_DIALECT:
      frame->first.types = loom_type_dialect_params(type);
      frame->first.count = loom_type_dialect_param_count(type);
      break;
    case LOOM_TYPE_REGISTER:
      frame->first.types = loom_type_register_value_type(type);
      frame->first.count = 1;
      break;
    case LOOM_TYPE_VIEW:
      frame->first.types = loom_type_reference_element(type);
      frame->first.count = 1;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("only composite types require an import frame");
  }
  if (frame->first.count && !frame->first.types) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type has %u children but a NULL payload",
                            frame->first.count);
  }
  return iree_ok_status();
}

static iree_status_t loom_type_import_run(loom_module_t* module,
                                          loom_type_import_frame_t* root,
                                          loom_type_id_t* out_type_id) {
  iree_arena_allocator_t scratch;
  iree_arena_initialize(module->arena.block_pool, &scratch);
  loom_type_id_t inline_children[16];
  root->children = inline_children;
  const uint32_t root_count = root->first.count + root->second.count;
  iree_status_t status = iree_ok_status();
  if (root_count > IREE_ARRAYSIZE(inline_children)) {
    status = iree_arena_allocate_array(
        &scratch, root_count, sizeof(loom_type_id_t), (void**)&root->children);
  }
  uintptr_t memo = 0;
  loom_type_import_frame_t* frame = root;
  while (frame && iree_status_is_ok(status)) {
    const uint32_t child_count = frame->first.count + frame->second.count;
    if (frame->completed == child_count) {
      status = loom_module_intern_topological_type_id(
          module, frame->source, frame->children, child_count, &frame->result);
      if (iree_status_is_ok(status)) {
        loom_type_import_frame_t* parent = frame->parent;
        if (parent) {
          loom_type_import_insert(&memo, frame);
          parent->children[parent->completed++] = frame->result;
        }
        frame = parent;
      }
      continue;
    }
    const loom_type_t child =
        frame->completed < frame->first.count
            ? frame->first.types[frame->completed]
            : frame->second.types[frame->completed - frame->first.count];
    loom_type_id_t child_id = LOOM_TYPE_ID_INVALID;
    if (!loom_type_import_is_composite(child)) {
      status = loom_module_intern_type_id(module, child, &child_id);
    } else {
      child_id = loom_type_identity_find(module, child);
      if (child_id == LOOM_TYPE_ID_INVALID) {
        child_id = loom_type_import_find(memo, child);
      }
      if (child_id == LOOM_TYPE_ID_INVALID) {
        loom_type_import_frame_t child_frame;
        status = loom_type_import_initialize_frame(child, &child_frame);
        loom_type_import_frame_t* pending = NULL;
        if (iree_status_is_ok(status)) {
          const iree_host_size_t allocation_size =
              sizeof(*pending) +
              child_frame.first.count * sizeof(loom_type_id_t);
          status =
              iree_arena_allocate(&scratch, allocation_size, (void**)&pending);
        }
        if (iree_status_is_ok(status)) {
          *pending = child_frame;
          pending->parent = frame;
          pending->children = (loom_type_id_t*)(pending + 1);
          frame = pending;
        }
        continue;
      }
    }
    if (iree_status_is_ok(status)) {
      frame->children[frame->completed++] = child_id;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_type_id = root->result;
  }
  iree_arena_deinitialize(&scratch);
  return status;
}

iree_status_t loom_type_import(loom_module_t* module, loom_type_t type,
                               loom_type_id_t* out_type_id) {
  loom_type_import_frame_t root;
  IREE_RETURN_IF_ERROR(loom_type_import_initialize_frame(type, &root));
  return loom_type_import_run(module, &root, out_type_id);
}

iree_status_t loom_type_import_function(loom_module_t* module,
                                        const loom_type_t* arg_types,
                                        uint16_t arg_count,
                                        const loom_type_t* result_types,
                                        uint16_t result_count,
                                        loom_type_id_t* out_type_id) {
  const loom_func_type_data_t signature = {.arg_count = arg_count,
                                           .result_count = result_count};
  loom_type_import_frame_t root = {
      .source = loom_type_function(&signature),
      .first = {.types = arg_types, .count = arg_count},
      .second = {.types = result_types, .count = result_count},
  };
  return loom_type_import_run(module, &root, out_type_id);
}
