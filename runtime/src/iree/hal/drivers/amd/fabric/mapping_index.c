// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/mapping_index.h"

// Nodes are stable across rotations and borrowed only inside a lookup/commit.
// Detached nodes reuse left as the free-list link.
struct iree_hal_amd_fabric_mapping_node_t {
  // Subtree containing earlier target offsets, or free-list successor.
  iree_hal_amd_fabric_mapping_node_t* left;
  // Subtree containing later target offsets.
  iree_hal_amd_fabric_mapping_node_t* right;
  // Parent used to rebalance and walk adjacent installed ranges.
  iree_hal_amd_fabric_mapping_node_t* parent;
  // Installed source coordinates, borrowed through the backing lifetime.
  iree_hal_amd_fabric_virtual_mapping_t mapping;
  // AVL subtree height; leaves have height one.
  uint8_t height;
};

struct iree_hal_amd_fabric_mapping_block_t {
  // Next owned allocation, newest first.
  iree_hal_amd_fabric_mapping_block_t* next;
  // Number of nodes following this header.
  iree_host_size_t capacity;
};

static uint8_t iree_hal_amd_fabric_mapping_node_height(
    const iree_hal_amd_fabric_mapping_node_t* node) {
  return node ? node->height : 0;
}

static void iree_hal_amd_fabric_mapping_node_update_height(
    iree_hal_amd_fabric_mapping_node_t* node) {
  node->height =
      1 + iree_max(iree_hal_amd_fabric_mapping_node_height(node->left),
                   iree_hal_amd_fabric_mapping_node_height(node->right));
}

static int iree_hal_amd_fabric_mapping_node_compare(
    const iree_hal_amd_fabric_mapping_node_t* lhs,
    const iree_hal_amd_fabric_mapping_node_t* rhs) {
  return lhs->mapping.native.target_byte_offset <
                 rhs->mapping.native.target_byte_offset
             ? -1
             : 1;
}

static void iree_hal_amd_fabric_mapping_tree_replace(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_hal_amd_fabric_mapping_node_t* old_node,
    iree_hal_amd_fabric_mapping_node_t* new_node) {
  if (!old_node->parent) {
    index->root = new_node;
  } else if (old_node == old_node->parent->left) {
    old_node->parent->left = new_node;
  } else {
    old_node->parent->right = new_node;
  }
  if (new_node) {
    new_node->parent = old_node->parent;
  }
}

static iree_hal_amd_fabric_mapping_node_t*
iree_hal_amd_fabric_mapping_tree_rotate_left(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_hal_amd_fabric_mapping_node_t* node) {
  iree_hal_amd_fabric_mapping_node_t* pivot = node->right;
  iree_hal_amd_fabric_mapping_tree_replace(index, node, pivot);
  node->right = pivot->left;
  if (node->right) {
    node->right->parent = node;
  }
  pivot->left = node;
  node->parent = pivot;
  iree_hal_amd_fabric_mapping_node_update_height(node);
  iree_hal_amd_fabric_mapping_node_update_height(pivot);
  return pivot;
}

static iree_hal_amd_fabric_mapping_node_t*
iree_hal_amd_fabric_mapping_tree_rotate_right(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_hal_amd_fabric_mapping_node_t* node) {
  iree_hal_amd_fabric_mapping_node_t* pivot = node->left;
  iree_hal_amd_fabric_mapping_tree_replace(index, node, pivot);
  node->left = pivot->right;
  if (node->left) {
    node->left->parent = node;
  }
  pivot->right = node;
  node->parent = pivot;
  iree_hal_amd_fabric_mapping_node_update_height(node);
  iree_hal_amd_fabric_mapping_node_update_height(pivot);
  return pivot;
}

static void iree_hal_amd_fabric_mapping_tree_rebalance(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_hal_amd_fabric_mapping_node_t* node) {
  while (node) {
    iree_hal_amd_fabric_mapping_node_update_height(node);
    const int balance =
        (int)iree_hal_amd_fabric_mapping_node_height(node->left) -
        (int)iree_hal_amd_fabric_mapping_node_height(node->right);
    if (balance > 1) {
      if (iree_hal_amd_fabric_mapping_node_height(node->left->left) <
          iree_hal_amd_fabric_mapping_node_height(node->left->right)) {
        iree_hal_amd_fabric_mapping_tree_rotate_left(index, node->left);
      }
      node = iree_hal_amd_fabric_mapping_tree_rotate_right(index, node);
    } else if (balance < -1) {
      if (iree_hal_amd_fabric_mapping_node_height(node->right->right) <
          iree_hal_amd_fabric_mapping_node_height(node->right->left)) {
        iree_hal_amd_fabric_mapping_tree_rotate_right(index, node->right);
      }
      node = iree_hal_amd_fabric_mapping_tree_rotate_left(index, node);
    }
    node = node->parent;
  }
}

static iree_hal_amd_fabric_mapping_node_t*
iree_hal_amd_fabric_mapping_node_successor(
    iree_hal_amd_fabric_mapping_node_t* node) {
  if (node->right) {
    node = node->right;
    while (node->left) {
      node = node->left;
    }
    return node;
  }
  while (node->parent && node == node->parent->right) {
    node = node->parent;
  }
  return node->parent;
}

static void iree_hal_amd_fabric_mapping_tree_insert(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_hal_amd_fabric_mapping_node_t* node) {
  node->left = NULL;
  node->right = NULL;
  node->parent = NULL;
  node->height = 1;
  if (!index->root) {
    index->root = node;
    return;
  }

  iree_hal_amd_fabric_mapping_node_t* parent = index->root;
  while (true) {
    iree_hal_amd_fabric_mapping_node_t** child =
        iree_hal_amd_fabric_mapping_node_compare(node, parent) < 0
            ? &parent->left
            : &parent->right;
    if (!*child) {
      *child = node;
      node->parent = parent;
      break;
    }
    parent = *child;
  }
  iree_hal_amd_fabric_mapping_tree_rebalance(index, parent);
}

static void iree_hal_amd_fabric_mapping_tree_erase(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_hal_amd_fabric_mapping_node_t* node) {
  iree_hal_amd_fabric_mapping_node_t* rebalance_from = node->parent;
  if (!node->left) {
    iree_hal_amd_fabric_mapping_tree_replace(index, node, node->right);
  } else if (!node->right) {
    iree_hal_amd_fabric_mapping_tree_replace(index, node, node->left);
  } else {
    iree_hal_amd_fabric_mapping_node_t* successor = node->right;
    while (successor->left) {
      successor = successor->left;
    }
    if (successor->parent != node) {
      rebalance_from = successor->parent;
      iree_hal_amd_fabric_mapping_tree_replace(index, successor,
                                               successor->right);
      successor->right = node->right;
      successor->right->parent = successor;
    } else {
      rebalance_from = successor;
    }
    iree_hal_amd_fabric_mapping_tree_replace(index, node, successor);
    successor->left = node->left;
    successor->left->parent = successor;
    iree_hal_amd_fabric_mapping_node_update_height(successor);
  }
  node->left = NULL;
  node->right = NULL;
  node->parent = NULL;
  node->height = 0;
  iree_hal_amd_fabric_mapping_tree_rebalance(index, rebalance_from);
}

static iree_hal_amd_fabric_mapping_node_t*
iree_hal_amd_fabric_mapping_node_previous(
    iree_hal_amd_fabric_mapping_node_t* node) {
  if (node->left) {
    node = node->left;
    while (node->right) {
      node = node->right;
    }
    return node;
  }
  while (node->parent && node == node->parent->left) {
    node = node->parent;
  }
  return node->parent;
}

// Last run beginning at or before position, whether or not it covers position.
static iree_hal_amd_fabric_mapping_node_t* iree_hal_amd_fabric_mapping_floor(
    const iree_hal_amd_fabric_mapping_index_t* index, uint64_t position) {
  iree_hal_amd_fabric_mapping_node_t* node = index->root;
  iree_hal_amd_fabric_mapping_node_t* result = NULL;
  while (node) {
    if (node->mapping.native.target_byte_offset > position) {
      node = node->left;
    } else {
      result = node;
      node = node->right;
    }
  }
  return result;
}

static iree_hal_amd_fabric_mapping_node_t*
iree_hal_amd_fabric_mapping_lower_bound(
    const iree_hal_amd_fabric_mapping_index_t* index, uint64_t position) {
  iree_hal_amd_fabric_mapping_node_t* node = index->root;
  iree_hal_amd_fabric_mapping_node_t* result = NULL;
  while (node) {
    if (node->mapping.native.target_byte_offset < position) {
      node = node->right;
    } else {
      result = node;
      node = node->left;
    }
  }
  return result;
}

void iree_hal_amd_fabric_mapping_index_deinitialize(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_allocator_t host_allocator) {
  iree_hal_amd_fabric_mapping_block_t* block = index->blocks;
  *index = (iree_hal_amd_fabric_mapping_index_t){0};
  while (block) {
    iree_hal_amd_fabric_mapping_block_t* next = block->next;
    iree_allocator_free(host_allocator, block);
    block = next;
  }
}

iree_status_t iree_hal_amd_fabric_mapping_index_prepare(
    iree_hal_amd_fabric_mapping_index_t* index, iree_host_size_t update_count,
    const amdf_memory_mapping_update_t* updates,
    iree_allocator_t host_allocator) {
  // Each update can add at most a binding and a split suffix. Once warm this
  // bound avoids searching the tree a second time on the ordinary path.
  if (update_count <= index->available_count / 2) {
    return iree_ok_status();
  }

  iree_host_size_t required_count = 0;
  for (iree_host_size_t i = 0; i < update_count; ++i) {
    const amdf_memory_mapping_update_t* update = &updates[i];
    const iree_hal_amd_fabric_mapping_node_t* node =
        iree_hal_amd_fabric_mapping_floor(index, update->target_byte_offset);
    const amdf_memory_mapping_update_t* old =
        node ? &node->mapping.native : NULL;
    const bool replaces_node =
        old && old->target_byte_offset == update->target_byte_offset &&
        old->byte_length == update->byte_length;
    const bool splits_node =
        old && old->target_byte_offset < update->target_byte_offset &&
        old->target_byte_offset + old->byte_length >
            update->target_byte_offset + update->byte_length;
    const iree_host_size_t added_count =
        (update->memory && !replaces_node ? 1 : 0) + (splits_node ? 1 : 0);
    if (!iree_host_size_checked_add(required_count, added_count,
                                    &required_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "mapping index capacity overflow");
    }
  }
  if (required_count <= index->available_count) {
    return iree_ok_status();
  }
  required_count -= index->available_count;

  // Keep newly allocated blocks separate until all growth succeeds. Queries
  // touch only installed nodes, and a failed prepare changes no retained state.
  iree_hal_amd_fabric_mapping_index_t pending = {0};
  iree_hal_amd_fabric_mapping_block_t* first_block = NULL;
  iree_hal_amd_fabric_mapping_node_t* first_node = NULL;
  iree_host_size_t capacity = index->blocks ? index->blocks->capacity : 1;
  iree_status_t status = iree_ok_status();
  while (pending.available_count < required_count &&
         iree_status_is_ok(status)) {
    // Small reservations start small; large ones grow in bounded ~5 KiB blocks.
    capacity = iree_min(capacity * 2, 64);
    iree_hal_amd_fabric_mapping_block_t* block = NULL;
    status = iree_allocator_malloc_uninitialized(
        host_allocator,
        sizeof(*block) + capacity * sizeof(iree_hal_amd_fabric_mapping_node_t),
        (void**)&block);
    if (iree_status_is_ok(status)) {
      block->next = pending.blocks;
      block->capacity = capacity;
      pending.blocks = block;
      iree_hal_amd_fabric_mapping_node_t* nodes = (void*)(block + 1);
      if (!first_block) {
        first_block = block;
        first_node = &nodes[0];
      }
      for (iree_host_size_t i = 0; i < capacity; ++i) {
        nodes[i].left = pending.available;
        pending.available = &nodes[i];
      }
      pending.available_count += capacity;
    }
  }
  if (iree_status_is_ok(status)) {
    first_block->next = index->blocks;
    first_node->left = index->available;
    index->blocks = pending.blocks;
    index->available = pending.available;
    index->available_count += pending.available_count;
  } else {
    iree_hal_amd_fabric_mapping_index_deinitialize(&pending, host_allocator);
  }
  return status;
}

static iree_hal_amd_fabric_mapping_node_t* iree_hal_amd_fabric_mapping_acquire(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_hal_amd_fabric_virtual_mapping_t mapping) {
  // prepare must cover every intermediate split, before native state changes.
  IREE_ASSERT(index->available);
  iree_hal_amd_fabric_mapping_node_t* node = index->available;
  index->available = node->left;
  --index->available_count;
  node->mapping = mapping;
  return node;
}

static void iree_hal_amd_fabric_mapping_recycle(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_hal_amd_fabric_mapping_node_t* node) {
  node->left = index->available;
  index->available = node;
  ++index->available_count;
}

static iree_hal_amd_fabric_virtual_mapping_t iree_hal_amd_fabric_mapping_tail(
    iree_hal_amd_fabric_virtual_mapping_t mapping, uint64_t offset) {
  mapping.native.target_byte_offset += offset;
  mapping.native.source_byte_offset += offset;
  mapping.native.byte_length -= offset;
  return mapping;
}

static bool iree_hal_amd_fabric_mapping_joins(
    const iree_hal_amd_fabric_virtual_mapping_t* left,
    const iree_hal_amd_fabric_virtual_mapping_t* right) {
  return left->native.memory == right->native.memory &&
         left->native.access == right->native.access &&
         left->source == right->source &&
         left->native.target_byte_offset + left->native.byte_length ==
             right->native.target_byte_offset &&
         left->native.source_byte_offset + left->native.byte_length ==
             right->native.source_byte_offset;
}

static iree_hal_amd_fabric_mapping_node_t* iree_hal_amd_fabric_mapping_coalesce(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_hal_amd_fabric_mapping_node_t* node) {
  iree_hal_amd_fabric_mapping_node_t* previous =
      iree_hal_amd_fabric_mapping_node_previous(node);
  if (previous &&
      iree_hal_amd_fabric_mapping_joins(&previous->mapping, &node->mapping)) {
    previous->mapping.native.byte_length += node->mapping.native.byte_length;
    iree_hal_amd_fabric_mapping_tree_erase(index, node);
    iree_hal_amd_fabric_mapping_recycle(index, node);
    node = previous;
  }
  iree_hal_amd_fabric_mapping_node_t* next =
      iree_hal_amd_fabric_mapping_node_successor(node);
  if (next &&
      iree_hal_amd_fabric_mapping_joins(&node->mapping, &next->mapping)) {
    node->mapping.native.byte_length += next->mapping.native.byte_length;
    iree_hal_amd_fabric_mapping_tree_erase(index, next);
    iree_hal_amd_fabric_mapping_recycle(index, next);
  }
  return node;
}

static iree_hal_amd_fabric_mapping_node_t*
iree_hal_amd_fabric_mapping_apply_one(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_hal_amd_fabric_virtual_mapping_t update,
    iree_hal_amd_fabric_mapping_node_t* hint) {
  const uint64_t start = update.native.target_byte_offset;
  const uint64_t end = start + update.native.byte_length;
  // A sorted batch can reuse the preceding edit and its immediate successor.
  // Gaps use the tree search rather than walking unrelated mappings.
  iree_hal_amd_fabric_mapping_node_t* covering = NULL;
  if (hint && hint->mapping.native.target_byte_offset <= start) {
    iree_hal_amd_fabric_mapping_node_t* next =
        iree_hal_amd_fabric_mapping_node_successor(hint);
    if (!next || next->mapping.native.target_byte_offset > start) {
      covering = hint;
    } else if (next->mapping.native.target_byte_offset +
                   next->mapping.native.byte_length >
               start) {
      covering = next;
    }
  }
  if (!covering) {
    covering = iree_hal_amd_fabric_mapping_floor(index, start);
  }
  if (covering && covering->mapping.native.target_byte_offset == start &&
      covering->mapping.native.byte_length == update.native.byte_length &&
      update.native.memory) {
    covering->mapping = update;
    return iree_hal_amd_fabric_mapping_coalesce(index, covering);
  }
  iree_hal_amd_fabric_mapping_node_t* previous = covering;
  if (previous && previous->mapping.native.target_byte_offset == start) {
    previous = iree_hal_amd_fabric_mapping_node_previous(previous);
  }
  if (previous && previous->mapping.native.target_byte_offset +
                          previous->mapping.native.byte_length >
                      start) {
    if (previous->mapping.native.target_byte_offset +
            previous->mapping.native.byte_length >
        end) {
      iree_hal_amd_fabric_mapping_node_t* suffix =
          iree_hal_amd_fabric_mapping_acquire(
              index, iree_hal_amd_fabric_mapping_tail(
                         previous->mapping,
                         end - previous->mapping.native.target_byte_offset));
      iree_hal_amd_fabric_mapping_tree_insert(index, suffix);
    }
    previous->mapping.native.byte_length =
        start - previous->mapping.native.target_byte_offset;
  }
  iree_hal_amd_fabric_mapping_node_t* node =
      previous ? iree_hal_amd_fabric_mapping_node_successor(previous)
               : iree_hal_amd_fabric_mapping_lower_bound(index, start);
  while (node && node->mapping.native.target_byte_offset < end) {
    iree_hal_amd_fabric_mapping_node_t* next =
        iree_hal_amd_fabric_mapping_node_successor(node);
    const bool has_suffix = node->mapping.native.target_byte_offset +
                                node->mapping.native.byte_length >
                            end;
    iree_hal_amd_fabric_mapping_tree_erase(index, node);
    if (has_suffix) {
      node->mapping = iree_hal_amd_fabric_mapping_tail(
          node->mapping, end - node->mapping.native.target_byte_offset);
      iree_hal_amd_fabric_mapping_tree_insert(index, node);
    } else {
      iree_hal_amd_fabric_mapping_recycle(index, node);
    }
    node = next;
  }
  if (!update.native.memory) {
    return previous;
  }
  node = iree_hal_amd_fabric_mapping_acquire(index, update);
  iree_hal_amd_fabric_mapping_tree_insert(index, node);
  return iree_hal_amd_fabric_mapping_coalesce(index, node);
}

void iree_hal_amd_fabric_mapping_index_apply(
    iree_hal_amd_fabric_mapping_index_t* index, iree_host_size_t update_count,
    const iree_hal_memory_mapping_update_t* updates,
    const amdf_memory_mapping_update_t* native_updates) {
  iree_hal_amd_fabric_mapping_node_t* hint = NULL;
  for (iree_host_size_t i = 0; i < update_count; ++i) {
    const iree_hal_amd_fabric_virtual_mapping_t mapping = {
        .native = native_updates[i],
        .source = native_updates[i].memory
                      ? updates[i].source.mapping_source.descriptor
                      : NULL,
    };
    hint = iree_hal_amd_fabric_mapping_apply_one(index, mapping, hint);
  }
}

bool iree_hal_amd_fabric_mapping_index_query(
    const iree_hal_amd_fabric_mapping_index_t* index, uint64_t position,
    uint64_t length, iree_hal_amd_fabric_virtual_mapping_t* out_mapping) {
  const iree_hal_amd_fabric_mapping_node_t* node =
      iree_hal_amd_fabric_mapping_floor(index, position);
  if (!node || position >= node->mapping.native.target_byte_offset +
                               node->mapping.native.byte_length) {
    *out_mapping = (iree_hal_amd_fabric_virtual_mapping_t){0};
    return false;
  }
  *out_mapping = iree_hal_amd_fabric_mapping_tail(
      node->mapping, position - node->mapping.native.target_byte_offset);
  out_mapping->native.byte_length =
      iree_min(out_mapping->native.byte_length, length);
  return true;
}
