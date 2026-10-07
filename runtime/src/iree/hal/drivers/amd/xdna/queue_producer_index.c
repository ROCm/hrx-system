// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_producer_index.h"

static uint8_t iree_hal_amd_xdna_queue_producer_node_height(
    const iree_hal_amd_xdna_queue_producer_node_t* node) {
  return node ? node->height : 0;
}

static void iree_hal_amd_xdna_queue_producer_node_update_height(
    iree_hal_amd_xdna_queue_producer_node_t* node) {
  node->height =
      1 + iree_max(iree_hal_amd_xdna_queue_producer_node_height(node->left),
                   iree_hal_amd_xdna_queue_producer_node_height(node->right));
}

static int iree_hal_amd_xdna_queue_producer_node_compare(
    const iree_hal_amd_xdna_queue_producer_node_t* lhs,
    const iree_hal_amd_xdna_queue_producer_node_t* rhs) {
  const uintptr_t lhs_semaphore = (uintptr_t)lhs->semaphore;
  const uintptr_t rhs_semaphore = (uintptr_t)rhs->semaphore;
  if (lhs_semaphore != rhs_semaphore) {
    return lhs_semaphore < rhs_semaphore ? -1 : 1;
  }
  if (lhs->value != rhs->value) {
    return lhs->value < rhs->value ? -1 : 1;
  }
  return (uintptr_t)lhs < (uintptr_t)rhs ? -1 : 1;
}

static void iree_hal_amd_xdna_queue_producer_index_replace(
    iree_hal_amd_xdna_queue_producer_index_t* index,
    iree_hal_amd_xdna_queue_producer_node_t* old_node,
    iree_hal_amd_xdna_queue_producer_node_t* new_node) {
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

static iree_hal_amd_xdna_queue_producer_node_t*
iree_hal_amd_xdna_queue_producer_index_rotate_left(
    iree_hal_amd_xdna_queue_producer_index_t* index,
    iree_hal_amd_xdna_queue_producer_node_t* node) {
  iree_hal_amd_xdna_queue_producer_node_t* pivot = node->right;
  iree_hal_amd_xdna_queue_producer_index_replace(index, node, pivot);
  node->right = pivot->left;
  if (node->right) {
    node->right->parent = node;
  }
  pivot->left = node;
  node->parent = pivot;
  iree_hal_amd_xdna_queue_producer_node_update_height(node);
  iree_hal_amd_xdna_queue_producer_node_update_height(pivot);
  return pivot;
}

static iree_hal_amd_xdna_queue_producer_node_t*
iree_hal_amd_xdna_queue_producer_index_rotate_right(
    iree_hal_amd_xdna_queue_producer_index_t* index,
    iree_hal_amd_xdna_queue_producer_node_t* node) {
  iree_hal_amd_xdna_queue_producer_node_t* pivot = node->left;
  iree_hal_amd_xdna_queue_producer_index_replace(index, node, pivot);
  node->left = pivot->right;
  if (node->left) {
    node->left->parent = node;
  }
  pivot->right = node;
  node->parent = pivot;
  iree_hal_amd_xdna_queue_producer_node_update_height(node);
  iree_hal_amd_xdna_queue_producer_node_update_height(pivot);
  return pivot;
}

static void iree_hal_amd_xdna_queue_producer_index_rebalance(
    iree_hal_amd_xdna_queue_producer_index_t* index,
    iree_hal_amd_xdna_queue_producer_node_t* node) {
  while (node) {
    iree_hal_amd_xdna_queue_producer_node_update_height(node);
    const int balance =
        (int)iree_hal_amd_xdna_queue_producer_node_height(node->left) -
        (int)iree_hal_amd_xdna_queue_producer_node_height(node->right);
    if (balance > 1) {
      if (iree_hal_amd_xdna_queue_producer_node_height(node->left->left) <
          iree_hal_amd_xdna_queue_producer_node_height(node->left->right)) {
        iree_hal_amd_xdna_queue_producer_index_rotate_left(index, node->left);
      }
      node = iree_hal_amd_xdna_queue_producer_index_rotate_right(index, node);
    } else if (balance < -1) {
      if (iree_hal_amd_xdna_queue_producer_node_height(node->right->right) <
          iree_hal_amd_xdna_queue_producer_node_height(node->right->left)) {
        iree_hal_amd_xdna_queue_producer_index_rotate_right(index, node->right);
      }
      node = iree_hal_amd_xdna_queue_producer_index_rotate_left(index, node);
    }
    node = node->parent;
  }
}

static iree_hal_amd_xdna_queue_producer_node_t*
iree_hal_amd_xdna_queue_producer_node_successor(
    iree_hal_amd_xdna_queue_producer_node_t* node) {
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

void iree_hal_amd_xdna_queue_producer_index_initialize(
    iree_hal_amd_xdna_queue_producer_index_t* out_index) {
  out_index->root = NULL;
}

void iree_hal_amd_xdna_queue_producer_index_insert(
    iree_hal_amd_xdna_queue_producer_index_t* index,
    iree_hal_amd_xdna_queue_producer_node_t* node) {
  node->left = NULL;
  node->right = NULL;
  node->parent = NULL;
  node->height = 1;
  if (!index->root) {
    index->root = node;
    return;
  }

  iree_hal_amd_xdna_queue_producer_node_t* parent = index->root;
  while (true) {
    iree_hal_amd_xdna_queue_producer_node_t** child =
        iree_hal_amd_xdna_queue_producer_node_compare(node, parent) < 0
            ? &parent->left
            : &parent->right;
    if (!*child) {
      *child = node;
      node->parent = parent;
      break;
    }
    parent = *child;
  }
  iree_hal_amd_xdna_queue_producer_index_rebalance(index, parent);
}

void iree_hal_amd_xdna_queue_producer_index_erase(
    iree_hal_amd_xdna_queue_producer_index_t* index,
    iree_hal_amd_xdna_queue_producer_node_t* node) {
  iree_hal_amd_xdna_queue_producer_node_t* rebalance_from = node->parent;
  if (!node->left) {
    iree_hal_amd_xdna_queue_producer_index_replace(index, node, node->right);
  } else if (!node->right) {
    iree_hal_amd_xdna_queue_producer_index_replace(index, node, node->left);
  } else {
    iree_hal_amd_xdna_queue_producer_node_t* successor = node->right;
    while (successor->left) {
      successor = successor->left;
    }
    if (successor->parent != node) {
      rebalance_from = successor->parent;
      iree_hal_amd_xdna_queue_producer_index_replace(index, successor,
                                                     successor->right);
      successor->right = node->right;
      successor->right->parent = successor;
    } else {
      rebalance_from = successor;
    }
    iree_hal_amd_xdna_queue_producer_index_replace(index, node, successor);
    successor->left = node->left;
    successor->left->parent = successor;
    iree_hal_amd_xdna_queue_producer_node_update_height(successor);
  }
  node->left = NULL;
  node->right = NULL;
  node->parent = NULL;
  node->height = 0;
  iree_hal_amd_xdna_queue_producer_index_rebalance(index, rebalance_from);
}

iree_hal_amd_xdna_queue_producer_node_t*
iree_hal_amd_xdna_queue_producer_index_find(
    const iree_hal_amd_xdna_queue_producer_index_t* index,
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value,
    const void* excluded_producer) {
  const uintptr_t semaphore_key = (uintptr_t)semaphore;
  iree_hal_amd_xdna_queue_producer_node_t* node = index->root;
  iree_hal_amd_xdna_queue_producer_node_t* candidate = NULL;
  while (node) {
    const uintptr_t node_semaphore = (uintptr_t)node->semaphore;
    if (node_semaphore < semaphore_key ||
        (node_semaphore == semaphore_key && node->value < minimum_value)) {
      node = node->right;
    } else {
      candidate = node;
      node = node->left;
    }
  }
  while (candidate && candidate->semaphore == semaphore &&
         candidate->producer == excluded_producer) {
    candidate = iree_hal_amd_xdna_queue_producer_node_successor(candidate);
  }
  return candidate && candidate->semaphore == semaphore ? candidate : NULL;
}
