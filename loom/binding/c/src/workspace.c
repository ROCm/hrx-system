// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "workspace.h"

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

enum {
  LOOMC_WORKSPACE_DEFAULT_BLOCK_SIZE = 32 * 1024,
};

struct loomc_workspace_t {
  // Atomic reference count for shared storage ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release workspace storage.
  loomc_allocator_t allocator;

  // Transient arena block pool reused across operations.
  iree_arena_block_pool_t block_pool;
};

static iree_allocator_t loomc_workspace_iree_allocator(
    loomc_allocator_t allocator) {
  return (iree_allocator_t){
      .self = allocator.self,
      .ctl = (iree_allocator_ctl_fn_t)allocator.ctl,
  };
}

static loomc_status_t loomc_workspace_validate_options(
    const loomc_workspace_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "workspace options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "workspace options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "workspace option extensions are not supported");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_workspace_create(const loomc_workspace_options_t* options,
                                      loomc_allocator_t allocator,
                                      loomc_workspace_t** out_workspace) {
  if (out_workspace == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_workspace must not be NULL");
  }
  *out_workspace = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_workspace_validate_options(options));

  loomc_workspace_t* workspace = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(allocator, sizeof(*workspace),
                                               (void**)&workspace));
  iree_atomic_ref_count_init(&workspace->ref_count);
  workspace->allocator = allocator;
  iree_host_size_t block_size = LOOMC_WORKSPACE_DEFAULT_BLOCK_SIZE;
  if (options && options->block_size != 0) {
    block_size = options->block_size;
  }
  iree_arena_block_pool_initialize(block_size,
                                   loomc_workspace_iree_allocator(allocator),
                                   &workspace->block_pool);
  *out_workspace = workspace;
  return loomc_ok_status();
}

void loomc_workspace_retain(loomc_workspace_t* workspace) {
  if (workspace == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&workspace->ref_count);
}

void loomc_workspace_trim(loomc_workspace_t* workspace) {
  if (workspace == NULL) {
    return;
  }
  iree_arena_block_pool_trim(&workspace->block_pool);
}

void loomc_workspace_release(loomc_workspace_t* workspace) {
  if (workspace == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&workspace->ref_count) != 1) {
    return;
  }
  loomc_allocator_t allocator = workspace->allocator;
  iree_arena_block_pool_deinitialize(&workspace->block_pool);
  loomc_allocator_free(allocator, workspace);
}

iree_arena_block_pool_t* loomc_workspace_block_pool(
    loomc_workspace_t* workspace) {
  return workspace ? &workspace->block_pool : NULL;
}
