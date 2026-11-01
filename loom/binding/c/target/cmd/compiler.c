// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/target/cmd/compiler.h"

#include "iree/base/api.h"
#include "loom/binding/c/src/link_index.h"
#include "loomc/iree.h"

struct loomc_cmd_compiler_t {
  // Atomic shared ownership across compilation jobs.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this handle.
  loomc_allocator_t allocator;
  // Retained source universe, including its context.
  loomc_link_index_t* index;
  // Completed native sources borrowing the immutable universe.
  loomc_cmd_native_source_cache_t* sources;
};

loomc_status_t loomc_cmd_compiler_create(
    loomc_link_index_t* link_index, const loomc_cmd_compiler_options_t* options,
    loomc_allocator_t allocator, loomc_cmd_compiler_t** out_compiler) {
  if (out_compiler == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_compiler must not be NULL");
  }
  *out_compiler = NULL;
  if (link_index == NULL || !loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link_index and allocator must be valid");
  }
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_CMD_COMPILER_OPTIONS) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "unknown command compiler options type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "command compiler options are too small");
    }
    if (options->next != NULL) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "unknown command compiler extension");
    }
  }
  loomc_cmd_compiler_t* compiler = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*compiler), (void**)&compiler));
  *compiler =
      (loomc_cmd_compiler_t){.allocator = allocator, .index = link_index};
  iree_atomic_ref_count_init(&compiler->ref_count);
  loomc_link_index_retain(link_index);
  const loomc_host_size_t byte_limit =
      options ? options->retained_byte_limit : 64u * 1024u * 1024u;
  loomc_status_t status = loomc_ok_status();
  if (byte_limit != 0) {
    status = loomc_status_from_iree(loomc_cmd_native_source_cache_allocate(
        loomc_link_index_module_index(link_index), byte_limit,
        iree_allocator_from_loomc(allocator), &compiler->sources));
  }
  if (loomc_status_is_ok(status)) {
    *out_compiler = compiler;
  } else {
    loomc_cmd_compiler_release(compiler);
  }
  return status;
}

void loomc_cmd_compiler_retain(loomc_cmd_compiler_t* compiler) {
  if (compiler != NULL) {
    iree_atomic_ref_count_inc(&compiler->ref_count);
  }
}

void loomc_cmd_compiler_release(loomc_cmd_compiler_t* compiler) {
  if (compiler == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&compiler->ref_count) != 1) {
    return;
  }
  loomc_allocator_t allocator = compiler->allocator;
  loomc_cmd_native_source_cache_free(compiler->sources);
  loomc_link_index_release(compiler->index);
  loomc_allocator_free(allocator, compiler);
}

loomc_link_index_t* loomc_cmd_compiler_index(loomc_cmd_compiler_t* compiler) {
  return compiler->index;
}

loomc_cmd_native_source_cache_t* loomc_cmd_compiler_sources(
    loomc_cmd_compiler_t* compiler) {
  return compiler->sources;
}
