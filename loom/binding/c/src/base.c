// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/base.h"

#include <string.h>

#include "iree/base/api.h"
#include "loomc/status.h"

static loomc_status_t loomc_allocator_validate(loomc_allocator_t allocator) {
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator.ctl must not be NULL");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_allocator_malloc(loomc_allocator_t allocator,
                                      loomc_host_size_t byte_length,
                                      void** out_ptr) {
  if (out_ptr == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_ptr must not be NULL");
  }
  *out_ptr = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_validate(allocator));
  loomc_allocator_alloc_params_t params = {
      .byte_length = byte_length,
  };
  return allocator.ctl(allocator.self, LOOMC_ALLOCATOR_COMMAND_CALLOC, &params,
                       out_ptr);
}

loomc_status_t loomc_allocator_malloc_uninitialized(
    loomc_allocator_t allocator, loomc_host_size_t byte_length,
    void** out_ptr) {
  if (out_ptr == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_ptr must not be NULL");
  }
  *out_ptr = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_validate(allocator));
  loomc_allocator_alloc_params_t params = {
      .byte_length = byte_length,
  };
  return allocator.ctl(allocator.self, LOOMC_ALLOCATOR_COMMAND_MALLOC, &params,
                       out_ptr);
}

loomc_status_t loomc_string_view_clone(loomc_string_view_t source,
                                       loomc_allocator_t allocator,
                                       loomc_string_view_t* out_string) {
  if (out_string == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_string must not be NULL");
  }
  *out_string = loomc_string_view_empty();
  if (source.data == NULL && source.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  if (loomc_string_view_is_empty(source)) {
    return loomc_ok_status();
  }
  char* target = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, source.size, (void**)&target));
  memcpy(target, source.data, source.size);
  *out_string = loomc_make_string_view(target, source.size);
  return loomc_ok_status();
}

void loomc_allocator_free(loomc_allocator_t allocator, void* ptr) {
  if (ptr == NULL) {
    return;
  }
  IREE_ASSERT_ARGUMENT(loomc_allocator_is_valid(allocator));
  void* inout_ptr = ptr;
  allocator.ctl(allocator.self, LOOMC_ALLOCATOR_COMMAND_FREE, NULL, &inout_ptr);
}
