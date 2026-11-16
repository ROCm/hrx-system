// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/proactor_thread_runner.h"

#include <stdio.h>

#include "iree/async/util/proactor_thread.h"

//===----------------------------------------------------------------------===//
// Thread runner factory callbacks
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_pool_thread_runner_create(
    void* user_data, iree_async_proactor_t* proactor, uint32_t node_id,
    const iree_thread_affinity_set_t* processor_set, iree_allocator_t allocator,
    void** out_runner) {
  iree_async_proactor_thread_options_t thread_options =
      iree_async_proactor_thread_options_default();
  thread_options.processor_set = processor_set;

  // Configure NUMA affinity if a node ID is specified.
  if (node_id != UINT32_MAX) {
    iree_thread_affinity_set_group_any(node_id, &thread_options.affinity);
  }

  // Generate a debug name from the node ID or a generic index.
  char thread_name_buffer[32];
  if (node_id != UINT32_MAX) {
    snprintf(thread_name_buffer, sizeof(thread_name_buffer), "iree-pro-%u",
             node_id);
  } else {
    snprintf(thread_name_buffer, sizeof(thread_name_buffer), "iree-pro");
  }
  thread_options.debug_name =
      iree_make_string_view(thread_name_buffer, strlen(thread_name_buffer));

  iree_async_proactor_thread_t* thread = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_thread_create(
      proactor, thread_options, allocator, &thread));
  *out_runner = thread;
  return iree_ok_status();
}

static void iree_async_proactor_pool_thread_runner_request_stop(void* user_data,
                                                                void* runner) {
  iree_async_proactor_thread_request_stop(
      (iree_async_proactor_thread_t*)runner);
}

static void iree_async_proactor_pool_thread_runner_destroy(void* user_data,
                                                           void* runner) {
  iree_async_proactor_thread_t* thread = (iree_async_proactor_thread_t*)runner;
  iree_async_proactor_thread_release(thread);
}

//===----------------------------------------------------------------------===//
// Public factory constructor
//===----------------------------------------------------------------------===//

iree_async_proactor_pool_runner_factory_t
iree_async_proactor_pool_thread_runner_factory(void) {
  iree_async_proactor_pool_runner_factory_t factory;
  memset(&factory, 0, sizeof(factory));
  factory.create = iree_async_proactor_pool_thread_runner_create;
  factory.request_stop = iree_async_proactor_pool_thread_runner_request_stop;
  factory.destroy = iree_async_proactor_pool_thread_runner_destroy;
  return factory;
}
