// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/session.h"

#include "loom/tooling/context/context.h"

static iree_status_t iree_benchmark_loom_register_context(
    void* user_data, loom_context_t* context) {
  const iree_benchmark_loom_configuration_t* configuration =
      (const iree_benchmark_loom_configuration_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_tooling_context_register_tool_dialects(context));
  if (configuration->register_context.fn == NULL) {
    return iree_ok_status();
  }
  return configuration->register_context.fn(
      configuration->register_context.user_data, context);
}

iree_status_t iree_benchmark_loom_session_initialize(
    const iree_benchmark_loom_configuration_t* configuration,
    iree_allocator_t host_allocator, loom_run_session_t* out_session) {
  IREE_ASSERT_ARGUMENT(configuration);
  IREE_ASSERT_ARGUMENT(out_session);
  loom_run_session_options_t session_options = {0};
  loom_run_session_options_initialize(&session_options);
  session_options.host_allocator = host_allocator;
  session_options.register_context = (loom_run_register_context_callback_t){
      .fn = iree_benchmark_loom_register_context,
      .user_data = (void*)configuration,
  };
  session_options.initialize_low_descriptor_registry =
      configuration->initialize_low_descriptor_registry;
  return loom_run_session_initialize(&session_options, out_session);
}
