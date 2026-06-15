// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "context.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "loom/ops/op_registry.h"
#include "loomc/iree.h"
#include "target.h"

struct loomc_context_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used to release the context.
  loomc_allocator_t allocator;
  // Finalized production Loom dialect and encoding context.
  loom_context_t context;
  // Optional target environment registered into the context.
  loomc_target_environment_t* target_environment;
};

static loomc_status_t loomc_context_validate_options(
    const loomc_context_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context options structure_size is too small");
  }
  return loomc_ok_status();
}

static void loomc_context_destroy(loomc_context_t* context) {
  loomc_allocator_t allocator = context->allocator;
  loom_context_deinitialize(&context->context);
  loomc_target_environment_release(context->target_environment);
  loomc_allocator_free(allocator, context);
}

loomc_status_t loomc_context_create(const loomc_context_options_t* options,
                                    loomc_allocator_t allocator,
                                    loomc_context_t** out_context) {
  if (out_context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_context must not be NULL");
  }
  *out_context = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_context_validate_options(options));
  loomc_target_environment_t* target_environment = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_context_target_options_resolve(options, &target_environment));

  loomc_context_t* context = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*context), (void**)&context));
  memset(context, 0, sizeof(*context));
  iree_atomic_ref_count_init(&context->ref_count);
  context->allocator = allocator;
  loom_context_initialize(iree_allocator_from_loomc(allocator),
                          &context->context);
  loomc_status_t status = loomc_status_from_iree(
      loom_op_registry_register_all_dialects(&context->context));
  if (loomc_status_is_ok(status) && target_environment != NULL) {
    status = loomc_target_environment_register_context(target_environment,
                                                       &context->context);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(loom_context_finalize(&context->context));
  }
  if (loomc_status_is_ok(status)) {
    context->target_environment = target_environment;
    loomc_target_environment_retain(context->target_environment);
    *out_context = context;
  } else {
    loom_context_deinitialize(&context->context);
    loomc_allocator_free(allocator, context);
  }
  return status;
}

void loomc_context_retain(loomc_context_t* context) {
  if (context == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&context->ref_count);
}

void loomc_context_release(loomc_context_t* context) {
  if (context == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&context->ref_count) == 1) {
    loomc_context_destroy(context);
  }
}

loom_context_t* loomc_context_loom_context(loomc_context_t* context) {
  return context ? &context->context : NULL;
}

loomc_target_environment_t* loomc_context_target_environment(
    const loomc_context_t* context) {
  return context ? context->target_environment : NULL;
}

const loomc_target_pass_environment_t* loomc_context_target_pass_environment(
    const loomc_context_t* context) {
  return context ? loomc_target_environment_pass_environment(
                       context->target_environment)
                 : NULL;
}
