// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/runtime_requirements.h"

#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/util/walk.h"

typedef struct loom_sanitizer_runtime_requirements_state_t {
  // Requirements discovered from executable operations and options.
  loom_sanitizer_runtime_requirements_t requirements;
  // All requirements that can be discovered under the reporting policy.
  loom_sanitizer_runtime_requirements_t complete_requirements;
  // Reporting policy applied to authored sanitizer operations.
  loom_sanitizer_reporting_mode_t reporting_mode;
} loom_sanitizer_runtime_requirements_state_t;

loom_sanitizer_runtime_requirements_t
loom_sanitizer_runtime_requirements_from_options(
    const loom_sanitizer_options_t* options) {
  if (options == NULL || !loom_sanitizer_options_is_enabled(options)) {
    return LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE;
  }

  loom_sanitizer_runtime_requirements_t requirements =
      options->reporting_mode == LOOM_SANITIZER_REPORTING_MODE_TRAP
          ? LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE
          : LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK;
  if (iree_any_bit_set(options->checks, LOOM_SANITIZER_CHECK_ACCESS)) {
    requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW;
  }
  if (iree_any_bit_set(options->checks, LOOM_SANITIZER_CHECK_RACE)) {
    requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW;
  }
  return requirements;
}

static void loom_sanitizer_runtime_requirements_add_feedback(
    loom_sanitizer_runtime_requirements_state_t* state) {
  if (state->reporting_mode != LOOM_SANITIZER_REPORTING_MODE_TRAP) {
    state->requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK;
  }
}

static iree_status_t loom_sanitizer_runtime_requirements_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  loom_sanitizer_runtime_requirements_state_t* state =
      (loom_sanitizer_runtime_requirements_state_t*)user_data;

  switch (op->kind) {
    case LOOM_OP_KERNEL_ASSERT:
    case LOOM_OP_SANITIZER_ASSERT_VALUE:
    case LOOM_OP_SANITIZER_ASSERT_OP:
    case LOOM_OP_SANITIZER_ASSERT_LAYOUT:
      loom_sanitizer_runtime_requirements_add_feedback(state);
      break;
    case LOOM_OP_SANITIZER_ASSERT_ACCESS:
    case LOOM_OP_SANITIZER_ASSERT_ACCESSES:
      state->requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW;
      loom_sanitizer_runtime_requirements_add_feedback(state);
      break;
    case LOOM_OP_SANITIZER_RACE_ACCESS:
    case LOOM_OP_SANITIZER_RACE_FRAGMENT_ACCESS:
      state->requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW;
      loom_sanitizer_runtime_requirements_add_feedback(state);
      break;
    case LOOM_OP_SANITIZER_RACE_SYNC:
      state->requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW;
      break;
    default:
      break;
  }

  *out_result = state->requirements == state->complete_requirements
                    ? LOOM_WALK_ABORT
                    : LOOM_WALK_CONTINUE;
  return iree_ok_status();
}

iree_status_t loom_sanitizer_runtime_requirements_query(
    const loom_module_t* module, const loom_sanitizer_options_t* options,
    loom_sanitizer_runtime_requirements_t* out_requirements) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_requirements);
  *out_requirements = LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE;

  const loom_sanitizer_reporting_mode_t reporting_mode =
      options != NULL ? options->reporting_mode
                      : LOOM_SANITIZER_REPORTING_MODE_DEFAULT;
  loom_sanitizer_runtime_requirements_state_t state = {
      .requirements = loom_sanitizer_runtime_requirements_from_options(options),
      .complete_requirements =
          LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW |
          LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW |
          (reporting_mode == LOOM_SANITIZER_REPORTING_MODE_TRAP
               ? LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE
               : LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK),
      .reporting_mode = reporting_mode,
  };
  if (state.requirements == state.complete_requirements) {
    *out_requirements = state.requirements;
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < module->symbols.count; ++i) {
    loom_func_like_t function = loom_func_like_const_cast(
        module, module->symbols.entries[i].defining_op);
    if (!loom_func_like_isa(function)) {
      continue;
    }

    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    status = loom_walk_function(
        module, function, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_sanitizer_runtime_requirements_visit,
                               &state},
        &walk_result);
    if (walk_result == LOOM_WALK_ABORT) {
      break;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_requirements = state.requirements;
  }
  return status;
}
