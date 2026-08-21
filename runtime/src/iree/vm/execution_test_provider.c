// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/execution_test_provider.h"

#include <string.h>

typedef struct iree_vm_execution_test_module_t {
  // Generic module base at offset zero.
  iree_vm_module_t base;
  // Allocator owning this module object.
  iree_allocator_t host_allocator;
  // Complete immutable descriptor published through |base|.
  iree_vm_module_descriptor_t descriptor;
  // Selected module implementation role.
  iree_vm_execution_test_module_kind_t kind;
  // Copied provider behavior.
  iree_vm_execution_test_options_t options;
  // Borrowed exclusive test observations.
  iree_vm_execution_test_counters_t* counters;
} iree_vm_execution_test_module_t;

static_assert(offsetof(iree_vm_execution_test_module_t, base) == 0,
              "test module base must remain at offset zero");

typedef uint32_t iree_vm_execution_test_process_phase_t;
enum iree_vm_execution_test_process_phase_e {
  IREE_VM_EXECUTION_TEST_PROCESS_PHASE_ZERO = 0u,
  IREE_VM_EXECUTION_TEST_PROCESS_PHASE_ATTACHED = 1u,
  IREE_VM_EXECUTION_TEST_PROCESS_PHASE_SEALED = 2u,
  IREE_VM_EXECUTION_TEST_PROCESS_PHASE_DETACHED = 3u,
};

typedef struct iree_vm_execution_test_process_state_t {
  // Current physical lifecycle phase.
  iree_vm_execution_test_process_phase_t phase;
  // Value stored by application.initialize.
  int32_t initialized_value;
} iree_vm_execution_test_process_state_t;

typedef uint32_t iree_vm_execution_test_frame_action_t;
enum iree_vm_execution_test_frame_action_e {
  IREE_VM_EXECUTION_TEST_FRAME_ACTION_INITIALIZE = 0u,
  IREE_VM_EXECUTION_TEST_FRAME_ACTION_YIELD_I32 = 1u,
  IREE_VM_EXECUTION_TEST_FRAME_ACTION_YIELD_REF = 2u,
  IREE_VM_EXECUTION_TEST_FRAME_ACTION_NESTED_PARENT = 3u,
};

typedef struct iree_vm_execution_test_frame_t {
  // Borrowed immutable module implementation.
  iree_vm_execution_test_module_t* module;
  // Stable physical call banks.
  iree_vm_call_packet_t call;
  // Frame-specific terminal behavior.
  iree_vm_execution_test_frame_action_t action;
  // Suspensions remaining before terminal work.
  uint32_t remaining_suspensions;
  // Copied level-triggered wake callback.
  iree_vm_invocation_wake_callback_t wake_callback;
} iree_vm_execution_test_frame_t;

//===----------------------------------------------------------------------===//
// Test Ref Type
//===----------------------------------------------------------------------===//

static const iree_vm_ref_type_table_t iree_vm_execution_test_type_table;

static void iree_vm_execution_test_object_destroy(void* object) {
  iree_vm_execution_test_object_t* test_object =
      (iree_vm_execution_test_object_t*)object;
  ++*test_object->destruction_count;
}

static const iree_vm_ref_type_descriptor_t
    iree_vm_execution_test_object_descriptor = {
        iree_vm_execution_test_object_destroy,
        &iree_vm_execution_test_type_table,
        IREE_SVL("object"),
};

static const iree_vm_ref_type_descriptor_t
    iree_vm_execution_test_wrong_object_descriptor = {
        iree_vm_execution_test_object_destroy,
        &iree_vm_execution_test_type_table,
        IREE_SVL("wrong_object"),
};

static const struct {
  // Canonical test object type at ordinal zero.
  iree_vm_ref_type_t object;
  // Deliberately incompatible test object type at ordinal one.
  iree_vm_ref_type_t wrong_object;
} iree_vm_execution_test_types = {
    &iree_vm_execution_test_object_descriptor,
    &iree_vm_execution_test_wrong_object_descriptor,
};

static const iree_vm_ref_type_table_t iree_vm_execution_test_type_table = {
    sizeof(iree_vm_execution_test_type_table),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SVL("execution.test"),
    {&iree_vm_execution_test_types, 2},
};

static const iree_vm_ref_type_t iree_vm_execution_test_module_ref_types[] = {
    &iree_vm_execution_test_object_descriptor,
    &iree_vm_execution_test_wrong_object_descriptor,
};

void iree_vm_execution_test_object_initialize(
    int* destruction_count, iree_vm_execution_test_object_t* out_object) {
  memset(out_object, 0, sizeof(*out_object));
  iree_vm_ref_object_initialize(&out_object->ref_object);
  out_object->destruction_count = destruction_count;
}

iree_vm_ref_type_t iree_vm_execution_test_object_type(void) {
  return &iree_vm_execution_test_object_descriptor;
}

//===----------------------------------------------------------------------===//
// Static Module Declarations
//===----------------------------------------------------------------------===//

static const iree_vm_module_signature_type_t
    iree_vm_execution_test_add_arguments[] = {
        {IREE_VM_SCALAR_TYPE_I32, 0},
        {IREE_VM_SCALAR_TYPE_I32, 0},
};

static const iree_vm_module_signature_type_t
    iree_vm_execution_test_i32_argument[] = {
        {IREE_VM_SCALAR_TYPE_I32, 0},
};

static const iree_vm_module_signature_type_t
    iree_vm_execution_test_i32_result[] = {
        {IREE_VM_SCALAR_TYPE_I32, 0},
};

static const iree_vm_module_signature_type_t
    iree_vm_execution_test_launch_config_arguments[] = {
        {IREE_VM_SCALAR_TYPE_I32, 0},
        {IREE_VM_SCALAR_TYPE_BF16, 0},
};

static const iree_vm_module_signature_type_t
    iree_vm_execution_test_launch_config_results[] = {
        {IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_SCALAR_TYPE_I64, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_SCALAR_TYPE_I64, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_SCALAR_TYPE_I64, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_SCALAR_TYPE_I64, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_SCALAR_TYPE_I64, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0},
};

static const iree_vm_module_signature_type_t
    iree_vm_execution_test_ref_argument[] = {
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
};

static const iree_vm_module_signature_type_t
    iree_vm_execution_test_ref_result[] = {
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
};

static const iree_vm_module_signature_type_t
    iree_vm_execution_test_function_arguments[] = {
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION, 0},
        {IREE_VM_SCALAR_TYPE_I32, 0},
        {IREE_VM_SCALAR_TYPE_I32, 0},
};

enum iree_vm_execution_test_callable_ordinal_e {
  IREE_VM_EXECUTION_TEST_CALLABLE_ADD = 0,
  IREE_VM_EXECUTION_TEST_CALLABLE_YIELD_I32 = 1,
  IREE_VM_EXECUTION_TEST_CALLABLE_ECHO_REF = 2,
  IREE_VM_EXECUTION_TEST_CALLABLE_FUNCTION = 3,
  IREE_VM_EXECUTION_TEST_CALLABLE_INITIALIZE = 4,
  IREE_VM_EXECUTION_TEST_CALLABLE_YIELD_REF = 5,
  IREE_VM_EXECUTION_TEST_CALLABLE_LAUNCH_CONFIG = 6,
};

static const iree_vm_module_callable_type_declaration_t
    iree_vm_execution_test_application_callable_types[] = {
        {
            .signature =
                {
                    .arguments = {iree_vm_execution_test_add_arguments,
                                  IREE_ARRAYSIZE(
                                      iree_vm_execution_test_add_arguments)},
                    .results = {iree_vm_execution_test_i32_result,
                                IREE_ARRAYSIZE(
                                    iree_vm_execution_test_i32_result)},
                },
            .flags = IREE_VM_CALLABLE_TYPE_FLAG_NONE,
        },
        {
            .signature =
                {
                    .arguments = {iree_vm_execution_test_i32_argument,
                                  IREE_ARRAYSIZE(
                                      iree_vm_execution_test_i32_argument)},
                    .results = {iree_vm_execution_test_i32_result,
                                IREE_ARRAYSIZE(
                                    iree_vm_execution_test_i32_result)},
                },
            .flags = IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
        },
        {
            .signature =
                {
                    .arguments = {iree_vm_execution_test_ref_argument,
                                  IREE_ARRAYSIZE(
                                      iree_vm_execution_test_ref_argument)},
                    .results = {iree_vm_execution_test_ref_result,
                                IREE_ARRAYSIZE(
                                    iree_vm_execution_test_ref_result)},
                },
            .flags = IREE_VM_CALLABLE_TYPE_FLAG_NONE,
        },
        {
            .signature =
                {
                    .arguments =
                        {iree_vm_execution_test_function_arguments,
                         IREE_ARRAYSIZE(
                             iree_vm_execution_test_function_arguments)},
                    .results = {iree_vm_execution_test_i32_result,
                                IREE_ARRAYSIZE(
                                    iree_vm_execution_test_i32_result)},
                },
            .flags = IREE_VM_CALLABLE_TYPE_FLAG_NONE,
        },
        {
            .signature =
                {
                    .arguments = {iree_vm_execution_test_i32_argument,
                                  IREE_ARRAYSIZE(
                                      iree_vm_execution_test_i32_argument)},
                    .results = {0},
                },
            .flags = IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
        },
        {
            .signature =
                {
                    .arguments = {iree_vm_execution_test_ref_argument,
                                  IREE_ARRAYSIZE(
                                      iree_vm_execution_test_ref_argument)},
                    .results = {iree_vm_execution_test_ref_result,
                                IREE_ARRAYSIZE(
                                    iree_vm_execution_test_ref_result)},
                },
            .flags = IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
        },
        {
            .signature =
                {
                    .arguments =
                        {iree_vm_execution_test_launch_config_arguments,
                         IREE_ARRAYSIZE(
                             iree_vm_execution_test_launch_config_arguments)},
                    .results =
                        {iree_vm_execution_test_launch_config_results,
                         IREE_ARRAYSIZE(
                             iree_vm_execution_test_launch_config_results)},
                },
            .flags = IREE_VM_CALLABLE_TYPE_FLAG_NONE,
        },
};

static const iree_vm_module_callable_type_declaration_t
    iree_vm_execution_test_math_callable_types[] = {
        {
            .signature =
                {
                    .arguments = {iree_vm_execution_test_add_arguments,
                                  IREE_ARRAYSIZE(
                                      iree_vm_execution_test_add_arguments)},
                    .results = {iree_vm_execution_test_i32_result,
                                IREE_ARRAYSIZE(
                                    iree_vm_execution_test_i32_result)},
                },
            .flags = IREE_VM_CALLABLE_TYPE_FLAG_NONE,
        },
        {
            .signature =
                {
                    .arguments = {iree_vm_execution_test_i32_argument,
                                  IREE_ARRAYSIZE(
                                      iree_vm_execution_test_i32_argument)},
                    .results = {iree_vm_execution_test_i32_result,
                                IREE_ARRAYSIZE(
                                    iree_vm_execution_test_i32_result)},
                },
            .flags = IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
        },
};

enum iree_vm_execution_test_application_function_ordinal_e {
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_ADD = 0,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_BAD_YIELD_REF = 1,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_CALL_FUNCTION = 2,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_CALL_IMPORT = 3,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_CALL_LOCAL = 4,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_ECHO_REF = 5,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_INITIALIZE = 6,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_NESTED_YIELD = 7,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_YIELD_REF = 8,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_YIELD_TWICE = 9,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_LAUNCH_CONFIG = 10,
  IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_COUNT = 11,
};

static const iree_vm_module_export_declaration_t
    iree_vm_execution_test_application_exports[] = {
        {IREE_SVL("add"), IREE_VM_EXECUTION_TEST_CALLABLE_ADD,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_ADD, 0},
        {IREE_SVL("bad_yield_ref"), IREE_VM_EXECUTION_TEST_CALLABLE_ECHO_REF,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_BAD_YIELD_REF, 0},
        {IREE_SVL("call_function"), IREE_VM_EXECUTION_TEST_CALLABLE_FUNCTION,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_CALL_FUNCTION, 0},
        {IREE_SVL("call_import"), IREE_VM_EXECUTION_TEST_CALLABLE_ADD,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_CALL_IMPORT, 0},
        {IREE_SVL("call_local"), IREE_VM_EXECUTION_TEST_CALLABLE_ADD,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_CALL_LOCAL, 0},
        {IREE_SVL("echo_ref"), IREE_VM_EXECUTION_TEST_CALLABLE_ECHO_REF,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_ECHO_REF, 0},
        {IREE_SVL("initialize"), IREE_VM_EXECUTION_TEST_CALLABLE_INITIALIZE,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_INITIALIZE, 0},
        {IREE_SVL("launch_config"),
         IREE_VM_EXECUTION_TEST_CALLABLE_LAUNCH_CONFIG,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_LAUNCH_CONFIG, 0},
        {IREE_SVL("nested_yield"), IREE_VM_EXECUTION_TEST_CALLABLE_YIELD_I32,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_NESTED_YIELD, 0},
        {IREE_SVL("yield_ref"), IREE_VM_EXECUTION_TEST_CALLABLE_YIELD_REF,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_YIELD_REF, 0},
        {IREE_SVL("yield_twice"), IREE_VM_EXECUTION_TEST_CALLABLE_YIELD_I32,
         IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_YIELD_TWICE, 0},
};

static const iree_vm_module_import_group_t
    iree_vm_execution_test_application_import_groups[] = {
        {IREE_SVL("execution.math"), 0, 2},
};

static const iree_vm_module_import_declaration_t
    iree_vm_execution_test_application_imports[] = {
        {IREE_SVL("execution.math"), IREE_SVL("add"),
         IREE_VM_EXECUTION_TEST_CALLABLE_ADD, IREE_VM_MODULE_IMPORT_FLAG_NONE,
         0},
        {IREE_SVL("execution.math"), IREE_SVL("suspend_add"),
         IREE_VM_EXECUTION_TEST_CALLABLE_YIELD_I32,
         IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
};

enum iree_vm_execution_test_math_function_ordinal_e {
  IREE_VM_EXECUTION_TEST_MATH_FUNCTION_ADD = 0,
  IREE_VM_EXECUTION_TEST_MATH_FUNCTION_SUSPEND_ADD = 1,
  IREE_VM_EXECUTION_TEST_MATH_FUNCTION_COUNT = 2,
};

static const iree_vm_module_export_declaration_t
    iree_vm_execution_test_math_exports[] = {
        {IREE_SVL("add"), IREE_VM_EXECUTION_TEST_CALLABLE_ADD,
         IREE_VM_EXECUTION_TEST_MATH_FUNCTION_ADD, 0},
        {IREE_SVL("suspend_add"), IREE_VM_EXECUTION_TEST_CALLABLE_YIELD_I32,
         IREE_VM_EXECUTION_TEST_MATH_FUNCTION_SUSPEND_ADD, 0},
};

//===----------------------------------------------------------------------===//
// Module Lifetime And Queries
//===----------------------------------------------------------------------===//

static iree_vm_execution_test_module_t* iree_vm_execution_test_module_cast(
    iree_vm_module_t* base_module) {
  return iree_containerof(base_module, iree_vm_execution_test_module_t, base);
}

static const iree_vm_execution_test_module_t*
iree_vm_execution_test_module_const_cast(const iree_vm_module_t* base_module) {
  return iree_containerof(base_module, iree_vm_execution_test_module_t, base);
}

static void iree_vm_execution_test_module_destroy(
    iree_vm_module_t* base_module) {
  iree_vm_execution_test_module_t* module =
      iree_vm_execution_test_module_cast(base_module);
  ++module->counters->module_destruction_count;
  iree_allocator_free(module->host_allocator, module);
}

static void iree_vm_execution_test_query_import_group(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group) {
  const iree_vm_execution_test_module_t* module =
      iree_vm_execution_test_module_const_cast(base_module);
  if (module->kind == IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION) {
    *out_group = iree_vm_execution_test_application_import_groups[ordinal];
  } else {
    memset(out_group, 0, sizeof(*out_group));
  }
}

static void iree_vm_execution_test_query_import(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import) {
  const iree_vm_execution_test_module_t* module =
      iree_vm_execution_test_module_const_cast(base_module);
  if (module->kind == IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION) {
    *out_import = iree_vm_execution_test_application_imports[ordinal];
  } else {
    memset(out_import, 0, sizeof(*out_import));
  }
}

static void iree_vm_execution_test_query_export(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export) {
  const iree_vm_execution_test_module_t* module =
      iree_vm_execution_test_module_const_cast(base_module);
  *out_export = module->kind == IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION
                    ? iree_vm_execution_test_application_exports[ordinal]
                    : iree_vm_execution_test_math_exports[ordinal];
}

static void iree_vm_execution_test_query_callable_type(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type) {
  const iree_vm_execution_test_module_t* module =
      iree_vm_execution_test_module_const_cast(base_module);
  *out_callable_type =
      module->kind == IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION
          ? iree_vm_execution_test_application_callable_types[ordinal]
          : iree_vm_execution_test_math_callable_types[ordinal];
}

//===----------------------------------------------------------------------===//
// Process State Lifecycle
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_execution_test_attach_state(
    iree_vm_module_t* base_module, iree_byte_span_t zeroed_storage,
    iree_allocator_t host_allocator) {
  iree_vm_execution_test_module_t* module =
      iree_vm_execution_test_module_cast(base_module);
  (void)host_allocator;
  ++module->counters->attach_count;
  if (module->kind == IREE_VM_EXECUTION_TEST_MODULE_KIND_MATH) {
    if (zeroed_storage.data || zeroed_storage.data_length != 0) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "zero-state math module received storage");
    }
  } else {
    if (!zeroed_storage.data ||
        zeroed_storage.data_length !=
            sizeof(iree_vm_execution_test_process_state_t)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "application process storage is invalid");
    }
    iree_vm_execution_test_process_state_t* state =
        (iree_vm_execution_test_process_state_t*)zeroed_storage.data;
    if (state->phase != IREE_VM_EXECUTION_TEST_PROCESS_PHASE_ZERO) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "application process storage is not zeroed");
    }
    state->phase = IREE_VM_EXECUTION_TEST_PROCESS_PHASE_ATTACHED;
    if (iree_any_bit_set(module->options.flags,
                         IREE_VM_EXECUTION_TEST_FLAG_FAIL_ATTACH)) {
      memset(state, 0, sizeof(*state));
      ++module->counters->attach_self_cleanup_count;
      return iree_make_status(IREE_STATUS_ABORTED, "scripted attach failure");
    }
  }
  if (iree_any_bit_set(module->options.flags,
                       IREE_VM_EXECUTION_TEST_FLAG_FAIL_ATTACH)) {
    ++module->counters->attach_self_cleanup_count;
    return iree_make_status(IREE_STATUS_ABORTED, "scripted attach failure");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_execution_test_seal_state(
    iree_vm_module_t* base_module, iree_byte_span_t storage) {
  iree_vm_execution_test_module_t* module =
      iree_vm_execution_test_module_cast(base_module);
  ++module->counters->seal_count;
  if (module->kind == IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION) {
    iree_vm_execution_test_process_state_t* state =
        (iree_vm_execution_test_process_state_t*)storage.data;
    if (!state ||
        state->phase != IREE_VM_EXECUTION_TEST_PROCESS_PHASE_ATTACHED ||
        state->initialized_value == 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "application process state is incomplete");
    }
    if (iree_any_bit_set(module->options.flags,
                         IREE_VM_EXECUTION_TEST_FLAG_FAIL_SEAL)) {
      return iree_make_status(IREE_STATUS_ABORTED, "scripted seal failure");
    }
    state->phase = IREE_VM_EXECUTION_TEST_PROCESS_PHASE_SEALED;
  } else if (storage.data || storage.data_length != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "zero-state math module received storage");
  } else if (iree_any_bit_set(module->options.flags,
                              IREE_VM_EXECUTION_TEST_FLAG_FAIL_SEAL)) {
    return iree_make_status(IREE_STATUS_ABORTED, "scripted seal failure");
  }
  return iree_ok_status();
}

static void iree_vm_execution_test_detach_state(iree_vm_module_t* base_module,
                                                iree_byte_span_t storage) {
  iree_vm_execution_test_module_t* module =
      iree_vm_execution_test_module_cast(base_module);
  ++module->counters->detach_count;
  if (module->kind == IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION) {
    iree_vm_execution_test_process_state_t* state =
        (iree_vm_execution_test_process_state_t*)storage.data;
    state->phase = IREE_VM_EXECUTION_TEST_PROCESS_PHASE_DETACHED;
    state->initialized_value = 0;
  }
}

//===----------------------------------------------------------------------===//
// Function Execution
//===----------------------------------------------------------------------===//

static void iree_vm_execution_test_wake(
    iree_vm_invocation_wake_callback_t wake_callback) {
  if (wake_callback.function) {
    wake_callback.function(wake_callback.user_data);
  }
}

static void iree_vm_execution_test_frame_cleanup(iree_vm_frame_t* frame) {
  iree_vm_execution_test_frame_t* payload =
      (iree_vm_execution_test_frame_t*)iree_vm_frame_storage(frame);
  ++payload->module->counters->frame_cleanup_count;
}

static iree_status_t iree_vm_execution_test_suspend(
    iree_vm_execution_test_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_test_frame_action_t action, uint32_t suspension_count,
    iree_vm_execution_outcome_t* out_outcome) {
  const iree_vm_frame_layout_t layout = {
      2048,
      iree_alignof(iree_vm_execution_test_frame_t),
  };
  iree_vm_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_invocation_push_frame(
      params, layout, iree_vm_execution_test_frame_cleanup, &frame));
  iree_vm_execution_test_frame_t* payload =
      (iree_vm_execution_test_frame_t*)iree_vm_frame_storage(frame);
  payload->module = module;
  payload->call = params->call;
  payload->action = action;
  payload->remaining_suspensions = suspension_count;
  payload->wake_callback =
      iree_vm_invocation_wake_callback(params->execution.invocation);
  iree_vm_execution_test_wake(payload->wake_callback);
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  return iree_ok_status();
}

static void iree_vm_execution_test_add(const iree_vm_call_packet_t* call) {
  const uint32_t lhs = (uint32_t)iree_vm_call_value_argument_load(call, 0);
  const uint32_t rhs = (uint32_t)iree_vm_call_value_argument_load(call, 1);
  iree_vm_call_value_result_store(call, 0, lhs + rhs);
}

static void iree_vm_execution_test_launch_config(
    const iree_vm_call_packet_t* call) {
  const uint32_t row_count =
      (uint32_t)iree_vm_call_value_argument_load(call, 0);
  const uint16_t scale_bits =
      (uint16_t)iree_vm_call_value_argument_load(call, 1);
  const uint32_t scale_f32_bits = (uint32_t)scale_bits << 16;
  float scale_value = 0.0f;
  memcpy(&scale_value, &scale_f32_bits, sizeof(scale_value));
  const uint32_t scale =
      scale_value >= 0.0f && scale_value < 0x1p32f ? (uint32_t)scale_value : 0;
  const uint64_t values[] = {
      (uint64_t)row_count * scale, 1, 1, 1, 1, 1, 1, 1, 1, 32, 256,
  };
  for (uint16_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    iree_vm_call_value_result_store(call, i, values[i]);
  }
}

static iree_status_t iree_vm_execution_test_application_start(
    iree_vm_execution_test_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  switch (params->function_ordinal) {
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_ADD:
      iree_vm_execution_test_add(&params->call);
      *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
      return iree_ok_status();
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_BAD_YIELD_REF:
      return iree_vm_execution_test_suspend(
          module, params, IREE_VM_EXECUTION_TEST_FRAME_ACTION_YIELD_REF, 1,
          out_outcome);
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_CALL_FUNCTION: {
      const iree_vm_function_ref_t function_ref =
          iree_vm_call_function_argument_load(&params->call, 0);
      return iree_vm_invocation_call_function_ref(
          &params->execution, function_ref, IREE_VM_EXECUTION_TEST_CALLABLE_ADD,
          &params->call, out_outcome);
    }
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_CALL_IMPORT:
      return iree_vm_invocation_call_import(&params->execution, 0,
                                            &params->call, out_outcome);
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_CALL_LOCAL: {
      const iree_vm_module_local_function_t local_function = {
          IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_ADD,
          IREE_VM_EXECUTION_TEST_CALLABLE_ADD,
          IREE_VM_MODULE_FUNCTION_FLAG_NONE,
      };
      return iree_vm_invocation_call_local(&params->execution, local_function,
                                           &params->call, out_outcome);
    }
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_ECHO_REF: {
      iree_vm_ref_t ref = iree_vm_ref_null();
      iree_vm_call_ref_argument_load_move(&params->call, 0, &ref);
      if (iree_any_bit_set(module->options.flags,
                           IREE_VM_EXECUTION_TEST_FLAG_RETURN_WRONG_REF_TYPE) &&
          ref.object) {
        ref.type_and_state =
            (ref.type_and_state & IREE_VM_REF_STATE_MASK) |
            (uintptr_t)iree_vm_execution_test_types.wrong_object;
      }
      iree_vm_call_ref_result_store_move(&params->call, 0, &ref);
      *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
      return iree_ok_status();
    }
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_INITIALIZE: {
      const uint32_t suspension_count =
          module->options.initializer_suspension_count;
      if (suspension_count != 0) {
        return iree_vm_execution_test_suspend(
            module, params, IREE_VM_EXECUTION_TEST_FRAME_ACTION_INITIALIZE,
            suspension_count, out_outcome);
      }
      iree_vm_execution_test_process_state_t* state =
          (iree_vm_execution_test_process_state_t*)
              params->execution.process_storage;
      state->initialized_value =
          (int32_t)(uint32_t)iree_vm_call_value_argument_load(&params->call, 0);
      *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
      return iree_ok_status();
    }
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_NESTED_YIELD: {
      const iree_vm_frame_layout_t layout = {
          sizeof(iree_vm_execution_test_frame_t),
          iree_alignof(iree_vm_execution_test_frame_t),
      };
      iree_vm_frame_t* frame = NULL;
      IREE_RETURN_IF_ERROR(iree_vm_invocation_push_frame(
          params, layout, iree_vm_execution_test_frame_cleanup, &frame));
      iree_vm_execution_test_frame_t* payload =
          (iree_vm_execution_test_frame_t*)iree_vm_frame_storage(frame);
      payload->module = module;
      payload->call = params->call;
      payload->action = IREE_VM_EXECUTION_TEST_FRAME_ACTION_NESTED_PARENT;
      payload->remaining_suspensions = 0;
      payload->wake_callback = (iree_vm_invocation_wake_callback_t){0};
      iree_status_t status = iree_vm_invocation_call_import(
          &params->execution, 1, &payload->call, out_outcome);
      if (iree_status_is_ok(status) &&
          *out_outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
        iree_vm_invocation_pop_frame(params->execution.invocation, frame);
      }
      return status;
    }
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_YIELD_REF:
      return iree_vm_execution_test_suspend(
          module, params, IREE_VM_EXECUTION_TEST_FRAME_ACTION_YIELD_REF, 1,
          out_outcome);
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_YIELD_TWICE:
      return iree_vm_execution_test_suspend(
          module, params, IREE_VM_EXECUTION_TEST_FRAME_ACTION_YIELD_I32, 2,
          out_outcome);
    case IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_LAUNCH_CONFIG:
      iree_vm_execution_test_launch_config(&params->call);
      *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown application function ordinal");
  }
}

static iree_status_t iree_vm_execution_test_math_start(
    iree_vm_execution_test_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  switch (params->function_ordinal) {
    case IREE_VM_EXECUTION_TEST_MATH_FUNCTION_ADD:
      iree_vm_execution_test_add(&params->call);
      *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
      return iree_ok_status();
    case IREE_VM_EXECUTION_TEST_MATH_FUNCTION_SUSPEND_ADD:
      return iree_vm_execution_test_suspend(
          module, params, IREE_VM_EXECUTION_TEST_FRAME_ACTION_YIELD_I32, 1,
          out_outcome);
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown math function ordinal");
  }
}

static iree_status_t iree_vm_execution_test_function_start(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_execution_test_module_t* module =
      iree_vm_execution_test_module_cast(base_module);
  ++module->counters->function_start_count;
  if (iree_any_bit_set(module->options.flags,
                       IREE_VM_EXECUTION_TEST_FLAG_FAIL_FUNCTION)) {
    return iree_make_status(IREE_STATUS_ABORTED, "scripted function failure");
  }
  return module->kind == IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION
             ? iree_vm_execution_test_application_start(module, params,
                                                        out_outcome)
             : iree_vm_execution_test_math_start(module, params, out_outcome);
}

static iree_status_t iree_vm_execution_test_function_resume(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_execution_test_module_t* module =
      iree_vm_execution_test_module_cast(base_module);
  ++module->counters->function_resume_count;
  iree_vm_execution_test_frame_t* payload =
      (iree_vm_execution_test_frame_t*)iree_vm_frame_storage(params->frame);
  if (payload->remaining_suspensions > 1) {
    --payload->remaining_suspensions;
    iree_vm_execution_test_wake(payload->wake_callback);
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
    return iree_ok_status();
  }

  switch (payload->action) {
    case IREE_VM_EXECUTION_TEST_FRAME_ACTION_INITIALIZE: {
      iree_vm_execution_test_process_state_t* state =
          (iree_vm_execution_test_process_state_t*)
              params->execution.process_storage;
      state->initialized_value =
          (int32_t)(uint32_t)iree_vm_call_value_argument_load(&payload->call,
                                                              0);
      break;
    }
    case IREE_VM_EXECUTION_TEST_FRAME_ACTION_YIELD_I32: {
      const uint32_t value =
          (uint32_t)iree_vm_call_value_argument_load(&payload->call, 0);
      iree_vm_call_value_result_store(&payload->call, 0, value + 1);
      break;
    }
    case IREE_VM_EXECUTION_TEST_FRAME_ACTION_YIELD_REF: {
      iree_vm_ref_t ref = iree_vm_ref_null();
      iree_vm_call_ref_argument_load_move(&payload->call, 0, &ref);
      iree_vm_call_ref_result_store_move(&payload->call, 0, &ref);
      break;
    }
    case IREE_VM_EXECUTION_TEST_FRAME_ACTION_NESTED_PARENT:
      break;
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown execution test frame action");
  }
  iree_vm_invocation_pop_frame(params->execution.invocation, params->frame);
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static const iree_vm_module_vtable_t iree_vm_execution_test_module_vtable = {
    sizeof(iree_vm_execution_test_module_vtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    iree_vm_execution_test_module_destroy,
    iree_vm_execution_test_function_start,
    iree_vm_execution_test_function_resume,
    iree_vm_execution_test_attach_state,
    iree_vm_execution_test_seal_state,
    iree_vm_execution_test_detach_state,
    iree_vm_execution_test_query_import_group,
    iree_vm_execution_test_query_import,
    iree_vm_execution_test_query_export,
    iree_vm_execution_test_query_callable_type,
    iree_vm_module_query_presentation_none,
    iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t iree_vm_execution_test_module_create(
    iree_vm_execution_test_module_kind_t kind,
    iree_vm_execution_test_options_t options,
    iree_vm_execution_test_counters_t* counters,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  if ((kind != IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION &&
       kind != IREE_VM_EXECUTION_TEST_MODULE_KIND_MATH) ||
      !counters || !out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid execution test module request");
  }
  *out_module = NULL;
  iree_vm_execution_test_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  module->host_allocator = host_allocator;
  module->kind = kind;
  module->options = options;
  module->counters = counters;
  if (kind == IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION) {
    module->descriptor = (iree_vm_module_descriptor_t){
        .name = IREE_SVL("execution.app"),
        .flags = IREE_VM_MODULE_FLAG_LINKABLE,
        .ref_types = {iree_vm_execution_test_module_ref_types,
                      IREE_ARRAYSIZE(iree_vm_execution_test_module_ref_types)},
        .counts =
            {
                .function_count =
                    IREE_VM_EXECUTION_TEST_APPLICATION_FUNCTION_COUNT,
                .callable_type_count = IREE_ARRAYSIZE(
                    iree_vm_execution_test_application_callable_types),
                .import_group_count = IREE_ARRAYSIZE(
                    iree_vm_execution_test_application_import_groups),
                .import_count =
                    IREE_ARRAYSIZE(iree_vm_execution_test_application_imports),
                .export_count =
                    IREE_ARRAYSIZE(iree_vm_execution_test_application_exports),
                .metadata_count = 0,
            },
        .process_storage_size = sizeof(iree_vm_execution_test_process_state_t),
    };
  } else {
    module->descriptor = (iree_vm_module_descriptor_t){
        .name = IREE_SVL("execution.math"),
        .flags = IREE_VM_MODULE_FLAG_LINKABLE,
        .ref_types = {0},
        .counts =
            {
                .function_count = IREE_VM_EXECUTION_TEST_MATH_FUNCTION_COUNT,
                .callable_type_count =
                    IREE_ARRAYSIZE(iree_vm_execution_test_math_callable_types),
                .import_group_count = 0,
                .import_count = 0,
                .export_count =
                    IREE_ARRAYSIZE(iree_vm_execution_test_math_exports),
                .metadata_count = 0,
            },
        .process_storage_size = 0,
    };
  }

  iree_status_t status =
      iree_vm_module_initialize(&iree_vm_execution_test_module_vtable,
                                &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
