// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/invocation_test_module.h"

#include <string.h>

#include "iree/vm/reflection.h"

enum {
  IREE_VM_TEST_CALLABLE_YIELD_I32 = 0,
  IREE_VM_TEST_CALLABLE_REF = 1,
  IREE_VM_TEST_CALLABLE_YIELD_REF = 2,
  IREE_VM_TEST_CALLABLE_ADD = 3,
  IREE_VM_TEST_CALLABLE_F32 = 4,
  IREE_VM_TEST_CALLABLE_RETURN_FUNCTION = 5,
  IREE_VM_TEST_CALLABLE_CALL_FUNCTION = 6,
};

enum {
  IREE_VM_TEST_FUNCTION_ADD = 0,
  IREE_VM_TEST_FUNCTION_BAD_REF_RESULT = 1,
  IREE_VM_TEST_FUNCTION_BAD_YIELD_REF = 2,
  IREE_VM_TEST_FUNCTION_CALL_FUNCTION = 3,
  IREE_VM_TEST_FUNCTION_CALL_IMPORT = 4,
  IREE_VM_TEST_FUNCTION_CALL_LOCAL = 5,
  IREE_VM_TEST_FUNCTION_CALL_OPTIONAL = 6,
  IREE_VM_TEST_FUNCTION_ECHO_REF = 7,
  IREE_VM_TEST_FUNCTION_EXHAUST = 8,
  IREE_VM_TEST_FUNCTION_FAIL = 9,
  IREE_VM_TEST_FUNCTION_MULTIPLY_F32 = 10,
  IREE_VM_TEST_FUNCTION_RETURN_IMPORT = 11,
  IREE_VM_TEST_FUNCTION_RETURN_LOCAL = 12,
  IREE_VM_TEST_FUNCTION_RETURN_OPTIONAL = 13,
  IREE_VM_TEST_FUNCTION_YIELD_REF = 14,
  IREE_VM_TEST_FUNCTION_YIELD_TWICE = 15,
  IREE_VM_TEST_FUNCTION_COUNT = 16,
};

typedef enum iree_vm_test_frame_action_e {
  IREE_VM_TEST_FRAME_ACTION_CHILD = 0,
  IREE_VM_TEST_FRAME_ACTION_I32 = 1,
  IREE_VM_TEST_FRAME_ACTION_REF = 2,
} iree_vm_test_frame_action_t;

typedef struct iree_vm_test_frame_t {
  // Owning module used for cleanup observations.
  iree_vm_invocation_test_module_t* module;
  // Durable root or parent packet bases.
  iree_vm_call_packet_t call;
  // Resume behavior selected by the starting function.
  iree_vm_test_frame_action_t action;
  // Number of host suspensions remaining.
  uint32_t remaining_suspensions;
  // Copied host callback used to publish test readiness.
  iree_vm_invocation_wake_callback_t wake_callback;
} iree_vm_test_frame_t;

//===----------------------------------------------------------------------===//
// Object types
//===----------------------------------------------------------------------===//

static const iree_vm_ref_type_table_t iree_vm_test_type_table;

static void iree_vm_test_object_destroy(void* object) {
  iree_vm_invocation_test_object_t* test_object =
      (iree_vm_invocation_test_object_t*)object;
  ++*test_object->destroy_count;
}

static const iree_vm_ref_type_descriptor_t iree_vm_test_object_descriptor = {
    iree_vm_test_object_destroy,
    &iree_vm_test_type_table,
    IREE_SVL("object"),
};

static const iree_vm_ref_type_descriptor_t
    iree_vm_test_wrong_object_descriptor = {
        iree_vm_test_object_destroy,
        &iree_vm_test_type_table,
        IREE_SVL("wrong_object"),
};

static const struct {
  // Primary object type at ordinal zero.
  iree_vm_ref_type_t object;
  // Deliberately incompatible object type at ordinal one.
  iree_vm_ref_type_t wrong_object;
} iree_vm_test_types = {
    &iree_vm_test_object_descriptor,
    &iree_vm_test_wrong_object_descriptor,
};

static const iree_vm_ref_type_table_t iree_vm_test_type_table = {
    sizeof(iree_vm_test_type_table),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SVL("invocation.test"),
    {&iree_vm_test_types, 2},
};

static const iree_vm_ref_type_t iree_vm_test_module_ref_types[] = {
    &iree_vm_test_object_descriptor,
    &iree_vm_test_wrong_object_descriptor,
};

void iree_vm_invocation_test_object_initialize(
    int* destroy_count, iree_vm_invocation_test_object_t* out_object) {
  memset(out_object, 0, sizeof(*out_object));
  iree_vm_ref_object_initialize(&out_object->ref_object);
  out_object->destroy_count = destroy_count;
}

iree_vm_ref_type_t iree_vm_invocation_test_object_type(void) {
  return &iree_vm_test_object_descriptor;
}

//===----------------------------------------------------------------------===//
// Module declaration
//===----------------------------------------------------------------------===//

static const iree_vm_module_signature_type_t iree_vm_test_i32[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
};
static const iree_vm_module_signature_type_t iree_vm_test_ref[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
};
static const iree_vm_module_signature_type_t iree_vm_test_add_arguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
};
static const iree_vm_module_signature_type_t iree_vm_test_f32_arguments[] = {
    {IREE_VM_SCALAR_TYPE_F32, 0},
    {IREE_VM_SCALAR_TYPE_F32, 0},
};
static const iree_vm_module_signature_type_t iree_vm_test_f32_result[] = {
    {IREE_VM_SCALAR_TYPE_F32, 0},
};
static const iree_vm_module_signature_type_t iree_vm_test_function_result[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION, IREE_VM_TEST_CALLABLE_ADD},
};
static const iree_vm_module_signature_type_t
    iree_vm_test_call_function_arguments[] = {
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION,
         IREE_VM_TEST_CALLABLE_ADD},
        {IREE_VM_SCALAR_TYPE_I32, 0},
        {IREE_VM_SCALAR_TYPE_I32, 0},
};

static const iree_vm_module_callable_type_declaration_t
    iree_vm_test_callable_types[] = {
        {{{iree_vm_test_i32, IREE_ARRAYSIZE(iree_vm_test_i32)},
          {iree_vm_test_i32, IREE_ARRAYSIZE(iree_vm_test_i32)}},
         IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
         0,
         0},
        {{{iree_vm_test_ref, IREE_ARRAYSIZE(iree_vm_test_ref)},
          {iree_vm_test_ref, IREE_ARRAYSIZE(iree_vm_test_ref)}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
        {{{iree_vm_test_ref, IREE_ARRAYSIZE(iree_vm_test_ref)},
          {iree_vm_test_ref, IREE_ARRAYSIZE(iree_vm_test_ref)}},
         IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
         0,
         0},
        {{{iree_vm_test_add_arguments,
           IREE_ARRAYSIZE(iree_vm_test_add_arguments)},
          {iree_vm_test_i32, IREE_ARRAYSIZE(iree_vm_test_i32)}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
        {{{iree_vm_test_f32_arguments,
           IREE_ARRAYSIZE(iree_vm_test_f32_arguments)},
          {iree_vm_test_f32_result, IREE_ARRAYSIZE(iree_vm_test_f32_result)}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
        {{{NULL, 0},
          {iree_vm_test_function_result,
           IREE_ARRAYSIZE(iree_vm_test_function_result)}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         1,
         0},
        {{{iree_vm_test_call_function_arguments,
           IREE_ARRAYSIZE(iree_vm_test_call_function_arguments)},
          {iree_vm_test_i32, IREE_ARRAYSIZE(iree_vm_test_i32)}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         1,
         0},
};

static const iree_vm_module_import_group_t iree_vm_test_import_groups[] = {
    {IREE_SVL("invocation.test"), 0, 2},
};
static const iree_vm_module_import_declaration_t iree_vm_test_imports[] = {
    {IREE_SVL("invocation.test"), IREE_SVL("add"), IREE_VM_TEST_CALLABLE_ADD,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
    {IREE_SVL("invocation.test"), IREE_SVL("missing"),
     IREE_VM_TEST_CALLABLE_ADD, IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL, 0},
};

static const iree_vm_module_export_declaration_t iree_vm_test_exports[] = {
    {IREE_SVL("add"), IREE_VM_TEST_CALLABLE_ADD, IREE_VM_TEST_FUNCTION_ADD, 0},
    {IREE_SVL("bad_ref_result"), IREE_VM_TEST_CALLABLE_REF,
     IREE_VM_TEST_FUNCTION_BAD_REF_RESULT, 0},
    {IREE_SVL("bad_yield_ref"), IREE_VM_TEST_CALLABLE_REF,
     IREE_VM_TEST_FUNCTION_BAD_YIELD_REF, 0},
    {IREE_SVL("call_function"), IREE_VM_TEST_CALLABLE_CALL_FUNCTION,
     IREE_VM_TEST_FUNCTION_CALL_FUNCTION, 0},
    {IREE_SVL("call_import"), IREE_VM_TEST_CALLABLE_ADD,
     IREE_VM_TEST_FUNCTION_CALL_IMPORT, 0},
    {IREE_SVL("call_local"), IREE_VM_TEST_CALLABLE_ADD,
     IREE_VM_TEST_FUNCTION_CALL_LOCAL, 0},
    {IREE_SVL("call_optional"), IREE_VM_TEST_CALLABLE_ADD,
     IREE_VM_TEST_FUNCTION_CALL_OPTIONAL, 0},
    {IREE_SVL("echo_ref"), IREE_VM_TEST_CALLABLE_REF,
     IREE_VM_TEST_FUNCTION_ECHO_REF, 0},
    {IREE_SVL("exhaust"), IREE_VM_TEST_CALLABLE_ADD,
     IREE_VM_TEST_FUNCTION_EXHAUST, 0},
    {IREE_SVL("fail"), IREE_VM_TEST_CALLABLE_ADD, IREE_VM_TEST_FUNCTION_FAIL,
     0},
    {IREE_SVL("multiply_f32"), IREE_VM_TEST_CALLABLE_F32,
     IREE_VM_TEST_FUNCTION_MULTIPLY_F32, 0},
    {IREE_SVL("return_import"), IREE_VM_TEST_CALLABLE_RETURN_FUNCTION,
     IREE_VM_TEST_FUNCTION_RETURN_IMPORT, 0},
    {IREE_SVL("return_local"), IREE_VM_TEST_CALLABLE_RETURN_FUNCTION,
     IREE_VM_TEST_FUNCTION_RETURN_LOCAL, 0},
    {IREE_SVL("return_optional"), IREE_VM_TEST_CALLABLE_RETURN_FUNCTION,
     IREE_VM_TEST_FUNCTION_RETURN_OPTIONAL, 0},
    {IREE_SVL("yield_ref"), IREE_VM_TEST_CALLABLE_YIELD_REF,
     IREE_VM_TEST_FUNCTION_YIELD_REF, 0},
    {IREE_SVL("yield_twice"), IREE_VM_TEST_CALLABLE_YIELD_I32,
     IREE_VM_TEST_FUNCTION_YIELD_TWICE, 0},
};

//===----------------------------------------------------------------------===//
// Function execution
//===----------------------------------------------------------------------===//

static iree_vm_invocation_test_module_t* iree_vm_test_module_cast(
    iree_vm_module_t* base_module) {
  return iree_containerof(base_module, iree_vm_invocation_test_module_t, base);
}

static void iree_vm_test_module_destroy(iree_vm_module_t* base_module) {
  ++iree_vm_test_module_cast(base_module)->counters->destroy_count;
}

static void iree_vm_test_wake(iree_vm_invocation_wake_callback_t callback) {
  if (callback.fn) callback.fn(callback.user_data);
}

static void iree_vm_test_frame_cleanup(iree_vm_frame_t* frame) {
  iree_vm_test_frame_t* payload =
      (iree_vm_test_frame_t*)iree_vm_frame_storage(frame);
  ++payload->module->counters->cleanup_count;
}

static iree_status_t iree_vm_test_push_frame(
    iree_vm_invocation_test_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_test_frame_action_t action, uint32_t suspension_count,
    iree_vm_frame_t** out_frame) {
  const iree_vm_frame_layout_t layout = {
      sizeof(iree_vm_test_frame_t),
      iree_alignof(iree_vm_test_frame_t),
  };
  iree_vm_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_invocation_push_frame(
      params, layout, iree_vm_test_frame_cleanup, &frame));
  iree_vm_test_frame_t* payload =
      (iree_vm_test_frame_t*)iree_vm_frame_storage(frame);
  *payload = (iree_vm_test_frame_t){
      .module = module,
      .call = params->call,
      .action = action,
      .remaining_suspensions = suspension_count,
      .wake_callback =
          iree_vm_invocation_wake_callback(params->execution.invocation),
  };
  *out_frame = frame;
  return iree_ok_status();
}

static iree_status_t iree_vm_test_suspend(
    iree_vm_invocation_test_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_test_frame_action_t action, uint32_t suspension_count,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_test_push_frame(module, params, action,
                                               suspension_count, &frame));
  iree_vm_test_frame_t* payload =
      (iree_vm_test_frame_t*)iree_vm_frame_storage(frame);
  iree_vm_test_wake(payload->wake_callback);
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  return iree_ok_status();
}

static iree_status_t iree_vm_test_call_child(
    iree_vm_invocation_test_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    uint16_t function_ordinal, iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_test_push_frame(
      module, params, IREE_VM_TEST_FRAME_ACTION_CHILD, 0, &frame));
  (void)frame;
  if (function_ordinal == IREE_VM_TEST_FUNCTION_CALL_LOCAL) {
    const iree_vm_module_local_function_t local_function = {
        IREE_VM_TEST_FUNCTION_ADD,
        IREE_VM_TEST_CALLABLE_ADD,
        IREE_VM_MODULE_FUNCTION_FLAG_NONE,
    };
    return iree_vm_invocation_call_local(&params->execution, local_function,
                                         &params->call, out_outcome);
  }
  const uint16_t import_ordinal =
      function_ordinal == IREE_VM_TEST_FUNCTION_CALL_IMPORT ? 0 : 1;
  return iree_vm_invocation_call_import(&params->execution, import_ordinal,
                                        &params->call, out_outcome);
}

static void iree_vm_test_add(const iree_vm_call_packet_t* call) {
  const uint32_t lhs = (uint32_t)iree_vm_call_value_argument_load(call, 0);
  const uint32_t rhs = (uint32_t)iree_vm_call_value_argument_load(call, 1);
  iree_vm_call_value_result_store(call, 0, lhs + rhs);
}

static void iree_vm_test_multiply_f32(const iree_vm_call_packet_t* call) {
  const uint32_t lhs_bits = (uint32_t)iree_vm_call_value_argument_load(call, 0);
  const uint32_t rhs_bits = (uint32_t)iree_vm_call_value_argument_load(call, 1);
  float lhs = 0.0f;
  float rhs = 0.0f;
  memcpy(&lhs, &lhs_bits, sizeof(lhs));
  memcpy(&rhs, &rhs_bits, sizeof(rhs));
  const float product = lhs * rhs;
  uint32_t product_bits = 0;
  memcpy(&product_bits, &product, sizeof(product_bits));
  iree_vm_call_value_result_store(call, 0, product_bits);
}

static iree_status_t iree_vm_test_return_function(
    const iree_vm_module_function_start_params_t* params,
    uint16_t function_ordinal, iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_function_ref_t function_ref = iree_vm_function_ref_null();
  iree_status_t status = iree_ok_status();
  if (function_ordinal == IREE_VM_TEST_FUNCTION_RETURN_LOCAL) {
    const iree_vm_module_local_function_t local_function = {
        IREE_VM_TEST_FUNCTION_ADD,
        IREE_VM_TEST_CALLABLE_ADD,
        IREE_VM_MODULE_FUNCTION_FLAG_NONE,
    };
    status = iree_vm_function_ref_from_local_function(
        &params->execution, local_function, &function_ref);
  } else {
    const uint16_t import_ordinal =
        function_ordinal == IREE_VM_TEST_FUNCTION_RETURN_IMPORT ? 0 : 1;
    status = iree_vm_function_ref_from_import(&params->execution,
                                              import_ordinal, &function_ref);
  }
  if (!iree_status_is_ok(status)) return status;
  iree_vm_call_function_result_store(&params->call, 0, function_ref);
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static iree_status_t iree_vm_test_function_start(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_invocation_test_module_t* module =
      iree_vm_test_module_cast(base_module);
  ++module->counters->start_count;
  switch (params->function_ordinal) {
    case IREE_VM_TEST_FUNCTION_ADD:
      iree_vm_test_add(&params->call);
      break;
    case IREE_VM_TEST_FUNCTION_BAD_REF_RESULT:
    case IREE_VM_TEST_FUNCTION_ECHO_REF: {
      iree_vm_ref_t ref = iree_vm_ref_null();
      iree_vm_call_ref_argument_load_move(&params->call, 0, &ref);
      if (params->function_ordinal == IREE_VM_TEST_FUNCTION_BAD_REF_RESULT &&
          ref.object) {
        ref.type_and_state = (ref.type_and_state & IREE_VM_REF_STATE_MASK) |
                             (uintptr_t)&iree_vm_test_wrong_object_descriptor;
      }
      iree_vm_call_ref_result_store_move(&params->call, 0, &ref);
      break;
    }
    case IREE_VM_TEST_FUNCTION_BAD_YIELD_REF:
    case IREE_VM_TEST_FUNCTION_YIELD_REF:
      return iree_vm_test_suspend(module, params, IREE_VM_TEST_FRAME_ACTION_REF,
                                  1, out_outcome);
    case IREE_VM_TEST_FUNCTION_CALL_FUNCTION: {
      iree_vm_frame_t* frame = NULL;
      IREE_RETURN_IF_ERROR(iree_vm_test_push_frame(
          module, params, IREE_VM_TEST_FRAME_ACTION_CHILD, 0, &frame));
      (void)frame;
      const iree_vm_function_ref_t function_ref =
          iree_vm_call_function_argument_load(&params->call, 0);
      return iree_vm_invocation_call_function_ref(
          &params->execution, function_ref, IREE_VM_TEST_CALLABLE_ADD,
          &params->call, out_outcome);
    }
    case IREE_VM_TEST_FUNCTION_CALL_IMPORT:
    case IREE_VM_TEST_FUNCTION_CALL_LOCAL:
    case IREE_VM_TEST_FUNCTION_CALL_OPTIONAL:
      return iree_vm_test_call_child(module, params, params->function_ordinal,
                                     out_outcome);
    case IREE_VM_TEST_FUNCTION_EXHAUST: {
      const iree_vm_frame_layout_t layout = {
          32 * 1024,
          iree_max_align_t,
      };
      iree_vm_frame_t* frame = NULL;
      IREE_RETURN_IF_ERROR(
          iree_vm_invocation_push_frame(params, layout, NULL, &frame));
      iree_vm_invocation_pop_frame(params->execution.invocation, frame);
      iree_vm_test_add(&params->call);
      break;
    }
    case IREE_VM_TEST_FUNCTION_FAIL:
      return iree_make_status(IREE_STATUS_ABORTED, "injected function failure");
    case IREE_VM_TEST_FUNCTION_MULTIPLY_F32:
      iree_vm_test_multiply_f32(&params->call);
      break;
    case IREE_VM_TEST_FUNCTION_RETURN_IMPORT:
    case IREE_VM_TEST_FUNCTION_RETURN_LOCAL:
    case IREE_VM_TEST_FUNCTION_RETURN_OPTIONAL:
      return iree_vm_test_return_function(params, params->function_ordinal,
                                          out_outcome);
    case IREE_VM_TEST_FUNCTION_YIELD_TWICE:
      return iree_vm_test_suspend(module, params, IREE_VM_TEST_FRAME_ACTION_I32,
                                  2, out_outcome);
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown test function ordinal");
  }
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static iree_status_t iree_vm_test_function_resume(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_invocation_test_module_t* module =
      iree_vm_test_module_cast(base_module);
  ++module->counters->resume_count;
  iree_vm_test_frame_t* payload =
      (iree_vm_test_frame_t*)iree_vm_frame_storage(params->frame);
  if (payload->remaining_suspensions > 1) {
    --payload->remaining_suspensions;
    iree_vm_test_wake(payload->wake_callback);
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
    return iree_ok_status();
  }
  if (payload->action == IREE_VM_TEST_FRAME_ACTION_I32) {
    const uint32_t value =
        (uint32_t)iree_vm_call_value_argument_load(&payload->call, 0);
    iree_vm_call_value_result_store(&payload->call, 0, value + 1);
  } else if (payload->action == IREE_VM_TEST_FRAME_ACTION_REF) {
    iree_vm_ref_t ref = iree_vm_ref_null();
    iree_vm_call_ref_argument_load_move(&payload->call, 0, &ref);
    iree_vm_call_ref_result_store_move(&payload->call, 0, &ref);
  }
  iree_vm_invocation_pop_frame(params->execution.invocation, params->frame);
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Module queries and lifetime
//===----------------------------------------------------------------------===//

static void iree_vm_test_query_import_group(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group) {
  (void)module;
  *out_group = iree_vm_test_import_groups[ordinal];
}

static void iree_vm_test_query_import(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import) {
  (void)module;
  *out_import = iree_vm_test_imports[ordinal];
}

static void iree_vm_test_query_export(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export) {
  (void)module;
  *out_export = iree_vm_test_exports[ordinal];
}

static void iree_vm_test_query_callable_type(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type) {
  (void)module;
  *out_callable_type = iree_vm_test_callable_types[ordinal];
}

static const iree_vm_module_vtable_t iree_vm_test_module_vtable = {
    sizeof(iree_vm_test_module_vtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    iree_vm_test_module_destroy,
    iree_vm_test_function_start,
    iree_vm_test_function_resume,
    NULL,
    NULL,
    NULL,
    iree_vm_test_query_import_group,
    iree_vm_test_query_import,
    iree_vm_test_query_export,
    iree_vm_test_query_callable_type,
    iree_vm_module_query_presentation_none,
    iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t iree_vm_invocation_test_module_initialize(
    iree_vm_invocation_test_counters_t* counters,
    iree_vm_invocation_test_module_t* out_module) {
  memset(out_module, 0, sizeof(*out_module));
  out_module->counters = counters;
  out_module->descriptor = (iree_vm_module_descriptor_t){
      IREE_SV("invocation.test"),
      IREE_VM_MODULE_FLAG_LINKABLE,
      {iree_vm_test_module_ref_types,
       IREE_ARRAYSIZE(iree_vm_test_module_ref_types)},
      {IREE_VM_TEST_FUNCTION_COUNT, IREE_ARRAYSIZE(iree_vm_test_callable_types),
       IREE_ARRAYSIZE(iree_vm_test_import_groups),
       IREE_ARRAYSIZE(iree_vm_test_imports),
       IREE_ARRAYSIZE(iree_vm_test_exports), 0},
      0,
  };
  return iree_vm_module_initialize(&iree_vm_test_module_vtable,
                                   &out_module->descriptor, &out_module->base);
}
