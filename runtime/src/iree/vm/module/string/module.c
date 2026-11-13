// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/module/string/module.h"

#include "iree/base/internal/unicode.h"
#include "iree/vm/reflection.h"

typedef struct iree_vm_string_module_t {
  // Native module prefix retained by every linked program.
  iree_vm_module_t base;
  // Allocator owning this module; calls require no additional storage.
  iree_allocator_t host_allocator;
} iree_vm_string_module_t;

enum {
  IREE_VM_STRING_MODULE_CODEPOINT_IS_DECIMAL,
  IREE_VM_STRING_MODULE_CODEPOINT_IS_DIGIT,
  IREE_VM_STRING_MODULE_CODEPOINT_IS_WHITESPACE,
  IREE_VM_STRING_MODULE_CODEPOINT_TO_LOWER,
  IREE_VM_STRING_MODULE_CODEPOINT_TO_UPPER,
};

static iree_status_t iree_vm_string_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  const iree_vm_call_packet_t* call = &params->call;
  const uint32_t codepoint =
      (uint32_t)iree_vm_call_value_argument_load(call, 0);
  switch (params->function_ordinal) {
    case IREE_VM_STRING_MODULE_CODEPOINT_IS_DECIMAL:
      iree_vm_call_value_result_store(call, 0,
                                      iree_unicode_is_decimal(codepoint));
      break;
    case IREE_VM_STRING_MODULE_CODEPOINT_IS_DIGIT:
      iree_vm_call_value_result_store(call, 0,
                                      iree_unicode_is_digit(codepoint));
      break;
    case IREE_VM_STRING_MODULE_CODEPOINT_IS_WHITESPACE:
      iree_vm_call_value_result_store(call, 0,
                                      iree_unicode_is_whitespace(codepoint));
      break;
    case IREE_VM_STRING_MODULE_CODEPOINT_TO_LOWER: {
      uint32_t values[2] = {0};
      const iree_host_size_t count = iree_unicode_to_lower(codepoint, values);
      iree_vm_call_value_result_store(call, 0, count);
      iree_vm_call_value_result_store(call, 1, values[0]);
      iree_vm_call_value_result_store(call, 2, values[1]);
      break;
    }
    case IREE_VM_STRING_MODULE_CODEPOINT_TO_UPPER: {
      uint32_t values[3] = {0};
      const iree_host_size_t count = iree_unicode_to_upper(codepoint, values);
      iree_vm_call_value_result_store(call, 0, count);
      iree_vm_call_value_result_store(call, 1, values[0]);
      iree_vm_call_value_result_store(call, 2, values[1]);
      iree_vm_call_value_result_store(call, 3, values[2]);
      break;
    }
    default:
      IREE_CHECK_UNREACHABLE("invalid string function ordinal");
  }
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static const iree_vm_module_signature_type_t iree_vm_string_module_i32[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
};
static const iree_vm_module_callable_type_declaration_t
    iree_vm_string_module_callables[] = {
        {.signature = {.arguments = {iree_vm_string_module_i32, 1, 1, 0, 0},
                       .results = {iree_vm_string_module_i32, 1, 1, 0, 0}}},
        {.signature = {.arguments = {iree_vm_string_module_i32, 1, 1, 0, 0},
                       .results = {iree_vm_string_module_i32, 3, 3, 0, 0}}},
        {.signature = {.arguments = {iree_vm_string_module_i32, 1, 1, 0, 0},
                       .results = {iree_vm_string_module_i32, 4, 4, 0, 0}}},
};
static const iree_vm_module_export_declaration_t
    iree_vm_string_module_exports[] = {
        {IREE_SVL("codepoint.is_decimal"), 0,
         IREE_VM_STRING_MODULE_CODEPOINT_IS_DECIMAL, 0},
        {IREE_SVL("codepoint.is_digit"), 0,
         IREE_VM_STRING_MODULE_CODEPOINT_IS_DIGIT, 0},
        {IREE_SVL("codepoint.is_whitespace"), 0,
         IREE_VM_STRING_MODULE_CODEPOINT_IS_WHITESPACE, 0},
        {IREE_SVL("codepoint.to_lower"), 1,
         IREE_VM_STRING_MODULE_CODEPOINT_TO_LOWER, 0},
        {IREE_SVL("codepoint.to_upper"), 2,
         IREE_VM_STRING_MODULE_CODEPOINT_TO_UPPER, 0},
};
static const iree_vm_module_descriptor_t iree_vm_string_module_descriptor = {
    .name = IREE_SVL("string"),
    .flags = IREE_VM_MODULE_FLAG_LINKABLE,
    .counts = {.function_count = IREE_ARRAYSIZE(iree_vm_string_module_exports),
               .callable_type_count =
                   IREE_ARRAYSIZE(iree_vm_string_module_callables),
               .export_count = IREE_ARRAYSIZE(iree_vm_string_module_exports),
               .callable_fields = {.value_count = 11}},
};

static void iree_vm_string_module_destroy(iree_vm_module_t* base) {
  iree_vm_string_module_t* module = (iree_vm_string_module_t*)base;
  iree_allocator_free(module->host_allocator, module);
}
static void iree_vm_string_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = iree_vm_string_module_exports[ordinal];
}
static void iree_vm_string_module_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = iree_vm_string_module_callables[ordinal];
}
static void iree_vm_string_module_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("String module has no imports");
}
static void iree_vm_string_module_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("String module has no imports");
}
static const iree_vm_module_vtable_t iree_vm_string_module_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = iree_vm_string_module_destroy,
    .function_start = iree_vm_string_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = iree_vm_string_module_query_import_group,
    .query_import = iree_vm_string_module_query_import,
    .query_export = iree_vm_string_module_query_export,
    .query_callable_type = iree_vm_string_module_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

IREE_API_EXPORT iree_status_t iree_vm_string_module_create(
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_vm_string_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  iree_status_t status = iree_vm_module_initialize(
      &iree_vm_string_module_vtable, &iree_vm_string_module_descriptor,
      &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
