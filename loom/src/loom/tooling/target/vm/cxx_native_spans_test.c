// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/vm/bytecode/module.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loom/tooling/target/vm/cxx_native_spans_bytecode.h"

// C++ projects each dynamic view to its extent and one borrowed span. Named
// positions keep the native consumer independent of literal packet offsets.
enum {
  SAMPLE_KEY_COUNT = 0,
  // Span descriptors occupy two values and start at an even bank ordinal.
  SAMPLE_KEYS = 2,
  SAMPLE_READING_COUNT = SAMPLE_KEYS + 2,
  SAMPLE_READINGS = SAMPLE_READING_COUNT + 2,
  SAMPLE_VALUE_COUNT = SAMPLE_READINGS + 2,
};

static const iree_vm_module_signature_type_t sample_arguments[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_SPAN |
         (iree_vm_module_signature_type_kind_t)IREE_VM_SCALAR_TYPE_I32,
     0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_SPAN |
         (iree_vm_module_signature_type_kind_t)IREE_VM_SCALAR_TYPE_I64,
     0},
};
static const iree_vm_module_signature_type_t sample_results[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0},
};
static const iree_vm_module_callable_type_declaration_t sample_callable = {
    .signature =
        {
            .arguments = {sample_arguments, IREE_ARRAYSIZE(sample_arguments),
                          SAMPLE_VALUE_COUNT, 0, 0},
            .results = {sample_results, IREE_ARRAYSIZE(sample_results), 1, 0,
                        0},
        },
};
static const iree_vm_module_descriptor_t sample_descriptor = {
    .name = IREE_SVL("samples"),
    .flags = IREE_VM_MODULE_FLAG_LINKABLE,
    .counts = {.function_count = 1,
               .callable_type_count = 1,
               .export_count = 1,
               .callable_fields = {.value_count = SAMPLE_VALUE_COUNT + 1}},
};

static iree_status_t sample_start(
    iree_vm_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* outcome) {
  const iree_vm_call_packet_t* call = &params->call;
  const uint64_t count =
      iree_vm_call_value_argument_load(call, SAMPLE_KEY_COUNT);
  const iree_vm_call_span_t keys =
      iree_vm_call_span_argument_load(call, SAMPLE_KEYS);
  const iree_vm_call_span_t readings =
      iree_vm_call_span_argument_load(call, SAMPLE_READINGS);
  if (count != keys.count || count != readings.count ||
      count != iree_vm_call_value_argument_load(call, SAMPLE_READING_COUNT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sample column counts must agree");
  }
  const uint8_t* key_data = keys.data;
  const uint8_t* reading_data = readings.data;
  uint64_t sum = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    sum +=
        (uint64_t)iree_unaligned_load_le_u32(key_data + i * sizeof(uint32_t)) *
        iree_unaligned_load_le_u64(reading_data + i * sizeof(uint64_t));
  }
  iree_vm_call_value_result_store(call, 0, sum);
  *outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static void sample_destroy(iree_vm_module_t* module) {}
static void sample_query_export(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = (iree_vm_module_export_declaration_t){IREE_SVL("sum"), 0, 0, 0};
}
static void sample_query_callable(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = sample_callable;
}
static void sample_query_import_group(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("sample module has no imports");
}
static void sample_query_import(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("sample module has no imports");
}
static const iree_vm_module_vtable_t sample_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = sample_destroy,
    .function_start = sample_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = sample_query_import_group,
    .query_import = sample_query_import,
    .query_export = sample_query_export,
    .query_callable_type = sample_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

static iree_status_t sample_run(iree_vm_invocation_t* invocation,
                                iree_vm_function_t function, uint32_t count,
                                uint64_t base) {
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(count),
                                   iree_vm_variant_from_i64(base)};
  iree_vm_variant_t result = iree_vm_variant_empty();
  IREE_RETURN_IF_ERROR(iree_vm_invoke(
      invocation, function, iree_vm_variant_span_from_array(arguments),
      (iree_vm_variant_span_t){&result, 1}));
  int64_t actual = 0;
  iree_status_t status = iree_vm_i64_from_variant(result, &actual);
  iree_vm_variant_reset(&result);
  IREE_RETURN_IF_ERROR(status);
  uint64_t expected = 777 + 999 + 555 + 333;
  for (uint32_t i = 0; i < count; ++i) {
    expected += 4 * (3 + i * 5) * (base + i * 7);
  }
  if (base && count) {
    expected += 3 * 3 * 11;
  }
  if ((uint64_t)actual != expected) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "native sample sum differs for count %u", count);
  }
  return iree_ok_status();
}

static iree_status_t sample_test(void) {
  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t native_module;
  iree_status_t status = iree_vm_module_initialize(
      &sample_vtable, &sample_descriptor, &native_module);
  bool native_initialized = iree_status_is_ok(status);
  iree_vm_module_t* module = NULL;
  const iree_file_toc_t* image = loom_vm_cxx_native_spans_bytecode_create();
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_create(
        environment, IREE_SV("client"),
        (iree_vm_bytecode_module_storage_t){
            iree_make_const_byte_span(image->data, image->size),
            iree_allocator_null()},
        iree_allocator_system(), &module);
  }
  iree_vm_program_t* program = NULL;
  if (iree_status_is_ok(status)) {
    iree_vm_module_t* dependencies[] = {&native_module};
    status = iree_vm_program_create(
        (iree_vm_program_modules_t){
            module, {dependencies, IREE_ARRAYSIZE(dependencies)}},
        iree_allocator_system(), &program);
  }
  iree_alignas(iree_max_align_t) uint8_t storage[64 * 1024];
  iree_vm_invocation_t* invocation = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_invocation_initialize(
        iree_make_byte_span(storage, sizeof(storage)), &invocation);
  }
  iree_vm_process_t* process = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_create(program, invocation,
                                    iree_vm_variant_span_empty(),
                                    iree_allocator_system(), &process);
  }
  iree_vm_function_t function;
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_lookup_function(process, IREE_SV("client"),
                                             IREE_SV("run"), &function);
  }
  const uint32_t counts[] = {0, 1, 2, 3, 19, 257};
  const uint64_t bases[] = {0, 11, UINT64_C(0x100000003)};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(counts) && iree_status_is_ok(status); ++i) {
    for (iree_host_size_t j = 0;
         j < IREE_ARRAYSIZE(bases) && iree_status_is_ok(status); ++j) {
      status = sample_run(invocation, function, counts[i], bases[j]);
    }
  }
  iree_vm_process_release(process);
  if (invocation) {
    iree_vm_invocation_deinitialize(invocation);
  }
  iree_vm_program_release(program);
  iree_vm_module_release(module);
  if (native_initialized) {
    iree_vm_module_release(&native_module);
  }
  iree_vm_environment_free(environment);
  return status;
}

int main(void) {
  iree_status_t status = sample_test();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
  }
  const int result = iree_status_is_ok(status) ? 0 : 1;
  iree_status_free(status);
  return result;
}
