// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/module_test_provider.h"

#include <string.h>

#include "iree/vm/reflection.h"

static const iree_vm_module_signature_type_t iree_vm_test_i32[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
};
static const iree_vm_module_signature_type_t iree_vm_test_ref[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
};
static const iree_vm_module_signature_type_t iree_vm_test_function[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION, 0},
};

static const iree_vm_module_callable_type_declaration_t
    iree_vm_test_callable_types[] = {
        {{{iree_vm_test_i32, IREE_ARRAYSIZE(iree_vm_test_i32), 1, 0, 0},
          {iree_vm_test_i32, IREE_ARRAYSIZE(iree_vm_test_i32), 1, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
        {{{iree_vm_test_ref, IREE_ARRAYSIZE(iree_vm_test_ref), 0, 1, 0},
          {iree_vm_test_ref, IREE_ARRAYSIZE(iree_vm_test_ref), 0, 1, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
        {{{iree_vm_test_function, IREE_ARRAYSIZE(iree_vm_test_function), 0, 0,
           1},
          {iree_vm_test_i32, IREE_ARRAYSIZE(iree_vm_test_i32), 1, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         1,
         0},
};

static const iree_vm_module_import_group_t iree_vm_test_import_groups[] = {
    {IREE_SVL("support"), 0, 1},
};
static const iree_vm_module_import_declaration_t iree_vm_test_imports[] = {
    {IREE_SVL("support"), IREE_SVL("apply"), 1,
     IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL, 1},
};
static const iree_vm_module_export_declaration_t iree_vm_test_exports[] = {
    {IREE_SVL("add_one"), 0, 0, 1},
    {IREE_SVL("increment"), 0, 0, 1},
};

typedef struct iree_vm_test_presentation_t {
  // Function documentation.
  iree_string_view_t documentation;
  // Authored function type.
  iree_string_view_t function_type;
  // Authored argument name.
  iree_string_view_t argument_name;
  // Authored argument type.
  iree_string_view_t argument_type;
  // Authored result name.
  iree_string_view_t result_name;
  // Authored result type.
  iree_string_view_t result_type;
} iree_vm_test_presentation_t;

static const iree_vm_test_presentation_t iree_vm_test_import_presentation = {
    IREE_SVL("Optionally transforms one buffer."),
    IREE_SVL("(vm.ref<vm, buffer>) -> vm.ref<vm, buffer>"),
    IREE_SVL("source"),
    IREE_SVL("vm.ref<vm, buffer>"),
    IREE_SVL("result"),
    IREE_SVL("vm.ref<vm, buffer>"),
};
static const iree_vm_test_presentation_t iree_vm_test_export_presentations[] = {
    {IREE_SVL("Adds one to a value."), IREE_SVL("(i32) -> i32"),
     IREE_SVL("value"), IREE_SVL("i32"), IREE_SVL("sum"), IREE_SVL("i32")},
    {IREE_SVL("Increments one value."), IREE_SVL("(i32) -> i32"),
     IREE_SVL("input"), IREE_SVL("i32"), IREE_SVL("output"), IREE_SVL("i32")},
};

static void iree_vm_test_destroy(iree_vm_module_t* base_module) {
  iree_vm_module_test_provider_t* provider =
      (iree_vm_module_test_provider_t*)base_module;
  ++*provider->destroy_count;
}

static iree_status_t iree_vm_test_function_start(
    iree_vm_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  (void)module;
  const uint64_t value = iree_vm_call_value_argument_load(&params->call, 0);
  iree_vm_call_value_result_store(&params->call, 0, value + 1);
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

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

static void iree_vm_test_query_presentation(
    const iree_vm_module_t* module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation) {
  (void)module;
  const iree_vm_test_presentation_t* source = NULL;
  switch (query->declaration.kind) {
    case IREE_VM_MODULE_DECLARATION_KIND_IMPORT:
      source = &iree_vm_test_import_presentation;
      break;
    case IREE_VM_MODULE_DECLARATION_KIND_EXPORT:
      source = &iree_vm_test_export_presentations[query->declaration.ordinal];
      break;
    default:
      break;
  }
  iree_vm_module_presentation_t presentation = {0};
  const iree_host_size_t transient_size =
      query->declaration.kind == IREE_VM_MODULE_DECLARATION_KIND_IMPORT ? 9 : 0;
  presentation.required_transient_storage_size = transient_size;
  if (source && query->fields.data && query->fields.count == 2 &&
      query->transient_storage.data_length >= transient_size) {
    query->fields.data[0].name = source->argument_name;
    query->fields.data[0].authored_type = source->argument_type;
    query->fields.data[1].name = source->result_name;
    query->fields.data[1].authored_type = source->result_type;
    if (transient_size) {
      memcpy(query->transient_storage.data, "generated", transient_size);
      presentation.documentation = iree_make_string_view(
          (const char*)query->transient_storage.data, transient_size);
    } else {
      presentation.documentation = source->documentation;
    }
    presentation.authored_type = source->function_type;
  }
  *out_presentation = presentation;
}

static iree_vm_metadata_entry_t iree_vm_test_make_metadata_entry(
    iree_string_view_t key, iree_vm_metadata_value_type_t type,
    const uint8_t* data, iree_host_size_t data_length) {
  const iree_vm_metadata_entry_t entry = {
      key,
      {type, {data, data_length}},
  };
  return entry;
}

static void iree_vm_test_metadata_by_ordinal(
    const iree_vm_module_t* module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry) {
  (void)module;
  static const uint8_t kRevision[] = {7, 0, 0, 0, 0, 0, 0, 0};
  static const uint8_t kTrue[] = {1};
  if (query->scope.kind == IREE_VM_MODULE_METADATA_SCOPE_KIND_MODULE) {
    *out_entry =
        query->ordinal == 0
            ? iree_vm_test_make_metadata_entry(IREE_SV("category"),
                                               IREE_VM_METADATA_VALUE_TYPE_UTF8,
                                               (const uint8_t*)"test", 4)
            : iree_vm_test_make_metadata_entry(
                  IREE_SV("revision"), IREE_VM_METADATA_VALUE_TYPE_U64,
                  kRevision, IREE_ARRAYSIZE(kRevision));
  } else if (query->scope.kind == IREE_VM_MODULE_METADATA_SCOPE_KIND_IMPORT) {
    *out_entry = iree_vm_test_make_metadata_entry(
        IREE_SV("optional"), IREE_VM_METADATA_VALUE_TYPE_BOOL, kTrue,
        IREE_ARRAYSIZE(kTrue));
  } else {
    *out_entry = iree_vm_test_make_metadata_entry(
        IREE_SV("alias"), IREE_VM_METADATA_VALUE_TYPE_UTF8,
        (const uint8_t*)iree_vm_test_export_presentations[query->scope.ordinal]
            .documentation.data,
        iree_vm_test_export_presentations[query->scope.ordinal]
            .documentation.size);
  }
}

static const iree_vm_module_vtable_t iree_vm_test_vtable = {
    sizeof(iree_vm_test_vtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    iree_vm_test_destroy,
    iree_vm_test_function_start,
    iree_vm_module_function_resume_unreachable,
    NULL,
    NULL,
    NULL,
    iree_vm_test_query_import_group,
    iree_vm_test_query_import,
    iree_vm_test_query_export,
    iree_vm_test_query_callable_type,
    iree_vm_test_query_presentation,
    iree_vm_test_metadata_by_ordinal,
};

iree_status_t iree_vm_module_test_provider_initialize(
    iree_vm_ref_type_t buffer_type, int* destroy_count,
    iree_vm_module_test_provider_t* out_provider) {
  memset(out_provider, 0, sizeof(*out_provider));
  out_provider->buffer_type = buffer_type;
  out_provider->destroy_count = destroy_count;
  const iree_vm_module_counts_t counts = {
      1,
      IREE_ARRAYSIZE(iree_vm_test_callable_types),
      IREE_ARRAYSIZE(iree_vm_test_import_groups),
      IREE_ARRAYSIZE(iree_vm_test_imports),
      IREE_ARRAYSIZE(iree_vm_test_exports),
      2,
      {3, 2, 1},
  };
  const iree_vm_module_descriptor_t descriptor = {
      IREE_SVL("fixture"),
      IREE_VM_MODULE_FLAG_LINKABLE,
      {&out_provider->buffer_type, 1},
      counts,
      0,
  };
  out_provider->descriptor = descriptor;
  return iree_vm_module_initialize(
      &iree_vm_test_vtable, &out_provider->descriptor, &out_provider->base);
}
