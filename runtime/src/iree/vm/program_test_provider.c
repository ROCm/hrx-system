// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/program_test_provider.h"

#include <string.h>

typedef struct iree_vm_program_test_module_t {
  // Generic module base at offset zero.
  iree_vm_module_t base;
  // Allocator owning this implementation object.
  iree_allocator_t host_allocator;
  // Borrowed immutable semantic definition.
  const iree_vm_program_test_module_definition_t* definition;
  // Optional externally owned final-release counter.
  int* destruction_count;
} iree_vm_program_test_module_t;

static_assert(offsetof(iree_vm_program_test_module_t, base) == 0,
              "test module base must remain at offset zero");

static void iree_vm_program_test_ref_destroy(void* object) { (void)object; }

static const iree_vm_ref_type_table_t iree_vm_program_test_ref_type_table;

static const iree_vm_ref_type_descriptor_t
    iree_vm_program_test_auxiliary_descriptor = {
        iree_vm_program_test_ref_destroy,
        &iree_vm_program_test_ref_type_table,
        IREE_SVL("auxiliary"),
};

static const iree_vm_ref_type_descriptor_t
    iree_vm_program_test_shared_descriptor = {
        iree_vm_program_test_ref_destroy,
        &iree_vm_program_test_ref_type_table,
        IREE_SVL("shared"),
};

static const struct {
  // Auxiliary type at ordinal zero.
  iree_vm_ref_type_t auxiliary;
  // Shared type at ordinal one.
  iree_vm_ref_type_t shared;
} iree_vm_program_test_ref_types = {
    &iree_vm_program_test_auxiliary_descriptor,
    &iree_vm_program_test_shared_descriptor,
};

static const iree_vm_ref_type_table_t iree_vm_program_test_ref_type_table = {
    sizeof(iree_vm_program_test_ref_type_table),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SVL("program.test"),
    {&iree_vm_program_test_ref_types, 2},
};

iree_vm_ref_type_t iree_vm_program_test_auxiliary_ref_type(void) {
  return &iree_vm_program_test_auxiliary_descriptor;
}

iree_vm_ref_type_t iree_vm_program_test_shared_ref_type(void) {
  return &iree_vm_program_test_shared_descriptor;
}

static iree_vm_program_test_module_t* iree_vm_program_test_module_cast(
    iree_vm_module_t* base_module) {
  return iree_containerof(base_module, iree_vm_program_test_module_t, base);
}

static const iree_vm_program_test_module_t*
iree_vm_program_test_module_const_cast(const iree_vm_module_t* base_module) {
  return iree_containerof(base_module, iree_vm_program_test_module_t, base);
}

static void iree_vm_program_test_module_destroy(iree_vm_module_t* base_module) {
  iree_vm_program_test_module_t* module =
      iree_vm_program_test_module_cast(base_module);
  if (module->destruction_count) ++*module->destruction_count;
  iree_allocator_free(module->host_allocator, module);
}

static iree_status_t iree_vm_program_test_module_function_start(
    iree_vm_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  (void)module;
  (void)params;
  (void)out_outcome;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "program-linking test module is not executable");
}

static void iree_vm_program_test_module_query_import_group(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group) {
  const iree_vm_program_test_module_t* module =
      iree_vm_program_test_module_const_cast(base_module);
  *out_group = module->definition->import_groups[ordinal];
}

static void iree_vm_program_test_module_query_import(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import) {
  const iree_vm_program_test_module_t* module =
      iree_vm_program_test_module_const_cast(base_module);
  *out_import = module->definition->imports[ordinal];
}

static void iree_vm_program_test_module_query_export(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export) {
  const iree_vm_program_test_module_t* module =
      iree_vm_program_test_module_const_cast(base_module);
  *out_export = module->definition->exports[ordinal];
}

static void iree_vm_program_test_module_query_callable_type(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type) {
  const iree_vm_program_test_module_t* module =
      iree_vm_program_test_module_const_cast(base_module);
  *out_callable_type = module->definition->callable_types[ordinal];
}

static const iree_vm_module_vtable_t iree_vm_program_test_module_vtable = {
    sizeof(iree_vm_program_test_module_vtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    iree_vm_program_test_module_destroy,
    iree_vm_program_test_module_function_start,
    iree_vm_module_function_resume_unreachable,
    NULL,
    NULL,
    NULL,
    iree_vm_program_test_module_query_import_group,
    iree_vm_program_test_module_query_import,
    iree_vm_program_test_module_query_export,
    iree_vm_program_test_module_query_callable_type,
    iree_vm_module_query_presentation_none,
    iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t iree_vm_program_test_module_create(
    const iree_vm_program_test_module_definition_t* definition,
    int* destruction_count, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  if (!definition || !out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "definition and out_module are required");
  }
  *out_module = NULL;

  iree_vm_program_test_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  module->host_allocator = host_allocator;
  module->definition = definition;
  module->destruction_count = destruction_count;

  iree_status_t status =
      iree_vm_module_initialize(&iree_vm_program_test_module_vtable,
                                &definition->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
