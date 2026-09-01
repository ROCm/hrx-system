// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/module_test_provider.h"

#include <string.h>

#include "iree/vm/buffer.h"

typedef struct iree_vm_test_module_t {
  // Generic module base at offset zero.
  iree_vm_module_t base;
  // Allocator owning this implementation object.
  iree_allocator_t host_allocator;
  // Optional externally owned lifecycle counters.
  iree_vm_test_module_counters_t* counters;
  // Module-local canonical ref-type handles.
  iree_vm_ref_type_t ref_types[1];
  // Stable fixed descriptor borrowing this implementation.
  iree_vm_module_descriptor_t descriptor;
} iree_vm_test_module_t;

static_assert(offsetof(iree_vm_test_module_t, base) == 0,
              "test module base must remain at offset zero");

typedef struct iree_vm_test_module_state_t {
  // Set while the process state is attached.
  uint32_t marker;
  // Set after the state has passed sealing.
  uint32_t is_sealed;
} iree_vm_test_module_state_t;

enum {
  IREE_VM_TEST_MODULE_STATE_MARKER = 0x564D544Du,
};

static const iree_vm_module_signature_type_t iree_vm_test_callback_arguments[] =
    {
        {IREE_VM_SCALAR_TYPE_I32, 0},
};
static const iree_vm_module_signature_type_t iree_vm_test_callback_results[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
};
static const iree_vm_module_signature_type_t iree_vm_test_run_arguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION, 0},
};
static const iree_vm_module_signature_type_t iree_vm_test_run_results[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION, 0},
};

static const iree_vm_module_callable_type_declaration_t
    iree_vm_test_callable_types[] = {
        {
            {
                {iree_vm_test_callback_arguments,
                 IREE_ARRAYSIZE(iree_vm_test_callback_arguments)},
                {iree_vm_test_callback_results,
                 IREE_ARRAYSIZE(iree_vm_test_callback_results)},
            },
            IREE_VM_CALLABLE_TYPE_FLAG_NONE,
            0,
            0,
        },
        {
            {
                {iree_vm_test_run_arguments,
                 IREE_ARRAYSIZE(iree_vm_test_run_arguments)},
                {iree_vm_test_run_results,
                 IREE_ARRAYSIZE(iree_vm_test_run_results)},
            },
            IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
            1,
            0,
        },
};

static const iree_vm_module_import_group_t iree_vm_test_import_groups[] = {
    {
        IREE_SVL("dependency.module"),
        0,
        1,
    },
};

static const iree_vm_module_import_declaration_t iree_vm_test_imports[] = {
    {
        IREE_SVL("dependency.module"),
        IREE_SVL("optional_callback"),
        0,
        IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL,
        1,
    },
};

static const iree_vm_module_export_declaration_t iree_vm_test_exports[] = {
    {
        IREE_SVL("alias"),
        1,
        0,
        1,
    },
    {
        IREE_SVL("run"),
        1,
        0,
        2,
    },
};

typedef struct iree_vm_test_field_presentation_t {
  // Optional source-level field name.
  iree_string_view_t name;
  // Optional source-level authored type.
  iree_string_view_t authored_type;
} iree_vm_test_field_presentation_t;

static const iree_vm_test_field_presentation_t
    iree_vm_test_import_presentation[] = {
        {IREE_SVL("value"), IREE_SVL("i32")},
        {IREE_SVL("result"), IREE_SVL("i32")},
};

static const iree_vm_test_field_presentation_t
    iree_vm_test_alias_presentation[] = {
        {IREE_SVL("value"), IREE_SVL("i32")},
        {IREE_SVL("input"), IREE_SVL("buffer")},
        {IREE_SVL("callback"), IREE_SVL("(i32) -> i32")},
        {IREE_SVL("sum"), IREE_SVL("i64")},
        {IREE_SVL("output"), IREE_SVL("buffer")},
        {IREE_SVL("callback_result"), IREE_SVL("(i32) -> i32")},
};

static const iree_vm_test_field_presentation_t iree_vm_test_run_presentation[] =
    {
        {IREE_SVL("value"), IREE_SVL("i32")},
        {IREE_SVL("input"), IREE_SVL("buffer")},
        {IREE_SVL("callback"), IREE_SVL("(i32) -> i32")},
        {IREE_SVL("sum"), IREE_SVL("i64")},
        {IREE_SVL("output"), IREE_SVL("buffer")},
        {IREE_SVL("callback_result"), IREE_SVL("(i32) -> i32")},
};

static const uint8_t iree_vm_test_metadata_true[] = {1};
static const uint8_t iree_vm_test_metadata_revision[] = {
    0x2A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};
static const uint8_t iree_vm_test_metadata_weight[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x40,
};

static const iree_vm_metadata_entry_t iree_vm_test_module_metadata[] = {
    {
        IREE_SVL("build"),
        {IREE_VM_METADATA_VALUE_TYPE_UTF8, {(const uint8_t*)"native-c", 8}},
    },
    {
        IREE_SVL("revision"),
        {IREE_VM_METADATA_VALUE_TYPE_U64,
         {iree_vm_test_metadata_revision,
          IREE_ARRAYSIZE(iree_vm_test_metadata_revision)}},
    },
};

static const iree_vm_metadata_entry_t iree_vm_test_import_metadata[] = {
    {
        IREE_SVL("optional"),
        {IREE_VM_METADATA_VALUE_TYPE_BOOL,
         {iree_vm_test_metadata_true,
          IREE_ARRAYSIZE(iree_vm_test_metadata_true)}},
    },
};

static const iree_vm_metadata_entry_t iree_vm_test_alias_metadata[] = {
    {
        IREE_SVL("route"),
        {IREE_VM_METADATA_VALUE_TYPE_UTF8, {(const uint8_t*)"alias", 5}},
    },
};

static const iree_vm_metadata_entry_t iree_vm_test_run_metadata[] = {
    {
        IREE_SVL("route"),
        {IREE_VM_METADATA_VALUE_TYPE_UTF8, {(const uint8_t*)"primary", 7}},
    },
    {
        IREE_SVL("weight"),
        {IREE_VM_METADATA_VALUE_TYPE_F64,
         {iree_vm_test_metadata_weight,
          IREE_ARRAYSIZE(iree_vm_test_metadata_weight)}},
    },
};

static void iree_vm_test_module_destroy(iree_vm_module_t* base_module) {
  iree_vm_test_module_t* module =
      iree_containerof(base_module, iree_vm_test_module_t, base);
  if (module->counters) ++module->counters->destroy_count;
  iree_allocator_free(module->host_allocator, module);
}

static iree_status_t iree_vm_test_module_function_start(
    iree_vm_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  (void)module;
  const uint64_t value_bits =
      iree_vm_call_value_argument_load(&params->call, 0);
  const int64_t result_value = (int64_t)(int32_t)value_bits + 7;
  iree_vm_call_value_result_store(&params->call, 0, (uint64_t)result_value);

  iree_vm_ref_t ref = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(&params->call, 0, &ref);
  iree_vm_call_ref_result_store_move(&params->call, 0, &ref);

  const iree_vm_function_ref_t function_ref =
      iree_vm_call_function_argument_load(&params->call, 0);
  iree_vm_call_function_result_store(&params->call, 0, function_ref);

  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static iree_status_t iree_vm_test_module_attach_state(
    iree_vm_module_t* base_module, iree_byte_span_t zeroed_storage,
    iree_allocator_t host_allocator) {
  (void)host_allocator;
  iree_vm_test_module_t* module =
      iree_containerof(base_module, iree_vm_test_module_t, base);
  if (zeroed_storage.data_length != sizeof(iree_vm_test_module_state_t) ||
      !zeroed_storage.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "test module state storage is invalid");
  }
  iree_vm_test_module_state_t* state =
      (iree_vm_test_module_state_t*)zeroed_storage.data;
  if (state->marker != 0 || state->is_sealed != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test module state storage is not zeroed");
  }
  state->marker = IREE_VM_TEST_MODULE_STATE_MARKER;
  if (module->counters) ++module->counters->attach_count;
  return iree_ok_status();
}

static iree_status_t iree_vm_test_module_seal_state(
    iree_vm_module_t* base_module, iree_byte_span_t storage) {
  iree_vm_test_module_t* module =
      iree_containerof(base_module, iree_vm_test_module_t, base);
  iree_vm_test_module_state_t* state =
      storage.data_length == sizeof(iree_vm_test_module_state_t)
          ? (iree_vm_test_module_state_t*)storage.data
          : NULL;
  if (!state || state->marker != IREE_VM_TEST_MODULE_STATE_MARKER ||
      state->is_sealed != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test module state is not attach-complete");
  }
  state->is_sealed = 1;
  if (module->counters) ++module->counters->seal_count;
  return iree_ok_status();
}

static void iree_vm_test_module_detach_state(iree_vm_module_t* base_module,
                                             iree_byte_span_t storage) {
  iree_vm_test_module_t* module =
      iree_containerof(base_module, iree_vm_test_module_t, base);
  if (storage.data_length == sizeof(iree_vm_test_module_state_t) &&
      storage.data) {
    memset(storage.data, 0, storage.data_length);
  }
  if (module->counters) ++module->counters->detach_count;
}

static void iree_vm_test_module_query_import_group(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group) {
  (void)module;
  *out_group = iree_vm_test_import_groups[ordinal];
}

static void iree_vm_test_module_query_import(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import) {
  (void)module;
  *out_import = iree_vm_test_imports[ordinal];
}

static void iree_vm_test_module_query_export(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export) {
  (void)module;
  *out_export = iree_vm_test_exports[ordinal];
}

static void iree_vm_test_module_query_callable_type(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type) {
  (void)module;
  *out_callable_type = iree_vm_test_callable_types[ordinal];
}

static void iree_vm_test_module_query_presentation(
    const iree_vm_module_t* module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation) {
  (void)module;
  const iree_vm_test_field_presentation_t* fields = NULL;
  iree_host_size_t field_count = 0;
  iree_host_size_t transient_storage_size = 0;
  iree_string_view_t documentation = iree_string_view_empty();
  iree_string_view_t authored_type = iree_string_view_empty();
  if (query->declaration.kind == IREE_VM_MODULE_DECLARATION_KIND_IMPORT) {
    fields = iree_vm_test_import_presentation;
    field_count = IREE_ARRAYSIZE(iree_vm_test_import_presentation);
    documentation = IREE_SV("Optional dependency callback.");
    authored_type = IREE_SV("(i32) -> i32");
  } else if (query->declaration.ordinal == 0) {
    fields = iree_vm_test_alias_presentation;
    field_count = IREE_ARRAYSIZE(iree_vm_test_alias_presentation);
    documentation = IREE_SV("Alias of the native test entry point.");
    authored_type =
        IREE_SV("(i32, buffer, (i32) -> i32) -> (i64, buffer, (i32) -> i32)");
  } else {
    fields = iree_vm_test_run_presentation;
    field_count = IREE_ARRAYSIZE(iree_vm_test_run_presentation);
    transient_storage_size = 8;
    authored_type =
        IREE_SV("(i32, buffer, (i32) -> i32) -> (i64, buffer, (i32) -> i32)");
  }

  iree_vm_module_presentation_t presentation = {
      transient_storage_size,
      iree_string_view_empty(),
      iree_string_view_empty(),
  };
  if (query->fields.count == field_count &&
      query->transient_storage.data_length >= transient_storage_size) {
    for (iree_host_size_t i = 0; i < field_count; ++i) {
      query->fields.data[i].name = fields[i].name;
      query->fields.data[i].authored_type = fields[i].authored_type;
    }
    if (transient_storage_size != 0) {
      memcpy(query->transient_storage.data, "dynamic", 7);
      documentation =
          iree_make_string_view((const char*)query->transient_storage.data, 7);
    }
    presentation.documentation = documentation;
    presentation.authored_type = authored_type;
  }
  *out_presentation = presentation;
}

static void iree_vm_test_module_metadata_by_ordinal(
    const iree_vm_module_t* module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry) {
  (void)module;
  switch (query->scope.kind) {
    case IREE_VM_MODULE_METADATA_SCOPE_KIND_MODULE:
      *out_entry = iree_vm_test_module_metadata[query->ordinal];
      break;
    case IREE_VM_MODULE_METADATA_SCOPE_KIND_IMPORT:
      *out_entry = iree_vm_test_import_metadata[query->ordinal];
      break;
    case IREE_VM_MODULE_METADATA_SCOPE_KIND_EXPORT:
      *out_entry = query->scope.ordinal == 0
                       ? iree_vm_test_alias_metadata[query->ordinal]
                       : iree_vm_test_run_metadata[query->ordinal];
      break;
    default:
      memset(out_entry, 0, sizeof(*out_entry));
      break;
  }
}

static const iree_vm_module_vtable_t iree_vm_test_module_vtable = {
    sizeof(iree_vm_test_module_vtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    iree_vm_test_module_destroy,
    iree_vm_test_module_function_start,
    iree_vm_module_function_resume_unreachable,
    iree_vm_test_module_attach_state,
    iree_vm_test_module_seal_state,
    iree_vm_test_module_detach_state,
    iree_vm_test_module_query_import_group,
    iree_vm_test_module_query_import,
    iree_vm_test_module_query_export,
    iree_vm_test_module_query_callable_type,
    iree_vm_test_module_query_presentation,
    iree_vm_test_module_metadata_by_ordinal,
};

iree_status_t iree_vm_test_module_create(
    iree_vm_environment_t* environment,
    iree_vm_test_module_counters_t* counters, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  if (!environment || !out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "environment and out_module are required");
  }
  *out_module = NULL;

  const iree_vm_ref_type_table_t* vm_type_table =
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm"));
  iree_vm_ref_types_t vm_types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(vm_type_table, &vm_types));

  iree_vm_test_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  module->host_allocator = host_allocator;
  module->counters = counters;
  module->ref_types[0] = vm_types.buffer;
  const iree_vm_module_descriptor_t descriptor = {
      IREE_SV("test.module"),
      IREE_VM_MODULE_FLAG_LINKABLE,
      {module->ref_types, IREE_ARRAYSIZE(module->ref_types)},
      {
          1,
          IREE_ARRAYSIZE(iree_vm_test_callable_types),
          IREE_ARRAYSIZE(iree_vm_test_import_groups),
          IREE_ARRAYSIZE(iree_vm_test_imports),
          IREE_ARRAYSIZE(iree_vm_test_exports),
          IREE_ARRAYSIZE(iree_vm_test_module_metadata),
      },
      sizeof(iree_vm_test_module_state_t),
  };
  module->descriptor = descriptor;

  iree_status_t status = iree_vm_module_initialize(
      &iree_vm_test_module_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
