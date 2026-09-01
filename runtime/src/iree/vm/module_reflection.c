// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/vm/module.h"

//===----------------------------------------------------------------------===//
// Description storage
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_module_validate_description_storage(
    iree_byte_span_t storage) {
  if (storage.data_length != 0 && !storage.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "description storage bytes are required");
  }
  if (storage.data_length != 0 &&
      !iree_host_ptr_has_alignment(storage.data, iree_max_align_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "description storage must have at least %" PRIhsz
                            "-byte alignment",
                            (iree_host_size_t)iree_max_align_t);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_import_identity(
    iree_vm_import_t import_value) {
  if (!import_value.module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import module is required");
  }
  if (import_value.ordinal >=
      import_value.module->descriptor->counts.import_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "import ordinal %" PRIhsz " is out of range",
                            import_value.ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_export_identity(
    iree_vm_export_t export_value) {
  if (!export_value.module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "export module is required");
  }
  if (export_value.ordinal >=
      export_value.module->descriptor->counts.export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIhsz " is out of range",
                            export_value.ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_callable_type_identity(
    iree_vm_callable_type_t callable_type) {
  if (!callable_type.module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "callable-type module is required");
  }
  if (callable_type.ordinal >=
      callable_type.module->descriptor->counts.callable_type_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "callable-type ordinal %" PRIhsz " is out of range",
                            callable_type.ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_calculate_field_storage_layout(
    iree_host_size_t field_count, iree_host_size_t* out_fields_offset,
    iree_host_size_t* out_transient_offset) {
  iree_host_size_t total_size = 0;
  return IREE_STRUCT_LAYOUT(
      0, &total_size,
      IREE_STRUCT_FIELD_ALIGNED(field_count, iree_vm_signature_field_t,
                                iree_alignof(iree_vm_signature_field_t),
                                out_fields_offset),
      IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, iree_max_align_t,
                                out_transient_offset));
}

static iree_status_t iree_vm_module_calculate_type_storage_layout(
    iree_host_size_t argument_count, iree_host_size_t result_count,
    iree_host_size_t* out_arguments_offset,
    iree_host_size_t* out_results_offset, iree_host_size_t* out_total_size) {
  return IREE_STRUCT_LAYOUT(
      0, out_total_size,
      IREE_STRUCT_FIELD_ALIGNED(argument_count, iree_vm_signature_type_t,
                                iree_alignof(iree_vm_signature_type_t),
                                out_arguments_offset),
      IREE_STRUCT_FIELD_ALIGNED(result_count, iree_vm_signature_type_t,
                                iree_alignof(iree_vm_signature_type_t),
                                out_results_offset));
}

static iree_vm_signature_type_t iree_vm_module_resolve_signature_type(
    const iree_vm_module_t* module, iree_vm_module_signature_type_t type) {
  iree_vm_signature_type_t resolved_type = {0};
  if (type.kind > IREE_VM_SCALAR_TYPE_NONE &&
      type.kind <= IREE_VM_SCALAR_TYPE_F64) {
    resolved_type.kind = IREE_VM_SIGNATURE_TYPE_KIND_SCALAR;
    resolved_type.value.scalar = (iree_vm_scalar_type_t)type.kind;
  } else if (type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
    resolved_type.kind = IREE_VM_SIGNATURE_TYPE_KIND_REF;
    resolved_type.value.ref =
        module->descriptor->ref_types.data[type.type_ordinal];
  } else {
    resolved_type.kind = IREE_VM_SIGNATURE_TYPE_KIND_FUNCTION;
    resolved_type.value.callable.module = module;
    resolved_type.value.callable.ordinal = type.type_ordinal;
  }
  return resolved_type;
}

static void iree_vm_module_resolve_signature_fields(
    const iree_vm_module_t* module,
    iree_vm_module_signature_type_span_t source_types,
    iree_vm_signature_field_t* target_fields) {
  for (iree_host_size_t i = 0; i < source_types.count; ++i) {
    target_fields[i].type =
        iree_vm_module_resolve_signature_type(module, source_types.data[i]);
  }
}

static void iree_vm_module_resolve_signature_types(
    const iree_vm_module_t* module,
    iree_vm_module_signature_type_span_t source_types,
    iree_vm_signature_type_t* target_types) {
  for (iree_host_size_t i = 0; i < source_types.count; ++i) {
    target_types[i] =
        iree_vm_module_resolve_signature_type(module, source_types.data[i]);
  }
}

static iree_status_t iree_vm_module_validate_presentation_text(
    iree_string_view_t value, const char* label) {
  if (value.size != 0 && !value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s bytes are required", label);
  }
  if (iree_string_view_find_char(value, '\0', 0) != IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s contains an embedded NUL byte", label);
  }
  if (!iree_unicode_utf8_validate(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s is not valid UTF-8", label);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_presentation(
    iree_vm_signature_field_t* fields, iree_host_size_t field_count,
    const iree_vm_module_presentation_t* presentation) {
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_presentation_text(
      presentation->documentation, "presentation documentation"));
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_presentation_text(
      presentation->authored_type, "presentation authored type"));
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_module_validate_presentation_text(
        fields[i].name, "presentation field name"));
    IREE_RETURN_IF_ERROR(iree_vm_module_validate_presentation_text(
        fields[i].authored_type, "presentation field authored type"));
  }
  return iree_ok_status();
}

typedef struct iree_vm_module_description_query_t {
  // Immutable module owning the queried declaration.
  const iree_vm_module_t* module;
  // Public import or export declaration identity.
  iree_vm_module_declaration_t declaration;
  // Exact machine signature shared with its callable type.
  iree_vm_module_signature_t signature;
} iree_vm_module_description_query_t;

typedef struct iree_vm_module_description_result_t {
  // Resolved argument fields in caller-owned storage.
  iree_vm_signature_field_span_t arguments;
  // Resolved result fields in caller-owned storage.
  iree_vm_signature_field_span_t results;
  // Complete provider-authored presentation.
  iree_vm_module_presentation_t presentation;
} iree_vm_module_description_result_t;

static iree_status_t iree_vm_module_query_declaration_description(
    const iree_vm_module_description_query_t* description_query,
    iree_byte_span_t storage, bool materialize,
    iree_host_size_t* out_required_storage_size,
    iree_vm_module_description_result_t* out_result) {
  iree_host_size_t field_count = 0;
  if (!iree_host_size_checked_add(description_query->signature.arguments.count,
                                  description_query->signature.results.count,
                                  &field_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "description field count overflows host size");
  }
  iree_host_size_t fields_offset = 0;
  iree_host_size_t transient_offset = 0;
  IREE_RETURN_IF_ERROR(iree_vm_module_calculate_field_storage_layout(
      field_count, &fields_offset, &transient_offset));

  iree_vm_module_presentation_query_t presentation_query = {
      description_query->declaration,
      {NULL, 0},
      iree_byte_span_empty(),
  };
  if (materialize && storage.data_length >= transient_offset) {
    if (field_count != 0) {
      presentation_query.fields.data =
          (iree_vm_signature_field_t*)(storage.data + fields_offset);
      presentation_query.fields.count = field_count;
    }
    if (storage.data_length != transient_offset) {
      presentation_query.transient_storage =
          iree_make_byte_span(storage.data + transient_offset,
                              storage.data_length - transient_offset);
    }
  }

  iree_vm_module_presentation_t presentation = {0};
  description_query->module->vtable->query_presentation(
      description_query->module, &presentation_query, &presentation);

  iree_host_size_t required_storage_size = 0;
  if (!iree_host_size_checked_add(transient_offset,
                                  presentation.required_transient_storage_size,
                                  &required_storage_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "description storage size overflows host size");
  }

  if (materialize && storage.data_length >= required_storage_size) {
    iree_vm_signature_field_t* fields = presentation_query.fields.data;
    IREE_RETURN_IF_ERROR(iree_vm_module_validate_presentation(
        fields, field_count, &presentation));
    iree_vm_signature_field_t* argument_fields =
        description_query->signature.arguments.count == 0 ? NULL : fields;
    iree_vm_module_resolve_signature_fields(
        description_query->module, description_query->signature.arguments,
        argument_fields);
    iree_vm_signature_field_t* result_fields =
        description_query->signature.results.count == 0
            ? NULL
            : fields + description_query->signature.arguments.count;
    iree_vm_module_resolve_signature_fields(
        description_query->module, description_query->signature.results,
        result_fields);

    const iree_vm_module_description_result_t result = {
        {argument_fields, description_query->signature.arguments.count},
        {result_fields, description_query->signature.results.count},
        presentation,
    };
    *out_result = result;
  }
  *out_required_storage_size = required_storage_size;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_import_query_description(
    iree_vm_import_t import_value, iree_byte_span_t storage,
    iree_host_size_t* out_required_storage_size,
    iree_vm_import_description_t* out_description) {
  if (!out_required_storage_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_required_storage_size is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_import_identity(import_value));
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_description_storage(storage));

  iree_vm_module_import_declaration_t import_declaration = {0};
  import_value.module->vtable->query_import(
      import_value.module, import_value.ordinal, &import_declaration);
  iree_vm_module_callable_type_declaration_t callable_type = {0};
  import_value.module->vtable->query_callable_type(
      import_value.module, import_declaration.callable_type_ordinal,
      &callable_type);

  const iree_vm_module_description_query_t description_query = {
      import_value.module,
      {IREE_VM_MODULE_DECLARATION_KIND_IMPORT, import_value.ordinal},
      callable_type.signature,
  };
  iree_host_size_t required_storage_size = 0;
  iree_vm_module_description_result_t result = {0};
  IREE_RETURN_IF_ERROR(iree_vm_module_query_declaration_description(
      &description_query, storage, out_description != NULL,
      &required_storage_size, &result));

  if (out_description && storage.data_length >= required_storage_size) {
    const iree_vm_import_description_t description = {
        {import_declaration.target_module_name,
         import_declaration.target_export_name},
        import_declaration.flags,
        callable_type.flags,
        result.arguments,
        result.results,
        result.presentation.documentation,
        result.presentation.authored_type,
    };
    *out_description = description;
  }
  *out_required_storage_size = required_storage_size;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_export_query_description(
    iree_vm_export_t export_value, iree_byte_span_t storage,
    iree_host_size_t* out_required_storage_size,
    iree_vm_export_description_t* out_description) {
  if (!out_required_storage_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_required_storage_size is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_export_identity(export_value));
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_description_storage(storage));

  iree_vm_module_export_declaration_t export_declaration = {0};
  export_value.module->vtable->query_export(
      export_value.module, export_value.ordinal, &export_declaration);
  iree_vm_module_callable_type_declaration_t callable_type = {0};
  export_value.module->vtable->query_callable_type(
      export_value.module, export_declaration.callable_type_ordinal,
      &callable_type);

  const iree_vm_module_description_query_t description_query = {
      export_value.module,
      {IREE_VM_MODULE_DECLARATION_KIND_EXPORT, export_value.ordinal},
      callable_type.signature,
  };
  iree_host_size_t required_storage_size = 0;
  iree_vm_module_description_result_t result = {0};
  IREE_RETURN_IF_ERROR(iree_vm_module_query_declaration_description(
      &description_query, storage, out_description != NULL,
      &required_storage_size, &result));

  if (out_description && storage.data_length >= required_storage_size) {
    const iree_vm_export_description_t description = {
        export_declaration.export_name,
        callable_type.flags,
        result.arguments,
        result.results,
        result.presentation.documentation,
        result.presentation.authored_type,
    };
    *out_description = description;
  }
  *out_required_storage_size = required_storage_size;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_callable_type_query_description(
    iree_vm_callable_type_t callable_type, iree_byte_span_t storage,
    iree_host_size_t* out_required_storage_size,
    iree_vm_callable_type_description_t* out_description) {
  if (!out_required_storage_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_required_storage_size is required");
  }
  IREE_RETURN_IF_ERROR(
      iree_vm_module_validate_callable_type_identity(callable_type));
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_description_storage(storage));

  iree_vm_module_callable_type_declaration_t declaration = {0};
  callable_type.module->vtable->query_callable_type(
      callable_type.module, callable_type.ordinal, &declaration);
  iree_host_size_t arguments_offset = 0;
  iree_host_size_t results_offset = 0;
  iree_host_size_t required_storage_size = 0;
  IREE_RETURN_IF_ERROR(iree_vm_module_calculate_type_storage_layout(
      declaration.signature.arguments.count,
      declaration.signature.results.count, &arguments_offset, &results_offset,
      &required_storage_size));

  if (out_description && storage.data_length >= required_storage_size) {
    iree_vm_signature_type_t* arguments =
        declaration.signature.arguments.count == 0
            ? NULL
            : (iree_vm_signature_type_t*)(storage.data + arguments_offset);
    iree_vm_signature_type_t* results =
        declaration.signature.results.count == 0
            ? NULL
            : (iree_vm_signature_type_t*)(storage.data + results_offset);
    iree_vm_module_resolve_signature_types(
        callable_type.module, declaration.signature.arguments, arguments);
    iree_vm_module_resolve_signature_types(
        callable_type.module, declaration.signature.results, results);
    const iree_vm_callable_type_description_t description = {
        declaration.flags,
        {arguments, declaration.signature.arguments.count},
        {results, declaration.signature.results.count},
    };
    *out_description = description;
  }
  *out_required_storage_size = required_storage_size;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_module_query_presentation_none(
    const iree_vm_module_t* module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation) {
  (void)module;
  for (iree_host_size_t i = 0; i < query->fields.count; ++i) {
    query->fields.data[i].name = iree_string_view_empty();
    query->fields.data[i].authored_type = iree_string_view_empty();
  }
  const iree_vm_module_presentation_t presentation = {
      0,
      iree_string_view_empty(),
      iree_string_view_empty(),
  };
  *out_presentation = presentation;
}

//===----------------------------------------------------------------------===//
// Metadata
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_module_metadata_scope_count(
    const iree_vm_module_t* module, iree_vm_module_metadata_scope_t scope,
    iree_host_size_t* out_count) {
  if (!module || !out_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and out_count are required");
  }
  switch (scope.kind) {
    case IREE_VM_MODULE_METADATA_SCOPE_KIND_MODULE:
      if (scope.ordinal != 0) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "module metadata scope ordinal must be zero");
      }
      *out_count = module->descriptor->counts.metadata_count;
      return iree_ok_status();
    case IREE_VM_MODULE_METADATA_SCOPE_KIND_IMPORT: {
      if (scope.ordinal >= module->descriptor->counts.import_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "import metadata scope is out of range");
      }
      iree_vm_module_import_declaration_t import_declaration = {0};
      module->vtable->query_import(module, scope.ordinal, &import_declaration);
      *out_count = import_declaration.metadata_count;
      return iree_ok_status();
    }
    case IREE_VM_MODULE_METADATA_SCOPE_KIND_EXPORT: {
      if (scope.ordinal >= module->descriptor->counts.export_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "export metadata scope is out of range");
      }
      iree_vm_module_export_declaration_t export_declaration = {0};
      module->vtable->query_export(module, scope.ordinal, &export_declaration);
      *out_count = export_declaration.metadata_count;
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "metadata scope kind is invalid");
  }
}

static iree_status_t iree_vm_module_metadata_by_scope_and_ordinal(
    const iree_vm_module_t* module, iree_vm_module_metadata_scope_t scope,
    iree_host_size_t ordinal, iree_vm_metadata_entry_t* out_entry) {
  if (!out_entry) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_entry is required");
  }
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(
      iree_vm_module_metadata_scope_count(module, scope, &count));
  if (ordinal >= count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "metadata ordinal %" PRIhsz " is out of range",
                            ordinal);
  }
  iree_vm_metadata_entry_t entry = {0};
  const iree_vm_module_metadata_query_t query = {scope, ordinal};
  module->vtable->metadata_by_ordinal(module, &query, &entry);
  *out_entry = entry;
  return iree_ok_status();
}

static iree_status_t iree_vm_module_try_lookup_metadata_by_scope(
    const iree_vm_module_t* module, iree_vm_module_metadata_scope_t scope,
    iree_string_view_t key, bool* out_found,
    iree_vm_metadata_value_t* out_value) {
  if (!out_found || !out_value || (key.size != 0 && !key.data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "coherent key, out_found, and out_value are required");
  }
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(
      iree_vm_module_metadata_scope_count(module, scope, &count));
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_vm_metadata_entry_t entry = {0};
    const iree_vm_module_metadata_query_t query = {scope, i};
    module->vtable->metadata_by_ordinal(module, &query, &entry);
    if (iree_string_view_equal(entry.key, key)) {
      *out_value = entry.value;
      *out_found = true;
      return iree_ok_status();
    }
  }
  *out_found = false;
  return iree_ok_status();
}

IREE_API_EXPORT iree_host_size_t
iree_vm_module_metadata_count(const iree_vm_module_t* module) {
  return module->descriptor->counts.metadata_count;
}

IREE_API_EXPORT iree_status_t iree_vm_module_metadata_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_metadata_entry_t* out_entry) {
  const iree_vm_module_metadata_scope_t scope = {
      IREE_VM_MODULE_METADATA_SCOPE_KIND_MODULE,
      0,
  };
  return iree_vm_module_metadata_by_scope_and_ordinal(module, scope, ordinal,
                                                      out_entry);
}

IREE_API_EXPORT iree_status_t iree_vm_module_try_lookup_metadata(
    const iree_vm_module_t* module, iree_string_view_t key, bool* out_found,
    iree_vm_metadata_value_t* out_value) {
  const iree_vm_module_metadata_scope_t scope = {
      IREE_VM_MODULE_METADATA_SCOPE_KIND_MODULE,
      0,
  };
  return iree_vm_module_try_lookup_metadata_by_scope(module, scope, key,
                                                     out_found, out_value);
}

IREE_API_EXPORT iree_host_size_t
iree_vm_import_metadata_count(iree_vm_import_t import_value) {
  iree_vm_module_import_declaration_t import_declaration = {0};
  import_value.module->vtable->query_import(
      import_value.module, import_value.ordinal, &import_declaration);
  return import_declaration.metadata_count;
}

IREE_API_EXPORT iree_status_t iree_vm_import_metadata_by_ordinal(
    iree_vm_import_t import_value, iree_host_size_t ordinal,
    iree_vm_metadata_entry_t* out_entry) {
  if (!import_value.module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import module is required");
  }
  const iree_vm_module_metadata_scope_t scope = {
      IREE_VM_MODULE_METADATA_SCOPE_KIND_IMPORT,
      import_value.ordinal,
  };
  return iree_vm_module_metadata_by_scope_and_ordinal(
      import_value.module, scope, ordinal, out_entry);
}

IREE_API_EXPORT iree_status_t iree_vm_import_try_lookup_metadata(
    iree_vm_import_t import_value, iree_string_view_t key, bool* out_found,
    iree_vm_metadata_value_t* out_value) {
  if (!import_value.module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import module is required");
  }
  const iree_vm_module_metadata_scope_t scope = {
      IREE_VM_MODULE_METADATA_SCOPE_KIND_IMPORT,
      import_value.ordinal,
  };
  return iree_vm_module_try_lookup_metadata_by_scope(import_value.module, scope,
                                                     key, out_found, out_value);
}

IREE_API_EXPORT iree_host_size_t
iree_vm_export_metadata_count(iree_vm_export_t export_value) {
  iree_vm_module_export_declaration_t export_declaration = {0};
  export_value.module->vtable->query_export(
      export_value.module, export_value.ordinal, &export_declaration);
  return export_declaration.metadata_count;
}

IREE_API_EXPORT iree_status_t iree_vm_export_metadata_by_ordinal(
    iree_vm_export_t export_value, iree_host_size_t ordinal,
    iree_vm_metadata_entry_t* out_entry) {
  if (!export_value.module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "export module is required");
  }
  const iree_vm_module_metadata_scope_t scope = {
      IREE_VM_MODULE_METADATA_SCOPE_KIND_EXPORT,
      export_value.ordinal,
  };
  return iree_vm_module_metadata_by_scope_and_ordinal(
      export_value.module, scope, ordinal, out_entry);
}

IREE_API_EXPORT iree_status_t iree_vm_export_try_lookup_metadata(
    iree_vm_export_t export_value, iree_string_view_t key, bool* out_found,
    iree_vm_metadata_value_t* out_value) {
  if (!export_value.module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "export module is required");
  }
  const iree_vm_module_metadata_scope_t scope = {
      IREE_VM_MODULE_METADATA_SCOPE_KIND_EXPORT,
      export_value.ordinal,
  };
  return iree_vm_module_try_lookup_metadata_by_scope(export_value.module, scope,
                                                     key, out_found, out_value);
}

static iree_status_t iree_vm_validate_metadata_value_payload(
    iree_vm_metadata_value_t value, iree_vm_metadata_value_type_t expected_type,
    iree_host_size_t expected_length) {
  if (value.type != expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metadata value has type %" PRIu32
                            " but type %" PRIu32 " is required",
                            value.type, expected_type);
  }
  if (value.data.data_length != expected_length ||
      (expected_length != 0 && !value.data.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metadata value has %" PRIhsz " bytes but %" PRIhsz
                            " are required",
                            value.data.data_length, expected_length);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_bool_from_metadata_value(
    iree_vm_metadata_value_t value, bool* out_value) {
  if (!out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_value is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_validate_metadata_value_payload(
      value, IREE_VM_METADATA_VALUE_TYPE_BOOL, 1));
  if (value.data.data[0] > 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BOOL metadata must be a canonical 0/1 byte");
  }
  *out_value = value.data.data[0] != 0;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_i64_from_metadata_value(
    iree_vm_metadata_value_t value, int64_t* out_value) {
  if (!out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_value is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_validate_metadata_value_payload(
      value, IREE_VM_METADATA_VALUE_TYPE_I64, 8));
  const uint64_t bits = iree_unaligned_load_le_u64(value.data.data);
  int64_t result = 0;
  memcpy(&result, &bits, sizeof(result));
  *out_value = result;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_u64_from_metadata_value(
    iree_vm_metadata_value_t value, uint64_t* out_value) {
  if (!out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_value is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_validate_metadata_value_payload(
      value, IREE_VM_METADATA_VALUE_TYPE_U64, 8));
  *out_value = iree_unaligned_load_le_u64(value.data.data);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_f64_from_metadata_value(
    iree_vm_metadata_value_t value, double* out_value) {
  if (!out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_value is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_validate_metadata_value_payload(
      value, IREE_VM_METADATA_VALUE_TYPE_F64, 8));
  *out_value = iree_unaligned_load_le_f64(value.data.data);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_string_view_from_metadata_value(
    iree_vm_metadata_value_t value, iree_string_view_t* out_value) {
  if (!out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_value is required");
  }
  if (value.type != IREE_VM_METADATA_VALUE_TYPE_UTF8 ||
      (value.data.data_length != 0 && !value.data.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "valid UTF8 metadata is required");
  }
  const iree_string_view_t result = iree_make_string_view(
      (const char*)value.data.data, value.data.data_length);
  if (!iree_unicode_utf8_validate(result)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "UTF8 metadata contains invalid UTF-8");
  }
  *out_value = result;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_const_byte_span_from_metadata_value(
    iree_vm_metadata_value_t value, iree_const_byte_span_t* out_value) {
  if (!out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_value is required");
  }
  if (value.type != IREE_VM_METADATA_VALUE_TYPE_BYTES ||
      (value.data.data_length != 0 && !value.data.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "valid BYTES metadata is required");
  }
  *out_value = value.data;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_module_metadata_by_ordinal_none(
    const iree_vm_module_t* module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry) {
  (void)module;
  (void)query;
  memset(out_entry, 0, sizeof(*out_entry));
}
