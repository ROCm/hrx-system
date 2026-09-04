// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/module.h"

#include "iree/base/internal/unicode.h"
#include "iree/vm/reflection.h"

#define IREE_VM_MODULE_DIRECT_ORDINAL_CAPACITY \
  ((iree_host_size_t)UINT16_MAX + 1)

static iree_status_t iree_vm_module_validate_symbol(iree_string_view_t value,
                                                    const char* label) {
  if (!value.data || value.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s must be nonempty",
                            label);
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

static iree_status_t iree_vm_module_validate_vtable(
    const iree_vm_module_vtable_t* vtable) {
  if (!vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module vtable is required");
  }
  if (vtable->structure_size < IREE_VM_MODULE_VTABLE_V0_REQUIRED_SIZE) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "module vtable has %" PRIu32 " bytes; ABI zero requires at least %zu",
        vtable->structure_size, (size_t)IREE_VM_MODULE_VTABLE_V0_REQUIRED_SIZE);
  }
  if (vtable->abi_version != IREE_VM_MODULE_ABI_VERSION_0) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "module ABI version %" PRIu32 " is unsupported",
                            vtable->abi_version);
  }
  if (!vtable->destroy || !vtable->function_start || !vtable->function_resume ||
      !vtable->query_import_group || !vtable->query_import ||
      !vtable->query_export || !vtable->query_callable_type ||
      !vtable->query_presentation || !vtable->metadata_by_ordinal) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module vtable is missing a required callback");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_descriptor(
    const iree_vm_module_descriptor_t* descriptor) {
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module descriptor is required");
  }
  IREE_RETURN_IF_ERROR(
      iree_vm_module_validate_symbol(descriptor->name, "module name"));
  if ((descriptor->flags &
       ~(iree_vm_module_flags_t)IREE_VM_MODULE_FLAG_LINKABLE) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module flags contain unsupported bits");
  }
  if (descriptor->ref_types.count != 0 && !descriptor->ref_types.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module ref-type storage is required");
  }
  if (descriptor->ref_types.count != 0 &&
      !iree_host_ptr_has_alignment(descriptor->ref_types.data,
                                   iree_alignof(iree_vm_ref_type_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module ref-type storage is misaligned");
  }
  if (descriptor->ref_types.count > IREE_VM_MODULE_DIRECT_ORDINAL_CAPACITY ||
      descriptor->counts.function_count >
          IREE_VM_MODULE_DIRECT_ORDINAL_CAPACITY ||
      descriptor->counts.callable_type_count >
          IREE_VM_MODULE_DIRECT_ORDINAL_CAPACITY ||
      descriptor->counts.import_group_count >
          IREE_VM_MODULE_DIRECT_ORDINAL_CAPACITY ||
      descriptor->counts.import_count >
          IREE_VM_MODULE_DIRECT_ORDINAL_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module direct-ordinal domains cannot exceed 65536 entries");
  }
#if defined(IREE_PTR_SIZE_64)
  if (descriptor->process_storage_size > (iree_host_size_t)UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module process storage cannot exceed %" PRIu32 " bytes", UINT32_MAX);
  }
#endif  // IREE_PTR_SIZE_64
  for (iree_host_size_t i = 0; i < descriptor->ref_types.count; ++i) {
    const iree_vm_ref_type_t type = descriptor->ref_types.data[i];
    if (!type ||
        !iree_host_ptr_has_alignment(
            type, iree_alignof(iree_vm_ref_type_descriptor_t)) ||
        !type->table) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "module ref type %" PRIhsz " is null, misaligned, or unowned", i);
    }
    if ((type->table->flags &
         ~(iree_vm_ref_type_table_flags_t)
             IREE_VM_REF_TYPE_TABLE_FLAG_REFLECTION_ONLY) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "module ref type %" PRIhsz " has unsupported provider flags", i);
    }
    if (iree_any_bit_set(descriptor->flags, IREE_VM_MODULE_FLAG_LINKABLE) &&
        iree_any_bit_set(type->table->flags,
                         IREE_VM_REF_TYPE_TABLE_FLAG_REFLECTION_ONLY)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "linkable module ref type %" PRIhsz " is reflection-only", i);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_signature_type(
    const iree_vm_module_descriptor_t* descriptor,
    iree_vm_module_signature_type_t type,
    iree_host_size_t current_callable_ordinal) {
  if (type.kind > IREE_VM_SCALAR_TYPE_NONE &&
      type.kind <= IREE_VM_SCALAR_TYPE_F64) {
    if (type.type_ordinal != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "scalar signature type has a nonzero ordinal");
    }
    return iree_ok_status();
  }
  if (type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
    if (type.type_ordinal >= descriptor->ref_types.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "signature ref-type ordinal is out of range");
    }
    return iree_ok_status();
  }
  if (type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION) {
    if (type.type_ordinal >= current_callable_ordinal) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "signature function type must precede its containing callable");
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "signature type kind 0x%04" PRIx16 " is invalid",
                          type.kind);
}

static int iree_vm_module_compare_u32(uint32_t lhs, uint32_t rhs) {
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

static int iree_vm_module_compare_signature_type(
    const iree_vm_module_t* module, iree_vm_module_signature_type_t lhs,
    iree_vm_module_signature_type_t rhs) {
  int comparison = iree_vm_module_compare_u32(lhs.kind, rhs.kind);
  if (comparison != 0) return comparison;
  if (lhs.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
    const iree_vm_ref_type_key_t lhs_key = iree_vm_ref_type_key(
        module->descriptor->ref_types.data[lhs.type_ordinal]);
    const iree_vm_ref_type_key_t rhs_key = iree_vm_ref_type_key(
        module->descriptor->ref_types.data[rhs.type_ordinal]);
    comparison = iree_string_view_compare(lhs_key.namespace_name,
                                          rhs_key.namespace_name);
    return comparison != 0
               ? comparison
               : iree_string_view_compare(lhs_key.type_name, rhs_key.type_name);
  }
  return iree_vm_module_compare_u32(lhs.type_ordinal, rhs.type_ordinal);
}

static int iree_vm_module_compare_signature_sides(
    const iree_vm_module_t* module, iree_vm_module_signature_side_t lhs,
    iree_vm_module_signature_side_t rhs) {
  int comparison =
      iree_vm_module_compare_u32((uint32_t)lhs.count, (uint32_t)rhs.count);
  if (comparison != 0) return comparison;
  for (iree_host_size_t i = 0; i < lhs.count; ++i) {
    comparison =
        iree_vm_module_compare_signature_type(module, lhs.data[i], rhs.data[i]);
    if (comparison != 0) return comparison;
  }
  return 0;
}

static int iree_vm_module_compare_callable_types(
    const iree_vm_module_t* module,
    const iree_vm_module_callable_type_declaration_t* lhs,
    const iree_vm_module_callable_type_declaration_t* rhs) {
  int comparison =
      iree_vm_module_compare_u32(lhs->nesting_depth, rhs->nesting_depth);
  if (comparison != 0) return comparison;
  comparison = iree_vm_module_compare_signature_sides(
      module, lhs->signature.arguments, rhs->signature.arguments);
  if (comparison != 0) return comparison;
  comparison = iree_vm_module_compare_signature_sides(
      module, lhs->signature.results, rhs->signature.results);
  return comparison != 0 ? comparison
                         : iree_vm_module_compare_u32(lhs->flags, rhs->flags);
}

static iree_status_t iree_vm_module_validate_signature_side(
    const iree_vm_module_t* module, iree_vm_module_signature_side_t side,
    iree_host_size_t current_callable_ordinal, const char* label,
    uint32_t* inout_nesting_depth) {
  if (side.count != 0 && !side.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s signature storage is required", label);
  }
  if (side.count != 0 &&
      !iree_host_ptr_has_alignment(
          side.data, iree_alignof(iree_vm_module_signature_type_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s signature storage is misaligned", label);
  }
  uint16_t expected_value_count = 0;
  uint16_t expected_ref_count = 0;
  uint16_t expected_function_count = 0;
  for (uint16_t i = 0; i < side.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_module_validate_signature_type(
        module->descriptor, side.data[i], current_callable_ordinal));
    if (side.data[i].kind > IREE_VM_SCALAR_TYPE_NONE &&
        side.data[i].kind <= IREE_VM_SCALAR_TYPE_F64) {
      ++expected_value_count;
    } else if (side.data[i].kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
      ++expected_ref_count;
    } else {
      ++expected_function_count;
      iree_vm_module_callable_type_declaration_t child = {0};
      module->vtable->query_callable_type(module, side.data[i].type_ordinal,
                                          &child);
      if (child.nesting_depth == UINT16_MAX) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "callable type nesting depth overflows u16");
      }
      *inout_nesting_depth =
          iree_max(*inout_nesting_depth, (uint32_t)child.nesting_depth + 1);
    }
  }
  if (side.value_count != expected_value_count ||
      side.ref_count != expected_ref_count ||
      side.function_count != expected_function_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s signature bank counts are not exact", label);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_callable_types(
    const iree_vm_module_t* module) {
  iree_vm_module_callable_type_declaration_t previous = {0};
  iree_vm_module_callable_field_counts_t total_fields = {0};
  for (iree_host_size_t i = 0;
       i < module->descriptor->counts.callable_type_count; ++i) {
    iree_vm_module_callable_type_declaration_t callable_type = {0};
    module->vtable->query_callable_type(module, i, &callable_type);
    if ((callable_type.flags & ~(iree_vm_callable_type_flags_t)
                                   IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD) != 0 ||
        callable_type.reserved != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "callable type %" PRIhsz
                              " has unsupported flags or reserved bits",
                              i);
    }
    uint32_t expected_nesting_depth = 0;
    IREE_RETURN_IF_ERROR(iree_vm_module_validate_signature_side(
        module, callable_type.signature.arguments, i, "argument",
        &expected_nesting_depth));
    IREE_RETURN_IF_ERROR(iree_vm_module_validate_signature_side(
        module, callable_type.signature.results, i, "result",
        &expected_nesting_depth));
    const iree_host_size_t value_count =
        callable_type.signature.arguments.value_count +
        callable_type.signature.results.value_count;
    const iree_host_size_t ref_count =
        callable_type.signature.arguments.ref_count +
        callable_type.signature.results.ref_count;
    const iree_host_size_t function_count =
        callable_type.signature.arguments.function_count +
        callable_type.signature.results.function_count;
    if (!iree_host_size_checked_add(total_fields.value_count, value_count,
                                    &total_fields.value_count) ||
        !iree_host_size_checked_add(total_fields.ref_count, ref_count,
                                    &total_fields.ref_count) ||
        !iree_host_size_checked_add(total_fields.function_count, function_count,
                                    &total_fields.function_count)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "callable signature field counts overflow");
    }
    if (callable_type.nesting_depth != expected_nesting_depth) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "callable type %" PRIhsz " has a noncanonical nesting depth", i);
    }
    if (i != 0 && iree_vm_module_compare_callable_types(module, &previous,
                                                        &callable_type) >= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "callable types must be unique and strictly ordered");
    }
    previous = callable_type;
  }
  const iree_vm_module_callable_field_counts_t declared_fields =
      module->descriptor->counts.callable_fields;
  if (declared_fields.value_count != total_fields.value_count ||
      declared_fields.ref_count != total_fields.ref_count ||
      declared_fields.function_count != total_fields.function_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "aggregate callable field counts are not exact");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_metadata_value(
    iree_vm_metadata_value_t value) {
  if (value.type == IREE_VM_METADATA_VALUE_TYPE_INVALID ||
      value.type > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metadata value type %" PRIu32
                            " is outside the 16-bit type domain",
                            value.type);
  }
  if (value.data.data_length != 0 && !value.data.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metadata value bytes are required");
  }
  switch (value.type) {
    case IREE_VM_METADATA_VALUE_TYPE_BOOL:
      if (value.data.data_length != 1 || value.data.data[0] > 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "BOOL metadata must be exactly one canonical 0/1 byte");
      }
      break;
    case IREE_VM_METADATA_VALUE_TYPE_I64:
    case IREE_VM_METADATA_VALUE_TYPE_U64:
    case IREE_VM_METADATA_VALUE_TYPE_F64:
      if (value.data.data_length != 8) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "64-bit metadata scalars must contain exactly eight bytes");
      }
      break;
    case IREE_VM_METADATA_VALUE_TYPE_UTF8:
      if (!iree_unicode_utf8_validate(iree_make_string_view(
              (const char*)value.data.data, value.data.data_length))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "UTF8 metadata contains invalid UTF-8");
      }
      break;
    default:
      break;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_metadata_scope(
    const iree_vm_module_t* module, iree_vm_module_metadata_scope_t scope,
    iree_host_size_t count) {
  iree_string_view_t previous_key = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_vm_metadata_entry_t entry = {0};
    const iree_vm_module_metadata_query_t query = {scope, i};
    module->vtable->metadata_by_ordinal(module, &query, &entry);
    IREE_RETURN_IF_ERROR(
        iree_vm_module_validate_symbol(entry.key, "metadata key"));
    if (i != 0 && iree_string_view_compare(previous_key, entry.key) >= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "metadata keys must be unique and strictly ordered within a scope");
    }
    IREE_RETURN_IF_ERROR(iree_vm_module_validate_metadata_value(entry.value));
    previous_key = entry.key;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_imports(
    const iree_vm_module_t* module) {
  const iree_vm_module_descriptor_t* descriptor = module->descriptor;
  iree_string_view_t previous_group_name = iree_string_view_empty();
  iree_host_size_t covered_import_count = 0;
  for (iree_host_size_t group_i = 0;
       group_i < descriptor->counts.import_group_count; ++group_i) {
    iree_vm_module_import_group_t group = {0};
    module->vtable->query_import_group(module, group_i, &group);
    IREE_RETURN_IF_ERROR(iree_vm_module_validate_symbol(
        group.target_module_name, "import target module name"));
    if (group_i != 0 &&
        iree_string_view_compare(previous_group_name,
                                 group.target_module_name) >= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "import groups must be strictly name-sorted");
    }
    if (group.import_count == 0 ||
        group.first_import_ordinal != covered_import_count ||
        covered_import_count > descriptor->counts.import_count ||
        group.import_count >
            descriptor->counts.import_count - covered_import_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "import groups must exactly partition the flat import domain");
    }

    iree_string_view_t previous_export_name = iree_string_view_empty();
    for (iree_host_size_t i = 0; i < group.import_count; ++i) {
      const iree_host_size_t import_ordinal = group.first_import_ordinal + i;
      iree_vm_module_import_declaration_t import_declaration = {0};
      module->vtable->query_import(module, import_ordinal, &import_declaration);
      if (!iree_string_view_equal(import_declaration.target_module_name,
                                  group.target_module_name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "import %" PRIhsz
                                " does not match its target module group",
                                import_ordinal);
      }
      IREE_RETURN_IF_ERROR(iree_vm_module_validate_symbol(
          import_declaration.target_export_name, "import target export name"));
      if (import_declaration.callable_type_ordinal >=
          descriptor->counts.callable_type_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "import %" PRIhsz " callable type is out of range", import_ordinal);
      }
      if ((import_declaration.flags &
           ~(iree_vm_module_import_flags_t)
               IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL) != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "import %" PRIhsz
                                " has unsupported behavior flags",
                                import_ordinal);
      }
      if (i != 0) {
        const int name_comparison = iree_string_view_compare(
            previous_export_name, import_declaration.target_export_name);
        if (name_comparison > 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "imports must be ordered by target export name");
        }
      }
      const iree_vm_module_metadata_scope_t metadata_scope = {
          IREE_VM_MODULE_METADATA_SCOPE_KIND_IMPORT,
          import_ordinal,
      };
      IREE_RETURN_IF_ERROR(iree_vm_module_validate_metadata_scope(
          module, metadata_scope, import_declaration.metadata_count));
      previous_export_name = import_declaration.target_export_name;
    }
    covered_import_count += group.import_count;
    previous_group_name = group.target_module_name;
  }
  if (covered_import_count != descriptor->counts.import_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "import groups do not cover the complete flat import domain");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_exports(
    const iree_vm_module_t* module) {
  iree_string_view_t previous_export_name = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < module->descriptor->counts.export_count;
       ++i) {
    iree_vm_module_export_declaration_t export_declaration = {0};
    module->vtable->query_export(module, i, &export_declaration);
    IREE_RETURN_IF_ERROR(iree_vm_module_validate_symbol(
        export_declaration.export_name, "export name"));
    if (i != 0 &&
        iree_string_view_compare(previous_export_name,
                                 export_declaration.export_name) >= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports must be strictly name-sorted");
    }
    if (export_declaration.callable_type_ordinal >=
        module->descriptor->counts.callable_type_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "export %" PRIhsz " callable type is out of range", i);
    }
    if (export_declaration.function_ordinal >=
        module->descriptor->counts.function_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "export %" PRIhsz " function ordinal is out of range", i);
    }
    const iree_vm_module_metadata_scope_t metadata_scope = {
        IREE_VM_MODULE_METADATA_SCOPE_KIND_EXPORT,
        i,
    };
    IREE_RETURN_IF_ERROR(iree_vm_module_validate_metadata_scope(
        module, metadata_scope, export_declaration.metadata_count));
    previous_export_name = export_declaration.export_name;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_module_validate_semantics(
    const iree_vm_module_t* module) {
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_callable_types(module));
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_imports(module));
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_exports(module));
  const iree_vm_module_metadata_scope_t scope = {
      IREE_VM_MODULE_METADATA_SCOPE_KIND_MODULE,
      0,
  };
  return iree_vm_module_validate_metadata_scope(
      module, scope, module->descriptor->counts.metadata_count);
}

IREE_API_EXPORT iree_status_t
iree_vm_module_initialize(const iree_vm_module_vtable_t* vtable,
                          const iree_vm_module_descriptor_t* descriptor,
                          iree_vm_module_t* out_module) {
  if (!out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_module is required");
  }
  out_module->vtable = NULL;
  out_module->descriptor = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_vtable(vtable));
  IREE_RETURN_IF_ERROR(iree_vm_module_validate_descriptor(descriptor));

  out_module->vtable = vtable;
  out_module->descriptor = descriptor;
  iree_status_t status = iree_vm_module_validate_semantics(out_module);
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&out_module->ref_count);
  } else {
    out_module->vtable = NULL;
    out_module->descriptor = NULL;
  }
  return status;
}

IREE_API_EXPORT void iree_vm_module_retain(iree_vm_module_t* module) {
  if (module) {
    iree_atomic_ref_count_inc(&module->ref_count);
  }
}

IREE_API_EXPORT void iree_vm_module_release(iree_vm_module_t* module) {
  if (module && iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    module->vtable->destroy(module);
  }
}

IREE_API_EXPORT iree_string_view_t
iree_vm_module_name(const iree_vm_module_t* module) {
  return module->descriptor->name;
}

IREE_API_EXPORT iree_host_size_t
iree_vm_module_import_count(const iree_vm_module_t* module) {
  return module->descriptor->counts.import_count;
}

IREE_API_EXPORT iree_host_size_t
iree_vm_module_export_count(const iree_vm_module_t* module) {
  return module->descriptor->counts.export_count;
}

IREE_API_EXPORT iree_host_size_t
iree_vm_module_function_count(const iree_vm_module_t* module) {
  return module->descriptor->counts.function_count;
}

IREE_API_EXPORT iree_host_size_t
iree_vm_module_ref_type_count(const iree_vm_module_t* module) {
  return module->descriptor->ref_types.count;
}

IREE_API_EXPORT iree_status_t iree_vm_module_ref_type_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_ref_type_t* out_type) {
  if (!module || !out_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and out_type are required");
  }
  if (ordinal >= module->descriptor->ref_types.count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ref-type ordinal %" PRIhsz " is out of range",
                            ordinal);
  }
  *out_type = module->descriptor->ref_types.data[ordinal];
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_module_import_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_import_t* out_import) {
  if (!module || !out_import) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and out_import are required");
  }
  if (ordinal >= module->descriptor->counts.import_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "import ordinal %" PRIhsz " is out of range",
                            ordinal);
  }
  const iree_vm_import_t import_value = {module, ordinal};
  *out_import = import_value;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_module_export_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_export_t* out_export) {
  if (!module || !out_export) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and out_export are required");
  }
  if (ordinal >= module->descriptor->counts.export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIhsz " is out of range",
                            ordinal);
  }
  const iree_vm_export_t export_value = {module, ordinal};
  *out_export = export_value;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_module_lookup_export(
    const iree_vm_module_t* module, iree_string_view_t name,
    iree_vm_export_t* out_export) {
  if (!module || !out_export || (name.size != 0 && !name.data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, coherent name, and out_export are required");
  }
  iree_host_size_t low = 0;
  iree_host_size_t high = module->descriptor->counts.export_count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    iree_vm_module_export_declaration_t export_declaration = {0};
    module->vtable->query_export(module, middle, &export_declaration);
    if (iree_string_view_compare(export_declaration.export_name, name) < 0) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low < module->descriptor->counts.export_count) {
    iree_vm_module_export_declaration_t export_declaration = {0};
    module->vtable->query_export(module, low, &export_declaration);
    if (iree_string_view_equal(export_declaration.export_name, name)) {
      const iree_vm_export_t export_value = {module, low};
      *out_export = export_value;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND, "export '%.*s%s' not found",
                          (int)iree_min(name.size, 128),
                          name.data ? name.data : "",
                          name.size <= 128 ? "" : "...");
}

IREE_API_EXPORT iree_status_t iree_vm_module_query_import_group(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group) {
  if (!module || !out_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and out_group are required");
  }
  if (ordinal >= module->descriptor->counts.import_group_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "import-group ordinal %" PRIhsz " is out of range",
                            ordinal);
  }
  iree_vm_module_import_group_t group = {0};
  module->vtable->query_import_group(module, ordinal, &group);
  *out_group = group;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_module_query_import(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import) {
  if (!module || !out_import) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and out_import are required");
  }
  if (ordinal >= module->descriptor->counts.import_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "import ordinal %" PRIhsz " is out of range",
                            ordinal);
  }
  iree_vm_module_import_declaration_t import_declaration = {0};
  module->vtable->query_import(module, ordinal, &import_declaration);
  *out_import = import_declaration;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_module_query_export(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export) {
  if (!module || !out_export) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and out_export are required");
  }
  if (ordinal >= module->descriptor->counts.export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIhsz " is out of range",
                            ordinal);
  }
  iree_vm_module_export_declaration_t export_declaration = {0};
  module->vtable->query_export(module, ordinal, &export_declaration);
  *out_export = export_declaration;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_module_query_callable_type(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type) {
  if (!module || !out_callable_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and out_callable_type are required");
  }
  if (ordinal >= module->descriptor->counts.callable_type_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "callable-type ordinal %" PRIhsz " is out of range",
                            ordinal);
  }
  iree_vm_module_callable_type_declaration_t callable_type = {0};
  module->vtable->query_callable_type(module, ordinal, &callable_type);
  *out_callable_type = callable_type;
  return iree_ok_status();
}

IREE_API_EXPORT iree_vm_import_target_t
iree_vm_import_target(iree_vm_import_t import_value) {
  iree_vm_module_import_declaration_t import_declaration = {0};
  import_value.module->vtable->query_import(
      import_value.module, import_value.ordinal, &import_declaration);
  const iree_vm_import_target_t target = {
      import_declaration.target_module_name,
      import_declaration.target_export_name,
  };
  return target;
}

IREE_API_EXPORT iree_string_view_t
iree_vm_export_name(iree_vm_export_t export_value) {
  iree_vm_module_export_declaration_t export_declaration = {0};
  export_value.module->vtable->query_export(
      export_value.module, export_value.ordinal, &export_declaration);
  return export_declaration.export_name;
}

#undef IREE_VM_MODULE_DIRECT_ORDINAL_CAPACITY
