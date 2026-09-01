// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_metadata.h"

#include <stdlib.h>
#include <string.h>

#include "iree/vm/bytecode/wire/module_format.h"
#include "loom/ir/module.h"
#include "loom/ops/metadata/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/emit/vm/module_layout.h"
#include "loom/target/emit/vm/module_types.h"

typedef struct loom_vm_module_metadata_counts_t {
  // Number of module-scope metadata entries.
  iree_host_size_t module_entry_count;
  // Total metadata entries across every scope.
  iree_host_size_t total_entry_count;
  // Number of nonempty import metadata scopes.
  iree_host_size_t import_scope_count;
  // Number of nonempty export metadata scopes.
  iree_host_size_t export_scope_count;
} loom_vm_module_metadata_counts_t;

static int loom_vm_module_metadata_compare_entries(const void* lhs_ptr,
                                                   const void* rhs_ptr) {
  const loom_vm_module_metadata_entry_layout_t* lhs =
      (const loom_vm_module_metadata_entry_layout_t*)lhs_ptr;
  const loom_vm_module_metadata_entry_layout_t* rhs =
      (const loom_vm_module_metadata_entry_layout_t*)rhs_ptr;
  return iree_string_view_compare(lhs->key, rhs->key);
}

static iree_status_t loom_vm_module_metadata_count_entries(
    iree_host_size_t entry_count, iree_host_size_t* inout_total_entry_count) {
  if (!iree_host_size_checked_add(*inout_total_entry_count, entry_count,
                                  inout_total_entry_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM metadata entry count exceeds host size");
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_metadata_count(
    loom_vm_module_layout_t* layout,
    loom_vm_module_metadata_counts_t* out_counts) {
  *out_counts = (loom_vm_module_metadata_counts_t){0};
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(layout->module), op) {
    if (!loom_metadata_module_isa(op)) continue;
    ++out_counts->module_entry_count;
  }
  if (out_counts->module_entry_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM module-scope metadata count exceeds the u16 domain");
  }
  out_counts->total_entry_count = out_counts->module_entry_count;

  for (iree_host_size_t i = 0; i < layout->import_count; ++i) {
    const loom_named_attr_slice_t metadata =
        loom_func_like_import_metadata(loom_func_like_cast(
            layout->module, layout->imports[i]->declaration_op));
    if (metadata.count == 0) continue;
    if (metadata.count > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM import metadata scope exceeds the u16 entry-count domain");
    }
    ++out_counts->import_scope_count;
    IREE_RETURN_IF_ERROR(loom_vm_module_metadata_count_entries(
        metadata.count, &out_counts->total_entry_count));
  }
  for (iree_host_size_t i = 0; i < layout->export_count; ++i) {
    const loom_named_attr_slice_t metadata = loom_func_like_export_metadata(
        loom_func_like_cast(layout->module, layout->exports[i]->function_op));
    if (metadata.count == 0) continue;
    if (metadata.count > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM export metadata scope exceeds the u16 entry-count domain");
    }
    ++out_counts->export_scope_count;
    IREE_RETURN_IF_ERROR(loom_vm_module_metadata_count_entries(
        metadata.count, &out_counts->total_entry_count));
  }
  if (out_counts->total_entry_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM metadata entry count exceeds the u32 domain");
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_metadata_initialize_entry(
    const loom_module_t* module, loom_string_id_t key_id,
    loom_attribute_t value, loom_vm_module_metadata_entry_layout_t* out_entry) {
  if (key_id == LOOM_STRING_ID_INVALID || key_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM metadata key has an invalid string ID");
  }
  const iree_string_view_t key = module->strings.entries[key_id];
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM metadata keys must be nonempty");
  }

  *out_entry = (loom_vm_module_metadata_entry_layout_t){
      .key = key,
      .key_string_ordinal = UINT16_MAX,
  };
  switch ((loom_attr_kind_t)value.kind) {
    case LOOM_ATTR_BOOL:
      out_entry->value_type = IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BOOL;
      out_entry->value.scalar[0] = loom_attr_as_bool(value) ? 1 : 0;
      break;
    case LOOM_ATTR_I64:
      out_entry->value_type = IREE_VM_BYTECODE_METADATA_VALUE_TYPE_I64;
      iree_unaligned_store_le_u64(out_entry->value.scalar,
                                  (uint64_t)loom_attr_as_i64(value));
      break;
    case LOOM_ATTR_U64:
      out_entry->value_type = IREE_VM_BYTECODE_METADATA_VALUE_TYPE_U64;
      iree_unaligned_store_le_u64(out_entry->value.scalar,
                                  loom_attr_as_u64(value));
      break;
    case LOOM_ATTR_F64: {
      out_entry->value_type = IREE_VM_BYTECODE_METADATA_VALUE_TYPE_F64;
      uint64_t bits = 0;
      const double f64 = loom_attr_as_f64(value);
      memcpy(&bits, &f64, sizeof(bits));
      iree_unaligned_store_le_u64(out_entry->value.scalar, bits);
      break;
    }
    case LOOM_ATTR_STRING: {
      const loom_string_id_t string_id = loom_attr_as_string_id(value);
      if (string_id == LOOM_STRING_ID_INVALID ||
          string_id >= module->strings.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "VM metadata '%.*s' UTF-8 value has an invalid string ID",
            (int)key.size, key.data);
      }
      const iree_string_view_t string = module->strings.entries[string_id];
      out_entry->value_type = IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8;
      out_entry->value.variable =
          iree_make_const_byte_span(string.data, string.size);
      break;
    }
    case LOOM_ATTR_BYTES:
      out_entry->value_type = IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BYTES;
      out_entry->value.variable = loom_attr_as_bytes(value);
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Loom metadata '%.*s' attribute kind %u has no VM metadata encoding",
          (int)key.size, key.data, (unsigned int)value.kind);
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_metadata_validate_entry_range(
    const loom_vm_module_metadata_entry_layout_t* entries,
    iree_host_size_t entry_count) {
  for (iree_host_size_t i = 1; i < entry_count; ++i) {
    if (iree_string_view_compare(entries[i - 1].key, entries[i].key) >= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM metadata keys must be strictly ordered within each scope");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_metadata_append_dictionary(
    const loom_module_t* module, loom_named_attr_slice_t metadata,
    loom_vm_module_metadata_entry_layout_t* entries) {
  for (iree_host_size_t i = 0; i < metadata.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vm_module_metadata_initialize_entry(
        module, metadata.entries[i].name_id, metadata.entries[i].value,
        &entries[i]));
  }
  return loom_vm_module_metadata_validate_entry_range(entries, metadata.count);
}

iree_const_byte_span_t loom_vm_module_metadata_entry_value(
    const loom_vm_module_metadata_entry_layout_t* entry) {
  switch (entry->value_type) {
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BOOL:
      return iree_make_const_byte_span(entry->value.scalar, 1);
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_I64:
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_U64:
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_F64:
      return iree_make_const_byte_span(entry->value.scalar, 8);
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8:
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BYTES:
      return entry->value.variable;
    default:
      IREE_ASSERT_UNREACHABLE("planned VM metadata value type");
      return iree_const_byte_span_empty();
  }
}

iree_status_t loom_vm_module_metadata_layout_build(
    iree_arena_allocator_t* arena, loom_vm_module_layout_t* layout) {
  layout->metadata = (loom_vm_module_metadata_layout_t){0};
  loom_vm_module_metadata_counts_t counts;
  IREE_RETURN_IF_ERROR(loom_vm_module_metadata_count(layout, &counts));
  if (counts.total_entry_count == 0) return iree_ok_status();

  loom_vm_module_metadata_entry_layout_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, counts.total_entry_count, sizeof(*entries), (void**)&entries));
  loom_vm_module_metadata_scope_layout_t* import_scopes = NULL;
  if (counts.import_scope_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, counts.import_scope_count, sizeof(*import_scopes),
        (void**)&import_scopes));
  }
  loom_vm_module_metadata_scope_layout_t* export_scopes = NULL;
  if (counts.export_scope_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, counts.export_scope_count, sizeof(*export_scopes),
        (void**)&export_scopes));
  }
  iree_host_size_t value_offset_count = 0;
  if (!iree_host_size_checked_add(counts.total_entry_count, 1,
                                  &value_offset_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM metadata offset count exceeds host size");
  }
  uint64_t* value_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, value_offset_count,
                                                 sizeof(*value_offsets),
                                                 (void**)&value_offsets));
  layout->metadata = (loom_vm_module_metadata_layout_t){
      .entries = entries,
      .module_entry_count = (uint32_t)counts.module_entry_count,
      .total_entry_count = (uint32_t)counts.total_entry_count,
      .import_scopes = import_scopes,
      .import_scope_count = (uint32_t)counts.import_scope_count,
      .export_scopes = export_scopes,
      .export_scope_count = (uint32_t)counts.export_scope_count,
      .value_offsets = value_offsets,
  };

  uint32_t entry_index = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(layout->module), op) {
    if (!loom_metadata_module_isa(op)) continue;
    IREE_RETURN_IF_ERROR(loom_vm_module_metadata_initialize_entry(
        layout->module, loom_metadata_module_key(op),
        loom_metadata_module_value(op), &entries[entry_index++]));
  }
  IREE_ASSERT_EQ(entry_index, layout->metadata.module_entry_count);
  qsort(entries, layout->metadata.module_entry_count, sizeof(*entries),
        loom_vm_module_metadata_compare_entries);
  IREE_RETURN_IF_ERROR(loom_vm_module_metadata_validate_entry_range(
      entries, layout->metadata.module_entry_count));

  uint32_t import_scope_index = 0;
  for (iree_host_size_t i = 0; i < layout->import_count; ++i) {
    const loom_named_attr_slice_t metadata =
        loom_func_like_import_metadata(loom_func_like_cast(
            layout->module, layout->imports[i]->declaration_op));
    if (metadata.count == 0) continue;
    import_scopes[import_scope_index++] =
        (loom_vm_module_metadata_scope_layout_t){
            .declaration_ordinal = (uint16_t)i,
            .entry_count = (uint16_t)metadata.count,
            .entry_base = entry_index,
        };
    IREE_RETURN_IF_ERROR(loom_vm_module_metadata_append_dictionary(
        layout->module, metadata, &entries[entry_index]));
    entry_index += (uint32_t)metadata.count;
  }
  IREE_ASSERT_EQ(import_scope_index, layout->metadata.import_scope_count);

  uint32_t export_scope_index = 0;
  for (iree_host_size_t i = 0; i < layout->export_count; ++i) {
    const loom_named_attr_slice_t metadata = loom_func_like_export_metadata(
        loom_func_like_cast(layout->module, layout->exports[i]->function_op));
    if (metadata.count == 0) continue;
    export_scopes[export_scope_index++] =
        (loom_vm_module_metadata_scope_layout_t){
            .declaration_ordinal = (uint16_t)i,
            .entry_count = (uint16_t)metadata.count,
            .entry_base = entry_index,
        };
    IREE_RETURN_IF_ERROR(loom_vm_module_metadata_append_dictionary(
        layout->module, metadata, &entries[entry_index]));
    entry_index += (uint32_t)metadata.count;
  }
  IREE_ASSERT_EQ(export_scope_index, layout->metadata.export_scope_count);
  IREE_ASSERT_EQ(entry_index, layout->metadata.total_entry_count);

  uint64_t value_data_length = 0;
  value_offsets[0] = 0;
  for (uint32_t i = 0; i < layout->metadata.total_entry_count; ++i) {
    const iree_const_byte_span_t value =
        loom_vm_module_metadata_entry_value(&entries[i]);
    if (value.data_length > UINT64_MAX - value_data_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM metadata value data exceeds the u64 domain");
    }
    value_data_length += value.data_length;
    value_offsets[i + 1] = value_data_length;
  }
  return iree_ok_status();
}

void loom_vm_module_metadata_resolve_string_ordinals(
    loom_vm_module_layout_t* layout) {
  for (uint32_t i = 0; i < layout->metadata.total_entry_count; ++i) {
    loom_vm_module_metadata_entry_layout_t* entry =
        &layout->metadata.entries[i];
    const bool resolved = loom_vm_module_type_tables_try_resolve_string_ordinal(
        &layout->type_tables, entry->key, &entry->key_string_ordinal);
    IREE_ASSERT(resolved);
  }
}
