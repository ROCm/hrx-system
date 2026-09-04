// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/program.h"

#include <string.h>

#include "iree/vm/program_storage.h"
#include "iree/vm/variant.h"

//===----------------------------------------------------------------------===//
// Linked module directory
//===----------------------------------------------------------------------===//

static int iree_vm_program_compare_linked_modules(
    const iree_vm_linked_module_t* lhs, const iree_vm_linked_module_t* rhs) {
  return iree_string_view_compare(lhs->module->descriptor->name,
                                  rhs->module->descriptor->name);
}

static void iree_vm_program_swap_linked_modules(iree_vm_linked_module_t* lhs,
                                                iree_vm_linked_module_t* rhs) {
  const iree_vm_linked_module_t value = *lhs;
  *lhs = *rhs;
  *rhs = value;
}

static void iree_vm_program_insertion_sort_linked_modules(
    iree_vm_linked_module_t* linked_modules, iree_host_size_t count) {
  for (iree_host_size_t i = 1; i < count; ++i) {
    const iree_vm_linked_module_t value = linked_modules[i];
    iree_host_size_t j = i;
    while (j != 0 && iree_vm_program_compare_linked_modules(
                         &value, &linked_modules[j - 1]) < 0) {
      linked_modules[j] = linked_modules[j - 1];
      --j;
    }
    linked_modules[j] = value;
  }
}

static void iree_vm_program_sift_down_linked_modules(
    iree_vm_linked_module_t* linked_modules, iree_host_size_t root,
    iree_host_size_t count) {
  while (root < count / 2) {
    iree_host_size_t child = root * 2 + 1;
    if (child + 1 < count &&
        iree_vm_program_compare_linked_modules(
            &linked_modules[child], &linked_modules[child + 1]) < 0) {
      ++child;
    }
    if (iree_vm_program_compare_linked_modules(&linked_modules[root],
                                               &linked_modules[child]) >= 0) {
      return;
    }
    iree_vm_program_swap_linked_modules(&linked_modules[root],
                                        &linked_modules[child]);
    root = child;
  }
}

static void iree_vm_program_sort_linked_modules(
    iree_vm_linked_module_t* linked_modules, iree_host_size_t count) {
  if (count <= 16) {
    iree_vm_program_insertion_sort_linked_modules(linked_modules, count);
    return;
  }
  for (iree_host_size_t i = count / 2; i != 0; --i) {
    iree_vm_program_sift_down_linked_modules(linked_modules, i - 1, count);
  }
  for (iree_host_size_t end = count; end > 1; --end) {
    iree_vm_program_swap_linked_modules(&linked_modules[0],
                                        &linked_modules[end - 1]);
    iree_vm_program_sift_down_linked_modules(linked_modules, 0, end - 1);
  }
}

static iree_host_size_t iree_vm_program_linked_module_lower_bound(
    const iree_vm_program_t* program, iree_string_view_t name) {
  iree_host_size_t low = 0;
  iree_host_size_t high = program->linked_module_count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    const int comparison = iree_string_view_compare(
        program->linked_modules[middle].module->descriptor->name, name);
    if (comparison < 0) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

const iree_vm_linked_module_t* iree_vm_program_find_linked_module(
    const iree_vm_program_t* program, const iree_vm_module_t* module,
    iree_host_size_t* out_ordinal) {
  const iree_host_size_t ordinal = iree_vm_program_linked_module_lower_bound(
      program, module->descriptor->name);
  if (ordinal >= program->linked_module_count ||
      program->linked_modules[ordinal].module != module) {
    return NULL;
  }
  *out_ordinal = ordinal;
  return &program->linked_modules[ordinal];
}

const iree_vm_linked_module_t* iree_vm_program_lookup_linked_module(
    const iree_vm_program_t* program, iree_string_view_t name,
    iree_host_size_t* out_ordinal) {
  const iree_host_size_t ordinal =
      iree_vm_program_linked_module_lower_bound(program, name);
  if (ordinal >= program->linked_module_count ||
      !iree_string_view_equal(
          program->linked_modules[ordinal].module->descriptor->name, name)) {
    return NULL;
  }
  *out_ordinal = ordinal;
  return &program->linked_modules[ordinal];
}

//===----------------------------------------------------------------------===//
// Structural callable linking
//===----------------------------------------------------------------------===//

typedef struct iree_vm_program_field_storage_t {
  // Next unassigned scalar field record.
  iree_vm_program_scalar_field_abi_t* scalar_fields;
  // Next unassigned ref field record.
  iree_vm_program_ref_field_abi_t* ref_fields;
  // Next unassigned function field record.
  iree_vm_program_function_field_abi_t* function_fields;
} iree_vm_program_field_storage_t;

static void iree_vm_program_build_signature_side_abi(
    const iree_vm_program_t* program,
    const iree_vm_linked_module_t* linked_module,
    iree_vm_module_signature_side_t signature_side,
    iree_vm_program_field_storage_t* field_storage,
    const iree_vm_program_scalar_field_abi_t** out_value_fields,
    const iree_vm_program_ref_field_abi_t** out_ref_fields,
    const iree_vm_program_function_field_abi_t** out_function_fields,
    iree_vm_program_bank_counts_t* out_counts) {
  const iree_vm_program_bank_counts_t counts = {
      signature_side.value_count,
      signature_side.ref_count,
      signature_side.function_count,
  };
  *out_value_fields = counts.value_count ? field_storage->scalar_fields : NULL;
  *out_ref_fields = counts.ref_count ? field_storage->ref_fields : NULL;
  *out_function_fields =
      counts.function_count ? field_storage->function_fields : NULL;
  for (uint16_t i = 0; i < signature_side.count; ++i) {
    const iree_vm_module_signature_type_t type = signature_side.data[i];
    if (type.kind > IREE_VM_SCALAR_TYPE_NONE &&
        type.kind <= IREE_VM_SCALAR_TYPE_F64) {
      *field_storage->scalar_fields++ = (iree_vm_program_scalar_field_abi_t){
          .payload_mask = iree_vm_scalar_type_payload_mask(type.kind),
          .variant_metadata =
              ((uint32_t)type.kind << 2) | IREE_VM_VARIANT_TAG_SCALAR,
          .variant_offset = (uint32_t)(i * sizeof(iree_vm_variant_t)),
      };
    } else if (type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
      *field_storage->ref_fields++ = (iree_vm_program_ref_field_abi_t){
          .type = linked_module->module->descriptor->ref_types
                      .data[type.type_ordinal],
          .variant_offset = (uint32_t)(i * sizeof(iree_vm_variant_t)),
      };
    } else {
      *field_storage->function_fields++ =
          (iree_vm_program_function_field_abi_t){
              .callable_mapping =
                  program->callable_mappings[linked_module->callable_base +
                                             type.type_ordinal],
              .variant_offset = (uint32_t)(i * sizeof(iree_vm_variant_t)),
          };
    }
  }
  *out_counts = counts;
}

static void iree_vm_program_append_root_bank_layout(
    uint16_t count, uint32_t element_size, uint32_t* inout_storage_size,
    iree_vm_program_root_bank_layout_t* out_layout) {
  const uint32_t direct_offset = *inout_storage_size;
  *out_layout = (iree_vm_program_root_bank_layout_t){
      .direct_offset = count ? direct_offset : 0,
      .overflow_offset =
          count > IREE_VM_CALL_DIRECT_REGISTER_COUNT
              ? direct_offset +
                    IREE_VM_CALL_DIRECT_REGISTER_COUNT * element_size
              : 0,
  };
  *inout_storage_size += count * element_size;
}

static iree_vm_program_root_layout_t iree_vm_program_build_root_layout(
    iree_vm_program_bank_counts_t argument_counts,
    iree_vm_program_bank_counts_t result_counts) {
  iree_vm_program_root_layout_t layout = {0};
  iree_vm_program_append_root_bank_layout(
      argument_counts.value_count, sizeof(uint64_t), &layout.storage_size,
      &layout.value_arguments);
  iree_vm_program_append_root_bank_layout(
      argument_counts.ref_count, sizeof(iree_vm_ref_t), &layout.storage_size,
      &layout.ref_arguments);
  iree_vm_program_append_root_bank_layout(
      result_counts.value_count, sizeof(uint64_t), &layout.storage_size,
      &layout.value_results);
  iree_vm_program_append_root_bank_layout(
      result_counts.ref_count, sizeof(iree_vm_ref_t), &layout.storage_size,
      &layout.ref_results);
  iree_vm_program_append_root_bank_layout(
      argument_counts.function_count, sizeof(iree_vm_function_ref_t),
      &layout.storage_size, &layout.function_arguments);
  iree_vm_program_append_root_bank_layout(
      result_counts.function_count, sizeof(iree_vm_function_ref_t),
      &layout.storage_size, &layout.function_results);
  return layout;
}

static iree_vm_program_callable_abi_t iree_vm_program_build_callable_abi(
    const iree_vm_program_t* program,
    const iree_vm_linked_module_t* linked_module,
    iree_vm_module_callable_type_declaration_t declaration,
    iree_vm_program_field_storage_t* field_storage) {
  iree_vm_program_callable_abi_t callable_abi = {0};
  iree_vm_program_build_signature_side_abi(
      program, linked_module, declaration.signature.arguments, field_storage,
      &callable_abi.value_arguments, &callable_abi.ref_arguments,
      &callable_abi.function_arguments, &callable_abi.argument_counts);
  iree_vm_program_build_signature_side_abi(
      program, linked_module, declaration.signature.results, field_storage,
      &callable_abi.value_results, &callable_abi.ref_results,
      &callable_abi.function_results, &callable_abi.result_counts);
  callable_abi.root_layout = iree_vm_program_build_root_layout(
      callable_abi.argument_counts, callable_abi.result_counts);
  return callable_abi;
}

static int iree_vm_program_compare_u32(uint32_t lhs, uint32_t rhs) {
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

static int iree_vm_program_compare_ref_types(iree_vm_ref_type_t lhs,
                                             iree_vm_ref_type_t rhs) {
  if (lhs == rhs) return 0;
  const iree_vm_ref_type_key_t lhs_key = iree_vm_ref_type_key(lhs);
  const iree_vm_ref_type_key_t rhs_key = iree_vm_ref_type_key(rhs);
  int comparison =
      iree_string_view_compare(lhs_key.namespace_name, rhs_key.namespace_name);
  if (comparison != 0) return comparison;
  comparison = iree_string_view_compare(lhs_key.type_name, rhs_key.type_name);
  if (comparison != 0) return comparison;
  return (uintptr_t)lhs < (uintptr_t)rhs ? -1 : 1;
}

static int iree_vm_program_compare_signature_type(
    const iree_vm_program_t* program, const iree_vm_linked_module_t* lhs_module,
    iree_vm_module_signature_type_t lhs,
    const iree_vm_linked_module_t* rhs_module,
    iree_vm_module_signature_type_t rhs) {
  int comparison = iree_vm_program_compare_u32(lhs.kind, rhs.kind);
  if (comparison != 0) return comparison;
  if (lhs.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
    return iree_vm_program_compare_ref_types(
        lhs_module->module->descriptor->ref_types.data[lhs.type_ordinal],
        rhs_module->module->descriptor->ref_types.data[rhs.type_ordinal]);
  }
  if (lhs.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION) {
    const uint32_t lhs_mapping =
        program
            ->callable_mappings[lhs_module->callable_base + lhs.type_ordinal];
    const uint32_t rhs_mapping =
        program
            ->callable_mappings[rhs_module->callable_base + rhs.type_ordinal];
    comparison = iree_vm_program_compare_u32(
        iree_vm_program_callable_token(lhs_mapping),
        iree_vm_program_callable_token(rhs_mapping));
    if (comparison != 0) return comparison;
    return iree_vm_program_compare_u32(
        iree_vm_program_callable_may_yield(lhs_mapping),
        iree_vm_program_callable_may_yield(rhs_mapping));
  }
  return 0;
}

static int iree_vm_program_compare_signature_sides(
    const iree_vm_program_t* program, const iree_vm_linked_module_t* lhs_module,
    iree_vm_module_signature_side_t lhs,
    const iree_vm_linked_module_t* rhs_module,
    iree_vm_module_signature_side_t rhs) {
  int comparison =
      iree_vm_program_compare_u32((uint32_t)lhs.count, (uint32_t)rhs.count);
  if (comparison != 0) return comparison;
  for (iree_host_size_t i = 0; i < lhs.count; ++i) {
    comparison = iree_vm_program_compare_signature_type(
        program, lhs_module, lhs.data[i], rhs_module, rhs.data[i]);
    if (comparison != 0) return comparison;
  }
  return 0;
}

static int iree_vm_program_compare_signatures(
    const iree_vm_program_t* program, const iree_vm_linked_module_t* lhs_module,
    iree_vm_module_signature_t lhs, const iree_vm_linked_module_t* rhs_module,
    iree_vm_module_signature_t rhs) {
  int comparison = iree_vm_program_compare_signature_sides(
      program, lhs_module, lhs.arguments, rhs_module, rhs.arguments);
  return comparison != 0
             ? comparison
             : iree_vm_program_compare_signature_sides(
                   program, lhs_module, lhs.results, rhs_module, rhs.results);
}

static int iree_vm_program_compare_callable_types(
    const iree_vm_program_t* program, const iree_vm_linked_module_t* lhs_module,
    const iree_vm_module_callable_type_declaration_t* lhs,
    const iree_vm_linked_module_t* rhs_module,
    const iree_vm_module_callable_type_declaration_t* rhs) {
  int comparison =
      iree_vm_program_compare_u32(lhs->nesting_depth, rhs->nesting_depth);
  if (comparison != 0) return comparison;
  comparison = iree_vm_program_compare_signatures(
      program, lhs_module, lhs->signature, rhs_module, rhs->signature);
  return comparison != 0 ? comparison
                         : iree_vm_program_compare_u32(lhs->flags, rhs->flags);
}

// Concurrently walks every module's canonical callable table and assigns one
// dense token and ABI to each unique structural signature. Exact callable
// contracts consume all equal heads together; sync and yieldable rows for one
// signature share the token and differ only in their mapping flag.
static void iree_vm_program_link_callable_types(
    iree_vm_program_t* program, uint32_t* module_cursors,
    iree_vm_program_field_storage_t field_storage) {
  const iree_vm_linked_module_t* previous_module = NULL;
  iree_vm_module_callable_type_declaration_t previous_type = {0};
  while (true) {
    iree_host_size_t next_module_ordinal = IREE_HOST_SIZE_MAX;
    iree_vm_module_callable_type_declaration_t next_type = {0};
    for (iree_host_size_t module_i = 0; module_i < program->linked_module_count;
         ++module_i) {
      const iree_vm_linked_module_t* linked_module =
          &program->linked_modules[module_i];
      const uint32_t cursor = module_cursors[module_i];
      if (cursor >=
          linked_module->module->descriptor->counts.callable_type_count) {
        continue;
      }
      iree_vm_module_callable_type_declaration_t candidate = {0};
      linked_module->module->vtable->query_callable_type(linked_module->module,
                                                         cursor, &candidate);
      if (next_module_ordinal == IREE_HOST_SIZE_MAX ||
          iree_vm_program_compare_callable_types(
              program, linked_module, &candidate,
              &program->linked_modules[next_module_ordinal], &next_type) < 0) {
        next_module_ordinal = module_i;
        next_type = candidate;
      }
    }
    if (next_module_ordinal == IREE_HOST_SIZE_MAX) break;

    const iree_vm_linked_module_t* next_module =
        &program->linked_modules[next_module_ordinal];
    if (!previous_module ||
        iree_vm_program_compare_signatures(program, previous_module,
                                           previous_type.signature, next_module,
                                           next_type.signature) != 0) {
      iree_vm_program_callable_abi_t* callable_abi =
          &program->callable_abis[program->callable_abi_count++];
      *callable_abi = iree_vm_program_build_callable_abi(
          program, next_module, next_type, &field_storage);
      previous_module = next_module;
      previous_type = next_type;
    }
    const uint32_t mapping =
        program->callable_abi_count |
        (iree_any_bit_set(next_type.flags, IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD)
             ? IREE_VM_PROGRAM_CALLABLE_MAY_YIELD
             : 0);
    for (iree_host_size_t module_i = 0; module_i < program->linked_module_count;
         ++module_i) {
      const iree_vm_linked_module_t* linked_module =
          &program->linked_modules[module_i];
      const uint32_t cursor = module_cursors[module_i];
      if (cursor >=
          linked_module->module->descriptor->counts.callable_type_count) {
        continue;
      }
      iree_vm_module_callable_type_declaration_t candidate = {0};
      linked_module->module->vtable->query_callable_type(linked_module->module,
                                                         cursor, &candidate);
      if (iree_vm_program_compare_callable_types(program, next_module,
                                                 &next_type, linked_module,
                                                 &candidate) != 0) {
        continue;
      }
      program->callable_mappings[linked_module->callable_base + cursor] =
          mapping;
      ++module_cursors[module_i];
    }
  }
}

//===----------------------------------------------------------------------===//
// Program planning and import resolution
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_program_plan_process_storage(
    iree_vm_program_t* program) {
  iree_host_size_t total_size = 0;
  for (iree_host_size_t i = 0; i < program->linked_module_count; ++i) {
    iree_vm_linked_module_t* linked_module = &program->linked_modules[i];
    const iree_host_size_t storage_size =
        linked_module->module->descriptor->process_storage_size;
    if (storage_size == 0) {
      linked_module->process_storage_offset = UINT32_MAX;
      continue;
    }
    iree_host_size_t aligned_offset = 0;
    if (!iree_host_size_checked_align(total_size, iree_max_align_t,
                                      &aligned_offset) ||
        aligned_offset >= UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "program process-storage offset exceeds 32 bits");
    }
    linked_module->process_storage_offset = (uint32_t)aligned_offset;
    if (!iree_host_size_checked_add(aligned_offset, storage_size,
                                    &total_size) ||
        total_size > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "program process storage exceeds 32 bits");
    }
  }
  program->process_storage_size = total_size;
  return iree_ok_status();
}

static iree_status_t iree_vm_program_resolve_imports(
    iree_vm_program_t* program, uint64_t* import_target_bits) {
  iree_host_size_t source_import_base = 0;
  for (iree_host_size_t source_i = 0; source_i < program->linked_module_count;
       ++source_i) {
    const iree_vm_linked_module_t* source_module =
        &program->linked_modules[source_i];
    const iree_vm_module_descriptor_t* source_descriptor =
        source_module->module->descriptor;
    uint64_t* source_import_target_bits =
        source_descriptor->counts.import_count
            ? &import_target_bits[source_import_base]
            : NULL;
    source_import_base += source_descriptor->counts.import_count;
    iree_host_size_t target_module_cursor = 0;
    for (iree_host_size_t group_i = 0;
         group_i < source_descriptor->counts.import_group_count; ++group_i) {
      iree_vm_module_import_group_t group = {0};
      source_module->module->vtable->query_import_group(source_module->module,
                                                        group_i, &group);
      while (
          target_module_cursor < program->linked_module_count &&
          iree_string_view_compare(program->linked_modules[target_module_cursor]
                                       .module->descriptor->name,
                                   group.target_module_name) < 0) {
        ++target_module_cursor;
      }
      const iree_vm_linked_module_t* target_module = NULL;
      uint16_t target_module_ordinal = 0;
      if (target_module_cursor < program->linked_module_count &&
          iree_string_view_equal(program->linked_modules[target_module_cursor]
                                     .module->descriptor->name,
                                 group.target_module_name)) {
        target_module = &program->linked_modules[target_module_cursor];
        target_module_ordinal = (uint16_t)target_module_cursor;
      }

      iree_host_size_t export_cursor = 0;
      for (iree_host_size_t import_i = 0; import_i < group.import_count;
           ++import_i) {
        const iree_host_size_t import_ordinal =
            group.first_import_ordinal + import_i;
        iree_vm_module_import_declaration_t import_declaration = {0};
        source_module->module->vtable->query_import(
            source_module->module, import_ordinal, &import_declaration);
        const bool is_optional = iree_any_bit_set(
            import_declaration.flags, IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL);
        if (!target_module) {
          if (!is_optional) {
            return iree_make_status(
                IREE_STATUS_NOT_FOUND,
                "module '%.*s' requires missing module '%.*s'",
                (int)iree_min(source_descriptor->name.size, 128),
                source_descriptor->name.data,
                (int)iree_min(group.target_module_name.size, 128),
                group.target_module_name.data);
          }
          continue;
        }

        iree_vm_module_export_declaration_t target_export = {0};
        bool found_export = false;
        while (export_cursor <
               target_module->module->descriptor->counts.export_count) {
          target_module->module->vtable->query_export(
              target_module->module, export_cursor, &target_export);
          const int comparison = iree_string_view_compare(
              target_export.export_name, import_declaration.target_export_name);
          if (comparison < 0) {
            ++export_cursor;
            continue;
          }
          found_export = comparison == 0;
          break;
        }
        if (!found_export) {
          if (!is_optional) {
            return iree_make_status(
                IREE_STATUS_NOT_FOUND,
                "module '%.*s' requires missing export '%.*s.%.*s'",
                (int)iree_min(source_descriptor->name.size, 128),
                source_descriptor->name.data,
                (int)iree_min(group.target_module_name.size, 128),
                group.target_module_name.data,
                (int)iree_min(import_declaration.target_export_name.size, 128),
                import_declaration.target_export_name.data);
          }
          continue;
        }

        const uint32_t source_mapping =
            program
                ->callable_mappings[source_module->callable_base +
                                    import_declaration.callable_type_ordinal];
        const uint32_t target_mapping =
            program->callable_mappings[target_module->callable_base +
                                       target_export.callable_type_ordinal];
        if (iree_vm_program_callable_token(source_mapping) !=
            iree_vm_program_callable_token(target_mapping)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "module '%.*s' import '%.*s.%.*s' has an incompatible callable "
              "type",
              (int)iree_min(source_descriptor->name.size, 128),
              source_descriptor->name.data,
              (int)iree_min(group.target_module_name.size, 128),
              group.target_module_name.data,
              (int)iree_min(import_declaration.target_export_name.size, 128),
              import_declaration.target_export_name.data);
        }
        if (iree_vm_program_callable_may_yield(target_mapping) &&
            !iree_vm_program_callable_may_yield(source_mapping)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "module '%.*s' import '%.*s.%.*s' forbids target suspension",
              (int)iree_min(source_descriptor->name.size, 128),
              source_descriptor->name.data,
              (int)iree_min(group.target_module_name.size, 128),
              group.target_module_name.data,
              (int)iree_min(import_declaration.target_export_name.size, 128),
              import_declaration.target_export_name.data);
        }
        source_import_target_bits[import_ordinal] =
            iree_vm_program_pack_target_bits(
                target_module_ordinal, (uint16_t)target_export.function_ordinal,
                target_mapping);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_program_select_initializer(
    iree_vm_program_t* program) {
  const iree_vm_linked_module_t* executable =
      &program->linked_modules[program->executable_module_ordinal];
  const iree_vm_module_descriptor_t* descriptor =
      executable->module->descriptor;
  const iree_string_view_t initialize_name = IREE_SV("initialize");
  iree_host_size_t low = 0;
  iree_host_size_t high = descriptor->counts.export_count;
  iree_vm_module_export_declaration_t initializer_export = {0};
  bool found = false;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    executable->module->vtable->query_export(executable->module, middle,
                                             &initializer_export);
    const int comparison = iree_string_view_compare(
        initializer_export.export_name, initialize_name);
    if (comparison < 0) {
      low = middle + 1;
    } else if (comparison > 0) {
      high = middle;
    } else {
      found = true;
      break;
    }
  }
  if (!found) return iree_ok_status();

  const uint32_t mapping =
      program->callable_mappings[executable->callable_base +
                                 initializer_export.callable_type_ordinal];
  const iree_vm_program_callable_abi_t* callable_abi =
      &program->callable_abis[iree_vm_program_callable_token(mapping) - 1];
  if (iree_vm_program_callable_abi_result_count(callable_abi) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module '%.*s' initialize must return no results",
                            (int)iree_min(descriptor->name.size, 128),
                            descriptor->name.data);
  }
  program->initializer = (iree_vm_program_initializer_t){
      .target_bits = iree_vm_program_pack_target_bits(
          program->executable_module_ordinal,
          (uint16_t)initializer_export.function_ordinal, mapping),
      .callable_abi = callable_abi,
  };
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Program lifetime and public queries
//===----------------------------------------------------------------------===//

static void iree_vm_program_destroy(iree_vm_program_t* program) {
  for (iree_host_size_t i = 0; i < program->linked_module_count; ++i) {
    iree_vm_module_release(program->linked_modules[i].module);
  }
  iree_allocator_free(program->host_allocator, program);
}

IREE_API_EXPORT iree_status_t iree_vm_program_create(
    iree_vm_program_modules_t modules, iree_allocator_t host_allocator,
    iree_vm_program_t** out_program) {
  if (!out_program) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_program is required");
  }
  *out_program = NULL;
  if (!modules.executable ||
      (modules.libraries.count != 0 && !modules.libraries.data) ||
      modules.libraries.count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program module composition is invalid");
  }

  const iree_host_size_t module_count = modules.libraries.count + 1;
  iree_host_size_t import_count = 0;
  iree_host_size_t callable_count = 0;
  iree_vm_module_callable_field_counts_t field_counts = {0};
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_vm_module_t* module =
        i == 0 ? modules.executable : modules.libraries.data[i - 1];
    if (!module || !iree_any_bit_set(module->descriptor->flags,
                                     IREE_VM_MODULE_FLAG_LINKABLE)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program contains a non-linkable module");
    }
    const iree_vm_module_callable_field_counts_t module_fields =
        module->descriptor->counts.callable_fields;
    if (!iree_host_size_checked_add(import_count,
                                    module->descriptor->counts.import_count,
                                    &import_count) ||
        !iree_host_size_checked_add(
            callable_count, module->descriptor->counts.callable_type_count,
            &callable_count) ||
        !iree_host_size_checked_add(field_counts.value_count,
                                    module_fields.value_count,
                                    &field_counts.value_count) ||
        !iree_host_size_checked_add(field_counts.ref_count,
                                    module_fields.ref_count,
                                    &field_counts.ref_count) ||
        !iree_host_size_checked_add(field_counts.function_count,
                                    module_fields.function_count,
                                    &field_counts.function_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "program declaration count overflows host size");
    }
  }
  if (callable_count > IREE_VM_PROGRAM_CALLABLE_TOKEN_MASK) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "program callable token domain is exhausted");
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t linked_modules_offset = 0;
  iree_host_size_t callable_cursors_offset = 0;
  iree_host_size_t import_target_bits_offset = 0;
  iree_host_size_t callable_mappings_offset = 0;
  iree_host_size_t callable_abis_offset = 0;
  iree_host_size_t scalar_fields_offset = 0;
  iree_host_size_t ref_fields_offset = 0;
  iree_host_size_t function_fields_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_vm_program_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(module_count, iree_vm_linked_module_t,
                                iree_alignof(iree_vm_linked_module_t),
                                &linked_modules_offset),
      IREE_STRUCT_FIELD_ALIGNED(module_count, uint32_t, iree_alignof(uint32_t),
                                &callable_cursors_offset),
      IREE_STRUCT_FIELD_ALIGNED(import_count, uint64_t, iree_alignof(uint64_t),
                                &import_target_bits_offset),
      IREE_STRUCT_FIELD_ALIGNED(callable_count, uint32_t,
                                iree_alignof(uint32_t),
                                &callable_mappings_offset),
      IREE_STRUCT_FIELD_ALIGNED(callable_count, iree_vm_program_callable_abi_t,
                                iree_alignof(iree_vm_program_callable_abi_t),
                                &callable_abis_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          field_counts.value_count, iree_vm_program_scalar_field_abi_t,
          iree_alignof(iree_vm_program_scalar_field_abi_t),
          &scalar_fields_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          field_counts.ref_count, iree_vm_program_ref_field_abi_t,
          iree_alignof(iree_vm_program_ref_field_abi_t), &ref_fields_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          field_counts.function_count, iree_vm_program_function_field_abi_t,
          iree_alignof(iree_vm_program_function_field_abi_t),
          &function_fields_offset)));

  iree_vm_program_t* program = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&program));
  memset(program, 0, sizeof(*program));
  program->host_allocator = host_allocator;
  program->linked_modules =
      (iree_vm_linked_module_t*)((uint8_t*)program + linked_modules_offset);
  uint32_t* callable_cursors =
      (uint32_t*)((uint8_t*)program + callable_cursors_offset);
  memset(callable_cursors, 0, module_count * sizeof(*callable_cursors));
  uint64_t* import_target_bits =
      import_count ? (uint64_t*)((uint8_t*)program + import_target_bits_offset)
                   : NULL;
  if (import_count) {
    memset(import_target_bits, 0, import_count * sizeof(*import_target_bits));
  }
  program->callable_mappings =
      callable_count ? (uint32_t*)((uint8_t*)program + callable_mappings_offset)
                     : NULL;
  program->callable_abis =
      callable_count ? (iree_vm_program_callable_abi_t*)((uint8_t*)program +
                                                         callable_abis_offset)
                     : NULL;
  program->callable_mapping_count = (uint32_t)callable_count;
  const iree_vm_program_field_storage_t field_storage = {
      .scalar_fields =
          field_counts.value_count
              ? (iree_vm_program_scalar_field_abi_t*)((uint8_t*)program +
                                                      scalar_fields_offset)
              : NULL,
      .ref_fields = field_counts.ref_count
                        ? (iree_vm_program_ref_field_abi_t*)((uint8_t*)program +
                                                             ref_fields_offset)
                        : NULL,
      .function_fields =
          field_counts.function_count
              ? (iree_vm_program_function_field_abi_t*)((uint8_t*)program +
                                                        function_fields_offset)
              : NULL,
  };

  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_vm_module_t* module =
        i == 0 ? modules.executable : modules.libraries.data[i - 1];
    iree_vm_module_retain(module);
    program->linked_modules[i] = (iree_vm_linked_module_t){
        .module = module,
        .import_target_bits = NULL,
        .callable_base = 0,
        .process_storage_offset = UINT32_MAX,
    };
  }
  program->linked_module_count = (uint32_t)module_count;
  iree_vm_program_sort_linked_modules(program->linked_modules, module_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < module_count && iree_status_is_ok(status);
       ++i) {
    iree_vm_module_t* module = program->linked_modules[i].module;
    if (i != 0 && iree_string_view_equal(
                      program->linked_modules[i - 1].module->descriptor->name,
                      module->descriptor->name)) {
      status =
          iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                           "program contains duplicate module name '%.*s'",
                           (int)iree_min(module->descriptor->name.size, 128),
                           module->descriptor->name.data);
    } else if (module == modules.executable) {
      program->executable_module_ordinal = (uint16_t)i;
    }
  }

  iree_host_size_t import_base = 0;
  iree_host_size_t callable_base = 0;
  for (iree_host_size_t i = 0; i < module_count && iree_status_is_ok(status);
       ++i) {
    iree_vm_linked_module_t* linked_module = &program->linked_modules[i];
    const iree_host_size_t local_import_count =
        linked_module->module->descriptor->counts.import_count;
    linked_module->import_target_bits =
        local_import_count ? &import_target_bits[import_base] : NULL;
    linked_module->callable_base = (uint32_t)callable_base;
    import_base += local_import_count;
    callable_base +=
        linked_module->module->descriptor->counts.callable_type_count;
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_program_plan_process_storage(program);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_program_link_callable_types(program, callable_cursors,
                                        field_storage);
    status = iree_vm_program_resolve_imports(program, import_target_bits);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_program_select_initializer(program);
  }

  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&program->ref_count);
    *out_program = program;
  } else {
    iree_vm_program_destroy(program);
  }
  return status;
}

IREE_API_EXPORT void iree_vm_program_retain(iree_vm_program_t* program) {
  if (program) {
    iree_atomic_ref_count_inc(&program->ref_count);
  }
}

IREE_API_EXPORT void iree_vm_program_release(iree_vm_program_t* program) {
  if (program && iree_atomic_ref_count_dec(&program->ref_count) == 1) {
    iree_vm_program_destroy(program);
  }
}

IREE_API_EXPORT iree_status_t iree_vm_function_ref_from_export(
    const iree_vm_program_t* program, iree_vm_export_t export_value,
    iree_vm_function_ref_t* out_function_ref) {
  if (!program || !export_value.module || !out_function_ref) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program, export module, and out_function_ref are required");
  }
  iree_host_size_t module_ordinal = 0;
  const iree_vm_linked_module_t* linked_module =
      iree_vm_program_find_linked_module(program, export_value.module,
                                         &module_ordinal);
  if (!linked_module ||
      export_value.ordinal >=
          export_value.module->descriptor->counts.export_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "export does not belong to the program");
  }

  iree_vm_module_export_declaration_t export_declaration = {0};
  export_value.module->vtable->query_export(
      export_value.module, export_value.ordinal, &export_declaration);
  const uint32_t mapping =
      program->callable_mappings[linked_module->callable_base +
                                 export_declaration.callable_type_ordinal];
  const iree_vm_function_ref_t function_ref = {
      (uint64_t)(uintptr_t)program,
      iree_vm_program_pack_target_bits(
          (uint16_t)module_ordinal,
          (uint16_t)export_declaration.function_ordinal, mapping),
  };
  *out_function_ref = function_ref;
  return iree_ok_status();
}
