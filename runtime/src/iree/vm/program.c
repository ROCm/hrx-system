// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/program.h"

#include <string.h>

#include "iree/vm/program_storage.h"

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

const iree_vm_linked_module_t* iree_vm_program_find_linked_module(
    const iree_vm_program_t* program, const iree_vm_module_t* module,
    iree_host_size_t* out_ordinal) {
  iree_host_size_t low = 0;
  iree_host_size_t high = program->linked_module_count;
  const iree_string_view_t name = module->descriptor->name;
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
  if (low >= program->linked_module_count ||
      program->linked_modules[low].module != module) {
    return NULL;
  }
  *out_ordinal = low;
  return &program->linked_modules[low];
}

const iree_vm_linked_module_t* iree_vm_program_lookup_linked_module(
    const iree_vm_program_t* program, iree_string_view_t name,
    iree_host_size_t* out_ordinal) {
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
  if (low >= program->linked_module_count ||
      !iree_string_view_equal(
          program->linked_modules[low].module->descriptor->name, name)) {
    return NULL;
  }
  *out_ordinal = low;
  return &program->linked_modules[low];
}

//===----------------------------------------------------------------------===//
// Structural callable interning
//===----------------------------------------------------------------------===//

static bool iree_vm_program_signature_types_equal(
    const iree_vm_program_t* program, const iree_vm_linked_module_t* lhs_module,
    iree_vm_module_signature_type_span_t lhs,
    const iree_vm_linked_module_t* rhs_module,
    iree_vm_module_signature_type_span_t rhs) {
  if (lhs.count != rhs.count) return false;
  for (iree_host_size_t i = 0; i < lhs.count; ++i) {
    const iree_vm_module_signature_type_t lhs_type = lhs.data[i];
    const iree_vm_module_signature_type_t rhs_type = rhs.data[i];
    if (lhs_type.kind != rhs_type.kind) return false;
    if (lhs_type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
      const iree_vm_ref_type_t lhs_ref_type =
          lhs_module->module->descriptor->ref_types.data[lhs_type.type_ordinal];
      const iree_vm_ref_type_t rhs_ref_type =
          rhs_module->module->descriptor->ref_types.data[rhs_type.type_ordinal];
      if (lhs_ref_type != rhs_ref_type) return false;
    } else if (lhs_type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION) {
      const uint32_t lhs_mapping =
          program->callables[lhs_module->callable_base + lhs_type.type_ordinal]
              .mapping;
      const uint32_t rhs_mapping =
          program->callables[rhs_module->callable_base + rhs_type.type_ordinal]
              .mapping;
      if (lhs_mapping != rhs_mapping) return false;
    }
  }
  return true;
}

static bool iree_vm_program_signatures_equal(
    const iree_vm_program_t* program, const iree_vm_linked_module_t* lhs_module,
    iree_vm_module_signature_t lhs, const iree_vm_linked_module_t* rhs_module,
    iree_vm_module_signature_t rhs) {
  return iree_vm_program_signature_types_equal(
             program, lhs_module, lhs.arguments, rhs_module, rhs.arguments) &&
         iree_vm_program_signature_types_equal(program, lhs_module, lhs.results,
                                               rhs_module, rhs.results);
}

static iree_vm_program_bank_counts_t iree_vm_program_count_signature_banks(
    iree_vm_module_signature_type_span_t types) {
  iree_vm_program_bank_counts_t counts = {0};
  for (iree_host_size_t i = 0; i < types.count; ++i) {
    const iree_vm_module_signature_type_t type = types.data[i];
    if (type.kind > IREE_VM_SCALAR_TYPE_INVALID &&
        type.kind <= IREE_VM_SCALAR_TYPE_F64) {
      ++counts.value_count;
    } else if (type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
      ++counts.ref_count;
    } else {
      ++counts.function_count;
    }
  }
  return counts;
}

static void iree_vm_program_intern_callable_types(iree_vm_program_t* program) {
  iree_host_size_t flat_ordinal = 0;
  for (iree_host_size_t module_i = 0; module_i < program->linked_module_count;
       ++module_i) {
    const iree_vm_linked_module_t* linked_module =
        &program->linked_modules[module_i];
    const iree_host_size_t callable_count =
        linked_module->module->descriptor->counts.callable_type_count;
    for (iree_host_size_t callable_i = 0; callable_i < callable_count;
         ++callable_i, ++flat_ordinal) {
      iree_vm_module_callable_type_declaration_t callable_type = {0};
      linked_module->module->vtable->query_callable_type(
          linked_module->module, callable_i, &callable_type);

      iree_vm_program_callable_t* callable = &program->callables[flat_ordinal];
      callable->argument_types = callable_type.signature.arguments.data;
      callable->result_types = callable_type.signature.results.data;
      callable->signature_module = linked_module;
      callable->argument_counts = iree_vm_program_count_signature_banks(
          callable_type.signature.arguments);
      callable->result_counts = iree_vm_program_count_signature_banks(
          callable_type.signature.results);
      if (callable_type.signature.results.count != 0 &&
          callable->result_counts.value_count ==
              callable_type.signature.results.count) {
        const iree_vm_scalar_type_t result_scalar_type =
            callable_type.signature.results.data[0].kind;
        callable->uniform_result_scalar_type = result_scalar_type;
        for (iree_host_size_t result_i = 1;
             result_i < callable_type.signature.results.count; ++result_i) {
          if (callable_type.signature.results.data[result_i].kind !=
              result_scalar_type) {
            callable->uniform_result_scalar_type = IREE_VM_SCALAR_TYPE_INVALID;
            break;
          }
        }
      }

      uint32_t token = 0;
      for (iree_host_size_t representative_i = 0;
           representative_i < flat_ordinal && token == 0; ++representative_i) {
        const iree_vm_program_callable_t* representative =
            &program->callables[representative_i];
        if (iree_vm_program_callable_token(representative->mapping) !=
            representative_i + 1) {
          continue;
        }
        if (iree_vm_program_signatures_equal(
                program, linked_module, callable_type.signature,
                representative->signature_module,
                iree_vm_program_callable_signature(representative))) {
          token = iree_vm_program_callable_token(representative->mapping);
        }
      }
      if (token == 0) token = (uint32_t)flat_ordinal + 1;
      callable->mapping =
          token | (iree_any_bit_set(callable_type.flags,
                                    IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD)
                       ? IREE_VM_PROGRAM_CALLABLE_MAY_YIELD
                       : 0);
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
    const iree_host_size_t storage_size =
        program->linked_modules[i].module->descriptor->process_storage_size;
    if (storage_size == 0) {
      program->linked_modules[i].process_storage_offset = UINT32_MAX;
      continue;
    }
    iree_host_size_t aligned_offset = 0;
    if (!iree_host_size_checked_align(total_size, iree_max_align_t,
                                      &aligned_offset) ||
        aligned_offset >= UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "program process-state offset exceeds 32 bits");
    }
    program->linked_modules[i].process_storage_offset =
        (uint32_t)aligned_offset;
    if (!iree_host_size_checked_add(aligned_offset, storage_size,
                                    &total_size) ||
        total_size > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "program process-state size exceeds 32 bits");
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
        source_descriptor->counts.import_count != 0
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
                ->callables[source_module->callable_base +
                            import_declaration.callable_type_ordinal]
                .mapping;
        const uint32_t target_mapping =
            program
                ->callables[target_module->callable_base +
                            target_export.callable_type_ordinal]
                .mapping;
        if (iree_vm_program_callable_token(source_mapping) !=
            iree_vm_program_callable_token(target_mapping)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "module '%.*s' import '%.*s.%.*s' has an incompatible "
              "callable type",
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

  iree_vm_module_callable_type_declaration_t callable_type = {0};
  executable->module->vtable->query_callable_type(
      executable->module, initializer_export.callable_type_ordinal,
      &callable_type);
  if (callable_type.signature.results.count != 0 ||
      callable_type.signature.arguments.count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module '%.*s' initialize must return no results and fit the "
        "physical call ABI",
        (int)iree_min(descriptor->name.size, 128), descriptor->name.data);
  }

  const iree_vm_program_callable_t* local_callable =
      &program->callables[executable->callable_base +
                          initializer_export.callable_type_ordinal];
  const uint32_t mapping = local_callable->mapping;
  const iree_vm_program_callable_t* callable = iree_vm_program_resolve_callable(
      program, iree_vm_program_callable_token(mapping));
  if (!callable) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "initializer callable token is invalid");
  }
  const iree_vm_program_initializer_t initializer = {
      iree_vm_program_pack_target_bits(
          program->executable_module_ordinal,
          (uint16_t)initializer_export.function_ordinal, mapping),
      callable,
  };
  program->initializer = initializer;
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
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_vm_module_t* module =
        i == 0 ? modules.executable : modules.libraries.data[i - 1];
    if (!module || !module->descriptor ||
        !iree_any_bit_set(module->descriptor->flags,
                          IREE_VM_MODULE_FLAG_LINKABLE)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program contains a non-linkable module");
    }
    if (!iree_host_size_checked_add(import_count,
                                    module->descriptor->counts.import_count,
                                    &import_count) ||
        !iree_host_size_checked_add(
            callable_count, module->descriptor->counts.callable_type_count,
            &callable_count)) {
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
  iree_host_size_t import_target_bits_offset = 0;
  iree_host_size_t callables_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_vm_program_t), &total_size,
      IREE_STRUCT_FIELD(module_count, iree_vm_linked_module_t,
                        &linked_modules_offset),
      IREE_STRUCT_FIELD(import_count, uint64_t, &import_target_bits_offset),
      IREE_STRUCT_FIELD(callable_count, iree_vm_program_callable_t,
                        &callables_offset)));

  iree_vm_program_t* program = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&program));
  memset(program, 0, total_size);
  program->host_allocator = host_allocator;
  program->linked_modules =
      (iree_vm_linked_module_t*)((uint8_t*)program + linked_modules_offset);
  program->linked_module_count = (uint32_t)module_count;
  uint64_t* import_target_bits =
      import_count != 0
          ? (uint64_t*)((uint8_t*)program + import_target_bits_offset)
          : NULL;
  program->callables =
      callable_count != 0
          ? (iree_vm_program_callable_t*)((uint8_t*)program + callables_offset)
          : NULL;
  program->callable_count = (uint32_t)callable_count;

  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_vm_module_t* module =
        i == 0 ? modules.executable : modules.libraries.data[i - 1];
    iree_vm_module_retain(module);
    program->linked_modules[i].module = module;
  }
  iree_vm_program_sort_linked_modules(program->linked_modules, module_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < module_count && iree_status_is_ok(status);
       ++i) {
    if (i != 0 && iree_string_view_equal(
                      program->linked_modules[i - 1].module->descriptor->name,
                      program->linked_modules[i].module->descriptor->name)) {
      const iree_string_view_t duplicate_name =
          program->linked_modules[i].module->descriptor->name;
      status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "program contains duplicate module name '%.*s'",
                                (int)iree_min(duplicate_name.size, 128),
                                duplicate_name.data);
    } else if (program->linked_modules[i].module == modules.executable) {
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
        local_import_count != 0 ? &import_target_bits[import_base] : NULL;
    linked_module->callable_base = (uint32_t)callable_base;
    import_base += local_import_count;
    callable_base +=
        linked_module->module->descriptor->counts.callable_type_count;
  }

  if (iree_status_is_ok(status)) {
    status = iree_vm_program_plan_process_storage(program);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_program_intern_callable_types(program);
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
  if (!program) return;
  iree_atomic_ref_count_inc(&program->ref_count);
}

IREE_API_EXPORT void iree_vm_program_release(iree_vm_program_t* program) {
  if (!program) return;
  if (iree_atomic_ref_count_dec(&program->ref_count) == 1) {
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
      program
          ->callables[linked_module->callable_base +
                      export_declaration.callable_type_ordinal]
          .mapping;
  const iree_vm_function_ref_t function_ref = {
      (uint64_t)(uintptr_t)program,
      iree_vm_program_pack_target_bits(
          (uint16_t)module_ordinal,
          (uint16_t)export_declaration.function_ordinal, mapping),
  };
  *out_function_ref = function_ref;
  return iree_ok_status();
}
