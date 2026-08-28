// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_types.h"

#include <stdlib.h>

#include "loom/codegen/low/function.h"
#include "loom/ops/func/reference.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/emit/vm/module_layout.h"
#include "loom/target/registers.h"

typedef struct loom_vm_module_ref_type_build_t {
  // Borrowed VM ref-type namespace.
  iree_string_view_t namespace_name;
  // Borrowed VM ref-type name within |namespace_name|.
  iree_string_view_t type_name;
  // Canonical flat ref-type ordinal assigned after sorting.
  uint16_t ordinal;
} loom_vm_module_ref_type_build_t;

typedef struct loom_vm_module_callable_type_build_t
    loom_vm_module_callable_type_build_t;

typedef struct loom_vm_module_signature_field_build_t {
  // Architectural scalar, REF, or FUNCTION signature kind.
  uint16_t kind;
  // Kind-specific type identity, unused for scalar fields.
  union {
    // Canonical ref-type build record for REF fields.
    const loom_vm_module_ref_type_build_t* ref_type;
    // Nested callable build record for FUNCTION fields.
    const loom_vm_module_callable_type_build_t* callable_type;
  } type;
} loom_vm_module_signature_field_build_t;

typedef uint8_t loom_vm_module_callable_state_t;
enum loom_vm_module_callable_state_e {
  // Candidate has not yet had its signature traversed.
  LOOM_VM_MODULE_CALLABLE_STATE_NEW = 0,
  // Candidate is on the active traversal path.
  LOOM_VM_MODULE_CALLABLE_STATE_VISITING = 1,
  // Candidate and all nested callable fields are complete.
  LOOM_VM_MODULE_CALLABLE_STATE_COMPLETE = 2,
};

struct loom_vm_module_callable_type_build_t {
  // Structural Loom function signature represented by this candidate.
  loom_type_t signature;
  // Callable permission flags encoded in the module row.
  uint16_t flags;
  // Maximum nested callable depth.
  uint16_t nesting_depth;
  // Canonical callable ordinal shared by structurally equal candidates.
  uint16_t ordinal;
  // Traversal state used to reject recursive callable types.
  loom_vm_module_callable_state_t state;
  // Source-ordered argument count.
  uint16_t argument_count;
  // Source-ordered result count.
  uint16_t result_count;
  // Arena-owned argument-then-result field descriptions.
  loom_vm_module_signature_field_build_t* fields;
  // Independently counted physical banks for the signature.
  iree_vm_bytecode_v0_signature_row_t signature_row;
};

typedef struct loom_vm_module_type_build_t {
  // Module supplying structural types and registered type names.
  loom_module_t* module;
  // Arena owning all build and final table storage.
  iree_arena_allocator_t* arena;
  // Stable storage for raw callable candidates.
  loom_vm_module_callable_type_build_t* callable_storage;
  // Sortable pointers into |callable_storage|.
  loom_vm_module_callable_type_build_t** callable_order;
  // Maximum number of entries in |callable_storage| and |callable_order|.
  iree_host_size_t callable_capacity;
  // Number of populated raw callable candidates.
  iree_host_size_t callable_count;
  // Stable storage for unique ref types.
  loom_vm_module_ref_type_build_t* ref_type_storage;
  // Sortable pointers into |ref_type_storage|.
  loom_vm_module_ref_type_build_t** ref_type_order;
  // Maximum number of entries in |ref_type_storage| and |ref_type_order|.
  iree_host_size_t ref_type_capacity;
  // Number of populated unique ref types.
  iree_host_size_t ref_type_count;
} loom_vm_module_type_build_t;

typedef struct loom_vm_module_type_capacities_t {
  // Maximum root and nested callable candidates present in the module.
  iree_host_size_t callable_count;
  // Maximum managed-reference types present in the module type table.
  iree_host_size_t ref_type_count;
} loom_vm_module_type_capacities_t;

typedef uint8_t loom_vm_module_signature_side_t;
enum loom_vm_module_signature_side_e {
  // Signature argument fields.
  LOOM_VM_MODULE_SIGNATURE_SIDE_ARGUMENT = 0,
  // Signature result fields.
  LOOM_VM_MODULE_SIGNATURE_SIDE_RESULT = 1,
};

static int loom_vm_module_compare_u16(uint16_t lhs, uint16_t rhs) {
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

static iree_status_t loom_vm_module_type_capacities_count(
    const loom_vm_module_layout_t* layout,
    loom_vm_module_type_capacities_t* out_capacities) {
  *out_capacities = (loom_vm_module_type_capacities_t){0};
  if (!iree_host_size_checked_add(layout->function_count,
                                  layout->import_declaration_count,
                                  &out_capacities->callable_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM callable candidate count exceeds host size");
  }
  for (iree_host_size_t i = 0; i < layout->module->types.count; ++i) {
    const loom_type_t type = layout->module->types.entries[i];
    if (loom_func_ref_type_isa(type)) {
      if (out_capacities->callable_count == IREE_HOST_SIZE_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "VM callable candidate count exceeds host size");
      }
      ++out_capacities->callable_count;
    } else if (loom_type_is_buffer(type)) {
      ++out_capacities->ref_type_count;
    } else if (loom_type_is_dialect(type)) {
      const loom_type_descriptor_t* descriptor =
          loom_type_registry_resolve(layout->module, type);
      if (descriptor != NULL && descriptor->semantics.semantic ==
                                    LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE) {
        ++out_capacities->ref_type_count;
      }
    }
  }
  return iree_ok_status();
}

static int loom_vm_module_compare_string_views(const void* lhs_ptr,
                                               const void* rhs_ptr) {
  const iree_string_view_t* lhs = (const iree_string_view_t*)lhs_ptr;
  const iree_string_view_t* rhs = (const iree_string_view_t*)rhs_ptr;
  return iree_string_view_compare(*lhs, *rhs);
}

static int loom_vm_module_compare_ref_types(const void* lhs_ptr,
                                            const void* rhs_ptr) {
  const loom_vm_module_ref_type_build_t* lhs =
      *(loom_vm_module_ref_type_build_t* const*)lhs_ptr;
  const loom_vm_module_ref_type_build_t* rhs =
      *(loom_vm_module_ref_type_build_t* const*)rhs_ptr;
  int comparison =
      iree_string_view_compare(lhs->namespace_name, rhs->namespace_name);
  return comparison != 0
             ? comparison
             : iree_string_view_compare(lhs->type_name, rhs->type_name);
}

static int loom_vm_module_compare_callable_depths(const void* lhs_ptr,
                                                  const void* rhs_ptr) {
  const loom_vm_module_callable_type_build_t* lhs =
      *(loom_vm_module_callable_type_build_t* const*)lhs_ptr;
  const loom_vm_module_callable_type_build_t* rhs =
      *(loom_vm_module_callable_type_build_t* const*)rhs_ptr;
  return loom_vm_module_compare_u16(lhs->nesting_depth, rhs->nesting_depth);
}

static int loom_vm_module_compare_signature_fields(
    const loom_vm_module_signature_field_build_t* lhs,
    const loom_vm_module_signature_field_build_t* rhs, uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    int comparison = loom_vm_module_compare_u16(lhs[i].kind, rhs[i].kind);
    if (comparison != 0) return comparison;
    if (lhs[i].kind == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
      const loom_vm_module_ref_type_build_t* lhs_type = lhs[i].type.ref_type;
      const loom_vm_module_ref_type_build_t* rhs_type = rhs[i].type.ref_type;
      comparison =
          loom_vm_module_compare_u16(lhs_type->ordinal, rhs_type->ordinal);
    } else if (lhs[i].kind == IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
      const loom_vm_module_callable_type_build_t* lhs_type =
          lhs[i].type.callable_type;
      const loom_vm_module_callable_type_build_t* rhs_type =
          rhs[i].type.callable_type;
      comparison =
          loom_vm_module_compare_u16(lhs_type->ordinal, rhs_type->ordinal);
    }
    if (comparison != 0) return comparison;
  }
  return 0;
}

static int loom_vm_module_compare_callable_structures(const void* lhs_ptr,
                                                      const void* rhs_ptr) {
  const loom_vm_module_callable_type_build_t* lhs =
      *(loom_vm_module_callable_type_build_t* const*)lhs_ptr;
  const loom_vm_module_callable_type_build_t* rhs =
      *(loom_vm_module_callable_type_build_t* const*)rhs_ptr;
  int comparison =
      loom_vm_module_compare_u16(lhs->argument_count, rhs->argument_count);
  if (comparison != 0) return comparison;
  comparison = loom_vm_module_compare_signature_fields(lhs->fields, rhs->fields,
                                                       lhs->argument_count);
  if (comparison != 0) return comparison;
  comparison = loom_vm_module_compare_u16(lhs->result_count, rhs->result_count);
  if (comparison != 0) return comparison;
  const loom_vm_module_signature_field_build_t* lhs_results =
      lhs->fields != NULL ? lhs->fields + lhs->argument_count : NULL;
  const loom_vm_module_signature_field_build_t* rhs_results =
      rhs->fields != NULL ? rhs->fields + rhs->argument_count : NULL;
  comparison = loom_vm_module_compare_signature_fields(lhs_results, rhs_results,
                                                       lhs->result_count);
  return comparison != 0 ? comparison
                         : loom_vm_module_compare_u16(lhs->flags, rhs->flags);
}

static iree_status_t loom_vm_module_signature_scalar_kind(loom_type_t type,
                                                          uint16_t* out_kind) {
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I64:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_I64;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_I8;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I16:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_I16;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I32:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_I32;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F8E4M3:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_F8E4M3FN;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_F8E5M2;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F16:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_F16;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_BF16:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_BF16;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F32:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_F32;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F64:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_F64;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "scalar type is not supported by the VM ABI");
  }
}

static iree_status_t loom_vm_module_resolve_logical_type(
    const loom_module_t* module, loom_type_t type, loom_type_t* out_type,
    loom_vm_call_abi_bank_t* out_bank) {
  *out_type = loom_type_none();
  *out_bank = LOOM_VM_CALL_ABI_BANK_NONE;
  if (loom_low_type_is_register(type)) {
    IREE_RETURN_IF_ERROR(
        loom_vm_call_abi_classify_type(module, type, out_bank));
    const loom_type_t* value_type = loom_type_register_value_type(type);
    *out_type = *value_type;
    return iree_ok_status();
  }
  if (!loom_vm_call_abi_try_classify_logical_type(module, type, out_bank)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM signature contains an unsupported type");
  }
  *out_type = type;
  return iree_ok_status();
}

static iree_status_t loom_vm_module_ref_type_find_or_add(
    loom_vm_module_type_build_t* build, loom_type_t type,
    loom_vm_module_ref_type_build_t** out_ref_type) {
  *out_ref_type = NULL;
  iree_string_view_t namespace_name;
  iree_string_view_t type_name;
  if (loom_type_is_buffer(type)) {
    namespace_name = IREE_SV("vm");
    type_name = IREE_SV("buffer");
  } else {
    const loom_type_descriptor_t* descriptor =
        loom_type_registry_resolve(build->module, type);
    if (descriptor == NULL || descriptor->semantics.semantic !=
                                  LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM ref field has no managed-reference type");
    }
    const iree_string_view_t full_name = loom_bstring_view(descriptor->name);
    const iree_host_size_t dot = iree_string_view_find_last_of(
        full_name, IREE_SV("."), IREE_STRING_VIEW_NPOS);
    if (dot == IREE_STRING_VIEW_NPOS || dot == 0 || dot + 1 == full_name.size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM ref type names must have nonempty namespace and local parts");
    }
    namespace_name = iree_string_view_substr(full_name, 0, dot);
    type_name = iree_string_view_substr(full_name, dot + 1, IREE_HOST_SIZE_MAX);
  }
  for (iree_host_size_t i = 0; i < build->ref_type_count; ++i) {
    loom_vm_module_ref_type_build_t* existing = &build->ref_type_storage[i];
    if (iree_string_view_equal(existing->namespace_name, namespace_name) &&
        iree_string_view_equal(existing->type_name, type_name)) {
      *out_ref_type = existing;
      return iree_ok_status();
    }
  }
  if (build->ref_type_count == build->ref_type_capacity) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "VM ref-type graph exceeds the module type-table bound");
  }
  loom_vm_module_ref_type_build_t* ref_type =
      &build->ref_type_storage[build->ref_type_count];
  *ref_type = (loom_vm_module_ref_type_build_t){
      .namespace_name = namespace_name,
      .type_name = type_name,
  };
  build->ref_type_order[build->ref_type_count++] = ref_type;
  *out_ref_type = ref_type;
  return iree_ok_status();
}

static iree_status_t loom_vm_module_callable_collect(
    loom_vm_module_type_build_t* build, loom_type_t signature, uint16_t flags,
    loom_vm_module_callable_type_build_t** out_callable);

static loom_vm_module_callable_type_build_t* loom_vm_module_callable_find(
    const loom_vm_module_type_build_t* build, loom_type_t signature,
    uint16_t flags) {
  for (iree_host_size_t i = 0; i < build->callable_count; ++i) {
    loom_vm_module_callable_type_build_t* callable =
        &build->callable_storage[i];
    if (callable->flags == flags &&
        loom_type_equal(callable->signature, signature)) {
      return callable;
    }
  }
  return NULL;
}

static uint16_t* loom_vm_module_signature_bank_count(
    loom_vm_module_callable_type_build_t* callable,
    loom_vm_module_signature_side_t side, loom_vm_call_abi_bank_t bank) {
  iree_vm_bytecode_v0_signature_row_t* row = &callable->signature_row;
  if (side == LOOM_VM_MODULE_SIGNATURE_SIDE_ARGUMENT) {
    switch (bank) {
      case LOOM_VM_CALL_ABI_BANK_VALUE:
        return &row->argument_value_count_u16;
      case LOOM_VM_CALL_ABI_BANK_REF:
        return &row->argument_ref_count_u16;
      case LOOM_VM_CALL_ABI_BANK_FUNCTION:
        return &row->argument_function_count_u16;
      default:
        break;
    }
  } else {
    switch (bank) {
      case LOOM_VM_CALL_ABI_BANK_VALUE:
        return &row->result_value_count_u16;
      case LOOM_VM_CALL_ABI_BANK_REF:
        return &row->result_ref_count_u16;
      case LOOM_VM_CALL_ABI_BANK_FUNCTION:
        return &row->result_function_count_u16;
      default:
        break;
    }
  }
  IREE_ASSERT_UNREACHABLE("valid VM signature bank and side");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_vm_module_signature_field_build(
    loom_vm_module_type_build_t* build,
    loom_vm_module_callable_type_build_t* callable,
    loom_vm_module_signature_side_t side, loom_type_t type,
    loom_vm_module_signature_field_build_t* out_field) {
  *out_field = (loom_vm_module_signature_field_build_t){0};
  loom_type_t logical_type = loom_type_none();
  loom_vm_call_abi_bank_t bank = LOOM_VM_CALL_ABI_BANK_NONE;
  IREE_RETURN_IF_ERROR(loom_vm_module_resolve_logical_type(
      build->module, type, &logical_type, &bank));
  ++*loom_vm_module_signature_bank_count(callable, side, bank);

  switch (bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE:
      return loom_vm_module_signature_scalar_kind(logical_type,
                                                  &out_field->kind);
    case LOOM_VM_CALL_ABI_BANK_REF: {
      loom_vm_module_ref_type_build_t* ref_type = NULL;
      IREE_RETURN_IF_ERROR(
          loom_vm_module_ref_type_find_or_add(build, logical_type, &ref_type));
      out_field->kind = IREE_VM_BYTECODE_SIGNATURE_KIND_REF;
      out_field->type.ref_type = ref_type;
      return iree_ok_status();
    }
    case LOOM_VM_CALL_ABI_BANK_FUNCTION: {
      const loom_type_t nested_signature =
          loom_func_ref_resolve_signature(build->module, logical_type);
      uint16_t nested_flags = 0;
      if (loom_func_ref_type_has_yieldability(logical_type)) {
        nested_flags = IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD;
      }
      loom_vm_module_callable_type_build_t* nested_callable = NULL;
      IREE_RETURN_IF_ERROR(loom_vm_module_callable_collect(
          build, nested_signature, nested_flags, &nested_callable));
      if (nested_callable->nesting_depth == UINT16_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "VM callable nesting depth exceeds u16");
      }
      callable->nesting_depth =
          iree_max(callable->nesting_depth,
                   (uint16_t)(nested_callable->nesting_depth + 1));
      out_field->kind = IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION;
      out_field->type.callable_type = nested_callable;
      return iree_ok_status();
    }
    default:
      IREE_ASSERT_UNREACHABLE("valid VM signature bank");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_vm_module_callable_collect(
    loom_vm_module_type_build_t* build, loom_type_t signature, uint16_t flags,
    loom_vm_module_callable_type_build_t** out_callable) {
  *out_callable = NULL;
  if (!loom_type_is_function(signature)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM callable type requires a function signature");
  }
  if ((flags & ~IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM callable type has unsupported flags");
  }

  loom_vm_module_callable_type_build_t* callable =
      loom_vm_module_callable_find(build, signature, flags);
  if (callable == NULL) {
    if (build->callable_count == build->callable_capacity) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "VM callable graph exceeds the module type-table bound");
    }
    callable = &build->callable_storage[build->callable_count];
    *callable = (loom_vm_module_callable_type_build_t){
        .signature = signature,
        .flags = flags,
    };
    build->callable_order[build->callable_count++] = callable;
  }
  *out_callable = callable;
  if (callable->state == LOOM_VM_MODULE_CALLABLE_STATE_COMPLETE) {
    return iree_ok_status();
  }
  if (callable->state == LOOM_VM_MODULE_CALLABLE_STATE_VISITING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recursive VM callable types are not supported");
  }

  callable->state = LOOM_VM_MODULE_CALLABLE_STATE_VISITING;
  callable->argument_count = loom_type_func_arg_count(signature);
  callable->result_count = loom_type_func_result_count(signature);
  const iree_host_size_t field_count =
      (iree_host_size_t)callable->argument_count + callable->result_count;
  if (field_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(build->arena, field_count,
                                                   sizeof(*callable->fields),
                                                   (void**)&callable->fields));
  }

  const loom_type_t* argument_types = loom_type_func_arg_types(signature);
  for (uint16_t i = 0; i < callable->argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vm_module_signature_field_build(
        build, callable, LOOM_VM_MODULE_SIGNATURE_SIDE_ARGUMENT,
        argument_types[i], &callable->fields[i]));
  }
  const loom_type_t* result_types = loom_type_func_result_types(signature);
  for (uint16_t i = 0; i < callable->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vm_module_signature_field_build(
        build, callable, LOOM_VM_MODULE_SIGNATURE_SIDE_RESULT, result_types[i],
        &callable->fields[callable->argument_count + i]));
  }
  callable->state = LOOM_VM_MODULE_CALLABLE_STATE_COMPLETE;
  return iree_ok_status();
}

static uint16_t loom_vm_module_function_ref_flags(loom_type_t type) {
  return loom_func_ref_type_has_yieldability(type)
             ? IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD
             : 0;
}

static iree_status_t loom_vm_module_collect_function_ref_type(
    loom_vm_module_type_build_t* build, loom_type_t type) {
  if (loom_low_type_is_register(type)) {
    const loom_type_t* logical_type = loom_type_register_value_type(type);
    if (logical_type == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM indirect-call target register has no logical value type");
    }
    type = *logical_type;
  }
  if (!loom_func_ref_type_isa(type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM indirect-call target is not a func.ref");
  }
  loom_vm_module_callable_type_build_t* callable = NULL;
  return loom_vm_module_callable_collect(
      build, loom_func_ref_resolve_signature(build->module, type),
      loom_vm_module_function_ref_flags(type), &callable);
}

static iree_status_t loom_vm_module_collect_indirect_call_types(
    loom_vm_module_type_build_t* build, const loom_vm_module_layout_t* layout) {
  for (iree_host_size_t function_i = 0; function_i < layout->function_count;
       ++function_i) {
    const loom_region_t* body =
        loom_low_function_const_body(layout->functions[function_i].function_op);
    if (body == NULL) continue;
    for (uint16_t block_i = 0; block_i < body->block_count; ++block_i) {
      const loom_op_t* op = NULL;
      loom_block_for_each_op(body->blocks[block_i], op) {
        if (!loom_low_func_call_indirect_isa(op)) continue;
        IREE_RETURN_IF_ERROR(loom_vm_module_collect_function_ref_type(
            build, loom_module_value_type(
                       build->module, loom_low_func_call_indirect_target(op))));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_collect_resource_types(
    loom_vm_module_type_build_t* build, loom_vm_module_layout_t* layout,
    loom_vm_module_ref_type_build_t** ref_global_types,
    loom_vm_module_callable_type_build_t** function_global_types) {
  for (uint32_t i = 0; i < layout->resources.ref_global_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vm_module_ref_type_find_or_add(
        build, layout->resources.ref_global_types[i], &ref_global_types[i]));
  }
  for (uint32_t i = 0; i < layout->resources.function_global_count; ++i) {
    const loom_type_t type = layout->resources.function_global_types[i];
    IREE_RETURN_IF_ERROR(loom_vm_module_callable_collect(
        build, loom_func_ref_resolve_signature(build->module, type),
        loom_vm_module_function_ref_flags(type), &function_global_types[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_ref_types_canonicalize(
    loom_vm_module_type_build_t* build) {
  if (build->ref_type_count > 65536u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM ref-type count exceeds the u16 ordinal domain");
  }
  if (build->ref_type_count == 0) return iree_ok_status();
  qsort(build->ref_type_order, build->ref_type_count,
        sizeof(*build->ref_type_order), loom_vm_module_compare_ref_types);
  for (iree_host_size_t i = 0; i < build->ref_type_count; ++i) {
    build->ref_type_order[i]->ordinal = (uint16_t)i;
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_callables_canonicalize(
    loom_vm_module_type_build_t* build,
    iree_host_size_t* out_callable_type_count) {
  *out_callable_type_count = 0;
  qsort(build->callable_order, build->callable_count,
        sizeof(*build->callable_order), loom_vm_module_compare_callable_depths);

  iree_host_size_t depth_begin = 0;
  while (depth_begin < build->callable_count) {
    const uint16_t depth = build->callable_order[depth_begin]->nesting_depth;
    iree_host_size_t depth_end = depth_begin + 1;
    while (depth_end < build->callable_count &&
           build->callable_order[depth_end]->nesting_depth == depth) {
      ++depth_end;
    }
    qsort(build->callable_order + depth_begin, depth_end - depth_begin,
          sizeof(*build->callable_order),
          loom_vm_module_compare_callable_structures);

    loom_vm_module_callable_type_build_t* previous = NULL;
    for (iree_host_size_t i = depth_begin; i < depth_end; ++i) {
      loom_vm_module_callable_type_build_t* callable = build->callable_order[i];
      if (previous == NULL || loom_vm_module_compare_callable_structures(
                                  &previous, &callable) != 0) {
        if (*out_callable_type_count == 65536u) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "VM callable-type count exceeds the u16 ordinal domain");
        }
        build->callable_order[*out_callable_type_count] = callable;
        ++*out_callable_type_count;
        previous = callable;
      }
      callable->ordinal = (uint16_t)(*out_callable_type_count - 1);
    }
    depth_begin = depth_end;
  }
  return iree_ok_status();
}

static uint16_t loom_vm_module_signature_field_type_ordinal(
    const loom_vm_module_signature_field_build_t* field) {
  if (field->kind == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
    return field->type.ref_type->ordinal;
  }
  if (field->kind == IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
    return field->type.callable_type->ordinal;
  }
  return 0;
}

static iree_status_t loom_vm_module_type_rows_build(
    loom_vm_module_type_build_t* build, iree_host_size_t callable_type_count,
    loom_vm_module_type_tables_t* tables) {
  iree_host_size_t descriptor_count = 0;
  for (iree_host_size_t i = 0; i < callable_type_count; ++i) {
    const loom_vm_module_callable_type_build_t* callable =
        build->callable_order[i];
    const iree_host_size_t callable_descriptor_count =
        (iree_host_size_t)callable->argument_count + callable->result_count;
    if (!iree_host_size_checked_add(descriptor_count, callable_descriptor_count,
                                    &descriptor_count) ||
        descriptor_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM signature descriptor count exceeds u32");
    }
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, callable_type_count, sizeof(*tables->signatures),
      (void**)&tables->signatures));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, callable_type_count, sizeof(*tables->callable_types),
      (void**)&tables->callable_types));
  if (descriptor_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->arena, descriptor_count, sizeof(*tables->signature_descriptors),
        (void**)&tables->signature_descriptors));
  }
  tables->signature_count = (uint32_t)callable_type_count;
  tables->signature_descriptor_count = (uint32_t)descriptor_count;
  tables->callable_type_count = (uint32_t)callable_type_count;

  uint32_t descriptor_base = 0;
  for (iree_host_size_t i = 0; i < callable_type_count; ++i) {
    const loom_vm_module_callable_type_build_t* callable =
        build->callable_order[i];
    tables->signatures[i] = callable->signature_row;
    tables->signatures[i].descriptor_base_u32 = descriptor_base;
    tables->callable_types[i] = (iree_vm_bytecode_v0_callable_type_row_t){
        .signature_ordinal_u16 = (uint16_t)i,
        .flags_u16 = callable->flags,
        .nesting_depth_u16 = callable->nesting_depth,
    };
    const uint32_t field_count =
        (uint32_t)callable->argument_count + callable->result_count;
    for (uint32_t j = 0; j < field_count; ++j) {
      const loom_vm_module_signature_field_build_t* field =
          &callable->fields[j];
      tables->signature_descriptors[descriptor_base++] =
          (iree_vm_bytecode_v0_signature_descriptor_row_t){
              .kind_u16 = field->kind,
              .type_ordinal_u16 =
                  loom_vm_module_signature_field_type_ordinal(field),
          };
    }
  }
  IREE_ASSERT_EQ(descriptor_base, descriptor_count);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_resource_type_rows_build(
    loom_vm_module_type_build_t* build,
    loom_vm_module_ref_type_build_t* const* ref_global_types,
    loom_vm_module_callable_type_build_t* const* function_global_types,
    loom_vm_module_layout_t* layout) {
  loom_vm_module_resource_layout_t* resources = &layout->resources;
  if (resources->ref_global_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(build->arena, resources->ref_global_count,
                                  sizeof(*resources->ref_global_descriptors),
                                  (void**)&resources->ref_global_descriptors));
    for (uint32_t i = 0; i < resources->ref_global_count; ++i) {
      resources->ref_global_descriptors[i] =
          (iree_vm_bytecode_v0_global_ref_descriptor_row_t){
              .ref_type_ordinal_u16 = ref_global_types[i]->ordinal,
          };
    }
  }
  if (resources->function_global_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->arena, resources->function_global_count,
        sizeof(*resources->function_global_descriptors),
        (void**)&resources->function_global_descriptors));
    for (uint32_t i = 0; i < resources->function_global_count; ++i) {
      resources->function_global_descriptors[i] =
          (iree_vm_bytecode_v0_global_function_descriptor_row_t){
              .callable_type_ordinal_u16 = function_global_types[i]->ordinal,
          };
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_callable_type_lookup_build(
    const loom_vm_module_type_build_t* build,
    loom_vm_module_type_tables_t* tables) {
  const iree_host_size_t signature_count = build->module->types.count;
  if (signature_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, signature_count,
      sizeof(*tables->callable_type_ordinals_by_signature),
      (void**)&tables->callable_type_ordinals_by_signature));
  tables->callable_type_ordinal_signature_count = signature_count;
  for (iree_host_size_t i = 0; i < signature_count; ++i) {
    tables->callable_type_ordinals_by_signature[i] =
        (loom_vm_module_callable_type_ordinals_t){
            .synchronous = UINT16_MAX,
            .yieldable = UINT16_MAX,
        };
  }

  for (iree_host_size_t i = 0; i < build->module->types.count; ++i) {
    const loom_type_t type = build->module->types.entries[i];
    if (!loom_func_ref_type_isa(type)) continue;
    const loom_type_id_t signature_id = loom_func_ref_type_signature(type);
    if (signature_id >= signature_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM function reference has an invalid signature type ID");
    }
    const uint16_t flags = loom_vm_module_function_ref_flags(type);
    const loom_vm_module_callable_type_build_t* callable =
        loom_vm_module_callable_find(
            build, loom_func_ref_resolve_signature(build->module, type), flags);
    if (callable == NULL) continue;
    loom_vm_module_callable_type_ordinals_t* ordinals =
        &tables->callable_type_ordinals_by_signature[signature_id];
    if (iree_any_bit_set(flags,
                         IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD)) {
      ordinals->yieldable = callable->ordinal;
    } else {
      ordinals->synchronous = callable->ordinal;
    }
  }
  return iree_ok_status();
}

static bool loom_vm_module_string_ordinal_find(
    const loom_vm_module_type_tables_t* tables, iree_string_view_t value,
    uint16_t* out_ordinal) {
  iree_host_size_t low = 0;
  iree_host_size_t high = tables->string_count;
  while (low < high) {
    const iree_host_size_t mid = low + (high - low) / 2;
    const int comparison =
        iree_string_view_compare(tables->strings[mid], value);
    if (comparison < 0) {
      low = mid + 1;
    } else if (comparison > 0) {
      high = mid;
    } else {
      *out_ordinal = (uint16_t)mid;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_vm_module_strings_build(
    const loom_vm_module_type_build_t* build, loom_vm_module_layout_t* layout) {
  iree_host_size_t string_capacity = layout->export_count;
  if (!iree_host_size_checked_add(string_capacity,
                                  layout->import_declaration_count,
                                  &string_capacity) ||
      !iree_host_size_checked_add(string_capacity,
                                  layout->import_declaration_count,
                                  &string_capacity) ||
      !iree_host_size_checked_add(string_capacity, build->ref_type_count,
                                  &string_capacity) ||
      !iree_host_size_checked_add(string_capacity, build->ref_type_count,
                                  &string_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM string candidate count exceeds host size");
  }
  if (string_capacity == 0) return iree_ok_status();

  iree_string_view_t* strings = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, string_capacity, sizeof(*strings), (void**)&strings));
  iree_host_size_t string_count = 0;
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    const iree_string_view_t export_name = layout->functions[i].export_name;
    if (!iree_string_view_is_empty(export_name)) {
      strings[string_count++] = export_name;
    }
  }
  for (iree_host_size_t i = 0; i < layout->import_declaration_count; ++i) {
    strings[string_count++] = layout->import_declarations[i].module_name;
    strings[string_count++] = layout->import_declarations[i].symbol_name;
  }
  for (iree_host_size_t i = 0; i < build->ref_type_count; ++i) {
    strings[string_count++] = build->ref_type_order[i]->namespace_name;
    strings[string_count++] = build->ref_type_order[i]->type_name;
  }
  IREE_ASSERT_EQ(string_count, string_capacity);

  qsort(strings, string_count, sizeof(*strings),
        loom_vm_module_compare_string_views);
  iree_host_size_t unique_count = 0;
  for (iree_host_size_t i = 0; i < string_count; ++i) {
    if (unique_count == 0 ||
        !iree_string_view_equal(strings[unique_count - 1], strings[i])) {
      strings[unique_count++] = strings[i];
    }
  }
  if (unique_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM string count exceeds the u16 ordinal domain");
  }
  layout->type_tables.strings = strings;
  layout->type_tables.string_count = (uint32_t)unique_count;
  return iree_ok_status();
}

static iree_status_t loom_vm_module_ref_type_rows_build(
    const loom_vm_module_type_build_t* build,
    loom_vm_module_type_tables_t* tables) {
  if (build->ref_type_count == 0) return iree_ok_status();

  iree_host_size_t group_count = 0;
  iree_string_view_t previous_namespace = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < build->ref_type_count; ++i) {
    const iree_string_view_t namespace_name =
        build->ref_type_order[i]->namespace_name;
    if (i == 0 || !iree_string_view_equal(previous_namespace, namespace_name)) {
      ++group_count;
      previous_namespace = namespace_name;
    }
  }
  if (group_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM ref-type namespace count exceeds the u16 ordinal domain");
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, group_count, sizeof(*tables->ref_type_groups),
      (void**)&tables->ref_type_groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, build->ref_type_count, sizeof(*tables->ref_type_entries),
      (void**)&tables->ref_type_entries));
  tables->ref_type_group_count = (uint32_t)group_count;
  tables->ref_type_entry_count = (uint32_t)build->ref_type_count;

  iree_host_size_t group_index = 0;
  iree_host_size_t entry_index = 0;
  while (entry_index < build->ref_type_count) {
    const iree_string_view_t namespace_name =
        build->ref_type_order[entry_index]->namespace_name;
    iree_host_size_t group_end = entry_index + 1;
    while (
        group_end < build->ref_type_count &&
        iree_string_view_equal(
            namespace_name, build->ref_type_order[group_end]->namespace_name)) {
      ++group_end;
    }
    uint16_t namespace_ordinal = 0;
    if (!loom_vm_module_string_ordinal_find(tables, namespace_name,
                                            &namespace_ordinal)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "VM ref-type namespace is missing from strings");
    }
    tables->ref_type_groups[group_index++] =
        (iree_vm_bytecode_v0_ref_type_group_row_t){
            .namespace_string_u16 = namespace_ordinal,
            .entry_count_u32 = (uint32_t)(group_end - entry_index),
        };
    while (entry_index < group_end) {
      uint16_t type_name_ordinal = 0;
      if (!loom_vm_module_string_ordinal_find(
              tables, build->ref_type_order[entry_index]->type_name,
              &type_name_ordinal)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "VM ref-type name is missing from strings");
      }
      tables->ref_type_entries[entry_index] =
          (iree_vm_bytecode_v0_ref_type_entry_row_t){
              .type_name_string_u16 = type_name_ordinal,
          };
      ++entry_index;
    }
  }
  IREE_ASSERT_EQ(group_index, group_count);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_root_ordinals_assign(
    loom_vm_module_callable_type_build_t* const* root_callables,
    loom_vm_module_layout_t* layout) {
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    loom_vm_module_function_layout_t* function = &layout->functions[i];
    function->callable_type_ordinal = root_callables[i]->ordinal;
    function->export_name_string_ordinal = UINT16_MAX;
    if (!iree_string_view_is_empty(function->export_name) &&
        !loom_vm_module_string_ordinal_find(
            &layout->type_tables, function->export_name,
            &function->export_name_string_ordinal)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "VM export name is missing from strings");
    }
  }
  for (iree_host_size_t i = 0; i < layout->import_declaration_count; ++i) {
    loom_vm_module_import_layout_t* import = &layout->import_declarations[i];
    import->callable_type_ordinal =
        root_callables[layout->function_count + i]->ordinal;
    if (!loom_vm_module_string_ordinal_find(
            &layout->type_tables, import->module_name,
            &import->module_name_string_ordinal)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "VM import module is missing from strings");
    }
    if (!loom_vm_module_string_ordinal_find(
            &layout->type_tables, import->symbol_name,
            &import->symbol_name_string_ordinal)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "VM import symbol is missing from strings");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_vm_module_type_tables_build(
    iree_arena_allocator_t* arena, loom_vm_module_layout_t* layout) {
  layout->type_tables = (loom_vm_module_type_tables_t){0};
  loom_vm_module_type_capacities_t capacities;
  IREE_RETURN_IF_ERROR(
      loom_vm_module_type_capacities_count(layout, &capacities));

  loom_vm_module_type_build_t build = {
      .module = layout->module,
      .arena = arena,
      .callable_capacity = capacities.callable_count,
      .ref_type_capacity = capacities.ref_type_count,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, build.callable_capacity, sizeof(*build.callable_storage),
      (void**)&build.callable_storage));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, build.callable_capacity, sizeof(*build.callable_order),
      (void**)&build.callable_order));
  if (build.ref_type_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, build.ref_type_capacity, sizeof(*build.ref_type_storage),
        (void**)&build.ref_type_storage));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, build.ref_type_capacity, sizeof(*build.ref_type_order),
        (void**)&build.ref_type_order));
  }

  loom_vm_module_callable_type_build_t** root_callables = NULL;
  iree_host_size_t root_callable_count = layout->function_count;
  if (!iree_host_size_checked_add(root_callable_count,
                                  layout->import_declaration_count,
                                  &root_callable_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM root callable count exceeds host size");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, root_callable_count,
                                                 sizeof(*root_callables),
                                                 (void**)&root_callables));
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    const uint16_t callable_flags =
        iree_any_bit_set(layout->functions[i].flags,
                         IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD)
            ? IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD
            : 0;
    IREE_RETURN_IF_ERROR(loom_vm_module_callable_collect(
        &build, layout->functions[i].logical_signature, callable_flags,
        &root_callables[i]));
  }
  for (iree_host_size_t i = 0; i < layout->import_declaration_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vm_module_callable_collect(
        &build, layout->import_declarations[i].logical_signature,
        layout->import_declarations[i].callable_flags,
        &root_callables[layout->function_count + i]));
  }
  IREE_RETURN_IF_ERROR(
      loom_vm_module_collect_indirect_call_types(&build, layout));

  loom_vm_module_ref_type_build_t** ref_global_types = NULL;
  if (layout->resources.ref_global_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, layout->resources.ref_global_count, sizeof(*ref_global_types),
        (void**)&ref_global_types));
  }
  loom_vm_module_callable_type_build_t** function_global_types = NULL;
  if (layout->resources.function_global_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, layout->resources.function_global_count,
        sizeof(*function_global_types), (void**)&function_global_types));
  }
  IREE_RETURN_IF_ERROR(loom_vm_module_collect_resource_types(
      &build, layout, ref_global_types, function_global_types));

  IREE_RETURN_IF_ERROR(loom_vm_module_ref_types_canonicalize(&build));
  iree_host_size_t callable_type_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_vm_module_callables_canonicalize(&build, &callable_type_count));
  IREE_RETURN_IF_ERROR(loom_vm_module_type_rows_build(
      &build, callable_type_count, &layout->type_tables));
  IREE_RETURN_IF_ERROR(loom_vm_module_resource_type_rows_build(
      &build, ref_global_types, function_global_types, layout));
  IREE_RETURN_IF_ERROR(
      loom_vm_module_callable_type_lookup_build(&build, &layout->type_tables));
  IREE_RETURN_IF_ERROR(loom_vm_module_strings_build(&build, layout));
  IREE_RETURN_IF_ERROR(
      loom_vm_module_ref_type_rows_build(&build, &layout->type_tables));
  return loom_vm_module_root_ordinals_assign(root_callables, layout);
}

bool loom_vm_module_type_tables_try_resolve_callable_ordinal(
    const loom_vm_module_type_tables_t* tables, loom_type_t function_ref_type,
    uint16_t* out_ordinal) {
  *out_ordinal = UINT16_MAX;
  if (!loom_func_ref_type_isa(function_ref_type)) return false;
  const loom_type_id_t signature_id =
      loom_func_ref_type_signature(function_ref_type);
  if (signature_id >= tables->callable_type_ordinal_signature_count) {
    return false;
  }
  const loom_vm_module_callable_type_ordinals_t ordinals =
      tables->callable_type_ordinals_by_signature[signature_id];
  const uint16_t ordinal =
      loom_func_ref_type_has_yieldability(function_ref_type)
          ? ordinals.yieldable
          : ordinals.synchronous;
  if (ordinal == UINT16_MAX) return false;
  *out_ordinal = ordinal;
  return true;
}
