// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/module_plan.h"

#include <limits.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/ireevm/descriptors/descriptors.h"
#include "loom/target/function_contract.h"
#include "loom/target/registers.h"

enum {
  LOOM_IREEVM_MAX_CALLING_CONVENTION_LENGTH = 256,
};

typedef struct loom_ireevm_calling_convention_buffer_t {
  // Stack storage for a VM calling convention string.
  char data[LOOM_IREEVM_MAX_CALLING_CONVENTION_LENGTH];
  // Number of initialized bytes in |data|.
  iree_host_size_t length;
} loom_ireevm_calling_convention_buffer_t;

static iree_string_view_t loom_ireevm_module_plan_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_ireevm_module_plan_symbol_name(
    const loom_module_t* module, loom_symbol_id_t symbol_id) {
  if (symbol_id >= module->symbols.count) {
    return IREE_SV("<unknown>");
  }
  return loom_ireevm_module_plan_string(
      module, module->symbols.entries[symbol_id].name_id);
}

static iree_status_t loom_ireevm_module_plan_copy_string(
    iree_arena_allocator_t* arena, iree_string_view_t source,
    iree_string_view_t* out_target) {
  *out_target = iree_string_view_empty();
  if (source.size == 0) {
    return iree_ok_status();
  }
  char* target_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, source.size, (void**)&target_data));
  memcpy(target_data, source.data, source.size);
  *out_target = iree_make_string_view(target_data, source.size);
  return iree_ok_status();
}

static bool loom_ireevm_module_plan_is_vm_bundle(
    const loom_target_bundle_t* bundle) {
  return bundle && bundle->snapshot && bundle->export_plan &&
         bundle->snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_VM &&
         bundle->snapshot->artifact_format ==
             LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_VM_MODULE_FUNCTION;
}

static bool loom_ireevm_module_plan_is_vm_import(const loom_op_t* op) {
  return loom_low_func_decl_isa(op) && loom_low_func_decl_import_kind(op) ==
                                           LOOM_LOW_FUNC_DECL_IMPORT_KIND_VM;
}

static iree_status_t loom_ireevm_calling_convention_append(
    loom_ireevm_calling_convention_buffer_t* buffer, char c) {
  if (buffer->length >= IREE_ARRAYSIZE(buffer->data)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IREE VM calling convention exceeds %u bytes",
                            (unsigned)IREE_ARRAYSIZE(buffer->data));
  }
  buffer->data[buffer->length++] = c;
  return iree_ok_status();
}

static bool loom_ireevm_module_plan_type_is_ref(const loom_module_t* module,
                                                loom_type_t type) {
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 1) {
    return false;
  }
  return iree_string_view_equal(
      loom_ireevm_module_plan_string(module, loom_type_dialect_name_id(type)),
      IREE_SV("ireevm.ref"));
}

static iree_status_t loom_ireevm_module_plan_append_calling_convention_type(
    const loom_module_t* module, loom_type_t type,
    loom_ireevm_calling_convention_buffer_t* buffer) {
  if (loom_low_type_is_register(type)) {
    const uint32_t unit_count = loom_low_register_type_unit_count(type);
    switch (loom_low_register_type_class_id(type)) {
      case IREEVM_CORE_REG_CLASS_ID_I32:
        if (unit_count == 1) {
          return loom_ireevm_calling_convention_append(buffer, 'i');
        }
        break;
      case IREEVM_CORE_REG_CLASS_ID_I64:
        if (unit_count == 2) {
          return loom_ireevm_calling_convention_append(buffer, 'I');
        }
        break;
      case IREEVM_CORE_REG_CLASS_ID_F32:
        if (unit_count == 1) {
          return loom_ireevm_calling_convention_append(buffer, 'f');
        }
        break;
      case IREEVM_CORE_REG_CLASS_ID_F64:
        if (unit_count == 2) {
          return loom_ireevm_calling_convention_append(buffer, 'F');
        }
        break;
      case IREEVM_CORE_REG_CLASS_ID_REF:
        if (unit_count == 1) {
          return loom_ireevm_calling_convention_append(buffer, 'r');
        }
        break;
    }
  }
  if (loom_type_is_scalar(type)) {
    const loom_scalar_type_t scalar_type = loom_type_element_type(type);
    switch (scalar_type) {
      case LOOM_SCALAR_TYPE_I1:
      case LOOM_SCALAR_TYPE_I32:
        return loom_ireevm_calling_convention_append(buffer, 'i');
      case LOOM_SCALAR_TYPE_I64:
        return loom_ireevm_calling_convention_append(buffer, 'I');
      case LOOM_SCALAR_TYPE_F32:
        return loom_ireevm_calling_convention_append(buffer, 'f');
      case LOOM_SCALAR_TYPE_F64:
        return loom_ireevm_calling_convention_append(buffer, 'F');
      default:
        break;
    }
  }
  if (loom_ireevm_module_plan_type_is_ref(module, type)) {
    return loom_ireevm_calling_convention_append(buffer, 'r');
  }
  return iree_make_status(
      IREE_STATUS_INTERNAL,
      "IREE VM target legality accepted an unsupported ABI value type");
}

static iree_status_t loom_ireevm_module_plan_build_calling_convention(
    const loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena, iree_string_view_t* out_calling_convention) {
  loom_ireevm_calling_convention_buffer_t buffer = {0};
  IREE_RETURN_IF_ERROR(loom_ireevm_calling_convention_append(&buffer, '0'));

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ireevm_module_plan_append_calling_convention_type(
        module, loom_module_value_type(module, argument_ids[i]), &buffer));
  }

  IREE_RETURN_IF_ERROR(loom_ireevm_calling_convention_append(&buffer, '_'));
  const loom_value_id_t* result_ids = loom_op_const_results(function.op);
  for (uint16_t i = 0; i < function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ireevm_module_plan_append_calling_convention_type(
        module, loom_module_value_type(module, result_ids[i]), &buffer));
  }
  return loom_ireevm_module_plan_copy_string(
      arena, iree_make_string_view(buffer.data, buffer.length),
      out_calling_convention);
}

static iree_status_t loom_ireevm_module_plan_lookup_func_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_id_t symbol_id,
    const loom_func_symbol_facts_t** out_func_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                     symbol_id, &base_facts));
  *out_func_facts = loom_func_symbol_facts_cast(base_facts);
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_plan_append_import(
    const loom_module_t* module, loom_symbol_id_t symbol_id,
    const loom_func_symbol_facts_t* func_facts, iree_arena_allocator_t* arena,
    loom_ireevm_module_plan_t* plan) {
  loom_ireevm_module_plan_import_t* import = &plan->imports[plan->import_count];
  *import = (loom_ireevm_module_plan_import_t){
      .symbol_ref = {.module_id = 0, .symbol_id = symbol_id},
      .op = func_facts->func_op,
      .full_name = loom_ireevm_module_plan_string(
          module, loom_low_func_decl_code_symbol(func_facts->func_op)),
  };
  IREE_RETURN_IF_ERROR(loom_ireevm_module_plan_build_calling_convention(
      module, loom_func_like_cast(module, func_facts->func_op), arena,
      &import->calling_convention));

  plan->callee_slots[symbol_id] = (loom_ireevm_callee_slot_t){
      .kind = LOOM_IREEVM_CALLEE_IMPORT,
      .ordinal = plan->import_count,
  };
  ++plan->import_count;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_plan_append_function(
    const loom_module_t* module, loom_symbol_id_t symbol_id,
    const loom_func_symbol_facts_t* func_facts,
    const loom_target_bundle_storage_t* bundle_storage,
    iree_arena_allocator_t* arena, loom_ireevm_module_plan_t* plan) {
  loom_ireevm_module_plan_function_t* function =
      &plan->functions[plan->function_count];
  *function = (loom_ireevm_module_plan_function_t){
      .symbol_ref = {.module_id = 0, .symbol_id = symbol_id},
      .op = func_facts->func_op,
      .symbol_name = func_facts->name,
      .bundle_storage = *bundle_storage,
  };
  loom_target_bundle_storage_rebind(&function->bundle_storage);
  IREE_RETURN_IF_ERROR(loom_ireevm_module_plan_build_calling_convention(
      module, loom_func_like_cast(module, func_facts->func_op), arena,
      &function->calling_convention));

  plan->callee_slots[symbol_id] = (loom_ireevm_callee_slot_t){
      .kind = LOOM_IREEVM_CALLEE_LOCAL,
      .ordinal = plan->function_count,
  };

  if (func_facts->exports || func_facts->visibility != 0) {
    iree_string_view_t export_name = bundle_storage->export_plan.export_symbol;
    if (iree_string_view_is_empty(export_name)) {
      export_name = func_facts->name;
    }
    plan->exports[plan->export_count] = (loom_ireevm_module_plan_export_t){
        .symbol_ref = {.module_id = 0, .symbol_id = symbol_id},
        .local_name = export_name,
        .internal_ordinal = plan->function_count,
    };
    ++plan->export_count;
  }

  ++plan->function_count;
  return iree_ok_status();
}

iree_status_t loom_ireevm_module_plan_build(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, loom_ireevm_module_plan_t* out_plan) {
  *out_valid = false;
  *out_plan = (loom_ireevm_module_plan_t){
      .module = module,
      .callee_slot_count = module->symbols.count,
  };

  loom_ireevm_callee_slot_t* callee_slots = NULL;
  loom_ireevm_module_plan_import_t* imports = NULL;
  loom_ireevm_module_plan_function_t* functions = NULL;
  loom_ireevm_module_plan_export_t* exports = NULL;
  if (module->symbols.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                   sizeof(*callee_slots),
                                                   (void**)&callee_slots));
    memset(callee_slots, 0, module->symbols.count * sizeof(*callee_slots));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->symbols.count, sizeof(*imports), (void**)&imports));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->symbols.count, sizeof(*functions), (void**)&functions));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->symbols.count, sizeof(*exports), (void**)&exports));
  }
  out_plan->callee_slots = callee_slots;
  out_plan->imports = imports;
  out_plan->functions = functions;
  out_plan->exports = exports;

  iree_status_t status = iree_ok_status();
  bool valid = true;
  for (iree_host_size_t i = 0;
       valid && i < module->symbols.count && iree_status_is_ok(status); ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!symbol->defining_op ||
        (!loom_low_func_def_isa(symbol->defining_op) &&
         !loom_ireevm_module_plan_is_vm_import(symbol->defining_op))) {
      continue;
    }

    const loom_symbol_id_t symbol_id = (loom_symbol_id_t)i;
    const loom_func_symbol_facts_t* func_facts = NULL;
    status = loom_ireevm_module_plan_lookup_func_facts(module, fact_table,
                                                       symbol_id, &func_facts);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (!func_facts) {
      continue;
    }

    loom_target_bundle_storage_t bundle_storage = {0};
    bool contract_valid = false;
    status = loom_target_function_contract_resolve(
        module, fact_table, func_facts, diagnostic_emitter, &contract_valid,
        &bundle_storage);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (!contract_valid) {
      valid = false;
      break;
    }
    if (!loom_ireevm_module_plan_is_vm_bundle(&bundle_storage.bundle)) {
      continue;
    }

    if (loom_low_func_def_isa(symbol->defining_op)) {
      status = loom_ireevm_module_plan_append_function(
          module, symbol_id, func_facts, &bundle_storage, arena, out_plan);
    } else {
      status = loom_ireevm_module_plan_append_import(
          module, symbol_id, func_facts, arena, out_plan);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_valid = valid;
  }
  return status;
}

iree_status_t loom_ireevm_module_plan_resolve_callee(
    const loom_ireevm_module_plan_t* plan, loom_symbol_ref_t callee,
    uint32_t* out_encoded_ordinal) {
  *out_encoded_ordinal = 0;
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= plan->callee_slot_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM call callee symbol reference is invalid");
  }
  const loom_ireevm_callee_slot_t slot = plan->callee_slots[callee.symbol_id];
  switch (slot.kind) {
    case LOOM_IREEVM_CALLEE_LOCAL:
      *out_encoded_ordinal = slot.ordinal;
      return iree_ok_status();
    case LOOM_IREEVM_CALLEE_IMPORT:
      if (slot.ordinal > INT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "VM import ordinal exceeds i31");
      }
      *out_encoded_ordinal = slot.ordinal | LOOM_IREEVM_IMPORT_ORDINAL_BIT;
      return iree_ok_status();
    case LOOM_IREEVM_CALLEE_NONE:
    default: {
      iree_string_view_t callee_name =
          loom_ireevm_module_plan_symbol_name(plan->module, callee.symbol_id);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "VM call callee '@%.*s' is outside the emitted module plan",
          (int)callee_name.size, callee_name.data);
    }
  }
}
