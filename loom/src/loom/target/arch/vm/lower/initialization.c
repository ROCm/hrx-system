// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/initialization.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/vm/ops/ops.h"

static bool loom_vm_initialization_target_is_vm(const loom_module_t* module,
                                                loom_symbol_ref_t target_ref) {
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
      target_ref.symbol_id >= module->symbols.count) {
    return false;
  }
  const loom_op_t* target_op =
      module->symbols.entries[target_ref.symbol_id].defining_op;
  return target_op != NULL && loom_vm_target_isa(target_op);
}

static iree_status_t loom_vm_initialization_emit(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = related_op_count,
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static iree_status_t loom_vm_initialization_emit_duplicate_initializer(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_op_t* op,
    const loom_op_t* prior_op) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("initialize")),
  };
  const loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("first initializer defined here"),
      .op = prior_op,
  }};
  return loom_vm_initialization_emit(
      diagnostic_emitter, op, LOOM_ERR_SYMBOL_005, params,
      IREE_ARRAYSIZE(params), related_ops, IREE_ARRAYSIZE(related_ops));
}

static iree_status_t loom_vm_initialization_emit_reserved_export(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_module_t* module,
    const loom_symbol_t* symbol, const loom_op_t* op) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(module->strings.entries[symbol->name_id]),
  };
  return loom_vm_initialization_emit(
      diagnostic_emitter, op, LOOM_ERR_TARGET_073, params,
      IREE_ARRAYSIZE(params),
      /*related_ops=*/NULL, /*related_op_count=*/0);
}

static iree_status_t loom_vm_initialization_emit_result_count(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_module_t* module,
    loom_func_like_t initializer) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, initializer.op)),
      loom_param_u32(initializer.op->result_count),
      loom_param_u32(0),
  };
  return loom_vm_initialization_emit(
      diagnostic_emitter, initializer.op, LOOM_ERR_STRUCTURE_002, params,
      IREE_ARRAYSIZE(params), /*related_ops=*/NULL,
      /*related_op_count=*/0);
}

static iree_status_t loom_vm_initialization_emit_export_name(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_module_t* module,
    loom_func_like_t initializer, loom_string_id_t export_name_id) {
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("export")),
          loom_diagnostic_field_ref(
              LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
              initializer.vtable->export_symbol_attr_index)),
      loom_param_string(module->strings.entries[export_name_id]),
      loom_param_string(IREE_SV("the canonical name \"initialize\"")),
  };
  return loom_vm_initialization_emit(
      diagnostic_emitter, initializer.op, LOOM_ERR_STRUCTURE_027, params,
      IREE_ARRAYSIZE(params), /*related_ops=*/NULL,
      /*related_op_count=*/0);
}

static iree_status_t loom_vm_initialization_emit_ambiguous_target(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_op_t* inline_global, iree_host_size_t target_count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u32((uint32_t)target_count),
  };
  return loom_vm_initialization_emit(
      diagnostic_emitter, inline_global, LOOM_ERR_TARGET_072, params,
      IREE_ARRAYSIZE(params), /*related_ops=*/NULL,
      /*related_op_count=*/0);
}

static iree_status_t loom_vm_initialization_reserve_function_symbol(
    loom_module_t* module, loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  for (uint32_t suffix_ordinal = 0; suffix_ordinal < UINT16_MAX;
       ++suffix_ordinal) {
    char name_storage[64];
    iree_string_view_t name = IREE_SV("__vm_initialize");
    if (suffix_ordinal != 0) {
      const int name_length =
          iree_snprintf(name_storage, sizeof(name_storage),
                        "__vm_initialize$%u", (unsigned)suffix_ordinal);
      if (name_length < 0 ||
          (iree_host_size_t)name_length >= IREE_ARRAYSIZE(name_storage)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "VM initializer symbol name overflow");
      }
      name = iree_make_string_view(name_storage, (iree_host_size_t)name_length);
    }

    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    if (name_id != LOOM_STRING_ID_INVALID &&
        loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    if (name_id == LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
    }
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_symbol_ref = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "no unique VM initializer symbol is available");
}

static iree_status_t loom_vm_initialization_build_function(
    loom_rewriter_t* rewriter, loom_symbol_ref_t target_ref,
    loom_string_id_t export_name_id, loom_location_id_t location,
    loom_op_t** out_function_op) {
  *out_function_op = NULL;
  loom_symbol_ref_t function_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_vm_initialization_reserve_function_symbol(
      rewriter->module, &function_ref));

  loom_builder_set_block(&rewriter->builder,
                         loom_module_block(rewriter->module));
  loom_op_t* function_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &rewriter->builder,
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC | LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET |
          LOOM_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL,
      /*visibility=*/0, /*retain=*/0, LOOM_FUNC_CC_INITIALIZER,
      /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0, target_ref,
      /*abi=*/0, loom_named_attr_slice_empty(), export_name_id,
      loom_named_attr_slice_empty(), function_ref,
      /*arg_types=*/NULL, /*arg_types_count=*/0, /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0, location, &function_op));

  loom_func_like_t function =
      loom_func_like_cast(rewriter->module, function_op);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &rewriter->builder, function_op, loom_func_like_body(function));
  loom_op_t* return_op = NULL;
  const iree_status_t status =
      loom_func_return_build(&rewriter->builder, /*values=*/NULL,
                             /*values_count=*/0, location, &return_op);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  *out_function_op = function_op;
  return iree_ok_status();
}

iree_status_t loom_vm_materialize_initializer(
    loom_module_t* module, iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_low_lower_prepare_module_result_t* out_result) {
  *out_result = (loom_low_lower_prepare_module_result_t){
      .valid = true,
  };
  loom_symbol_ref_t vm_target_ref = loom_symbol_ref_null();
  iree_host_size_t vm_target_count = 0;
  loom_func_like_t initializer = {0};
  const loom_op_t* inline_global = NULL;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    loom_op_t* op = module->symbols.entries[i].defining_op;
    if (op == NULL) continue;
    if (inline_global == NULL &&
        ((loom_global_constant_isa(op) &&
          !loom_attr_is_absent(loom_global_constant_initializer(op))) ||
         (loom_global_variable_isa(op) &&
          !loom_attr_is_absent(loom_global_variable_initializer(op))))) {
      inline_global = op;
    }
    if (loom_vm_target_isa(op)) {
      if (vm_target_count == 0) {
        vm_target_ref = (loom_symbol_ref_t){
            .module_id = 0,
            .symbol_id = (loom_symbol_id_t)i,
        };
      }
      ++vm_target_count;
      continue;
    }

    loom_func_like_t function = loom_func_like_cast(module, op);
    if (!loom_func_like_isa(function) ||
        loom_func_like_body(function) == NULL ||
        !loom_vm_initialization_target_is_vm(module,
                                             loom_func_like_target(function))) {
      continue;
    }
    const iree_string_view_t effective_export_name =
        loom_func_like_export_name(module, symbol, function);
    if (loom_func_like_cc(function) != LOOM_FUNC_CC_INITIALIZER) {
      if (iree_string_view_equal(effective_export_name,
                                 IREE_SV("initialize"))) {
        out_result->valid = false;
        return loom_vm_initialization_emit_reserved_export(diagnostic_emitter,
                                                           module, symbol, op);
      }
      continue;
    }
    if (initializer.op != NULL) {
      out_result->valid = false;
      return loom_vm_initialization_emit_duplicate_initializer(
          diagnostic_emitter, op, initializer.op);
    }
    initializer = function;
  }
  if (vm_target_count == 0) return iree_ok_status();
  if (initializer.op != NULL && initializer.op->result_count != 0) {
    out_result->valid = false;
    return loom_vm_initialization_emit_result_count(diagnostic_emitter, module,
                                                    initializer);
  }

  if (initializer.op != NULL) {
    loom_string_id_t export_name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, IREE_SV("initialize"), &export_name_id));
    const loom_string_id_t existing_export =
        loom_func_like_export_symbol(initializer);
    if (existing_export != LOOM_STRING_ID_INVALID &&
        existing_export != export_name_id) {
      out_result->valid = false;
      return loom_vm_initialization_emit_export_name(
          diagnostic_emitter, module, initializer, existing_export);
    }
    if (existing_export == export_name_id) return iree_ok_status();

    loom_rewriter_t rewriter = {0};
    IREE_RETURN_IF_ERROR(
        loom_rewriter_initialize(&rewriter, module, scratch_arena));
    const iree_status_t status = loom_rewriter_set_attr(
        &rewriter, initializer.op, initializer.vtable->export_symbol_attr_index,
        loom_attr_string(export_name_id));
    loom_rewriter_deinitialize(&rewriter);
    IREE_RETURN_IF_ERROR(status);
    out_result->changed = true;
    return iree_ok_status();
  }

  if (inline_global == NULL) return iree_ok_status();
  if (vm_target_count != 1) {
    out_result->valid = false;
    return loom_vm_initialization_emit_ambiguous_target(
        diagnostic_emitter, inline_global, vm_target_count);
  }

  loom_string_id_t export_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("initialize"),
                                                 &export_name_id));
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  const loom_op_t* target_op =
      module->symbols.entries[vm_target_ref.symbol_id].defining_op;
  loom_op_t* function_op = NULL;
  const iree_status_t status = loom_vm_initialization_build_function(
      &rewriter, vm_target_ref, export_name_id, target_op->location,
      &function_op);
  loom_rewriter_deinitialize(&rewriter);
  IREE_RETURN_IF_ERROR(status);
  out_result->changed = true;
  return iree_ok_status();
}
