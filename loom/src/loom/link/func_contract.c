// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/func_contract.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"

typedef struct loom_link_func_contract_attr_t {
  // Source contract attribute index.
  uint8_t source_attr_index;
  // Selected contract attribute index.
  uint8_t selected_attr_index;
  // Stable field name used in mismatch reports.
  iree_string_view_t field_name;
} loom_link_func_contract_attr_t;

typedef enum loom_link_func_contract_apply_mode_e {
  LOOM_LINK_FUNC_CONTRACT_APPLY_CHECK = 0,
  LOOM_LINK_FUNC_CONTRACT_APPLY_MERGE = 1,
} loom_link_func_contract_apply_mode_t;

static void loom_link_func_contract_mismatch_field(
    iree_string_view_t field_name,
    loom_link_func_contract_mismatch_t* out_mismatch) {
  *out_mismatch = (loom_link_func_contract_mismatch_t){
      .kind = LOOM_LINK_FUNC_CONTRACT_MISMATCH_FIELD,
      .field_name = field_name,
  };
}

static void loom_link_func_contract_mismatch_count(
    iree_string_view_t field_name, iree_host_size_t source_count,
    iree_host_size_t selected_count,
    loom_link_func_contract_mismatch_t* out_mismatch) {
  *out_mismatch = (loom_link_func_contract_mismatch_t){
      .kind = LOOM_LINK_FUNC_CONTRACT_MISMATCH_COUNT,
      .field_name = field_name,
      .detail.counts =
          {
              .source = source_count,
              .selected = selected_count,
          },
  };
}

static void loom_link_func_contract_mismatch_type(
    iree_string_view_t field_name, iree_host_size_t type_ordinal,
    loom_link_func_contract_mismatch_t* out_mismatch) {
  *out_mismatch = (loom_link_func_contract_mismatch_t){
      .kind = LOOM_LINK_FUNC_CONTRACT_MISMATCH_TYPE,
      .field_name = field_name,
      .detail.type_ordinal = type_ordinal,
  };
}

loom_link_func_contract_t loom_link_func_contract_from_op(
    const loom_module_t* module, loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  IREE_ASSERT(vtable != NULL);
  IREE_ASSERT(vtable->symbol_def != NULL);
  IREE_ASSERT(vtable->func_like != NULL);
  loom_func_like_t function = loom_func_like_cast(module, op);
  IREE_ASSERT(loom_func_like_isa(function));

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  loom_value_slice_t workload_arguments = {0};
  if (loom_symbol_definition_implements(vtable->symbol_def,
                                        LOOM_SYMBOL_INTERFACE_KERNEL)) {
    workload_arguments = loom_kernel_workload_arg_ids(module, op);
  }
  return (loom_link_func_contract_t){
      .module = module,
      .symbol_definition = vtable->symbol_def,
      .function = vtable->func_like,
      .workload_arguments = workload_arguments,
      .arguments =
          {
              .values = (loom_value_id_t*)arguments,
              .count = argument_count,
          },
      .results =
          {
              .values = loom_op_results(op),
              .count = op->result_count,
          },
      .tied_results = loom_op_tied_results(op),
      .tied_result_count = op->tied_result_count,
      .attributes = loom_op_attrs(op),
  };
}

static iree_status_t loom_link_func_contract_map_values(
    loom_ir_remap_t* remap, loom_value_slice_t source_values,
    loom_value_slice_t selected_values, iree_string_view_t field_name,
    loom_link_func_contract_mismatch_t* out_mismatch) {
  if (source_values.count != selected_values.count) {
    loom_link_func_contract_mismatch_count(field_name, source_values.count,
                                           selected_values.count, out_mismatch);
    return iree_ok_status();
  }
  return loom_ir_remap_map_values(remap, source_values.values,
                                  selected_values.values, source_values.count);
}

static iree_status_t loom_link_func_contract_check_value_types(
    loom_ir_remap_t* remap, const loom_link_func_contract_t* source,
    loom_value_slice_t source_values, const loom_link_func_contract_t* selected,
    loom_value_slice_t selected_values, iree_string_view_t field_name,
    loom_link_func_contract_mismatch_t* out_mismatch) {
  for (uint16_t i = 0; i < source_values.count; ++i) {
    const loom_type_t source_type =
        loom_module_value_type(source->module, source_values.values[i]);
    loom_type_t remapped_type = {0};
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(remap, source_type, &remapped_type));
    const loom_type_t selected_type =
        loom_module_value_type(selected->module, selected_values.values[i]);
    if (!loom_type_equal(remapped_type, selected_type)) {
      loom_link_func_contract_mismatch_type(field_name, i, out_mismatch);
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_link_func_contract_attr_present(
    const loom_link_func_contract_t* contract, uint8_t attr_index) {
  return attr_index != LOOM_ATTR_INDEX_NONE &&
         !loom_attr_is_absent(contract->attributes[attr_index]);
}

static iree_status_t loom_link_func_contract_apply_attr(
    loom_ir_remap_t* remap, const loom_link_func_contract_t* source,
    uint8_t source_attr_index, const loom_link_func_contract_t* selected,
    uint8_t selected_attr_index, iree_string_view_t field_name,
    loom_link_func_contract_apply_mode_t mode,
    loom_link_func_contract_mismatch_t* out_mismatch) {
  if (!loom_link_func_contract_attr_present(source, source_attr_index)) {
    return iree_ok_status();
  }
  if (selected_attr_index == LOOM_ATTR_INDEX_NONE) {
    loom_link_func_contract_mismatch_field(field_name, out_mismatch);
    return iree_ok_status();
  }

  loom_attribute_t remapped_attr = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_attribute(
      remap, source->attributes[source_attr_index], &remapped_attr));
  loom_attribute_t* selected_attr = &selected->attributes[selected_attr_index];
  if (loom_attr_is_absent(*selected_attr)) {
    if (mode == LOOM_LINK_FUNC_CONTRACT_APPLY_MERGE) {
      *selected_attr = remapped_attr;
    }
    return iree_ok_status();
  }
  if (!loom_attribute_equal(selected_attr, &remapped_attr)) {
    loom_link_func_contract_mismatch_field(field_name, out_mismatch);
  }
  return iree_ok_status();
}

static iree_status_t loom_link_func_contract_apply_attrs(
    loom_ir_remap_t* remap, const loom_link_func_contract_t* source,
    const loom_link_func_contract_t* selected,
    const loom_link_func_contract_attr_t* attrs, iree_host_size_t attr_count,
    loom_link_func_contract_apply_mode_t mode,
    loom_link_func_contract_mismatch_t* out_mismatch) {
  for (iree_host_size_t i = 0;
       i < attr_count &&
       !loom_link_func_contract_mismatch_present(out_mismatch);
       ++i) {
    IREE_RETURN_IF_ERROR(loom_link_func_contract_apply_attr(
        remap, source, attrs[i].source_attr_index, selected,
        attrs[i].selected_attr_index, attrs[i].field_name, mode, out_mismatch));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_func_contract_apply(
    const loom_link_func_contract_t* source,
    const loom_link_func_contract_t* selected, loom_ir_remap_t* remap,
    loom_link_func_contract_apply_mode_t mode,
    loom_link_func_contract_merge_flags_t flags,
    loom_link_func_contract_mismatch_t* out_mismatch) {
  *out_mismatch = (loom_link_func_contract_mismatch_t){0};
  IREE_ASSERT(source->module == remap->source_module);
  IREE_ASSERT(selected->module == remap->target_module);

  if (!source->symbol_definition || !selected->symbol_definition ||
      !loom_symbol_definition_satisfies(
          selected->symbol_definition, source->symbol_definition->interfaces)) {
    loom_link_func_contract_mismatch_field(IREE_SV("symbol interfaces"),
                                           out_mismatch);
    return iree_ok_status();
  }
  if (!source->function || !selected->function) {
    loom_link_func_contract_mismatch_field(IREE_SV("func_like"), out_mismatch);
    return iree_ok_status();
  }

  if (loom_symbol_definition_implements(source->symbol_definition,
                                        LOOM_SYMBOL_INTERFACE_KERNEL)) {
    IREE_RETURN_IF_ERROR(loom_link_func_contract_map_values(
        remap, source->workload_arguments, selected->workload_arguments,
        IREE_SV("kernel workload args"), out_mismatch));
    if (!loom_link_func_contract_mismatch_present(out_mismatch)) {
      IREE_RETURN_IF_ERROR(loom_link_func_contract_check_value_types(
          remap, source, source->workload_arguments, selected,
          selected->workload_arguments, IREE_SV("kernel workload arg"),
          out_mismatch));
    }
  }
  if (loom_link_func_contract_mismatch_present(out_mismatch)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_link_func_contract_map_values(
      remap, source->arguments, selected->arguments, IREE_SV("args"),
      out_mismatch));
  if (loom_link_func_contract_mismatch_present(out_mismatch)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_link_func_contract_map_values(
      remap, source->results, selected->results, IREE_SV("results"),
      out_mismatch));
  if (loom_link_func_contract_mismatch_present(out_mismatch)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_link_func_contract_check_value_types(
      remap, source, source->arguments, selected, selected->arguments,
      IREE_SV("arg"), out_mismatch));
  if (loom_link_func_contract_mismatch_present(out_mismatch)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_link_func_contract_check_value_types(
      remap, source, source->results, selected, selected->results,
      IREE_SV("result"), out_mismatch));
  if (loom_link_func_contract_mismatch_present(out_mismatch)) {
    return iree_ok_status();
  }

  if (source->tied_result_count != selected->tied_result_count) {
    loom_link_func_contract_mismatch_count(
        IREE_SV("tied results"), source->tied_result_count,
        selected->tied_result_count, out_mismatch);
    return iree_ok_status();
  }
  if (source->tied_result_count != 0 &&
      memcmp(source->tied_results, selected->tied_results,
             (iree_host_size_t)source->tied_result_count *
                 sizeof(*source->tied_results)) != 0) {
    loom_link_func_contract_mismatch_field(IREE_SV("tied_results"),
                                           out_mismatch);
    return iree_ok_status();
  }

  const loom_link_func_contract_attr_t attrs[] = {
      {
          .source_attr_index = source->function->cc_attr_index,
          .selected_attr_index = selected->function->cc_attr_index,
          .field_name = IREE_SV("cc"),
      },
      {
          .source_attr_index = source->function->purity_attr_index,
          .selected_attr_index = selected->function->purity_attr_index,
          .field_name = IREE_SV("purity"),
      },
      {
          .source_attr_index = source->function->temperature_attr_index,
          .selected_attr_index = selected->function->temperature_attr_index,
          .field_name = IREE_SV("temperature"),
      },
      {
          .source_attr_index = source->function->inline_policy_attr_index,
          .selected_attr_index = selected->function->inline_policy_attr_index,
          .field_name = IREE_SV("inline_policy"),
      },
      {
          .source_attr_index = source->function->target_attr_index,
          .selected_attr_index = selected->function->target_attr_index,
          .field_name = IREE_SV("target"),
      },
      {
          .source_attr_index = source->function->requires_attr_index,
          .selected_attr_index = selected->function->requires_attr_index,
          .field_name = IREE_SV("requires"),
      },
      {
          .source_attr_index = source->function->repr_contract_attr_index,
          .selected_attr_index = selected->function->repr_contract_attr_index,
          .field_name = IREE_SV("repr_contract"),
      },
      {
          .source_attr_index = source->function->abi_attr_index,
          .selected_attr_index = selected->function->abi_attr_index,
          .field_name = IREE_SV("abi"),
      },
      {
          .source_attr_index = source->function->abi_attrs_attr_index,
          .selected_attr_index = selected->function->abi_attrs_attr_index,
          .field_name = IREE_SV("abi_attrs"),
      },
      {
          .source_attr_index = source->function->predicates_attr_index,
          .selected_attr_index = selected->function->predicates_attr_index,
          .field_name = IREE_SV("predicates"),
      },
      {
          .source_attr_index = source->function->template_family_attr_index,
          .selected_attr_index = selected->function->template_family_attr_index,
          .field_name = IREE_SV("implements"),
      },
      {
          .source_attr_index =
              source->function->specialization_count_attr_index,
          .selected_attr_index =
              selected->function->specialization_count_attr_index,
          .field_name = IREE_SV("specialization_count"),
      },
  };
  IREE_RETURN_IF_ERROR(loom_link_func_contract_apply_attrs(
      remap, source, selected, attrs, IREE_ARRAYSIZE(attrs), mode,
      out_mismatch));
  if (loom_link_func_contract_mismatch_present(out_mismatch) ||
      !iree_any_bit_set(flags, LOOM_LINK_FUNC_CONTRACT_MERGE_FLAG_OUTPUT)) {
    return iree_ok_status();
  }

  const loom_link_func_contract_attr_t output_attrs[] = {
      {
          .source_attr_index = source->function->visibility_attr_index,
          .selected_attr_index = selected->function->visibility_attr_index,
          .field_name = IREE_SV("visibility"),
      },
      {
          .source_attr_index = source->function->import_module_attr_index,
          .selected_attr_index = selected->function->import_module_attr_index,
          .field_name = IREE_SV("import_module"),
      },
      {
          .source_attr_index = source->function->import_symbol_attr_index,
          .selected_attr_index = selected->function->import_symbol_attr_index,
          .field_name = IREE_SV("import_symbol"),
      },
      {
          .source_attr_index = source->function->export_symbol_attr_index,
          .selected_attr_index = selected->function->export_symbol_attr_index,
          .field_name = IREE_SV("export_symbol"),
      },
      {
          .source_attr_index = source->function->export_attrs_attr_index,
          .selected_attr_index = selected->function->export_attrs_attr_index,
          .field_name = IREE_SV("export_attrs"),
      },
      {
          .source_attr_index = source->function->export_linkage_attr_index,
          .selected_attr_index = selected->function->export_linkage_attr_index,
          .field_name = IREE_SV("export_linkage"),
      },
  };
  return loom_link_func_contract_apply_attrs(
      remap, source, selected, output_attrs, IREE_ARRAYSIZE(output_attrs), mode,
      out_mismatch);
}

iree_status_t loom_link_func_contract_check(
    const loom_link_func_contract_t* source,
    const loom_link_func_contract_t* selected, loom_ir_remap_t* remap,
    loom_link_func_contract_mismatch_t* out_mismatch) {
  return loom_link_func_contract_apply(source, selected, remap,
                                       LOOM_LINK_FUNC_CONTRACT_APPLY_CHECK,
                                       /*flags=*/0, out_mismatch);
}

iree_status_t loom_link_func_contract_merge(
    const loom_link_func_contract_t* source,
    const loom_link_func_contract_t* selected, loom_ir_remap_t* remap,
    loom_link_func_contract_merge_flags_t flags,
    loom_link_func_contract_mismatch_t* out_mismatch) {
  return loom_link_func_contract_apply(source, selected, remap,
                                       LOOM_LINK_FUNC_CONTRACT_APPLY_MERGE,
                                       flags, out_mismatch);
}
