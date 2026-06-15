// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func_symbol_facts.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

static iree_status_t loom_func_symbol_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t field_name, iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (string_id >= module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "func %.*s string id %u is invalid",
        (int)field_name.size, field_name.data, (uint32_t)string_id);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static bool loom_func_symbol_attr_present(loom_func_like_t func,
                                          uint8_t attr_index) {
  if (attr_index == LOOM_ATTR_INDEX_NONE) {
    return false;
  }
  return !loom_attr_is_absent(loom_op_attrs(func.op)[attr_index]);
}

static iree_status_t loom_func_symbol_string_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    iree_string_view_t field_name, iree_string_view_t* out_string) {
  if (attr->kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func contract field %.*s must be a string",
                            (int)field_name.size, field_name.data);
  }
  return loom_func_symbol_string_from_id(module, loom_attr_as_string_id(*attr),
                                         field_name, out_string);
}

static iree_status_t loom_func_symbol_linkage_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    loom_target_linkage_t* out_linkage) {
  if (attr->kind == LOOM_ATTR_ENUM) {
    *out_linkage = (loom_target_linkage_t)loom_attr_as_enum(*attr);
    return iree_ok_status();
  }
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_func_symbol_string_attr(module, attr, IREE_SV("linkage"), &value));
  if (iree_string_view_equal(value, IREE_SV("default"))) {
    *out_linkage = LOOM_TARGET_LINKAGE_DEFAULT;
  } else if (iree_string_view_equal(value, IREE_SV("dso_local"))) {
    *out_linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func contract linkage '%.*s' is unknown",
                            (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_func_symbol_apply_export_attr(
    const loom_module_t* module, iree_string_view_t name,
    const loom_attribute_t* value, loom_func_symbol_facts_t* facts) {
  if (iree_string_view_equal(name, IREE_SV("linkage"))) {
    IREE_RETURN_IF_ERROR(
        loom_func_symbol_linkage_attr(module, value, &facts->export_linkage));
    facts->has_export_linkage = true;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "func export contract field '%.*s' is not supported", (int)name.size,
        name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_func_symbol_apply_export_attrs(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_func_symbol_facts_t* facts) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_func_symbol_string_from_id(
        module, entry->name_id, IREE_SV("export field"), &name));
    IREE_RETURN_IF_ERROR(
        loom_func_symbol_apply_export_attr(module, name, &entry->value, facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_func_symbol_apply_direct_export_attrs(
    loom_func_like_t func, loom_func_symbol_facts_t* facts) {
  uint8_t export_linkage = 0;
  if (loom_func_like_export_linkage(func, &export_linkage)) {
    facts->export_linkage = (loom_target_linkage_t)export_linkage;
    facts->has_export_linkage = true;
    facts->exports = true;
  }

  return iree_ok_status();
}

static bool loom_func_symbol_is_kernel_entry(loom_func_like_t func) {
  const loom_op_kind_t kernel_def = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 0);
  const loom_op_kind_t low_kernel_def = LOOM_OP_KIND(LOOM_DIALECT_LOW, 1);
  return func.op->kind == kernel_def || func.op->kind == low_kernel_def;
}

static iree_status_t loom_func_symbol_apply_imports(
    const loom_module_t* module, loom_func_like_t func,
    loom_func_symbol_facts_t* facts) {
  loom_string_id_t import_module_id = loom_func_like_import_module(func);
  loom_string_id_t import_symbol_id = loom_func_like_import_symbol(func);
  facts->imports = import_module_id != LOOM_STRING_ID_INVALID ||
                   import_symbol_id != LOOM_STRING_ID_INVALID;
  if (import_module_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_func_symbol_string_from_id(
        module, import_module_id, IREE_SV("import_module"),
        &facts->import_module));
  }
  if (import_symbol_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_func_symbol_string_from_id(
        module, import_symbol_id, IREE_SV("import_symbol"),
        &facts->import_symbol));
  } else if (facts->imports) {
    facts->import_symbol = facts->name;
  }
  return iree_ok_status();
}

static iree_status_t loom_func_symbol_fact_compute(
    const loom_symbol_fact_domain_t* domain,
    loom_symbol_fact_context_t* context, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_t* symbol,
    const loom_symbol_facts_base_t** out_facts) {
  *out_facts = NULL;

  loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "func symbol fact domain attached to a non-FuncLike op");
  }

  loom_func_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_allocate(
      context, sizeof(*facts), (void**)&facts));
  memset(facts, 0, sizeof(*facts));

  facts->base.domain = domain;
  facts->base.symbol_kind = symbol->kind;
  facts->func_op = func.op;
  facts->symbol = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  IREE_RETURN_IF_ERROR(loom_func_symbol_string_from_id(
      module, symbol->name_id, IREE_SV("symbol"), &facts->name));
  facts->visibility = loom_func_like_visibility(func);
  facts->calling_convention = loom_func_like_cc(func);
  facts->purity = loom_func_like_purity(func);
  facts->temperature = loom_func_like_temperature(func);
  facts->inline_policy = loom_func_like_inline_policy(func);
  facts->has_body = loom_func_like_body(func) != NULL;
  facts->implements_id = loom_func_like_implements(func);
  if (facts->implements_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_func_symbol_string_from_id(
        module, facts->implements_id, IREE_SV("implements"),
        &facts->implements));
  }
  facts->priority = loom_func_like_priority(func);
  facts->argument_ids = loom_func_like_arg_ids(func, &facts->argument_count);
  facts->result_ids = loom_op_const_results(func.op);
  facts->result_count = func.op->result_count;
  facts->predicates = loom_func_like_predicates(func, &facts->predicate_count);
  IREE_RETURN_IF_ERROR(loom_func_symbol_apply_imports(module, func, facts));
  facts->target_symbol = loom_func_like_target(func);

  const bool has_abi_attr =
      loom_func_symbol_attr_present(func, func.vtable->abi_attr_index);
  if (has_abi_attr) {
    facts->has_abi = true;
    facts->abi_kind = (loom_target_abi_kind_t)loom_func_like_abi(func);
  } else if (loom_func_symbol_is_kernel_entry(func)) {
    facts->has_abi = true;
    facts->abi_kind = LOOM_TARGET_ABI_HAL_KERNEL;
  }
  facts->abi_attrs = loom_func_like_abi_attrs(func);
  loom_string_id_t export_symbol_id = loom_func_like_export_symbol(func);
  bool has_export_symbol = export_symbol_id != LOOM_STRING_ID_INVALID;
  loom_named_attr_slice_t export_attrs = loom_func_like_export_attrs(func);
  if (has_export_symbol) {
    IREE_RETURN_IF_ERROR(loom_func_symbol_string_from_id(
        module, export_symbol_id, IREE_SV("export_symbol"),
        &facts->export_symbol));
    facts->exports = true;
  }
  IREE_RETURN_IF_ERROR(
      loom_func_symbol_apply_export_attrs(module, export_attrs, facts));
  IREE_RETURN_IF_ERROR(loom_func_symbol_apply_direct_export_attrs(func, facts));
  if (export_attrs.count > 0 || loom_func_symbol_is_kernel_entry(func)) {
    facts->exports = true;
  }

  (void)context;
  *out_facts = &facts->base;
  return iree_ok_status();
}

const loom_symbol_fact_domain_t loom_func_symbol_fact_domain = {
    .compute = loom_func_symbol_fact_compute,
};

const loom_func_symbol_facts_t* loom_func_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts) {
  if (!facts || facts->domain != &loom_func_symbol_fact_domain) {
    return NULL;
  }
  return (const loom_func_symbol_facts_t*)facts;
}
