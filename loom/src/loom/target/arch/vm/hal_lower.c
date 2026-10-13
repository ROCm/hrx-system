// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/hal_lower.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/module/hal/native_schema.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/value_tree.h"
#include "loom/ops/hal/ops.h"
#include "loom/ops/hal/schema.h"
#include "loom/target/arch/vm/hal_tables.inl"

typedef struct loom_vm_hal_state_t {
  // True once all immutable method signatures have been interned.
  bool initialized;
  // Canonical module types indexed by the declared method ordinal.
  loom_type_id_t signatures[LOOM_OP_HAL_COUNT_];
} loom_vm_hal_state_t;

// Declaration types are built once per function, independently of callsite
// cardinalities. Module type interning shares their storage across functions.
static iree_status_t loom_vm_hal_signatures(loom_low_lower_context_t* context,
                                            loom_vm_hal_state_t** out_state) {
  static const char kStateKey = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &kStateKey, sizeof(**out_state), (void**)out_state));
  loom_vm_hal_state_t* state = *out_state;
  if (state->initialized) {
    return iree_ok_status();
  }
  loom_module_t* module = loom_low_lower_context_module(context);
  loom_string_id_t record_name, range_name;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("vm.record"), &record_name));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("vm.range"), &range_name));
  loom_type_t reference_types[IREE_ARRAYSIZE(kReferenceNames)];
  loom_type_t field_types[IREE_ARRAYSIZE(kIreeHalNativeFields)];
  loom_type_t ranges[IREE_ARRAYSIZE(kIreeHalNativeLayouts)];
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < IREE_ARRAYSIZE(kReferenceNames); ++i) {
    loom_string_id_t name;
    status = loom_module_intern_string(module, kReferenceNames[i], &name);
    if (iree_status_is_ok(status)) {
      reference_types[i] = loom_type_dialect_opaque(name);
    }
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < IREE_ARRAYSIZE(kIreeHalNativeLayouts);
       ++i) {
    const iree_vm_native_call_layout_t* layout = &kIreeHalNativeLayouts[i];
    for (uint16_t j = 0; j < layout->field_count; ++j) {
      uint32_t field_index = layout->field_base + j;
      const iree_vm_native_call_field_t* field =
          &kIreeHalNativeFields[field_index];
      field_types[field_index] =
          field->kind == IREE_VM_NATIVE_CALL_FIELD_RANGE
              ? ranges[field->type_ordinal]
          : field->kind == IREE_VM_NATIVE_CALL_FIELD_REF
              ? reference_types[field->type_ordinal]
              : loom_type_scalar(kFieldScalarTypes[field_index]);
    }
    status = loom_module_intern_type(
        module,
        loom_type_dialect(range_name, layout->field_count,
                          &field_types[layout->field_base]),
        &ranges[i]);
  }
  for (uint16_t i = 0; iree_status_is_ok(status) && i < LOOM_OP_HAL_COUNT_;
       ++i) {
    const iree_vm_native_call_layout_t* root =
        &kIreeHalNativeLayouts[kIreeHalNativeSchemas[i].root_ordinal];
    status = loom_module_intern_type_id(
        module,
        loom_type_dialect(record_name, root->field_count,
                          &field_types[root->field_base]),
        &state->signatures[i]);
  }
  state->initialized = iree_status_is_ok(status);
  return status;
}

typedef struct loom_vm_hal_projection_t {
  // Next verified semantic source node, consumed exactly once in preorder.
  const int64_t* shape;
  // Output tree with fixed records flattened and optionals made into ranges.
  int64_t* nodes;
  // Number of populated output tree nodes.
  uint32_t node_count;
} loom_vm_hal_projection_t;

// Recursion is bounded by the immutable declaration, not source cardinality.
// ABI grouping stays symbolic; the native call planner owns binary geometry.
static void loom_vm_hal_project(loom_vm_hal_projection_t* projection,
                                uint16_t node_ordinal) {
  const loom_hal_schema_node_t* node = &loom_hal_schema_nodes[node_ordinal];
  uint32_t token = (uint32_t)*projection->shape;
  if (node->kind == LOOM_HAL_SCHEMA_VALUE) {
    ++projection->shape;
    projection->nodes[projection->node_count++] = LOOM_VALUE_TREE_VALUE;
    return;
  }
  if (node->kind == LOOM_HAL_SCHEMA_RECORD) {
    ++projection->shape;
    for (uint16_t i = 0; i < node->child_count; ++i) {
      uint16_t child = loom_hal_schema_children[node->child_base + i];
      loom_vm_hal_project(projection, child);
    }
    return;
  }
  uint32_t count;
  if (node->kind == LOOM_HAL_SCHEMA_OPTIONAL) {
    count = token != LOOM_VALUE_TREE_ABSENT;
    projection->shape += count == 0;
  } else {
    count = token >> 2;
    ++projection->shape;
  }
  uint16_t child = loom_hal_schema_children[node->child_base];
  projection->nodes[projection->node_count++] =
      (count << 2) | LOOM_VALUE_TREE_RANGE;
  for (uint32_t i = 0; i < count; ++i) {
    if (kFieldCounts[child] != 1) {
      projection->nodes[projection->node_count++] =
          (kFieldCounts[child] << 2) | LOOM_VALUE_TREE_RECORD;
    }
    loom_vm_hal_project(projection, child);
  }
}

typedef struct loom_vm_hal_plan_t {
  // Immutable declaration signature shared by every shape of this method.
  loom_type_id_t signature;
  // Semantic ABI grouping, independent of binary layout and allocation.
  loom_attribute_t shape;
  // Physical carriers required by the native declaration and call.
  loom_type_t* argument_types;
} loom_vm_hal_plan_t;

iree_status_t loom_vm_hal_select(loom_low_lower_context_t* context,
                                 const loom_op_t* source_op,
                                 loom_low_lower_plan_t* out_plan) {
  uint16_t ordinal = loom_op_dialect_index(source_op->kind);
  const loom_hal_method_t* method = &loom_hal_methods[ordinal];
  loom_attribute_t shape = loom_op_const_attrs(source_op)[0];
  const uint32_t capacity = shape.count * LOOM_VM_HAL_MAXIMUM_TOKEN_NODES;
  int64_t* nodes = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_planning_array(
      context, capacity, sizeof(*nodes), (void**)&nodes));
  nodes[0] = (kFieldCounts[method->arguments] << 2) | LOOM_VALUE_TREE_RECORD;
  loom_vm_hal_projection_t projection = {
      .shape = shape.i64_array,
      .nodes = nodes,
      .node_count = 1,
  };
  loom_vm_hal_project(&projection, method->arguments);
  if (projection.node_count > UINT16_MAX) {
    const loom_diagnostic_param_t parameters[] = {
        loom_param_string(loom_low_lower_context_target_key(context)),
        loom_param_string(loom_low_lower_context_export_name(context)),
        loom_param_string(loom_low_lower_context_config_key(context)),
        loom_param_string(loom_low_lower_context_function_name(context)),
        loom_param_string(
            loom_op_name(loom_low_lower_context_module(context), source_op)),
        loom_param_string(IREE_SV("native call grouping must fit 65535 nodes")),
    };
    return loom_low_lower_emit_error_ref(context, source_op,
                                         LOOM_ERR_TARGET_032_REF, parameters,
                                         IREE_ARRAYSIZE(parameters));
  }
  loom_vm_hal_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_vm_hal_signatures(context, &state));
  loom_vm_hal_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  int64_t* retained_nodes = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, projection.node_count, sizeof(*nodes), (void**)&retained_nodes));
  memcpy(retained_nodes, nodes, projection.node_count * sizeof(*nodes));
  plan->signature = state->signatures[ordinal];
  plan->shape = loom_attr_i64_array(retained_nodes, projection.node_count);
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, source_op->operand_count, sizeof(*plan->argument_types),
      (void**)&plan->argument_types));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       iree_status_is_ok(status) && i < source_op->operand_count; ++i) {
    status = loom_low_lower_map_value(context, source_op,
                                      loom_op_const_operands(source_op)[i],
                                      &plan->argument_types[i]);
  }
  *out_plan = loom_low_lower_plan_make(ordinal, plan);
  return status;
}

// A fresh interned spelling cannot collide with an authored symbol. The
// existing string index provides this fact without a module symbol scan.
static iree_status_t loom_vm_hal_add_symbol(loom_module_t* module,
                                            loom_symbol_ref_t* out_symbol) {
  uint32_t discriminator = module->symbols.count;
  char name_storage[32];
  iree_string_view_t name;
  do {
    int length = snprintf(name_storage, sizeof(name_storage), "__hal_%" PRIu32,
                          discriminator++);
    name = iree_make_string_view(name_storage, length);
  } while (loom_module_lookup_string(module, name) != LOOM_STRING_ID_INVALID);
  loom_string_id_t name_id;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  *out_symbol = loom_symbol_ref_null();
  out_symbol->module_id = 0;
  return loom_module_add_symbol(module, name_id, &out_symbol->symbol_id);
}

iree_status_t loom_vm_hal_emit(loom_low_lower_context_t* context,
                               const loom_op_t* source_op,
                               loom_low_lower_plan_t selected) {
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_hal_method_t* method = &loom_hal_methods[selected.id];
  const loom_vm_hal_plan_t* plan = selected.target_data;
  loom_named_attr_t attributes[2];
  const iree_string_view_t names[] = {
      IREE_SVL("signature"),
      IREE_SVL("version"),
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < IREE_ARRAYSIZE(names); ++i) {
    status =
        loom_module_intern_string(module, names[i], &attributes[i].name_id);
  }
  IREE_RETURN_IF_ERROR(status);
  attributes[0].value = loom_attr_type(plan->signature);
  attributes[1].value = loom_attr_i64(method->version);
  loom_string_id_t import_module, code_symbol, descriptor_set;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("hal"), &import_module));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, method->name, &code_symbol));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("vm.core"), &descriptor_set));
  loom_symbol_ref_t callee;
  IREE_RETURN_IF_ERROR(loom_vm_hal_add_symbol(module, &callee));
  loom_builder_t declaration_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &declaration_builder);
  const loom_symbol_ref_t target =
      loom_low_func_def_target(loom_low_lower_context_low_function(context));
  loom_op_t* declaration = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_decl_build(
      &declaration_builder,
      LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_KIND |
          LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_CODE_SYMBOL |
          LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_MODULE |
          (loom_symbol_ref_is_valid(target)
               ? LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_TARGET
               : 0) |
          LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_ABI |
          LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_ABI_LAYOUT,
      0, 0, 0, 0, 0, 0, 0, LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE, code_symbol,
      import_module, descriptor_set, target, LOOM_TARGET_ABI_VM_NATIVE,
      loom_named_attr_slice_empty(),
      (loom_named_attr_slice_t){attributes, IREE_ARRAYSIZE(attributes)},
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), callee,
      plan->argument_types, source_op->operand_count, NULL, 0, NULL, 0, NULL, 0,
      source_op->location, &declaration));
  IREE_RETURN_IF_ERROR(loom_low_lower_publish_function(context, declaration));
  loom_value_id_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, loom_op_const_operands(source_op),
      source_op->operand_count, plan->argument_types, &arguments));
  loom_op_t* call = NULL;
  return loom_low_func_call_build(
      loom_low_lower_context_builder(context),
      LOOM_LOW_FUNC_CALL_BUILD_FLAG_HAS_SHAPE, 0, 0, callee,
      plan->shape.i64_array, plan->shape.count, arguments,
      source_op->operand_count, NULL, 0, NULL, 0, source_op->location, &call);
}
