// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower.h"

#include "loom/codegen/low/lower/source_memory.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/vm/contracts/core.h"
#include "loom/target/arch/vm/contracts/core_lower_rules.h"
#include "loom/target/arch/vm/descriptors/descriptors.h"
#include "loom/target/arch/vm/ops/type_builders.h"
#include "loom/target/arch/vm/types.h"
#include "loom/util/fact_table.h"

static bool loom_vm_source_type_supported(void* user_data,
                                          const loom_module_t* module,
                                          loom_type_t type) {
  loom_type_reference_key_t key;
  return loom_vm_type_reference_key(module, type, &key) ||
         (loom_type_is_scalar(type) &&
          loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_ADDRESS |
                                            LOOM_SCALAR_TYPE_SET_INTEGER |
                                            LOOM_SCALAR_TYPE_SET_FLOAT,
                                        loom_type_element_type(type)));
}

static bool loom_vm_source_vector_carrier_supported(
    void* user_data, const loom_target_facts_t* target_facts,
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_type_t type) {
  (void)target_facts;
  (void)fact_table;
  (void)value_id;
  return loom_vm_source_type_supported(user_data, module, type);
}

static const loom_target_vector_packet_policy_t kVmVectorPacketPolicy = {
    .index_bit_count = 64u,
    .offset_bit_count = 64u,
    .flags = LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_BOUNDARY_PROJECTION |
             LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_SCALAR_BOUNDARY_COMPONENTS,
    // VM has no native vector packets. Keep ordinary vector legalization on
    // the existing scalar-reference path while boundary projection expands
    // its bounded signatures into scalar fields.
    .maximum_unpacketized_bit_count =
        LOOM_TARGET_VECTOR_PACKET_UNBOUNDED_BIT_COUNT,
};

static iree_status_t loom_vm_map_type(void* user_data,
                                      loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_type_t source_type,
                                      loom_type_t* out_low_type) {
  (void)user_data;
  // Both address domains use the target's 64-bit value carrier. Signedness is
  // expressed by the selected operations, not a distinct physical register.
  if (loom_type_is_scalar(source_type) &&
      (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_INDEX ||
       loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_OFFSET)) {
    source_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  }
  if (loom_vm_source_type_supported(
          user_data, loom_low_lower_context_module(context), source_type)) {
    return loom_low_lower_make_typed_register_type(
        context,
        loom_type_is_scalar(source_type) ? VM_CORE_REG_CLASS_ID_VALUE
                                         : VM_CORE_REG_CLASS_ID_REF,
        1, source_type, out_low_type);
  }
  return iree_ok_status();
}

#include "loom/target/arch/vm/contracts/tables.inl"

static iree_status_t loom_vm_map_value(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_value_id_t source_value,
                                       loom_type_t source_type,
                                       loom_type_t* out_low_type) {
  if (!loom_type_is_view(source_type)) {
    return loom_vm_map_type(user_data, context, source_op, source_type,
                            out_low_type);
  }
  const loom_value_fact_table_t* facts =
      loom_low_lower_context_fact_table(context);
  if (facts == NULL) {
    const loom_type_t element = loom_type_element(source_type);
    const loom_module_t* module = loom_low_lower_context_module(context);
    if (loom_value_def_op(loom_module_value(module, source_value)) == NULL &&
        loom_func_like_import_symbol(loom_low_lower_context_source_function(
            context)) != LOOM_STRING_ID_INVALID &&
        loom_type_rank(source_type) == 1 &&
        loom_vm_source_type_supported(
            user_data, loom_low_lower_context_module(context), element) &&
        (!loom_type_is_scalar(element) ||
         loom_type_element_type(element) != LOOM_SCALAR_TYPE_I1)) {
      loom_type_id_t element_id = LOOM_TYPE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_module_intern_type_id(
          loom_low_lower_context_module(context), element, &element_id));
      loom_type_t span_type;
      IREE_RETURN_IF_ERROR(loom_vm_span_type_make(
          loom_low_lower_context_module(context), element_id, &span_type));
      return loom_low_lower_make_typed_register_type(
          context, VM_CORE_REG_CLASS_ID_VALUE, 2, span_type, out_low_type);
    }
    return loom_vm_map_type(user_data, context, source_op, source_type,
                            out_low_type);
  }
  loom_value_fact_view_reference_t reference;
  if (!loom_value_facts_query_view_reference(
          &facts->context, loom_value_fact_table_lookup(facts, source_value),
          &reference)) {
    return iree_ok_status();
  }
  if (reference.buffer_value_id != LOOM_VALUE_ID_INVALID) {
    return loom_low_lower_make_typed_register_type(
        context, VM_CORE_REG_CLASS_ID_REF, 1, loom_type_buffer(), out_low_type);
  }
  if (reference.origin.kind == LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION &&
      reference.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE) {
    return loom_low_lower_make_register_type(
        context, VM_CORE_REG_CLASS_ID_VALUE, 1, out_low_type);
  }
  return iree_ok_status();
}

// The source symbol distinguishes immutable bytes from process value globals.
// Keep that symbol in Low; the module writer assigns its final data ordinal.
static iree_status_t loom_vm_select_op(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (!loom_global_load_isa(source_op)) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_symbol_ref_t symbol = loom_global_load_global(source_op);
  const loom_op_t* definition =
      module->symbols.entries[symbol.symbol_id].defining_op;
  if (loom_global_rodata_def_isa(definition)) {
    loom_type_t result_type;
    IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
        context, VM_CORE_REG_CLASS_ID_REF, 1, loom_type_buffer(),
        &result_type));
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
        context, loom_global_load_result(source_op).values[0], result_type));
    *out_plan = loom_low_lower_plan_make(
        VM_CORE_DESCRIPTOR_REF_BUFFER_RODATA_LOAD, NULL);
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_emit_op(void* user_data,
                                     loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(builder, IREE_SV("rodata_u16"), &name));
  const loom_named_attr_t attr = {
      .name_id = name,
      .value = loom_attr_symbol(loom_global_load_global(source_op)),
  };
  const loom_type_t result_type = loom_low_lower_value_binding_type(
      context, loom_global_load_result(source_op).values[0]);
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor =
          &loom_low_lower_context_descriptor_set(context)->descriptors[plan.id],
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, NULL, 0,
      (loom_named_attr_slice_t){.entries = &attr, .count = 1}, &result_type, 1,
      NULL, 0, source_op->location, &low_op));
  return loom_low_lower_bind_value(context,
                                   loom_global_load_result(source_op).values[0],
                                   loom_op_results(low_op)[0]);
}

// Reference allocations start null on every execution, including loop reuse.
// Coordinates use the descriptor cell width, independent of native pointer
// size.
static iree_status_t loom_vm_initialize_reference_storage(
    loom_low_lower_context_t* context, loom_value_id_t address,
    int64_t byte_length, loom_location_id_t location) {
  const loom_low_descriptor_set_t* set =
      loom_low_lower_context_descriptor_set(context);
  const uint32_t cell_bytes =
      set->reg_classes[VM_CORE_REG_CLASS_ID_REF].alloc_unit_bits / 8;
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      loom_low_lower_context_builder(context), IREE_SV("count_u32"), &name));
  const loom_named_attr_t attr = {
      .name_id = name,
      .value = loom_attr_i64(byte_length / cell_bytes),
  };
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor =
          &set->descriptors[VM_CORE_DESCRIPTOR_REF_REF_STACK_DISCARD_RANGE],
  };
  loom_op_t* op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, &address, 1,
      (loom_named_attr_slice_t){.entries = &attr, .count = 1}, NULL, 0, NULL, 0,
      location, &op);
}

static const loom_low_lower_function_storage_mapping_t kFunctionStorage[] = {
    {LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, LOOM_STORAGE_SPACE_STACK,
     VM_CORE_REG_CLASS_ID_VALUE,
     LOOM_LOW_LOWER_FUNCTION_STORAGE_SOURCE_SCALAR_VIEW},
    {LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, LOOM_STORAGE_SPACE_PRIVATE,
     VM_CORE_REG_CLASS_ID_VALUE,
     LOOM_LOW_LOWER_FUNCTION_STORAGE_SOURCE_REFERENCE_VIEW,
     loom_vm_initialize_reference_storage},
};

typedef struct loom_vm_span_operand_plan_t {
  // Retained root-relative address, including all subview terms.
  loom_low_source_memory_access_plan_t access;
  // Dynamic count source, or INVALID for a static count.
  loom_value_id_t count;
  // Element count when count is INVALID.
  int64_t static_count;
} loom_vm_span_operand_plan_t;

static iree_status_t loom_vm_reject_span_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, iree_string_view_t reason) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_type(loom_module_value_type(module, source_value)),
      loom_param_string(reason),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       loom_error_def_ref(LOOM_ERR_TARGET_097),
                                       params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vm_prepare_span_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value,
    loom_type_t actual_type, loom_type_t required_type, const void** out_plan) {
  (void)user_data;
  (void)actual_type;
  const loom_type_t* payload = loom_type_register_value_type(required_type);
  if (!payload || !loom_vm_span_type_isa(*payload)) {
    return loom_vm_reject_span_operand(
        context, source_op, source_value,
        IREE_SV("the receiving signature is not a native span"));
  }
  const loom_type_t type = loom_module_value_type(
      loom_low_lower_context_module(context), source_value);
  const loom_value_fact_table_t* facts =
      loom_low_lower_context_fact_table(context);
  loom_value_fact_view_reference_t reference;
  if (!loom_value_facts_query_view_reference(
          &facts->context, loom_value_fact_table_lookup(facts, source_value),
          &reference) ||
      reference.buffer_value_id != LOOM_VALUE_ID_INVALID ||
      reference.origin.kind != LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION) {
    return loom_vm_reject_span_operand(
        context, source_op, source_value,
        IREE_SV("VM span arguments require frame-owned view storage"));
  }
  const loom_view_region_table_t* regions = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_context_view_regions(context, &regions));
  loom_vm_span_operand_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  loom_low_source_memory_access_diagnostic_t diagnostic;
  if (!loom_low_source_memory_access_plan_build_view_origin(
          regions, source_value, &plan->access, &diagnostic) ||
      plan->access.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE ||
      plan->access.address_layout !=
          LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR) {
    return loom_vm_reject_span_operand(
        context, source_op, source_value,
        IREE_SV("VM span arguments require a contiguous local view"));
  }
  plan->count = loom_type_dim_is_dynamic_at(type, 0)
                    ? loom_type_dim_value_id_at(type, 0)
                    : LOOM_VALUE_ID_INVALID;
  plan->static_count = plan->count == LOOM_VALUE_ID_INVALID
                           ? loom_type_dim_static_size_at(type, 0)
                           : 0;
  loom_low_lower_require_source_memory_complete_byte_offset_storage(
      context, &plan->access);
  if (plan->count != LOOM_VALUE_ID_INVALID) {
    loom_low_lower_require_source_value_storage(context, plan->count);
  }
  *out_plan = plan;
  return iree_ok_status();
}

static iree_status_t loom_vm_emit_span_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value,
    loom_value_id_t low_value, loom_type_t required_type, const void* plan_ptr,
    loom_value_id_t* out_value) {
  (void)user_data;
  (void)source_value;
  (void)low_value;
  const loom_vm_span_operand_plan_t* plan = plan_ptr;
  loom_value_id_t parts[2];
  IREE_RETURN_IF_ERROR(
      loom_low_lower_materialize_source_memory_complete_byte_offset(
          context, &loom_vm_core_lower_rule_set,
          &loom_vm_core_lower_rule_set
               .source_memory_byte_offset_materializers[0],
          source_op, &plan->access, &parts[0]));
  const loom_value_id_t address_parts[] = {
      loom_low_lower_lookup_value(
          context,
          loom_low_source_memory_access_address_value_id(&plan->access)),
      parts[0],
  };
  const loom_low_lower_resolved_descriptor_t add_descriptor = {
      .descriptor = &loom_low_lower_context_descriptor_set(context)
                         ->descriptors[VM_CORE_DESCRIPTOR_REF_INTEGER_ADD_I64]};
  loom_type_t address_type;
  IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
      context, VM_CORE_REG_CLASS_ID_VALUE, 1,
      loom_type_scalar(LOOM_SCALAR_TYPE_I64), &address_type));
  loom_op_t* address = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &add_descriptor, address_parts, 2, loom_named_attr_slice_empty(),
      &address_type, 1, NULL, 0, source_op->location, &address));
  parts[0] = loom_op_results(address)[0];
  if (plan->count != LOOM_VALUE_ID_INVALID) {
    parts[1] = loom_low_lower_lookup_value(context, plan->count);
  } else {
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        loom_low_lower_context_builder(context), IREE_SV("bits"), &name));
    const loom_named_attr_t attr = {.name_id = name,
                                    .value = loom_attr_i64(plan->static_count)};
    const loom_low_lower_resolved_descriptor_t descriptor = {
        .descriptor = &loom_low_lower_context_descriptor_set(context)
                           ->descriptors[VM_CORE_DESCRIPTOR_REF_CONSTANT_I64]};
    loom_type_t count_type;
    IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
        context, VM_CORE_REG_CLASS_ID_VALUE, 1,
        loom_type_scalar(LOOM_SCALAR_TYPE_I64), &count_type));
    loom_op_t* constant = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
        context, &descriptor,
        (loom_named_attr_slice_t){.entries = &attr, .count = 1}, count_type,
        source_op->location, &constant));
    parts[1] = loom_op_results(constant)[0];
  }
  loom_op_t* concat = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context), parts, 2,
                            required_type, source_op->location, &concat));
  *out_value = loom_op_results(concat)[0];
  return iree_ok_status();
}

static const loom_low_lower_policy_t kPolicy = {
    .name = IREE_SVL("vm-lower"),
    .import_decl_kind = LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE,
    .error_catalog = &loom_error_catalog_core,
    .vector_packet_policy = &kVmVectorPacketPolicy,
    .source_type_supported = {.fn = loom_vm_source_type_supported},
    .source_vector_carrier_supported =
        {.fn = loom_vm_source_vector_carrier_supported},
    .map_type = {.fn = loom_vm_map_type},
    .map_value = {.fn = loom_vm_map_value},
    .control_operand = {.prepare = loom_vm_prepare_span_operand,
                        .emit = loom_vm_emit_span_operand},
    .function_storage = {.mappings = kFunctionStorage,
                         .count = IREE_ARRAYSIZE(kFunctionStorage)},
    .contract = LOOM_VM_CORE_CONTRACT,
    .select_op = {.fn = loom_vm_select_op},
    .emit_op = {.fn = loom_vm_emit_op},
};

void loom_vm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {.contract_set_stable_id = UINT64_C(0x32476f14332ccb19),
       .policy = &kPolicy},
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      kEntries, IREE_ARRAYSIZE(kEntries), out_registry);
}
