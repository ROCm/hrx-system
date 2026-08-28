// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/resources.h"

#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/lower/constants.h"
#include "loom/target/arch/vm/ops/ops.h"
#include "loom/target/arch/vm/records/target_records.h"

typedef uint8_t loom_vm_module_resource_kind_t;
enum loom_vm_module_resource_kind_e {
  LOOM_VM_MODULE_RESOURCE_KIND_NONE = 0,
#define LOOM_VM_MODULE_RESOURCE_ROW(kind, load, store, store_preserve) \
  LOOM_VM_MODULE_RESOURCE_KIND_##kind,
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_MODULE_RESOURCE_ROW
  LOOM_VM_MODULE_RESOURCE_KIND_COUNT_,
};
static_assert(LOOM_VM_MODULE_RESOURCE_KIND_COUNT_ <= UINT8_MAX,
              "VM module resource kinds must fit in eight bits");

typedef struct loom_vm_module_resource_entry_t {
  // Dense ordinal in the kind-selected physical resource domain.
  uint16_t ordinal;
  // Physical resource kind determining its bank and mutability partition.
  loom_vm_module_resource_kind_t kind;
  // Reserved for future resource-plan flags.
  uint8_t reserved;
} loom_vm_module_resource_entry_t;
static_assert(sizeof(loom_vm_module_resource_entry_t) == 4,
              "VM module resource entries must remain four bytes");

typedef struct loom_vm_module_resource_plan_t {
  // True once all symbol entries have been populated.
  bool initialized;
  // Number of source symbol slots covered by |entries_by_symbol|.
  iree_host_size_t symbol_count;
  // Resource entries indexed directly by source module symbol ID.
  loom_vm_module_resource_entry_t* entries_by_symbol;
} loom_vm_module_resource_plan_t;

typedef struct loom_vm_module_resource_descriptors_t {
  // Descriptor ordinal selected for a global load.
  uint16_t load;
  // Descriptor ordinal selected for a global store, or UINT16_MAX.
  uint16_t store;
  // Descriptor preserving a non-consuming store operand, or UINT16_MAX.
  uint16_t store_preserve;
} loom_vm_module_resource_descriptors_t;

static const loom_vm_module_resource_descriptors_t
    kVmModuleResourceDescriptors[LOOM_VM_MODULE_RESOURCE_KIND_COUNT_] = {
        {UINT16_MAX, UINT16_MAX, UINT16_MAX},
#define LOOM_VM_MODULE_RESOURCE_ROW(kind, load, store, store_preserve) \
  {load, store, store_preserve},
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_MODULE_RESOURCE_ROW
};

static const loom_vm_module_resource_kind_t kVmGlobalResourceKindPairs[][2] = {
    {LOOM_VM_MODULE_RESOURCE_KIND_VALUE_IMMUTABLE,
     LOOM_VM_MODULE_RESOURCE_KIND_VALUE_MUTABLE},
    {LOOM_VM_MODULE_RESOURCE_KIND_REF_IMMUTABLE,
     LOOM_VM_MODULE_RESOURCE_KIND_REF_MUTABLE},
    {LOOM_VM_MODULE_RESOURCE_KIND_FUNCTION_IMMUTABLE,
     LOOM_VM_MODULE_RESOURCE_KIND_FUNCTION_MUTABLE},
};

static const char loom_vm_module_resource_plan_key;

static loom_vm_module_resource_kind_t loom_vm_module_global_resource_kind(
    const loom_module_t* module, const loom_op_t* op) {
  const bool is_immutable = loom_global_constant_isa(op);
  loom_value_id_t global_value = LOOM_VALUE_ID_INVALID;
  if (is_immutable) {
    global_value = loom_global_constant_type(op);
  } else if (loom_global_variable_isa(op)) {
    global_value = loom_global_variable_type(op);
  } else {
    return LOOM_VM_MODULE_RESOURCE_KIND_NONE;
  }

  loom_vm_call_abi_bank_t bank = LOOM_VM_CALL_ABI_BANK_NONE;
  const loom_type_t type = loom_module_value_type(module, global_value);
  if (!loom_vm_call_abi_try_classify_logical_type(module, type, &bank)) {
    return LOOM_VM_MODULE_RESOURCE_KIND_NONE;
  }
  switch (bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE:
      return is_immutable ? LOOM_VM_MODULE_RESOURCE_KIND_VALUE_IMMUTABLE
                          : LOOM_VM_MODULE_RESOURCE_KIND_VALUE_MUTABLE;
    case LOOM_VM_CALL_ABI_BANK_REF:
      return is_immutable ? LOOM_VM_MODULE_RESOURCE_KIND_REF_IMMUTABLE
                          : LOOM_VM_MODULE_RESOURCE_KIND_REF_MUTABLE;
    case LOOM_VM_CALL_ABI_BANK_FUNCTION:
      return is_immutable ? LOOM_VM_MODULE_RESOURCE_KIND_FUNCTION_IMMUTABLE
                          : LOOM_VM_MODULE_RESOURCE_KIND_FUNCTION_MUTABLE;
    default:
      return LOOM_VM_MODULE_RESOURCE_KIND_NONE;
  }
}

static loom_vm_module_resource_kind_t loom_vm_module_resource_kind(
    const loom_module_t* module, const loom_op_t* op) {
  if (loom_global_rodata_def_isa(op)) {
    return LOOM_VM_MODULE_RESOURCE_KIND_RODATA;
  }
  return loom_vm_module_global_resource_kind(module, op);
}

static iree_status_t loom_vm_module_resource_plan_build(
    const loom_module_t* module, loom_low_lower_module_state_t* module_state,
    loom_vm_module_resource_plan_t* plan) {
  plan->symbol_count = module->symbols.count;
  if (plan->symbol_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_module_state_allocate_array(
        module_state, plan->symbol_count, sizeof(*plan->entries_by_symbol),
        (void**)&plan->entries_by_symbol));
    memset(plan->entries_by_symbol, 0,
           plan->symbol_count * sizeof(*plan->entries_by_symbol));
  }

  uint32_t resource_counts[LOOM_VM_MODULE_RESOURCE_KIND_COUNT_] = {0};
  for (iree_host_size_t i = 0; i < plan->symbol_count; ++i) {
    const loom_op_t* defining_op = module->symbols.entries[i].defining_op;
    if (defining_op == NULL) continue;
    const loom_vm_module_resource_kind_t kind =
        loom_vm_module_resource_kind(module, defining_op);
    if (kind == LOOM_VM_MODULE_RESOURCE_KIND_NONE) continue;
    plan->entries_by_symbol[i].kind = kind;
    if (resource_counts[kind] > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM module resource count exceeds the u16 ordinal domain");
    }
    ++resource_counts[kind];
  }

  uint32_t next_ordinals[LOOM_VM_MODULE_RESOURCE_KIND_COUNT_] = {0};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kVmGlobalResourceKindPairs);
       ++i) {
    const loom_vm_module_resource_kind_t immutable_kind =
        kVmGlobalResourceKindPairs[i][0];
    const loom_vm_module_resource_kind_t mutable_kind =
        kVmGlobalResourceKindPairs[i][1];
    if (resource_counts[immutable_kind] + resource_counts[mutable_kind] >
        UINT16_MAX + UINT32_C(1)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM global count exceeds the u16 ordinal domain");
    }
    next_ordinals[mutable_kind] = resource_counts[immutable_kind];
  }

  for (iree_host_size_t i = 0; i < plan->symbol_count; ++i) {
    loom_vm_module_resource_entry_t* entry = &plan->entries_by_symbol[i];
    const loom_vm_module_resource_kind_t kind = entry->kind;
    if (kind == LOOM_VM_MODULE_RESOURCE_KIND_NONE) continue;
    entry->ordinal = (uint16_t)next_ordinals[kind]++;
  }
  plan->initialized = true;
  return iree_ok_status();
}

static iree_status_t loom_vm_module_resource_plan_get(
    const loom_module_t* module, loom_low_lower_module_state_t* module_state,
    loom_vm_module_resource_plan_t** out_plan) {
  *out_plan = NULL;
  loom_vm_module_resource_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_module_state_get_or_allocate(
      module_state, &loom_vm_module_resource_plan_key, sizeof(*plan),
      (void**)&plan));
  if (!plan->initialized) {
    IREE_RETURN_IF_ERROR(
        loom_vm_module_resource_plan_build(module, module_state, plan));
  }
  *out_plan = plan;
  return iree_ok_status();
}

static loom_symbol_ref_t loom_vm_module_resource_source_ref(
    const loom_op_t* source_op) {
  if (loom_global_load_isa(source_op)) {
    return loom_global_load_global(source_op);
  }
  if (loom_global_store_isa(source_op)) {
    return loom_global_store_global(source_op);
  }
  return loom_symbol_ref_null();
}

static iree_status_t loom_vm_module_resource_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_vm_target_bundle_is_core(
          loom_target_low_legality_bundle(context))) {
    return iree_ok_status();
  }
  if (!loom_global_load_isa(source_op) && !loom_global_store_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_symbol_ref_t source_ref =
      loom_vm_module_resource_source_ref(source_op);
  if (!loom_symbol_ref_is_valid(source_ref) || source_ref.module_id != 0 ||
      source_ref.symbol_id >= module->symbols.count) {
    return iree_ok_status();
  }
  const loom_op_t* defining_op =
      module->symbols.entries[source_ref.symbol_id].defining_op;
  if (defining_op == NULL) return iree_ok_status();
  const loom_vm_module_resource_kind_t kind =
      loom_vm_module_resource_kind(module, defining_op);
  if (kind == LOOM_VM_MODULE_RESOURCE_KIND_NONE) return iree_ok_status();
  if (loom_global_store_isa(source_op) &&
      kVmModuleResourceDescriptors[kind].store == UINT16_MAX) {
    return iree_ok_status();
  }
  *out_handled = true;
  return iree_ok_status();
}

const loom_target_low_legality_provider_t
    loom_vm_module_resource_low_legality_provider = {
        .name = IREE_SVL("vm-module-resources"),
        .builtin_dialect_bits = UINT64_C(1) << LOOM_DIALECT_GLOBAL,
        .try_verify_op = loom_vm_module_resource_try_verify_op,
};

iree_status_t loom_vm_module_resource_try_select_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (!loom_global_load_isa(source_op) && !loom_global_store_isa(source_op)) {
    return iree_ok_status();
  }

  loom_vm_module_resource_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(loom_vm_module_resource_plan_get(
      loom_low_lower_context_module(context),
      loom_low_lower_context_module_state(context), &plan));
  const loom_symbol_ref_t source_ref =
      loom_vm_module_resource_source_ref(source_op);
  if (!loom_symbol_ref_is_valid(source_ref) || source_ref.module_id != 0 ||
      source_ref.symbol_id >= plan->symbol_count) {
    return iree_ok_status();
  }
  const loom_vm_module_resource_entry_t* entry =
      &plan->entries_by_symbol[source_ref.symbol_id];
  if (entry->kind == LOOM_VM_MODULE_RESOURCE_KIND_NONE) {
    return iree_ok_status();
  }

  const loom_vm_module_resource_descriptors_t descriptors =
      kVmModuleResourceDescriptors[entry->kind];
  const uint16_t descriptor_ordinal =
      loom_global_load_isa(source_op) ? descriptors.load : descriptors.store;
  if (descriptor_ordinal == UINT16_MAX) return iree_ok_status();
  *out_plan = loom_low_lower_plan_make(descriptor_ordinal, entry);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_resource_preserve_store_operand(
    loom_low_lower_context_t* context, uint16_t descriptor_ordinal,
    loom_value_id_t source_operand, loom_location_id_t location,
    loom_value_id_t* out_preserved_operand) {
  *out_preserved_operand = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  const loom_low_lower_resolved_descriptor_t resolved_descriptor = {
      .descriptor = descriptor,
  };
  const loom_type_t result_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_operand);
  loom_op_t* preserve_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_descriptor, &source_operand, /*operand_count=*/1,
      loom_named_attr_slice_empty(), &result_type, /*result_count=*/1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &preserve_op));
  *out_preserved_operand = loom_op_const_results(preserve_op)[0];
  return iree_ok_status();
}

iree_status_t loom_vm_module_resource_emit_op(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan,
                                              bool* out_handled) {
  *out_handled =
      loom_global_load_isa(source_op) || loom_global_store_isa(source_op);
  if (!*out_handled) return iree_ok_status();

  const loom_vm_module_resource_entry_t* entry =
      (const loom_vm_module_resource_entry_t*)plan.target_data;
  IREE_ASSERT(entry != NULL);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, (uint32_t)plan.id);
  IREE_ASSERT(descriptor != NULL);
  IREE_ASSERT_EQ(descriptor->immediate_count, 1);

  loom_value_id_t low_operand = LOOM_VALUE_ID_INVALID;
  if (source_op->operand_count != 0) {
    IREE_ASSERT_EQ(source_op->operand_count, 1);
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, loom_op_const_operands(source_op)[0], &low_operand));
    const uint16_t preserve_descriptor =
        kVmModuleResourceDescriptors[entry->kind].store_preserve;
    if (preserve_descriptor != UINT16_MAX) {
      IREE_RETURN_IF_ERROR(loom_vm_module_resource_preserve_store_operand(
          context, preserve_descriptor, low_operand, source_op->location,
          &low_operand));
    }
  }
  loom_type_t low_result_type = loom_type_none();
  if (source_op->result_count != 0) {
    IREE_ASSERT_EQ(source_op->result_count, 1);
    IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
        context, source_op, loom_op_const_results(source_op)[0],
        &low_result_type));
  }

  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  loom_named_attr_t ordinal_attr = {
      .value = loom_attr_i64(entry->ordinal),
  };
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      loom_low_lower_context_builder(context),
      loom_low_descriptor_set_string(descriptor_set,
                                     immediate->field_name_string_offset),
      &ordinal_attr.name_id));
  const loom_low_lower_resolved_descriptor_t resolved_descriptor = {
      .descriptor = descriptor,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_descriptor,
      source_op->operand_count != 0 ? &low_operand : NULL,
      source_op->operand_count, loom_make_named_attr_slice(&ordinal_attr, 1),
      source_op->result_count != 0 ? &low_result_type : NULL,
      source_op->result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_op));
  if (source_op->result_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, loom_op_const_results(source_op)[0],
                                  loom_op_const_results(low_op)[0]));
  }
  return iree_ok_status();
}

static bool loom_vm_module_resource_inline_initializer(
    const loom_op_t* source_op, loom_value_id_t* out_type_value,
    loom_attribute_t* out_initializer) {
  if (loom_global_constant_isa(source_op)) {
    *out_type_value = loom_global_constant_type(source_op);
    *out_initializer = loom_global_constant_initializer(source_op);
  } else if (loom_global_variable_isa(source_op)) {
    *out_type_value = loom_global_variable_type(source_op);
    *out_initializer = loom_global_variable_initializer(source_op);
  } else {
    return false;
  }
  return !loom_attr_is_absent(*out_initializer);
}

static iree_status_t loom_vm_module_resource_build_value_store(
    loom_builder_t* builder, uint16_t descriptor_ordinal,
    uint16_t global_ordinal, loom_value_id_t value,
    loom_location_id_t location) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_vm_core_descriptor_set();
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  IREE_ASSERT_EQ(descriptor->immediate_count, 1u);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  loom_named_attr_t ordinal_attr = {
      .value = loom_attr_i64(global_ordinal),
  };
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      builder,
      loom_low_descriptor_set_string(descriptor_set,
                                     immediate->field_name_string_offset),
      &ordinal_attr.name_id));
  loom_op_t* store_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, &value, /*operand_count=*/1,
      loom_make_named_attr_slice(&ordinal_attr, 1), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &store_op);
}

iree_status_t loom_vm_module_resources_emit_initializer_preamble(
    loom_low_lower_context_t* context) {
  loom_module_t* module = loom_low_lower_context_module(context);
  loom_vm_module_resource_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(loom_vm_module_resource_plan_get(
      module, loom_low_lower_context_module_state(context), &plan));

  for (iree_host_size_t i = 0; i < plan->symbol_count; ++i) {
    const loom_op_t* source_op = module->symbols.entries[i].defining_op;
    if (source_op == NULL) continue;
    loom_value_id_t type_value = LOOM_VALUE_ID_INVALID;
    loom_attribute_t initializer = loom_attr_absent();
    if (!loom_vm_module_resource_inline_initializer(source_op, &type_value,
                                                    &initializer)) {
      continue;
    }

    const loom_type_t source_type = loom_module_value_type(module, type_value);
    loom_type_t low_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_low_lower_map_type(context, source_op, source_type, &low_type));
    if (loom_type_kind(low_type) == LOOM_TYPE_NONE) continue;

    const loom_vm_module_resource_entry_t entry = plan->entries_by_symbol[i];
    if (entry.kind != LOOM_VM_MODULE_RESOURCE_KIND_VALUE_IMMUTABLE &&
        entry.kind != LOOM_VM_MODULE_RESOURCE_KIND_VALUE_MUTABLE) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "VM scalar initializer did not map to a value global");
    }
    const uint64_t bits = loom_vm_constant_bits_from_scalar_attr(
        loom_type_element_type(source_type), initializer);
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vm_inline_constant_build(
        loom_low_lower_context_builder(context), bits, low_type,
        source_op->location, &value));
    IREE_RETURN_IF_ERROR(loom_vm_module_resource_build_value_store(
        loom_low_lower_context_builder(context),
        kVmModuleResourceDescriptors[entry.kind].store, entry.ordinal, value,
        source_op->location));
  }
  return iree_ok_status();
}

iree_status_t loom_vm_module_resources_finalize(
    loom_module_t* module, loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena) {
  (void)scratch_arena;
  loom_vm_module_resource_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vm_module_resource_plan_get(module, module_state, &plan));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  for (iree_host_size_t i = 0; i < plan->symbol_count; ++i) {
    const loom_vm_module_resource_entry_t entry = plan->entries_by_symbol[i];
    if (entry.kind == LOOM_VM_MODULE_RESOURCE_KIND_NONE) continue;
    const loom_symbol_ref_t source_ref = {
        .module_id = 0,
        .symbol_id = (loom_symbol_id_t)i,
    };
    const loom_op_t* source_op = module->symbols.entries[i].defining_op;
    IREE_ASSERT(source_op != NULL);
    loom_op_t* resource_op = NULL;
    if (entry.kind == LOOM_VM_MODULE_RESOURCE_KIND_RODATA) {
      IREE_RETURN_IF_ERROR(
          loom_vm_rodata_build(&builder, source_ref, entry.ordinal,
                               source_op->location, &resource_op));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_vm_global_build(&builder, source_ref, entry.ordinal,
                               source_op->location, &resource_op));
    }
  }
  return iree_ok_status();
}
