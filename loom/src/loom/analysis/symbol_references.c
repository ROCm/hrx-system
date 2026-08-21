// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_references.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/template/ops.h"

typedef struct loom_symbol_reference_builder_t {
  // Module being scanned.
  const loom_module_t* module;
  // Arena receiving table storage.
  iree_arena_allocator_t* arena;
  // Mutable per-symbol occurrence heads.
  loom_symbol_reference_symbol_occurrences_t* symbols;
  // Mutable occurrence storage.
  loom_symbol_reference_occurrence_t* occurrences;
  // Number of live occurrence entries.
  iree_host_size_t occurrence_count;
  // Number of allocated occurrence slots.
  iree_host_size_t occurrence_capacity;
  // First module-root occurrence.
  loom_symbol_reference_occurrence_id_t first_module_occurrence_id;
  // Number of module-root occurrences.
  uint32_t module_occurrence_count;
  // Mutable template-demand storage and family summary.
  struct {
    // Demand entries.
    loom_template_demand_t* values;
    // Number of live entries.
    iree_host_size_t count;
    // Number of allocated entries.
    iree_host_size_t capacity;
    // Dense bitset indexed by module symbol ID for demanded families.
    uint64_t* family_bits;
  } template_demands;
} loom_symbol_reference_builder_t;

static void loom_symbol_reference_initialize_symbol_occurrences(
    loom_symbol_reference_symbol_occurrences_t* symbols,
    iree_host_size_t symbol_count) {
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    symbols[i] = (loom_symbol_reference_symbol_occurrences_t){
        .first_outgoing_occurrence_id =
            LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID,
        .first_incoming_occurrence_id =
            LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID,
        .first_template_demand_id = LOOM_TEMPLATE_DEMAND_ID_INVALID,
    };
  }
}

static iree_status_t loom_symbol_reference_builder_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_symbol_reference_builder_t* builder) {
  *builder = (loom_symbol_reference_builder_t){
      .module = module,
      .arena = arena,
      .first_module_occurrence_id = LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID,
  };
  if (module->symbols.count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*builder->symbols),
                                                 (void**)&builder->symbols));
  loom_symbol_reference_initialize_symbol_occurrences(builder->symbols,
                                                      module->symbols.count);
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_builder_append_occurrence(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    loom_symbol_id_t target_symbol_id,
    loom_symbol_reference_occurrence_kind_t kind,
    loom_symbol_reference_role_t role, uint16_t attr_index,
    const loom_op_t* user_op) {
  if (source_symbol_id != LOOM_SYMBOL_ID_INVALID &&
      source_symbol_id >= builder->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol id %u is out of range for %" PRIhsz " symbols",
        (unsigned)source_symbol_id, builder->module->symbols.count);
  }
  if (target_symbol_id >= builder->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target symbol id %u is out of range for %" PRIhsz " symbols",
        (unsigned)target_symbol_id, builder->module->symbols.count);
  }
  if (builder->occurrence_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "symbol reference table exceeds %u occurrences",
                            (unsigned)(UINT32_MAX - 1));
  }
  if (builder->occurrence_count >= builder->occurrence_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->arena, builder->occurrence_count,
        builder->occurrence_count + 1, sizeof(*builder->occurrences),
        &builder->occurrence_capacity, (void**)&builder->occurrences));
  }

  const loom_symbol_reference_occurrence_id_t occurrence_id =
      (loom_symbol_reference_occurrence_id_t)builder->occurrence_count++;
  loom_symbol_reference_occurrence_t* occurrence =
      &builder->occurrences[occurrence_id];
  *occurrence = (loom_symbol_reference_occurrence_t){
      .source_symbol_id = source_symbol_id,
      .target_symbol_id = target_symbol_id,
      .kind = kind,
      .role = role,
      .attr_index = attr_index,
      .user_op = user_op,
      .next_outgoing_occurrence_id =
          LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID,
      .next_incoming_occurrence_id =
          builder->symbols[target_symbol_id].first_incoming_occurrence_id,
  };
  builder->symbols[target_symbol_id].first_incoming_occurrence_id =
      occurrence_id;
  ++builder->symbols[target_symbol_id].incoming_count;

  if (source_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    occurrence->next_outgoing_occurrence_id =
        builder->first_module_occurrence_id;
    builder->first_module_occurrence_id = occurrence_id;
    ++builder->module_occurrence_count;
    return iree_ok_status();
  }
  occurrence->next_outgoing_occurrence_id =
      builder->symbols[source_symbol_id].first_outgoing_occurrence_id;
  builder->symbols[source_symbol_id].first_outgoing_occurrence_id =
      occurrence_id;
  ++builder->symbols[source_symbol_id].outgoing_count;
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_add_ref(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    loom_symbol_ref_t target_ref, loom_symbol_reference_occurrence_kind_t kind,
    loom_symbol_reference_role_t role, uint16_t attr_index,
    const loom_op_t* user_op) {
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0) {
    return iree_ok_status();
  }
  return loom_symbol_reference_builder_append_occurrence(
      builder, source_symbol_id, target_ref.symbol_id, kind, role, attr_index,
      user_op);
}

static iree_status_t loom_symbol_reference_append_template_demand(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    const loom_op_t* apply_op) {
  if (source_symbol_id >= builder->module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template.apply is not owned by a module symbol");
  }
  const loom_symbol_ref_t family = loom_template_apply_family(apply_op);
  if (!loom_symbol_ref_is_valid(family) || family.module_id != 0 ||
      family.symbol_id >= builder->module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template.apply has an invalid family symbol");
  }
  if (builder->template_demands.count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "symbol reference table exceeds %u abstract "
                            "provider demands",
                            (unsigned)(UINT32_MAX - 1));
  }
  if (builder->template_demands.count >= builder->template_demands.capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(builder->arena, builder->template_demands.count,
                              builder->template_demands.count + 1,
                              sizeof(*builder->template_demands.values),
                              &builder->template_demands.capacity,
                              (void**)&builder->template_demands.values));
  }
  if (!builder->template_demands.family_bits) {
    iree_host_size_t rounded_symbol_count = 0;
    if (!iree_host_size_checked_add(builder->module->symbols.count, 63,
                                    &rounded_symbol_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "template demand bitmap size overflow");
    }
    const iree_host_size_t word_count = rounded_symbol_count / 64;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        iree_arena_allocator(builder->arena), word_count,
        sizeof(*builder->template_demands.family_bits),
        (void**)&builder->template_demands.family_bits));
  }

  const loom_template_demand_id_t demand_id =
      (loom_template_demand_id_t)builder->template_demands.count++;
  loom_symbol_reference_symbol_occurrences_t* source =
      &builder->symbols[source_symbol_id];
  builder->template_demands.values[demand_id] = (loom_template_demand_t){
      .family_symbol_id = family.symbol_id,
      .source_symbol_id = source_symbol_id,
      .apply_op = apply_op,
      .next_source_demand_id = source->first_template_demand_id,
  };
  source->first_template_demand_id = demand_id;
  ++source->template_demand_count;
  builder->template_demands.family_bits[family.symbol_id >> 6] |=
      UINT64_C(1) << (family.symbol_id & 63u);
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_type(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    loom_type_t type, loom_symbol_reference_occurrence_kind_t kind,
    uint16_t attr_index, const loom_op_t* user_op);

static iree_status_t loom_symbol_reference_visit_attr(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_symbol_reference_occurrence_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op, uint8_t dict_depth);

static iree_status_t loom_symbol_reference_visit_encoding(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    const loom_encoding_t* encoding,
    loom_symbol_reference_occurrence_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op) {
  if (!encoding || encoding->attribute_count == 0) return iree_ok_status();
  if (!encoding->attributes) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty encoding attribute list has a NULL entry pointer");
  }
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_attr(
        builder, source_symbol_id, encoding->attributes[i].value,
        /*descriptor=*/NULL, kind, attr_index, user_op, /*dict_depth=*/0));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_static_encoding(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    uint16_t encoding_id, loom_symbol_reference_occurrence_kind_t kind,
    uint16_t attr_index, const loom_op_t* user_op) {
  if (encoding_id == 0) return iree_ok_status();
  const loom_encoding_t* encoding =
      loom_module_encoding(builder->module, encoding_id);
  if (!encoding) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "static encoding id %u is out of range for module with %" PRIhsz
        " encodings",
        (unsigned)encoding_id, builder->module->encodings.count);
  }
  return loom_symbol_reference_visit_encoding(
      builder, source_symbol_id, encoding, kind, attr_index, user_op);
}

static iree_status_t loom_symbol_reference_visit_type_sequence(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    const loom_type_t* types, uint16_t type_count,
    loom_symbol_reference_occurrence_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op) {
  if (!types) return iree_ok_status();
  for (uint16_t i = 0; i < type_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_type(
        builder, source_symbol_id, types[i], kind, attr_index, user_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_type(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    loom_type_t type, loom_symbol_reference_occurrence_kind_t kind,
    uint16_t attr_index, const loom_op_t* user_op) {
  loom_type_kind_t type_kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(type_kind)) return iree_ok_status();

  if (loom_type_has_static_encoding(type)) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_static_encoding(
        builder, source_symbol_id, type.encoding_id, kind, attr_index,
        user_op));
  }

  switch (type_kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return iree_ok_status();
      return loom_symbol_reference_visit_type_sequence(
          builder, source_symbol_id, data->types,
          (uint16_t)(data->arg_count + data->result_count), kind, attr_index,
          user_op);
    }
    case LOOM_TYPE_DIALECT:
      return loom_symbol_reference_visit_type_sequence(
          builder, source_symbol_id, loom_type_dialect_params(type),
          loom_type_dialect_param_count(type), kind, attr_index, user_op);
    case LOOM_TYPE_PARAMETERIZED: {
      const loom_parameterized_type_descriptor_t* descriptor =
          loom_type_parameterized_descriptor(type);
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      uint8_t parameter_count = loom_type_parameterized_parameter_count(type);
      for (uint8_t i = 0; i < parameter_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_attr(
            builder, source_symbol_id, parameters[i],
            &descriptor->parameter_descriptors[i], kind, attr_index, user_op,
            /*dict_depth=*/1));
      }
      return iree_ok_status();
    }
    case LOOM_TYPE_REGISTER: {
      const loom_type_t* value_type = loom_type_register_value_type(type);
      return value_type ? loom_symbol_reference_visit_type(
                              builder, source_symbol_id, *value_type, kind,
                              attr_index, user_op)
                        : iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_symbol_reference_visit_attr(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_symbol_reference_occurrence_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op, uint8_t dict_depth) {
  switch ((loom_attr_kind_t)attr.kind) {
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_reference_role_t role = LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY;
      if (descriptor && descriptor->attr_kind == LOOM_ATTR_SYMBOL &&
          descriptor->reference.symbol_ref) {
        role = descriptor->reference.symbol_ref->role;
      }
      return loom_symbol_reference_add_ref(builder, source_symbol_id,
                                           loom_attr_as_symbol(attr), kind,
                                           role, attr_index, user_op);
    }
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      loom_symbol_reference_role_t role = LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY;
      if (descriptor && descriptor->attr_kind == attr.kind &&
          descriptor->reference.symbol_ref) {
        role = descriptor->reference.symbol_ref->role;
      }
      loom_symbol_ref_array_t refs = attr.kind == LOOM_ATTR_SYMBOL_SET
                                         ? loom_attr_as_symbol_set(attr)
                                         : loom_attr_as_symbol_array(attr);
      for (uint16_t i = 0; i < refs.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_symbol_reference_add_ref(
            builder, source_symbol_id, refs.values[i], kind, role, attr_index,
            user_op));
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_TYPE:
      if (attr.type_id == LOOM_TYPE_ID_INVALID) return iree_ok_status();
      if (attr.type_id >= builder->module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute id %u is out of range for module with %" PRIhsz
            " types",
            (unsigned)attr.type_id, builder->module->types.count);
      }
      return loom_symbol_reference_visit_type(
          builder, source_symbol_id,
          builder->module->types.entries[attr.type_id],
          LOOM_SYMBOL_REFERENCE_OCCURRENCE_TYPE_ATTR, attr_index, user_op);
    case LOOM_ATTR_ENCODING:
      return loom_symbol_reference_visit_static_encoding(
          builder, source_symbol_id, attr.encoding_id,
          LOOM_SYMBOL_REFERENCE_OCCURRENCE_ENCODING_ATTR, attr_index, user_op);
    case LOOM_ATTR_DICT:
      if (dict_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dict attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      if (attr.count == 0) return iree_ok_status();
      if (!attr.dict_entries) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty dict attribute has a NULL entry pointer");
      }
      loom_symbol_reference_occurrence_kind_t nested_kind = kind;
      if (kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR ||
          kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL ||
          kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_GLOBAL_ACCESS) {
        nested_kind = LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR;
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_attr(
            builder, source_symbol_id, attr.dict_entries[i].value,
            /*descriptor=*/NULL, nested_kind, attr_index, user_op,
            (uint8_t)(dict_depth + 1)));
      }
      return iree_ok_status();
    case LOOM_ATTR_PARAMETERIZED:
      if (dict_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      const loom_parameterized_attr_descriptor_t* family_descriptor =
          loom_context_resolve_parameterized_attr(
              builder->module->context, loom_attr_as_parameterized_kind(attr));
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_attr(
            builder, source_symbol_id, attr.parameterized_slots[i],
            &family_descriptor->parameter_descriptors[i],
            LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR, attr_index, user_op,
            (uint8_t)(dict_depth + 1)));
      }
      return iree_ok_status();
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (dict_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_attr(
            builder, source_symbol_id, attr.parameterized_array[i],
            /*descriptor=*/NULL, LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR,
            attr_index, user_op, (uint8_t)(dict_depth + 1)));
      }
      return iree_ok_status();
    default:
      return iree_ok_status();
  }
}

static loom_symbol_reference_occurrence_kind_t
loom_symbol_reference_direct_attr_kind(const loom_op_vtable_t* vtable,
                                       const loom_attr_descriptor_t* descriptor,
                                       uint8_t attr_index) {
  if (vtable && vtable->call_like &&
      attr_index == vtable->call_like->callee_attr_index) {
    return LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL;
  }
  if (descriptor && descriptor->attr_kind == LOOM_ATTR_SYMBOL &&
      descriptor->reference.symbol_ref &&
      iree_any_bit_set(descriptor->reference.symbol_ref->interfaces,
                       LOOM_SYMBOL_INTERFACE_GLOBAL)) {
    return LOOM_SYMBOL_REFERENCE_OCCURRENCE_GLOBAL_ACCESS;
  }
  return LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR;
}

static iree_status_t loom_symbol_reference_visit_value_type(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    loom_value_id_t value_id, const loom_op_t* user_op) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= builder->module->values.count) {
    return iree_ok_status();
  }
  return loom_symbol_reference_visit_type(
      builder, source_symbol_id,
      loom_module_value_type(builder->module, value_id),
      LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE,
      LOOM_SYMBOL_REFERENCE_ATTR_INDEX_NONE, user_op);
}

static iree_status_t loom_symbol_reference_visit_op_value_types(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_value_type(
        builder, source_symbol_id, operands[i], op));
  }
  const loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_value_type(
        builder, source_symbol_id, results[i], op));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_block_arg_types(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    const loom_block_t* block) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_value_type(
        builder, source_symbol_id, loom_block_arg_id(block, i),
        /*user_op=*/NULL));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_op_attrs(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    const loom_op_t* op, const loom_op_vtable_t* vtable) {
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (vtable && vtable->symbol_def &&
        i == vtable->symbol_def->name_attr_index) {
      continue;
    }
    const loom_attr_descriptor_t* descriptor = NULL;
    if (vtable && vtable->attr_descriptors && i < vtable->attribute_count) {
      descriptor = &vtable->attr_descriptors[i];
    }
    loom_symbol_reference_occurrence_kind_t kind =
        loom_symbol_reference_direct_attr_kind(vtable, descriptor, i);
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_attr(
        builder, source_symbol_id, attrs[i], descriptor, kind, i, op,
        /*dict_depth=*/0));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_region(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t source_symbol_id,
    const loom_region_t* region) {
  if (!region) return iree_ok_status();
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_block_arg_types(
        builder, source_symbol_id, block));
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_op_vtable_t* vtable = loom_op_vtable(builder->module, op);
      loom_symbol_id_t nested_source_symbol_id = source_symbol_id;
      loom_symbol_ref_t op_symbol_ref = loom_symbol_ref_null();
      if (loom_op_defining_symbol_ref(builder->module, op, &op_symbol_ref)) {
        nested_source_symbol_id = op_symbol_ref.symbol_id;
      }
      if (loom_template_apply_isa(op)) {
        IREE_RETURN_IF_ERROR(loom_symbol_reference_append_template_demand(
            builder, nested_source_symbol_id, op));
      }
      IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_op_value_types(
          builder, nested_source_symbol_id, op));
      IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_op_attrs(
          builder, nested_source_symbol_id, op, vtable));
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_region(
            builder, nested_source_symbol_id, regions[i]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_module_encodings(
    loom_symbol_reference_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->module->encodings.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_encoding(
        builder, LOOM_SYMBOL_ID_INVALID, &builder->module->encodings.entries[i],
        LOOM_SYMBOL_REFERENCE_OCCURRENCE_MODULE_ENCODING,
        LOOM_SYMBOL_REFERENCE_ATTR_INDEX_NONE, /*user_op=*/NULL));
  }
  return iree_ok_status();
}

iree_status_t loom_symbol_reference_table_build(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_symbol_reference_table_t* out_table) {
  if (!out_table) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol reference output table is NULL");
  }
  *out_table = (loom_symbol_reference_table_t){0};
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol reference module is NULL");
  }
  if (!arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol reference arena is NULL");
  }

  loom_symbol_reference_builder_t builder = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_builder_initialize(module, arena, &builder));
  IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_region(
      &builder, LOOM_SYMBOL_ID_INVALID, module->body));
  IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_module_encodings(&builder));

  *out_table = (loom_symbol_reference_table_t){
      .module = module,
      .symbols = builder.symbols,
      .symbol_count = module->symbols.count,
      .occurrences = builder.occurrences,
      .occurrence_count = builder.occurrence_count,
      .first_module_occurrence_id = builder.first_module_occurrence_id,
      .module_occurrence_count = builder.module_occurrence_count,
      .template_demands =
          {
              .values = builder.template_demands.values,
              .count = builder.template_demands.count,
              .family_bits = builder.template_demands.family_bits,
          },
  };
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_dependency_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const loom_symbol_reference_table_t* table =
      (const loom_symbol_reference_table_t*)user_data;
  if (node >= table->symbol_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol reference SCC node %" PRIhsz
                            " out of range for %" PRIhsz " symbols",
                            node, table->symbol_count);
  }
  loom_symbol_reference_occurrence_id_t occurrence_id =
      table->symbols[node].first_outgoing_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &table->occurrences[occurrence_id];
    if (!loom_symbol_reference_occurrence_is_dependency(occurrence)) {
      occurrence_id = occurrence->next_outgoing_occurrence_id;
      continue;
    }
    IREE_RETURN_IF_ERROR(
        successor.fn(successor.user_data, occurrence->target_symbol_id));
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
  return iree_ok_status();
}

loom_scc_graph_t loom_symbol_reference_dependency_scc_graph(
    const loom_symbol_reference_table_t* table) {
  return (loom_scc_graph_t){
      .node_count = table ? table->symbol_count : 0,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_symbol_reference_visit_dependency_successors, (void*)table),
  };
}
