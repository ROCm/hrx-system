// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_references.h"

#include <string.h>

#include "loom/analysis/symbol_reference_summary.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/util/adaptive_sort.h"

static bool loom_symbol_reference_symbol_id_less(const loom_symbol_id_t* lhs,
                                                 const loom_symbol_id_t* rhs) {
  return *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_symbol_reference_sort_symbol_ids,
                          loom_symbol_id_t,
                          loom_symbol_reference_symbol_id_less)

typedef struct loom_symbol_reference_builder_t {
  // Module being scanned.
  const loom_module_t* module;
  // Arena receiving table storage.
  iree_arena_allocator_t* arena;
  // Invocation-owned structural facts, separate from occurrence storage.
  loom_symbol_reference_summary_t* summary;
  // Direct directory of lazily allocated mutable symbol occurrence rows.
  const loom_symbol_reference_symbol_occurrences_t** symbol_segments;
  // Append-only occurrence rows staged until the complete table is published.
  struct {
    // Stable row segments and their pointer directory.
    loom_segmented_storage_t segments;
    // Current segment receiving rows; NULL before the first append.
    loom_symbol_reference_occurrence_t* tail;
    // Number of initialized occurrence rows.
    iree_host_size_t count;
  } occurrences;
  // First module-root occurrence.
  loom_symbol_reference_occurrence_id_t first_module_occurrence_id;
  // Number of module-root occurrences.
  uint32_t module_occurrence_count;
  // Direct call counts accumulated as occurrences are published.
  loom_symbol_reference_call_counts_t calls;
  // Execution target references for independent regions, in traversal order.
  struct {
    // Target references; null entries inherit the owning function's target.
    loom_symbol_ref_t* values;
    // Number of initialized entries.
    iree_host_size_t count;
    // Number of allocated entries.
    iree_host_size_t capacity;
  } execution_targets;
  // Mutable template-demand storage and family summary.
  struct {
    // Demand entries.
    loom_template_demand_t* values;
    // Number of live entries.
    iree_host_size_t count;
    // Number of allocated entries.
    iree_host_size_t capacity;
    // Unique demanded family symbol IDs.
    loom_symbol_id_t* family_symbol_ids;
    // Number of unique demanded family symbol IDs.
    iree_host_size_t family_count;
    // Number of allocated family symbol ID entries.
    iree_host_size_t family_capacity;
    // Dense bitset indexed by module symbol ID for demanded families.
    uint64_t* family_bits;
  } template_demands;

  // Number of valid module-local template providers.
  uint32_t template_provider_count;
  // Symbols with contextual operations inheriting the symbol's target.
  uint64_t* contextual_symbols;
  // Symbols directly owning func.exit operations.
  uint64_t* callable_exit_symbols;
} loom_symbol_reference_builder_t;

// Symbol definition and root region currently owning a reference occurrence.
typedef struct loom_symbol_reference_source_scope_t {
  // Module-local owning symbol, or invalid for module-root references.
  loom_symbol_id_t symbol_id;
  // Root region slot on symbol_id plus one, or zero for its contract.
  uint8_t root_region_index_plus_one;

  // True when an enclosing structured condition or CFG edge can supply facts.
  bool has_path_condition;
  // Independent region ordinal plus one, or zero for the owning function.
  loom_symbol_reference_execution_scope_id_t execution_scope;
} loom_symbol_reference_source_scope_t;

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

static void loom_symbol_reference_builder_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_symbol_reference_builder_t* builder) {
  *builder = (loom_symbol_reference_builder_t){
      .module = module,
      .arena = arena,
      .first_module_occurrence_id = LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID,
  };
  loom_segmented_storage_initialize(
      LOOM_SYMBOL_REFERENCE_OCCURRENCE_SEGMENT_CAPACITY *
          sizeof(loom_symbol_reference_occurrence_t),
      iree_alignof(loom_symbol_reference_occurrence_t),
      &builder->occurrences.segments);
}

static iree_status_t loom_symbol_reference_builder_set_symbol_bit(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t symbol_id,
    uint64_t** bitset) {
  if (*bitset == NULL) {
    const iree_host_size_t word_count =
        builder->module->symbols.count / 64 +
        (builder->module->symbols.count % 64 != 0);
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, word_count, sizeof(**bitset), (void**)bitset));
    memset(*bitset, 0, word_count * sizeof(**bitset));
  }
  (*bitset)[symbol_id >> 6] |= UINT64_C(1) << (symbol_id & 63u);
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_builder_get_or_create_symbol(
    loom_symbol_reference_builder_t* builder, loom_symbol_id_t symbol_id,
    loom_symbol_reference_symbol_occurrences_t** out_symbol) {
  *out_symbol = NULL;
  const iree_host_size_t segment_capacity =
      LOOM_SYMBOL_REFERENCE_SYMBOL_SEGMENT_CAPACITY;
  const iree_host_size_t directory_count =
      (builder->module->symbols.count + segment_capacity - 1) >>
      LOOM_SYMBOL_REFERENCE_SYMBOL_SEGMENT_SHIFT;
  if (!builder->symbol_segments) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, directory_count, sizeof(*builder->symbol_segments),
        (void**)&builder->symbol_segments));
    memset(builder->symbol_segments, 0,
           directory_count * sizeof(*builder->symbol_segments));
  }

  const iree_host_size_t segment_index =
      symbol_id >> LOOM_SYMBOL_REFERENCE_SYMBOL_SEGMENT_SHIFT;
  loom_symbol_reference_symbol_occurrences_t* segment =
      (loom_symbol_reference_symbol_occurrences_t*)
          builder->symbol_segments[segment_index];
  if (!segment) {
    const iree_host_size_t segment_base = segment_index * segment_capacity;
    const iree_host_size_t remaining_count =
        builder->module->symbols.count - segment_base;
    const iree_host_size_t segment_count =
        remaining_count < segment_capacity ? remaining_count : segment_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, segment_count, sizeof(*segment), (void**)&segment));
    loom_symbol_reference_initialize_symbol_occurrences(segment, segment_count);
    builder->symbol_segments[segment_index] = segment;
  }

  *out_symbol = &segment[symbol_id &
                         (LOOM_SYMBOL_REFERENCE_SYMBOL_SEGMENT_CAPACITY - 1u)];
  return iree_ok_status();
}

// Source ownership comes from loom_op_defining_symbol_id. Direct references
// and structural summaries establish target validity before publication.
static iree_status_t loom_symbol_reference_builder_append_occurrence(
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope,
    loom_symbol_id_t target_symbol_id,
    loom_symbol_reference_occurrence_kind_t kind,
    loom_symbol_reference_role_t role,
    loom_symbol_interface_flags_t target_interfaces, uint8_t attr_index,
    const loom_op_t* user_op) {
  if (builder->occurrences.count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "symbol reference table exceeds %u occurrences",
                            (unsigned)UINT32_MAX);
  }
  loom_symbol_reference_symbol_occurrences_t* target_symbol = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_reference_builder_get_or_create_symbol(
      builder, target_symbol_id, &target_symbol));
  loom_symbol_reference_symbol_occurrences_t* source_symbol = NULL;
  if (source_scope.symbol_id != LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_builder_get_or_create_symbol(
        builder, source_scope.symbol_id, &source_symbol));
  }
  const uint32_t segment_offset =
      builder->occurrences.count &
      (LOOM_SYMBOL_REFERENCE_OCCURRENCE_SEGMENT_CAPACITY - 1u);
  if (segment_offset == 0) {
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &builder->occurrences.segments, builder->arena,
        (void**)&builder->occurrences.tail));
  }

  const loom_symbol_reference_occurrence_id_t occurrence_id =
      (loom_symbol_reference_occurrence_id_t)builder->occurrences.count++;
  loom_symbol_reference_occurrence_t* occurrence =
      &builder->occurrences.tail[segment_offset];
  *occurrence = (loom_symbol_reference_occurrence_t){
      .source_symbol_id = source_scope.symbol_id,
      .target_symbol_id = target_symbol_id,
      .target_interfaces = target_interfaces,
      .kind = kind,
      .role = role,
      .source_root_region_index_plus_one =
          source_scope.root_region_index_plus_one,
      .execution_scope = source_scope.execution_scope,
      .attr_index = attr_index,
      .user_op = user_op,
      .next_outgoing_occurrence_id =
          LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID,
      .next_incoming_occurrence_id =
          target_symbol->first_incoming_occurrence_id,
  };
  target_symbol->first_incoming_occurrence_id = occurrence_id;

  if (kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL) {
    ++builder->calls.count;
    builder->calls.template_count += loom_template_call_isa(user_op);
  }

  if (source_scope.symbol_id == LOOM_SYMBOL_ID_INVALID) {
    occurrence->next_outgoing_occurrence_id =
        builder->first_module_occurrence_id;
    builder->first_module_occurrence_id = occurrence_id;
    ++builder->module_occurrence_count;
    return iree_ok_status();
  }
  occurrence->next_outgoing_occurrence_id =
      source_symbol->first_outgoing_occurrence_id;
  source_symbol->first_outgoing_occurrence_id = occurrence_id;
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_add_ref(
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope,
    loom_symbol_ref_t target_ref, loom_symbol_reference_occurrence_kind_t kind,
    loom_symbol_reference_role_t role,
    loom_symbol_interface_flags_t target_interfaces, uint8_t attr_index,
    const loom_op_t* user_op) {
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0) {
    return iree_ok_status();
  }
  if (target_ref.symbol_id >= builder->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target symbol id %u is out of range for %" PRIhsz " symbols",
        (unsigned)target_ref.symbol_id, builder->module->symbols.count);
  }
  return loom_symbol_reference_builder_append_occurrence(
      builder, source_scope, target_ref.symbol_id, kind, role,
      target_interfaces, attr_index, user_op);
}

static iree_status_t loom_symbol_reference_append_template_demand(
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope,
    const loom_op_t* apply_op) {
  if (source_scope.symbol_id >= builder->module->symbols.count) {
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
                            (unsigned)UINT32_MAX);
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

  const uint64_t family_mask = UINT64_C(1) << (family.symbol_id & 63u);
  uint64_t* family_word =
      &builder->template_demands.family_bits[family.symbol_id >> 6];
  if ((*family_word & family_mask) == 0) {
    if (builder->template_demands.family_count >=
        builder->template_demands.family_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          builder->arena, builder->template_demands.family_count,
          builder->template_demands.family_count + 1,
          sizeof(*builder->template_demands.family_symbol_ids),
          &builder->template_demands.family_capacity,
          (void**)&builder->template_demands.family_symbol_ids));
    }
    builder->template_demands
        .family_symbol_ids[builder->template_demands.family_count++] =
        family.symbol_id;
    *family_word |= family_mask;
  }

  loom_symbol_reference_symbol_occurrences_t* source = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_reference_builder_get_or_create_symbol(
      builder, source_scope.symbol_id, &source));
  const loom_template_demand_id_t demand_id =
      (loom_template_demand_id_t)builder->template_demands.count++;
  builder->template_demands.values[demand_id] = (loom_template_demand_t){
      .family_symbol_id = family.symbol_id,
      .source_symbol_id = source_scope.symbol_id,
      .source_root_region_index_plus_one =
          source_scope.root_region_index_plus_one,
      .has_path_condition = source_scope.has_path_condition,
      .execution_scope = source_scope.execution_scope,
      .apply_op = apply_op,
      .next_source_demand_id = source->first_template_demand_id,
  };
  source->first_template_demand_id = demand_id;
  ++source->template_demand_count;
  return iree_ok_status();
}

static void loom_symbol_reference_count_template_provider(
    loom_symbol_reference_builder_t* builder, const loom_symbol_t* symbol) {
  if (!loom_symbol_implements(symbol,
                              LOOM_SYMBOL_INTERFACE_TEMPLATE_PROVIDER)) {
    return;
  }
  const loom_func_like_t provider =
      loom_func_like_cast(builder->module, symbol->defining_op);
  IREE_ASSERT(loom_func_like_isa(provider));
  const loom_symbol_ref_t family = loom_func_like_template_family(provider);
  if (!loom_symbol_ref_is_valid(family) || family.module_id != 0 ||
      family.symbol_id >= builder->module->symbols.count) {
    return;
  }
  ++builder->template_provider_count;
}

static_assert(LOOM_ATTR_COUNT_ == 23,
              "update symbol-bearing attr classification for new kinds");

static bool loom_symbol_reference_attr_may_contain_ref(loom_attribute_t attr) {
  switch ((loom_attr_kind_t)attr.kind) {
    case LOOM_ATTR_SYMBOL:
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET:
    case LOOM_ATTR_TYPE:
    case LOOM_ATTR_ENCODING:
    case LOOM_ATTR_DICT:
    case LOOM_ATTR_PARAMETERIZED:
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_symbol_reference_emit_summary(
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope, uint8_t attr_index,
    const loom_op_t* user_op) {
  loom_symbol_reference_summary_span_t span;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         loom_symbol_reference_summary_next(builder->summary, &span)) {
    for (iree_host_size_t i = 0; i < span.count && iree_status_is_ok(status);
         ++i) {
      status = loom_symbol_reference_builder_append_occurrence(
          builder, source_scope, span.targets[i].symbol_id, span.kind,
          span.role, span.interfaces, attr_index, user_op);
    }
  }
  return status;
}

static iree_status_t loom_symbol_reference_visit_type(
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope, loom_type_t type,
    loom_symbol_reference_occurrence_kind_t kind, uint8_t attr_index,
    const loom_op_t* user_op) {
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_summary_query_type(builder->summary, type, kind));
  return loom_symbol_reference_emit_summary(builder, source_scope, attr_index,
                                            user_op);
}

static iree_status_t loom_symbol_reference_visit_attr(
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_symbol_reference_occurrence_kind_t kind, uint8_t attr_index,
    const loom_op_t* user_op) {
  // Direct scalar references need no structural discovery or scratch storage.
  if (attr.kind == LOOM_ATTR_SYMBOL || ((attr.kind == LOOM_ATTR_SYMBOL_ARRAY ||
                                         attr.kind == LOOM_ATTR_SYMBOL_SET) &&
                                        attr.count <= 1)) {
    loom_symbol_reference_role_t role = LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY;
    loom_symbol_interface_flags_t interfaces = 0;
    if (descriptor && descriptor->attr_kind == attr.kind &&
        descriptor->reference.symbol_ref) {
      role = descriptor->reference.symbol_ref->role;
      interfaces = descriptor->reference.symbol_ref->interfaces;
    }
    if (attr.kind == LOOM_ATTR_SYMBOL) {
      return loom_symbol_reference_add_ref(
          builder, source_scope, loom_attr_as_symbol(attr), kind, role,
          interfaces, attr_index, user_op);
    }
    loom_symbol_ref_array_t refs = attr.kind == LOOM_ATTR_SYMBOL_SET
                                       ? loom_attr_as_symbol_set(attr)
                                       : loom_attr_as_symbol_array(attr);
    if (!refs.count) {
      return iree_ok_status();
    }
    return loom_symbol_reference_add_ref(builder, source_scope, refs.values[0],
                                         kind, role, interfaces, attr_index,
                                         user_op);
  }
  IREE_RETURN_IF_ERROR(loom_symbol_reference_summary_query_attr(
      builder->summary, attr, descriptor, kind));
  return loom_symbol_reference_emit_summary(builder, source_scope, attr_index,
                                            user_op);
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
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope, loom_value_id_t value_id,
    const loom_op_t* user_op) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= builder->module->values.count) {
    return iree_ok_status();
  }
  const loom_type_t type = loom_module_value_type(builder->module, value_id);
  if (!loom_symbol_reference_type_may_contain_ref(type)) {
    return iree_ok_status();
  }
  return loom_symbol_reference_visit_type(
      builder, source_scope, type, LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE,
      LOOM_SYMBOL_REFERENCE_ATTR_INDEX_NONE, user_op);
}

static iree_status_t loom_symbol_reference_visit_op_defined_value_types(
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope, const loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  if (vtable && vtable->func_like &&
      vtable->func_like->args_operand_field_index != LOOM_OPERAND_INDEX_NONE) {
    // Operand-backed signatures define every operand. Kernel declarations may
    // divide those definitions between ABI and workload operand fields.
    const loom_value_id_t* operands = loom_op_operands(op);
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_value_type(
          builder, source_scope, operands[i], op));
    }
  }
  const loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_value_type(
        builder, source_scope, results[i], op));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_block_arg_types(
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope,
    const loom_block_t* block) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_value_type(
        builder, source_scope, loom_block_arg_id(block, i),
        /*user_op=*/NULL));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_op_attrs(
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope, const loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (vtable && vtable->symbol_def &&
        i == vtable->symbol_def->name_attr_index) {
      continue;
    }
    if (!loom_symbol_reference_attr_may_contain_ref(attrs[i])) {
      continue;
    }
    const loom_attr_descriptor_t* descriptor = NULL;
    if (vtable && vtable->attr_descriptors && i < vtable->attribute_count) {
      descriptor = &vtable->attr_descriptors[i];
    }
    loom_symbol_reference_occurrence_kind_t kind =
        loom_symbol_reference_direct_attr_kind(vtable, descriptor, i);
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_attr(
        builder, source_scope, attrs[i], descriptor, kind, i, op));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_append_execution_target(
    loom_symbol_reference_builder_t* builder, loom_symbol_ref_t target,
    loom_symbol_reference_execution_scope_id_t* out_scope) {
  if (builder->execution_targets.count == UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "symbol reference table exceeds %u execution scopes",
        (unsigned)UINT32_MAX);
  }
  if (builder->execution_targets.count == builder->execution_targets.capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(builder->arena, builder->execution_targets.count,
                              builder->execution_targets.count + 1,
                              sizeof(*builder->execution_targets.values),
                              &builder->execution_targets.capacity,
                              (void**)&builder->execution_targets.values));
  }
  builder->execution_targets.values[builder->execution_targets.count++] =
      target;
  *out_scope = (loom_symbol_reference_execution_scope_id_t)
                   builder->execution_targets.count;
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_region(
    loom_symbol_reference_builder_t* builder,
    loom_symbol_reference_source_scope_t source_scope,
    const loom_region_t* region) {
  if (!region) {
    return iree_ok_status();
  }
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_block_arg_types(
        builder, source_scope, block));
    loom_symbol_reference_source_scope_t block_source_scope = source_scope;
    block_source_scope.has_path_condition |= block->region_index != 0;
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_op_vtable_t* vtable = loom_op_vtable(builder->module, op);
      loom_symbol_reference_source_scope_t nested_source_scope =
          block_source_scope;
      const loom_symbol_id_t op_symbol_id =
          loom_op_defining_symbol_id(builder->module, op, vtable);
      if (op_symbol_id != LOOM_SYMBOL_ID_INVALID) {
        nested_source_scope = (loom_symbol_reference_source_scope_t){
            .symbol_id = op_symbol_id,
        };
        loom_symbol_reference_count_template_provider(
            builder, &builder->module->symbols.entries[op_symbol_id]);
      }
      if (iree_any_bit_set(op->traits, LOOM_TRAIT_CONTEXTUAL) &&
          nested_source_scope.symbol_id != LOOM_SYMBOL_ID_INVALID &&
          (nested_source_scope.execution_scope == 0 ||
           !loom_symbol_ref_is_valid(
               builder->execution_targets
                   .values[nested_source_scope.execution_scope - 1]))) {
        IREE_RETURN_IF_ERROR(loom_symbol_reference_builder_set_symbol_bit(
            builder, nested_source_scope.symbol_id,
            &builder->contextual_symbols));
      }
      if ((loom_func_exit_isa(op) || loom_low_func_exit_isa(op)) &&
          nested_source_scope.symbol_id != LOOM_SYMBOL_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_symbol_reference_builder_set_symbol_bit(
            builder, nested_source_scope.symbol_id,
            &builder->callable_exit_symbols));
      }
      if (loom_template_apply_isa(op)) {
        IREE_RETURN_IF_ERROR(loom_symbol_reference_append_template_demand(
            builder, nested_source_scope, op));
      }
      IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_op_defined_value_types(
          builder, nested_source_scope, op, vtable));
      IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_op_attrs(
          builder, nested_source_scope, op, vtable));
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        loom_symbol_reference_source_scope_t child_source_scope =
            nested_source_scope;
        child_source_scope.has_path_condition |= loom_scf_if_isa(op);
        if (op_symbol_id != LOOM_SYMBOL_ID_INVALID) {
          child_source_scope.root_region_index_plus_one = (uint8_t)(i + 1);
        }
        const loom_region_descriptor_t* descriptor =
            loom_op_vtable_region_descriptor(vtable, i);
        if (descriptor && descriptor->execution_target_attr_index_plus_one) {
          const loom_attribute_t attr = loom_op_const_attrs(
              op)[descriptor->execution_target_attr_index_plus_one - 1];
          if (!loom_attr_is_absent(attr) ||
              child_source_scope.execution_scope == 0) {
            IREE_RETURN_IF_ERROR(loom_symbol_reference_append_execution_target(
                builder,
                loom_attr_is_absent(attr) ? loom_symbol_ref_null()
                                          : loom_attr_as_symbol(attr),
                &child_source_scope.execution_scope));
          }
        }
        IREE_RETURN_IF_ERROR(loom_symbol_reference_visit_region(
            builder, child_source_scope, regions[i]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_visit_module_encodings(
    loom_symbol_reference_builder_t* builder) {
  const loom_symbol_reference_source_scope_t module_scope = {
      .symbol_id = LOOM_SYMBOL_ID_INVALID,
  };
  for (iree_host_size_t i = 0; i < builder->module->encodings.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_summary_query_encoding(
        builder->summary, (uint16_t)(i + 1),
        LOOM_SYMBOL_REFERENCE_OCCURRENCE_MODULE_ENCODING));
    IREE_RETURN_IF_ERROR(loom_symbol_reference_emit_summary(
        builder, module_scope, LOOM_SYMBOL_REFERENCE_ATTR_INDEX_NONE,
        /*user_op=*/NULL));
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
  loom_symbol_reference_builder_initialize(module, arena, &builder);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  loom_symbol_reference_summary_t summary;
  loom_symbol_reference_summary_initialize(module, &scratch_arena, &summary);
  builder.summary = &summary;
  const loom_symbol_reference_source_scope_t module_scope = {
      .symbol_id = LOOM_SYMBOL_ID_INVALID,
  };
  iree_status_t status =
      loom_symbol_reference_visit_region(&builder, module_scope, module->body);
  if (iree_status_is_ok(status)) {
    status = loom_symbol_reference_visit_module_encodings(&builder);
  }
  iree_arena_deinitialize(&scratch_arena);
  IREE_RETURN_IF_ERROR(status);
  loom_symbol_reference_sort_symbol_ids(
      builder.template_demands.family_symbol_ids,
      builder.template_demands.family_count);

  *out_table = (loom_symbol_reference_table_t){
      .module = module,
      .symbol_segments = builder.symbol_segments,
      .symbol_count = module->symbols.count,
      .occurrences = builder.occurrences.segments,
      .occurrence_count = builder.occurrences.count,
      .first_module_occurrence_id = builder.first_module_occurrence_id,
      .module_occurrence_count = builder.module_occurrence_count,
      .calls = builder.calls,
      .template_provider_count = builder.template_provider_count,
      .contextual_symbols = builder.contextual_symbols,
      .callable_exit_symbols = builder.callable_exit_symbols,
      .execution_targets =
          {
              .values = builder.execution_targets.values,
              .count = builder.execution_targets.count,
          },
      .template_demands =
          {
              .values = builder.template_demands.values,
              .count = builder.template_demands.count,
              .family_symbol_ids = builder.template_demands.family_symbol_ids,
              .family_count = builder.template_demands.family_count,
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
      loom_symbol_reference_table_symbol(table, (loom_symbol_id_t)node)
          .first_outgoing_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        loom_symbol_reference_table_occurrence(table, occurrence_id);
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
