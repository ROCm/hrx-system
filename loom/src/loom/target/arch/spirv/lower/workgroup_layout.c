// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/workgroup_layout.h"

#include <stdint.h>
#include <string.h>

#include "loom/analysis/source_storage_packing.h"
#include "loom/analysis/storage_interference.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/facts.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/atomic.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/arch/spirv/scalar_types.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/util/fact_table.h"

enum {
  LOOM_SPIRV_WORKGROUP_SCALAR_TYPE_COUNT = LOOM_SPIRV_SCALAR_TYPE_U64 + 1,
};

typedef struct loom_spirv_workgroup_root_carrier_t {
  // Common bit width of every typed view over the allocation.
  uint8_t bit_width;
  // Floating-point carrier used when no access requires integer atomics.
  loom_spirv_scalar_type_t float_scalar_type;
  // Whether an access requires a same-width integer carrier.
  bool requires_integer;
  // Whether the allocation has views with incompatible bit widths.
  bool incompatible;
} loom_spirv_workgroup_root_carrier_t;

typedef uint8_t loom_spirv_workgroup_layout_entry_flags_t;

#define LOOM_SPIRV_WORKGROUP_LAYOUT_ENTRY_PACKED ((uint8_t)1u << 0)

typedef struct loom_spirv_workgroup_layout_entry_t {
  // Entry state bits.
  loom_spirv_workgroup_layout_entry_flags_t flags;
  // Exact scalar carrier shared by every typed view of the source root.
  loom_spirv_scalar_type_t scalar_type;
  // Packed byte offset within the carrier arena.
  uint64_t byte_offset;
} loom_spirv_workgroup_layout_entry_t;

typedef struct loom_spirv_workgroup_layout_segment_t {
  // Shared stable byte-range packing for one exact scalar carrier.
  loom_source_storage_packing_t* packing;
  // Emitted Low storage arena root, or invalid before entry setup.
  loom_value_id_t low_storage_value_id;
} loom_spirv_workgroup_layout_segment_t;

typedef struct loom_spirv_workgroup_layout_t {
  // Function-local value domain covered by all dense arrays.
  const loom_local_value_domain_t* value_domain;
  // Source facts used to resolve Workgroup allocations and aliases.
  const loom_value_fact_table_t* fact_table;
  // Feature set used to decide whether float atomics need integer fallback.
  loom_spirv_feature_bits_t feature_bits;
  // Selected scalar carrier indexed by function-local allocation-root ordinal.
  loom_spirv_scalar_type_t* root_scalar_types;
  // Number of entries allocated in the dense carrier array.
  loom_value_ordinal_t scalar_type_count;
  // Whether carrier arrays have been initialized, including for empty domains.
  bool carriers_initialized;

  // Physical layout entries indexed by function-local root ordinal.
  loom_spirv_workgroup_layout_entry_t* entries;
  // Number of physical layout entry slots.
  iree_host_size_t entry_count;
  // Per-carrier physical Workgroup storage arenas.
  loom_spirv_workgroup_layout_segment_t
      segments[LOOM_SPIRV_WORKGROUP_SCALAR_TYPE_COUNT];
  // Scalar carriers in first-allocation order for stable arena emission.
  loom_spirv_scalar_type_t
      segment_order[LOOM_SPIRV_WORKGROUP_SCALAR_TYPE_COUNT];
  // Number of populated entries in |segment_order|.
  iree_host_size_t segment_count;
  // Shared lifetime and interference facts for source allocations.
  loom_storage_interference_t* interference;
  // Whether physical layout state has been initialized for the source function.
  bool layout_initialized;
} loom_spirv_workgroup_layout_t;

static const uint8_t kLoomSpirvWorkgroupLayoutStateKey;

static bool loom_spirv_workgroup_view_scalar_type(
    loom_type_t view_type, loom_spirv_scalar_type_t* out_scalar_type) {
  *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  if (!loom_type_is_view(view_type)) {
    return false;
  }
  loom_spirv_value_type_t value_type = {0};
  if (!loom_spirv_value_type_from_loom_type(
          loom_type_scalar(loom_type_element_type(view_type)), &value_type) ||
      value_type.value_class != LOOM_SPIRV_VALUE_CLASS_SCALAR) {
    return false;
  }
  *out_scalar_type = value_type.scalar_type;
  return true;
}

static bool loom_spirv_workgroup_view_root_is_alloca(
    const loom_module_t* module, loom_value_id_t root_value_id) {
  if (root_value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* root = loom_module_value(module, root_value_id);
  return !loom_value_is_block_arg(root) &&
         loom_buffer_alloca_isa(loom_value_def_op(root));
}

static bool loom_spirv_workgroup_view_reference(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_fact_view_reference_t* out_reference) {
  *out_reference = (loom_value_fact_view_reference_t){0};
  if (!loom_value_facts_query_view_reference(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, value_id), out_reference)) {
    return false;
  }
  out_reference->root_value_id =
      loom_value_fact_view_reference_resolve_root_value(*out_reference,
                                                        value_id);
  return out_reference->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
}

static loom_spirv_scalar_type_t loom_spirv_workgroup_signed_integer_carrier(
    uint8_t bit_width) {
  switch (bit_width) {
    case 8:
      return LOOM_SPIRV_SCALAR_TYPE_S8;
    case 16:
      return LOOM_SPIRV_SCALAR_TYPE_S16;
    case 32:
      return LOOM_SPIRV_SCALAR_TYPE_S32;
    case 64:
      return LOOM_SPIRV_SCALAR_TYPE_S64;
    default:
      return LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  }
}

static loom_spirv_feature_bits_t
loom_spirv_workgroup_native_float_atomic_feature(
    loom_spirv_scalar_type_t scalar_type, loom_atomic_kind_t atomic_kind) {
  switch (atomic_kind) {
    case LOOM_ATOMIC_KIND_XCHGF:
      switch (scalar_type) {
        case LOOM_SPIRV_SCALAR_TYPE_F16:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS;
        case LOOM_SPIRV_SCALAR_TYPE_F32:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMICS;
        case LOOM_SPIRV_SCALAR_TYPE_F64:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS;
        default:
          return 0;
      }
    case LOOM_ATOMIC_KIND_ADDF:
      switch (scalar_type) {
        case LOOM_SPIRV_SCALAR_TYPE_F16:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD;
        case LOOM_SPIRV_SCALAR_TYPE_F32:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMIC_ADD;
        case LOOM_SPIRV_SCALAR_TYPE_F64:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD;
        default:
          return 0;
      }
    default:
      return 0;
  }
}

static bool loom_spirv_workgroup_view_requires_integer_carrier(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_spirv_scalar_type_t scalar_type,
    loom_spirv_feature_bits_t feature_bits) {
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (loom_view_atomic_cmpxchg_isa(user_op) &&
        loom_view_atomic_cmpxchg_view(user_op) == value_id) {
      return true;
    }

    loom_atomic_kind_t atomic_kind = LOOM_ATOMIC_KIND_COUNT_;
    if (loom_view_atomic_reduce_isa(user_op) &&
        loom_view_atomic_reduce_view(user_op) == value_id) {
      atomic_kind = loom_view_atomic_reduce_kind(user_op);
    } else if (loom_view_atomic_rmw_isa(user_op) &&
               loom_view_atomic_rmw_view(user_op) == value_id) {
      atomic_kind = loom_view_atomic_rmw_kind(user_op);
    } else {
      continue;
    }

    if (iree_any_bit_set(user_op->instance_flags,
                         LOOM_MEMORY_ACCESS_FLAG_NOFTZ)) {
      return true;
    }

    const loom_spirv_feature_bits_t native_feature =
        loom_spirv_workgroup_native_float_atomic_feature(scalar_type,
                                                         atomic_kind);
    if (native_feature == 0 ||
        !iree_all_bits_set(feature_bits, native_feature)) {
      return true;
    }
  }
  return false;
}

static loom_spirv_scalar_type_t loom_spirv_workgroup_select_root_carrier(
    loom_spirv_workgroup_root_carrier_t root_carrier) {
  if (root_carrier.incompatible || root_carrier.bit_width == 0) {
    return LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  }
  return root_carrier.requires_integer
             ? loom_spirv_workgroup_signed_integer_carrier(
                   root_carrier.bit_width)
             : root_carrier.float_scalar_type;
}

static iree_status_t loom_spirv_prepare_workgroup_carriers(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    loom_spirv_feature_bits_t feature_bits, iree_arena_allocator_t* arena,
    loom_spirv_workgroup_layout_t* layout) {
  if (layout->carriers_initialized && layout->value_domain == value_domain &&
      layout->fact_table == fact_table &&
      layout->feature_bits == feature_bits &&
      layout->scalar_type_count >= value_domain->value_count) {
    return iree_ok_status();
  }

  loom_spirv_scalar_type_t* root_scalar_types = NULL;
  loom_spirv_workgroup_root_carrier_t* root_carriers = NULL;
  if (value_domain->value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_domain->value_count, sizeof(*root_scalar_types),
        (void**)&root_scalar_types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_domain->value_count, sizeof(*root_carriers),
        (void**)&root_carriers));
    memset(root_scalar_types, 0,
           value_domain->value_count * sizeof(*root_scalar_types));
    memset(root_carriers, 0,
           value_domain->value_count * sizeof(*root_carriers));
  }

  for (loom_value_ordinal_t value_ordinal = 0;
       value_ordinal < value_domain->value_count; ++value_ordinal) {
    const loom_value_id_t value_id = value_domain->value_ids[value_ordinal];
    const loom_type_t value_type = loom_module_value_type(module, value_id);
    loom_spirv_scalar_type_t scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
    if (!loom_spirv_workgroup_view_scalar_type(value_type, &scalar_type)) {
      continue;
    }
    loom_value_fact_view_reference_t reference = {0};
    if (!loom_spirv_workgroup_view_reference(fact_table, value_id,
                                             &reference) ||
        !loom_spirv_workgroup_view_root_is_alloca(module,
                                                  reference.root_value_id)) {
      continue;
    }
    const loom_value_ordinal_t root_ordinal =
        loom_local_value_domain_try_ordinal(value_domain,
                                            reference.root_value_id);
    if (root_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      continue;
    }

    const loom_spirv_scalar_type_descriptor_t* scalar_descriptor =
        loom_spirv_scalar_type_descriptor(scalar_type);
    if (scalar_descriptor == NULL) {
      continue;
    }
    loom_spirv_workgroup_root_carrier_t* root_carrier =
        &root_carriers[root_ordinal];
    if (root_carrier->bit_width == 0) {
      root_carrier->bit_width = scalar_descriptor->bit_width;
    } else if (root_carrier->bit_width != scalar_descriptor->bit_width) {
      root_carrier->incompatible = true;
      continue;
    }

    // Software BF16 keeps its payload in i32 registers but occupies one i16
    // storage element. Retain that choice with the allocation's other views.
    if (scalar_descriptor->kind != LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT ||
        (scalar_type == LOOM_SPIRV_SCALAR_TYPE_BF16 &&
         !iree_all_bits_set(feature_bits,
                            LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR))) {
      root_carrier->requires_integer = true;
      continue;
    }
    if (root_carrier->float_scalar_type == LOOM_SPIRV_SCALAR_TYPE_UNKNOWN) {
      root_carrier->float_scalar_type = scalar_type;
    } else if (root_carrier->float_scalar_type != scalar_type) {
      root_carrier->requires_integer = true;
    }
    if (loom_spirv_workgroup_view_requires_integer_carrier(
            module, value_id, scalar_type, feature_bits)) {
      root_carrier->requires_integer = true;
    }
  }

  for (loom_value_ordinal_t value_ordinal = 0;
       value_ordinal < value_domain->value_count; ++value_ordinal) {
    root_scalar_types[value_ordinal] =
        loom_spirv_workgroup_select_root_carrier(root_carriers[value_ordinal]);
  }

  layout->value_domain = value_domain;
  layout->fact_table = fact_table;
  layout->feature_bits = feature_bits;
  layout->root_scalar_types = root_scalar_types;
  layout->scalar_type_count = value_domain->value_count;
  layout->carriers_initialized = true;
  layout->layout_initialized = false;
  return iree_ok_status();
}

static iree_status_t loom_spirv_resolve_workgroup_view_reg_class_from_layout(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    loom_spirv_feature_bits_t feature_bits, iree_arena_allocator_t* arena,
    loom_spirv_workgroup_layout_t* layout, loom_value_id_t source_value_id,
    bool* out_is_workgroup, uint16_t* out_reg_class_id) {
  *out_is_workgroup = false;
  *out_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_spirv_workgroup_view_reference(fact_table, source_value_id,
                                           &reference)) {
    return iree_ok_status();
  }
  *out_is_workgroup = true;
  if (!loom_spirv_workgroup_view_root_is_alloca(module,
                                                reference.root_value_id)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_spirv_prepare_workgroup_carriers(
      module, fact_table, value_domain, feature_bits, arena, layout));
  const loom_value_ordinal_t root_ordinal = loom_local_value_domain_try_ordinal(
      value_domain, reference.root_value_id);
  if (root_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      root_ordinal >= layout->scalar_type_count) {
    return iree_ok_status();
  }
  *out_reg_class_id = loom_spirv_ptr_workgroup_array_reg_class_id(
      layout->root_scalar_types[root_ordinal]);
  return iree_ok_status();
}

iree_status_t loom_spirv_resolve_workgroup_view_reg_class(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_is_workgroup, uint16_t* out_reg_class_id) {
  loom_spirv_workgroup_layout_t* layout = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &kLoomSpirvWorkgroupLayoutStateKey, sizeof(*layout),
      (void**)&layout));
  return loom_spirv_resolve_workgroup_view_reg_class_from_layout(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context),
      loom_low_lower_context_value_domain(context),
      loom_low_lower_context_bundle(context)->config->contract_feature_bits,
      loom_low_lower_context_function_arena(context), layout, source_value_id,
      out_is_workgroup, out_reg_class_id);
}

iree_status_t loom_spirv_resolve_workgroup_contract_view_reg_class(
    const loom_target_contract_query_environment_t* environment,
    loom_value_id_t source_value_id, bool* out_is_workgroup,
    uint16_t* out_reg_class_id) {
  *out_is_workgroup = false;
  *out_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  loom_target_contract_query_environment_t analyzed_environment = *environment;
  IREE_RETURN_IF_ERROR(loom_target_contract_query_resolve_function_analysis(
      &analyzed_environment));
  environment = &analyzed_environment;
  if (environment->fact_table == NULL || environment->value_domain == NULL ||
      environment->arena == NULL) {
    return iree_ok_status();
  }
  loom_spirv_workgroup_layout_t local_layout = {0};
  loom_spirv_workgroup_layout_t* layout = &local_layout;
  if (environment->target_state_allocator.fn != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_target_contract_query_get_or_allocate_target_state(
            environment, &kLoomSpirvWorkgroupLayoutStateKey, sizeof(*layout),
            (void**)&layout));
    if (layout == NULL) {
      return iree_ok_status();
    }
  }
  const loom_target_bundle_t* bundle =
      loom_target_contract_query_environment_bundle(environment);
  return loom_spirv_resolve_workgroup_view_reg_class_from_layout(
      environment->module, environment->fact_table, environment->value_domain,
      bundle->config->contract_feature_bits, environment->arena, layout,
      source_value_id, out_is_workgroup, out_reg_class_id);
}

static iree_status_t loom_spirv_workgroup_layout_initialize(
    loom_low_lower_context_t* context, loom_spirv_workgroup_layout_t* layout) {
  const loom_local_value_domain_t* value_domain =
      loom_low_lower_context_value_domain(context);
  IREE_RETURN_IF_ERROR(loom_spirv_prepare_workgroup_carriers(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context), value_domain,
      loom_low_lower_context_bundle(context)->config->contract_feature_bits,
      loom_low_lower_context_function_arena(context), layout));
  if (layout->layout_initialized) {
    return iree_ok_status();
  }

  layout->entry_count = value_domain->value_count;
  layout->entries = NULL;
  if (layout->entry_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
        context, layout->entry_count, sizeof(*layout->entries),
        (void**)&layout->entries));
    memset(layout->entries, 0, layout->entry_count * sizeof(*layout->entries));
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(layout->segments); ++i) {
    layout->segments[i] = (loom_spirv_workgroup_layout_segment_t){
        .low_storage_value_id = LOOM_VALUE_ID_INVALID,
    };
  }
  layout->segment_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_context_storage_interference(
      context, &layout->interference));
  layout->layout_initialized = true;
  return iree_ok_status();
}

static iree_status_t loom_spirv_workgroup_layout_for_context(
    loom_low_lower_context_t* context,
    loom_spirv_workgroup_layout_t** out_layout) {
  *out_layout = NULL;
  loom_spirv_workgroup_layout_t* layout = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &kLoomSpirvWorkgroupLayoutStateKey, sizeof(*layout),
      (void**)&layout));
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_layout_initialize(context, layout));
  *out_layout = layout;
  return iree_ok_status();
}

static iree_status_t loom_spirv_workgroup_layout_query_interference(
    void* user_data, loom_value_id_t lhs_root_value_id,
    loom_value_id_t rhs_root_value_id, bool* out_interferes) {
  loom_spirv_workgroup_layout_t* layout =
      (loom_spirv_workgroup_layout_t*)user_data;
  *out_interferes = true;
  bool proven_nonoverlap = false;
  IREE_RETURN_IF_ERROR(loom_storage_interference_prove_workgroup_nonoverlap(
      layout->interference, lhs_root_value_id, rhs_root_value_id,
      &proven_nonoverlap));
  *out_interferes = !proven_nonoverlap;
  return iree_ok_status();
}

iree_status_t loom_spirv_workgroup_layout_record_alloca(
    loom_low_lower_context_t* context, const loom_op_t* alloca_op,
    uint64_t byte_length, uint64_t byte_alignment, bool* out_packed) {
  *out_packed = false;
  loom_spirv_workgroup_layout_t* layout = NULL;
  IREE_RETURN_IF_ERROR(
      loom_spirv_workgroup_layout_for_context(context, &layout));
  const loom_value_id_t root_value_id = loom_buffer_alloca_result(alloca_op);
  const loom_value_ordinal_t root_ordinal =
      loom_local_value_domain_try_ordinal(layout->value_domain, root_value_id);
  if (root_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      root_ordinal >= layout->entry_count) {
    return iree_ok_status();
  }
  loom_spirv_workgroup_layout_entry_t* entry = &layout->entries[root_ordinal];
  if (iree_all_bits_set(entry->flags,
                        LOOM_SPIRV_WORKGROUP_LAYOUT_ENTRY_PACKED)) {
    *out_packed = true;
    return iree_ok_status();
  }

  const loom_spirv_scalar_type_t scalar_type =
      layout->root_scalar_types[root_ordinal];
  const loom_spirv_scalar_type_descriptor_t* scalar_descriptor =
      loom_spirv_scalar_type_descriptor(scalar_type);
  if (scalar_descriptor == NULL ||
      !loom_storage_interference_root_may_be_accessed(layout->interference,
                                                      root_value_id)) {
    return iree_ok_status();
  }
  const uint64_t scalar_byte_count = scalar_descriptor->bit_width / 8u;
  if (scalar_byte_count == 0 || byte_length % scalar_byte_count != 0 ||
      byte_alignment < scalar_byte_count) {
    return iree_ok_status();
  }

  IREE_ASSERT_LT((uint32_t)scalar_type,
                 (uint32_t)IREE_ARRAYSIZE(layout->segments));
  loom_spirv_workgroup_layout_segment_t* segment =
      &layout->segments[scalar_type];
  if (segment->packing == NULL) {
    IREE_RETURN_IF_ERROR(loom_source_storage_packing_allocate(
        loom_source_storage_packing_interference_callback_make(
            loom_spirv_workgroup_layout_query_interference, layout),
        NULL, 0, loom_low_lower_context_function_arena(context),
        &segment->packing));
    IREE_ASSERT_LT(layout->segment_count,
                   IREE_ARRAYSIZE(layout->segment_order));
    layout->segment_order[layout->segment_count++] = scalar_type;
  }
  uint64_t byte_offset = 0;
  IREE_RETURN_IF_ERROR(loom_source_storage_packing_append(
      segment->packing, root_value_id, byte_length, byte_alignment,
      &byte_offset));
  *entry = (loom_spirv_workgroup_layout_entry_t){
      .flags = LOOM_SPIRV_WORKGROUP_LAYOUT_ENTRY_PACKED,
      .scalar_type = scalar_type,
      .byte_offset = byte_offset,
  };
  *out_packed = true;
  return iree_ok_status();
}

static const loom_spirv_workgroup_layout_t* loom_spirv_workgroup_layout_lookup(
    const loom_low_lower_context_t* context) {
  return (const loom_spirv_workgroup_layout_t*)
      loom_low_lower_lookup_target_state(context,
                                         &kLoomSpirvWorkgroupLayoutStateKey,
                                         sizeof(loom_spirv_workgroup_layout_t));
}

static loom_spirv_scalar_type_t loom_spirv_workgroup_layout_scalar_type_at(
    const loom_spirv_workgroup_layout_t* layout, iree_host_size_t index) {
  IREE_ASSERT(layout != NULL && layout->layout_initialized);
  IREE_ASSERT_LT(index, layout->segment_count);
  return layout->segment_order[index];
}

iree_host_size_t loom_spirv_workgroup_layout_storage_root_count(
    const loom_low_lower_context_t* context) {
  const loom_spirv_workgroup_layout_t* layout =
      loom_spirv_workgroup_layout_lookup(context);
  return layout != NULL && layout->layout_initialized ? layout->segment_count
                                                      : 0;
}

loom_spirv_workgroup_storage_root_requirement_t
loom_spirv_workgroup_layout_storage_root_requirement(
    const loom_low_lower_context_t* context, iree_host_size_t index) {
  const loom_spirv_workgroup_layout_t* layout =
      loom_spirv_workgroup_layout_lookup(context);
  const loom_spirv_scalar_type_t scalar_type =
      loom_spirv_workgroup_layout_scalar_type_at(layout, index);
  const loom_spirv_workgroup_layout_segment_t* segment =
      &layout->segments[scalar_type];
  IREE_ASSERT(segment->packing != NULL);
  const loom_source_storage_packing_requirement_t requirement =
      loom_source_storage_packing_requirement(segment->packing);
  return (loom_spirv_workgroup_storage_root_requirement_t){
      .byte_length = requirement.byte_length,
      .byte_alignment = requirement.byte_alignment,
  };
}

iree_status_t loom_spirv_workgroup_layout_emit_storage_roots(
    loom_low_lower_context_t* context) {
  const loom_spirv_workgroup_layout_t* existing_layout =
      loom_spirv_workgroup_layout_lookup(context);
  if (existing_layout == NULL || !existing_layout->layout_initialized) {
    return iree_ok_status();
  }
  loom_spirv_workgroup_layout_t* layout =
      (loom_spirv_workgroup_layout_t*)existing_layout;
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_op_t* source_function_op =
      loom_low_lower_context_source_function(context).op;
  for (iree_host_size_t i = 0; i < layout->segment_count; ++i) {
    const loom_spirv_scalar_type_t scalar_type =
        loom_spirv_workgroup_layout_scalar_type_at(layout, i);
    loom_spirv_workgroup_layout_segment_t* segment =
        &layout->segments[scalar_type];
    IREE_ASSERT(segment->packing != NULL);
    const loom_source_storage_packing_requirement_t requirement =
        loom_source_storage_packing_requirement(segment->packing);
    IREE_ASSERT_NE(requirement.byte_length, 0);
    IREE_ASSERT_EQ(segment->low_storage_value_id, LOOM_VALUE_ID_INVALID);
    loom_op_t* storage_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
        builder, (int64_t)requirement.byte_length,
        (int64_t)requirement.byte_alignment,
        loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP),
        source_function_op->location, &storage_op));
    segment->low_storage_value_id =
        loom_low_storage_reserve_storage(storage_op);
  }
  return iree_ok_status();
}

static const loom_spirv_workgroup_layout_entry_t*
loom_spirv_workgroup_layout_lookup_entry(
    const loom_spirv_workgroup_layout_t* layout,
    loom_value_id_t root_value_id) {
  const loom_value_ordinal_t root_ordinal =
      loom_local_value_domain_try_ordinal(layout->value_domain, root_value_id);
  if (root_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      root_ordinal >= layout->entry_count) {
    return NULL;
  }
  const loom_spirv_workgroup_layout_entry_t* entry =
      &layout->entries[root_ordinal];
  return iree_all_bits_set(entry->flags,
                           LOOM_SPIRV_WORKGROUP_LAYOUT_ENTRY_PACKED)
             ? entry
             : NULL;
}

void loom_spirv_workgroup_layout_lookup_low_storage(
    const loom_low_lower_context_t* context, loom_value_id_t root_value_id,
    loom_value_id_t* out_low_storage_value_id) {
  IREE_ASSERT_ARGUMENT(out_low_storage_value_id);
  *out_low_storage_value_id = LOOM_VALUE_ID_INVALID;
  const loom_spirv_workgroup_layout_t* layout =
      (const loom_spirv_workgroup_layout_t*)loom_low_lower_lookup_target_state(
          context, &kLoomSpirvWorkgroupLayoutStateKey, sizeof(*layout));
  IREE_ASSERT(layout != NULL && layout->layout_initialized);
  const loom_spirv_workgroup_layout_entry_t* entry =
      loom_spirv_workgroup_layout_lookup_entry(layout, root_value_id);
  IREE_ASSERT(entry != NULL);
  const loom_spirv_workgroup_layout_segment_t* segment =
      &layout->segments[entry->scalar_type];
  IREE_ASSERT_NE(segment->low_storage_value_id, LOOM_VALUE_ID_INVALID);
  *out_low_storage_value_id = segment->low_storage_value_id;
}

uint64_t loom_spirv_workgroup_layout_root_byte_offset(
    const loom_low_lower_context_t* context, loom_value_id_t root_value_id) {
  const loom_spirv_workgroup_layout_t* layout =
      (const loom_spirv_workgroup_layout_t*)loom_low_lower_lookup_target_state(
          context, &kLoomSpirvWorkgroupLayoutStateKey, sizeof(*layout));
  if (layout == NULL || !layout->layout_initialized) {
    return 0;
  }
  const loom_spirv_workgroup_layout_entry_t* entry =
      loom_spirv_workgroup_layout_lookup_entry(layout, root_value_id);
  return entry != NULL ? entry->byte_offset : 0;
}

uint64_t loom_spirv_workgroup_layout_source_memory_root_byte_offset(
    void* user_data, const loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_memory_access) {
  (void)user_data;
  return source_memory_access->memory_space ==
                 LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
             ? loom_spirv_workgroup_layout_root_byte_offset(
                   context, source_memory_access->root_value_id)
             : 0;
}
