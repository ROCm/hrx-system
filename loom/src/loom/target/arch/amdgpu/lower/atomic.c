// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/atomic.h"

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/candidates/atomic_candidates.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

typedef uint32_t loom_amdgpu_atomic_rejection_flags_t;

#define LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP ((uint32_t)1u << 0)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND ((uint32_t)1u << 1)
#define LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE ((uint32_t)1u << 2)
#define LOOM_AMDGPU_ATOMIC_REJECTION_WORKGROUP_ROOT ((uint32_t)1u << 3)
#define LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE ((uint32_t)1u << 4)
#define LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND ((uint32_t)1u << 5)
#define LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE ((uint32_t)1u << 6)
#define LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT ((uint32_t)1u << 7)
#define LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING ((uint32_t)1u << 8)
#define LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE ((uint32_t)1u << 9)
#define LOOM_AMDGPU_ATOMIC_REJECTION_CACHE_POLICY ((uint32_t)1u << 10)
#define LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING ((uint32_t)1u << 11)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE ((uint32_t)1u << 12)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE ((uint32_t)1u << 13)

typedef struct loom_amdgpu_atomic_diagnostic_t {
  // Rejection bits explaining why a source atomic is not legal.
  loom_amdgpu_atomic_rejection_flags_t rejection_bits;
} loom_amdgpu_atomic_diagnostic_t;

typedef struct loom_amdgpu_atomic_rejection_key_t {
  // Rejection bit matched by this diagnostic row.
  loom_amdgpu_atomic_rejection_flags_t rejection_bit;
  // Stable diagnostic constraint key for the matched rejection bit.
  iree_string_view_t constraint_key;
} loom_amdgpu_atomic_rejection_key_t;

typedef struct loom_amdgpu_atomic_wait_packet_template_t {
  // Stable descriptor ref emitted for this wait packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Static immediate rows for draining the selected counter to zero.
  loom_amdgpu_explicit_packet_immediate_template_t
      immediates[LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY];
  // Number of populated immediate rows.
  iree_host_size_t immediate_count;
} loom_amdgpu_atomic_wait_packet_template_t;

typedef struct loom_amdgpu_atomic_explicit_packet_selection_t {
  // Stable descriptor ref selected for this explicit packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Immediate rows emitted on the descriptor.
  loom_amdgpu_explicit_packet_immediate_template_t
      immediates[LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY];
  // Number of populated immediate rows.
  iree_host_size_t immediate_count;
} loom_amdgpu_atomic_explicit_packet_selection_t;

typedef struct loom_amdgpu_atomic_ordering_selection_t {
  // Explicit waits emitted before the atomic packet.
  loom_amdgpu_atomic_explicit_packet_selection_t
      pre_atomic_waits[LOOM_AMDGPU_ATOMIC_WAIT_CAPACITY];
  // Number of populated pre-atomic wait packets.
  iree_host_size_t pre_atomic_wait_count;
  // Explicit waits emitted after the atomic packet.
  loom_amdgpu_atomic_explicit_packet_selection_t
      post_atomic_waits[LOOM_AMDGPU_ATOMIC_WAIT_CAPACITY];
  // Number of populated post-atomic wait packets.
  iree_host_size_t post_atomic_wait_count;
  // Explicit cache controls emitted after the atomic packet.
  loom_amdgpu_atomic_explicit_packet_selection_t
      post_atomic_cache_controls[LOOM_AMDGPU_ATOMIC_CACHE_CONTROL_CAPACITY];
  // Number of populated post-atomic cache-control packets.
  iree_host_size_t post_atomic_cache_control_descriptor_count;
} loom_amdgpu_atomic_ordering_selection_t;

typedef struct loom_amdgpu_atomic_selection_t {
  // Target-independent source memory access plan being wrapped.
  loom_low_source_memory_access_plan_t source;
  // Target-specific lowering flags derived from the selected descriptor.
  loom_amdgpu_atomic_plan_flags_t flags;
  // Source atomic operation form being lowered.
  loom_amdgpu_atomic_operation_kind_t operation_kind;
  // Selected target addressing form for the atomic packet.
  loom_amdgpu_memory_address_form_t address_form;
  // Target operand path selected for each source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t
      dynamic_term_kinds[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Static offset value encoded in the descriptor offset immediate.
  int64_t immediate_offset;
  // Static byte offset materialized through the scalar SOFFSET operand.
  uint32_t scalar_byte_offset;
  // Stable descriptor ref selected for the active descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Descriptor attrs emitted directly on the selected atomic packet.
  loom_amdgpu_atomic_packet_attrs_t packet_attrs;
  // Explicit packets required to implement source atomic ordering.
  loom_amdgpu_atomic_ordering_selection_t ordering;
} loom_amdgpu_atomic_selection_t;

static const loom_amdgpu_atomic_rejection_key_t kAmdgpuAtomicRejectionKeys[] = {
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP,
        .constraint_key = IREE_SVL("atomic.source_op"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND,
        .constraint_key = IREE_SVL("atomic.operation_kind"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE,
        .constraint_key = IREE_SVL("atomic.memory_space"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_WORKGROUP_ROOT,
        .constraint_key = IREE_SVL("atomic.workgroup_root"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE,
        .constraint_key = IREE_SVL("atomic.shape"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND,
        .constraint_key = IREE_SVL("atomic.kind"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE,
        .constraint_key = IREE_SVL("atomic.value_type"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT,
        .constraint_key = IREE_SVL("atomic.value_placement"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING,
        .constraint_key = IREE_SVL("atomic.ordering"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE,
        .constraint_key = IREE_SVL("atomic.scope"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_CACHE_POLICY,
        .constraint_key = IREE_SVL("atomic.cache_policy"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING,
        .constraint_key = IREE_SVL("atomic.descriptor_missing"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE,
        .constraint_key = IREE_SVL("atomic.offset_immediate"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE,
        .constraint_key = IREE_SVL("atomic.offset_range"),
    },
};

// GFX11 VMEM load drain used by global atomic acquire/release ordering.
static const loom_amdgpu_atomic_wait_packet_template_t
    kAmdgpuGfx11VmemLoadWaitPacket = {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT,
        .immediates =
            {
                {IREE_SVL("vmcnt"), 0},
                {IREE_SVL("lgkmcnt"), 15},
            },
        .immediate_count = 2,
};

// GFX11 VMEM store drain used by global atomic acquire/release ordering.
static const loom_amdgpu_atomic_wait_packet_template_t
    kAmdgpuGfx11VmemStoreWaitPacket = {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT,
        .immediates =
            {
                {IREE_SVL("vscnt"), 0},
            },
        .immediate_count = 1,
};

// GFX12 global/device scope encoded in VGLOBAL SCOPE immediates.
#define LOOM_AMDGPU_GFX12_SCOPE_DEVICE 2

// GFX12 VMEM load drain used by global atomic acquire ordering.
static const loom_amdgpu_atomic_wait_packet_template_t
    kAmdgpuGfx12VmemLoadWaitPacket = {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_LOADCNT,
        .immediates =
            {
                {IREE_SVL("loadcnt"), 0},
            },
        .immediate_count = 1,
};

// GFX12 VMEM store drain used by global atomic acquire ordering.
static const loom_amdgpu_atomic_wait_packet_template_t
    kAmdgpuGfx12VmemStoreWaitPacket = {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_STORECNT,
        .immediates =
            {
                {IREE_SVL("storecnt"), 0},
            },
        .immediate_count = 1,
};

static bool loom_amdgpu_view_atomic_isa(const loom_op_t* op) {
  return loom_view_atomic_reduce_isa(op) || loom_view_atomic_rmw_isa(op) ||
         loom_view_atomic_cmpxchg_isa(op);
}

static loom_value_id_t loom_amdgpu_atomic_value(const loom_op_t* op) {
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_value(op);
  }
  return loom_view_atomic_rmw_value(op);
}

static uint8_t loom_amdgpu_atomic_kind(const loom_op_t* op) {
  if (loom_view_atomic_cmpxchg_isa(op)) {
    return LOOM_AMDGPU_ATOMIC_KIND_NONE;
  }
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_kind(op);
  }
  return loom_view_atomic_rmw_kind(op);
}

static uint8_t loom_amdgpu_atomic_ordering(const loom_op_t* op) {
  if (loom_view_atomic_cmpxchg_isa(op)) {
    return loom_view_atomic_cmpxchg_success_ordering(op);
  }
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_ordering(op);
  }
  return loom_view_atomic_rmw_ordering(op);
}

static uint8_t loom_amdgpu_atomic_failure_ordering(const loom_op_t* op) {
  if (loom_view_atomic_cmpxchg_isa(op)) {
    return loom_view_atomic_cmpxchg_failure_ordering(op);
  }
  return loom_amdgpu_atomic_ordering(op);
}

static uint8_t loom_amdgpu_atomic_scope(const loom_op_t* op) {
  if (loom_view_atomic_cmpxchg_isa(op)) {
    return loom_view_atomic_cmpxchg_scope(op);
  }
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_scope(op);
  }
  return loom_view_atomic_rmw_scope(op);
}

static loom_amdgpu_vector_memory_cache_policy_encoding_t
loom_amdgpu_atomic_vector_cache_encoding(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  return descriptor_set_info == NULL
             ? LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE
             : descriptor_set_info->vector_memory.cache_policy_encoding;
}

static bool loom_amdgpu_atomic_prefers_global_saddr(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_memory_space_t memory_space) {
  return memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
         loom_amdgpu_atomic_vector_cache_encoding(descriptor_set) ==
             LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH;
}

static bool loom_amdgpu_atomic_ordering_has_acquire(uint8_t ordering) {
  switch (ordering) {
    case LOOM_ATOMIC_ORDERING_ACQUIRE:
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_atomic_ordering_has_release(uint8_t ordering) {
  switch (ordering) {
    case LOOM_ATOMIC_ORDERING_RELEASE:
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_atomic_has_acquire_ordering(const loom_op_t* op) {
  return loom_amdgpu_atomic_ordering_has_acquire(
             loom_amdgpu_atomic_ordering(op)) ||
         loom_amdgpu_atomic_ordering_has_acquire(
             loom_amdgpu_atomic_failure_ordering(op));
}

static bool loom_amdgpu_atomic_has_release_ordering(const loom_op_t* op) {
  return loom_amdgpu_atomic_ordering_has_release(
      loom_amdgpu_atomic_ordering(op));
}

static bool loom_amdgpu_atomic_global_ordering_supported(
    const loom_low_descriptor_set_t* descriptor_set, uint8_t ordering) {
  if (ordering != LOOM_ATOMIC_ORDERING_ACQUIRE &&
      ordering != LOOM_ATOMIC_ORDERING_RELEASE &&
      ordering != LOOM_ATOMIC_ORDERING_ACQ_REL &&
      ordering != LOOM_ATOMIC_ORDERING_SEQ_CST) {
    return false;
  }
  const loom_amdgpu_vector_memory_cache_policy_encoding_t encoding =
      loom_amdgpu_atomic_vector_cache_encoding(descriptor_set);
  return encoding ==
             LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC ||
         encoding ==
             LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH;
}

static bool loom_amdgpu_atomic_memory_space_is_device_visible(
    loom_value_fact_memory_space_t memory_space) {
  return memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
         memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC;
}

static bool loom_amdgpu_atomic_ordering_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_memory_space_t memory_space, uint8_t ordering) {
  if (ordering == LOOM_ATOMIC_ORDERING_RELAXED) {
    return true;
  }
  if (loom_amdgpu_atomic_memory_space_is_device_visible(memory_space)) {
    return loom_amdgpu_atomic_global_ordering_supported(descriptor_set,
                                                        ordering);
  }
  if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return false;
  }
  return ordering == LOOM_ATOMIC_ORDERING_ACQUIRE ||
         ordering == LOOM_ATOMIC_ORDERING_RELEASE ||
         ordering == LOOM_ATOMIC_ORDERING_ACQ_REL ||
         ordering == LOOM_ATOMIC_ORDERING_SEQ_CST;
}

static bool loom_amdgpu_atomic_orderings_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_memory_space_t memory_space, const loom_op_t* op) {
  return loom_amdgpu_atomic_ordering_supported(
             descriptor_set, memory_space, loom_amdgpu_atomic_ordering(op)) &&
         loom_amdgpu_atomic_ordering_supported(
             descriptor_set, memory_space,
             loom_amdgpu_atomic_failure_ordering(op));
}

static bool loom_amdgpu_atomic_value_kind_matches(
    loom_type_t value_type, loom_amdgpu_atomic_value_kind_t value_kind) {
  switch (value_kind) {
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32:
      return loom_amdgpu_type_is_i32(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32:
      return loom_amdgpu_type_is_f32(value_type);
  }
  return false;
}

static bool loom_amdgpu_atomic_value_as_i32_constant(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, int64_t* out_value) {
  *out_value = 0;
  return fact_table != NULL &&
         loom_amdgpu_type_is_i32(loom_module_value_type(module, value_id)) &&
         loom_amdgpu_value_facts_as_exact_i32(
             loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

static bool loom_amdgpu_atomic_value_as_f32_bit_pattern(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, uint32_t* out_bit_pattern) {
  *out_bit_pattern = 0;
  return fact_table != NULL &&
         loom_amdgpu_type_is_f32(loom_module_value_type(module, value_id)) &&
         loom_amdgpu_value_facts_as_f32_bit_pattern(
             loom_value_fact_table_lookup(fact_table, value_id),
             out_bit_pattern);
}

static bool loom_amdgpu_atomic_value_can_feed_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_amdgpu_atomic_value_kind_t value_kind) {
  if (loom_amdgpu_source_value_prefers_vgpr(module, fact_table,
                                            /*view_regions=*/NULL, value_id)) {
    return true;
  }
  if (value_kind == LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32) {
    uint32_t unused_bit_pattern = 0;
    return loom_amdgpu_atomic_value_as_f32_bit_pattern(
        module, fact_table, value_id, &unused_bit_pattern);
  }
  int64_t unused_value = 0;
  return loom_amdgpu_atomic_value_as_i32_constant(module, fact_table, value_id,
                                                  &unused_value);
}

static bool loom_amdgpu_atomic_value_can_feed_vgpr_operand(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op,
    const loom_amdgpu_atomic_descriptor_candidate_t* candidate) {
  if (candidate->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    const loom_value_id_t expected =
        loom_view_atomic_cmpxchg_expected(source_op);
    const loom_value_id_t replacement =
        loom_view_atomic_cmpxchg_replacement(source_op);
    return loom_amdgpu_atomic_value_can_feed_vgpr(
               module, fact_table, expected,
               LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32) &&
           loom_amdgpu_atomic_value_can_feed_vgpr(
               module, fact_table, replacement,
               LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32);
  }
  const loom_value_id_t value_id = loom_amdgpu_atomic_value(source_op);
  return loom_amdgpu_atomic_value_can_feed_vgpr(module, fact_table, value_id,
                                                candidate->value_kind);
}

static bool loom_amdgpu_atomic_source_plan_proves_workgroup_root(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source) {
  if (source->root_value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* root_value =
      loom_module_value(module, source->root_value_id);
  if (loom_value_is_block_arg(root_value)) {
    return false;
  }
  const loom_op_t* root_op = loom_value_def_op(root_value);
  return root_op != NULL && loom_buffer_alloca_isa(root_op);
}

static bool loom_amdgpu_atomic_select_descriptor(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_atomic_selection_t* selection, loom_type_t value_type,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  selection->descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const uint8_t atomic_kind = loom_amdgpu_atomic_kind(source_op);
  const bool prefer_global_saddr = loom_amdgpu_atomic_prefers_global_saddr(
      descriptor_set, selection->source.memory_space);
  bool found_kind = false;
  bool found_type = false;
  const iree_host_size_t pass_count = prefer_global_saddr ? 2 : 1;
  for (iree_host_size_t pass = 0; pass < pass_count; ++pass) {
    const bool global_saddr_only = prefer_global_saddr && pass == 0;
    for (iree_host_size_t i = 0; i < kLoomAmdgpuAtomicDescriptorCandidateCount;
         ++i) {
      const loom_amdgpu_atomic_descriptor_candidate_t* candidate =
          &kLoomAmdgpuAtomicDescriptorCandidates[i];
      if (prefer_global_saddr) {
        if (global_saddr_only &&
            candidate->address_form !=
                LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
          continue;
        }
        if (!global_saddr_only &&
            candidate->address_form ==
                LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
          continue;
        }
      }
      if (candidate->memory_space != selection->source.memory_space ||
          candidate->operation_kind != selection->operation_kind) {
        continue;
      }
      if (selection->operation_kind != LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG &&
          candidate->atomic_kind != atomic_kind) {
        continue;
      }
      found_kind = true;
      if (!loom_amdgpu_atomic_value_kind_matches(value_type,
                                                 candidate->value_kind)) {
        continue;
      }
      found_type = true;
      if (!loom_amdgpu_atomic_value_can_feed_vgpr_operand(
              module, fact_table, source_op, candidate)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT;
        return false;
      }
      const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
          descriptor_set, candidate->descriptor_ref);
      if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        continue;
      }
      const loom_low_descriptor_t* descriptor =
          loom_low_descriptor_set_descriptor_at(descriptor_set,
                                                descriptor_ordinal);
      IREE_ASSERT(descriptor != NULL);
      selection->address_form = candidate->address_form;
      selection->descriptor_ref = candidate->descriptor_ref;
      if (loom_amdgpu_descriptor_has_implicit_resource_operand(descriptor_set,
                                                               descriptor)) {
        selection->flags |= LOOM_AMDGPU_ATOMIC_PLAN_REQUIRES_M0;
      }
      return true;
    }
  }
  if (found_type) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
  } else {
    diagnostic->rejection_bits |=
        found_kind ? LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE
                   : LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND;
  }
  return false;
}

static bool loom_amdgpu_atomic_uses_buffer_resource(
    const loom_amdgpu_atomic_plan_t* plan) {
  return plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
         plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
}

static bool loom_amdgpu_atomic_selection_uses_buffer_resource(
    const loom_amdgpu_atomic_selection_t* selection) {
  return selection->source.memory_space ==
             LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
         selection->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
}

static bool loom_amdgpu_atomic_uses_flat_address(
    const loom_amdgpu_atomic_plan_t* plan) {
  return plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC &&
         plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT;
}

static void loom_amdgpu_atomic_append_packet_resource_operands(
    const loom_amdgpu_atomic_plan_t* plan, loom_value_id_t low_saddr,
    loom_value_id_t low_m0, loom_value_id_t* operands,
    iree_host_size_t* operand_count) {
  if (plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    operands[(*operand_count)++] = low_saddr;
  }
  if (iree_any_bit_set(plan->flags, LOOM_AMDGPU_ATOMIC_PLAN_REQUIRES_M0)) {
    operands[(*operand_count)++] = low_m0;
  }
}

static bool loom_amdgpu_atomic_select_offset(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_atomic_selection_t* selection,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  const uint32_t descriptor_ordinal =
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  loom_low_immediate_kind_t expected_kind = LOOM_LOW_IMMEDIATE_KIND_UNSIGNED;
  if (selection->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    expected_kind = LOOM_LOW_IMMEDIATE_KIND_SIGNED;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (selection->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
    if (!loom_amdgpu_descriptor_offset_immediate_info(
            descriptor_set, descriptor_ordinal, 1, expected_kind,
            &offset_info) ||
        offset_info.unit_byte_count == 0) {
      expected_kind = LOOM_LOW_IMMEDIATE_KIND_SIGNED;
      if (!loom_amdgpu_descriptor_offset_immediate_info(
              descriptor_set, descriptor_ordinal, 1, expected_kind,
              &offset_info) ||
          offset_info.unit_byte_count == 0) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE;
        return false;
      }
    }
  } else if (!loom_amdgpu_descriptor_offset_immediate_info(
                 descriptor_set, descriptor_ordinal, 1, expected_kind,
                 &offset_info) ||
             offset_info.unit_byte_count == 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE;
    return false;
  }
  if (expected_kind == LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    const int64_t signed_max = offset_info.unsigned_max > INT64_MAX
                                   ? INT64_MAX
                                   : (int64_t)offset_info.unsigned_max;
    if (offset_info.unit_byte_count != 1 ||
        selection->source.static_byte_offset < offset_info.signed_min ||
        selection->source.static_byte_offset > signed_max) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    selection->immediate_offset = selection->source.static_byte_offset;
    selection->scalar_byte_offset = 0;
    if (!loom_amdgpu_source_memory_offset_fits_u32(&selection->source,
                                                   /*static_byte_offset=*/0)) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    return true;
  }
  if (selection->source.static_byte_offset < 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  const uint64_t static_byte_offset =
      (uint64_t)selection->source.static_byte_offset;
  if (loom_amdgpu_atomic_selection_uses_buffer_resource(selection)) {
    if (offset_info.unit_byte_count != 1) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    const uint64_t immediate_offset =
        iree_min(static_byte_offset, offset_info.unsigned_max);
    const uint64_t scalar_byte_offset = static_byte_offset - immediate_offset;
    if (scalar_byte_offset > UINT32_MAX) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    selection->immediate_offset = (int64_t)immediate_offset;
    selection->scalar_byte_offset = (uint32_t)scalar_byte_offset;
    if (!loom_amdgpu_source_memory_offset_fits_u32(
            &selection->source, selection->source.static_byte_offset)) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    return true;
  }
  if ((static_byte_offset % offset_info.unit_byte_count) != 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  const uint64_t encoded_offset =
      static_byte_offset / offset_info.unit_byte_count;
  if (encoded_offset > offset_info.unsigned_max) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  selection->immediate_offset = (int64_t)encoded_offset;
  selection->scalar_byte_offset = 0;
  if (!loom_amdgpu_source_memory_offset_fits_u32(
          &selection->source, selection->source.static_byte_offset)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_atomic_append_explicit_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_explicit_packet_immediate_template_t* immediates,
    iree_host_size_t immediate_count,
    loom_amdgpu_atomic_explicit_packet_selection_t* packets,
    iree_host_size_t packet_capacity, iree_host_size_t* inout_packet_count) {
  if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref)) {
    return false;
  }
  IREE_ASSERT(*inout_packet_count < packet_capacity);
  IREE_ASSERT(immediate_count <=
              LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY);
  loom_amdgpu_atomic_explicit_packet_selection_t* packet =
      &packets[(*inout_packet_count)++];
  *packet = (loom_amdgpu_atomic_explicit_packet_selection_t){
      .descriptor_ref = descriptor_ref,
      .immediate_count = immediate_count,
  };
  for (iree_host_size_t i = 0; i < immediate_count; ++i) {
    packet->immediates[i] = (loom_amdgpu_explicit_packet_immediate_template_t){
        .name = immediates[i].name,
        .value = immediates[i].value,
    };
  }
  return true;
}

static bool loom_amdgpu_atomic_append_wait_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_atomic_wait_packet_template_t* wait_packet,
    loom_amdgpu_atomic_explicit_packet_selection_t* waits,
    iree_host_size_t wait_capacity, iree_host_size_t* inout_wait_count) {
  return loom_amdgpu_atomic_append_explicit_packet(
      descriptor_set, wait_packet->descriptor_ref, wait_packet->immediates,
      wait_packet->immediate_count, waits, wait_capacity, inout_wait_count);
}

static bool loom_amdgpu_atomic_append_cache_control_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_atomic_explicit_packet_selection_t* cache_controls,
    iree_host_size_t cache_control_capacity,
    iree_host_size_t* inout_cache_control_count) {
  return loom_amdgpu_atomic_append_explicit_packet(
      descriptor_set, descriptor_ref, /*immediates=*/NULL,
      /*immediate_count=*/0, cache_controls, cache_control_capacity,
      inout_cache_control_count);
}

static bool loom_amdgpu_atomic_append_gfx12_global_cache_control_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_atomic_explicit_packet_selection_t* cache_controls,
    iree_host_size_t cache_control_capacity,
    iree_host_size_t* inout_cache_control_count) {
  const loom_amdgpu_explicit_packet_immediate_template_t immediates[] = {
      {IREE_SVL("scope"), LOOM_AMDGPU_GFX12_SCOPE_DEVICE},
  };
  return loom_amdgpu_atomic_append_explicit_packet(
      descriptor_set, descriptor_ref, immediates, IREE_ARRAYSIZE(immediates),
      cache_controls, cache_control_capacity, inout_cache_control_count);
}

static bool loom_amdgpu_atomic_select_global_release_waits(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding,
    loom_amdgpu_atomic_ordering_selection_t* ordering) {
  switch (encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      if (!loom_amdgpu_atomic_append_wait_packet(
              descriptor_set, &kAmdgpuGfx11VmemLoadWaitPacket,
              ordering->pre_atomic_waits,
              IREE_ARRAYSIZE(ordering->pre_atomic_waits),
              &ordering->pre_atomic_wait_count)) {
        return false;
      }
      return loom_amdgpu_atomic_append_wait_packet(
          descriptor_set, &kAmdgpuGfx11VmemStoreWaitPacket,
          ordering->pre_atomic_waits,
          IREE_ARRAYSIZE(ordering->pre_atomic_waits),
          &ordering->pre_atomic_wait_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_atomic_select_global_acquire_waits(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding,
    loom_amdgpu_atomic_ordering_selection_t* ordering,
    loom_amdgpu_atomic_operation_kind_t operation_kind) {
  const loom_amdgpu_atomic_wait_packet_template_t* wait_packet = NULL;
  switch (encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      wait_packet = operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE
                        ? &kAmdgpuGfx11VmemStoreWaitPacket
                        : &kAmdgpuGfx11VmemLoadWaitPacket;
      break;
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      wait_packet = operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE
                        ? &kAmdgpuGfx12VmemStoreWaitPacket
                        : &kAmdgpuGfx12VmemLoadWaitPacket;
      break;
    default:
      return false;
  }
  return loom_amdgpu_atomic_append_wait_packet(
      descriptor_set, wait_packet, ordering->post_atomic_waits,
      IREE_ARRAYSIZE(ordering->post_atomic_waits),
      &ordering->post_atomic_wait_count);
}

static bool loom_amdgpu_atomic_select_global_acquire_cache_controls(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding,
    loom_amdgpu_atomic_ordering_selection_t* ordering) {
  switch (encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      if (!loom_amdgpu_atomic_append_cache_control_packet(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV,
              ordering->post_atomic_cache_controls,
              IREE_ARRAYSIZE(ordering->post_atomic_cache_controls),
              &ordering->post_atomic_cache_control_descriptor_count)) {
        return false;
      }
      return loom_amdgpu_atomic_append_cache_control_packet(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV,
          ordering->post_atomic_cache_controls,
          IREE_ARRAYSIZE(ordering->post_atomic_cache_controls),
          &ordering->post_atomic_cache_control_descriptor_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return loom_amdgpu_atomic_append_gfx12_global_cache_control_packet(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV,
          ordering->post_atomic_cache_controls,
          IREE_ARRAYSIZE(ordering->post_atomic_cache_controls),
          &ordering->post_atomic_cache_control_descriptor_count);
    default:
      return false;
  }
}

static bool loom_amdgpu_atomic_select_global_ordering(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_atomic_selection_t* selection, const loom_op_t* source_op,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  selection->ordering = (loom_amdgpu_atomic_ordering_selection_t){0};
  const loom_amdgpu_vector_memory_cache_policy_encoding_t encoding =
      loom_amdgpu_atomic_vector_cache_encoding(descriptor_set);
  if (!loom_amdgpu_atomic_memory_space_is_device_visible(
          selection->source.memory_space) ||
      (!loom_amdgpu_atomic_has_release_ordering(source_op) &&
       !loom_amdgpu_atomic_has_acquire_ordering(source_op))) {
    return true;
  }

  if (loom_amdgpu_atomic_has_release_ordering(source_op)) {
    if (!loom_amdgpu_atomic_select_global_release_waits(
            descriptor_set, encoding, &selection->ordering)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
      return false;
    }
  }
  if (loom_amdgpu_atomic_has_acquire_ordering(source_op)) {
    if (!loom_amdgpu_atomic_select_global_acquire_waits(
            descriptor_set, encoding, &selection->ordering,
            selection->operation_kind)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
      return false;
    }
    if (!loom_amdgpu_atomic_select_global_acquire_cache_controls(
            descriptor_set, encoding, &selection->ordering)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
      return false;
    }
  }
  return true;
}

static void loom_amdgpu_atomic_select_packet_attrs(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_atomic_selection_t* selection) {
  selection->packet_attrs = (loom_amdgpu_atomic_packet_attrs_t){0};
  selection->packet_attrs.scope_attr_name_id = LOOM_STRING_ID_INVALID;
  if (!loom_amdgpu_atomic_memory_space_is_device_visible(
          selection->source.memory_space)) {
    return;
  }
  const loom_amdgpu_vector_memory_cache_policy_encoding_t encoding =
      loom_amdgpu_atomic_vector_cache_encoding(descriptor_set);
  if (encoding !=
      LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH) {
    return;
  }
  selection->packet_attrs.flags |= LOOM_AMDGPU_ATOMIC_PACKET_ATTR_SCOPE;
  selection->packet_attrs.scope = LOOM_AMDGPU_GFX12_SCOPE_DEVICE;
}

static bool loom_amdgpu_atomic_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_atomic_selection_t* out_selection,
    loom_low_source_memory_access_diagnostic_t* source_diagnostic,
    loom_amdgpu_memory_access_diagnostic_t* memory_diagnostic,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  *out_selection = (loom_amdgpu_atomic_selection_t){
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  *source_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  *memory_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};
  *diagnostic = (loom_amdgpu_atomic_diagnostic_t){0};
  if (!loom_amdgpu_view_atomic_isa(source_op)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP;
    return false;
  }

  if (!loom_low_source_memory_access_plan_build(module, fact_table, source_op,
                                                &out_selection->source,
                                                source_diagnostic)) {
    return false;
  }
  switch (out_selection->source.operation_kind) {
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE:
      out_selection->operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE;
      break;
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW:
      out_selection->operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_RMW;
      break;
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_CMPXCHG:
      out_selection->operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG;
      break;
    default:
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND;
      return false;
  }
  switch (out_selection->source.memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      out_selection->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
      if (!loom_amdgpu_atomic_source_plan_proves_workgroup_root(
              module, &out_selection->source)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_ATOMIC_REJECTION_WORKGROUP_ROOT;
        return false;
      }
      break;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      out_selection->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
      break;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      out_selection->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT;
      if (loom_low_source_memory_access_is_dynamic(&out_selection->source)) {
        loom_amdgpu_memory_access_record_flat_dynamic_address_rejection(
            module, &out_selection->source, memory_diagnostic);
        return false;
      }
      break;
    default:
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE;
      return false;
  }
  if (out_selection->source.element_byte_count != 4 ||
      out_selection->source.vector_lane_count != 1 ||
      out_selection->source.vector_lane_byte_stride != 4) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE;
    return false;
  }
  if (loom_amdgpu_memory_cache_policy_is_present(
          &out_selection->source.cache_policy)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_CACHE_POLICY;
    return false;
  }
  if (!loom_amdgpu_atomic_orderings_supported(
          descriptor_set, out_selection->source.memory_space, source_op)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING;
    return false;
  }
  const uint8_t expected_scope = out_selection->source.memory_space ==
                                         LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
                                     ? LOOM_ATOMIC_SCOPE_WORKGROUP
                                     : LOOM_ATOMIC_SCOPE_DEVICE;
  if (loom_amdgpu_atomic_scope(source_op) != expected_scope) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE;
    return false;
  }

  loom_amdgpu_memory_access_t memory_access = {
      .source = out_selection->source,
      .address_form = out_selection->address_form,
  };
  if (!loom_amdgpu_memory_access_select_dynamic_term_kinds(
          module, /*fact_table=*/NULL, /*view_regions=*/NULL, &memory_access,
          memory_diagnostic)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++i) {
    out_selection->dynamic_term_kinds[i] = memory_access.dynamic_term_kinds[i];
  }

  const loom_type_t value_type =
      out_selection->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG
          ? loom_module_value_type(module,
                                   loom_view_atomic_cmpxchg_old(source_op))
          : loom_module_value_type(module, loom_amdgpu_atomic_value(source_op));
  if (!loom_amdgpu_atomic_select_descriptor(module, fact_table, descriptor_set,
                                            source_op, out_selection,
                                            value_type, diagnostic)) {
    return false;
  }
  loom_amdgpu_atomic_select_packet_attrs(descriptor_set, out_selection);
  if (!loom_amdgpu_atomic_select_global_ordering(descriptor_set, out_selection,
                                                 source_op, diagnostic)) {
    return false;
  }
  return loom_amdgpu_atomic_select_offset(
      descriptor_set, out_selection->descriptor_ref, out_selection, diagnostic);
}

static iree_status_t loom_amdgpu_atomic_resolve_explicit_packet_selection(
    loom_low_lower_context_t* context,
    const loom_amdgpu_atomic_explicit_packet_selection_t* selection,
    loom_amdgpu_explicit_packet_plan_t* out_plan) {
  bool present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_explicit_packet_plan(
      context, selection->descriptor_ref, selection->immediates,
      selection->immediate_count, out_plan, &present));
  if (!present) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "selected AMDGPU explicit atomic ordering packet "
                            "descriptor is not present");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_atomic_resolve_ordering_selection(
    loom_low_lower_context_t* context,
    const loom_amdgpu_atomic_ordering_selection_t* selection,
    loom_amdgpu_atomic_ordering_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_atomic_ordering_plan_t){0};
  out_plan->pre_atomic_wait_count = selection->pre_atomic_wait_count;
  for (iree_host_size_t i = 0; i < selection->pre_atomic_wait_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_atomic_resolve_explicit_packet_selection(
        context, &selection->pre_atomic_waits[i],
        &out_plan->pre_atomic_waits[i]));
  }
  out_plan->post_atomic_wait_count = selection->post_atomic_wait_count;
  for (iree_host_size_t i = 0; i < selection->post_atomic_wait_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_atomic_resolve_explicit_packet_selection(
        context, &selection->post_atomic_waits[i],
        &out_plan->post_atomic_waits[i]));
  }
  out_plan->post_atomic_cache_control_descriptor_count =
      selection->post_atomic_cache_control_descriptor_count;
  for (iree_host_size_t i = 0;
       i < selection->post_atomic_cache_control_descriptor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_atomic_resolve_explicit_packet_selection(
        context, &selection->post_atomic_cache_controls[i],
        &out_plan->post_atomic_cache_controls[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_atomic_resolve_selection(
    loom_low_lower_context_t* context,
    const loom_amdgpu_atomic_selection_t* selection,
    loom_amdgpu_atomic_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_atomic_plan_t){
      .source = selection->source,
      .flags = selection->flags,
      .operation_kind = selection->operation_kind,
      .address_form = selection->address_form,
      .immediate_offset = selection->immediate_offset,
      .scalar_byte_offset = selection->scalar_byte_offset,
      .packet_attrs = selection->packet_attrs,
  };
  for (iree_host_size_t i = 0; i < LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++i) {
    out_plan->dynamic_term_kinds[i] = selection->dynamic_term_kinds[i];
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, selection->descriptor_ref, &out_plan->descriptor));
  if (iree_any_bit_set(out_plan->packet_attrs.flags,
                       LOOM_AMDGPU_ATOMIC_PACKET_ATTR_SCOPE)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_intern(
        context, IREE_SV("scope"), &out_plan->packet_attrs.scope_attr_name_id));
  }
  return loom_amdgpu_atomic_resolve_ordering_selection(
      context, &selection->ordering, &out_plan->ordering);
}

static iree_string_view_t loom_amdgpu_atomic_rejection_key(
    loom_amdgpu_atomic_rejection_flags_t rejection_bits) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kAmdgpuAtomicRejectionKeys);
       ++i) {
    const loom_amdgpu_atomic_rejection_key_t* row =
        &kAmdgpuAtomicRejectionKeys[i];
    if (iree_any_bit_set(rejection_bits, row->rejection_bit)) {
      return row->constraint_key;
    }
  }
  return IREE_SV("atomic.representability");
}

iree_status_t loom_amdgpu_select_view_atomic_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_atomic_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_atomic_plan_t){0};
  *out_selected = false;
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  loom_amdgpu_atomic_diagnostic_t diagnostic = {0};
  loom_amdgpu_atomic_selection_t selection = {0};
  if (!loom_amdgpu_atomic_select(loom_low_lower_context_module(context),
                                 loom_low_lower_context_fact_table(context),
                                 loom_low_lower_context_descriptor_set(context),
                                 source_op, &selection, &source_diagnostic,
                                 &memory_diagnostic, &diagnostic)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_atomic_resolve_selection(context, &selection, out_plan));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_atomic_value_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t source_value = loom_amdgpu_atomic_value(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, source_value);
  if (loom_amdgpu_type_is_i32(source_type)) {
    return loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, source_value, out_low_value);
  }
  if (loom_amdgpu_type_is_f32(source_type)) {
    return loom_amdgpu_lookup_or_materialize_vgpr_f32(
        context, source_op, source_value, out_low_value);
  }

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));
  loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU atomic selected a non-VGPR dynamic update value");
}

static iree_status_t loom_amdgpu_materialize_atomic_value_as_fresh_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t source_value = loom_amdgpu_atomic_value(source_op);
  int64_t i32_value = 0;
  if (loom_amdgpu_value_as_i32_constant(context, source_value, &i32_value)) {
    loom_type_t vgpr_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
    return loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        (uint32_t)(int32_t)i32_value, vgpr_type, out_low_value);
  }
  uint32_t f32_bits = 0;
  if (loom_amdgpu_value_as_f32_constant(context, source_value, &f32_bits)) {
    loom_type_t vgpr_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                                      f32_bits, vgpr_type, out_low_value);
  }

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));
  return loom_amdgpu_emit_vgpr_b32_copy(context, source_op, low_value,
                                        out_low_value);
}

static iree_status_t loom_amdgpu_lookup_atomic_cmpxchg_values_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_expected, loom_value_id_t* out_low_replacement) {
  *out_low_expected = LOOM_VALUE_ID_INVALID;
  *out_low_replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, loom_view_atomic_cmpxchg_expected(source_op),
      out_low_expected));
  return loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, loom_view_atomic_cmpxchg_replacement(source_op),
      out_low_replacement);
}

static iree_status_t loom_amdgpu_emit_atomic_cmpxchg_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_expected, loom_value_id_t low_replacement,
    loom_type_t pair_type, loom_value_id_t* out_low_pair) {
  *out_low_pair = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[] = {low_expected, low_replacement};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), operands,
      IREE_ARRAYSIZE(operands), pair_type, source_op->location, &concat_op));
  *out_low_pair = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_atomic_buffer_soffset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_plan_t* plan, loom_value_id_t* out_low_soffset) {
  return loom_amdgpu_emit_sgpr_byte_offset_terms(
      context, source_op, &plan->source, plan->dynamic_term_kinds,
      plan->scalar_byte_offset, out_low_soffset);
}

static iree_status_t loom_amdgpu_emit_atomic_ordering_waits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_explicit_packet_plan_t* waits,
    iree_host_size_t wait_count) {
  for (iree_host_size_t i = 0; i < wait_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_explicit_packet_plan(context, source_op, &waits[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_atomic_cache_controls(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_explicit_packet_plan_t* cache_controls,
    iree_host_size_t cache_control_count) {
  for (iree_host_size_t i = 0; i < cache_control_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_explicit_packet_plan(
        context, source_op, &cache_controls[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_atomic_packet_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_atomic_packet_attrs_t* packet_attrs,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  if (iree_any_bit_set(packet_attrs->flags,
                       LOOM_AMDGPU_ATOMIC_PACKET_ATTR_SCOPE)) {
    IREE_ASSERT(packet_attrs->scope_attr_name_id != LOOM_STRING_ID_INVALID);
    if (*inout_attr_count >= attr_capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AMDGPU low attr capacity exceeded");
    }
    attrs[*inout_attr_count] = (loom_named_attr_t){
        .name_id = packet_attrs->scope_attr_name_id,
        .value = loom_attr_i64(packet_attrs->scope),
    };
    *inout_attr_count += 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_atomic_post_ordering(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_ordering_plan_t* ordering) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_ordering_waits(
      context, source_op, ordering->post_atomic_waits,
      ordering->post_atomic_wait_count));
  return loom_amdgpu_emit_atomic_cache_controls(
      context, source_op, ordering->post_atomic_cache_controls,
      ordering->post_atomic_cache_control_descriptor_count);
}

iree_status_t loom_amdgpu_lower_view_atomic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_plan_t* plan) {
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, plan->source.view_value_id, &low_resource));

  loom_amdgpu_memory_access_t access = {
      .source = plan->source,
      .address_form = plan->address_form,
      .immediate_offset = plan->immediate_offset,
      .scalar_byte_offset = plan->scalar_byte_offset,
      .payload_register_count = 1,
      .packet_byte_count = 4,
  };
  for (iree_host_size_t i = 0; i < LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++i) {
    access.dynamic_term_kinds[i] = plan->dynamic_term_kinds[i];
  }
  const loom_value_id_t low_base_addr =
      plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
          ? low_resource
          : LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_atomic_uses_flat_address(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_vaddr(
        context, source_op, &access, low_resource, &low_vaddr));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
        context, source_op, &access, low_base_addr, &low_vaddr));
  }

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, &access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_atomic_packet_attrs(
      context, &plan->packet_attrs, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_named_attr_slice_t packet_attrs =
      loom_make_named_attr_slice(attrs, attr_count);

  loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
  if (plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, &access, low_resource, &low_saddr));
  }
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_buffer_soffset(
        context, source_op, plan, &low_soffset));
  }
  loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
        context, source_op, low_resource, &access.source, &low_descriptor));
  }
  loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
  if (iree_any_bit_set(plan->flags, LOOM_AMDGPU_ATOMIC_PLAN_REQUIRES_M0)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_m0_u32(
        context, source_op, &plan->descriptor, 0, &low_m0));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_ordering_waits(
      context, source_op, plan->ordering.pre_atomic_waits,
      plan->ordering.pre_atomic_wait_count));

  if (plan->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    loom_value_id_t low_expected = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_replacement = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_atomic_cmpxchg_values_as_vgpr(
        context, source_op, &low_expected, &low_replacement));
    loom_type_t old_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &old_type));
    loom_value_id_t low_old = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
      loom_type_t pair_type = loom_type_none();
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_make_vgpr_range_type(context, 2, &pair_type));
      loom_value_id_t low_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_cmpxchg_pair(
          context, source_op, low_expected, low_replacement, pair_type,
          &low_pair));
      loom_value_id_t operands[] = {low_pair, low_descriptor, low_vaddr,
                                    low_soffset};
      const loom_tied_result_t tied_result = {
          .result_index = 0,
          .operand_index = 0,
      };
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
          packet_attrs, &pair_type, 1, &tied_result, 1, source_op->location,
          &low_op));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op,
          loom_value_slice_get(loom_low_op_results(low_op), 0), 0, old_type,
          &low_old));
    } else if (plan->address_form ==
                   LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR ||
               loom_amdgpu_atomic_uses_flat_address(plan)) {
      loom_type_t pair_type = loom_type_none();
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_make_vgpr_range_type(context, 2, &pair_type));
      loom_value_id_t low_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_cmpxchg_pair(
          context, source_op, low_expected, low_replacement, pair_type,
          &low_pair));
      loom_value_id_t operands[4] = {low_vaddr, low_pair};
      iree_host_size_t operand_count = 2;
      loom_amdgpu_atomic_append_packet_resource_operands(
          plan, low_saddr, low_m0, operands, &operand_count);
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->descriptor, operands, operand_count, packet_attrs,
          &old_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
          source_op->location, &low_op));
      low_old = loom_value_slice_get(loom_low_op_results(low_op), 0);
    } else {
      loom_value_id_t operands[] = {low_vaddr, low_expected, low_replacement};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
          packet_attrs, &old_type, 1, /*tied_results=*/NULL,
          /*tied_result_count=*/0, source_op->location, &low_op));
      low_old = loom_value_slice_get(loom_low_op_results(low_op), 0);
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_post_ordering(
        context, source_op, &plan->ordering));
    return loom_low_lower_bind_value(
        context, loom_view_atomic_cmpxchg_old(source_op), low_old);
  }

  if (plan->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_RMW) {
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
        context, source_op, loom_view_atomic_rmw_result(source_op),
        &result_type));
    if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
      loom_value_id_t low_fresh_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_atomic_value_as_fresh_vgpr(
          context, source_op, &low_fresh_value));
      loom_value_id_t operands[] = {low_fresh_value, low_descriptor, low_vaddr,
                                    low_soffset};
      const loom_tied_result_t tied_result = {
          .result_index = 0,
          .operand_index = 0,
      };
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
          packet_attrs, &result_type, 1, &tied_result, 1, source_op->location,
          &low_op));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_post_ordering(
          context, source_op, &plan->ordering));
      return loom_low_lower_bind_value(
          context, loom_view_atomic_rmw_result(source_op),
          loom_value_slice_get(loom_low_op_results(low_op), 0));
    }

    loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_atomic_value_as_vgpr(
        context, source_op, &low_value));
    loom_value_id_t operands[4] = {low_vaddr, low_value};
    iree_host_size_t operand_count = 2;
    loom_amdgpu_atomic_append_packet_resource_operands(
        plan, low_saddr, low_m0, operands, &operand_count);
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->descriptor, operands, operand_count, packet_attrs,
        &result_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &low_op));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_post_ordering(
        context, source_op, &plan->ordering));
    return loom_low_lower_bind_value(
        context, loom_view_atomic_rmw_result(source_op),
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_lookup_atomic_value_as_vgpr(context, source_op, &low_value));
  if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
    loom_value_id_t operands[] = {low_value, low_descriptor, low_vaddr,
                                  low_soffset};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
        packet_attrs, /*result_types=*/NULL, /*result_count=*/0,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &low_op));
    return loom_amdgpu_emit_atomic_post_ordering(context, source_op,
                                                 &plan->ordering);
  }

  loom_value_id_t operands[4] = {low_vaddr, low_value};
  iree_host_size_t operand_count = 2;
  loom_amdgpu_atomic_append_packet_resource_operands(plan, low_saddr, low_m0,
                                                     operands, &operand_count);
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, operand_count, packet_attrs,
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_op));
  return loom_amdgpu_emit_atomic_post_ordering(context, source_op,
                                               &plan->ordering);
}

iree_status_t loom_amdgpu_low_legality_verify_view_atomic(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_view_atomic_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_atomic_selection_t selection = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  loom_amdgpu_atomic_diagnostic_t diagnostic = {0};
  const loom_module_t* module = loom_target_low_legality_module(context);
  if (loom_amdgpu_atomic_select(
          module, loom_target_low_legality_fact_table(context),
          loom_target_low_legality_descriptor_set(context), op, &selection,
          &source_diagnostic, &memory_diagnostic, &diagnostic)) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = IREE_SV("atomic.representability");
  if (source_diagnostic.rejection_bits != 0) {
    constraint_key = loom_low_source_memory_access_rejection_key(
        source_diagnostic.rejection_bits);
  } else if (memory_diagnostic.rejection_bits != 0) {
    bool handled = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_access_rejection_diagnostic(
        context, op, &selection.source, &memory_diagnostic, &handled));
    if (handled) {
      return iree_ok_status();
    }
    constraint_key = loom_amdgpu_memory_access_rejection_key(
        memory_diagnostic.rejection_bits);
  } else {
    constraint_key =
        loom_amdgpu_atomic_rejection_key(diagnostic.rejection_bits);
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}
