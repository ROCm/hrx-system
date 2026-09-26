// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Self-contained physical program plans for AIE2P leaf-object emission.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_PROGRAM_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_PROGRAM_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_aie2p_planned_slot_flag_bits_e {
  // Slot is a planner-synthesized architectural NOP.
  LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP = 1u << 0,
  // Slot materializes a structural Low control operation.
  LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL = 1u << 1,
  // Slot materializes one allocation-planned structural register move.
  LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_MOVE = 1u << 2,
  // Slot materializes one symbolic function-local storage address.
  LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_STORAGE_ADDRESS = 1u << 3,
};
typedef uint16_t loom_aie2p_planned_slot_flags_t;

// One physical instruction assigned to a bundle slot.
typedef struct loom_aie2p_planned_slot_t {
  // Encoded instruction value and its physical slot.
  loom_aie2p_encoded_slot_t encoded_slot;
  // Scheduled Low packet represented by this slot, or UINT32_MAX when the
  // planner synthesized the slot.
  uint32_t scheduled_packet_index;
  // Planner provenance flags.
  loom_aie2p_planned_slot_flags_t flags;
} loom_aie2p_planned_slot_t;

// One variable-width physical VLIW bundle.
typedef struct loom_aie2p_planned_bundle_t {
  // Physical issue cycle in the core program, including implicit NOP gaps.
  uint32_t issue_cycle;
  // Source-order Low block containing this bundle.
  uint32_t block_index;
  // Logical Low schedule cycle expanded into this physical bundle.
  uint32_t logical_issue_cycle;
  // Byte offset of this bundle in the contribution code section.
  uint32_t byte_offset;
  // First slot record in the owning plan.
  uint32_t slot_start;
  // Physical bundle format selected from the owned target table.
  loom_aie2p_bundle_format_id_t format;
  // Number of slot records in this bundle.
  uint8_t slot_count;
} loom_aie2p_planned_bundle_t;

// One contribution-relative branch target requiring final placement.
typedef struct loom_aie2p_planned_branch_fixup_t {
  // Bundle containing the branch instruction.
  uint32_t bundle_index;
  // Source-order Low block targeted by the branch.
  uint32_t target_block_index;
} loom_aie2p_planned_branch_fixup_t;

// One contribution-relative local-storage address requiring final placement.
typedef struct loom_aie2p_planned_storage_fixup_t {
  // Bundle containing the address-materialization instruction.
  uint32_t bundle_index;
  // Function-local storage space referenced by the instruction.
  loom_storage_space_t storage_space;
  // Byte offset from the placed storage-space base.
  uint64_t byte_offset;
} loom_aie2p_planned_storage_fixup_t;

// Exact storage required in one placement domain.
typedef struct loom_aie2p_leaf_storage_requirement_t {
  // Required byte length, excluding placement padding outside this domain.
  uint64_t byte_length;
  // Minimum placement alignment, or zero when no storage is required.
  uint64_t minimum_alignment;
} loom_aie2p_leaf_storage_requirement_t;

enum loom_aie2p_leaf_resource_flag_bits_e {
  // Resource extent is supplied through the Low extent operand.
  LOOM_AIE2P_LEAF_RESOURCE_FLAG_DYNAMIC_EXTENT = 1u << 0,
  // Resource carries a static byte extent.
  LOOM_AIE2P_LEAF_RESOURCE_FLAG_STATIC_EXTENT = 1u << 1,
  // Resource carries a cache-swizzle byte stride.
  LOOM_AIE2P_LEAF_RESOURCE_FLAG_CACHE_SWIZZLE_STRIDE = 1u << 2,
};
typedef uint16_t loom_aie2p_leaf_resource_flags_t;

// One detached Low resource import and its final physical-register binding.
typedef struct loom_aie2p_leaf_resource_import_t {
  // Resource table index selected by low.resource.
  uint64_t index;
  // Static resource extent when STATIC_EXTENT is set, otherwise zero.
  uint64_t extent;
  // Cache-swizzle byte stride when CACHE_SWIZZLE_STRIDE is set, otherwise
  // zero.
  uint32_t cache_swizzle_stride;
  // First AIE2P physical-register ID occupied by the imported value.
  uint32_t physical_register;
  // Number of logical physical-register units occupied by the imported value.
  uint32_t physical_register_count;
  // Physical register carrying a dynamic extent, or UINT32_MAX when absent.
  uint32_t extent_physical_register;
  // AIE2P descriptor-set register class owning the physical register.
  uint16_t descriptor_register_class_id;
  // AIE2P register class carrying a dynamic extent, or zero when absent.
  uint16_t extent_descriptor_register_class_id;
  // Number of physical-register units carrying a dynamic extent.
  uint32_t extent_physical_register_count;
  // Optional resource metadata carried by this record.
  loom_aie2p_leaf_resource_flags_t flags;
  // Low ABI import kind.
  loom_low_resource_import_kind_t import_kind;
  // Outer Loom type kind of the imported source value.
  loom_type_kind_t source_type_kind;
} loom_aie2p_leaf_resource_import_t;

// Immutable physical AIE2P program produced for one Low leaf function.
//
// The function name and all pointer tables are owned by the plan arena. The
// plan contains no compiler IR, schedule, allocation, storage-layout, or
// diagnostic state and may be emitted repeatedly without mutation.
typedef struct loom_aie2p_leaf_program_plan_t {
  // Arena-owned function name used for native sections and symbols.
  iree_string_view_t function_name;
  // Final requirements indexed by valid loom_storage_space_t values.
  loom_aie2p_leaf_storage_requirement_t
      storage_requirements[LOOM_STORAGE_SPACE_COUNT_];
  // Materialized spill payload already included in one storage requirement.
  loom_aie2p_leaf_storage_requirement_t spill;
  // Resource imports in Low entry-block order with final physical bindings.
  const loom_aie2p_leaf_resource_import_t* resource_imports;
  // Number of entries in |resource_imports|.
  iree_host_size_t resource_import_count;
  // Contribution-relative byte offsets in source block order.
  const uint32_t* block_byte_offsets;
  // Number of records in |block_byte_offsets|.
  iree_host_size_t block_count;
  // Physical bundles in increasing issue-cycle order.
  const loom_aie2p_planned_bundle_t* bundles;
  // Number of explicitly stored bundles; implicit timing gaps have no rows.
  iree_host_size_t bundle_count;
  // Total physical issue cycles, including NOPs represented by timing gaps.
  uint32_t issue_cycle_count;
  // Encoded physical slots grouped by |bundles|.
  const loom_aie2p_planned_slot_t* slots;
  // Number of encoded physical slots.
  iree_host_size_t slot_count;
  // Branch sites whose absolute targets require final contribution placement.
  const loom_aie2p_planned_branch_fixup_t* branch_fixups;
  // Number of records in |branch_fixups|.
  iree_host_size_t branch_fixup_count;
  // Storage-address sites whose domains require final contribution placement.
  const loom_aie2p_planned_storage_fixup_t* storage_fixups;
  // Number of records in |storage_fixups|.
  iree_host_size_t storage_fixup_count;
  // Exact byte length after variable-width bundle packing.
  iree_host_size_t encoded_byte_length;
  // Union of atomic register units written by committed native instructions,
  // including implicit and allocation-generated writes. One bit per unit in
  // the AIE2P descriptor set.
  loom_aie2p_register_unit_set_t register_writes;
} loom_aie2p_leaf_program_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_PROGRAM_H_
