// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU collective payload, lane exchange, and cross-wave publication plans.
// Selection retains descriptor and shape decisions for emission.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_PLAN_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/synchronization_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_subgroup_payload_kind_e {
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE = 0,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR = 1,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR = 2,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR = 3,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR = 4,
} loom_amdgpu_subgroup_payload_kind_t;

typedef enum loom_amdgpu_subgroup_broadcast_strategy_e {
  // Read the named lane directly through the target's native subgroup crossbar.
  LOOM_AMDGPU_SUBGROUP_BROADCAST_STRATEGY_BPERMUTE = 0,
  // Read one subgroup-uniform lane into an SGPR and publish it to every lane.
  LOOM_AMDGPU_SUBGROUP_BROADCAST_STRATEGY_SCALAR_READLANE = 1,
} loom_amdgpu_subgroup_broadcast_strategy_t;

typedef struct loom_amdgpu_subgroup_broadcast_plan_t {
  // Descriptor row selected for the strategy's native lane read.
  loom_low_lower_resolved_descriptor_t exchange_descriptor;
  // Descriptor row publishing an SGPR read result back to VGPRs.
  loom_low_lower_resolved_descriptor_t scalar_copy_descriptor;
  // Descriptor row extracting a proven-uniform lane index from a VGPR.
  loom_low_lower_resolved_descriptor_t scalar_lane_descriptor;
  // Source value broadcast from source_lane.
  loom_value_id_t value;
  // Result value receiving the broadcast payload.
  loom_value_id_t result;
  // Subgroup lane SSA value read by the broadcast.
  loom_value_id_t source_lane;
  // Exact source lane when known during planning, or UINT32_MAX when dynamic.
  uint32_t exact_source_lane;
  // Number of 32-bit registers in the broadcast payload.
  uint32_t register_count;
  // Native exchange and publication strategy selected during planning.
  loom_amdgpu_subgroup_broadcast_strategy_t strategy;
  // Whether the mapped result requires copying the scalar read into VGPRs.
  bool result_in_vgpr;
} loom_amdgpu_subgroup_broadcast_plan_t;

typedef struct loom_amdgpu_subgroup_broadcast_first_plan_t {
  // Source value broadcast from the first active subgroup lane.
  loom_value_id_t value;
  // Descriptor row selected to read the first active lane into an SGPR.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Result value receiving the broadcast payload.
  loom_value_id_t result;
  // Number of 32-bit registers in the broadcast payload.
  uint32_t register_count;
  // Whether the mapped result requires copying the scalar read into VGPRs.
  bool result_in_vgpr;
} loom_amdgpu_subgroup_broadcast_first_plan_t;

typedef enum loom_amdgpu_crosslane_kind_e {
  // Use DS bpermute with a byte-addressed source lane.
  LOOM_AMDGPU_CROSSLANE_BPERMUTE = 0,
  // Use DPP row-lane moves with an immediate control value.
  LOOM_AMDGPU_CROSSLANE_DPP = 1,
  // Use DS swizzle with a bitmask permutation immediate.
  LOOM_AMDGPU_CROSSLANE_SWIZZLE = 2,
} loom_amdgpu_crosslane_kind_t;

typedef struct loom_amdgpu_direct_xor_lane_recipe_t {
  // Descriptor row implementing the lane exchange.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Optional descriptor fusing the exchange into a conditional false operand.
  loom_amdgpu_descriptor_ref_t conditional_ref;
  // Optional tied descriptor updating selected DPP destination banks.
  loom_amdgpu_descriptor_ref_t masked_move_ref;
  // DPP control or DS swizzle offset interpreted by |crosslane_kind|.
  uint32_t immediate;
  // Native packet family selected for the lane exchange.
  loom_amdgpu_crosslane_kind_t crosslane_kind;
} loom_amdgpu_direct_xor_lane_recipe_t;

typedef struct loom_amdgpu_subgroup_shuffle_plan_t {
  // Source value moved across subgroup lanes.
  loom_value_id_t value;
  // Dynamic lane offset interpreted by |mode| when |exact_offset| is absent.
  loom_value_id_t source_offset;
  // Descriptor row selected for the native cross-lane read.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Result value receiving the moved payload.
  loom_value_id_t result;
  // Used per-lane participation result, or INVALID when the result is unused.
  loom_value_id_t valid;
  // Number of 32-bit registers in the shuffled payload.
  uint32_t register_count;
  // Cross-lane packet family selected for the shuffle.
  loom_amdgpu_crosslane_kind_t crosslane_kind;
  // Full-width lane addressing mode selected by the source op.
  loom_kernel_subgroup_shuffle_mode_t mode;
  // DPP control or DS swizzle offset used by direct cross-lane packets.
  uint32_t crosslane_immediate;
  // Exact lane offset or lane index, or UINT32_MAX when dynamic.
  uint32_t exact_offset;
  // Exact shuffle segment width from the source op.
  uint32_t width;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
} loom_amdgpu_subgroup_shuffle_plan_t;

typedef enum loom_amdgpu_subgroup_reduce_crosslane_kind_e {
  // Use DS bpermute for every subgroup tree exchange.
  LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_BPERMUTE = 0,
  // Use DPP within 16-lane rows and DS bpermute between rows.
  LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_BPERMUTE = 1,
  // Use DPP within 16-lane rows and permlanex16 between row pairs.
  LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_PERMLANEX16 = 2,
} loom_amdgpu_subgroup_reduce_crosslane_kind_t;

typedef enum loom_amdgpu_subgroup_reduce_publication_kind_e {
  // Publish a correct reduced VGPR payload to every active subgroup lane.
  LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_ALL_LANES = 0,
  // Combine 32-lane halves through SGPRs and broadcast the completed payload.
  LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_SCALAR_BROADCAST = 1,
} loom_amdgpu_subgroup_reduce_publication_kind_t;

typedef struct loom_amdgpu_subgroup_reduce_plan_t {
  // Source value reduced across subgroup lanes.
  loom_value_id_t value;
  // Optional literal recipe for the scalar integer payload's VGPR carrier.
  const void* payload_materialization;
  // Descriptor row selected for each native cross-lane read.
  loom_low_lower_resolved_descriptor_t bpermute_descriptor;
  // Descriptor row selected for all-lane DPP row moves.
  loom_low_lower_resolved_descriptor_t dpp_descriptor;
  // Descriptor row selected for fused DPP row moves and lane combines.
  loom_low_lower_resolved_descriptor_t dpp_combine_descriptor;
  // Descriptor row selected for paired 16-lane row exchanges.
  loom_low_lower_resolved_descriptor_t permlanex16_descriptor;
  // Descriptor row selected for reading a fixed VGPR lane into an SGPR.
  loom_low_lower_resolved_descriptor_t readlane_descriptor;
  // Descriptor row selected for each native lane combine.
  loom_low_lower_resolved_descriptor_t combine_descriptor;
  // Descriptor row selected to guard inactive source lanes.
  loom_low_lower_resolved_descriptor_t guard_descriptor;
  // Descriptor row selected to retain the accumulator for inactive peers.
  loom_low_lower_resolved_descriptor_t select_descriptor;
  // Result value receiving the reduced payload.
  loom_value_id_t result;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the reduced payload.
  uint32_t register_count;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
  // Number of low-numbered lanes participating in the emitted reduce tree.
  uint32_t active_lane_count;
  // Cross-lane exchange strategy selected for the subgroup tree.
  loom_amdgpu_subgroup_reduce_crosslane_kind_t crosslane_kind;
  // Strategy used to publish the final reduced value to result users.
  loom_amdgpu_subgroup_reduce_publication_kind_t publication_kind;
} loom_amdgpu_subgroup_reduce_plan_t;

typedef enum loom_amdgpu_workgroup_reduce_publication_kind_e {
  // One wave reduces LDS-published per-wave partials and publishes the final
  // value back through LDS for all workitems to reload.
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LDS = 0,
  // Every wave reloads the per-wave partials, redundantly reduces them within
  // the wave, and broadcasts the wave-local lane-zero result to its lanes.
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP = 1,
  // One wave reduces LDS-published per-wave partials and leaves the result
  // valid only for leader-guarded uses.
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM = 2,
  // Every wave redundantly reduces LDS-published per-wave partials and leaves
  // the result valid only in subgroup leader lanes.
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE = 3,
} loom_amdgpu_workgroup_reduce_publication_kind_t;

typedef struct loom_amdgpu_workgroup_collective_cross_wave_descriptors_t {
  // Descriptor row selected for LDS reads between waves.
  loom_low_lower_resolved_descriptor_t lds_read_descriptor;
  // Descriptor row selected for LDS writes between waves.
  loom_low_lower_resolved_descriptor_t lds_write_descriptor;
  // Target-selected packet plan used to synchronize LDS publication.
  loom_amdgpu_kernel_barrier_plan_t barrier;
  // Descriptor row selected to restrict publication to producer lanes.
  loom_low_lower_resolved_descriptor_t saveexec_descriptor;
  // Descriptor row selected to restore EXEC after lane-restricted regions.
  loom_low_lower_resolved_descriptor_t restore_exec_descriptor;
} loom_amdgpu_workgroup_collective_cross_wave_descriptors_t;

typedef struct loom_amdgpu_workgroup_reduce_plan_t {
  // Source value reduced across workgroup lanes.
  loom_value_id_t value;
  // Optional literal recipe for the scalar integer payload's VGPR carrier.
  const void* payload_materialization;
  // Descriptor row selected for each native cross-lane read.
  loom_low_lower_resolved_descriptor_t bpermute_descriptor;
  // Descriptor row selected for all-lane DPP row moves.
  loom_low_lower_resolved_descriptor_t dpp_descriptor;
  // Descriptor row selected for fused DPP row moves and lane combines.
  loom_low_lower_resolved_descriptor_t dpp_combine_descriptor;
  // Descriptor row selected for paired 16-lane row exchanges.
  loom_low_lower_resolved_descriptor_t permlanex16_descriptor;
  // Descriptor row selected for each native lane combine.
  loom_low_lower_resolved_descriptor_t combine_descriptor;
  // Descriptor row selected to guard inactive source lanes.
  loom_low_lower_resolved_descriptor_t guard_descriptor;
  // Descriptor row selected to identify the partial tail wave.
  loom_low_lower_resolved_descriptor_t lane_ge_descriptor;
  // Descriptor row selected to retain the accumulator for inactive peers.
  loom_low_lower_resolved_descriptor_t select_descriptor;
  // Descriptor bundle used for cross-wave LDS publication.
  loom_amdgpu_workgroup_collective_cross_wave_descriptors_t cross_wave;
  // Result value receiving the reduced payload.
  loom_value_id_t result;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the reduced payload.
  uint32_t register_count;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
  // Execution wavefront width used to partition cross-wave workgroup staging.
  uint32_t partition_wavefront_size;
  // Exact flattened workgroup size selected by launch configuration.
  uint32_t flat_workgroup_size;
  // Exact LDS scratch extent selected for cross-wave staging.
  uint32_t scratch_byte_length;
  // Cross-lane strategies selected for the two reduction stages.
  struct {
    // Exchange strategy for the partial reduction within each wave.
    loom_amdgpu_subgroup_reduce_crosslane_kind_t per_wave;
    // Exchange strategy for combining the LDS-published wave partials.
    loom_amdgpu_subgroup_reduce_crosslane_kind_t cross_wave;
  } crosslane;
  // Strategy used to publish the final reduced value to all workitems.
  loom_amdgpu_workgroup_reduce_publication_kind_t publication_kind;
} loom_amdgpu_workgroup_reduce_plan_t;

typedef struct loom_amdgpu_subgroup_scan_plan_t {
  // Source value scanned across subgroup lanes.
  loom_value_id_t value;
  // Optional literal recipe for the scalar integer payload's VGPR carrier.
  const void* payload_materialization;
  // Descriptor row selected for each native cross-lane read.
  loom_low_lower_resolved_descriptor_t bpermute_descriptor;
  // Optional descriptor selected for shifts within 16-lane DPP rows.
  loom_low_lower_resolved_descriptor_t dpp_descriptor;
  // Optional descriptor fusing a DPP shift and the lane combine.
  loom_low_lower_resolved_descriptor_t dpp_combine_descriptor;
  // Descriptor row selected for each native lane combine.
  loom_low_lower_resolved_descriptor_t combine_descriptor;
  // Descriptor row selected to guard each prefix step.
  loom_low_lower_resolved_descriptor_t guard_descriptor;
  // Descriptor row selected to merge guarded prefix-step results.
  loom_low_lower_resolved_descriptor_t select_descriptor;
  // Result value receiving the scanned payload.
  loom_value_id_t result;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the scanned payload.
  uint32_t register_count;
  // Inclusive or exclusive scan mode selected by the source op.
  loom_kernel_subgroup_scan_mode_t mode;
  // Lane order selected by the source op.
  loom_kernel_subgroup_scan_direction_t direction;
  // 32-bit value produced for an empty exclusive prefix.
  uint32_t empty_prefix_bits;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
  // Number of low-numbered lanes participating in the emitted scan tree.
  uint32_t active_lane_count;
} loom_amdgpu_subgroup_scan_plan_t;

typedef struct loom_amdgpu_workgroup_scan_plan_t {
  // Source value scanned across workgroup lanes.
  loom_value_id_t value;
  // Optional literal recipe for the scalar integer payload's VGPR carrier.
  const void* payload_materialization;
  // Descriptor row selected for each native cross-lane read.
  loom_low_lower_resolved_descriptor_t bpermute_descriptor;
  // Optional descriptor selected for shifts within 16-lane DPP rows.
  loom_low_lower_resolved_descriptor_t dpp_descriptor;
  // Optional descriptor fusing a DPP shift and the lane combine.
  loom_low_lower_resolved_descriptor_t dpp_combine_descriptor;
  // Descriptor row selected for each native lane combine.
  loom_low_lower_resolved_descriptor_t combine_descriptor;
  // Descriptor row selected to guard each prefix step.
  loom_low_lower_resolved_descriptor_t guard_descriptor;
  // Descriptor row selected to merge guarded prefix-step results.
  loom_low_lower_resolved_descriptor_t select_descriptor;
  // Descriptor row selected for first-wave predicates.
  loom_low_lower_resolved_descriptor_t lane_lt_descriptor;
  // Descriptor row selected for tail-wave predicates.
  loom_low_lower_resolved_descriptor_t lane_ge_descriptor;
  // Descriptor bundle used for cross-wave LDS publication.
  loom_amdgpu_workgroup_collective_cross_wave_descriptors_t cross_wave;
  // Result value receiving the scanned payload.
  loom_value_id_t result;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the scanned payload.
  uint32_t register_count;
  // Inclusive or exclusive scan mode selected by the source op.
  loom_kernel_subgroup_scan_mode_t mode;
  // Lane order selected by the source op.
  loom_kernel_subgroup_scan_direction_t direction;
  // 32-bit value produced for an empty exclusive or cross-partition prefix.
  uint32_t empty_prefix_bits;
  // Whether the empty-prefix value can contribute to a nonempty scan unchanged.
  bool empty_prefix_is_identity;
  // Native cross-lane width used to partition workgroup scan staging.
  uint32_t partition_wavefront_size;
  // Exact flattened workgroup size selected by launch configuration.
  uint32_t flat_workgroup_size;
  // Exact LDS scratch extent selected for cross-wave staging.
  uint32_t scratch_byte_length;
} loom_amdgpu_workgroup_scan_plan_t;

typedef struct loom_amdgpu_subgroup_active_mask_plan_t {
  // Descriptor row selected to read the native EXEC lane mask.
  loom_low_lower_resolved_descriptor_t exec_read_descriptor;
  // Source mask result receiving the active-lane payload.
  loom_value_id_t mask;
  // Static bit width of the source integer mask result.
  uint32_t mask_bit_count;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
} loom_amdgpu_subgroup_active_mask_plan_t;

typedef struct loom_amdgpu_subgroup_predicate_mask_descriptors_t {
  // Descriptor row selected to read the active EXEC lane mask.
  loom_low_lower_resolved_descriptor_t exec_read;
  // Descriptor row selected to intersect predicate and active masks.
  loom_low_lower_resolved_descriptor_t intersect;
} loom_amdgpu_subgroup_predicate_mask_descriptors_t;

typedef struct loom_amdgpu_subgroup_ballot_plan_t {
  // Source predicate already materialized as a native EXEC-width mask.
  loom_value_id_t predicate;
  // Descriptors that restrict the predicate to lanes active at this use.
  loom_amdgpu_subgroup_predicate_mask_descriptors_t active;
  // Source mask result receiving predicate bits for active lanes.
  loom_value_id_t mask;
  // Static bit width of the source integer mask result.
  uint32_t mask_bit_count;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
} loom_amdgpu_subgroup_ballot_plan_t;

typedef struct loom_amdgpu_subgroup_vote_any_plan_t {
  // Source predicate already materialized as a native EXEC-width mask.
  loom_value_id_t predicate;
  // Descriptors that restrict the predicate to lanes active at this use.
  loom_amdgpu_subgroup_predicate_mask_descriptors_t active;
  // Descriptor row selected to compare the predicate mask against zero.
  loom_low_lower_resolved_descriptor_t compare_descriptor;
  // Descriptor row selected to materialize each half of the zero mask.
  loom_low_lower_resolved_descriptor_t zero_descriptor;
  // Subgroup-uniform i1 source result receiving SCC.
  loom_value_id_t result;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
} loom_amdgpu_subgroup_vote_any_plan_t;

typedef struct loom_amdgpu_subgroup_vote_all_plan_t {
  // Source predicate already materialized as a native EXEC-width mask.
  loom_value_id_t predicate;
  // Descriptor row selected to compare predicate and active EXEC masks.
  loom_low_lower_resolved_descriptor_t compare_descriptor;
  // Descriptors that restrict the predicate to lanes active at this use.
  loom_amdgpu_subgroup_predicate_mask_descriptors_t active;
  // Subgroup-uniform i1 source result receiving SCC.
  loom_value_id_t result;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
} loom_amdgpu_subgroup_vote_all_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_PLAN_H_
