// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_KERNEL_OPS_H_
#define LOOM_OPS_KERNEL_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/ir/facts.h"
#include "loom/ops/atomic.h"
#include "loom/ops/cache.h"
#include "loom/ops/combining.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_KERNEL_DEF = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 0),
  LOOM_OP_KERNEL_LAUNCH_CONFIG = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 1),
  LOOM_OP_KERNEL_RETURN = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 2),
  LOOM_OP_KERNEL_EXIT = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 3),
  LOOM_OP_KERNEL_BARRIER = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 4),
  LOOM_OP_KERNEL_ASYNC_COPY = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 5),
  LOOM_OP_KERNEL_ASYNC_COPY_MASK = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 6),
  LOOM_OP_KERNEL_ASYNC_GATHER = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 7),
  LOOM_OP_KERNEL_ASYNC_GATHER_MASK = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 8),
  LOOM_OP_KERNEL_ASYNC_GROUP = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 9),
  LOOM_OP_KERNEL_ASYNC_WAIT = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 10),
  LOOM_OP_KERNEL_TENSOR_LDS_DESCRIPTOR = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 11),
  LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 12),
  LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 13),
  LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 14),
  LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 15),
  LOOM_OP_KERNEL_WORKITEM_ID = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 16),
  LOOM_OP_KERNEL_WORKGROUP_ID = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 17),
  LOOM_OP_KERNEL_WORKGROUP_SIZE = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 18),
  LOOM_OP_KERNEL_WORKGROUP_COUNT = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 19),
  LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 20),
  LOOM_OP_KERNEL_SUBGROUP_ID = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 21),
  LOOM_OP_KERNEL_SUBGROUP_COUNT = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 22),
  LOOM_OP_KERNEL_SUBGROUP_SIZE = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 23),
  LOOM_OP_KERNEL_SUBGROUP_LANE_ID = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 24),
  LOOM_OP_KERNEL_SUBGROUP_SHUFFLE = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 25),
  LOOM_OP_KERNEL_SUBGROUP_BROADCAST = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 26),
  LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 27),
  LOOM_OP_KERNEL_SUBGROUP_REDUCE = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 28),
  LOOM_OP_KERNEL_SUBGROUP_SCAN = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 29),
  LOOM_OP_KERNEL_SUBGROUP_VOTE_ANY = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 30),
  LOOM_OP_KERNEL_SUBGROUP_VOTE_ALL = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 31),
  LOOM_OP_KERNEL_SUBGROUP_VOTE_BALLOT = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 32),
  LOOM_OP_KERNEL_SUBGROUP_ACTIVE_MASK = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 33),
  LOOM_OP_KERNEL_SUBGROUP_MATCH_ANY = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 34),
  LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 35),
  LOOM_OP_KERNEL_WORKGROUP_REDUCE = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 36),
  LOOM_OP_KERNEL_WORKGROUP_SCAN = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 37),
  LOOM_OP_KERNEL_WORKGROUP_VOTE_ANY = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 38),
  LOOM_OP_KERNEL_WORKGROUP_VOTE_ALL = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 39),
  LOOM_OP_KERNEL_WORKGROUP_VOTE_COUNT = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 40),
  LOOM_OP_KERNEL_ASSERT = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 41),
  LOOM_OP_KERNEL_COUNT_ = 42,
};

// Required async copy direction.
typedef enum loom_kernel_direction_e {
  LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP = 0,
  LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL = 1,
  LOOM_KERNEL_DIRECTION_COUNT_ = 2,
} loom_kernel_direction_t;

// Three-dimensional kernel coordinate axis.
typedef enum loom_kernel_dimension_e {
  LOOM_KERNEL_DIMENSION_X = 0,
  LOOM_KERNEL_DIMENSION_Y = 1,
  LOOM_KERNEL_DIMENSION_Z = 2,
  LOOM_KERNEL_DIMENSION_COUNT_ = 3,
} loom_kernel_dimension_t;

// Subgroup lane shuffle addressing mode.
typedef enum loom_kernel_subgroup_shuffle_mode_e {
  LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_XOR = 0,
  LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_UP = 1,
  LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_DOWN = 2,
  LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_INDEX = 3,
  LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_COUNT_ = 4,
} loom_kernel_subgroup_shuffle_mode_t;

// Subgroup scan inclusivity.
typedef enum loom_kernel_subgroup_scan_mode_e {
  LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE = 0,
  LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE = 1,
  LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_ = 2,
} loom_kernel_subgroup_scan_mode_t;

// Subgroup scan lane order.
typedef enum loom_kernel_subgroup_scan_direction_e {
  LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD = 0,
  LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE = 1,
  LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_ = 2,
} loom_kernel_subgroup_scan_direction_t;

// Workgroup scan inclusivity.
typedef enum loom_kernel_workgroup_scan_mode_e {
  LOOM_KERNEL_WORKGROUP_SCAN_MODE_INCLUSIVE = 0,
  LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE = 1,
  LOOM_KERNEL_WORKGROUP_SCAN_MODE_COUNT_ = 2,
} loom_kernel_workgroup_scan_mode_t;

// Workgroup scan workitem order.
typedef enum loom_kernel_workgroup_scan_direction_e {
  LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_FORWARD = 0,
  LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_REVERSE = 1,
  LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_COUNT_ = 2,
} loom_kernel_workgroup_scan_direction_t;

// LOOM_OP_KERNEL_DEF: Dispatchable source-level kernel entry. Kernel entries own launch and export contracts; ordinary func.def bodies remain helper/callable code.
// kernel.def @entry() {
//   %one = index.constant 1 : index
//   kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
// } launch(%buffer: buffer) {
//   kernel.return
// }
LOOM_DEFINE_ISA(loom_kernel_def_isa, LOOM_OP_KERNEL_DEF)
LOOM_DEFINE_ATTR_SYMBOL(loom_kernel_def_callee, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_kernel_def_target, 1)
LOOM_DEFINE_ATTR_STRING(loom_kernel_def_export_symbol, 2)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_def_export_linkage, 3, loom_target_linkage_t)
LOOM_DEFINE_REGION(loom_kernel_def_config, 0)
LOOM_DEFINE_REGION(loom_kernel_def_body, 1)
enum loom_kernel_def_build_flag_bits_e {
  LOOM_KERNEL_DEF_BUILD_FLAG_HAS_TARGET = 1u << 0,
  LOOM_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 1,
  LOOM_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_LINKAGE = 1u << 2,
};
typedef uint32_t loom_kernel_def_build_flags_t;
iree_status_t loom_kernel_def_build(
    loom_builder_t* builder,
    loom_kernel_def_build_flags_t build_flags,
    loom_optional loom_symbol_ref_t target,
    loom_optional loom_string_id_t export_symbol,
    loom_optional uint8_t export_linkage,
    loom_symbol_ref_t callee,
    const loom_type_t* config_arg_types,
    iree_host_size_t config_arg_types_count,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_LAUNCH_CONFIG: Terminate a kernel launch configuration region with the computed workgroup grid and required workgroup size.
// kernel.launch.config workgroups(%gx, %gy, %gz) workgroup_size(%sx, %sy, %sz) : index
LOOM_DEFINE_ISA(loom_kernel_launch_config_isa, LOOM_OP_KERNEL_LAUNCH_CONFIG)
LOOM_DEFINE_OPERAND(loom_kernel_launch_config_workgroup_count_x, 0)
LOOM_DEFINE_OPERAND(loom_kernel_launch_config_workgroup_count_y, 1)
LOOM_DEFINE_OPERAND(loom_kernel_launch_config_workgroup_count_z, 2)
LOOM_DEFINE_OPERAND(loom_kernel_launch_config_workgroup_size_x, 3)
LOOM_DEFINE_OPERAND(loom_kernel_launch_config_workgroup_size_y, 4)
LOOM_DEFINE_OPERAND(loom_kernel_launch_config_workgroup_size_z, 5)
iree_status_t loom_kernel_launch_config_build(
    loom_builder_t* builder,
    loom_value_id_t workgroup_count_x,
    loom_value_id_t workgroup_count_y,
    loom_value_id_t workgroup_count_z,
    loom_value_id_t workgroup_size_x,
    loom_value_id_t workgroup_size_y,
    loom_value_id_t workgroup_size_z,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_KERNEL_RETURN: Return from a dispatchable kernel entry.
// kernel.return
LOOM_DEFINE_ISA(loom_kernel_return_isa, LOOM_OP_KERNEL_RETURN)
iree_status_t loom_kernel_return_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_KERNEL_EXIT: Conditionally leaves the current kernel before executing the following top-level kernel-body operations.
// kernel.exit %done : i1
LOOM_DEFINE_ISA(loom_kernel_exit_isa, LOOM_OP_KERNEL_EXIT)
LOOM_DEFINE_OPERAND(loom_kernel_exit_condition, 0)
LOOM_DEFINE_OPTIONAL_REGION(loom_kernel_exit_body, 0)
enum loom_kernel_exit_build_flag_bits_e {
  LOOM_KERNEL_EXIT_BUILD_FLAG_HAS_BODY = 1u << 0,
};
typedef uint32_t loom_kernel_exit_build_flags_t;
iree_status_t loom_kernel_exit_build(
    loom_builder_t* builder,
    loom_kernel_exit_build_flags_t build_flags,
    loom_value_id_t condition,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_exit_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);

// LOOM_OP_KERNEL_BARRIER: Synchronize invocations in an explicit execution scope and fence a named memory space with a required ordering. Supported source-level kernel barriers synchronize either the current subgroup or workgroup while fencing workgroup memory with acquire-release ordering. Async-copy completion is modeled by kernel.async.wait; use kernel.barrier only when invocations must rendezvous before consuming shared memory.
// kernel.barrier<workgroup> {ordering = acq_rel, scope = subgroup}
LOOM_DEFINE_ISA(loom_kernel_barrier_isa, LOOM_OP_KERNEL_BARRIER)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_barrier_memory_space, 0, loom_value_fact_memory_space_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_barrier_ordering, 1, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_barrier_scope, 2, loom_atomic_scope_t)
iree_status_t loom_kernel_barrier_build(
    loom_builder_t* builder,
    loom_value_fact_memory_space_t memory_space,
    loom_atomic_ordering_t ordering,
    loom_atomic_scope_t scope,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_barrier_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_COPY: Initiate an asynchronous byte-for-byte transfer between two already originated views. The source and destination view types may use different logical element types or shapes, but they must describe the same static byte footprint. The direction attribute makes the required memory-space flow explicit. The returned token must be committed to exactly one kernel.async.group before the copied bytes are waited or consumed.
// %copy = kernel.async.copy %src to %dst {cache_scope = cu, cache_temporal = regular, direction = global_to_workgroup} : view<16xi8> to view<16xi8> -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_copy_isa, LOOM_OP_KERNEL_ASYNC_COPY)
LOOM_DEFINE_OPERAND(loom_kernel_async_copy_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_copy_dest, 1)
LOOM_DEFINE_RESULT(loom_kernel_async_copy_token, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_copy_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_copy_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_copy_direction, 2, loom_kernel_direction_t)
iree_status_t loom_kernel_async_copy_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_cache_scope_t cache_scope,
    loom_cache_temporal_t cache_temporal,
    loom_kernel_direction_t direction,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_copy_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_COPY_MASK: Predicated form of kernel.async.copy. When predicate is true, the op initiates the same transfer as kernel.async.copy. When predicate is false, the op performs no memory access and produces an already complete token so grouping and waiting remain structurally uniform.
// %copy = kernel.async.copy.mask %src to %dst, %in_bounds {cache_scope = cu, cache_temporal = non_temporal, direction = global_to_workgroup} : view<16xi8> to view<16xi8>, i1 -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_copy_mask_isa, LOOM_OP_KERNEL_ASYNC_COPY_MASK)
LOOM_DEFINE_OPERAND(loom_kernel_async_copy_mask_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_copy_mask_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_copy_mask_predicate, 2)
LOOM_DEFINE_RESULT(loom_kernel_async_copy_mask_token, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_copy_mask_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_copy_mask_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_copy_mask_direction, 2, loom_kernel_direction_t)
iree_status_t loom_kernel_async_copy_mask_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t predicate,
    loom_cache_scope_t cache_scope,
    loom_cache_temporal_t cache_temporal,
    loom_kernel_direction_t direction,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_copy_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_GATHER: Initiate a subgroup-collective asynchronous gather from each invocation's source view into a lane-contiguous workgroup destination view. The destination view has one leading subgroup-lane axis and a trailing lane slot with enough static bytes to hold one source payload. If the lane slot is larger than the source footprint, the extra destination bytes are padding bytes with unspecified contents. The destination denotes the subgroup-uniform base tile; the current subgroup lane is applied by the op semantics and must not be pre-applied by forming a lane subview. This directly represents AMDGPU global_load_lds-style staging, including padded narrow loads, without requiring a later pass to rediscover that a set of per-lane copies was really one subgroup LDS DMA operation.
// %copy = kernel.async.gather %src_lane to %lds_tile {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<[%wave]x4xi8> -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_gather_isa, LOOM_OP_KERNEL_ASYNC_GATHER)
LOOM_DEFINE_OPERAND(loom_kernel_async_gather_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_gather_dest, 1)
LOOM_DEFINE_RESULT(loom_kernel_async_gather_token, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_gather_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_gather_cache_temporal, 1, loom_cache_temporal_t)
iree_status_t loom_kernel_async_gather_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_cache_scope_t cache_scope,
    loom_cache_temporal_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_gather_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_GATHER_MASK: Predicated form of kernel.async.gather. False predicates perform no source or destination access for the current invocation but still produce a completed token, preserving a uniform async group shape for tails and guarded tiles.
// %copy = kernel.async.gather.mask %src_lane to %lds_tile, %in_bounds {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<[%wave]x4xi8>, i1 -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_gather_mask_isa, LOOM_OP_KERNEL_ASYNC_GATHER_MASK)
LOOM_DEFINE_OPERAND(loom_kernel_async_gather_mask_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_gather_mask_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_gather_mask_predicate, 2)
LOOM_DEFINE_RESULT(loom_kernel_async_gather_mask_token, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_gather_mask_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_gather_mask_cache_temporal, 1, loom_cache_temporal_t)
iree_status_t loom_kernel_async_gather_mask_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t predicate,
    loom_cache_scope_t cache_scope,
    loom_cache_temporal_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_GROUP: Commit zero or more async copy/gather/cluster/tensor tokens into the ordered async stream. Empty groups are valid pipeline markers. The resulting group completes after all committed transfers complete. Groups are ordered by program order; waiting a group also completes older groups in the same stream.
// %empty = kernel.async.group -> kernel.async.group
LOOM_DEFINE_ISA(loom_kernel_async_group_isa, LOOM_OP_KERNEL_ASYNC_GROUP)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_kernel_async_group_tokens, 0)
LOOM_DEFINE_RESULT(loom_kernel_async_group_group, 0)
iree_status_t loom_kernel_async_group_build(
    loom_builder_t* builder,
    loom_may_consume const loom_value_id_t* tokens,
    iree_host_size_t tokens_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_group_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_WAIT: Wait until a committed async-copy group has completed. This completes the named group and all older groups in the same ordered async stream. The required newer_groups value states how many younger groups are allowed to remain outstanding after the wait, matching AMDGPU wait.asyncmark and NVVM wait_group count semantics without making lowering rediscover the software-pipeline distance from scratch. It is not a workgroup barrier; use kernel.barrier separately when other invocations must observe the copied destination data.
// kernel.async.wait %group {newer_groups = 0} : kernel.async.group
LOOM_DEFINE_ISA(loom_kernel_async_wait_isa, LOOM_OP_KERNEL_ASYNC_WAIT)
LOOM_DEFINE_OPERAND(loom_kernel_async_wait_group, 0)
LOOM_DEFINE_ATTR_I64(loom_kernel_async_wait_newer_groups, 0)
iree_status_t loom_kernel_async_wait_build(
    loom_builder_t* builder,
    loom_value_id_t group,
    int64_t newer_groups,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_wait_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_TENSOR_LDS_DESCRIPTOR: Bundle AMDGPU tensor-memory descriptor groups into one typed SSA value. The dgroups are the exact operands lowered to llvm.amdgcn.tensor.load.to.lds or llvm.amdgcn.tensor.store.from.lds: D0 is vector<4xi32>, D1 is vector<8xi32>, and optional D2/D3 are vector<4xi32>. Gfx1250 uses two-group and four-group descriptor forms; the LLVM intrinsic's fifth D4 operand is lowered as zero because gfx1250 ignores it. The op is pure and contains no memory endpoints; endpoint views are operands of the async tensor ops so fact propagation and alias analysis do not need to decode hardware bitfields.
// %desc = kernel.tensor.lds.descriptor dgroups(%d0, %d1) : vector<4xi32>, vector<8xi32> -> kernel.tensor.lds.descriptor
LOOM_DEFINE_ISA(loom_kernel_tensor_lds_descriptor_isa, LOOM_OP_KERNEL_TENSOR_LDS_DESCRIPTOR)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_kernel_tensor_lds_descriptor_dgroups, 0)
LOOM_DEFINE_RESULT(loom_kernel_tensor_lds_descriptor_descriptor, 0)
iree_status_t loom_kernel_tensor_lds_descriptor_build(
    loom_builder_t* builder,
    loom_may_consume const loom_value_id_t* dgroups,
    iree_host_size_t dgroups_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_tensor_lds_descriptor_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS: Initiate an AMDGPU gfx1250+ tensor-memory load from a global-like source view into a workgroup/LDS destination view using an explicit kernel.tensor.lds.descriptor. The descriptor supplies the exact hardware dgroups, while the source and destination views keep the logical rank, element type, layout, and memory-space facts visible. The endpoints must have the same rank in [1, 5], the same 1/2/4/8 byte element type, and memory spaces global/constant/descriptor to workgroup. The returned token must be committed to exactly one kernel.async.group.
// %copy = kernel.async.tensor.load.to.lds %global_tile to %lds_tile using %desc {cache_scope = cu, cache_temporal = regular} : view<64x64xf32> to view<64x64xf32>, kernel.tensor.lds.descriptor -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_tensor_load_to_lds_isa, LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_load_to_lds_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_load_to_lds_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_load_to_lds_descriptor, 2)
LOOM_DEFINE_RESULT(loom_kernel_async_tensor_load_to_lds_token, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_tensor_load_to_lds_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_tensor_load_to_lds_cache_temporal, 1, loom_cache_temporal_t)
iree_status_t loom_kernel_async_tensor_load_to_lds_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t descriptor,
    loom_cache_scope_t cache_scope,
    loom_cache_temporal_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_tensor_load_to_lds_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS: Initiate an AMDGPU gfx1250+ tensor-memory store from a workgroup/LDS source view into a global-like destination view using an explicit kernel.tensor.lds.descriptor. The descriptor supplies the exact hardware dgroups, while the source and destination views keep the logical rank, element type, layout, and memory-space facts visible. The endpoints must have the same rank in [1, 5], the same 1/2/4/8 byte element type, and memory spaces workgroup to global/descriptor. The returned token must be committed to exactly one kernel.async.group.
// %copy = kernel.async.tensor.store.from.lds %lds_tile to %global_tile using %desc {cache_scope = device, cache_temporal = non_temporal_writeback} : view<64x64xf32> to view<64x64xf32>, kernel.tensor.lds.descriptor -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_tensor_store_from_lds_isa, LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_store_from_lds_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_store_from_lds_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_store_from_lds_descriptor, 2)
LOOM_DEFINE_RESULT(loom_kernel_async_tensor_store_from_lds_token, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_tensor_store_from_lds_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_tensor_store_from_lds_cache_temporal, 1, loom_cache_temporal_t)
iree_status_t loom_kernel_async_tensor_store_from_lds_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t descriptor,
    loom_cache_scope_t cache_scope,
    loom_cache_temporal_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_tensor_store_from_lds_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER: Initiate an AMDGPU gfx1250+ cluster asynchronous load from a global-like source view into a workgroup/LDS destination view. The required i32 cluster_mask is the hardware workgroup broadcast mask loaded through M0. Source and destination must have the same static byte footprint, and that footprint must be exactly 1, 4, 8, or 16 bytes; target lowering maps those widths to llvm.amdgcn.cluster.load.async.to.lds.b8/b32/b64/b128. The LLVM offset and cache-policy immediate operands are lowering choices derived from the view address and cache attributes, not separate Loom semantics. The returned token must be committed to exactly one kernel.async.group.
// %copy = kernel.async.cluster.gather %src to %lds using %mask {cache_scope = se, cache_temporal = high_temporal} : view<16xi8> to view<16xi8>, i32 -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_cluster_gather_isa, LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_cluster_mask, 2)
LOOM_DEFINE_RESULT(loom_kernel_async_cluster_gather_token, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_cluster_gather_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_cluster_gather_cache_temporal, 1, loom_cache_temporal_t)
iree_status_t loom_kernel_async_cluster_gather_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t cluster_mask,
    loom_cache_scope_t cache_scope,
    loom_cache_temporal_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_cluster_gather_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK: Predicated form of kernel.async.cluster.gather. False predicates perform no source or destination access for the current invocation but still produce a completed token, preserving a uniform async group shape for tails and guarded tiles. The cluster_mask remains the target workgroup broadcast mask and is distinct from the scalar i1 predicate.
// %copy = kernel.async.cluster.gather.mask %src to %lds using %mask, %in_bounds {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<4xi8>, i32, i1 -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_cluster_gather_mask_isa, LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_mask_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_mask_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_mask_cluster_mask, 2)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_mask_predicate, 3)
LOOM_DEFINE_RESULT(loom_kernel_async_cluster_gather_mask_token, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_cluster_gather_mask_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_async_cluster_gather_mask_cache_temporal, 1, loom_cache_temporal_t)
iree_status_t loom_kernel_async_cluster_gather_mask_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t cluster_mask,
    loom_may_consume loom_value_id_t predicate,
    loom_cache_scope_t cache_scope,
    loom_cache_temporal_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_cluster_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_WORKITEM_ID: Read one coordinate of the current invocation within its workgroup. The result is a logical index value, not a byte offset; target lowering decides whether the coordinate is carried in scalar, vector, or dedicated target registers.
// %tid = kernel.workitem.id<x> : index
LOOM_DEFINE_ISA(loom_kernel_workitem_id_isa, LOOM_OP_KERNEL_WORKITEM_ID)
LOOM_DEFINE_RESULT(loom_kernel_workitem_id_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_workitem_id_dimension, 0, loom_kernel_dimension_t)
iree_status_t loom_kernel_workitem_id_build(
    loom_builder_t* builder,
    loom_kernel_dimension_t dimension,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_workitem_id_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_WORKGROUP_ID: Read one coordinate of the current workgroup within the dispatch grid. The result is a logical index value; target lowering decides whether the coordinate is carried in scalar registers, ABI state, or target-specific builtin values.
// %bid = kernel.workgroup.id<x> : index
LOOM_DEFINE_ISA(loom_kernel_workgroup_id_isa, LOOM_OP_KERNEL_WORKGROUP_ID)
LOOM_DEFINE_RESULT(loom_kernel_workgroup_id_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_workgroup_id_dimension, 0, loom_kernel_dimension_t)
iree_status_t loom_kernel_workgroup_id_build(
    loom_builder_t* builder,
    loom_kernel_dimension_t dimension,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_workgroup_id_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_WORKGROUP_SIZE: Read the selected workgroup size dimension. A launch configuration contract can make this an exact fact; otherwise target facts bound the dynamic launch value.
// %size = kernel.workgroup.size<x> : index
LOOM_DEFINE_ISA(loom_kernel_workgroup_size_isa, LOOM_OP_KERNEL_WORKGROUP_SIZE)
LOOM_DEFINE_RESULT(loom_kernel_workgroup_size_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_workgroup_size_dimension, 0, loom_kernel_dimension_t)
iree_status_t loom_kernel_workgroup_size_build(
    loom_builder_t* builder,
    loom_kernel_dimension_t dimension,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_workgroup_size_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_WORKGROUP_COUNT: Read the dispatched workgroup count in one grid dimension.
// %count = kernel.workgroup.count<x> : index
LOOM_DEFINE_ISA(loom_kernel_workgroup_count_isa, LOOM_OP_KERNEL_WORKGROUP_COUNT)
LOOM_DEFINE_RESULT(loom_kernel_workgroup_count_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_workgroup_count_dimension, 0, loom_kernel_dimension_t)
iree_status_t loom_kernel_workgroup_count_build(
    loom_builder_t* builder,
    loom_kernel_dimension_t dimension,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_workgroup_count_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID: Read one coordinate of the current invocation in the whole dispatch. This is the logical coordinate formed from workgroup id, workgroup size, and workitem id; target lowering may materialize it directly or derive it from lower-level launch registers.
// %gid = kernel.workitem.dispatch.id<x> : index
LOOM_DEFINE_ISA(loom_kernel_workitem_dispatch_id_isa, LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID)
LOOM_DEFINE_RESULT(loom_kernel_workitem_dispatch_id_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_workitem_dispatch_id_dimension, 0, loom_kernel_dimension_t)
iree_status_t loom_kernel_workitem_dispatch_id_build(
    loom_builder_t* builder,
    loom_kernel_dimension_t dimension,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_workitem_dispatch_id_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_SUBGROUP_ID: Read the current subgroup coordinate within the workgroup.
// %sg = kernel.subgroup.id : index
LOOM_DEFINE_ISA(loom_kernel_subgroup_id_isa, LOOM_OP_KERNEL_SUBGROUP_ID)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_id_result, 0)
iree_status_t loom_kernel_subgroup_id_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_id_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_SUBGROUP_COUNT: Read the number of subgroups in the current workgroup.
// %count = kernel.subgroup.count : index
LOOM_DEFINE_ISA(loom_kernel_subgroup_count_isa, LOOM_OP_KERNEL_SUBGROUP_COUNT)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_count_result, 0)
iree_status_t loom_kernel_subgroup_count_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_count_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_SUBGROUP_SIZE: Read the invocation count of the current subgroup.
// %size = kernel.subgroup.size : index
LOOM_DEFINE_ISA(loom_kernel_subgroup_size_isa, LOOM_OP_KERNEL_SUBGROUP_SIZE)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_size_result, 0)
iree_status_t loom_kernel_subgroup_size_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_size_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_SUBGROUP_LANE_ID: Read the current invocation coordinate within its subgroup.
// %lane = kernel.subgroup.lane.id : index
LOOM_DEFINE_ISA(loom_kernel_subgroup_lane_id_isa, LOOM_OP_KERNEL_SUBGROUP_LANE_ID)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_lane_id_result, 0)
iree_status_t loom_kernel_subgroup_lane_id_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_lane_id_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_SUBGROUP_SHUFFLE: Move a scalar or rank-1 vector value across lanes of the current subgroup. The result value has the same type as the input value, and the valid result reports whether the named source lane participated.
// %r, %valid = kernel.subgroup.shuffle<xor> %v, %offset, %width : f32, i32, i32
LOOM_DEFINE_ISA(loom_kernel_subgroup_shuffle_isa, LOOM_OP_KERNEL_SUBGROUP_SHUFFLE)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_shuffle_value, 0)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_shuffle_offset, 1)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_shuffle_width, 2)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_shuffle_result, 0)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_shuffle_valid, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_subgroup_shuffle_mode, 0, loom_kernel_subgroup_shuffle_mode_t)
iree_status_t loom_kernel_subgroup_shuffle_build(
    loom_builder_t* builder,
    loom_kernel_subgroup_shuffle_mode_t mode,
    loom_value_id_t value,
    loom_value_id_t offset,
    loom_value_id_t width,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_shuffle_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_SUBGROUP_BROADCAST: Broadcast a scalar or rank-1 vector value from one named subgroup lane.
// %r = kernel.subgroup.broadcast %v from %lane : f32, i32
LOOM_DEFINE_ISA(loom_kernel_subgroup_broadcast_isa, LOOM_OP_KERNEL_SUBGROUP_BROADCAST)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_broadcast_value, 0)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_broadcast_lane, 1)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_broadcast_result, 0)
iree_status_t loom_kernel_subgroup_broadcast_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_value_id_t lane,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_broadcast_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST: Broadcast a scalar or rank-1 vector value from the first active subgroup lane.
// %r = kernel.subgroup.broadcast.first %v : f32
LOOM_DEFINE_ISA(loom_kernel_subgroup_broadcast_first_isa, LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_broadcast_first_value, 0)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_broadcast_first_result, 0)
iree_status_t loom_kernel_subgroup_broadcast_first_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_value_result_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_SUBGROUP_REDUCE: Reduce a scalar or rank-1 vector value across the current subgroup.
// %sum = kernel.subgroup.reduce<addf> %v : f32
LOOM_DEFINE_ISA(loom_kernel_subgroup_reduce_isa, LOOM_OP_KERNEL_SUBGROUP_REDUCE)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_reduce_value, 0)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_reduce_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_subgroup_reduce_kind, 0, loom_combining_kind_t)
LOOM_DEFINE_ATTR_I64(loom_kernel_subgroup_reduce_cluster_size, 1)
LOOM_DEFINE_ATTR_I64(loom_kernel_subgroup_reduce_cluster_stride, 2)
enum loom_kernel_subgroup_reduce_build_flag_bits_e {
  LOOM_KERNEL_SUBGROUP_REDUCE_BUILD_FLAG_HAS_CLUSTER_SIZE = 1u << 0,
  LOOM_KERNEL_SUBGROUP_REDUCE_BUILD_FLAG_HAS_CLUSTER_STRIDE = 1u << 1,
};
typedef uint32_t loom_kernel_subgroup_reduce_build_flags_t;
iree_status_t loom_kernel_subgroup_reduce_build(
    loom_builder_t* builder,
    loom_kernel_subgroup_reduce_build_flags_t build_flags,
    loom_combining_kind_t kind,
    loom_value_id_t value,
    loom_optional int64_t cluster_size,
    loom_optional int64_t cluster_stride,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_SUBGROUP_SCAN: Prefix-scan a scalar or rank-1 vector value across the current subgroup.
// %prefix = kernel.subgroup.scan<addf> %v {mode = inclusive, direction = forward} : f32
LOOM_DEFINE_ISA(loom_kernel_subgroup_scan_isa, LOOM_OP_KERNEL_SUBGROUP_SCAN)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_scan_value, 0)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_scan_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_subgroup_scan_kind, 0, loom_combining_kind_t)
LOOM_DEFINE_ATTR_I64(loom_kernel_subgroup_scan_cluster_size, 1)
LOOM_DEFINE_ATTR_I64(loom_kernel_subgroup_scan_cluster_stride, 2)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_subgroup_scan_mode, 3, loom_kernel_subgroup_scan_mode_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_subgroup_scan_direction, 4, loom_kernel_subgroup_scan_direction_t)
enum loom_kernel_subgroup_scan_build_flag_bits_e {
  LOOM_KERNEL_SUBGROUP_SCAN_BUILD_FLAG_HAS_CLUSTER_SIZE = 1u << 0,
  LOOM_KERNEL_SUBGROUP_SCAN_BUILD_FLAG_HAS_CLUSTER_STRIDE = 1u << 1,
};
typedef uint32_t loom_kernel_subgroup_scan_build_flags_t;
iree_status_t loom_kernel_subgroup_scan_build(
    loom_builder_t* builder,
    loom_kernel_subgroup_scan_build_flags_t build_flags,
    loom_combining_kind_t kind,
    loom_value_id_t value,
    loom_optional int64_t cluster_size,
    loom_optional int64_t cluster_stride,
    loom_kernel_subgroup_scan_mode_t mode,
    loom_kernel_subgroup_scan_direction_t direction,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_scan_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_SUBGROUP_VOTE_ANY: Return true when any active subgroup lane has a true predicate.
// %any = kernel.subgroup.vote.any %p : i1
LOOM_DEFINE_ISA(loom_kernel_subgroup_vote_any_isa, LOOM_OP_KERNEL_SUBGROUP_VOTE_ANY)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_vote_any_predicate, 0)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_vote_any_result, 0)
iree_status_t loom_kernel_subgroup_vote_any_build(
    loom_builder_t* builder,
    loom_value_id_t predicate,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_vote_any_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_SUBGROUP_VOTE_ALL: Return true when all active subgroup lanes have a true predicate.
// %all = kernel.subgroup.vote.all %p : i1
LOOM_DEFINE_ISA(loom_kernel_subgroup_vote_all_isa, LOOM_OP_KERNEL_SUBGROUP_VOTE_ALL)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_vote_all_predicate, 0)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_vote_all_result, 0)
iree_status_t loom_kernel_subgroup_vote_all_build(
    loom_builder_t* builder,
    loom_value_id_t predicate,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_vote_all_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_KERNEL_SUBGROUP_VOTE_BALLOT: Return an integer mask of active subgroup lanes whose predicate is true.
// %mask = kernel.subgroup.vote.ballot %p : i1 -> i64
LOOM_DEFINE_ISA(loom_kernel_subgroup_vote_ballot_isa, LOOM_OP_KERNEL_SUBGROUP_VOTE_BALLOT)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_vote_ballot_predicate, 0)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_vote_ballot_mask, 0)
iree_status_t loom_kernel_subgroup_vote_ballot_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t predicate,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_vote_ballot_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_kernel_subgroup_mask_result_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_SUBGROUP_ACTIVE_MASK: Return an integer mask of the currently active subgroup lanes.
// %mask = kernel.subgroup.active.mask : i64
LOOM_DEFINE_ISA(loom_kernel_subgroup_active_mask_isa, LOOM_OP_KERNEL_SUBGROUP_ACTIVE_MASK)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_active_mask_mask, 0)
iree_status_t loom_kernel_subgroup_active_mask_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_active_mask_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_kernel_subgroup_mask_result_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_SUBGROUP_MATCH_ANY: Return a lane mask of active subgroup invocations with the same scalar value.
// %mask = kernel.subgroup.match.any %v : i32 -> i64
LOOM_DEFINE_ISA(loom_kernel_subgroup_match_any_isa, LOOM_OP_KERNEL_SUBGROUP_MATCH_ANY)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_match_any_value, 0)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_match_any_mask, 0)
iree_status_t loom_kernel_subgroup_match_any_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_match_any_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_kernel_subgroup_mask_result_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL: Return a lane mask and predicate describing whether all active subgroup lanes hold the same scalar value.
// %mask, %all = kernel.subgroup.match.all %v : i32 -> i64, i1
LOOM_DEFINE_ISA(loom_kernel_subgroup_match_all_isa, LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL)
LOOM_DEFINE_OPERAND(loom_kernel_subgroup_match_all_value, 0)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_match_all_mask, 0)
LOOM_DEFINE_RESULT(loom_kernel_subgroup_match_all_all_equal, 1)
iree_status_t loom_kernel_subgroup_match_all_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_subgroup_match_all_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_kernel_subgroup_match_all_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_WORKGROUP_REDUCE: Reduce a scalar or rank-1 vector value across the current workgroup.
// %sum = kernel.workgroup.reduce<addf> %v : f32
LOOM_DEFINE_ISA(loom_kernel_workgroup_reduce_isa, LOOM_OP_KERNEL_WORKGROUP_REDUCE)
LOOM_DEFINE_OPERAND(loom_kernel_workgroup_reduce_value, 0)
LOOM_DEFINE_RESULT(loom_kernel_workgroup_reduce_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_workgroup_reduce_kind, 0, loom_combining_kind_t)
iree_status_t loom_kernel_workgroup_reduce_build(
    loom_builder_t* builder,
    loom_combining_kind_t kind,
    loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_workgroup_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_WORKGROUP_SCAN: Prefix-scan a scalar or rank-1 vector value across the current workgroup.
// %prefix = kernel.workgroup.scan<addf> %v {mode = inclusive, direction = forward} : f32
LOOM_DEFINE_ISA(loom_kernel_workgroup_scan_isa, LOOM_OP_KERNEL_WORKGROUP_SCAN)
LOOM_DEFINE_OPERAND(loom_kernel_workgroup_scan_value, 0)
LOOM_DEFINE_RESULT(loom_kernel_workgroup_scan_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_workgroup_scan_kind, 0, loom_combining_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_workgroup_scan_mode, 1, loom_kernel_workgroup_scan_mode_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_kernel_workgroup_scan_direction, 2, loom_kernel_workgroup_scan_direction_t)
iree_status_t loom_kernel_workgroup_scan_build(
    loom_builder_t* builder,
    loom_combining_kind_t kind,
    loom_value_id_t value,
    loom_kernel_workgroup_scan_mode_t mode,
    loom_kernel_workgroup_scan_direction_t direction,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_workgroup_scan_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_WORKGROUP_VOTE_ANY: Return true when any workgroup invocation has a true predicate.
// %any = kernel.workgroup.vote.any %p : i1
LOOM_DEFINE_ISA(loom_kernel_workgroup_vote_any_isa, LOOM_OP_KERNEL_WORKGROUP_VOTE_ANY)
LOOM_DEFINE_OPERAND(loom_kernel_workgroup_vote_any_predicate, 0)
LOOM_DEFINE_RESULT(loom_kernel_workgroup_vote_any_result, 0)
iree_status_t loom_kernel_workgroup_vote_any_build(
    loom_builder_t* builder,
    loom_value_id_t predicate,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_KERNEL_WORKGROUP_VOTE_ALL: Return true when all workgroup invocations have a true predicate.
// %all = kernel.workgroup.vote.all %p : i1
LOOM_DEFINE_ISA(loom_kernel_workgroup_vote_all_isa, LOOM_OP_KERNEL_WORKGROUP_VOTE_ALL)
LOOM_DEFINE_OPERAND(loom_kernel_workgroup_vote_all_predicate, 0)
LOOM_DEFINE_RESULT(loom_kernel_workgroup_vote_all_result, 0)
iree_status_t loom_kernel_workgroup_vote_all_build(
    loom_builder_t* builder,
    loom_value_id_t predicate,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_KERNEL_WORKGROUP_VOTE_COUNT: Count workgroup invocations with a true predicate.
// %count = kernel.workgroup.vote.count %p : i1 -> i32
LOOM_DEFINE_ISA(loom_kernel_workgroup_vote_count_isa, LOOM_OP_KERNEL_WORKGROUP_VOTE_COUNT)
LOOM_DEFINE_OPERAND(loom_kernel_workgroup_vote_count_predicate, 0)
LOOM_DEFINE_RESULT(loom_kernel_workgroup_vote_count_result, 0)
iree_status_t loom_kernel_workgroup_vote_count_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t predicate,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_workgroup_vote_count_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASSERT: Runtime assertion inside a dispatchable kernel. The condition is expected to be true; if it is false, target lowering must preserve runtime failure semantics through a trap/assert path or reject the kernel when assertions cannot be represented. This is not an optimization assume.
// kernel.assert %ok : i1
LOOM_DEFINE_ISA(loom_kernel_assert_isa, LOOM_OP_KERNEL_ASSERT)
LOOM_DEFINE_OPERAND(loom_kernel_assert_condition, 0)
LOOM_DEFINE_ATTR_STRING(loom_kernel_assert_message, 0)
enum loom_kernel_assert_build_flag_bits_e {
  LOOM_KERNEL_ASSERT_BUILD_FLAG_HAS_MESSAGE = 1u << 0,
};
typedef uint32_t loom_kernel_assert_build_flags_t;
iree_status_t loom_kernel_assert_build(
    loom_builder_t* builder,
    loom_kernel_assert_build_flags_t build_flags,
    loom_value_id_t condition,
    loom_optional loom_string_id_t message,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the kernel dialect.
const loom_op_vtable_t* const* loom_kernel_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the kernel dialect.
const loom_op_semantics_t* loom_kernel_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a kernel op kind, or empty metadata.
loom_op_semantics_t loom_kernel_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_KERNEL_OPS_H_
