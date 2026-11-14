// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_BUFFER_OPS_H_
#define LOOM_OPS_BUFFER_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/ir/memory_space.h"
#include "loom/ops/atomic.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_BUFFER_ALLOCA = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 0),
  LOOM_OP_BUFFER_ASSUME_ALIGNMENT = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 1),
  LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 2),
  LOOM_OP_BUFFER_ASSUME_NOALIAS = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 3),
  LOOM_OP_BUFFER_ASSUME_SAME_ROOT = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 4),
  LOOM_OP_BUFFER_VIEW = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 5),
  LOOM_OP_BUFFER_PACK = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 6),
  LOOM_OP_BUFFER_LENGTH = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 7),
  LOOM_OP_BUFFER_LOAD_I8_U = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 8),
  LOOM_OP_BUFFER_STORE_I8 = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 9),
  LOOM_OP_BUFFER_COPY = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 10),
  LOOM_OP_BUFFER_FILL = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 11),
  LOOM_OP_BUFFER_COMPARE = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 12),
  LOOM_OP_BUFFER_FENCE = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 13),
  LOOM_OP_BUFFER_CONSTANT = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 14),
  LOOM_OP_BUFFER_ALLOCA_INIT = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 15),
  LOOM_OP_BUFFER_DUPLICATE = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 16),
  LOOM_OP_BUFFER_NULL = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 17),
  LOOM_OP_BUFFER_COMPARE_NULL = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 18),
  LOOM_OP_BUFFER_COMPARE_ADDRESS = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 19),
  LOOM_OP_BUFFER_DISTANCE = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 20),
  LOOM_OP_BUFFER_RESTRICT_BEGIN = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 21),
  LOOM_OP_BUFFER_RESTRICT_END = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 22),
  LOOM_OP_BUFFER_COUNT_ = 23,
};

// Equality predicates for opaque buffer positions.
typedef enum loom_buffer_compare_address_predicate_e {
  LOOM_BUFFER_COMPARE_ADDRESS_PREDICATE_EQ = 0,
  LOOM_BUFFER_COMPARE_ADDRESS_PREDICATE_NE = 1,
  LOOM_BUFFER_COMPARE_ADDRESS_PREDICATE_COUNT_ = 2,
} loom_buffer_compare_address_predicate_t;

// LOOM_OP_BUFFER_ALLOCA: Create a fixed-frame scratch buffer root in an allocatable memory space. Each execution produces a distinct storage identity; identical allocas must not be commoned. The byte length is the requested physical byte count for the execution. Targets requiring a static frame reserve its proven finite non-negative maximum. base_alignment is the minimum byte alignment of the root storage base. Target lowering determines which allocatable spaces are legal for the containing program kind. The execution frame owns the allocation; the result borrows that storage and has no explicit release obligation. An explicit pool selects the backing allocation resource; memory_space remains an access requirement that resource must satisfy. The pool is borrowed and does not extend the allocation lifetime. Omitting it uses the containing execution's allocation resource for that space, which must be unambiguous before materialization. Lowering cannot replace an explicit pool with an ambient allocation resource.
// %scratch = buffer.alloca<workgroup> align(64) %bytes : buffer
LOOM_DEFINE_ISA(loom_buffer_alloca_isa, LOOM_OP_BUFFER_ALLOCA)
LOOM_DEFINE_SEGMENTED_OPTIONAL_OPERAND(loom_buffer_alloca_pool, 0)
LOOM_DEFINE_SEGMENTED_OPERAND(loom_buffer_alloca_byte_length, 1)
LOOM_DEFINE_RESULT(loom_buffer_alloca_result, 0)
LOOM_DEFINE_ATTR_I64(loom_buffer_alloca_base_alignment, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_buffer_alloca_memory_space, 1, loom_value_fact_memory_space_t)
enum loom_buffer_alloca_build_flag_bits_e {
  LOOM_BUFFER_ALLOCA_BUILD_FLAG_HAS_POOL = 1u << 0,
};
typedef uint32_t loom_buffer_alloca_build_flags_t;
iree_status_t loom_buffer_alloca_build(
    loom_builder_t* builder,
    loom_buffer_alloca_build_flags_t build_flags,
    loom_value_fact_memory_space_t memory_space,
    loom_optional loom_may_consume loom_value_id_t pool,
    int64_t base_alignment,
    loom_may_consume loom_value_id_t byte_length,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_alloca_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_alloca_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_ASSUME_ALIGNMENT: Refine existing buffer roots with an explicit minimum byte alignment contract. The result preserves the same storage identity, extent, memory-space, alias, and nullability facts while strengthening the root base alignment fact.
// %aligned = buffer.assume.alignment %buffer {minimum_alignment = 16} : buffer
LOOM_DEFINE_ISA(loom_buffer_assume_alignment_isa, LOOM_OP_BUFFER_ASSUME_ALIGNMENT)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_buffer_assume_alignment_buffers, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_buffer_assume_alignment_results, 0)
LOOM_DEFINE_ATTR_I64(loom_buffer_assume_alignment_minimum_alignment, 0)
iree_status_t loom_buffer_assume_alignment_build(
    loom_builder_t* builder,
    const loom_value_id_t* buffers,
    iree_host_size_t buffers_count,
    int64_t minimum_alignment,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_assume_alignment_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_assume_alignment_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE: Refine an existing buffer root with a concrete target-independent memory-space fact while preserving the same storage identity, extent, alignment, and nullability facts.
// %global = buffer.assume.memory_space<global> %buffer : buffer
LOOM_DEFINE_ISA(loom_buffer_assume_memory_space_isa, LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE)
LOOM_DEFINE_OPERAND(loom_buffer_assume_memory_space_buffer, 0)
LOOM_DEFINE_RESULT(loom_buffer_assume_memory_space_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_buffer_assume_memory_space_memory_space, 0, loom_value_fact_memory_space_t)
iree_status_t loom_buffer_assume_memory_space_build(
    loom_builder_t* builder,
    loom_value_fact_memory_space_t memory_space,
    loom_value_id_t buffer,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_assume_memory_space_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_assume_memory_space_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_ASSUME_NOALIAS: Refine an existing buffer root with an explicit noalias contract. The result preserves the same storage identity, extent, memory-space, alignment, and nullability facts, and marks the root identity as comparable for disjointness proofs. External buffer arguments do not gain this proof by default. The promise separates distinct scoped roots in the same execution. Repeated executions may bind a root to different storage; noalias alone does not establish disjointness across executions.
// %unique = buffer.assume.noalias %buffer : buffer
LOOM_DEFINE_ISA(loom_buffer_assume_noalias_isa, LOOM_OP_BUFFER_ASSUME_NOALIAS)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_buffer_assume_noalias_buffers, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_buffer_assume_noalias_results, 0)
iree_status_t loom_buffer_assume_noalias_build(
    loom_builder_t* builder,
    const loom_value_id_t* buffers,
    iree_host_size_t buffers_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_assume_noalias_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_ASSUME_SAME_ROOT: Refine an existing buffer root to share another buffer's storage root. This is a dominance-scoped assertion for internally specialized dispatches that know two incoming handles refer to the same allocation. The result keeps the first operand's value while inheriting the second operand's root identity and comparable alias scope.
// %same = buffer.assume.same_root %buffer, %root : buffer
LOOM_DEFINE_ISA(loom_buffer_assume_same_root_isa, LOOM_OP_BUFFER_ASSUME_SAME_ROOT)
LOOM_DEFINE_OPERAND(loom_buffer_assume_same_root_buffer, 0)
LOOM_DEFINE_OPERAND(loom_buffer_assume_same_root_root, 1)
LOOM_DEFINE_RESULT(loom_buffer_assume_same_root_result, 0)
iree_status_t loom_buffer_assume_same_root_build(
    loom_builder_t* builder,
    loom_value_id_t buffer,
    loom_value_id_t root,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_assume_same_root_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_VIEW: Form a typed non-owning view from an opaque buffer root and base byte offset. The result view type carries the address layout.
// %view = buffer.view %buffer[%offset] : buffer -> view<[%M]xf32, %layout>
LOOM_DEFINE_ISA(loom_buffer_view_isa, LOOM_OP_BUFFER_VIEW)
LOOM_DEFINE_OPERAND(loom_buffer_view_buffer, 0)
LOOM_DEFINE_OPERAND(loom_buffer_view_byte_offset, 1)
LOOM_DEFINE_RESULT(loom_buffer_view_result, 0)
iree_status_t loom_buffer_view_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_may_consume loom_value_id_t byte_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_view_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_view_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_PACK: Lay out simultaneously live physical byte ranges in one dense slab. Ranges retain operand order, begin at offsets satisfying their minimum alignments, and never alias. The total byte length is rounded up to the greatest range alignment so the result can be used as a repeatable record stride. Byte lengths may remain dynamic through specialization. The complete aligned extent must be representable in the selected target's offset domain. Allocation lifetime packing is a separate compiler responsibility and must not be encoded with this operation.
// %total, %header_offset, %payload_offset = buffer.pack [align(16) %header_bytes, align(256) %payload_bytes] : offset
LOOM_DEFINE_ISA(loom_buffer_pack_isa, LOOM_OP_BUFFER_PACK)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_buffer_pack_byte_lengths, 0)
LOOM_DEFINE_RESULT(loom_buffer_pack_total_byte_length, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_buffer_pack_byte_offsets, 1)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_buffer_pack_minimum_alignments, 0)
iree_status_t loom_buffer_pack_build(
    loom_builder_t* builder,
    loom_may_consume const loom_value_id_t* byte_lengths,
    iree_host_size_t byte_lengths_count,
    const int64_t* minimum_alignments,
    iree_host_size_t minimum_alignments_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_pack_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_pack_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_LENGTH: Query the physical byte length of a buffer root without accessing its payload. Returns zero when the buffer is null.
// %byte_length = buffer.length %buffer
LOOM_DEFINE_ISA(loom_buffer_length_isa, LOOM_OP_BUFFER_LENGTH)
LOOM_DEFINE_OPERAND(loom_buffer_length_buffer, 0)
LOOM_DEFINE_RESULT(loom_buffer_length_byte_length, 0)
iree_status_t loom_buffer_length_build(
    loom_builder_t* builder,
    loom_value_id_t buffer,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_length_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_LOAD_I8_U: Load one unsigned byte from a buffer root and zero-extend it to the canonical i32 carrier. The byte offset must identify an accessible byte in the buffer.
// %byte = buffer.load.i8.u %source[%byte_offset]
LOOM_DEFINE_ISA(loom_buffer_load_i8_u_isa, LOOM_OP_BUFFER_LOAD_I8_U)
LOOM_DEFINE_OPERAND(loom_buffer_load_i8_u_source, 0)
LOOM_DEFINE_OPERAND(loom_buffer_load_i8_u_byte_offset, 1)
LOOM_DEFINE_RESULT(loom_buffer_load_i8_u_result, 0)
iree_status_t loom_buffer_load_i8_u_build(
    loom_builder_t* builder,
    loom_value_id_t source,
    loom_value_id_t byte_offset,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_load_i8_u_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_STORE_I8: Store the low eight bits of an i32 carrier to one byte in a buffer root. The byte offset must identify an accessible byte in the buffer.
// buffer.store.i8 %byte, %target[%byte_offset]
LOOM_DEFINE_ISA(loom_buffer_store_i8_isa, LOOM_OP_BUFFER_STORE_I8)
LOOM_DEFINE_OPERAND(loom_buffer_store_i8_value, 0)
LOOM_DEFINE_OPERAND(loom_buffer_store_i8_target, 1)
LOOM_DEFINE_OPERAND(loom_buffer_store_i8_byte_offset, 2)
iree_status_t loom_buffer_store_i8_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_value_id_t target,
    loom_value_id_t byte_offset,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_BUFFER_COPY: Copy an exact non-overlapping byte range between buffer roots. Source and target ranges must not overlap; programs may not depend on an overlap-safe target implementation. A zero byte length performs no byte access.
// buffer.copy %source[%source_offset], %target[%target_offset], %byte_length
LOOM_DEFINE_ISA(loom_buffer_copy_isa, LOOM_OP_BUFFER_COPY)
LOOM_DEFINE_OPERAND(loom_buffer_copy_source, 0)
LOOM_DEFINE_OPERAND(loom_buffer_copy_source_offset, 1)
LOOM_DEFINE_OPERAND(loom_buffer_copy_target, 2)
LOOM_DEFINE_OPERAND(loom_buffer_copy_target_offset, 3)
LOOM_DEFINE_OPERAND(loom_buffer_copy_byte_length, 4)
iree_status_t loom_buffer_copy_build(
    loom_builder_t* builder,
    loom_value_id_t source,
    loom_value_id_t source_offset,
    loom_value_id_t target,
    loom_value_id_t target_offset,
    loom_value_id_t byte_length,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_BUFFER_FILL: Repeat the raw little-endian bytes of an 8-, 16-, 32-, or 64-bit integer or floating-point scalar across an exact writable byte range. A final partial repetition writes the low-address prefix of the pattern bytes. A zero byte length performs no byte access.
// buffer.fill %pattern, %target[%target_offset], %byte_length : bf16
LOOM_DEFINE_ISA(loom_buffer_fill_isa, LOOM_OP_BUFFER_FILL)
LOOM_DEFINE_OPERAND(loom_buffer_fill_pattern, 0)
LOOM_DEFINE_OPERAND(loom_buffer_fill_target, 1)
LOOM_DEFINE_OPERAND(loom_buffer_fill_target_offset, 2)
LOOM_DEFINE_OPERAND(loom_buffer_fill_byte_length, 3)
iree_status_t loom_buffer_fill_build(
    loom_builder_t* builder,
    loom_value_id_t pattern,
    loom_value_id_t target,
    loom_value_id_t target_offset,
    loom_value_id_t byte_length,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_BUFFER_COMPARE: Lexicographically compare equal-length ranges as unsigned bytes and return canonical i32 -1, 0, or +1. A zero byte length returns zero and performs no byte access.
// %order = buffer.compare %lhs[%lhs_offset], %rhs[%rhs_offset], %byte_length
LOOM_DEFINE_ISA(loom_buffer_compare_isa, LOOM_OP_BUFFER_COMPARE)
LOOM_DEFINE_OPERAND(loom_buffer_compare_lhs, 0)
LOOM_DEFINE_OPERAND(loom_buffer_compare_lhs_offset, 1)
LOOM_DEFINE_OPERAND(loom_buffer_compare_rhs, 2)
LOOM_DEFINE_OPERAND(loom_buffer_compare_rhs_offset, 3)
LOOM_DEFINE_OPERAND(loom_buffer_compare_byte_length, 4)
LOOM_DEFINE_RESULT(loom_buffer_compare_order, 0)
iree_status_t loom_buffer_compare_build(
    loom_builder_t* builder,
    loom_value_id_t lhs,
    loom_value_id_t lhs_offset,
    loom_value_id_t rhs,
    loom_value_id_t rhs_offset,
    loom_value_id_t byte_length,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_compare_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_FENCE: Order the executing thread's memory accesses across storage roots using an explicit atomic ordering and synchronization scope. This applies to ordinary functions and kernels, including accesses through typed views and vectors. Acquire and release fences participate in synchronization through matching atomic observations and publications; they do not rendezvous with other invocations or complete independent asynchronous transfers. The backing memory and target must support the requested synchronization domain.
// buffer.fence scope(system) ordering(acquire)
LOOM_DEFINE_ISA(loom_buffer_fence_isa, LOOM_OP_BUFFER_FENCE)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_buffer_fence_scope, 0, loom_atomic_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_buffer_fence_ordering, 1, loom_atomic_ordering_t)
iree_status_t loom_buffer_fence_build(
    loom_builder_t* builder,
    loom_atomic_scope_t scope,
    loom_atomic_ordering_t ordering,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_BUFFER_CONSTANT: Borrow anonymous immutable executable storage containing the exact payload bytes. Evaluation performs no per-invocation allocation or initialization. Equal contents may share storage, with no distinct address identity guarantee. The byte extent excludes alignment padding; UTF-8 adds no terminator or normalization. Empty payloads produce a nonnull zero-length buffer. Omitted alignment means one byte. Named objects with identity or linkage requirements use global.rodata.def.
// %message = buffer.constant utf8("ready") : buffer
LOOM_DEFINE_ISA(loom_buffer_constant_isa, LOOM_OP_BUFFER_CONSTANT)
LOOM_DEFINE_RESULT(loom_buffer_constant_result, 0)
LOOM_DEFINE_ATTR_BYTES(loom_buffer_constant_contents, 0)
LOOM_DEFINE_ATTR_I64(loom_buffer_constant_alignment, 1)
enum loom_buffer_constant_build_flag_bits_e {
  LOOM_BUFFER_CONSTANT_BUILD_FLAG_HAS_ALIGNMENT = 1u << 0,
};
typedef uint32_t loom_buffer_constant_build_flags_t;
iree_status_t loom_buffer_constant_build(
    loom_builder_t* builder,
    loom_buffer_constant_build_flags_t build_flags,
    loom_optional int64_t alignment,
    iree_const_byte_span_t contents,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_constant_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_constant_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_ALLOCA_INIT: Create fresh fixed-frame storage initialized with the exact payload bytes. Extent is the payload length, without implicit terminators or alignment padding. Ownership, lifetime, alignment, and backing-pool selection follow buffer.alloca; equal initializers never merge writable allocations. Workgroup initialization is collective: every workitem encounters the same dynamic instance in workgroup-uniform control, and the initialized bytes are visible to the whole workgroup before participants proceed. Other allocatable spaces initialize each invocation's independent storage without collective synchronization.
// %scratch = buffer.alloca.init<private> align(1) utf8("ready") : buffer
LOOM_DEFINE_ISA(loom_buffer_alloca_init_isa, LOOM_OP_BUFFER_ALLOCA_INIT)
LOOM_DEFINE_OPTIONAL_OPERAND(loom_buffer_alloca_init_pool, 0)
LOOM_DEFINE_RESULT(loom_buffer_alloca_init_result, 0)
LOOM_DEFINE_ATTR_I64(loom_buffer_alloca_init_base_alignment, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_buffer_alloca_init_memory_space, 1, loom_value_fact_memory_space_t)
LOOM_DEFINE_ATTR_BYTES(loom_buffer_alloca_init_contents, 2)
enum loom_buffer_alloca_init_build_flag_bits_e {
  LOOM_BUFFER_ALLOCA_INIT_BUILD_FLAG_HAS_POOL = 1u << 0,
};
typedef uint32_t loom_buffer_alloca_init_build_flags_t;
iree_status_t loom_buffer_alloca_init_build(
    loom_builder_t* builder,
    loom_buffer_alloca_init_build_flags_t build_flags,
    loom_value_fact_memory_space_t memory_space,
    loom_optional loom_may_consume loom_value_id_t pool,
    int64_t base_alignment,
    iree_const_byte_span_t contents,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
loom_trait_flags_t loom_buffer_alloca_init_effective_traits(
    const loom_module_t* module, const loom_op_t* op);
iree_status_t loom_buffer_alloca_init_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_alloca_init_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_DUPLICATE: Create another local owner of a buffer root. The duplicate identifies the same opaque storage and may lower to the same physical carrier, but remains a distinct SSA ownership identity.
// %copy = buffer.duplicate %source : buffer
LOOM_DEFINE_ISA(loom_buffer_duplicate_isa, LOOM_OP_BUFFER_DUPLICATE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_buffer_duplicate_sources, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_buffer_duplicate_duplicates, 0)
iree_status_t loom_buffer_duplicate_build(
    loom_builder_t* builder,
    const loom_value_id_t* sources,
    iree_host_size_t sources_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_duplicate_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_NULL: Produce the null buffer, which identifies no storage. The result has zero byte length, allocates nothing, and carries no release obligation. It may be selected, passed, returned, and tested for null, but cannot back a memory access. Null is distinct from every allocated buffer, including zero-length storage. Its physical representation is target and memory-space dependent; it is not an integer address.
// %empty = buffer.null : buffer
LOOM_DEFINE_ISA(loom_buffer_null_isa, LOOM_OP_BUFFER_NULL)
LOOM_DEFINE_RESULT(loom_buffer_null_result, 0)
iree_status_t loom_buffer_null_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_null_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_COMPARE_NULL: Test whether a buffer is null. Returns true exactly when the operand identifies no storage, independently of its byte length. The query does not access memory or change ownership.
// %missing = buffer.compare.null %input : buffer
LOOM_DEFINE_ISA(loom_buffer_compare_null_isa, LOOM_OP_BUFFER_COMPARE_NULL)
LOOM_DEFINE_OPERAND(loom_buffer_compare_null_buffer, 0)
LOOM_DEFINE_RESULT(loom_buffer_compare_null_result, 0)
iree_status_t loom_buffer_compare_null_build(
    loom_builder_t* builder,
    loom_value_id_t buffer,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_compare_null_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_COMPARE_ADDRESS: Compare two byte positions without exposing their addresses. Each nonnull buffer must be live and its offset must be in bounds or at the one-past endpoint. A null buffer permits only offset zero and differs from every nonnull buffer, including empty storage. Aliased buffers compare by their selected positions, not wrapper identity. An empty root without native storage retains a distinct logical position shared by its subspans. The operation reads no payload bytes and changes no ownership.
// %done = buffer.compare.address eq, %cursor[%position], %end[%limit]
LOOM_DEFINE_ISA(loom_buffer_compare_address_isa, LOOM_OP_BUFFER_COMPARE_ADDRESS)
LOOM_DEFINE_OPERAND(loom_buffer_compare_address_lhs, 0)
LOOM_DEFINE_OPERAND(loom_buffer_compare_address_lhs_offset, 1)
LOOM_DEFINE_OPERAND(loom_buffer_compare_address_rhs, 2)
LOOM_DEFINE_OPERAND(loom_buffer_compare_address_rhs_offset, 3)
LOOM_DEFINE_RESULT(loom_buffer_compare_address_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_buffer_compare_address_predicate, 0, loom_buffer_compare_address_predicate_t)
iree_status_t loom_buffer_compare_address_build(
    loom_builder_t* builder,
    loom_buffer_compare_address_predicate_t predicate,
    loom_value_id_t lhs,
    loom_value_id_t lhs_offset,
    loom_value_id_t rhs,
    loom_value_id_t rhs_offset,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_compare_address_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_buffer_compare_address_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_DISTANCE: Return lhs_offset minus rhs_offset as a signed byte distance. Both positions must derive from the same buffer byte origin, proven before target lowering; otherwise compilation fails. Sharing an allocation without a common byte origin is insufficient. Nonnull positions must be live and in bounds or one-past. Two null positions at offset zero have distance zero. Offset values lie in [0, INT64_MAX], so their signed difference fits i64 and must also fit the signed result type. No address or payload bytes are observed.
// %bytes = buffer.distance %buffer[%end], %buffer[%begin] : i64
LOOM_DEFINE_ISA(loom_buffer_distance_isa, LOOM_OP_BUFFER_DISTANCE)
LOOM_DEFINE_OPERAND(loom_buffer_distance_lhs, 0)
LOOM_DEFINE_OPERAND(loom_buffer_distance_lhs_offset, 1)
LOOM_DEFINE_OPERAND(loom_buffer_distance_rhs, 2)
LOOM_DEFINE_OPERAND(loom_buffer_distance_rhs_offset, 3)
LOOM_DEFINE_RESULT(loom_buffer_distance_result, 0)
iree_status_t loom_buffer_distance_build(
    loom_builder_t* builder,
    loom_value_id_t lhs,
    loom_value_id_t lhs_offset,
    loom_value_id_t rhs,
    loom_value_id_t rhs_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_distance_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);

// LOOM_OP_BUFFER_RESTRICT_BEGIN: Begin a source-language restrict access scope over buffer roots and typed views. Restricted references name independent access paths; peers name the other incoming storage paths covered by the same scope. The results preserve physical storage identity and differ only in access-provenance facts. A restrict scope may prove that overlapping accesses do not occur in a defined program only when at least one access writes. It does not prove that roots are physically disjoint, and shared read-only storage remains valid. buffer.restrict.end removes the scope from values that leave it.
// %source_r = buffer.restrict.begin %source : buffer
LOOM_DEFINE_ISA(loom_buffer_restrict_begin_isa, LOOM_OP_BUFFER_RESTRICT_BEGIN)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_buffer_restrict_begin_restricted, 0)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_buffer_restrict_begin_peers, 1)
LOOM_DEFINE_VARIADIC_RESULTS(loom_buffer_restrict_begin_results, 0)
iree_status_t loom_buffer_restrict_begin_build(
    loom_builder_t* builder,
    const loom_value_id_t* restricted,
    iree_host_size_t restricted_count,
    const loom_value_id_t* peers,
    iree_host_size_t peers_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_restrict_begin_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_restrict_begin_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_RESTRICT_END: End source-language restrict access scopes carried by buffer roots or typed views. The results preserve every physical storage fact while removing the scoped access association, preventing it from escaping a lexical or invocation boundary.
// %returned = buffer.restrict.end %scoped : buffer
LOOM_DEFINE_ISA(loom_buffer_restrict_end_isa, LOOM_OP_BUFFER_RESTRICT_END)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_buffer_restrict_end_references, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_buffer_restrict_end_results, 0)
iree_status_t loom_buffer_restrict_end_build(
    loom_builder_t* builder,
    const loom_value_id_t* references,
    iree_host_size_t references_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_restrict_end_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_restrict_end_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the buffer dialect.
const loom_op_vtable_t* const* loom_buffer_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the buffer dialect.
const loom_op_semantics_t* loom_buffer_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a buffer op kind, or empty metadata.
loom_op_semantics_t loom_buffer_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

// Additional named attribute helpers are generated with this dialect.
#include "loom/ops/buffer/ops.inc"

#endif  // LOOM_OPS_BUFFER_OPS_H_
