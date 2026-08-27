// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU vector-memory, atomics, and prefetch lowering helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_H_

#include <stdint.h>

#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_source_alloca_layout_t
    loom_amdgpu_source_alloca_layout_t;

// Bounded caller-owned workspace for selecting one direct memory plan. The
// workspace lives only for one selection call and is never retained.
typedef struct loom_amdgpu_memory_access_selection_t {
  // Packet candidates populated in increasing source-register order.
  loom_amdgpu_memory_packet_plan_t packets[LOOM_AMDGPU_MAX_MEMORY_PACKET_COUNT];
  // Number of populated packet candidates.
  uint32_t packet_count;
} loom_amdgpu_memory_access_selection_t;

typedef struct loom_amdgpu_memory_dynamic_term_sequence_t {
  // Dynamic terms selected for emission in address-expression order.
  const loom_low_source_memory_dynamic_term_t*
      terms[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Target operand path selected for each emitted term.
  loom_amdgpu_memory_dynamic_index_kind_t
      kinds[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Number of populated term and kind entries.
  uint8_t count;
} loom_amdgpu_memory_dynamic_term_sequence_t;

typedef uint32_t loom_amdgpu_memory_access_rejection_flags_t;

#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE ((uint32_t)1u << 0)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE \
  ((uint32_t)1u << 1)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE \
  ((uint32_t)1u << 2)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE ((uint32_t)1u << 3)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT \
  ((uint32_t)1u << 4)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET \
  ((uint32_t)1u << 5)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT \
  ((uint32_t)1u << 6)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING \
  ((uint32_t)1u << 7)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE \
  ((uint32_t)1u << 8)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE \
  ((uint32_t)1u << 9)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE ((uint32_t)1u << 10)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT ((uint32_t)1u << 11)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SCALAR_DYNAMIC_STRIDE \
  ((uint32_t)1u << 12)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE \
  ((uint32_t)1u << 13)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE \
  ((uint32_t)1u << 14)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_ADDRESS \
  ((uint32_t)1u << 15)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_OFFSET_RANGE \
  ((uint32_t)1u << 16)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PACKED_REGISTER_FOOTPRINT \
  ((uint32_t)1u << 17)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_FLAT_DYNAMIC_ADDRESS \
  ((uint32_t)1u << 18)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE \
  ((uint32_t)1u << 19)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET_RANGE \
  ((uint32_t)1u << 20)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_LOW16_DESCRIPTOR_MISSING \
  ((uint32_t)1u << 21)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SIGNED_I16_REPAIR_DESCRIPTOR_MISSING \
  ((uint32_t)1u << 22)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PRIVATE_ROOT ((uint32_t)1u << 23)

typedef struct loom_amdgpu_memory_access_diagnostic_t {
  // Rejection bits explaining why an access is not legal for this target.
  loom_amdgpu_memory_access_rejection_flags_t rejection_bits;
  // Source vector type involved in vector-width diagnostics.
  loom_type_t vector_type;
  // Source vector lane count involved in vector-width diagnostics.
  uint32_t vector_lane_count;
  // Byte count of one source vector element in vector-width diagnostics.
  uint32_t element_byte_count;
  // Required 32-bit payload lanes for the source vector.
  uint32_t required_32bit_lane_count;
  // Maximum source lanes representable by one native memory packet.
  uint32_t native_max_vector_lane_count;
  // Maximum source lanes representable by scalarized fallback lowering.
  uint32_t scalarized_max_vector_lane_count;
  // Source payload type involved in register-footprint diagnostics.
  loom_type_t payload_type;
  // Number of source payload bits involved in register-footprint diagnostics.
  uint32_t payload_bit_count;
  // Number of selected register-footprint bits for the payload.
  uint32_t register_bit_count;
  // First dynamic address term involved in dynamic-address diagnostics.
  uint32_t dynamic_term_index;
} loom_amdgpu_memory_access_diagnostic_t;

typedef struct loom_amdgpu_descriptor_offset_immediate_info_t {
  // Low immediate kind required by all offset immediate fields.
  loom_low_immediate_kind_t kind;
  // Minimum encoded value accepted by signed offset immediate fields.
  int64_t signed_min;
  // Maximum encoded value accepted by every offset immediate field.
  uint64_t unsigned_max;
  // Byte count represented by one encoded offset unit.
  uint32_t unit_byte_count;
} loom_amdgpu_descriptor_offset_immediate_info_t;

typedef uint32_t loom_amdgpu_memory_cache_policy_attr_flags_t;

#define LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_SCOPE ((uint32_t)1u << 0)
#define LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_TH ((uint32_t)1u << 1)
#define LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_NT ((uint32_t)1u << 2)

typedef struct loom_amdgpu_memory_cache_policy_attrs_t {
  // Encoded attribute bits present for the selected descriptor-set encoding.
  loom_amdgpu_memory_cache_policy_attr_flags_t flags;
  // SCOPE immediate value for GFX12 vector memory packets.
  int64_t scope;
  // TH immediate value for GFX12 vector memory packets.
  int64_t th;
  // NT immediate value for GFX950 vector memory packets.
  int64_t nt;
} loom_amdgpu_memory_cache_policy_attrs_t;

// Result of resolving an advisory source cache policy for a descriptor set.
typedef enum loom_amdgpu_memory_cache_policy_resolution_e {
  // The source memory operation has no cache policy.
  LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_ABSENT = 0,
  // The selected descriptor set can encode the source cache policy.
  LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_ENCODED = 1,
  // The policy is valid but unsupported and must be omitted from the packet.
  LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_DROPPED = 2,
  // The policy is invalid for the selected memory access or descriptor set.
  LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_REJECTED = 3,
} loom_amdgpu_memory_cache_policy_resolution_t;

// Reads common offset-immediate limits from a descriptor.
bool loom_amdgpu_descriptor_offset_immediate_info(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint16_t expected_offset_immediate_count,
    loom_low_immediate_kind_t expected_kind,
    loom_amdgpu_descriptor_offset_immediate_info_t* out_info);

// Returns true when the source memory byte offset selected for a 32-bit AMDGPU
// packet operand is fully proven non-negative and u32 after adding the static
// byte contribution named by the selected address form.
bool loom_amdgpu_source_memory_offset_fits_u32(
    const loom_low_source_memory_access_plan_t* source,
    int64_t static_byte_offset);

// Selects a complete AMDGPU memory packet sequence from source IR and facts
// into caller-owned bounded workspace.
bool loom_amdgpu_memory_access_plan_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions,
    loom_func_like_t source_function, const loom_target_bundle_t* bundle,
    loom_amdgpu_instruction_constraint_bits_t instruction_constraints,
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    const loom_op_t* source_op,
    loom_low_source_memory_access_plan_t* out_source,
    loom_amdgpu_memory_access_selection_t* out_selection,
    loom_low_source_memory_access_diagnostic_t* out_source_diagnostic,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic);

// Selects a flat global address plan for instrumentation that needs the actual
// application pointer rather than a memory packet for the source access.
bool loom_amdgpu_memory_access_select_flat_global_address(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic);

// Selects an explicit VGPR byte-offset expression for source access
// instrumentation.
//
// The selected access has all static byte offset, including target-assigned
// workgroup/private alloca layout, and all dynamic byte terms routed through
// VADDR. It does not select a memory packet descriptor. Callers can pass the
// result to loom_amdgpu_emit_memory_vaddr with no base value to materialize the
// exact u32 memory-space-relative byte offset.
bool loom_amdgpu_memory_access_select_u32_vaddr_byte_offset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic);

// Folds target-assigned workgroup/private alloca layout into a source memory
// access so later offset materialization uses the same address domain as real
// AMDGPU memory packets.
bool loom_amdgpu_memory_access_include_alloca_root_byte_offset(
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic);

// Selects target operand paths for dynamic source memory address terms.
bool loom_amdgpu_memory_access_select_dynamic_term_kinds(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic);

// Resolves canonical address terms to their emission sequence. An equivalent
// source realization replaces its canonical term range only when the source
// value is already materialized for another use.
void loom_amdgpu_memory_access_resolve_dynamic_terms(
    const loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_dynamic_index_kind_t* dynamic_term_kinds,
    loom_amdgpu_memory_dynamic_term_sequence_t* out_sequence);

// Routes all dynamic source terms through the VGPR byte-address operand.
void loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(
    loom_amdgpu_memory_access_t* access);

// Emits the VGPR address operand for a selected memory access.
iree_status_t loom_amdgpu_emit_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t low_base_addr,
    loom_value_id_t* out_low_vaddr);

// Emits the SGPR SADDR operand for a low HAL binding pointer.
iree_status_t loom_amdgpu_emit_memory_saddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t low_binding,
    loom_value_id_t* out_low_saddr);

// Emits the target buffer descriptor consumed by MUBUF-style packets from a low
// HAL binding pointer. When the low resource has no explicit extent,
// |source_access| may provide source view facts used to derive the descriptor
// range word.
iree_status_t loom_amdgpu_emit_hal_buffer_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_binding,
    const loom_low_source_memory_access_plan_t* source_access,
    loom_value_id_t* out_low_descriptor);

// Emits the 64-bit flat VGPR address sliced from a low HAL binding pointer and
// extended by the selected source memory byte offset.
iree_status_t loom_amdgpu_emit_memory_flat_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t low_binding,
    loom_value_id_t* out_low_vaddr);

// Builds descriptor offset and cache-policy attrs for a memory packet.
iree_status_t loom_amdgpu_make_memory_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count);

// Builds only the descriptor cache-policy attrs for a memory-like packet that
// has no address offset immediate.
iree_status_t loom_amdgpu_make_memory_cache_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count);

// Returns true when an access carries an explicit cache policy.
bool loom_amdgpu_memory_cache_policy_is_present(
    const loom_vector_memory_cache_policy_t* policy);

// Returns the stable diagnostic key for the selected descriptor-set cache
// policy encoding.
iree_string_view_t loom_amdgpu_memory_cache_policy_encoding_key(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns the vector-memory cache-policy encoding for |descriptor_set|.
loom_amdgpu_vector_memory_cache_policy_encoding_t
loom_amdgpu_memory_cache_policy_descriptor_encoding(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns the stable diagnostic decision key for a selected cache-policy
// encoding.
iree_string_view_t loom_amdgpu_memory_cache_policy_selected_key(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns true when the cache policy can be encoded or safely omitted for the
// selected descriptor set and memory access plan.
bool loom_amdgpu_memory_cache_policy_can_lower(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access);

// Returns the stable diagnostic name for a memory space.
iree_string_view_t loom_amdgpu_memory_space_name(
    loom_value_fact_memory_space_t memory_space);

// Returns the stable report/diagnostic name for a memory operation kind.
iree_string_view_t loom_amdgpu_memory_operation_name(
    loom_low_source_memory_operation_kind_t kind);

// Returns the stable report/diagnostic name for a memory address form.
iree_string_view_t loom_amdgpu_memory_address_form_name(
    loom_amdgpu_memory_address_form_t address_form);

// Returns the stable report/diagnostic name for the selected dynamic term path.
iree_string_view_t loom_amdgpu_memory_access_dynamic_term_kind_name(
    const loom_amdgpu_memory_access_t* access);

// Returns the stable report/diagnostic reason key explaining why an LDS access
// did or did not select the DS addtid address form.
iree_string_view_t loom_amdgpu_memory_ds_addtid_reason_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_target_bundle_t* bundle,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind);

// Returns the stable diagnostic name for a cache scope.
iree_string_view_t loom_amdgpu_cache_scope_name(uint8_t scope);

// Returns the stable diagnostic name for a cache temporal policy.
iree_string_view_t loom_amdgpu_cache_temporal_name(uint8_t temporal);

// Resolves target-specific cache-policy attributes for a selected memory
// access plan. Absent and dropped policies produce an empty attrs struct.
loom_amdgpu_memory_cache_policy_resolution_t
loom_amdgpu_memory_cache_policy_resolve(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_cache_policy_attrs_t* out_attrs);

// Returns a stable diagnostic constraint key for the cache policy selected by
// the memory access.
iree_string_view_t loom_amdgpu_memory_cache_policy_rejection_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    const loom_vector_memory_cache_policy_t* policy);

// Returns the stable diagnostic constraint key for target-specific
// memory-access rejection bits.
iree_string_view_t loom_amdgpu_memory_access_rejection_key(
    loom_amdgpu_memory_access_rejection_flags_t rejection_bits);

// Records the first source dynamic term relevant to a flat-address rejection.
void loom_amdgpu_memory_access_record_flat_dynamic_address_rejection(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic);

// Emits a dedicated AMDGPU memory-access rejection diagnostic when one exists
// for the recorded rejection bits.
iree_status_t loom_amdgpu_emit_memory_access_rejection_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic,
    bool* out_handled);

// Records optional memory diagnostics for a selected memory access plan.
iree_status_t loom_amdgpu_record_memory_access_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind);

// Records optional cache-policy diagnostics for a selected memory access plan.
iree_status_t loom_amdgpu_record_memory_cache_policy_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_cache_policy_resolution_t resolution,
    const loom_amdgpu_memory_cache_policy_attrs_t* cache_attrs);

// Records optional cache-policy diagnostics for a rejected memory access plan.
iree_status_t loom_amdgpu_record_memory_cache_policy_rejection_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access);

// Populates optional source storage-schema fields on a memory report row.
void loom_amdgpu_memory_report_row_populate_storage_schema(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source,
    loom_low_lower_memory_report_row_t* row);

// Selects and retains an exact-sized AMDGPU source memory-load packet plan.
iree_status_t loom_amdgpu_select_memory_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Selects and retains an exact-sized AMDGPU source memory-store packet plan.
iree_status_t loom_amdgpu_select_memory_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Lowers a source memory-load op to an AMDGPU memory packet.
iree_status_t loom_amdgpu_lower_memory_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan);

// Lowers a source memory-store op to an AMDGPU memory packet.
iree_status_t loom_amdgpu_lower_memory_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan);

// Marks source values required by a selected source memory access plan.
void loom_amdgpu_mark_source_memory_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source);

// Marks the storage root and dynamic address terms required by a selected
// source memory access plan.
void loom_amdgpu_mark_source_memory_plan_root_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source);

// Marks only dynamic address terms required by a source memory access plan.
// This is used when target-assigned static layout replaces the source root.
void loom_amdgpu_mark_source_memory_plan_dynamic_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source);

// Marks the physical source values needed by a selected AMDGPU memory plan.
void loom_amdgpu_mark_memory_access_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan);

// Returns whether the selected descriptor set contains a native packet
// candidate for the atomic semantic tuple. This queries packet availability
// only; source address, ordering, and operand representability remain the
// responsibility of atomic plan selection.
bool loom_amdgpu_atomic_has_descriptor_candidate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_atomic_operation_kind_t operation_kind, uint8_t atomic_kind,
    loom_type_t value_type);

// Selects an AMDGPU atomic packet plan.
iree_status_t loom_amdgpu_select_atomic_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_atomic_plan_t* out_plan, bool* out_selected);

// Lowers a source atomic op to an AMDGPU atomic packet.
iree_status_t loom_amdgpu_lower_atomic(loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       const loom_amdgpu_atomic_plan_t* plan);

// Marks the physical source values needed by a selected AMDGPU atomic plan.
void loom_amdgpu_mark_atomic_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_plan_t* plan);

// Selects an AMDGPU scalar-buffer data prefetch plan.
iree_status_t loom_amdgpu_select_view_prefetch_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_prefetch_plan_t* out_plan, bool* out_selected);

// Lowers a source view.prefetch to an AMDGPU scalar-buffer data prefetch
// packet.
iree_status_t loom_amdgpu_lower_view_prefetch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_prefetch_plan_t* plan);

// Marks the physical source values needed by a selected AMDGPU prefetch plan.
void loom_amdgpu_mark_prefetch_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_prefetch_plan_t* plan);

// Records optional prefetch diagnostics for source view.prefetch hints.
iree_status_t loom_amdgpu_record_view_prefetch_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* source_op,
    const loom_low_descriptor_set_t* descriptor_set);

// Verifies source memory legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies source atomic legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_atomic(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_H_
