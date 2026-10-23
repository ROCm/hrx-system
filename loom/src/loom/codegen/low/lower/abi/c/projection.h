// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent Low projection of retained C source objects.

#ifndef LOOM_CODEGEN_LOW_LOWER_ABI_C_PROJECTION_H_
#define LOOM_CODEGEN_LOW_LOWER_ABI_C_PROJECTION_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for an absent object, carrier, or physical result index.
#define LOOM_LOW_C_ABI_INDEX_NONE UINT16_MAX

typedef uint8_t loom_low_c_abi_byte_order_t;
enum loom_low_c_abi_byte_order_e {
  LOOM_LOW_C_ABI_BYTE_ORDER_LITTLE = 0,
  LOOM_LOW_C_ABI_BYTE_ORDER_BIG = 1,
};

typedef uint8_t loom_low_c_abi_object_transport_t;
enum loom_low_c_abi_object_transport_e {
  // Physical carriers contain the source object's byte image.
  LOOM_LOW_C_ABI_OBJECT_TRANSPORT_VALUE = 0,
  // One physical carrier points at the source object's byte image.
  LOOM_LOW_C_ABI_OBJECT_TRANSPORT_ADDRESS = 1,
};

typedef uint8_t loom_low_c_abi_leaf_flags_t;
enum loom_low_c_abi_leaf_flag_bits_e {
  // Leaf projects a buffer base and byte offset into one source pointer.
  LOOM_LOW_C_ABI_LEAF_FLAG_POINTER = 1u << 0,
  // Leaf is one native vector value.
  LOOM_LOW_C_ABI_LEAF_FLAG_VECTOR = 1u << 1,
  // Loaded scalar bits must be normalized to a one-bit logical value.
  LOOM_LOW_C_ABI_LEAF_FLAG_NORMALIZE_BOOLEAN = 1u << 2,
  // Logical and physical carrier values are already the same Low register.
  LOOM_LOW_C_ABI_LEAF_FLAG_DIRECT_IDENTITY = 1u << 3,
};

// Exact descriptor operation selected for a terminal source leaf.
typedef struct loom_low_c_abi_leaf_projection_t {
  // Descriptor loading this leaf from an address.
  uint32_t load_descriptor;
  // Descriptor storing this leaf to an address.
  uint32_t store_descriptor;
  // Descriptor moving the logical value into its integer bit carrier.
  uint32_t pack_move_descriptor;
  // Descriptor moving integer bits into the logical value carrier.
  uint32_t unpack_move_descriptor;
  // Physical carrier receiving this leaf, or LOOM_LOW_C_ABI_INDEX_NONE.
  uint16_t physical_index;
  // loom_low_c_abi_leaf_flag_bits_e bitset.
  loom_low_c_abi_leaf_flags_t flags;
  // Reserved for compact recipe flags.
  uint8_t reserved;
} loom_low_c_abi_leaf_projection_t;
static_assert(sizeof(loom_low_c_abi_leaf_projection_t) == 20,
              "C ABI leaf projections must be exactly 20 bytes");

// Transport recipe for one root source argument or result object.
typedef struct loom_low_c_abi_object_projection_t {
  // Private byte image required by fallback pack/unpack projection.
  uint32_t storage_byte_length;
  // Required alignment of the private byte image.
  uint32_t storage_byte_alignment;
  // First physical carrier in the corresponding signature array.
  uint16_t physical_offset;
  // Number of contiguous physical carriers transporting this object.
  uint16_t physical_count;
  // loom_low_c_abi_object_transport_t.
  loom_low_c_abi_object_transport_t transport;
  // True when every leaf maps directly into one physical carrier.
  bool has_direct_projection;
  // Reserved for compact object recipe flags.
  uint16_t reserved;
} loom_low_c_abi_object_projection_t;
static_assert(sizeof(loom_low_c_abi_object_projection_t) == 16,
              "C ABI object projections must be exactly 16 bytes");

// Byte placement and exact conversion operations for one physical carrier.
typedef struct loom_low_c_abi_carrier_projection_t {
  // Byte offset from the beginning of the source object.
  uint32_t byte_offset;
  // Descriptor moving packed integer bits into the physical carrier.
  uint32_t pack_move_descriptor;
  // Descriptor moving the physical carrier into integer bits.
  uint32_t unpack_move_descriptor;
  // Number of source leaves assigned to this carrier.
  uint16_t leaf_count;
  // Reserved for compact carrier recipe flags.
  uint16_t reserved;
} loom_low_c_abi_carrier_projection_t;
static_assert(sizeof(loom_low_c_abi_carrier_projection_t) == 16,
              "C ABI carrier projections must be exactly 16 bytes");

// One target descriptor converting between register classes. Rows are searched
// only while finalizing the retained projection plan; emission retains the
// selected descriptor ordinal directly.
typedef struct loom_low_c_abi_move_descriptor_t {
  // Descriptor-local source register class.
  uint16_t source_register_class;
  // Descriptor-local result register class.
  uint16_t result_register_class;
  // Descriptor ordinal implementing the bit-preserving move.
  uint32_t descriptor;
} loom_low_c_abi_move_descriptor_t;

// Address load/store descriptors for one native vector register class.
typedef struct loom_low_c_abi_vector_memory_descriptor_t {
  // Descriptor-local vector register class.
  uint16_t register_class;
  // Full physical register width used by private spill storage.
  uint16_t register_byte_length;
  // Descriptor ordinal loading the complete register.
  uint32_t load_descriptor;
  // Descriptor ordinal storing the complete register.
  uint32_t store_descriptor;
} loom_low_c_abi_vector_memory_descriptor_t;

// Fixed target operations copied into a finalized projection plan.
typedef struct loom_low_c_abi_projection_ops_t {
  // Stable descriptor-set identity shared by every projection register class.
  uint64_t register_descriptor_set_stable_id;
  // Integer register class carrying scalar values up to 32 bits.
  uint16_t bits32_register_class;
  // Integer register class carrying scalar values up to 64 bits.
  uint16_t bits64_register_class;
  // Native register class used by private storage and source pointers.
  uint16_t address_register_class;
  // Native address width in bytes; the current projection family requires 8.
  uint8_t address_byte_length;
  // loom_low_c_abi_byte_order_t.
  loom_low_c_abi_byte_order_t byte_order;
  // Unsigned byte load into bits32_register_class.
  uint32_t load_u8;
  // Unsigned halfword load into bits32_register_class.
  uint32_t load_u16;
  // Word load into bits32_register_class.
  uint32_t load_u32;
  // Doubleword load into bits64_register_class.
  uint32_t load_u64;
  // Byte store from bits32_register_class.
  uint32_t store_u8;
  // Halfword store from bits32_register_class.
  uint32_t store_u16;
  // Word store from bits32_register_class.
  uint32_t store_u32;
  // Doubleword store from bits64_register_class.
  uint32_t store_u64;
  // Address addition over address_register_class values.
  uint32_t add_address;
  // Immediate mask over bits32_register_class values.
  uint32_t mask_u32;
  // Zero extension from bits32_register_class to bits64_register_class.
  uint32_t widen_u32;
  // Truncation from bits64_register_class to bits32_register_class.
  uint32_t truncate_u64;
  // Immediate left shift over bits64_register_class values.
  uint32_t shift_left_u64;
  // Immediate logical right shift over bits64_register_class values.
  uint32_t shift_right_u64;
  // Bitwise union over bits64_register_class values.
  uint32_t or_u64;
  // Immediate materialization into bits64_register_class.
  uint32_t constant_u64;
  // Null-terminated load/store displacement immediate name.
  const char* memory_offset_name;
  // Null-terminated mask immediate name.
  const char* mask_name;
  // Null-terminated shift immediate name.
  const char* shift_name;
  // Null-terminated constant immediate name. Target literals must remain valid
  // for the plan lifetime.
  const char* constant_name;
} loom_low_c_abi_projection_ops_t;

// Target vocabulary used only while finalizing exact projection recipes.
typedef struct loom_low_c_abi_projection_target_t {
  // Fixed register types and operations available to all recipes.
  loom_low_c_abi_projection_ops_t ops;
  // Register-class moves searched while finalizing leaf and carrier recipes.
  const loom_low_c_abi_move_descriptor_t* moves;
  // Number of entries in moves.
  uint16_t move_count;
  // Native vector memory operations searched while finalizing leaf recipes.
  const loom_low_c_abi_vector_memory_descriptor_t* vector_memory;
  // Number of entries in vector_memory.
  uint16_t vector_memory_count;
} loom_low_c_abi_projection_target_t;

// Immutable retained C object projection. Target planners populate carrier
// placement through the functions below and then finalize descriptor recipes.
// Emitters only read this structure linearly.
typedef struct loom_low_c_abi_projection_plan_t {
  // Normalized source object and terminal leaf layout.
  const loom_c_abi_signature_layout_t* layout;
  // Flat logical argument types indexed by argument leaves.
  const loom_type_t* logical_argument_types;
  // Flat logical result types indexed by result leaves.
  const loom_type_t* logical_result_types;
  // Physical argument types indexed by argument carriers.
  const loom_type_t* physical_argument_types;
  // Physical result types indexed by result carriers.
  const loom_type_t* physical_result_types;
  // Argument object recipes in source signature order.
  loom_low_c_abi_object_projection_t* arguments;
  // Result object recipes in source signature order.
  loom_low_c_abi_object_projection_t* results;
  // Argument leaf recipes in normalized layout order.
  loom_low_c_abi_leaf_projection_t* argument_leaves;
  // Result leaf recipes in normalized layout order.
  loom_low_c_abi_leaf_projection_t* result_leaves;
  // Argument carrier recipes in physical signature order.
  loom_low_c_abi_carrier_projection_t* argument_carriers;
  // Result carrier recipes in physical signature order.
  loom_low_c_abi_carrier_projection_t* result_carriers;
  // Number of source argument objects.
  uint16_t argument_object_count;
  // Number of source result objects.
  uint16_t result_object_count;
  // Number of populated argument carrier recipes.
  uint16_t argument_carrier_count;
  // Number of populated result carrier recipes.
  uint16_t result_carrier_count;
  // Allocated argument carrier recipe capacity.
  uint16_t argument_carrier_capacity;
  // Allocated result carrier recipe capacity.
  uint16_t result_carrier_capacity;
  // Result object transported through caller-owned storage, if any.
  uint16_t indirect_result_object;
  // Physical argument carrying the indirect result address.
  uint16_t indirect_result_argument;
  // Optional physical result echoing the indirect result address.
  uint16_t indirect_result_echo;
  // Fixed operations copied from the selected target vocabulary.
  loom_low_c_abi_projection_ops_t ops;
} loom_low_c_abi_projection_plan_t;

// Allocates a projection plan and initializes source object/leaf recipes from a
// normalized signature layout. Accepted root byte lengths and alignments must
// fit uint32_t; the platform classifier owns that support decision.
iree_status_t loom_low_c_abi_projection_plan_allocate(
    const loom_c_abi_signature_layout_t* layout,
    const loom_type_t* logical_argument_types, uint16_t logical_argument_count,
    const loom_type_t* logical_result_types, uint16_t logical_result_count,
    uint16_t argument_carrier_capacity, uint16_t result_carrier_capacity,
    iree_arena_allocator_t* arena, loom_low_c_abi_projection_plan_t** out_plan);

// Appends one physical carrier and optionally assigns it to a source argument
// object. Assigned carriers for each object must be contiguous.
uint16_t loom_low_c_abi_projection_append_argument_carrier(
    loom_low_c_abi_projection_plan_t* plan, uint16_t object_index,
    uint32_t byte_offset, uint32_t storage_byte_length);

// Appends one physical carrier and optionally assigns it to a source result
// object. Assigned carriers for each object must be contiguous.
uint16_t loom_low_c_abi_projection_append_result_carrier(
    loom_low_c_abi_projection_plan_t* plan, uint16_t object_index,
    uint32_t byte_offset, uint32_t storage_byte_length);

// Marks one source argument as transported through its single address carrier.
void loom_low_c_abi_projection_set_argument_address(
    loom_low_c_abi_projection_plan_t* plan, uint16_t object_index);

// Marks one source result as transported through caller-owned storage. The
// address argument is required; |echo_result| may be
// LOOM_LOW_C_ABI_INDEX_NONE when the platform returns no address echo.
void loom_low_c_abi_projection_set_indirect_result(
    loom_low_c_abi_projection_plan_t* plan, uint16_t object_index,
    uint16_t address_argument, uint16_t echo_result);

// Resolves exact leaf/carrier descriptor operations and direct projections.
// Returns false with a static reason when the selected target vocabulary cannot
// express a source-dependent projection. No allocation or IR mutation occurs.
bool loom_low_c_abi_projection_plan_finalize(
    loom_low_c_abi_projection_plan_t* plan,
    const loom_low_c_abi_projection_target_t* target,
    const loom_type_t* physical_argument_types,
    uint16_t physical_argument_count, const loom_type_t* physical_result_types,
    uint16_t physical_result_count, iree_string_view_t* out_reason);

// Emits all definition/call pack and unpack phases from a finalized plan.
iree_status_t loom_low_c_abi_projection_emit_source_signature(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_source_signature_emission_phase_t phase,
    const loom_op_t* source_op, const void* target_plan,
    const loom_value_id_t* input_values, uint16_t input_count,
    loom_value_id_t* output_values, uint16_t output_count,
    iree_arena_allocator_t* state_arena, void** inout_state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_ABI_C_PROJECTION_H_
