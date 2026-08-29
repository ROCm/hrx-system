// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Table-driven AIE2P physical-register and machine-form model.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_MACHINE_MACHINE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_MACHINE_MACHINE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dense build-local physical-register identifier aligned with Low descriptor
// physical-register row ordinals.
typedef uint16_t loom_aie2p_physical_register_id_t;
#define LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID \
  ((loom_aie2p_physical_register_id_t)0xFFFFu)

// Dense build-local register-class identifier. Class zero is reserved.
typedef uint16_t loom_aie2p_register_class_id_t;
#define LOOM_AIE2P_REGISTER_CLASS_ID_INVALID ((loom_aie2p_register_class_id_t)0)

// Dense build-local operand-adapter identifier. Adapter zero means direct
// physical-register hardware encoding.
typedef uint16_t loom_aie2p_register_adapter_id_t;
#define LOOM_AIE2P_REGISTER_ADAPTER_ID_DIRECT \
  ((loom_aie2p_register_adapter_id_t)0)
// Name lookup failure; never a generated adapter ID.
#define LOOM_AIE2P_REGISTER_ADAPTER_ID_INVALID \
  ((loom_aie2p_register_adapter_id_t)0xFFFFu)

// Dense build-local immediate-domain identifier. Immediate zero is reserved.
typedef uint16_t loom_aie2p_immediate_id_t;
#define LOOM_AIE2P_IMMEDIATE_ID_INVALID ((loom_aie2p_immediate_id_t)0)

// Dense build-local machine-form identifier. Form zero is reserved. Generated
// tables keep these IDs aligned with AIE2P instruction encoding IDs.
typedef uint16_t loom_aie2p_machine_form_id_t;
#define LOOM_AIE2P_MACHINE_FORM_ID_INVALID ((loom_aie2p_machine_form_id_t)0)

// Atomic register-unit identifiers are zero-based because every value denotes
// storage. The all-ones value is reserved for failed queries.
typedef uint16_t loom_aie2p_atomic_unit_id_t;
#define LOOM_AIE2P_ATOMIC_UNIT_ID_INVALID ((loom_aie2p_atomic_unit_id_t)0xFFFFu)

enum loom_aie2p_register_class_flag_bits_e {
  // Class participates in physical-register allocation.
  LOOM_AIE2P_REGISTER_CLASS_FLAG_ALLOCATABLE = 1u << 0,
  // Class contributes to pre-allocation scheduling pressure.
  LOOM_AIE2P_REGISTER_CLASS_FLAG_PRE_RA_SCHEDULING = 1u << 1,
  // Class owns a generated pressure set.
  LOOM_AIE2P_REGISTER_CLASS_FLAG_PRESSURE_SET = 1u << 2,
};
typedef uint8_t loom_aie2p_register_class_flags_t;

enum loom_aie2p_immediate_flag_bits_e {
  // Immediate domain uses signed interpretation.
  LOOM_AIE2P_IMMEDIATE_FLAG_SIGNED = 1u << 0,
  // Immediate domain contains only negative values.
  LOOM_AIE2P_IMMEDIATE_FLAG_NEGATIVE = 1u << 1,
  // Immediate accepts a symbolic relocation reference.
  LOOM_AIE2P_IMMEDIATE_FLAG_SYMBOL_REFERENCE = 1u << 2,
};
typedef uint8_t loom_aie2p_immediate_flags_t;

enum loom_aie2p_machine_property_flag_bits_e {
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_HAS_DELAY_SLOT = 1u << 0,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_EXTRA_DEF_REGISTER = 1u << 1,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_EXTRA_USE_REGISTER = 1u << 2,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_SIDE_EFFECTS = 1u << 3,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_BARRIER = 1u << 4,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_BRANCH = 1u << 5,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_CALL = 1u << 6,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_CODEGEN_ONLY = 1u << 7,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_COMMUTABLE = 1u << 8,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_MOVE_IMMEDIATE = 1u << 9,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_PSEUDO = 1u << 10,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_REMATERIALIZABLE = 1u << 11,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_RETURN = 1u << 12,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_TERMINATOR = 1u << 13,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_MAY_LOAD = 1u << 14,
  LOOM_AIE2P_MACHINE_PROPERTY_FLAG_MAY_STORE = 1u << 15,
};
typedef uint16_t loom_aie2p_machine_property_flags_t;

typedef enum loom_aie2p_machine_operand_kind_e {
  LOOM_AIE2P_MACHINE_OPERAND_KIND_REGISTER_CLASS = 0,
  LOOM_AIE2P_MACHINE_OPERAND_KIND_REGISTER_ADAPTER = 1,
  LOOM_AIE2P_MACHINE_OPERAND_KIND_IMMEDIATE = 2,
} loom_aie2p_machine_operand_kind_t;

typedef enum loom_aie2p_control_flow_kind_e {
  LOOM_AIE2P_CONTROL_FLOW_NONE = 0,
  LOOM_AIE2P_CONTROL_FLOW_BRANCH_CONDITIONAL_DECREMENT = 1,
  LOOM_AIE2P_CONTROL_FLOW_BRANCH_CONDITIONAL_NONZERO = 2,
  LOOM_AIE2P_CONTROL_FLOW_BRANCH_CONDITIONAL_ZERO = 3,
  LOOM_AIE2P_CONTROL_FLOW_BRANCH_DIRECT = 4,
  LOOM_AIE2P_CONTROL_FLOW_BRANCH_INDIRECT = 5,
  LOOM_AIE2P_CONTROL_FLOW_CALL_DIRECT = 6,
  LOOM_AIE2P_CONTROL_FLOW_CALL_INDIRECT = 7,
  LOOM_AIE2P_CONTROL_FLOW_RETURN = 8,
} loom_aie2p_control_flow_kind_t;

// Immutable metadata for one physical register.
typedef struct loom_aie2p_physical_register_info_t {
  // Stable target register name.
  iree_string_view_t name;
  // Native assembly spelling.
  iree_string_view_t assembly_name;
  // Direct hardware encoding used by raw register-class operands.
  uint16_t hardware_encoding;
  // Number of atomic storage units occupied by the register.
  uint8_t atomic_unit_count;
  // Number of named subregisters covering the register.
  uint8_t subregister_count;
} loom_aie2p_physical_register_info_t;

// One named subregister relation.
typedef struct loom_aie2p_subregister_info_t {
  // Physical register selected as the subregister.
  loom_aie2p_physical_register_id_t register_id;
  // Stable target subregister-index name.
  iree_string_view_t index_name;
} loom_aie2p_subregister_info_t;

// Immutable metadata for one ordered physical-register class.
typedef struct loom_aie2p_register_class_info_t {
  // Stable target register-class name.
  iree_string_view_t name;
  // In-register value width in bits.
  uint16_t register_size_bits;
  // Required register alignment in bits.
  uint16_t alignment_bits;
  // Spill-slot width in bits.
  uint16_t spill_size_bits;
  // Spill-slot alignment in bits.
  uint16_t spill_alignment_bits;
  // Register-class allocation policy.
  loom_aie2p_register_class_flags_t flags;
  // Number of ordered physical-register candidates.
  uint8_t candidate_count;
  // Number of accepted machine value types.
  uint8_t value_type_count;
} loom_aie2p_register_class_info_t;

// Immutable metadata for one operand-local register encoding adapter.
typedef struct loom_aie2p_register_adapter_info_t {
  // Stable target adapter name.
  iree_string_view_t name;
  // Register class whose complete candidate set is mapped.
  loom_aie2p_register_class_id_t register_class_id;
} loom_aie2p_register_adapter_info_t;

// Immutable metadata for one scaled immediate domain.
typedef struct loom_aie2p_immediate_info_t {
  // Stable target immediate name.
  iree_string_view_t name;
  // Semantic width including fixed low-zero and implicit sign bits.
  uint8_t semantic_width_bits;
  // Width physically carried by the instruction field.
  uint8_t encoded_width_bits;
  // Required semantic value granularity.
  uint32_t step;
  // Signed, negative-only, and symbol-reference behavior.
  loom_aie2p_immediate_flags_t flags;
} loom_aie2p_immediate_info_t;

// Immutable metadata for one concrete physical instruction form.
typedef struct loom_aie2p_machine_form_info_t {
  // Stable target instruction name.
  iree_string_view_t name;
  // Native assembly template retained for diagnostics and printing.
  iree_string_view_t assembly;
  // Machine selection and side-effect properties.
  loom_aie2p_machine_property_flags_t property_flags;
  // Normalized control-flow behavior.
  loom_aie2p_control_flow_kind_t control_flow_kind;
  // Number of explicit definition operands.
  uint8_t output_count;
  // Number of explicit use operands.
  uint8_t input_count;
  // Number of implicit physical-register definitions.
  uint8_t implicit_def_count;
  // Number of implicit physical-register uses.
  uint8_t implicit_use_count;
  // Number of definition/use equality constraints.
  uint8_t tie_count;
} loom_aie2p_machine_form_info_t;

// One explicit machine-form operand.
typedef struct loom_aie2p_machine_operand_info_t {
  // Stable operand field name.
  iree_string_view_t name;
  // Register-class, adapter, or immediate identifier selected by |kind|.
  uint16_t type_id;
  // Domain interpreting |type_id|.
  loom_aie2p_machine_operand_kind_t kind;
} loom_aie2p_machine_operand_info_t;

// One tied explicit definition/use pair, addressed within their respective
// output and input lists.
typedef struct loom_aie2p_machine_tie_info_t {
  // Definition operand ordinal.
  uint8_t definition_ordinal;
  // Use operand ordinal.
  uint8_t use_ordinal;
} loom_aie2p_machine_tie_info_t;

iree_host_size_t loom_aie2p_machine_atomic_unit_count(void);
iree_host_size_t loom_aie2p_machine_physical_register_count(void);
iree_host_size_t loom_aie2p_machine_register_class_count(void);
iree_host_size_t loom_aie2p_machine_register_adapter_count(void);
iree_host_size_t loom_aie2p_machine_immediate_count(void);
iree_host_size_t loom_aie2p_machine_form_count(void);

iree_string_view_t loom_aie2p_machine_atomic_unit_name(
    loom_aie2p_atomic_unit_id_t atomic_unit);

loom_aie2p_physical_register_id_t loom_aie2p_machine_find_physical_register(
    iree_string_view_t name);
loom_aie2p_register_class_id_t loom_aie2p_machine_find_register_class(
    iree_string_view_t name);
loom_aie2p_register_adapter_id_t loom_aie2p_machine_find_register_adapter(
    iree_string_view_t name);
loom_aie2p_immediate_id_t loom_aie2p_machine_find_immediate(
    iree_string_view_t name);
loom_aie2p_machine_form_id_t loom_aie2p_machine_find_form(
    iree_string_view_t name);

bool loom_aie2p_machine_query_physical_register(
    loom_aie2p_physical_register_id_t register_id,
    loom_aie2p_physical_register_info_t* out_info);
loom_aie2p_atomic_unit_id_t loom_aie2p_machine_physical_register_atomic_unit(
    loom_aie2p_physical_register_id_t register_id, uint8_t ordinal);
bool loom_aie2p_machine_query_subregister(
    loom_aie2p_physical_register_id_t register_id, uint8_t ordinal,
    loom_aie2p_subregister_info_t* out_info);

bool loom_aie2p_machine_query_register_class(
    loom_aie2p_register_class_id_t register_class_id,
    loom_aie2p_register_class_info_t* out_info);
loom_aie2p_physical_register_id_t loom_aie2p_machine_register_class_candidate(
    loom_aie2p_register_class_id_t register_class_id, uint8_t ordinal);
iree_string_view_t loom_aie2p_machine_register_class_value_type(
    loom_aie2p_register_class_id_t register_class_id, uint8_t ordinal);

bool loom_aie2p_machine_query_register_adapter(
    loom_aie2p_register_adapter_id_t adapter_id,
    loom_aie2p_register_adapter_info_t* out_info);
// Returns the architectural operand-encoder value for a physical register.
// An instruction field consumes only the low bits declared by that form.
bool loom_aie2p_machine_encode_register(
    loom_aie2p_register_adapter_id_t adapter_id,
    loom_aie2p_physical_register_id_t register_id, uint8_t* out_value);

// Adapts one allocator-produced physical register to its operand-encoder value.
// Generation proves that |adapter_id| covers the operand register class and
// allocation proves that |register_id| is a candidate in that class. Final
// instruction packing consumes the field's declared low bits.
uint8_t loom_aie2p_machine_adapt_allocated_register(
    loom_aie2p_register_adapter_id_t adapter_id,
    loom_aie2p_physical_register_id_t register_id);

bool loom_aie2p_machine_query_immediate(loom_aie2p_immediate_id_t immediate_id,
                                        loom_aie2p_immediate_info_t* out_info);
iree_status_t loom_aie2p_machine_encode_immediate(
    loom_aie2p_immediate_id_t immediate_id, int64_t value,
    uint64_t* out_encoded_value);

// Encodes an immediate already verified against its Low descriptor range and
// value step. The descriptor generator keeps |immediate_id| aligned with the
// machine table.
uint64_t loom_aie2p_machine_encode_verified_immediate(
    loom_aie2p_immediate_id_t immediate_id, int64_t value);
iree_status_t loom_aie2p_machine_decode_immediate(
    loom_aie2p_immediate_id_t immediate_id, uint64_t encoded_value,
    int64_t* out_value);

bool loom_aie2p_machine_query_form(loom_aie2p_machine_form_id_t form_id,
                                   loom_aie2p_machine_form_info_t* out_info);
bool loom_aie2p_machine_query_form_operand(
    loom_aie2p_machine_form_id_t form_id, uint8_t ordinal,
    loom_aie2p_machine_operand_info_t* out_info);
loom_aie2p_physical_register_id_t loom_aie2p_machine_form_implicit_def(
    loom_aie2p_machine_form_id_t form_id, uint8_t ordinal);
loom_aie2p_physical_register_id_t loom_aie2p_machine_form_implicit_use(
    loom_aie2p_machine_form_id_t form_id, uint8_t ordinal);
bool loom_aie2p_machine_query_form_tie(loom_aie2p_machine_form_id_t form_id,
                                       uint8_t ordinal,
                                       loom_aie2p_machine_tie_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_MACHINE_MACHINE_H_
