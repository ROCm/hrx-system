// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Text low assembly interface.
//
// The text parser owns the surface syntax for low asm regions, but target
// descriptor tables and canonical low operation builders live outside the text
// package. This interface keeps that boundary explicit: parser/printer code
// receives packet shapes, build hooks, and print hooks from an environment
// supplied by the tool or compiler pipeline.

#ifndef LOOM_FORMAT_TEXT_LOW_ASM_H_
#define LOOM_FORMAT_TEXT_LOW_ASM_H_

#include "iree/base/api.h"
#include "loom/error/error_defs.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_text_low_asm_environment_state_t
    loom_text_low_asm_environment_state_t;
typedef struct loom_text_low_asm_descriptor_set_t
    loom_text_low_asm_descriptor_set_t;
typedef struct loom_text_low_asm_form_t loom_text_low_asm_form_t;
typedef struct loom_text_low_asm_descriptor_handle_t
    loom_text_low_asm_descriptor_handle_t;

typedef struct loom_text_low_asm_packet_descriptor_t {
  // Opaque descriptor-set handle owned by the environment implementation.
  const loom_text_low_asm_descriptor_set_t* descriptor_set;
  // Opaque asm-form handle owned by the environment implementation.
  const loom_text_low_asm_form_t* form;
  // Opaque low descriptor handle owned by the environment implementation.
  const loom_text_low_asm_descriptor_handle_t* descriptor;
  // Stable canonical opcode key to store on the emitted low operation.
  iree_string_view_t opcode_key;
  // Surface mnemonic emitted or parsed for this asm packet.
  iree_string_view_t mnemonic;
  // Number of SSA results produced by this asm packet.
  uint16_t result_count;
  // Number of SSA operands consumed by this asm packet.
  uint16_t operand_count;
  // Number of immediate attributes addressed by the compact asm form.
  uint16_t asm_immediate_count;
  // Number of descriptor immediate attributes addressable by this asm packet.
  uint16_t immediate_count;
  // Operation attribute field index storing packet immediate attributes.
  uint16_t immediate_attribute_field_index;
  // True when at least one immediate requires named-dictionary syntax.
  bool has_named_immediates;
  // True when this packet should canonicalize to a low const operation.
  bool builds_as_const;
} loom_text_low_asm_packet_descriptor_t;

typedef struct loom_text_low_asm_immediate_descriptor_t {
  // Canonical low attribute field name stored in the emitted operation.
  iree_string_view_t field_name;
  // Surface spelling accepted by the asm parser for this immediate.
  iree_string_view_t spelling;
  // True when the packet may omit this immediate attribute.
  bool has_default_value;
  // Effective i64 value used when the packet omits this immediate.
  int64_t default_value;
} loom_text_low_asm_immediate_descriptor_t;

enum {
  LOOM_TEXT_LOW_ASM_DIAGNOSTIC_PARAM_CAPACITY = 12,
};

typedef struct loom_text_low_asm_diagnostic_t {
  // Structured diagnostic to emit, or NULL when no target-owned diagnostic
  // matches.
  const loom_error_def_t* error;
  // Inline diagnostic parameters owned by this result object.
  loom_diagnostic_param_t params[LOOM_TEXT_LOW_ASM_DIAGNOSTIC_PARAM_CAPACITY];
  // Number of populated entries in |params|.
  iree_host_size_t param_count;
} loom_text_low_asm_diagnostic_t;

typedef enum loom_text_low_asm_statement_kind_e {
  // Unknown or uninitialized statement kind. Printers may use this to report
  // that a canonical operation has no lossless low asm spelling.
  LOOM_TEXT_LOW_ASM_STATEMENT_UNKNOWN = 0,
  // Descriptor-backed packet printed as an instruction mnemonic.
  LOOM_TEXT_LOW_ASM_STATEMENT_PACKET = 1,
  // Low return packet printed as `return`.
  LOOM_TEXT_LOW_ASM_STATEMENT_RETURN = 2,
  // Target-low structural intrinsic printed in asm syntax.
  LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL = 3,
} loom_text_low_asm_statement_kind_t;

typedef enum loom_text_low_asm_structural_kind_e {
  // Unknown or uninitialized structural kind.
  LOOM_TEXT_LOW_ASM_STRUCTURAL_UNKNOWN = 0,
  // Function-local target resource import.
  LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE = 1,
  // Target ABI live-in register import.
  LOOM_TEXT_LOW_ASM_STRUCTURAL_LIVE_IN = 2,
  // Register range concatenation.
  LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT = 3,
  // Register range slice projection.
  LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE = 4,
  // Function-local storage reservation.
  LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_RESERVE = 5,
  // Function-local storage address materialization.
  LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_ADDRESS = 6,
  // Explicit virtual-register copy/coalescing boundary.
  LOOM_TEXT_LOW_ASM_STRUCTURAL_COPY = 7,
  // Function-local storage subspan projection.
  LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_VIEW = 8,
} loom_text_low_asm_structural_kind_t;

typedef struct loom_text_low_asm_structural_attribute_t {
  // Surface attribute name to print in the structural intrinsic dictionary.
  iree_string_view_t name;
  // Attribute payload owned by the canonical operation.
  const loom_attribute_t* value;
  // Optional descriptor used to print enum attributes by spelling.
  const loom_attr_descriptor_t* descriptor;
} loom_text_low_asm_structural_attribute_t;

typedef struct loom_text_low_asm_statement_t {
  // Statement kind describing which fields below are meaningful.
  loom_text_low_asm_statement_kind_t kind;
  // Original canonical operation represented by this asm statement.
  const loom_op_t* op;
  // Structural intrinsic kind when |kind| is STRUCTURAL.
  loom_text_low_asm_structural_kind_t structural_kind;
  // Structural intrinsic key printed in angle brackets, if any.
  iree_string_view_t structural_key;
  // Static slice offset when |structural_kind| is SLICE.
  int64_t structural_offset;
  // Packet descriptor for descriptor-backed packet statements.
  loom_text_low_asm_packet_descriptor_t packet;
  // SSA results defined by a packet statement.
  const loom_value_id_t* results;
  // Number of SSA results in |results|.
  uint16_t result_count;
  // SSA operands consumed by a packet statement, or return values for returns.
  const loom_value_id_t* operands;
  // Number of SSA values in |operands|.
  uint16_t operand_count;
  // Canonical immediate attributes stored on the packet operation.
  loom_named_attr_slice_t attributes;
  // Structural attributes with direct descriptors for typed printing.
  loom_text_low_asm_structural_attribute_t structural_attributes[8];
  // Number of entries populated in |structural_attributes|.
  uint8_t structural_attribute_count;
  // True when immediate attributes correspond to a concrete op attr field.
  bool has_immediate_attribute_field;
  // Operation attribute field index storing packet immediate attributes.
  uint16_t immediate_attribute_field_index;
  // Source location attached to the represented operation.
  loom_location_id_t location;
} loom_text_low_asm_statement_t;

// Contract for descriptor-backed statement descriptions.
//
// A describe callback is called after the canonical IR has been verified by the
// owning low dialect/descriptor registry. It either returns UNKNOWN for a valid
// operation that has no lossless low asm spelling, or returns a fully
// well-formed statement:
// - PACKET result/operand counts match |packet| and all result/operand IDs are
//   valid in the module value table.
// - PACKET immediate attributes are canonical for |packet|; missing optional
//   immediates are represented by descriptor defaults.
// - RETURN operands and STRUCTURAL results/operands are valid module values.
// - STRUCTURAL statements carry the result/operand/attribute shape required by
//   their structural kind.
//
// The text printer relies on this contract and only performs formatting and
// lossless-spelling availability checks. Semantic validation belongs in the
// verifier and descriptor-backed describe implementation, not in the printer.

// Resolves an `asm<...>` descriptor-set key to an environment-owned handle.
// Returns OK with NULL when no descriptor set matches.
typedef iree_status_t (*loom_text_low_asm_lookup_descriptor_set_fn_t)(
    const loom_text_low_asm_environment_state_t* state, iree_string_view_t key,
    const loom_text_low_asm_descriptor_set_t** out_descriptor_set);

// Resolves a target symbol reference to the descriptor set selected by that
// target. Returns OK with NULL when the target cannot select a descriptor set.
typedef iree_status_t (*loom_text_low_asm_lookup_target_descriptor_set_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_module_t* module, loom_attribute_t target_attr,
    const loom_text_low_asm_descriptor_set_t** out_descriptor_set);

// Resolves a mnemonic within a descriptor set to a packet descriptor. Returns
// OK with |out_packet->descriptor| NULL when no packet matches.
typedef iree_status_t (*loom_text_low_asm_lookup_packet_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    iree_string_view_t mnemonic,
    loom_text_low_asm_packet_descriptor_t* out_packet);

// Attempts to explain an otherwise unknown mnemonic with a target-owned
// structured diagnostic. Returns OK with |out_diagnostic->error| NULL when no
// diagnostic matches.
typedef iree_status_t (*loom_text_low_asm_diagnose_unknown_mnemonic_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    iree_string_view_t mnemonic,
    loom_text_low_asm_diagnostic_t* out_diagnostic);

typedef iree_status_t (*loom_text_low_asm_infer_result_type_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_module_t* module, loom_type_t* out_type,
    iree_string_view_t* out_diagnostic_detail);

typedef iree_status_t (*loom_text_low_asm_validate_result_type_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_module_t* module, loom_type_t type,
    iree_string_view_t* out_diagnostic_detail);

typedef iree_status_t (*loom_text_low_asm_result_type_annotation_required_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, const loom_module_t* module, loom_type_t type,
    bool* out_required, iree_string_view_t* out_diagnostic_detail);

typedef iree_status_t (*loom_text_low_asm_immediate_descriptor_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t immediate_index,
    loom_text_low_asm_immediate_descriptor_t* out_immediate);

typedef iree_status_t (*loom_text_low_asm_build_packet_fn_t)(
    const loom_text_low_asm_environment_state_t* state, loom_builder_t* builder,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attributes, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_location_id_t location,
    loom_op_t** out_op);

typedef iree_status_t (*loom_text_low_asm_build_return_fn_t)(
    const loom_text_low_asm_environment_state_t* state, loom_builder_t* builder,
    const loom_value_id_t* values, iree_host_size_t value_count,
    loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_text_low_asm_structural_attr_descriptor_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    loom_text_low_asm_structural_kind_t kind, iree_string_view_t attr_name,
    const loom_attr_descriptor_t** out_descriptor);

typedef iree_status_t (*loom_text_low_asm_build_structural_fn_t)(
    const loom_text_low_asm_environment_state_t* state, loom_builder_t* builder,
    loom_text_low_asm_structural_kind_t kind, iree_string_view_t key,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attributes, int64_t offset, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_text_low_asm_describe_operation_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement);

// Resolves a textual register class in |descriptor_set| to a compact target-low
// register type. Returns OK with |out_found| false when no class matches.
typedef iree_status_t (*loom_text_low_asm_resolve_register_type_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    iree_string_view_t register_class_name, uint32_t unit_count,
    loom_type_t* out_type, bool* out_found);

// Resolves the descriptor set that owns a compact target-low register type.
// Returns OK with |out_descriptor_set| NULL when |type| is not a known register
// type in this environment.
typedef iree_status_t (*loom_text_low_asm_lookup_register_descriptor_set_fn_t)(
    const loom_text_low_asm_environment_state_t* state, loom_type_t type,
    const loom_text_low_asm_descriptor_set_t** out_descriptor_set);

// Resolves a compact target-low register type to its textual class name in
// |descriptor_set|. Returns OK with |out_found| false when the type does not
// belong to the descriptor set.
typedef iree_status_t (*loom_text_low_asm_describe_register_type_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set, loom_type_t type,
    iree_string_view_t* out_register_class_name, uint32_t* out_unit_count,
    bool* out_found);

typedef struct loom_text_low_asm_vtable_t {
  // Resolves an `asm<...>` descriptor-set key to an environment-owned handle.
  loom_text_low_asm_lookup_descriptor_set_fn_t lookup_descriptor_set;
  // Resolves a target symbol reference to its descriptor-set handle.
  loom_text_low_asm_lookup_target_descriptor_set_fn_t
      lookup_target_descriptor_set;
  // Resolves a mnemonic within a descriptor-set handle to a packet descriptor.
  loom_text_low_asm_lookup_packet_fn_t lookup_packet;
  // Optional target-owned explanation for unknown mnemonics.
  loom_text_low_asm_diagnose_unknown_mnemonic_fn_t diagnose_unknown_mnemonic;
  // Infers a result type when the asm packet omits explicit type annotations.
  loom_text_low_asm_infer_result_type_fn_t infer_result_type;
  // Validates an explicit asm result type annotation against the descriptor.
  loom_text_low_asm_validate_result_type_fn_t validate_result_type;
  // Returns whether printing must emit an explicit result type annotation.
  loom_text_low_asm_result_type_annotation_required_fn_t
      result_type_annotation_required;
  // Returns canonical field and surface spelling metadata for one immediate.
  loom_text_low_asm_immediate_descriptor_fn_t immediate_descriptor;
  // Builds the canonical low operation for a parsed non-return asm packet.
  loom_text_low_asm_build_packet_fn_t build_packet;
  // Builds the canonical low return operation for an asm `return` packet.
  loom_text_low_asm_build_return_fn_t build_return;
  // Looks up a typed structural intrinsic attribute descriptor by name.
  loom_text_low_asm_structural_attr_descriptor_fn_t structural_attr_descriptor;
  // Builds the canonical low operation for a parsed structural intrinsic.
  loom_text_low_asm_build_structural_fn_t build_structural;
  // Describes a canonical operation as a printable low asm statement. Returns
  // OK with UNKNOWN when the operation is valid but has no lossless asm form.
  // See the statement description contract above for the validity guarantees
  // required when a concrete statement kind is returned.
  loom_text_low_asm_describe_operation_fn_t describe_operation;
  // Resolves textual register classes while parsing target-low register types.
  loom_text_low_asm_resolve_register_type_fn_t resolve_register_type;
  // Resolves the descriptor-set handle owning a compact register type.
  loom_text_low_asm_lookup_register_descriptor_set_fn_t
      lookup_register_descriptor_set;
  // Resolves compact target-low register types while printing text.
  loom_text_low_asm_describe_register_type_fn_t describe_register_type;
} loom_text_low_asm_vtable_t;

typedef struct loom_text_low_asm_environment_t {
  // Function table implementing low asm lookup, type inference, and builders.
  const loom_text_low_asm_vtable_t* vtable;
  // Environment-owned state passed to every vtable callback.
  const loom_text_low_asm_environment_state_t* state;
} loom_text_low_asm_environment_t;

static inline bool loom_text_low_asm_environment_is_configured(
    const loom_text_low_asm_environment_t* environment) {
  return environment && environment->vtable && environment->state &&
         environment->vtable->lookup_descriptor_set &&
         environment->vtable->lookup_packet &&
         environment->vtable->resolve_register_type &&
         environment->vtable->infer_result_type &&
         environment->vtable->validate_result_type &&
         environment->vtable->immediate_descriptor &&
         environment->vtable->build_packet &&
         environment->vtable->build_return &&
         environment->vtable->structural_attr_descriptor &&
         environment->vtable->build_structural;
}

static inline bool loom_text_low_asm_environment_supports_printing(
    const loom_text_low_asm_environment_t* environment) {
  return environment && environment->vtable && environment->state &&
         environment->vtable->lookup_descriptor_set &&
         environment->vtable->result_type_annotation_required &&
         environment->vtable->immediate_descriptor &&
         environment->vtable->describe_operation &&
         environment->vtable->lookup_register_descriptor_set &&
         environment->vtable->describe_register_type;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_LOW_ASM_H_
