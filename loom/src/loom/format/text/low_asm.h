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
//
// Packet result annotations use the canonical result-list syntax after `:`:
// `reg<...>` defines an independent result and `%operand as reg<...>` defines
// an ownership tie. An explicit list supplies every result's type and the
// complete ownership tie set. Without a list, the descriptor supplies both
// types and default ties. Physical instruction constraints apply in either
// form; a physical register tie need not consume the source's IR ownership.

#ifndef LOOM_FORMAT_TEXT_LOW_ASM_H_
#define LOOM_FORMAT_TEXT_LOW_ASM_H_

#include "iree/base/api.h"
#include "loom/error/parameter.h"
#include "loom/format/low_repr.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_text_low_asm_environment_state_t
    loom_text_low_asm_environment_state_t;
typedef loom_low_repr_descriptor_set_t loom_text_low_asm_descriptor_set_t;
typedef struct loom_text_low_asm_form_t loom_text_low_asm_form_t;
typedef struct loom_text_low_asm_descriptor_handle_t
    loom_text_low_asm_descriptor_handle_t;

typedef enum loom_text_low_asm_operand_segment_delimiter_e {
  // Angle brackets: `<...>`.
  LOOM_TEXT_LOW_ASM_OPERAND_SEGMENT_DELIMITER_ANGLE = 1,
  // Square brackets: `[...]`.
  LOOM_TEXT_LOW_ASM_OPERAND_SEGMENT_DELIMITER_SQUARE = 2,
  // Parentheses: `(...)`.
  LOOM_TEXT_LOW_ASM_OPERAND_SEGMENT_DELIMITER_PAREN = 3,
} loom_text_low_asm_operand_segment_delimiter_t;

typedef struct loom_text_low_asm_operand_segment_descriptor_t {
  // Delimiter pair enclosing this operand segment.
  loom_text_low_asm_operand_segment_delimiter_t delimiter;
  // Number of fixed operands in this segment.
  uint16_t fixed_operand_count;
  // True when the segment consumes all remaining operands after its fixed
  // prefix. Only the final segment may be variadic.
  bool is_variadic;
} loom_text_low_asm_operand_segment_descriptor_t;

// Active target-low representation contract for contextual types and regions.
typedef struct loom_text_low_repr_context_t {
  // Canonical representation-contract key selected by the enclosing wrapper.
  iree_string_view_t contract_key;
  // Descriptor-set handle resolved once from |contract_key|.
  const loom_text_low_asm_descriptor_set_t* descriptor_set;
} loom_text_low_repr_context_t;

typedef struct loom_text_low_asm_packet_descriptor_t {
  // Opaque descriptor-set handle owned by the environment implementation.
  const loom_text_low_asm_descriptor_set_t* descriptor_set;
  // Opaque asm-form handle owned by the environment implementation.
  const loom_text_low_asm_form_t* form;
  // Opaque low descriptor handle owned by the environment implementation.
  const loom_text_low_asm_descriptor_handle_t* descriptor;
  // Stable canonical descriptor key used only in source diagnostics.
  iree_string_view_t descriptor_key;
  // Surface mnemonic emitted or parsed for this asm packet.
  iree_string_view_t mnemonic;
  // Number of SSA results produced by this asm packet.
  uint16_t result_count;
  // Minimum number of SSA operands consumed by this asm packet.
  uint16_t minimum_operand_count;
  // Number of delimited operand segments, or zero for the flat legacy form.
  uint16_t operand_segment_count;
  // True when the final operand segment accepts a variadic suffix.
  bool has_variadic_operands;
  // Number of immediate attributes addressed by the compact asm form.
  uint16_t asm_immediate_count;
  // Number of descriptor immediate attributes addressable by this asm packet.
  uint16_t immediate_count;
  // Operation attribute field index storing packet immediate attributes.
  uint16_t immediate_attribute_field_index;
  // True when at least one immediate requires named-dictionary syntax.
  bool has_named_immediates;
  // Whether the named dictionary is required after any positional prefix.
  bool requires_named_immediates;
  // Canonical operation kind supplying the packet's instance-flag vocabulary.
  loom_op_kind_t operation_kind;
} loom_text_low_asm_packet_descriptor_t;

typedef struct loom_text_low_asm_immediate_descriptor_t {
  // Canonical low attribute field name stored in the emitted operation.
  iree_string_view_t field_name;
  // Surface spelling accepted by the asm parser for this immediate.
  iree_string_view_t spelling;
  // True when the packet may omit this immediate attribute.
  bool has_default_value;
  // Contract-local enum domain, or UINT16_MAX for a non-enum immediate.
  uint16_t enum_domain;
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
  // Ordinary operation interpreted through its declared format and optional
  // assembly alias. Operation legality belongs to the dialect verifier.
  LOOM_TEXT_LOW_ASM_STATEMENT_CANONICAL = 0,
  // Descriptor-backed packet printed as an instruction mnemonic.
  LOOM_TEXT_LOW_ASM_STATEMENT_PACKET = 1,
  // Descriptor-backed operation with no lossless packet spelling. The packet
  // descriptor_key identifies it for diagnostics; canonical fallback is
  // invalid.
  LOOM_TEXT_LOW_ASM_STATEMENT_UNAVAILABLE = 2,
} loom_text_low_asm_statement_kind_t;

enum loom_text_low_asm_packet_build_flag_bits_e {
  // No result annotation was authored. Infer ownership ties from the packet
  // descriptor. Explicit result annotations instead carry the complete tie set.
  LOOM_TEXT_LOW_ASM_PACKET_BUILD_FLAG_INFER_TIES = 1u << 0,
};
typedef uint32_t loom_text_low_asm_packet_build_flags_t;

typedef struct loom_text_low_asm_statement_t {
  // Statement kind describing which fields below are meaningful.
  loom_text_low_asm_statement_kind_t kind;
  // Original canonical operation represented by this asm statement.
  const loom_op_t* op;
  // Packet descriptor for descriptor-backed packet statements.
  loom_text_low_asm_packet_descriptor_t packet;
  // SSA results defined by a packet statement.
  const loom_value_id_t* results;
  // Number of SSA results in |results|.
  uint16_t result_count;
  // Ownership ties differ from descriptor inference, requiring a complete
  // result annotation even when the result types alone could be inferred.
  bool requires_result_tie_annotation;
  // SSA operands consumed by a packet statement.
  const loom_value_id_t* operands;
  // Number of SSA values in |operands|.
  uint16_t operand_count;
  // Canonical immediate attributes stored on the packet operation.
  loom_named_attr_slice_t attributes;
  // True when immediate attributes correspond to a concrete op attr field.
  bool has_immediate_attribute_field;
  // Operation attribute field index storing packet immediate attributes.
  uint16_t immediate_attribute_field_index;
  // Source location attached to the represented operation.
  loom_location_id_t location;
} loom_text_low_asm_statement_t;

// Contract for descriptor-backed statement descriptions.
//
// A describe callback consumes canonical IR and classifies descriptor-backed
// operations independently of ordinary operation formatting. CANONICAL uses the
// operation's generated grammar. UNAVAILABLE retains the stable descriptor key
// for a packet that cannot be printed losslessly. PACKET carries valid module
// values, descriptor-matching result/operand counts and canonical immediates;
// omitted optional immediates use descriptor defaults.
//
// The text printer relies on this contract and only performs formatting and
// lossless-spelling availability checks. Semantic validation belongs in the
// verifier and descriptor-backed describe implementation, not in the printer.

// Resolves a mnemonic within a descriptor set to a packet descriptor. Sets
// |out_packet->descriptor| to NULL when no packet matches.
typedef void (*loom_text_low_asm_lookup_packet_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    iree_string_view_t mnemonic,
    loom_text_low_asm_packet_descriptor_t* out_packet);

// Attempts to explain an otherwise unknown stable descriptor key or compact
// assembly mnemonic with a target-owned structured diagnostic. Returns OK with
// |out_diagnostic->error| NULL when no diagnostic matches.
typedef iree_status_t (*loom_text_low_asm_diagnose_unknown_packet_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    iree_string_view_t packet_name,
    loom_text_low_asm_diagnostic_t* out_diagnostic);

// Result-type queries below receive an index below packet->result_count and
// operands in the packet's declared assembly order. Type compatibility depends
// on user IR; the descriptor metadata and its projections are generated facts.
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

// Resolves a generation-owned field in the packet's complete immediate order.
// The caller supplies an index below packet->immediate_count.
typedef void (*loom_text_low_asm_immediate_descriptor_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t immediate_index,
    loom_text_low_asm_immediate_descriptor_t* out_immediate);

// Resolves a generation-owned segment. The caller supplies an index below
// packet->operand_segment_count.
typedef void (*loom_text_low_asm_operand_segment_descriptor_fn_t)(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet, uint16_t segment_index,
    loom_text_low_asm_operand_segment_descriptor_t* out_segment);

typedef iree_status_t (*loom_text_low_asm_build_packet_fn_t)(
    const loom_text_low_asm_environment_state_t* state, loom_builder_t* builder,
    const loom_text_low_asm_packet_descriptor_t* packet,
    loom_text_low_asm_packet_build_flags_t build_flags, uint8_t instance_flags,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attributes, const loom_type_t* result_types,
    iree_host_size_t result_count, const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count, loom_location_id_t location,
    loom_op_t** out_op);

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
  // All callbacks are required except diagnose_unknown_packet. Environment
  // presence is checked once when entering a Low function; packet parsing and
  // printing then call this contract directly.
  // Resolves a mnemonic within a descriptor-set handle to a packet descriptor.
  loom_text_low_asm_lookup_packet_fn_t lookup_packet;
  // Optional target-owned explanation for unknown mnemonics.
  loom_text_low_asm_diagnose_unknown_packet_fn_t diagnose_unknown_packet;
  // Infers a result type when the asm packet omits explicit type annotations.
  loom_text_low_asm_infer_result_type_fn_t infer_result_type;
  // Validates an explicit asm result type annotation against the descriptor.
  loom_text_low_asm_validate_result_type_fn_t validate_result_type;
  // Returns whether printing must emit an explicit result type annotation.
  loom_text_low_asm_result_type_annotation_required_fn_t
      result_type_annotation_required;
  // Returns canonical field and surface spelling metadata for one immediate.
  loom_text_low_asm_immediate_descriptor_fn_t immediate_descriptor;
  // Returns delimiter and cardinality metadata for one operand segment.
  loom_text_low_asm_operand_segment_descriptor_fn_t operand_segment_descriptor;
  // Builds the canonical low operation for a parsed descriptor-backed packet.
  loom_text_low_asm_build_packet_fn_t build_packet;
  // Describes a canonical operation as a printable low asm statement.
  // CANONICAL delegates to ordinary formats; UNAVAILABLE rejects packet
  // fallback. See the statement description contract above for the validity
  // guarantees required when a concrete statement kind is returned.
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
  // Generated short spellings, or NULL to use only canonical operation names.
  const loom_op_assembly_format_table_t* operation_formats;
  // Stable-key codec for canonical Low representation values.
  loom_low_repr_environment_t low_repr;
  // Function table implementing low asm lookup, type inference, and builders.
  const loom_text_low_asm_vtable_t* vtable;
  // Environment-owned state passed to every vtable callback.
  const loom_text_low_asm_environment_state_t* state;
} loom_text_low_asm_environment_t;

static inline bool loom_text_low_asm_environment_is_configured(
    const loom_text_low_asm_environment_t* environment) {
  return environment && environment->vtable && environment->state &&
         environment->low_repr.vtable && environment->low_repr.state &&
         environment->vtable->lookup_packet &&
         environment->vtable->resolve_register_type &&
         environment->vtable->infer_result_type &&
         environment->vtable->validate_result_type &&
         environment->vtable->immediate_descriptor &&
         environment->vtable->operand_segment_descriptor &&
         environment->vtable->build_packet;
}

static inline bool loom_text_low_asm_environment_supports_printing(
    const loom_text_low_asm_environment_t* environment) {
  return environment && environment->vtable && environment->state &&
         environment->low_repr.vtable && environment->low_repr.state &&
         environment->vtable->result_type_annotation_required &&
         environment->vtable->immediate_descriptor &&
         environment->vtable->operand_segment_descriptor &&
         environment->vtable->describe_operation &&
         environment->vtable->lookup_register_descriptor_set &&
         environment->vtable->describe_register_type;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_LOW_ASM_H_
