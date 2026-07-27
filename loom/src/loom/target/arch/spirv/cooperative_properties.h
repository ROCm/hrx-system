// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V cooperative matrix/vector property facts.
//
// These tables model API/device and target-profile cooperative operation facts
// separately from SPIR-V extension/capability availability. Feature atoms
// answer whether a target profile can emit a family of instructions; property
// rows answer whether a specific operation shape and interpretation is legal
// for that family.

#ifndef LOOM_TARGET_ARCH_SPIRV_COOPERATIVE_PROPERTIES_H_
#define LOOM_TARGET_ARCH_SPIRV_COOPERATIVE_PROPERTIES_H_

#include "iree/base/api.h"
#include "loom/analysis/policy.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/scalar_types.h"
#include "loom/util/numeric_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_spirv_cooperative_matrix_layout_flag_bits_e {
  // RowMajorKHR layout is accepted.
  LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_BIT = 1u << 0,
  // ColumnMajorKHR layout is accepted.
  LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_BIT = 1u << 1,
} loom_spirv_cooperative_matrix_layout_flag_bits_t;

// Bitset of loom_spirv_cooperative_matrix_layout_flag_bits_t values.
typedef uint32_t loom_spirv_cooperative_matrix_layout_flags_t;

typedef enum loom_spirv_cooperative_matrix_operand_flag_bits_e {
  // Matrix A signed-components operand is required.
  LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_A_SIGNED_COMPONENTS = 1u << 0,
  // Matrix B signed-components operand is required.
  LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_B_SIGNED_COMPONENTS = 1u << 1,
  // Matrix C signed-components operand is required.
  LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_C_SIGNED_COMPONENTS = 1u << 2,
  // Matrix result signed-components operand is required.
  LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_RESULT_SIGNED_COMPONENTS = 1u << 3,
  // Saturating accumulation operand is required.
  LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION = 1u << 4,
} loom_spirv_cooperative_matrix_operand_flag_bits_t;

// Bitset of loom_spirv_cooperative_matrix_operand_flag_bits_t values.
typedef uint32_t loom_spirv_cooperative_matrix_operand_flags_t;

typedef enum loom_spirv_cooperative_vector_matrix_layout_flag_bits_e {
  // RowMajorNV layout is accepted.
  LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_BIT = 1u << 0,
  // ColumnMajorNV layout is accepted.
  LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_BIT = 1u << 1,
  // InferencingOptimalNV layout is accepted.
  LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_BIT = 1u << 2,
  // TrainingOptimalNV layout is accepted.
  LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_BIT = 1u << 3,
} loom_spirv_cooperative_vector_matrix_layout_flag_bits_t;

// Bitset of loom_spirv_cooperative_vector_matrix_layout_flag_bits_t values.
typedef uint32_t loom_spirv_cooperative_vector_matrix_layout_flags_t;

typedef enum loom_spirv_cooperative_vector_flag_bits_e {
  // Transposed opaque matrix layout is accepted.
  LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRANSPOSE = 1u << 0,
  // Row is intended for training operations rather than inference only.
  LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING = 1u << 1,
} loom_spirv_cooperative_vector_flag_bits_t;

// Bitset of loom_spirv_cooperative_vector_flag_bits_t values.
typedef uint32_t loom_spirv_cooperative_vector_flags_t;

typedef enum loom_spirv_storage_class_flag_bits_e {
  // Workgroup storage class is accepted.
  LOOM_SPIRV_STORAGE_CLASS_BIT_WORKGROUP = 1u << 0,
  // StorageBuffer storage class is accepted.
  LOOM_SPIRV_STORAGE_CLASS_BIT_STORAGE_BUFFER = 1u << 1,
  // PhysicalStorageBuffer storage class is accepted.
  LOOM_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER = 1u << 2,
} loom_spirv_storage_class_flag_bits_t;

// Bitset of loom_spirv_storage_class_flag_bits_t values.
typedef uint32_t loom_spirv_storage_class_flags_t;

typedef enum loom_spirv_cooperative_selection_status_e {
  // Unknown or uninitialized selection status.
  LOOM_SPIRV_COOPERATIVE_SELECTION_UNKNOWN = 0,
  // A property row matched the request.
  LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED = 1,
  // The required feature family is not selected by the target profile.
  LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_FEATURE_MISSING = 2,
  // The feature is selected, but no property row matched the request.
  LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING = 3,
  // No property row matched, but the lowering policy permits reference
  // fallback.
  LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED = 4,
} loom_spirv_cooperative_selection_status_t;

typedef enum loom_spirv_cooperative_rejection_flag_bits_e {
  // No rejection reason was recorded.
  LOOM_SPIRV_COOPERATIVE_REJECTION_NONE = 0u,
  // The target profile does not select the required feature atom.
  LOOM_SPIRV_COOPERATIVE_REJECTION_FEATURE = 1u << 0,
  // No candidate has the requested shape.
  LOOM_SPIRV_COOPERATIVE_REJECTION_SHAPE = 1u << 1,
  // No shape-compatible candidate has the requested component types.
  LOOM_SPIRV_COOPERATIVE_REJECTION_COMPONENT_TYPE = 1u << 2,
  // No type-compatible candidate has the requested scope.
  LOOM_SPIRV_COOPERATIVE_REJECTION_SCOPE = 1u << 3,
  // No type-compatible candidate accepts the requested layout.
  LOOM_SPIRV_COOPERATIVE_REJECTION_LAYOUT = 1u << 4,
  // No type-compatible candidate accepts the requested storage class.
  LOOM_SPIRV_COOPERATIVE_REJECTION_STORAGE_CLASS = 1u << 5,
  // No type-compatible candidate carries the requested operand flags.
  LOOM_SPIRV_COOPERATIVE_REJECTION_OPERANDS = 1u << 6,
  // The lowering policy permits reference fallback for this miss.
  LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK = 1u << 7,
  // The cooperative vector Input object component type differs.
  LOOM_SPIRV_COOPERATIVE_REJECTION_INPUT_TYPE = 1u << 8,
  // The cooperative vector InputInterpretation operand differs.
  LOOM_SPIRV_COOPERATIVE_REJECTION_INPUT_INTERPRETATION = 1u << 9,
  // The cooperative vector MatrixInterpretation operand differs.
  LOOM_SPIRV_COOPERATIVE_REJECTION_MATRIX_INTERPRETATION = 1u << 10,
  // The cooperative vector BiasInterpretation operand differs.
  LOOM_SPIRV_COOPERATIVE_REJECTION_BIAS_INTERPRETATION = 1u << 11,
  // The cooperative vector Result object component type differs.
  LOOM_SPIRV_COOPERATIVE_REJECTION_RESULT_TYPE = 1u << 12,
  // A property row matched, but no target-low descriptor maps it yet.
  LOOM_SPIRV_COOPERATIVE_REJECTION_DESCRIPTOR = 1u << 13,
} loom_spirv_cooperative_rejection_flag_bits_t;

// Bitset of loom_spirv_cooperative_rejection_flag_bits_t values.
typedef uint32_t loom_spirv_cooperative_rejection_flags_t;

typedef struct loom_spirv_cooperative_matrix_property_t {
  // Stable row name for diagnostics and tests.
  iree_string_view_t name;
  // Feature bits required before this row is legal.
  loom_spirv_feature_bits_t required_feature_bits;
  // Result row count and Matrix A row count.
  uint16_t m_size;
  // Result column count and Matrix B column count.
  uint16_t n_size;
  // Matrix A column count and Matrix B row count.
  uint16_t k_size;
  // Matrix A scalar component type.
  loom_spirv_scalar_type_t lhs_type;
  // Matrix B scalar component type.
  loom_spirv_scalar_type_t rhs_type;
  // Matrix C accumulator scalar component type.
  loom_spirv_scalar_type_t accumulator_type;
  // Result scalar component type.
  loom_spirv_scalar_type_t result_type;
  // Cooperative matrix scope.
  loom_spirv_scope_t scope;
  // Accepted load/store memory layouts.
  loom_spirv_cooperative_matrix_layout_flags_t layout_flags;
  // Accepted pointer storage classes for load/store operations.
  loom_spirv_storage_class_flags_t storage_class_flags;
  // Required cooperative matrix operand bits for this row.
  loom_spirv_cooperative_matrix_operand_flags_t operand_flags;
  // Target-low descriptor ref selected for the matrix multiply-add operation.
  uint32_t mul_add_descriptor_ref;
} loom_spirv_cooperative_matrix_property_t;

typedef struct loom_spirv_cooperative_vector_property_t {
  // Stable row name for diagnostics and tests.
  iree_string_view_t name;
  // Feature bits required before this row is legal.
  loom_spirv_feature_bits_t required_feature_bits;
  // Output vector component count.
  uint16_t m_size;
  // Logical input vector component count.
  uint16_t k_size;
  // Component type of the Input vector object.
  loom_spirv_component_type_t input_type;
  // InputInterpretation operand value.
  loom_spirv_component_type_t input_interpretation;
  // MatrixInterpretation operand value.
  loom_spirv_component_type_t matrix_interpretation;
  // BiasInterpretation operand value.
  loom_spirv_component_type_t bias_interpretation;
  // Component type of the Result vector object.
  loom_spirv_component_type_t result_type;
  // Accepted matrix layout operands.
  loom_spirv_cooperative_vector_matrix_layout_flags_t matrix_layout_flags;
  // Accepted matrix pointer storage classes.
  loom_spirv_storage_class_flags_t storage_class_flags;
  // Additional cooperative vector behavior flags.
  loom_spirv_cooperative_vector_flags_t flags;
} loom_spirv_cooperative_vector_property_t;

typedef struct loom_spirv_cooperative_property_span_t {
  // Shape key used for binary lookup.
  uint64_t shape_key;
  // First row in the corresponding property table.
  uint16_t start;
  // Number of contiguous rows sharing |shape_key|.
  uint16_t count;
} loom_spirv_cooperative_property_span_t;

typedef struct loom_spirv_cooperative_property_set_t {
  // Feature bits selected by the prepared SPIR-V target profile.
  loom_spirv_feature_bits_t feature_bits;
  // Matrix property rows sorted by shape key.
  const loom_spirv_cooperative_matrix_property_t* matrix_properties;
  // Number of entries in |matrix_properties|.
  uint16_t matrix_property_count;
  // Shape-key spans into |matrix_properties|.
  const loom_spirv_cooperative_property_span_t* matrix_shape_spans;
  // Number of entries in |matrix_shape_spans|.
  uint16_t matrix_shape_span_count;
  // Cooperative vector property rows sorted by shape key.
  const loom_spirv_cooperative_vector_property_t* vector_properties;
  // Number of entries in |vector_properties|.
  uint16_t vector_property_count;
  // Shape-key spans into |vector_properties|.
  const loom_spirv_cooperative_property_span_t* vector_shape_spans;
  // Number of entries in |vector_shape_spans|.
  uint16_t vector_shape_span_count;
} loom_spirv_cooperative_property_set_t;

typedef struct loom_spirv_cooperative_property_storage_t {
  // Borrowable property set view over rows owned by this storage object.
  loom_spirv_cooperative_property_set_t set;
  // Matrix property rows owned by this storage object.
  loom_spirv_cooperative_matrix_property_t* matrix_properties;
  // Matrix shape-key spans owned by this storage object.
  loom_spirv_cooperative_property_span_t* matrix_shape_spans;
  // Vector property rows owned by this storage object.
  loom_spirv_cooperative_vector_property_t* vector_properties;
  // Vector shape-key spans owned by this storage object.
  loom_spirv_cooperative_property_span_t* vector_shape_spans;
} loom_spirv_cooperative_property_storage_t;

typedef struct loom_spirv_cooperative_matrix_query_t {
  // Result row count and Matrix A row count.
  uint16_t m_size;
  // Result column count and Matrix B column count.
  uint16_t n_size;
  // Matrix A column count and Matrix B row count.
  uint16_t k_size;
  // Matrix A scalar component type.
  loom_spirv_scalar_type_t lhs_type;
  // Matrix B scalar component type.
  loom_spirv_scalar_type_t rhs_type;
  // Matrix C accumulator scalar component type.
  loom_spirv_scalar_type_t accumulator_type;
  // Result scalar component type.
  loom_spirv_scalar_type_t result_type;
  // Cooperative matrix scope.
  loom_spirv_scope_t scope;
  // Requested load/store memory layout.
  loom_spirv_cooperative_matrix_layout_t layout;
  // Requested pointer storage class for load/store operations.
  loom_spirv_storage_class_t storage_class;
  // Required cooperative matrix operand bits.
  loom_spirv_cooperative_matrix_operand_flags_t operand_flags;
  // Fallback and target primitive selection policy.
  loom_lowering_policy_t policy;
} loom_spirv_cooperative_matrix_query_t;

typedef struct loom_spirv_cooperative_vector_query_t {
  // Output vector component count.
  uint16_t m_size;
  // Logical input vector component count.
  uint16_t k_size;
  // Component type of the Input vector object.
  loom_spirv_component_type_t input_type;
  // InputInterpretation operand value.
  loom_spirv_component_type_t input_interpretation;
  // MatrixInterpretation operand value.
  loom_spirv_component_type_t matrix_interpretation;
  // BiasInterpretation operand value.
  loom_spirv_component_type_t bias_interpretation;
  // Component type of the Result vector object.
  loom_spirv_component_type_t result_type;
  // Requested matrix layout operand.
  loom_spirv_cooperative_vector_matrix_layout_t matrix_layout;
  // Requested matrix pointer storage class.
  loom_spirv_storage_class_t storage_class;
  // Required cooperative vector behavior flags.
  loom_spirv_cooperative_vector_flags_t flags;
  // Fallback and target primitive selection policy.
  loom_lowering_policy_t policy;
} loom_spirv_cooperative_vector_query_t;

typedef struct loom_spirv_cooperative_diagnostic_t {
  // Final selection status.
  loom_spirv_cooperative_selection_status_t status;
  // Feature atom family required by the query.
  loom_spirv_feature_atom_t feature_atom;
  // Structural rejection flags explaining the miss.
  loom_spirv_cooperative_rejection_flags_t rejection_flags;
  // Number of rows in the shape-key candidate span.
  uint16_t shape_candidate_count;
  // Number of shape-compatible rows that matched component types.
  uint16_t type_candidate_count;
  // Number of candidates that matched layout after earlier filters.
  uint16_t layout_candidate_count;
  // Number of candidates that matched storage class after earlier filters.
  uint16_t storage_candidate_count;
} loom_spirv_cooperative_diagnostic_t;

// Returns the stable diagnostic spelling for a component type.
iree_string_view_t loom_spirv_component_type_name(
    loom_spirv_component_type_t component_type);

// Maps a target-independent encoded numeric format to a SPIR-V ComponentType
// operand value. Returns false when the numeric format is unknown,
// multi-valued, or not representable by the currently modeled SPIR-V extension
// component vocabulary.
bool loom_spirv_component_type_from_numeric_format(
    loom_value_fact_numeric_format_flags_t numeric_format,
    loom_spirv_component_type_t* out_component_type);

// Returns the stable diagnostic spelling for a scope.
iree_string_view_t loom_spirv_scope_name(loom_spirv_scope_t scope);

// Returns the stable diagnostic spelling for a matrix layout.
iree_string_view_t loom_spirv_cooperative_matrix_layout_name(
    loom_spirv_cooperative_matrix_layout_t layout);

// Returns the stable diagnostic spelling for a vector matrix layout.
iree_string_view_t loom_spirv_cooperative_vector_matrix_layout_name(
    loom_spirv_cooperative_vector_matrix_layout_t layout);

// Returns the stable diagnostic spelling for a storage class.
iree_string_view_t loom_spirv_storage_class_name(
    loom_spirv_storage_class_t storage_class);

// Returns the bit corresponding to a storage class, or zero when unsupported.
loom_spirv_storage_class_flags_t loom_spirv_storage_class_bit(
    loom_spirv_storage_class_t storage_class);

// Returns the sorted static cooperative matrix model rows.
const loom_spirv_cooperative_matrix_property_t*
loom_spirv_cooperative_matrix_model_properties(iree_host_size_t* out_count);

// Returns the sorted static cooperative vector model rows.
const loom_spirv_cooperative_vector_property_t*
loom_spirv_cooperative_vector_model_properties(iree_host_size_t* out_count);

// Returns the shape key used to group matrix property rows.
uint64_t loom_spirv_cooperative_matrix_shape_key(uint16_t m_size,
                                                 uint16_t n_size,
                                                 uint16_t k_size);

// Prepares static cooperative property rows for a prepared feature set.
void loom_spirv_cooperative_property_set_prepare(
    const loom_spirv_feature_set_t* feature_set,
    loom_spirv_cooperative_property_set_t* out_property_set);

// Initializes storage with copied, sorted cooperative property rows.
//
// This helper owns allocation, stable shape-key sorting, and span construction;
// selection remains the same direct shape-key lookup used by static property
// sets.
iree_status_t loom_spirv_cooperative_property_storage_initialize(
    loom_spirv_feature_bits_t feature_bits,
    const loom_spirv_cooperative_matrix_property_t* matrix_properties,
    iree_host_size_t matrix_property_count,
    const loom_spirv_cooperative_vector_property_t* vector_properties,
    iree_host_size_t vector_property_count, iree_allocator_t allocator,
    loom_spirv_cooperative_property_storage_t* out_storage);

// Releases storage allocated by
// loom_spirv_cooperative_property_storage_initialize.
void loom_spirv_cooperative_property_storage_deinitialize(
    loom_spirv_cooperative_property_storage_t* storage,
    iree_allocator_t allocator);

// Selects a cooperative matrix property row for |query|. |out_diagnostic| may
// be NULL when the caller only needs the selected row.
const loom_spirv_cooperative_matrix_property_t*
loom_spirv_cooperative_matrix_property_select(
    const loom_spirv_cooperative_property_set_t* property_set,
    const loom_spirv_cooperative_matrix_query_t* query,
    loom_spirv_cooperative_diagnostic_t* out_diagnostic);

// Selects a cooperative vector property row for |query|. |out_diagnostic| may
// be NULL when the caller only needs the selected row.
const loom_spirv_cooperative_vector_property_t*
loom_spirv_cooperative_vector_property_select(
    const loom_spirv_cooperative_property_set_t* property_set,
    const loom_spirv_cooperative_vector_query_t* query,
    loom_spirv_cooperative_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_COOPERATIVE_PROPERTIES_H_
