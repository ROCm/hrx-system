// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_SPIRV_PROFILE_H_
#define LOOMC_TARGET_SPIRV_PROFILE_H_

#include "loomc/result.h"
#include "loomc/target/spirv/base.h"

/// @file
/// SPIR-V target profile facts.
///
/// This leaf owns the normalized, header-light SPIR-V profile API. It does not
/// include Vulkan, IREE HAL, or platform probe headers. Live adapters and saved
/// profile importers should normalize their observations into these facts and
/// then create ordinary `loomc_target_profile_t` handles.
///
/// Profiles are partial by construction. Unknown feature facts remain unknown
/// until a caller supplies a stronger observation; false means a feature is
/// known to be unavailable. Contradictory true/false facts are reported through
/// the returned `loomc_result_t` with provenance strings preserved in the
/// diagnostic text. Numeric limits and environment facts follow the same
/// tri-state model: true means the value is known, false means the fact is
/// known not to apply to this profile, and unknown preserves partial-target
/// compilation.
///
/// @par Example
/// Create a reusable Vulkan-style SPIR-V profile and target selection:
///
/// @code{.c}
/// loomc_spirv_feature_fact_t facts[] = {
///     {
///         .feature = LOOMC_SPIRV_FEATURE_FLOAT16,
///         .state = LOOMC_TARGET_FACT_STATE_TRUE,
///         .provenance = loomc_make_cstring_view("vulkaninfo:shaderFloat16"),
///     },
/// };
/// loomc_spirv_limit_fact_t limits[] = {
///     {
///         .limit = LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
///         .state = LOOMC_TARGET_FACT_STATE_TRUE,
///         .value = 1024,
///         .provenance =
///             loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupSize[0]"),
///     },
/// };
/// loomc_spirv_environment_fact_t environment[] = {
///     {
///         .environment = LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
///         .state = LOOMC_TARGET_FACT_STATE_TRUE,
///         .value = LOOMC_SPIRV_VERSION_1_3,
///         .provenance = loomc_make_cstring_view("vulkaninfo:apiVersion"),
///     },
/// };
/// loomc_spirv_profile_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
///     .structure_size = sizeof(loomc_spirv_profile_options_t),
///     .identifier = loomc_make_cstring_view("offline-vulkan13"),
///     .preset = LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
///     .feature_facts = facts,
///     .feature_fact_count = 1,
///     .limit_facts = limits,
///     .limit_fact_count = 1,
///     .environment_facts = environment,
///     .environment_fact_count = 1,
/// };
/// loomc_target_profile_t* profile = NULL;
/// loomc_result_t* result = NULL;
/// loomc_status_t status = loomc_target_profile_create_spirv(
///     target_environment, &options, loomc_allocator_system(), &profile,
///     &result);
/// if (!loomc_status_is_ok(status)) return status;
/// if (!loomc_result_succeeded(result)) {
///   // Inspect structured diagnostics; profile is NULL on profile failure.
/// }
/// loomc_result_release(result);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Stable SPIR-V feature fact identifier.
///
/// These identifiers are Loom profile facts, not Vulkan feature struct fields.
/// A live Vulkan probe, saved vulkaninfo profile, or non-Vulkan driver may all
/// map their own capability model into the same feature IDs.
typedef enum loomc_spirv_feature_e {
  /// Unknown or uninitialized feature.
  LOOMC_SPIRV_FEATURE_UNKNOWN = 0,

  /// Vulkan shader-module baseline.
  LOOMC_SPIRV_FEATURE_VULKAN_SHADER = 1,

  /// SPV_KHR_physical_storage_buffer support.
  LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER = 2,

  /// 16-bit floating-point scalar support.
  LOOMC_SPIRV_FEATURE_FLOAT16 = 3,

  /// 64-bit floating-point scalar support.
  LOOMC_SPIRV_FEATURE_FLOAT64 = 4,

  /// 8-bit integer scalar support.
  LOOMC_SPIRV_FEATURE_INT8 = 5,

  /// 16-bit integer scalar support.
  LOOMC_SPIRV_FEATURE_INT16 = 6,

  /// 8-bit storage-buffer access support.
  LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS = 7,

  /// 16-bit storage-buffer access support.
  LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS = 8,

  /// SPV_NV_cooperative_vector support.
  LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV = 9,

  /// SPV_NV_cooperative_vector training-operation support.
  LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV = 10,

  /// SPV_KHR_cooperative_matrix support.
  LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR = 11,

  /// SPV_KHR_bfloat16 scalar type support.
  LOOMC_SPIRV_FEATURE_BFLOAT16_TYPE_KHR = 12,

  /// SPV_KHR_bfloat16 dot-product support.
  LOOMC_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR = 13,

  /// SPV_KHR_bfloat16 cooperative-matrix support.
  LOOMC_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR = 14,

  /// 64-bit integer scalar support.
  LOOMC_SPIRV_FEATURE_INT64 = 15,

  /// Number of public SPIR-V feature identifiers.
  LOOMC_SPIRV_FEATURE_COUNT = 16,
} loomc_spirv_feature_t;

/// Bitset of `loomc_spirv_feature_t` values.
typedef uint64_t loomc_spirv_feature_bits_t;

/// Encodes a SPIR-V binary version word.
#define LOOMC_SPIRV_VERSION(major, minor) \
  ((((uint32_t)(major)) << 16) | (((uint32_t)(minor)) << 8))

/// SPIR-V 1.0 binary version word.
#define LOOMC_SPIRV_VERSION_1_0 LOOMC_SPIRV_VERSION(1, 0)

/// SPIR-V 1.1 binary version word.
#define LOOMC_SPIRV_VERSION_1_1 LOOMC_SPIRV_VERSION(1, 1)

/// SPIR-V 1.2 binary version word.
#define LOOMC_SPIRV_VERSION_1_2 LOOMC_SPIRV_VERSION(1, 2)

/// SPIR-V 1.3 binary version word.
#define LOOMC_SPIRV_VERSION_1_3 LOOMC_SPIRV_VERSION(1, 3)

/// SPIR-V 1.4 binary version word.
#define LOOMC_SPIRV_VERSION_1_4 LOOMC_SPIRV_VERSION(1, 4)

/// SPIR-V 1.5 binary version word.
#define LOOMC_SPIRV_VERSION_1_5 LOOMC_SPIRV_VERSION(1, 5)

/// SPIR-V 1.6 binary version word.
#define LOOMC_SPIRV_VERSION_1_6 LOOMC_SPIRV_VERSION(1, 6)

/// Returns the maximum SPIR-V version implied by a Vulkan API version pair.
///
/// The returned value is a SPIR-V binary version word such as
/// `LOOMC_SPIRV_VERSION_1_6`.
static inline uint32_t loomc_spirv_max_version_from_vulkan_api_version(
    uint32_t vulkan_major, uint32_t vulkan_minor) {
  if (vulkan_major > 1 || (vulkan_major == 1 && vulkan_minor >= 3)) {
    return LOOMC_SPIRV_VERSION_1_6;
  }
  if (vulkan_major == 1 && vulkan_minor >= 2) {
    return LOOMC_SPIRV_VERSION_1_5;
  }
  if (vulkan_major == 1 && vulkan_minor >= 1) {
    return LOOMC_SPIRV_VERSION_1_3;
  }
  return LOOMC_SPIRV_VERSION_1_0;
}

/// Returns the direct bit for a SPIR-V feature.
static inline loomc_spirv_feature_bits_t loomc_spirv_feature_bit(
    loomc_spirv_feature_t feature) {
  return feature > LOOMC_SPIRV_FEATURE_UNKNOWN &&
                 feature < LOOMC_SPIRV_FEATURE_COUNT
             ? (UINT64_C(1) << feature)
             : 0u;
}

/// Stable SPIR-V numeric limit fact identifier.
///
/// Limits use Loom profile names rather than Vulkan field names so the same
/// fact can come from Vulkan, another SPIR-V-capable API, a saved profile, or a
/// synthetic cross-compilation target.
typedef enum loomc_spirv_limit_e {
  /// Unknown or uninitialized limit.
  LOOMC_SPIRV_LIMIT_UNKNOWN = 0,

  /// Maximum local workgroup size along the x dimension.
  LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X = 1,

  /// Maximum local workgroup size along the y dimension.
  LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y = 2,

  /// Maximum local workgroup size along the z dimension.
  LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z = 3,

  /// Maximum product of local workgroup dimensions.
  LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE = 4,

  /// Maximum bytes reserved in SPIR-V Workgroup storage by one entry point.
  LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES = 5,

  /// Fixed target-wide subgroup size in invocations.
  LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE = 6,

  /// Maximum dispatched workgroup count along the x dimension.
  LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X = 7,

  /// Maximum dispatched workgroup count along the y dimension.
  LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y = 8,

  /// Maximum dispatched workgroup count along the z dimension.
  LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z = 9,

  /// Maximum dispatched grid size along the x dimension, in workitems.
  LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_X = 10,

  /// Maximum dispatched grid size along the y dimension, in workitems.
  LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Y = 11,

  /// Maximum dispatched grid size along the z dimension, in workitems.
  LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Z = 12,

  /// Maximum product of dispatched grid dimensions, in workitems.
  LOOMC_SPIRV_LIMIT_MAX_FLAT_GRID_SIZE = 13,

  /// Number of public SPIR-V limit identifiers.
  LOOMC_SPIRV_LIMIT_COUNT = 14,
} loomc_spirv_limit_t;

/// Stable SPIR-V numeric environment fact identifier.
///
/// Environment facts describe the SPIR-V module environment accepted by the
/// target independently from individual feature bits. They constrain feature
/// selection but do not themselves imply extensions or capabilities.
typedef enum loomc_spirv_environment_e {
  /// Unknown or uninitialized environment fact.
  LOOMC_SPIRV_ENVIRONMENT_UNKNOWN = 0,

  /// Maximum SPIR-V binary version accepted by the target.
  LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION = 1,

  /// Number of public SPIR-V environment identifiers.
  LOOMC_SPIRV_ENVIRONMENT_COUNT = 2,
} loomc_spirv_environment_t;

/// SPIR-V semantic scalar type fact.
///
/// These values describe the scalar element types Loom selects for SPIR-V type
/// declarations. They are separate from SPIR-V component-type enum operands,
/// which are only used by instructions or extension APIs that literally encode
/// those operands.
typedef enum loomc_spirv_scalar_type_e {
  /// Unknown or uninitialized scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_UNKNOWN = 0,

  /// IEEE binary16 floating-point scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_F16 = 1,

  /// IEEE binary32 floating-point scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_F32 = 2,

  /// IEEE binary64 floating-point scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_F64 = 3,

  /// KHR bfloat16 scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_BF16 = 4,

  /// Signed 8-bit integer scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_S8 = 5,

  /// Signed 16-bit integer scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_S16 = 6,

  /// Signed 32-bit integer scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_S32 = 7,

  /// Signed 64-bit integer scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_S64 = 8,

  /// Unsigned 8-bit integer scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_U8 = 9,

  /// Unsigned 16-bit integer scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_U16 = 10,

  /// Unsigned 32-bit integer scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_U32 = 11,

  /// Unsigned 64-bit integer scalar type.
  LOOMC_SPIRV_SCALAR_TYPE_U64 = 12,
} loomc_spirv_scalar_type_t;

/// SPIR-V Scope operand values used by public profile fact rows.
typedef enum loomc_spirv_scope_e {
  /// CrossDevice scope.
  LOOMC_SPIRV_SCOPE_CROSS_DEVICE = 0,

  /// Device scope.
  LOOMC_SPIRV_SCOPE_DEVICE = 1,

  /// Workgroup scope.
  LOOMC_SPIRV_SCOPE_WORKGROUP = 2,

  /// Subgroup scope.
  LOOMC_SPIRV_SCOPE_SUBGROUP = 3,

  /// Invocation scope.
  LOOMC_SPIRV_SCOPE_INVOCATION = 4,

  /// QueueFamily scope.
  LOOMC_SPIRV_SCOPE_QUEUE_FAMILY = 5,

  /// ShaderCallKHR scope.
  LOOMC_SPIRV_SCOPE_SHADER_CALL_KHR = 6,
} loomc_spirv_scope_t;

/// Cooperative matrix layout fact bits.
typedef enum loomc_spirv_cooperative_matrix_layout_flag_bits_e {
  /// RowMajorKHR layout is accepted.
  LOOMC_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_BIT = 1u << 0,

  /// ColumnMajorKHR layout is accepted.
  LOOMC_SPIRV_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_BIT = 1u << 1,
} loomc_spirv_cooperative_matrix_layout_flag_bits_t;

/// Bitset of `loomc_spirv_cooperative_matrix_layout_flag_bits_t` values.
typedef uint32_t loomc_spirv_cooperative_matrix_layout_flags_t;

/// Cooperative matrix operand fact bits.
typedef enum loomc_spirv_cooperative_matrix_operand_flag_bits_e {
  /// Matrix A signed-components operand is required.
  LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_A_SIGNED_COMPONENTS = 1u << 0,

  /// Matrix B signed-components operand is required.
  LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_B_SIGNED_COMPONENTS = 1u << 1,

  /// Matrix C signed-components operand is required.
  LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_C_SIGNED_COMPONENTS = 1u << 2,

  /// Matrix result signed-components operand is required.
  LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_RESULT_SIGNED_COMPONENTS = 1u << 3,

  /// Saturating accumulation operand is required.
  LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION = 1u << 4,
} loomc_spirv_cooperative_matrix_operand_flag_bits_t;

/// Bitset of `loomc_spirv_cooperative_matrix_operand_flag_bits_t` values.
typedef uint32_t loomc_spirv_cooperative_matrix_operand_flags_t;

/// SPIR-V cooperative vector ComponentType operand values.
typedef enum loomc_spirv_component_type_e {
  /// Float16NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_FLOAT16_NV = 0,

  /// Float32NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_FLOAT32_NV = 1,

  /// Float64NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_FLOAT64_NV = 2,

  /// SignedInt8NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV = 3,

  /// SignedInt16NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT16_NV = 4,

  /// SignedInt32NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV = 5,

  /// SignedInt64NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT64_NV = 6,

  /// UnsignedInt8NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV = 7,

  /// UnsignedInt16NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT16_NV = 8,

  /// UnsignedInt32NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV = 9,

  /// UnsignedInt64NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT64_NV = 10,

  /// SignedInt8PackedNV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV = 1000491000,

  /// UnsignedInt8PackedNV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_PACKED_NV = 1000491001,

  /// FloatE4M3NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV = 1000491002,

  /// FloatE5M2NV component type.
  LOOMC_SPIRV_COMPONENT_TYPE_FLOAT_E5_M2_NV = 1000491003,
} loomc_spirv_component_type_t;

/// Cooperative vector matrix-layout fact bits.
typedef enum loomc_spirv_cooperative_vector_matrix_layout_flag_bits_e {
  /// RowMajorNV layout is accepted.
  LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_BIT = 1u << 0,

  /// ColumnMajorNV layout is accepted.
  LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_BIT = 1u << 1,

  /// InferencingOptimalNV layout is accepted.
  LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_BIT =
      (1u << 2),

  /// TrainingOptimalNV layout is accepted.
  LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_BIT = 1u << 3,
} loomc_spirv_cooperative_vector_matrix_layout_flag_bits_t;

/// Bitset of
/// `loomc_spirv_cooperative_vector_matrix_layout_flag_bits_t` values.
typedef uint32_t loomc_spirv_cooperative_vector_matrix_layout_flags_t;

/// Cooperative vector behavior fact bits.
typedef enum loomc_spirv_cooperative_vector_flag_bits_e {
  /// Transposed opaque matrix layout is accepted.
  LOOMC_SPIRV_COOPERATIVE_VECTOR_FLAG_TRANSPOSE = 1u << 0,

  /// Row is intended for training operations rather than inference only.
  LOOMC_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING = 1u << 1,
} loomc_spirv_cooperative_vector_flag_bits_t;

/// Bitset of `loomc_spirv_cooperative_vector_flag_bits_t` values.
typedef uint32_t loomc_spirv_cooperative_vector_flags_t;

/// Storage-class fact bits used by cooperative operation rows.
typedef enum loomc_spirv_storage_class_flag_bits_e {
  /// Workgroup storage class is accepted.
  LOOMC_SPIRV_STORAGE_CLASS_BIT_WORKGROUP = 1u << 0,

  /// StorageBuffer storage class is accepted.
  LOOMC_SPIRV_STORAGE_CLASS_BIT_STORAGE_BUFFER = 1u << 1,

  /// PhysicalStorageBuffer storage class is accepted.
  LOOMC_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER = 1u << 2,
} loomc_spirv_storage_class_flag_bits_t;

/// Bitset of `loomc_spirv_storage_class_flag_bits_t` values.
typedef uint32_t loomc_spirv_storage_class_flags_t;

/// Built-in SPIR-V profile preset.
typedef enum loomc_spirv_profile_preset_e {
  /// No preset facts.
  LOOMC_SPIRV_PROFILE_PRESET_NONE = 0,

  /// Vulkan 1.3 shader profile with physical-storage-buffer addressing.
  LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA = 1,
} loomc_spirv_profile_preset_t;

/// One SPIR-V feature observation.
typedef struct loomc_spirv_feature_fact_t {
  /// Feature being observed.
  loomc_spirv_feature_t feature;

  /// Observed state for the feature.
  loomc_target_fact_state_t state;

  /// Borrowed provenance string used in diagnostics.
  loomc_string_view_t provenance;
} loomc_spirv_feature_fact_t;

/// One SPIR-V numeric limit observation.
typedef struct loomc_spirv_limit_fact_t {
  /// Limit being observed.
  loomc_spirv_limit_t limit;

  /// Observed state for the limit.
  loomc_target_fact_state_t state;

  /// Observed limit value when `state` is `LOOMC_TARGET_FACT_STATE_TRUE`.
  uint64_t value;

  /// Borrowed provenance string used in diagnostics.
  loomc_string_view_t provenance;
} loomc_spirv_limit_fact_t;

/// Queried SPIR-V numeric limit state.
typedef struct loomc_spirv_limit_value_t {
  /// Known state for the limit.
  loomc_target_fact_state_t state;

  /// Limit value when `state` is `LOOMC_TARGET_FACT_STATE_TRUE`.
  uint64_t value;
} loomc_spirv_limit_value_t;

/// One SPIR-V environment observation.
typedef struct loomc_spirv_environment_fact_t {
  /// Environment fact being observed.
  loomc_spirv_environment_t environment;

  /// Observed state for the environment fact.
  loomc_target_fact_state_t state;

  /// Observed environment value when `state` is `LOOMC_TARGET_FACT_STATE_TRUE`.
  uint64_t value;

  /// Borrowed provenance string used in diagnostics.
  loomc_string_view_t provenance;
} loomc_spirv_environment_fact_t;

/// Queried SPIR-V environment fact state.
typedef struct loomc_spirv_environment_value_t {
  /// Known state for the environment fact.
  loomc_target_fact_state_t state;

  /// Environment value when `state` is `LOOMC_TARGET_FACT_STATE_TRUE`.
  uint64_t value;
} loomc_spirv_environment_value_t;

/// Forward declaration for a cooperative matrix operation fact row.
typedef struct loomc_spirv_cooperative_matrix_row_t
    loomc_spirv_cooperative_matrix_row_t;

/// Forward declaration for a cooperative vector operation fact row.
typedef struct loomc_spirv_cooperative_vector_row_t
    loomc_spirv_cooperative_vector_row_t;

/// SPIR-V target profile creation options.
typedef struct loomc_spirv_profile_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// Stable profile identifier used in diagnostics.
  loomc_string_view_t identifier;

  /// Built-in preset facts applied before explicit feature facts.
  loomc_spirv_profile_preset_t preset;

  /// Borrowed feature fact array.
  const loomc_spirv_feature_fact_t* feature_facts;

  /// Number of entries in `feature_facts`.
  loomc_host_size_t feature_fact_count;

  /// Borrowed numeric limit fact array.
  const loomc_spirv_limit_fact_t* limit_facts;

  /// Number of entries in `limit_facts`.
  loomc_host_size_t limit_fact_count;

  /// Borrowed environment fact array.
  const loomc_spirv_environment_fact_t* environment_facts;

  /// Number of entries in `environment_facts`.
  loomc_host_size_t environment_fact_count;

  /// Borrowed cooperative matrix row fact array.
  const loomc_spirv_cooperative_matrix_row_t* cooperative_matrix_rows;

  /// Number of entries in `cooperative_matrix_rows`.
  loomc_host_size_t cooperative_matrix_row_count;

  /// Borrowed cooperative vector row fact array.
  const loomc_spirv_cooperative_vector_row_t* cooperative_vector_rows;

  /// Number of entries in `cooperative_vector_rows`.
  loomc_host_size_t cooperative_vector_row_count;
} loomc_spirv_profile_options_t;

/// Prepared SPIR-V profile summary.
typedef struct loomc_spirv_profile_info_t {
  /// Minimum SPIR-V binary version selected by known-true features.
  uint32_t minimum_spirv_version;

  /// SPIR-V addressing model numeric value.
  uint32_t addressing_model;

  /// SPIR-V memory model numeric value.
  uint32_t memory_model;

  /// Number of OpExtension rows selected by known-true features.
  loomc_host_size_t extension_count;

  /// Number of OpCapability rows selected by known-true features.
  loomc_host_size_t capability_count;

  /// Number of opcode rows selected by known-true features.
  loomc_host_size_t opcode_count;

  /// Number of storage-class rows selected by known-true features.
  loomc_host_size_t storage_class_count;

  /// Number of decoration rows selected by known-true features.
  loomc_host_size_t decoration_count;

  /// Number of cooperative matrix rows selected by known-true features.
  loomc_host_size_t cooperative_matrix_row_count;

  /// Number of cooperative vector rows selected by known-true features.
  loomc_host_size_t cooperative_vector_row_count;
} loomc_spirv_profile_info_t;

/// Cooperative matrix operation fact row.
typedef struct loomc_spirv_cooperative_matrix_row_t {
  /// Stable row name for diagnostics, caches, and tests.
  loomc_string_view_t name;

  /// Known availability state for this row.
  loomc_target_fact_state_t state;

  /// Borrowed provenance string used in diagnostics.
  loomc_string_view_t provenance;

  /// Feature bits required before this row is legal.
  loomc_spirv_feature_bits_t required_features;

  /// Result row count and Matrix A row count.
  uint16_t m_size;

  /// Result column count and Matrix B column count.
  uint16_t n_size;

  /// Matrix A column count and Matrix B row count.
  uint16_t k_size;

  /// Matrix A scalar component type.
  loomc_spirv_scalar_type_t lhs_type;

  /// Matrix B scalar component type.
  loomc_spirv_scalar_type_t rhs_type;

  /// Matrix C accumulator scalar component type.
  loomc_spirv_scalar_type_t accumulator_type;

  /// Result scalar component type.
  loomc_spirv_scalar_type_t result_type;

  /// Cooperative matrix scope operand.
  loomc_spirv_scope_t scope;

  /// Accepted load/store memory layouts.
  loomc_spirv_cooperative_matrix_layout_flags_t layout_flags;

  /// Accepted pointer storage classes for load/store operations.
  loomc_spirv_storage_class_flags_t storage_class_flags;

  /// Required cooperative matrix operand bits for this row.
  loomc_spirv_cooperative_matrix_operand_flags_t operand_flags;
} loomc_spirv_cooperative_matrix_row_t;

/// Cooperative vector operation fact row.
typedef struct loomc_spirv_cooperative_vector_row_t {
  /// Stable row name for diagnostics, caches, and tests.
  loomc_string_view_t name;

  /// Known availability state for this row.
  loomc_target_fact_state_t state;

  /// Borrowed provenance string used in diagnostics.
  loomc_string_view_t provenance;

  /// Feature bits required before this row is legal.
  loomc_spirv_feature_bits_t required_features;

  /// Output vector component count.
  uint16_t m_size;

  /// Logical input vector component count.
  uint16_t k_size;

  /// Component type of the Input vector object.
  loomc_spirv_component_type_t input_type;

  /// InputInterpretation operand value.
  loomc_spirv_component_type_t input_interpretation;

  /// MatrixInterpretation operand value.
  loomc_spirv_component_type_t matrix_interpretation;

  /// BiasInterpretation operand value.
  loomc_spirv_component_type_t bias_interpretation;

  /// Component type of the Result vector object.
  loomc_spirv_component_type_t result_type;

  /// Accepted matrix layout operands.
  loomc_spirv_cooperative_vector_matrix_layout_flags_t matrix_layout_flags;

  /// Accepted matrix pointer storage classes.
  loomc_spirv_storage_class_flags_t storage_class_flags;

  /// Additional cooperative vector behavior flags.
  loomc_spirv_cooperative_vector_flags_t flags;
} loomc_spirv_cooperative_vector_row_t;

/// Creates a reusable SPIR-V target profile.
///
/// @param target_environment SPIR-V-capable target environment.
/// @param options Profile facts, or `NULL` for an empty partial profile.
/// @param allocator Host allocator used for profile and result storage.
/// @param out_profile Receives one retained profile when `out_result`
/// succeeds.
/// @param out_result Receives one retained result describing profile
/// preparation.
/// @return OK when profile preparation completed far enough to report a result.
/// Non-OK statuses represent API misuse or infrastructure failures before a
/// result could be produced.
///
/// When known-true facts select the Vulkan 1.3 physical-storage-buffer profile,
/// the returned target profile contains a compiler-facing target bundle whose
/// snapshot includes known numeric limit facts. More partial profiles still
/// carry SPIR-V target data, but let source IR target records select the
/// compiler target snapshot.
///
/// @ownership
/// The caller owns `out_profile` when the returned result succeeds and releases
/// it with `loomc_target_profile_release`. The caller always owns `out_result`
/// on an OK return and releases it with `loomc_result_release`.
///
/// @thread_safety
/// Created profiles are immutable and may be shared across threads.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_spirv(
    loomc_target_environment_t* target_environment,
    const loomc_spirv_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile, loomc_result_t** out_result);

/// Refines a SPIR-V target profile with additional facts.
///
/// @param base_profile Existing SPIR-V profile to refine.
/// @param options Additional profile facts, or `NULL` to clone the base
/// profile.
/// @param allocator Host allocator used for refined profile and result storage.
/// @param out_profile Receives one retained refined profile when `out_result`
/// succeeds.
/// @param out_result Receives one retained result describing profile
/// refinement.
/// @return OK when refinement completed far enough to report a result. Non-OK
/// statuses represent API misuse or infrastructure failures before a result
/// could be produced.
///
/// Refinement never mutates `base_profile`. Known facts from `base_profile`
/// are applied first, then `options->preset`, then explicit option facts in
/// array order. Repeating the same fact is accepted when the known state and
/// value match. Contradictory facts fail `out_result` and preserve provenance
/// from both the base fact and the refining fact. Empty `options->identifier`
/// carries the base profile identifier into the refined profile.
///
/// @ownership
/// The caller owns `out_profile` when the returned result succeeds and releases
/// it with `loomc_target_profile_release`. The caller always owns `out_result`
/// on an OK return and releases it with `loomc_result_release`.
///
/// @thread_safety
/// Refinement reads the immutable base profile and may run concurrently with
/// other queries or refinements of the same profile.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_refine(
    const loomc_target_profile_t* base_profile,
    const loomc_spirv_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile, loomc_result_t** out_result);

/// Returns the known state for one SPIR-V feature fact.
///
/// @param profile SPIR-V target profile to query.
/// @param feature Feature fact to inspect.
/// @param out_state Receives the feature state.
/// @return OK when the feature state was returned.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_query_feature(
    const loomc_target_profile_t* profile, loomc_spirv_feature_t feature,
    loomc_target_fact_state_t* out_state);

/// Returns the known state and value for one SPIR-V numeric limit.
///
/// @param profile SPIR-V target profile to query.
/// @param limit Limit fact to inspect.
/// @param out_value Receives the limit state and value.
/// @return OK when the limit state was returned.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_query_limit(
    const loomc_target_profile_t* profile, loomc_spirv_limit_t limit,
    loomc_spirv_limit_value_t* out_value);

/// Returns the known state and value for one SPIR-V environment fact.
///
/// @param profile SPIR-V target profile to query.
/// @param environment Environment fact to inspect.
/// @param out_value Receives the environment state and value.
/// @return OK when the environment state was returned.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_query_environment(
    const loomc_target_profile_t* profile,
    loomc_spirv_environment_t environment,
    loomc_spirv_environment_value_t* out_value);

/// Returns prepared SPIR-V profile summary rows.
///
/// @param profile SPIR-V target profile to query.
/// @param out_info Receives counts for extension, capability, opcode,
/// storage-class, decoration, and cooperative operation rows.
/// @return OK when summary information was returned.
LOOMC_API_EXPORT loomc_status_t
loomc_spirv_target_profile_query_info(const loomc_target_profile_t* profile,
                                      loomc_spirv_profile_info_t* out_info);

/// Returns an OpExtension row by index.
///
/// @param profile SPIR-V target profile to query.
/// @param index Zero-based extension row index.
/// @param out_extension Receives the extension name.
/// @return OK when the extension row was returned.
///
/// @lifetime
/// The returned string view remains valid until `profile` is released.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_extension_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    loomc_string_view_t* out_extension);

/// Returns an OpCapability numeric row by index.
///
/// @param profile SPIR-V target profile to query.
/// @param index Zero-based capability row index.
/// @param out_capability Receives the SPIR-V capability enumerant value.
/// @return OK when the capability row was returned.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_capability_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_capability);

/// Returns an opcode numeric row by index.
///
/// @param profile SPIR-V target profile to query.
/// @param index Zero-based opcode row index.
/// @param out_opcode Receives the SPIR-V opcode enumerant value.
/// @return OK when the opcode row was returned.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_opcode_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_opcode);

/// Returns a storage-class numeric row by index.
///
/// @param profile SPIR-V target profile to query.
/// @param index Zero-based storage-class row index.
/// @param out_storage_class Receives the SPIR-V storage-class enumerant value.
/// @return OK when the storage-class row was returned.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_storage_class_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_storage_class);

/// Returns a decoration numeric row by index.
///
/// @param profile SPIR-V target profile to query.
/// @param index Zero-based decoration row index.
/// @param out_decoration Receives the SPIR-V decoration enumerant value.
/// @return OK when the decoration row was returned.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_decoration_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_decoration);

/// Returns a cooperative matrix fact row by index.
///
/// @param profile SPIR-V target profile to query.
/// @param index Zero-based cooperative matrix row index.
/// @param out_row Receives the cooperative matrix fact row.
/// @return OK when the cooperative matrix row was returned.
///
/// @lifetime
/// The returned string view remains valid until `profile` is released.
LOOMC_API_EXPORT loomc_status_t
loomc_spirv_target_profile_cooperative_matrix_row_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    loomc_spirv_cooperative_matrix_row_t* out_row);

/// Returns a cooperative vector fact row by index.
///
/// @param profile SPIR-V target profile to query.
/// @param index Zero-based cooperative vector row index.
/// @param out_row Receives the cooperative vector fact row.
/// @return OK when the cooperative vector row was returned.
///
/// @lifetime
/// The returned string view remains valid until `profile` is released.
LOOMC_API_EXPORT loomc_status_t
loomc_spirv_target_profile_cooperative_vector_row_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    loomc_spirv_cooperative_vector_row_t* out_row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_SPIRV_PROFILE_H_
