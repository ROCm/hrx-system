// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/lower/source_representation.h"

#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/descriptors.h"

enum loom_test_low_source_representation_string_offset_e {
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CANONICAL = 0,
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ALTERNATE =
      LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CANONICAL +
      sizeof("test.canonical"),
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ADDI_GROUP =
      LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ALTERNATE +
      sizeof("test.alternate"),
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CAST_GROUP =
      LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ADDI_GROUP +
      sizeof("test.addi.frame"),
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ADDI_CANONICAL =
      LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CAST_GROUP +
      sizeof("test.cast.transition"),
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ADDI_ALTERNATE =
      LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ADDI_CANONICAL +
      sizeof("test.addi.canonical"),
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CAST_CANONICAL =
      LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ADDI_ALTERNATE +
      sizeof("test.addi.alternate"),
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CAST_ALTERNATE =
      LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CAST_CANONICAL +
      sizeof("test.cast.canonical"),
};

static const uint8_t kStrings[] = LOOM_BSTRING_LITERAL(14, "test.canonical")
    LOOM_BSTRING_LITERAL(14, "test.alternate")
        LOOM_BSTRING_LITERAL(15, "test.addi.frame")
            LOOM_BSTRING_LITERAL(20, "test.cast.transition")
                LOOM_BSTRING_LITERAL(19, "test.addi.canonical")
                    LOOM_BSTRING_LITERAL(19, "test.addi.alternate")
                        LOOM_BSTRING_LITERAL(19, "test.cast.canonical")
                            LOOM_BSTRING_LITERAL(19, "test.cast.alternate");

static const loom_low_source_representation_t kRepresentations[] = {
    {
        .stable_key = LOOM_TEST_LOW_SOURCE_REPRESENTATION_CANONICAL_KEY,
        .name_string_offset =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CANONICAL,
    },
    {
        .stable_key = LOOM_TEST_LOW_SOURCE_REPRESENTATION_ALTERNATE_KEY,
        .name_string_offset =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ALTERNATE,
    },
};

static const uint16_t kTestOperationIndices[LOOM_OP_TEST_COUNT_] = {
    [LOOM_OP_TEST_ADDI & 0xFFu] = 1,
    [LOOM_OP_TEST_CAST & 0xFFu] = 2,
};

static const uint16_t kScalarOperationIndices[LOOM_OP_SCALAR_COUNT_] = {
    [LOOM_OP_SCALAR_ADDI & 0xFFu] = 1,
};

static const loom_low_source_representation_dialect_table_t kDialectTables[] = {
    {
        .operation_count = IREE_ARRAYSIZE(kTestOperationIndices),
        .operation_indices = kTestOperationIndices,
    },
    {
        .operation_count = IREE_ARRAYSIZE(kScalarOperationIndices),
        .operation_indices = kScalarOperationIndices,
    },
};

static const loom_low_source_representation_port_t kPorts[] = {
    // test.addi
    {LOOM_LOW_SOURCE_REPRESENTATION_PORT_OPERAND_FIELD, 0, 0, 0, 0, 0},
    {LOOM_LOW_SOURCE_REPRESENTATION_PORT_OPERAND_FIELD, 1, 0, 0, 0, 0},
    {LOOM_LOW_SOURCE_REPRESENTATION_PORT_RESULT_FIELD, 0, 0, 0, 0, 0},
    // test.cast
    {LOOM_LOW_SOURCE_REPRESENTATION_PORT_OPERAND_FIELD, 0, 0, 0, 0, 0},
    {LOOM_LOW_SOURCE_REPRESENTATION_PORT_RESULT_FIELD, 0, 1, 0, 0, 0},
};

static const loom_low_source_representation_binding_t kBindings[] = {
    // test.addi canonical and alternate.
    {0, LOOM_LOW_SOURCE_REPRESENTATION_BINDING_CANONICAL, 0},
    {1, 0, 0},
    // test.cast canonical source/destination.
    {0, LOOM_LOW_SOURCE_REPRESENTATION_BINDING_CANONICAL, 0},
    {0, LOOM_LOW_SOURCE_REPRESENTATION_BINDING_CANONICAL, 0},
    // test.cast fixed canonical source and alternate destination.
    {0, LOOM_LOW_SOURCE_REPRESENTATION_BINDING_CANONICAL, 0},
    {1, 0, 0},
};

static const loom_low_descriptor_recipe_entry_t kRecipeEntries[] = {
    {TEST_LOW_CORE_DESCRIPTOR_REF_TEST_MUL_I32, 4},
    {TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32, 1},
    {TEST_LOW_CORE_DESCRIPTOR_REF_TEST_MUL_I32, 4},
    {TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32, 1},
};

static iree_status_t loom_test_low_source_representation_alternate_enabled(
    void* user_data,
    const loom_low_source_representation_environment_t* environment,
    const loom_op_t* source_op, bool* out_matches) {
  (void)user_data;
  (void)source_op;
  const loom_test_low_source_representation_configuration_t* configuration =
      (const loom_test_low_source_representation_configuration_t*)
          environment->configuration;
  *out_matches =
      configuration != NULL &&
      iree_any_bit_set(configuration->flags,
                       LOOM_TEST_LOW_SOURCE_REPRESENTATION_ENABLE_ALTERNATE);
  return iree_ok_status();
}

static const loom_low_source_representation_predicate_t kPredicates[] = {
    {.fn = loom_test_low_source_representation_alternate_enabled},
};

typedef struct loom_test_low_source_representation_operation_data_t {
  // Normalized operation class captured once for all candidate matches.
  loom_test_low_source_representation_operation_t operation;
} loom_test_low_source_representation_operation_data_t;

static iree_status_t loom_test_low_source_representation_prepare_operation(
    void* user_data,
    const loom_low_source_representation_environment_t* environment,
    const loom_op_t* source_op, iree_arena_allocator_t* arena,
    const void** out_operation_data) {
  (void)user_data;
  (void)environment;
  loom_test_low_source_representation_operation_data_t* operation_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*operation_data),
                                           (void**)&operation_data));
  switch (source_op->kind) {
    case LOOM_OP_TEST_ADDI:
    case LOOM_OP_SCALAR_ADDI:
      operation_data->operation =
          LOOM_TEST_LOW_SOURCE_REPRESENTATION_OPERATION_ADDI;
      break;
    case LOOM_OP_TEST_CAST:
      operation_data->operation =
          LOOM_TEST_LOW_SOURCE_REPRESENTATION_OPERATION_CAST;
      break;
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "test source representation provider received operation kind %u",
          source_op->kind);
  }
  *out_operation_data = operation_data;
  return iree_ok_status();
}

static iree_status_t loom_test_low_source_representation_match_candidate(
    void* user_data,
    const loom_low_source_representation_environment_t* environment,
    const loom_op_t* source_op, const void* opaque_operation_data,
    const void* opaque_target_data,
    loom_low_source_representation_candidate_match_t* out_match) {
  (void)user_data;
  (void)environment;
  (void)source_op;
  const loom_test_low_source_representation_operation_data_t* operation_data =
      (const loom_test_low_source_representation_operation_data_t*)
          opaque_operation_data;
  const loom_test_low_source_representation_target_data_t* target_data =
      (const loom_test_low_source_representation_target_data_t*)
          opaque_target_data;
  if (operation_data == NULL || target_data == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "test source representation matching requires prepared operation and "
        "target data");
  }
  const bool matches = operation_data->operation == target_data->operation;
  *out_match = (loom_low_source_representation_candidate_match_t){
      .fallback_rank = target_data->fallback_rank,
      .target_variant = target_data->realization,
      .rejection_bits = matches ? 0u : (uint32_t)1u << target_data->realization,
      .rejection_rank =
          matches ? 0u : (uint8_t)(target_data->fallback_rank + 1),
      .matches = matches,
  };
  return iree_ok_status();
}

static const loom_low_source_representation_candidate_t kCandidates[] = {
    {
        .stable_key = UINT64_C(0x073f328929d07320),
        .name_string_offset =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ADDI_CANONICAL,
        .target_data_ordinal =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_ADDI_CANONICAL,
        .binding_start = 0,
        .binding_count = 1,
        .recipe_entry_start = 0,
        .recipe_entry_count = 1,
        .proof = LOOM_LOW_SOURCE_REPRESENTATION_PROOF_EXACT,
    },
    {
        .stable_key = UINT64_C(0x2b020723225b48c4),
        .name_string_offset =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ADDI_ALTERNATE,
        .target_data_ordinal =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_ADDI_ALTERNATE,
        .binding_start = 1,
        .binding_count = 1,
        .recipe_entry_start = 1,
        .recipe_entry_count = 1,
        .predicate_index_plus_one = 1,
        .proof = LOOM_LOW_SOURCE_REPRESENTATION_PROOF_EXACT,
    },
    {
        .stable_key = UINT64_C(0x4b71fbb9c6161290),
        .name_string_offset =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CAST_CANONICAL,
        .target_data_ordinal =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_CAST_CANONICAL,
        .binding_start = 2,
        .binding_count = 2,
        .recipe_entry_start = 2,
        .recipe_entry_count = 1,
        .proof = LOOM_LOW_SOURCE_REPRESENTATION_PROOF_EXACT,
    },
    {
        .stable_key = UINT64_C(0x0a6f19fb8441784e),
        .name_string_offset =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CAST_ALTERNATE,
        .target_data_ordinal =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_CAST_ALTERNATE,
        .binding_start = 4,
        .binding_count = 2,
        .recipe_entry_start = 3,
        .recipe_entry_count = 1,
        .predicate_index_plus_one = 1,
        .proof = LOOM_LOW_SOURCE_REPRESENTATION_PROOF_EXACT,
    },
};

static const loom_low_source_representation_group_t kGroups[] = {
    {
        .stable_key = LOOM_TEST_LOW_SOURCE_REPRESENTATION_ADDI_GROUP_KEY,
        .name_string_offset =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_ADDI_GROUP,
        .port_start = 0,
        .candidate_start = 0,
        .port_count = 3,
        .candidate_count = 2,
        .component_count = 1,
    },
    {
        .stable_key = LOOM_TEST_LOW_SOURCE_REPRESENTATION_CAST_GROUP_KEY,
        .name_string_offset =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_STRING_CAST_GROUP,
        .port_start = 3,
        .candidate_start = 2,
        .port_count = 2,
        .candidate_count = 2,
        .component_count = 2,
    },
};

static const loom_low_source_representation_operation_t kOperations[] = {
    {.group_start = 0, .group_count = 1},
    {.group_start = 1, .group_count = 1},
};

static const loom_test_low_source_representation_target_data_t kTargetData[] = {
    {
        .realization =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_ADDI_CANONICAL,
        .fallback_rank = 1,
        .operation = LOOM_TEST_LOW_SOURCE_REPRESENTATION_OPERATION_ADDI,
    },
    {
        .realization =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_ADDI_ALTERNATE,
        .fallback_rank = 0,
        .operation = LOOM_TEST_LOW_SOURCE_REPRESENTATION_OPERATION_ADDI,
    },
    {
        .realization =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_CAST_CANONICAL,
        .fallback_rank = 1,
        .operation = LOOM_TEST_LOW_SOURCE_REPRESENTATION_OPERATION_CAST,
    },
    {
        .realization =
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_CAST_ALTERNATE,
        .fallback_rank = 0,
        .operation = LOOM_TEST_LOW_SOURCE_REPRESENTATION_OPERATION_CAST,
    },
};

static iree_status_t loom_test_low_source_representation_seed_value(
    void* user_data,
    const loom_low_source_representation_environment_t* environment,
    loom_value_id_t value_id,
    loom_low_source_representation_domain_t* out_domain) {
  (void)user_data;
  *out_domain = (loom_low_source_representation_domain_t){
      .canonical_representation_index =
          LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE,
  };
  const loom_type_t type =
      loom_module_value_type(environment->module, value_id);
  if (!loom_type_is_scalar(type)) return iree_ok_status();
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (element_type != LOOM_SCALAR_TYPE_I32 &&
      element_type != LOOM_SCALAR_TYPE_F32) {
    return iree_ok_status();
  }
  const loom_test_low_source_representation_configuration_t* configuration =
      (const loom_test_low_source_representation_configuration_t*)
          environment->configuration;
  if (element_type == LOOM_SCALAR_TYPE_I32 && configuration != NULL &&
      iree_any_bit_set(configuration->flags,
                       LOOM_TEST_LOW_SOURCE_REPRESENTATION_EMPTY_I32_DOMAIN)) {
    out_domain->constrained = true;
    return iree_ok_status();
  }
  static const uint16_t kDomain[] = {0, 1};
  *out_domain = (loom_low_source_representation_domain_t){
      .representation_indices = kDomain,
      .representation_count = IREE_ARRAYSIZE(kDomain),
      .canonical_representation_index = 0,
      .constrained = true,
  };
  return iree_ok_status();
}

const loom_low_source_representation_provider_t
    loom_test_low_source_representation_provider = {
        .name = IREE_SVL("test-low-source-representation"),
        .string_table =
            {
                .data = kStrings,
                .data_length = sizeof(kStrings) - 1,
            },
        .representation_count = IREE_ARRAYSIZE(kRepresentations),
        .representations = kRepresentations,
        .dialect_base_id = LOOM_DIALECT_TEST,
        .dialect_count = IREE_ARRAYSIZE(kDialectTables),
        .dialects = kDialectTables,
        .operation_count = IREE_ARRAYSIZE(kOperations),
        .operations = kOperations,
        .group_count = IREE_ARRAYSIZE(kGroups),
        .groups = kGroups,
        .port_count = IREE_ARRAYSIZE(kPorts),
        .ports = kPorts,
        .candidate_count = IREE_ARRAYSIZE(kCandidates),
        .candidates = kCandidates,
        .binding_count = IREE_ARRAYSIZE(kBindings),
        .bindings = kBindings,
        .recipe_entry_count = IREE_ARRAYSIZE(kRecipeEntries),
        .recipe_entries = kRecipeEntries,
        .predicate_count = IREE_ARRAYSIZE(kPredicates),
        .predicates = kPredicates,
        .candidate_matcher =
            {
                .prepare_operation =
                    loom_test_low_source_representation_prepare_operation,
                .match_candidate =
                    loom_test_low_source_representation_match_candidate,
            },
        .target_data_stride = sizeof(kTargetData[0]),
        .target_data_count = IREE_ARRAYSIZE(kTargetData),
        .target_data = (const uint8_t*)kTargetData,
        .seed_value = {.fn = loom_test_low_source_representation_seed_value},
};
