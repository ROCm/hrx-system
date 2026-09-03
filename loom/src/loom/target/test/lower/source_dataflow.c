// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/lower/source_dataflow.h"

#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"

static iree_status_t loom_test_low_source_dataflow_seed_value(
    void* user_data, const loom_source_dataflow_environment_t* environment,
    loom_value_id_t value_id, loom_source_dataflow_bits_t* out_bits) {
  (void)user_data;
  *out_bits = 0;
  const loom_module_t* module = environment->program->module;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value) &&
      loom_value_def_block(value) ==
          loom_region_const_entry_block(environment->program->root_region)) {
    *out_bits |= LOOM_TEST_LOW_SOURCE_DATAFLOW_ENTRY_DERIVED |
                 LOOM_TEST_LOW_SOURCE_DATAFLOW_ALL_ENTRY_DERIVED;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (loom_type_is_scalar(type) &&
      loom_type_element_type(type) == LOOM_SCALAR_TYPE_I8) {
    *out_bits |= LOOM_TEST_LOW_SOURCE_DATAFLOW_SCALAR_CANDIDATE_REJECTED;
  }
  return iree_ok_status();
}

static iree_status_t loom_test_low_source_dataflow_result_is_i32(
    void* user_data, const loom_source_dataflow_environment_t* environment,
    const loom_op_t* op, bool* out_matches) {
  (void)user_data;
  *out_matches = false;
  if (op->result_count != 1) return iree_ok_status();
  const loom_type_t type = loom_module_value_type(environment->program->module,
                                                  loom_op_const_results(op)[0]);
  *out_matches = loom_type_is_scalar(type) &&
                 loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
  return iree_ok_status();
}

static const uint16_t kTestOperationIndices[LOOM_OP_TEST_COUNT_] = {
    [LOOM_OP_TEST_YIELD & 0xFFu] = 3,
};

static const uint16_t kScalarOperationIndices[LOOM_OP_SCALAR_COUNT_] = {
    [LOOM_OP_SCALAR_ADDI & 0xFFu] = 2,
    [LOOM_OP_SCALAR_CONSTANT & 0xFFu] = 1,
};

static const loom_source_dataflow_dialect_table_t kDialectTables[] = {
    {
        .operation_count = IREE_ARRAYSIZE(kTestOperationIndices),
        .operation_indices = kTestOperationIndices,
    },
    {
        .operation_count = IREE_ARRAYSIZE(kScalarOperationIndices),
        .operation_indices = kScalarOperationIndices,
    },
};

static const loom_source_dataflow_port_t kPorts[] = {
    // scalar.constant
    {LOOM_SOURCE_DATAFLOW_PORT_RESULT_FIELD, 0},
    // scalar.addi
    {LOOM_SOURCE_DATAFLOW_PORT_OPERAND_FIELD, 0},
    {LOOM_SOURCE_DATAFLOW_PORT_OPERAND_FIELD, 1},
    {LOOM_SOURCE_DATAFLOW_PORT_RESULT_FIELD, 0},
    // test.yield
    {LOOM_SOURCE_DATAFLOW_PORT_OPERAND_FIELD, 0},
};

static const loom_source_dataflow_rule_t kRules[] = {
    {
        .target_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_ENTRY_DERIVED |
                       LOOM_TEST_LOW_SOURCE_DATAFLOW_ALL_ENTRY_DERIVED,
        .target_port_mask = 0b1,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_SEED,
        .predicate_index_plus_one = 1,
    },
    {
        .source_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_ENTRY_DERIVED,
        .target_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_ENTRY_DERIVED,
        .source_port_mask = 0b011,
        .target_port_mask = 0b100,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_ANY,
    },
    {
        .source_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_ALL_ENTRY_DERIVED,
        .target_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_ALL_ENTRY_DERIVED,
        .source_port_mask = 0b011,
        .target_port_mask = 0b100,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_ALL,
    },
    {
        .source_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_SCALAR_CANDIDATE_REJECTED,
        .target_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_SCALAR_CANDIDATE_REJECTED,
        .source_port_mask = 0b011,
        .target_port_mask = 0b100,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_ANY,
    },
    {
        .source_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_BOUNDARY_REQUIRED,
        .target_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_BOUNDARY_REQUIRED,
        .source_port_mask = 0b100,
        .target_port_mask = 0b011,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_ANY,
    },
    {
        .target_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_BOUNDARY_REQUIRED,
        .target_port_mask = 0b1,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_SEED,
    },
};

static const loom_source_dataflow_operation_t kOperations[] = {
    {
        .port_start = 0,
        .rule_start = 0,
        .port_count = 1,
        .rule_count = 1,
    },
    {
        .port_start = 1,
        .rule_start = 1,
        .port_count = 3,
        .rule_count = 4,
    },
    {
        .port_start = 4,
        .rule_start = 5,
        .port_count = 1,
        .rule_count = 1,
    },
};

static const loom_source_dataflow_predicate_t kPredicates[] = {
    {.fn = loom_test_low_source_dataflow_result_is_i32},
};

const loom_source_dataflow_provider_t loom_test_low_source_dataflow = {
    .name = IREE_SVL("test-low-source-dataflow"),
    .valid_bits = LOOM_TEST_LOW_SOURCE_DATAFLOW_ENTRY_DERIVED |
                  LOOM_TEST_LOW_SOURCE_DATAFLOW_ALL_ENTRY_DERIVED |
                  LOOM_TEST_LOW_SOURCE_DATAFLOW_SCALAR_CANDIDATE_REJECTED |
                  LOOM_TEST_LOW_SOURCE_DATAFLOW_BOUNDARY_REQUIRED,
    .structural_copy_bits =
        LOOM_TEST_LOW_SOURCE_DATAFLOW_ENTRY_DERIVED |
        LOOM_TEST_LOW_SOURCE_DATAFLOW_ALL_ENTRY_DERIVED |
        LOOM_TEST_LOW_SOURCE_DATAFLOW_SCALAR_CANDIDATE_REJECTED |
        LOOM_TEST_LOW_SOURCE_DATAFLOW_BOUNDARY_REQUIRED,
    .dialect_base_id = LOOM_DIALECT_TEST,
    .dialect_count = IREE_ARRAYSIZE(kDialectTables),
    .operation_count = IREE_ARRAYSIZE(kOperations),
    .dialects = kDialectTables,
    .operations = kOperations,
    .port_count = IREE_ARRAYSIZE(kPorts),
    .ports = kPorts,
    .rule_count = IREE_ARRAYSIZE(kRules),
    .rules = kRules,
    .predicate_count = IREE_ARRAYSIZE(kPredicates),
    .predicates = kPredicates,
    .seed_value = {.fn = loom_test_low_source_dataflow_seed_value},
};
