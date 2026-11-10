// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/type.h"

#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/format/bytecode/reader/type_validator.h"
#include "loom/ops/type_registry.h"

namespace loom {
namespace {

struct DiagnosticSnapshot {
  // Static error descriptor identifying the failed validation rule.
  const loom_error_def_t* error = nullptr;
  // First byte of the diagnosed wire range.
  iree_host_size_t start = 0;
  // Exclusive end of the diagnosed wire range.
  iree_host_size_t end = 0;
};

static iree_status_t CaptureDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  auto* snapshot = static_cast<DiagnosticSnapshot*>(user_data);
  *snapshot = {diagnostic->error, diagnostic->origin.start,
               diagnostic->origin.end};
  return iree_ok_status();
}

static const loom_attr_descriptor_t kParameters[] = {
    {
        .name = LOOM_BSTRING_REF(5, "first"),
        .attr_kind = LOOM_ATTR_I64,
        .flags = 0,
    },
    {
        .name = LOOM_BSTRING_REF(6, "middle"),
        .attr_kind = LOOM_ATTR_I64,
        .flags = LOOM_ATTR_OPTIONAL,
    },
    {
        .name = LOOM_BSTRING_REF(4, "last"),
        .attr_kind = LOOM_ATTR_I64,
        .flags = 0,
    },
};
extern const loom_parameterized_type_descriptor_t kParameterDescriptor;
static const loom_type_descriptor_t kTypeDescriptor = {
    .name = LOOM_BSTRING_REF(15, "wire.parameters<"),
    .ir_kind = LOOM_TYPE_PARAMETERIZED,
    .param_count = IREE_ARRAYSIZE(kParameters),
    .fact_domain = nullptr,
    .semantics = {},
    .format_elements = nullptr,
    .format_element_count = 0,
    .parameterized = &kParameterDescriptor,
};
const loom_parameterized_type_descriptor_t kParameterDescriptor = {
    .type_descriptor = &kTypeDescriptor,
    .parameter_descriptors = kParameters,
    .fact_domain = nullptr,
    .ir_kind = LOOM_TYPE_PARAMETERIZED,
    .type_flags = 0,
    .parameter_count = IREE_ARRAYSIZE(kParameters),
    .flags = 0,
};
static const loom_type_registry_entry_t kTypeRegistry[] = {
    {IREE_SV("wire.parameters"), &kTypeDescriptor},
};

class BytecodeTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_type_registry_register_types(
        &context_, kTypeRegistry, IREE_ARRAYSIZE(kTypeRegistry)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("type_test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{CaptureDiagnostic, &diagnostic_},
        IREE_SV("type_test.loombc"), &error_count_, &decoder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t BuildPlan(const uint8_t* data, iree_host_size_t length) {
    loom_bytecode_type_validation_t validation;
    return Validate(data, length, LOOM_BYTECODE_TYPE_RETAIN_PLAN,
                    &scratch_arena_, &validation);
  }

  iree_status_t Validate(const uint8_t* data, iree_host_size_t length,
                         loom_bytecode_type_retention_t retention,
                         iree_arena_allocator_t* arena,
                         loom_bytecode_type_validation_t* out_validation) {
    IREE_RETURN_IF_ERROR(loom_bytecode_type_validation_begin(
        &decoder_, &context_, &module_view_,
        iree_make_const_byte_span(data, length), 0, retention, arena,
        out_validation));
    return loom_bytecode_type_validation_finish(out_validation);
  }

  iree_status_t DecodeEntry(loom_type_id_t type_index, const uint8_t* data,
                            iree_host_size_t length, uint64_t absolute_offset,
                            loom_bytecode_type_plan_entry_t* out_entry,
                            loom_bytecode_type_fact_t** out_fact) {
    return loom_bytecode_type_plan_decode_indexed_entry(
        &decoder_, &context_, &module_view_, &scratch_arena_, type_index,
        iree_make_const_byte_span(data, length), absolute_offset, out_entry,
        out_fact);
  }

  loom_bytecode_type_materializer_t MakeMaterializer(const uint8_t* data,
                                                     iree_host_size_t length) {
    return loom_bytecode_type_materializer_t{
        .decoder = &decoder_,
        .bytecode = iree_make_const_byte_span(data, length),
        .context = &context_,
        .module_view = &module_view_,
        .scratch_arena = &scratch_arena_,
        .output_module = module_,
        .position = 0,
        .next_fact = module_view_.types.facts,
    };
  }

  void CheckValidationModes(const uint8_t* data, iree_host_size_t length,
                            iree_status_code_t expected_status) {
    module_view_.types = {};
    error_count_ = 0;
    diagnostic_ = {};
    auto status = BuildPlan(data, length);
    const auto plan_status = iree_status_code(status);
    iree_status_ignore(status);
    ASSERT_EQ(plan_status, expected_status);
    const DiagnosticSnapshot expected_diagnostic = diagnostic_;
    const auto expected_count = module_view_.types.count;
    std::vector<uint64_t> offsets;
    if (expected_status == IREE_STATUS_OK) {
      for (iree_host_size_t i = 0; i < expected_count; ++i) {
        offsets.push_back(module_view_.types.entries[i].bytecode_offset);
      }
      offsets.push_back(length);
    }
    iree_arena_reset(&scratch_arena_);
    module_view_.types = {};
    error_count_ = 0;
    diagnostic_ = {};
    loom_bytecode_type_validation_t validation;
    status = Validate(data, length, LOOM_BYTECODE_TYPE_RETAIN_NONE, nullptr,
                      &validation);
    EXPECT_EQ(iree_status_code(status), expected_status);
    iree_status_ignore(status);
    EXPECT_EQ(module_view_.types.count, expected_count);
    EXPECT_EQ(module_view_.types.entries, nullptr);
    EXPECT_EQ(module_view_.types.facts, nullptr);
    EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
    EXPECT_EQ(diagnostic_.error, expected_diagnostic.error);
    EXPECT_EQ(diagnostic_.start, expected_diagnostic.start);
    EXPECT_EQ(diagnostic_.end, expected_diagnostic.end);

    module_view_.types = {};
    error_count_ = 0;
    diagnostic_ = {};
    iree_arena_allocator_t retained_arena;
    iree_arena_initialize(&block_pool_, &retained_arena);
    status = Validate(data, length, LOOM_BYTECODE_TYPE_RETAIN_RANGES,
                      &retained_arena, &validation);
    EXPECT_EQ(iree_status_code(status), expected_status);
    iree_status_ignore(status);
    EXPECT_EQ(module_view_.types.entries, nullptr);
    EXPECT_EQ(module_view_.types.facts, nullptr);
    EXPECT_EQ(diagnostic_.error, expected_diagnostic.error);
    EXPECT_EQ(diagnostic_.start, expected_diagnostic.start);
    EXPECT_EQ(diagnostic_.end, expected_diagnostic.end);
    if (expected_status == IREE_STATUS_OK) {
      EXPECT_EQ(validation.position, expected_count);
      EXPECT_EQ(retained_arena.used_allocation_size,
                expected_count * sizeof(*validation.entries));
      // Index entries remain usable after validation scratch is reset.
      iree_arena_reset(&scratch_arena_);
      for (iree_host_size_t i = 0; i < expected_count; ++i) {
        EXPECT_EQ(validation.entries[i].entry_offset, offsets[i]);
        EXPECT_EQ(validation.entries[i].entry_length,
                  offsets[i + 1] - offsets[i]);
      }
    }
    iree_arena_deinitialize(&retained_arena);
  }

  // Last diagnostic's stable identity and wire range.
  DiagnosticSnapshot diagnostic_;
  // Minimal validated module facts receiving each test's type plan.
  loom_bytecode_reader_module_view_t module_view_ = {};
  // Bounded wire decoder sharing this fixture's diagnostic count.
  loom_bytecode_reader_decoder_t decoder_ = {};
  // Number of accepted malformed-input diagnostics.
  uint32_t error_count_ = 0;
  // Block source shared by scratch and output-module arenas.
  iree_arena_block_pool_t block_pool_;
  // Storage owning the immutable plan and temporary type payloads.
  iree_arena_allocator_t scratch_arena_;
  // Finalized registry with one descriptor-backed wire fixture family.
  loom_context_t context_;
  // Output module receiving canonical materialized types.
  loom_module_t* module_ = nullptr;
};

TEST_F(BytecodeTypeTest, RejectsCountBeforeRetainingPlan) {
  const uint8_t data[] = {
      0x02,                     // Two types.
      LOOM_BYTECODE_TYPE_NONE,  // Only one minimum record.
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED, BuildPlan(data, sizeof(data)));

  EXPECT_EQ(error_count_, 1u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
}

TEST_F(BytecodeTypeTest, RejectsFunctionChildCountBeforeRetainingFact) {
  const uint8_t data[] = {
      LOOM_BYTECODE_TYPE_FUNCTION,
      0x02,  // Two arguments.
      0x00,  // No results or child references.
  };
  loom_bytecode_type_plan_entry_t entry = {};
  loom_bytecode_type_fact_t* fact = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        DecodeEntry(/*type_index=*/0, data, sizeof(data),
                                    /*absolute_offset=*/0, &entry, &fact));

  EXPECT_EQ(error_count_, 1u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
  EXPECT_EQ(fact, nullptr);
}

TEST_F(BytecodeTypeTest, RejectsDialectChildCountBeforeRetainingFact) {
  iree_string_view_t strings[] = {IREE_SV("example.wrapper")};
  module_view_.strings.values = strings;
  module_view_.strings.count = IREE_ARRAYSIZE(strings);
  const uint8_t data[] = {
      LOOM_BYTECODE_TYPE_DIALECT,
      0x00,  // Type name.
      0x02,  // Two parameters with no references.
  };
  loom_bytecode_type_plan_entry_t entry = {};
  loom_bytecode_type_fact_t* fact = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        DecodeEntry(/*type_index=*/0, data, sizeof(data),
                                    /*absolute_offset=*/0, &entry, &fact));

  EXPECT_EQ(error_count_, 1u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
  EXPECT_EQ(fact, nullptr);
}

TEST_F(BytecodeTypeTest, RejectsParameterizedCountBeforeRetainingFact) {
  iree_string_view_t strings[] = {IREE_SV("wire.parameters")};
  module_view_.strings.values = strings;
  module_view_.strings.count = IREE_ARRAYSIZE(strings);
  const uint8_t data[] = {
      LOOM_BYTECODE_TYPE_PARAMETERIZED,
      0x00,  // Family name.
      0x02,  // Two parameters with no records.
  };
  loom_bytecode_type_plan_entry_t entry = {};
  loom_bytecode_type_fact_t* fact = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        DecodeEntry(/*type_index=*/0, data, sizeof(data),
                                    /*absolute_offset=*/0, &entry, &fact));

  EXPECT_EQ(error_count_, 1u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
  EXPECT_EQ(fact, nullptr);
}

TEST_F(BytecodeTypeTest, BuildsAndMaterializesTopologicalPlan) {
  const uint8_t data[] = {
      0x03,
      LOOM_BYTECODE_TYPE_NONE,
      LOOM_BYTECODE_TYPE_SCALAR,
      LOOM_SCALAR_TYPE_I32,
      LOOM_BYTECODE_TYPE_FUNCTION,
      0x01,
      0x01,
      0x01,
      0x00,
  };
  IREE_ASSERT_OK(BuildPlan(data, sizeof(data)));

  ASSERT_EQ(module_view_.types.count, 3u);
  EXPECT_EQ(module_view_.types.entries[0].bytecode_offset, 1u);
  EXPECT_EQ(module_view_.types.entries[1].bytecode_offset, 2u);
  EXPECT_EQ(module_view_.types.entries[2].bytecode_offset, 4u);
  ASSERT_NE(module_view_.types.facts, nullptr);
  EXPECT_EQ(module_view_.types.facts->type_id, 2u);
  EXPECT_EQ(module_view_.types.facts->next, nullptr);

  loom_bytecode_type_materializer_t materializer =
      MakeMaterializer(data, sizeof(data));
  IREE_ASSERT_OK(loom_bytecode_type_materialize_prefix(
      &materializer, module_view_.types.count));
  ASSERT_EQ(module_->types.count, 3u);
  for (loom_type_id_t i = 0; i < module_view_.types.count; ++i) {
    EXPECT_EQ(module_view_.types.entries[i].completed_type, i);
  }
  EXPECT_EQ(loom_type_kind(loom_type_table_get(&module_->types, 0)),
            LOOM_TYPE_NONE);
  EXPECT_EQ(loom_type_kind(loom_type_table_get(&module_->types, 1)),
            LOOM_TYPE_SCALAR);
  EXPECT_EQ(loom_type_kind(loom_type_table_get(&module_->types, 2)),
            LOOM_TYPE_FUNCTION);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeTypeTest, AdvancesSparseFactsAndConstructionAcrossPrefixes) {
  const uint8_t data[] = {
      5,
      LOOM_BYTECODE_TYPE_NONE,
      LOOM_BYTECODE_TYPE_SCALAR,
      LOOM_SCALAR_TYPE_I32,
      LOOM_BYTECODE_TYPE_FUNCTION,
      1,
      0,
      1,
      LOOM_BYTECODE_TYPE_SCALAR,
      LOOM_SCALAR_TYPE_F32,
      LOOM_BYTECODE_TYPE_FUNCTION,
      1,
      1,
      2,
      3,
  };
  loom_bytecode_type_validation_t validation;
  IREE_ASSERT_OK(loom_bytecode_type_validation_begin(
      &decoder_, &context_, &module_view_,
      iree_make_const_byte_span(data, sizeof(data)), 0,
      LOOM_BYTECODE_TYPE_RETAIN_PLAN, &scratch_arena_, &validation));
  IREE_ASSERT_OK(loom_bytecode_type_validation_advance(&validation, 0));
  EXPECT_EQ(validation.position, 0u);
  EXPECT_EQ(module_view_.types.facts, nullptr);
  IREE_ASSERT_OK(loom_bytecode_type_validation_advance(&validation, 2));
  EXPECT_EQ(validation.position, 2u);
  EXPECT_EQ(module_view_.types.facts, nullptr);
  IREE_ASSERT_OK(loom_bytecode_type_validation_advance(&validation, 3));
  ASSERT_NE(module_view_.types.facts, nullptr);
  EXPECT_EQ(module_view_.types.facts->type_id, 2u);
  EXPECT_EQ(module_view_.types.facts->next, nullptr);
  IREE_ASSERT_OK(loom_bytecode_type_validation_advance(&validation, 3));
  IREE_ASSERT_OK(loom_bytecode_type_validation_finish(&validation));
  EXPECT_EQ(validation.position, 5u);
  ASSERT_NE(module_view_.types.facts->next, nullptr);
  EXPECT_EQ(module_view_.types.facts->next->type_id, 4u);
  EXPECT_EQ(module_view_.types.facts->next->next, nullptr);

  auto materializer = MakeMaterializer(data, sizeof(data));
  IREE_ASSERT_OK(loom_bytecode_type_materialize_prefix(&materializer, 2));
  EXPECT_EQ(materializer.position, 2u);
  EXPECT_EQ(materializer.next_fact, module_view_.types.facts);
  IREE_ASSERT_OK(loom_bytecode_type_materialize_prefix(&materializer, 3));
  EXPECT_EQ(materializer.next_fact, module_view_.types.facts->next);
  IREE_ASSERT_OK(loom_bytecode_type_materialize_prefix(&materializer, 3));
  IREE_ASSERT_OK(loom_bytecode_type_materialize_prefix(&materializer, 5));
  EXPECT_EQ(materializer.position, 5u);
  EXPECT_EQ(materializer.next_fact, nullptr);
  ASSERT_EQ(module_->types.count, 5u);
  const auto* signature =
      loom_type_func_data(loom_type_table_get(&module_->types, 4));
  ASSERT_NE(signature, nullptr);
  EXPECT_EQ(signature->arg_count, 1u);
  EXPECT_EQ(signature->result_count, 1u);
  EXPECT_TRUE(loom_type_equal(signature->types[0],
                              loom_type_table_get(&module_->types, 2)));
  EXPECT_TRUE(loom_type_equal(signature->types[1],
                              loom_type_scalar(LOOM_SCALAR_TYPE_F32)));
}

TEST_F(BytecodeTypeTest, StructuralTypesRetainChildrenBeyondPlanLifetime) {
  iree_string_view_t strings[] = {IREE_SV("example.wrapper")};
  module_view_.strings = {strings, IREE_ARRAYSIZE(strings)};
  const uint8_t data[] = {
      4,
      LOOM_BYTECODE_TYPE_SCALAR,
      LOOM_SCALAR_TYPE_I32,
      LOOM_BYTECODE_TYPE_FUNCTION,
      1,
      1,
      0,
      0,
      LOOM_BYTECODE_TYPE_DIALECT,
      0,
      1,
      1,
      LOOM_BYTECODE_TYPE_REGISTER,
      1,
      0x80,
      0x80,
      0x04,
      1,
      2,
  };
  IREE_ASSERT_OK(BuildPlan(data, sizeof(data)));

  loom_type_id_t unused_type_id = LOOM_TYPE_ID_INVALID;
  loom_type_id_t scalar_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_I64), &unused_type_id));
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &scalar_type_id));
  ASSERT_NE(scalar_type_id, 0u);
  loom_string_id_t unused_name_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("unused"), &unused_name_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, strings[0], &target_name_id));
  ASSERT_NE(target_name_id, 0u);

  const loom_bytecode_type_fact_t* fact = module_view_.types.facts;
  ASSERT_NE(fact, nullptr);
  ASSERT_EQ(fact->kind, LOOM_TYPE_FUNCTION);
  const loom_type_id_t signature_ids[] = {scalar_type_id, scalar_type_id};
  loom_type_id_t function_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_type_materialize_structural(
      &module_view_.types.entries[1].structural,
      reinterpret_cast<const loom_bytecode_structural_type_fact_t*>(fact),
      signature_ids, module_, &function_type_id));

  fact = fact->next;
  ASSERT_NE(fact, nullptr);
  ASSERT_EQ(fact->kind, LOOM_TYPE_DIALECT);
  loom_bytecode_structural_type_plan_t dialect_plan =
      module_view_.types.entries[2].structural;
  dialect_plan.name_id = target_name_id;
  loom_type_id_t dialect_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_type_materialize_structural(
      &dialect_plan,
      reinterpret_cast<const loom_bytecode_structural_type_fact_t*>(fact),
      &function_type_id, module_, &dialect_type_id));

  fact = fact->next;
  ASSERT_NE(fact, nullptr);
  ASSERT_EQ(fact->kind, LOOM_TYPE_REGISTER);
  loom_type_id_t register_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_type_materialize_structural(
      &module_view_.types.entries[3].structural,
      reinterpret_cast<const loom_bytecode_structural_type_fact_t*>(fact),
      &dialect_type_id, module_, &register_type_id));
  EXPECT_EQ(fact->next, nullptr);

  // Published metadata and children survive overwriting their source plan.
  iree_arena_reset(&scratch_arena_);
  void* overwritten_plan = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&scratch_arena_, 1024, &overwritten_plan));
  std::memset(overwritten_plan, 0xA5, 1024);

  const loom_func_type_data_t* signature = loom_type_func_data(
      loom_type_table_get(&module_->types, function_type_id));
  ASSERT_NE(signature, nullptr);
  EXPECT_EQ(signature->arg_count, 1u);
  EXPECT_EQ(signature->result_count, 1u);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(
        loom_type_equal(signature->types[i],
                        loom_type_table_get(&module_->types, scalar_type_id)));
  }
  const loom_type_t dialect =
      loom_type_table_get(&module_->types, dialect_type_id);
  EXPECT_EQ(loom_type_dialect_name_id(dialect), target_name_id);
  ASSERT_EQ(loom_type_dialect_param_count(dialect), 1u);
  EXPECT_TRUE(
      loom_type_equal(loom_type_dialect_params(dialect)[0],
                      loom_type_table_get(&module_->types, function_type_id)));
  EXPECT_EQ(loom_type_dialect_params(dialect)[0].dims[0],
            loom_type_table_get(&module_->types, function_type_id).dims[0]);
  const loom_register_type_data_t* carrier = loom_type_register_data(
      loom_type_table_get(&module_->types, register_type_id));
  ASSERT_NE(carrier, nullptr);
  EXPECT_EQ(carrier->carrier_payload0, 1u);
  EXPECT_EQ(carrier->carrier_payload1, UINT64_C(1) << 16);
  EXPECT_TRUE(
      loom_type_equal(carrier->value_type,
                      loom_type_table_get(&module_->types, dialect_type_id)));
  EXPECT_EQ(carrier->value_type.dims[0],
            loom_type_table_get(&module_->types, dialect_type_id).dims[0]);
  for (const auto type_id :
       {function_type_id, dialect_type_id, register_type_id}) {
    loom_type_id_t duplicate_id = LOOM_TYPE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_type_id(
        module_, loom_type_table_get(&module_->types, type_id), &duplicate_id));
    EXPECT_EQ(duplicate_id, type_id);
  }
}

TEST_F(BytecodeTypeTest, RetainsOnlyRequestedFactsForMixedTypes) {
  iree_string_view_t strings[] = {
      IREE_SV("wire.parameters"), IREE_SV("first"),
      IREE_SV("middle"),          IREE_SV("last"),
      IREE_SV("example.wrapper"),
  };
  module_view_.strings = {strings, IREE_ARRAYSIZE(strings)};
  const uint8_t data[] = {
      7,
      LOOM_BYTECODE_TYPE_NONE,
      LOOM_BYTECODE_TYPE_SCALAR,
      LOOM_SCALAR_TYPE_I32,
      LOOM_BYTECODE_TYPE_TILE,
      LOOM_SCALAR_TYPE_I32,
      3,
      LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE,
      0,
      0,
      1,
      0,
      2,
      0,
      3,
      LOOM_BYTECODE_TYPE_FUNCTION,
      1,
      1,
      1,
      2,
      LOOM_BYTECODE_TYPE_DIALECT,
      4,
      1,
      3,
      LOOM_BYTECODE_TYPE_REGISTER,
      1,
      0x80,
      0x80,
      0x04,
      1,
      1,
      LOOM_BYTECODE_TYPE_PARAMETERIZED,
      0,
      2,
      1,
      LOOM_BYTECODE_ATTR_I64,
      2,
      3,
      LOOM_BYTECODE_ATTR_I64,
      4,
  };
  ASSERT_NO_FATAL_FAILURE(
      CheckValidationModes(data, sizeof(data), IREE_STATUS_OK));
  for (iree_host_size_t length = 0; length < sizeof(data); ++length) {
    SCOPED_TRACE(length);
    ASSERT_NO_FATAL_FAILURE(
        CheckValidationModes(data, length, IREE_STATUS_DEFERRED));
  }
}

TEST_F(BytecodeTypeTest, EmptyValidationAndIndexNeedNoStorage) {
  const uint8_t data[] = {0};
  ASSERT_NO_FATAL_FAILURE(
      CheckValidationModes(data, sizeof(data), IREE_STATUS_OK));
}

TEST_F(BytecodeTypeTest, RetentionDoesNotChangeParameterValidation) {
  iree_string_view_t strings[] = {IREE_SV("wire.parameters"), IREE_SV("first"),
                                  IREE_SV("middle"), IREE_SV("last")};
  module_view_.strings = {strings, IREE_ARRAYSIZE(strings)};
  const std::vector<std::vector<uint8_t>> entries = {
      // Both required parameters absent.
      {0},
      // Required leading parameter absent.
      {1, 3, LOOM_BYTECODE_ATTR_I64, 2},
      // Required trailing parameter absent.
      {1, 1, LOOM_BYTECODE_ATTR_I64, 2},
      // Declared parameters out of order.
      {2, 3, LOOM_BYTECODE_ATTR_I64, 2, 1, LOOM_BYTECODE_ATTR_I64, 2},
      // Repeated parameter.
      {2, 1, LOOM_BYTECODE_ATTR_I64, 2, 1, LOOM_BYTECODE_ATTR_I64, 2},
      // Unknown parameter name.
      {1, 0, LOOM_BYTECODE_ATTR_I64, 2},
      // Value kind mismatches its descriptor.
      {1, 1, LOOM_BYTECODE_ATTR_TYPE, 0},
      // Present count exceeds the descriptor.
      {4},
  };
  for (const auto& parameters : entries) {
    std::vector<uint8_t> data = {1, LOOM_BYTECODE_TYPE_PARAMETERIZED, 0};
    data.insert(data.end(), parameters.begin(), parameters.end());
    ASSERT_NO_FATAL_FAILURE(
        CheckValidationModes(data.data(), data.size(), IREE_STATUS_DEFERRED));
  }
}

TEST_F(BytecodeTypeTest, RejectsNonTopologicalTypeReference) {
  const uint8_t data[] = {
      0x01, LOOM_BYTECODE_TYPE_FUNCTION, 0x01, 0x00, 0x00,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED, BuildPlan(data, sizeof(data)));
  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeTypeTest, ParameterKindsAreCheckedWithValidReferences) {
  iree_string_view_t strings[] = {IREE_SV("wire.parameters"), IREE_SV("first"),
                                  IREE_SV("last")};
  module_view_.strings = {strings, IREE_ARRAYSIZE(strings)};
  const uint8_t data[] = {
      2,
      LOOM_BYTECODE_TYPE_SCALAR,
      LOOM_SCALAR_TYPE_I32,
      LOOM_BYTECODE_TYPE_PARAMETERIZED,
      0,
      2,
      1,
      LOOM_BYTECODE_ATTR_TYPE,
      0,
      2,
      LOOM_BYTECODE_ATTR_I64,
      2,
  };
  ASSERT_NO_FATAL_FAILURE(
      CheckValidationModes(data, sizeof(data), IREE_STATUS_DEFERRED));
}

TEST_F(BytecodeTypeTest, MaterializesEmptyFunctionPayload) {
  const uint8_t data[] = {0x01, LOOM_BYTECODE_TYPE_FUNCTION, 0x00, 0x00};
  IREE_ASSERT_OK(BuildPlan(data, sizeof(data)));
  loom_bytecode_type_materializer_t materializer =
      MakeMaterializer(data, sizeof(data));
  IREE_ASSERT_OK(loom_bytecode_type_materialize_prefix(
      &materializer, module_view_.types.count));
  ASSERT_EQ(module_->types.count, 1u);
  const loom_func_type_data_t* payload =
      loom_type_func_data(loom_type_table_get(&module_->types, 0));
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->arg_count, 0u);
  EXPECT_EQ(payload->result_count, 0u);
  EXPECT_EQ(payload->reserved, 0u);
}

TEST_F(BytecodeTypeTest, MaterializesFullWidthFunctionSignature) {
  std::vector<uint8_t> data = {
      0x03,
      LOOM_BYTECODE_TYPE_SCALAR,
      LOOM_SCALAR_TYPE_I32,
      LOOM_BYTECODE_TYPE_SCALAR,
      LOOM_SCALAR_TYPE_F32,
      LOOM_BYTECODE_TYPE_FUNCTION,
      0xFF,
      0xFF,
      0x03,  // UINT16_MAX arguments.
      0xFF,
      0xFF,
      0x03,  // UINT16_MAX results.
  };
  data.insert(data.end(), UINT16_MAX, /*argument_type_id=*/0);
  data.insert(data.end(), UINT16_MAX, /*result_type_id=*/1);
  IREE_ASSERT_OK(BuildPlan(data.data(), data.size()));
  EXPECT_EQ(module_view_.types.entries[2].structural.dependency_count,
            2u * UINT16_MAX);
  loom_bytecode_type_materializer_t materializer =
      MakeMaterializer(data.data(), data.size());
  IREE_ASSERT_OK(loom_bytecode_type_materialize_prefix(
      &materializer, module_view_.types.count));
  ASSERT_EQ(module_->types.count, 3u);
  const loom_func_type_data_t* payload =
      loom_type_func_data(loom_type_table_get(&module_->types, 2));
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->arg_count, UINT16_MAX);
  EXPECT_EQ(payload->result_count, UINT16_MAX);
  EXPECT_EQ(payload->reserved, 0u);
  for (iree_host_size_t i = 0; i < 2u * UINT16_MAX; ++i) {
    EXPECT_TRUE(loom_type_equal(
        payload->types[i],
        loom_type_table_get(&module_->types, i < UINT16_MAX ? 0 : 1)));
  }
}

TEST_F(BytecodeTypeTest, RejectsOversizedFunctionCountsBeforeAllocation) {
  const uint8_t oversized_arguments[] = {
      0x01, LOOM_BYTECODE_TYPE_FUNCTION, 0x80, 0x80, 0x04, 0x00};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      BuildPlan(oversized_arguments, sizeof(oversized_arguments)));
  const uint8_t oversized_results[] = {
      0x01, LOOM_BYTECODE_TYPE_FUNCTION, 0x00, 0x80, 0x80, 0x04};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      BuildPlan(oversized_results, sizeof(oversized_results)));
  EXPECT_EQ(error_count_, 2u);
  EXPECT_EQ(module_->types.count, 0u);
}

TEST_F(BytecodeTypeTest, ViewAlignmentValidationModesAgree) {
  for (uint8_t alignment : {0, 1, 2, 4, 8, 3, 16, 128, 255}) {
    const uint8_t data[] = {
        1,
        LOOM_BYTECODE_TYPE_VIEW,
        LOOM_SCALAR_TYPE_I64,
        /*rank=*/0,
        LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE,
        /*encoding_instance=*/0,
        alignment,
    };
    CheckValidationModes(data, sizeof(data),
                         alignment == 0 || loom_type_view_alignment_is_valid(
                                               LOOM_SCALAR_TYPE_I64, alignment)
                             ? IREE_STATUS_OK
                             : IREE_STATUS_DEFERRED);
  }
}

TEST_F(BytecodeTypeTest, RejectsNoneScalarType) {
  const uint8_t data[] = {
      LOOM_BYTECODE_TYPE_SCALAR,
      LOOM_SCALAR_TYPE_NONE,
  };
  loom_bytecode_type_plan_entry_t entry = {};
  loom_bytecode_type_fact_t* fact = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        DecodeEntry(/*type_index=*/0, data, sizeof(data),
                                    /*absolute_offset=*/0, &entry, &fact));
  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeTypeTest, RejectsNoneShapedElementType) {
  const uint8_t data[] = {
      LOOM_BYTECODE_TYPE_TENSOR,
      LOOM_SCALAR_TYPE_NONE,
      /*rank=*/0,
      LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE,
      /*encoding_instance=*/0,
  };
  loom_bytecode_type_plan_entry_t entry = {};
  loom_bytecode_type_fact_t* fact = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        DecodeEntry(/*type_index=*/0, data, sizeof(data),
                                    /*absolute_offset=*/0, &entry, &fact));
  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeTypeTest, GlobalTypesRequireScopeIndependentEncodings) {
  const uint8_t data[] = {
      /*type_count=*/1,
      LOOM_BYTECODE_TYPE_TENSOR,
      LOOM_SCALAR_TYPE_F32,
      /*rank=*/0,
      LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA,
      /*encoding_instance=*/0,
  };
  CheckValidationModes(data, sizeof(data), IREE_STATUS_DEFERRED);
}

TEST_F(BytecodeTypeTest, DecodesOneIndexedEntry) {
  const uint8_t data[] = {
      LOOM_BYTECODE_TYPE_FUNCTION, 0x01, 0x01, 0x01, 0x00,
  };
  loom_bytecode_type_plan_entry_t entry = {};
  loom_bytecode_type_fact_t* fact = nullptr;
  IREE_ASSERT_OK(DecodeEntry(/*type_index=*/2, data, sizeof(data),
                             /*absolute_offset=*/41, &entry, &fact));

  EXPECT_EQ(entry.bytecode_offset, 41u);
  ASSERT_NE(fact, nullptr);
  EXPECT_EQ(fact->type_id, 2u);
  EXPECT_EQ(fact->kind, LOOM_TYPE_FUNCTION);
  const auto* structural_fact =
      reinterpret_cast<const loom_bytecode_structural_type_fact_t*>(fact);
  loom_func_type_data_t payload_header;
  memcpy(&payload_header, structural_fact->payload_prefix,
         sizeof(payload_header));
  EXPECT_EQ(payload_header.arg_count, 1u);
  EXPECT_EQ(payload_header.result_count, 1u);
  EXPECT_EQ(payload_header.reserved, 0u);
  EXPECT_EQ(entry.structural.dependency_count, 2u);
  EXPECT_EQ(structural_fact->type_ids[0], 1u);
  EXPECT_EQ(structural_fact->type_ids[1], 0u);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeTypeTest, ScopedShapesRetainSourceCoordinatesAcrossRanks) {
  for (loom_type_kind_t kind :
       {LOOM_TYPE_GROUP, LOOM_TYPE_TENSOR, LOOM_TYPE_TILE, LOOM_TYPE_VECTOR,
        LOOM_TYPE_VIEW}) {
    for (uint8_t rank = 0; rank <= LOOM_TYPE_MAX_RANK; ++rank) {
      if (rank == 0 && (kind == LOOM_TYPE_GROUP || kind == LOOM_TYPE_VECTOR)) {
        continue;
      }
      SCOPED_TRACE(::testing::Message()
                   << "kind=" << kind << " rank=" << +rank);
      const bool has_encoding =
          kind != LOOM_TYPE_GROUP && kind != LOOM_TYPE_VECTOR;
      std::vector<uint8_t> data;
      if (kind != LOOM_TYPE_GROUP) {
        data.push_back(LOOM_SCALAR_TYPE_F32);
      }
      data.push_back(rank);
      if (kind != LOOM_TYPE_GROUP) {
        data.push_back(has_encoding ? LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA
                                    : LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE);
        data.push_back(has_encoding ? 2 : 0);
      }
      if (kind == LOOM_TYPE_VIEW) {
        data.push_back(1);
      }
      uint16_t expected_dynamic_axes = 0;
      for (uint8_t axis = 0; axis < rank; ++axis) {
        const bool dynamic = (axis % 3) != 1;
        data.push_back(dynamic);
        // Axis zero is deliberately unbound; subsequent dynamic dimensions
        // use the one-based wire coordinates, independent of any IR module.
        data.push_back(dynamic ? axis : 31);
        expected_dynamic_axes |= uint16_t(dynamic << axis);
      }
      loom_bytecode_reader_cursor_t cursor;
      loom_bytecode_reader_cursor_initialize(data.data(), data.size(), 0,
                                             IREE_SV("SCOPED_TYPES"), &cursor);
      loom_bytecode_scoped_shape_t shape;
      IREE_ASSERT_OK(loom_bytecode_type_read_scoped_shape(
          &decoder_, &cursor, kind, LOOM_TYPE_MAX_RANK,
          /*encoding_count=*/0, &shape));
      EXPECT_EQ(cursor.cursor.position, data.size());
      EXPECT_EQ(shape.rank, rank);
      EXPECT_EQ(shape.element_type, kind == LOOM_TYPE_GROUP
                                        ? LOOM_SCALAR_TYPE_NONE
                                        : LOOM_SCALAR_TYPE_F32);
      EXPECT_EQ(shape.dynamic_axes, expected_dynamic_axes);
      EXPECT_EQ(shape.attachment, has_encoding
                                      ? LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA
                                      : LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE);
      EXPECT_EQ(shape.encoding_ordinal, has_encoding ? 2u : 0u);
      EXPECT_EQ(shape.access_alignment, kind == LOOM_TYPE_VIEW ? 1u : 0u);
      for (uint8_t axis = 0; axis < rank; ++axis) {
        EXPECT_EQ(shape.dimensions[axis], (axis % 3) != 1 ? axis : 31);
      }
      EXPECT_EQ(module_->values.count, 0u);
      EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
    }
  }
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeTypeTest, ScopedShapesRejectInvalidWireFields) {
  struct InvalidShape {
    // Native kind already established by the owning record decoder.
    loom_type_kind_t kind;
    // Wire payload following the kind, with one invalid field per row.
    std::vector<uint8_t> payload;
  };
  const InvalidShape cases[] = {
      {LOOM_TYPE_GROUP, {0}},
      {LOOM_TYPE_GROUP, {LOOM_TYPE_MAX_RANK + 1}},
      {LOOM_TYPE_GROUP, {1, 2, 1}},
      {LOOM_TYPE_GROUP, {1, 1, 4}},
      {LOOM_TYPE_VECTOR, {LOOM_SCALAR_TYPE_F32, 0, 0, 0}},
      {LOOM_TYPE_VECTOR, {LOOM_SCALAR_TYPE_F32, 1, 2, 0, 1, 1}},
      {LOOM_TYPE_TENSOR, {LOOM_SCALAR_TYPE_NONE, 0, 0, 0}},
      {LOOM_TYPE_TENSOR, {LOOM_SCALAR_TYPE_F32, 16, 0, 0}},
      {LOOM_TYPE_TILE, {LOOM_SCALAR_TYPE_F32, 0, 3, 0}},
      {LOOM_TYPE_TILE, {LOOM_SCALAR_TYPE_F32, 0, 0, 1}},
      {LOOM_TYPE_TILE, {LOOM_SCALAR_TYPE_F32, 0, 1, 0}},
      {LOOM_TYPE_TILE, {LOOM_SCALAR_TYPE_F32, 0, 1, 3}},
      {LOOM_TYPE_TILE, {LOOM_SCALAR_TYPE_F32, 0, 2, 3}},
      {LOOM_TYPE_VIEW, {LOOM_SCALAR_TYPE_F32, 0, 0, 0, 3}},
      {LOOM_TYPE_VIEW, {LOOM_SCALAR_TYPE_F32, 0, 0, 0, 0x80, 2}},
      {LOOM_TYPE_VIEW, {LOOM_SCALAR_TYPE_F32, 1, 0, 0, 0, 2, 1}},
      {LOOM_TYPE_VIEW, {LOOM_SCALAR_TYPE_F32, 1, 0, 0, 0, 1, 4}},
      {LOOM_TYPE_VIEW,
       {LOOM_SCALAR_TYPE_F32, 1, 0, 0, 0, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 1}},
  };
  for (const auto& invalid : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << (&invalid - cases));
    loom_bytecode_reader_cursor_t cursor;
    loom_bytecode_reader_cursor_initialize(invalid.payload.data(),
                                           invalid.payload.size(), 0,
                                           IREE_SV("SCOPED_TYPES"), &cursor);
    loom_bytecode_scoped_shape_t shape;
    error_count_ = 0;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DEFERRED,
        loom_bytecode_type_read_scoped_shape(&decoder_, &cursor, invalid.kind,
                                             /*value_count=*/3,
                                             /*encoding_count=*/2, &shape));
    EXPECT_EQ(error_count_, 1u);
    EXPECT_EQ(module_->values.count, 0u);
    EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
  }
}

TEST_F(BytecodeTypeTest, ScopedShapeRejectsEveryTruncatedPrefix) {
  const uint8_t data[] = {LOOM_SCALAR_TYPE_F32, 3, 2, 2, 1, 1, 1, 0, 31, 1, 3};
  for (size_t length = 0; length < sizeof(data); ++length) {
    SCOPED_TRACE(length);
    loom_bytecode_reader_cursor_t cursor;
    loom_bytecode_reader_cursor_initialize(data, length, 0,
                                           IREE_SV("SCOPED_TYPES"), &cursor);
    loom_bytecode_scoped_shape_t shape;
    error_count_ = 0;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DEFERRED,
        loom_bytecode_type_read_scoped_shape(&decoder_, &cursor, LOOM_TYPE_VIEW,
                                             /*value_count=*/3,
                                             /*encoding_count=*/0, &shape));
    EXPECT_EQ(error_count_, 1u);
    EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
  }
}

TEST_F(BytecodeTypeTest, ScopedRecordFramingPreservesItsEnclosingCursor) {
  // Empty extensions select static types; zero-node nonempty extensions may
  // select a type completed by an earlier signature value.
  const uint8_t data[] = {0, 2, 0, 2, 99};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 100,
                                         IREE_SV("SIGNATURE"), &cursor);
  loom_bytecode_reader_cursor_t records;
  uint64_t count = UINT64_MAX;
  IREE_ASSERT_OK(loom_bytecode_type_open_scoped_records(
      &decoder_, &cursor, /*reference=*/0, &records, &count));
  EXPECT_EQ(records.cursor.length, 0u);
  EXPECT_EQ(count, 0u);
  EXPECT_EQ(cursor.cursor.position, 1u);

  IREE_ASSERT_OK(loom_bytecode_type_open_scoped_records(
      &decoder_, &cursor, /*reference=*/1, &records, &count));
  EXPECT_EQ(count, 0u);
  EXPECT_EQ(records.absolute_offset, 102u);
  EXPECT_EQ(records.cursor.length, 2u);
  EXPECT_EQ(cursor.cursor.position, 4u);
  uint64_t root = UINT64_MAX;
  IREE_ASSERT_OK(loom_bytecode_type_close_scoped_records(
      &decoder_, &records, /*completed_count=*/2, &root));
  EXPECT_EQ(root, 1u);
  EXPECT_EQ(records.cursor.position, records.cursor.length);
  EXPECT_EQ(cursor.cursor.data[cursor.cursor.position], 99u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
}

TEST_F(BytecodeTypeTest, ScopedRecordFramingRejectsInvalidExtensions) {
  struct Case {
    // Encoded type reference whose extension is tested.
    uint64_t reference;
    // Extension length, node count, root, and any trailing malformed bytes.
    std::vector<uint8_t> bytes;
  };
  const Case cases[] = {
      {1, {0}},           // Scoped reference without an extension.
      {0, {2, 0, 1}},     // Static reference carrying an extension.
      {3, {2, 0, 1}},     // Only the scoped sentinel introduces an extension.
      {1, {3, 0, 1}},     // Truncated bounded payload.
      {1, {1, 0}},        // Missing root.
      {1, {2, 0, 0}},     // Zero is not a completed root.
      {1, {2, 0, 3}},     // Root outside the completed prefix.
      {1, {3, 0, 2, 9}},  // Bytes after the selected root.
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << (&test_case - cases));
    loom_bytecode_reader_cursor_t cursor;
    loom_bytecode_reader_cursor_initialize(test_case.bytes.data(),
                                           test_case.bytes.size(), 0,
                                           IREE_SV("SIGNATURE"), &cursor);
    loom_bytecode_reader_cursor_t records;
    uint64_t count = 0;
    uint64_t root = UINT64_MAX;
    error_count_ = 0;
    iree_status_t status = loom_bytecode_type_open_scoped_records(
        &decoder_, &cursor, test_case.reference, &records, &count);
    if (iree_status_is_ok(status)) {
      status = loom_bytecode_type_close_scoped_records(
          &decoder_, &records, /*completed_count=*/2, &root);
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED, status);
    EXPECT_EQ(error_count_, 1u);
    EXPECT_EQ(root, UINT64_MAX);
    EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
  }
}

}  // namespace
}  // namespace loom
