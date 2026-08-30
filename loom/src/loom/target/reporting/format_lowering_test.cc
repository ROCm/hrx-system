// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/types.h"
#include "loom/target/math_policy.h"
#include "loom/target/reporting/format.h"

namespace loom {
namespace {

constexpr uint32_t kTestSourceRejectionDetail = 4;

static loom_native_contraction_role_facts_t MakeNativeContractionRoleFacts(
    loom_contract_operand_role_t role, uint16_t element_bit_count,
    uint16_t payload_element_count, uint32_t physical_position_count,
    uint16_t owner_multiplicity) {
  loom_native_contraction_role_facts_t facts = {};
  facts.role = role;
  facts.evidence = LOOM_NATIVE_LAYOUT_EVIDENCE_EXACT;
  facts.element_bit_count = element_bit_count;
  facts.register_count = 8;
  facts.payload_element_count = payload_element_count;
  facts.physical_position_count = physical_position_count;
  facts.logical_coordinate_count = 256;
  facts.owner_multiplicity_minimum = owner_multiplicity;
  facts.owner_multiplicity_maximum = owner_multiplicity;
  return facts;
}

static loom_native_contraction_facts_t MakeNativeContractionFacts() {
  loom_native_contraction_facts_t facts = {};
  facts.shape.block_count = 1;
  facts.shape.result_row_count = 16;
  facts.shape.result_column_count = 16;
  facts.shape.reduction_count = 16;
  facts.participant_count = 32;
  facts.lhs = MakeNativeContractionRoleFacts(LOOM_CONTRACT_OPERAND_ROLE_LHS, 16,
                                             16, 512, 2);
  facts.rhs = MakeNativeContractionRoleFacts(LOOM_CONTRACT_OPERAND_ROLE_RHS, 16,
                                             16, 512, 2);
  facts.accumulator = MakeNativeContractionRoleFacts(
      LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 32, 8, 256, 1);
  facts.result = MakeNativeContractionRoleFacts(
      LOOM_CONTRACT_OPERAND_ROLE_RESULT, 32, 8, 256, 1);
  return facts;
}

static loom_native_transition_owner_factor_t MakeNativeTransitionOwnerFactor(
    loom_native_physical_dimension_t destination_dimension,
    loom_native_physical_dimension_t source_owner_dimension,
    uint32_t destination_divisor, uint32_t destination_modulus,
    uint32_t source_owner_multiplier) {
  loom_native_transition_owner_factor_t factor = {};
  factor.destination_dimension = destination_dimension;
  factor.source_owner_dimension = source_owner_dimension;
  factor.destination_divisor = destination_divisor;
  factor.destination_modulus = destination_modulus;
  factor.source_owner_multiplier = source_owner_multiplier;
  return factor;
}

static const loom_native_contraction_facts_t kNativeContractionFacts =
    MakeNativeContractionFacts();

static const loom_native_transition_owner_factor_t
    kNativeTransitionOwnerFactors[] = {
        MakeNativeTransitionOwnerFactor(
            LOOM_NATIVE_PHYSICAL_DIMENSION_PARTICIPANT,
            LOOM_NATIVE_PHYSICAL_DIMENSION_PARTICIPANT, 1, 16, 1),
        MakeNativeTransitionOwnerFactor(
            LOOM_NATIVE_PHYSICAL_DIMENSION_POSITION,
            LOOM_NATIVE_PHYSICAL_DIMENSION_PARTICIPANT, 1, 2, 16),
        MakeNativeTransitionOwnerFactor(LOOM_NATIVE_PHYSICAL_DIMENSION_POSITION,
                                        LOOM_NATIVE_PHYSICAL_DIMENSION_POSITION,
                                        2, 0, 1),
};

static loom_native_transition_facts_t MakeNativeTransitionFacts() {
  loom_native_transition_facts_t facts = {};
  facts.source_role = LOOM_CONTRACT_OPERAND_ROLE_RESULT;
  facts.destination_role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
  facts.destination_position_count = 512;
  facts.participant_change_count = 256;
  facts.local_position_change_count = 480;
  facts.destination_positions_per_source_minimum = 2;
  facts.destination_positions_per_source_maximum = 2;
  facts.source_owner_factors = kNativeTransitionOwnerFactors;
  facts.source_owner_factor_count =
      IREE_ARRAYSIZE(kNativeTransitionOwnerFactors);
  return facts;
}

static const loom_native_transition_facts_t kNativeTransitionFacts =
    MakeNativeTransitionFacts();

static iree_string_view_t ParseJsonDocument(iree_string_view_t json) {
  iree_string_view_t cursor = json;
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_consume_value(&cursor, &value));
  IREE_EXPECT_OK(iree_json_consume_insignificant(&cursor));
  EXPECT_TRUE(iree_string_view_is_empty(cursor));
  return value;
}

static iree_string_view_t LookupObject(iree_string_view_t object,
                                       iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(object, key, &value));
  return value;
}

static void ExpectObjectValueEquals(iree_string_view_t object,
                                    iree_string_view_t key,
                                    iree_string_view_t expected) {
  EXPECT_TRUE(iree_string_view_equal(LookupObject(object, key), expected));
}

static void ExpectObjectUint64Equals(iree_string_view_t object,
                                     iree_string_view_t key,
                                     uint64_t expected) {
  uint64_t actual = 0;
  IREE_EXPECT_OK(iree_json_parse_uint64(LookupObject(object, key), &actual));
  EXPECT_EQ(actual, expected);
}

static iree_string_view_t LookupArrayElement(iree_string_view_t array,
                                             iree_host_size_t index) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_array_get(array, index, &value));
  return value;
}

TEST(CompileReportFormatTest, FormatsSourceToLowSelectionAndMemory) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  report.source_low_selected_op_count = 4;
  report.source_low_emitted_op_count = 5;

  loom_target_compile_report_source_low_row_t selection = {};
  selection.function_name = IREE_SVL("branchy");
  selection.source_op_name = IREE_SVL("scalar.addi");
  selection.source_op_kind = 42;
  selection.selection_kind =
      LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE;
  selection.plan_key = IREE_SVL("test.scalar_addi.strategy.native");
  selection.native_contraction_facts = &kNativeContractionFacts;
  selection.native_transition_facts = &kNativeTransitionFacts;
  selection.native_transition_source_type = LOOM_SCALAR_TYPE_F32;
  selection.native_transition_destination_type = LOOM_SCALAR_TYPE_F16;
  selection.descriptor_key = IREE_SVL("test.add.i32");
  selection.descriptor_semantic_tag = IREE_SVL("integer.add.i32");
  selection.emitted_low_op_count = 1;
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_row(&report, &selection));

  loom_target_compile_report_source_low_memory_row_t memory = {};
  memory.function_name = IREE_SVL("branchy");
  memory.source_op_name = IREE_SVL("vector.load");
  memory.source_op_kind = 43;
  memory.source_root_name = IREE_SVL("lhs");
  memory.source_root_argument_index = 0;
  memory.memory_space = IREE_SVL("workgroup");
  memory.operation_kind = IREE_SVL("load");
  memory.packet_key = IREE_SVL("amdgpu.ds_read2_b32");
  memory.strategy_key = IREE_SVL("ds_2addr_memory_report");
  memory.address_form = IREE_SVL("ds_2addr");
  memory.dynamic_term_kind = IREE_SVL("vaddr");
  memory.fallback_reason = IREE_SVL("cross_wave_workgroup");
  memory.element_byte_count = 4;
  memory.vector_lane_count = 2;
  memory.issued_read_byte_count = 8;
  memory.dynamic_stride_bytes = 32;
  memory.vector_lane_stride_bytes = 8;
  memory.storage_element_format = IREE_SVL("f8e4m3fn");
  memory.storage_scale_format = IREE_SVL("f32");
  memory.storage_payload_packing = IREE_SVL("dense_lanes");
  memory.storage_scale_topology = IREE_SVL("block_1d");
  memory.storage_affine_policy = IREE_SVL("scale_only");
  memory.storage_rounding_policy = IREE_SVL("finite_only");
  memory.bank_service.proof = IREE_SVL("exact");
  memory.bank_service.classification = IREE_SVL("conflicted");
  memory.bank_service.model_key = IREE_SVL("test.ds_read_b128");
  memory.bank_service.model_revision = IREE_SVL("test-model@1");
  memory.bank_service.model_evidence = IREE_SVL("silicon-calibrated");
  memory.bank_service.request_policy = IREE_SVL("count-each");
  memory.bank_service.lane_address_proof = IREE_SVL("affine-lane-stride");
  memory.bank_service.active_lane_proof = IREE_SVL("full-wave");
  memory.bank_service.base_residue_proof = IREE_SVL("common-translation");
  memory.bank_service.wave_size = 32;
  memory.bank_service.bank_count = 32;
  memory.bank_service.bank_word_byte_count = 4;
  memory.bank_service.packet_word_count = 4;
  memory.bank_service.phase_count = 2;
  memory.bank_service.phase_lane_counts[0] = 4;
  memory.bank_service.phase_lane_counts[1] = 4;
  memory.bank_service.base_residue_count = 32;
  memory.bank_service.phase_required_rounds[0] = 2;
  memory.bank_service.phase_required_rounds[1] = 2;
  memory.bank_service.required_rounds = 4;
  memory.bank_service.uncontended_rounds = 2;
  memory.bank_service.extra_rounds = 2;
  memory.bank_service.maximum_request_multiplicity = 2;
  memory.subgroup_access.proof = IREE_SVL("exact");
  memory.subgroup_access.lane_address_proof =
      IREE_SVL("compiled-fragment-lane-register-layout");
  memory.subgroup_access.active_lane_proof = IREE_SVL("full-wave");
  memory.subgroup_access.lane_mapping = IREE_SVL("linear");
  memory.subgroup_access.interval_coverage = IREE_SVL("gapped");
  memory.subgroup_access.subgroup_size = 32;
  memory.subgroup_access.lane_term_count = 1;
  memory.subgroup_access.lane_terms[0] = {
      /*.divisor=*/1,
      /*.modulus=*/0,
      /*.byte_stride=*/64,
  };
  memory.subgroup_access.per_lane_packet_byte_count = 8;
  memory.subgroup_access.linear_lane_byte_stride = 64;
  memory.subgroup_access.subgroup_requested_byte_count = 256;
  memory.subgroup_access.subgroup_unique_byte_count = 256;
  memory.subgroup_access.subgroup_span_byte_count = 1992;
  memory.subgroup_access.maximum_adjacent_lane_delta_bytes = 64;
  memory.subgroup_access.maximum_uncovered_byte_gap_bytes = 56;
  memory.subgroup_access.distinct_lane_address_count = 32;
  memory.subgroup_access.contiguous_region_count = 32;
  memory.execution_count_plus_one = 2;
  IREE_ASSERT_OK(loom_target_compile_report_record_source_low_memory_row(
      &report, &memory));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  const iree_string_view_t text = iree_string_builder_view(&builder);
  EXPECT_NE(
      iree_string_view_find(text, IREE_SV("source_low selected_ops=4"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text, IREE_SV("source_low[0] function=branchy"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                text, IREE_SV("plan_key=test.scalar_addi.strategy.native"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                text, IREE_SV("native_contraction={tile:1x16x16x16"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text,
                            IREE_SV("native_transition={result:f32->rhs:f16,"
                                    "positions:512,participant_changes:256"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                text, IREE_SV("participant+=(participant/1%16)*1"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text,
                            IREE_SV("source_low_memory[0] function=branchy "
                                    "source_op=vector.load"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          text, IREE_SV("subgroup_access={proof:exact,lane_address_proof:"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(text, IREE_SV("interval_coverage:gapped"), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  const iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  const iree_string_view_t source_low =
      LookupObject(root, IREE_SV("source_low"));
  ExpectObjectUint64Equals(source_low, IREE_SV("selected_op_count"), 4);
  const iree_string_view_t memory_summary =
      LookupObject(source_low, IREE_SV("memory"));
  const iree_string_view_t bank_service_summary =
      LookupObject(memory_summary, IREE_SV("bank_service"));
  ExpectObjectUint64Equals(bank_service_summary,
                           IREE_SV("modeled_packet_count"), 1);
  ExpectObjectUint64Equals(bank_service_summary, IREE_SV("exact_packet_count"),
                           1);
  const iree_string_view_t structural_bank_service =
      LookupObject(bank_service_summary, IREE_SV("structural"));
  ExpectObjectUint64Equals(structural_bank_service,
                           IREE_SV("conflicted_packet_count"), 1);
  ExpectObjectUint64Equals(structural_bank_service,
                           IREE_SV("extra_round_count"), 2);
  const iree_string_view_t dynamic_bank_service =
      LookupObject(bank_service_summary, IREE_SV("dynamic"));
  ExpectObjectUint64Equals(dynamic_bank_service, IREE_SV("packet_count"), 1);
  ExpectObjectUint64Equals(dynamic_bank_service, IREE_SV("extra_round_count"),
                           2);
  ExpectObjectUint64Equals(memory_summary, IREE_SV("bank_service_group_count"),
                           1);
  const iree_string_view_t bank_service_group = LookupArrayElement(
      LookupObject(memory_summary, IREE_SV("bank_service_groups")),
      /*index=*/0);
  ExpectObjectValueEquals(bank_service_group, IREE_SV("function"),
                          IREE_SV("branchy"));
  ExpectObjectValueEquals(bank_service_group, IREE_SV("source_root"),
                          IREE_SV("lhs"));
  const iree_string_view_t group_model =
      LookupObject(bank_service_group, IREE_SV("model"));
  ExpectObjectValueEquals(group_model, IREE_SV("key"),
                          IREE_SV("test.ds_read_b128"));
  const iree_string_view_t group_summary =
      LookupObject(bank_service_group, IREE_SV("summary"));
  ExpectObjectUint64Equals(group_summary, IREE_SV("exact_packet_count"), 1);
  const iree_string_view_t subgroup_access_summary =
      LookupObject(memory_summary, IREE_SV("subgroup_access"));
  ExpectObjectUint64Equals(subgroup_access_summary,
                           IREE_SV("exact_packet_count"), 1);
  const iree_string_view_t structural_subgroup_access =
      LookupObject(subgroup_access_summary, IREE_SV("structural"));
  ExpectObjectUint64Equals(structural_subgroup_access,
                           IREE_SV("gapped_packet_count"), 1);
  ExpectObjectUint64Equals(memory_summary,
                           IREE_SV("subgroup_access_group_count"), 1);
  const iree_string_view_t subgroup_access_group = LookupArrayElement(
      LookupObject(memory_summary, IREE_SV("subgroup_access_groups")),
      /*index=*/0);
  const iree_string_view_t group_access =
      LookupObject(subgroup_access_group, IREE_SV("access"));
  const iree_string_view_t group_geometry =
      LookupObject(group_access, IREE_SV("geometry"));
  ExpectObjectValueEquals(group_geometry, IREE_SV("interval_coverage"),
                          IREE_SV("gapped"));
  ExpectObjectUint64Equals(group_geometry, IREE_SV("subgroup_span_bytes"),
                           1992);
  const iree_string_view_t selection_row = LookupArrayElement(
      LookupObject(source_low, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(selection_row, IREE_SV("plan_key"),
                          IREE_SV("test.scalar_addi.strategy.native"));
  ExpectObjectValueEquals(selection_row, IREE_SV("descriptor_semantic_tag"),
                          IREE_SV("integer.add.i32"));
  const iree_string_view_t native_contraction =
      LookupObject(selection_row, IREE_SV("native_contraction"));
  ExpectObjectUint64Equals(native_contraction, IREE_SV("participants"), 32);
  const iree_string_view_t native_tile =
      LookupObject(native_contraction, IREE_SV("tile"));
  ExpectObjectUint64Equals(native_tile, IREE_SV("m"), 16);
  const iree_string_view_t native_lhs =
      LookupObject(native_contraction, IREE_SV("lhs"));
  ExpectObjectValueEquals(native_lhs, IREE_SV("evidence"), IREE_SV("exact"));
  ExpectObjectUint64Equals(native_lhs, IREE_SV("physical_positions"), 512);
  const iree_string_view_t native_transition =
      LookupObject(selection_row, IREE_SV("native_transition"));
  ExpectObjectValueEquals(native_transition, IREE_SV("source_role"),
                          IREE_SV("result"));
  ExpectObjectValueEquals(native_transition, IREE_SV("destination_type"),
                          IREE_SV("f16"));
  ExpectObjectUint64Equals(native_transition, IREE_SV("participant_changes"),
                           256);
  const iree_string_view_t native_factor = LookupArrayElement(
      LookupObject(native_transition, IREE_SV("source_owner_factors")),
      /*index=*/1);
  ExpectObjectValueEquals(native_factor, IREE_SV("destination_dimension"),
                          IREE_SV("position"));
  ExpectObjectUint64Equals(native_factor, IREE_SV("multiplier"), 16);
  const iree_string_view_t memory_row = LookupArrayElement(
      LookupObject(source_low, IREE_SV("memory_rows")), /*index=*/0);
  ExpectObjectValueEquals(memory_row, IREE_SV("packet"),
                          IREE_SV("amdgpu.ds_read2_b32"));
  ExpectObjectValueEquals(memory_row, IREE_SV("strategy"),
                          IREE_SV("ds_2addr_memory_report"));
  const iree_string_view_t storage =
      LookupObject(memory_row, IREE_SV("storage"));
  ExpectObjectValueEquals(storage, IREE_SV("element_format"),
                          IREE_SV("f8e4m3fn"));
  ExpectObjectValueEquals(storage, IREE_SV("rounding_policy"),
                          IREE_SV("finite_only"));
  const iree_string_view_t bank_service =
      LookupObject(memory_row, IREE_SV("bank_service"));
  ExpectObjectValueEquals(bank_service, IREE_SV("proof"), IREE_SV("exact"));
  ExpectObjectValueEquals(bank_service, IREE_SV("classification"),
                          IREE_SV("conflicted"));
  ExpectObjectUint64Equals(bank_service, IREE_SV("required_rounds"), 4);
  const iree_string_view_t bank_model =
      LookupObject(bank_service, IREE_SV("model"));
  ExpectObjectValueEquals(bank_model, IREE_SV("key"),
                          IREE_SV("test.ds_read_b128"));
  ExpectObjectValueEquals(bank_model, IREE_SV("revision"),
                          IREE_SV("test-model@1"));
  const iree_string_view_t bank_address =
      LookupObject(bank_service, IREE_SV("address"));
  ExpectObjectValueEquals(bank_address, IREE_SV("active_lane_proof"),
                          IREE_SV("full-wave"));
  const iree_string_view_t subgroup_access =
      LookupObject(memory_row, IREE_SV("subgroup_access"));
  const iree_string_view_t subgroup_address =
      LookupObject(subgroup_access, IREE_SV("address"));
  ExpectObjectValueEquals(subgroup_address, IREE_SV("lane_mapping"),
                          IREE_SV("linear"));
  const iree_string_view_t subgroup_geometry =
      LookupObject(subgroup_access, IREE_SV("geometry"));
  ExpectObjectUint64Equals(subgroup_geometry,
                           IREE_SV("maximum_uncovered_gap_bytes"), 56);
  uint64_t second_phase_rounds = 0;
  IREE_ASSERT_OK(iree_json_parse_uint64(
      LookupArrayElement(
          LookupObject(bank_service, IREE_SV("phase_required_rounds")),
          /*index=*/1),
      &second_phase_rounds));
  EXPECT_EQ(second_phase_rounds, 2u);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsMathAndTargetLegalization) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS;

  loom_target_compile_report_math_row_t math = {};
  math.function_name = IREE_SVL("branchy");
  math.source_op_name = IREE_SVL("scalar.roundf");
  math.source_op_kind = 44;
  math.target_bundle_name = IREE_SVL("test_target");
  math.target_config_name = IREE_SVL("test_o0");
  math.policy_name = IREE_SVL("amdgpu-math");
  math.constraint_key = IREE_SVL("math.recipe.round_away_f32");
  math.math_op = LOOM_TARGET_MATH_OP_ROUNDF;
  math.lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR;
  math.element_type = LOOM_SCALAR_TYPE_F32;
  math.action = LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REWRITTEN;
  math.recipe = LOOM_TARGET_MATH_RECIPE_ROUND_AWAY_F32;
  math.created_op_count = 10;
  math.erased_op_count = 1;
  IREE_ASSERT_OK(loom_target_compile_report_record_math_row(&report, &math));

  loom_target_compile_report_legalization_row_t legalization = {};
  legalization.function_name = IREE_SVL("branchy");
  legalization.source_op_name = IREE_SVL("vector.reduce.axes");
  legalization.source_op_kind = 73;
  legalization.target_bundle_name = IREE_SVL("test_target");
  legalization.target_config_name = IREE_SVL("test_o0");
  legalization.legalizer_name = IREE_SVL("vector");
  legalization.legalizer_strategy =
      LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE;
  legalization.mode = LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_FINAL;
  legalization.policy =
      LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REFERENCE_ONLY;
  legalization.action =
      LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN;
  legalization.legalization_outcome =
      LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REFERENCE_FALLBACK;
  legalization.contract_outcome =
      LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNSUPPORTED;
  legalization.descriptor_key = IREE_SVL("test.legalized.descriptor");
  legalization.source_rejection_bits = 0x1;
  legalization.source_rejection_detail = kTestSourceRejectionDetail;
  legalization.target_rejection_bits = 0x2;
  legalization.missing_feature_bits = 0x4;
  legalization.missing_fact_bits = 0x8;
  legalization.created_op_count = 6;
  legalization.erased_op_count = 1;
  IREE_ASSERT_OK(loom_target_compile_report_record_legalization_row(
      &report, &legalization));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  const iree_string_view_t text = iree_string_builder_view(&builder);
  EXPECT_NE(
      iree_string_view_find(text, IREE_SV("math_legalization rewritten=1"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text,
                            IREE_SV("math_legalization[0] function=branchy "
                                    "source_op=scalar.roundf"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text,
                            IREE_SV("target_legalization[0] function=branchy "
                                    "source_op=vector.reduce.axes"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                text, IREE_SV("descriptor_key=test.legalized.descriptor"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(text, IREE_SV(" descriptor="), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  const iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  const iree_string_view_t math_json =
      LookupObject(root, IREE_SV("math_legalization"));
  const iree_string_view_t math_row =
      LookupArrayElement(LookupObject(math_json, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(math_row, IREE_SV("recipe"),
                          IREE_SV("round-away-f32"));
  ExpectObjectUint64Equals(math_row, IREE_SV("created_op_count"), 10);
  const iree_string_view_t legalization_json =
      LookupObject(root, IREE_SV("target_legalization"));
  const iree_string_view_t legalization_row = LookupArrayElement(
      LookupObject(legalization_json, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(legalization_row, IREE_SV("legalizer_strategy"),
                          IREE_SV("reference"));
  ExpectObjectValueEquals(legalization_row, IREE_SV("descriptor_key"),
                          IREE_SV("test.legalized.descriptor"));
  ExpectObjectUint64Equals(legalization_row, IREE_SV("source_rejection_detail"),
                           4);
  ExpectObjectUint64Equals(legalization_row, IREE_SV("created_op_count"), 6);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

}  // namespace
}  // namespace loom
