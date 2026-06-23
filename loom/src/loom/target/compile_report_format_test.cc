// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report_format.h"

#include <stdint.h>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/types.h"
#include "loom/target/math_policy.h"

namespace loom {
namespace {

constexpr uint32_t kTestSourceRejectionDetail = 4;

TEST(CompileReportFormatTest, FormatsSummaryAndDetails) {
  loom_target_compile_report_pressure_row_t pressure_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy"),
          /*.register_class=*/IREE_SVL("amdgpu.sgpr"),
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/LOOM_SCALAR_TYPE_I32,
          /*.peak_live_units=*/32,
          /*.peak_live_values=*/4,
          /*.peak_point=*/3,
          /*.peak_block_name=*/IREE_SVL("entry"),
          /*.peak_operation_name=*/IREE_SVL("low.op<amdgpu.s_add_u32>"),
      },
      {
          /*.function_name=*/IREE_SVL("branchy"),
          /*.register_class=*/IREE_SVL("amdgpu.vgpr"),
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/LOOM_SCALAR_TYPE_I32,
          /*.peak_live_units=*/96,
          /*.peak_live_values=*/16,
          /*.peak_point=*/4,
          /*.peak_block_name=*/IREE_SVL("entry"),
          /*.peak_operation_name=*/IREE_SVL("low.op<amdgpu.v_add_u32>"),
      },
  };
  loom_target_compile_report_pressure_origin_row_t pressure_origin_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy_export"),
          /*.register_class=*/IREE_SVL("amdgpu.vgpr"),
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/LOOM_SCALAR_TYPE_I32,
          /*.peak_point=*/4,
          /*.peak_block_name=*/IREE_SVL("entry"),
          /*.peak_operation_name=*/IREE_SVL("low.op<amdgpu.v_add_u32>"),
          /*.origin_kind=*/
          LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_DOT,
          /*.origin_operation_name=*/IREE_SVL("low.op<amdgpu.v_dot4_i32_i8>"),
          /*.semantic_tag=*/IREE_SVL("dot.i32.i8"),
          /*.sample_value_name=*/IREE_SVL("acc"),
          /*.live_units=*/64,
          /*.live_values=*/8,
      },
  };
  loom_target_compile_report_schedule_band_row_t schedule_band_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy_export"),
          /*.block_name=*/IREE_SVL("body"),
          /*.block_index=*/2,
          /*.first_packet_index=*/17,
          /*.first_scheduled_ordinal=*/5,
          /*.node_count=*/4,
          /*.origin_kind=*/
          LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_LOCAL_MEMORY,
          /*.origin_operation_name=*/
          IREE_SVL("low.op<amdgpu.ds_read2_b32>"),
          /*.semantic_tag=*/IREE_SVL("memory.workgroup.load2.u32"),
          /*.sample_value_name=*/IREE_SVL("tile"),
          /*.static_instruction_mix=*/
          {
              /*.descriptor_count=*/4,
              /*.unknown_count=*/0,
              /*.scalar_alu_count=*/0,
              /*.vector_alu_count=*/0,
              /*.matrix_count=*/0,
              /*.mfma_count=*/0,
              /*.wmma_count=*/0,
              /*.dot_count=*/0,
              /*.global_memory_count=*/0,
              /*.global_load_count=*/0,
              /*.global_store_count=*/0,
              /*.buffer_load_count=*/0,
              /*.buffer_store_count=*/0,
              /*.flat_memory_count=*/0,
              /*.local_memory_count=*/4,
              /*.scalar_memory_count=*/0,
              /*.generic_memory_count=*/0,
              /*.atomic_count=*/0,
              /*.branch_count=*/0,
              /*.barrier_count=*/0,
              /*.control_count=*/0,
              /*.conversion_count=*/0,
              /*.cache_count=*/0,
              /*.register_move_count=*/0,
          },
          /*.result_value_count=*/4,
          /*.result_unit_count=*/16,
      },
  };
  loom_target_compile_report_schedule_band_summary_row_t
      schedule_band_summary_rows[] = {
          {
              /*.function_name=*/IREE_SVL("branchy_export"),
              /*.block_name=*/IREE_SVL("body"),
              /*.block_index=*/2,
              /*.first_packet_index=*/17,
              /*.band_count=*/3,
              /*.node_count=*/12,
              /*.max_band_node_count=*/4,
              /*.origin_kind=*/
              LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_LOCAL_MEMORY,
              /*.origin_operation_name=*/
              IREE_SVL("low.op<amdgpu.ds_read2_b32>"),
              /*.semantic_tag=*/IREE_SVL("memory.workgroup.load2.u32"),
              /*.sample_value_name=*/IREE_SVL("tile"),
              /*.static_instruction_mix=*/
              {
                  /*.descriptor_count=*/12,
                  /*.unknown_count=*/0,
                  /*.scalar_alu_count=*/0,
                  /*.vector_alu_count=*/0,
                  /*.matrix_count=*/0,
                  /*.mfma_count=*/0,
                  /*.wmma_count=*/0,
                  /*.dot_count=*/0,
                  /*.global_memory_count=*/0,
                  /*.global_load_count=*/0,
                  /*.global_store_count=*/0,
                  /*.buffer_load_count=*/0,
                  /*.buffer_store_count=*/0,
                  /*.flat_memory_count=*/0,
                  /*.local_memory_count=*/12,
                  /*.scalar_memory_count=*/0,
                  /*.generic_memory_count=*/0,
                  /*.atomic_count=*/0,
                  /*.branch_count=*/0,
                  /*.barrier_count=*/0,
                  /*.control_count=*/0,
                  /*.conversion_count=*/0,
                  /*.cache_count=*/0,
                  /*.register_move_count=*/0,
              },
              /*.result_value_count=*/12,
              /*.result_unit_count=*/48,
          },
      };
  loom_target_compile_report_spill_row_t spill_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy"),
          /*.value_name=*/IREE_SVL("rhs"),
          /*.register_class=*/IREE_SVL("test.i32"),
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/LOOM_SCALAR_TYPE_I32,
          /*.assignment_index=*/2,
          /*.slot_index=*/1,
          /*.slot_space=*/IREE_SVL("stack"),
          /*.byte_size=*/4,
          /*.byte_alignment=*/4,
          /*.store_count=*/1,
          /*.reload_count=*/2,
      },
  };
  loom_target_compile_report_allocation_failure_row_t allocation_failure_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy"),
          /*.value_name=*/IREE_SVL("blocked"),
          /*.register_class=*/IREE_SVL("test.scc"),
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/LOOM_SCALAR_TYPE_INDEX,
          /*.failure_code=*/IREE_SVL("unspillable-register-exhausted"),
          /*.blocking_kind=*/
          LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT,
          /*.origin_operation_name=*/IREE_SVL("low.return"),
          /*.origin_block_name=*/IREE_SVL("entry"),
          /*.start_point=*/2,
          /*.end_point=*/5,
          /*.required_unit_count=*/1,
          /*.budget_units=*/1,
          /*.peak_live_units=*/2,
          /*.location_kind=*/IREE_SVL("physical_register"),
          /*.location_base=*/0,
          /*.location_count=*/1,
          /*.conflict_assignment_index=*/0,
          /*.conflict_value_name=*/IREE_SVL("leader"),
          /*.conflict_start_point=*/0,
          /*.conflict_end_point=*/5,
          /*.conflict_location_kind=*/IREE_SVL("physical_register"),
          /*.conflict_location_base=*/0,
          /*.conflict_location_count=*/1,
      },
  };
  loom_target_compile_report_allocation_high_water_row_t
      allocation_high_water_rows[] = {
          {
              /*.function_name=*/IREE_SVL("branchy_export"),
              /*.value_name=*/IREE_SVL("rhs_window"),
              /*.register_class=*/IREE_SVL("amdgpu.vgpr"),
              /*.type_kind=*/LOOM_TYPE_REGISTER,
              /*.element_type=*/LOOM_SCALAR_TYPE_I32,
              /*.assignment_index=*/5,
              /*.origin_operation_name=*/
              IREE_SVL("low.op<amdgpu.ds_load_b128>"),
              /*.origin_kind=*/
              LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_LOCAL_MEMORY,
              /*.semantic_tag=*/IREE_SVL("memory.workgroup.load.u128"),
              /*.start_point=*/17,
              /*.end_point=*/24,
              /*.required_unit_count=*/4,
              /*.location_kind=*/IREE_SVL("physical_register"),
              /*.location_base=*/248,
              /*.location_count=*/4,
              /*.high_water_units=*/252,
              /*.lower_free_unit_count=*/13,
              /*.lower_free_run_count=*/3,
              /*.lower_largest_free_run_unit_count=*/6,
              /*.lower_pressure_releasable_free_unit_count=*/21,
              /*.lower_pressure_releasable_free_run_count=*/2,
              /*.lower_pressure_releasable_largest_free_run_unit_count=*/
              14,
              /*.active_assignment_blocker_count=*/47,
              /*.active_assignment_blocker_units=*/244,
              /*.active_storage_lease_blocker_count=*/3,
              /*.active_storage_lease_blocker_units=*/12,
              /*.active_pressure_storage_lease_blocker_count=*/2,
              /*.active_pressure_storage_lease_blocker_units=*/8,
              /*.active_fallback_storage_lease_blocker_count=*/1,
              /*.active_fallback_storage_lease_blocker_units=*/4,
          },
      };
  const loom_target_compile_report_wait_plan_t wait_plan = {
      /*.action_count=*/5,
      /*.explicit_action_count=*/1,
      /*.planned_action_count=*/4,
      /*.full_drain_count=*/2,
      /*.partial_wait_count=*/3,
      /*.max_outstanding_before=*/6,
      /*.max_full_drain_outstanding_before=*/6,
  };
  loom_target_compile_report_wait_counter_row_t wait_counter_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy_export"),
          /*.counter_name=*/IREE_SVL("vmem_load"),
          /*.counter_id=*/1,
          /*.summary=*/
          {
              /*.action_count=*/3,
              /*.explicit_action_count=*/0,
              /*.planned_action_count=*/3,
              /*.full_drain_count=*/1,
              /*.partial_wait_count=*/2,
              /*.max_outstanding_before=*/6,
              /*.max_full_drain_outstanding_before=*/6,
          },
      },
      {
          /*.function_name=*/IREE_SVL("branchy_export"),
          /*.counter_name=*/IREE_SVL("lds"),
          /*.counter_id=*/3,
          /*.summary=*/
          {
              /*.action_count=*/2,
              /*.explicit_action_count=*/1,
              /*.planned_action_count=*/1,
              /*.full_drain_count=*/1,
              /*.partial_wait_count=*/1,
              /*.max_outstanding_before=*/2,
              /*.max_full_drain_outstanding_before=*/2,
          },
      },
  };
  loom_target_compile_report_wait_action_row_t wait_action_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy_export"),
          /*.counter_name=*/IREE_SVL("vmem_load"),
          /*.action_name=*/IREE_SVL("planned"),
          /*.reason_name=*/IREE_SVL("amdgpu.ssa_use"),
          /*.counter_id=*/1,
          /*.action_id=*/2,
          /*.reason_id=*/2,
          /*.block_index=*/1,
          /*.node_index=*/42,
          /*.scheduled_ordinal=*/17,
          /*.producer_node=*/8,
          /*.producer_scheduled_ordinal=*/3,
          /*.producer_operation_name=*/
          IREE_SVL("low.op<amdgpu.global_load_b32>"),
          /*.producer_descriptor_key=*/IREE_SVL("amdgpu.global_load_b32"),
          /*.producer_semantic_tag=*/IREE_SVL("memory.load.u32"),
          /*.consumer_node=*/42,
          /*.consumer_scheduled_ordinal=*/17,
          /*.consumer_operation_name=*/IREE_SVL("low.op<amdgpu.v_add_u32>"),
          /*.consumer_descriptor_key=*/IREE_SVL("amdgpu.v_add_u32"),
          /*.consumer_semantic_tag=*/IREE_SVL("vector.add.i32"),
          /*.target_count=*/2,
          /*.outstanding_before=*/6,
      },
      {
          /*.function_name=*/IREE_SVL("branchy_export"),
          /*.counter_name=*/IREE_SVL("smem"),
          /*.action_name=*/IREE_SVL("explicit"),
          /*.reason_name=*/IREE_SVL("amdgpu.explicit_packet"),
          /*.counter_id=*/4,
          /*.action_id=*/1,
          /*.reason_id=*/1,
          /*.block_index=*/1,
          /*.node_index=*/44,
          /*.scheduled_ordinal=*/19,
          /*.producer_node=*/UINT32_MAX,
          /*.producer_scheduled_ordinal=*/UINT32_MAX,
          /*.producer_operation_name=*/IREE_SVL(""),
          /*.producer_descriptor_key=*/IREE_SVL(""),
          /*.producer_semantic_tag=*/IREE_SVL(""),
          /*.consumer_node=*/UINT32_MAX,
          /*.consumer_scheduled_ordinal=*/UINT32_MAX,
          /*.consumer_operation_name=*/IREE_SVL(""),
          /*.consumer_descriptor_key=*/IREE_SVL(""),
          /*.consumer_semantic_tag=*/IREE_SVL(""),
          /*.target_count=*/0,
          /*.outstanding_before=*/2,
      },
  };
  loom_target_compile_report_source_low_row_t source_low_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy"),
          /*.source_op_name=*/IREE_SVL("scalar.addi"),
          /*.source_op_kind=*/42,
          /*.selection_kind=*/
          LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE,
          /*.rule_set_index=*/0,
          /*.rule_index=*/1,
          /*.plan_id=*/UINT64_MAX,
          /*.plan_key=*/IREE_SVL("test.scalar_addi.strategy.native"),
          /*.descriptor_id=*/7,
          /*.emitted_low_op_count=*/1,
      },
  };
  loom_target_compile_report_source_low_memory_row_t source_low_memory_rows[] =
      {
          {
              /*.function_name=*/IREE_SVL("branchy"),
              /*.source_op_name=*/IREE_SVL("vector.load"),
              /*.source_op_kind=*/43,
              /*.memory_space=*/IREE_SVL("workgroup"),
              /*.operation_kind=*/IREE_SVL("load"),
              /*.packet_key=*/IREE_SVL("amdgpu.ds_read2_b32"),
              /*.address_form=*/IREE_SVL("ds_2addr"),
              /*.dynamic_term_kind=*/IREE_SVL("vaddr"),
              /*.fallback_reason=*/IREE_SVL("cross_wave_workgroup"),
              /*.static_offset_bytes=*/0,
              /*.element_byte_count=*/4,
              /*.vector_lane_count=*/2,
              /*.dynamic_stride_bytes=*/32,
              /*.vector_lane_stride_bytes=*/8,
              /*.bank_stride_words=*/8,
              /*.bank_conflict_degree=*/8,
              /*.bank_conflict_kind=*/IREE_SVL("bank-conflict-risk"),
          },
      };
  loom_target_compile_report_math_row_t math_legalization_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy"),
          /*.source_op_name=*/IREE_SVL("scalar.roundf"),
          /*.source_op_kind=*/44,
          /*.target_bundle_name=*/IREE_SVL("vm_target"),
          /*.target_config_name=*/IREE_SVL("vm_o0"),
          /*.policy_name=*/IREE_SVL("amdgpu-math"),
          /*.constraint_key=*/IREE_SVL("math.recipe.round_away_f32"),
          /*.math_op=*/LOOM_TARGET_MATH_OP_ROUNDF,
          /*.lane_domain=*/LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
          /*.element_type=*/LOOM_SCALAR_TYPE_F32,
          /*.action=*/LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REWRITTEN,
          /*.recipe=*/LOOM_TARGET_MATH_RECIPE_ROUND_AWAY_F32,
          /*.source_fastmath_flags=*/LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
          /*.recipe_fastmath_flags=*/LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
          /*.created_op_count=*/10,
          /*.erased_op_count=*/1,
      },
  };
  loom_target_compile_report_legalization_row_t target_legalization_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy"),
          /*.source_op_name=*/IREE_SVL("vector.reduce.axes"),
          /*.source_op_kind=*/73,
          /*.target_bundle_name=*/IREE_SVL("vm_target"),
          /*.target_config_name=*/IREE_SVL("vm_o0"),
          /*.legalizer_name=*/IREE_SVL("vector"),
          /*.legalizer_strategy=*/
          LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE,
          /*.mode=*/LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_FINAL,
          /*.policy=*/
          LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REFERENCE_ONLY,
          /*.action=*/LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN,
          /*.legalization_outcome=*/
          LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REFERENCE_FALLBACK,
          /*.contract_outcome=*/
          LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNSUPPORTED,
          /*.binding_index=*/0,
          /*.case_index=*/2,
          /*.rule_set_index=*/3,
          /*.rule_index=*/4,
          /*.diagnostic_index=*/UINT16_MAX,
          /*.descriptor_id=*/UINT64_MAX,
          /*.source_rejection_bits=*/0x1,
          /*.source_rejection_detail=*/kTestSourceRejectionDetail,
          /*.target_rejection_bits=*/0x2,
          /*.missing_feature_bits=*/0x4,
          /*.missing_fact_bits=*/0x8,
          /*.created_op_count=*/6,
          /*.erased_op_count=*/1,
      },
  };
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN;
  report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
  report.function_name = IREE_SVL("branchy");
  report.target_bundle_name = IREE_SVL("vm_target");
  report.target_export_name = IREE_SVL("vm_export");
  report.target_export_symbol = IREE_SVL("branchy_export");
  report.target_config_name = IREE_SVL("vm_o0");
  report.lowered_symbol = IREE_SVL("branchy");
  loom_target_compile_report_record_artifact_size(&report, 128);
  loom_target_compile_report_record_schedule(&report, 5, 5, 4, 2, 1, 1, 2, 96);
  loom_target_compile_report_record_allocation(&report, 6, 1, 1, 2, 0, 11, 9,
                                               3);
  loom_target_compile_report_record_move_cause(
      &report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
      3, 3);
  loom_target_compile_report_record_move_cause(
      &report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT, 2, 8);
  loom_target_compile_report_record_move_cause(
      &report,
      LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION, 1, 1);
  const loom_target_compile_report_static_instruction_mix_t instruction_mix = {
      /*.descriptor_count=*/9,
      /*.unknown_count=*/{},
      /*.scalar_alu_count=*/2,
      /*.vector_alu_count=*/3,
      /*.matrix_count=*/1,
      /*.mfma_count=*/{},
      /*.wmma_count=*/1,
      /*.dot_count=*/{},
      /*.global_memory_count=*/2,
      /*.global_load_count=*/1,
      /*.global_store_count=*/{},
      /*.buffer_load_count=*/1,
      /*.buffer_store_count=*/{},
      /*.flat_memory_count=*/{},
      /*.local_memory_count=*/{},
      /*.scalar_memory_count=*/{},
      /*.generic_memory_count=*/{},
      /*.atomic_count=*/{},
      /*.branch_count=*/{},
      /*.barrier_count=*/1,
      /*.control_count=*/{},
      /*.conversion_count=*/1,
  };
  loom_target_compile_report_record_static_instruction_mix(&report,
                                                           &instruction_mix);
  loom_target_compile_report_record_emission(&report, 8, 64, 80);
  loom_target_compile_report_record_memory(&report, 16, 32);
  const loom_target_compile_report_target_resources_t target_resources = {
      /*.scalar_register_class=*/IREE_SVL("amdgpu.sgpr"),
      /*.scalar_register_count=*/38,
      /*.scalar_pressure_peak_live_units=*/{},
      /*.scalar_register_overhead_units=*/{},
      /*.vector_register_class=*/IREE_SVL("amdgpu.vgpr"),
      /*.vector_register_count=*/112,
      /*.vector_pressure_peak_live_units=*/{},
      /*.vector_register_overhead_units=*/{},
      /*.subgroup_size=*/32,
      /*.max_subgroups_per_simd=*/16,
      /*.resident_subgroups_per_simd=*/8,
      /*.occupancy_percent=*/50,
      /*.limiting_resource=*/IREE_SVL("amdgpu.vgpr"),
  };
  loom_target_compile_report_t entry_report = {};
  loom_target_compile_report_initialize(&entry_report, iree_allocator_system());
  entry_report.requested_detail_flags = report.requested_detail_flags;
  entry_report.function_name = IREE_SVL("branchy_export");
  entry_report.lowered_symbol = IREE_SVL("branchy");
  entry_report.target_bundle_name = IREE_SVL("vm_target");
  entry_report.target_export_name = IREE_SVL("vm_export");
  entry_report.target_export_symbol = IREE_SVL("branchy_export");
  entry_report.target_config_name = IREE_SVL("vm_o0");
  loom_target_compile_report_record_schedule(&entry_report, 5, 5, 4, 2, 1, 1, 2,
                                             96);
  loom_target_compile_report_record_allocation(&entry_report, 6, 1, 1, 2, 0, 11,
                                               9, 3);
  loom_target_compile_report_record_move_cause(
      &entry_report,
      LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION, 3, 3);
  loom_target_compile_report_record_move_cause(
      &entry_report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT, 2, 8);
  loom_target_compile_report_record_move_cause(
      &entry_report,
      LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION, 1, 1);
  loom_target_compile_report_record_emission(&entry_report, 8, 64, 80);
  loom_target_compile_report_record_memory(&entry_report, 16, 32);
  IREE_ASSERT_OK(loom_target_compile_report_record_pressure_row(
      &entry_report, &pressure_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_pressure_row(
      &entry_report, &pressure_rows[1]));
  IREE_ASSERT_OK(loom_target_compile_report_record_pressure_origin_row(
      &entry_report, &pressure_origin_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_schedule_band_row(
      &entry_report, &schedule_band_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_schedule_band_summary_row(
      &entry_report, &schedule_band_summary_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_allocation_high_water_row(
      &entry_report, &allocation_high_water_rows[0]));
  loom_target_compile_report_record_wait_plan(&entry_report, &wait_plan);
  IREE_ASSERT_OK(loom_target_compile_report_record_wait_counter_row(
      &entry_report, &wait_counter_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_wait_counter_row(
      &entry_report, &wait_counter_rows[1]));
  IREE_ASSERT_OK(loom_target_compile_report_record_wait_action_row(
      &entry_report, &wait_action_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_wait_action_row(
      &entry_report, &wait_action_rows[1]));
  loom_target_compile_report_record_target_resources(&entry_report,
                                                     &target_resources);
  loom_target_compile_report_record_static_instruction_mix(&entry_report,
                                                           &instruction_mix);
  IREE_ASSERT_OK(
      loom_target_compile_report_record_entry_report(&report, &entry_report));
  IREE_ASSERT_OK(
      loom_target_compile_report_record_spill_row(&report, &spill_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_allocation_failure_row(
      &report, &allocation_failure_rows[0]));
  report.source_low_selected_op_count = 4;
  report.source_low_emitted_op_count = 5;
  IREE_ASSERT_OK(loom_target_compile_report_record_source_low_row(
      &report, &source_low_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_source_low_memory_row(
      &report, &source_low_memory_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_math_row(
      &report, &math_legalization_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_legalization_row(
      &report, &target_legalization_rows[0]));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("artifact=vm-archive"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("pressure_classes=2"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("pressure_rows count=2"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("pressure_origin_rows count=1"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("schedule_band_rows count=1"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("schedule_band_summary_rows count=1"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("move_causes kinds=3 packets=6 "
                                          "units=12"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("move_cause[low_concat] packets=2 "
                                          "units=8"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("wait_plan actions=5 explicit=1 "
                                          "planned=4 full_drains=2 "
                                          "partial_waits=3 "
                                          "max_outstanding=6 "
                                          "max_full_drain_outstanding=6 "
                                          "counter_rows=2 action_rows=2"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("wait_counter[0] "
                                          "function=branchy_export "
                                          "counter=vmem_load counter_id=1 "
                                          "actions=3 explicit=0 planned=3 "
                                          "full_drains=1 partial_waits=2 "
                                          "max_outstanding=6 "
                                          "max_full_drain_outstanding=6"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("wait_counter[1] "
                                          "function=branchy_export "
                                          "counter=lds counter_id=3 "
                                          "actions=2 explicit=1 planned=1 "
                                          "full_drains=1 partial_waits=1 "
                                          "max_outstanding=2 "
                                          "max_full_drain_outstanding=2"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("wait_action[0] "
                                          "function=branchy_export "
                                          "counter=vmem_load counter_id=1 "
                                          "action=planned action_id=2 "
                                          "reason=amdgpu.ssa_use reason_id=2 "
                                          "block=1 node=42 ordinal=17 "
                                          "producer_node=8 "
                                          "producer_ordinal=3 "
                                          "producer_operation=low.op<amdgpu."
                                          "global_load_b32> "
                                          "producer_descriptor_key=amdgpu."
                                          "global_load_b32 "
                                          "producer_semantic_tag="
                                          "memory.load.u32 consumer_node=42 "
                                          "consumer_ordinal=17 "
                                          "consumer_operation=low.op<amdgpu."
                                          "v_add_u32> "
                                          "consumer_descriptor_key=amdgpu."
                                          "v_add_u32 "
                                          "consumer_semantic_tag="
                                          "vector.add.i32 "
                                          "target_count=2 "
                                          "outstanding_before=6"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("wait_action[1] function=branchy_export counter=smem "
                        "counter_id=4 action=explicit action_id=1 "
                        "reason=amdgpu.explicit_packet reason_id=1 block=1 "
                        "node=44 ordinal=19 producer_node=- "
                        "producer_ordinal=- producer_operation=- "
                        "producer_descriptor_key=- producer_semantic_tag=- "
                        "consumer_node=- "
                        "consumer_ordinal=- consumer_operation=- "
                        "consumer_descriptor_key=- consumer_semantic_tag=- "
                        "target_count=0 outstanding_before=2"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("move_cause[operand_bank_materialization] packets=1 "
                        "units=1"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("static_instruction_mix "
                                          "descriptors=9"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output, IREE_SV("global_load=1 global_store=0 buffer_load=1"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("target_resources scalar_register_class=amdgpu.sgpr "
                        "scalar_registers=38 "
                        "scalar_pressure_peak=32 "
                        "scalar_register_overhead=6 "
                        "vector_register_class=amdgpu.vgpr "
                        "vector_registers=112 vector_pressure_peak=96 "
                        "vector_register_overhead=16 subgroup_size=32 "
                        "resident_subgroups_per_simd=8 "
                        "max_subgroups_per_simd=16 "
                        "occupancy_percent=50 limiting=amdgpu.vgpr"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("vector_alu=3"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("pressure[0] function=branchy "
                                          "class=amdgpu.sgpr"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("pressure[1] function=branchy "
                                          "class=amdgpu.vgpr"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("pressure_origin[0] function=branchy_export "
                        "class=amdgpu.vgpr type=register element=i32 "
                        "point=4 block=entry op=low.op<amdgpu.v_add_u32> "
                        "origin=dot origin_op=low.op<amdgpu.v_dot4_i32_i8> "
                        "semantic=dot.i32.i8 sample=acc live_units=64 "
                        "live_values=8"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("spill[0] function=branchy "
                                          "value=rhs"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("allocation_failure[0] function=branchy "
                                    "value=blocked class=test.scc "
                                    "code=unspillable-register-exhausted "
                                    "blocking=active-assignment"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("conflict_value=leader "
                                          "conflict_start=0 conflict_end=5"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("allocation_high_water[0] function=branchy_export "
                        "value=rhs_window class=amdgpu.vgpr type=register "
                        "element=i32 assignment=5 origin=local-memory "
                        "origin_op=low.op<amdgpu.ds_load_b128> "
                        "semantic=memory.workgroup.load.u128 start=17 end=24 "
                        "required_units=4 location=physical_register[248:4] "
                        "high_water=252 lower_free_units=13 "
                        "lower_free_runs=3 lower_largest_free_run_units=6 "
                        "lower_pressure_releasable_free_units=21 "
                        "lower_pressure_releasable_free_runs=2 "
                        "lower_pressure_releasable_largest_free_run_units=14 "
                        "active_assignment_blockers=47 "
                        "active_assignment_blocker_units=244 "
                        "active_storage_lease_blockers=3 "
                        "active_storage_lease_blocker_units=12 "
                        "active_pressure_storage_lease_blockers=2 "
                        "active_pressure_storage_lease_blocker_units=8 "
                        "active_fallback_storage_lease_blockers=1 "
                        "active_fallback_storage_lease_blocker_units=4"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("entry[0] "
                                          "function=branchy_export "
                                          "source=branchy"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("resource_uses=2 hazard_gaps=1 "
                                          "model_summaries=1 "
                                          "pressure_summaries=2 peak_live=96"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("instructions=8 code_bytes=64 storage_bytes=80 "
                        "private_bytes=16 local_bytes=32 "
                        "scalar_register_class=amdgpu.sgpr "
                        "scalar_registers=38 "
                        "scalar_pressure_peak=32 "
                        "scalar_register_overhead=6 "
                        "vector_register_class=amdgpu.vgpr "
                        "vector_registers=112 vector_pressure_peak=96 "
                        "vector_register_overhead=16 subgroup_size=32 "
                        "resident_subgroups_per_simd=8 "
                        "max_subgroups_per_simd=16 "
                        "occupancy_percent=50 limiting=amdgpu.vgpr"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("spill_plans=1 coalesced_copies=2 "
                                          "materialized_copies=0 "
                                          "storage_leases=11 "
                                          "storage_lease_instances=9 "
                                          "storage_release_actions=3 "
                                          "move_kinds=3 move_packets=6 "
                                          "move_units=12"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("wait_actions=5 wait_explicit=1 "
                                          "wait_planned=4 "
                                          "wait_full_drains=2 "
                                          "wait_partial=3 "
                                          "wait_max_outstanding=6 "
                                          "wait_max_full_drain_outstanding=6"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("pressure_rows=2 "
                                          "pressure_origin_rows=1 "
                                          "schedule_band_rows=1 "
                                          "schedule_band_summary_rows=1 "
                                          "spill_rows=0 "
                                          "allocation_high_water_rows=1 "
                                          "wait_counter_rows=2 "
                                          "wait_action_rows=2"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("schedule_band[0] function=branchy_export "
                        "block=body block_index=2 first_packet=17 "
                        "first_ordinal=5 nodes=4 origin=local-memory"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("schedule_band_summary[0] function=branchy_export "
                        "block=body block_index=2 first_packet=17 bands=3 "
                        "nodes=12 max_band_nodes=4 origin=local-memory"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("source_low selected_ops=4"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("math_legalization rewritten=1 "
                                          "rejected=0 missing_policy=0 "
                                          "missing_recipe=0 rows=1"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("math_legalization[0] function=branchy "
                        "source_op=scalar.roundf action=rewritten "
                        "policy=amdgpu-math math_op=roundf domain=scalar "
                        "element=f32 recipe=round-away-f32 "
                        "constraint=math.recipe.round_away_f32"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("source_low_memory rows=1"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("source_low[0] function=branchy"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("selection=rule rule_set=0 rule=1 "
                                    "plan_key=test.scalar_addi.strategy.native "
                                    "descriptor=7 emitted_ops=1"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("source_low_memory[0] function=branchy "
                        "source_op=vector.load memory_space=workgroup "
                        "operation=load packet=amdgpu.ds_read2_b32 "
                        "address_form=ds_2addr dynamic_term_kind=vaddr "
                        "fallback_reason=cross_wave_workgroup "
                        "static_offset_bytes=0 element_bytes=4 vector_lanes=2 "
                        "dynamic_stride_bytes=32 "
                        "vector_lane_stride_bytes=8 bank_stride_words=8 "
                        "bank_conflict_degree=8 "
                        "bank_conflict_kind=bank-conflict-risk"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("target_legalization legal=0 "
                                          "rewritten=1 target_rewritten=0 "
                                          "reference_rewritten=1 deferred=0"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("target_legalization[0] function=branchy source_op="
                        "vector.reduce.axes mode=final policy=reference-only "
                        "action=rewritten outcome=reference-fallback"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("strategy=reference"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("source_rejection_detail=4"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("created_ops=6 erased_ops=1"), 0),
      IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));

  output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"artifact_kind\":\"vm-archive\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"schedule\":{\"node_count\":5"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"entries\":{\"count\":1,"
                                          "\"rows\":[{\"index\":0,"
                                          "\"function\":\"branchy_export\""),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"schedule_resource_use_count\":2"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"move_causes\":{\"kind_count\":3,"
                        "\"packet_count\":6,\"unit_count\":12,\"causes\""),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("\"wait_plan\":{\"action_count\":5,"
                                    "\"explicit_action_count\":1,"
                                    "\"planned_action_count\":4,"
                                    "\"full_drain_count\":2,"
                                    "\"partial_wait_count\":3,"
                                    "\"max_outstanding_before\":6,"
                                    "\"max_full_drain_outstanding_before\":6}"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("\"wait_counter_row_count\":2"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("\"wait_action_row_count\":2"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output, IREE_SV("\"wait_counter_rows\":{\"count\":2,\"rows\":["), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"counter\":\"vmem_load\",\"counter_id\":1,"
                        "\"summary\":{\"action_count\":3,"
                        "\"explicit_action_count\":0,"
                        "\"planned_action_count\":3,"
                        "\"full_drain_count\":1,"
                        "\"partial_wait_count\":2,"
                        "\"max_outstanding_before\":6,"
                        "\"max_full_drain_outstanding_before\":6}"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("\"counter\":\"lds\",\"counter_id\":3,"
                                    "\"summary\":{\"action_count\":2,"
                                    "\"explicit_action_count\":1,"
                                    "\"planned_action_count\":1,"
                                    "\"full_drain_count\":1,"
                                    "\"partial_wait_count\":1,"
                                    "\"max_outstanding_before\":2,"
                                    "\"max_full_drain_outstanding_before\":2}"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output, IREE_SV("\"wait_action_rows\":{\"count\":2,\"rows\":["), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"counter\":\"vmem_load\",\"counter_id\":1,"
                        "\"action\":\"planned\",\"action_id\":2,"
                        "\"reason\":\"amdgpu.ssa_use\",\"reason_id\":2,"
                        "\"block_index\":1,\"node_index\":42,"
                        "\"scheduled_ordinal\":17,\"producer_node\":8,"
                        "\"producer_scheduled_ordinal\":3,"
                        "\"producer_operation\":"
                        "\"low.op<amdgpu.global_load_b32>\","
                        "\"producer_descriptor_key\":"
                        "\"amdgpu.global_load_b32\","
                        "\"producer_semantic_tag\":\"memory.load.u32\","
                        "\"consumer_node\":42,"
                        "\"consumer_scheduled_ordinal\":17,"
                        "\"consumer_operation\":"
                        "\"low.op<amdgpu.v_add_u32>\","
                        "\"consumer_descriptor_key\":\"amdgpu.v_add_u32\","
                        "\"consumer_semantic_tag\":\"vector.add.i32\","
                        "\"target_count\":2,"
                        "\"outstanding_before\":6"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"counter\":\"smem\",\"counter_id\":4,"
                        "\"action\":\"explicit\",\"action_id\":1,"
                        "\"reason\":\"amdgpu.explicit_packet\","
                        "\"reason_id\":1,\"block_index\":1,"
                        "\"node_index\":44,\"scheduled_ordinal\":19,"
                        "\"producer_node\":null,"
                        "\"producer_scheduled_ordinal\":null,"
                        "\"producer_operation\":null,"
                        "\"producer_descriptor_key\":null,"
                        "\"producer_semantic_tag\":null,"
                        "\"consumer_node\":null,"
                        "\"consumer_scheduled_ordinal\":null,"
                        "\"consumer_operation\":null,"
                        "\"consumer_descriptor_key\":null,"
                        "\"consumer_semantic_tag\":null,"
                        "\"target_count\":0,\"outstanding_before\":2"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"move_causes\":{\"kind_count\":3"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output, IREE_SV("\"static_instruction_mix\":{\"descriptor_count\":9"),
          0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("\"allocation\":{\"assignment_count\":6,"
                                    "\"spill_count\":1,\"spill_plan_count\":1,"
                                    "\"coalesced_copy_count\":2,"
                                    "\"materialized_copy_count\":0,"
                                    "\"storage_lease_count\":11,"
                                    "\"storage_lease_instance_count\":9,"
                                    "\"storage_release_action_count\":3}"),
                            0),
      IREE_STRING_VIEW_NPOS);
  const iree_string_view_t split_memory_mix_fields = IREE_SV(
      "\"global_load_count\":1,\"global_store_count\":0,"
      "\"buffer_load_count\":1");
  const iree_host_size_t first_split_memory_mix =
      iree_string_view_find(output, split_memory_mix_fields, 0);
  EXPECT_NE(first_split_memory_mix, IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, split_memory_mix_fields,
                                  first_split_memory_mix + 1),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"target_resources\":{\"scalar_register_class\":"
                        "\"amdgpu.sgpr\",\"scalar_register_count\":38,"
                        "\"scalar_pressure_peak_live_units\":32,"
                        "\"scalar_register_overhead_units\":6,"
                        "\"vector_register_class\":\"amdgpu.vgpr\","
                        "\"vector_register_count\":112,"
                        "\"vector_pressure_peak_live_units\":96,"
                        "\"vector_register_overhead_units\":16,"
                        "\"subgroup_size\":32,"
                        "\"max_subgroups_per_simd\":16,"
                        "\"resident_subgroups_per_simd\":8,"
                        "\"occupancy_percent\":50,\"limiting_resource\":"
                        "\"amdgpu.vgpr\"}"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("\"wmma_count\":1"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("\"cause\":\"low_concat\""), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"pressure_rows\":{\"count\":2,"
                                          "\"rows\":["),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"pressure_origin_rows\":{\"count\":1,\"rows\":["), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output, IREE_SV("\"schedule_band_rows\":{\"count\":1,\"rows\":["), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"schedule_band_rows\":{\"count\":1,\"rows\":[{"
                        "\"index\":0,\"function\":\"branchy_export\","
                        "\"block\":\"body\",\"block_index\":2"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"schedule_band_summary_rows\":{\"count\":1,"
                        "\"rows\":["),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output,
          IREE_SV("\"schedule_band_summary_rows\":{\"count\":1,"
                  "\"rows\":[{\"index\":0,\"function\":\"branchy_export\","
                  "\"block\":\"body\",\"block_index\":2"),
          0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("\"band_count\":3"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("\"max_band_node_count\":4"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output, IREE_SV("\"origin_kind\":13,\"origin\":\"local-memory\""), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"semantic_tag\":\"memory.workgroup.load2.u32\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output, IREE_SV("\"static_instruction_mix\":{\"descriptor_count\":4"),
          0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("\"result_unit_count\":16"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"origin_kind\":11,\"origin\":\"dot\","
                        "\"origin_operation\":"
                        "\"low.op<amdgpu.v_dot4_i32_i8>\","
                        "\"semantic_tag\":\"dot.i32.i8\","
                        "\"sample_value\":\"acc\",\"live_units\":64,"
                        "\"live_values\":8"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"register_class\":\"amdgpu.sgpr\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"register_class\":\"amdgpu.vgpr\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"spill_rows\":{\"count\":1,"
                                          "\"rows\":["),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output,
          IREE_SV("\"allocation_failure_rows\":{\"count\":1,"
                  "\"rows\":[{\"index\":0,\"function\":\"branchy\","
                  "\"value\":\"blocked\",\"register_class\":\"test.scc\""),
          0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"failure_code\":\"unspillable-register-exhausted\","
                        "\"blocking_kind\":\"active-assignment\""),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"conflict_value\":\"leader\","
                                          "\"conflict_start_point\":0,"
                                          "\"conflict_end_point\":5"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"allocation_high_water_rows\":{\"count\":1,"
                        "\"rows\":[{\"index\":0,\"function\":"
                        "\"branchy_export\",\"value\":\"rhs_window\","
                        "\"register_class\":\"amdgpu.vgpr\""),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"origin_kind\":13,\"origin\":\"local-memory\","
                        "\"origin_operation\":"
                        "\"low.op<amdgpu.ds_load_b128>\","
                        "\"semantic_tag\":\"memory.workgroup.load.u128\""),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"location_kind\":\"physical_register\","
                        "\"location_base\":248,\"location_count\":4,"
                        "\"high_water_units\":252,"
                        "\"lower_free_unit_count\":13,"
                        "\"lower_free_run_count\":3,"
                        "\"lower_largest_free_run_unit_count\":6,"
                        "\"lower_pressure_releasable_free_unit_count\":21,"
                        "\"lower_pressure_releasable_free_run_count\":2,"
                        "\"lower_pressure_releasable_largest_free_run_unit_"
                        "count\":14,"
                        "\"active_assignment_blocker_count\":47,"
                        "\"active_assignment_blocker_units\":244,"
                        "\"active_storage_lease_blocker_count\":3,"
                        "\"active_storage_lease_blocker_units\":12,"
                        "\"active_pressure_storage_lease_blocker_count\":2,"
                        "\"active_pressure_storage_lease_blocker_units\":8,"
                        "\"active_fallback_storage_lease_blocker_count\":1,"
                        "\"active_fallback_storage_lease_blocker_units\":4"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"math_legalization\":{\"rewritten_op_count\":1,"
                        "\"rejected_op_count\":0,"
                        "\"missing_policy_op_count\":0,"
                        "\"missing_recipe_op_count\":0,\"count\":1,"
                        "\"rows\":[{\"index\":0,\"function\":\"branchy\","
                        "\"source_op\":\"scalar.roundf\""),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"policy\":\"amdgpu-math\","
                        "\"constraint_key\":\"math.recipe.round_away_f32\","
                        "\"math_op\":\"roundf\",\"lane_domain\":\"scalar\","
                        "\"element_type\":\"f32\",\"action\":\"rewritten\","
                        "\"recipe\":\"round-away-f32\","
                        "\"source_fastmath_flags\":0,"
                        "\"recipe_fastmath_flags\":0,"
                        "\"created_op_count\":10,\"erased_op_count\":1"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"source_low\":{\"selected_op_count\":4"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("\"memory_count\":1"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("\"rule_set_index\":0,"
                                    "\"rule_index\":1,\"plan_id\":null,"
                                    "\"plan_key\":"
                                    "\"test.scalar_addi.strategy.native\","
                                    "\"descriptor_id\":7"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"memory_rows\":[{\"index\":0,\"function\":"
                        "\"branchy\",\"source_op\":\"vector.load\","
                        "\"source_op_kind\":43,\"memory_space\":\"workgroup\","
                        "\"operation\":\"load\",\"packet\":"
                        "\"amdgpu.ds_read2_b32\","
                        "\"address_form\":\"ds_2addr\","
                        "\"dynamic_term_kind\":\"vaddr\","
                        "\"fallback_reason\":\"cross_wave_workgroup\","
                        "\"static_offset_bytes\":0,"
                        "\"element_bytes\":4,\"vector_lanes\":2,"
                        "\"dynamic_stride_bytes\":32,"
                        "\"vector_lane_stride_bytes\":8,"
                        "\"bank_stride_words\":8,"
                        "\"bank_conflict_degree\":8,\"bank_conflict_kind\":"
                        "\"bank-conflict-risk\"}]"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"target_legalization\":{\"legal_op_count\":0,"
                        "\"rewritten_op_count\":1,"
                        "\"target_rewritten_op_count\":0,"
                        "\"reference_rewritten_op_count\":1"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("\"legalizer\":\"vector\","
                                    "\"legalizer_strategy\":\"reference\","
                                    "\"mode\":\"final\","
                                    "\"policy\":\"reference-only\","
                                    "\"action\":\"rewritten\","
                                    "\"legalization_outcome\":"
                                    "\"reference-fallback\","
                                    "\"contract_outcome\":\"unsupported\""),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"source_rejection_detail\":4"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"created_op_count\":6,"
                                          "\"erased_op_count\":1"),
                                  0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&entry_report);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsJsonSummaryWithoutDetailRows) {
  loom_target_compile_report_pressure_row_t pressure_rows[] = {
      {
          /*.function_name=*/IREE_SVL("summary_only"),
          /*.register_class=*/IREE_SVL("test.i32"),
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/LOOM_SCALAR_TYPE_I32,
          /*.peak_live_units=*/7,
          /*.peak_live_values=*/4,
          /*.peak_point=*/3,
      },
  };
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE;
  report.backend_name = IREE_SVL("hal");
  loom_target_compile_report_record_artifact_size(&report, 256);
  IREE_ASSERT_OK(loom_target_compile_report_record_pressure_row(
      &report, &pressure_rows[0]));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"artifact_kind\":\"hal-executable\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("\"backend\":\"hal\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("\"artifact_size\":256"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"entries\":{\"count\":0,\"rows\":[]}"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"pressure_rows\":{\"count\":1}"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(
                output, IREE_SV("\"pressure_rows\":{\"count\":1,\"rows\""), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsJsonEscapedStrings) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.backend_name = IREE_SVL("quote\"line\n");

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"backend\":\"quote\\\"line\\n\""), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, JsonModeNoneWritesNothing) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  EXPECT_EQ(iree_string_builder_size(&builder), 0u);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, ParsesModes) {
  loom_target_compile_report_format_mode_t mode =
      LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE;
  IREE_ASSERT_OK(
      loom_target_compile_report_format_mode_parse(IREE_SV("summary"), &mode));
  EXPECT_EQ(mode, LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_compile_report_format_mode_parse(IREE_SV("verbose"), &mode));
}

}  // namespace
}  // namespace loom
