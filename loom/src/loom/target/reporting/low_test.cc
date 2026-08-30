// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/low.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/target/facts_builder.h"
#include "loom/target/registers.h"
#include "loom/target/test/target_records.h"

namespace loom {
namespace {

template <typename T>
static const T* CompileReportRowAt(
    const loom_target_compile_report_row_list_t& row_list,
    iree_host_size_t index) {
  for (const loom_target_compile_report_vec_t* vec = row_list.head; vec != NULL;
       vec = vec->next) {
    if (index < vec->count) {
      const T* rows =
          static_cast<const T*>(loom_target_compile_report_vec_const_rows(vec));
      return &rows[index];
    }
    index -= vec->count;
  }
  return NULL;
}

TEST(CompileReportLowTest, RecordsPressureSpillAndAllocationFailureRows) {
  constexpr uint32_t kSourceAssignmentIndex = 0;
  constexpr uint32_t kResultAssignmentIndex = 1;
  constexpr uint32_t kEdgeCopyCount = 1;
  constexpr uint32_t kRegisterCopyTagOffset = 0;
  constexpr uint32_t kMemoryGlobalTagOffset =
      kRegisterCopyTagOffset + sizeof("register.copy.b32");
  constexpr uint32_t kMemoryStackLoadTagOffset =
      kMemoryGlobalTagOffset + sizeof("memory.global.load.u32");
  constexpr uint32_t kMemoryStackStoreTagOffset =
      kMemoryStackLoadTagOffset + sizeof("memory.stack.load.u32");
  constexpr uint32_t kMatrixWmmaTagOffset =
      kMemoryStackStoreTagOffset + sizeof("memory.stack.store.u128");
  constexpr uint32_t kMatrixSwmmacTagOffset =
      kMatrixWmmaTagOffset + sizeof("matrix.wmma.f32");
  constexpr uint32_t kMatrixSmfmacTagOffset =
      kMatrixSwmmacTagOffset + sizeof("matrix.swmmac.f32");
  constexpr uint32_t kRegisterClassGprOffset =
      kMatrixSmfmacTagOffset + sizeof("matrix.smfmac.f32");
  static const uint8_t kDescriptorStringTable[] =
      "\x11"
      "register.copy.b32"
      "\x16"
      "memory.global.load.u32"
      "\x15"
      "memory.stack.load.u32"
      "\x17"
      "memory.stack.store.u128"
      "\x0f"
      "matrix.wmma.f32"
      "\x11"
      "matrix.swmmac.f32"
      "\x11"
      "matrix.smfmac.f32"
      "\x08"
      "test.gpr";
  loom_low_descriptor_t descriptors[8] = {};
  loom_low_descriptor_view_t descriptor_views[8] = {};
  descriptors[0].semantic_tag_string_offset = kRegisterCopyTagOffset;
  descriptors[1].semantic_tag_string_offset = kMemoryGlobalTagOffset;
  descriptors[2].semantic_tag_string_offset = kMemoryStackLoadTagOffset;
  descriptors[2].effect_count = 1;
  descriptors[3].semantic_tag_string_offset = kMemoryStackStoreTagOffset;
  descriptors[3].effect_start = 1;
  descriptors[3].effect_count = 1;
  descriptors[4].semantic_tag_string_offset = kMatrixWmmaTagOffset;
  descriptors[5].semantic_tag_string_offset = kMemoryGlobalTagOffset;
  descriptors[6].semantic_tag_string_offset = kMatrixSwmmacTagOffset;
  descriptors[7].semantic_tag_string_offset = kMatrixSmfmacTagOffset;
  const loom_low_effect_t effects[] = {
      {
          /*.kind=*/LOOM_LOW_EFFECT_KIND_READ,
          /*.memory_space=*/{},
          /*.scope_id=*/{},
          /*.flags=*/{},
          /*.counter_id=*/{},
          /*.width_bits=*/32,
      },
      {
          /*.kind=*/LOOM_LOW_EFFECT_KIND_WRITE,
          /*.memory_space=*/{},
          /*.scope_id=*/{},
          /*.flags=*/{},
          /*.counter_id=*/{},
          /*.width_bits=*/128,
      },
  };
  const loom_low_schedule_class_t schedule_classes[5] = {};
  const loom_low_reg_class_t reg_classes[] = {
      {
          /*.name_string_offset=*/kRegisterClassGprOffset,
          /*.target_bank_id=*/{},
          /*.flags=*/LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
          /*.alloc_unit_bits=*/32,
          /*.allocatable_count=*/{},
          /*.fixed_location_base=*/{},
          /*.fixed_location_count=*/{},
          /*.alias_set_id=*/{},
          /*.spill_class_id=*/LOOM_LOW_REG_CLASS_NONE,
          /*.full_register_part_mask=*/1,
          /*.spill_slot_space=*/LOOM_LOW_SPILL_SLOT_SPACE_STACK,
      },
  };
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.stable_id = 1;
  descriptor_set.string_table = {
      /*.data=*/kDescriptorStringTable,
      /*.data_length=*/sizeof(kDescriptorStringTable) - 1,
  };
  descriptor_set.descriptors = descriptors;
  descriptor_set.descriptor_views = descriptor_views;
  descriptor_set.descriptor_count = IREE_ARRAYSIZE(descriptors);
  descriptor_set.effects = effects;
  descriptor_set.effect_count = IREE_ARRAYSIZE(effects);
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = IREE_ARRAYSIZE(reg_classes);
  descriptor_set.schedule_classes = schedule_classes;
  descriptor_set.schedule_class_count = IREE_ARRAYSIZE(schedule_classes);
  descriptor_views[0].schedule_class_id = 0;
  descriptor_views[1].schedule_class_id = 1;
  descriptor_views[2].schedule_class_id = 1;
  descriptor_views[3].schedule_class_id = 2;
  descriptor_views[4].schedule_class_id = 3;
  descriptor_views[5].schedule_class_id = 1;
  descriptor_views[6].schedule_class_id = 3;
  descriptor_views[7].schedule_class_id = 4;
  descriptor_views[0].instruction_class_flags =
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_VECTOR_ALU |
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_REGISTER_MOVE;
  descriptor_views[1].instruction_class_flags =
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_GLOBAL_MEMORY |
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_GLOBAL_LOAD;
  descriptor_views[2].instruction_class_flags =
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_PRIVATE_MEMORY;
  descriptor_views[3].instruction_class_flags =
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_PRIVATE_MEMORY;
  descriptor_views[4].instruction_class_flags =
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_MATRIX |
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_WMMA;
  descriptor_views[5].instruction_class_flags =
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_GLOBAL_MEMORY |
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_BUFFER_LOAD;
  descriptor_views[6].instruction_class_flags =
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_MATRIX |
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_WMMA |
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_SWMMAC;
  descriptor_views[7].instruction_class_flags =
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_MATRIX |
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_MFMA |
      LOOM_LOW_INSTRUCTION_CLASS_FLAG_SMFMAC;
  loom_low_schedule_node_t schedule_nodes[13] = {};
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                   &block_pool);
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_context_finalize(&context));
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("test"), &block_pool,
                                      NULL, iree_allocator_system(), &module));
  for (uint32_t i = 0; i < 7; ++i) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(
        loom_module_define_value(module, (loom_type_t){0}, &value_id));
    EXPECT_EQ(value_id, i);
  }
  IREE_ASSERT_OK(loom_module_set_value_type(
      module, 4,
      loom_low_register_type(/*descriptor_set_stable_id=*/1,
                             /*register_class_id=*/0, 1)));
  IREE_ASSERT_OK(loom_module_set_value_type(
      module, 5,
      loom_low_register_type(/*descriptor_set_stable_id=*/1,
                             /*register_class_id=*/0, 2)));
  IREE_ASSERT_OK(loom_module_set_value_type(
      module, 6,
      loom_low_register_type(/*descriptor_set_stable_id=*/1,
                             /*register_class_id=*/0, 2)));
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS;

  const loom_op_t peak_op = {};
  const loom_liveness_pressure_summary_t pressure_summaries[] = {
      {
          /*.value_class=*/{
              /*.type_kind=*/LOOM_TYPE_REGISTER,
              /*.element_type=*/LOOM_SCALAR_TYPE_I32,
              /*.register_descriptor_set_stable_id=*/1,
              /*.register_class_id=*/0,
          },
          /*.peak_live_units=*/7,
          /*.peak_live_values=*/5,
          /*.peak_block=*/{},
          /*.peak_op=*/{},
          /*.peak_point=*/3,
      },
      {
          /*.value_class=*/{
              /*.type_kind=*/LOOM_TYPE_REGISTER,
              /*.element_type=*/LOOM_SCALAR_TYPE_F32,
              /*.register_descriptor_set_stable_id=*/1,
              /*.register_class_id=*/0,
          },
          /*.peak_live_units=*/11,
          /*.peak_live_values=*/2,
          /*.peak_block=*/{},
          /*.peak_op=*/&peak_op,
          /*.peak_point=*/9,
      },
  };
  const loom_low_allocation_assignment_t assignments[] = {
      {
          /*.value_id=*/4,
          /*.value_class=*/pressure_summaries[0].value_class,
          /*.descriptor_reg_class_id=*/0,
          /*.flags=*/{},
          /*.start_point=*/{},
          /*.end_point=*/{},
          /*.unit_count=*/{},
          /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
          /*.location_base=*/0,
          /*.location_count=*/1,
      },
      {
          /*.value_id=*/5,
          /*.value_class=*/pressure_summaries[1].value_class,
          /*.descriptor_reg_class_id=*/0,
          /*.flags=*/{},
          /*.start_point=*/{},
          /*.end_point=*/{},
          /*.unit_count=*/{},
          /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
          /*.location_base=*/1,
          /*.location_count=*/1,
      },
      {
          /*.value_id=*/6,
          /*.value_class=*/pressure_summaries[0].value_class,
          /*.descriptor_reg_class_id=*/0,
          /*.flags=*/{},
          /*.start_point=*/2,
          /*.end_point=*/6,
          /*.unit_count=*/2,
          /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          /*.location_base=*/7,
          /*.location_count=*/2,
      },
  };
  const loom_value_id_t liveness_value_ids[] = {
      4,
      5,
      6,
  };
  const uint32_t liveness_value_interval_indices[] = {
      0,
      1,
      2,
  };
  const loom_liveness_interval_t liveness_intervals[] = {
      {
          /*.value_id=*/4,
          /*.start_point=*/0,
          /*.end_point=*/8,
          /*.value_class=*/pressure_summaries[0].value_class,
          /*.unit_count=*/1,
      },
      {
          /*.value_id=*/5,
          /*.start_point=*/1,
          /*.end_point=*/12,
          /*.value_class=*/pressure_summaries[1].value_class,
          /*.unit_count=*/2,
      },
      {
          /*.value_id=*/6,
          /*.start_point=*/2,
          /*.end_point=*/6,
          /*.value_class=*/pressure_summaries[0].value_class,
          /*.unit_count=*/2,
      },
  };
  const uint32_t assignment_indices_by_value_ordinal[] = {
      0,
      1,
      2,
  };
  const loom_low_allocation_copy_decision_t copy_decisions[] = {
      {
          /*.source_value_id=*/4,
          /*.result_value_id=*/5,
          /*.source_assignment_index=*/kSourceAssignmentIndex,
          /*.result_assignment_index=*/kResultAssignmentIndex,
          /*.kind=*/LOOM_LOW_ALLOCATION_COPY_MATERIALIZED,
      },
  };
  const loom_low_move_t moves[] = {
      {
          /*.destination=*/
          {
              /*.location_kind=*/
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
              /*.value_class=*/pressure_summaries[0].value_class,
              /*.descriptor_reg_class_id=*/0,
              /*.location=*/1,
          },
          /*.source=*/
          {
              /*.location_kind=*/
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
              /*.value_class=*/pressure_summaries[0].value_class,
              /*.descriptor_reg_class_id=*/0,
              /*.location=*/0,
          },
      },
      {
          /*.destination=*/
          {
              /*.location_kind=*/
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
              /*.value_class=*/pressure_summaries[0].value_class,
              /*.descriptor_reg_class_id=*/0,
              /*.location=*/2,
          },
          /*.source=*/
          {
              /*.location_kind=*/
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
              /*.value_class=*/pressure_summaries[0].value_class,
              /*.descriptor_reg_class_id=*/0,
              /*.location=*/1,
          },
      },
  };
  const loom_low_allocation_packet_move_group_t packet_move_groups[] = {
      {
          /*.source_ordinal=*/0,
          /*.cause=*/LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY,
          /*.move_group=*/
          {
              /*.moves=*/
              {
                  /*.start=*/0,
                  /*.count=*/1,
              },
          },
      },
  };
  const loom_low_allocation_edge_copy_t edge_copies[kEdgeCopyCount] = {
      {
          /*.payload_index=*/0,
          /*.source_value_id=*/4,
          /*.destination_value_id=*/5,
          /*.source_assignment_index=*/kSourceAssignmentIndex,
          /*.destination_assignment_index=*/kResultAssignmentIndex,
          /*.source_unit_offset=*/0,
          /*.destination_unit_offset=*/0,
          /*.unit_count=*/1,
      },
  };
  const loom_low_allocation_edge_copy_group_t edge_copy_groups[] = {
      {
          /*.terminator_op=*/{},
          /*.source_ordinal=*/{},
          /*.copy_start=*/0,
          /*.copy_count=*/kEdgeCopyCount,
          /*.move_group=*/
          {
              /*.moves=*/
              {
                  /*.start=*/1,
                  /*.count=*/1,
              },
          },
      },
  };
  const loom_low_allocation_spill_plan_t spill_plans[] = {
      {
          /*.value_id=*/4,
          /*.assignment_index=*/0,
          /*.slot_index=*/0,
          /*.slot_space=*/LOOM_LOW_SPILL_SLOT_SPACE_STACK,
          /*.byte_size=*/16,
          /*.byte_alignment=*/8,
          /*.store_count=*/1,
          /*.reload_count=*/2,
      },
      {
          /*.value_id=*/5,
          /*.assignment_index=*/1,
          /*.slot_index=*/1,
          /*.slot_space=*/LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH,
          /*.byte_size=*/32,
          /*.byte_alignment=*/16,
          /*.store_count=*/3,
          /*.reload_count=*/4,
      },
  };
  const loom_low_allocation_materialized_spill_t materialized_spills[] = {
      {
          /*.value_id=*/6,
          /*.value_class=*/pressure_summaries[0].value_class,
          /*.flags=*/0,
          /*.assignment_index=*/2,
          /*.slot_index=*/7,
          /*.slot_space=*/LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE,
          /*.byte_size=*/64,
          /*.byte_alignment=*/16,
          /*.store_count=*/5,
          /*.store_bytes=*/320,
          /*.reload_count=*/6,
          /*.reload_bytes=*/384,
      },
      {
          /*.value_id=*/4,
          /*.value_class=*/pressure_summaries[0].value_class,
          /*.flags=*/
          LOOM_LOW_ALLOCATION_MATERIALIZED_SPILL_FLAG_VALUE_WAS_BLOCK_ARGUMENT,
          /*.assignment_index=*/0,
          /*.slot_index=*/8,
          /*.slot_space=*/LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE,
          /*.byte_size=*/4,
          /*.byte_alignment=*/4,
          /*.store_count=*/2,
          /*.store_bytes=*/8,
          /*.reload_count=*/3,
          /*.reload_bytes=*/12,
      },
  };
  loom_low_allocation_materialized_spill_vec_t materialized_spill_vec = {
      /*.records=*/materialized_spills,
      /*.record_count=*/IREE_ARRAYSIZE(materialized_spills),
      /*.next=*/{},
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(descriptors); ++i) {
    schedule_nodes[i].descriptor = &descriptors[i];
    schedule_nodes[i].schedule_class =
        &schedule_classes[descriptor_views[i].schedule_class_id];
    schedule_nodes[i].source_ordinal = i;
    schedule_nodes[i].scheduled_ordinal = i;
    schedule_nodes[i].memory_access_record_index =
        LOOM_LOW_SCHEDULE_MEMORY_ACCESS_RECORD_NONE;
    schedule_nodes[i].kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
  }
  schedule_nodes[0].operand_count = 1;
  schedule_nodes[0].result_count = 1;
  schedule_nodes[0].value_ordinals.inline_value_ordinals[0] = 0;
  schedule_nodes[0].value_ordinals.inline_value_ordinals[1] = 1;
  schedule_nodes[1].result_count = 1;
  schedule_nodes[1].value_ordinals.inline_value_ordinals[0] = 2;
  const loom_low_schedule_block_t schedule_blocks[] = {
      {
          /*.block=*/{},
          /*.node_start=*/0,
          /*.node_count=*/8,
          /*.scheduled_node_start=*/0,
          /*.scheduled_node_count=*/8,
      },
  };
  const uint32_t scheduled_node_indices[] = {0, 1, 2, 3, 4, 5, 6, 7};
  loom_target_facts_t target_facts = {};
  const loom_target_bundle_t* target_bundle = loom_target_bundle_table_lookup(
      &loom_test_target_bundles, LOOM_TEST_TARGET_KIND_LOW_CORE);
  ASSERT_NE(target_bundle, nullptr);
  loom_target_facts_builder_initialize(&loom_test_target_fact_type,
                                       target_bundle, &target_facts);
  const loom_low_resolved_target_t target = {
      /*.target_facts=*/&target_facts,
      /*.target_name=*/target_bundle->name,
      /*.descriptor_set_key=*/target_bundle->config->contract_set_key,
      /*.feature_bits=*/target_bundle->config->contract_feature_bits,
      /*.descriptor_set=*/&descriptor_set,
  };
  loom_low_schedule_table_t schedule = {};
  schedule.module = module;
  schedule.target = target;
  schedule.blocks = schedule_blocks;
  schedule.block_count = IREE_ARRAYSIZE(schedule_blocks);
  schedule.nodes = schedule_nodes;
  schedule.node_count = 13;
  schedule.dependencies.count = 6;
  schedule.scheduled_node_indices = scheduled_node_indices;
  schedule.scheduled_node_count = IREE_ARRAYSIZE(scheduled_node_indices);
  schedule.resource_use_count = 4;
  schedule.hazard_gap_count = 2;
  schedule.model_summary_count = 1;

  loom_low_emission_frame_t frame = {};
  frame.target = target;
  frame.schedule = schedule;
  frame.allocation.module = module;
  frame.allocation.target = target;
  frame.allocation.liveness.intervals = liveness_intervals;
  frame.allocation.liveness.interval_count = IREE_ARRAYSIZE(liveness_intervals);
  frame.allocation.liveness.value_ids = liveness_value_ids;
  frame.allocation.liveness.value_count = IREE_ARRAYSIZE(liveness_value_ids);
  frame.allocation.liveness.value_interval_indices =
      liveness_value_interval_indices;
  frame.allocation.liveness.pressure_summaries = pressure_summaries;
  frame.allocation.liveness.pressure_summary_count =
      IREE_ARRAYSIZE(pressure_summaries);
  frame.allocation.error_count = 1;
  frame.allocation.assignments = assignments;
  frame.allocation.assignment_count = IREE_ARRAYSIZE(assignments);
  frame.allocation.assignment_indices_by_value_ordinal =
      assignment_indices_by_value_ordinal;
  frame.allocation.spill_plans = spill_plans;
  frame.allocation.spill_plan_count = IREE_ARRAYSIZE(spill_plans);
  frame.allocation.failure.failure_code =
      IREE_SVL("unspillable-register-exhausted");
  frame.allocation.failure.value_id = 5;
  frame.allocation.failure.value_class = pressure_summaries[1].value_class;
  frame.allocation.failure.descriptor_reg_class_id = 0;
  frame.allocation.failure.start_point = 3;
  frame.allocation.failure.end_point = 8;
  frame.allocation.failure.required_unit_count = 2;
  frame.allocation.failure.budget_units = 1;
  frame.allocation.failure.peak_live_units = 11;
  frame.allocation.failure.location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  frame.allocation.failure.location_base = 0;
  frame.allocation.failure.location_count = 2;
  frame.allocation.failure.blocking_kind =
      LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT;
  frame.allocation.failure.conflict_assignment_index = 0;
  frame.allocation.failure.conflict_value_id = 4;
  frame.allocation.failure.conflict_start_point = 0;
  frame.allocation.failure.conflict_end_point = 8;
  frame.allocation.failure.conflict_location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  frame.allocation.failure.conflict_location_base = 0;
  frame.allocation.failure.conflict_location_count = 1;
  frame.allocation.copy_decisions = copy_decisions;
  frame.allocation.copy_decision_count = IREE_ARRAYSIZE(copy_decisions);
  frame.allocation.edge_copies = edge_copies;
  frame.allocation.edge_copy_count = IREE_ARRAYSIZE(edge_copies);
  frame.allocation.edge_copy_groups = edge_copy_groups;
  frame.allocation.edge_copy_group_count = IREE_ARRAYSIZE(edge_copy_groups);
  frame.allocation.packet_move_groups = packet_move_groups;
  frame.allocation.packet_move_group_count = IREE_ARRAYSIZE(packet_move_groups);
  frame.allocation.moves = moves;
  frame.allocation.packet_move_count = 1;
  frame.allocation.spill_count = IREE_ARRAYSIZE(spill_plans);
  frame.allocation.coalesced_copy_count = 3;
  frame.allocation.materialized_copy_count = 1;
  frame.materialized_spill_storage_count = 4;
  frame.materialized_spill_storage_bytes = 40;
  frame.materialized_spill_store_count = 5;
  frame.materialized_spill_store_bytes = 50;
  frame.materialized_reload_count = 6;
  frame.materialized_reload_bytes = 60;
  frame.materialized_spills.head = &materialized_spill_vec;
  frame.materialized_spills.tail = &materialized_spill_vec;
  frame.materialized_spills.record_count = IREE_ARRAYSIZE(materialized_spills);

  IREE_ASSERT_OK(
      loom_target_compile_report_record_low_emission_frame(&report, &frame));

  EXPECT_TRUE(iree_all_bits_set(report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE));
  EXPECT_TRUE(iree_all_bits_set(report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION));
  EXPECT_TRUE(iree_all_bits_set(
      report.detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX));
  EXPECT_TRUE(iree_all_bits_set(report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES));
  EXPECT_TRUE(iree_all_bits_set(
      report.detail_flags, LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS));
  EXPECT_TRUE(iree_all_bits_set(
      report.detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS));
  EXPECT_TRUE(
      iree_all_bits_set(report.detail_flags,
                        LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS));
  EXPECT_TRUE(iree_all_bits_set(
      report.detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS));
  EXPECT_TRUE(iree_all_bits_set(report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS));
  EXPECT_TRUE(iree_all_bits_set(
      report.detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS));
  EXPECT_TRUE(iree_all_bits_set(
      report.detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS));
  EXPECT_TRUE(
      iree_string_view_equal(report.function_name, IREE_SV("<unnamed>")));
  EXPECT_EQ(report.schedule_node_count, 13u);
  EXPECT_EQ(report.register_pressure_summary_count, 2u);
  EXPECT_EQ(report.register_pressure_peak_live_units, 11u);
  EXPECT_EQ(report.allocation_spill_count, 2u);
  EXPECT_EQ(report.allocation_materialized_spill_storage_count, 4u);
  EXPECT_EQ(report.allocation_materialized_spill_store_count, 5u);
  EXPECT_EQ(report.allocation_materialized_reload_count, 6u);
  EXPECT_EQ(report.static_instruction_mix.descriptor_count, 8u);
  EXPECT_EQ(report.static_instruction_mix.vector_alu_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.global_memory_count, 2u);
  EXPECT_EQ(report.static_instruction_mix.global_load_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.buffer_load_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.private_memory_count, 2u);
  EXPECT_EQ(report.static_instruction_mix.memory_read_byte_count, 4u);
  EXPECT_EQ(report.static_instruction_mix.memory_write_byte_count, 16u);
  EXPECT_EQ(report.static_instruction_mix.private_read_byte_count, 4u);
  EXPECT_EQ(report.static_instruction_mix.private_write_byte_count, 16u);
  EXPECT_EQ(report.static_instruction_mix.unclassified_read_byte_count, 0u);
  EXPECT_EQ(report.static_instruction_mix.unclassified_write_byte_count, 0u);
  EXPECT_EQ(report.static_instruction_mix.matrix_count, 3u);
  EXPECT_EQ(report.static_instruction_mix.mfma_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.smfmac_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.wmma_count, 2u);
  EXPECT_EQ(report.static_instruction_mix.swmmac_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.register_move_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.unknown_count, 0u);
  EXPECT_EQ(report.move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY]
                .packet_count,
            1u);
  EXPECT_EQ(report.move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY]
                .unit_count,
            1u);
  EXPECT_EQ(
      report.move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE]
          .packet_count,
      1u);
  EXPECT_EQ(
      report.move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE]
          .unit_count,
      1u);
  EXPECT_EQ(
      report
          .move_causes
              [LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION]
          .packet_count,
      1u);
  EXPECT_EQ(
      report
          .move_causes
              [LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION]
          .unit_count,
      2u);
  EXPECT_EQ(report.pressure_rows.count, 2u);
  ASSERT_NE(report.pressure_rows.head, nullptr);
  const auto* pressure_rows =
      static_cast<const loom_target_compile_report_pressure_row_t*>(
          loom_target_compile_report_vec_const_rows(report.pressure_rows.head));
  EXPECT_TRUE(iree_string_view_equal(pressure_rows[0].function_name,
                                     IREE_SV("<unnamed>")));
  EXPECT_EQ(pressure_rows[0].peak_live_units, 7u);
  EXPECT_EQ(pressure_rows[0].peak_live_values, 5u);
  EXPECT_TRUE(iree_string_view_equal(pressure_rows[0].peak_operation_name,
                                     IREE_SV("<block-boundary>")));
  EXPECT_EQ(pressure_rows[1].peak_live_units, 11u);
  EXPECT_EQ(report.pressure_origin_rows.count, 3u);
  ASSERT_NE(report.pressure_origin_rows.head, nullptr);
  const auto* pressure_origin_rows =
      static_cast<const loom_target_compile_report_pressure_origin_row_t*>(
          loom_target_compile_report_vec_const_rows(
              report.pressure_origin_rows.head));
  EXPECT_EQ(pressure_origin_rows[0].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN);
  EXPECT_EQ(pressure_origin_rows[0].live_units, 1u);
  EXPECT_EQ(pressure_origin_rows[0].live_values, 1u);
  EXPECT_EQ(pressure_origin_rows[1].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY);
  EXPECT_TRUE(iree_string_view_equal(pressure_origin_rows[1].semantic_tag,
                                     IREE_SV("memory.global.load.u32")));
  EXPECT_EQ(pressure_origin_rows[1].live_units, 2u);
  EXPECT_EQ(pressure_origin_rows[1].live_values, 1u);
  EXPECT_EQ(pressure_origin_rows[2].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_REGISTER_MOVE);
  EXPECT_TRUE(iree_string_view_equal(pressure_origin_rows[2].semantic_tag,
                                     IREE_SV("register.copy.b32")));
  EXPECT_EQ(pressure_origin_rows[2].live_units, 2u);
  EXPECT_EQ(pressure_origin_rows[2].live_values, 1u);
  EXPECT_EQ(report.schedule_band_rows.count, 8u);
  ASSERT_NE(report.schedule_band_rows.head, nullptr);
  loom_target_compile_report_schedule_band_row_t schedule_band_rows[8] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(schedule_band_rows); ++i) {
    const auto* row =
        CompileReportRowAt<loom_target_compile_report_schedule_band_row_t>(
            report.schedule_band_rows, i);
    ASSERT_NE(row, nullptr);
    schedule_band_rows[i] = *row;
  }
  EXPECT_EQ(schedule_band_rows[0].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_REGISTER_MOVE);
  EXPECT_EQ(schedule_band_rows[0].block_index, 0u);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_rows[0].semantic_tag,
                                     IREE_SV("register.copy.b32")));
  EXPECT_EQ(schedule_band_rows[0].node_count, 1u);
  EXPECT_EQ(schedule_band_rows[0].first_scheduled_ordinal, 0u);
  EXPECT_EQ(schedule_band_rows[0].static_instruction_mix.descriptor_count, 1u);
  EXPECT_EQ(schedule_band_rows[0].static_instruction_mix.vector_alu_count, 1u);
  EXPECT_EQ(schedule_band_rows[0].static_instruction_mix.register_move_count,
            1u);
  EXPECT_EQ(schedule_band_rows[0].result_value_count, 1u);
  EXPECT_EQ(schedule_band_rows[0].result_unit_count, 2u);
  EXPECT_EQ(schedule_band_rows[1].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY);
  EXPECT_EQ(schedule_band_rows[1].block_index, 0u);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_rows[1].semantic_tag,
                                     IREE_SV("memory.global.load.u32")));
  EXPECT_EQ(schedule_band_rows[1].static_instruction_mix.global_memory_count,
            1u);
  EXPECT_EQ(schedule_band_rows[1].static_instruction_mix.global_load_count, 1u);
  EXPECT_EQ(schedule_band_rows[1].result_value_count, 1u);
  EXPECT_EQ(schedule_band_rows[1].result_unit_count, 2u);
  EXPECT_EQ(schedule_band_rows[2].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_PRIVATE_MEMORY);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_rows[2].semantic_tag,
                                     IREE_SV("memory.stack.load.u32")));
  EXPECT_EQ(schedule_band_rows[2].static_instruction_mix.global_memory_count,
            0u);
  EXPECT_EQ(schedule_band_rows[2].static_instruction_mix.private_memory_count,
            1u);
  EXPECT_EQ(
      schedule_band_rows[2].static_instruction_mix.private_read_byte_count, 4u);
  EXPECT_EQ(schedule_band_rows[3].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_PRIVATE_MEMORY);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_rows[3].semantic_tag,
                                     IREE_SV("memory.stack.store.u128")));
  EXPECT_EQ(schedule_band_rows[3].static_instruction_mix.global_memory_count,
            0u);
  EXPECT_EQ(schedule_band_rows[3].static_instruction_mix.private_memory_count,
            1u);
  EXPECT_EQ(
      schedule_band_rows[3].static_instruction_mix.private_write_byte_count,
      16u);
  EXPECT_EQ(schedule_band_rows[4].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_rows[4].semantic_tag,
                                     IREE_SV("matrix.wmma.f32")));
  EXPECT_EQ(schedule_band_rows[4].static_instruction_mix.matrix_count, 1u);
  EXPECT_EQ(schedule_band_rows[4].static_instruction_mix.wmma_count, 1u);
  EXPECT_EQ(schedule_band_rows[4].static_instruction_mix.swmmac_count, 0u);
  EXPECT_EQ(schedule_band_rows[5].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_rows[5].semantic_tag,
                                     IREE_SV("memory.global.load.u32")));
  EXPECT_EQ(schedule_band_rows[5].static_instruction_mix.buffer_load_count, 1u);
  EXPECT_EQ(schedule_band_rows[6].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_rows[6].semantic_tag,
                                     IREE_SV("matrix.swmmac.f32")));
  EXPECT_EQ(schedule_band_rows[6].static_instruction_mix.matrix_count, 1u);
  EXPECT_EQ(schedule_band_rows[6].static_instruction_mix.wmma_count, 1u);
  EXPECT_EQ(schedule_band_rows[6].static_instruction_mix.swmmac_count, 1u);
  EXPECT_EQ(schedule_band_rows[7].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_rows[7].semantic_tag,
                                     IREE_SV("matrix.smfmac.f32")));
  EXPECT_EQ(schedule_band_rows[7].static_instruction_mix.matrix_count, 1u);
  EXPECT_EQ(schedule_band_rows[7].static_instruction_mix.mfma_count, 1u);
  EXPECT_EQ(schedule_band_rows[7].static_instruction_mix.smfmac_count, 1u);
  EXPECT_EQ(report.schedule_band_summary_rows.count, 7u);
  ASSERT_NE(report.schedule_band_summary_rows.head, nullptr);
  loom_target_compile_report_schedule_band_summary_row_t
      schedule_band_summary_rows[7] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(schedule_band_summary_rows);
       ++i) {
    const auto* row = CompileReportRowAt<
        loom_target_compile_report_schedule_band_summary_row_t>(
        report.schedule_band_summary_rows, i);
    ASSERT_NE(row, nullptr);
    schedule_band_summary_rows[i] = *row;
  }
  EXPECT_EQ(schedule_band_summary_rows[0].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_REGISTER_MOVE);
  EXPECT_EQ(schedule_band_summary_rows[0].block_index, 0u);
  EXPECT_EQ(schedule_band_summary_rows[0].band_count, 1u);
  EXPECT_EQ(schedule_band_summary_rows[0].node_count, 1u);
  EXPECT_EQ(schedule_band_summary_rows[0].max_band_node_count, 1u);
  EXPECT_EQ(schedule_band_summary_rows[1].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY);
  EXPECT_EQ(schedule_band_summary_rows[1].block_index, 0u);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_summary_rows[1].semantic_tag,
                                     IREE_SV("memory.global.load.u32")));
  EXPECT_EQ(schedule_band_summary_rows[1].band_count, 2u);
  EXPECT_EQ(schedule_band_summary_rows[1].node_count, 2u);
  EXPECT_EQ(schedule_band_summary_rows[1].max_band_node_count, 1u);
  EXPECT_EQ(
      schedule_band_summary_rows[1].static_instruction_mix.global_memory_count,
      2u);
  EXPECT_EQ(
      schedule_band_summary_rows[1].static_instruction_mix.global_load_count,
      1u);
  EXPECT_EQ(
      schedule_band_summary_rows[1].static_instruction_mix.buffer_load_count,
      1u);
  EXPECT_EQ(schedule_band_summary_rows[2].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_PRIVATE_MEMORY);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_summary_rows[2].semantic_tag,
                                     IREE_SV("memory.stack.load.u32")));
  EXPECT_EQ(schedule_band_summary_rows[2].band_count, 1u);
  EXPECT_EQ(schedule_band_summary_rows[2].node_count, 1u);
  EXPECT_EQ(
      schedule_band_summary_rows[2].static_instruction_mix.private_memory_count,
      1u);
  EXPECT_EQ(schedule_band_summary_rows[2]
                .static_instruction_mix.private_read_byte_count,
            4u);
  EXPECT_EQ(schedule_band_summary_rows[3].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_PRIVATE_MEMORY);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_summary_rows[3].semantic_tag,
                                     IREE_SV("memory.stack.store.u128")));
  EXPECT_EQ(
      schedule_band_summary_rows[3].static_instruction_mix.private_memory_count,
      1u);
  EXPECT_EQ(schedule_band_summary_rows[3]
                .static_instruction_mix.private_write_byte_count,
            16u);
  EXPECT_EQ(schedule_band_summary_rows[4].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_summary_rows[4].semantic_tag,
                                     IREE_SV("matrix.wmma.f32")));
  EXPECT_EQ(schedule_band_summary_rows[4].static_instruction_mix.matrix_count,
            1u);
  EXPECT_EQ(schedule_band_summary_rows[4].static_instruction_mix.wmma_count,
            1u);
  EXPECT_EQ(schedule_band_summary_rows[5].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_summary_rows[5].semantic_tag,
                                     IREE_SV("matrix.swmmac.f32")));
  EXPECT_EQ(schedule_band_summary_rows[5].static_instruction_mix.matrix_count,
            1u);
  EXPECT_EQ(schedule_band_summary_rows[5].static_instruction_mix.wmma_count,
            1u);
  EXPECT_EQ(schedule_band_summary_rows[5].static_instruction_mix.swmmac_count,
            1u);
  EXPECT_EQ(schedule_band_summary_rows[6].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX);
  EXPECT_TRUE(iree_string_view_equal(schedule_band_summary_rows[6].semantic_tag,
                                     IREE_SV("matrix.smfmac.f32")));
  EXPECT_EQ(schedule_band_summary_rows[6].static_instruction_mix.matrix_count,
            1u);
  EXPECT_EQ(schedule_band_summary_rows[6].static_instruction_mix.mfma_count,
            1u);
  EXPECT_EQ(schedule_band_summary_rows[6].static_instruction_mix.smfmac_count,
            1u);
  EXPECT_EQ(report.spill_rows.count, 4u);
  ASSERT_NE(report.spill_rows.head, nullptr);
  const auto* spill_rows =
      static_cast<const loom_target_compile_report_spill_row_t*>(
          loom_target_compile_report_vec_const_rows(report.spill_rows.head));
  EXPECT_EQ(spill_rows[0].kind, LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_PLANNED);
  EXPECT_TRUE(iree_string_view_equal(spill_rows[0].function_name,
                                     IREE_SV("<unnamed>")));
  EXPECT_EQ(spill_rows[0].assignment_index, 0u);
  EXPECT_EQ(spill_rows[0].slot_index, 0u);
  EXPECT_TRUE(
      iree_string_view_equal(spill_rows[0].slot_space, IREE_SV("stack")));
  EXPECT_EQ(spill_rows[0].byte_size, 16u);
  EXPECT_EQ(spill_rows[0].store_count, 1u);
  EXPECT_EQ(spill_rows[0].store_bytes, 16u);
  EXPECT_EQ(spill_rows[0].reload_count, 2u);
  EXPECT_EQ(spill_rows[0].reload_bytes, 32u);
  EXPECT_EQ(spill_rows[0].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN);
  EXPECT_EQ(spill_rows[1].kind, LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_PLANNED);
  EXPECT_EQ(spill_rows[1].slot_index, 1u);
  EXPECT_EQ(spill_rows[1].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_REGISTER_MOVE);
  EXPECT_TRUE(iree_string_view_equal(spill_rows[1].origin_operation_name,
                                     IREE_SV("<unknown>")));
  EXPECT_TRUE(iree_string_view_equal(spill_rows[1].semantic_tag,
                                     IREE_SV("register.copy.b32")));
  EXPECT_EQ(spill_rows[2].kind,
            LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_MATERIALIZED);
  EXPECT_EQ(spill_rows[2].assignment_index, 2u);
  EXPECT_EQ(spill_rows[2].slot_index, 7u);
  EXPECT_TRUE(
      iree_string_view_equal(spill_rows[2].slot_space, IREE_SV("private")));
  EXPECT_EQ(spill_rows[2].byte_size, 64u);
  EXPECT_EQ(spill_rows[2].store_count, 5u);
  EXPECT_EQ(spill_rows[2].store_bytes, 320u);
  EXPECT_EQ(spill_rows[2].reload_count, 6u);
  EXPECT_EQ(spill_rows[2].reload_bytes, 384u);
  EXPECT_EQ(spill_rows[2].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY);
  EXPECT_TRUE(iree_string_view_equal(spill_rows[2].origin_operation_name,
                                     IREE_SV("<unknown>")));
  EXPECT_TRUE(iree_string_view_equal(spill_rows[2].semantic_tag,
                                     IREE_SV("memory.global.load.u32")));
  EXPECT_EQ(spill_rows[3].kind,
            LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_MATERIALIZED);
  EXPECT_EQ(spill_rows[3].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BLOCK_ARGUMENT);
  EXPECT_TRUE(iree_string_view_equal(spill_rows[3].origin_operation_name,
                                     IREE_SV("<block-argument>")));
  EXPECT_EQ(spill_rows[3].slot_index, 8u);
  EXPECT_EQ(spill_rows[3].byte_size, 4u);
  EXPECT_EQ(spill_rows[3].store_count, 2u);
  EXPECT_EQ(spill_rows[3].store_bytes, 8u);
  EXPECT_EQ(spill_rows[3].reload_count, 3u);
  EXPECT_EQ(spill_rows[3].reload_bytes, 12u);
  EXPECT_EQ(report.allocation_failure_rows.count, 1u);
  ASSERT_NE(report.allocation_failure_rows.head, nullptr);
  const auto* allocation_failure_rows =
      static_cast<const loom_target_compile_report_allocation_failure_row_t*>(
          loom_target_compile_report_vec_const_rows(
              report.allocation_failure_rows.head));
  EXPECT_TRUE(
      iree_string_view_equal(allocation_failure_rows[0].failure_code,
                             IREE_SV("unspillable-register-exhausted")));
  EXPECT_TRUE(iree_string_view_equal(allocation_failure_rows[0].register_class,
                                     IREE_SV("test.gpr")));
  EXPECT_EQ(
      allocation_failure_rows[0].blocking_kind,
      LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT);
  EXPECT_TRUE(iree_string_view_equal(allocation_failure_rows[0].value_name,
                                     IREE_SV("<unnamed>")));
  EXPECT_EQ(allocation_failure_rows[0].required_unit_count, 2u);
  EXPECT_EQ(allocation_failure_rows[0].budget_units, 1u);
  EXPECT_EQ(allocation_failure_rows[0].peak_live_units, 11u);
  EXPECT_EQ(allocation_failure_rows[0].conflict_assignment_index, 0u);
  EXPECT_TRUE(iree_string_view_equal(
      allocation_failure_rows[0].conflict_value_name, IREE_SV("<unnamed>")));
  EXPECT_EQ(allocation_failure_rows[0].conflict_location_base, 0u);
  EXPECT_EQ(report.allocation_high_water_rows.count, 1u);
  ASSERT_NE(report.allocation_high_water_rows.head, nullptr);
  const auto* allocation_high_water_rows = static_cast<
      const loom_target_compile_report_allocation_high_water_row_t*>(
      loom_target_compile_report_vec_const_rows(
          report.allocation_high_water_rows.head));
  EXPECT_TRUE(iree_string_view_equal(
      allocation_high_water_rows[0].function_name, IREE_SV("<unnamed>")));
  EXPECT_TRUE(iree_string_view_equal(
      allocation_high_water_rows[0].register_class, IREE_SV("test.gpr")));
  EXPECT_EQ(allocation_high_water_rows[0].assignment_index, 2u);
  EXPECT_EQ(allocation_high_water_rows[0].origin_kind,
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY);
  EXPECT_TRUE(iree_string_view_equal(allocation_high_water_rows[0].semantic_tag,
                                     IREE_SV("memory.global.load.u32")));
  EXPECT_EQ(allocation_high_water_rows[0].location_base, 7u);
  EXPECT_EQ(allocation_high_water_rows[0].location_count, 2u);
  EXPECT_EQ(allocation_high_water_rows[0].high_water_units, 9u);
  EXPECT_EQ(allocation_high_water_rows[0].lower_free_unit_count, 7u);
  EXPECT_EQ(allocation_high_water_rows[0].lower_free_run_count, 1u);
  EXPECT_EQ(allocation_high_water_rows[0].lower_largest_free_run_unit_count,
            7u);
  EXPECT_EQ(
      allocation_high_water_rows[0].lower_pressure_releasable_free_unit_count,
      7u);
  EXPECT_EQ(
      allocation_high_water_rows[0].lower_pressure_releasable_free_run_count,
      1u);
  EXPECT_EQ(allocation_high_water_rows[0]
                .lower_pressure_releasable_largest_free_run_unit_count,
            7u);
  EXPECT_EQ(allocation_high_water_rows[0].active_assignment_blocker_count, 0u);
  EXPECT_EQ(allocation_high_water_rows[0].active_assignment_blocker_units, 0u);
  EXPECT_EQ(allocation_high_water_rows[0].active_storage_lease_blocker_count,
            0u);
  EXPECT_EQ(allocation_high_water_rows[0].active_storage_lease_blocker_units,
            0u);
  EXPECT_EQ(
      allocation_high_water_rows[0].active_pressure_storage_lease_blocker_count,
      0u);
  EXPECT_EQ(
      allocation_high_water_rows[0].active_pressure_storage_lease_blocker_units,
      0u);
  EXPECT_EQ(
      allocation_high_water_rows[0].active_fallback_storage_lease_blocker_count,
      0u);
  EXPECT_EQ(
      allocation_high_water_rows[0].active_fallback_storage_lease_blocker_units,
      0u);

  loom_target_compile_report_t summary_report = {};
  loom_target_compile_report_initialize(&summary_report,
                                        iree_allocator_system());
  summary_report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS;
  IREE_ASSERT_OK(loom_target_compile_report_record_low_emission_frame(
      &summary_report, &frame));
  EXPECT_FALSE(
      iree_any_bit_set(summary_report.detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS));
  EXPECT_TRUE(iree_all_bits_set(
      summary_report.detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS));
  EXPECT_EQ(summary_report.schedule_band_rows.count, 0u);
  EXPECT_EQ(summary_report.schedule_band_summary_rows.count, 7u);
  loom_target_compile_report_deinitialize(&summary_report);

  loom_target_compile_report_deinitialize(&report);
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
