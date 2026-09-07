// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline_plan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/testing/context.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class PipelinePlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr Parse(const char* source) {
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("pipeline_plan_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  loom_func_like_t FindPipeline(loom_module_t* module,
                                iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    EXPECT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    EXPECT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    loom_func_like_t pipeline = loom_func_like_cast(
        module, module->symbols.entries[symbol_id].defining_op);
    EXPECT_TRUE(loom_pipeline_def_isa(pipeline.op));
    return pipeline;
  }

  iree_status_t BuildPlan(loom_module_t* module, iree_string_view_t name,
                          loom_pipeline_plan_t* out_plan) {
    const loom_func_like_t pipeline = FindPipeline(module, name);
    loom_value_fact_table_t facts = {};
    IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(
        &facts, &analysis_arena_, module->values.count));
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_compute(&facts, module, pipeline));
    return loom_pipeline_plan_build(module, pipeline, &facts,
                                    (loom_pipeline_plan_limits_t){
                                        /*.instance_count=*/16,
                                    },
                                    &analysis_arena_, out_plan);
  }

  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t analysis_arena_ = {};
  loom_context_t context_ = {};
};

TEST_F(PipelinePlanTest, ExpandsSplitKDataflowOnce) {
  ModulePtr module = Parse(R"(
func.def @product(%lhs: buffer, %rhs: buffer, %partial: buffer) {
  func.return
}

func.def @reduce(%partial0: buffer, %partial1: buffer, %bias: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @split_k() launch(%lhs: buffer, %rhs: buffer, %bias: buffer, %output: buffer) {
  %product_lanes = index.constant 2 : index
  %reducer_lanes = index.constant 1 : index
  %ring_capacity = index.constant 2 : index
  %record_rows = index.constant 2 : index
  %record_columns = index.constant 2 : index
  %tile_extent = index.constant 8 : index
  %base = index.constant 0 : offset
  %products = group.create %product_lanes : index -> group
  %reducers = group.create %reducer_lanes : index -> group
  %lhs_view = buffer.view %lhs[%base] : buffer -> view<[%product_lanes]x[%record_rows]x[%record_columns]x[%tile_extent]x[%tile_extent]xi8>
  %rhs_view = buffer.view %rhs[%base] : buffer -> view<[%product_lanes]x[%record_rows]x[%record_columns]x[%tile_extent]x[%tile_extent]xi8>
  %bias_view = buffer.view %bias[%base] : buffer -> view<[%tile_extent]x[%tile_extent]xi32>
  %output_view = buffer.view %output[%base] : buffer -> view<[%tile_extent]x[%tile_extent]xi32>
  %lhs_tiles = pipeline.scatter %lhs_view across %products : view<[%product_lanes]x[%record_rows]x[%record_columns]x[%tile_extent]x[%tile_extent]xi8>, group -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>
  %rhs_tiles = pipeline.scatter %rhs_view across %products : view<[%product_lanes]x[%record_rows]x[%record_columns]x[%tile_extent]x[%tile_extent]xi8>, group -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>
  %bias_tile = pipeline.read %bias_view on %reducers : view<[%tile_extent]x[%tile_extent]xi32>, group -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>
  %partials = pipeline.stage @product on %products(%lhs_tiles, %rhs_tiles) : (group, pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>, pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>) -> (pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>)
  %folded_partials = pipeline.fold<addi> %partials : pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>
  %buffered_partials = pipeline.buffer %folded_partials capacity %ring_capacity : (pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>, index) -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>
  %result = pipeline.reduce @reduce from %products(%buffered_partials) to %reducers(%bias_tile) : (group, pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>) to (group, pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>) -> (pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>)
  pipeline.write %result to %output_view : pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>, view<[%tile_extent]x[%tile_extent]xi32>
  pipeline.return
}
)");

  const loom_func_like_t pipeline =
      FindPipeline(module.get(), IREE_SV("split_k"));
  loom_value_fact_table_t facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &analysis_arena_,
                                                  module->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_compute(&facts, module.get(), pipeline));

  loom_pipeline_plan_t plan = {};
  IREE_ASSERT_OK(loom_pipeline_plan_build(module.get(), pipeline, &facts,
                                          (loom_pipeline_plan_limits_t){
                                              /*.instance_count=*/16,
                                          },
                                          &analysis_arena_, &plan));

  ASSERT_EQ(plan.binding_count, 4u);
  EXPECT_EQ(plan.bindings[0].access, LOOM_PIPELINE_BINDING_ACCESS_FLAG_READ);
  EXPECT_EQ(plan.bindings[1].access, LOOM_PIPELINE_BINDING_ACCESS_FLAG_READ);
  EXPECT_EQ(plan.bindings[2].access, LOOM_PIPELINE_BINDING_ACCESS_FLAG_READ);
  EXPECT_EQ(plan.bindings[3].access, LOOM_PIPELINE_BINDING_ACCESS_FLAG_WRITE);

  ASSERT_EQ(plan.group_count, 2u);
  EXPECT_EQ(plan.groups[0].lane_count, 2u);
  EXPECT_EQ(plan.groups[0].instance_start, 0u);
  EXPECT_EQ(plan.groups[1].lane_count, 1u);
  EXPECT_EQ(plan.groups[1].instance_start, 2u);

  ASSERT_EQ(plan.instance_count, 3u);
  EXPECT_EQ(plan.instances[0].group_index, 0u);
  EXPECT_EQ(plan.instances[0].lane, 0u);
  EXPECT_EQ(plan.instances[0].fold_record_count, 4u);
  EXPECT_EQ(plan.instances[0].fold_output_port, 2u);
  EXPECT_EQ(plan.instances[0].fold_kind, LOOM_COMBINING_KIND_ADDI);
  EXPECT_EQ(plan.instances[1].group_index, 0u);
  EXPECT_EQ(plan.instances[1].lane, 1u);
  EXPECT_EQ(plan.instances[1].fold_record_count, 4u);
  EXPECT_EQ(plan.instances[2].group_index, 1u);
  EXPECT_EQ(plan.instances[2].lane, 0u);
  EXPECT_EQ(plan.instances[2].fold_record_count, 0u);

  EXPECT_EQ(plan.flow_count, 7u);
  EXPECT_EQ(plan.edge_count, 8u);
  EXPECT_EQ(plan.flows[0].record_count, 4u);
  EXPECT_EQ(plan.flows[1].record_count, 4u);
  EXPECT_EQ(plan.flows[2].record_count, 1u);
  EXPECT_EQ(plan.flows[3].record_count, 4u);
  EXPECT_EQ(plan.flows[4].record_count, 1u);
  EXPECT_EQ(plan.flows[5].record_count, 1u);
  EXPECT_EQ(plan.flows[5].minimum_capacity, 2u);
  EXPECT_EQ(plan.flows[6].record_count, 1u);
  for (uint32_t i = 0; i < plan.flow_count; ++i) {
    EXPECT_TRUE(loom_type_is_all_static(plan.flows[i].tile_type));
    EXPECT_EQ(loom_type_dim_static_size_at(plan.flows[i].tile_type, 0), 8);
    EXPECT_EQ(loom_type_dim_static_size_at(plan.flows[i].tile_type, 1), 8);
  }
  ASSERT_TRUE(plan.flows[0].partitioned);
  EXPECT_TRUE(loom_type_is_all_static(plan.flows[0].partition_source_type));
  EXPECT_EQ(
      loom_type_dim_static_size_at(plan.flows[0].partition_source_type, 0), 2);
  EXPECT_EQ(
      loom_type_dim_static_size_at(plan.flows[0].partition_source_type, 1), 2);
  EXPECT_EQ(
      loom_type_dim_static_size_at(plan.flows[0].partition_source_type, 2), 2);

  const loom_pipeline_plan_edge_t& product0_to_reducer = plan.edges[4];
  EXPECT_EQ(product0_to_reducer.source_kind,
            LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  EXPECT_EQ(product0_to_reducer.source_index, 0u);
  EXPECT_EQ(product0_to_reducer.source_port, 2u);
  EXPECT_EQ(product0_to_reducer.target_kind,
            LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  EXPECT_EQ(product0_to_reducer.target_index, 2u);
  EXPECT_EQ(product0_to_reducer.target_port, 0u);

  const loom_pipeline_plan_edge_t& product1_to_reducer = plan.edges[5];
  EXPECT_EQ(product1_to_reducer.source_kind,
            LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  EXPECT_EQ(product1_to_reducer.source_index, 1u);
  EXPECT_EQ(product1_to_reducer.source_port, 2u);
  EXPECT_EQ(product1_to_reducer.target_kind,
            LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  EXPECT_EQ(product1_to_reducer.target_index, 2u);
  EXPECT_EQ(product1_to_reducer.target_port, 1u);

  loom_pipeline_plan_t undersized_plan = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_pipeline_plan_build(module.get(), pipeline, &facts,
                               (loom_pipeline_plan_limits_t){
                                   /*.instance_count=*/2,
                               },
                               &analysis_arena_, &undersized_plan));
}

TEST_F(PipelinePlanTest, ConnectsEqualCardinalityStageGroupsPointwise) {
  ModulePtr module = Parse(R"(
func.def @first(%input: buffer, %intermediate: buffer) {
  func.return
}

func.def @second(%intermediate: buffer, %shared_input: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @chain() launch(%input: buffer, %output: buffer) {
  %lane_count = index.constant 1 : index
  %base = index.constant 0 : offset
  %first_group = group.create %lane_count : index -> group
  %second_group = group.create %lane_count : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<16xi8>
  %output_view = buffer.view %output[%base] : buffer -> view<16xi8>
  %input_flow = pipeline.read %input_view on %first_group : view<16xi8>, group -> pipeline.flow<tile<16xi8>>
  %intermediate_flow = pipeline.stage @first on %first_group(%input_flow) : (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  %output_flow = pipeline.stage @second on %second_group(%intermediate_flow, %input_flow) : (group, pipeline.flow<tile<16xi8>>, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  pipeline.write %output_flow to %output_view : pipeline.flow<tile<16xi8>>, view<16xi8>
  pipeline.return
}
)");

  const loom_func_like_t pipeline =
      FindPipeline(module.get(), IREE_SV("chain"));
  loom_value_fact_table_t facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &analysis_arena_,
                                                  module->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_compute(&facts, module.get(), pipeline));

  loom_pipeline_plan_t plan = {};
  IREE_ASSERT_OK(loom_pipeline_plan_build(module.get(), pipeline, &facts,
                                          (loom_pipeline_plan_limits_t){
                                              /*.instance_count=*/16,
                                          },
                                          &analysis_arena_, &plan));

  ASSERT_EQ(plan.instance_count, 2u);
  ASSERT_EQ(plan.edge_count, 4u);
  const loom_pipeline_plan_edge_t& stage_edge = plan.edges[1];
  EXPECT_EQ(stage_edge.source_kind, LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  EXPECT_EQ(stage_edge.source_index, 0u);
  EXPECT_EQ(stage_edge.source_port, 1u);
  EXPECT_EQ(stage_edge.target_kind, LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  EXPECT_EQ(stage_edge.target_index, 1u);
  EXPECT_EQ(stage_edge.target_port, 0u);

  const loom_pipeline_plan_edge_t& first_binding_edge = plan.edges[0];
  const loom_pipeline_plan_edge_t& second_binding_edge = plan.edges[2];
  EXPECT_EQ(first_binding_edge.flow_index, second_binding_edge.flow_index);
  EXPECT_EQ(first_binding_edge.source_kind,
            LOOM_PIPELINE_ENDPOINT_KIND_BINDING);
  EXPECT_EQ(second_binding_edge.source_kind,
            LOOM_PIPELINE_ENDPOINT_KIND_BINDING);
  EXPECT_EQ(first_binding_edge.source_index, second_binding_edge.source_index);
  EXPECT_EQ(first_binding_edge.source_port, second_binding_edge.source_port);
  EXPECT_EQ(first_binding_edge.target_index, 0u);
  EXPECT_EQ(second_binding_edge.target_index, 1u);
}

TEST_F(PipelinePlanTest, ExpandsFlowFanoutWithoutCloningProducer) {
  ModulePtr module = Parse(R"(
func.def @producer(%input: buffer, %intermediate: buffer) {
  func.return
}

func.def @consumer(%intermediate: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @fanout() launch(%input: buffer, %output0: buffer, %output1: buffer) {
  %lane_count = index.constant 1 : index
  %base = index.constant 0 : offset
  %producers = group.create %lane_count : index -> group
  %consumers0 = group.create %lane_count : index -> group
  %consumers1 = group.create %lane_count : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<16xi8>
  %output_view0 = buffer.view %output0[%base] : buffer -> view<16xi8>
  %output_view1 = buffer.view %output1[%base] : buffer -> view<16xi8>
  %input_flow = pipeline.read %input_view on %producers : view<16xi8>, group -> pipeline.flow<tile<16xi8>>
  %shared_flow = pipeline.stage @producer on %producers(%input_flow) : (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  %output_flow0 = pipeline.stage @consumer on %consumers0(%shared_flow) : (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  %output_flow1 = pipeline.stage @consumer on %consumers1(%shared_flow) : (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  pipeline.write %output_flow0 to %output_view0 : pipeline.flow<tile<16xi8>>, view<16xi8>
  pipeline.write %output_flow1 to %output_view1 : pipeline.flow<tile<16xi8>>, view<16xi8>
  pipeline.return
}
)");

  const loom_func_like_t pipeline =
      FindPipeline(module.get(), IREE_SV("fanout"));
  loom_value_fact_table_t facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &analysis_arena_,
                                                  module->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_compute(&facts, module.get(), pipeline));

  loom_pipeline_plan_t plan = {};
  IREE_ASSERT_OK(loom_pipeline_plan_build(module.get(), pipeline, &facts,
                                          (loom_pipeline_plan_limits_t){
                                              /*.instance_count=*/16,
                                          },
                                          &analysis_arena_, &plan));

  ASSERT_EQ(plan.instance_count, 3u);
  ASSERT_EQ(plan.edge_count, 5u);
  EXPECT_EQ(plan.edges[1].flow_index, plan.edges[2].flow_index);
  EXPECT_EQ(plan.edges[1].source_kind, LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  EXPECT_EQ(plan.edges[1].source_index, 0u);
  EXPECT_EQ(plan.edges[2].source_index, 0u);
  EXPECT_EQ(plan.edges[1].source_port, 1u);
  EXPECT_EQ(plan.edges[2].source_port, 1u);
  EXPECT_EQ(plan.edges[1].target_index, 1u);
  EXPECT_EQ(plan.edges[2].target_index, 2u);
}

TEST_F(PipelinePlanTest, RejectsUnresolvedCardinalityAtConcreteBoundary) {
  ModulePtr module = Parse(R"(
func.def @stage() {
  func.return
}

pipeline.def<kernel> @dynamic(%lanes: index) launch() {
  %workers = group.create %lanes : index -> group
  pipeline.stage @stage on %workers() : (group) -> ()
  pipeline.return
}
)");

  const loom_func_like_t pipeline =
      FindPipeline(module.get(), IREE_SV("dynamic"));
  loom_value_fact_table_t facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &analysis_arena_,
                                                  module->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_compute(&facts, module.get(), pipeline));

  loom_pipeline_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pipeline_plan_build(module.get(), pipeline, &facts,
                                                 (loom_pipeline_plan_limits_t){
                                                     /*.instance_count=*/16,
                                                 },
                                                 &analysis_arena_, &plan));
}

TEST_F(PipelinePlanTest, RejectsUnresolvedRecordShapeAtConcreteBoundary) {
  ModulePtr module = Parse(R"(
func.def @copy(%input: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @dynamic(%extent: index) launch(%input: buffer, %output: buffer) {
  %lanes = index.constant 1 : index
  %base = index.constant 0 : offset
  %workers = group.create %lanes : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<[%extent]xi8>
  %output_view = buffer.view %output[%base] : buffer -> view<[%extent]xi8>
  %input_tile = pipeline.read %input_view on %workers : view<[%extent]xi8>, group -> pipeline.flow<tile<[%extent]xi8>>
  %output_tile = pipeline.stage @copy on %workers(%input_tile) : (group, pipeline.flow<tile<[%extent]xi8>>) -> (pipeline.flow<tile<[%extent]xi8>>)
  pipeline.write %output_tile to %output_view : pipeline.flow<tile<[%extent]xi8>>, view<[%extent]xi8>
  pipeline.return
}
)");

  const loom_func_like_t pipeline =
      FindPipeline(module.get(), IREE_SV("dynamic"));
  loom_value_fact_table_t facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &analysis_arena_,
                                                  module->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_compute(&facts, module.get(), pipeline));

  loom_pipeline_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pipeline_plan_build(module.get(), pipeline, &facts,
                                                 (loom_pipeline_plan_limits_t){
                                                     /*.instance_count=*/16,
                                                 },
                                                 &analysis_arena_, &plan));
}

TEST_F(PipelinePlanTest, RejectsMismatchedStageRecordCounts) {
  ModulePtr module = Parse(R"(
func.def @join(%lhs: buffer, %rhs: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @mismatch() launch(%lhs: buffer, %rhs: buffer) {
  %lanes = index.constant 1 : index
  %base = index.constant 0 : offset
  %workers = group.create %lanes : index -> group
  %lhs_view = buffer.view %lhs[%base] : buffer -> view<4x8xi8>
  %rhs_view = buffer.view %rhs[%base] : buffer -> view<2x8xi8>
  %lhs_records = pipeline.read %lhs_view on %workers : view<4x8xi8>, group -> pipeline.flow<tile<8xi8>>
  %rhs_records = pipeline.read %rhs_view on %workers : view<2x8xi8>, group -> pipeline.flow<tile<8xi8>>
  %output = pipeline.stage @join on %workers(%lhs_records, %rhs_records) : (group, pipeline.flow<tile<8xi8>>, pipeline.flow<tile<8xi8>>) -> (pipeline.flow<tile<8xi8>>)
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        BuildPlan(module.get(), IREE_SV("mismatch"), &plan));
}

TEST_F(PipelinePlanTest, RejectsMismatchedOutputRecordCount) {
  ModulePtr module = Parse(R"(
func.def @copy(%input: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @mismatch() launch(%input: buffer, %output: buffer) {
  %lanes = index.constant 1 : index
  %base = index.constant 0 : offset
  %workers = group.create %lanes : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<4x8xi8>
  %output_view = buffer.view %output[%base] : buffer -> view<2x8xi8>
  %input_records = pipeline.read %input_view on %workers : view<4x8xi8>, group -> pipeline.flow<tile<8xi8>>
  %output_records = pipeline.stage @copy on %workers(%input_records) : (group, pipeline.flow<tile<8xi8>>) -> (pipeline.flow<tile<8xi8>>)
  pipeline.write %output_records to %output_view : pipeline.flow<tile<8xi8>>, view<2x8xi8>
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        BuildPlan(module.get(), IREE_SV("mismatch"), &plan));
}

TEST_F(PipelinePlanTest, RejectsEmptyRecordSequence) {
  ModulePtr module = Parse(R"(
func.def @copy(%input: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @empty() launch(%input: buffer) {
  %lanes = index.constant 1 : index
  %base = index.constant 0 : offset
  %workers = group.create %lanes : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<0x8xi8>
  %input_records = pipeline.read %input_view on %workers : view<0x8xi8>, group -> pipeline.flow<tile<8xi8>>
  %output_records = pipeline.stage @copy on %workers(%input_records) : (group, pipeline.flow<tile<8xi8>>) -> (pipeline.flow<tile<8xi8>>)
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        BuildPlan(module.get(), IREE_SV("empty"), &plan));
}

TEST_F(PipelinePlanTest, RejectsRecordCountOverflow) {
  ModulePtr module = Parse(R"(
pipeline.def<kernel> @overflow() launch(%input: buffer) {
  %lanes = index.constant 1 : index
  %base = index.constant 0 : offset
  %workers = group.create %lanes : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<65536x65536x8xi8>
  %input_records = pipeline.read %input_view on %workers : view<65536x65536x8xi8>, group -> pipeline.flow<tile<8xi8>>
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        BuildPlan(module.get(), IREE_SV("overflow"), &plan));
}

TEST_F(PipelinePlanTest, RejectsMismatchedReductionRecordCounts) {
  ModulePtr module = Parse(R"(
func.def @produce(%input: buffer, %output: buffer) {
  func.return
}

func.def @reduce(%source: buffer, %bias: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @mismatch() launch(%input: buffer, %bias: buffer) {
  %lanes = index.constant 1 : index
  %base = index.constant 0 : offset
  %producers = group.create %lanes : index -> group
  %reducers = group.create %lanes : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<4x8xi8>
  %bias_view = buffer.view %bias[%base] : buffer -> view<2x8xi8>
  %input_records = pipeline.read %input_view on %producers : view<4x8xi8>, group -> pipeline.flow<tile<8xi8>>
  %source_records = pipeline.stage @produce on %producers(%input_records) : (group, pipeline.flow<tile<8xi8>>) -> (pipeline.flow<tile<8xi8>>)
  %bias_records = pipeline.read %bias_view on %reducers : view<2x8xi8>, group -> pipeline.flow<tile<8xi8>>
  %output_records = pipeline.reduce @reduce from %producers(%source_records) to %reducers(%bias_records) : (group, pipeline.flow<tile<8xi8>>) to (group, pipeline.flow<tile<8xi8>>) -> (pipeline.flow<tile<8xi8>>)
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        BuildPlan(module.get(), IREE_SV("mismatch"), &plan));
}

TEST_F(PipelinePlanTest, RejectsRecordwiseUseBeforeFold) {
  ModulePtr module = Parse(R"(
func.def @copy(%input: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @fanout() launch(%input: buffer) {
  %lanes = index.constant 1 : index
  %base = index.constant 0 : offset
  %producers = group.create %lanes : index -> group
  %consumers = group.create %lanes : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<4x8xi8>
  %input_records = pipeline.read %input_view on %producers : view<4x8xi8>, group -> pipeline.flow<tile<8xi8>>
  %produced = pipeline.stage @copy on %producers(%input_records) : (group, pipeline.flow<tile<8xi8>>) -> (pipeline.flow<tile<8xi8>>)
  %consumed = pipeline.stage @copy on %consumers(%produced) : (group, pipeline.flow<tile<8xi8>>) -> (pipeline.flow<tile<8xi8>>)
  %folded = pipeline.fold<addi> %produced : pipeline.flow<tile<8xi8>>
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        BuildPlan(module.get(), IREE_SV("fanout"), &plan));
}

}  // namespace
}  // namespace loom
