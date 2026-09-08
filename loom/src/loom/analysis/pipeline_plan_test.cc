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
#include "loom/ops/encoding/storage.h"
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
  %bias_view = buffer.view %bias[%base] : buffer -> view<[%record_rows]x[%tile_extent]x[%tile_extent]xi32>
  %output_view = buffer.view %output[%base] : buffer -> view<[%record_rows]x[%tile_extent]x[%tile_extent]xi32>
  %lhs_tiles = pipeline.scatter %lhs_view across %products : view<[%product_lanes]x[%record_rows]x[%record_columns]x[%tile_extent]x[%tile_extent]xi8>, group -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>
  %rhs_tiles = pipeline.scatter %rhs_view across %products : view<[%product_lanes]x[%record_rows]x[%record_columns]x[%tile_extent]x[%tile_extent]xi8>, group -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>
  %bias_tile = pipeline.read %bias_view on %reducers : view<[%record_rows]x[%tile_extent]x[%tile_extent]xi32>, group -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>
  %partials = pipeline.stage @product on %products(%lhs_tiles, %rhs_tiles) : (group, pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>, pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>) -> (pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>)
  %folded_partials = pipeline.fold<addi> %partials : pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>
  %buffered_partials = pipeline.buffer %folded_partials capacity %ring_capacity : (pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>, index) -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>
  %result = pipeline.reduce @reduce from %products(%buffered_partials) to %reducers(%bias_tile) : (group, pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>) to (group, pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>) -> (pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>)
  pipeline.write %result to %output_view : pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>, view<[%record_rows]x[%tile_extent]x[%tile_extent]xi32>
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
  EXPECT_EQ(plan.instances[0].fold_record_count, 2u);
  EXPECT_EQ(plan.instances[0].fold_output_port, 2u);
  EXPECT_EQ(plan.instances[0].fold_output_count, 1u);
  EXPECT_EQ(plan.instances[0].fold_kind, LOOM_COMBINING_KIND_ADDI);
  EXPECT_EQ(plan.instances[1].group_index, 0u);
  EXPECT_EQ(plan.instances[1].lane, 1u);
  EXPECT_EQ(plan.instances[1].fold_record_count, 2u);
  EXPECT_EQ(plan.instances[2].group_index, 1u);
  EXPECT_EQ(plan.instances[2].lane, 0u);
  EXPECT_EQ(plan.instances[2].fold_record_count, 0u);
  EXPECT_EQ(plan.instances[2].fold_output_count, 0u);

  EXPECT_EQ(plan.flow_count, 7u);
  EXPECT_EQ(plan.edge_count, 8u);
  EXPECT_EQ(plan.flows[0].record_count, 4u);
  EXPECT_EQ(plan.flows[1].record_count, 4u);
  EXPECT_EQ(plan.flows[2].record_count, 2u);
  EXPECT_EQ(plan.flows[3].record_count, 4u);
  EXPECT_EQ(plan.flows[4].record_count, 2u);
  EXPECT_EQ(plan.flows[5].record_count, 2u);
  EXPECT_EQ(plan.flows[5].minimum_capacity, 2u);
  EXPECT_EQ(plan.flows[6].record_count, 2u);
  EXPECT_EQ(plan.flows[0].record_shape.rank, 2u);
  EXPECT_EQ(plan.flows[0].record_shape.dimensions[0], 2u);
  EXPECT_EQ(plan.flows[0].record_shape.dimensions[1], 2u);
  EXPECT_EQ(plan.flows[4].record_shape.rank, 1u);
  EXPECT_EQ(plan.flows[4].record_shape.dimensions[0], 2u);
  for (uint32_t i = 0; i < plan.flow_count; ++i) {
    EXPECT_TRUE(loom_type_is_all_static(plan.flows[i].tile_type));
    EXPECT_EQ(loom_type_dim_static_size_at(plan.flows[i].tile_type, 0), 8);
    EXPECT_EQ(loom_type_dim_static_size_at(plan.flows[i].tile_type, 1), 8);
  }
  ASSERT_NE(plan.flows[0].binding_view_index, UINT32_MAX);
  const loom_type_t lhs_binding_type =
      plan.binding_views[plan.flows[0].binding_view_index].binding_type;
  EXPECT_TRUE(loom_type_is_all_static(lhs_binding_type));
  EXPECT_EQ(loom_type_dim_static_size_at(lhs_binding_type, 0), 2);
  EXPECT_EQ(loom_type_dim_static_size_at(lhs_binding_type, 1), 2);
  EXPECT_EQ(loom_type_dim_static_size_at(lhs_binding_type, 2), 2);

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

TEST_F(PipelinePlanTest, CombinesParallelFoldedStagesIntoOneWorkerBehavior) {
  ModulePtr module = Parse(R"(
func.def @add(%lhs: buffer, %rhs: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @parallel_folds() launch(%left: buffer, %right: buffer, %shared: buffer, %left_output: buffer, %right_output: buffer) {
  %lane_count = index.constant 1 : index
  %base = index.constant 0 : offset
  %workers = group.create %lane_count : index -> group
  %left_view = buffer.view %left[%base] : buffer -> view<2x1xf32>
  %right_view = buffer.view %right[%base] : buffer -> view<2x1xf32>
  %shared_view = buffer.view %shared[%base] : buffer -> view<2x1xf32>
  %left_output_view = buffer.view %left_output[%base] : buffer -> view<1xf32>
  %right_output_view = buffer.view %right_output[%base] : buffer -> view<1xf32>
  %left_flow = pipeline.read %left_view on %workers : view<2x1xf32>, group -> pipeline.flow<tile<1xf32>>
  %right_flow = pipeline.read %right_view on %workers : view<2x1xf32>, group -> pipeline.flow<tile<1xf32>>
  %shared_flow = pipeline.read %shared_view on %workers : view<2x1xf32>, group -> pipeline.flow<tile<1xf32>>
  %left_records = pipeline.stage @add on %workers(%left_flow, %shared_flow) : (group, pipeline.flow<tile<1xf32>>, pipeline.flow<tile<1xf32>>) -> (pipeline.flow<tile<1xf32>>)
  %right_records = pipeline.stage @add on %workers(%right_flow, %shared_flow) : (group, pipeline.flow<tile<1xf32>>, pipeline.flow<tile<1xf32>>) -> (pipeline.flow<tile<1xf32>>)
  %left_sum = pipeline.fold<addf> %left_records : pipeline.flow<tile<1xf32>>
  %right_sum = pipeline.fold<addf> %right_records : pipeline.flow<tile<1xf32>>
  pipeline.write %left_sum to %left_output_view : pipeline.flow<tile<1xf32>>, view<1xf32>
  pipeline.write %right_sum to %right_output_view : pipeline.flow<tile<1xf32>>, view<1xf32>
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_ASSERT_OK(BuildPlan(module.get(), IREE_SV("parallel_folds"), &plan));

  ASSERT_EQ(plan.group_count, 1u);
  ASSERT_EQ(plan.instance_count, 1u);
  EXPECT_FALSE(loom_symbol_ref_is_valid(plan.instances[0].entry));
  EXPECT_EQ(plan.instances[0].fold_record_count, 2u);
  EXPECT_EQ(plan.instances[0].fold_output_port, 3u);
  EXPECT_EQ(plan.instances[0].fold_output_count, 2u);
  EXPECT_EQ(plan.instances[0].fold_kind, LOOM_COMBINING_KIND_ADDF);

  ASSERT_EQ(plan.group_port_count, 5u);
  EXPECT_EQ(plan.group_ports[3].direction,
            LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_SEND);
  EXPECT_EQ(plan.group_ports[4].direction,
            LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_SEND);
}

TEST_F(PipelinePlanTest, PreservesFixedRecordStorageAcrossPartitions) {
  ModulePtr module = Parse(R"(
func.def @consume(%weight: buffer, %activation: buffer) {
  func.return
}

pipeline.def<kernel> @encoded() launch(%weight: buffer, %activation: buffer) {
  %lane_count = index.constant 2 : index
  %base = index.constant 0 : offset
  %workers = group.create %lane_count : index -> group
  %weight_view = buffer.view %weight[%base] : buffer -> view<2x3x176xi8, #encoding.storage<layout=#encoding.layout.dense, schema=#ggml.q5_k>>
  %activation_view = buffer.view %activation[%base] : buffer -> view<2x3x2x144xi8, #encoding.storage<layout=#encoding.layout.dense, schema=#ggml.q8_1_x4>>
  %weight_records = pipeline.scatter %weight_view across %workers : view<2x3x176xi8, #encoding.storage<layout=#encoding.layout.dense, schema=#ggml.q5_k>>, group -> pipeline.flow<tile<176xi8>>
  %activation_records = pipeline.scatter %activation_view across %workers : view<2x3x2x144xi8, #encoding.storage<layout=#encoding.layout.dense, schema=#ggml.q8_1_x4>>, group -> pipeline.flow<tile<2x144xi8>>
  pipeline.stage @consume on %workers(%weight_records, %activation_records) : (group, pipeline.flow<tile<176xi8>>, pipeline.flow<tile<2x144xi8>>) -> ()
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_ASSERT_OK(BuildPlan(module.get(), IREE_SV("encoded"), &plan));

  ASSERT_EQ(plan.flow_count, 2u);
  EXPECT_EQ(plan.flows[0].record_count, 3u);
  EXPECT_EQ(plan.flows[1].record_count, 3u);
  const loom_type_t weight_binding_type =
      plan.binding_views[plan.flows[0].binding_view_index].binding_type;
  const loom_type_t activation_binding_type =
      plan.binding_views[plan.flows[1].binding_view_index].binding_type;
  const loom_encoding_record_layout_t* weight_layout = nullptr;
  ASSERT_TRUE(loom_encoding_query_type_record_layout(
      nullptr, module.get(), weight_binding_type, &weight_layout));
  ASSERT_NE(weight_layout, nullptr);
  EXPECT_EQ(weight_layout->geometry.logical_element_count, 256u);
  EXPECT_EQ(weight_layout->geometry.storage_byte_count, 176u);
  EXPECT_EQ(weight_layout->geometry.required_alignment, 2u);
  const loom_encoding_record_layout_t* activation_layout = nullptr;
  ASSERT_TRUE(loom_encoding_query_type_record_layout(
      nullptr, module.get(), activation_binding_type, &activation_layout));
  ASSERT_NE(activation_layout, nullptr);
  EXPECT_EQ(activation_layout->geometry.logical_element_count, 128u);
  EXPECT_EQ(activation_layout->geometry.storage_byte_count, 144u);
  EXPECT_EQ(activation_layout->geometry.required_alignment, 16u);
  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  loom_value_fact_address_layout_t address_layout = {};
  ASSERT_TRUE(loom_encoding_query_type_address_layout(
      nullptr, module.get(), weight_binding_type, stride_storage,
      IREE_ARRAYSIZE(stride_storage), &address_layout));
  EXPECT_EQ(address_layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE);
}

TEST_F(PipelinePlanTest, RejectsPartialFixedStorageRecordTiles) {
  ModulePtr module = Parse(R"(
func.def @consume(%weight: buffer) {
  func.return
}

pipeline.def<kernel> @partial_record() launch(%weight: buffer) {
  %lane_count = index.constant 2 : index
  %base = index.constant 0 : offset
  %workers = group.create %lane_count : index -> group
  %weight_view = buffer.view %weight[%base] : buffer -> view<2x3x175xi8, #encoding.storage<layout=#encoding.layout.dense, schema=#ggml.q5_k>>
  %weight_records = pipeline.scatter %weight_view across %workers : view<2x3x175xi8, #encoding.storage<layout=#encoding.layout.dense, schema=#ggml.q5_k>>, group -> pipeline.flow<tile<175xi8>>
  pipeline.stage @consume on %workers(%weight_records) : (group, pipeline.flow<tile<175xi8>>) -> ()
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      BuildPlan(module.get(), IREE_SV("partial_record"), &plan));
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

TEST_F(PipelinePlanTest, FusesPointwiseStagesInOneSchedulingGroup) {
  ModulePtr module = Parse(R"(
func.def @first(%input: buffer, %intermediate: buffer) {
  func.return
}

func.def @second(%intermediate: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @chain() launch(%input: buffer, %output: buffer) {
  %lane_count = index.constant 1 : index
  %base = index.constant 0 : offset
  %workers = group.create %lane_count : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<16xi8>
  %output_view = buffer.view %output[%base] : buffer -> view<16xi8>
  %input_flow = pipeline.read %input_view on %workers : view<16xi8>, group -> pipeline.flow<tile<16xi8>>
  %intermediate_flow = pipeline.stage @first on %workers(%input_flow) : (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  %output_flow = pipeline.stage @second on %workers(%intermediate_flow) : (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  pipeline.write %output_flow to %output_view : pipeline.flow<tile<16xi8>>, view<16xi8>
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_ASSERT_OK(BuildPlan(module.get(), IREE_SV("chain"), &plan));

  ASSERT_EQ(plan.group_count, 1u);
  EXPECT_EQ(plan.groups[0].stage_count, 2u);
  ASSERT_EQ(plan.instance_count, 1u);
  EXPECT_FALSE(loom_symbol_ref_is_valid(plan.instances[0].entry));
  ASSERT_EQ(plan.stage_count, 2u);
  EXPECT_EQ(plan.stages[0].group_index, 0u);
  EXPECT_EQ(plan.stages[0].port_start, 0u);
  EXPECT_EQ(plan.stages[0].input_count, 1u);
  EXPECT_EQ(plan.stages[0].output_count, 1u);
  EXPECT_EQ(plan.stages[1].group_index, 0u);
  EXPECT_EQ(plan.stages[1].port_start, 2u);
  ASSERT_EQ(plan.stage_port_count, 4u);
  EXPECT_EQ(plan.stage_ports[0].flow_index, 0u);
  EXPECT_EQ(plan.stage_ports[1].flow_index, 1u);
  EXPECT_EQ(plan.stage_ports[2].flow_index, 1u);
  EXPECT_EQ(plan.stage_ports[3].flow_index, 2u);

  ASSERT_EQ(plan.group_port_count, 2u);
  EXPECT_EQ(plan.group_ports[0].direction,
            LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_RECEIVE);
  EXPECT_EQ(plan.group_ports[0].flow_index, 0u);
  EXPECT_EQ(plan.group_ports[0].port, 0u);
  EXPECT_EQ(plan.group_ports[1].direction,
            LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_SEND);
  EXPECT_EQ(plan.group_ports[1].flow_index, 2u);
  EXPECT_EQ(plan.group_ports[1].port, 1u);

  ASSERT_EQ(plan.edge_count, 2u);
  EXPECT_EQ(plan.edges[0].source_kind, LOOM_PIPELINE_ENDPOINT_KIND_BINDING);
  EXPECT_EQ(plan.edges[0].target_kind, LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  EXPECT_EQ(plan.edges[0].target_port, 0u);
  EXPECT_EQ(plan.edges[1].source_kind, LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  EXPECT_EQ(plan.edges[1].source_port, 1u);
  EXPECT_EQ(plan.edges[1].target_kind, LOOM_PIPELINE_ENDPOINT_KIND_BINDING);
}

TEST_F(PipelinePlanTest, SharesCompositeBoundaryAndInternalFanoutStorage) {
  ModulePtr module = Parse(R"(
func.def @branch(%input: buffer, %output: buffer) {
  func.return
}

func.def @join(%lhs: buffer, %rhs: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @fanout() launch(%input: buffer, %output: buffer) {
  %lane_count = index.constant 1 : index
  %base = index.constant 0 : offset
  %workers = group.create %lane_count : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<16xi8>
  %output_view = buffer.view %output[%base] : buffer -> view<16xi8>
  %input_flow = pipeline.read %input_view on %workers : view<16xi8>, group -> pipeline.flow<tile<16xi8>>
  %lhs_flow = pipeline.stage @branch on %workers(%input_flow) : (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  %rhs_flow = pipeline.stage @branch on %workers(%input_flow) : (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  %output_flow = pipeline.stage @join on %workers(%lhs_flow, %rhs_flow) : (group, pipeline.flow<tile<16xi8>>, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  pipeline.write %output_flow to %output_view : pipeline.flow<tile<16xi8>>, view<16xi8>
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_ASSERT_OK(BuildPlan(module.get(), IREE_SV("fanout"), &plan));

  ASSERT_EQ(plan.group_count, 1u);
  EXPECT_EQ(plan.groups[0].stage_count, 3u);
  ASSERT_EQ(plan.instance_count, 1u);
  EXPECT_FALSE(loom_symbol_ref_is_valid(plan.instances[0].entry));
  ASSERT_EQ(plan.stage_count, 3u);
  ASSERT_EQ(plan.stage_port_count, 7u);
  EXPECT_EQ(plan.stage_ports[0].flow_index, 0u);
  EXPECT_EQ(plan.stage_ports[2].flow_index, 0u);
  EXPECT_EQ(plan.stage_ports[4].flow_index, 1u);
  EXPECT_EQ(plan.stage_ports[5].flow_index, 2u);

  ASSERT_EQ(plan.group_port_count, 2u);
  EXPECT_EQ(plan.group_ports[0].flow_index, 0u);
  EXPECT_EQ(plan.group_ports[0].port, 0u);
  EXPECT_EQ(plan.group_ports[1].flow_index, 3u);
  EXPECT_EQ(plan.group_ports[1].port, 1u);
  ASSERT_EQ(plan.edge_count, 2u);
}

TEST_F(PipelinePlanTest, PreservesExpandedReductionLanesOnCompositePorts) {
  ModulePtr module = Parse(R"(
func.def @produce(%input: buffer, %output: buffer) {
  func.return
}

func.def @prepare(%input: buffer, %output: buffer) {
  func.return
}

func.def @reduce(%partial0: buffer, %partial1: buffer, %bias: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @composite_reduce() launch(%input: buffer, %bias: buffer, %output: buffer) {
  %producer_lanes = index.constant 2 : index
  %reducer_lanes = index.constant 1 : index
  %base = index.constant 0 : offset
  %producers = group.create %producer_lanes : index -> group
  %reducers = group.create %reducer_lanes : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<2x16xi8>
  %bias_view = buffer.view %bias[%base] : buffer -> view<16xi8>
  %output_view = buffer.view %output[%base] : buffer -> view<16xi8>
  %input_flow = pipeline.scatter %input_view across %producers : view<2x16xi8>, group -> pipeline.flow<tile<16xi8>>
  %partial_flow = pipeline.stage @produce on %producers(%input_flow) : (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  %bias_flow = pipeline.read %bias_view on %reducers : view<16xi8>, group -> pipeline.flow<tile<16xi8>>
  %prepared_bias = pipeline.stage @prepare on %reducers(%bias_flow) : (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  %result = pipeline.reduce @reduce from %producers(%partial_flow) to %reducers(%prepared_bias) : (group, pipeline.flow<tile<16xi8>>) to (group, pipeline.flow<tile<16xi8>>) -> (pipeline.flow<tile<16xi8>>)
  pipeline.write %result to %output_view : pipeline.flow<tile<16xi8>>, view<16xi8>
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_ASSERT_OK(BuildPlan(module.get(), IREE_SV("composite_reduce"), &plan));

  ASSERT_EQ(plan.group_count, 2u);
  EXPECT_EQ(plan.groups[0].stage_count, 1u);
  EXPECT_EQ(plan.groups[1].stage_count, 2u);
  ASSERT_EQ(plan.instance_count, 3u);
  EXPECT_TRUE(loom_symbol_ref_is_valid(plan.instances[0].entry));
  EXPECT_TRUE(loom_symbol_ref_is_valid(plan.instances[1].entry));
  EXPECT_FALSE(loom_symbol_ref_is_valid(plan.instances[2].entry));

  ASSERT_EQ(plan.group_port_count, 6u);
  EXPECT_EQ(plan.group_ports[1].group_index, 1u);
  EXPECT_EQ(plan.group_ports[1].flow_index, 2u);
  EXPECT_EQ(plan.group_ports[1].source_lane, UINT32_MAX);
  EXPECT_EQ(plan.group_ports[1].port, 0u);
  EXPECT_EQ(plan.group_ports[2].group_index, 1u);
  EXPECT_EQ(plan.group_ports[2].flow_index, 1u);
  EXPECT_EQ(plan.group_ports[2].source_lane, 0u);
  EXPECT_EQ(plan.group_ports[2].port, 1u);
  EXPECT_EQ(plan.group_ports[3].group_index, 1u);
  EXPECT_EQ(plan.group_ports[3].flow_index, 1u);
  EXPECT_EQ(plan.group_ports[3].source_lane, 1u);
  EXPECT_EQ(plan.group_ports[3].port, 2u);
  EXPECT_EQ(plan.group_ports[5].group_index, 1u);
  EXPECT_EQ(plan.group_ports[5].flow_index, 4u);
  EXPECT_EQ(plan.group_ports[5].port, 3u);

  ASSERT_EQ(plan.edge_count, 6u);
  EXPECT_EQ(plan.edges[3].source_index, 0u);
  EXPECT_EQ(plan.edges[3].target_index, 2u);
  EXPECT_EQ(plan.edges[3].target_port, 1u);
  EXPECT_EQ(plan.edges[4].source_index, 1u);
  EXPECT_EQ(plan.edges[4].target_index, 2u);
  EXPECT_EQ(plan.edges[4].target_port, 2u);
}

TEST_F(PipelinePlanTest, BroadcastsMultiLaneReadsFromOneBindingPort) {
  ModulePtr module = Parse(R"(
func.def @consume(%input: buffer) {
  func.return
}

pipeline.def<kernel> @broadcast() launch(%input: buffer) {
  %lane_count = index.constant 2 : index
  %base = index.constant 4294967360 : offset
  %workers = group.create %lane_count : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<3x4xi8>
  %input_records = pipeline.read %input_view on %workers : view<3x4xi8>, group -> pipeline.flow<tile<4xi8>>
  pipeline.stage @consume on %workers(%input_records) : (group, pipeline.flow<tile<4xi8>>) -> ()
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_ASSERT_OK(BuildPlan(module.get(), IREE_SV("broadcast"), &plan));

  ASSERT_EQ(plan.binding_count, 1u);
  EXPECT_EQ(plan.bindings[0].access, LOOM_PIPELINE_BINDING_ACCESS_FLAG_READ);
  ASSERT_EQ(plan.flow_count, 1u);
  EXPECT_EQ(plan.flows[0].record_count, 3u);
  EXPECT_EQ(plan.flows[0].producer_port, 0u);
  ASSERT_EQ(plan.binding_view_count, 1u);
  EXPECT_EQ(plan.flows[0].binding_view_index, 0u);
  EXPECT_EQ(plan.binding_views[0].byte_offset, UINT64_C(4294967360));
  ASSERT_EQ(plan.edge_count, 2u);
  for (uint32_t lane = 0; lane < 2; ++lane) {
    const loom_pipeline_plan_edge_t& edge = plan.edges[lane];
    EXPECT_EQ(edge.flow_index, 0u);
    EXPECT_EQ(edge.source_kind, LOOM_PIPELINE_ENDPOINT_KIND_BINDING);
    EXPECT_EQ(edge.source_index, 0u);
    EXPECT_EQ(edge.source_port, 0u);
    EXPECT_EQ(edge.binding_view_index, 0u);
    EXPECT_EQ(edge.target_kind, LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
    EXPECT_EQ(edge.target_index, lane);
    EXPECT_EQ(edge.target_port, 0u);
  }
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

TEST_F(PipelinePlanTest, PartitionsMultiLaneWritesByLeadingDimension) {
  ModulePtr module = Parse(R"(
func.def @copy(%input: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @distributed_copy() launch(%input: buffer, %output: buffer) {
  %lane_count = index.constant 2 : index
  %input_base = index.constant 32 : offset
  %output_base = index.constant 64 : offset
  %workers = group.create %lane_count : index -> group
  %input_view = buffer.view %input[%input_base] : buffer -> view<2x3x4xi8>
  %output_view = buffer.view %output[%output_base] : buffer -> view<2x3x4xi8>
  %input_records = pipeline.scatter %input_view across %workers : view<2x3x4xi8>, group -> pipeline.flow<tile<4xi8>>
  %output_records = pipeline.stage @copy on %workers(%input_records) : (group, pipeline.flow<tile<4xi8>>) -> (pipeline.flow<tile<4xi8>>)
  pipeline.write %output_records to %output_view : pipeline.flow<tile<4xi8>>, view<2x3x4xi8>
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_ASSERT_OK(BuildPlan(module.get(), IREE_SV("distributed_copy"), &plan));

  ASSERT_EQ(plan.binding_count, 2u);
  EXPECT_EQ(plan.bindings[0].access, LOOM_PIPELINE_BINDING_ACCESS_FLAG_READ);
  EXPECT_EQ(plan.bindings[1].access, LOOM_PIPELINE_BINDING_ACCESS_FLAG_WRITE);
  ASSERT_EQ(plan.flow_count, 2u);
  EXPECT_EQ(plan.flows[1].record_count, 3u);
  ASSERT_EQ(plan.binding_view_count, 2u);
  EXPECT_EQ(plan.binding_views[0].byte_offset, 32u);
  EXPECT_EQ(plan.binding_views[1].byte_offset, 64u);
  ASSERT_EQ(plan.edge_count, 4u);
  for (uint32_t lane = 0; lane < 2; ++lane) {
    const loom_pipeline_plan_edge_t& input_edge = plan.edges[lane];
    EXPECT_EQ(input_edge.binding_view_lane, lane);
    ASSERT_NE(input_edge.binding_view_index, UINT32_MAX);

    const loom_pipeline_plan_edge_t& output_edge = plan.edges[2 + lane];
    EXPECT_EQ(output_edge.source_kind, LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
    EXPECT_EQ(output_edge.source_index, lane);
    EXPECT_EQ(output_edge.target_kind, LOOM_PIPELINE_ENDPOINT_KIND_BINDING);
    EXPECT_EQ(output_edge.target_index, 1u);
    EXPECT_EQ(output_edge.binding_view_lane, lane);
    ASSERT_NE(output_edge.binding_view_index, UINT32_MAX);
    const loom_pipeline_plan_binding_view_t& output_binding_view =
        plan.binding_views[output_edge.binding_view_index];
    EXPECT_TRUE(loom_type_is_all_static(output_binding_view.binding_type));
    EXPECT_EQ(loom_type_rank(output_binding_view.binding_type), 3u);
    EXPECT_EQ(loom_type_dim_static_size_at(output_binding_view.binding_type, 0),
              2);
  }
}

TEST_F(PipelinePlanTest, RejectsMismatchedOutputLaneDimension) {
  ModulePtr module = Parse(R"(
func.def @copy(%input: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @mismatch() launch(%input: buffer, %output: buffer) {
  %lane_count = index.constant 2 : index
  %base = index.constant 0 : offset
  %workers = group.create %lane_count : index -> group
  %input_view = buffer.view %input[%base] : buffer -> view<2x3x4xi8>
  %output_view = buffer.view %output[%base] : buffer -> view<3x3x4xi8>
  %input_records = pipeline.scatter %input_view across %workers : view<2x3x4xi8>, group -> pipeline.flow<tile<4xi8>>
  %output_records = pipeline.stage @copy on %workers(%input_records) : (group, pipeline.flow<tile<4xi8>>) -> (pipeline.flow<tile<4xi8>>)
  pipeline.write %output_records to %output_view : pipeline.flow<tile<4xi8>>, view<3x3x4xi8>
  pipeline.return
}
)");

  loom_pipeline_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        BuildPlan(module.get(), IREE_SV("mismatch"), &plan));
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

TEST_F(PipelinePlanTest, RejectsDifferentlyShapedStageRecordSequences) {
  ModulePtr module = Parse(R"(
func.def @join(%lhs: buffer, %rhs: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> @mismatch() launch(%lhs: buffer, %rhs: buffer) {
  %lanes = index.constant 1 : index
  %base = index.constant 0 : offset
  %workers = group.create %lanes : index -> group
  %lhs_view = buffer.view %lhs[%base] : buffer -> view<2x2x8xi8>
  %rhs_view = buffer.view %rhs[%base] : buffer -> view<4x8xi8>
  %lhs_records = pipeline.read %lhs_view on %workers : view<2x2x8xi8>, group -> pipeline.flow<tile<8xi8>>
  %rhs_records = pipeline.read %rhs_view on %workers : view<4x8xi8>, group -> pipeline.flow<tile<8xi8>>
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
