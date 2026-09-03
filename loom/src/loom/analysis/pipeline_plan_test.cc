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
  %tile_extent = index.constant 8 : index
  %base = index.constant 0 : offset
  %products = group.create %product_lanes : index -> group
  %reducers = group.create %reducer_lanes : index -> group
  %lhs_view = buffer.view %lhs[%base] : buffer -> view<[%product_lanes]x[%tile_extent]x[%tile_extent]xi8>
  %rhs_view = buffer.view %rhs[%base] : buffer -> view<[%product_lanes]x[%tile_extent]x[%tile_extent]xi8>
  %bias_view = buffer.view %bias[%base] : buffer -> view<[%tile_extent]x[%tile_extent]xi32>
  %output_view = buffer.view %output[%base] : buffer -> view<[%tile_extent]x[%tile_extent]xi32>
  %lhs_tiles = pipeline.scatter %lhs_view across %products : view<[%product_lanes]x[%tile_extent]x[%tile_extent]xi8>, group -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>
  %rhs_tiles = pipeline.scatter %rhs_view across %products : view<[%product_lanes]x[%tile_extent]x[%tile_extent]xi8>, group -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>
  %bias_tile = pipeline.read %bias_view on %reducers : view<[%tile_extent]x[%tile_extent]xi32>, group -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>
  %partials = pipeline.stage @product on %products(%lhs_tiles, %rhs_tiles) : (group, pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>, pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi8>>) -> (pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>)
  %buffered_partials = pipeline.buffer %partials capacity %ring_capacity : (pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>, index) -> pipeline.flow<tile<[%tile_extent]x[%tile_extent]xi32>>
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
  EXPECT_EQ(plan.instances[1].group_index, 0u);
  EXPECT_EQ(plan.instances[1].lane, 1u);
  EXPECT_EQ(plan.instances[2].group_index, 1u);
  EXPECT_EQ(plan.instances[2].lane, 0u);

  EXPECT_EQ(plan.flow_count, 6u);
  EXPECT_EQ(plan.edge_count, 8u);
  EXPECT_EQ(plan.flows[4].minimum_capacity, 2u);
  for (uint32_t i = 0; i < plan.flow_count; ++i) {
    EXPECT_TRUE(loom_type_is_all_static(plan.flows[i].tile_type));
    EXPECT_EQ(loom_type_dim_static_size_at(plan.flows[i].tile_type, 0), 8);
    EXPECT_EQ(loom_type_dim_static_size_at(plan.flows[i].tile_type, 1), 8);
  }
  ASSERT_TRUE(plan.flows[0].partitioned);
  EXPECT_TRUE(loom_type_is_all_static(plan.flows[0].partition_source_type));
  EXPECT_EQ(
      loom_type_dim_static_size_at(plan.flows[0].partition_source_type, 0), 2);

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

}  // namespace
}  // namespace loom
