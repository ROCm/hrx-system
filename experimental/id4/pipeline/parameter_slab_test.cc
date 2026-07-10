// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_slab.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(PipelineParameterSlab, PackSpanAlignsFromCurrentSlabEnd) {
  iree_device_size_t slab_byte_length = 0;

  iree_io_parameter_span_t first_span;
  IREE_ASSERT_OK(id4_pipeline_parameter_slab_pack_span(
      /*byte_length=*/5, /*alignment=*/16, &slab_byte_length, &first_span));
  EXPECT_EQ(first_span.parameter_offset, 0u);
  EXPECT_EQ(first_span.buffer_offset, 0u);
  EXPECT_EQ(first_span.length, 5u);
  EXPECT_EQ(slab_byte_length, 5u);

  iree_io_parameter_span_t second_span;
  IREE_ASSERT_OK(id4_pipeline_parameter_slab_pack_span(
      /*byte_length=*/7, /*alignment=*/16, &slab_byte_length, &second_span));
  EXPECT_EQ(second_span.parameter_offset, 0u);
  EXPECT_EQ(second_span.buffer_offset, 16u);
  EXPECT_EQ(second_span.length, 7u);
  EXPECT_EQ(slab_byte_length, 23u);
}

TEST(PipelineParameterSlab, PackSpanRejectsInvalidLengthsAndAlignments) {
  iree_device_size_t slab_byte_length = 0;
  iree_io_parameter_span_t span;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_parameter_slab_pack_span(
          /*byte_length=*/0, /*alignment=*/16, &slab_byte_length, &span));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_parameter_slab_pack_span(
          /*byte_length=*/1, /*alignment=*/3, &slab_byte_length, &span));
}

TEST(PipelineParameterSlab, MakeDeviceLocalPlanUsesSlabAlignment) {
  id4_pipeline_parameter_request_t request = id4_pipeline_parameter_request(
      IREE_SV("weight"),
      id4_pipeline_parameter_span(/*parameter_offset=*/0, /*buffer_offset=*/0,
                                  /*length=*/16));
  id4_pipeline_parameter_slab_plan_t plan =
      id4_pipeline_make_device_local_parameter_slab_plan(
          /*placement_id=*/2, /*binding_slot=*/3, IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ, /*byte_length=*/16,
          /*alignment=*/64);
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(/*count=*/1, &request);
  EXPECT_EQ(plan.placement_id, 2u);
  EXPECT_EQ(plan.binding_slot, 3u);
  EXPECT_EQ(plan.byte_length, 16u);
  EXPECT_EQ(plan.alignment, 64u);
  EXPECT_EQ(plan.target_params.type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  EXPECT_EQ(plan.target_params.access, IREE_HAL_MEMORY_ACCESS_ALL);
  EXPECT_EQ(plan.target_params.min_alignment, 64u);
  EXPECT_EQ(request_table.count, 1u);
  EXPECT_EQ(request_table.values, &request);
}

TEST(PipelineParameterSlab, ValidateLoadStepRange) {
  id4_pipeline_parameter_request_t requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV("weight.0"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0, /*length=*/16)),
      id4_pipeline_parameter_request(
          IREE_SV("weight.1"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/16, /*length=*/16)),
  };
  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/1, IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ, /*byte_length=*/32,
          /*alignment=*/16);
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(IREE_ARRAYSIZE(requests),
                                                requests);

  id4_pipeline_parameter_load_step_t step =
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("parameters.gather.tail"), IREE_SV("scope"),
          /*target_slab_index=*/0, /*request_offset=*/1,
          /*request_count=*/1);
  IREE_EXPECT_OK(id4_pipeline_parameter_load_step_validate(
      &step, /*slab_count=*/1, &slab, &request_table));

  step.request_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_parameter_load_step_validate(
                            &step, /*slab_count=*/1, &slab, &request_table));

  step.request_count = 2;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        id4_pipeline_parameter_load_step_validate(
                            &step, /*slab_count=*/1, &slab, &request_table));

  step = id4_pipeline_parameter_gather_load_step(
      iree_string_view_empty(), IREE_SV("scope"), /*target_slab_index=*/0,
      /*request_offset=*/0, /*request_count=*/1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_parameter_load_step_validate(
                            &step, /*slab_count=*/1, &slab, &request_table));
}

TEST(PipelineParameterSlab, ValidateIndexedGatherLoadStepRange) {
  id4_pipeline_parameter_request_t requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV("weight.0"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0, /*length=*/16)),
      id4_pipeline_parameter_request(
          IREE_SV("weight.1"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/16, /*length=*/16)),
  };
  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/1, IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ, /*byte_length=*/32,
          /*alignment=*/16);
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(IREE_ARRAYSIZE(requests),
                                                requests);

  const iree_host_size_t request_indices[] = {1, 0};
  id4_pipeline_parameter_load_step_t step =
      id4_pipeline_parameter_indexed_gather_load_step(
          IREE_SV("parameters.gather.indexed"), IREE_SV("scope"),
          /*target_slab_index=*/0, IREE_ARRAYSIZE(request_indices),
          request_indices);
  IREE_EXPECT_OK(id4_pipeline_parameter_load_step_validate(
      &step, /*slab_count=*/1, &slab, &request_table));

  const iree_host_size_t bad_request_indices[] = {1, 2};
  step.request_indices = bad_request_indices;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        id4_pipeline_parameter_load_step_validate(
                            &step, /*slab_count=*/1, &slab, &request_table));
}

TEST(PipelineParameterSlab, ValidatesFp8ScaledEncodeLoadStep) {
  id4_pipeline_parameter_request_t requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV("weight.0"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0, /*length=*/32)),
  };
  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/1, IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ, /*byte_length=*/32,
          /*alignment=*/16);
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(IREE_ARRAYSIZE(requests),
                                                requests);
  id4_pipeline_tensor_shape_t weight_shape = {
      /*.rank=*/2,
      /*.dims=*/{4, 4},
  };
  id4_pipeline_tensor_shape_t scale_shape = {
      /*.rank=*/1,
      /*.dims=*/{4},
  };
  const id4_pipeline_parameter_load_source_t sources[] = {
      id4_pipeline_parameter_load_source(IREE_SV("fp8"), IREE_SV("weight.0"),
                                         ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3,
                                         weight_shape,
                                         /*byte_length=*/16),
      id4_pipeline_parameter_load_source(
          IREE_SV("fp8"), IREE_SV("weight.0_scale"),
          ID4_PIPELINE_TENSOR_DTYPE_F32, scale_shape,
          /*byte_length=*/16),
  };
  id4_pipeline_parameter_load_step_t step =
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("parameters.encode_fp8"), IREE_ARRAYSIZE(sources), sources,
          /*target_slab_index=*/0, /*request_offset=*/0);
  IREE_EXPECT_OK(id4_pipeline_parameter_load_step_validate(
      &step, /*slab_count=*/1, &slab, &request_table));

  id4_pipeline_parameter_load_source_t bad_sources[] = {
      sources[0],
      sources[1],
  };
  bad_sources[1].shape.dims[0] = 5;
  step.sources = bad_sources;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_parameter_load_step_validate(
                            &step, /*slab_count=*/1, &slab, &request_table));
}

TEST(PipelineParameterSlab, GroupsDirectGathersAndContiguousEncoders) {
  id4_pipeline_tensor_shape_t weight_shape = {
      /*.rank=*/2,
      /*.dims=*/{16, 16},
  };
  id4_pipeline_tensor_shape_t scale_shape = {
      /*.rank=*/1,
      /*.dims=*/{16},
  };
  const id4_pipeline_parameter_load_source_t fp8_sources[] = {
      id4_pipeline_parameter_load_source(
          IREE_SV("model.fp8"), IREE_SV("layers.0.attn.q.weight"),
          ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3, weight_shape,
          /*byte_length=*/256),
      id4_pipeline_parameter_load_source(
          IREE_SV("model.fp8"), IREE_SV("layers.0.attn.q.weight_scale"),
          ID4_PIPELINE_TENSOR_DTYPE_F32, scale_shape,
          /*byte_length=*/64),
  };
  const id4_pipeline_parameter_load_step_t load_steps[] = {
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("parameters.gather.embeddings"), IREE_SV("model.bf16"),
          /*target_slab_index=*/0, /*request_offset=*/0,
          /*request_count=*/1),
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("parameters.encode.layer0.q"),
          /*source_count=*/IREE_ARRAYSIZE(fp8_sources), fp8_sources,
          /*target_slab_index=*/0, /*request_offset=*/1),
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("parameters.encode.layer0.k"),
          /*source_count=*/IREE_ARRAYSIZE(fp8_sources), fp8_sources,
          /*target_slab_index=*/0, /*request_offset=*/2),
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("parameters.gather.norm"), IREE_SV("model.bf16"),
          /*target_slab_index=*/0, /*request_offset=*/3,
          /*request_count=*/1),
  };

  iree_host_size_t group_count = 0;
  IREE_ASSERT_OK(id4_pipeline_parameter_load_group_count(
      IREE_ARRAYSIZE(load_steps), load_steps, &group_count));
  EXPECT_EQ(group_count, 3u);

  id4_pipeline_parameter_load_group_t group;
  IREE_ASSERT_OK(id4_pipeline_parameter_load_group_at(
      IREE_ARRAYSIZE(load_steps), load_steps, /*group_index=*/0, &group));
  EXPECT_EQ(group.step_offset, 0u);
  EXPECT_EQ(group.step_count, 1u);
  EXPECT_EQ(group.kind, ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_GATHER);
  EXPECT_EQ(group.target_slab_index, 0u);

  IREE_ASSERT_OK(id4_pipeline_parameter_load_group_at(
      IREE_ARRAYSIZE(load_steps), load_steps, /*group_index=*/1, &group));
  EXPECT_EQ(group.step_offset, 1u);
  EXPECT_EQ(group.step_count, 2u);
  EXPECT_EQ(group.kind, ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE);
  EXPECT_EQ(group.target_slab_index, 0u);

  IREE_ASSERT_OK(id4_pipeline_parameter_load_group_at(
      IREE_ARRAYSIZE(load_steps), load_steps, /*group_index=*/2, &group));
  EXPECT_EQ(group.step_offset, 3u);
  EXPECT_EQ(group.step_count, 1u);
  EXPECT_EQ(group.kind, ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_GATHER);
  EXPECT_EQ(group.target_slab_index, 0u);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        id4_pipeline_parameter_load_group_at(
                            IREE_ARRAYSIZE(load_steps), load_steps,
                            /*group_index=*/3, &group));
}

TEST(PipelineParameterSlab, SplitsEncodedLoadGroupsByReadinessKey) {
  id4_pipeline_tensor_shape_t weight_shape = {
      /*.rank=*/2,
      /*.dims=*/{4, 4},
  };
  id4_pipeline_tensor_shape_t scale_shape = {
      /*.rank=*/1,
      /*.dims=*/{4},
  };
  const id4_pipeline_parameter_load_source_t fp8_sources[] = {
      id4_pipeline_parameter_load_source(
          IREE_SV("model.fp8"), IREE_SV("weight"),
          ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3, weight_shape,
          /*byte_length=*/16),
      id4_pipeline_parameter_load_source(
          IREE_SV("model.fp8"), IREE_SV("weight_scale"),
          ID4_PIPELINE_TENSOR_DTYPE_F32, scale_shape,
          /*byte_length=*/16),
  };
  id4_pipeline_parameter_load_step_t load_steps[] = {
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("parameters.encode.layer0.q"),
          /*source_count=*/IREE_ARRAYSIZE(fp8_sources), fp8_sources,
          /*target_slab_index=*/0, /*request_offset=*/0),
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("parameters.encode.layer0.k"),
          /*source_count=*/IREE_ARRAYSIZE(fp8_sources), fp8_sources,
          /*target_slab_index=*/0, /*request_offset=*/1),
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("parameters.encode.layer1.q"),
          /*source_count=*/IREE_ARRAYSIZE(fp8_sources), fp8_sources,
          /*target_slab_index=*/0, /*request_offset=*/2),
  };
  load_steps[0].readiness_group_key = 0;
  load_steps[1].readiness_group_key = 0;
  load_steps[2].readiness_group_key = 1;

  iree_host_size_t group_count = 0;
  IREE_ASSERT_OK(id4_pipeline_parameter_load_group_count(
      IREE_ARRAYSIZE(load_steps), load_steps, &group_count));
  EXPECT_EQ(group_count, 2u);

  id4_pipeline_parameter_load_group_t group;
  IREE_ASSERT_OK(id4_pipeline_parameter_load_group_at(
      IREE_ARRAYSIZE(load_steps), load_steps, /*group_index=*/0, &group));
  EXPECT_EQ(group.step_offset, 0u);
  EXPECT_EQ(group.step_count, 2u);
  EXPECT_EQ(group.kind, ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE);

  IREE_ASSERT_OK(id4_pipeline_parameter_load_group_at(
      IREE_ARRAYSIZE(load_steps), load_steps, /*group_index=*/1, &group));
  EXPECT_EQ(group.step_offset, 2u);
  EXPECT_EQ(group.step_count, 1u);
  EXPECT_EQ(group.kind, ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE);
}

TEST(PipelineParameterSlab, EnumeratorCoversSelectedRequestRange) {
  id4_pipeline_parameter_request_t requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV("weight.0"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0, /*length=*/16)),
      id4_pipeline_parameter_request(
          IREE_SV("weight.1"),
          id4_pipeline_parameter_span(/*parameter_offset=*/8,
                                      /*buffer_offset=*/16, /*length=*/32)),
  };
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(IREE_ARRAYSIZE(requests),
                                                requests);
  id4_pipeline_parameter_slab_enumerator_state_t state = {
      // Full provider request table.
      /*.request_table=*/&request_table,
      // First selected request ordinal.
      /*.request_offset=*/1,
      // Number of selected requests.
      /*.request_count=*/1,
  };
  iree_io_parameter_enumerator_t enumerator =
      id4_pipeline_parameter_slab_enumerator(&state);

  iree_string_view_t key = iree_string_view_empty();
  iree_io_parameter_span_t span;
  IREE_EXPECT_OK(enumerator.fn(enumerator.user_data, /*i=*/0, &key, &span));
  EXPECT_TRUE(iree_string_view_equal(key, IREE_SV("weight.1")));
  EXPECT_EQ(span.parameter_offset, 8u);
  EXPECT_EQ(span.buffer_offset, 16u);
  EXPECT_EQ(span.length, 32u);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      enumerator.fn(enumerator.user_data, /*i=*/1, &key, &span));
}

TEST(PipelineParameterSlab, EnumeratorCoversExplicitRequestIndices) {
  id4_pipeline_parameter_request_t requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV("weight.0"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0, /*length=*/16)),
      id4_pipeline_parameter_request(
          IREE_SV("weight.1"),
          id4_pipeline_parameter_span(/*parameter_offset=*/8,
                                      /*buffer_offset=*/16, /*length=*/32)),
  };
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(IREE_ARRAYSIZE(requests),
                                                requests);
  const iree_host_size_t request_indices[] = {1, 0};
  id4_pipeline_parameter_slab_enumerator_state_t state = {
      // Full provider request table.
      /*.request_table=*/&request_table,
      // Contiguous offset ignored when explicit request indices are present.
      /*.request_offset=*/0,
      // Number of selected requests.
      /*.request_count=*/IREE_ARRAYSIZE(request_indices),
      // Explicit selected request ordinals.
      /*.request_indices=*/request_indices,
  };
  iree_io_parameter_enumerator_t enumerator =
      id4_pipeline_parameter_slab_enumerator(&state);

  iree_string_view_t key = iree_string_view_empty();
  iree_io_parameter_span_t span;
  IREE_EXPECT_OK(enumerator.fn(enumerator.user_data, /*i=*/0, &key, &span));
  EXPECT_TRUE(iree_string_view_equal(key, IREE_SV("weight.1")));
  EXPECT_EQ(span.parameter_offset, 8u);
  EXPECT_EQ(span.buffer_offset, 16u);
  EXPECT_EQ(span.length, 32u);

  IREE_EXPECT_OK(enumerator.fn(enumerator.user_data, /*i=*/1, &key, &span));
  EXPECT_TRUE(iree_string_view_equal(key, IREE_SV("weight.0")));
  EXPECT_EQ(span.parameter_offset, 0u);
  EXPECT_EQ(span.buffer_offset, 0u);
  EXPECT_EQ(span.length, 16u);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      enumerator.fn(enumerator.user_data, /*i=*/2, &key, &span));
}

}  // namespace
