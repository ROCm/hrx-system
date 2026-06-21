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
          IREE_SV("scope"), /*placement_id=*/2, /*binding_slot=*/3,
          IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ, /*byte_length=*/16,
          /*alignment=*/64, /*request_count=*/1, &request);
  EXPECT_EQ(plan.placement_id, 2u);
  EXPECT_EQ(plan.binding_slot, 3u);
  EXPECT_EQ(plan.byte_length, 16u);
  EXPECT_EQ(plan.alignment, 64u);
  EXPECT_EQ(plan.target_params.type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  EXPECT_EQ(plan.target_params.access, IREE_HAL_MEMORY_ACCESS_ALL);
  EXPECT_EQ(plan.target_params.min_alignment, 64u);
  EXPECT_EQ(plan.request_count, 1u);
  EXPECT_EQ(plan.requests, &request);
}

}  // namespace
