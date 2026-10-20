// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/dispatch.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {
class DispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    function_.metadata = &metadata_;
    function_.kernarg = &layout_;
  }
  iree_status_t Prepare(
      iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags = 0,
      iree_const_byte_span_t constants = iree_const_byte_span_empty(),
      iree_host_size_t binding_count = 0) {
    return iree_hal_amd_gpu_dispatch_prepare(&function_, 65536, true, config,
                                             constants, binding_count, flags,
                                             &dispatch_);
  }
  // Image-owned resource and geometry constraints.
  iree_hal_amd_gpu_metadata_kernel_t metadata_ = {};
  // Prepared native argument representation.
  iree_hal_amd_gpu_kernarg_layout_t layout_ = {.alignment = 8};
  // Placed reflection borrowing this fixture's immutable facts.
  iree_hal_amd_gpu_function_t function_ = {};
  // Output of public admission.
  iree_hal_amd_gpu_dispatch_geometry_t dispatch_ = {};
};

TEST_F(DispatchTest, DefaultGeometryUsesRequiredMetadata) {
  auto config = iree_hal_make_static_dispatch_config(7, 3, 2);
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.workgroup_size[0], 1u);
  metadata_.has_required_workgroup_size = true;
  metadata_.required_workgroup_size[0] = 32;
  metadata_.required_workgroup_size[1] = 2;
  metadata_.required_workgroup_size[2] = 1;
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.workgroup_size[0], 32u);
  EXPECT_EQ(dispatch_.workgroup_size[1], 2u);
  EXPECT_EQ(dispatch_.workgroup_count[0], 7u);
  config.workgroup_size[0] = 16;
  config.workgroup_size[1] = 2;
  config.workgroup_size[2] = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config));
}

TEST_F(DispatchTest, ExactItemsAdmitPartialGroupsAndUniformRequirements) {
  iree_hal_dispatch_config_t config = {};
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 4;
  config.workgroup_size[2] = 1;
  config.workitem_count[0] = 65;
  config.workitem_count[1] = 8;
  config.workitem_count[2] = 1;
  const auto flags = IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT;
  IREE_ASSERT_OK(Prepare(config, flags));
  EXPECT_EQ(dispatch_.workgroup_count[0], 3u);
  EXPECT_EQ(dispatch_.workgroup_count[1], 2u);
  EXPECT_EQ(dispatch_.partial_size[0], 1u);
  EXPECT_EQ(dispatch_.partial_size[1], 4u);
  EXPECT_EQ(dispatch_.partial_size[2], 1u);
  metadata_.uniform_workgroup_size = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config, flags));
  config.workitem_count[0] = 64;
  IREE_ASSERT_OK(Prepare(config, flags));
  EXPECT_EQ(dispatch_.partial_size[0], 0u);
  EXPECT_EQ(dispatch_.partial_size[1], 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config));
}

TEST_F(DispatchTest, GeometryAndLdsRespectExecutableLimits) {
  auto config = iree_hal_make_static_dispatch_config(5, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  metadata_.max_flat_workgroup_size = 16;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Prepare(config));
  metadata_.max_flat_workgroup_size = 32;
  metadata_.max_workgroup_count[0] = 4;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Prepare(config));
  metadata_.max_workgroup_count[0] = 5;
  metadata_.group_segment_fixed_size = 32768;
  config.dynamic_workgroup_local_memory = 32768;
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.group_segment_byte_length, 65536u);
  ++config.dynamic_workgroup_local_memory;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Prepare(config));
  config.dynamic_workgroup_local_memory = UINT32_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Prepare(config));
}

TEST_F(DispatchTest, DirectArgumentsRespectDenseAndNativeExtents) {
  const auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  std::array<uint8_t, 24> constants = {};
  layout_.byte_length = 24;
  layout_.native_byte_length = 12;
  layout_.constant_byte_length = 8;
  IREE_ASSERT_OK(Prepare(config, 0, {constants.data(), 8}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Prepare(config, 0, {constants.data(), 12}));
  const auto native = IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
  IREE_ASSERT_OK(Prepare(config, native, {constants.data(), 12}));
  IREE_ASSERT_OK(Prepare(config, native, {constants.data(), 24}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Prepare(config, native, {constants.data(), 11}));
}

TEST_F(DispatchTest, DirectBindingRangesAcceptRemainingExtents) {
  std::array<uint32_t, 4> storage = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_USAGE_STORAGE, sizeof(storage),
      iree_make_byte_span(storage.data(), sizeof(storage)), {},
      iree_allocator_system(), &buffer));
  layout_.byte_length = 8;
  for (iree_hal_dispatch_flags_t flags : {
           IREE_HAL_DISPATCH_FLAG_NONE,
           IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_ARGUMENTS,
           IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS,
       }) {
    const auto validate = [&](iree_hal_buffer_ref_t reference) {
      return iree_hal_amd_gpu_dispatch_validate_direct_bindings(
          &layout_, {}, {1, &reference}, flags);
    };
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, validate({}));
    for (iree_device_size_t offset : {0u, 8u}) {
      IREE_EXPECT_OK(validate(iree_hal_make_buffer_ref(buffer, offset, 8)));
      IREE_EXPECT_OK(validate(
          iree_hal_make_buffer_ref(buffer, offset, IREE_HAL_WHOLE_BUFFER)));
    }
    for (auto reference : {
             iree_hal_make_buffer_ref(buffer, 16, 1),
             iree_hal_make_buffer_ref(buffer, 17, 0),
             iree_hal_make_buffer_ref(buffer, 17, IREE_HAL_WHOLE_BUFFER),
             iree_hal_make_buffer_ref(buffer, UINT64_MAX, 0),
             iree_hal_make_buffer_ref(buffer, 8, UINT64_MAX - 1),
         }) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, validate(reference));
    }
    for (auto reference : {
             iree_hal_make_buffer_ref(buffer, 12, 4),
             iree_hal_make_buffer_ref(buffer, 12, IREE_HAL_WHOLE_BUFFER),
             iree_hal_make_buffer_ref(buffer, 16, 0),
             iree_hal_make_buffer_ref(buffer, 16, IREE_HAL_WHOLE_BUFFER),
         }) {
      IREE_EXPECT_STATUS_IS(flags ? IREE_STATUS_OUT_OF_RANGE : IREE_STATUS_OK,
                            validate(reference));
    }
  }
  iree_hal_buffer_release(buffer);
}

TEST_F(DispatchTest, IndirectParameterReferencesCoverThreeAlignedCounts) {
  std::array<uint32_t, 4> storage = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS, sizeof(storage),
      iree_make_byte_span(storage.data(), sizeof(storage)), {},
      iree_allocator_system(), &buffer));
  for (auto flags : {IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                     IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
    const auto validate = [&](iree_hal_buffer_ref_t reference) {
      return iree_hal_amd_gpu_dispatch_validate_direct_bindings(
          &layout_, reference, {}, flags);
    };
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, validate({}));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          validate(iree_hal_make_buffer_ref(buffer, 1, 12)));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          validate(iree_hal_make_buffer_ref(buffer, 4, 8)));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        validate(iree_hal_make_buffer_ref(buffer, 8, IREE_HAL_WHOLE_BUFFER)));
    IREE_ASSERT_OK(validate(iree_hal_make_buffer_ref(buffer, 4, 12)));
    IREE_ASSERT_OK(
        validate(iree_hal_make_buffer_ref(buffer, 4, IREE_HAL_WHOLE_BUFFER)));
  }
  iree_hal_buffer_release(buffer);
}

TEST_F(DispatchTest, PartialGeometryPacksNativeHiddenArguments) {
  const iree_hal_amd_gpu_metadata_argument_t arguments[] = {
      {
          .offset = 0,
          .size = 4,
          .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X,
      },
      {
          .offset = 4,
          .size = 2,
          .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GROUP_SIZE_X,
      },
      {
          .offset = 6,
          .size = 2,
          .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_REMAINDER_X,
      },
      {
          .offset = 8,
          .size = 2,
          .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GRID_DIMS,
      },
      {
          .offset = 12,
          .size = 4,
          .kind =
              IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_DYNAMIC_LDS_SIZE,
      },
  };
  metadata_.has_argument_metadata = true;
  metadata_.arguments = arguments;
  metadata_.argument_count = IREE_ARRAYSIZE(arguments);
  metadata_.kernarg_segment_size = 16;
  metadata_.kernarg_segment_alignment = 8;
  metadata_.group_segment_fixed_size = 128;
  iree_hal_amd_gpu_kernarg_layout_t* layout = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_kernarg_layout_allocate(
      &metadata_, iree_allocator_system(), &layout));
  function_.kernarg = layout;
  iree_hal_dispatch_config_t config = {
      .workgroup_size = {32, 1, 1},
      .dynamic_workgroup_local_memory = 384,
      .workitem_count = {65, 1, 1},
  };
  const auto exact = IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT;
  IREE_ASSERT_OK(Prepare(config, exact));
  std::array<uint32_t, 4> bytes;
  bytes.fill(UINT32_MAX);
  iree_hal_amd_gpu_dispatch_emplace_arguments(&function_, &dispatch_, {},
                                              nullptr, exact, bytes.data());
  EXPECT_EQ(bytes, (std::array<uint32_t, 4>{2, 0x00010020, 1, 384}));
  // Caller-owned native arguments never receive host writes.
  bytes.fill(0xcafebabe);
  iree_hal_amd_gpu_dispatch_emplace_arguments(
      &function_, &dispatch_, {}, nullptr,
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS, bytes.data());
  EXPECT_EQ(bytes, (std::array<uint32_t, 4>{0xcafebabe, 0xcafebabe, 0xcafebabe,
                                            0xcafebabe}));
  iree_hal_amd_gpu_kernarg_layout_free(layout);
}

}  // namespace
