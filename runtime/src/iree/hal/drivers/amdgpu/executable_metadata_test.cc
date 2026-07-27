// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable_metadata.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

TEST(ExecutableMetadataTest, AllocatesHotAndColdTables) {
  iree_hal_amdgpu_executable_metadata_counts_t counts = {
      /*.export_count=*/2,
      /*.parameter_count=*/3,
      /*.layout_blob_byte_length=*/128,
  };

  iree_hal_amdgpu_executable_metadata_t* metadata = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_allocate(
      &counts, iree_allocator_system(), &metadata));

  EXPECT_EQ(metadata->export_count, 2);
  EXPECT_NE(metadata->exports, nullptr);
  EXPECT_NE(metadata->reflection, nullptr);
  EXPECT_EQ(metadata->parameter_count, 3);
  EXPECT_NE(metadata->parameters, nullptr);
  EXPECT_EQ(metadata->layout_blob_capacity, 128);
  EXPECT_EQ(metadata->layout_blob_used, 0);
  EXPECT_NE(metadata->layout_blob, nullptr);
  EXPECT_FALSE(iree_hal_amdgpu_kernarg_layout_ref_is_valid(
      metadata->exports[0].kernarg_layout));
  EXPECT_FALSE(iree_hal_amdgpu_kernarg_layout_ref_is_valid(
      metadata->exports[1].kernarg_layout));

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(ExecutableMetadataTest, AppendsAndResolvesLayout) {
  iree_host_size_t layout_byte_length = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_kernarg_layout_storage_size(
      /*binding_count=*/1, /*constant_span_count=*/1, &layout_byte_length));
  iree_hal_amdgpu_executable_metadata_counts_t counts = {
      /*.export_count=*/1,
      /*.parameter_count=*/{},
      /*.layout_blob_byte_length=*/layout_byte_length,
  };
  iree_hal_amdgpu_executable_metadata_t* metadata = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_allocate(
      &counts, iree_allocator_system(), &metadata));

  iree_hal_amdgpu_kernarg_layout_ref_t layout_ref =
      iree_hal_amdgpu_kernarg_layout_ref_invalid();
  iree_byte_span_t layout_storage = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_append_layout(
      metadata, layout_byte_length, &layout_ref, &layout_storage));
  EXPECT_TRUE(iree_hal_amdgpu_kernarg_layout_ref_is_valid(layout_ref));
  EXPECT_EQ(layout_storage.data_length, layout_byte_length);

  const iree_hal_amdgpu_kernarg_binding_slot_t binding_slots[] = {
      {/*.target_qword_index=*/0},
  };
  const iree_hal_amdgpu_kernarg_constant_span_t constant_spans[] = {
      {
          /*.target_byte_offset=*/8,
          /*.source_byte_offset=*/0,
          /*.byte_length=*/4,
      },
  };
  iree_hal_amdgpu_kernarg_layout_params_t params = {
      /*.kernarg_byte_length=*/16,
      /*.kernarg_alignment=*/8,
      /*.constant_byte_length=*/4,
      /*.implicit_args_byte_offset=*/
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE,
      /*.binding_count=*/IREE_ARRAYSIZE(binding_slots),
      /*.binding_slots=*/binding_slots,
      /*.constant_span_count=*/IREE_ARRAYSIZE(constant_spans),
      /*.constant_spans=*/constant_spans,
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_kernarg_layout_initialize(
      &params, layout_storage.data_length,
      reinterpret_cast<iree_hal_amdgpu_kernarg_layout_t*>(
          layout_storage.data)));

  metadata->exports[0].kernarg_layout = layout_ref;
  const iree_hal_amdgpu_kernarg_layout_t* resolved_layout = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_resolve_layout(
      metadata, metadata->exports[0].kernarg_layout, &resolved_layout));
  ASSERT_NE(resolved_layout, nullptr);
  EXPECT_EQ(resolved_layout->binding_count, 1);
  EXPECT_EQ(resolved_layout->constant_byte_length, 4);

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(ExecutableMetadataTest, RejectsLayoutBlobOverflow) {
  iree_hal_amdgpu_executable_metadata_counts_t counts = {
      /*.export_count=*/1,
      /*.parameter_count=*/{},
      /*.layout_blob_byte_length=*/8,
  };
  iree_hal_amdgpu_executable_metadata_t* metadata = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_allocate(
      &counts, iree_allocator_system(), &metadata));

  iree_hal_amdgpu_kernarg_layout_ref_t layout_ref =
      iree_hal_amdgpu_kernarg_layout_ref_invalid();
  iree_byte_span_t layout_storage = iree_byte_span_empty();
  EXPECT_THAT(
      Status(iree_hal_amdgpu_executable_metadata_append_layout(
          metadata, /*layout_byte_length=*/16, &layout_ref, &layout_storage)),
      StatusIs(StatusCode::kResourceExhausted));
  EXPECT_FALSE(iree_hal_amdgpu_kernarg_layout_ref_is_valid(layout_ref));
  EXPECT_EQ(layout_storage.data, nullptr);

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(ExecutableMetadataTest, RejectsInvalidLayoutReference) {
  iree_hal_amdgpu_executable_metadata_counts_t counts = {
      /*.export_count=*/1,
      /*.parameter_count=*/{},
      /*.layout_blob_byte_length=*/16,
  };
  iree_hal_amdgpu_executable_metadata_t* metadata = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_allocate(
      &counts, iree_allocator_system(), &metadata));

  const iree_hal_amdgpu_kernarg_layout_t* resolved_layout = nullptr;
  EXPECT_THAT(Status(iree_hal_amdgpu_executable_metadata_resolve_layout(
                  metadata, iree_hal_amdgpu_kernarg_layout_ref_invalid(),
                  &resolved_layout)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(resolved_layout, nullptr);

  iree_hal_amdgpu_kernarg_layout_ref_t layout_ref =
      iree_hal_amdgpu_kernarg_layout_ref_invalid();
  iree_byte_span_t layout_storage = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_append_layout(
      metadata, /*layout_byte_length=*/16, &layout_ref, &layout_storage));
  EXPECT_THAT(
      Status(iree_hal_amdgpu_executable_metadata_resolve_layout(
          metadata, iree_hal_amdgpu_kernarg_layout_ref_t{/*.byte_offset=*/1},
          &resolved_layout)),
      StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(resolved_layout, nullptr);

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

}  // namespace
}  // namespace iree::hal::amdgpu
