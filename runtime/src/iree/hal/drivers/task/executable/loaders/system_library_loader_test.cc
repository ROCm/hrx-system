// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/executable/loaders/system_library_loader.h"

#include <vector>

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/hal/drivers/task/executable/elf/testdata/elementwise_mul_native.h"
#include "iree/hal/drivers/task/executable/executable.h"
#include "iree/hal/drivers/task/executable/loader.h"
#include "iree/task/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Version-zero producers append this native struct after the library and debug
// database, padding the debug database and footer starts to 16-byte boundaries.
struct Footer {
  // Format marker, including the trailing zero byte.
  uint8_t magic[8] = {'I', 'R', 'E', 'E', 'D', 'B', 'G', 0};
  // Native footer format version.
  uint32_t version = 0;
  // Reserved version-zero flags.
  uint32_t flags = 0;
  // Reserved version-zero library offset.
  uint64_t library_offset = 0;
  // Library byte length, excluding padding.
  uint64_t library_size = 0;
  // Debug byte offset from the start of the image.
  uint64_t debug_offset = 0;
  // Debug byte length, excluding padding.
  uint64_t debug_size = 16;
};
static_assert(sizeof(Footer) == 48);

class SystemLibraryLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(1, &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    iree_status_t status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(iree_allocator_system(),
                                                  &device_allocator_));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactor_pool_));
    IREE_ASSERT_OK(iree_hal_system_library_loader_create(
        iree_allocator_system(), &loader_));
    iree_hal_task_device_params_t device_params;
    iree_hal_task_device_params_initialize(&device_params);
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool_;
    IREE_ASSERT_OK(iree_hal_task_device_create(
        IREE_SV("system_library_test"), &device_params, 1, &executor_, 1,
        &loader_, device_allocator_, &create_params, iree_allocator_system(),
        &device_));
    iree_hal_executable_target_selection_t selection = {
        .family = IREE_SV("cpu"),
        .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
    };
    auto selected = iree_hal_device_spec_select_executable_target(
        iree_hal_device_spec(device_), &selection);
    ASSERT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
              selected.outcome);
    target_ = selected.target;
    ASSERT_NE(nullptr, target_);
    ASSERT_TRUE(
        iree_hal_executable_loader_query_target_support(loader_, target_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    iree_hal_device_release(device_);
    iree_hal_executable_loader_release(loader_);
    iree_async_proactor_pool_release(proactor_pool_);
    iree_hal_allocator_release(device_allocator_);
    iree_task_executor_release(executor_);
  }

  Footer MakeFooter() {
    Footer footer;
    footer.library_size = elementwise_mul_native_create()->size;
    footer.debug_offset = (footer.library_size + 15) & ~uint64_t{15};
    return footer;
  }

  std::vector<uint8_t> MakeImage(const Footer& footer) {
    const auto* file = elementwise_mul_native_create();
    const size_t debug_offset = (file->size + 15) & ~size_t{15};
    std::vector<uint8_t> image(debug_offset + 16 + sizeof(Footer));
    memcpy(image.data(), file->data, file->size);
    memcpy(image.data() + image.size() - sizeof(Footer), &footer,
           sizeof(footer));
    return image;
  }

  iree_status_t Load(iree_const_byte_span_t data) {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.flags = IREE_HAL_EXECUTABLE_LOAD_FLAG_ENABLE_DEBUGGING;
    params.executable_data = data;
    EXPECT_TRUE(iree_hal_executable_loader_claims_executable(loader_, target_,
                                                             &params));
    return iree_hal_executable_loader_load(
        loader_, iree_hal_device_queue_family(device_, 0), target_, &params, 1,
        &executable_);
  }

  void ExpectRuns(iree_const_byte_span_t data) {
    IREE_ASSERT_OK(Load(data));
    float lhs[] = {1, 2, 3, 4};
    float rhs[] = {100, 200, 300, 400};
    float result[4] = {};
    void* binding_ptrs[] = {lhs, rhs, result};
    size_t binding_lengths[] = {sizeof(lhs), sizeof(rhs), sizeof(result)};
    iree_hal_executable_dispatch_state_v0_t dispatch = {
        .workgroup_size_x = 1,
        .workgroup_size_y = 1,
        .workgroup_size_z = 1,
        .workgroup_count_x = 1,
        .workgroup_count_y = 1,
        .workgroup_count_z = 1,
        .max_concurrency = 1,
        .binding_count = 3,
        .binding_ptrs = binding_ptrs,
        .binding_lengths = binding_lengths,
    };
    iree_hal_executable_workgroup_state_v0_t workgroup = {};
    IREE_ASSERT_OK(iree_hal_task_executable_issue_call(
        iree_hal_task_executable_cast(executable_), 0, &dispatch, &workgroup,
        0));
    EXPECT_THAT(result, ::testing::ElementsAre(100, 400, 900, 1600));
  }

  // Executor backing the device's actual queue family.
  iree_task_executor_t* executor_ = nullptr;
  // Heap allocator retained by the device.
  iree_hal_allocator_t* device_allocator_ = nullptr;
  // Proactor pool retained by the device.
  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
  // System loader installed on the device.
  iree_hal_executable_loader_t* loader_ = nullptr;
  // Device owning the queue family and CPU target used for each load.
  iree_hal_device_t* device_ = nullptr;
  // CPU target borrowed from the device specification.
  const iree_hal_executable_target_t* target_ = nullptr;
  // Most recently loaded executable, released before its device.
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_F(SystemLibraryLoaderTest, LoadsWithoutFooter) {
  const auto* file = elementwise_mul_native_create();
  ExpectRuns(iree_make_const_byte_span(file->data, file->size));
}

TEST_F(SystemLibraryLoaderTest, LoadsVersionZeroFooterAtAnyInputAlignment) {
  for (size_t alignment = 0; alignment < alignof(Footer); ++alignment) {
    SCOPED_TRACE(alignment);
    auto image = MakeImage(MakeFooter());
    std::vector<uint8_t> shifted(image.size() + alignment);
    memcpy(shifted.data() + alignment, image.data(), image.size());
    ExpectRuns(
        iree_make_const_byte_span(shifted.data() + alignment, image.size()));
  }
}

TEST_F(SystemLibraryLoaderTest, LoadsEmptyDebugSpansAtEitherBoundary) {
  Footer footer = MakeFooter();
  footer.debug_size = 0;
  for (uint64_t offset : {uint64_t{0}, footer.debug_offset + 16}) {
    SCOPED_TRACE(offset);
    footer.debug_offset = offset;
    auto image = MakeImage(footer);
    ExpectRuns(iree_make_const_byte_span(image.data(), image.size()));
  }
}

TEST_F(SystemLibraryLoaderTest, LoadsLibraryEndingAtFooter) {
  Footer footer = MakeFooter();
  footer.library_size = footer.debug_offset + footer.debug_size;
  footer.debug_offset = footer.library_size;
  footer.debug_size = 0;
  auto image = MakeImage(footer);
  ExpectRuns(iree_make_const_byte_span(image.data(), image.size()));
}

TEST_F(SystemLibraryLoaderTest, IgnoresUnrecognizedFooterMagic) {
  Footer footer = MakeFooter();
  footer.magic[7] = 1;
  auto image = MakeImage(footer);
  ExpectRuns(iree_make_const_byte_span(image.data(), image.size()));
}

TEST_F(SystemLibraryLoaderTest, RejectsUnsupportedFooterMetadata) {
  for (int field = 0; field < 3; ++field) {
    SCOPED_TRACE(field);
    Footer footer = MakeFooter();
    if (field == 0) {
      footer.version = 1;
    }
    if (field == 1) {
      footer.flags = 1;
    }
    if (field == 2) {
      footer.library_offset = 1;
    }
    auto image = MakeImage(footer);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNIMPLEMENTED,
        Load(iree_make_const_byte_span(image.data(), image.size())));
    EXPECT_EQ(nullptr, executable_);
  }
}

TEST_F(SystemLibraryLoaderTest, RejectsOutOfBoundsFooterSpans) {
  Footer valid_footer = MakeFooter();
  const uint64_t data_length =
      valid_footer.debug_offset + valid_footer.debug_size;
  for (int field = 0; field < 3; ++field) {
    for (uint64_t value : {data_length + 1, UINT64_MAX}) {
      SCOPED_TRACE(field);
      SCOPED_TRACE(value);
      Footer footer = valid_footer;
      if (field == 0) {
        footer.library_size = value;
      }
      if (field == 1) {
        footer.debug_offset = value;
      }
      if (field == 2) {
        footer.debug_size = value;
      }
      auto image = MakeImage(footer);
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_OUT_OF_RANGE,
          Load(iree_make_const_byte_span(image.data(), image.size())));
      EXPECT_EQ(nullptr, executable_);
    }
  }
  for (uint64_t debug_size : {uint64_t{0}, uint64_t{8}}) {
    Footer footer = valid_footer;
    footer.debug_offset = UINT64_MAX - 3;
    footer.debug_size = debug_size;
    auto image = MakeImage(footer);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        Load(iree_make_const_byte_span(image.data(), image.size())));
    EXPECT_EQ(nullptr, executable_);
  }
}

TEST_F(SystemLibraryLoaderTest, TruncatedLibraryHasNoFooter) {
  const auto* file = elementwise_mul_native_create();
  for (size_t length = 8; length < sizeof(Footer); ++length) {
    SCOPED_TRACE(length);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                          Load(iree_make_const_byte_span(file->data, length)));
    EXPECT_EQ(nullptr, executable_);
  }
}

}  // namespace
