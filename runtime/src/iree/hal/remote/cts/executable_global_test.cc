// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests remote forwarding of executable global buffer aliases.

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {
namespace {

class RemoteExecutableGlobalTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), executable_cache_.out()));

    iree_const_byte_span_t executable_data =
        this->executable_data(IREE_SV("executable_global_test.bin"));
    ASSERT_GT(executable_data.data_length, 0u);

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(executable_format());
    executable_params.executable_data = executable_data;

    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, executable_.out()));
  }

  Ref<iree_hal_executable_cache_t> executable_cache_;
  Ref<iree_hal_executable_t> executable_;
};

TEST_P(RemoteExecutableGlobalTest, LookupGlobalByName) {
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_lookup_global_by_name(
      executable_, IREE_SV("cts_lookup_global"), &global));

  iree_hal_executable_global_info_t info;
  IREE_ASSERT_OK(iree_hal_executable_global_info(executable_, global, &info));
  EXPECT_EQ(info.byte_length, 4);

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, global, IREE_HAL_QUEUE_AFFINITY_ANY, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 4);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(RemoteExecutableGlobalTest);

}  // namespace
}  // namespace iree::hal::cts
