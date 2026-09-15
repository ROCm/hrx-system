// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#include "iree/base/api.h"
#include "iree/base/testing/dynamic_library_test_library_embed.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace iree::hal::amdgpu {
namespace {

TEST(LibHSATest, MissingRequiredSymbolsAreNotUnavailable) {
  iree::testing::TempFilePath library_path("iree_libhsa_missing_symbols",
                                           ".so");
  const iree_file_toc_t* file_toc = dynamic_library_test_library_create();
  IREE_ASSERT_OK(iree_io_file_contents_write(
      library_path.path_view(),
      iree_make_const_byte_span(file_toc->data, file_toc->size),
      iree_allocator_system()));

  iree_string_view_t search_path = library_path.path_view();
  iree_hal_amdgpu_libhsa_t libhsa = {};
  // The library exists and loads, but it is not an HSA implementation.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_hal_amdgpu_libhsa_initialize(
                            IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE,
                            iree_string_view_list_t{1, &search_path},
                            iree_allocator_system(), &libhsa));
  EXPECT_FALSE(libhsa.initialized);
  iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
}

TEST(LibHSATest, DeviceFailureStatusesPreserveCause) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_status_from_hsa_status(__FILE__, __LINE__,
                                                    HSA_STATUS_ERROR_EXCEPTION,
                                                    "exception", nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_status_from_hsa_status(__FILE__, __LINE__, HSA_STATUS_ERROR_FATAL,
                                  "fatal", nullptr));
}

// Tests that we can find, load, and unload HSA.
// In ASAN builds it tests that we don't leak the library (though ROCR itself
// leaks a bunch). If the library cannot be found then we skip the test so that
// it doesn't fail on machines without HSA installed.
TEST(LibHSATest, Load) {
  iree_hal_amdgpu_libhsa_t libhsa;
  iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
      IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
      iree_allocator_system(), &libhsa);
  if (iree_status_is_unavailable(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    GTEST_SKIP() << "HSA not available, skipping tests";
  }
  IREE_ASSERT_OK(status);

  // Ensure resolved symbols are callable without perturbing the HSA runtime
  // lifetime beyond the one owned by libhsa.
  uint16_t version_major = 0;
  IREE_ASSERT_OK(iree_hsa_system_get_info(
      IREE_LIBHSA(&libhsa), HSA_SYSTEM_INFO_VERSION_MAJOR, &version_major));
  EXPECT_NE(version_major, 0u);

  iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
}

}  // namespace
}  // namespace iree::hal::amdgpu
