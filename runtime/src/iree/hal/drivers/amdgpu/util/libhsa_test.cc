// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#include "iree/base/api.h"
#include "iree/base/testing/dynamic_library_test_library_embed.h"
#include "iree/hal/drivers/amdgpu/util/libhsa_test_library_embed.h"
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

#if !IREE_HAL_AMDGPU_LIBHSA_STATIC
TEST(LibHSATest, LoadsWithoutOptionalQueueCreateEntryPoint) {
#if defined(IREE_PLATFORM_WINDOWS)
  static constexpr const char* extension = ".dll";
#else
  static constexpr const char* extension = ".so";
#endif  // IREE_PLATFORM_WINDOWS
  iree::testing::TempFilePath library_path("iree_libhsa_test", extension);
  const iree_file_toc_t* file_toc = libhsa_test_library_create();
  IREE_ASSERT_OK(iree_io_file_contents_write(
      library_path.path_view(),
      iree_make_const_byte_span(file_toc->data, file_toc->size),
      iree_allocator_system()));

  const iree_string_view_t search_path = library_path.path_view();
  const iree_string_view_list_t search_paths = {1, &search_path};
  iree_hal_amdgpu_libhsa_t libhsa;
  IREE_ASSERT_OK(iree_hal_amdgpu_libhsa_initialize(
      IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, search_paths, iree_allocator_system(),
      &libhsa));
  EXPECT_FALSE(iree_hal_amdgpu_libhsa_has_hsa_amd_queue_create(&libhsa));
  iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
}
#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

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
