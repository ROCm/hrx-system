// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_library_util.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::local {
namespace {

TEST(ExecutableLibraryUtilTest, ReportsFixedWorkgroupLocalMemory) {
  const iree_hal_executable_dispatch_attrs_v0_t attributes = {
      /*.flags=*/{},
      /*.local_memory_pages=*/3,
      /*.binding_count=*/{},
      /*.reserved_0=*/{},
      /*.workgroup_size_x=*/4,
      /*.workgroup_size_y=*/2,
      /*.workgroup_size_z=*/1,
      /*.parameter_count=*/{},
  };
  const iree_hal_executable_library_v0_t library = {
      /*.header=*/{},
      /*.imports=*/{},
      /*.exports=*/
      {
          /*.count=*/1,
          /*.ptrs=*/{},
          /*.attrs=*/&attributes,
      },
  };

  iree_hal_executable_function_info_t output;
  memset(&output, 0xA5, sizeof(output));
  IREE_ASSERT_OK(iree_hal_executable_library_export_info(
      &library, iree_hal_executable_function_from_index(0), &output));

  EXPECT_EQ(IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_WORKGROUP_LOCAL_MEMORY,
            output.resource_usage.provided_flags);
  EXPECT_EQ(3u * IREE_HAL_EXECUTABLE_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE,
            output.resource_usage.fixed_workgroup_local_memory_size);
  EXPECT_EQ(0u, output.resource_usage.fixed_private_memory_size);
  EXPECT_EQ(0u, output.resource_usage.invocation_register_count);
  EXPECT_EQ(0u, output.maximum_workgroup_invocations);
}

TEST(ExecutableLibraryUtilTest, InitializesUnspecifiedParameterFields) {
  const iree_hal_executable_dispatch_parameter_v0_t parameter = {
      /*.type=*/IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
      /*.size=*/sizeof(void*),
      /*.flags=*/IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
      /*.name=*/UINT16_MAX,
      /*.offset=*/0,
  };
  const iree_hal_executable_dispatch_parameter_v0_t* parameters[] = {
      &parameter,
  };
  const iree_hal_executable_dispatch_attrs_v0_t attributes = {
      /*.flags=*/{},
      /*.local_memory_pages=*/{},
      /*.binding_count=*/{},
      /*.reserved_0=*/{},
      /*.workgroup_size_x=*/{},
      /*.workgroup_size_y=*/{},
      /*.workgroup_size_z=*/{},
      /*.parameter_count=*/1,
  };
  const iree_hal_executable_library_v0_t library = {
      /*.header=*/{},
      /*.imports=*/{},
      /*.exports=*/
      {
          /*.count=*/1,
          /*.ptrs=*/{},
          /*.attrs=*/&attributes,
          /*.params=*/parameters,
      },
  };

  iree_hal_executable_function_parameter_t output;
  memset(&output, 0xA5, sizeof(output));
  IREE_ASSERT_OK(iree_hal_executable_library_export_parameters(
      &library, iree_hal_executable_function_from_index(0), /*capacity=*/1,
      &output));

  EXPECT_EQ(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING, output.type);
  EXPECT_EQ(sizeof(void*), output.size);
  EXPECT_EQ(0u, output.flags);
  EXPECT_EQ(0u, output.offset);
  EXPECT_EQ(0u, output.native_abi_offset);
  EXPECT_TRUE(iree_string_view_is_empty(output.name));
}

}  // namespace
}  // namespace iree::hal::local
