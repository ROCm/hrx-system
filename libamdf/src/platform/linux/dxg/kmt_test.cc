// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/dxg/kmt.h"

#include "gtest/gtest.h"

namespace {

struct Invocation {
  // Explicit connection forwarded as the transport context.
  const amdf_linux_dxg_connection_t* connection = nullptr;
  // Encoded Linux ioctl request observed by the transport.
  uint64_t request = 0;
  // Typed operation argument storage forwarded without copying.
  void* arguments = nullptr;
  // Native result returned through the typed operation wrapper.
  uint32_t result = 0;
};

Invocation invocation;

TEST(DxgKmtTest, ForwardsCompletePublishedOperationFamily) {
  const amdf_linux_dxg_connection_t connection = {.descriptor = 17};
  invocation.result = 0xC0000001u;

#define AMDF_LINUX_DXG_KMT_OPERATION(name, type, operation)                 \
  {                                                                         \
    type arguments = {};                                                    \
    invocation = {.result = 0xC0000001u};                                   \
    EXPECT_EQ((uint32_t)amdf_linux_dxg_kmt_##name(&connection, &arguments), \
              invocation.result);                                           \
    EXPECT_EQ(invocation.connection, &connection);                          \
    EXPECT_EQ(invocation.request, (uint64_t)_IOWR('G', operation, type));   \
    EXPECT_EQ(invocation.arguments, &arguments);                            \
  }
#include "libamdf/src/platform/linux/dxg/kmt_operations.inc"
#undef AMDF_LINUX_DXG_KMT_OPERATION
}

}  // namespace

extern "C" uint32_t __wrap_amdf_linux_dxg_connection_invoke(void* context,
                                                            uint64_t request,
                                                            void* arguments) {
  invocation.connection =
      static_cast<const amdf_linux_dxg_connection_t*>(context);
  invocation.request = request;
  invocation.arguments = arguments;
  return invocation.result;
}
