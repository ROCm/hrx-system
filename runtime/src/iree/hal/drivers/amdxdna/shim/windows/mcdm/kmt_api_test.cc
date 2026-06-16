// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kmt_api.h"

#include <cstring>

#include "iree/testing/gtest.h"

namespace iree::hal::amdxdna::mcdm {
namespace {

int g_close_count = 0;
D3DKMT_HANDLE g_closed_adapters[4] = {};

void ResetFakes() {
  g_close_count = 0;
  std::memset(g_closed_adapters, 0, sizeof(g_closed_adapters));
}

NTSTATUS APIENTRY FakeEnumTooManyAdapters(D3DKMT_ENUMADAPTERS3* args) {
  args->NumAdapters = 257;
  return 0;
}

NTSTATUS APIENTRY FakeCreateDeviceFails(D3DKMT_CREATEDEVICE* args) {
  args->hDevice = 0;
  return static_cast<NTSTATUS>(0xC0000001u);
}

NTSTATUS APIENTRY FakeCloseAdapter(CONST D3DKMT_CLOSEADAPTER* args) {
  if (g_close_count < static_cast<int>(sizeof(g_closed_adapters) /
                                       sizeof(g_closed_adapters[0]))) {
    g_closed_adapters[g_close_count] = args->hAdapter;
  }
  ++g_close_count;
  return 0;
}

TEST(KmtApiTest, ErrorMessageDefaultsWhenEmpty) {
  Error error = {};
  EXPECT_STREQ(ErrorMessage(&error), "unknown MCDM error");
}

TEST(KmtApiTest, FindNpuAdapterRejectsExcessiveAdapterCount) {
  KmtApi api = {};
  api.enum_adapters3 = FakeEnumTooManyAdapters;
  Adapter adapter = {};
  Error error = {};

  EXPECT_FALSE(FindNpuAdapter(api, &adapter, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "too many adapters"), nullptr);
}

TEST(KmtApiTest, CreateDeviceClosesRetainedHandlesOnCreateFailure) {
  ResetFakes();
  KmtApi api = {};
  api.create_device = FakeCreateDeviceFails;
  api.close_adapter = FakeCloseAdapter;

  Adapter adapter = {};
  adapter.handle = 0x10;
  adapter.retained_handles[0] = 0x20;
  adapter.retained_handles[1] = 0x30;
  adapter.retained_handle_count = 2;
  Device device = {};
  Error error = {};

  EXPECT_FALSE(CreateDevice(api, adapter, &device, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "D3DKMTCreateDevice"), nullptr);
  EXPECT_EQ(g_close_count, 2);
  EXPECT_EQ(g_closed_adapters[0], 0x20u);
  EXPECT_EQ(g_closed_adapters[1], 0x30u);
}

}  // namespace
}  // namespace iree::hal::amdxdna::mcdm
