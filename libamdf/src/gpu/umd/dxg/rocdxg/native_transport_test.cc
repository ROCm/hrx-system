// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/native_transport.h"

#include <sys/ioctl.h>

#include "gtest/gtest.h"
#include "shared/include/dxcore_loader.h"

namespace {

struct Invocation {
  // Native status returned to the thunk proxy.
  uint32_t result = 0;
  // Request received from the sidecar trampoline.
  uint64_t request = 0;
  // Typed argument storage forwarded without copying.
  void* arguments = nullptr;
};

uint32_t Invoke(void* context, uint64_t request, void* arguments) {
  auto* invocation = static_cast<Invocation*>(context);
  invocation->request = request;
  invocation->arguments = arguments;
  return invocation->result;
}

NTSTATUS QueryAdapterInfo(D3DKMT_QUERYADAPTERINFO* query) {
  auto& loader = wsl::thunk::dxcore::DxcoreLoader::Instance();
  EXPECT_TRUE(loader.Initialize());
  return loader.pfn_D3DKMTQueryAdapterInfo(query);
}

NTSTATUS CreateAllocation(D3DKMT_CREATEALLOCATION* create) {
  auto& loader = wsl::thunk::dxcore::DxcoreLoader::Instance();
  EXPECT_TRUE(loader.Initialize());
  return loader.pfn_D3DKMTCreateAllocation2(create);
}

TEST(RocdxgNativeTransportTest, RejectsCallsOutsideAnOwningBridgeScope) {
  D3DKMT_QUERYADAPTERINFO query = {};
  EXPECT_EQ(static_cast<uint32_t>(QueryAdapterInfo(&query)), 0xC0000184u);
  D3DKMT_CREATEALLOCATION create = {};
  EXPECT_EQ(static_cast<uint32_t>(CreateAllocation(&create)), 0xC0000184u);
}

TEST(RocdxgNativeTransportTest, ForwardsAndRestoresNestedTransportScopes) {
  Invocation outer_invocation = {.result = 0xC0000001u};
  Invocation inner_invocation = {.result = 0xC000000Du};
  const amdf_dxg_bridge_native_transport_t outer_transport = {
      .context = &outer_invocation,
      .invoke = Invoke,
  };
  const amdf_dxg_bridge_native_transport_t inner_transport = {
      .context = &inner_invocation,
      .invoke = Invoke,
  };
  D3DKMT_QUERYADAPTERINFO query = {};
  const uint64_t expected_request = _IOWR('G', 0x09, D3DKMT_QUERYADAPTERINFO);

  {
    amdf::rocdxg_bridge::NativeTransportScope outer_scope(outer_transport);
    EXPECT_EQ(static_cast<uint32_t>(QueryAdapterInfo(&query)),
              outer_invocation.result);
    EXPECT_EQ(outer_invocation.request, expected_request);
    EXPECT_EQ(outer_invocation.arguments, &query);
    D3DKMT_CREATEALLOCATION create = {};
    EXPECT_EQ(static_cast<uint32_t>(CreateAllocation(&create)),
              outer_invocation.result);
    EXPECT_EQ(outer_invocation.request,
              _IOWR('G', 0x06, D3DKMT_CREATEALLOCATION));
    EXPECT_EQ(outer_invocation.arguments, &create);
    {
      amdf::rocdxg_bridge::NativeTransportScope inner_scope(inner_transport);
      EXPECT_EQ(static_cast<uint32_t>(QueryAdapterInfo(&query)),
                inner_invocation.result);
      EXPECT_EQ(inner_invocation.request, expected_request);
      EXPECT_EQ(inner_invocation.arguments, &query);
    }
    outer_invocation = {.result = 0xC0000022u};
    EXPECT_EQ(static_cast<uint32_t>(QueryAdapterInfo(&query)),
              outer_invocation.result);
  }
  EXPECT_EQ(static_cast<uint32_t>(QueryAdapterInfo(&query)), 0xC0000184u);
}

}  // namespace
