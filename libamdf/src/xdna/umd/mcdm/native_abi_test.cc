// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/native_abi.h"

#include <cstring>

#include "gtest/gtest.h"

namespace {

struct QueryState {
  // KMD build returned through the standard adapter query.
  uint64_t version = UINT64_C(0x0020000000CB00F0);
  // Private response returned by the retained miniport ABI.
  uint32_t private_info[2] = {0, 3};
  // Whether the miniport actually writes a private-query response.
  bool writes_private_info = true;
  // Native failure injected at the query boundary.
  NTSTATUS status = 0;
  // Number of private queries issued after the standard query.
  uint32_t private_query_count = 0;
};

QueryState* query_state = nullptr;

NTSTATUS APIENTRY Query(const D3DKMT_QUERYADAPTERINFO* query) {
  if (query_state->status < 0) return query_state->status;
  if (query->Type == KMTQAITYPE_KMD_DRIVER_VERSION) {
    EXPECT_EQ(query->PrivateDriverDataSize, sizeof(D3DKMT_KMD_DRIVER_VERSION));
    auto* version =
        static_cast<D3DKMT_KMD_DRIVER_VERSION*>(query->pPrivateDriverData);
    version->DriverVersion.QuadPart = query_state->version;
  } else {
    EXPECT_EQ(query->Type, KMTQAITYPE_UMDRIVERPRIVATE);
    ++query_state->private_query_count;
    if (query_state->writes_private_info) {
      std::memcpy(query->pPrivateDriverData, query_state->private_info,
                  sizeof(query_state->private_info));
    }
  }
  return 0;
}

class WindowsXdnaNativeAbiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    query_state = &state_;
    kmt_.query_adapter_info = Query;
  }
  void TearDown() override { query_state = nullptr; }
  // Per-test native query responses and observations.
  QueryState state_;
  // Existing native API dependency, with no production test platform.
  amdf_kmt_api_t kmt_ = {};
};

TEST_F(WindowsXdnaNativeAbiTest, AdmitsExactBuildWithoutUnwrittenPrivateQuery) {
  state_.writes_private_info = false;
  const amdf_windows_xdna_native_abi_t* abi = nullptr;
  ASSERT_EQ(amdf_windows_xdna_native_abi_query(&kmt_, 1, &abi), AMDF_STATUS_OK);
  EXPECT_EQ(abi->context_encoding, AMDF_WINDOWS_XDNA_CONTEXT_ENCODING_METADATA);
  EXPECT_EQ(abi->context_cookie_byte_offset, 0x30u);
  EXPECT_EQ(abi->submission_header_byte_length, 88u);
  EXPECT_EQ(state_.private_query_count, 0u);
}

TEST_F(WindowsXdnaNativeAbiTest, PreservesExplicitPrivateAbi) {
  state_.version = 0;
  const amdf_windows_xdna_native_abi_t* abi = nullptr;
  ASSERT_EQ(amdf_windows_xdna_native_abi_query(&kmt_, 1, &abi), AMDF_STATUS_OK);
  EXPECT_EQ(abi->context_encoding, AMDF_WINDOWS_XDNA_CONTEXT_ENCODING_XCLBIN);
  EXPECT_EQ(abi->context_cookie_byte_offset, 0x40u);
  EXPECT_EQ(abi->submission_header_byte_length, 104u);
  EXPECT_EQ(state_.private_query_count, 1u);
}

TEST_F(WindowsXdnaNativeAbiTest, DoesNotInferVersionRangeOrUnwrittenTag) {
  ++state_.version;
  state_.writes_private_info = false;
  auto* const sentinel =
      reinterpret_cast<const amdf_windows_xdna_native_abi_t*>(uintptr_t{1});
  const amdf_windows_xdna_native_abi_t* abi = sentinel;
  EXPECT_EQ(
      amdf_status_code(amdf_windows_xdna_native_abi_query(&kmt_, 1, &abi)),
      AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(abi, sentinel);
  EXPECT_EQ(state_.private_query_count, 1u);
}

TEST_F(WindowsXdnaNativeAbiTest, PropagatesNativeQueryFailure) {
  state_.status = static_cast<NTSTATUS>(0xC0000001u);
  const amdf_windows_xdna_native_abi_t* abi = nullptr;
  EXPECT_EQ(amdf_windows_xdna_native_abi_query(&kmt_, 1, &abi),
            amdf_kmt_make_status(state_.status));
  EXPECT_EQ(abi, nullptr);
  EXPECT_EQ(state_.private_query_count, 0u);
}

}  // namespace
