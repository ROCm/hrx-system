// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk_session.h"

#include <cstdint>

#include "iree/net/channel/bulk/transfer_table.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ClientBulkSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_remote_client_bulk_session_initialize(
        iree_allocator_system(), &session_));
  }

  void TearDown() override {
    iree_host_size_t ignored_transfer_count = 0;
    iree_hal_remote_client_bulk_session_deinitialize_transfers(
        &session_, ClearTransfers, &ignored_transfer_count);
    iree_hal_remote_client_bulk_session_deinitialize(&session_);
  }

  static void ClearTransfers(void* user_data,
                             iree_net_bulk_transfer_table_t* transfers) {
    iree_host_size_t* transfer_count =
        static_cast<iree_host_size_t*>(user_data);
    *transfer_count += iree_net_bulk_transfer_table_count(transfers);
    iree_net_bulk_transfer_table_clear(transfers);
  }

  iree_status_t InitializeTransfers() {
    iree_hal_remote_client_bulk_session_transfer_options_t options =
        iree_hal_remote_client_bulk_session_transfer_options_default();
    options.transfer_user_storage_size = sizeof(uint64_t);
    options.transfer_user_storage_alignment = iree_alignof(uint64_t);
    options.chunk_user_storage_size = sizeof(uint64_t);
    options.chunk_user_storage_alignment = iree_alignof(uint64_t);
    return iree_hal_remote_client_bulk_session_initialize_transfers(
        &session_, &options, iree_allocator_system());
  }

  iree_hal_remote_client_bulk_session_t session_ = {};
};

TEST_F(ClientBulkSessionTest, ChannelPublicationExchangesPointers) {
  uintptr_t channel_a_storage = 0;
  uintptr_t channel_b_storage = 0;
  auto* channel_a =
      reinterpret_cast<iree_net_bulk_channel_t*>(&channel_a_storage);
  auto* channel_b =
      reinterpret_cast<iree_net_bulk_channel_t*>(&channel_b_storage);

  EXPECT_EQ(iree_hal_remote_client_bulk_session_load_channel(&session_),
            nullptr);
  EXPECT_EQ(iree_hal_remote_client_bulk_session_exchange_channel(&session_,
                                                                 channel_a),
            nullptr);
  EXPECT_EQ(iree_hal_remote_client_bulk_session_load_channel(&session_),
            channel_a);
  EXPECT_EQ(iree_hal_remote_client_bulk_session_exchange_channel(&session_,
                                                                 channel_b),
            channel_a);
  EXPECT_EQ(
      iree_hal_remote_client_bulk_session_exchange_channel(&session_, nullptr),
      channel_b);
  EXPECT_EQ(iree_hal_remote_client_bulk_session_load_channel(&session_),
            nullptr);
}

TEST_F(ClientBulkSessionTest, TransferStateOwnsTablesAndPools) {
  IREE_ASSERT_OK(InitializeTransfers());

  EXPECT_NE(session_.transfers, nullptr);
  EXPECT_NE(session_.send_chunks, nullptr);
  EXPECT_NE(session_.receive_chunks, nullptr);

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_allocate_transfer(
      session_.transfers, /*total_size=*/4096, /*user_value=*/0, &transfer));

  iree_host_size_t transfer_count = 0;
  iree_hal_remote_client_bulk_session_deinitialize_transfers(
      &session_, ClearTransfers, &transfer_count);

  EXPECT_EQ(transfer_count, 1u);
  EXPECT_EQ(session_.transfers, nullptr);
  EXPECT_EQ(session_.send_chunks, nullptr);
  EXPECT_EQ(session_.receive_chunks, nullptr);
}

TEST_F(ClientBulkSessionTest, TransferStateDeinitializesWhenEmpty) {
  IREE_ASSERT_OK(InitializeTransfers());

  iree_host_size_t transfer_count = 0;
  iree_hal_remote_client_bulk_session_deinitialize_transfers(
      &session_, ClearTransfers, &transfer_count);

  EXPECT_EQ(transfer_count, 0u);
  EXPECT_EQ(session_.transfers, nullptr);
  EXPECT_EQ(session_.send_chunks, nullptr);
  EXPECT_EQ(session_.receive_chunks, nullptr);
}

}  // namespace
