// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/connection_data.h"

#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

iree_net_rdma_connection_data_t MakeTestConnectionData() {
  iree_net_rdma_connection_data_t data = {};
  data.send_queue_depth = 256;
  data.recv_queue_depth = 128;
  data.max_send_sge = 4;
  data.max_recv_sge = 1;
  data.max_inline_data = 64;
  data.initial_recv_credits = 128;
  data.credit_memory.address = 0x1122334455667788ull;
  data.credit_memory.rkey = 0xAABBCCDDu;
  data.credit_memory.length = sizeof(uint32_t);
  return data;
}

void ExpectConnectionDataEq(const iree_net_rdma_connection_data_t& expected,
                            const iree_net_rdma_connection_data_t& actual) {
  EXPECT_EQ(expected.flags, actual.flags);
  EXPECT_EQ(expected.send_queue_depth, actual.send_queue_depth);
  EXPECT_EQ(expected.recv_queue_depth, actual.recv_queue_depth);
  EXPECT_EQ(expected.max_send_sge, actual.max_send_sge);
  EXPECT_EQ(expected.max_recv_sge, actual.max_recv_sge);
  EXPECT_EQ(expected.max_inline_data, actual.max_inline_data);
  EXPECT_EQ(expected.initial_recv_credits, actual.initial_recv_credits);
  EXPECT_EQ(expected.credit_memory.address, actual.credit_memory.address);
  EXPECT_EQ(expected.credit_memory.rkey, actual.credit_memory.rkey);
  EXPECT_EQ(expected.credit_memory.length, actual.credit_memory.length);
}

void SerializeTestConnectionData(uint8_t* storage,
                                 iree_host_size_t storage_length) {
  const iree_net_rdma_connection_data_t data = MakeTestConnectionData();
  iree_host_size_t data_length = 0;
  IREE_ASSERT_OK(iree_net_rdma_connection_data_serialize(
      &data, iree_make_byte_span(storage, storage_length), &data_length));
  ASSERT_EQ(IREE_NET_RDMA_CONNECTION_DATA_LENGTH, data_length);
}

TEST(RdmaConnectionDataTest, RoundTrip) {
  uint8_t storage[64] = {0};
  memset(storage, 0xCD, sizeof(storage));

  const iree_net_rdma_connection_data_t expected = MakeTestConnectionData();
  iree_host_size_t data_length = 0;
  IREE_ASSERT_OK(iree_net_rdma_connection_data_serialize(
      &expected, iree_make_byte_span(storage, sizeof(storage)), &data_length));
  ASSERT_EQ(IREE_NET_RDMA_CONNECTION_DATA_LENGTH, data_length);
  EXPECT_EQ(0xCDu, storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH]);

  iree_net_rdma_connection_data_t actual = {};
  IREE_ASSERT_OK(iree_net_rdma_connection_data_deserialize(
      iree_make_const_byte_span(storage, data_length), &actual));
  ExpectConnectionDataEq(expected, actual);
}

TEST(RdmaConnectionDataTest, SerializesLittleEndianWireLayout) {
  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH] = {0};
  SerializeTestConnectionData(storage, sizeof(storage));

  EXPECT_EQ(0x49u, storage[0]);
  EXPECT_EQ(0x52u, storage[1]);
  EXPECT_EQ(0x44u, storage[2]);
  EXPECT_EQ(0x4Du, storage[3]);
  EXPECT_EQ(0x01u, storage[4]);
  EXPECT_EQ(0x00u, storage[5]);
  EXPECT_EQ(0x00u, storage[6]);
  EXPECT_EQ(0x00u, storage[7]);

  EXPECT_EQ(0x88u, storage[32]);
  EXPECT_EQ(0x77u, storage[33]);
  EXPECT_EQ(0x66u, storage[34]);
  EXPECT_EQ(0x55u, storage[35]);
  EXPECT_EQ(0x44u, storage[36]);
  EXPECT_EQ(0x33u, storage[37]);
  EXPECT_EQ(0x22u, storage[38]);
  EXPECT_EQ(0x11u, storage[39]);
  EXPECT_EQ(0xDDu, storage[40]);
  EXPECT_EQ(0xCCu, storage[41]);
  EXPECT_EQ(0xBBu, storage[42]);
  EXPECT_EQ(0xAAu, storage[43]);
  for (iree_host_size_t i = 48; i < IREE_NET_RDMA_CONNECTION_DATA_LENGTH; ++i) {
    EXPECT_EQ(0x00u, storage[i]);
  }
}

TEST(RdmaConnectionDataTest, RejectsShortBuffer) {
  const iree_net_rdma_connection_data_t data = MakeTestConnectionData();
  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH - 1] = {0};

  iree_host_size_t data_length = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_rdma_connection_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
  EXPECT_EQ(0u, data_length);

  iree_net_rdma_connection_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_net_rdma_connection_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaConnectionDataTest, RejectsInvalidArguments) {
  const iree_net_rdma_connection_data_t data = MakeTestConnectionData();
  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 1;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_connection_data_serialize(
          nullptr, iree_make_byte_span(storage, sizeof(storage)),
          &data_length));
  EXPECT_EQ(0u, data_length);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_connection_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), nullptr));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_connection_data_serialize(
          &data, iree_make_byte_span(nullptr, sizeof(storage)), &data_length));

  iree_net_rdma_connection_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_connection_data_deserialize(
          iree_make_const_byte_span(nullptr, sizeof(storage)), &actual));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_connection_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), nullptr));
}

TEST(RdmaConnectionDataTest, RejectsBadMagic) {
  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH] = {0};
  SerializeTestConnectionData(storage, sizeof(storage));

  storage[0] = 0;
  iree_net_rdma_connection_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_net_rdma_connection_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaConnectionDataTest, RejectsUnsupportedVersion) {
  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH] = {0};
  SerializeTestConnectionData(storage, sizeof(storage));

  storage[4] = 2;
  iree_net_rdma_connection_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_net_rdma_connection_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaConnectionDataTest, RejectsReservedBytes) {
  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH] = {0};
  SerializeTestConnectionData(storage, sizeof(storage));

  storage[48] = 1;
  iree_net_rdma_connection_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_net_rdma_connection_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaConnectionDataTest, AllowsOpaqueZeroRkey) {
  iree_net_rdma_connection_data_t data = MakeTestConnectionData();
  data.credit_memory.rkey = 0;

  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 0;
  IREE_ASSERT_OK(iree_net_rdma_connection_data_serialize(
      &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));

  iree_net_rdma_connection_data_t actual = {};
  IREE_ASSERT_OK(iree_net_rdma_connection_data_deserialize(
      iree_make_const_byte_span(storage, data_length), &actual));
  ExpectConnectionDataEq(data, actual);
}

TEST(RdmaConnectionDataTest, RejectsInvalidCreditMemory) {
  iree_net_rdma_connection_data_t data = MakeTestConnectionData();
  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 0;

  data.credit_memory.address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_connection_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));

  data = MakeTestConnectionData();
  data.credit_memory.length = sizeof(uint32_t) - 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_connection_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
}

TEST(RdmaConnectionDataTest, RejectsCreditOverflow) {
  iree_net_rdma_connection_data_t data = MakeTestConnectionData();
  data.initial_recv_credits = data.recv_queue_depth + 1;

  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_connection_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
}

TEST(RdmaConnectionDataTest, RejectsTooManySendSge) {
  iree_net_rdma_connection_data_t data = MakeTestConnectionData();
  data.max_send_sge = IREE_NET_RDMA_CONNECTION_DATA_MAX_SEND_SGE + 1;

  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_connection_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
}

}  // namespace
