// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/endpoint_data.h"

#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

iree_net_rdma_endpoint_data_t MakeTestEndpointData() {
  iree_net_rdma_endpoint_data_t data = {};
  data.group_id = 0x0102030405060708ull;
  data.endpoint_index = 2;
  data.endpoint_count = 4;
  data.bootstrap_recv_buffer_size = 4096;
  data.bootstrap_recv_credits = 128;
  return data;
}

void ExpectEndpointDataEq(const iree_net_rdma_endpoint_data_t& expected,
                          const iree_net_rdma_endpoint_data_t& actual) {
  EXPECT_EQ(expected.flags, actual.flags);
  EXPECT_EQ(expected.group_id, actual.group_id);
  EXPECT_EQ(expected.endpoint_index, actual.endpoint_index);
  EXPECT_EQ(expected.endpoint_count, actual.endpoint_count);
  EXPECT_EQ(expected.bootstrap_recv_buffer_size,
            actual.bootstrap_recv_buffer_size);
  EXPECT_EQ(expected.bootstrap_recv_credits, actual.bootstrap_recv_credits);
}

void SerializeTestEndpointData(uint8_t* storage,
                               iree_host_size_t storage_length) {
  const iree_net_rdma_endpoint_data_t data = MakeTestEndpointData();
  iree_host_size_t data_length = 0;
  IREE_ASSERT_OK(iree_net_rdma_endpoint_data_serialize(
      &data, iree_make_byte_span(storage, storage_length), &data_length));
  ASSERT_EQ(IREE_NET_RDMA_ENDPOINT_DATA_LENGTH, data_length);
}

TEST(RdmaEndpointDataTest, RoundTrip) {
  uint8_t storage[128] = {0};
  memset(storage, 0xCD, sizeof(storage));

  const iree_net_rdma_endpoint_data_t expected = MakeTestEndpointData();
  iree_host_size_t data_length = 0;
  IREE_ASSERT_OK(iree_net_rdma_endpoint_data_serialize(
      &expected, iree_make_byte_span(storage, sizeof(storage)), &data_length));
  ASSERT_EQ(IREE_NET_RDMA_ENDPOINT_DATA_LENGTH, data_length);
  EXPECT_EQ(0xCDu, storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH]);

  iree_net_rdma_endpoint_data_t actual = {};
  IREE_ASSERT_OK(iree_net_rdma_endpoint_data_deserialize(
      iree_make_const_byte_span(storage, data_length), &actual));
  ExpectEndpointDataEq(expected, actual);
}

TEST(RdmaEndpointDataTest, SerializesLittleEndianWireLayout) {
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  SerializeTestEndpointData(storage, sizeof(storage));

  EXPECT_EQ(0x49u, storage[0]);
  EXPECT_EQ(0x52u, storage[1]);
  EXPECT_EQ(0x44u, storage[2]);
  EXPECT_EQ(0x45u, storage[3]);
  EXPECT_EQ(0x03u, storage[4]);
  EXPECT_EQ(0x00u, storage[5]);
  EXPECT_EQ(0x00u, storage[6]);
  EXPECT_EQ(0x00u, storage[7]);

  EXPECT_EQ(0x08u, storage[8]);
  EXPECT_EQ(0x07u, storage[9]);
  EXPECT_EQ(0x06u, storage[10]);
  EXPECT_EQ(0x05u, storage[11]);
  EXPECT_EQ(0x04u, storage[12]);
  EXPECT_EQ(0x03u, storage[13]);
  EXPECT_EQ(0x02u, storage[14]);
  EXPECT_EQ(0x01u, storage[15]);
  EXPECT_EQ(0x02u, storage[16]);
  EXPECT_EQ(0x00u, storage[17]);
  EXPECT_EQ(0x04u, storage[18]);
  EXPECT_EQ(0x00u, storage[19]);
  EXPECT_EQ(0x00u, storage[20]);
  EXPECT_EQ(0x10u, storage[21]);
  EXPECT_EQ(0x00u, storage[22]);
  EXPECT_EQ(0x00u, storage[23]);
  EXPECT_EQ(0x80u, storage[24]);
  EXPECT_EQ(0x00u, storage[25]);
  EXPECT_EQ(0x00u, storage[26]);
  EXPECT_EQ(0x00u, storage[27]);
  EXPECT_EQ(0x00u, storage[28]);
  EXPECT_EQ(0x00u, storage[29]);
  EXPECT_EQ(0x00u, storage[30]);
  EXPECT_EQ(0x00u, storage[31]);
  for (iree_host_size_t i = 28; i < IREE_NET_RDMA_ENDPOINT_DATA_LENGTH; ++i) {
    EXPECT_EQ(0x00u, storage[i]);
  }
}

TEST(RdmaEndpointDataTest, RejectsShortBuffer) {
  const iree_net_rdma_endpoint_data_t data = MakeTestEndpointData();
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH - 1] = {0};

  iree_host_size_t data_length = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_rdma_endpoint_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
  EXPECT_EQ(0u, data_length);

  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaEndpointDataTest, RejectsInvalidArguments) {
  const iree_net_rdma_endpoint_data_t data = MakeTestEndpointData();
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 1;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_serialize(
          nullptr, iree_make_byte_span(storage, sizeof(storage)),
          &data_length));
  EXPECT_EQ(0u, data_length);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), nullptr));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_serialize(
          &data, iree_make_byte_span(nullptr, sizeof(storage)), &data_length));

  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(nullptr, sizeof(storage)), &actual));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), nullptr));
}

TEST(RdmaEndpointDataTest, RejectsBadMagic) {
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  SerializeTestEndpointData(storage, sizeof(storage));

  storage[0] = 0;
  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaEndpointDataTest, RejectsUnsupportedVersion) {
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  SerializeTestEndpointData(storage, sizeof(storage));

  storage[4] = 1;
  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaEndpointDataTest, RejectsReservedBytes) {
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  SerializeTestEndpointData(storage, sizeof(storage));

  storage[28] = 1;
  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaEndpointDataTest, RejectsReservedFlags) {
  iree_net_rdma_endpoint_data_t data = MakeTestEndpointData();
  data.flags = 1;

  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
}

TEST(RdmaEndpointDataTest, RejectsDecodedReservedFlags) {
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  SerializeTestEndpointData(storage, sizeof(storage));
  storage[6] = 1;

  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaEndpointDataTest, RejectsZeroGroupId) {
  iree_net_rdma_endpoint_data_t data = MakeTestEndpointData();
  data.group_id = 0;

  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
}

TEST(RdmaEndpointDataTest, RejectsDecodedZeroGroupId) {
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  SerializeTestEndpointData(storage, sizeof(storage));
  memset(storage + 8, 0, sizeof(uint64_t));

  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaEndpointDataTest, RejectsZeroEndpointCount) {
  iree_net_rdma_endpoint_data_t data = MakeTestEndpointData();
  data.endpoint_count = 0;

  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
}

TEST(RdmaEndpointDataTest, RejectsDecodedZeroEndpointCount) {
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  SerializeTestEndpointData(storage, sizeof(storage));
  storage[18] = 0;
  storage[19] = 0;

  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaEndpointDataTest, RejectsEndpointIndexAtCount) {
  iree_net_rdma_endpoint_data_t data = MakeTestEndpointData();
  data.endpoint_index = data.endpoint_count;

  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
}

TEST(RdmaEndpointDataTest, RejectsDecodedEndpointIndexAtCount) {
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  SerializeTestEndpointData(storage, sizeof(storage));
  storage[16] = storage[18];
  storage[17] = storage[19];

  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaEndpointDataTest, RejectsSmallBootstrapRecvBuffer) {
  iree_net_rdma_endpoint_data_t data = MakeTestEndpointData();
  data.bootstrap_recv_buffer_size = IREE_NET_RDMA_CONNECTION_DATA_LENGTH - 1;

  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
}

TEST(RdmaEndpointDataTest, RejectsZeroBootstrapRecvCredits) {
  iree_net_rdma_endpoint_data_t data = MakeTestEndpointData();
  data.bootstrap_recv_credits = 0;

  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  iree_host_size_t data_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_serialize(
          &data, iree_make_byte_span(storage, sizeof(storage)), &data_length));
}

TEST(RdmaEndpointDataTest, RejectsDecodedSmallBootstrapRecvBuffer) {
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  SerializeTestEndpointData(storage, sizeof(storage));
  memset(storage + 20, 0, sizeof(uint32_t));

  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

TEST(RdmaEndpointDataTest, RejectsDecodedZeroBootstrapRecvCredits) {
  uint8_t storage[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH] = {0};
  SerializeTestEndpointData(storage, sizeof(storage));
  memset(storage + 24, 0, sizeof(uint32_t));

  iree_net_rdma_endpoint_data_t actual = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_endpoint_data_deserialize(
          iree_make_const_byte_span(storage, sizeof(storage)), &actual));
}

}  // namespace
