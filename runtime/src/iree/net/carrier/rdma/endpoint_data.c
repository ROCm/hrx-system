// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/endpoint_data.h"

#include <string.h>

#include "iree/base/alignment.h"

enum iree_net_rdma_endpoint_data_offset_e {
  IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_MAGIC = 0,
  IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_VERSION = 4,
  IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_FLAGS = 6,
  IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_GROUP_ID = 8,
  IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_ENDPOINT_INDEX = 16,
  IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_ENDPOINT_COUNT = 18,
  IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_BOOTSTRAP_RECV_BUFFER_SIZE = 20,
  IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_BOOTSTRAP_RECV_CREDITS = 24,
  IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_RESERVED0 = 28,
};

static void iree_net_rdma_endpoint_data_store_u16(uint8_t* target,
                                                  iree_host_size_t offset,
                                                  uint16_t value) {
  iree_unaligned_store_le_u16((uint16_t*)(target + offset), value);
}

static void iree_net_rdma_endpoint_data_store_u32(uint8_t* target,
                                                  iree_host_size_t offset,
                                                  uint32_t value) {
  iree_unaligned_store_le_u32((uint32_t*)(target + offset), value);
}

static void iree_net_rdma_endpoint_data_store_u64(uint8_t* target,
                                                  iree_host_size_t offset,
                                                  uint64_t value) {
  iree_unaligned_store_le_u64((uint64_t*)(target + offset), value);
}

static uint16_t iree_net_rdma_endpoint_data_load_u16(const uint8_t* source,
                                                     iree_host_size_t offset) {
  return iree_unaligned_load_le_u16((const uint16_t*)(source + offset));
}

static uint32_t iree_net_rdma_endpoint_data_load_u32(const uint8_t* source,
                                                     iree_host_size_t offset) {
  return iree_unaligned_load_le_u32((const uint32_t*)(source + offset));
}

static uint64_t iree_net_rdma_endpoint_data_load_u64(const uint8_t* source,
                                                     iree_host_size_t offset) {
  return iree_unaligned_load_le_u64((const uint64_t*)(source + offset));
}

static iree_status_t iree_net_rdma_endpoint_data_validate_reserved(
    const uint8_t* source) {
  for (iree_host_size_t i = IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_RESERVED0;
       i < IREE_NET_RDMA_ENDPOINT_DATA_LENGTH; ++i) {
    if (source[i] != 0) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "RDMA endpoint bootstrap data uses reserved byte %" PRIhsz, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_endpoint_data_validate(
    const iree_net_rdma_endpoint_data_t* data) {
  if (data->flags != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported RDMA endpoint flags: 0x%04X",
                            data->flags);
  }
  if (data->group_id == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RDMA endpoint group_id must be non-zero");
  }
  if (data->endpoint_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RDMA endpoint_count must be non-zero");
  }
  if (data->endpoint_index >= data->endpoint_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RDMA endpoint_index %" PRIu16
                            " must be less than endpoint_count %" PRIu16,
                            data->endpoint_index, data->endpoint_count);
  }
  if (data->bootstrap_recv_buffer_size < IREE_NET_RDMA_CONNECTION_DATA_LENGTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RDMA bootstrap recv_buffer_size %" PRIu32
                            " is too small for connection data length %" PRIhsz,
                            data->bootstrap_recv_buffer_size,
                            IREE_NET_RDMA_CONNECTION_DATA_LENGTH);
  }
  if (data->bootstrap_recv_credits == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RDMA bootstrap_recv_credits must be non-zero");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_endpoint_data_serialize(
    const iree_net_rdma_endpoint_data_t* data, iree_byte_span_t target,
    iree_host_size_t* out_length) {
  if (!out_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_length must not be NULL");
  }
  *out_length = 0;

  if (!data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "endpoint data must not be NULL");
  }
  IREE_RETURN_IF_ERROR(iree_net_rdma_endpoint_data_validate(data));
  if (target.data_length < IREE_NET_RDMA_ENDPOINT_DATA_LENGTH) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "RDMA endpoint bootstrap data buffer too small: %" PRIhsz " < %" PRIhsz,
        target.data_length, IREE_NET_RDMA_ENDPOINT_DATA_LENGTH);
  }
  if (!target.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target buffer must not be NULL");
  }

  memset(target.data, 0, IREE_NET_RDMA_ENDPOINT_DATA_LENGTH);
  iree_net_rdma_endpoint_data_store_u32(
      target.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_MAGIC,
      IREE_NET_RDMA_ENDPOINT_DATA_MAGIC);
  iree_net_rdma_endpoint_data_store_u16(
      target.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_VERSION,
      IREE_NET_RDMA_ENDPOINT_DATA_VERSION);
  iree_net_rdma_endpoint_data_store_u16(
      target.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_FLAGS, data->flags);
  iree_net_rdma_endpoint_data_store_u64(
      target.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_GROUP_ID, data->group_id);
  iree_net_rdma_endpoint_data_store_u16(
      target.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_ENDPOINT_INDEX,
      data->endpoint_index);
  iree_net_rdma_endpoint_data_store_u16(
      target.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_ENDPOINT_COUNT,
      data->endpoint_count);
  iree_net_rdma_endpoint_data_store_u32(
      target.data,
      IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_BOOTSTRAP_RECV_BUFFER_SIZE,
      data->bootstrap_recv_buffer_size);
  iree_net_rdma_endpoint_data_store_u32(
      target.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_BOOTSTRAP_RECV_CREDITS,
      data->bootstrap_recv_credits);

  *out_length = IREE_NET_RDMA_ENDPOINT_DATA_LENGTH;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_endpoint_data_deserialize(
    iree_const_byte_span_t source, iree_net_rdma_endpoint_data_t* out_data) {
  if (!out_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_data must not be NULL");
  }
  memset(out_data, 0, sizeof(*out_data));

  if (source.data_length < IREE_NET_RDMA_ENDPOINT_DATA_LENGTH) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "RDMA endpoint bootstrap data truncated: %" PRIhsz " < %" PRIhsz,
        source.data_length, IREE_NET_RDMA_ENDPOINT_DATA_LENGTH);
  }
  if (!source.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source buffer must not be NULL");
  }

  uint32_t magic = iree_net_rdma_endpoint_data_load_u32(
      source.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_MAGIC);
  if (magic != IREE_NET_RDMA_ENDPOINT_DATA_MAGIC) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "RDMA endpoint bootstrap data has bad magic");
  }

  uint16_t version = iree_net_rdma_endpoint_data_load_u16(
      source.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_VERSION);
  if (version != IREE_NET_RDMA_ENDPOINT_DATA_VERSION) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported RDMA endpoint bootstrap data version: %" PRIu16, version);
  }
  IREE_RETURN_IF_ERROR(
      iree_net_rdma_endpoint_data_validate_reserved(source.data));

  out_data->flags = iree_net_rdma_endpoint_data_load_u16(
      source.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_FLAGS);
  out_data->group_id = iree_net_rdma_endpoint_data_load_u64(
      source.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_GROUP_ID);
  out_data->endpoint_index = iree_net_rdma_endpoint_data_load_u16(
      source.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_ENDPOINT_INDEX);
  out_data->endpoint_count = iree_net_rdma_endpoint_data_load_u16(
      source.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_ENDPOINT_COUNT);
  out_data->bootstrap_recv_buffer_size = iree_net_rdma_endpoint_data_load_u32(
      source.data,
      IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_BOOTSTRAP_RECV_BUFFER_SIZE);
  out_data->bootstrap_recv_credits = iree_net_rdma_endpoint_data_load_u32(
      source.data, IREE_NET_RDMA_ENDPOINT_DATA_OFFSET_BOOTSTRAP_RECV_CREDITS);

  return iree_net_rdma_endpoint_data_validate(out_data);
}
