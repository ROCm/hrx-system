// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/handshake_message.h"

#include <string.h>

#include "iree/net/carrier/shm/region.h"

static iree_status_t iree_net_shm_handshake_message_validate_common(
    const iree_net_shm_handshake_header_t* header,
    iree_net_shm_handshake_message_type_t expected_type) {
  if (header->magic != IREE_NET_SHM_HANDSHAKE_MAGIC) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SHM handshake magic mismatch: got 0x%08x, expected 0x%08x",
        header->magic, IREE_NET_SHM_HANDSHAKE_MAGIC);
  }
  if (header->version != IREE_NET_SHM_HANDSHAKE_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM handshake version mismatch: got %" PRIu32
                            ", expected %" PRIu32,
                            header->version, IREE_NET_SHM_HANDSHAKE_VERSION);
  }
  if (header->type != expected_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unexpected SHM handshake message type: got %u, expected %u",
        (unsigned)header->type, (unsigned)expected_type);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(header->reserved); ++i) {
    if (header->reserved[i] != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SHM handshake reserved byte %" PRIhsz " must be zero", i);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_net_shm_handshake_message_validate_wake_extent(
    uint32_t wake_epoch_size) {
  if (wake_epoch_size < IREE_NET_SHM_REGION_MIN_MAPPING_ALIGNMENT ||
      wake_epoch_size > IREE_NET_SHM_REGION_MAX_MAPPING_ALIGNMENT ||
      !iree_host_size_is_power_of_two((iree_host_size_t)wake_epoch_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "wake epoch size %" PRIu32 " must be a power of two between %" PRIhsz
        " and %" PRIhsz,
        wake_epoch_size, IREE_NET_SHM_REGION_MIN_MAPPING_ALIGNMENT,
        IREE_NET_SHM_REGION_MAX_MAPPING_ALIGNMENT);
  }
  return iree_ok_status();
}

iree_status_t iree_net_shm_handshake_message_validate_offer(
    const iree_net_shm_handshake_header_t* header,
    iree_net_shm_region_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));

  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_message_validate_common(
      header, IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER));
  iree_net_shm_region_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_net_shm_region_layout_calculate(
      header->ring_capacity, (iree_host_size_t)header->wake_epoch_size,
      &layout));
  if (header->transport_region_size != (uint64_t)layout.region_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "offered transport region size %" PRIu64
                            " does not match expected %" PRIhsz,
                            header->transport_region_size, layout.region_size);
  }

  *out_layout = layout;
  return iree_ok_status();
}

iree_status_t iree_net_shm_handshake_message_validate_accept(
    const iree_net_shm_handshake_header_t* header) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_message_validate_common(
      header, IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT));
  if (header->transport_region_size != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "ACCEPT transport region size must be zero, got %" PRIu64,
        header->transport_region_size);
  }
  if (header->ring_capacity != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ACCEPT ring capacity must be zero, got %" PRIu32,
                            header->ring_capacity);
  }
  return iree_net_shm_handshake_message_validate_wake_extent(
      header->wake_epoch_size);
}

iree_status_t iree_net_shm_handshake_message_validate_ready(
    const iree_net_shm_handshake_header_t* header) {
  IREE_ASSERT_ARGUMENT(header);
  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_message_validate_common(
      header, IREE_NET_SHM_HANDSHAKE_MESSAGE_READY));
  if (header->transport_region_size != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "READY transport region size must be zero, got %" PRIu64,
        header->transport_region_size);
  }
  if (header->ring_capacity != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "READY ring capacity must be zero, got %" PRIu32,
                            header->ring_capacity);
  }
  if (header->wake_epoch_size != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "READY wake epoch size must be zero, got %" PRIu32,
                            header->wake_epoch_size);
  }
  return iree_ok_status();
}
