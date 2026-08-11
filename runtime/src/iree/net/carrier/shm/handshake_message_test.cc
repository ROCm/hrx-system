// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/handshake_message.h"

#include "iree/net/carrier/shm/region.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_net_shm_handshake_header_t MakeMessage(
    iree_net_shm_handshake_message_type_t type) {
  iree_net_shm_handshake_header_t header = {};
  header.magic = IREE_NET_SHM_HANDSHAKE_MAGIC;
  header.version = IREE_NET_SHM_HANDSHAKE_VERSION;
  header.type = type;
  return header;
}

static iree_net_shm_handshake_header_t MakeOffer(
    uint32_t ring_capacity = 4096, uint32_t wake_epoch_size = 4096) {
  iree_net_shm_region_layout_t layout;
  IREE_CHECK_OK(iree_net_shm_region_layout_calculate(ring_capacity,
                                                     wake_epoch_size, &layout));
  iree_net_shm_handshake_header_t header =
      MakeMessage(IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER);
  header.transport_region_size = layout.region_size;
  header.ring_capacity = ring_capacity;
  header.wake_epoch_size = wake_epoch_size;
  return header;
}

TEST(ShmHandshakeMessageTest, ValidOfferReturnsCanonicalLayout) {
  iree_net_shm_handshake_header_t header = MakeOffer();
  iree_net_shm_region_layout_t layout;
  IREE_ASSERT_OK(
      iree_net_shm_handshake_message_validate_offer(&header, &layout));
  EXPECT_EQ(layout.ring_capacity, header.ring_capacity);
  EXPECT_EQ(layout.mapping_alignment, header.wake_epoch_size);
  EXPECT_EQ(layout.region_size, header.transport_region_size);
}

TEST(ShmHandshakeMessageTest, OfferAcceptsSupportedCreatorPages) {
  const uint32_t page_sizes[] = {4096, 8192, 16384, 32768, 65536};
  for (uint32_t page_size : page_sizes) {
    iree_net_shm_handshake_header_t header =
        MakeOffer(/*ring_capacity=*/4096, page_size);
    iree_net_shm_region_layout_t layout;
    IREE_EXPECT_OK(
        iree_net_shm_handshake_message_validate_offer(&header, &layout));
  }
}

TEST(ShmHandshakeMessageTest, OfferCarriesExtentAboveFourGiB) {
#if defined(IREE_PTR_SIZE_64)
  iree_net_shm_handshake_header_t header = MakeOffer(UINT32_C(0x80000000));
  EXPECT_GT(header.transport_region_size, (uint64_t)UINT32_MAX);
  iree_net_shm_region_layout_t layout;
  IREE_EXPECT_OK(
      iree_net_shm_handshake_message_validate_offer(&header, &layout));
#endif  // IREE_PTR_SIZE_64
}

TEST(ShmHandshakeMessageTest, OfferRejectsIncorrectExtent) {
  iree_net_shm_handshake_header_t header = MakeOffer();
  iree_net_shm_region_layout_t layout;
  --header.transport_region_size;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_message_validate_offer(&header, &layout));

  header = MakeOffer();
  ++header.transport_region_size;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_message_validate_offer(&header, &layout));
}

TEST(ShmHandshakeMessageTest, OfferRejectsInvalidCapacity) {
  iree_net_shm_handshake_header_t header = MakeOffer();
  iree_net_shm_region_layout_t layout;
  header.ring_capacity = IREE_MPSC_QUEUE_MIN_CAPACITY / 2;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_message_validate_offer(&header, &layout));

  header = MakeOffer();
  header.ring_capacity = IREE_MPSC_QUEUE_MIN_CAPACITY + 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_message_validate_offer(&header, &layout));
}

TEST(ShmHandshakeMessageTest, OfferRejectsInvalidCreatorPage) {
  const uint32_t page_sizes[] = {0, 2048, 12288, 131072};
  for (uint32_t page_size : page_sizes) {
    iree_net_shm_handshake_header_t header = MakeOffer();
    header.wake_epoch_size = page_size;
    iree_net_shm_region_layout_t layout;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_net_shm_handshake_message_validate_offer(&header, &layout));
  }
}

TEST(ShmHandshakeMessageTest, RejectsInvalidCommonFields) {
  iree_net_shm_region_layout_t layout;
  iree_net_shm_handshake_header_t header = MakeOffer();
  header.magic ^= 1u;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_message_validate_offer(&header, &layout));

  header = MakeOffer();
  --header.version;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_message_validate_offer(&header, &layout));

  header = MakeOffer();
  header.type = IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_message_validate_offer(&header, &layout));

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(header.reserved); ++i) {
    header = MakeOffer();
    header.reserved[i] = 1;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_net_shm_handshake_message_validate_offer(&header, &layout));
  }
}

TEST(ShmHandshakeMessageTest, ValidAccept) {
  iree_net_shm_handshake_header_t header =
      MakeMessage(IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT);
  header.wake_epoch_size = 16384;
  IREE_EXPECT_OK(iree_net_shm_handshake_message_validate_accept(&header));
}

TEST(ShmHandshakeMessageTest, AcceptRejectsInapplicableGeometry) {
  iree_net_shm_handshake_header_t header =
      MakeMessage(IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT);
  header.wake_epoch_size = 4096;
  header.transport_region_size = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_message_validate_accept(&header));

  header = MakeMessage(IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT);
  header.wake_epoch_size = 4096;
  header.ring_capacity = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_message_validate_accept(&header));

  header = MakeMessage(IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_message_validate_accept(&header));
}

TEST(ShmHandshakeMessageTest, ValidReady) {
  iree_net_shm_handshake_header_t header =
      MakeMessage(IREE_NET_SHM_HANDSHAKE_MESSAGE_READY);
  IREE_EXPECT_OK(iree_net_shm_handshake_message_validate_ready(&header));
}

TEST(ShmHandshakeMessageTest, ReadyRejectsGeometry) {
  iree_net_shm_handshake_header_t header =
      MakeMessage(IREE_NET_SHM_HANDSHAKE_MESSAGE_READY);
  header.transport_region_size = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_shm_handshake_message_validate_ready(&header));

  header = MakeMessage(IREE_NET_SHM_HANDSHAKE_MESSAGE_READY);
  header.ring_capacity = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_shm_handshake_message_validate_ready(&header));

  header = MakeMessage(IREE_NET_SHM_HANDSHAKE_MESSAGE_READY);
  header.wake_epoch_size = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_shm_handshake_message_validate_ready(&header));
}

}  // namespace
