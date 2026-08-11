// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "iree/net/carrier/shm/handshake_message.h"
#include "iree/net/carrier/shm/region.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size != sizeof(iree_net_shm_handshake_header_t)) return 0;

  iree_net_shm_handshake_header_t header;
  std::memcpy(&header, data, sizeof(header));

  iree_net_shm_region_layout_t layout;
  iree_status_free(
      iree_net_shm_handshake_message_validate_offer(&header, &layout));
  iree_status_free(iree_net_shm_handshake_message_validate_accept(&header));
  iree_status_free(iree_net_shm_handshake_message_validate_ready(&header));
  return 0;
}
