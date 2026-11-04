// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_DXG_ENDPOINT_H_
#define AMDF_SRC_PLATFORM_LINUX_DXG_ENDPOINT_H_

#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/platform/linux/dxg/instance.h"
#include "libamdf/src/platform/linux/dxg/kmt.h"

struct amdf_platform_endpoint_t {
  // Platform instance borrowed by the retained adapter handle.
  amdf_platform_instance_t* instance;
  // Query-only KMT adapter handle owned by this endpoint.
  D3DKMT_HANDLE adapter;
  // Physical-adapter index represented by this endpoint.
  uint32_t physical_adapter_index;
  // Immutable endpoint properties qualified while opening the adapter.
  amdf_endpoint_info_t info;
  // Unavailable PCI attachment address in the WSL guest namespace.
  amdf_endpoint_pci_address_t pci_address;
};

#endif  // AMDF_SRC_PLATFORM_LINUX_DXG_ENDPOINT_H_
