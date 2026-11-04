// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_ENDPOINT_STATE_H_
#define AMDF_SRC_PLATFORM_LINUX_ENDPOINT_STATE_H_

#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/platform/linux/dxg/kmt.h"
#include "libamdf/src/platform/linux/instance.h"

struct amdf_platform_endpoint_t {
  // Instance borrowed for native identity validation and provider operations.
  amdf_platform_instance_t* instance;
  // Immutable identity and PCI properties established on open.
  amdf_endpoint_info_t info;
  // OS attachment node with online CPUs at open, or UINT32_MAX when unknown.
  uint32_t host_numa_node;
  // Query-only KMT adapter handle owned by a DXG endpoint, or zero otherwise.
  D3DKMT_HANDLE adapter;
  // Physical adapter represented by a DXG endpoint, or zero otherwise.
  uint32_t physical_adapter_index;
  // Physical PCI attachment, unavailable in the WSL guest namespace.
  amdf_endpoint_pci_address_t pci_address;
};

// Returns the immutable native service selected by the endpoint's instance.
static inline amdf_linux_platform_kind_t amdf_linux_endpoint_platform_kind(
    const amdf_platform_endpoint_t* endpoint) {
  return endpoint->instance->kind;
}

#endif  // AMDF_SRC_PLATFORM_LINUX_ENDPOINT_STATE_H_
