// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/pci_product_name.h"

#include <assert.h>

#include "pci_product_name.inl"

static_assert(AMDF_PCI_PRODUCT_MAX_NAME_LENGTH < AMDF_ENDPOINT_NAME_CAPACITY,
              "product names must fit endpoint name storage");

const char* amdf_pci_product_name(const amdf_pci_info_t* pci) {
  if (pci->vendor_id != 0x1002u) {
    return NULL;
  }
  const uint32_t key = (pci->device_id << 8) | pci->revision_id;
  size_t low = 0, high = AMDF_PCI_PRODUCT_DEVICE_COUNT;
  while (low < high) {
    const size_t middle = low + (high - low) / 2;
    if (amdf_pci_product_device_keys[middle] < key) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low == AMDF_PCI_PRODUCT_DEVICE_COUNT ||
      amdf_pci_product_device_keys[low] != key) {
    return NULL;
  }
  const size_t device_index = low;
#if AMDF_PCI_PRODUCT_SUBSYSTEM_COUNT
  const uint64_t subsystem_key = ((uint64_t)key << 32) |
                                 ((uint64_t)pci->subsystem_vendor_id << 16) |
                                 pci->subsystem_device_id;
  low = 0;
  high = AMDF_PCI_PRODUCT_SUBSYSTEM_COUNT;
  while (low < high) {
    const size_t middle = low + (high - low) / 2;
    if (amdf_pci_product_subsystem_keys[middle] < subsystem_key) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low < AMDF_PCI_PRODUCT_SUBSYSTEM_COUNT &&
      amdf_pci_product_subsystem_keys[low] == subsystem_key) {
    return amdf_pci_product_names + amdf_pci_product_subsystem_offsets[low];
  }
#endif  // AMDF_PCI_PRODUCT_SUBSYSTEM_COUNT
  return amdf_pci_product_names + amdf_pci_product_device_offsets[device_index];
}
