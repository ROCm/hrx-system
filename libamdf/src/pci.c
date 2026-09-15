// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/pci.h"

#define AMDF_PCI_VENDOR_ID_ATI 0x1002u
#define AMDF_PCI_VENDOR_ID_AMD 0x1022u

bool amdf_pci_is_amd(const amdf_pci_info_t* pci) {
  return pci->vendor_id == AMDF_PCI_VENDOR_ID_ATI ||
         pci->vendor_id == AMDF_PCI_VENDOR_ID_AMD;
}
