// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PCI_H_
#define AMDF_SRC_PCI_H_

#include <stdbool.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true when `pci` identifies an AMD-owned endpoint vendor.
bool amdf_pci_is_amd(const amdf_pci_info_t* pci);

// Parses a canonical hexadecimal domain:bus:device.function address.
// Success sets availability and components, preserving the output header.
// An invalid address leaves the output unchanged.
bool amdf_pci_parse_address(const char* text,
                            amdf_endpoint_pci_address_t* out_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PCI_H_
