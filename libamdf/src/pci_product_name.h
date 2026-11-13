// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PCI_PRODUCT_NAME_H_
#define AMDF_SRC_PCI_PRODUCT_NAME_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Looks up a native PCI identity in the compiled AMD GPU product registry.
// Returns a library-lifetime UTF-8 string fitting AMDF_ENDPOINT_NAME_CAPACITY,
// or NULL when the identity is absent. Performs no allocation or platform IO.
const char* amdf_pci_product_name(const amdf_pci_info_t* pci);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PCI_PRODUCT_NAME_H_
