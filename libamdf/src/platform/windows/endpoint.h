// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_H_
#define AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_H_

#include <stdint.h>

#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/platform/windows/instance.h"

struct amdf_platform_endpoint_t {
  // Platform instance borrowed by the retained adapter handle.
  amdf_platform_instance_t* instance;
  // Query-only KMT adapter handle owned by this endpoint.
  D3DKMT_HANDLE adapter;
  // Physical-adapter index represented by this endpoint.
  uint32_t physical_adapter_index;
  // Opaque public identity cached when the endpoint was opened.
  amdf_endpoint_id_t id;
};

#endif  // AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_H_
