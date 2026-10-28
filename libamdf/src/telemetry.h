// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_TELEMETRY_H_
#define AMDF_SRC_TELEMETRY_H_

#include "amdf/telemetry.h"

// Acquires the immutable caller-scheduled telemetry API table without native
// initialization, allocation, or device discovery.
amdf_status_t amdf_telemetry_extension_query(uint32_t minimum_version,
                                             uint32_t maximum_version,
                                             const void** out_api);

#endif  // AMDF_SRC_TELEMETRY_H_
