// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_TEST_UTIL_H_
#define EXPERIMENTAL_ID4_STAGES_TEST_UTIL_H_

#include <string>
#include <vector>

#include "experimental/id4/pipeline/diagnostics.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

namespace id4::test {

typedef struct StageDiagnostics {
  // Number of diagnostic events observed.
  iree_host_size_t count;
  // Event keys observed in order.
  std::vector<std::string> keys;
  // Number of kernel diagnostic events observed.
  iree_host_size_t kernel_event_count;
} StageDiagnostics;

std::string ToString(iree_string_view_t value);

bool ContainsKey(const std::vector<std::string>& keys, const char* key);

id4_pipeline_diagnostics_sink_t DiagnosticsSink(StageDiagnostics* diagnostics);

// Returns a retained reference to the process-wide local-sync test device
// group. Callers must release the returned reference.
iree_hal_device_group_t* CreateLocalSyncDeviceGroup();

}  // namespace id4::test

#endif  // EXPERIMENTAL_ID4_STAGES_TEST_UTIL_H_
