// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_PM4_COMMAND_FIXTURE_H_
#define AMDF_CTS_GPU_PM4_COMMAND_FIXTURE_H_

#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

// Selects the target's compute recipe and admits its semantic queue
// requirements before borrowing the cached native device.
class Pm4CommandTest : public GpuCommandTest {
 protected:
  explicit Pm4CommandTest(amdf_queue_roles_t additional_roles = 0,
                          amdf_queue_publication_modes_t publication_modes =
                              AMDF_QUEUE_PUBLICATION_MODE_USER |
                              AMDF_QUEUE_PUBLICATION_MODE_KERNEL)
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
            .roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL |
                     additional_roles,
            .format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR,
            .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
            .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
            .publication_modes = publication_modes,
        }) {}

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    amdf_gpu_endpoint_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
    info.structure_size = sizeof(info);
    const amdf_status_t status = gpu_api_->endpoint_query_info(endpoint, &info);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    pm4_profile_ = Pm4CommandProfile::Find(info);
    if (!pm4_profile_) {
      *out_matches = false;
      return AMDF_STATUS_OK;
    }
    return GpuCommandTest::MatchGpuEndpoint(endpoint, out_matches);
  }

  // Static command encoding selected from the passively matched endpoint.
  const Pm4CommandProfile* pm4_profile_ = nullptr;
};

#endif  // AMDF_CTS_GPU_PM4_COMMAND_FIXTURE_H_
