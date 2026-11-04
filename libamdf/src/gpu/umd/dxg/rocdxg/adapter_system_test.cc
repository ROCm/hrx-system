// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/api.h"
#include "libamdf/src/platform/linux/dxg/connection.h"
#include "libamdf/src/platform/linux/dxg/kmt.h"

namespace {

enum class AdapterProbeResult {
  kSuccess,
  kUnavailable,
  kFailure,
};

bool CloseAdapters(const amdf_linux_dxg_kmt_t* kmt,
                   const std::vector<D3DKMT_ADAPTERINFO>& adapters) {
  bool succeeded = true;
  for (const D3DKMT_ADAPTERINFO& adapter_info : adapters) {
    if (adapter_info.hAdapter == 0) {
      continue;
    }
    D3DKMT_CLOSEADAPTER close = {.hAdapter = adapter_info.hAdapter};
    if (amdf_linux_dxg_kmt_close_adapter(kmt, &close) != 0) {
      succeeded = false;
    }
  }
  return succeeded;
}

AdapterProbeResult ExerciseAdapter(
    amdf_dxg_bridge_gpu_properties_t* out_properties, std::string* out_error) {
  const amdf_dxg_bridge_api_t* bridge = amdf_gpu_dxg_rocdxg_default_api();
  amdf_linux_dxg_connection_t connection = {.descriptor = -1};
  const amdf_status_t connection_status =
      amdf_linux_dxg_connection_initialize(&connection);
  if (!amdf_status_is_ok(connection_status)) {
    const bool unavailable =
        amdf_status_domain(connection_status) == AMDF_STATUS_DOMAIN_ERRNO &&
        (amdf_status_code(connection_status) == ENOENT ||
         amdf_status_code(connection_status) == ENODEV);
    if (!unavailable) {
      *out_error = "opening /dev/dxg failed";
    }
    return unavailable ? AdapterProbeResult::kUnavailable
                       : AdapterProbeResult::kFailure;
  }
  const amdf_allocator_t host_allocator = amdf_allocator_system();
  amdf_linux_dxg_kmt_t kmt = {
      .connection = &connection,
      .host_allocator = &host_allocator,
  };

  D3DKMT_ENUMADAPTERS2 enumeration = {};
  NTSTATUS native_status =
      amdf_linux_dxg_kmt_enumerate_adapters(&kmt, &enumeration);
  if (native_status != 0 || enumeration.NumAdapters == 0) {
    *out_error = "adapter count query failed";
    amdf_linux_dxg_connection_deinitialize(&connection);
    return AdapterProbeResult::kFailure;
  }
  std::vector<D3DKMT_ADAPTERINFO> adapters(enumeration.NumAdapters);
  enumeration.pAdapters = adapters.data();
  native_status = amdf_linux_dxg_kmt_enumerate_adapters(&kmt, &enumeration);
  if (native_status != 0 || enumeration.NumAdapters > adapters.size()) {
    *out_error = "adapter enumeration failed";
    CloseAdapters(&kmt, adapters);
    amdf_linux_dxg_connection_deinitialize(&connection);
    return AdapterProbeResult::kFailure;
  }
  adapters.resize(enumeration.NumAdapters);

  bool found_adapter = false;
  bool cleanup_succeeded = true;
  for (const D3DKMT_ADAPTERINFO& adapter_info : adapters) {
    D3DKMT_CREATEDEVICE create = {.hAdapter = adapter_info.hAdapter};
    if (amdf_linux_dxg_kmt_create_device(&kmt, &create) != 0 ||
        create.hDevice == 0) {
      continue;
    }

    const amdf_dxg_bridge_gpu_adapter_open_info_t open_info = {
        .structure_size = sizeof(amdf_dxg_bridge_gpu_adapter_open_info_t),
        .adapter_handle = adapter_info.hAdapter,
        .device_handle = create.hDevice,
        .physical_adapter_index = 0,
        .adapter_luid = (static_cast<uint64_t>(static_cast<uint32_t>(
                             adapter_info.AdapterLuid.HighPart))
                         << 32) |
                        adapter_info.AdapterLuid.LowPart,
        .native_transport =
            {
                .context = &kmt,
                .invoke = amdf_linux_dxg_kmt_invoke,
            },
    };
    amdf_dxg_bridge_gpu_adapter_t* adapter = nullptr;
    amdf_dxg_bridge_gpu_properties_t properties = {};
    uint32_t bridge_native_status = 0;
    const amdf_dxg_bridge_result_t open_result =
        bridge->gpu_adapter_open(&open_info, &host_allocator, &adapter,
                                 &properties, &bridge_native_status);
    if (open_result == AMDF_DXG_BRIDGE_RESULT_SUCCESS) {
      found_adapter = true;
      *out_properties = properties;
      if (bridge->gpu_adapter_close(adapter, &bridge_native_status) !=
          AMDF_DXG_BRIDGE_RESULT_SUCCESS) {
        cleanup_succeeded = false;
      }
    } else if (open_result != AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED) {
      *out_error = "bridge adapter open failed";
    }

    D3DKMT_DESTROYDEVICE destroy = {.hDevice = create.hDevice};
    if (amdf_linux_dxg_kmt_destroy_device(&kmt, &destroy) != 0) {
      cleanup_succeeded = false;
    }
    if (found_adapter || !out_error->empty()) {
      break;
    }
  }

  if (!CloseAdapters(&kmt, adapters) ||
      !amdf_status_is_ok(amdf_linux_dxg_connection_deinitialize(&connection))) {
    cleanup_succeeded = false;
  }
  if (!out_error->empty() || !cleanup_succeeded) {
    if (out_error->empty()) {
      *out_error = "native teardown failed";
    }
    return AdapterProbeResult::kFailure;
  }
  if (!found_adapter) {
    *out_error = "no supported AMD DXG adapter was found";
    return AdapterProbeResult::kFailure;
  }
  return AdapterProbeResult::kSuccess;
}

TEST(RocdxgAdapterSystemTest, QueriesRealAdapterThroughExplicitDxgConnection) {
  amdf_dxg_bridge_gpu_properties_t properties = {};
  std::string error;
  ::testing::internal::CaptureStdout();
  ::testing::internal::CaptureStderr();
  const AdapterProbeResult result = ExerciseAdapter(&properties, &error);
  const std::string stderr_output = ::testing::internal::GetCapturedStderr();
  const std::string stdout_output = ::testing::internal::GetCapturedStdout();

  if (result == AdapterProbeResult::kUnavailable) {
    GTEST_SKIP() << "/dev/dxg is unavailable on this runner";
  }
  EXPECT_EQ(result, AdapterProbeResult::kSuccess) << error;
  EXPECT_TRUE(stdout_output.empty()) << stdout_output;
  EXPECT_TRUE(stderr_output.empty()) << stderr_output;
  EXPECT_GT(properties.gfx_ip_major, 0);
  EXPECT_GT(properties.wavefront_size, 0u);
  EXPECT_GT(properties.compute_unit_count, 0u);
  EXPECT_GT(properties.xcc_count, 0u);
  EXPECT_GT(properties.shader_engine_count, 0u);
}

}  // namespace
