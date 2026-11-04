// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/bridge_api.h"
#include "shared/include/d3dkmt_types.h"

namespace {

using EnumAdaptersFn = NTSTATUS (*)(const D3DKMT_ENUMADAPTERS2* arguments);
using CreateDeviceFn = NTSTATUS (*)(D3DKMT_CREATEDEVICE* arguments);
using DestroyDeviceFn = NTSTATUS (*)(const D3DKMT_DESTROYDEVICE* arguments);
using CloseAdapterFn = NTSTATUS (*)(const D3DKMT_CLOSEADAPTER* arguments);

enum class AdapterProbeResult {
  kSuccess,
  kUnavailable,
  kFailure,
};

template <typename Procedure>
Procedure LoadProcedure(void* module, const char* name) {
  static_assert(sizeof(Procedure) == sizeof(void*));
  void* symbol = dlsym(module, name);
  Procedure procedure = nullptr;
  std::memcpy(&procedure, &symbol, sizeof(symbol));
  return procedure;
}

bool ModuleIsMapped(const char* path) {
  void* module = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  if (module == nullptr) {
    return false;
  }
  dlclose(module);
  return true;
}

bool CloseAdapters(const std::vector<D3DKMT_ADAPTERINFO>& adapters,
                   CloseAdapterFn close_adapter) {
  bool succeeded = true;
  for (const D3DKMT_ADAPTERINFO& adapter_info : adapters) {
    if (adapter_info.hAdapter == 0) {
      continue;
    }
    const D3DKMT_CLOSEADAPTER close = {.hAdapter = adapter_info.hAdapter};
    if (close_adapter(&close) != 0) {
      succeeded = false;
    }
  }
  return succeeded;
}

AdapterProbeResult ExerciseAdapter(
    const char* sidecar_path, amdf_dxg_bridge_gpu_properties_t* out_properties,
    std::string* out_error) {
  void* sidecar = dlopen(sidecar_path, RTLD_NOW | RTLD_LOCAL);
  if (sidecar == nullptr) {
    *out_error = dlerror();
    return AdapterProbeResult::kFailure;
  }
  const auto query_api = LoadProcedure<amdf_dxg_bridge_query_api_fn_t>(
      sidecar, "amdf_dxg_bridge_query_api");
  if (query_api == nullptr) {
    *out_error = "sidecar query symbol is absent";
    dlclose(sidecar);
    return AdapterProbeResult::kFailure;
  }
  const amdf_dxg_bridge_api_t* bridge = nullptr;
  if (query_api(AMDF_DXG_BRIDGE_ABI_VERSION_LATEST,
                AMDF_DXG_BRIDGE_ABI_VERSION_LATEST,
                &bridge) != AMDF_DXG_BRIDGE_RESULT_SUCCESS ||
      bridge == nullptr) {
    *out_error = "sidecar API negotiation failed";
    dlclose(sidecar);
    return AdapterProbeResult::kFailure;
  }

  void* dxcore = dlopen("libdxcore.so", RTLD_NOW | RTLD_LOCAL);
  if (dxcore == nullptr) {
    dlclose(sidecar);
    return AdapterProbeResult::kUnavailable;
  }
  const auto enum_adapters =
      LoadProcedure<EnumAdaptersFn>(dxcore, "D3DKMTEnumAdapters2");
  const auto create_device =
      LoadProcedure<CreateDeviceFn>(dxcore, "D3DKMTCreateDevice");
  const auto destroy_device =
      LoadProcedure<DestroyDeviceFn>(dxcore, "D3DKMTDestroyDevice");
  const auto close_adapter =
      LoadProcedure<CloseAdapterFn>(dxcore, "D3DKMTCloseAdapter");
  if (enum_adapters == nullptr || create_device == nullptr ||
      destroy_device == nullptr || close_adapter == nullptr) {
    *out_error = "libdxcore lacks required adapter procedures";
    dlclose(dxcore);
    dlclose(sidecar);
    return AdapterProbeResult::kFailure;
  }

  D3DKMT_ENUMADAPTERS2 enumeration = {};
  NTSTATUS native_status = enum_adapters(&enumeration);
  if (native_status != 0 || enumeration.NumAdapters == 0) {
    *out_error = "adapter count query failed";
    dlclose(dxcore);
    dlclose(sidecar);
    return AdapterProbeResult::kFailure;
  }
  std::vector<D3DKMT_ADAPTERINFO> adapters(enumeration.NumAdapters);
  enumeration.pAdapters = adapters.data();
  native_status = enum_adapters(&enumeration);
  if (native_status != 0 || enumeration.NumAdapters > adapters.size()) {
    *out_error = "adapter enumeration failed";
    CloseAdapters(adapters, close_adapter);
    dlclose(dxcore);
    dlclose(sidecar);
    return AdapterProbeResult::kFailure;
  }
  adapters.resize(enumeration.NumAdapters);

  const amdf_allocator_t host_allocator = amdf_allocator_system();
  bool found_adapter = false;
  bool cleanup_succeeded = true;
  for (const D3DKMT_ADAPTERINFO& adapter_info : adapters) {
    D3DKMT_CREATEDEVICE create = {.hAdapter = adapter_info.hAdapter};
    if (create_device(&create) != 0 || create.hDevice == 0) {
      continue;
    }

    const amdf_dxg_bridge_gpu_adapter_open_info_t open_info = {
        sizeof(amdf_dxg_bridge_gpu_adapter_open_info_t),
        adapter_info.hAdapter,
        create.hDevice,
        0,
        (static_cast<uint64_t>(
             static_cast<uint32_t>(adapter_info.AdapterLuid.HighPart))
         << 32) |
            adapter_info.AdapterLuid.LowPart,
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

    const D3DKMT_DESTROYDEVICE destroy = {.hDevice = create.hDevice};
    if (destroy_device(&destroy) != 0) {
      cleanup_succeeded = false;
    }
    if (found_adapter || !out_error->empty()) {
      break;
    }
  }

  if (!CloseAdapters(adapters, close_adapter)) {
    cleanup_succeeded = false;
  }
  if (dlclose(sidecar) != 0 || dlclose(dxcore) != 0) {
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

TEST(RocdxgAdapterSystemTest, QueriesRealAdapterAndReleasesNativeState) {
  const char* sidecar_path = std::getenv("AMDF_ROCDXG_BRIDGE_TEST_PATH");
  ASSERT_NE(sidecar_path, nullptr);
  const bool dxcore_was_mapped = ModuleIsMapped("libdxcore.so");

  amdf_dxg_bridge_gpu_properties_t properties = {};
  std::string error;
  ::testing::internal::CaptureStdout();
  ::testing::internal::CaptureStderr();
  const AdapterProbeResult result =
      ExerciseAdapter(sidecar_path, &properties, &error);
  const std::string stderr_output = ::testing::internal::GetCapturedStderr();
  const std::string stdout_output = ::testing::internal::GetCapturedStdout();

  if (result == AdapterProbeResult::kUnavailable) {
    GTEST_SKIP() << "libdxcore is unavailable on this runner";
  }
  EXPECT_EQ(result, AdapterProbeResult::kSuccess) << error;
  EXPECT_TRUE(stdout_output.empty()) << stdout_output;
  EXPECT_TRUE(stderr_output.empty()) << stderr_output;
  EXPECT_FALSE(ModuleIsMapped(sidecar_path));
  if (!dxcore_was_mapped) {
    EXPECT_FALSE(ModuleIsMapped("libdxcore.so"));
  }
  EXPECT_GT(properties.gfx_ip_major, 0);
  EXPECT_GT(properties.wavefront_size, 0u);
  EXPECT_GT(properties.compute_unit_count, 0u);
  EXPECT_GT(properties.xcc_count, 0u);
  EXPECT_GT(properties.shader_engine_count, 0u);
}

}  // namespace
