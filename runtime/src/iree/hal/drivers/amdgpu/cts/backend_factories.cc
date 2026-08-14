// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/cts/backend_factories.h"

#include <string>

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/executable/amdgpu/executable_target.h"

namespace iree::hal::cts {
namespace {

static void InitializeAmdgpuCtsDeviceOptions(
    AmdgpuCtsBackendMode mode,
    iree_hal_amdgpu_logical_device_options_t* out_options) {
  iree_hal_amdgpu_logical_device_options_initialize(out_options);
  switch (mode) {
    case AmdgpuCtsBackendMode::kDefault:
      break;
    case AmdgpuCtsBackendMode::kAsan:
      out_options->asan.enabled = 1;
      break;
    case AmdgpuCtsBackendMode::kTsan:
      out_options->tsan.enabled = 1;
      out_options->tsan.report_policy =
          IREE_HAL_AMDGPU_TSAN_REPORT_POLICY_REPORT_ONLY;
      break;
  }
}

static std::string AmdgpuHostCompatibilityReason(
    iree_hal_amdgpu_logical_device_host_compatibility_t compatibility) {
  switch (compatibility) {
    case IREE_HAL_AMDGPU_LOGICAL_DEVICE_HOST_COMPATIBILITY_COMPATIBLE:
      return "";
    case IREE_HAL_AMDGPU_LOGICAL_DEVICE_HOST_COMPATIBILITY_INCOMPATIBLE_HOST_TSAN_ASAN:
      return "AMDGPU ASAN is not supported in host ThreadSanitizer builds";
    default:
      return "AMDGPU logical-device options are not compatible with this host";
  }
}

static bool IsAmdgpuCtsBackendHostCompatible(AmdgpuCtsBackendMode mode,
                                             std::string* out_reason) {
  iree_hal_amdgpu_logical_device_options_t options;
  InitializeAmdgpuCtsDeviceOptions(mode, &options);
  const iree_hal_amdgpu_logical_device_host_compatibility_t compatibility =
      iree_hal_amdgpu_logical_device_options_query_host_compatibility(&options);
  if (compatibility ==
      IREE_HAL_AMDGPU_LOGICAL_DEVICE_HOST_COMPATIBILITY_COMPATIBLE) {
    return true;
  }
  if (out_reason) {
    *out_reason = AmdgpuHostCompatibilityReason(compatibility);
  }
  return false;
}

static iree_status_t SelectAmdgpuCtsExecutableTarget(
    const iree_hal_device_spec_t* device_spec,
    iree_string_view_t artifact_target_family,
    iree_string_view_t artifact_target_key,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    iree_hal_executable_target_selection_result_t* out_result) {
  if (!iree_string_view_equal(artifact_target_family, IREE_SV("amdgpu"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU CTS executable target family must be 'amdgpu'; got '%.*s'",
        (int)artifact_target_family.size, artifact_target_family.data);
  }
  return iree_hal_amdgpu_device_spec_select_executable_target(
      device_spec, artifact_target_key, physical_device_affinity, out_result);
}

static iree_status_t CreateAmdgpuCtsDevice(
    AmdgpuCtsBackendMode mode,
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  InitializeAmdgpuCtsDeviceOptions(mode, &options.default_device_options);

  iree_hal_driver_t* driver = nullptr;
  iree_status_t status = iree_hal_amdgpu_driver_create(
      IREE_SV("amdgpu"), &options, iree_allocator_system(), &driver);

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_default_device(
        driver, create_params, iree_allocator_system(), &device);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

}  // namespace

BackendInfo MakeAmdgpuCtsBackendInfo(const char* name,
                                     AmdgpuCtsBackendMode mode) {
  BackendInfo info;
  info.name = name;
  info.factory = [mode](const iree_hal_device_create_params_t* create_params,
                        iree_hal_driver_t** out_driver,
                        iree_hal_device_t** out_device) {
    return CreateAmdgpuCtsDevice(mode, create_params, out_driver, out_device);
  };
  info.unsupported_tests = {
      // Features and API surface not currently implemented.
  };
  info.host_compatibility_fn = [mode](std::string* out_reason) {
    return IsAmdgpuCtsBackendHostCompatible(mode, out_reason);
  };
  info.executable_target_selector = SelectAmdgpuCtsExecutableTarget;
  return info;
}

}  // namespace iree::hal::cts
