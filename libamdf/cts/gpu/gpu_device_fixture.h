// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_GPU_DEVICE_FIXTURE_H_
#define AMDF_CTS_GPU_GPU_DEVICE_FIXTURE_H_

#include <vector>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "gtest/gtest.h"
#include "util/device_cache.h"
#include "util/provider.h"

// Finds a backing profile qualified for the requested live device access.
inline uint32_t FindGpuMemoryProfileOrdinal(
    const amdf_api_t* api, amdf_memory_scope_t* scope, amdf_device_t* device,
    amdf_memory_profile_roles_t required_roles,
    amdf_memory_flags_t required_flags,
    amdf_memory_access_requirements_t requirements) {
  const amdf_memory_device_access_t access = {device, requirements};
  amdf_memory_scope_info_t scope_info = {};
  scope_info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
  scope_info.structure_size = sizeof(scope_info);
  const amdf_status_t scope_status =
      api->memory_scope_query_info(scope, &scope_info);
  if (!amdf_status_is_ok(scope_status)) {
    ADD_FAILURE() << "memory scope query failed: domain="
                  << amdf_status_domain(scope_status)
                  << " code=" << amdf_status_code(scope_status);
    return AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  }
  for (uint32_t ordinal = 0; ordinal < scope_info.memory_profile_count;
       ++ordinal) {
    amdf_memory_profile_t profile = {};
    profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile.structure_size = sizeof(profile);
    amdf_memory_access_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    const amdf_status_t status = api->memory_scope_query_device_profile(
        scope, ordinal, 1, &access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
    if (!amdf_status_is_ok(status)) {
      ADD_FAILURE() << "memory profile query failed: domain="
                    << amdf_status_domain(status)
                    << " code=" << amdf_status_code(status);
      break;
    }
    if ((profile.roles & required_roles) == required_roles &&
        (required_flags & ~profile.supported_flags) == 0) {
      return ordinal;
    }
  }
  ADD_FAILURE() << "no matching memory profile";
  return AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
}

// Borrows the shared device for the selected GPU endpoint and instance policy.
class GpuDeviceFixture : public ::testing::Test {
 protected:
  // Selects which opened GPU endpoint should back this fixture. A failure
  // leaves `out_matches` unchanged.
  virtual amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                         bool* out_matches) {
    (void)endpoint;
    *out_matches = true;
    return AMDF_STATUS_OK;
  }

  void SetUp() override {
    ASSERT_TRUE(amdf_status_is_ok(amdf_cts_provider_query_api()(
        AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api_)));
    ASSERT_NE(api_, nullptr);

    const void* extension_api = nullptr;
    ASSERT_TRUE(amdf_status_is_ok(api_->query_extension(
        AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
        AMDF_GPU_EXTENSION_VERSION_LATEST, &extension_api)));
    gpu_api_ = static_cast<const amdf_gpu_api_t*>(extension_api);
    ASSERT_NE(gpu_api_, nullptr);

    amdf_status_t status = GetCtsDeviceCache().GetInstance(&instance_);
    if (amdf_status_domain(status) == AMDF_STATUS_DOMAIN_API &&
        amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
      GTEST_SKIP() << "platform provider is not implemented";
    }
    ASSERT_TRUE(amdf_status_is_ok(status));

    uint32_t scope_count = 0;
    ASSERT_EQ(amdf_status_code(api_->instance_enumerate_memory_scopes(
                  instance_, 0, nullptr, &scope_count)),
              AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    std::vector<amdf_memory_scope_t*> scopes(scope_count);
    ASSERT_EQ(api_->instance_enumerate_memory_scopes(
                  instance_, scope_count, scopes.data(), &scope_count),
              AMDF_STATUS_OK);
    for (amdf_memory_scope_t* scope : scopes) {
      amdf_memory_scope_info_t info = {};
      info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
      info.structure_size = sizeof(info);
      ASSERT_EQ(api_->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
      if (info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) system_scope_ = scope;
    }
    ASSERT_NE(system_scope_, nullptr);

    uint32_t endpoint_count = 0;
    ASSERT_TRUE(amdf_status_is_ok(
        api_->endpoint_enumerate(instance_, 0, nullptr, &endpoint_count)));
    std::vector<amdf_endpoint_summary_t> summaries(endpoint_count);
    if (endpoint_count != 0) {
      ASSERT_TRUE(amdf_status_is_ok(api_->endpoint_enumerate(
          instance_, endpoint_count, summaries.data(), &endpoint_count)));
    }
    for (const amdf_endpoint_summary_t& summary : summaries) {
      if (summary.engine_kind != AMDF_ENGINE_KIND_GPU) continue;
      ASSERT_TRUE(amdf_status_is_ok(
          GetCtsDeviceCache().OpenEndpoint(summary.id, &endpoint_)));
      bool matches = false;
      status = MatchGpuEndpoint(endpoint_, &matches);
      ASSERT_EQ(status, AMDF_STATUS_OK)
          << "domain=" << amdf_status_domain(status)
          << " code=" << amdf_status_code(status);
      if (matches) break;
      endpoint_ = nullptr;
    }
    if (endpoint_ == nullptr) {
      GTEST_SKIP() << "no qualified GPU endpoint present";
    }

    amdf_gpu_device_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    status =
        gpu_api_->endpoint_query_device_capabilities(endpoint_, &capabilities);
    if (amdf_status_domain(status) == AMDF_STATUS_DOMAIN_API &&
        amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
      GTEST_SKIP() << "requested native lifetime is unavailable";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);

    scope_count = 0;
    status = api_->endpoint_enumerate_memory_scopes(endpoint_, 0, nullptr,
                                                    &scope_count);
    ASSERT_EQ(status,
              scope_count == 0
                  ? AMDF_STATUS_OK
                  : amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
    scopes.resize(scope_count);
    ASSERT_EQ(api_->endpoint_enumerate_memory_scopes(
                  endpoint_, scope_count, scopes.data(), &scope_count),
              AMDF_STATUS_OK);
    for (amdf_memory_scope_t* scope : scopes) {
      amdf_memory_scope_info_t info = {};
      info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
      info.structure_size = sizeof(info);
      ASSERT_EQ(api_->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
      if (info.kind == AMDF_MEMORY_SCOPE_KIND_LOCAL) local_scope_ = scope;
    }

    status = GetCtsDeviceCache().GetGpuDevice(endpoint_, &device_);
    // Passive capabilities describe the provider; activation admits the
    // installed native ABI under the requested lifetime policy.
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "native GPU activation is unavailable for this lifetime";
    }
    ASSERT_TRUE(amdf_status_is_ok(status))
        << "domain=" << amdf_status_domain(status)
        << " code=" << amdf_status_code(status);
    amdf_gpu_device_info_t device_info = {};
    device_info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
    device_info.structure_size = sizeof(device_info);
    ASSERT_EQ(gpu_api_->device_query_info(device_, &device_info),
              AMDF_STATUS_OK);
    features_ = device_info.features;
    memory_access_.device = device_;
    memory_access_.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    memory_access_.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  }

  amdf_status_t QueryMemoryProfile(
      amdf_memory_scope_t* scope, uint32_t ordinal,
      amdf_memory_access_requirements_t requirements,
      amdf_memory_profile_t* out_profile,
      amdf_memory_access_capabilities_t* out_capabilities) const {
    const amdf_memory_device_access_t access = {device_, requirements};
    return api_->memory_scope_query_device_profile(
        scope, ordinal, 1, &access, out_profile, out_capabilities);
  }

  uint32_t FindMemoryProfileOrdinal(
      amdf_memory_scope_t* scope, amdf_memory_profile_roles_t required_roles,
      amdf_memory_flags_t required_flags,
      amdf_memory_access_requirements_t requirements) const {
    return FindGpuMemoryProfileOrdinal(api_, scope, device_, required_roles,
                                       required_flags, requirements);
  }

  // Core table borrowed from the CTS provider.
  const amdf_api_t* api_ = nullptr;
  // GPU table borrowed from the CTS provider.
  const amdf_gpu_api_t* gpu_api_ = nullptr;
  // Shared instance borrowed from the process-lifetime CTS cache.
  amdf_instance_t* instance_ = nullptr;
  // Shared query endpoint borrowed from the CTS cache.
  amdf_endpoint_t* endpoint_ = nullptr;
  // Shared native device; each case releases only its workload children.
  amdf_device_t* device_ = nullptr;
  // Borrowed system scope discovered before device activation.
  amdf_memory_scope_t* system_scope_ = nullptr;
  // Borrowed physical local scope, absent when the endpoint has no local heap.
  amdf_memory_scope_t* local_scope_ = nullptr;
  // Default explicit consumer used by fixture-owned construction requests.
  amdf_memory_device_access_t memory_access_ = {};
  // Achieved capabilities of the shared native device.
  amdf_gpu_device_features_t features_ = 0;
};

#endif  // AMDF_CTS_GPU_GPU_DEVICE_FIXTURE_H_
