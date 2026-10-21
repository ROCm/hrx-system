// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/driver.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "amdf/api.h"
#include "amdf/gpu.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amd/registration/driver_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Tests mutate only these scheme variables and restore the caller's
// environment. A whitespace-only value exercises the empty-list contract on
// both CRTs: the Windows CRT setter removes a variable when given an empty
// string.
class ScopedEnvironment {
 public:
  explicit ScopedEnvironment(const char* name) : name_(name) {
    if (const char* value = std::getenv(name)) {
      original_ = value;
    }
    Set(nullptr);
  }
  ~ScopedEnvironment() { Set(original_ ? original_->c_str() : nullptr); }

  void Set(const char* value) {
#if defined(IREE_PLATFORM_WINDOWS)
    EXPECT_EQ(_putenv_s(name_, value ? value : ""), 0);
#else
    EXPECT_EQ(value ? setenv(name_, value, 1) : unsetenv(name_), 0);
#endif
  }

 private:
  // Process environment key borrowed from a string literal.
  const char* name_;
  // Original value, preserving the distinction between unset and empty.
  std::optional<std::string> original_;
};

// The native dependency supplies passive topology and observes the selected
// endpoint at activation. Successful native execution is covered by GPU CTS.
struct NativeEndpoint {
  // Stable instance-local identity and engine classification.
  amdf_endpoint_summary_t summary = {};
  // Architecture facts copied through the GPU extension.
  amdf_gpu_endpoint_info_t gpu = {};
  // Cached family descriptors, independent of any active device.
  std::vector<amdf_queue_family_info_t> families;
};

amdf_queue_family_info_t ComputeFamily(amdf_queue_publication_modes_t mode) {
  return {
      .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
      .structure_size = sizeof(amdf_queue_family_info_t),
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
      .publication_modes = mode,
      .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
      .format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR,
      .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER |
               AMDF_QUEUE_ROLE_ATOMIC | AMDF_QUEUE_ROLE_CACHE_CONTROL,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      .atomic_capabilities =
          {
              .wait_conditions_32 = AMDF_ATOMIC_WAIT_CONDITION_EQUAL,
              .operations_without_dispatch_32 =
                  AMDF_ATOMIC_OPERATION_STORE | AMDF_ATOMIC_OPERATION_WAIT,
          },
      .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
  };
}

struct NativeInstance {
  NativeInstance() {
    for (uint32_t i = 0; i < endpoints.size(); ++i) {
      endpoints[i].summary.id.words[0] = i + 1;
      endpoints[i].summary.engine_kind = AMDF_ENGINE_KIND_GPU;
      std::snprintf(endpoints[i].summary.name,
                    sizeof(endpoints[i].summary.name), "GPU %u", i);
      endpoints[i].gpu.gfx_ip.major = 11;
    }
    endpoints[0].gpu.gfx_ip = {10, 3, 6};
    endpoints[1].gpu.gfx_ip = {12, 0, 1};
    endpoints[1].families.push_back(
        ComputeFamily(AMDF_QUEUE_PUBLICATION_MODE_KERNEL));
    endpoints[1].families.push_back(
        {.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA});
    endpoints[2].families.push_back(
        ComputeFamily(AMDF_QUEUE_PUBLICATION_MODE_KERNEL));
    endpoints[2]
        .families[0]
        .atomic_capabilities.operations_without_dispatch_32 = 0;
    endpoints[3].families.push_back(
        ComputeFamily(AMDF_QUEUE_PUBLICATION_MODE_USER));
    endpoints[4].summary.engine_kind = AMDF_ENGINE_KIND_XDNA;
    endpoints[4].families.push_back({
        .command_type = AMDF_QUEUE_COMMAND_TYPE_XDNA,
        .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
    });
    endpoints[5].gpu.gfx_ip = {9, 4, 2};
    endpoints[5].families.push_back({
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(amdf_queue_family_info_t),
        .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
        .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
        .format_version = AMDF_GPU_AQL_QUEUE_FORMAT_VERSION_1,
        .format_features = AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE,
        .roles = AMDF_QUEUE_ROLE_COMPUTE,
        .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                                   AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
    });
  }

  // One fake dependency instance per test; no native driver is loaded.
  static NativeInstance* current;
  // Mixed native topology with two usable PM4 GPUs, one NPU, and one AQL GPU.
  std::array<NativeEndpoint, 6> endpoints;
  // Instance-owned identity for the system memory scope.
  int system_scope = 0;
  // Live query-only handles, balanced on success and query failure.
  uint32_t open_count = 0;
  // Number of passive opens across discovery and device construction.
  uint32_t query_count = 0;
  // Number of GPU extension negotiations during passive discovery.
  uint32_t extension_query_count = 0;
  // Number of instance creations through this fake dependency.
  uint32_t instance_count = 0;
  // Number of explicit device activation attempts.
  uint32_t activation_count = 0;
  // Physical endpoint selected by the most recent activation attempt.
  amdf_endpoint_id_t activated = {};
  // Optional native family-query error, never interpreted as absent support.
  amdf_status_t query_failure = AMDF_STATUS_OK;
};
NativeInstance* NativeInstance::current = nullptr;

amdf_status_t AMDF_CALL CreateInstance(const amdf_instance_create_info_t*,
                                       amdf_instance_t** out_instance) {
  ++NativeInstance::current->instance_count;
  *out_instance = reinterpret_cast<amdf_instance_t*>(NativeInstance::current);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL DestroyInstance(amdf_instance_t* instance) {
  auto* native = reinterpret_cast<NativeInstance*>(instance);
  EXPECT_EQ(native->open_count, 0u);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL EnumerateScopes(amdf_instance_t* instance,
                                        uint32_t capacity,
                                        amdf_memory_scope_t** scopes,
                                        uint32_t* out_count) {
  *out_count = 1;
  if (!capacity) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
  }
  scopes[0] = reinterpret_cast<amdf_memory_scope_t*>(
      &reinterpret_cast<NativeInstance*>(instance)->system_scope);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryScope(amdf_memory_scope_t*,
                                   amdf_memory_scope_info_t* out_info) {
  out_info->kind = AMDF_MEMORY_SCOPE_KIND_SYSTEM;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL EnumerateEndpoints(amdf_instance_t* instance,
                                           uint32_t capacity,
                                           amdf_endpoint_summary_t* summaries,
                                           uint32_t* out_count) {
  const auto& endpoints =
      reinterpret_cast<NativeInstance*>(instance)->endpoints;
  *out_count = static_cast<uint32_t>(endpoints.size());
  for (uint32_t i = 0; i < capacity && i < endpoints.size(); ++i) {
    summaries[i] = endpoints[i].summary;
  }
  return capacity && capacity < endpoints.size()
             ? amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)
             : AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL OpenEndpoint(amdf_instance_t* instance,
                                     const amdf_endpoint_id_t* id,
                                     amdf_endpoint_t** out_endpoint) {
  auto* native = reinterpret_cast<NativeInstance*>(instance);
  for (auto& endpoint : native->endpoints) {
    if (amdf_endpoint_id_is_equal(&endpoint.summary.id, id)) {
      ++native->open_count;
      ++native->query_count;
      *out_endpoint = reinterpret_cast<amdf_endpoint_t*>(&endpoint);
      return AMDF_STATUS_OK;
    }
  }
  return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
}

amdf_status_t AMDF_CALL CloseEndpoint(amdf_endpoint_t*) {
  --NativeInstance::current->open_count;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryEndpoint(amdf_endpoint_t* endpoint,
                                      amdf_endpoint_info_t* out_info) {
  out_info->queue_family_count = static_cast<uint32_t>(
      reinterpret_cast<NativeEndpoint*>(endpoint)->families.size());
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryGpu(amdf_endpoint_t* endpoint,
                                 amdf_gpu_endpoint_info_t* out_info) {
  out_info->gfx_ip = reinterpret_cast<NativeEndpoint*>(endpoint)->gpu.gfx_ip;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryFamily(amdf_endpoint_t* endpoint, uint32_t ordinal,
                                    amdf_queue_family_info_t* out_info) {
  if (!amdf_status_is_ok(NativeInstance::current->query_failure)) {
    return NativeInstance::current->query_failure;
  }
  *out_info = reinterpret_cast<NativeEndpoint*>(endpoint)->families.at(ordinal);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL CreateGpu(amdf_endpoint_t* endpoint,
                                  const amdf_gpu_device_create_info_t*,
                                  amdf_device_t**) {
  ++NativeInstance::current->activation_count;
  NativeInstance::current->activated =
      reinterpret_cast<NativeEndpoint*>(endpoint)->summary.id;
  // Stop at the external construction boundary; the HAL must preserve this
  // resource failure instead of falling back to another physical device.
  return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
}

amdf_status_t AMDF_CALL QueryExtension(amdf_extension_id_t extension,
                                       uint32_t minimum, uint32_t maximum,
                                       const void** out_api) {
  ++NativeInstance::current->extension_query_count;
  EXPECT_EQ(extension, AMDF_EXTENSION_GPU);
  EXPECT_LE(minimum, AMDF_GPU_EXTENSION_VERSION_LATEST);
  EXPECT_GE(maximum, AMDF_GPU_EXTENSION_VERSION_LATEST);
  static const amdf_gpu_api_t api = {
      .structure_size = sizeof(amdf_gpu_api_t),
      .extension_version = AMDF_GPU_EXTENSION_VERSION_LATEST,
      .endpoint_query_info = QueryGpu,
      .device_create = CreateGpu,
  };
  *out_api = &api;
  return AMDF_STATUS_OK;
}

const amdf_api_t kNativeApi = {
    .structure_size = sizeof(amdf_api_t),
    .abi_version = AMDF_ABI_VERSION_LATEST,
    .instance_create = CreateInstance,
    .instance_destroy = DestroyInstance,
    .endpoint_enumerate = EnumerateEndpoints,
    .endpoint_open = OpenEndpoint,
    .endpoint_query_info = QueryEndpoint,
    .endpoint_close = CloseEndpoint,
    .query_extension = QueryExtension,
    .endpoint_query_queue_family_info = QueryFamily,
    .instance_enumerate_memory_scopes = EnumerateScopes,
    .memory_scope_query_info = QueryScope,
};

class DriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    NativeInstance::current = &native_;
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactors_));
  }
  void TearDown() override {
    iree_hal_driver_release(driver_);
    iree_async_proactor_pool_release(proactors_);
    NativeInstance::current = nullptr;
  }

  iree_status_t CreateDriver() {
    return iree_hal_amd_driver_create(AMDF_NATIVE_LIFETIME_PROCESS,
                                      iree_allocator_system(), &driver_);
  }

  std::vector<iree_hal_device_id_t> Enumerate(iree_string_view_t scheme) {
    iree_host_size_t count = 0;
    iree_hal_device_info_t* infos = nullptr;
    IREE_EXPECT_OK(iree_hal_driver_query_available_devices(
        driver_, scheme, iree_allocator_system(), &count, &infos));
    std::vector<iree_hal_device_id_t> ids;
    for (iree_host_size_t i = 0; i < count; ++i) {
      ids.push_back(infos[i].device_id);
    }
    iree_allocator_free(iree_allocator_system(), infos);
    return ids;
  }

  void CreatePath(iree_string_view_t path, iree_status_code_t expected_status,
                  uint64_t expected_identity,
                  iree_string_view_t scheme = IREE_SV("rdna")) {
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    iree_hal_device_t* device = nullptr;
    const uint32_t before = NativeInstance::current->activation_count;
    IREE_EXPECT_STATUS_IS(expected_status,
                          iree_hal_driver_create_device_by_path(
                              driver_, scheme, path, 0, nullptr, &params,
                              iree_allocator_system(), &device));
    EXPECT_EQ(device, nullptr);
    EXPECT_EQ(NativeInstance::current->activation_count,
              before + (expected_identity ? 1u : 0u));
    if (expected_identity) {
      EXPECT_EQ(NativeInstance::current->activated.words[0], expected_identity);
    }
    EXPECT_EQ(NativeInstance::current->open_count, 0u);
  }

  // Scheme selections are independent even when they name the same endpoints.
  ScopedEnvironment rdna_visibility_{"HAL_VISIBLE_RDNA_DEVICES"};
  // CDNA visibility is independent of RDNA and the combined GPU view.
  ScopedEnvironment cdna_visibility_{"HAL_VISIBLE_CDNA_DEVICES"};
  // Unified GPU selection shares the driver, not RDNA's visibility policy.
  ScopedEnvironment gpu_visibility_{"HAL_VISIBLE_AMDGPU2_DEVICES"};
  // NPU ordinals are independent of every GPU namespace.
  ScopedEnvironment xdna_visibility_{"HAL_VISIBLE_XDNA_DEVICES"};
  // Passive native facts configured before driver creation.
  NativeInstance native_;
  // Real HAL driver; only its native libamdf dependency is replaced.
  iree_hal_driver_t* driver_ = nullptr;
  // Real caller-owned service satisfying the public device creation contract.
  iree_async_proactor_pool_t* proactors_ = nullptr;
};

TEST_F(DriverTest, EnumerationQualifiesComputeBeforeAdvertisingEndpoints) {
  IREE_ASSERT_OK(CreateDriver());
  const uint32_t query_count = native_.query_count;
  iree_host_size_t count = 0;
  iree_hal_device_info_t* infos = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, IREE_SV("rdna"), iree_allocator_system(), &count, &infos));
  EXPECT_EQ(count, 2u);
  if (count == 2) {
    EXPECT_EQ(infos[0].device_id, 2u);
    EXPECT_EQ(infos[1].device_id, 4u);
    EXPECT_TRUE(iree_string_view_equal(
        infos[0].path, IREE_SV("id/00000000000000020000000000000000")));
    EXPECT_TRUE(iree_string_view_equal(
        infos[1].path, IREE_SV("id/00000000000000040000000000000000")));
  }
  iree_allocator_free(iree_allocator_system(), infos);
  EXPECT_EQ(NativeInstance::current->activation_count, 0u);
  EXPECT_EQ(NativeInstance::current->open_count, 0u);
  EXPECT_EQ(Enumerate(IREE_SV("amdgpu2")),
            (std::vector<iree_hal_device_id_t>{2, 4, 6}));
  EXPECT_EQ(Enumerate(IREE_SV("xdna")), (std::vector<iree_hal_device_id_t>{5}));
  EXPECT_EQ(Enumerate(IREE_SV("")),
            (std::vector<iree_hal_device_id_t>{2, 4, 5, 6}));
  EXPECT_EQ(Enumerate(IREE_SV("cdna")), (std::vector<iree_hal_device_id_t>{6}));
  EXPECT_EQ(native_.query_count, query_count);
  EXPECT_EQ(native_.extension_query_count, 1u);
}

TEST_F(DriverTest, DefaultOrdinalAndIdentityUseTheSameExecutableSet) {
  IREE_ASSERT_OK(CreateDriver());
  CreatePath(IREE_SV(""), IREE_STATUS_RESOURCE_EXHAUSTED, 2);
  CreatePath(IREE_SV("0"), IREE_STATUS_RESOURCE_EXHAUSTED, 2);
  CreatePath(IREE_SV("id/00000000000000020000000000000000"),
             IREE_STATUS_RESOURCE_EXHAUSTED, 2);
  CreatePath(IREE_SV("1"), IREE_STATUS_RESOURCE_EXHAUSTED, 4);
  CreatePath(IREE_SV("id/00000000000000040000000000000000"),
             IREE_STATUS_RESOURCE_EXHAUSTED, 4);
  CreatePath(IREE_SV("2"), IREE_STATUS_NOT_FOUND, 0);
  CreatePath(IREE_SV("id/00000000000000010000000000000000"),
             IREE_STATUS_UNAVAILABLE, 0);
  CreatePath(IREE_SV("id/00000000000000030000000000000000"),
             IREE_STATUS_UNAVAILABLE, 0);
  CreatePath(IREE_SV("id/00000000000000990000000000000000"),
             IREE_STATUS_NOT_FOUND, 0);
}

TEST_F(DriverTest, NumericIdsCannotBypassPassiveQualification) {
  IREE_ASSERT_OK(CreateDriver());
  auto params = iree_hal_device_create_params_default();
  params.proactor_pool = proactors_;
  for (iree_hal_device_id_t id = 1; id <= 4; ++id) {
    iree_hal_device_t* device = nullptr;
    const bool supported = id == 2 || id == 4;
    const uint32_t before = NativeInstance::current->activation_count;
    IREE_EXPECT_STATUS_IS(
        supported ? IREE_STATUS_RESOURCE_EXHAUSTED : IREE_STATUS_UNAVAILABLE,
        iree_hal_driver_create_device_by_id(driver_, id, 0, nullptr, &params,
                                            iree_allocator_system(), &device));
    EXPECT_EQ(device, nullptr);
    EXPECT_EQ(NativeInstance::current->activation_count,
              before + (supported ? 1u : 0u));
    if (supported) {
      EXPECT_EQ(NativeInstance::current->activated.words[0], id);
    }
    EXPECT_EQ(NativeInstance::current->open_count, 0u);
  }
}

TEST_F(DriverTest, CdnaQualifiesNativeProtocolsIndependentlyOfArchitecture) {
  auto& endpoints = NativeInstance::current->endpoints;
  for (auto& endpoint : endpoints) {
    if (endpoint.summary.engine_kind == AMDF_ENGINE_KIND_GPU) {
      endpoint.gpu.gfx_ip = {9, 4, 2};
    }
  }
  const amdf_queue_family_info_t aql = endpoints[5].families[0];
  endpoints[0].families = {aql};
  endpoints[0].families[0].user_queue_capabilities =
      AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  endpoints[1].families = {aql};
  IREE_ASSERT_OK(CreateDriver());

  EXPECT_EQ(Enumerate(IREE_SV("cdna")),
            (std::vector<iree_hal_device_id_t>{2, 4, 6}));
  EXPECT_EQ(NativeInstance::current->activation_count, 0u);
  EXPECT_EQ(NativeInstance::current->open_count, 0u);
  CreatePath(IREE_SV(""), IREE_STATUS_RESOURCE_EXHAUSTED, 2, IREE_SV("cdna"));
  CreatePath(IREE_SV("1"), IREE_STATUS_RESOURCE_EXHAUSTED, 4, IREE_SV("cdna"));
  CreatePath(IREE_SV("id/00000000000000010000000000000000"),
             IREE_STATUS_UNAVAILABLE, 0, IREE_SV("cdna"));
  CreatePath(IREE_SV("id/00000000000000020000000000000000"),
             IREE_STATUS_RESOURCE_EXHAUSTED, 2, IREE_SV("cdna"));
  CreatePath(IREE_SV("0"), IREE_STATUS_NOT_FOUND, 0);
}

TEST_F(DriverTest, NativeQueryFailureIsNotAnUnsupportedEndpoint) {
  native_.query_failure =
      amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, CreateDriver());
  EXPECT_EQ(driver_, nullptr);
  EXPECT_EQ(native_.open_count, 0u);
  EXPECT_EQ(native_.activation_count, 0u);
}

TEST_F(DriverTest, SchemeViewsReorderTheirOwnUnmaskedOrdinalsIndependently) {
  rdna_visibility_.Set("1, 0");
  gpu_visibility_.Set("0");
  IREE_ASSERT_OK(CreateDriver());
  EXPECT_EQ(Enumerate(IREE_SV("rdna")),
            (std::vector<iree_hal_device_id_t>{4, 2}));
  EXPECT_EQ(Enumerate(IREE_SV("amdgpu2")),
            (std::vector<iree_hal_device_id_t>{2}));
  EXPECT_EQ(Enumerate(IREE_SV("xdna")), (std::vector<iree_hal_device_id_t>{5}));
  EXPECT_EQ(Enumerate(IREE_SV("")),
            (std::vector<iree_hal_device_id_t>{2, 4, 5, 6}));
  CreatePath(IREE_SV(""), IREE_STATUS_RESOURCE_EXHAUSTED, 4);
  CreatePath(IREE_SV("1"), IREE_STATUS_RESOURCE_EXHAUSTED, 2);
  CreatePath(IREE_SV("0"), IREE_STATUS_RESOURCE_EXHAUSTED, 2,
             IREE_SV("amdgpu2"));
  CreatePath(IREE_SV("1"), IREE_STATUS_NOT_FOUND, 0, IREE_SV("amdgpu2"));
}

TEST_F(DriverTest, VisibilityIsCapturedOnceAndIdentityPathsRespectIt) {
  rdna_visibility_.Set("1");
  IREE_ASSERT_OK(CreateDriver());
  rdna_visibility_.Set("0");
  EXPECT_EQ(Enumerate(IREE_SV("rdna")), (std::vector<iree_hal_device_id_t>{4}));
  CreatePath(IREE_SV("0"), IREE_STATUS_RESOURCE_EXHAUSTED, 4);
  CreatePath(IREE_SV("id/00000000000000040000000000000000"),
             IREE_STATUS_RESOURCE_EXHAUSTED, 4);
  CreatePath(IREE_SV("id/00000000000000020000000000000000"),
             IREE_STATUS_UNAVAILABLE, 0);
  CreatePath(IREE_SV("id/00000000000000020000000000000000"),
             IREE_STATUS_RESOURCE_EXHAUSTED, 2, IREE_SV("amdgpu2"));
}

TEST_F(DriverTest, UnifiedGpuSelectionIncludesCdnaIndependently) {
  gpu_visibility_.Set("2,1");
  rdna_visibility_.Set("0");
  cdna_visibility_.Set(" \t");
  IREE_ASSERT_OK(CreateDriver());
  EXPECT_EQ(Enumerate(IREE_SV("amdgpu2")),
            (std::vector<iree_hal_device_id_t>{6, 4}));
  EXPECT_EQ(Enumerate(IREE_SV("rdna")), (std::vector<iree_hal_device_id_t>{2}));
  EXPECT_TRUE(Enumerate(IREE_SV("cdna")).empty());
  EXPECT_EQ(Enumerate(IREE_SV("xdna")), (std::vector<iree_hal_device_id_t>{5}));
  CreatePath(IREE_SV("0"), IREE_STATUS_RESOURCE_EXHAUSTED, 6,
             IREE_SV("amdgpu2"));
  CreatePath(IREE_SV("1"), IREE_STATUS_RESOURCE_EXHAUSTED, 4,
             IREE_SV("amdgpu2"));
  CreatePath(IREE_SV("id/00000000000000060000000000000000"),
             IREE_STATUS_UNAVAILABLE, 0, IREE_SV("cdna"));
  CreatePath(IREE_SV("id/00000000000000060000000000000000"),
             IREE_STATUS_RESOURCE_EXHAUSTED, 6, IREE_SV("amdgpu2"));
}

TEST_F(DriverTest, EmptyVisibilityHidesOnlyItsScheme) {
  rdna_visibility_.Set(" \t");
  xdna_visibility_.Set(" \t");
  IREE_ASSERT_OK(CreateDriver());
  EXPECT_TRUE(Enumerate(IREE_SV("rdna")).empty());
  EXPECT_TRUE(Enumerate(IREE_SV("xdna")).empty());
  EXPECT_EQ(Enumerate(IREE_SV("amdgpu2")),
            (std::vector<iree_hal_device_id_t>{2, 4, 6}));
  EXPECT_EQ(Enumerate(IREE_SV("")), (std::vector<iree_hal_device_id_t>{2, 4, 6}));
  CreatePath(IREE_SV(""), IREE_STATUS_NOT_FOUND, 0);
  CreatePath(IREE_SV(""), IREE_STATUS_NOT_FOUND, 0, IREE_SV("xdna"));
  CreatePath(IREE_SV("id/00000000000000040000000000000000"),
             IREE_STATUS_UNAVAILABLE, 0);

  // Opaque IDs from another view name physical endpoints, not scheme ordinals.
  auto params = iree_hal_device_create_params_default();
  params.proactor_pool = proactors_;
  iree_hal_device_t* device = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_driver_create_device_by_id(driver_, 4, 0, nullptr, &params,
                                          iree_allocator_system(), &device));
  EXPECT_EQ(device, nullptr);
  EXPECT_EQ(native_.activated.words[0], 4u);
}

TEST_F(DriverTest, NonemptyVisibilityRequiresAnAvailableEndpoint) {
  for (auto& endpoint : native_.endpoints) {
    endpoint.families.clear();
  }
  gpu_visibility_.Set("0");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, CreateDriver());
  EXPECT_EQ(driver_, nullptr);
  EXPECT_EQ(native_.open_count, 0u);
  EXPECT_EQ(native_.activation_count, 0u);
}

TEST_F(DriverTest, GenericOrdinalCreationUsesTheVisibleSchemeOrder) {
  gpu_visibility_.Set("1,0");
  IREE_ASSERT_OK(CreateDriver());
  auto params = iree_hal_device_create_params_default();
  params.proactor_pool = proactors_;
  iree_hal_device_t* device = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_driver_create_device_by_ordinal(
                            driver_, IREE_SV("amdgpu2"), 1, 0, nullptr, &params,
                            iree_allocator_system(), &device));
  EXPECT_EQ(device, nullptr);
  EXPECT_EQ(native_.activated.words[0], 2u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_driver_create_device_by_ordinal(
                            driver_, IREE_SV(""), 1, 0, nullptr, &params,
                            iree_allocator_system(), &device));
  EXPECT_EQ(device, nullptr);
  EXPECT_EQ(native_.activated.words[0], 4u);
}

class InvalidVisibilityTest
    : public DriverTest,
      public ::testing::WithParamInterface<const char*> {};

TEST_P(InvalidVisibilityTest, RejectsMalformedOrUnavailableOrdinals) {
  rdna_visibility_.Set(GetParam());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, CreateDriver());
  EXPECT_EQ(driver_, nullptr);
  EXPECT_EQ(native_.open_count, 0u);
  EXPECT_EQ(native_.activation_count, 0u);
}

INSTANTIATE_TEST_SUITE_P(Selection, InvalidVisibilityTest,
                         ::testing::Values("2", "0,0", "1,1", "-1", "+1", "0x1",
                                           "4294967296", "0,", ",0", "0,,1",
                                           "x", "0 1"));

TEST_F(DriverTest, RegistryAliasesShareOneNativeOwner) {
  iree_hal_driver_registry_t* registry = nullptr;
  IREE_ASSERT_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_ASSERT_OK(iree_hal_amd_driver_module_register(registry));
  IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
      registry, IREE_SV("rdna"), &driver_));
  iree_hal_driver_t* gpu = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
      registry, IREE_SV("amdgpu2"), &gpu));
  iree_hal_driver_t* xdna = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
      registry, IREE_SV("xdna"), &xdna));
  iree_hal_driver_t* cdna = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
      registry, IREE_SV("cdna"), &cdna));
  EXPECT_EQ(driver_, gpu);
  EXPECT_EQ(driver_, xdna);
  EXPECT_EQ(driver_, cdna);
  EXPECT_EQ(native_.instance_count, 1u);
  iree_hal_driver_registry_free(registry);
  iree_hal_driver_release(cdna);
  iree_hal_driver_release(xdna);
  iree_hal_driver_release(gpu);
  CreatePath(IREE_SV("0"), IREE_STATUS_RESOURCE_EXHAUSTED, 2,
             IREE_SV("amdgpu2"));
}

}  // namespace

// Resolve the driver's native dependency to the fixture without changing the
// shipping driver API or loading a real native device in this unit test.
extern "C" amdf_status_t AMDF_CALL amdf_query_api(amdf_abi_version_t minimum,
                                                  amdf_abi_version_t maximum,
                                                  const amdf_api_t** out_api) {
  EXPECT_LE(minimum, AMDF_ABI_VERSION_LATEST);
  EXPECT_GE(maximum, AMDF_ABI_VERSION_LATEST);
  *out_api = &kNativeApi;
  return AMDF_STATUS_OK;
}
