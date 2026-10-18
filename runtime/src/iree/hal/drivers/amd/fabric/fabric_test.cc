// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/fabric.h"

#include <atomic>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct NativeInstance {
  // Distinct instance-owned storage used as the native system scope identity.
  int system_scope = 0;
  // Native policy captured at the public construction boundary.
  amdf_native_lifetime_t native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
  // Number of instances destroyed by the dependency mock.
  static std::atomic<int> destroy_count;
};
std::atomic<int> NativeInstance::destroy_count = 0;

amdf_status_t AMDF_CALL CreateInstance(const amdf_instance_create_info_t* info,
                                       amdf_instance_t** out_instance) {
  auto* instance = new NativeInstance;
  instance->native_lifetime = info->native_lifetime;
  *out_instance = reinterpret_cast<amdf_instance_t*>(instance);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL DestroyInstance(amdf_instance_t* instance) {
  delete reinterpret_cast<NativeInstance*>(instance);
  ++NativeInstance::destroy_count;
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

amdf_status_t AMDF_CALL QueryScope(amdf_memory_scope_t* scope,
                                   amdf_memory_scope_info_t* out_info) {
  out_info->kind = AMDF_MEMORY_SCOPE_KIND_SYSTEM;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL EnumerateEndpoints(amdf_instance_t* instance,
                                           uint32_t capacity,
                                           amdf_endpoint_summary_t* summaries,
                                           uint32_t* out_count) {
  *out_count = 3;
  if (!capacity) {
    return AMDF_STATUS_OK;
  }
  if (capacity < 3) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
  }
  for (uint32_t i = 0; i < 3; ++i) {
    summaries[i] = {};
    summaries[i].id.words[0] = i + 1;
    summaries[i].engine_kind =
        i == 1 ? AMDF_ENGINE_KIND_XDNA : AMDF_ENGINE_KIND_GPU;
    std::strcpy(summaries[i].name, i == 1 ? "NPU" : "GPU");
  }
  return AMDF_STATUS_OK;
}

amdf_api_t NativeApi() {
  amdf_api_t api = {
      .instance_create = CreateInstance,
      .instance_destroy = DestroyInstance,
      .endpoint_enumerate = EnumerateEndpoints,
      .instance_enumerate_memory_scopes = EnumerateScopes,
      .memory_scope_query_info = QueryScope,
  };
  // No engine extension or device creation callbacks exist in passive
  // discovery.
  return api;
}

TEST(AmdFabricTest, PreservesAllEngineIdentitiesWithoutActivation) {
  auto api = NativeApi();
  iree_hal_amd_fabric_t* fabric = nullptr;
  const int before = NativeInstance::destroy_count;
  IREE_ASSERT_OK(iree_hal_amd_fabric_create(&api, AMDF_NATIVE_LIFETIME_PROCESS,
                                            iree_allocator_system(), &fabric));
  EXPECT_EQ(fabric->api, &api);
  EXPECT_NE(fabric->system_scope, nullptr);
  ASSERT_EQ(fabric->endpoint_count, 3u);
  EXPECT_EQ(fabric->endpoints[0].engine_kind, AMDF_ENGINE_KIND_GPU);
  EXPECT_EQ(fabric->endpoints[1].engine_kind, AMDF_ENGINE_KIND_XDNA);
  EXPECT_EQ(fabric->endpoints[2].engine_kind, AMDF_ENGINE_KIND_GPU);
  EXPECT_EQ(fabric->endpoints[2].id.words[0], 3u);
  iree_hal_amd_fabric_retain(fabric);
  iree_hal_amd_fabric_release(fabric);
  EXPECT_EQ(NativeInstance::destroy_count, before);
  EXPECT_EQ(fabric->endpoints[1].id.words[0], 2u);
  iree_hal_amd_fabric_release(fabric);
  EXPECT_EQ(NativeInstance::destroy_count, before + 1);
}

TEST(AmdFabricTest, ExplicitCreationIsolatesNativeInstancesAndScopes) {
  auto api = NativeApi();
  iree_hal_amd_fabric_t *first = nullptr, *second = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_fabric_create(&api, AMDF_NATIVE_LIFETIME_PROCESS,
                                            iree_allocator_system(), &first));
  IREE_ASSERT_OK(iree_hal_amd_fabric_create(&api, AMDF_NATIVE_LIFETIME_PROCESS,
                                            iree_allocator_system(), &second));
  EXPECT_NE(first->instance, second->instance);
  EXPECT_NE(first->system_scope, second->system_scope);
  iree_hal_amd_fabric_release(first);
  EXPECT_EQ(second->endpoint_count, 3u);
  iree_hal_amd_fabric_release(second);
}

TEST(AmdFabricTest, ForwardsExplicitNativeLifetimePolicy) {
  auto api = NativeApi();
  for (amdf_native_lifetime_t lifetime :
       {AMDF_NATIVE_LIFETIME_PROCESS, AMDF_NATIVE_LIFETIME_INSTANCE}) {
    iree_hal_amd_fabric_t* fabric = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_fabric_create(
        &api, lifetime, iree_allocator_system(), &fabric));
    EXPECT_EQ(
        reinterpret_cast<NativeInstance*>(fabric->instance)->native_lifetime,
        lifetime);
    iree_hal_amd_fabric_release(fabric);
  }
}

TEST(AmdFabricTest, EmptyDiscoveryRetainsTheNativeInstance) {
  auto api = NativeApi();
  api.endpoint_enumerate = +[](amdf_instance_t*, uint32_t,
                               amdf_endpoint_summary_t*, uint32_t* count) {
    *count = 0;
    return AMDF_STATUS_OK;
  };
  iree_hal_amd_fabric_t* fabric = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_fabric_create(&api, AMDF_NATIVE_LIFETIME_PROCESS,
                                            iree_allocator_system(), &fabric));
  EXPECT_EQ(fabric->endpoint_count, 0u);
  EXPECT_EQ(fabric->endpoints, nullptr);
  EXPECT_NE(fabric->system_scope, nullptr);
  iree_hal_amd_fabric_release(fabric);
}

TEST(AmdFabricTest, EnginesAcquireTheSameNodeAllocationQueue) {
  auto api = NativeApi();
  iree_hal_amd_fabric_t* fabric = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_fabric_create(&api, AMDF_NATIVE_LIFETIME_PROCESS,
                                            iree_allocator_system(), &fabric));
  iree_hal_amd_fabric_allocation_queue_t* first = nullptr;
  iree_hal_amd_fabric_allocation_queue_t* second = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_fabric_acquire_allocation_queue(fabric, UINT32_MAX, &first));
  IREE_ASSERT_OK(iree_hal_amd_fabric_acquire_allocation_queue(
      fabric, UINT32_MAX, &second));
  EXPECT_EQ(first, second);
  iree_hal_amd_fabric_allocation_queue_release(first);
  iree_hal_amd_fabric_release(fabric);
  iree_hal_amd_fabric_allocation_queue_release(second);
}

amdf_status_t AMDF_CALL FailEnumeration(amdf_instance_t* instance,
                                        uint32_t capacity,
                                        amdf_endpoint_summary_t* summaries,
                                        uint32_t* out_count) {
  return amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
}

TEST(AmdFabricTest, FailedDiscoveryReleasesTheUnpublishedOwner) {
  auto api = NativeApi();
  api.endpoint_enumerate = FailEnumeration;
  iree_hal_amd_fabric_t* fabric = nullptr;
  const int before = NativeInstance::destroy_count;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_amd_fabric_create(&api, AMDF_NATIVE_LIFETIME_PROCESS,
                                 iree_allocator_system(), &fabric));
  EXPECT_EQ(fabric, nullptr);
  EXPECT_EQ(NativeInstance::destroy_count, before + 1);
}

}  // namespace
