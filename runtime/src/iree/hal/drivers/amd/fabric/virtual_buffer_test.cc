// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/virtual_buffer.h"

#include <algorithm>
#include <future>

#include "iree/hal/drivers/amd/fabric/file_transfer.h"
#include "iree/hal/drivers/amd/fabric/memory_backend_test_base.h"
#include "iree/hal/utils/fd_file.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/testing/temp_file.h"

namespace iree::hal::amd::fabric {

class VirtualBufferContractTest : public MemoryBackendTest {
 protected:
  void SetUp() override {
    MemoryBackendTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
    native_.mapping_enabled = true;
    native_.host_reservation_enabled = true;
    native_.host_source_enabled = true;
    native_.host_alias_coherent = true;
    native_.host_mapping_features |= AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS;
    native_.required_flags |= AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
    native_.atomics[0] = {
        AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE |
            AMDF_ATOMIC_OPERATION_ADD,
        AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE};
    native_.atomic_reach[0] = {AMDF_ATOMIC_SCOPE_SYSTEM,
                               AMDF_ATOMIC_SCOPE_SYSTEM};
    binding_descriptions_[0] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[0] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_families_[0].host_usage = 0;
    accesses_[0].family = families_[0];
    accesses_[1].family = families_[1];
    accesses_[2].family = iree_hal_device_queue_family(cpu_device_, 0);
    for (auto& access : accesses_) {
      access.usage = IREE_HAL_BUFFER_USAGE_STORAGE;
    }
  }

  iree_hal_pool_scope_t Scope() {
    return {accesses_.size(),
            accesses_.data(),
            {IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_MAPPING_MODE_PERSISTENT}};
  }

  iree_status_t CreateSource(iree_hal_pool_t** out_pool) {
    const uint64_t mapping = UINT64_C(1)
                             << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
    return CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, out_pool, mapping,
                      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_STORAGE,
                      {}, mapping);
  }

  // CPU, GPU, and NPU consumers of each replaceable physical extent.
  std::array<iree_hal_pool_family_access_t, 3> accesses_ = {};
};

TEST_F(VirtualBufferContractTest,
       CopyProtocolSurvivesSourcesAndConstrainsAdmission) {
  constexpr uint32_t kCopyFamily = 97;
  native_families_[0].copy_family_ordinal = kCopyFamily;
  native_.copy_family_ordinals[0] = kCopyFamily;
  iree_hal_pool_t* coherent = nullptr;
  IREE_ASSERT_OK(CreateSource(&coherent));
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, Scope(), {1, &coherent},
                                                &options, allocator_, &target));
  const auto gpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto* protocols =
      static_cast<const iree_hal_amd_fabric_memory_copy_protocol_t*>(
          target->memory.contract->backend_data);
  ASSERT_NE(protocols, nullptr);
  EXPECT_EQ(protocols[gpu.id].begin.release.bits, 0u);
  EXPECT_EQ(protocols[gpu.id].end.acquire.bits, 0u);
  native_.api.memory_scope_query_pair_info =
      [](amdf_memory_scope_t* scope,
         const amdf_memory_profile_pair_query_t* query,
         amdf_memory_pair_info_t* info) -> amdf_status_t {
    const auto status =
        FakeNative::MemoryScopeQueryPairInfo(scope, query, info);
    if (amdf_status_is_ok(status) &&
        query->producer.kind == AMDF_MEMORY_SITE_KIND_DEVICE &&
        query->consumer.kind == AMDF_MEMORY_SITE_KIND_DEVICE &&
        query->consumer.value.device.queue_family_ordinal == kCopyFamily) {
      info->release = {
          .kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
          .executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
          .operation = AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
      };
    }
    return status;
  };
  iree_hal_pool_t* cached = nullptr;
  IREE_ASSERT_OK(CreateSource(&cached));
  EXPECT_FALSE(
      iree_hal_virtual_buffer_query_pool_compatibility(target, gpu, cached));
  iree_hal_pool_t* sources[] = {coherent, cached};
  iree_hal_buffer_t* joined = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, Scope(), {2, sources},
                                                &options, allocator_, &joined));
  for (auto* source : sources) {
    EXPECT_TRUE(
        iree_hal_virtual_buffer_query_pool_compatibility(joined, gpu, source));
    iree_hal_pool_release(source);
  }
  const int queries = native_.pair_query_count;
  const auto* joined_protocols =
      static_cast<const iree_hal_amd_fabric_memory_copy_protocol_t*>(
          joined->memory.contract->backend_data);
  EXPECT_EQ(joined_protocols[gpu.id].begin.release.bits,
            IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM);
  EXPECT_EQ(protocols[gpu.id].begin.release.bits, 0u);
  EXPECT_EQ(native_.pair_query_count, queries);
  EXPECT_EQ(native_.create_count, 0);
  iree_hal_buffer_release(joined);
  iree_hal_buffer_release(target);
}

TEST_F(VirtualBufferContractTest, AtomicsIntersectSourcesAtEachExecutionSite) {
  std::array<iree_hal_pool_t*, 2> sources = {};
  IREE_ASSERT_OK(CreateSource(&sources[0]));
  native_.atomics[0][0] &= ~AMDF_ATOMIC_OPERATION_ADD;
  IREE_ASSERT_OK(CreateSource(&sources[1]));
  const iree_hal_virtual_buffer_sources_t declaration = {sources.size(),
                                                         sources.data()};
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, Scope(), declaration,
                                                &options, allocator_, &target));
  const auto gpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto npu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[1]);
  const auto cpu =
      ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, accesses_[2].family);
  const auto host = ResolveScope(IREE_HAL_MEMORY_SITE_HOST);
  const auto gpu_operations =
      iree_hal_buffer_query_atomic_operations(target, gpu);
  EXPECT_EQ(gpu_operations.system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_buffer_query_atomic_operations(target, npu)));
  for (auto site : {cpu, host}) {
    EXPECT_EQ(
        iree_hal_buffer_query_atomic_operations(target, site).system_scope_32,
        IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  }
  for (auto* source : sources) {
    EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
        target,
        {iree_hal_device_group_memory_domain(group_),
         IREE_HAL_MEMORY_SCOPE_ANY},
        source));
    iree_hal_pool_release(source);
  }
  // The declaration borrows pools. A later pool with the same guarantees is
  // compatible after both original pool identities have been destroyed.
  iree_hal_pool_t* replacement = nullptr;
  IREE_ASSERT_OK(CreateSource(&replacement));
  EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
      target,
      {iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY},
      replacement));
  EXPECT_EQ(
      iree_hal_buffer_query_atomic_operations(target, gpu).system_scope_32,
      gpu_operations.system_scope_32);
  EXPECT_EQ(native_.create_count, 0);
  iree_hal_pool_release(replacement);
  iree_hal_buffer_release(target);
}

TEST_F(VirtualBufferContractTest, PartialSourceMapsOnlyItsNativeProjection) {
  native_.alias_projections = false;
  iree_hal_pool_t* shared_source = nullptr;
  IREE_ASSERT_OK(CreateSource(&shared_source));
  native_.host_source_enabled = false;
  native_.required_flags = 0;
  const iree_hal_pool_family_access_t gpu_family = {
      .family = families_[0],
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
  };
  iree_hal_pool_t* device_source = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, {1, &gpu_family, {}},
                                           &pool_options_, allocator_,
                                           &device_source));
  iree_hal_pool_t* sources[] = {device_source, shared_source};
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, Scope(), {IREE_ARRAYSIZE(sources), sources}, allocator_,
      &capabilities));
  ASSERT_NE(capabilities.max_byte_length, 0u);
  EXPECT_EQ(native_.create_count, 0);
  EXPECT_EQ(native_.reservation_count, 0);
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 8192};
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, Scope(), {IREE_ARRAYSIZE(sources), sources}, &options, allocator_,
      &target));
  const auto gpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto host = ResolveScope(IREE_HAL_MEMORY_SITE_HOST);
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};
  EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(target, gpu,
                                                               device_source));
  EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(target, host,
                                                                device_source));
  EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(target, all,
                                                                device_source));
  EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(target, all,
                                                               shared_source));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      device_source, iree_hal_make_pool_reservation_request(4096, 4096),
      iree_infinite_timeout(), &backing));
  iree_hal_memory_mapping_t mapping = {
      .target = target,
      .scope = host,
      .target_range = {0, 4096},
      .source = backing,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  auto* queue = iree_hal_device_queue(cpu_device_, 0, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_queue_map(queue, {}, {}, 1, &mapping, 0));
  EXPECT_EQ(native_.update_count, 0);
  mapping.scope = gpu;
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done));
  uint64_t value = 1;
  IREE_ASSERT_OK(
      iree_hal_queue_map(queue, {}, {1, &done, &value}, 1, &mapping, 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count, 1);
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, all));
  iree_hal_semaphore_release(done);
  iree_hal_buffer_release(target);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(device_source);
  iree_hal_pool_release(shared_source);
}

TEST_F(VirtualBufferContractTest, QueueShaderContractCoversEveryReplacement) {
  native_families_[0].program_site_kind = AMDF_MEMORY_SITE_KIND_PROGRAM;
  native_families_[0].program_atomics = true;
  native_.program_atomics[0] = {
      .system_scope_32 = AMDF_ATOMIC_OPERATION_LOAD |
                         AMDF_ATOMIC_OPERATION_STORE |
                         AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE,
  };
  std::array<iree_hal_pool_t*, 2> sources = {};
  IREE_ASSERT_OK(CreateSource(&sources[0]));
  native_.program_atomics[0].system_scope_32 &=
      ~AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE;
  IREE_ASSERT_OK(CreateSource(&sources[1]));
  const auto queue = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  // Both physical sources provide CP ADD, but only the first can use the
  // shader ADD recipe fixed by a recording for the replaceable range.
  for (auto* source : sources) {
    EXPECT_TRUE(iree_all_bits_set(
        iree_hal_pool_query_atomic_operations(source, queue).system_scope_32,
        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD));
  }
  const iree_hal_virtual_buffer_sources_t declaration = {sources.size(),
                                                         sources.data()};
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, Scope(), declaration,
                                                &options, allocator_, &target));
  EXPECT_EQ(
      iree_hal_buffer_query_atomic_operations(target, queue).system_scope_32,
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
          IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  iree_hal_buffer_release(target);
  const auto reservations = native_.reservation_count;
  accesses_[0].atomics.queue.system_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, Scope(), declaration, allocator_, &capabilities));
  EXPECT_EQ(capabilities.max_byte_length, 0u);
  EXPECT_EQ(native_.reservation_count, reservations);
  for (auto* source : sources) {
    iree_hal_pool_release(source);
  }
}

TEST_F(VirtualBufferContractTest, PairGuaranteesFollowTheVirtualShaderRecipe) {
  native_families_[0].program_site_kind = AMDF_MEMORY_SITE_KIND_PROGRAM;
  native_families_[0].program_atomics = true;
  native_.atomics[0] = {AMDF_ATOMIC_OPERATION_STORE, 0};
  const auto transactions = AMDF_ATOMIC_OPERATION_LOAD |
                            AMDF_ATOMIC_OPERATION_STORE |
                            AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE;
  native_.program_atomics[0] = {
      .device_scope_32 = transactions,
      .system_scope_32 = transactions,
  };
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(&source));
  const auto queue = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  EXPECT_EQ(PairInfo(source->memory_contract, queue, queue)
                .producer_atomics.device_scope_32,
            0u);
  const auto* alias =
      iree_hal_memory_contract_mapping_alias(source->memory_contract);
  EXPECT_EQ(PairInfo(alias, queue, queue).producer_atomics.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, Scope(), {1, &source},
                                                &options, allocator_, &target));
  const auto* contract = iree_hal_buffer_memory_view(target).contract;
  const auto pair = PairInfo(contract, queue, queue);
  EXPECT_EQ(pair.producer_atomics.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(pair.consumer_atomics.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
      target,
      {iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY},
      source));

  // A new source keeps identical one-site support and cache actions, but loses
  // only the exact peer guarantee. It cannot replace this declared backing.
  native_.api.memory_scope_query_pair_info =
      [](amdf_memory_scope_t* scope,
         const amdf_memory_profile_pair_query_t* query,
         amdf_memory_pair_info_t* info) -> amdf_status_t {
    auto status = FakeNative::MemoryScopeQueryPairInfo(scope, query, info);
    if (amdf_status_is_ok(status) && info->next &&
        query->producer.kind == AMDF_MEMORY_SITE_KIND_PROGRAM &&
        query->consumer.kind == AMDF_MEMORY_SITE_KIND_PROGRAM) {
      auto* pair = static_cast<amdf_memory_pair_atomics_t*>(info->next);
      pair->producer.device_scope_32 &= ~AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE;
    }
    return status;
  };
  iree_hal_pool_t* replacement = nullptr;
  IREE_ASSERT_OK(CreateSource(&replacement));
  const int queries = native_.pair_query_count;
  EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
      target,
      {iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY},
      replacement));
  EXPECT_EQ(PairInfo(contract, queue, queue).producer_atomics.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(native_.pair_query_count, queries);
  EXPECT_EQ(native_.create_count, 0);
  iree_hal_pool_release(replacement);
  iree_hal_pool_release(source);
  iree_hal_buffer_release(target);
}

TEST_F(VirtualBufferContractTest, HostCoherencePreservesDeviceSideBarrier) {
  native_.api.memory_scope_query_pair_info =
      [](amdf_memory_scope_t* scope,
         const amdf_memory_profile_pair_query_t* query,
         amdf_memory_pair_info_t* info) -> amdf_status_t {
    const amdf_status_t status =
        FakeNative::MemoryScopeQueryPairInfo(scope, query, info);
    if (amdf_status_is_ok(status) &&
        query->producer.kind == AMDF_MEMORY_SITE_KIND_HOST &&
        query->consumer.kind == AMDF_MEMORY_SITE_KIND_DEVICE) {
      info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL;
      info->acquire.executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE;
      info->acquire.operation = AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM;
    }
    return status;
  };
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(&source));
  accesses_[0].requirements = IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST;
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, Scope(), {1, &source},
                                                &options, allocator_, &target));
  const auto host = ResolveScope(IREE_HAL_MEMORY_SITE_HOST);
  const auto gpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto transition = iree_hal_buffer_query_transition(target, host, gpu);
  EXPECT_TRUE(iree_hal_memory_effects_is_empty(transition.release));
  EXPECT_EQ(transition.acquire.bits,
            IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM);
  iree_hal_buffer_release(target);
  iree_hal_pool_release(source);
}

TEST_F(VirtualBufferContractTest, ZeroComputeRequirementRejectsShaderAliases) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(&source));
  accesses_[0].atomics.queue_zero_compute.device_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, Scope(), {1, &source}, allocator_, &capabilities));
  EXPECT_EQ(capabilities.max_byte_length, 0u);
  EXPECT_EQ(native_.reservation_count, 0);
  EXPECT_EQ(native_.create_count, 0);
  iree_hal_pool_release(source);
}

TEST_F(VirtualBufferContractTest, MissingRequiredAtomicRejectsBeforeAddresses) {
  native_.atomics[0][0] &= ~AMDF_ATOMIC_OPERATION_ADD;
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(&source));
  accesses_[0].atomics.queue.system_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, Scope(), {1, &source}, allocator_, &capabilities));
  EXPECT_EQ(capabilities.max_byte_length, 0u);
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* target = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_virtual_buffer_create(group_, Scope(), {1, &source}, &options,
                                     allocator_, &target));
  EXPECT_EQ(target, nullptr);
  EXPECT_EQ(native_.reservation_count, 0);
  EXPECT_EQ(native_.create_count, 0);
  iree_hal_pool_release(source);
}

TEST_F(VirtualBufferContractTest,
       NewCacheObligationRejectsBeforeNativeMutation) {
  iree_hal_pool_t* coherent = nullptr;
  IREE_ASSERT_OK(CreateSource(&coherent));
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, Scope(), {1, &coherent},
                                                &options, allocator_, &target));
  iree_hal_pool_release(coherent);
  const auto host = ResolveScope(IREE_HAL_MEMORY_SITE_HOST);
  const auto gpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto before = iree_hal_buffer_query_transition(target, host, gpu);
  EXPECT_TRUE(iree_hal_memory_effects_is_empty(before.release));
  native_.host_alias_coherent = false;
  iree_hal_pool_t* noncoherent = nullptr;
  IREE_ASSERT_OK(CreateSource(&noncoherent));
  EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
      target,
      {iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY},
      noncoherent));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      noncoherent, iree_hal_make_pool_reservation_request(4096, 4096),
      iree_infinite_timeout(), &backing));
  const iree_hal_memory_mapping_t mapping = {
      .target = target,
      .scope = gpu,
      .target_range = {0, 4096},
      .source = backing,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_queue_map(iree_hal_device_queue(cpu_device_, 0, 0), {}, {}, 1,
                         &mapping, 0));
  EXPECT_EQ(native_.update_count, 0);
  const auto after = iree_hal_buffer_query_transition(target, host, gpu);
  EXPECT_EQ(after.release.bits, before.release.bits);
  EXPECT_EQ(after.acquire.bits, before.acquire.bits);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(noncoherent);
  iree_hal_buffer_release(target);
}

TEST_F(VirtualBufferContractTest, DeclaredCacheObligationsCoverEverySource) {
  std::array<iree_hal_pool_t*, 2> sources = {};
  IREE_ASSERT_OK(CreateSource(&sources[0]));
  native_.host_alias_coherent = false;
  IREE_ASSERT_OK(CreateSource(&sources[1]));
  const auto host = ResolveScope(IREE_HAL_MEMORY_SITE_HOST);
  const auto gpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  for (int order = 0; order < 2; ++order) {
    iree_hal_buffer_t* target = nullptr;
    IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
        group_, Scope(), {sources.size(), sources.data()}, &options, allocator_,
        &target));
    iree_hal_memory_transition_pair_t pair;
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        {target->memory.contract}, host, gpu,
        IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
    const auto info =
        iree_hal_memory_transition_query_info({target->memory.contract}, pair);
    EXPECT_EQ(info.release.operation,
              IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH);
    EXPECT_EQ(info.release.executor,
              IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API);
    for (auto* source : sources) {
      EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
          target,
          {iree_hal_device_group_memory_domain(group_),
           IREE_HAL_MEMORY_SCOPE_ANY},
          source));
    }
    iree_hal_buffer_release(target);
    std::swap(sources[0], sources[1]);
  }
  EXPECT_EQ(native_.create_count, 0);
  for (auto* source : sources) {
    iree_hal_pool_release(source);
  }
}

TEST_F(MemoryBackendTest, SourceCapturesReservationAliasProtocolBeforePayload) {
  native_.mapping_enabled = true;
  native_.host_reservation_enabled = true;
  native_.host_source_enabled = true;
  native_.host_alias_coherent = true;
  native_.required_flags |= AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
  const uint64_t mapping_source = UINT64_C(1)
                                  << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                            mapping_source, IREE_HAL_MEMORY_ACCESS_ALL,
                            IREE_HAL_BUFFER_USAGE_STORAGE, {}, mapping_source));
  const auto* alias =
      iree_hal_memory_contract_mapping_alias(pool->memory_contract);
  ASSERT_NE(alias, pool->memory_contract);
  const auto gpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto host = ResolveScope(IREE_HAL_MEMORY_SITE_HOST);
  const auto cpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE,
                                iree_hal_device_queue_family(cpu_device_, 0));
  const int query_count = native_.pair_query_count;
  for (auto producer : {host, cpu}) {
    iree_hal_memory_transition_pair_t pair;
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        {alias}, producer, gpu, IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
    EXPECT_EQ(iree_hal_memory_transition_query({alias}, pair).release.bits, 0u);
    EXPECT_NE(iree_hal_pool_query_transition(pool, producer, gpu).release.bits,
              0u);
  }
  EXPECT_EQ(native_.pair_query_count, query_count);
  EXPECT_EQ(native_.create_count, 0);
  EXPECT_EQ(native_.reservation_count, 0);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, RejectsUnavailableNativeNpuReservations) {
  const iree_hal_pool_family_access_t access = {
      .family = families_[0],
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
  };
  const iree_hal_pool_scope_t scope = {1, &access, {}};
  iree_hal_pool_t* source_pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &source_pool));
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, allocator_, &capabilities));
  EXPECT_EQ(capabilities.max_byte_length, 0u);
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_virtual_buffer_create(group_, scope, {1, &source_pool}, &options,
                                     allocator_, &buffer));
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(native_.reservation_count, 0);
  iree_hal_pool_release(source_pool);
}

TEST_F(MemoryBackendTest, QueriesReservationGeometryBeforeAcquiringAddresses) {
  native_.mapping_enabled = true;
  native_.reservation_granularities = {4096, 65536};
  std::array<iree_hal_pool_family_access_t, 2> accesses = {};
  for (size_t i = 0; i < accesses.size(); ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_families_[i].host_usage = 0;
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    accesses[i].family = families_[i];
    accesses[i].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  }
  native_.required_flags = 0;
  for (auto& access : accesses) {
    access.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  const iree_hal_pool_scope_t source_scope = {
      accesses.size(), accesses.data(), {}};
  iree_hal_pool_t* source_pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, source_scope, &pool_options_,
                                           allocator_, &source_pool));
  for (auto& access : accesses) {
    access.interfaces = 0;
  }
  iree_hal_pool_scope_t scope = {accesses.size(), accesses.data(), {}};
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, allocator_, &capabilities));
  EXPECT_EQ(capabilities.reservation_granularity, 65536u);
  EXPECT_EQ(capabilities.mapping_granularity, 65536u);
  EXPECT_EQ(capabilities.max_byte_length, UINT64_C(1) << 40);
  EXPECT_EQ(capabilities.max_alignment, UINT64_C(1) << 30);
  EXPECT_EQ(capabilities.supported_flags, 0u);
  EXPECT_EQ(native_.reservation_count, 0);
  EXPECT_EQ(native_.create_count, 0);

  iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* reservation = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_virtual_buffer_create(group_, scope, {1, &source_pool}, &options,
                                     allocator_, &reservation));
  EXPECT_EQ(reservation, nullptr);
  EXPECT_EQ(native_.reservation_count, 0);
  options.byte_length = capabilities.reservation_granularity;
  options.mapping_granularity = 4 * capabilities.mapping_granularity;
  options.flags = UINT32_C(0x80000000);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_virtual_buffer_create(group_, scope, {1, &source_pool}, &options,
                                     allocator_, &reservation));
  EXPECT_EQ(native_.reservation_count, 0);
  options.flags = IREE_HAL_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS;
  native_.reservation_flags[0] =
      AMDF_MEMORY_RESERVATION_FLAG_REQUIRE_COMMON_ADDRESS;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, allocator_, &capabilities));
  EXPECT_EQ(capabilities.supported_flags, 0u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_virtual_buffer_create(group_, scope, {1, &source_pool}, &options,
                                     allocator_, &reservation));
  EXPECT_EQ(reservation, nullptr);
  EXPECT_EQ(native_.reservation_count, 0);
  native_.reservation_flags[1] =
      AMDF_MEMORY_RESERVATION_FLAG_REQUIRE_COMMON_ADDRESS;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, allocator_, &capabilities));
  EXPECT_EQ(capabilities.supported_flags,
            IREE_HAL_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS);
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, scope, {1, &source_pool}, &options, allocator_, &reservation));
  iree_hal_virtual_buffer_info_t info;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_info(reservation, &info));
  EXPECT_EQ(info.mapping_granularity, options.mapping_granularity);
  EXPECT_EQ(native_.reservation_count, 1);
  EXPECT_EQ(native_.last_reservation_flags,
            AMDF_MEMORY_RESERVATION_FLAG_REQUIRE_COMMON_ADDRESS);
  EXPECT_EQ(native_.create_count, 0);
  iree_hal_buffer_release(reservation);

  // Adding host access makes this device-only native reservation inadmissible.
  // A capability query reports absence without manufacturing an error status.
  scope.host.access = IREE_HAL_MEMORY_ACCESS_READ;
  scope.host.modes = IREE_HAL_MAPPING_MODE_PERSISTENT;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, allocator_, &capabilities));
  EXPECT_EQ(capabilities.max_byte_length, 0u);
  EXPECT_EQ(capabilities.reservation_granularity, 0u);
  EXPECT_EQ(capabilities.mapping_granularity, 0u);
  EXPECT_EQ(capabilities.max_alignment, 0u);
  EXPECT_EQ(native_.reservation_count, 0);
  iree_hal_pool_release(source_pool);
}

class ReservationBackendTest : public MemoryBackendTest,
                               public ::testing::WithParamInterface<bool> {};

TEST_F(MemoryBackendTest, SharedTranslationsDoNotGrantSourceConsumerAccess) {
  native_.mapping_enabled = true;
  native_.alias_projections = true;
  native_.required_flags = 0;
  std::array<iree_hal_pool_family_access_t, 2> families = {};
  for (uint32_t i = 0; i < families.size(); ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_families_[i].host_usage = 0;
    families[i] = {
        .family = families_[i],
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
    };
  }
  iree_hal_pool_t* pool = nullptr;
  const iree_hal_pool_scope_t source_scope = {1, families.data(), {}};
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, source_scope, &pool_options_,
                                           allocator_, &pool));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(4096, 4096),
      iree_infinite_timeout(), &backing));
  iree_hal_pool_t* complete_source = nullptr;
  const iree_hal_pool_scope_t complete_scope = {
      families.size(), families.data(), {}};
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, complete_scope, &pool_options_, allocator_, &complete_source));
  for (auto& family : families) {
    family.interfaces = 0;
  }
  const iree_hal_pool_scope_t target_scope = {
      families.size(), families.data(), {}};
  const iree_hal_virtual_buffer_options_t options = {.byte_length = 4096};
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, target_scope,
                                                {1, &complete_source}, &options,
                                                allocator_, &target));
  const auto first = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const iree_hal_memory_mapping_t mapping = {
      .target = target,
      .scope = first,
      .target_range = {0, 4096},
      .source = backing,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done));
  uint64_t value = 1;
  EXPECT_FALSE(
      iree_hal_virtual_buffer_query_pool_compatibility(target, first, pool));
  EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
      target, first, complete_source));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_queue_map(iree_hal_device_queue(cpu_device_, 0, 0), {},
                         {1, &done, &value}, 1, &mapping, 0));
  EXPECT_EQ(native_.update_count, 0);
  iree_hal_pool_release(complete_source);
  iree_hal_semaphore_release(done);
  iree_hal_buffer_release(target);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, FileTransfersUseExplicitPhysicalSources) {
  native_.mapping_enabled = true;
  native_.host_mappable = false;
  native_.required_flags = 0;
  for (uint32_t i = 0; i < families_.size(); ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_families_[i].host_usage = 0;
  }
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_TRANSFER, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
                 IREE_HAL_MEMORY_ACCESS_NONE));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(16384, 4096),
      iree_infinite_timeout(), &backing));
  iree_hal_buffer_t* target = nullptr;
  IREE_ASSERT_OK(
      CreateReservation(&target, pool, IREE_HAL_BUFFER_USAGE_TRANSFER));
  iree::testing::TempFilePath path("iree_fabric_file_sources");
  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_create(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_ASYNC,
      path.path_view(), 16384, allocator_, &handle));
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get(proactor_pool_, 0, &proactor));
  std::array<iree_hal_file_t*, 2> files = {};
  iree_status_t status = iree_hal_fd_file_from_handle(
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle,
      proactor, allocator_, &files[0]);
  iree_io_file_handle_release(handle);
  IREE_ASSERT_OK(status);
  std::vector<uint8_t> contents(16384);
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(contents.data(), contents.size()), {}, allocator_,
      &handle));
  status = iree_hal_memory_file_wrap(
      nullptr, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      handle, IREE_HAL_MEMORY_FILE_FLAG_NONE, allocator_, &files[1]);
  iree_io_file_handle_release(handle);
  IREE_ASSERT_OK(status);

  for (auto* buffer : {backing, target}) {
    for (auto* file : files) {
      iree_hal_amd_fabric_file_transfer_t transfer = {};
      IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_initialize(
          file, IREE_HAL_MEMORY_ACCESS_READ, 0, buffer, 0, 16384, allocator_,
          &transfer));
      iree_hal_memory_effects_t before, after;
      IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare_source(
          &transfer, &native_.api, families_[0],
          native_families_[0].queue_family_ordinal, &before, &after));
      const bool physical =
          buffer == backing && iree_hal_file_async_handle(file);
      EXPECT_EQ(transfer.source.contents.data_length, physical ? 16384u : 0u);
      EXPECT_EQ(transfer.source.contents.data, nullptr);
      if (physical) {
        EXPECT_NE(transfer.source.storage, nullptr);
      }
      EXPECT_EQ(native_.map_count, 0);
      iree_hal_amd_fabric_file_transfer_deinitialize(&transfer);
    }
  }
  for (auto* file : files) {
    iree_hal_file_release(file);
  }
  EXPECT_TRUE(path.Remove());
  iree_hal_buffer_release(target);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
}

TEST_P(ReservationBackendTest, MapsExactConsumersThroughNativeProjections) {
  native_.mapping_enabled = true;
  native_.alias_projections = GetParam();
  native_.native_prefix = 4096;
  for (uint32_t i = 0; i < 2; ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_families_[i].host_usage = 0;
  }
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(32768, 4096),
      iree_infinite_timeout(), &backing));
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(backing, 8192, 16384, allocator_, &view));
  std::array<iree_hal_pool_family_access_t, 2> families = {};
  for (uint32_t i = 0; i < 2; ++i) {
    families[i].family = families_[1 - i];
    families[i].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  }
  const iree_hal_pool_scope_t scope = {families.size(), families.data(), {}};
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = 16384,
  };
  std::array<iree_hal_buffer_t*, 2> reservations = {};
  for (auto& reservation : reservations) {
    IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
        group_, scope, {1, &pool}, &options, allocator_, &reservation));
  }
  const auto first = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const auto second = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[1]);
  uint32_t first_projection, second_projection;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(
      reservations[0], first, &first_projection));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(
      reservations[0], second, &second_projection));
  EXPECT_EQ(first_projection == second_projection, native_.alias_projections);

  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &gate));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done));
  uint64_t value = 1;
  const iree_hal_semaphore_list_t waits = {1, &gate, &value};
  const iree_hal_semaphore_list_t signals = {1, &done, &value};
  std::array<iree_hal_memory_mapping_t, 2> mappings = {};
  for (uint32_t i = 0; i < 2; ++i) {
    mappings[i] = {
        .target = reservations[i],
        .scope = i ? second : first,
        .target_range = {4096, 8192},
        .source = view,
        .source_offset = 4096,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
    };
  }
  auto* queue = iree_hal_device_queue(cpu_device_, 0, 0);
  IREE_ASSERT_OK(iree_hal_queue_map(queue, waits, signals, mappings.size(),
                                    mappings.data(), 0));
  EXPECT_EQ(native_.update_count, 0);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count, 2);
  ASSERT_EQ(native_.last_updates.size(), 1u);
  // The slab prefix participates in native alignment, but libamdf consumes
  // logical source coordinates. Nested HAL view offsets are included once.
  EXPECT_EQ(native_.last_updates[0].source_byte_offset, 12288u);
  EXPECT_EQ(native_.last_updates[0].target_byte_offset, 4096u);
  EXPECT_EQ(native_.last_updates[0].byte_length, 8192u);
  // ANY expands distinct projections while deduplicating native aliases.
  mappings[0].scope = {first.domain, IREE_HAL_MEMORY_SCOPE_ANY};
  ++value;
  IREE_ASSERT_OK(iree_hal_queue_map(queue, {}, signals, 1, mappings.data(), 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count, native_.alias_projections ? 3 : 4);
  mappings[0].scope = first;

  // Distinct consumers of one page table still produce a single native vector.
  mappings[1].target = reservations[0];
  mappings[0].target_range = {0, 4096};
  mappings[1].target_range = {8192, 4096};
  const int previous_updates = native_.update_count;
  ++value;
  IREE_ASSERT_OK(iree_hal_queue_map(queue, {}, signals, mappings.size(),
                                    mappings.data(), 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count - previous_updates,
            native_.alias_projections ? 1 : 2);
  EXPECT_EQ(native_.last_updates.size(), native_.alias_projections ? 2u : 1u);
  mappings[0].target_range = mappings[1].target_range = {4096, 8192};

  // Conflicts are checked by shared projection, even across exact consumers.
  mappings[1].target = reservations[0];
  if (!native_.alias_projections) {
    mappings[1].scope = first;
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_map(queue, {}, {}, mappings.size(), mappings.data(), 0));
  for (uint32_t i = 0; i < 2; ++i) {
    mappings[i].target = reservations[i];
    mappings[i].source = nullptr;
    mappings[i].source_offset = 0;
    mappings[i].access = 0;
  }
  ++value;
  IREE_ASSERT_OK(iree_hal_queue_map(queue, {}, signals, mappings.size(),
                                    mappings.data(), 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.last_updates[0].memory, nullptr);
  EXPECT_EQ(native_.last_updates[0].access, 0u);
  for (auto* reservation : reservations) {
    iree_hal_buffer_release(reservation);
  }
  EXPECT_EQ(native_.reservation_count, 0);
  iree_hal_semaphore_release(gate);
  iree_hal_semaphore_release(done);
  iree_hal_buffer_release(view);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
}

INSTANTIATE_TEST_SUITE_P(ProjectionIdentity, ReservationBackendTest,
                         ::testing::Values(false, true));

TEST_F(MemoryBackendTest, NativeMappingFailureReachesQueueCompletion) {
  iree_hal_buffer_t* reservation = nullptr;
  IREE_ASSERT_OK(CreateReservation(&reservation));
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done));
  uint64_t value = 1;
  const iree_hal_semaphore_list_t signals = {1, &done, &value};
  const iree_hal_memory_mapping_t mapping = {
      .target = reservation,
      .scope = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]),
      .target_range = {0, 4096},
  };
  native_.update_failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_ASSERT_OK(iree_hal_queue_map(iree_hal_device_queue(cpu_device_, 0, 0),
                                    {}, signals, 1, &mapping, 0));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count, 1);
  EXPECT_EQ(native_.reservation_count, 1);
  iree_hal_buffer_release(reservation);
  EXPECT_EQ(native_.reservation_count, 0);
  iree_hal_semaphore_release(done);
}

// Exercises the real HAL capture/resolution/backend path. Only the native page
// table implementation is replaced; metadata, source admission, and publication
// are the production implementation.
class MappingPublicationTest : public MemoryBackendTest {
 protected:
  void SetUp() override {
    MemoryBackendTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
    native_.mapping_enabled = true;
    native_.alias_projections = true;
    for (size_t i = 0; i < families_.size(); ++i) {
      binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                  AMDF_MEMORY_ADDRESS_GPU};
      native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
      native_families_[i].host_usage = 0;
    }
    IREE_ASSERT_OK(
        CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool_,
                   UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE));
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(32768, 4096),
        iree_infinite_timeout(), &backing_));
    std::array<iree_hal_pool_family_access_t, 2> families = {};
    for (size_t i = 0; i < families.size(); ++i) {
      families[i].family = families_[i];
      families[i].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
    }
    const iree_hal_virtual_buffer_options_t options = {.byte_length = 16384};
    IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
        group_, {families.size(), families.data(), {}}, {1, &pool_}, &options,
        allocator_, &target_));
    scope_ = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  }

  void TearDown() override {
    iree_hal_buffer_release(target_);
    iree_hal_buffer_release(backing_);
    iree_hal_pool_release(pool_);
    MemoryBackendTest::TearDown();
  }

  iree_hal_memory_mapping_t Mapping(uint64_t position, uint64_t length,
                                    uint64_t source_offset = 0) {
    return {
        .target = target_,
        .scope = scope_,
        .target_range = {position, length},
        .source = backing_,
        .source_offset = source_offset,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE,
    };
  }

  iree_status_t Update(const std::vector<iree_hal_memory_mapping_t>& mappings) {
    iree_hal_host_call_t call;
    IREE_RETURN_IF_ERROR(iree_hal_memory_mapping_prepare(
        iree_hal_device_queue(cpu_device_, 0, 0), mappings.size(),
        mappings.data(), 0, allocator_, &call));
    const uint64_t args[4] = {};
    iree_status_t status = call.fn(call.user_data, args, nullptr);
    iree_hal_resource_release(call.resource);
    return status;
  }

  // Pool admitted for both native consumers.
  iree_hal_pool_t* pool_ = nullptr;
  // Physical source kept live until all native aliases have been destroyed.
  iree_hal_buffer_t* backing_ = nullptr;
  // Virtual buffer owning the native address reservation.
  iree_hal_buffer_t* target_ = nullptr;
  // First native consumer's exact program scope.
  iree_hal_memory_scope_t scope_ = {};
};

TEST_F(MappingPublicationTest, NativeRunsPreserveOriginalViewOwnership) {
  const auto backing_references =
      iree_atomic_ref_count_load(&backing_->resource.ref_count);
  const auto target_references =
      iree_atomic_ref_count_load(&target_->resource.ref_count);
  // Cancellation and successful resolution own the same captured views.
  for (bool resolve : {false, true}) {
    std::array<iree_hal_memory_mapping_t, 3> mappings;
    for (size_t i = 0; i < mappings.size(); ++i) {
      const uint64_t offset = (mappings.size() - i) * 4096;
      mappings[i] = Mapping(0, 4096);
      IREE_ASSERT_OK(iree_hal_buffer_subspan(target_, offset, 4096, allocator_,
                                             &mappings[i].target));
      IREE_ASSERT_OK(iree_hal_buffer_subspan(backing_, offset, 4096, allocator_,
                                             &mappings[i].source));
    }
    iree_hal_memory_mapping_batch_t* batch = nullptr;
    IREE_ASSERT_OK(iree_hal_memory_mapping_batch_create(
        iree_hal_device_queue(cpu_device_, 0, 0), mappings.size(),
        mappings.data(), 0, allocator_, &batch));
    EXPECT_EQ(batch->update_count, 3u);
    for (const auto& mapping : mappings) {
      iree_hal_buffer_release(mapping.target);
      iree_hal_buffer_release(mapping.source);
    }
    if (resolve) {
      // Resolution may be repeated without changing captured ownership or
      // reusing the previous compaction count as the original input count.
      for (int pass = 0; pass < 2; ++pass) {
        IREE_ASSERT_OK(iree_hal_memory_mapping_batch_resolve(batch));
        ASSERT_EQ(batch->entry_count, 1u);
        ASSERT_EQ(batch->update_count, 1u);
        ASSERT_EQ(batch->entries[0].update_count, 1u);
        const auto& update = batch->entries[0].updates[0];
        EXPECT_EQ(update.target_range.byte_offset, 4096u);
        EXPECT_EQ(update.target_range.byte_length, 12288u);
        EXPECT_EQ(update.source.mapping_source.offset, 4096u);
        EXPECT_NE(update.source.mapping_source.descriptor, nullptr);
      }
    }
    for (const auto& mapping : mappings) {
      EXPECT_EQ(iree_atomic_ref_count_load(&mapping.target->resource.ref_count),
                1);
      EXPECT_EQ(iree_atomic_ref_count_load(&mapping.source->resource.ref_count),
                1);
    }
    iree_hal_memory_mapping_batch_release(batch);
    EXPECT_EQ(iree_atomic_ref_count_load(&backing_->resource.ref_count),
              backing_references);
    EXPECT_EQ(iree_atomic_ref_count_load(&target_->resource.ref_count),
              target_references);
  }
}

TEST_F(MappingPublicationTest, CoalescesEachMappingOperation) {
  enum class SourceKind { kRemoved, kNoAccess, kReadWrite };
  for (auto operation : {IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE,
                         IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
                         IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND,
                         IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT}) {
    // Bound ranges can revoke all access without becoming logical removals.
    // REPLACE also admits an explicitly absent source, while UNBIND requires
    // it.
    for (auto source_kind : {SourceKind::kRemoved, SourceKind::kNoAccess,
                             SourceKind::kReadWrite}) {
      const bool removal = source_kind == SourceKind::kRemoved;
      if ((operation == IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND && !removal) ||
          ((operation == IREE_HAL_MEMORY_MAPPING_OPERATION_BIND ||
            operation == IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT) &&
           removal)) {
        continue;
      }
      std::array<iree_hal_memory_mapping_t, 2> mappings = {
          Mapping(0, 4096), Mapping(4096, 4096, 4096)};
      for (auto& mapping : mappings) {
        mapping.operation = operation;
        if (source_kind != SourceKind::kReadWrite) {
          mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
        }
        if (removal) {
          mapping.previous_source = mapping.source;
          mapping.previous_source_offset = mapping.source_offset;
          mapping.source = nullptr;
          mapping.source_offset = 0;
          mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
        }
      }
      iree_hal_memory_mapping_batch_t* batch = nullptr;
      IREE_ASSERT_OK(iree_hal_memory_mapping_batch_create(
          iree_hal_device_queue(cpu_device_, 0, 0), mappings.size(),
          mappings.data(), 0, allocator_, &batch));
      IREE_ASSERT_OK(iree_hal_memory_mapping_batch_resolve(batch));
      ASSERT_EQ(batch->entry_count, 1u);
      ASSERT_EQ(batch->update_count, 1u);
      ASSERT_EQ(batch->entries[0].update_count, 1u);
      const auto& update = batch->entries[0].updates[0];
      EXPECT_EQ(update.target_range.byte_offset, 0u);
      EXPECT_EQ(update.target_range.byte_length, 8192u);
      EXPECT_EQ(update.operation, operation);
      EXPECT_EQ(update.access, mappings[0].access);
      EXPECT_EQ(update.source.mapping_source.descriptor != nullptr,
                mappings[0].source != nullptr);
      iree_hal_memory_mapping_batch_release(batch);
    }
  }
}

TEST_F(MappingPublicationTest, PreservesNativeAndSemanticRunBoundaries) {
  iree_hal_buffer_t* other_backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(8192, 4096),
      iree_infinite_timeout(), &other_backing));
  iree_hal_buffer_t* other_target = nullptr;
  IREE_ASSERT_OK(CreateReservation(&other_target));
  const auto baseline = Mapping(4096, 4096, 4096);
  std::array<iree_hal_memory_mapping_t, 7> boundaries;
  boundaries.fill(baseline);
  boundaries[0].target_range.byte_offset = 8192;
  boundaries[1].source_offset = 8192;
  boundaries[2].source = other_backing;
  boundaries[3].access = IREE_HAL_MEMORY_ACCESS_READ;
  boundaries[4].operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND;
  boundaries[5].scope =
      ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[1]);
  boundaries[6].target = other_target;
  for (const auto& boundary : boundaries) {
    const iree_hal_memory_mapping_t mappings[] = {Mapping(0, 4096), boundary};
    iree_hal_memory_mapping_batch_t* batch = nullptr;
    IREE_ASSERT_OK(iree_hal_memory_mapping_batch_create(
        iree_hal_device_queue(cpu_device_, 0, 0), IREE_ARRAYSIZE(mappings),
        mappings, 0, allocator_, &batch));
    IREE_ASSERT_OK(iree_hal_memory_mapping_batch_resolve(batch));
    EXPECT_EQ(batch->update_count, 2u);
    EXPECT_EQ(batch->entry_count, boundary.target == target_ ? 1u : 2u);
    for (size_t e = 0; e < batch->entry_count; ++e) {
      for (size_t i = 0; i < batch->entries[e].update_count; ++i) {
        EXPECT_EQ(batch->entries[e].updates[i].target_range.byte_length, 4096u);
      }
    }
    iree_hal_memory_mapping_batch_release(batch);
  }
  iree_hal_buffer_release(other_target);
  iree_hal_buffer_release(other_backing);
}

TEST_F(MappingPublicationTest, CompactedEntriesKeepSeparateUpdateVectors) {
  iree_hal_buffer_t* other_target = nullptr;
  IREE_ASSERT_OK(CreateReservation(&other_target));
  auto other_first = Mapping(4096, 4096, 8192);
  auto other_second = Mapping(8192, 4096, 12288);
  other_first.target = other_second.target = other_target;
  const iree_hal_memory_mapping_t mappings[] = {
      Mapping(0, 4096), other_first, Mapping(4096, 4096, 4096), other_second};
  iree_hal_memory_mapping_batch_t* batch = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_mapping_batch_create(
      iree_hal_device_queue(cpu_device_, 0, 0), IREE_ARRAYSIZE(mappings),
      mappings, 0, allocator_, &batch));
  IREE_ASSERT_OK(iree_hal_memory_mapping_batch_resolve(batch));
  ASSERT_EQ(batch->entry_count, 2u);
  ASSERT_EQ(batch->update_count, 2u);
  for (size_t i = 0; i < batch->entry_count; ++i) {
    const auto& entry = batch->entries[i];
    ASSERT_EQ(entry.update_count, 1u);
    const bool first = &entry.target->buffer == target_;
    EXPECT_EQ(entry.updates[0].target_range.byte_offset, first ? 0u : 4096u);
    EXPECT_EQ(entry.updates[0].source.mapping_source.offset,
              first ? 0u : 8192u);
    EXPECT_EQ(entry.updates[0].target_range.byte_length, 8192u);
  }
  iree_hal_memory_mapping_batch_release(batch);
  iree_hal_buffer_release(other_target);
}

TEST_F(MappingPublicationTest, RejectsOverlapBeforePublishingCompactedPrefix) {
  iree_hal_buffer_t* alias = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(target_, 4096, 8192, allocator_, &alias));
  auto overlap = Mapping(0, 8192);
  overlap.target = alias;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Update({Mapping(0, 4096), Mapping(4096, 4096, 4096), overlap}));
  EXPECT_EQ(native_.update_count, 0);
  iree_hal_buffer_release(alias);
}

TEST_F(MappingPublicationTest,
       PriorViewsSurviveCaptureAndReachNativePublication) {
  iree_hal_buffer_t* previous = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(backing_, 4096, 8192, allocator_, &previous));
  auto removal = Mapping(4096, 8192);
  removal.source = nullptr;
  removal.access = IREE_HAL_MEMORY_ACCESS_NONE;
  removal.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
  removal.previous_source = previous;
  iree_hal_host_call_t call;
  IREE_ASSERT_OK(
      iree_hal_memory_mapping_prepare(iree_hal_device_queue(cpu_device_, 0, 0),
                                      1, &removal, 0, allocator_, &call));
  iree_hal_buffer_release(previous);
  const uint64_t args[4] = {};
  IREE_ASSERT_OK(call.fn(call.user_data, args, nullptr));
  ASSERT_EQ(native_.last_updates.size(), 1u);
  EXPECT_NE(native_.last_updates[0].previous_memory, nullptr);
  EXPECT_EQ(native_.last_updates[0].previous_source_byte_offset, 4096u);
  EXPECT_EQ(native_.last_updates[0].byte_length, 8192u);
  EXPECT_EQ(native_.last_updates[0].memory, nullptr);
  iree_hal_resource_release(call.resource);
}

TEST_F(MappingPublicationTest, PriorSourceBoundariesPreventCoalescing) {
  auto first = Mapping(0, 4096);
  first.previous_source = backing_;
  first.previous_source_offset = 8192;
  auto second = Mapping(4096, 4096, 4096);
  second.previous_source = backing_;
  second.previous_source_offset = 16384;
  IREE_ASSERT_OK(Update({first, second}));
  ASSERT_EQ(native_.last_updates.size(), 2u);
  EXPECT_EQ(native_.last_updates[0].previous_source_byte_offset, 8192u);
  EXPECT_EQ(native_.last_updates[1].previous_source_byte_offset, 16384u);
  second.previous_source_offset = 12288;
  IREE_ASSERT_OK(Update({first, second}));
  ASSERT_EQ(native_.last_updates.size(), 1u);
  EXPECT_EQ(native_.last_updates[0].previous_source_byte_offset, 8192u);
  EXPECT_EQ(native_.last_updates[0].byte_length, 8192u);
}

TEST_F(MappingPublicationTest, InvalidPriorSourceRejectsTheWholeBatch) {
  const auto prior_references =
      iree_atomic_ref_count_load(&backing_->resource.ref_count);
  for (uint32_t failure = 0; failure < 5; ++failure) {
    auto replacement = Mapping(4096, 4096);
    replacement.previous_source = backing_;
    iree_status_code_t expected = IREE_STATUS_INVALID_ARGUMENT;
    switch (failure) {
      case 0:
        replacement.previous_source = nullptr;
        replacement.previous_source_offset = 4096;
        break;
      case 1:
        replacement.previous_source_offset =
            iree_hal_buffer_byte_length(backing_);
        expected = IREE_STATUS_OUT_OF_RANGE;
        break;
      case 2:
        replacement.previous_source_offset = 1;
        expected = IREE_STATUS_FAILED_PRECONDITION;
        break;
      case 3:
        replacement.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND;
        break;
      case 4:
        replacement.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT;
        break;
    }
    IREE_EXPECT_STATUS_IS(expected, Update({Mapping(0, 4096), replacement}));
    EXPECT_EQ(native_.update_count, 0);
    EXPECT_EQ(iree_atomic_ref_count_load(&backing_->resource.ref_count),
              prior_references);
  }
}

TEST_F(MappingPublicationTest, PartialNativeFailureRetainsCheckedTeardown) {
  IREE_ASSERT_OK(Update({Mapping(0, 16384)}));
  native_.api.memory_reservation_update =
      +[](amdf_memory_reservation_t* handle, uint32_t ordinal, uint32_t count,
          const amdf_memory_mapping_update_t* updates) -> amdf_status_t {
    EXPECT_EQ(count, 2u);
    FakeNative::ReservationUpdate(handle, ordinal, 1, updates);
    return amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  };
  auto first = Mapping(0, 4096, 16384);
  first.previous_source = backing_;
  auto second = Mapping(8192, 4096, 24576);
  second.previous_source = backing_;
  second.previous_source_offset = 8192;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Update({first, second}));
  ASSERT_EQ(native_.last_updates.size(), 1u);
  native_.detach_failure =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_virtual_buffer_detach(target_, scope_));
  native_.detach_failure = AMDF_STATUS_OK;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target_, scope_));
}

class ConcurrentMemoryBackendTest : public MemoryBackendTest {
 protected:
  void SetUp() override {
    cpu_worker_count_ = 2;
    MemoryBackendTest::SetUp();
  }
};

TEST_F(ConcurrentMemoryBackendTest,
       SlowNativeMappingLeavesOtherWorkersAvailable) {
  iree_hal_buffer_t* reservation = nullptr;
  IREE_ASSERT_OK(CreateReservation(&reservation));
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      cpu_device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done));
  struct NativeWork {
    // Protects the native entry and release predicates.
    std::mutex mutex;
    // Publishes native entry and independent CPU work completion.
    std::condition_variable condition;
    // True once the actual native update callback is blocked.
    bool entered = false;
    // Set only by the independent CPU queue operation.
    bool released = false;
  } work;
  native_.before_update = [&] {
    std::unique_lock<std::mutex> lock(work.mutex);
    work.entered = true;
    work.condition.notify_all();
    work.condition.wait(lock, [&] { return work.released; });
  };
  uint64_t value = 1;
  const iree_hal_semaphore_list_t signals = {1, &done, &value};
  const iree_hal_memory_mapping_t mapping = {
      .target = reservation,
      .scope = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]),
      .target_range = {0, 4096},
  };
  auto* queue = iree_hal_device_queue(cpu_device_, 0, 0);
  IREE_ASSERT_OK(iree_hal_queue_map(queue, {}, signals, 1, &mapping, 0));
  {
    std::unique_lock<std::mutex> lock(work.mutex);
    work.condition.wait(lock, [&] { return work.entered; });
  }
  const uint64_t args[4] = {};
  iree_status_t status = iree_hal_queue_host_call(
      queue, {}, {},
      iree_hal_make_host_call(
          +[](void* user_data, const uint64_t*, iree_hal_host_call_context_t*) {
            auto* work = static_cast<NativeWork*>(user_data);
            std::lock_guard<std::mutex> lock(work->mutex);
            work->released = true;
            work->condition.notify_all();
            return iree_ok_status();
          },
          &work),
      args, 0);
  if (!iree_status_is_ok(status)) {
    std::lock_guard<std::mutex> lock(work.mutex);
    work.released = true;
    work.condition.notify_all();
  }
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  EXPECT_EQ(native_.update_count, 1);
  native_.before_update = {};
  iree_hal_buffer_release(reservation);
  iree_hal_semaphore_release(done);
}

TEST_F(MemoryBackendTest, NativeMappingFailureFailsFrontierWithoutSignals) {
  iree_hal_buffer_t* reservation = nullptr;
  IREE_ASSERT_OK(CreateReservation(&reservation));
  alignas(iree_async_frontier_t)
      uint8_t frontier_storage[sizeof(iree_async_frontier_t) +
                               sizeof(iree_async_frontier_entry_t)] = {};
  auto* frontier = reinterpret_cast<iree_async_frontier_t*>(frontier_storage);
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {
      iree_hal_device_topology_info(cpu_device_)->frontier.base_axis, 1};
  struct Completion {
    // Terminal status transferred from the queue's frontier failure.
    iree_status_t status = iree_ok_status();
    // Publishes status after the callback's last non-atomic access.
    std::atomic<bool> ready{false};
  } completion;
  iree_async_frontier_waiter_t waiter = {};
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      tracker_, frontier,
      +[](void* user_data, iree_status_t status) {
        auto* completion = static_cast<Completion*>(user_data);
        completion->status = status;
        completion->ready.store(true, std::memory_order_release);
      },
      &completion, &waiter));
  const iree_hal_memory_mapping_t mapping = {
      .target = reservation,
      .scope = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]),
      .target_range = {0, 4096},
  };
  native_.update_failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_ASSERT_OK(iree_hal_queue_map(iree_hal_device_queue(cpu_device_, 0, 0),
                                    {}, {}, 1, &mapping, 0));
  while (!completion.ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, completion.status);
  EXPECT_EQ(native_.update_count, 1);
  iree_hal_buffer_release(reservation);
}

TEST_F(MemoryBackendTest, FinalBufferReleaseOwnsNativeDestruction) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(CreateReservation(&buffer));
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(buffer, 4096, 4096, allocator_, &view));
  native_.destroy_failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  iree_hal_buffer_release(buffer);
  EXPECT_EQ(native_.reservation_count, 1);
  iree_hal_virtual_buffer_info_t info;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_info(view, &info));
  EXPECT_EQ(info.mapping_byte_offset, 4096u);
  EXPECT_TRUE(native_failures_.empty());
  iree_hal_buffer_release(view);
  EXPECT_EQ(native_.reservation_count, 0);
  EXPECT_EQ(native_failures_, (std::vector<uint64_t>{native_.destroy_failure}));
}

TEST_F(MemoryBackendTest, CheckedRemovalDoesNotReleaseAddresses) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(CreateReservation(&buffer));
  const auto scope = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  native_.detach_failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_virtual_buffer_detach(buffer, scope));
  EXPECT_EQ(native_.reservation_count, 1);
  native_.detach_failure = AMDF_STATUS_OK;
  IREE_EXPECT_OK(iree_hal_virtual_buffer_detach(buffer, scope));
  EXPECT_EQ(native_.reservation_count, 1);
  EXPECT_EQ(native_.detached_accesses, (std::vector<uint32_t>{0, 0}));
  iree_hal_buffer_release(buffer);
  EXPECT_EQ(native_.reservation_count, 0);
}

TEST_F(MemoryBackendTest,
       RejectsUnavailableReservationConsumersBeforeAllocation) {
  iree_hal_pool_family_access_t family = {
      .family = families_[0],
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
  };
  const iree_hal_pool_scope_t scope = {1, &family, {}};
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = 16384,
  };
  iree_hal_pool_t* ordinary = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &ordinary));
  iree_hal_buffer_t* reservation = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_virtual_buffer_create(group_, scope, {1, &ordinary}, &options,
                                     allocator_, &reservation));
  EXPECT_EQ(reservation, nullptr);
  EXPECT_EQ(native_.reservation_count, 0);
  EXPECT_EQ(native_.create_count, 0);
  iree_hal_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE));
  EXPECT_EQ(pool, nullptr);
  iree_hal_pool_release(ordinary);
}

enum class NativeBatchKind { kAcquire, kRelease, kRollback };

class MappingPriorityTest
    : public MemoryBackendTest,
      public ::testing::WithParamInterface<NativeBatchKind> {};

TEST_P(MappingPriorityTest, MappingOvertakesRemainingNativeSteps) {
  native_.mapping_enabled = true;
  for (uint32_t i = 0; i < 2; ++i) {
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_families_[i].host_usage = 0;
  }
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(8192, 4096),
      iree_infinite_timeout(), &backing));
  iree_hal_buffer_t* reservation = nullptr;
  IREE_ASSERT_OK(CreateReservation(&reservation));
  const auto scope = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const iree_hal_memory_mapping_t mappings[] = {
      {
          .target = reservation,
          .scope = scope,
          .target_range = {0, 4096},
          .source = backing,
          .source_offset = 0,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
      },
      {
          .target = reservation,
          .scope = scope,
          .target_range = {4096, 4096},
          .source = backing,
          .source_offset = 4096,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
      },
  };
  iree_hal_memory_mapping_batch_t* batch = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_mapping_batch_create(
      iree_hal_device_queue(cpu_device_, 0, 0), IREE_ARRAYSIZE(mappings),
      mappings, 0, allocator_, &batch));
  IREE_ASSERT_OK(iree_hal_memory_mapping_batch_resolve(batch));

  // Private backing and pool slabs share the same native owner. The test gates
  // real native entry points underneath that owner, not its scheduler.
  iree_hal_amd_fabric_memory_options_t options = {.scope = native_.scope()};
  options.profile.roles = AMDF_MEMORY_PROFILE_ROLE_CREATE;
  options.profile.allocation.byte_length_granularity = 4096;
  options.profile.allocation.minimum_alignment = 4096;
  options.profile.allocation.maximum_byte_length = UINT64_C(1) << 30;
  options.required_flags = native_.required_flags;
  std::array<iree_hal_amd_fabric_memory_request_t, 100> requests;
  std::array<iree_hal_amd_fabric_memory_t, 100> memories = {};
  for (auto& request : requests) {
    request = {&options, 4096, 4096};
  }
  std::promise<iree::Status> acquired;
  iree_hal_amd_fabric_memory_acquire_request_t acquire = {
      .operation =
          {
              .api = &native_.api,
              .event_sink = iree_hal_device_event_sink_discard(),
          },
      .count = requests.size(),
      .requests = requests.data(),
      .memories = memories.data(),
      .completion =
          {
              .fn =
                  [](void* user_data, iree_status_t status) {
                    static_cast<std::promise<iree::Status>*>(user_data)
                        ->set_value(iree::Status(std::move(status)));
                  },
              .user_data = &acquired,
          },
  };
  std::promise<void> released;
  iree_hal_amd_fabric_memory_release_request_t release = {
      .operation =
          {
              .api = &native_.api,
              .event_sink = iree_hal_device_event_sink_discard(),
          },
      .count = memories.size(),
      .memories = memories.data(),
      .completion =
          {
              .fn =
                  [](void* user_data) {
                    static_cast<std::promise<void>*>(user_data)->set_value();
                  },
              .user_data = &released,
          },
  };
  if (GetParam() == NativeBatchKind::kRelease) {
    iree_hal_amd_fabric_allocation_queue_acquire_memory(allocation_queue_,
                                                        &acquire);
    IREE_ASSERT_OK(acquired.get_future().get());
  }

  std::vector<char> sequence;
  std::promise<void> entered;
  std::promise<void> resume;
  auto resumed = resume.get_future().share();
  uint32_t acquisition_count = 0, release_count = 0;
  native_.before_create = [&] {
    sequence.push_back('A');
    if (++acquisition_count == 1 && GetParam() == NativeBatchKind::kAcquire) {
      entered.set_value();
      resumed.wait();
    }
    if (acquisition_count == 3 && GetParam() == NativeBatchKind::kRollback) {
      native_.create_failure =
          amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
  };
  native_.before_destroy = [&] {
    sequence.push_back('R');
    if (++release_count == 1 && GetParam() != NativeBatchKind::kAcquire) {
      entered.set_value();
      resumed.wait();
    }
  };
  native_.before_update = [&] { sequence.push_back('M'); };
  if (GetParam() == NativeBatchKind::kRelease) {
    iree_hal_amd_fabric_allocation_queue_release_memory(allocation_queue_,
                                                        &release);
  } else {
    iree_hal_amd_fabric_allocation_queue_acquire_memory(allocation_queue_,
                                                        &acquire);
  }
  entered.get_future().get();
  std::promise<iree::Status> mapped;
  iree_hal_amd_fabric_memory_map_request_t map = {
      .batch = batch,
      .completion =
          {
              .fn =
                  [](void* user_data, iree_status_t status) {
                    static_cast<std::promise<iree::Status>*>(user_data)
                        ->set_value(iree::Status(std::move(status)));
                  },
              .user_data = &mapped,
          },
  };
  iree_hal_amd_fabric_allocation_queue_map_memory(allocation_queue_, &map);
  resume.set_value();
  IREE_EXPECT_OK(mapped.get_future().get());
  if (GetParam() == NativeBatchKind::kRelease) {
    released.get_future().get();
  } else if (GetParam() == NativeBatchKind::kRollback) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          acquired.get_future().get());
    EXPECT_EQ(sequence, (std::vector<char>{'A', 'A', 'A', 'R', 'M', 'R'}));
    for (const auto& memory : memories) {
      EXPECT_EQ(memory.handle, nullptr);
    }
  } else {
    IREE_EXPECT_OK(acquired.get_future().get());
  }
  const auto mapping_position =
      std::find(sequence.begin(), sequence.end(), 'M');
  EXPECT_EQ(mapping_position - sequence.begin(),
            GetParam() == NativeBatchKind::kRollback ? 4 : 1);
  EXPECT_EQ(native_.update_count, 1);
  ASSERT_EQ(native_.last_updates.size(), 1u);
  EXPECT_EQ(native_.last_updates[0].target_byte_offset, 0u);
  EXPECT_EQ(native_.last_updates[0].source_byte_offset, 0u);
  EXPECT_EQ(native_.last_updates[0].byte_length, 8192u);

  native_.before_create = {};
  native_.before_destroy = {};
  native_.before_update = {};
  if (GetParam() == NativeBatchKind::kAcquire) {
    iree_hal_amd_fabric_allocation_queue_release_memory(allocation_queue_,
                                                        &release);
    released.get_future().get();
  }
  iree_hal_memory_mapping_batch_release(batch);
  IREE_EXPECT_OK(iree_hal_virtual_buffer_detach(reservation, scope));
  iree_hal_buffer_release(reservation);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
}

INSTANTIATE_TEST_SUITE_P(NativeBatches, MappingPriorityTest,
                         ::testing::Values(NativeBatchKind::kAcquire,
                                           NativeBatchKind::kRelease,
                                           NativeBatchKind::kRollback));

}  // namespace iree::hal::amd::fabric
