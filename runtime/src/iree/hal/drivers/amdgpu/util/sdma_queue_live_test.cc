// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "iree/hal/drivers/amdgpu/util/sdma_emitter.h"
#include "iree/hal/drivers/amdgpu/util/sdma_program.h"
#include "iree/hal/drivers/amdgpu/util/sdma_queue.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SdmaQueueLiveTest : public ::testing::Test {
 protected:
  iree_hal_amdgpu_libhsa_t lib{};
  hsa_agent_t gpu{}, cpu{};
  hsa_amd_memory_pool_t host_pool{}, device_pool{};
  iree_hal_amdgpu_gfxip_version_t version{};
  iree_hal_amdgpu_sdma_queue_t queue{};
  std::vector<void*> allocations;
  static hsa_status_t HSA_API Agent(hsa_agent_t agent, void* user) {
    auto* self = static_cast<SdmaQueueLiveTest*>(user);
    hsa_device_type_t type;
    auto status = iree_hsa_agent_get_info_raw(&self->lib, agent,
                                              HSA_AGENT_INFO_DEVICE, &type);
    if (status != HSA_STATUS_SUCCESS) return status;
    if (type == HSA_DEVICE_TYPE_GPU && !self->gpu.handle) self->gpu = agent;
    if (type == HSA_DEVICE_TYPE_CPU && !self->cpu.handle) self->cpu = agent;
    return HSA_STATUS_SUCCESS;
  }
  struct PoolQuery {
    const iree_hal_amdgpu_libhsa_t* lib;
    bool host;
    hsa_amd_memory_pool_t pool{};
  };
  static hsa_status_t HSA_API Pool(hsa_amd_memory_pool_t pool, void* user) {
    auto* q = static_cast<PoolQuery*>(user);
    hsa_amd_segment_t segment;
    auto status = iree_hsa_amd_memory_pool_get_info_raw(
        q->lib, pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (status != HSA_STATUS_SUCCESS || segment != HSA_AMD_SEGMENT_GLOBAL)
      return status;
    uint32_t flags = 0;
    bool allowed = false;
    status = iree_hsa_amd_memory_pool_get_info_raw(
        q->lib, pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = iree_hsa_amd_memory_pool_get_info_raw(
        q->lib, pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allowed);
    if (status != HSA_STATUS_SUCCESS) return status;
    if (allowed &&
        (q->host ? (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
                 : (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)))
      q->pool = pool;
    return HSA_STATUS_SUCCESS;
  }
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        iree_allocator_system(), &lib));
    IREE_ASSERT_OK(iree_hsa_iterate_agents(IREE_LIBHSA(&lib), Agent, this));
    if (!gpu.handle || !cpu.handle)
      GTEST_SKIP() << "GPU and CPU agents required";
    char name[64]{};
    IREE_ASSERT_OK(iree_hsa_agent_get_info(IREE_LIBHSA(&lib), gpu,
                                           HSA_AGENT_INFO_NAME, name));
    iree_hal_amdgpu_target_identity_t identity{};
    IREE_ASSERT_OK(iree_hal_amdgpu_target_identity_parse_processor(
        iree_make_cstring_view(name), &identity));
    version = identity.version;
    iree_hal_amdgpu_sdma_capabilities_t caps{};
    auto status = iree_hal_amdgpu_sdma_query_capabilities(version, &caps);
    if (iree_status_is_unimplemented(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "No SDMA encoding profile for " << name;
    }
    IREE_ASSERT_OK(status);
    PoolQuery host{&lib, true}, device{&lib, false};
    IREE_ASSERT_OK(iree_hsa_amd_agent_iterate_memory_pools(IREE_LIBHSA(&lib),
                                                           cpu, Pool, &host));
    IREE_ASSERT_OK(iree_hsa_amd_agent_iterate_memory_pools(IREE_LIBHSA(&lib),
                                                           gpu, Pool, &device));
    host_pool = host.pool;
    device_pool = device.pool;
    ASSERT_NE(host_pool.handle, 0u);
    ASSERT_NE(device_pool.handle, 0u);
  }
  void TearDown() override {
    iree_hal_amdgpu_sdma_queue_deinitialize(&queue);
    for (void* p : allocations)
      IREE_EXPECT_OK(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&lib), p));
    iree_hal_amdgpu_libhsa_deinitialize(&lib);
  }
  void* Allocate(size_t bytes, bool device = false) {
    void* p = nullptr;
    IREE_CHECK_OK(iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(&lib), device ? device_pool : host_pool, bytes, 0, &p));
    allocations.push_back(p);
    IREE_CHECK_OK(iree_hsa_amd_agents_allow_access(IREE_LIBHSA(&lib), 1, &gpu,
                                                   nullptr, p));
    return p;
  }
  void Open(uint32_t engine_id = UINT32_MAX) {
    iree_hal_amdgpu_sdma_queue_params_t p{};
    p.libhsa = &lib;
    p.agent = gpu;
    p.gfxip_version = version;
    p.capacity_bytes = 1024;
    p.engine_id = engine_id;
    p.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    IREE_ASSERT_OK(iree_hal_amdgpu_sdma_queue_initialize(&p, &queue));
  }
  void Submit(const iree_hal_amdgpu_sdma_program_params_t& p) {
    uint32_t count = 0;
    uint32_t* words = nullptr;
    IREE_ASSERT_OK(iree_hal_amdgpu_sdma_program_measure(&p, &count));
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
      auto status =
          iree_hal_amdgpu_sdma_ring_try_reserve(&queue.ring, count * 4, &words);
      if (!iree_status_is_unavailable(status)) {
        IREE_ASSERT_OK(status);
        break;
      }
      iree_status_ignore(status);
      ASSERT_LT(std::chrono::steady_clock::now(), deadline);
      std::this_thread::yield();
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_sdma_program_emit(&p, count, words, &count));
    iree_hal_amdgpu_sdma_ring_commit(&queue.ring);
  }
  void SubmitIndirect(const iree_hal_amdgpu_sdma_program_params_t& p) {
    uint32_t count = 0;
    IREE_ASSERT_OK(iree_hal_amdgpu_sdma_program_measure(&p, &count));
    uint32_t padded_count = (count + 7u) & ~7u;
    std::vector<uint32_t> words(padded_count, 0);
    IREE_ASSERT_OK(
        iree_hal_amdgpu_sdma_program_emit(&p, count, words.data(), &count));
    void* ib = Allocate(padded_count * 4, true);
    IREE_ASSERT_OK(iree_hsa_memory_copy(IREE_LIBHSA(&lib), ib, words.data(),
                                        padded_count * 4));
    uint32_t* ring_words = nullptr;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
      auto status = iree_hal_amdgpu_sdma_ring_try_reserve(&queue.ring, 16 * 4,
                                                          &ring_words);
      if (!iree_status_is_unavailable(status)) {
        IREE_ASSERT_OK(status);
        break;
      }
      iree_status_ignore(status);
      ASSERT_LT(std::chrono::steady_clock::now(), deadline);
      std::this_thread::yield();
    }
    std::memset(ring_words, 0, 16 * 4);
    ASSERT_NE(iree_hal_amdgpu_sdma_emit_indirect(
                  16, ring_words, ring_words - queue.ring.base,
                  reinterpret_cast<uint64_t>(ib), padded_count, 0, 0),
              0u);
    iree_hal_amdgpu_sdma_ring_commit(&queue.ring);
  }
  void Wait(uint32_t* done, uint32_t value) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((uint32_t)iree_atomic_load((iree_atomic_int32_t*)done,
                                      iree_memory_order_acquire) != value) {
      ASSERT_LT(std::chrono::steady_clock::now(), deadline);
      std::this_thread::yield();
    }
  }
};

TEST_F(SdmaQueueLiveTest, DeviceToHostCopyWaitTimestampAndWrap) {
  constexpr size_t bytes = (1u << 22) + 17;
  auto* source = Allocate(bytes, true);
  auto* input = static_cast<uint8_t*>(Allocate(bytes));
  auto* output = static_cast<uint8_t*>(Allocate(bytes));
  auto* state = static_cast<uint32_t*>(Allocate(128));
  std::memset(state, 0, 128);
  for (size_t i = 0; i < bytes; ++i) input[i] = (i * 17 + i / 113) & 255;
  IREE_ASSERT_OK(iree_hsa_memory_copy(IREE_LIBHSA(&lib), source, input, bytes));
  Open();
  ASSERT_NE(queue.handle, nullptr);
  iree_hal_amdgpu_sdma_copy_t copy{reinterpret_cast<uint64_t>(source),
                                   reinterpret_cast<uint64_t>(output), bytes};
  iree_hal_amdgpu_sdma_program_params_t p{};
  p.capabilities = queue.capabilities;
  p.copy_count = 1;
  p.copies = &copy;
  p.wait_address = reinterpret_cast<uint64_t>(state + 16);
  p.completion_address = reinterpret_cast<uint64_t>(state);
  p.start_timestamp_address = reinterpret_cast<uint64_t>(state + 8);
  p.end_timestamp_address = reinterpret_cast<uint64_t>(state + 24);
  for (uint32_t epoch = 1; epoch <= 40; ++epoch) {
    std::memset(output, 0, bytes);
    p.wait_value = p.completion_value = epoch;
    Submit(p);
    ASSERT_FALSE(HasFatalFailure());
    // The unresolved ready milestone must prevent completion publication.
    EXPECT_EQ((uint32_t)iree_atomic_load((iree_atomic_int32_t*)state,
                                         iree_memory_order_acquire),
              epoch - 1);
    iree_atomic_store((iree_atomic_int32_t*)(state + 16), epoch,
                      iree_memory_order_release);
    Wait(state, epoch);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(std::memcmp(output, input, bytes), 0);
    double ns = 0;
    IREE_ASSERT_OK(iree_hal_amdgpu_sdma_queue_elapsed_ns(
        &queue, *reinterpret_cast<uint64_t*>(state + 8),
        *reinterpret_cast<uint64_t*>(state + 24), &ns));
    EXPECT_GT(ns, 0.0);
  }
  EXPECT_GT(queue.ring.committed_position, queue.ring.capacity * 4u);
}
// Requires the experimental KFD IB_ENABLE patch. Explicitly opt in.
TEST_F(SdmaQueueLiveTest, DISABLED_IndirectCopyWaitTimestampAndWrap) {
  constexpr size_t bytes = (1u << 22) + 17;
  auto* source = Allocate(bytes, true);
  auto* input = static_cast<uint8_t*>(Allocate(bytes));
  auto* output = static_cast<uint8_t*>(Allocate(bytes));
  auto* state = static_cast<uint32_t*>(Allocate(128));
  std::memset(state, 0, 128);
  for (size_t i = 0; i < bytes; ++i) input[i] = (i * 17 + i / 113) & 255;
  IREE_ASSERT_OK(iree_hsa_memory_copy(IREE_LIBHSA(&lib), source, input, bytes));
  Open();
  ASSERT_NE(queue.handle, nullptr);
  iree_hal_amdgpu_sdma_copy_t copy{reinterpret_cast<uint64_t>(source),
                                   reinterpret_cast<uint64_t>(output), bytes};
  iree_hal_amdgpu_sdma_program_params_t p{};
  p.capabilities = queue.capabilities;
  p.copy_count = 1;
  p.copies = &copy;
  p.wait_address = reinterpret_cast<uint64_t>(state + 16);
  p.completion_address = reinterpret_cast<uint64_t>(state);
  p.start_timestamp_address = reinterpret_cast<uint64_t>(state + 8);
  p.end_timestamp_address = reinterpret_cast<uint64_t>(state + 24);
  for (uint32_t epoch = 1; epoch <= 40; ++epoch) {
    std::memset(output, 0, bytes);
    p.wait_value = p.completion_value = epoch;
    SubmitIndirect(p);
    ASSERT_FALSE(HasFatalFailure());
    // The unresolved ready milestone must prevent completion publication.
    EXPECT_EQ((uint32_t)iree_atomic_load((iree_atomic_int32_t*)state,
                                         iree_memory_order_acquire),
              epoch - 1);
    iree_atomic_store((iree_atomic_int32_t*)(state + 16), epoch,
                      iree_memory_order_release);
    Wait(state, epoch);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(std::memcmp(output, input, bytes), 0);
    double ns = 0;
    IREE_ASSERT_OK(iree_hal_amdgpu_sdma_queue_elapsed_ns(
        &queue, *reinterpret_cast<uint64_t*>(state + 8),
        *reinterpret_cast<uint64_t*>(state + 24), &ns));
    EXPECT_GT(ns, 0.0);
  }
  EXPECT_GT(queue.ring.committed_position, queue.ring.capacity * 2u);
}
// Explicit diagnostic: stock KFD may disable INDIRECT while accepting the
// packet. Run with --gtest_also_run_disabled_tests and filter this case. A
// failed body assertion with successful suffix/drain distinguishes skipping
// from a hang. Packet layout and placement follow sdma_v7_0_ring_emit_ib in
// Linux v7.0:
// https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/amd/amdgpu/sdma_v7_0.c
// VMID=0 and CSA=0 are diagnostic inputs, not a qualified user-queue IB ABI.
TEST_F(SdmaQueueLiveTest, DISABLED_IndirectBufferDiagnostic) {
  if (version.major != 12 || version.minor != 0 || version.stepping != 1)
    GTEST_SKIP() << "Diagnostic qualified only for gfx1201";
  for (uint32_t engine : {0u, 1u}) {
    Open(engine);
    ASSERT_FALSE(HasFatalFailure());
    for (bool device : {false, true}) {
      for (uint32_t body_dw : {8u, 32u, 128u}) {
        auto* state = static_cast<uint32_t*>(Allocate(256));
        std::memset(state, 0, 256);
        auto* ib = Allocate(body_dw * 4, device);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(ib) & 31u, 0u);
        std::vector<uint32_t> body(body_dw, 0);
        ASSERT_EQ(iree_hal_amdgpu_sdma_emit_fence32(
                      body_dw, body.data(),
                      reinterpret_cast<uint64_t>(state + 16), 0x22222222),
                  4u);
        IREE_ASSERT_OK(iree_hsa_memory_copy(IREE_LIBHSA(&lib), ib, body.data(),
                                            body_dw * 4));
        // Verify the command bytes are accessible through this native SDMA
        // queue before testing instruction fetch from the same allocation.
        auto* readback = Allocate(body_dw * 4);
        iree_hal_amdgpu_sdma_copy_t copy{reinterpret_cast<uint64_t>(ib),
                                         reinterpret_cast<uint64_t>(readback),
                                         body_dw * 4};
        iree_hal_amdgpu_sdma_program_params_t params{};
        params.capabilities = queue.capabilities;
        params.copy_count = 1;
        params.copies = &copy;
        params.completion_address = reinterpret_cast<uint64_t>(state + 48);
        params.completion_value = 1;
        Submit(params);
        ASSERT_FALSE(HasFatalFailure());
        Wait(state + 48, 1);
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(std::memcmp(readback, body.data(), body_dw * 4), 0);
        for (bool indirect : {false, true}) {
          std::memset(state, 0, 192);
          // Reservation may wrap to zero; calculate INDIRECT packet end
          // alignment from the returned address, not the previous write index.
          uint32_t* words = nullptr;
          const uint32_t count = body_dw + 32;
          auto deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(10);
          for (;;) {
            auto status = iree_hal_amdgpu_sdma_ring_try_reserve(
                &queue.ring, count * 4, &words);
            if (!iree_status_is_unavailable(status)) {
              IREE_ASSERT_OK(status);
              break;
            }
            iree_status_ignore(status);
            ASSERT_LT(std::chrono::steady_clock::now(), deadline);
            std::this_thread::yield();
          }
          std::memset(words, 0, count * 4);
          uint32_t n = iree_hal_amdgpu_sdma_emit_fence32(
              count, words, reinterpret_cast<uint64_t>(state), 0x11111111);
          if (indirect) {
            uint64_t offset =
                static_cast<uint64_t>(words - queue.ring.base) + n;
            uint32_t emitted = iree_hal_amdgpu_sdma_emit_indirect(
                count - n, words + n, offset, reinterpret_cast<uint64_t>(ib),
                body_dw, 0, 0);
            ASSERT_NE(emitted, 0u);
            n += emitted;
          } else {
            std::memcpy(words + n, body.data(), body_dw * 4);
            n += body_dw;
          }
          n += iree_hal_amdgpu_sdma_emit_fence32(
              count - n, words + n, reinterpret_cast<uint64_t>(state + 32),
              0x33333333);
          ASSERT_LE(n, count);
          iree_hal_amdgpu_sdma_ring_commit(&queue.ring);
          Wait(state + 32, 0x33333333);
          ASSERT_FALSE(HasFatalFailure());
          deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(10);
          while (static_cast<uint64_t>(iree_atomic_load(
                     queue.ring.read_position, iree_memory_order_acquire)) !=
                 queue.ring.committed_position) {
            ASSERT_LT(std::chrono::steady_clock::now(), deadline);
            std::this_thread::yield();
          }
          SCOPED_TRACE(::testing::Message()
                       << "engine=" << engine << " device_memory=" << device
                       << "body_dwords=" << body_dw
                       << " indirect=" << indirect);
          EXPECT_EQ(state[0], 0x11111111u);
          EXPECT_EQ(state[16], 0x22222222u);
        }
      }
    }
    iree_hal_amdgpu_sdma_queue_deinitialize(&queue);
  }
}
}  // namespace
