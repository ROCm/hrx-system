// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/sdma_queue.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

#if !IREE_HAL_AMDGPU_LIBHSA_STATIC
struct State {
  hsa_queue_t native{};
  amd_signal_t signal{};
  alignas(64) uint32_t storage[64]{};
  uint64_t read = 0, write = 0, bell = 0;
  int creates = 0, destroys = 0, queries = 0, fail_query = 0;
  hsa_status_t create_status = HSA_STATUS_SUCCESS;
  State() {
    native.base_address = storage;
    native.size = sizeof(storage);
    native.doorbell_signal.handle = reinterpret_cast<uint64_t>(&signal);
    signal.kind = AMD_SIGNAL_KIND_DOORBELL;
    signal.hardware_doorbell_ptr = &bell;
  }
};
hsa_status_t HSA_API Create(hsa_agent_t agent, hsa_amd_queue_create_desc_t* d,
                            uint32_t n) {
  auto* s = reinterpret_cast<State*>(agent.handle);
  ++s->creates;
  EXPECT_EQ(n, 1u);
  EXPECT_EQ(d->engine_type, HSA_AMD_QUEUE_ENGINE_SDMA);
  EXPECT_EQ(d->queue_size_bytes, 256u);
  EXPECT_EQ(d->engine.sdma.sdma_engine_id, 3u);
  EXPECT_EQ(d->version, HSA_AMD_QUEUE_CREATE_DESC_VERSION);
  d->queue = &s->native;
  return s->create_status;
}
hsa_status_t HSA_API Destroy(hsa_queue_t* q) {
  ++reinterpret_cast<State*>(q)->destroys;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t HSA_API Info(hsa_queue_t* q, hsa_queue_info_attribute_t attr,
                          void* out) {
  auto* s = reinterpret_cast<State*>(q);
  if (++s->queries == s->fail_query) return HSA_STATUS_ERROR_INVALID_QUEUE;
  switch (attr) {
    case HSA_AMD_QUEUE_INFO_READ_POINTER:
      *static_cast<uint64_t*>(out) = reinterpret_cast<uint64_t>(&s->read);
      break;
    case HSA_AMD_QUEUE_INFO_WRITE_POINTER:
      *static_cast<uint64_t*>(out) = reinterpret_cast<uint64_t>(&s->write);
      break;
    case HSA_AMD_QUEUE_INFO_SDMA_ENGINE_ID:
      *static_cast<uint32_t*>(out) = 3;
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}
iree_hal_amdgpu_libhsa_t Lib() {
  iree_hal_amdgpu_libhsa_t l{};
  l.hsa_amd_queue_create = Create;
  l.hsa_amd_queue_get_info = Info;
  l.hsa_queue_destroy = Destroy;
  return l;
}
iree_hal_amdgpu_sdma_queue_params_t Params(State& s,
                                           const iree_hal_amdgpu_libhsa_t& l) {
  iree_hal_amdgpu_sdma_queue_params_t p{};
  p.libhsa = &l;
  p.agent.handle = reinterpret_cast<uint64_t>(&s);
  p.gfxip_version = {12, 0, 1};
  p.capacity_bytes = 256;
  p.engine_id = 3;
  p.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  return p;
}

TEST(SdmaQueueTest, BorrowsRuntimeAndSelectedAgent) {
  State s;
  auto l = Lib();
  auto p = Params(s, l);
  iree_hal_amdgpu_sdma_queue_t q{};
  IREE_ASSERT_OK(iree_hal_amdgpu_sdma_queue_initialize(&p, &q));
  EXPECT_EQ(q.libhsa, &l);
  EXPECT_EQ(q.agent.handle, p.agent.handle);
  EXPECT_EQ(q.engine_id, 3u);
  EXPECT_FALSE(q.capabilities.supports_indirect_buffers);
  EXPECT_EQ(q.ring.base, s.storage);
  iree_hal_amdgpu_sdma_queue_deinitialize(&q);
  EXPECT_EQ(s.creates, 1);
  EXPECT_EQ(s.destroys, 1);
  EXPECT_EQ(q.handle, nullptr);
  iree_hal_amdgpu_sdma_queue_deinitialize(&q);
  EXPECT_EQ(s.destroys, 1);
}

TEST(SdmaQueueTest, QueryFailuresDestroyUnpublishedQueue) {
  for (int fail = 1; fail <= 3; ++fail) {
    State s;
    s.fail_query = fail;
    auto l = Lib();
    auto p = Params(s, l);
    iree_hal_amdgpu_sdma_queue_t q{};
    q.engine_id = 42;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_hal_amdgpu_sdma_queue_initialize(&p, &q));
    EXPECT_EQ(s.destroys, 1);
    EXPECT_EQ(q.engine_id, 42u);
    EXPECT_EQ(q.handle, nullptr);
  }
}

TEST(SdmaQueueTest, PartialCreateFailureAndMissingDoorbellCleanUp) {
  State s;
  auto l = Lib();
  auto p = Params(s, l);
  iree_hal_amdgpu_sdma_queue_t q{};
  s.create_status = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amdgpu_sdma_queue_initialize(&p, &q));
  EXPECT_EQ(s.destroys, 1);
  EXPECT_EQ(s.queries, 0);
  s.create_status = HSA_STATUS_SUCCESS;
  s.signal.hardware_doorbell_ptr = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amdgpu_sdma_queue_initialize(&p, &q));
  EXPECT_EQ(s.destroys, 2);
}

TEST(SdmaQueueTest, UnsupportedTargetDoesNotCreateQueue) {
  State s;
  auto l = Lib();
  auto p = Params(s, l);
  p.gfxip_version = {9, 4, 2};
  iree_hal_amdgpu_sdma_queue_t q{};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amdgpu_sdma_queue_initialize(&p, &q));
  EXPECT_EQ(s.creates, 0);
}
#endif
}  // namespace
