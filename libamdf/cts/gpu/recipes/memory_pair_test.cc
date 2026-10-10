// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/pm4/command_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"

namespace {

enum class PairQuery { kConcrete, kProfile };

class MemoryPairRecipeTest : public Pm4CommandTest {
 protected:
  void QueryPair(const GpuMemory& memory, PairQuery query_kind,
                 amdf_memory_site_kind_t producer,
                 amdf_memory_site_kind_t consumer,
                 amdf_memory_pair_info_t* pair) {
    pair->type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    pair->structure_size = sizeof(*pair);
    if (query_kind == PairQuery::kConcrete) {
      const auto source = producer == AMDF_MEMORY_SITE_KIND_HOST
                              ? memory.HostSite()
                              : memory.DeviceSite(family_.ordinal);
      const auto target = consumer == AMDF_MEMORY_SITE_KIND_HOST
                              ? memory.HostSite()
                              : memory.DeviceSite(family_.ordinal);
      ASSERT_EQ(api_->memory_query_pair_info(&source, &target, pair),
                AMDF_STATUS_OK);
    } else {
      amdf_memory_profile_pair_query_t query = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
          .structure_size = sizeof(query),
          .memory_profile_ordinal = memory.creation.memory_profile_ordinal,
          .access_count = memory.creation.access_count,
          .required_flags = memory.creation.required_flags,
          .accesses = memory.creation.accesses,
      };
      query.producer.kind = producer;
      query.consumer.kind = consumer;
      if (producer == AMDF_MEMORY_SITE_KIND_HOST) {
        query.producer.value.host_access = AMDF_MEMORY_MAP_FLAG_WRITE;
      } else {
        query.producer.value.device.queue_family_ordinal = family_.ordinal;
      }
      if (consumer == AMDF_MEMORY_SITE_KIND_HOST) {
        query.consumer.value.host_access = AMDF_MEMORY_MAP_FLAG_READ;
      } else {
        query.consumer.value.device.queue_family_ordinal = family_.ordinal;
      }
      ASSERT_EQ(api_->memory_scope_query_pair_info(system_scope_, &query, pair),
                AMDF_STATUS_OK);
    }
    ASSERT_NE(pair->flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
  }

  void CheckQueueTransition(const amdf_cache_transition_t& transition,
                            amdf_cache_operation_t operation) {
    ASSERT_EQ(transition.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    ASSERT_EQ(transition.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
    ASSERT_EQ(transition.operation, operation);
  }

  void RunCoherentHandoff(PairQuery query_kind) {
    constexpr amdf_memory_access_t kReadWrite =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    GpuMemory* source = nullptr;
    GpuMemory* intermediate = nullptr;
    GpuMemory* target = nullptr;
    GpuMemory* control = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &source));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &intermediate));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &target));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &control));

    amdf_memory_pair_info_t ingress = {}, handoff = {}, egress = {};
    ASSERT_NO_FATAL_FAILURE(QueryPair(*source, query_kind,
                                      AMDF_MEMORY_SITE_KIND_HOST,
                                      AMDF_MEMORY_SITE_KIND_DEVICE, &ingress));
    ASSERT_NO_FATAL_FAILURE(QueryPair(*intermediate, query_kind,
                                      AMDF_MEMORY_SITE_KIND_DEVICE,
                                      AMDF_MEMORY_SITE_KIND_DEVICE, &handoff));
    ASSERT_NO_FATAL_FAILURE(QueryPair(*target, query_kind,
                                      AMDF_MEMORY_SITE_KIND_DEVICE,
                                      AMDF_MEMORY_SITE_KIND_HOST, &egress));
    // Qualified coherent CPU edges execute no cache operation. The release
    // publication and completion-acquire loads still supply ordering.
    ASSERT_EQ(ingress.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(ingress.release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
    ASSERT_EQ(egress.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(egress.acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
    ASSERT_NO_FATAL_FAILURE(CheckQueueTransition(
        ingress.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
    ASSERT_NO_FATAL_FAILURE(CheckQueueTransition(
        handoff.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));
    ASSERT_NO_FATAL_FAILURE(CheckQueueTransition(
        handoff.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
    ASSERT_NO_FATAL_FAILURE(CheckQueueTransition(
        egress.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));

    GpuCommandQueue* producer = nullptr;
    GpuCommandQueue* consumer = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&producer));
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&consumer));
    EXPECT_TRUE(amdf_device_id_is_equal(&producer->device_id(),
                                        &consumer->device_id()));
    ASSERT_NE(producer->native_handle(), consumer->native_handle());
    ASSERT_GE(producer->words().size_bytes(), 2048u);
    ASSERT_GE(consumer->words().size_bytes(), 2048u);
    auto* input = static_cast<uint32_t*>(source->host.pointer);
    auto* output = static_cast<uint32_t*>(target->host.pointer);
    auto* staging = static_cast<uint32_t*>(intermediate->host.pointer);
    auto* milestones = static_cast<uint32_t*>(control->host.pointer);
    milestones[0] = milestones[16] = 0;
    uint64_t producer_index = 0, consumer_index = 0;
    constexpr size_t kWordCount = 16;
    for (uint32_t epoch = 1; epoch <= 2; ++epoch) {
      for (size_t i = 0; i < kWordCount; ++i) {
        input[i] = epoch * 0x17390000u + static_cast<uint32_t>(i) * 0x00110101u;
        staging[i] = output[i] = ~input[i];
      }
      Pm4CommandWriter produce(producer->words().data() + producer_index,
                               *pm4_profile_);
      produce.SystemBarrier();  // ingress.acquire.
      for (size_t i = 0; i < kWordCount; ++i) {
        produce.CopyData32(source->device_address + i * sizeof(uint32_t),
                           intermediate->device_address + i * sizeof(uint32_t));
      }
      produce.SystemBarrier();  // handoff.release.
      produce.WriteData32(control->device_address, epoch);
      produce.PadToEightWords();
      producer_index += produce.word_count();

      Pm4CommandWriter consume(consumer->words().data() + consumer_index,
                               *pm4_profile_);
      consume.WaitMemory32(control->device_address, epoch);
      consume.SystemBarrier();  // handoff.acquire, after the ordering edge.
      for (size_t i = 0; i < kWordCount; ++i) {
        consume.CopyData32(intermediate->device_address + i * sizeof(uint32_t),
                           target->device_address + i * sizeof(uint32_t));
      }
      consume.SystemBarrier();  // egress.release.
      consume.WriteData32(control->device_address + 64, epoch);
      consume.PadToEightWords();
      consumer_index += consume.word_count();

      // Submit the consumer first. The device memory dependency, not host
      // retirement or queue submission order, makes the payload usable.
      ASSERT_NO_FATAL_FAILURE(
          consumer->Publish(api_, gpu_api_, consumer_index));
      ASSERT_NO_FATAL_FAILURE(
          producer->Publish(api_, gpu_api_, producer_index));
      GpuWaitEqual<uint32_t>(reinterpret_cast<uintptr_t>(milestones + 16),
                             epoch);
      for (size_t i = 0; i < kWordCount; ++i) {
        EXPECT_EQ(output[i], input[i]) << i;
      }
      ASSERT_NO_FATAL_FAILURE(producer->WaitRetired(api_));
      ASSERT_NO_FATAL_FAILURE(consumer->WaitRetired(api_));
    }
  }
};

TEST_F(MemoryPairRecipeTest, ConcreteCoherentHostAndTwoQueueHandoff) {
  RunCoherentHandoff(PairQuery::kConcrete);
}

TEST_F(MemoryPairRecipeTest, ProfileCoherentHostAndTwoQueueHandoff) {
  RunCoherentHandoff(PairQuery::kProfile);
}

}  // namespace
