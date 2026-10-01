// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/peer/device_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"
#include "libamdf/cts/util/mapped_memory.h"

namespace {

constexpr GpuQueueRequirements kRequirements = {
    .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
    .roles = AMDF_QUEUE_ROLE_TRANSFER,
    .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
};
constexpr size_t kSlotCount = 4;
enum Region {
  kInput,
  kFirstIntermediate,
  kSecondIntermediate,
  kOutput,
  kRegionCount
};
constexpr size_t kRegionWordCount = 2048;
constexpr size_t kDataWordCount = kSlotCount * kRegionCount * kRegionWordCount;
constexpr size_t kControlWordCount = 1024;
constexpr size_t kBackingWordCount = kDataWordCount + kControlWordCount;
constexpr size_t kCopyWordCount = 257;
// Each copy crosses a 4 KiB boundary at a different offset within its region.
constexpr std::array<size_t, kRegionCount> kOffsets = {1020, 1009, 1013, 1017};
// The initiator uses two polls, two copies and two fences per slot. Its peer
// uses two polls, one copy and three fences. These finite streams never wrap.
constexpr size_t kInitiatorWordsPerSlot = 2 * 6 + 2 * 7 + 2 * 4;
constexpr size_t kPeerWordsPerSlot = 2 * 6 + 7 + 3 * 4;

size_t DataWord(size_t slot, size_t region) {
  return (slot * kRegionCount + region) * kRegionWordCount + kOffsets[region];
}

enum Control {
  kArrival,
  kInitiatorReady,
  kPeerReady,
  kInitiatorDone,
  kPeerDone
};

size_t ControlWord(size_t slot, Control field) {
  // Each single-writer DWORD occupies its own 64-byte line. A terminal value
  // remains stable through both readers; rearming follows both retirements.
  return kDataWordCount + (slot * 5 + field) * 16;
}

class PeerSdmaSystemTest : public GpuPeerDeviceFixture {
 protected:
  amdf_status_t MatchGpuPair(amdf_endpoint_t* primary, amdf_endpoint_t* peer,
                             bool* out_matches) override {
    bool primary_matches = false;
    const auto status = FindGpuQueueFamily(api_, primary, kRequirements,
                                           &families_[0], &primary_matches);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (!primary_matches) {
      *out_matches = false;
      return AMDF_STATUS_OK;
    }
    return FindGpuQueueFamily(api_, peer, kRequirements, &families_[1],
                              out_matches);
  }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(GpuPeerDeviceFixture::SetUp());
    if (IsSkipped()) {
      return;
    }
    devices_ = {device_, peer_device_};
    for (size_t i = 0; i < 2; ++i) {
      accesses_[i].device = devices_[i];
      accesses_[i].requirements.access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
      accesses_[i].requirements.flags =
          AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
      accesses_[i].requirements.address_kinds = UINT64_C(1)
                                                << AMDF_MEMORY_ADDRESS_GPU;
      RecordProperty("sdma_format_features_" + std::to_string(i),
                     std::to_string(families_[i].format_features));
    }
  }

  void SelectBacking() {
    amdf_memory_scope_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &info),
              AMDF_STATUS_OK);
    amdf_memory_profile_t selected = {};
    selected.ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t ordinal = 0; ordinal < info.memory_profile_count; ++ordinal) {
      amdf_memory_profile_t profile = {};
      profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      profile.structure_size = sizeof(profile);
      std::array<amdf_memory_access_capabilities_t, 2> capabilities = {};
      for (auto& capability : capabilities) {
        capability.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
        capability.structure_size = sizeof(capability);
      }
      const auto status = api_->memory_scope_query_device_profile(
          system_scope_, ordinal, accesses_.size(), accesses_.data(), &profile,
          capabilities.data());
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      constexpr auto roles =
          AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
      if ((profile.roles & roles) == roles &&
          (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE)) {
        selected = profile;
        break;
      }
    }
    if (selected.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
      GTEST_SKIP()
          << "no jointly accessible coherent SYSTEM construction profile";
    }
    creation_.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    creation_.structure_size = sizeof(creation_);
    creation_.memory_profile_ordinal = selected.ordinal;
    creation_.access_count = accesses_.size();
    creation_.accesses = accesses_.data();
    creation_.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    creation_.minimum_alignment = selected.allocation.minimum_alignment;
    const uint64_t granularity = selected.allocation.byte_length_granularity;
    ASSERT_NE(granularity, 0u);
    creation_.byte_length =
        (kBackingWordCount * sizeof(uint32_t) + granularity - 1) / granularity *
        granularity;
    ASSERT_LE(creation_.byte_length, selected.allocation.maximum_byte_length);
  }

  void QueryProspectivePair(int from, int to,
                            amdf_memory_pair_info_t* prospective) {
    amdf_memory_profile_pair_query_t query = {};
    query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
    query.structure_size = sizeof(query);
    query.memory_profile_ordinal = creation_.memory_profile_ordinal;
    query.required_flags = creation_.required_flags;
    query.access_count = accesses_.size();
    query.accesses = accesses_.data();
    query.producer.kind =
        from < 0 ? AMDF_MEMORY_SITE_KIND_HOST : AMDF_MEMORY_SITE_KIND_DEVICE;
    query.consumer.kind =
        to < 0 ? AMDF_MEMORY_SITE_KIND_HOST : AMDF_MEMORY_SITE_KIND_DEVICE;
    if (from < 0) {
      query.producer.value.host_access = AMDF_MEMORY_MAP_FLAG_WRITE;
    } else {
      query.producer.value.device = {static_cast<uint32_t>(from),
                                     families_[from].ordinal};
    }
    if (to < 0) {
      query.consumer.value.host_access = AMDF_MEMORY_MAP_FLAG_READ;
    } else {
      query.consumer.value.device = {static_cast<uint32_t>(to),
                                     families_[to].ordinal};
    }
    prospective->type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    prospective->structure_size = sizeof(*prospective);
    const auto status =
        api_->memory_scope_query_pair_info(system_scope_, &query, prospective);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "directional SYSTEM visibility is not advertised";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    if (prospective->release.kind != AMDF_CACHE_TRANSITION_KIND_NONE ||
        prospective->acquire.kind != AMDF_CACHE_TRANSITION_KIND_NONE) {
      GTEST_SKIP() << "SYSTEM pair requires payload cache operations";
    }
    ASSERT_NO_FATAL_FAILURE(CheckTransitions(*prospective));
  }

  void CheckTransitions(const amdf_memory_pair_info_t& pair) {
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_EQ(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(pair.release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
    ASSERT_EQ(pair.acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  }

  void CheckConcretePair(int from, int to,
                         const amdf_memory_pair_info_t& prospective) {
    const auto source =
        from < 0 ? backing_.HostSite()
                 : backing_.DeviceSite(from, families_[from].ordinal);
    const auto target = to < 0 ? backing_.HostSite()
                               : backing_.DeviceSite(to, families_[to].ordinal);
    amdf_memory_pair_info_t concrete = {};
    concrete.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    concrete.structure_size = sizeof(concrete);
    ASSERT_EQ(api_->memory_query_pair_info(&source, &target, &concrete),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(CheckTransitions(concrete));
    ASSERT_EQ(concrete.flags, prospective.flags);
    ASSERT_EQ(concrete.atomic_reach.scope_32,
              prospective.atomic_reach.scope_32);
    ASSERT_EQ(concrete.atomic_reach.scope_64,
              prospective.atomic_reach.scope_64);
  }

  void InitializeWork() {
    ASSERT_NO_FATAL_FAILURE(SelectBacking());
    if (IsSkipped()) {
      return;
    }
    std::array<amdf_memory_pair_info_t, 9> prospective = {};
    for (int from = -1; from < 2; ++from) {
      for (int to = -1; to < 2; ++to) {
        SCOPED_TRACE(::testing::Message() << from << " -> " << to);
        ASSERT_NO_FATAL_FAILURE(QueryProspectivePair(
            from, to, &prospective[(from + 1) * 3 + to + 1]));
        if (IsSkipped()) {
          return;
        }
      }
    }
    ASSERT_NO_FATAL_FAILURE(backing_.Create(api_, system_scope_, creation_));
    ASSERT_EQ(backing_.info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
    ASSERT_EQ(backing_.info.access_count, 2u);
    ASSERT_EQ(backing_.host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
    for (size_t i = 0; i < gpu_addresses_.size(); ++i) {
      ASSERT_EQ(
          api_->memory_query_address(
              backing_.memory, i, AMDF_MEMORY_ADDRESS_GPU, &gpu_addresses_[i]),
          AMDF_STATUS_OK);
    }
    for (int from = -1; from < 2; ++from) {
      for (int to = -1; to < 2; ++to) {
        SCOPED_TRACE(::testing::Message() << from << " -> " << to);
        ASSERT_NO_FATAL_FAILURE(
            CheckConcretePair(from, to, prospective[(from + 1) * 3 + to + 1]));
      }
    }
    RecordProperty("queried_pairs", 9);
    for (size_t i = 0; i < queues_.size(); ++i) {
      ASSERT_NO_FATAL_FAILURE(queues_[i].Initialize(
          api_, gpu_api_, devices_[i], system_scope_, families_[i],
          AMDF_QUEUE_PUBLICATION_MODE_USER));
      std::fill(queues_[i].words().begin(), queues_[i].words().end(), 0u);
    }
    ASSERT_FALSE(amdf_device_id_is_equal(&queues_[0].device_id(),
                                         &queues_[1].device_id()));
  }

  void RunBatch(uint32_t initiator) {
    SCOPED_TRACE(initiator);
    const uint32_t peer = 1 - initiator;
    std::vector<uint32_t> expected(creation_.byte_length / sizeof(uint32_t));
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] = 0xa5397dc1u ^ (initiator * 0x1634a593u + i * 0x1739050bu);
    }
    for (size_t slot = 0; slot < kSlotCount; ++slot) {
      for (auto field :
           {kArrival, kInitiatorReady, kPeerReady, kInitiatorDone, kPeerDone}) {
        expected[ControlWord(slot, field)] = 0;
      }
    }
    std::memcpy(backing_.host.pointer, expected.data(), creation_.byte_length);
    const auto data = [&](uint32_t device, size_t slot, size_t region) {
      return gpu_addresses_[device] + DataWord(slot, region) * sizeof(uint32_t);
    };
    const auto control = [&](uint32_t device, size_t slot, Control field) {
      return gpu_addresses_[device] +
             ControlWord(slot, field) * sizeof(uint32_t);
    };
    SdmaCommandWriter first(
        queues_[initiator].words().data() + word_frontiers_[initiator],
        families_[initiator].format_features);
    SdmaCommandWriter second(
        queues_[peer].words().data() + word_frontiers_[peer],
        families_[peer].format_features);
    // These two finite streams fit with a free ring word; no wrap/reuse occurs.
    ASSERT_LT(word_frontiers_[initiator] + kSlotCount * kInitiatorWordsPerSlot,
              queues_[initiator].words().size());
    ASSERT_LT(word_frontiers_[peer] + kSlotCount * kPeerWordsPerSlot,
              queues_[peer].words().size());
    for (size_t slot = 0; slot < kSlotCount; ++slot) {
      first.WaitMemory32(control(initiator, slot, kArrival), 1);
      first.CopyLinear(data(initiator, slot, kInput),
                       data(initiator, slot, kFirstIntermediate),
                       kCopyWordCount * sizeof(uint32_t));
      first.Fence32(control(initiator, slot, kInitiatorReady), 1);
      first.WaitMemory32(control(initiator, slot, kPeerReady), 1);
      first.CopyLinear(data(initiator, slot, kSecondIntermediate),
                       data(initiator, slot, kOutput),
                       kCopyWordCount * sizeof(uint32_t));
      first.Fence32(control(initiator, slot, kInitiatorDone), 1);
      second.Fence32(control(peer, slot, kArrival), 1);
      second.WaitMemory32(control(peer, slot, kInitiatorReady), 1);
      second.CopyLinear(data(peer, slot, kFirstIntermediate),
                        data(peer, slot, kSecondIntermediate),
                        kCopyWordCount * sizeof(uint32_t));
      second.Fence32(control(peer, slot, kPeerReady), 1);
      second.WaitMemory32(control(peer, slot, kInitiatorDone), 1);
      second.Fence32(control(peer, slot, kPeerDone), 1);
    }
    ASSERT_EQ(first.word_count(), kSlotCount * kInitiatorWordsPerSlot);
    ASSERT_EQ(second.word_count(), kSlotCount * kPeerWordsPerSlot);
    word_frontiers_[initiator] += first.word_count();
    word_frontiers_[peer] += second.word_count();
    std::array<std::vector<uint32_t>, 2> expected_commands;
    for (size_t i = 0; i < queues_.size(); ++i) {
      expected_commands[i].assign(queues_[i].words().begin(),
                                  queues_[i].words().end());
    }
    ASSERT_NO_FATAL_FAILURE(
        queues_[peer].Publish(api_, gpu_api_, word_frontiers_[peer]));
    const uintptr_t host = reinterpret_cast<uintptr_t>(backing_.host.pointer);
    GpuWaitEqual<uint32_t>(host + ControlWord(0, kArrival) * sizeof(uint32_t),
                           1);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(host + ControlWord(0, kInitiatorReady) *
                                                  sizeof(uint32_t)),
              0u);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(host + ControlWord(0, kPeerDone) *
                                                  sizeof(uint32_t)),
              0u);
    // Publishing the producer resolves the pending first dependency. There is
    // no host participation in any stage or slot dependency after this point.
    ASSERT_NO_FATAL_FAILURE(
        queues_[initiator].Publish(api_, gpu_api_, word_frontiers_[initiator]));
    GpuWaitEqual<uint32_t>(
        host + ControlWord(kSlotCount - 1, kPeerDone) * sizeof(uint32_t), 1);
    std::vector<uint32_t> observed(expected.size());
    std::memcpy(observed.data(), backing_.host.pointer, creation_.byte_length);
    std::array<std::vector<uint32_t>, 2> observed_commands;
    for (size_t i = 0; i < queues_.size(); ++i) {
      observed_commands[i].assign(queues_[i].words().begin(),
                                  queues_[i].words().end());
    }
    for (size_t slot = 0; slot < kSlotCount; ++slot) {
      for (size_t region = kFirstIntermediate; region < kRegionCount;
           ++region) {
        for (size_t word = 0; word < kCopyWordCount; ++word) {
          expected[DataWord(slot, region) + word] =
              expected[DataWord(slot, kInput) + word];
        }
      }
      for (auto field :
           {kArrival, kInitiatorReady, kPeerReady, kInitiatorDone, kPeerDone}) {
        expected[ControlWord(slot, field)] = 1;
      }
    }
    EXPECT_EQ(observed, expected);
    EXPECT_EQ(observed_commands, expected_commands);
    for (auto& queue : queues_) {
      EXPECT_NO_FATAL_FAILURE(queue.WaitRetired(api_));
    }
  }

  void TearDown() override {
    for (auto& queue : queues_) {
      ASSERT_TRUE(queue.Release(api_));
    }
    ASSERT_TRUE(backing_.Release(api_));
    GpuPeerDeviceFixture::TearDown();
  }

  // Borrowed cached devices, independently selected physical endpoints.
  std::array<amdf_device_t*, 2> devices_ = {};
  // Exact endpoint-local command families.
  std::array<amdf_queue_family_info_t, 2> families_ = {};
  // Immutable caller-ordered access set for the one shared backing.
  std::array<amdf_memory_device_access_t, 2> accesses_ = {};
  // The actual construction request used by prospective pair queries.
  amdf_memory_create_info_t creation_ = {};
  // Owned shared allocation and explicit coherent host view.
  CtsMappedMemory backing_;
  // Device-specific addresses from the actual access ordinals.
  std::array<uint64_t, 2> gpu_addresses_ = {};
  // One case-owned command queue per physical GPU.
  std::array<GpuCommandQueue, 2> queues_;
  // Cumulative complete command DWORDs, converted by Publish to native bytes.
  std::array<size_t, 2> word_frontiers_ = {};
};

TEST_F(PeerSdmaSystemTest, DeviceDrivenRoundTrips) {
  ASSERT_NO_FATAL_FAILURE(InitializeWork());
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(RunBatch(0));
  if (HasFailure()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(RunBatch(1));
  if (HasFailure()) {
    return;
  }
  RecordProperty("completed_round_trips", 8);
  RecordProperty("peer_payload_edges", 16);
  RecordProperty("copied_bytes", 24 * kCopyWordCount * sizeof(uint32_t));
  RecordProperty("checked_backing_bytes",
                 std::to_string(2 * creation_.byte_length));
  RecordProperty(
      "checked_command_bytes",
      static_cast<int>(2 *
                       (queues_[0].words().size() + queues_[1].words().size()) *
                       sizeof(uint32_t)));
  RecordProperty("retired_frontier_bytes_0",
                 static_cast<int>(word_frontiers_[0] * 4));
  RecordProperty("retired_frontier_bytes_1",
                 static_cast<int>(word_frontiers_[1] * 4));
}

}  // namespace
