// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "libamdf/cts/gpu/aql/executable.h"
#include "libamdf/cts/gpu/aql/publication.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/peer/device_fixture.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

constexpr GpuQueueRequirements kRequirements = {
    .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
    .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_CACHE_CONTROL,
    .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                        AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
    .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
    .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
};
constexpr size_t kMinimumWordCount = 4096;
constexpr size_t kOutputOffset = 16;
constexpr uint32_t kProducerOffset = 1023;
constexpr uint32_t kProducerCount = 257;
constexpr uint8_t kControlGuard = 0xa7;
enum SignalIndex { kArrival, kProduced, kConsumed, kInitialized, kSignalCount };
constexpr aql::FenceScopes kNoFences = {aql::FenceScope::kNone,
                                        aql::FenceScope::kNone};

struct JointMemory {
  JointMemory() = default;
  JointMemory(const JointMemory&) = delete;
  JointMemory& operator=(const JointMemory&) = delete;

  // Borrowed scope establishing physical placement.
  amdf_memory_scope_t* scope = nullptr;
  // Caller-ordered live consumers, retained for prospective query inputs.
  std::array<amdf_memory_device_access_t, 2> accesses = {};
  // Exact selected construction contract.
  amdf_memory_create_info_t creation = {};
  // Case-owned shared native allocation.
  amdf_memory_t* memory = nullptr;
  // Achieved backing properties.
  amdf_memory_info_t info = {};
  // Optional case-owned host mapping; never created for LOCAL memory.
  amdf_host_mapping_t* mapping = nullptr;
  // Borrowed mapped host extent.
  amdf_host_mapping_info_t host = {};
  // Stable addresses, independently queried for both consumers.
  std::array<uint64_t, 2> addresses = {};

  amdf_memory_site_t Site(int device, uint32_t family = 0) const {
    amdf_memory_site_t result = {};
    result.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE;
    result.structure_size = sizeof(result);
    result.kind =
        device < 0 ? AMDF_MEMORY_SITE_KIND_HOST : AMDF_MEMORY_SITE_KIND_DEVICE;
    if (device < 0) {
      result.value.host_mapping = mapping;
    } else {
      result.value.device = {memory, static_cast<uint32_t>(device), family};
    }
    return result;
  }

  bool Release(const amdf_api_t* api) {
    if (mapping) {
      const auto status =
          api->host_mapping_destroy(std::exchange(mapping, nullptr));
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (!amdf_status_is_ok(status)) {
        return false;
      }
    }
    if (memory) {
      const auto status = api->memory_destroy(std::exchange(memory, nullptr));
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (!amdf_status_is_ok(status)) {
        return false;
      }
    }
    return true;
  }
};

class PeerAqlMemoryTest : public GpuPeerDeviceFixture {
 protected:
  amdf_status_t MatchGpuPair(amdf_endpoint_t* primary, amdf_endpoint_t* peer,
                             bool* out_matches) override {
    bool matches = false;
    const auto status = FindGpuQueueFamily(api_, primary, kRequirements,
                                           &families_[0], &matches);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (!matches) {
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
    endpoints_ = {gpu_endpoint_info_, peer_gpu_endpoint_info_};
  }

  void CheckTransition(const amdf_cache_transition_t& transition, int device,
                       amdf_cache_operation_t operation) {
    ASSERT_EQ(transition.kind, device < 0 ? AMDF_CACHE_TRANSITION_KIND_NONE
                                          : AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    ASSERT_EQ(transition.executor, device < 0
                                       ? AMDF_CACHE_TRANSITION_EXECUTOR_NONE
                                       : AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
    if (device >= 0) {
      ASSERT_EQ(transition.operation, operation);
    }
  }

  void QueryPair(JointMemory& memory, int from, int to,
                 amdf_memory_pair_info_t* prospective) {
    amdf_memory_profile_pair_query_t query = {};
    query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
    query.structure_size = sizeof(query);
    query.memory_profile_ordinal = memory.creation.memory_profile_ordinal;
    query.required_flags = memory.creation.required_flags;
    query.access_count = memory.accesses.size();
    query.accesses = memory.accesses.data();
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
        api_->memory_scope_query_pair_info(memory.scope, &query, prospective);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "directional peer visibility is not advertised";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    if ((from >= 0 &&
         prospective->release.kind != AMDF_CACHE_TRANSITION_KIND_GLOBAL) ||
        (to >= 0 &&
         prospective->acquire.kind != AMDF_CACHE_TRANSITION_KIND_GLOBAL) ||
        (from < 0 &&
         prospective->release.kind != AMDF_CACHE_TRANSITION_KIND_NONE) ||
        (to < 0 &&
         prospective->acquire.kind != AMDF_CACHE_TRANSITION_KIND_NONE)) {
      GTEST_SKIP()
          << "peer backing does not admit the SYSTEM-scoped AQL recipe";
    }
    ASSERT_NE(
        prospective->flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
        0u);
    ASSERT_NO_FATAL_FAILURE(CheckTransition(
        prospective->release, from, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));
    ASSERT_NO_FATAL_FAILURE(CheckTransition(
        prospective->acquire, to, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
  }

  void CreateJoint(amdf_memory_scope_t* scope, amdf_memory_class_t memory_class,
                   uint64_t byte_length, JointMemory* memory) {
    memory->scope = scope;
    const bool system = memory_class == AMDF_MEMORY_CLASS_SYSTEM;
    amdf_memory_scope_info_t scope_info = {};
    scope_info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
    scope_info.structure_size = sizeof(scope_info);
    ASSERT_EQ(api_->memory_scope_query_info(scope, &scope_info),
              AMDF_STATUS_OK);
    for (size_t i = 0; i < 2; ++i) {
      memory->accesses[i].device = devices_[i];
      memory->accesses[i].requirements = {
          .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
          .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                   (system ? AMDF_MEMORY_FLAG_HOST_COHERENT : UINT64_C(0)),
          .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU};
    }
    amdf_memory_profile_t selected = {};
    selected.ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t ordinal = 0; ordinal < scope_info.memory_profile_count;
         ++ordinal) {
      amdf_memory_profile_t profile = {};
      profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      profile.structure_size = sizeof(profile);
      std::array<amdf_memory_access_capabilities_t, 2> capabilities = {};
      for (auto& capability : capabilities) {
        capability.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
        capability.structure_size = sizeof(capability);
      }
      const auto status = api_->memory_scope_query_device_profile(
          scope, ordinal, 2, memory->accesses.data(), &profile,
          capabilities.data());
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      const auto roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                         (system ? AMDF_MEMORY_PROFILE_ROLE_HOST_MAP : 0);
      if (profile.memory_class == memory_class &&
          (profile.roles & roles) == roles &&
          (!system ||
           (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE))) {
        selected = profile;
        break;
      }
    }
    if (selected.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
      GTEST_SKIP() << "no joint peer construction profile for memory class "
                   << memory_class;
    }
    auto& creation = memory->creation;
    creation.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    creation.structure_size = sizeof(creation);
    creation.memory_profile_ordinal = selected.ordinal;
    creation.access_count = 2;
    creation.accesses = memory->accesses.data();
    creation.required_flags =
        system ? AMDF_MEMORY_FLAG_HOST_VISIBLE : AMDF_MEMORY_FLAG_DEVICE_LOCAL;
    creation.minimum_alignment = selected.allocation.minimum_alignment;
    const uint64_t granularity = selected.allocation.byte_length_granularity;
    ASSERT_NE(granularity, 0u);
    creation.byte_length =
        (byte_length + granularity - 1) / granularity * granularity;
    ASSERT_LE(creation.byte_length, selected.allocation.maximum_byte_length);
    std::array<amdf_memory_pair_info_t, 9> pairs = {};
    for (int from = system ? -1 : 0; from < 2; ++from) {
      for (int to = system ? -1 : 0; to < 2; ++to) {
        SCOPED_TRACE(::testing::Message()
                     << memory_class << ": " << from << " -> " << to);
        ASSERT_NO_FATAL_FAILURE(
            QueryPair(*memory, from, to, &pairs[(from + 1) * 3 + to + 1]));
        if (IsSkipped()) {
          return;
        }
      }
    }
    ASSERT_EQ(api_->memory_create(scope, &creation, &memory->memory),
              AMDF_STATUS_OK);
    memory->info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    memory->info.structure_size = sizeof(memory->info);
    ASSERT_EQ(api_->memory_query_info(memory->memory, &memory->info),
              AMDF_STATUS_OK);
    ASSERT_EQ(memory->info.memory_class, memory_class);
    ASSERT_EQ(memory->info.access_count, 2u);
    ASSERT_EQ(memory->info.byte_length, creation.byte_length);
    if (system) {
      amdf_memory_map_info_t map = {};
      map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
      map.structure_size = sizeof(map);
      map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
      map.byte_length = creation.byte_length;
      ASSERT_EQ(api_->memory_map(memory->memory, &map, &memory->mapping),
                AMDF_STATUS_OK);
      memory->host.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
      memory->host.structure_size = sizeof(memory->host);
      ASSERT_EQ(api_->host_mapping_query_info(memory->mapping, &memory->host),
                AMDF_STATUS_OK);
      ASSERT_EQ(memory->host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
      ASSERT_EQ(memory->host.byte_length, creation.byte_length);
    } else {
      ASSERT_EQ(memory->info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
      ASSERT_EQ(memory->mapping, nullptr);
    }
    for (uint32_t i = 0; i < 2; ++i) {
      ASSERT_EQ(
          api_->memory_query_address(memory->memory, i, AMDF_MEMORY_ADDRESS_GPU,
                                     &memory->addresses[i]),
          AMDF_STATUS_OK);
    }
    for (int from = system ? -1 : 0; from < 2; ++from) {
      for (int to = system ? -1 : 0; to < 2; ++to) {
        const auto source =
            memory->Site(from, from < 0 ? 0 : families_[from].ordinal);
        const auto target =
            memory->Site(to, to < 0 ? 0 : families_[to].ordinal);
        amdf_memory_pair_info_t concrete = {};
        concrete.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
        concrete.structure_size = sizeof(concrete);
        ASSERT_EQ(api_->memory_query_pair_info(&source, &target, &concrete),
                  AMDF_STATUS_OK);
        const auto& prospective = pairs[(from + 1) * 3 + to + 1];
        ASSERT_NO_FATAL_FAILURE(CheckTransition(
            concrete.release, from, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));
        ASSERT_NO_FATAL_FAILURE(CheckTransition(
            concrete.acquire, to, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
        ASSERT_EQ(concrete.flags, prospective.flags);
        ASSERT_EQ(concrete.atomic_reach.scope_32,
                  prospective.atomic_reach.scope_32);
        ASSERT_EQ(concrete.atomic_reach.scope_64,
                  prospective.atomic_reach.scope_64);
        ++queried_pairs_;
      }
    }
  }

  void InitializeControl() {
    ASSERT_NO_FATAL_FAILURE(
        CreateJoint(system_scope_, AMDF_MEMORY_CLASS_SYSTEM, 4096, &control_));
    if (IsSkipped()) {
      return;
    }
    for (size_t i = 0; i < 2; ++i) {
      ASSERT_EQ(control_.addresses[i] % alignof(aql::Signal), 0u);
      ASSERT_NE(control_.addresses[i],
                0u);  // Zero is a null native signal operand.
      ASSERT_NO_FATAL_FAILURE(
          queues_[i].Initialize(api_, gpu_api_, devices_[i], families_[i],
                                AMDF_QUEUE_PRODUCER_MODE_SINGLE, {}));
      RecordProperty("signal_gpu_base_" + std::to_string(i),
                     std::to_string(control_.addresses[i]));
    }
    ASSERT_FALSE(amdf_device_id_is_equal(&queues_[0].info.device_id,
                                         &queues_[1].info.device_id));
  }

  void ResetSignals() {
    std::memset(control_.host.pointer, kControlGuard,
                control_.host.byte_length);
    auto* signals = static_cast<aql::Signal*>(control_.host.pointer);
    for (size_t i = 0; i < kSignalCount; ++i) {
      signals[i] = {};
      signals[i].kind = 1;
      signals[i].value = 1;
    }
  }

  uint64_t SignalAddress(uint32_t device, SignalIndex signal) const {
    return control_.addresses[device] + signal * sizeof(aql::Signal);
  }

  uint64_t HostSignal(SignalIndex signal) const {
    return reinterpret_cast<uintptr_t>(control_.host.pointer) +
           signal * sizeof(aql::Signal) + offsetof(aql::Signal, value);
  }

  void Publish(uint32_t device, const aql::Packet& packet) {
    const uint64_t index = indices_[device]++;
    GpuStoreRelease(queues_[device].host.write_index_address, index + 1);
    aql::Publish(queues_[device], index, packet);
  }

  void Retire() {
    for (size_t i = 0; i < 2; ++i) {
      EXPECT_NO_FATAL_FAILURE(queues_[i].WaitConsumed(api_, indices_[i]));
      amdf_user_queue_status_t status = {};
      status.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS;
      status.structure_size = sizeof(status);
      EXPECT_EQ(api_->user_queue_query_status(queues_[i].queue, &status),
                AMDF_STATUS_OK);
      EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
      EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
      EXPECT_EQ(status.producer_index, indices_[i]);
      EXPECT_EQ(status.consumed_index, indices_[i]);
    }
  }

  void CheckControl(int64_t initialized_value) {
    std::vector<uint8_t> snapshot(control_.host.byte_length);
    std::memcpy(snapshot.data(), control_.host.pointer, snapshot.size());
    for (size_t i = 0; i < kSignalCount; ++i) {
      aql::Signal signal;
      std::memcpy(&signal, snapshot.data() + i * sizeof(signal),
                  sizeof(signal));
      EXPECT_EQ(signal.kind, 1);
      EXPECT_EQ(signal.value, i == kInitialized ? initialized_value : 0);
      // Dispatch timestamps are native outputs, not unused guard bytes.
      for (size_t field : {0u, 1u, 4u, 5u}) {
        EXPECT_EQ(signal.reserved[field], 0u);
      }
    }
    for (size_t i = kSignalCount * sizeof(aql::Signal); i < snapshot.size();
         ++i) {
      EXPECT_EQ(snapshot[i], kControlGuard) << "control byte " << i;
    }
  }

  void CreateSingle(uint32_t device, amdf_memory_access_t access,
                    uint64_t byte_length, GpuMemory* memory) {
    const amdf_memory_device_access_t attachment = {
        devices_[device],
        {.access = access,
         .flags =
             AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.memory_profile_ordinal = FindGpuMemoryProfileOrdinal(
        api_, system_scope_, devices_[device],
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        AMDF_MEMORY_FLAG_HOST_VISIBLE, attachment.requirements);
    ASSERT_NE(create.memory_profile_ordinal,
              AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.access_count = 1;
    create.accesses = &attachment;
    create.byte_length = byte_length;
    create.minimum_alignment = 4096;
    ASSERT_NO_FATAL_FAILURE(memory->Initialize(api_, system_scope_, create));
  }

  void PublishImages() {
    for (uint32_t i = 0; i < 2; ++i) {
      const auto* product = kernels::transform::kKernels.Find(endpoints_[i]);
      ASSERT_NE(product, nullptr);
      kernels_[i] = product;
      const auto& kernel = *product;
      ASSERT_EQ(kernel.private_segment_byte_length, 0u);
      ASSERT_EQ(kernel.workgroup_size(), 64u);
      ASSERT_EQ(kernel.arguments.byte_length, 24u);
      ASSERT_TRUE(std::equal(kernel.arguments.byte_offsets.begin(),
                             kernel.arguments.byte_offsets.end(),
                             kernels::transform::kArgumentByteOffsets.begin(),
                             kernels::transform::kArgumentByteOffsets.end()));
      ASSERT_TRUE(endpoints_[i].gfx_ip.major == 9 ||
                  Pm4CommandProfile::Find(endpoints_[i]) != nullptr);
      ASSERT_NO_FATAL_FAILURE(
          CreateSingle(i, AMDF_MEMORY_ACCESS_READ, 4096, &arguments_[i]));
      ASSERT_EQ(arguments_[i].device_address % kernel.arguments.alignment, 0u);
      const auto prefix = "kernel_" + std::to_string(i);
      ASSERT_NO_FATAL_FAILURE(executables_[i].Initialize(
          api_, system_scope_, devices_[i], endpoints_[i], kernel,
          prefix.c_str(), queues_[i], &indices_[i]));
      RecordProperty(prefix + "_target", kernel.target);
      RecordProperty(prefix + "_hsaco_sha256", kernel.hsaco_sha256);
    }
  }

  void InitializePayload(uint32_t owner) {
    uint32_t scope_count = 0;
    const auto query_status = api_->device_enumerate_memory_scopes(
        devices_[owner], 0, nullptr, &scope_count);
    ASSERT_EQ(query_status,
              scope_count == 0
                  ? AMDF_STATUS_OK
                  : amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
    std::vector<amdf_memory_scope_t*> scopes(scope_count);
    ASSERT_EQ(api_->device_enumerate_memory_scopes(devices_[owner], scope_count,
                                                   scopes.data(), &scope_count),
              AMDF_STATUS_OK);
    amdf_memory_scope_t* selected = nullptr;
    for (auto* scope : scopes) {
      amdf_memory_scope_info_t info = {};
      info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
      info.structure_size = sizeof(info);
      ASSERT_EQ(api_->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
      if (info.kind == AMDF_MEMORY_SCOPE_KIND_LOCAL) {
        const auto& endpoint =
            owner == 0 ? *GetCtsDeviceCache().gpu_endpoint_id()
                       : *GetCtsDeviceCache().gpu_peer_endpoint_id();
        ASSERT_TRUE(
            amdf_endpoint_id_is_equal(&info.physical_endpoint_id, &endpoint));
        selected = scope;
      }
    }
    if (selected == nullptr) {
      GTEST_SKIP() << "selected LOCAL memory owner has no LOCAL scope";
    }
    ASSERT_NO_FATAL_FAILURE(CreateJoint(selected, AMDF_MEMORY_CLASS_LOCAL,
                                        kMinimumWordCount * 4, &local_memory_));
    if (IsSkipped()) {
      return;
    }
    ASSERT_EQ(local_memory_.info.byte_length % sizeof(uint32_t), 0u);
    ASSERT_LE(local_memory_.info.byte_length / sizeof(uint32_t),
              INT32_MAX - 63);
    word_count_ = static_cast<uint32_t>(local_memory_.info.byte_length /
                                        sizeof(uint32_t));
    ASSERT_NO_FATAL_FAILURE(CreateJoint(system_scope_, AMDF_MEMORY_CLASS_SYSTEM,
                                        local_memory_.info.byte_length,
                                        &input_));
    if (IsSkipped()) {
      return;
    }
    ASSERT_NO_FATAL_FAILURE(CreateJoint(
        system_scope_, AMDF_MEMORY_CLASS_SYSTEM,
        local_memory_.info.byte_length + 2 * kOutputOffset * sizeof(uint32_t),
        &output_));
    if (IsSkipped()) {
      return;
    }
    ASSERT_EQ(input_.info.byte_length % sizeof(uint32_t), 0u);
    ASSERT_EQ(output_.info.byte_length % sizeof(uint32_t), 0u);
    RecordProperty("local_memory_byte_length",
                   std::to_string(local_memory_.info.byte_length));
    for (size_t i = 0; i < 2; ++i) {
      RecordProperty("local_memory_gpu_base_" + std::to_string(i),
                     std::to_string(local_memory_.addresses[i]));
    }
  }

  void SetArguments(uint32_t device, size_t offset,
                    const kernels::transform::Arguments& arguments) {
    ASSERT_EQ(offset % kernels_[device]->arguments.alignment, 0u);
    ASSERT_LE(offset + sizeof(arguments), arguments_[device].info.byte_length);
    std::memcpy(static_cast<uint8_t*>(arguments_[device].host.pointer) + offset,
                &arguments, kernels_[device]->arguments.byte_length);
  }

  aql::Packet Dispatch(uint32_t device, size_t argument_offset, uint32_t count,
                       SignalIndex completion) const {
    const auto& kernel = *kernels_[device];
    const uint32_t width = kernel.workgroup_size();
    return aql::Dispatch(aql::HeaderBarrier::kDisabled,
                         {1,
                          {static_cast<uint16_t>(width), 1, 1},
                          {(count + width - 1) / width * width, 1, 1}},
                         kernel.private_segment_byte_length,
                         kernel.group_segment_byte_length,
                         executables_[device].descriptor_address(),
                         arguments_[device].device_address + argument_offset,
                         SignalAddress(device, completion));
  }

  void RunGeneration(uint32_t owner, uint32_t producer, uint32_t generation) {
    SCOPED_TRACE(::testing::Message()
                 << "owner=" << owner << " producer=" << producer
                 << " generation=" << generation);
    const uint32_t consumer = 1 - producer;
    std::vector<uint32_t> input(input_.info.byte_length / sizeof(uint32_t));
    std::vector<uint32_t> expected(output_.info.byte_length / sizeof(uint32_t),
                                   0x68d329b7u);
    const uint32_t initialize_addend = 0x21345687u + generation * 0x11234567u;
    const uint32_t produce_addend = 0x82345679u + generation * 0x23456789u;
    const uint32_t consume_addend = 0x13579bdfu + generation * 0x31234567u;
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = 0xfffffff0u + i * 0x01030507u + generation * 0x11111111u;
    }
    for (size_t i = 0; i < word_count_; ++i) {
      const bool changed =
          i >= kProducerOffset && i < kProducerOffset + kProducerCount;
      const uint32_t first_addend =
          changed ? produce_addend : initialize_addend;
      expected[kOutputOffset + i] = static_cast<uint32_t>(
          uint64_t{input[i]} * 9 + uint64_t{first_addend} * 3 + consume_addend);
    }
    std::memcpy(input_.host.pointer, input.data(), input_.info.byte_length);
    auto initial_output = expected;
    for (size_t i = 0; i < word_count_; ++i) {
      initial_output[kOutputOffset + i] ^= UINT32_MAX;
    }
    std::memcpy(output_.host.pointer, initial_output.data(),
                output_.info.byte_length);
    for (auto& arguments : arguments_) {
      std::memset(arguments.host.pointer, 0, arguments.info.byte_length);
    }
    ResetSignals();
    ASSERT_NO_FATAL_FAILURE(
        SetArguments(owner, 0,
                     {input_.addresses[owner], local_memory_.addresses[owner],
                      word_count_, initialize_addend}));
    // Initialize all LOCAL memory through the owner shader before the tested
    // peer edge.
    ASSERT_NO_FATAL_FAILURE(
        Publish(owner, Dispatch(owner, 0, word_count_, kInitialized)));
    GpuWaitEqual<int64_t>(HostSignal(kInitialized), 0);
    ASSERT_NO_FATAL_FAILURE(queues_[owner].WaitConsumed(api_, indices_[owner]));
    // Initialization kernargs stay immutable. Work uses the next aligned slot.
    ASSERT_NO_FATAL_FAILURE(
        SetArguments(producer, 64,
                     {input_.addresses[producer] + kProducerOffset * 4,
                      local_memory_.addresses[producer] + kProducerOffset * 4,
                      kProducerCount, produce_addend}));
    ASSERT_NO_FATAL_FAILURE(
        SetArguments(consumer, 64,
                     {local_memory_.addresses[consumer],
                      output_.addresses[consumer] + kOutputOffset * 4,
                      word_count_, consume_addend}));
    std::array<std::vector<uint8_t>, 2> expected_arguments;
    for (size_t i = 0; i < 2; ++i) {
      const auto* bytes =
          static_cast<const uint8_t*>(arguments_[i].host.pointer);
      expected_arguments[i].assign(bytes,
                                   bytes + arguments_[i].info.byte_length);
    }
    ASSERT_NO_FATAL_FAILURE(Publish(
        consumer,
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                     SignalAddress(consumer, kArrival), {}, kNoFences)));
    ASSERT_NO_FATAL_FAILURE(Publish(
        consumer,
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
                     {SignalAddress(consumer, kProduced)}, kNoFences)));
    ASSERT_NO_FATAL_FAILURE(
        Publish(consumer, Dispatch(consumer, 64, word_count_, kConsumed)));
    GpuWaitEqual<int64_t>(HostSignal(kArrival), 0);
    EXPECT_EQ(GpuLoadAcquire<int64_t>(HostSignal(kProduced)), 1);
    EXPECT_EQ(GpuLoadAcquire<int64_t>(HostSignal(kConsumed)), 1);
    ASSERT_NO_FATAL_FAILURE(
        Publish(producer, Dispatch(producer, 64, kProducerCount, kProduced)));
    // No host dependency or payload relay follows producer publication.
    GpuWaitEqual<int64_t>(HostSignal(kConsumed), 0);
    std::vector<uint32_t> observed(expected.size()),
        unchanged_input(input.size());
    std::memcpy(observed.data(), output_.host.pointer,
                output_.info.byte_length);
    std::memcpy(unchanged_input.data(), input_.host.pointer,
                input_.info.byte_length);
    GpuWaitEqual<int64_t>(HostSignal(kProduced), 0);
    std::array<std::vector<uint8_t>, 2> observed_arguments;
    for (size_t i = 0; i < 2; ++i) {
      const auto* bytes =
          static_cast<const uint8_t*>(arguments_[i].host.pointer);
      observed_arguments[i].assign(bytes,
                                   bytes + arguments_[i].info.byte_length);
    }
    EXPECT_EQ(observed, expected);
    EXPECT_EQ(unchanged_input, input);
    EXPECT_EQ(observed_arguments, expected_arguments);
    EXPECT_NO_FATAL_FAILURE(CheckControl(0));
    EXPECT_NO_FATAL_FAILURE(Retire());
  }

  void TearDown() override {
    for (auto& queue : queues_) {
      ASSERT_TRUE(queue.Release(api_));
    }
    for (auto& argument : arguments_) {
      ASSERT_TRUE(argument.Release(api_));
    }
    for (auto& executable : executables_) {
      ASSERT_TRUE(executable.Release(api_));
    }
    ASSERT_TRUE(local_memory_.Release(api_));
    ASSERT_TRUE(output_.Release(api_));
    ASSERT_TRUE(input_.Release(api_));
    ASSERT_TRUE(control_.Release(api_));
    GpuPeerDeviceFixture::TearDown();
  }

  // Cached native endpoint owners, retained through every workload resource.
  std::array<amdf_device_t*, 2> devices_ = {};
  // Exact physical target metadata, independently selecting each kernel image.
  std::array<amdf_gpu_endpoint_info_t, 2> endpoints_ = {};
  // Queried native command-family contracts.
  std::array<amdf_queue_family_info_t, 2> families_ = {};
  // One case-owned queue per physical endpoint.
  std::array<GpuUserQueue, 2> queues_;
  // Complete AQL packet reservation frontiers.
  std::array<uint64_t, 2> indices_ = {};
  // Borrowed immutable compiler products.
  std::array<const kernels::Kernel*, 2> kernels_ = {};
  // Independently published images retained through both queue removals.
  std::array<aql::Executable, 2> executables_;
  // Per-device immutable argument slots retained through the final reader.
  std::array<GpuMemory, 2> arguments_;
  // Jointly attached native signal storage with an explicit coherent host view.
  JointMemory control_;
  // Jointly attached host-written input and its immutable pattern.
  JointMemory input_;
  // Jointly attached host-readable output, including leading/trailing guards.
  JointMemory output_;
  // Device-only peer LOCAL memory; its owner is selected independently of the
  // producer.
  JointMemory local_memory_;
  // Complete allocation-rounded LOCAL memory extent processed by both shader
  // stages.
  uint32_t word_count_ = 0;
  // Number of matching prospective/concrete directed visibility queries.
  uint32_t queried_pairs_ = 0;
};

TEST_F(PeerAqlMemoryTest, NativeSignalsAcrossBothDevices) {
  ASSERT_NO_FATAL_FAILURE(InitializeControl());
  if (IsSkipped()) {
    return;
  }
  for (uint32_t producer = 0; producer < 2; ++producer) {
    SCOPED_TRACE(producer);
    const uint32_t consumer = 1 - producer;
    ResetSignals();
    ASSERT_NO_FATAL_FAILURE(Publish(
        consumer,
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                     SignalAddress(consumer, kArrival), {}, kNoFences)));
    ASSERT_NO_FATAL_FAILURE(Publish(
        consumer,
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                     SignalAddress(consumer, kConsumed),
                     {SignalAddress(consumer, kProduced)})));
    GpuWaitEqual<int64_t>(HostSignal(kArrival), 0);
    EXPECT_EQ(GpuLoadAcquire<int64_t>(HostSignal(kProduced)), 1);
    EXPECT_EQ(GpuLoadAcquire<int64_t>(HostSignal(kConsumed)), 1);
    ASSERT_NO_FATAL_FAILURE(Publish(
        producer,
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                     SignalAddress(producer, kProduced))));
    GpuWaitEqual<int64_t>(HostSignal(kConsumed), 0);
    GpuWaitEqual<int64_t>(HostSignal(kProduced), 0);
    EXPECT_NO_FATAL_FAILURE(CheckControl(1));
    EXPECT_NO_FATAL_FAILURE(Retire());
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("completed_signal_edges", 2);
  RecordProperty("queried_pairs", queried_pairs_);
  for (size_t i = 0; i < 2; ++i) {
    RecordProperty("retired_packets_" + std::to_string(i),
                   std::to_string(indices_[i]));
  }
}

class PeerAqlLocalMemoryTest
    : public PeerAqlMemoryTest,
      public ::testing::WithParamInterface<std::array<uint32_t, 2>> {};

INSTANTIATE_TEST_SUITE_P(Placement, PeerAqlLocalMemoryTest,
                         ::testing::Values(std::array<uint32_t, 2>{0, 0},
                                           std::array<uint32_t, 2>{0, 1},
                                           std::array<uint32_t, 2>{1, 0},
                                           std::array<uint32_t, 2>{1, 1}),
                         [](const auto& info) {
                           return "Owner" + std::to_string(info.param[0]) +
                                  "Producer" + std::to_string(info.param[1]);
                         });

TEST_P(PeerAqlLocalMemoryTest, TransformAcrossPhysicalDevices) {
  const auto [owner, producer] = GetParam();
  ASSERT_NO_FATAL_FAILURE(InitializeControl());
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(InitializePayload(owner));
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(PublishImages());
  for (uint32_t generation = 0; generation < 2; ++generation) {
    ASSERT_NO_FATAL_FAILURE(RunGeneration(owner, producer, generation));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("local_memory_owner", owner);
  RecordProperty("producer", producer);
  RecordProperty("completed_generations", 2);
  RecordProperty("queried_pairs", queried_pairs_);
  RecordProperty(
      "checked_output_words",
      std::to_string(2 * output_.info.byte_length / sizeof(uint32_t)));
  RecordProperty(
      "checked_input_words",
      std::to_string(2 * input_.info.byte_length / sizeof(uint32_t)));
  RecordProperty("changed_local_memory_words", 2 * kProducerCount);
  for (size_t i = 0; i < 2; ++i) {
    RecordProperty("retired_packets_" + std::to_string(i),
                   std::to_string(indices_[i]));
  }
}

}  // namespace
