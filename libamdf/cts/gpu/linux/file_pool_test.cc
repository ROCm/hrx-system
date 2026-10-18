// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/linux/file_pool_io.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

namespace {

class SdmaFilePoolBase : public GpuCommandTest {
 protected:
  SdmaFilePoolBase()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER |
                                 AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        }) {}
};

// Only dispatch encoding varies. The owned pool, original pinned file view,
// slot reuse, alias rebinding, and independent file oracle are identical.
template <typename DispatchTest>
class GpuFilePoolTest : public DispatchTest {
 protected:
  static constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  static constexpr uint32_t kUnit = 64 * 1024;
  static constexpr uint32_t kWordCount = kUnit / sizeof(uint32_t);
  static constexpr uint32_t kGenerationCount = 4;
  static constexpr uint32_t kGuard = 0x6bd783e5u;
  static constexpr bool kAql = std::is_same_v<DispatchTest, AqlDispatchTest>;
  static constexpr bool kSdma = std::is_same_v<DispatchTest, SdmaFilePoolBase>;

  void TearDown() override {
    // No submitted GPU work waits for file or host publication. Removing its
    // queues first makes all other case owners safe to retire independently.
    ASSERT_NO_FATAL_FAILURE(DispatchTest::TearDown());
    ASSERT_NO_FATAL_FAILURE(io_.Release());
    if (reservation_ != nullptr) {
      ASSERT_EQ(this->api_->memory_reservation_destroy(
                    std::exchange(reservation_, nullptr)),
                AMDF_STATUS_OK);
    }
    if (shared_mapping_ != nullptr) {
      ASSERT_EQ(this->api_->host_mapping_destroy(shared_mapping_),
                AMDF_STATUS_OK);
      shared_mapping_ = nullptr;
    }
    if (shared_memory_ != nullptr) {
      ASSERT_EQ(
          this->api_->memory_destroy(std::exchange(shared_memory_, nullptr)),
          AMDF_STATUS_OK);
    }
    ASSERT_TRUE(pool_.Release(this->api_));
    if (second_device_ != nullptr) {
      ASSERT_EQ(
          this->api_->device_destroy(std::exchange(second_device_, nullptr)),
          AMDF_STATUS_OK);
    }
  }

  void PreparePool(bool default_address) {
    const amdf_memory_device_access_t access = {
        this->device_,
        {
            .access = kReadWrite,
            .flags = AMDF_MEMORY_FLAG_MAPPING_SOURCE |
                     (default_address ? AMDF_MEMORY_FLAG_DEVICE_ADDRESS
                                      : UINT64_C(0)),
        },
    };
    const amdf_memory_flags_t flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_FILE_IO;
    amdf_memory_scope_info_t scope = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(scope),
    };
    ASSERT_EQ(this->api_->memory_scope_query_info(this->system_scope_, &scope),
              AMDF_STATUS_OK);
    uint32_t selected = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    uint64_t minimum_alignment = 0;
    for (uint32_t ordinal = 0; ordinal < scope.memory_profile_count;
         ++ordinal) {
      amdf_memory_profile_t profile = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
          .structure_size = sizeof(profile),
      };
      amdf_memory_access_capabilities_t capabilities = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(capabilities),
      };
      const auto status = this->api_->memory_scope_query_device_profile(
          this->system_scope_, ordinal, 1, &access, &profile, &capabilities);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      if ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) == 0 ||
          (flags & ~profile.supported_flags) != 0) {
        continue;
      }
      if (!amdf_status_is_ok(profile.availability)) {
        this->RecordProperty("file_pool_unavailable_domain",
                             amdf_status_domain(profile.availability));
        this->RecordProperty("file_pool_unavailable_code",
                             amdf_status_code(profile.availability));
        // Admission preserves the discovered native reason and caller output.
        const amdf_memory_create_info_t create = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
            .structure_size = sizeof(create),
            .memory_profile_ordinal = ordinal,
            .access_count = 1,
            .required_flags = flags,
            .byte_length = 8 * kUnit,
            .minimum_alignment = profile.allocation.minimum_alignment,
            .accesses = &access,
        };
        auto* output = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
        EXPECT_EQ(
            this->api_->memory_create(this->system_scope_, &create, &output),
            profile.availability);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
        continue;
      }
      ASSERT_LE(8 * kUnit, profile.allocation.maximum_byte_length);
      ASSERT_EQ(kUnit % profile.allocation.byte_length_granularity, 0u);
      selected = ordinal;
      minimum_alignment = profile.allocation.minimum_alignment;
      break;
    }
    if (selected == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
      GTEST_SKIP() << "owned file-I/O GPU pool is unavailable; native reason "
                      "is recorded separately from ordinary allocation support";
    }
    const amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal = selected,
        .access_count = 1,
        .required_flags = flags,
        .byte_length = 8 * kUnit,
        .minimum_alignment = minimum_alignment,
        .accesses = &access,
    };
    ASSERT_NO_FATAL_FAILURE(
        pool_.Initialize(this->api_, this->system_scope_, create));
    EXPECT_EQ(pool_.host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
    ASSERT_NE(pool_.host.features & AMDF_HOST_MAPPING_FEATURE_FILE_IO, 0u);
    EXPECT_EQ(pool_.device_address != 0, default_address);

    const amdf_memory_reservation_access_t projection = {this->device_, 0,
                                                         kReadWrite, 0};
    const amdf_memory_reservation_create_info_t reserve = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
        .structure_size = sizeof(reserve),
        .byte_length = 2 * kUnit,
        .minimum_alignment = kUnit,
        .access_count = 1,
        .accesses = &projection,
    };
    ASSERT_EQ(this->api_->memory_reservation_create(this->instance_, &reserve,
                                                    &reservation_),
              AMDF_STATUS_OK);
    amdf_memory_reservation_access_info_t view = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
        .structure_size = sizeof(view),
    };
    ASSERT_EQ(this->api_->memory_reservation_query_access_info(reservation_, 0,
                                                               &view),
              AMDF_STATUS_OK);
    ASSERT_EQ(kUnit % view.mapping_granularity, 0u);
    alias_address_ = view.address;
    ASSERT_NO_FATAL_FAILURE(QueryFileTransitions());
  }

  void QueryFileTransitions() {
    auto file = pool_.HostSite();
    file.kind = AMDF_MEMORY_SITE_KIND_FILE_IO;
    auto device = pool_.DeviceSite(this->family_.ordinal);
    amdf_memory_profile_pair_query_t query = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
        .structure_size = sizeof(query),
        .memory_profile_ordinal = pool_.creation.memory_profile_ordinal,
        .access_count = 1,
        .required_flags = pool_.creation.required_flags,
        .accesses = pool_.creation.accesses,
    };
    query.producer.kind = AMDF_MEMORY_SITE_KIND_FILE_IO;
    query.producer.value.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    query.consumer.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    query.consumer.value.device = {0, this->family_.ordinal};
    for (uint32_t direction = 0; direction < 2; ++direction) {
      amdf_memory_pair_info_t prospective = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(prospective),
      };
      auto concrete = prospective;
      ASSERT_EQ(this->api_->memory_scope_query_pair_info(this->system_scope_,
                                                         &query, &prospective),
                AMDF_STATUS_OK);
      ASSERT_EQ(this->api_->memory_query_pair_info(&file, &device, &concrete),
                AMDF_STATUS_OK);
      ASSERT_NE(concrete.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                0u);
      EXPECT_EQ(concrete.flags, prospective.flags);
      EXPECT_EQ(concrete.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
      EXPECT_EQ(concrete.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
      for (const auto* pair : {&concrete, &prospective}) {
        const auto& file_transition =
            direction == 0 ? pair->release : pair->acquire;
        EXPECT_EQ(file_transition.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
        EXPECT_EQ(file_transition.executor,
                  AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
        const auto& gpu = direction == 0 ? pair->acquire : pair->release;
        if (gpu.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
          continue;
        }
        const auto operation = direction == 0
                                   ? AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
                                   : AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM;
        ASSERT_EQ(gpu.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
        ASSERT_EQ(gpu.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
        ASSERT_EQ(gpu.operation, operation);
        ASSERT_NE(this->family_.cache_operations & (UINT64_C(1) << operation),
                  0u);
        cache_operations_ |= UINT64_C(1) << operation;
      }
      std::swap(file, device);
      std::swap(query.producer, query.consumer);
    }
  }

  void PrepareDispatch() {
    ASSERT_NO_FATAL_FAILURE(this->CreateMemory(
        kReadWrite, 4096, &control_,
        AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE));
    std::memset(control_->host.pointer, 0, control_->info.byte_length);
    if constexpr (!kSdma) {
      kernel_ = kernels::transform::kKernels.Find(this->gpu_endpoint_info_);
      ASSERT_NE(kernel_, nullptr) << "missing target transform product";
      ASSERT_EQ(kernel_->private_segment_byte_length, 0u);
      ASSERT_EQ(kernel_->group_segment_byte_length, 0u);
      this->RecordProperty("transform_target", kernel_->target);
      ASSERT_NO_FATAL_FAILURE(
          this->CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments_));
    }
    if constexpr (kAql) {
      ASSERT_NO_FATAL_FAILURE(this->CreateQueue(&aql_queue_));
      ASSERT_NO_FATAL_FAILURE(this->PublishKernel(*aql_queue_, *kernel_,
                                                  "file_pool", &packet_index_,
                                                  &descriptor_address_));
      *static_cast<aql::Signal*>(control_->host.pointer) = aql::UserSignal();
    } else {
      ASSERT_NO_FATAL_FAILURE(this->CreateQueue(&command_queue_));
      if constexpr (!kSdma) {
        program_ = {0,
                    kernel_->program.resource1,
                    kernel_->program.resource2,
                    kernel_->program.resource3,
                    kernel_->group_segment_byte_length,
                    kernel_->wavefront_size,
                    {kernel_->workgroup_size(), 1, 1}};
        ASSERT_NO_FATAL_FAILURE(this->PrepareProgram(kernel_->executable,
                                                     kernel_->entry_byte_offset,
                                                     &program_, "file_pool"));
      }
    }
  }

  void Publish(uint64_t input, uint64_t output, uint32_t count, uint32_t addend,
               uint32_t generation) {
    if constexpr (!kSdma) {
      const kernels::transform::Arguments arguments = {input, output, count,
                                                       addend};
      std::memcpy(arguments_->host.pointer, &arguments, sizeof(arguments));
      ASSERT_NO_FATAL_FAILURE(arguments_->TransferHostCache(
          this->api_, this->family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
    }
    if constexpr (kAql) {
      auto& signal = *static_cast<aql::Signal*>(control_->host.pointer);
      signal.value = 1;
      const uint32_t grid = (count + kernel_->workgroup_size() - 1) /
                            kernel_->workgroup_size() *
                            kernel_->workgroup_size();
      const auto packet = aql::Dispatch(
          aql::HeaderBarrier::kDisabled,
          {1,
           {static_cast<uint16_t>(kernel_->workgroup_size()), 1, 1},
           {grid, 1, 1}},
          0, 0, descriptor_address_, arguments_->device_address,
          control_->device_address);
      GpuStoreRelease(aql_queue_->host.write_index_address, packet_index_ + 1);
      ASSERT_TRUE(aql::Publish(*aql_queue_, packet_index_++, packet));
    } else {
      ASSERT_GE(command_queue_->words().size(), command_word_count_ + 256);
      auto* words = command_queue_->words().data() + command_word_count_;
      if constexpr (kSdma) {
        SdmaCommandWriter commands(words, this->family_.format_features);
        if ((cache_operations_ & AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) !=
            0) {
          commands.AcquireFromSystem();
        }
        commands.CopyLinear(input, output, count * sizeof(uint32_t));
        if ((cache_operations_ & AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) !=
            0) {
          commands.ReleaseToSystem();
        }
        commands.Fence32(control_->device_address, generation);
        command_word_count_ += commands.word_count();
      } else {
        Pm4CommandWriter commands(words, *this->pm4_profile_);
        commands.SystemBarrier();
        commands.BindCompute(program_, arguments_->device_address,
                             this->family_.format_features);
        const uint32_t grid = (count + kernel_->workgroup_size() - 1) /
                              kernel_->workgroup_size() *
                              kernel_->workgroup_size();
        commands.Dispatch(program_, grid, 1, 1);
        commands.ReleaseSystem32(control_->device_address, generation);
        commands.PadToEightWords();
        command_word_count_ += commands.word_count();
      }
      ASSERT_NO_FATAL_FAILURE(command_queue_->Publish(
          this->api_, this->gpu_api_, command_word_count_));
    }
  }

  void Wait(uint32_t generation) {
    if constexpr (kAql) {
      ASSERT_NO_FATAL_FAILURE(this->WaitCompletionAndConsumption(
          *aql_queue_, *static_cast<aql::Signal*>(control_->host.pointer),
          packet_index_));
    } else {
      ASSERT_TRUE(GpuWaitEqual<uint32_t>(
          reinterpret_cast<uintptr_t>(control_->host.pointer), generation));
      ASSERT_NO_FATAL_FAILURE(command_queue_->WaitRetired(this->api_));
    }
  }

  void Run(bool direct, bool default_address) {
    ASSERT_NO_FATAL_FAILURE(PreparePool(default_address));
    if (this->IsSkipped()) {
      return;
    }
    std::vector<uint32_t> expected_file(12 * kWordCount, kGuard);
    for (uint32_t generation = 0; generation < kGenerationCount; ++generation) {
      for (uint32_t word = 0; word < kWordCount; ++word) {
        expected_file[(1 + generation) * kWordCount + word] =
            0xf17b35a9u + generation * 0x01030507u + word * 0x9e3779b9u;
      }
    }
    ASSERT_NO_FATAL_FAILURE(
        io_.CreateFile(expected_file, direct, pool_.info.alignment));
    if (this->IsSkipped()) {
      return;
    }
    std::vector<uint32_t> expected_pool(8 * kWordCount, kGuard);
    std::memcpy(pool_.host.pointer, expected_pool.data(),
                pool_.info.byte_length);
    ASSERT_EQ(this->api_->host_mapping_cache_control(
                  pool_.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                  pool_.info.byte_length),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(
        io_.Register(pool_.host.pointer, pool_.info.byte_length));
    if (this->IsSkipped()) {
      return;
    }
    ASSERT_NO_FATAL_FAILURE(PrepareDispatch());
    auto* bytes = static_cast<uint8_t*>(pool_.host.pointer);
    ASSERT_NO_FATAL_FAILURE(io_.Begin(false, bytes + kUnit, kUnit, kUnit, 1));
    ASSERT_NO_FATAL_FAILURE(io_.Complete(1, kUnit));
    for (uint32_t generation = 0; generation < kGenerationCount; ++generation) {
      SCOPED_TRACE(generation);
      const uint32_t input_slot = 1 + 2 * (generation % 2);
      const uint32_t output_slot = 5 + 2 * (generation % 2);
      const uint32_t count = kWordCount - 13 - generation * 11;
      const uint32_t addend = 0x80000023u + generation * 0x01010101u;
      const std::array<amdf_memory_mapping_update_t, 2> updates = {{
          {
              .byte_length = kUnit,
              .memory = pool_.memory,
              .access = AMDF_MEMORY_ACCESS_READ,
              .source_byte_offset = input_slot * kUnit,
          },
          {
              .target_byte_offset = kUnit,
              .byte_length = kUnit,
              .memory = pool_.memory,
              .access = kReadWrite,
              .source_byte_offset = output_slot * kUnit,
          },
      }};
      ASSERT_EQ(this->api_->memory_reservation_update(
                    reservation_, 0, updates.size(), updates.data()),
                AMDF_STATUS_OK);
      // Keep a disjoint file request outstanding while publishing GPU work.
      // Completion may be fast; this tests ownership, not physical overlap.
      if (generation + 1 < kGenerationCount) {
        const uint32_t next_slot = 1 + 2 * ((generation + 1) % 2);
        ASSERT_NO_FATAL_FAILURE(io_.Begin(false, bytes + next_slot * kUnit,
                                          kUnit, (generation + 2) * kUnit,
                                          generation + 2));
      }
      const uint64_t input = default_address && generation == 0
                                 ? pool_.device_address + input_slot * kUnit
                                 : alias_address_;
      ASSERT_NO_FATAL_FAILURE(Publish(input, alias_address_ + kUnit, count,
                                      addend, generation + 1));
      if (generation + 1 < kGenerationCount) {
        ASSERT_NO_FATAL_FAILURE(io_.Complete(generation + 2, kUnit));
      }
      ASSERT_NO_FATAL_FAILURE(Wait(generation + 1));
      ASSERT_NO_FATAL_FAILURE(io_.Begin(true, bytes + output_slot * kUnit,
                                        kUnit, (7 + generation) * kUnit,
                                        100 + generation));
      ASSERT_NO_FATAL_FAILURE(io_.Complete(100 + generation, kUnit));
      std::copy_n(expected_file.data() + (1 + generation) * kWordCount,
                  kWordCount, expected_pool.data() + input_slot * kWordCount);
      for (uint32_t word = 0; word < count; ++word) {
        const uint32_t source =
            expected_file[(1 + generation) * kWordCount + word];
        expected_pool[output_slot * kWordCount + word] =
            kSdma ? source
                  : static_cast<uint32_t>(uint64_t{source} * 3 + addend);
      }
      std::copy_n(expected_pool.data() + output_slot * kWordCount, kWordCount,
                  expected_file.data() + (7 + generation) * kWordCount);
    }
    ASSERT_EQ(this->api_->host_mapping_cache_control(
                  pool_.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                  pool_.info.byte_length),
              AMDF_STATUS_OK);
    EXPECT_EQ(std::memcmp(pool_.host.pointer, expected_pool.data(),
                          pool_.info.byte_length),
              0);
    // Cancellation completion, including a successful-I/O race, precedes reuse.
    ASSERT_NO_FATAL_FAILURE(io_.CancelAndRetire(bytes + kUnit, kUnit, kUnit));
    ASSERT_NO_FATAL_FAILURE(
        io_.Begin(false, bytes + kUnit, kUnit, 2 * kUnit, 2001));
    ASSERT_NO_FATAL_FAILURE(io_.Complete(2001, kUnit));
    std::copy_n(expected_file.data() + 2 * kWordCount, kWordCount,
                expected_pool.data() + kWordCount);
    ASSERT_EQ(this->api_->host_mapping_cache_control(
                  pool_.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                  pool_.info.byte_length),
              AMDF_STATUS_OK);
    EXPECT_EQ(std::memcmp(pool_.host.pointer, expected_pool.data(),
                          pool_.info.byte_length),
              0);
    ASSERT_NO_FATAL_FAILURE(io_.VerifyFile(expected_file));
    this->RecordProperty("completed_generations", kGenerationCount);
  }

  void RunSharedAttachments() {
    ASSERT_NO_FATAL_FAILURE(PreparePool(true));
    if (this->IsSkipped()) {
      return;
    }
    // Two logical consumers of one endpoint exercise independent native
    // owners inside the same render file and KFD process. Both see the same
    // physical pages; neither owns the other's GEM handle or eviction fence.
    const amdf_gpu_device_create_info_t device_create = {
        .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
        .structure_size = sizeof(device_create),
    };
    ASSERT_EQ(this->gpu_api_->device_create(this->endpoint_, &device_create,
                                            &second_device_),
              AMDF_STATUS_OK);
    auto secondary = pool_.attachment;
    secondary.device = second_device_;
    const std::array<amdf_memory_device_access_t, 2> accesses = {
        pool_.attachment, secondary};
    auto create = pool_.creation;
    create.access_count = accesses.size();
    create.accesses = accesses.data();
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    std::array<amdf_memory_access_capabilities_t, 2> capabilities;
    for (auto& capability : capabilities) {
      capability = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(capability),
      };
    }
    ASSERT_EQ(this->api_->memory_scope_query_device_profile(
                  this->system_scope_, create.memory_profile_ordinal,
                  create.access_count, create.accesses, &profile,
                  capabilities.data()),
              AMDF_STATUS_OK);
    ASSERT_EQ(profile.availability, AMDF_STATUS_OK);
    ASSERT_EQ(this->api_->memory_create(this->system_scope_, &create,
                                        &shared_memory_),
              AMDF_STATUS_OK);
    std::array<uint64_t, 2> addresses = {};
    for (uint32_t i = 0; i < addresses.size(); ++i) {
      ASSERT_EQ(this->api_->memory_query_address(
                    shared_memory_, i, AMDF_MEMORY_ADDRESS_GPU, &addresses[i]),
                AMDF_STATUS_OK);
    }
    ASSERT_NE(addresses[0], addresses[1]);
    const amdf_memory_map_info_t map = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        .structure_size = sizeof(map),
        .byte_length = create.byte_length,
        .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    ASSERT_EQ(this->api_->memory_map(shared_memory_, &map, &shared_mapping_),
              AMDF_STATUS_OK);
    amdf_host_mapping_info_t host = {
        .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
        .structure_size = sizeof(host),
    };
    ASSERT_EQ(this->api_->host_mapping_query_info(shared_mapping_, &host),
              AMDF_STATUS_OK);
    ASSERT_NE(host.features & AMDF_HOST_MAPPING_FEATURE_FILE_IO, 0u);
    std::vector<uint32_t> expected(8 * kWordCount, kGuard);
    for (uint32_t i = 0; i < kWordCount; ++i) {
      expected[i] = 0x81e35b07u + i * 0x9e3779b9u;
    }
    std::memcpy(host.pointer, expected.data(), host.byte_length);
    ASSERT_EQ(this->api_->host_mapping_cache_control(
                  shared_mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                  host.byte_length),
              AMDF_STATUS_OK);
    std::vector<uint32_t> expected_pool(8 * kWordCount, kGuard);
    std::memcpy(pool_.host.pointer, expected_pool.data(),
                pool_.info.byte_length);
    ASSERT_EQ(this->api_->host_mapping_cache_control(
                  pool_.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                  pool_.info.byte_length),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(PrepareDispatch());
    for (uint32_t generation = 0; generation < 2; ++generation) {
      const uint32_t count = kWordCount - 17 - generation;
      const uint32_t addend = 0x7b31e907u + generation;
      ASSERT_NO_FATAL_FAILURE(
          Publish(addresses[generation] + generation * kUnit,
                  addresses[1 - generation] + (generation + 1) * kUnit, count,
                  addend, generation + 1));
      ASSERT_NO_FATAL_FAILURE(Wait(generation + 1));
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t input = expected[generation * kWordCount + i];
        expected[(generation + 1) * kWordCount + i] =
            kSdma ? input : static_cast<uint32_t>(uint64_t{input} * 3 + addend);
      }
    }
    // A pool can grow through independent slabs without changing its virtual
    // region. Rebind both slots after their final readers, then use the prior
    // slab's output through the same addresses in the reverse direction.
    for (uint32_t generation = 0; generation < 2; ++generation) {
      SCOPED_TRACE(generation);
      const bool forward = generation == 0;
      const uint32_t input_slot = forward ? 2 : 3;
      const uint32_t output_slot = forward ? 3 : 4;
      const std::array<amdf_memory_mapping_update_t, 2> updates = {{
          {
              .byte_length = kUnit,
              .memory = forward ? shared_memory_ : pool_.memory,
              .access = AMDF_MEMORY_ACCESS_READ,
              .source_byte_offset = input_slot * kUnit,
          },
          {
              .target_byte_offset = kUnit,
              .byte_length = kUnit,
              .memory = forward ? pool_.memory : shared_memory_,
              .access = kReadWrite,
              .source_byte_offset = output_slot * kUnit,
          },
      }};
      ASSERT_EQ(this->api_->memory_reservation_update(
                    reservation_, 0, updates.size(), updates.data()),
                AMDF_STATUS_OK);
      const uint32_t count = kWordCount - 31 - generation;
      const uint32_t addend = 0xc31f6907u + generation;
      ASSERT_NO_FATAL_FAILURE(Publish(alias_address_, alias_address_ + kUnit,
                                      count, addend, generation + 3));
      ASSERT_NO_FATAL_FAILURE(Wait(generation + 3));
      const auto& source = forward ? expected : expected_pool;
      auto& destination = forward ? expected_pool : expected;
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t input = source[input_slot * kWordCount + i];
        destination[output_slot * kWordCount + i] =
            kSdma ? input : static_cast<uint32_t>(uint64_t{input} * 3 + addend);
      }
    }
    ASSERT_EQ(this->api_->host_mapping_cache_control(
                  pool_.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                  pool_.info.byte_length),
              AMDF_STATUS_OK);
    EXPECT_EQ(std::memcmp(pool_.host.pointer, expected_pool.data(),
                          pool_.info.byte_length),
              0);
    ASSERT_EQ(this->api_->host_mapping_cache_control(
                  shared_mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                  host.byte_length),
              AMDF_STATUS_OK);
    EXPECT_EQ(std::memcmp(host.pointer, expected.data(), host.byte_length), 0);
    // Checked teardown releases both imports independently, then the pages.
    // Reusing one native export would double-close its deduplicated GEM handle.
  }

  // Additional logical device used only to exercise independent attachments.
  amdf_device_t* second_device_ = nullptr;
  // Joint allocation with two native owners of the same physical pages.
  amdf_memory_t* shared_memory_ = nullptr;
  // Original ordinary host view, retired before its joint allocation.
  amdf_host_mapping_t* shared_mapping_ = nullptr;
  // Independently owned pool survives queue, file-pin, and alias retirement.
  GpuMemory pool_;
  // Native file and fixed-buffer registrations borrow the pool's original view.
  FilePoolIo io_;
  // Stable two-slot GPU projection, rebound only after the last GPU reader.
  amdf_memory_reservation_t* reservation_ = nullptr;
  // Base address of the explicit GPU projection.
  uint64_t alias_address_ = 0;
  // Queried file/device cache operations for the chosen engine.
  amdf_cache_operations_t cache_operations_ = 0;
  // Complete signal or CP fence, separate from cached payload storage.
  GpuMemory* control_ = nullptr;
  // Immutable during each dispatch; retained by the engine fixture.
  GpuMemory* arguments_ = nullptr;
  // Target-specific compiler product, shared by the two compute engines.
  const kernels::Kernel* kernel_ = nullptr;
  // AQL carrier owned by the engine fixture.
  GpuUserQueue* aql_queue_ = nullptr;
  // Next AQL packet index, including cold code publication.
  uint64_t packet_index_ = 0;
  // Published AQL code descriptor.
  uint64_t descriptor_address_ = 0;
  // PM4/SDMA command carrier owned by the engine fixture.
  GpuCommandQueue* command_queue_ = nullptr;
  // Published command extent, never overwritten while the queue is live.
  size_t command_word_count_ = 0;
  // PM4 launch facts with the code owner's entry address.
  Pm4ComputeProgram program_ = {};
};

using Engines =
    ::testing::Types<Pm4DispatchTest, AqlDispatchTest, SdmaFilePoolBase>;
TYPED_TEST_SUITE(GpuFilePoolTest, Engines);

TYPED_TEST(GpuFilePoolTest, BufferedSourceOnly) { this->Run(false, false); }
TYPED_TEST(GpuFilePoolTest, DirectSourceOnly) { this->Run(true, false); }
TYPED_TEST(GpuFilePoolTest, BufferedDefaultAddress) { this->Run(false, true); }
TYPED_TEST(GpuFilePoolTest, DirectDefaultAddress) { this->Run(true, true); }
TYPED_TEST(GpuFilePoolTest, IndependentAttachmentsSharePages) {
  this->RunSharedAttachments();
}

}  // namespace
