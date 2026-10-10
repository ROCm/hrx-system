// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <iterator>
#include <span>
#include <string_view>
#include <vector>

#include "libamdf/cts/xdna/programs/mul_i32.h"
#include "libamdf/cts/xdna/util/executable.h"
#include "libamdf/cts/xdna/util/execution.h"
#include "libamdf/cts/xdna/xdna_device_fixture.h"
#include "util/mapped_memory.h"

namespace {

constexpr size_t kBindingByteOffset = 64;
constexpr size_t kBindingStorageByteLength = 192;
constexpr std::array<uint32_t, 16> kValues = {
    0,          1,          2,          3,          7,          31,
    65535,      65536,      0x7fffffff, 0x80000000, 0x80000001, 0xfffffffd,
    0xfffffffe, 0xffffffff, 0x12345678, 0x87654321,
};

// Native arithmetic fixtures use little-endian words on both host platforms.
void StoreU32(std::span<uint8_t> bytes, size_t byte_offset, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) {
    bytes[byte_offset + i] = value >> (8 * i);
  }
}

void CheckTransition(const amdf_cache_transition_t& actual,
                     const amdf_cache_transition_t& expected) {
  ASSERT_EQ(actual.kind, expected.kind);
  ASSERT_EQ(actual.executor, expected.executor);
  ASSERT_EQ(actual.operation, expected.operation);
  ASSERT_EQ(actual.host_operation, expected.host_operation);
  ASSERT_EQ(actual.host_instruction, expected.host_instruction);
  ASSERT_EQ(actual.host_fence_before, expected.host_fence_before);
  ASSERT_EQ(actual.host_fence_after, expected.host_fence_after);
  ASSERT_EQ(actual.range_granularity, expected.range_granularity);
}

class CpuXdnaRecipeTest : public XdnaDeviceFixture {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(XdnaDeviceFixture::SetUp());
    amdf_xdna_endpoint_info_t endpoint_info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
        .structure_size = sizeof(endpoint_info),
    };
    ASSERT_EQ(xdna_api_->endpoint_query_info(endpoint_, &endpoint_info),
              AMDF_STATUS_OK);
    RecordProperty("amdf_xdna_target", endpoint_info.target_id);
    amdf_xdna_device_info_t device_info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO,
        .structure_size = sizeof(device_info),
    };
    ASSERT_EQ(xdna_api_->device_query_info(device_, &device_info),
              AMDF_STATUS_OK);
    ASSERT_TRUE(FindXdnaKernelQueueFamily(api_, endpoint_, &family_ordinal_));

    const iree_file_toc_t* image = nullptr;
    const std::string_view target = endpoint_info.target_id;
    if (target == "amd.xdna.strix_halo.17f0_11") {
      image = &amdf_cts_xdna_mul_i32_create()[1];
    } else if (target == "amd.xdna.strix.17f0_10" ||
               target == "amd.xdna.krackan.17f0_20") {
      image = &amdf_cts_xdna_mul_i32_create()[0];
    } else {
      GTEST_SKIP() << "no finite arithmetic fixture for " << target;
    }
    constexpr std::array<amdf_memory_access_t, 3> binding_accesses = {
        AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_READ,
        AMDF_MEMORY_ACCESS_WRITE};
    ASSERT_TRUE(executable_.Initialize(
        {reinterpret_cast<const uint8_t*>(image->data), image->size},
        endpoint_info, device_info, 1, binding_accesses));
    memory_access_.requirements.address_kinds = UINT64_C(1)
                                                << AMDF_MEMORY_ADDRESS_XDNA_DMA;
  }

  void TearDown() override {
    if (!execution_.Release(api_, xdna_api_)) {
      return;
    }
    for (auto& binding : bindings_) {
      if (!binding.Release(api_)) {
        return;
      }
    }
    for (auto& backing : registered_storage_) {
      if (!backing.Release(api_)) {
        return;
      }
    }
  }

  void CreateBindings(amdf_memory_profile_roles_t role) {
    const uint32_t ordinal =
        FindMemoryProfileOrdinal(role | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
                                 AMDF_MEMORY_FLAG_HOST_VISIBLE);
    ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    ASSERT_EQ(QueryMemoryProfile(ordinal, &profile, &capabilities),
              AMDF_STATUS_OK);
    const auto& geometry = role == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                               ? profile.registration
                               : profile.allocation;
    ASSERT_GT(geometry.byte_length_granularity, 0u);
    for (size_t i = 0; i < bindings_.size(); ++i) {
      SCOPED_TRACE(i);
      amdf_memory_create_info_t create = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
          .structure_size = sizeof(create),
          .memory_profile_ordinal = ordinal,
          .access_count = 1,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
          .byte_length = (kBindingStorageByteLength +
                          geometry.byte_length_granularity - 1) /
                         geometry.byte_length_granularity *
                         geometry.byte_length_granularity,
          .minimum_alignment = geometry.minimum_alignment,
          .accesses = &memory_access_,
      };
      if (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
        amdf_memory_create_info_t host_create = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
            .structure_size = sizeof(host_create),
            .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
            .byte_length = create.byte_length,
            .minimum_alignment = geometry.registered_host_pointer_alignment,
        };
        ASSERT_NO_FATAL_FAILURE(
            registered_storage_[i].Create(api_, system_scope_, host_create));
        ASSERT_EQ(registered_storage_[i].host.cacheability,
                  geometry.registered_host_cacheability);
        create.registered_host_pointer = registered_storage_[i].host.pointer;
        create.registered_host_cacheability =
            geometry.registered_host_cacheability;
      }
      ASSERT_NO_FATAL_FAILURE(bindings_[i].Create(api_, system_scope_, create));
      uint64_t address = 0;
      ASSERT_EQ(
          api_->memory_query_address(bindings_[i].memory, 0,
                                     AMDF_MEMORY_ADDRESS_XDNA_DMA, &address),
          AMDF_STATUS_OK);
      binding_addresses_[i] = address + kBindingByteOffset;
      const auto host = bindings_[i].HostSite();
      const auto device = bindings_[i].DeviceSite(0, family_ordinal_);
      ASSERT_NO_FATAL_FAILURE(QueryHostPair(host, device, &ingress_[i]));
      ASSERT_NO_FATAL_FAILURE(QueryHostPair(device, host, &egress_[i]));
      ASSERT_EQ(ingress_[i].release.host_operation,
                AMDF_HOST_CACHE_OPERATION_FLUSH);
      ASSERT_EQ(ingress_[i].release.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
      ASSERT_EQ(egress_[i].acquire.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
      ASSERT_EQ(egress_[i].acquire.host_operation,
                AMDF_HOST_CACHE_OPERATION_INVALIDATE);
      const amdf_cache_transition_t no_op = {
          .kind = AMDF_CACHE_TRANSITION_KIND_NONE,
      };
      ASSERT_NO_FATAL_FAILURE(
          CheckTransition(ingress_[i].release, bindings_[i].host.flush));
      ASSERT_NO_FATAL_FAILURE(CheckTransition(ingress_[i].acquire, no_op));
      ASSERT_NO_FATAL_FAILURE(CheckTransition(egress_[i].release, no_op));
      ASSERT_NO_FATAL_FAILURE(
          CheckTransition(egress_[i].acquire, bindings_[i].host.invalidate));
    }
  }

  void QueryHostPair(const amdf_memory_site_t& producer,
                     const amdf_memory_site_t& consumer,
                     amdf_memory_pair_info_t* pair) {
    pair->type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    pair->structure_size = sizeof(*pair);
    ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, pair),
              AMDF_STATUS_OK);
    ASSERT_NE(pair->flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_EQ(pair->atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
    ASSERT_EQ(pair->atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
  }

  void RunRoundTrips(amdf_memory_profile_roles_t role) {
    ASSERT_NO_FATAL_FAILURE(CreateBindings(role));
    std::vector<uint8_t> image_storage(executable_.allocation_byte_length());
    executable_.Load(image_storage);
    ASSERT_TRUE(executable_.Bind(image_storage, binding_addresses_));
    ASSERT_NO_FATAL_FAILURE(
        execution_.Prepare(api_, xdna_api_, device_, family_ordinal_, 1,
                           executable_.ResolveInvocation(image_storage),
                           executable_.allocation_alignment()));
    const auto commands = execution_.instructions.bytes();
    const std::vector<uint8_t> original_commands(commands.begin(),
                                                 commands.end());
    std::vector<uint8_t> observed_commands(commands.size());
    std::array<std::vector<uint8_t>, 3> observed;
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      observed[ordinal].resize(bindings_[ordinal].host.byte_length);
    }
    // Command publication is separate from the tested payload ownership edges.
    ASSERT_EQ(api_->host_mapping_cache_control(execution_.instructions.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, commands.size()),
              AMDF_STATUS_OK);
    for (uint32_t generation = 0; generation < 8 && !HasFailure();
         ++generation) {
      SCOPED_TRACE(generation);
      std::array<std::vector<uint8_t>, 3> expected;
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        const auto bytes = bindings_[ordinal].bytes();
        std::fill(bytes.begin(), bytes.end(),
                  static_cast<uint8_t>(0xA5 ^ generation));
        expected[ordinal].assign(bytes.begin(), bytes.end());
      }
      for (size_t i = 0; i < kValues.size(); ++i) {
        const uint32_t lhs = kValues[(i + generation) % kValues.size()];
        const uint32_t rhs = kValues[(i * 3 + generation + 5) % kValues.size()];
        const uint32_t product = lhs * rhs;
        const size_t offset = kBindingByteOffset + i * sizeof(uint32_t);
        StoreU32(bindings_[0].bytes(), offset, lhs);
        StoreU32(bindings_[1].bytes(), offset, rhs);
        StoreU32(bindings_[2].bytes(), offset, ~product);
        StoreU32(expected[0], offset, lhs);
        StoreU32(expected[1], offset, rhs);
        StoreU32(expected[2], offset, product);
      }
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        ASSERT_EQ(api_->host_mapping_cache_control(
                      bindings_[ordinal].mapping,
                      ingress_[ordinal].release.host_operation, 0,
                      bindings_[ordinal].host.byte_length),
                  AMDF_STATUS_OK);
      }

      amdf_xdna_kernel_queue_submission_info_t submit = {
          .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
          .structure_size = sizeof(submit),
          .command_count = 1,
          .commands = &execution_.command,
      };
      uint64_t point = 0;
      ASSERT_EQ(
          xdna_api_->kernel_queue_submit(execution_.queue, &submit, &point),
          AMDF_STATUS_OK);
      ASSERT_EQ(api_->kernel_queue_wait(execution_.queue, point,
                                        AMDF_TIMEOUT_INFINITE, 0),
                AMDF_STATUS_OK);
      // The output transfer joins both input records and all output writes.
      // Remaining resident activity accesses only tile-local state.
      // Native completion supplies ordering; these queried actions supply CPU
      // visibility. Capture output first, then the inputs and command owner. No
      // oracle, progress query or cleanup precedes the complete observation
      // snapshots.
      for (size_t ordinal : {2u, 0u, 1u}) {
        ASSERT_EQ(api_->host_mapping_cache_control(
                      bindings_[ordinal].mapping,
                      egress_[ordinal].acquire.host_operation, 0,
                      bindings_[ordinal].host.byte_length),
                  AMDF_STATUS_OK);
        const auto bytes = bindings_[ordinal].bytes();
        std::copy(bytes.begin(), bytes.end(), observed[ordinal].begin());
      }
      ASSERT_EQ(api_->host_mapping_cache_control(
                    execution_.instructions.mapping,
                    AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, commands.size()),
                AMDF_STATUS_OK);
      std::copy(commands.begin(), commands.end(), observed_commands.begin());
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        SCOPED_TRACE(ordinal);
        const auto mismatch =
            std::mismatch(observed[ordinal].begin(), observed[ordinal].end(),
                          expected[ordinal].begin());
        EXPECT_EQ(mismatch.first, observed[ordinal].end())
            << "binding byte "
            << std::distance(observed[ordinal].begin(), mismatch.first);
      }
      EXPECT_EQ(observed_commands, original_commands);
      amdf_kernel_queue_status_t status = {
          .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
          .structure_size = sizeof(status),
      };
      ASSERT_EQ(api_->kernel_queue_query_status(execution_.queue, &status),
                AMDF_STATUS_OK);
      EXPECT_EQ(status.retired_submission, point);
      EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
    }
  }

  // Native family selected before allocating instructions or payloads.
  uint32_t family_ordinal_ = UINT32_MAX;
  // Immutable, admitted fixture borrowing build-generated Loom output.
  XdnaExecutable executable_;
  // Private context, command backing and one native queue.
  XdnaExecution execution_;
  // CPU-only storage retained through native registration release.
  std::array<CtsMappedMemory, 3> registered_storage_;
  // Lhs, rhs and output, each with a complete guarded host view.
  std::array<CtsMappedMemory, 3> bindings_;
  // Independently queried native DMA addresses including each logical offset.
  std::array<uint64_t, 3> binding_addresses_ = {};
  // Queried CPU publication actions paired with NPU ingress.
  std::array<amdf_memory_pair_info_t, 3> ingress_ = {};
  // Queried CPU acquisition actions paired with drained NPU output.
  std::array<amdf_memory_pair_info_t, 3> egress_ = {};
};

TEST_F(CpuXdnaRecipeTest, AllocatedRoundTrip) {
  RunRoundTrips(AMDF_MEMORY_PROFILE_ROLE_CREATE);
}

TEST_F(CpuXdnaRecipeTest, RegisteredRoundTrip) {
  RunRoundTrips(AMDF_MEMORY_PROFILE_ROLE_REGISTER);
}

}  // namespace
