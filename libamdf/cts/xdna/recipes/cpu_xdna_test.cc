// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "libamdf/cts/xdna/programs/mul_i32.h"
#if defined(AMDF_CTS_XDNA_CXX_WORKER)
#include "libamdf/cts/xdna/programs/mul_i32_cxx.h"
#endif  // AMDF_CTS_XDNA_CXX_WORKER
#include "libamdf/cts/xdna/programs/recurrence_i32.h"
#include "libamdf/cts/xdna/util/executable.h"
#include "libamdf/cts/xdna/util/execution.h"
#include "libamdf/cts/xdna/xdna_device_fixture.h"
#include "util/mapped_memory.h"
#include "util/observation.h"
#include "util/wait.h"

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

// One caller-correlated command, with device timestamps in format-2
// nanoseconds.
struct FirmwareCommandInterval {
  // Independent observations in the common library host clock ticks.
  struct {
    // Lower bound, immediately before submitting the command.
    uint64_t begin = 0;
    // Upper bound, immediately after observing completion.
    uint64_t end = 0;
  } host;
  // Firmware timestamps in this uninterrupted recorder's clock.
  struct {
    // Command start in nanoseconds.
    uint64_t begin = 0;
    // Command completion in nanoseconds.
    uint64_t end = 0;
  } device;
};

// Caller-owned global recorder and fixed read storage. The ordinary numeric
// workload supplies execution and retirement; observation never gates its work.
class FirmwareTraceCapture {
 public:
  FirmwareTraceCapture(const amdf_xdna_api_t* api, amdf_device_t* device,
                       const ObservationHostClock& host_clock,
                       std::span<FirmwareCommandInterval> intervals = {})
      : api_(api),
        device_(device),
        host_clock_(host_clock),
        intervals_(intervals) {}
  ~FirmwareTraceCapture() {
    if (owns_configuration_) {
      EXPECT_EQ(api_->device_configure_firmware_trace(device_, 0, 0),
                AMDF_STATUS_OK);
    }
  }

  void Start(const amdf_xdna_firmware_trace_info_t& initial) {
    ASSERT_EQ(initial.enabled, 0u) << "firmware recorder is already owned";
    storage_.resize((initial.recommended_buffer_byte_length + 7) / 8);
    owns_configuration_ = true;
    ASSERT_EQ(api_->device_configure_firmware_trace(
                  device_, 1,
                  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_SCHEDULING |
                      AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_CLOCK_POWER),
              AMDF_STATUS_OK);
    amdf_xdna_firmware_trace_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_FIRMWARE_TRACE_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->device_query_firmware_trace_info(device_, &info),
              AMDF_STATUS_OK);
    ASSERT_EQ(info.enabled, 1u);
    format_version_ = info.format_version;
    ASSERT_EQ(format_version_, AMDF_XDNA_FIRMWARE_TRACE_FORMAT_2_0);
    ASSERT_EQ(info.clock.frequency_hz, UINT64_C(1000000000));
    ASSERT_EQ(info.clock.counter_bit_count, 64u);
  }

  void Read() {
    amdf_observation_read_result_t result = {};
    ASSERT_EQ(api_->device_read_firmware_trace(
                  device_, format_version_, position_, storage_.data(),
                  static_cast<uint32_t>(storage_.size() * 8), &result),
              AMDF_STATUS_OK);
    ASSERT_EQ(result.flags, 0u);
    ASSERT_EQ(result.lost_byte_length, 0u);
    ASSERT_GE(result.next_position, position_);
    position_ = result.next_position;
    ASSERT_LE(result.byte_offset + result.byte_length, storage_.size() * 8);
    ASSERT_EQ(result.byte_length % sizeof(amdf_xdna_firmware_trace_event_t),
              0u);
    const auto* bytes = reinterpret_cast<const uint8_t*>(storage_.data());
    for (uint64_t offset = 0; offset < result.byte_length;
         offset += sizeof(amdf_xdna_firmware_trace_event_t)) {
      amdf_xdna_firmware_trace_event_t event;
      std::memcpy(&event, bytes + result.byte_offset + offset, sizeof(event));
      if (event.event_id != AMDF_XDNA_FIRMWARE_TRACE_EVENT_COMMAND_BEGIN &&
          event.event_id != AMDF_XDNA_FIRMWARE_TRACE_EVENT_COMMAND_END) {
        continue;
      }
      // EXEC_NPU_CMD is the unchanged recipe's native carrier.
      if (amdf_xdna_firmware_trace_command_opcode(&event) != 23) {
        continue;
      }
      const uint32_t context =
          amdf_xdna_firmware_trace_command_context(&event) |
          (amdf_xdna_firmware_trace_command_virtual_context(&event) << 4);
      if (event.event_id == AMDF_XDNA_FIRMWARE_TRACE_EVENT_COMMAND_BEGIN) {
        ASSERT_FALSE(pending_.active);
        pending_ = {true, context, event.timestamp};
        ++begin_count_;
      } else {
        ASSERT_TRUE(pending_.active);
        ASSERT_EQ(context, pending_.context);
        ASSERT_GE(event.timestamp, pending_.timestamp);
        if (!intervals_.empty()) {
          ASSERT_LT(end_count_, intervals_.size());
          auto& interval = intervals_[end_count_];
          interval.device.begin = pending_.timestamp;
          interval.device.end = event.timestamp;
        }
        pending_.active = false;
        ++end_count_;
      }
    }
    if (result.byte_length) {
      ++nonempty_reads_;
    }
  }

  void Drain(uint32_t expected_count) {
    // The numeric producer has retired. Native publication may still progress;
    // known message count, not an empty snapshot or a sleep, closes this read.
    const amdf::cts::WaitDeadline deadline;
    while (end_count_ < expected_count) {
      ASSERT_NO_FATAL_FAILURE(Read());
      if (end_count_ < expected_count) {
        ASSERT_TRUE(deadline.Pause()) << "firmware command ends " << end_count_
                                      << ", expected " << expected_count;
      }
    }
    EXPECT_EQ(begin_count_, expected_count);
    EXPECT_EQ(end_count_, expected_count);
    EXPECT_FALSE(pending_.active);
    EXPECT_GT(nonempty_reads_, 0u);
  }

  void BeginCommand(uint32_t ordinal) {
    if (intervals_.empty()) {
      return;
    }
    // A deliberate ordinary idle gap checks source continuity. It is neither
    // a readiness wait nor a performance measurement or power-policy change.
    if (ordinal == intervals_.size() / 2) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_LT(ordinal, intervals_.size());
    ASSERT_EQ(host_clock_.Read(&intervals_[ordinal].host.begin),
              AMDF_STATUS_OK);
  }

  void EndCommand(uint32_t ordinal) {
    if (!intervals_.empty()) {
      ASSERT_LT(ordinal, intervals_.size());
      ASSERT_EQ(host_clock_.Read(&intervals_[ordinal].host.end),
                AMDF_STATUS_OK);
    }
  }

  uint32_t nonempty_reads() const { return nonempty_reads_; }

 private:
  // Borrowed immutable API table.
  const amdf_xdna_api_t* api_;
  // Borrowed corpus device; configuration is explicitly scoped to this test.
  amdf_device_t* device_;
  // Borrowed common host clock, live for the entire capture.
  const ObservationHostClock& host_clock_;
  // Optional bounded caller storage for this isolated command sequence.
  std::span<FirmwareCommandInterval> intervals_;
  // True after this caller has attempted to enable its initially idle recorder.
  bool owns_configuration_ = false;
  // Aligned, fixed-capacity native request and decoded-event storage.
  std::vector<uint64_t> storage_;
  // Queried native event schema for this capture.
  uint32_t format_version_ = 0;
  // Next native read position accepted by this consumer.
  uint64_t position_ = 0;
  // Number of observed workload command beginnings.
  uint32_t begin_count_ = 0;
  // Number of observed workload command completions.
  uint32_t end_count_ = 0;
  // Number of useful incremental reads.
  uint32_t nonempty_reads_ = 0;
  // One serial workload message awaiting its corresponding completion event.
  struct {
    // True between the start and done events.
    bool active = false;
    // Physical and virtual native context identity.
    uint32_t context = 0;
    // Start timestamp, compared only with this message's done timestamp.
    uint64_t timestamp = 0;
  } pending_;
};

class CpuXdnaRecipeTest : public XdnaDeviceFixture {
 protected:
  enum class RegisteredContents { kUntouched, kHostWritten };
  enum class ContextWidth { kSingleColumn, kFullArray };

  // Portable clock used by finite markers and automatic firmware capture.
  ObservationHostClock host_clock_;

  virtual const iree_file_toc_t* ProgramImages() const {
    return amdf_cts_xdna_mul_i32_create();
  }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(XdnaDeviceFixture::SetUp());
    if (IsSkipped()) {
      return;
    }
    ASSERT_NO_FATAL_FAILURE(host_clock_.Initialize(api_));
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
      image = &ProgramImages()[1];
    } else if (target == "amd.xdna.strix.17f0_10" ||
               target == "amd.xdna.krackan.17f0_20") {
      image = &ProgramImages()[0];
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
    if (!consumer_.Release(api_, xdna_api_)) {
      return;
    }
    if (!execution_.Release(api_, xdna_api_)) {
      return;
    }
    api_->external_memory_release(&external_memory_);
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
    EXPECT_TRUE(export_source_.Release(api_));
  }

  void CreateBindings(
      amdf_memory_profile_roles_t role,
      RegisteredContents registered_contents = RegisteredContents::kUntouched,
      uint64_t storage_byte_length = kBindingStorageByteLength) {
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
          .byte_length =
              (storage_byte_length + geometry.byte_length_granularity - 1) /
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
        if (i == 2 && registered_contents == RegisteredContents::kHostWritten) {
          const auto bytes = registered_storage_[i].bytes();
          std::fill(bytes.begin(), bytes.end(), 0xA5);
        }
        ASSERT_EQ(registered_storage_[i].host.cacheability,
                  geometry.registered_host_cacheability);
        create.registered_host_pointer = registered_storage_[i].host.pointer;
        create.registered_host_cacheability =
            geometry.registered_host_cacheability;
      }
      ASSERT_NO_FATAL_FAILURE(bindings_[i].Create(api_, system_scope_, create));
    }
    ASSERT_NO_FATAL_FAILURE(QueryBindingRecipes());
  }

  void FindExternalProfile(amdf_memory_profile_roles_t role,
                           amdf_memory_profile_t* profile,
                           amdf_external_memory_support_t* support) {
    amdf_memory_scope_info_t scope = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(scope),
    };
    ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &scope),
              AMDF_STATUS_OK);
    const auto roles = role | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    const auto required_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE |
        (role == AMDF_MEMORY_PROFILE_ROLE_IMPORT ? 0
                                                 : AMDF_MEMORY_FLAG_SHAREABLE);
    const auto flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                       (role == AMDF_MEMORY_PROFILE_ROLE_IMPORT
                            ? AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT
                            : AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT);
    for (uint32_t ordinal = 0; ordinal < scope.memory_profile_count;
         ++ordinal) {
      profile->type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      profile->structure_size = sizeof(*profile);
      amdf_memory_access_capabilities_t capabilities = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(capabilities),
      };
      const auto status = QueryMemoryProfile(ordinal, profile, &capabilities);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      if ((profile->roles & roles) != roles ||
          (profile->supported_flags & required_flags) != required_flags) {
        continue;
      }
      for (uint32_t i = 0; i < profile->external_memory_support_count; ++i) {
        const auto& candidate = profile->external_memory_support[i];
        if (candidate.type == AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD &&
            (candidate.flags & flags) == flags) {
          *support = candidate;
          return;
        }
      }
    }
    GTEST_SKIP() << "direct-host external subranges are not advertised";
  }

  void ImportBindings() {
    amdf_memory_profile_t owned = {}, imported = {};
    amdf_external_memory_support_t export_support = {}, import_support = {};
    ASSERT_NO_FATAL_FAILURE(FindExternalProfile(
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_EXPORT,
        &owned, &export_support));
    if (IsSkipped()) {
      return;
    }
    ASSERT_NO_FATAL_FAILURE(FindExternalProfile(AMDF_MEMORY_PROFILE_ROLE_IMPORT,
                                                &imported, &import_support));
    if (IsSkipped()) {
      return;
    }
    ASSERT_TRUE(amdf_external_memory_provenance_is_equal(
        &export_support.provenance, &import_support.provenance));
    ASSERT_GT(export_support.source_offset_alignment, 0u);
    ASSERT_GT(import_support.source_offset_alignment, 0u);
    ASSERT_GT(export_support.byte_length_alignment, 0u);
    ASSERT_GT(import_support.byte_length_alignment, 0u);
    ASSERT_GT(imported.import.byte_length_granularity, 0u);
    ASSERT_GT(owned.allocation.byte_length_granularity, 0u);
    const uint64_t alignment = std::max<uint64_t>(
        kBindingByteOffset, imported.import.minimum_alignment);
    ASSERT_LE(alignment, imported.import.maximum_alignment);
    const uint64_t offset_alignment =
        std::lcm(alignment, std::lcm(export_support.source_offset_alignment,
                                     import_support.source_offset_alignment));
    const uint64_t source_offset =
        (owned.allocation.native_byte_length_granularity + offset_alignment -
         1) /
            offset_alignment * offset_alignment +
        offset_alignment;
    const uint64_t length_alignment =
        std::lcm(imported.import.byte_length_granularity,
                 std::lcm(export_support.byte_length_alignment,
                          import_support.byte_length_alignment));
    const uint64_t byte_length =
        (kBindingStorageByteLength + length_alignment - 1) / length_alignment *
        length_alignment;
    const uint64_t allocation_granularity =
        owned.allocation.byte_length_granularity;
    const uint64_t source_length =
        (source_offset + byte_length + allocation_granularity - 1) /
        allocation_granularity * allocation_granularity;
    ASSERT_LE(source_length, owned.allocation.maximum_byte_length);
    ASSERT_LE(byte_length, imported.import.maximum_byte_length);
    const amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal = owned.ordinal,
        .access_count = 1,
        .required_flags =
            AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE,
        .byte_length = source_length,
        .minimum_alignment = owned.allocation.minimum_alignment,
        .accesses = &memory_access_,
    };
    const amdf_memory_export_info_t export_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
        .structure_size = sizeof(export_info),
        .external_memory_type = export_support.type,
        .byte_offset = source_offset,
        .byte_length = byte_length,
    };
    const amdf_memory_import_info_t import_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO,
        .structure_size = sizeof(import_info),
        .memory_profile_ordinal = imported.ordinal,
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .minimum_alignment = alignment,
        .accesses = &memory_access_,
    };
    for (size_t i = 0; i < bindings_.size(); ++i) {
      SCOPED_TRACE(i);
      ASSERT_NO_FATAL_FAILURE(
          export_source_.Create(api_, system_scope_, create));
      const auto source = export_source_.bytes();
      std::fill(source.begin(), source.end(), uint8_t(0xA5));
      std::fill_n(source.begin() + source_offset, byte_length, uint8_t(0x3C));
      ASSERT_EQ(api_->host_mapping_cache_control(
                    export_source_.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                    source_length),
                AMDF_STATUS_OK);
      ASSERT_EQ(api_->memory_export(export_source_.memory, &export_info,
                                    &external_memory_),
                AMDF_STATUS_OK);
      const auto identity = external_memory_.physical_backing_id;
      ASSERT_TRUE(amdf_physical_memory_id_is_valid(&identity));
      const uint64_t physical_offset = external_memory_.source_byte_offset;
      ASSERT_EQ(physical_offset,
                export_source_.info.source_byte_offset + source_offset);
      ASSERT_EQ(
          api_->memory_import(system_scope_, &import_info, &external_memory_,
                              &bindings_[i].memory, nullptr),
          AMDF_STATUS_OK);
      ASSERT_EQ(external_memory_.type, AMDF_EXTERNAL_MEMORY_TYPE_NONE);
      // No exporter mapping or attachment survives to supply cache maintenance
      // or preserve the native device address of this imported range.
      ASSERT_TRUE(export_source_.Release(api_));
      ASSERT_NO_FATAL_FAILURE(bindings_[i].Map(api_));
      ASSERT_EQ(bindings_[i].info.source_byte_offset, physical_offset);
      ASSERT_EQ(bindings_[i].info.byte_length, byte_length);
      ASSERT_TRUE(amdf_physical_memory_id_is_equal(
          &bindings_[i].info.physical_backing_id, &identity));
      const auto bytes = bindings_[i].bytes();
      EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(),
                              [](uint8_t value) { return value == 0x3C; }));
    }
    ASSERT_NO_FATAL_FAILURE(QueryBindingRecipes());
  }

  void QueryBindingRecipes() {
    for (size_t i = 0; i < bindings_.size(); ++i) {
      SCOPED_TRACE(i);
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

  void PrepareExecution(XdnaExecution* execution, uint32_t column_count,
                        std::span<const uint64_t> binding_addresses) {
    std::vector<uint8_t> image_storage(executable_.allocation_byte_length());
    executable_.Load(image_storage);
    ASSERT_TRUE(executable_.Bind(image_storage, binding_addresses));
    ASSERT_NO_FATAL_FAILURE(execution->Prepare(
        api_, xdna_api_, device_, family_ordinal_, column_count,
        executable_.ResolveInvocation(image_storage),
        executable_.allocation_alignment()));
  }

  void RunFirstDeviceWriter(
      amdf_memory_profile_roles_t role,
      RegisteredContents registered_contents = RegisteredContents::kUntouched) {
    ASSERT_NO_FATAL_FAILURE(CreateBindings(role, registered_contents));
    ASSERT_NO_FATAL_FAILURE(
        PrepareExecution(&execution_, 1, binding_addresses_));
    constexpr size_t kPayloadByteLength = kValues.size() * sizeof(uint32_t);
    std::array<uint8_t, kPayloadByteLength> expected;
    for (size_t i = 0; i < kValues.size(); ++i) {
      const uint32_t lhs = static_cast<uint32_t>(i + 17);
      const uint32_t rhs = static_cast<uint32_t>(i * 3 + 11);
      const size_t offset = i * sizeof(uint32_t);
      StoreU32(bindings_[0].bytes(), kBindingByteOffset + offset, lhs);
      StoreU32(bindings_[1].bytes(), kBindingByteOffset + offset, rhs);
      StoreU32(expected, offset, lhs * rhs);
    }
    // No CPU access or cache operation touches output after construction.
    // Initial contents, if any, were written before registration.
    for (size_t ordinal : {0u, 1u}) {
      ASSERT_EQ(api_->host_mapping_cache_control(
                    bindings_[ordinal].mapping,
                    ingress_[ordinal].release.host_operation,
                    kBindingByteOffset, kPayloadByteLength),
                AMDF_STATUS_OK);
    }
    ASSERT_EQ(
        api_->host_mapping_cache_control(
            execution_.instructions.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
            execution_.instructions.host.byte_length),
        AMDF_STATUS_OK);
    amdf_xdna_kernel_queue_submission_info_t submit = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submit),
        .command_count = 1,
        .commands = &execution_.command,
    };
    uint64_t point = 0;
    ASSERT_EQ(xdna_api_->kernel_queue_submit(execution_.queue, &submit, &point),
              AMDF_STATUS_OK);
    ASSERT_EQ(api_->kernel_queue_wait(execution_.queue, point,
                                      amdf::cts::kWaitTimeoutNs,
                                      amdf::cts::kWaitPollIntervalNs),
              AMDF_STATUS_OK);
    const bool has_initial_contents =
        registered_contents == RegisteredContents::kHostWritten;
    ASSERT_EQ(api_->host_mapping_cache_control(
                  bindings_[2].mapping, egress_[2].acquire.host_operation,
                  has_initial_contents ? 0 : kBindingByteOffset,
                  has_initial_contents ? bindings_[2].host.byte_length
                                       : kPayloadByteLength),
              AMDF_STATUS_OK);
    const auto output =
        bindings_[2].bytes().subspan(kBindingByteOffset, kPayloadByteLength);
    const auto mismatch =
        std::mismatch(output.begin(), output.end(), expected.begin());
    if (mismatch.first != output.end()) {
      EXPECT_EQ(*mismatch.first, *mismatch.second)
          << "output byte " << std::distance(output.begin(), mismatch.first);
    }
    if (has_initial_contents) {
      const auto bytes = bindings_[2].bytes();
      const auto prefix = bytes.first(kBindingByteOffset);
      const auto suffix =
          bytes.subspan(kBindingByteOffset + kPayloadByteLength);
      EXPECT_TRUE(std::all_of(prefix.begin(), prefix.end(),
                              [](uint8_t value) { return value == 0xA5; }));
      EXPECT_TRUE(std::all_of(suffix.begin(), suffix.end(),
                              [](uint8_t value) { return value == 0xA5; }));
    }
  }

  void RunRoundTrips(uint32_t count = 8,
                     FirmwareTraceCapture* capture = nullptr) {
    ASSERT_NO_FATAL_FAILURE(
        PrepareExecution(&execution_, 1, binding_addresses_));
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
    for (uint32_t generation = 0; generation < count && !HasFailure();
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
      if (capture != nullptr) {
        ASSERT_NO_FATAL_FAILURE(capture->BeginCommand(generation));
      }
      ASSERT_EQ(
          xdna_api_->kernel_queue_submit(execution_.queue, &submit, &point),
          AMDF_STATUS_OK);
      ASSERT_EQ(api_->kernel_queue_wait(execution_.queue, point,
                                        amdf::cts::kWaitTimeoutNs,
                                        amdf::cts::kWaitPollIntervalNs),
                AMDF_STATUS_OK);
      if (capture != nullptr) {
        ASSERT_NO_FATAL_FAILURE(capture->EndCommand(generation));
      }
      // The output transfer joins both input records and all output writes.
      // Remaining resident activity accesses only tile-local state.
      // Native completion supplies ordering; these queried actions supply CPU
      // visibility. Snapshot all payloads before checking their values. The
      // streaming case checks the immutable command owner after its final use;
      // ordinary memory recipes check that owner after every submission.
      for (size_t ordinal : {2u, 0u, 1u}) {
        ASSERT_EQ(api_->host_mapping_cache_control(
                      bindings_[ordinal].mapping,
                      egress_[ordinal].acquire.host_operation, 0,
                      bindings_[ordinal].host.byte_length),
                  AMDF_STATUS_OK);
        const auto bytes = bindings_[ordinal].bytes();
        std::copy(bytes.begin(), bytes.end(), observed[ordinal].begin());
      }
      if (capture == nullptr || generation + 1 == count) {
        ASSERT_EQ(api_->host_mapping_cache_control(
                      execution_.instructions.mapping,
                      AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, commands.size()),
                  AMDF_STATUS_OK);
        std::copy(commands.begin(), commands.end(), observed_commands.begin());
        EXPECT_EQ(observed_commands, original_commands);
      }
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        SCOPED_TRACE(ordinal);
        const auto mismatch =
            std::mismatch(observed[ordinal].begin(), observed[ordinal].end(),
                          expected[ordinal].begin());
        EXPECT_EQ(mismatch.first, observed[ordinal].end())
            << "binding byte "
            << std::distance(observed[ordinal].begin(), mismatch.first);
      }
      amdf_kernel_queue_status_t status = {
          .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
          .structure_size = sizeof(status),
      };
      ASSERT_EQ(api_->kernel_queue_query_status(execution_.queue, &status),
                AMDF_STATUS_OK);
      EXPECT_EQ(status.retired_submission, point);
      EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
      if (capture != nullptr && generation % 64 == 63) {
        ASSERT_NO_FATAL_FAILURE(capture->Read());
      }
    }
    if (capture != nullptr) {
      ASSERT_NO_FATAL_FAILURE(capture->Drain(count));
    }
  }

  void RunContextHandoffs(ContextWidth width) {
    amdf_xdna_device_info_t device_info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO,
        .structure_size = sizeof(device_info),
    };
    ASSERT_EQ(xdna_api_->device_query_info(device_, &device_info),
              AMDF_STATUS_OK);
    const uint32_t column_count =
        width == ContextWidth::kFullArray ? device_info.array.column_count : 1;
    const auto& context = device_info.context;
    ASSERT_GT(context.column_count_granularity, 0u);
    if (column_count < context.minimum_column_count ||
        column_count > context.maximum_column_count ||
        (column_count - context.minimum_column_count) %
                context.column_count_granularity !=
            0) {
      GTEST_SKIP() << "requested logical context width is not advertised";
    }
    RecordProperty("logical_column_count", column_count);
    ASSERT_NO_FATAL_FAILURE(
        PrepareExecution(&execution_, column_count, binding_addresses_));
    const std::array<uint64_t, 3> consumer_addresses = {
        binding_addresses_[2], binding_addresses_[1], binding_addresses_[0]};
    ASSERT_NO_FATAL_FAILURE(
        PrepareExecution(&consumer_, column_count, consumer_addresses));

    // The producer writes (lhs * rhs) to shared backing. The consumer reads it
    // directly and writes (product * rhs) to the original lhs. Full-width
    // contexts cannot satisfy this handoff with disjoint physical placements.
    const std::array<XdnaExecution*, 2> executions = {&execution_, &consumer_};
    std::array<std::vector<uint8_t>, 2> original_commands;
    std::array<std::vector<uint8_t>, 2> observed_commands;
    for (size_t i = 0; i < executions.size(); ++i) {
      const auto& instructions = executions[i]->instructions;
      const auto bytes = instructions.bytes();
      original_commands[i].assign(bytes.begin(), bytes.end());
      observed_commands[i].resize(bytes.size());
      ASSERT_EQ(api_->host_mapping_cache_control(
                    instructions.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                    bytes.size()),
                AMDF_STATUS_OK);
    }
    std::array<std::vector<uint8_t>, 3> expected;
    std::array<std::vector<uint8_t>, 3> observed;
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      observed[ordinal].resize(bindings_[ordinal].host.byte_length);
    }
    std::array<uint32_t, kValues.size()> products = {};
    for (uint32_t generation = 0; generation < 5 && !HasFailure();
         ++generation) {
      SCOPED_TRACE(generation);
      if (generation == 3) {
        // Ordinary backing borrows the device, not this context or queue. The
        // remaining consumer must continue using the last producer result.
        ASSERT_TRUE(execution_.Release(api_, xdna_api_));
      }
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        if (generation >= 3 && ordinal == 2) {
          continue;
        }
        const auto bytes = bindings_[ordinal].bytes();
        std::fill(bytes.begin(), bytes.end(), uint8_t(0xA5 ^ generation));
        expected[ordinal].assign(bytes.begin(), bytes.end());
      }
      for (size_t i = 0; i < kValues.size(); ++i) {
        const size_t offset = kBindingByteOffset + i * sizeof(uint32_t);
        const uint32_t factor =
            static_cast<uint32_t>(i * 2 + generation * 2 + 3);
        StoreU32(bindings_[1].bytes(), offset, factor);
        StoreU32(expected[1], offset, factor);
        if (generation < 3) {
          const uint32_t lhs = kValues[(i + generation) % kValues.size()];
          products[i] = lhs * factor;
          StoreU32(bindings_[0].bytes(), offset, lhs);
          StoreU32(bindings_[2].bytes(), offset, ~products[i]);
          StoreU32(expected[2], offset, products[i]);
        } else {
          StoreU32(bindings_[0].bytes(), offset, ~(products[i] * factor));
        }
        StoreU32(expected[0], offset, products[i] * factor);
      }
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        if (generation >= 3 && ordinal == 2) {
          continue;
        }
        ASSERT_EQ(api_->host_mapping_cache_control(
                      bindings_[ordinal].mapping,
                      ingress_[ordinal].release.host_operation, 0,
                      bindings_[ordinal].host.byte_length),
                  AMDF_STATUS_OK);
      }
      std::array<uint64_t, 2> points = {};
      for (size_t i = 0; i < executions.size(); ++i) {
        const auto& execution = *executions[i];
        if (!execution.queue) {
          continue;
        }
        const amdf_xdna_kernel_queue_submission_info_t submit = {
            .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
            .structure_size = sizeof(submit),
            .command_count = 1,
            .commands = &execution.command,
        };
        ASSERT_EQ(xdna_api_->kernel_queue_submit(execution.queue, &submit,
                                                 &points[i]),
                  AMDF_STATUS_OK);
        ASSERT_EQ(api_->kernel_queue_wait(execution.queue, points[i],
                                          AMDF_TIMEOUT_INFINITE, 0),
                  AMDF_STATUS_OK);
        // No CPU payload access or cache operation separates these commands.
      }
      // Capture both device-written buffers before checking numerical output,
      // command immutability or terminal status.
      for (size_t ordinal : {0u, 2u, 1u}) {
        ASSERT_EQ(api_->host_mapping_cache_control(
                      bindings_[ordinal].mapping,
                      egress_[ordinal].acquire.host_operation, 0,
                      bindings_[ordinal].host.byte_length),
                  AMDF_STATUS_OK);
        const auto bytes = bindings_[ordinal].bytes();
        std::copy(bytes.begin(), bytes.end(), observed[ordinal].begin());
      }
      for (size_t i = 0; i < executions.size(); ++i) {
        if (!executions[i]->queue) {
          continue;
        }
        const auto& instructions = executions[i]->instructions;
        ASSERT_EQ(
            api_->host_mapping_cache_control(
                instructions.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                instructions.host.byte_length),
            AMDF_STATUS_OK);
        const auto bytes = instructions.bytes();
        std::copy(bytes.begin(), bytes.end(), observed_commands[i].begin());
      }
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        SCOPED_TRACE(ordinal);
        EXPECT_EQ(observed[ordinal], expected[ordinal]);
      }
      for (size_t i = 0; i < executions.size(); ++i) {
        if (!executions[i]->queue) {
          continue;
        }
        SCOPED_TRACE(i);
        EXPECT_EQ(observed_commands[i], original_commands[i]);
        amdf_kernel_queue_status_t status = {
            .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
            .structure_size = sizeof(status),
        };
        ASSERT_EQ(
            api_->kernel_queue_query_status(executions[i]->queue, &status),
            AMDF_STATUS_OK);
        EXPECT_EQ(status.retired_submission, points[i]);
        EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
      }
    }
  }

  // Native family selected before allocating instructions or payloads.
  uint32_t family_ordinal_ = UINT32_MAX;
  // Immutable, admitted fixture borrowing build-generated Loom output.
  XdnaExecutable executable_;
  // Private context, command backing and one native queue.
  XdnaExecution execution_;
  // Independent context and commands consuming ordinary shared payloads.
  XdnaExecution consumer_;
  // CPU-only storage retained through native registration release.
  std::array<CtsMappedMemory, 3> registered_storage_;
  // Temporary exporter, closed before the imported range is mapped or used.
  CtsMappedMemory export_source_;
  // Move-owned export consumed by import, or released after a failed import.
  amdf_external_memory_t external_memory_ = {};
  // Lhs, rhs and output, each with a complete guarded host view.
  std::array<CtsMappedMemory, 3> bindings_;
  // Independently queried native DMA addresses including each logical offset.
  std::array<uint64_t, 3> binding_addresses_ = {};
  // Queried CPU publication actions paired with NPU ingress.
  std::array<amdf_memory_pair_info_t, 3> ingress_ = {};
  // Queried CPU acquisition actions paired with drained NPU output.
  std::array<amdf_memory_pair_info_t, 3> egress_ = {};
};

#if defined(AMDF_CTS_XDNA_CXX_WORKER)
class CxxWorkerCpuXdnaRecipeTest : public CpuXdnaRecipeTest {
 protected:
  const iree_file_toc_t* ProgramImages() const override {
    return amdf_cts_xdna_mul_i32_cxx_create();
  }
};
#endif  // AMDF_CTS_XDNA_CXX_WORKER

// Finite capture controls share the workload context but borrow no storage on
// its behalf. Cleanup deliberately stops if native retirement becomes unknown.
class FirmwareResultRecipeTest : public CpuXdnaRecipeTest {
 protected:
  const iree_file_toc_t* ProgramImages() const override {
    return amdf_cts_xdna_recurrence_i32_create();
  }

  void FindCoreEnable(std::span<const uint8_t> original) {
    // This finite one-worker fixture has one core enable after configuration.
    // Locate it once during preparation; binding only patches DMA addresses.
    size_t offset = 16;
    uint32_t count = 0;
    while (offset < original.size()) {
      const auto operation = original.subspan(offset);
      ASSERT_GE(operation.size(), 8u);
      size_t length_offset = 0;
      switch (operation[0]) {
        case 0:
          length_offset = 20;
          break;  // WRITE32.
        case 1:
          length_offset = 12;
          break;  // BLOCKWRITE32.
        case 3:
        case 4:
          length_offset = 24;
          break;  // MASKWRITE/POLL32.
        case 128:
        case 129:
        case 130:
          length_offset = 4;
          break;
        default:
          FAIL() << "unexpected transaction opcode " << +operation[0];
      }
      ASSERT_GE(operation.size(), length_offset + 4);
      auto word = [&](size_t position) {
        uint32_t value;
        std::memcpy(&value, operation.data() + position, sizeof(value));
        return value;
      };
      const uint32_t length = word(length_offset);
      ASSERT_GE(length, length_offset + 4);
      ASSERT_LE(length, operation.size());
      if ((operation[0] == 0 || operation[0] == 3) &&
          (word(8) & 0xfffff) == 0x32000 && (word(16) & 1) != 0 &&
          (operation[0] == 0 || (word(20) & 1) != 0)) {
        ASSERT_EQ(core_enable_offset_, 0u);
        core_enable_offset_ = offset;
      }
      offset += length;
      ++count;
    }
    ASSERT_EQ(count, operation_count_);
    ASSERT_NE(core_enable_offset_, 0u);
  }

  void TearDown() override {
    if (control_failed_) {
      ADD_FAILURE() << "retaining firmware result backing after native failure";
      return;
    }
    if (last_point_ != 0) {
      const auto status = api_->kernel_queue_wait(
          execution_.queue, last_point_, amdf::cts::kWaitTimeoutNs,
          amdf::cts::kWaitPollIntervalNs);
      ASSERT_EQ(status, AMDF_STATUS_OK);
    }
    if (attached_) {
      ASSERT_NO_FATAL_FAILURE(Detach());
    }
    if (!result_storage_.Release(api_)) {
      return;
    }
    CpuXdnaRecipeTest::TearDown();
  }

  void PrepareResults(uint32_t dispatch_count = 4) {
    dispatch_count_ = dispatch_count;
    ASSERT_NO_FATAL_FAILURE(CreateBindings(
        AMDF_MEMORY_PROFILE_ROLE_CREATE, RegisteredContents::kUntouched,
        dispatch_count_ * kBindingStorageByteLength));
    image_storage_.resize(executable_.allocation_byte_length());
    executable_.Load(image_storage_);
    ASSERT_TRUE(executable_.Bind(image_storage_, binding_addresses_));
    const auto original = executable_.ResolveInvocation(image_storage_);
    // Version 0.1 controller header followed by the unchanged Loom program.
    ASSERT_GE(original.size(), 16u);
    ASSERT_EQ(original[0], 0u);
    ASSERT_EQ(original[1], 1u);
    std::memcpy(&operation_count_, original.data() + 8, 4);
    ASSERT_NO_FATAL_FAILURE(FindCoreEnable(original));
    command_length_ = original.size() + 3 * 12;
    const uint64_t alignment = executable_.allocation_alignment();
    command_stride_ = (command_length_ + alignment - 1) / alignment * alignment;
    amdf_xdna_context_create_info_t context = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO,
        .structure_size = sizeof(context),
        .logical_column_count = 1,
        .physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY,
        .acceptable_scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
    };
    ASSERT_EQ(xdna_api_->context_create(device_, &context, &execution_.context),
              AMDF_STATUS_OK);
    requirements_.type = AMDF_STRUCTURE_TYPE_XDNA_RESULT_BUFFER_INFO;
    requirements_.structure_size = sizeof(requirements_);
    const auto status = xdna_api_->context_query_result_buffer_info(
        execution_.context, &requirements_);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "native context protocol has no finite result service";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    EXPECT_EQ(requirements_.timer_clock.counter_bit_count, 64u);
    EXPECT_EQ(requirements_.timer_clock.frequency_hz, 0u);
    // Two complete batches append before rearming. Native collection does not
    // reset the destination, so its capacity covers both batches together.
    const uint64_t result_granularity = requirements_.byte_length_granularity;
    const uint64_t result_length =
        (std::max<uint64_t>(
             requirements_.minimum_byte_length,
             2 * dispatch_count_ * 3 * sizeof(amdf_xdna_timer_record_t)) +
         result_granularity - 1) /
        result_granularity * result_granularity;
    ASSERT_LE(result_length, requirements_.maximum_byte_length);
    amdf_memory_device_access_t access = {.device = device_};
    access.requirements.access = AMDF_MEMORY_ACCESS_READ |
                                 AMDF_MEMORY_ACCESS_WRITE |
                                 AMDF_MEMORY_ACCESS_EXECUTE;
    access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access.requirements.address_kinds = UINT64_C(1)
                                        << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    ASSERT_EQ(
        api_->memory_scope_query_device_profile(
            requirements_.memory_scope, requirements_.memory_profile_ordinal, 1,
            &access, &profile, &capabilities),
        AMDF_STATUS_OK);
    const uint64_t granularity = profile.allocation.byte_length_granularity;
    // Leave a guard page between instructions and the result slice. Whole-BO
    // providers instead use a separate allocation with the exact admitted size.
    const uint64_t storage_length =
        dispatch_count_ * command_stride_ + alignment +
        2 * requirements_.byte_offset_alignment + result_length;
    amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal = profile.ordinal,
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .byte_length =
            (storage_length + granularity - 1) / granularity * granularity,
        .minimum_alignment = profile.allocation.minimum_alignment,
        .accesses = &access,
    };
    ASSERT_NO_FATAL_FAILURE(execution_.instructions.Create(
        api_, requirements_.memory_scope, create));
    auto instructions = execution_.instructions.bytes();
    std::fill(instructions.begin(), instructions.end(), 0xA5);
    uint64_t address = 0;
    ASSERT_EQ(
        api_->memory_query_address(execution_.instructions.memory, 0,
                                   AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE, &address),
        AMDF_STATUS_OK);
    command_offset_ = (alignment - address % alignment) % alignment;
    if (requirements_.binding_mode ==
        AMDF_XDNA_RESULT_BUFFER_BINDING_WHOLE_MEMORY) {
      create.byte_length = result_length;
      ASSERT_NO_FATAL_FAILURE(
          result_storage_.Create(api_, requirements_.memory_scope, create));
      result_memory_ = &result_storage_;
    } else {
      ASSERT_EQ(requirements_.binding_mode,
                AMDF_XDNA_RESULT_BUFFER_BINDING_RANGE);
      result_memory_ = &execution_.instructions;
      const uint64_t result_alignment = requirements_.byte_offset_alignment;
      binding_.byte_offset =
          (command_offset_ + dispatch_count_ * command_stride_ +
           2 * result_alignment - 1) /
          result_alignment * result_alignment;
    }
    binding_.memory = result_memory_->memory;
    binding_.byte_length = result_length;
    execution_.command = {
        .memory = execution_.instructions.memory,
        .byte_offset = command_offset_,
        .byte_length = command_length_,
    };
    amdf_xdna_kernel_queue_create_info_t queue = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO,
        .structure_size = sizeof(queue),
        .queue_family_ordinal = family_ordinal_,
        // Arrays consume command packets as well as their carrier. Prepare
        // enough native storage to publish the full batch before its one wait.
        .maximum_pending_submission_count = 2 * dispatch_count_,
    };
    ASSERT_EQ(xdna_api_->kernel_queue_create(execution_.context, &queue,
                                             &execution_.queue),
              AMDF_STATUS_OK);
  }

  std::span<uint8_t> ResultBytes() {
    return result_memory_->bytes().subspan(binding_.byte_offset,
                                           binding_.byte_length);
  }

  void Attach() {
    auto bytes = ResultBytes();
    std::fill(bytes.begin(), bytes.end(), 0);
    ASSERT_EQ(api_->host_mapping_cache_control(
                  result_memory_->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH,
                  binding_.byte_offset, binding_.byte_length),
              AMDF_STATUS_OK);
    const auto status =
        xdna_api_->context_attach_result_buffer(execution_.context, &binding_);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "native driver supplies no finite result attachment";
    }
    control_failed_ = !amdf_status_is_ok(status);
    ASSERT_EQ(status, AMDF_STATUS_OK);
    attached_ = true;
    expected_ids_.clear();
  }

  void Detach() {
    const auto status =
        xdna_api_->context_detach_result_buffer(execution_.context, &binding_);
    control_failed_ = !amdf_status_is_ok(status);
    ASSERT_EQ(status, AMDF_STATUS_OK);
    attached_ = false;
  }

  void Collect() {
    const auto status =
        xdna_api_->context_collect_result_buffer(execution_.context, &binding_);
    control_failed_ = !amdf_status_is_ok(status);
    ASSERT_EQ(status, AMDF_STATUS_OK);
    ASSERT_EQ(api_->host_mapping_cache_control(
                  result_memory_->mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                  binding_.byte_offset, binding_.byte_length),
              AMDF_STATUS_OK);
    auto bytes = ResultBytes();
    uint64_t previous = 0;
    for (size_t i = 0; i < expected_ids_.size(); ++i) {
      amdf_xdna_timer_record_t record;
      std::memcpy(&record, bytes.data() + i * sizeof(record), sizeof(record));
      ASSERT_EQ(record.id, expected_ids_[i]);
      const uint64_t timestamp = amdf_xdna_timer_record_timestamp(&record);
      ASSERT_NE(timestamp, 0u);
      // The final pair brackets core enable through output DMA completion,
      // after all array reconfiguration. The pre-configuration marker can
      // belong to a different timer epoch.
      if (i % 3 == 2) {
        EXPECT_GT(timestamp, previous);
      }
      previous = timestamp;
    }
    if (result_memory_ == &execution_.instructions) {
      const uint64_t guard_offset =
          command_offset_ + dispatch_count_ * command_stride_;
      const uint64_t guard_length = binding_.byte_offset - guard_offset;
      ASSERT_EQ(
          api_->host_mapping_cache_control(result_memory_->mapping,
                                           AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                                           guard_offset, guard_length),
          AMDF_STATUS_OK);
      const auto guard =
          result_memory_->bytes().subspan(guard_offset, guard_length);
      EXPECT_TRUE(std::all_of(guard.begin(), guard.end(),
                              [](uint8_t value) { return value == 0xA5; }));
    }
    const auto unused =
        bytes.subspan(expected_ids_.size() * sizeof(amdf_xdna_timer_record_t));
    EXPECT_TRUE(std::all_of(unused.begin(), unused.end(),
                            [](uint8_t value) { return value == 0; }));
  }

  void RunBatch(uint32_t generation, uint32_t iterations = 1,
                uint32_t commands_per_submission = 1,
                bool reuse_commands = false) {
    for (auto& memory : bindings_) {
      auto bytes = memory.bytes();
      std::fill(bytes.begin(), bytes.end(), 0xA5);
    }
    for (uint32_t dispatch = 0; dispatch < dispatch_count_; ++dispatch) {
      for (uint32_t i = 0; i < 16; ++i) {
        const size_t offset =
            dispatch * kBindingStorageByteLength + kBindingByteOffset + i * 4;
        StoreU32(bindings_[0].bytes(), offset, generation * 17 + i + dispatch);
        StoreU32(bindings_[1].bytes(), offset,
                 generation * 11 + i * 3 + dispatch);
      }
      StoreU32(
          bindings_[1].bytes(),
          dispatch * kBindingStorageByteLength + kBindingByteOffset + 15 * 4,
          iterations);
      if (!reuse_commands) {
        auto addresses = binding_addresses_;
        for (auto& address : addresses) {
          address += dispatch * kBindingStorageByteLength;
        }
        ASSERT_TRUE(executable_.Bind(image_storage_, addresses));
        const auto original = executable_.ResolveInvocation(image_storage_);
        auto command = execution_.instructions.bytes().subspan(
            command_offset_ + dispatch * command_stride_, command_length_);
        std::copy_n(original.begin(), 16, command.begin());
        StoreU32(command, 8, operation_count_ + 3);
        StoreU32(command, 12, static_cast<uint32_t>(command_length_));
        std::copy(original.begin() + 16, original.begin() + core_enable_offset_,
                  command.begin() + 28);
        std::copy(original.begin() + core_enable_offset_, original.end(),
                  command.begin() + core_enable_offset_ + 24);
        for (uint32_t marker = 0; marker < 3; ++marker) {
          const size_t offset = marker == 0   ? 16
                                : marker == 1 ? core_enable_offset_ + 12
                                              : command_length_ - 12;
          const uint32_t id = marker + 1;
          StoreU32(command, offset, 0x83);  // RECORD_TIMER.
          StoreU32(command, offset + 4, 12);
          StoreU32(command, offset + 8, id);
        }
      }
      for (uint32_t marker = 0; marker < 3; ++marker) {
        expected_ids_.push_back(marker + 1);
      }
    }
    for (size_t i = 0; i < bindings_.size(); ++i) {
      ASSERT_EQ(api_->host_mapping_cache_control(
                    bindings_[i].mapping, ingress_[i].release.host_operation, 0,
                    bindings_[i].host.byte_length),
                AMDF_STATUS_OK);
    }
    if (!reuse_commands) {
      ASSERT_EQ(
          api_->host_mapping_cache_control(
              execution_.instructions.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH,
              command_offset_, dispatch_count_ * command_stride_),
          AMDF_STATUS_OK);
    }
    const auto commands = execution_.instructions.bytes().subspan(
        command_offset_, dispatch_count_ * command_stride_);
    const std::vector<uint8_t> command_snapshot(commands.begin(),
                                                commands.end());
    std::vector<amdf_xdna_kernel_command_t> native_commands(dispatch_count_,
                                                            execution_.command);
    for (uint32_t dispatch = 0; dispatch < dispatch_count_; ++dispatch) {
      native_commands[dispatch].byte_offset += dispatch * command_stride_;
    }
    for (uint32_t dispatch = 0; dispatch < dispatch_count_;) {
      const uint32_t count =
          std::min(commands_per_submission, dispatch_count_ - dispatch);
      const amdf_xdna_kernel_queue_submission_info_t submit = {
          .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
          .structure_size = sizeof(submit),
          .command_count = count,
          .commands = native_commands.data() + dispatch,
      };
      uint64_t point = 0;
      ASSERT_EQ(
          xdna_api_->kernel_queue_submit(execution_.queue, &submit, &point),
          AMDF_STATUS_OK);
      ASSERT_GT(point, last_point_);
      last_point_ = point;
      dispatch += count;
    }
    ASSERT_EQ(api_->kernel_queue_wait(execution_.queue, last_point_,
                                      amdf::cts::kWaitTimeoutNs,
                                      amdf::cts::kWaitPollIntervalNs),
              AMDF_STATUS_OK);
    ASSERT_EQ(
        api_->host_mapping_cache_control(execution_.instructions.mapping,
                                         AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                                         command_offset_, commands.size()),
        AMDF_STATUS_OK);
    EXPECT_TRUE(std::equal(command_snapshot.begin(), command_snapshot.end(),
                           commands.begin()));
    ASSERT_EQ(api_->host_mapping_cache_control(
                  bindings_[2].mapping, egress_[2].acquire.host_operation, 0,
                  bindings_[2].host.byte_length),
              AMDF_STATUS_OK);
    for (uint32_t dispatch = 0; dispatch < dispatch_count_; ++dispatch) {
      for (uint32_t i = 0; i < 16; ++i) {
        uint32_t actual;
        std::memcpy(&actual,
                    bindings_[2].bytes().data() +
                        dispatch * kBindingStorageByteLength +
                        kBindingByteOffset + i * 4,
                    4);
        const uint32_t lhs = generation * 17 + i + dispatch;
        const uint32_t rhs =
            i == 15 ? iterations : generation * 11 + i * 3 + dispatch;
        uint32_t expected = lhs;
        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
          expected = expected * lhs + rhs;
        }
        ASSERT_EQ(actual, expected);
      }
    }
    auto output = bindings_[2].bytes();
    for (size_t i = 0; i < output.size(); ++i) {
      if (i >= dispatch_count_ * kBindingStorageByteLength ||
          i % kBindingStorageByteLength < kBindingByteOffset ||
          i % kBindingStorageByteLength >= kBindingByteOffset + 64) {
        ASSERT_EQ(output[i], 0xA5) << i;
      }
    }
    amdf_kernel_queue_status_t status = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(status),
    };
    ASSERT_EQ(api_->kernel_queue_query_status(execution_.queue, &status),
              AMDF_STATUS_OK);
    EXPECT_EQ(status.retired_submission, last_point_);
    EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
  }

  void RunTimerObservations(bool use_firmware_clock) {
    ASSERT_NO_FATAL_FAILURE(PrepareResults());
    if (IsSkipped()) {
      return;
    }
    std::array<FirmwareCommandInterval, 12> firmware_intervals;
    FirmwareTraceCapture capture(xdna_api_, device_, host_clock_,
                                 firmware_intervals);
    if (use_firmware_clock) {
      amdf_xdna_firmware_trace_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_XDNA_FIRMWARE_TRACE_INFO,
          .structure_size = sizeof(info),
      };
      const auto status =
          xdna_api_->device_query_firmware_trace_info(device_, &info);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        GTEST_SKIP() << "native driver supplies no independent firmware clock";
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(capture.Start(info));
    }
    constexpr std::array<uint32_t, 3> kIterations = {1, 4096, 16384};
    for (uint32_t batch = 0; batch < kIterations.size(); ++batch) {
      SCOPED_TRACE(batch);
      ASSERT_NO_FATAL_FAILURE(Attach());
      if (IsSkipped()) {
        return;
      }
      amdf_observation_clock_info_t before = {};
      const auto status =
          xdna_api_->device_query_timer_clock_info(device_, &before);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        GTEST_SKIP() << "native driver supplies no timer operating rate";
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      ASSERT_GT(before.frequency_hz, 0u);
      uint64_t host_begin = 0;
      ASSERT_EQ(host_clock_.Read(&host_begin), AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(RunBatch(batch + 1, kIterations[batch]));
      uint64_t host_end = 0;
      ASSERT_EQ(host_clock_.Read(&host_end), AMDF_STATUS_OK);
      const double host_duration_ns =
          host_clock_.Nanoseconds(host_end - host_begin);
      ASSERT_GT(host_duration_ns, 0);
      amdf_observation_clock_info_t after = {};
      ASSERT_EQ(xdna_api_->device_query_timer_clock_info(device_, &after),
                AMDF_STATUS_OK);
      ASSERT_GT(after.frequency_hz, 0u);
      ASSERT_NO_FATAL_FAILURE(Collect());
      if (use_firmware_clock) {
        ASSERT_NO_FATAL_FAILURE(capture.Drain((batch + 1) * 4));
      }
      const std::string prefix = "timer_batch_" + std::to_string(batch);
      RecordProperty(prefix + "_iterations", kIterations[batch]);
      RecordProperty(prefix + "_frequency_before_hz",
                     std::to_string(before.frequency_hz));
      RecordProperty(prefix + "_frequency_after_hz",
                     std::to_string(after.frequency_hz));
      // Endpoint snapshots neither hold the rate nor report intervening
      // changes. Retain this conversion as a model to evaluate against the
      // independent brackets, without making it the marker transport oracle.
      bool endpoint_rate_model_fits = before.frequency_hz == after.frequency_hz;
      const double nanoseconds_per_tick = 1e9 / before.frequency_hz;
      double estimated_sum_ns = 0;
      uint64_t firmware_sum_ns = 0;
      for (uint32_t dispatch = 0; dispatch < 4; ++dispatch) {
        std::array<amdf_xdna_timer_record_t, 3> records;
        std::memcpy(records.data(),
                    ResultBytes().data() + dispatch * sizeof(records),
                    sizeof(records));
        const uint64_t begin = amdf_xdna_timer_record_timestamp(&records[1]);
        const uint64_t end = amdf_xdna_timer_record_timestamp(&records[2]);
        ASSERT_GT(end, begin);
        const double estimate_ns = (end - begin) * nanoseconds_per_tick;
        estimated_sum_ns += estimate_ns;
        const std::string sample =
            prefix + "_dispatch_" + std::to_string(dispatch);
        RecordProperty(sample + "_begin_ticks", std::to_string(begin));
        RecordProperty(sample + "_end_ticks", std::to_string(end));
        RecordProperty(sample + "_ticks", std::to_string(end - begin));
        RecordProperty(sample + "_endpoint_rate_estimate_ns",
                       std::to_string(estimate_ns));
        if (use_firmware_clock) {
          const auto& interval =
              firmware_intervals[batch * 4 + dispatch].device;
          ASSERT_GT(interval.end, interval.begin);
          const uint64_t firmware_ns = interval.end - interval.begin;
          firmware_sum_ns += firmware_ns;
          endpoint_rate_model_fits &= estimate_ns <= firmware_ns;
          RecordProperty(sample + "_firmware_duration_ns",
                         std::to_string(firmware_ns));
        }
      }
      // Firmware events supply nanoseconds independently of the array clock.
      // Their serial command intervals must fit the enclosing host batch.
      if (use_firmware_clock) {
        EXPECT_LE(double(firmware_sum_ns), host_duration_ns);
      }
      endpoint_rate_model_fits &= estimated_sum_ns <= host_duration_ns;
      RecordProperty(prefix + "_endpoint_rate_model_fits",
                     endpoint_rate_model_fits);
      RecordProperty(prefix + "_host_bracket_ns",
                     std::to_string(host_duration_ns));
      ASSERT_NO_FATAL_FAILURE(Detach());
    }
  }

  // Queried immutable scope, native binding geometry and timer interpretation.
  amdf_xdna_result_buffer_info_t requirements_ = {};
  // Whole-BO providers require a separate result allocation.
  CtsMappedMemory result_storage_;
  // Borrowed host view of either result_storage_ or the instruction aperture.
  CtsMappedMemory* result_memory_ = nullptr;
  // Exact caller-owned association supplied unchanged to all three controls.
  amdf_xdna_result_buffer_binding_t binding_ = {};
  // True only while a successful attachment has not been removed.
  bool attached_ = false;
  // A failed native control prevents speculative resource release in teardown.
  bool control_failed_ = false;
  // Last accepted application submission, independent of native control points.
  uint64_t last_point_ = 0;
  // Number of independently bound numeric invocations in each batch.
  uint32_t dispatch_count_ = 0;
  // Case-owned image used to resolve the distinct numeric invocations.
  std::vector<uint8_t> image_storage_;
  // Native operation count before inserting the three marker instructions.
  uint32_t operation_count_ = 0;
  // Native position after configuration and immediately before core enable.
  size_t core_enable_offset_ = 0;
  // Complete transaction length including markers.
  uint64_t command_length_ = 0;
  // Firmware-aligned distance between immutable command instances.
  uint64_t command_stride_ = 0;
  // Offset aligning the first command to its queried firmware address.
  uint64_t command_offset_ = 0;
  // Known finite record protocol accumulated since the most recent attachment.
  std::vector<uint32_t> expected_ids_;
};

TEST_F(FirmwareResultRecipeTest,
       AppendsCollectsAndRearmsWithoutChangingQueuePoints) {
  ASSERT_NO_FATAL_FAILURE(PrepareResults());
  if (IsSkipped()) {
    return;
  }
  for (auto control : {xdna_api_->context_attach_result_buffer,
                       xdna_api_->context_collect_result_buffer,
                       xdna_api_->context_detach_result_buffer}) {
    auto invalid = binding_;
    invalid.byte_offset += 1;
    EXPECT_EQ(control(execution_.context, &invalid),
              amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
    invalid = binding_;
    invalid.byte_length = 0;
    EXPECT_EQ(control(execution_.context, &invalid),
              amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
    invalid = binding_;
    invalid.memory = bindings_[0].memory;
    EXPECT_EQ(control(execution_.context, &invalid),
              amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  }
  ASSERT_NO_FATAL_FAILURE(Attach());
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(Collect());
  for (uint32_t generation = 1; generation <= 3; ++generation) {
    SCOPED_TRACE(generation);
    if (generation == 3) {
      ASSERT_NO_FATAL_FAILURE(Detach());
      ASSERT_NO_FATAL_FAILURE(Attach());
      if (IsSkipped()) {
        return;
      }
    }
    ASSERT_NO_FATAL_FAILURE(RunBatch(generation));
    ASSERT_NO_FATAL_FAILURE(Collect());
    const auto bytes = ResultBytes();
    const std::vector<uint8_t> snapshot(bytes.begin(), bytes.end());
    ASSERT_NO_FATAL_FAILURE(Collect());
    EXPECT_TRUE(std::equal(snapshot.begin(), snapshot.end(), bytes.begin()));
  }
  ASSERT_NO_FATAL_FAILURE(Detach());
}

TEST_F(FirmwareResultRecipeTest, MarkerTicksAndRateSnapshots) {
  RunTimerObservations(false);
}

TEST_F(FirmwareResultRecipeTest, MarkerTicksAndFirmwareIntervals) {
  RunTimerObservations(true);
}

TEST_F(FirmwareResultRecipeTest, NativeArraysAppendCollectAndRearm) {
  // Cross both the Linux 64-command and Windows 36-command packet boundaries.
  ASSERT_NO_FATAL_FAILURE(PrepareResults(65));
  if (IsSkipped()) {
    return;
  }
  amdf_kernel_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(api_->kernel_queue_query_info(execution_.queue, &info),
            AMDF_STATUS_OK);
  if (info.maximum_command_count < 2) {
    GTEST_SKIP() << "queue does not support command arrays";
  }
  RecordProperty("commands_per_batch", dispatch_count_);
  RecordProperty("maximum_command_count", info.maximum_command_count);
  RecordProperty("result_capacity_bytes", std::to_string(binding_.byte_length));
  ASSERT_NO_FATAL_FAILURE(Attach());
  if (IsSkipped()) {
    return;
  }
  for (uint32_t generation = 1; generation <= 3; ++generation) {
    SCOPED_TRACE(generation);
    if (generation == 3) {
      ASSERT_NO_FATAL_FAILURE(Detach());
      ASSERT_NO_FATAL_FAILURE(Attach());
      if (IsSkipped()) {
        return;
      }
    }
    ASSERT_NO_FATAL_FAILURE(
        RunBatch(generation, 1, info.maximum_command_count, generation != 1));
    ASSERT_NO_FATAL_FAILURE(Collect());
    const auto bytes = ResultBytes();
    const std::vector<uint8_t> snapshot(bytes.begin(), bytes.end());
    ASSERT_NO_FATAL_FAILURE(Collect());
    EXPECT_TRUE(std::equal(snapshot.begin(), snapshot.end(), bytes.begin()));
  }
  ASSERT_NO_FATAL_FAILURE(Detach());
}

TEST_F(CpuXdnaRecipeTest, AllocatedRoundTrip) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_CREATE));
  RunRoundTrips();
}

#if defined(AMDF_CTS_XDNA_CXX_WORKER)
TEST_F(CxxWorkerCpuXdnaRecipeTest, AllocatedRoundTrip) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_CREATE));
  RunRoundTrips();
}
#endif  // AMDF_CTS_XDNA_CXX_WORKER

TEST_F(CpuXdnaRecipeTest, AllocatedRoundTripWithFirmwareTrace) {
  amdf_xdna_firmware_trace_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_FIRMWARE_TRACE_INFO,
      .structure_size = sizeof(info),
  };
  const amdf_status_t status =
      xdna_api_->device_query_firmware_trace_info(device_, &info);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "native driver does not expose firmware tracing";
  }
  ASSERT_EQ(status, AMDF_STATUS_OK);
  FirmwareTraceCapture capture(xdna_api_, device_, host_clock_);
  ASSERT_NO_FATAL_FAILURE(capture.Start(info));
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_CREATE));
  RunRoundTrips(16384, &capture);
  EXPECT_GT(capture.nonempty_reads(), 1u);
}

TEST_F(CpuXdnaRecipeTest, FirmwareDurationsAndHostClockBounds) {
  amdf_xdna_firmware_trace_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_FIRMWARE_TRACE_INFO,
      .structure_size = sizeof(info),
  };
  const auto status =
      xdna_api_->device_query_firmware_trace_info(device_, &info);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "native driver does not expose firmware tracing";
  }
  ASSERT_EQ(status, AMDF_STATUS_OK);
  std::array<FirmwareCommandInterval, 32> intervals;
  FirmwareTraceCapture capture(xdna_api_, device_, host_clock_, intervals);
  ASSERT_NO_FATAL_FAILURE(capture.Start(info));
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_CREATE));
  ASSERT_NO_FATAL_FAILURE(
      RunRoundTrips(static_cast<uint32_t>(intervals.size()), &capture));

  // For an independently attributed command in one uninterrupted source,
  // host_begin <= device_begin + offset and device_end + offset <= host_end.
  // Each command constrains a constant-offset model. Nanosecond units do not
  // guarantee a fixed offset against the host oscillator throughout a capture.
  double lower = -std::numeric_limits<double>::infinity();
  double upper = std::numeric_limits<double>::infinity();
  const uint64_t host_origin = intervals.front().host.begin;
  const uint64_t device_origin = intervals.front().device.begin;
  uint64_t minimum_duration = UINT64_MAX;
  uint64_t maximum_duration = 0;
  uint64_t previous_end = 0;
  for (const auto& interval : intervals) {
    ASSERT_GE(interval.host.begin, host_origin);
    ASSERT_GE(interval.host.end, interval.host.begin);
    ASSERT_GE(interval.device.begin, previous_end);
    ASSERT_GT(interval.device.end, interval.device.begin);
    const uint64_t duration = interval.device.end - interval.device.begin;
    ASSERT_LE(duration,
              host_clock_.Nanoseconds(interval.host.end - interval.host.begin));
    lower = std::max(
        lower, host_clock_.Nanoseconds(interval.host.begin - host_origin) -
                   double(interval.device.begin - device_origin));
    upper = std::min(upper,
                     host_clock_.Nanoseconds(interval.host.end - host_origin) -
                         double(interval.device.end - device_origin));
    minimum_duration = std::min(minimum_duration, duration);
    maximum_duration = std::max(maximum_duration, duration);
    previous_end = interval.device.end;
  }
  // These are capture observations, not performance thresholds. A slow host
  // widens uncertainty instead of causing an arbitrary latency failure.
  RecordProperty("firmware_minimum_duration_ns",
                 std::to_string(minimum_duration));
  RecordProperty("firmware_maximum_duration_ns",
                 std::to_string(maximum_duration));
  RecordProperty("firmware_host_offset_lower_ns", std::to_string(lower));
  RecordProperty("firmware_host_offset_upper_ns", std::to_string(upper));
  RecordProperty("firmware_host_constant_offset_consistent", lower <= upper);
}

TEST_F(CpuXdnaRecipeTest, RegisteredRoundTrip) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_REGISTER));
  RunRoundTrips();
}

TEST_F(CpuXdnaRecipeTest, ImportedRoundTripAfterExporterCloses) {
  ASSERT_NO_FATAL_FAILURE(ImportBindings());
  if (IsSkipped()) {
    return;
  }
  RunRoundTrips();
}

TEST_F(CpuXdnaRecipeTest, AllocatedFirstDeviceWriter) {
  RunFirstDeviceWriter(AMDF_MEMORY_PROFILE_ROLE_CREATE);
}

TEST_F(CpuXdnaRecipeTest, RegisteredFirstDeviceWriter) {
  RunFirstDeviceWriter(AMDF_MEMORY_PROFILE_ROLE_REGISTER);
}

TEST_F(CpuXdnaRecipeTest, RegisteredDirtyFirstDeviceWriter) {
  RunFirstDeviceWriter(AMDF_MEMORY_PROFILE_ROLE_REGISTER,
                       RegisteredContents::kHostWritten);
}

TEST_F(CpuXdnaRecipeTest, AllocatedSharedContextLifetime) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_CREATE));
  RunContextHandoffs(ContextWidth::kSingleColumn);
}

TEST_F(CpuXdnaRecipeTest, RegisteredSharedContextLifetime) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_REGISTER));
  RunContextHandoffs(ContextWidth::kSingleColumn);
}

TEST_F(CpuXdnaRecipeTest, ImportedSharedContextLifetime) {
  ASSERT_NO_FATAL_FAILURE(ImportBindings());
  if (IsSkipped()) {
    return;
  }
  RunContextHandoffs(ContextWidth::kSingleColumn);
}

TEST_F(CpuXdnaRecipeTest, AllocatedFullWidthContextHandoff) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_CREATE));
  RunContextHandoffs(ContextWidth::kFullArray);
}

TEST_F(CpuXdnaRecipeTest, RegisteredFullWidthContextHandoff) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_REGISTER));
  RunContextHandoffs(ContextWidth::kFullArray);
}

TEST_F(CpuXdnaRecipeTest, ImportedFullWidthContextHandoff) {
  ASSERT_NO_FATAL_FAILURE(ImportBindings());
  if (IsSkipped()) {
    return;
  }
  RunContextHandoffs(ContextWidth::kFullArray);
}

}  // namespace
