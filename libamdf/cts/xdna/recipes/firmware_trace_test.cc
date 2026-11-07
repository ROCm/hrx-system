// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "libamdf/cts/xdna/programs/mul_i32.h"
#include "libamdf/cts/xdna/util/executable.h"
#include "libamdf/cts/xdna/util/execution.h"
#include "libamdf/cts/xdna/xdna_device_fixture.h"
#include "util/mapped_memory.h"
#include "util/observation.h"
#include "util/wait.h"

namespace {

constexpr size_t kPayloadOffset = 64;
constexpr size_t kValueCount = 16;

// One ordinary queue submission and its explicitly bounded host observations.
struct CommandSample {
  // Independent observations in the portable host clock domain.
  struct {
    // Sample immediately before native publication, in host ticks.
    uint64_t begin = 0;
    // Sample after successful native retirement, in host ticks.
    uint64_t end = 0;
  } host;
  // Matched command events in the firmware nanosecond clock domain.
  struct {
    // Beginning of this command's firmware processing.
    uint64_t begin = 0;
    // Completion of this command's firmware processing.
    uint64_t end = 0;
  } firmware;
};

struct CapturedWork {
  // Independent context, immutable bound commands, and one native queue.
  XdnaExecution execution;
  // Private operands and output retained through queue and context retirement.
  std::array<CtsMappedMemory, 3> bindings;
  // Complete expected allocations, including input and padding guards.
  std::array<std::vector<uint8_t>, 3> expected;
  // Queried directional host cache actions for each operand.
  struct Edge {
    // CPU publication before device access.
    amdf_host_cache_operation_t release = AMDF_HOST_CACHE_OPERATION_NONE;
    // CPU visibility after native retirement.
    amdf_host_cache_operation_t acquire = AMDF_HOST_CACHE_OPERATION_NONE;
  };
  // Independent ingress and egress contracts, indexed by binding.
  std::array<Edge, 3> edges;
  // Original complete command owner, checked after its last execution.
  std::vector<uint8_t> commands;
  // Most recently submitted native retirement point.
  uint64_t point = 0;
  // Native identity queried before this workload's first submission.
  uint32_t firmware_context = AMDF_XDNA_FIRMWARE_CONTEXT_ID_UNASSIGNED;
  // One in-flight command at a time within this context; other contexts may
  // run.
  std::vector<CommandSample> samples;
};

class FirmwareTraceRecipeTest : public XdnaDeviceFixture {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(XdnaDeviceFixture::SetUp());
    ASSERT_GE(xdna_api_->extension_version, AMDF_XDNA_EXTENSION_VERSION_3);
    ASSERT_NO_FATAL_FAILURE(host_clock_.Initialize(api_));
    amdf_xdna_firmware_trace_info_t trace = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_FIRMWARE_TRACE_INFO,
        .structure_size = sizeof(trace),
    };
    const auto trace_status =
        xdna_api_->device_query_firmware_trace_info(device_, &trace);
    if (amdf_status_code(trace_status) == AMDF_STATUS_CODE_UNSUPPORTED) {
      GTEST_SKIP() << "native firmware recorder is unavailable";
    }
    ASSERT_EQ(trace_status, AMDF_STATUS_OK);
    ASSERT_EQ(trace.enabled, 0u) << "recorder is already owned";
    read_storage_.resize((trace.recommended_buffer_byte_length + 7) / 8);

    amdf_xdna_endpoint_info_t endpoint = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
        .structure_size = sizeof(endpoint),
    };
    ASSERT_EQ(xdna_api_->endpoint_query_info(endpoint_, &endpoint),
              AMDF_STATUS_OK);
    amdf_xdna_device_info_t device = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO,
        .structure_size = sizeof(device),
    };
    ASSERT_EQ(xdna_api_->device_query_info(device_, &device), AMDF_STATUS_OK);
    const auto* images = amdf_cts_xdna_mul_i32_create();
    const std::string_view target = endpoint.target_id;
    size_t image_index = 0;
    if (target == "amd.xdna.strix_halo.17f0_11") {
      image_index = 1;
    } else if (target != "amd.xdna.strix.17f0_10" &&
               target != "amd.xdna.krackan.17f0_20") {
      GTEST_SKIP() << "finite numeric fixture is unavailable for " << target;
    }
    constexpr std::array<amdf_memory_access_t, 3> accesses = {
        AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_READ,
        AMDF_MEMORY_ACCESS_WRITE};
    ASSERT_TRUE(executable_.Initialize(
        {reinterpret_cast<const uint8_t*>(images[image_index].data),
         images[image_index].size},
        endpoint, device, 1, accesses));
    ASSERT_TRUE(FindXdnaKernelQueueFamily(api_, endpoint_, &family_));
    memory_access_.requirements.address_kinds = UINT64_C(1)
                                                << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    const uint32_t ordinal = FindMemoryProfileOrdinal(
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        AMDF_MEMORY_FLAG_HOST_VISIBLE);
    ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    profile_.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile_.structure_size = sizeof(profile_);
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    ASSERT_EQ(QueryMemoryProfile(ordinal, &profile_, &capabilities),
              AMDF_STATUS_OK);
    ASSERT_GT(profile_.allocation.byte_length_granularity, 0u);
  }

  void TearDown() override {
    // Recorder ownership is independent of workload retirement. A failed
    // resource release must not leave the device-global recorder enabled.
    if (owns_trace_) {
      EXPECT_EQ(xdna_api_->device_configure_firmware_trace(device_, 0, 0),
                AMDF_STATUS_OK);
      owns_trace_ = false;
    }
    for (auto& work : work_) {
      if (!work.execution.Release(api_, xdna_api_)) {
        return;
      }
      for (auto& binding : work.bindings) {
        if (!binding.Release(api_)) {
          return;
        }
      }
    }
  }

  void StartTrace() {
    owns_trace_ = true;
    ASSERT_EQ(xdna_api_->device_configure_firmware_trace(
                  device_, 1,
                  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_SCHEDULING |
                      AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_CLOCK_POWER),
              AMDF_STATUS_OK);
    amdf_xdna_firmware_trace_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_FIRMWARE_TRACE_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(xdna_api_->device_query_firmware_trace_info(device_, &info),
              AMDF_STATUS_OK);
    ASSERT_EQ(info.enabled, 1u);
    ASSERT_EQ(info.format_version, AMDF_XDNA_FIRMWARE_TRACE_FORMAT_2_0);
    ASSERT_EQ(info.clock.frequency_hz, UINT64_C(1000000000));
  }

  void QueryIdentity(size_t owner, const char* phase, bool admission = false) {
    amdf_xdna_context_firmware_trace_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_FIRMWARE_TRACE_INFO,
        .structure_size = sizeof(info),
    };
    const auto status = xdna_api_->context_query_firmware_trace_info(
        work_[owner].execution.context, &info);
    if (admission && amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
      GTEST_SKIP() << "native context identity query is unavailable";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    ASSERT_NE(info.firmware_context_id,
              AMDF_XDNA_FIRMWARE_CONTEXT_ID_UNASSIGNED);
    ASSERT_LT(info.firmware_context_id, 16u)
        << "format-2 events carry a four-bit physical context";
    auto& firmware_context = work_[owner].firmware_context;
    if (firmware_context != AMDF_XDNA_FIRMWARE_CONTEXT_ID_UNASSIGNED) {
      ASSERT_EQ(info.firmware_context_id, firmware_context);
    }
    firmware_context = info.firmware_context_id;
    snapshots_ += std::to_string(owner) + "," + phase + "," +
                  std::to_string(firmware_context) + ";";
  }

  void CreateWork(size_t owner) {
    auto& work = work_[owner];
    const auto& geometry = profile_.allocation;
    const uint64_t byte_length = (192 + geometry.byte_length_granularity - 1) /
                                 geometry.byte_length_granularity *
                                 geometry.byte_length_granularity;
    std::array<uint64_t, 3> addresses;
    for (size_t i = 0; i < work.bindings.size(); ++i) {
      amdf_memory_create_info_t create = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
          .structure_size = sizeof(create),
          .memory_profile_ordinal = profile_.ordinal,
          .access_count = 1,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
          .byte_length = byte_length,
          .minimum_alignment = geometry.minimum_alignment,
          .accesses = &memory_access_,
      };
      auto& binding = work.bindings[i];
      ASSERT_NO_FATAL_FAILURE(binding.Create(api_, system_scope_, create));
      ASSERT_EQ(
          api_->memory_query_address(
              binding.memory, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA, &addresses[i]),
          AMDF_STATUS_OK);
      addresses[i] += kPayloadOffset;
      const auto host = binding.HostSite();
      const auto device = binding.DeviceSite(0, family_);
      amdf_memory_pair_info_t pair = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(pair),
      };
      ASSERT_EQ(api_->memory_query_pair_info(&host, &device, &pair),
                AMDF_STATUS_OK);
      ASSERT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
      work.edges[i].release = pair.release.host_operation;
      ASSERT_EQ(api_->memory_query_pair_info(&device, &host, &pair),
                AMDF_STATUS_OK);
      ASSERT_EQ(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
      work.edges[i].acquire = pair.acquire.host_operation;
    }
    std::vector<uint8_t> image(executable_.allocation_byte_length());
    executable_.Load(image);
    ASSERT_TRUE(executable_.Bind(image, addresses));
    ASSERT_NO_FATAL_FAILURE(
        work.execution.Prepare(api_, xdna_api_, device_, family_, 1,
                               executable_.ResolveInvocation(image),
                               executable_.allocation_alignment()));
    const auto commands = work.execution.instructions.bytes();
    work.commands.assign(commands.begin(), commands.end());
    ASSERT_EQ(api_->host_mapping_cache_control(
                  work.execution.instructions.mapping,
                  AMDF_HOST_CACHE_OPERATION_FLUSH, 0, commands.size()),
              AMDF_STATUS_OK);
  }

  void Submit(size_t owner, uint32_t generation) {
    auto& work = work_[owner];
    for (size_t i = 0; i < work.bindings.size(); ++i) {
      const auto bytes = work.bindings[i].bytes();
      std::fill(bytes.begin(), bytes.end(), uint8_t(0xA5 ^ generation ^ owner));
      work.expected[i].assign(bytes.begin(), bytes.end());
    }
    for (size_t i = 0; i < kValueCount; ++i) {
      const uint32_t lhs = uint32_t(17 + 131 * owner + 13 * generation + i);
      const uint32_t rhs = uint32_t(23 + 103 * owner + 19 * generation + 3 * i);
      const uint32_t product = lhs * rhs;
      const std::array<uint32_t, 3> values = {lhs, rhs, product};
      for (size_t b = 0; b < values.size(); ++b) {
        const size_t offset = kPayloadOffset + sizeof(uint32_t) * i;
        const uint32_t initial = b == 2 ? ~product : values[b];
        std::memcpy(work.bindings[b].bytes().data() + offset, &initial, 4);
        std::memcpy(work.expected[b].data() + offset, &values[b], 4);
      }
    }
    for (size_t i = 0; i < work.bindings.size(); ++i) {
      ASSERT_EQ(api_->host_mapping_cache_control(
                    work.bindings[i].mapping, work.edges[i].release, 0,
                    work.bindings[i].host.byte_length),
                AMDF_STATUS_OK);
    }
    amdf_xdna_kernel_queue_submission_info_t submit = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submit),
        .command_count = 1,
        .commands = &work.execution.command,
    };
    work.samples.emplace_back();
    ASSERT_EQ(host_clock_.Read(&work.samples.back().host.begin),
              AMDF_STATUS_OK);
    ASSERT_EQ(xdna_api_->kernel_queue_submit(work.execution.queue, &submit,
                                             &work.point),
              AMDF_STATUS_OK);
  }

  void Retire(size_t owner, uint32_t generation) {
    auto& work = work_[owner];
    ASSERT_EQ(api_->kernel_queue_wait(work.execution.queue, work.point,
                                      amdf::cts::kWaitTimeoutNs,
                                      amdf::cts::kWaitPollIntervalNs),
              AMDF_STATUS_OK);
    ASSERT_EQ(host_clock_.Read(&work.samples.back().host.end), AMDF_STATUS_OK);
    ASSERT_GE(work.samples.back().host.end, work.samples.back().host.begin);
    for (size_t i : {2u, 0u, 1u}) {
      ASSERT_EQ(api_->host_mapping_cache_control(
                    work.bindings[i].mapping, work.edges[i].acquire, 0,
                    work.bindings[i].host.byte_length),
                AMDF_STATUS_OK);
      const auto bytes = work.bindings[i].bytes();
      ASSERT_TRUE(
          std::equal(bytes.begin(), bytes.end(), work.expected[i].begin()))
          << "owner=" << owner << " generation=" << generation
          << " binding=" << i;
    }
  }

  void ReleaseWork(size_t owner) {
    auto& work = work_[owner];
    ASSERT_EQ(
        api_->host_mapping_cache_control(work.execution.instructions.mapping,
                                         AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                                         0, work.commands.size()),
        AMDF_STATUS_OK);
    const auto commands = work.execution.instructions.bytes();
    ASSERT_TRUE(
        std::equal(commands.begin(), commands.end(), work.commands.begin()));
    ASSERT_TRUE(work.execution.Release(api_, xdna_api_));
  }

  void ReadTrace() {
    amdf_observation_read_result_t result = {};
    ASSERT_EQ(
        xdna_api_->device_read_firmware_trace(
            device_, AMDF_XDNA_FIRMWARE_TRACE_FORMAT_2_0, position_,
            read_storage_.data(), uint32_t(read_storage_.size() * 8), &result),
        AMDF_STATUS_OK);
    ASSERT_EQ(result.flags, 0u);
    ASSERT_EQ(result.lost_byte_length, 0u);
    ASSERT_GE(result.next_position, position_);
    position_ = result.next_position;
    ASSERT_LE(result.byte_offset + result.byte_length,
              read_storage_.size() * 8);
    ASSERT_EQ(result.byte_length % sizeof(amdf_xdna_firmware_trace_event_t),
              0u);
    const auto* bytes = reinterpret_cast<const uint8_t*>(read_storage_.data());
    for (uint64_t offset = 0; offset < result.byte_length;
         offset += sizeof(amdf_xdna_firmware_trace_event_t)) {
      amdf_xdna_firmware_trace_event_t event;
      std::memcpy(&event, bytes + result.byte_offset + offset, sizeof(event));
      if (last_sequence_ != 0) {
        ASSERT_EQ(event.sequence,
                  last_sequence_ == 65535 ? 1u : last_sequence_ + 1);
      }
      last_sequence_ = event.sequence;
      events_.push_back(event);
      ASSERT_LE(events_.size(), 4096u);
    }
  }

  void DrainDeletes(size_t expected) {
    const amdf::cts::WaitDeadline deadline;
    size_t observed = 0;
    while (observed < expected) {
      ASSERT_NO_FATAL_FAILURE(ReadTrace());
      observed =
          std::count_if(events_.begin(), events_.end(), [&](const auto& event) {
            if (event.event_id !=
                AMDF_XDNA_FIRMWARE_TRACE_EVENT_CONTEXT_DELETE) {
              return false;
            }
            return std::any_of(
                work_.begin(), work_.end(), [&](const auto& work) {
                  return work.firmware_context == (event.arguments & 15);
                });
          });
      if (observed < expected) {
        ASSERT_TRUE(deadline.Pause())
            << "observed context deletions " << observed;
      }
    }
    ASSERT_EQ(observed, expected);
  }

  void CheckCapture() {
    struct Lifetime {
      // Explicitly constructed caller owner, or no owner after destruction.
      size_t owner = SIZE_MAX;
      // Virtual context decoded from this native creation, not a driver ID.
      uint32_t virtual_context = 0;
      // Beginning of this firmware lifetime in the event clock's nanoseconds.
      uint64_t begin_timestamp = 0;
      // True between one finite command's begin and end records.
      bool pending = false;
    };
    // Format 2 has sixteen physical slots. Multiple lifetimes can reuse one.
    std::array<Lifetime, 16> active;
    std::array<size_t, 3> command_counts = {};
    size_t created_count = 0;
    size_t deleted_count = 0;
    for (const auto& event : events_) {
      const uint32_t physical = uint32_t(event.arguments & 15);
      if (std::none_of(work_.begin(), work_.end(), [&](const auto& work) {
            return work.firmware_context == physical;
          })) {
        continue;
      }
      auto& lifetime = active[physical];
      if (event.event_id == AMDF_XDNA_FIRMWARE_TRACE_EVENT_CONTEXT_CREATE) {
        ASSERT_EQ(lifetime.owner, SIZE_MAX);
        ASSERT_LT(created_count, work_.size());
        // Host creation calls are serialized; the independent native query
        // supplies each owner identity. No global command-order join is used.
        ASSERT_EQ(physical, work_[created_count].firmware_context);
        lifetime.owner = created_count++;
        lifetime.virtual_context = uint32_t((event.arguments >> 28) & 255);
        lifetime.begin_timestamp = event.timestamp;
      } else if (event.event_id ==
                 AMDF_XDNA_FIRMWARE_TRACE_EVENT_CONTEXT_DELETE) {
        ASSERT_NE(lifetime.owner, SIZE_MAX);
        ASSERT_FALSE(lifetime.pending);
        ASSERT_EQ(lifetime.virtual_context,
                  uint32_t((event.arguments >> 36) & 255));
        ASSERT_EQ(uint32_t(event.arguments >> 4), 0u) << "native delete status";
        const auto& work = work_[lifetime.owner];
        ASSERT_EQ(command_counts[lifetime.owner], work.samples.size());
        ASSERT_LE(work.samples.back().firmware.end, event.timestamp);
        lifetime.owner = SIZE_MAX;
        ++deleted_count;
      } else if ((event.event_id ==
                      AMDF_XDNA_FIRMWARE_TRACE_EVENT_COMMAND_BEGIN ||
                  event.event_id ==
                      AMDF_XDNA_FIRMWARE_TRACE_EVENT_COMMAND_END) &&
                 amdf_xdna_firmware_trace_command_opcode(&event) == 23) {
        // Each establishing invocation uses one ordinary EXEC_NPU_CMD. Each
        // queue has one in-flight command, so its local ordinal is independent
        // of the other queue's publication and observed completion order.
        ASSERT_NE(lifetime.owner, SIZE_MAX);
        ASSERT_EQ(lifetime.virtual_context,
                  amdf_xdna_firmware_trace_command_virtual_context(&event));
        auto& work = work_[lifetime.owner];
        const size_t ordinal = command_counts[lifetime.owner];
        ASSERT_LT(ordinal, work.samples.size());
        auto& sample = work.samples[ordinal];
        if (event.event_id == AMDF_XDNA_FIRMWARE_TRACE_EVENT_COMMAND_BEGIN) {
          ASSERT_FALSE(lifetime.pending);
          ASSERT_GE(event.timestamp, lifetime.begin_timestamp);
          if (ordinal != 0) {
            ASSERT_GE(event.timestamp, work.samples[ordinal - 1].firmware.end);
          }
          sample.firmware.begin = event.timestamp;
          lifetime.pending = true;
        } else {
          ASSERT_TRUE(lifetime.pending);
          ASSERT_GT(event.timestamp, sample.firmware.begin);
          sample.firmware.end = event.timestamp;
          EXPECT_LE(
              double(sample.firmware.end - sample.firmware.begin),
              host_clock_.Nanoseconds(sample.host.end - sample.host.begin));
          lifetime.pending = false;
          ++command_counts[lifetime.owner];
        }
      }
    }
    ASSERT_EQ(created_count, work_.size());
    ASSERT_EQ(deleted_count, work_.size());
    for (const auto& lifetime : active) {
      ASSERT_EQ(lifetime.owner, SIZE_MAX);
      ASSERT_FALSE(lifetime.pending);
    }
    ASSERT_EQ(command_counts, (std::array<size_t, 3>{8, 16, 8}));

    // A finite intact capture supplies host bounds for every separately
    // attributed command. Clock drift can make one constant offset fail to
    // fit all records; that rejects the clock model, not native publication.
    const uint64_t host_origin = work_[0].samples.front().host.begin;
    const uint64_t firmware_origin = work_[0].samples.front().firmware.begin;
    double offset_lower = -std::numeric_limits<double>::infinity();
    double offset_upper = std::numeric_limits<double>::infinity();
    std::string intervals;
    size_t overlap_count = 0;
    for (size_t owner = 0; owner < work_.size(); ++owner) {
      for (size_t ordinal = 0; ordinal < work_[owner].samples.size();
           ++ordinal) {
        const auto& sample = work_[owner].samples[ordinal];
        ASSERT_GE(sample.host.begin, host_origin);
        const double begin_delta =
            sample.firmware.begin >= firmware_origin
                ? double(sample.firmware.begin - firmware_origin)
                : -double(firmware_origin - sample.firmware.begin);
        const double end_delta =
            sample.firmware.end >= firmware_origin
                ? double(sample.firmware.end - firmware_origin)
                : -double(firmware_origin - sample.firmware.end);
        offset_lower =
            std::max(offset_lower,
                     host_clock_.Nanoseconds(sample.host.begin - host_origin) -
                         begin_delta);
        offset_upper = std::min(
            offset_upper,
            host_clock_.Nanoseconds(sample.host.end - host_origin) - end_delta);
        intervals += std::to_string(owner) + "," + std::to_string(ordinal) +
                     "," + std::to_string(sample.host.begin) + "," +
                     std::to_string(sample.host.end) + "," +
                     std::to_string(sample.firmware.begin) + "," +
                     std::to_string(sample.firmware.end) + ";";
        for (size_t other = owner + 1; other < work_.size(); ++other) {
          for (const auto& other_sample : work_[other].samples) {
            overlap_count +=
                std::max(sample.firmware.begin, other_sample.firmware.begin) <
                std::min(sample.firmware.end, other_sample.firmware.end);
          }
        }
      }
    }
    RecordProperty("host_frequency_hz",
                   std::to_string(host_clock_.frequency_hz()));
    RecordProperty("native_snapshots", snapshots_);
    RecordProperty("command_intervals", intervals);
    RecordProperty("firmware_record_count", std::to_string(events_.size()));
    RecordProperty("overlapping_command_pairs", std::to_string(overlap_count));
    RecordProperty("recreated_slot_reused",
                   work_[0].firmware_context == work_[2].firmware_context);
    RecordProperty("offset_lower_ns", std::to_string(offset_lower));
    RecordProperty("offset_upper_ns", std::to_string(offset_upper));
    RecordProperty("constant_offset_fits", offset_lower <= offset_upper);
  }

  // Shared finite program; each execution has independently patched bindings.
  XdnaExecutable executable_;
  // Three independently owned lifetimes, with at most two alive together.
  std::array<CapturedWork, 3> work_;
  // Native allocation geometry selected before resources are constructed.
  amdf_memory_profile_t profile_ = {};
  // Native transaction queue family.
  uint32_t family_ = UINT32_MAX;
  // Explicit recorder ownership, released independently of workload cleanup.
  bool owns_trace_ = false;
  // Fixed-capacity native read storage, with whole retained-window capacity.
  std::vector<uint64_t> read_storage_;
  // Next accepted native read position.
  uint64_t position_ = 0;
  // Last published sequence, retained across native read calls.
  uint32_t last_sequence_ = 0;
  // Bounded raw evidence; native storage is copied before reuse.
  std::vector<amdf_xdna_firmware_trace_event_t> events_;
  // Portable host clock; no native client API is needed.
  ObservationHostClock host_clock_;
  // Context identity receipts independent of event arrival order.
  std::string snapshots_;
};

TEST_F(FirmwareTraceRecipeTest, AttributesConcurrentContextLifetimes) {
  ASSERT_NO_FATAL_FAILURE(StartTrace());
  ASSERT_NO_FATAL_FAILURE(CreateWork(0));
  ASSERT_NO_FATAL_FAILURE(QueryIdentity(0, "created", true));
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(CreateWork(1));
  ASSERT_NO_FATAL_FAILURE(QueryIdentity(1, "created"));
  ASSERT_NE(work_[0].firmware_context, work_[1].firmware_context);
  for (uint32_t generation = 0; generation < 8 && !HasFailure(); ++generation) {
    const size_t first = generation & 1;
    ASSERT_NO_FATAL_FAILURE(Submit(first, generation));
    ASSERT_NO_FATAL_FAILURE(Submit(1 - first, generation));
    ASSERT_NO_FATAL_FAILURE(Retire(1 - first, generation));
    ASSERT_NO_FATAL_FAILURE(Retire(first, generation));
    ASSERT_NO_FATAL_FAILURE(ReadTrace());
  }
  ASSERT_NO_FATAL_FAILURE(QueryIdentity(0, "retired"));
  ASSERT_NO_FATAL_FAILURE(QueryIdentity(1, "sibling_retired"));
  ASSERT_NO_FATAL_FAILURE(ReleaseWork(0));
  ASSERT_NO_FATAL_FAILURE(DrainDeletes(1));
  ASSERT_NO_FATAL_FAILURE(CreateWork(2));
  ASSERT_NO_FATAL_FAILURE(QueryIdentity(2, "created"));
  ASSERT_NE(work_[2].firmware_context, work_[1].firmware_context);
  for (uint32_t generation = 8; generation < 16 && !HasFailure();
       ++generation) {
    const size_t first = 1 + (generation & 1);
    ASSERT_NO_FATAL_FAILURE(Submit(first, generation));
    ASSERT_NO_FATAL_FAILURE(Submit(3 - first, generation));
    ASSERT_NO_FATAL_FAILURE(Retire(3 - first, generation));
    ASSERT_NO_FATAL_FAILURE(Retire(first, generation));
    ASSERT_NO_FATAL_FAILURE(ReadTrace());
  }
  ASSERT_NO_FATAL_FAILURE(QueryIdentity(1, "retired"));
  ASSERT_NO_FATAL_FAILURE(QueryIdentity(2, "retired"));
  ASSERT_NO_FATAL_FAILURE(ReleaseWork(2));
  ASSERT_NO_FATAL_FAILURE(ReleaseWork(1));
  ASSERT_NO_FATAL_FAILURE(DrainDeletes(3));
  ASSERT_NO_FATAL_FAILURE(CheckCapture());
}

}  // namespace
