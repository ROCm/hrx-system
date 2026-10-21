// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdlib>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/queue.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::amd::gpu::cts::DeviceAllocator;
using iree::hal::amd::gpu::cts::GpuNativeTest;
using iree::hal::amd::gpu::cts::MatchesNativeIdentity;
using iree::hal::cts::Ref;

// Both protocol owners remain alive through every direction and recording
// mode. By default both protocols use the selected physical endpoint. An
// explicit peer identity admits a second physical endpoint without allowing
// ordinary single-GPU runs to activate another device.
class GpuProtocolInteropTest : public GpuNativeTest {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const amdf_endpoint_id_t* peer_id = &context_->endpoint_info.id;
    const char* selected_peer =
        std::getenv("AMDF_CTS_GPU_PEER_NATIVE_IDENTITY");
    if (selected_peer) {
      ASSERT_NE(std::getenv("AMDF_CTS_GPU_NATIVE_IDENTITY"), nullptr)
          << "Peer selection requires an explicit primary GPU";
      peer_id = nullptr;
      for (uint32_t i = 0; i < fabric_->endpoint_count && !peer_id; ++i) {
        if (fabric_->endpoints[i].engine_kind != AMDF_ENGINE_KIND_GPU) {
          continue;
        }
        bool selected = false;
        IREE_ASSERT_OK(MatchesNativeIdentity(fabric_, &fabric_->endpoints[i].id,
                                             selected_peer, &selected));
        if (selected) {
          peer_id = &fabric_->endpoints[i].id;
        }
      }
      ASSERT_NE(peer_id, nullptr) << "Selected peer GPU was not discovered";
      ASSERT_FALSE(
          amdf_endpoint_id_is_equal(peer_id, &context_->endpoint_info.id));
      RecordProperty("peer_native_identity", selected_peer);
    }
    uint32_t pm4_family = UINT32_MAX;
    IREE_ASSERT_OK(FindPm4Family(peer_id, &pm4_family));
    if (pm4_family == UINT32_MAX) {
      GTEST_SKIP() << "Selected endpoint has no supported PM4 family";
    }
    IREE_ASSERT_OK(iree_hal_amd_gpu_context_allocate(
        fabric_, peer_id, iree_hal_device_event_sink_stderr(),
        iree_allocator_system(), &peer_context_));
    options_.queue_count = 1;
    options_.queue.ring_byte_length = 4096;
    options_.queue.retirement_capacity = 4;
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    auto* context = context_;
    context_ = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_device_create(
        context, IREE_SV("AQL interoperability"), &options_, &params,
        allocator_.allocator(), &device_));
    auto peer_options = options_;
    peer_options.queue_family_ordinal = pm4_family;
    context = peer_context_;
    peer_context_ = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_device_create(
        context, IREE_SV("PM4 interoperability"), &peer_options, &params,
        peer_allocator_.allocator(), &peer_));

    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder, tracker_);
    iree_status_t status =
        iree_hal_device_group_builder_add_device(&builder, device_);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_add_device(&builder, peer_);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_finalize(
          &builder, iree_allocator_system(), &group_);
    }
    iree_hal_device_group_builder_deinitialize(&builder);
    IREE_ASSERT_OK(status);
    protocol_queues_ = {iree_hal_device_queue(device_, 0, 0),
                        iree_hal_device_queue(peer_, 0, 0)};
    for (auto* queue : protocol_queues_) {
      ASSERT_NE(queue, nullptr);
    }
  }

  void TearDown() override {
    iree_hal_device_group_release(group_);
    group_ = nullptr;
    iree_hal_device_release(peer_);
    EXPECT_EQ(peer_allocator_.live_count.load(), 0u);
    iree_hal_amd_gpu_context_free(peer_context_);
    GpuNativeTest::TearDown();
  }

  // Qualifies the requested peer's packet family before native activation.
  iree_status_t FindPm4Family(const amdf_endpoint_id_t* id,
                              uint32_t* out_family) {
    amdf_endpoint_t* endpoint = nullptr;
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        fabric_->api->endpoint_open(fabric_->instance, id, &endpoint),
        "endpoint_open"));
    amdf_endpoint_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
        .structure_size = sizeof(info),
    };
    iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        fabric_->api->endpoint_query_info(endpoint, &info),
        "endpoint_query_info");
    for (uint32_t i = 0; i < info.queue_family_count &&
                         *out_family == UINT32_MAX && iree_status_is_ok(status);
         ++i) {
      amdf_queue_family_info_t family = {
          .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
          .structure_size = sizeof(family),
      };
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          fabric_->api->endpoint_query_queue_family_info(endpoint, i, &family),
          "endpoint_query_queue_family_info");
      if (iree_status_is_ok(status) &&
          iree_hal_amd_gpu_pm4_queue_publication_modes(&family)) {
        *out_family = i;
      }
    }
    return iree_status_join(
        status, IREE_HAL_AMD_STATUS_FROM_AMDF(
                    fabric_->api->endpoint_close(endpoint), "endpoint_close"));
  }

  // Pending native ownership transfers only after complete peer construction.
  iree_hal_amd_gpu_context_t* peer_context_ = nullptr;
  // Allocator outlives all peer resources and their synchronous teardown.
  DeviceAllocator peer_allocator_;
  // Independent PM4 device sharing the fixture's process fabric.
  iree_hal_device_t* peer_ = nullptr;
  // Borrowed AQL and PM4 queues, respectively, from the sealed group.
  std::array<iree_hal_queue_t*, 2> protocol_queues_ = {};
};

TEST_F(GpuProtocolInteropTest, ConsumerFirstHandoffsPreserveSharedBacking) {
  constexpr uint32_t kGuard = 0xDEADC0DEu;
  constexpr size_t kWordCount = 64;
  constexpr size_t kRangeWords = 4;
  constexpr size_t kRangeBytes = kRangeWords * sizeof(uint32_t);
  constexpr std::array<size_t, 3> kOffsets = {16, 80, 144};
  std::array<iree_hal_pool_family_access_t, 2> families = {};
  std::array<Ref<iree_hal_semaphore_t>, 2> timeline_owners;
  std::array<iree_hal_semaphore_t*, 2> timelines = {};
  for (size_t i = 0; i < families.size(); ++i) {
    families[i] = {
        .family = iree_hal_queue_family(protocol_queues_[i]),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        i == 0 ? device_ : peer_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, timeline_owners[i].out()));
    timelines[i] = timeline_owners[i];
  }
  const iree_hal_pool_scope_t scope = {
      .family_count = families.size(),
      .families = families.data(),
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), pool.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool,
      iree_hal_make_pool_reservation_request(kWordCount * sizeof(uint32_t),
                                             sizeof(uint32_t)),
      iree_infinite_timeout(), buffer.out()));

  std::array<Ref<iree_hal_command_buffer_t>, 2> recordings;
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  for (size_t i = 0; i < recordings.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        families[i].family, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, 2, recordings[i].out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recordings[i]));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recordings[i], &acquire));
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
        recordings[i], iree_hal_make_indirect_buffer_ref(0, 0, kRangeBytes),
        iree_hal_make_indirect_buffer_ref(1, 0, kRangeBytes), 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recordings[i], &release));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recordings[i]));
  }

  std::array<uint64_t, 2> epochs = {};
  for (bool recorded : {false, true}) {
    for (size_t first = 0; first < 2; ++first) {
      const size_t second = 1 - first;
      for (uint32_t iteration = 0; iteration < 16; ++iteration) {
        SCOPED_TRACE(::testing::Message()
                     << "recorded=" << recorded << " first=" << first
                     << " iteration=" << iteration);
        const uint32_t pattern = 0xCAFE0000u + iteration + first * 32;
        uint64_t initialized = ++epochs[first];
        uint64_t produced = ++epochs[first];
        uint64_t returned = ++epochs[first];
        uint64_t downloaded = ++epochs[first];
        uint64_t copied = ++epochs[second];
        IREE_ASSERT_OK(iree_hal_queue_fill(
            protocol_queues_[first], iree_hal_semaphore_list_empty(),
            {1, &timelines[first], &initialized}, buffer, 0,
            kWordCount * sizeof(uint32_t), &kGuard, sizeof(kGuard), nullptr,
            0));
        const auto copy = [&](size_t participant, size_t source, size_t target,
                              iree_hal_semaphore_list_t waits,
                              iree_hal_semaphore_list_t signals) {
          if (recorded) {
            const iree_hal_buffer_binding_t bindings[] = {
                {buffer, source, kRangeBytes}, {buffer, target, kRangeBytes}};
            return iree_hal_queue_execute(protocol_queues_[participant], waits,
                                          signals, recordings[participant],
                                          {IREE_ARRAYSIZE(bindings), bindings},
                                          0);
          }
          return iree_hal_queue_copy(protocol_queues_[participant], waits,
                                     signals, buffer, source, buffer, target,
                                     kRangeBytes, nullptr, 0);
        };
        // Enqueue both consumers before their producers. These are distinct
        // native domains, so readiness crosses asynchronous host admission;
        // a blocked consumer must not obstruct its later producer.
        IREE_ASSERT_OK(copy(first, kOffsets[1], kOffsets[2],
                            {1, &timelines[second], &copied},
                            {1, &timelines[first], &returned}));
        IREE_ASSERT_OK(copy(second, kOffsets[0], kOffsets[1],
                            {1, &timelines[first], &produced},
                            {1, &timelines[second], &copied}));
        IREE_ASSERT_OK(iree_hal_queue_fill(
            protocol_queues_[first], {1, &timelines[first], &initialized},
            {1, &timelines[first], &produced}, buffer, kOffsets[0], kRangeBytes,
            &pattern, sizeof(pattern), nullptr, 0));
        std::array<uint32_t, kWordCount> output = {};
        const iree_hal_transfer_operation_t download = {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
            .download =
                {
                    .source_buffer = buffer,
                    .target = output.data(),
                    .length = sizeof(output),
                },
        };
        IREE_ASSERT_OK(iree_hal_queue_transfer(
            protocol_queues_[first], {1, &timelines[first], &returned},
            {1, &timelines[first], &downloaded}, 1, &download, nullptr));
        if (recorded && first == 1 && iteration == 15) {
          for (auto& recording : recordings) {
            recording.reset();
          }
          buffer.reset();
        }
        IREE_ASSERT_OK(iree_hal_semaphore_wait(timelines[first], downloaded,
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE));
        std::array<uint32_t, kWordCount> expected;
        expected.fill(kGuard);
        for (size_t offset : kOffsets) {
          for (size_t i = 0; i < kRangeWords; ++i) {
            expected[offset / sizeof(uint32_t) + i] = pattern;
          }
        }
        EXPECT_EQ(output, expected);
      }
    }
  }
}

}  // namespace
