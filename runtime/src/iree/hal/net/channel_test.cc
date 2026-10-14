// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/net/channel_provider.h"
#include "iree/net/carrier/tcp/factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ChannelTest : public ::testing::Test {
 protected:
  virtual int32_t RankCount() const { return 2; }

  void SetUp() override {
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(1, &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    iree_status_t status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(
        iree_hal_allocator_create_heap(iree_allocator_system(), &allocator_));
    auto pool_options = iree_async_proactor_pool_options_default();
    pool_options.runner = {};
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, pool_options, iree_allocator_system(), &pool_));
    IREE_ASSERT_OK(iree_async_proactor_pool_get(pool_, 0, &proactor_));
    iree_hal_task_device_params_t device_params;
    iree_hal_task_device_params_initialize(&device_params);
    auto create_params = iree_hal_device_create_params_default();
    create_params.proactor_pool = pool_;
    IREE_ASSERT_OK(iree_hal_task_device_create(
        IREE_SV("channel_test"), &device_params, 1, &executor_, 0, nullptr,
        allocator_, &create_params, iree_allocator_system(), &device_));
    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    status = iree_hal_device_group_create_from_device(
        device_, tracker, iree_allocator_system(), &group_);
    iree_async_frontier_tracker_release(tracker);
    IREE_ASSERT_OK(status);
    queue_ = iree_hal_device_queue(device_, 0, 0);
    iree_hal_pool_family_access_t family = {
        .family = iree_hal_queue_family(queue_),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
    };
    iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family,
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_ALL,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED |
                         IREE_HAL_MAPPING_MODE_PERSISTENT,
            },
    };
    const iree_hal_pool_import_class_t import_class = {
        iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
        {}};
    iree_hal_slab_pool_options_t storage_options;
    iree_hal_slab_pool_options_initialize(&storage_options);
    storage_options.imports.class_count = 1;
    storage_options.imports.classes = &import_class;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &storage_options,
                                             iree_allocator_system(),
                                             &payload_pool_));
    auto factory_options = iree_net_tcp_factory_options_default();
    factory_options.connection_options.carrier_options.max_send_operations = 1;
    factory_options.connection_options.receive_mode =
        IREE_NET_TCP_RECEIVE_MODE_PLACEMENT;
    IREE_ASSERT_OK(iree_net_tcp_factory_create(
        &factory_options, iree_allocator_system(), &factory_));
    const int32_t count = RankCount();
    channels_.resize(count);
    closing_.resize(count);
    receipts_.resize(count);
    std::vector<std::array<char, IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH>> storage(
        count);
    std::vector<iree_string_view_t> addresses(count);
    for (int32_t rank = 0; rank < count; ++rank) {
      auto options = iree_hal_net_channel_options_default();
      options.rank = rank;
      options.count = count;
      options.tag_capacity = 4;
      options.peer_addresses = {static_cast<iree_host_size_t>(rank),
                                addresses.data()};
      IREE_ASSERT_OK(CreateChannel(options, &channels_[rank]));
      if (rank + 1 < count) {
        IREE_ASSERT_OK(iree_hal_net_channel_query_address(
            channels_[rank], storage[rank].size(), storage[rank].data(),
            &addresses[rank]));
      }
    }
  }

  void TearDown() override {
    for (size_t rank = 0; rank < channels_.size(); ++rank) {
      if (channels_[rank] && !closing_[rank]) {
        Close(rank);
      }
    }
    PumpUntil([&] {
      for (size_t rank = 0; rank < channels_.size(); ++rank) {
        if (channels_[rank] && !receipts_[rank].retired) {
          return false;
        }
      }
      return true;
    });
    for (auto* transfer : transfers_) {
      iree_hal_channel_transfer_release(transfer);
    }
    for (auto* buffer : buffers_) {
      iree_hal_buffer_release(buffer);
    }
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    for (size_t rank = 0; rank < channels_.size(); ++rank) {
      if (expect_failure_) {
        // These tests deliberately fail accepted exchanges. Individual tests
        // assert their terminal signals; this consumes the channel diagnostic.
        iree_status_free(receipts_[rank].status);
      } else {
        IREE_EXPECT_OK(receipts_[rank].status);
      }
      iree_hal_channel_release(channels_[rank]);
    }
    iree_net_transport_factory_release(factory_);
    iree_hal_pool_release(payload_pool_);
    iree_hal_device_group_release(group_);
    iree_hal_device_release(device_);
    iree_async_proactor_end_polling(proactor_);
    iree_async_proactor_pool_release(pool_);
    iree_hal_allocator_release(allocator_);
    iree_task_executor_release(executor_);
  }

  iree_status_t CreateChannel(iree_hal_net_channel_options_t options,
                              iree_hal_channel_t** out_channel) {
    iree_hal_channel_provider_t* provider = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_net_channel_provider_create(
        factory_, proactor_, &options, iree_allocator_system(), &provider));
    iree_hal_device_replace_channel_provider(device_, provider);
    iree_hal_channel_provider_release(provider);
    iree_hal_channel_params_t params = {
        .group = IREE_SV("prepared-transfers"),
        .rank = options.rank,
        .count = options.count,
    };
    return iree_hal_channel_create(device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                   params, out_channel);
  }

  void PumpUntil(const std::function<bool()>& ready) {
    while (!ready()) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
  }

  void Await(iree_hal_semaphore_list_t points) {
    Receipt receipt;
    iree_async_semaphore_wait_operation_t wait = {};
    wait.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
    wait.base.completion_fn =
        [](void* user_data, iree_async_operation_t*, iree_status_t status,
           iree_async_completion_flags_t) { OnClosed(user_data, status); };
    wait.base.user_data = &receipt;
    wait.semaphores =
        reinterpret_cast<iree_async_semaphore_t**>(points.semaphores);
    wait.values = points.payload_values;
    wait.count = points.count;
    wait.mode = IREE_ASYNC_WAIT_MODE_ALL;
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait.base));
    PumpUntil([&] { return receipt.retired; });
    IREE_ASSERT_OK(receipt.status);
  }

  static void OnClosed(void* user_data, iree_status_t status) {
    auto* receipt = static_cast<Receipt*>(user_data);
    receipt->status = status;
    receipt->retired = true;
  }

  void Close(size_t rank, iree_status_t reason = iree_ok_status()) {
    IREE_ASSERT_OK(iree_hal_channel_deactivate(channels_[rank], reason,
                                               {OnClosed, &receipts_[rank]}));
    closing_[rank] = true;
  }

  iree_hal_semaphore_t* Semaphore(uint64_t value = 0) {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_EXPECT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, value,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
    semaphores_.push_back(semaphore);
    return semaphore;
  }

  static bool Reached(iree_hal_semaphore_t* semaphore, uint64_t value = 1) {
    uint64_t observed = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(semaphore, &observed));
    return observed >= value;
  }

  static iree_hal_semaphore_list_t Point(iree_hal_semaphore_t** semaphore,
                                         uint64_t* value) {
    return {1, semaphore, value};
  }

  iree_hal_buffer_t* Buffer(uint8_t* data, size_t length) {
    iree_hal_buffer_t* buffer = nullptr;
    iree_hal_external_storage_t external = {
        .storage_class = iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
        .payload = {.host_pointer = data},
        .byte_length = length,
        .host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK,
        .maintenance_range = {0, length},
    };
    IREE_EXPECT_OK(iree_hal_pool_import_buffer(payload_pool_, &external,
                                               {0, length}, &buffer));
    buffers_.push_back(buffer);
    return buffer;
  }

  struct PreparedTransfer {
    // Prepared metadata owned by the fixture independently of the payload.
    iree_hal_channel_transfer_t* transfer;
    // Explicit execution buffer owned by the fixture.
    iree_hal_buffer_t* buffer;
  };

  PreparedTransfer* Prepare(size_t rank, iree_hal_channel_transfer_mode_t mode,
                            uint32_t tag, uint8_t* data, size_t length,
                            int32_t peer_rank = -1) {
    iree_hal_buffer_t* buffer = Buffer(data, length);
    const iree_hal_channel_transfer_params_t params = {
        mode, peer_rank < 0 ? static_cast<int32_t>(1 - rank) : peer_rank, tag,
        payload_pool_, length};
    iree_hal_channel_transfer_t* transfer = nullptr;
    IREE_EXPECT_OK(
        iree_hal_channel_prepare_transfer(channels_[rank], &params, &transfer));
    transfers_.push_back(transfer);
    preparations_.push_back({transfer, buffer});
    return &preparations_.back();
  }

  struct Receipt {
    // Owned terminal diagnostic consumed after native retirement.
    iree_status_t status = iree_ok_status();
    // Whether the channel's native retirement callback has arrived.
    bool retired = false;
  };
  // Native CPU executor used by the real HAL device.
  iree_task_executor_t* executor_ = nullptr;
  // Heap allocation provider shared by the device.
  iree_hal_allocator_t* allocator_ = nullptr;
  // Externally polled device/network progress service owner.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Borrowed service driven only by PumpUntil.
  iree_async_proactor_t* proactor_ = nullptr;
  // Real local-task device carrying the installed channel provider.
  iree_hal_device_t* device_ = nullptr;
  // Group assigning queue frontiers for actual queue work.
  iree_hal_device_group_t* group_ = nullptr;
  // Exact queue shared by both channels and independent work.
  iree_hal_queue_t* queue_ = nullptr;
  // TCP factory; no transport or channel implementation is mocked.
  iree_net_transport_factory_t* factory_ = nullptr;
  // Owned rank-local channels with independent matching namespaces.
  std::vector<iree_hal_channel_t*> channels_;
  // Whether explicit deactivation has been requested for each channel.
  std::vector<bool> closing_;
  // Stable callback storage through native retirement.
  std::vector<Receipt> receipts_;
  // Qualified storage owner shared by every preparation and execution.
  iree_hal_pool_t* payload_pool_ = nullptr;
  // Stable caller-side pairs used by fixed-payload protocol tests.
  std::deque<PreparedTransfer> preparations_;
  // Prepared resources released only after their channel drains.
  std::vector<iree_hal_channel_transfer_t*> transfers_;
  // Borrowing wrappers whose backing is held by each test until drain.
  std::vector<iree_hal_buffer_t*> buffers_;
  // Dependency objects retained through accepted queue operations.
  std::vector<iree_hal_semaphore_t*> semaphores_;
  // Deliberate operation-failure cases consume their asserted diagnostics.
  bool expect_failure_ = false;
  // Payload backing survives channel and buffer teardown, including assertions.
  std::array<std::array<uint8_t, 257>, 2> source_ = {};
  // Destination backing survives channel and buffer teardown.
  std::array<std::array<uint8_t, 257>, 2> target_ = {};
  // Independent bounded source slots for the held-reader witness.
  std::array<std::array<uint8_t, 32>, 2> small_source_ = {};
  // Independent bounded destination slots for the held-reader witness.
  std::array<std::array<uint8_t, 32>, 2> small_target_ = {};
};

class PackedChannelTest : public ChannelTest {
 protected:
  // One application slab shared by both roles, retained through native drain.
  std::vector<uint8_t> slab_ = std::vector<uint8_t>(2 * (40976 * 512 + 128));
};

TEST_F(PackedChannelTest, ChangingPackedStageRangesReusePreparedTransfers) {
  constexpr size_t kMaximumLength = 40976 * 512;
  constexpr size_t kCapacity = kMaximumLength + 128;
  auto* buffer = Buffer(slab_.data(), slab_.size());
  iree_hal_channel_transfer_t* plans[2] = {};
  for (size_t rank = 0; rank < 2; ++rank) {
    const iree_hal_channel_transfer_params_t params = {
        rank ? IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE
             : IREE_HAL_CHANNEL_TRANSFER_MODE_SEND,
        static_cast<int32_t>(1 - rank), 0, payload_pool_, kMaximumLength};
    IREE_ASSERT_OK(iree_hal_channel_prepare_transfer(channels_[rank], &params,
                                                     &plans[rank]));
    transfers_.push_back(plans[rank]);
  }
  auto* returned = Semaphore();
  auto* arrived = Semaphore();
  uint64_t sequence = 0;
  for (size_t tokens : {1, 16, 256, 512, 1, 512}) {
    ++sequence;
    const size_t length = 40976 * tokens;
    const size_t source_base = sequence % 2 ? 0 : kCapacity;
    const size_t target_base = sequence % 2 ? kCapacity : 0;
    const size_t offset = 32 + sequence * 3;
    std::fill(slab_.begin(), slab_.end(), 0xCC);
    for (size_t i = 0; i < length; ++i) {
      slab_[source_base + offset + i] = static_cast<uint8_t>(i * 13 + sequence);
    }
    IREE_ASSERT_OK(iree_hal_queue_channel_send(
        queue_, {}, Point(&returned, &sequence), plans[0], sequence, buffer,
        source_base + offset, length));
    IREE_ASSERT_OK(iree_hal_queue_channel_recv(
        queue_, {}, Point(&arrived, &sequence), plans[1], sequence, buffer,
        target_base + offset, length));
    PumpUntil([&] {
      return Reached(returned, sequence) && Reached(arrived, sequence);
    });
    EXPECT_EQ(memcmp(slab_.data() + source_base + offset,
                     slab_.data() + target_base + offset, length),
              0);
    EXPECT_TRUE(std::all_of(slab_.begin() + target_base,
                            slab_.begin() + target_base + offset,
                            [](uint8_t byte) { return byte == 0xCC; }));
    EXPECT_TRUE(std::all_of(slab_.begin() + target_base + offset + length,
                            slab_.begin() + target_base + kCapacity,
                            [](uint8_t byte) { return byte == 0xCC; }));
  }
}

TEST_F(ChannelTest, PreparedTransfersReuseWithoutRendezvous) {
  auto& source = source_[0];
  auto& target = target_[0];
  auto* send = Prepare(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 0, source.data(),
                       source.size());
  auto* receive = Prepare(1, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, 0,
                          target.data(), target.size());
  auto* returned = Semaphore();
  auto* arrived = Semaphore();
  for (uint64_t sequence = 1; sequence <= 12; ++sequence) {
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = i * 17 + sequence;
    }
    IREE_ASSERT_OK(iree_hal_queue_channel_send(
        queue_, {}, Point(&returned, &sequence), send->transfer, sequence,
        send->buffer, 0, source.size()));
    IREE_ASSERT_OK(iree_hal_queue_channel_recv(
        queue_, {}, Point(&arrived, &sequence), receive->transfer, sequence,
        receive->buffer, 0, target.size()));
    PumpUntil([&] {
      return Reached(returned, sequence) && Reached(arrived, sequence);
    });
    EXPECT_EQ(source, target);
  }
  Close(0);
  Close(1);
  PumpUntil([&] { return receipts_[0].retired && receipts_[1].retired; });
}

TEST_F(ChannelTest, HeldReaderGatesOverwriteWhileAnotherTagProgresses) {
  auto& source = small_source_;
  auto& target = small_target_;
  std::array<PreparedTransfer*, 2> sends;
  std::array<PreparedTransfer*, 2> receives;
  std::array<iree_hal_semaphore_t*, 2> returned;
  std::array<iree_hal_semaphore_t*, 2> arrived;
  for (uint32_t tag = 0; tag < 2; ++tag) {
    source[tag].fill(19 + tag);
    sends[tag] = Prepare(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, tag,
                         source[tag].data(), source[tag].size());
    receives[tag] = Prepare(1, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, tag,
                            target[tag].data(), target[tag].size());
    returned[tag] = Semaphore();
    arrived[tag] = Semaphore();
  }
  uint64_t one = 1;
  uint64_t two = 2;
  // First populate a real generation, then admit its final queue consumer
  // behind an independent release gate before posting the next receive.
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(
      queue_, {}, Point(&arrived[0], &one), receives[0]->transfer, one,
      receives[0]->buffer, 0, 32));
  IREE_ASSERT_OK(iree_hal_queue_channel_send(
      queue_, {}, Point(&returned[0], &one), sends[0]->transfer, one,
      sends[0]->buffer, 0, 32));
  PumpUntil([&] { return Reached(returned[0]) && Reached(arrived[0]); });
  const auto previous = target[0];
  auto* reader = Semaphore();
  auto* release = Semaphore();
  iree_hal_semaphore_t* read_waits[] = {arrived[0], release};
  uint64_t read_values[] = {one, one};
  auto* output = Buffer(source_[1].data(), source_[1].size());
  IREE_ASSERT_OK(iree_hal_queue_copy(
      queue_, {2, read_waits, read_values}, Point(&reader, &one),
      receives[0]->buffer, 0, output, 0, 32, nullptr, IREE_HAL_COPY_FLAG_NONE));
  source[0].fill(71);
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(
      queue_, Point(&reader, &one), Point(&arrived[0], &two),
      receives[0]->transfer, two, receives[0]->buffer, 0, 32));
  IREE_ASSERT_OK(iree_hal_queue_channel_send(
      queue_, {}, Point(&returned[0], &two), sends[0]->transfer, two,
      sends[0]->buffer, 0, 32));
  for (uint64_t sequence = 1; sequence <= 5; ++sequence) {
    source[1].fill(20 + sequence);
    IREE_ASSERT_OK(iree_hal_queue_channel_recv(
        queue_, {}, Point(&arrived[1], &sequence), receives[1]->transfer,
        sequence, receives[1]->buffer, 0, 32));
    IREE_ASSERT_OK(iree_hal_queue_channel_send(
        queue_, {}, Point(&returned[1], &sequence), sends[1]->transfer,
        sequence, sends[1]->buffer, 0, 32));
    PumpUntil([&] {
      return Reached(returned[1], sequence) && Reached(arrived[1], sequence);
    });
    EXPECT_EQ(source[1], target[1]);
    EXPECT_FALSE(Reached(arrived[0], two));
    EXPECT_FALSE(Reached(reader));
    EXPECT_EQ(target[0], previous);
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(release, one, nullptr));
  PumpUntil(
      [&] { return Reached(returned[0], two) && Reached(arrived[0], two); });
  EXPECT_EQ(source[0], target[0]);
  EXPECT_EQ(0, memcmp(source_[1].data(), previous.data(), previous.size()));
  Close(0);
  Close(1);
  PumpUntil([&] { return receipts_[0].retired && receipts_[1].retired; });
}

TEST_F(ChannelTest, ReceiveOwnerClosesWithoutWaitingForRemoteSourceReturn) {
  auto* send = Prepare(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 0,
                       source_[0].data(), source_[0].size());
  auto* receive = Prepare(1, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, 0,
                          target_[0].data(), target_[0].size());
  source_[0].fill(91);
  auto* returned = Semaphore();
  auto* arrived = Semaphore();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(
      queue_, {}, Point(&arrived, &one), receive->transfer, one,
      receive->buffer, 0, target_[0].size()));
  IREE_ASSERT_OK(iree_hal_queue_channel_send(queue_, {}, Point(&returned, &one),
                                             send->transfer, one, send->buffer,
                                             0, source_[0].size()));
  PumpUntil([&] { return Reached(arrived); });
  EXPECT_EQ(source_[0], target_[0]);
  Close(1);
  PumpUntil([&] { return receipts_[1].retired && Reached(returned); });
  Close(0);
  PumpUntil([&] { return receipts_[0].retired; });
}

TEST_F(ChannelTest, CancellationJoinsUnsignaledDependencies) {
  expect_failure_ = true;
  auto& target = target_[0];
  auto* receive = Prepare(1, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, 0,
                          target.data(), target.size());
  auto* reader = Semaphore();
  auto* arrived = Semaphore();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(
      queue_, Point(&reader, &one), Point(&arrived, &one), receive->transfer,
      one, receive->buffer, 0, 64));
  // Complete an independent exchange after admission so cancellation joins an
  // installed dependency wait as well as already posted issue callbacks.
  auto* other_send = Prepare(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 1,
                             source_[1].data(), source_[1].size());
  auto* other_receive = Prepare(1, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, 1,
                                target_[1].data(), target_[1].size());
  auto* other_returned = Semaphore();
  auto* other_arrived = Semaphore();
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(
      queue_, {}, Point(&other_arrived, &one), other_receive->transfer, one,
      other_receive->buffer, 0, 32));
  IREE_ASSERT_OK(iree_hal_queue_channel_send(
      queue_, {}, Point(&other_returned, &one), other_send->transfer, one,
      other_send->buffer, 0, 32));
  PumpUntil([&] { return Reached(other_returned) && Reached(other_arrived); });
  Close(1, iree_make_status(IREE_STATUS_ABORTED, "application cancelled"));
  Close(0);
  PumpUntil([&] { return receipts_[0].retired && receipts_[1].retired; });
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_semaphore_query(arrived, &value));
  EXPECT_FALSE(Reached(reader));
  EXPECT_EQ(target, (std::array<uint8_t, 257>{}));
}

TEST_F(ChannelTest, RejectedExtentCapturesNoSignalOrSequence) {
  auto& source = source_[0];
  auto* send =
      Prepare(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 0, source.data(), 16);
  auto* returned = Semaphore();
  uint64_t one = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_channel_send(queue_, {}, Point(&returned, &one),
                                  send->transfer, one, send->buffer, 0, 17));
  EXPECT_FALSE(Reached(returned));
  IREE_ASSERT_OK(iree_hal_queue_channel_send(queue_, {}, Point(&returned, &one),
                                             send->transfer, one, send->buffer,
                                             0, 16));
  expect_failure_ = true;
  Close(0);
  Close(1);
  PumpUntil([&] { return receipts_[0].retired && receipts_[1].retired; });
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_query(returned, &value));
}

TEST_F(ChannelTest, BusyRejectionPreservesDependenciesAndNextSequence) {
  auto* send = Prepare(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 0,
                       source_[0].data(), source_[0].size());
  auto* receive = Prepare(1, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, 0,
                          target_[0].data(), target_[0].size());
  auto* produced = Semaphore();
  auto* returned = Semaphore();
  auto* arrived = Semaphore();
  auto* rejected_wait = Semaphore();
  auto* rejected_returned = Semaphore();
  auto* rejected_arrived = Semaphore();
  uint64_t one = 1;
  uint64_t two = 2;
  source_[0].fill(19);
  IREE_ASSERT_OK(iree_hal_queue_channel_send(
      queue_, Point(&produced, &one), Point(&returned, &one), send->transfer,
      one, send->buffer, 0, source_[0].size()));
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(
      queue_, {}, Point(&arrived, &one), receive->transfer, one,
      receive->buffer, 0, target_[0].size()));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_queue_channel_send(
          queue_, Point(&rejected_wait, &one), Point(&rejected_returned, &two),
          send->transfer, two, send->buffer, 0, source_[0].size()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_queue_channel_recv(
          queue_, Point(&rejected_wait, &one), Point(&rejected_arrived, &two),
          receive->transfer, two, receive->buffer, 0, target_[0].size()));
  // Rejected dependencies cannot fail the accepted exchange or either channel.
  iree_hal_semaphore_fail(rejected_wait, iree_make_status(IREE_STATUS_ABORTED));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(produced, one, nullptr));
  PumpUntil([&] { return Reached(returned) && Reached(arrived); });
  EXPECT_EQ(source_[0], target_[0]);
  EXPECT_FALSE(Reached(rejected_returned));
  EXPECT_FALSE(Reached(rejected_arrived));

  // The same rejected sequence remains admissible after actual completion.
  source_[0].fill(71);
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(
      queue_, {}, Point(&arrived, &two), receive->transfer, two,
      receive->buffer, 0, target_[0].size()));
  IREE_ASSERT_OK(iree_hal_queue_channel_send(queue_, {}, Point(&returned, &two),
                                             send->transfer, two, send->buffer,
                                             0, source_[0].size()));
  PumpUntil([&] { return Reached(returned, two) && Reached(arrived, two); });
  EXPECT_EQ(source_[0], target_[0]);
  Close(0);
  Close(1);
  PumpUntil([&] { return receipts_[0].retired && receipts_[1].retired; });
  EXPECT_FALSE(Reached(rejected_returned));
  EXPECT_FALSE(Reached(rejected_arrived));
}

TEST_F(ChannelTest, SubmissionFromAnotherThreadRetainsDependencies) {
  auto* send = Prepare(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 0,
                       source_[0].data(), source_[0].size());
  auto* receive = Prepare(1, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, 0,
                          target_[0].data(), target_[0].size());
  auto* returned = Semaphore();
  auto* arrived = Semaphore();
  auto* produced = Semaphore();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(
      queue_, {}, Point(&arrived, &one), receive->transfer, one,
      receive->buffer, 0, target_[0].size()));
  struct {
    // Owned result published before finished and consumed after join.
    iree_status_t status = iree_ok_status();
    // Publishes the worker's result to the poll owner.
    std::atomic<bool> finished = false;
  } submission;
  std::thread submitter([&] {
    source_[0].fill(63);
    iree_status_t status = iree_hal_queue_channel_send(
        queue_, Point(&produced, &one), Point(&returned, &one), send->transfer,
        one, send->buffer, 0, source_[0].size());
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_signal(produced, one, nullptr);
    }
    submission.status = status;
    submission.finished.store(true, std::memory_order_release);
    iree_async_proactor_wake(proactor_);
  });
  PumpUntil([&] {
    return submission.finished.load(std::memory_order_acquire) &&
           (!iree_status_is_ok(submission.status) ||
            (Reached(returned) && Reached(arrived)));
  });
  submitter.join();
  IREE_ASSERT_OK(submission.status);
  EXPECT_EQ(source_[0], target_[0]);
  Close(0);
  Close(1);
  PumpUntil([&] { return receipts_[0].retired && receipts_[1].retired; });
}

TEST_F(ChannelTest, InvalidAddressStorageCapturesNoProvider) {
  auto options = iree_hal_net_channel_options_default();
  options.address = {nullptr, 4};
  iree_hal_channel_provider_t* provider = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_net_channel_provider_create(factory_, proactor_, &options,
                                           iree_allocator_system(), &provider));
  EXPECT_EQ(provider, nullptr);
}

TEST_F(ChannelTest, MismatchedExtentFailsBothRanksAndJoinsRetirement) {
  expect_failure_ = true;
  auto* send = Prepare(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 0,
                       source_[0].data(), source_[0].size());
  auto* receive = Prepare(1, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, 0,
                          target_[0].data(), target_[0].size());
  auto* returned = Semaphore();
  auto* arrived = Semaphore();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_channel_send(queue_, {}, Point(&returned, &one),
                                             send->transfer, one, send->buffer,
                                             0, 32));
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(queue_, {}, Point(&arrived, &one),
                                             receive->transfer, one,
                                             receive->buffer, 0, 31));
  PumpUntil([&] {
    uint64_t value = 0;
    iree_status_t status = iree_hal_semaphore_query(returned, &value);
    bool failed = !iree_status_is_ok(status);
    iree_status_free(status);
    return failed;
  });
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_semaphore_query(returned, &value));
  Close(0);
  Close(1);
  PumpUntil([&] { return receipts_[0].retired && receipts_[1].retired; });
  IREE_EXPECT_NOT_OK(iree_hal_semaphore_query(arrived, &value));
  EXPECT_EQ(target_[0], (std::array<uint8_t, 257>{}));
}

class ThreeRankChannelTest : public ChannelTest {
 protected:
  int32_t RankCount() const override { return 3; }

  // Every directed edge has independent application-owned backing.
  std::array<std::array<uint8_t, 257>, 9> sources_ = {};
  // Receiver storage survives the fixture's channel drain.
  std::array<std::array<uint8_t, 257>, 9> targets_ = {};
};

TEST_F(ThreeRankChannelTest, ReusesSameTagForEveryDirectedPeer) {
  std::array<PreparedTransfer*, 9> sends = {};
  std::array<PreparedTransfer*, 9> receives = {};
  std::array<iree_hal_semaphore_t*, 9> returned = {};
  std::array<iree_hal_semaphore_t*, 9> arrived = {};
  for (int32_t rank = 0; rank < 3; ++rank) {
    EXPECT_EQ(iree_hal_channel_rank(channels_[rank]), rank);
    EXPECT_EQ(iree_hal_channel_count(channels_[rank]), 3);
    for (int32_t peer = 0; peer < 3; ++peer) {
      if (rank == peer) {
        continue;
      }
      const size_t edge = rank * 3 + peer;
      sends[edge] = Prepare(rank, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 0,
                            sources_[edge].data(), sources_[edge].size(), peer);
      receives[edge] =
          Prepare(peer, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, 0,
                  targets_[edge].data(), targets_[edge].size(), rank);
      returned[edge] = Semaphore();
      arrived[edge] = Semaphore();
    }
  }
  for (uint64_t sequence = 1; sequence <= 17; ++sequence) {
    for (size_t edge = 0; edge < sends.size(); ++edge) {
      if (!sends[edge]) {
        continue;
      }
      const size_t length = 1 + (sequence * 31 + edge * 17) % 257;
      sources_[edge].fill(static_cast<uint8_t>(sequence * 7 + edge));
      targets_[edge].fill(0xFA);
      IREE_ASSERT_OK(iree_hal_queue_channel_send(
          queue_, {}, Point(&returned[edge], &sequence), sends[edge]->transfer,
          sequence, sends[edge]->buffer, 0, length));
      IREE_ASSERT_OK(iree_hal_queue_channel_recv(
          queue_, {}, Point(&arrived[edge], &sequence),
          receives[edge]->transfer, sequence, receives[edge]->buffer, 0,
          length));
    }
    PumpUntil([&] {
      for (size_t edge = 0; edge < sends.size(); ++edge) {
        if (sends[edge] && (!Reached(returned[edge], sequence) ||
                            !Reached(arrived[edge], sequence))) {
          return false;
        }
      }
      return true;
    });
    for (size_t edge = 0; edge < sends.size(); ++edge) {
      if (!sends[edge]) {
        continue;
      }
      const size_t length = 1 + (sequence * 31 + edge * 17) % 257;
      for (size_t i = 0; i < targets_[edge].size(); ++i) {
        EXPECT_EQ(targets_[edge][i], i < length ? sources_[edge][i] : 0xFA);
      }
    }
  }
}

TEST_F(ThreeRankChannelTest, HeldPeerDoesNotBlockAnotherPeerWithSameTag) {
  std::array<PreparedTransfer*, 2> sends;
  std::array<PreparedTransfer*, 2> receives;
  std::array<iree_hal_semaphore_t*, 2> returned;
  std::array<iree_hal_semaphore_t*, 2> arrived;
  auto* reader_gate = Semaphore();
  uint64_t one = 1;
  for (int32_t peer = 1; peer < 3; ++peer) {
    const size_t edge = peer - 1;
    sources_[edge].fill(static_cast<uint8_t>(peer * 19));
    sends[edge] = Prepare(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 0,
                          sources_[edge].data(), sources_[edge].size(), peer);
    receives[edge] = Prepare(peer, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, 0,
                             targets_[edge].data(), targets_[edge].size(), 0);
    returned[edge] = Semaphore();
    arrived[edge] = Semaphore();
  }
  IREE_ASSERT_OK(iree_hal_queue_channel_send(
      queue_, {}, Point(&returned[0], &one), sends[0]->transfer, one,
      sends[0]->buffer, 0, sources_[0].size()));
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(
      queue_, Point(&reader_gate, &one), Point(&arrived[0], &one),
      receives[0]->transfer, one, receives[0]->buffer, 0, targets_[0].size()));
  for (uint64_t sequence = 1; sequence <= 17; ++sequence) {
    sources_[1].fill(static_cast<uint8_t>(sequence));
    IREE_ASSERT_OK(iree_hal_queue_channel_send(
        queue_, {}, Point(&returned[1], &sequence), sends[1]->transfer,
        sequence, sends[1]->buffer, 0, sources_[1].size()));
    IREE_ASSERT_OK(iree_hal_queue_channel_recv(
        queue_, {}, Point(&arrived[1], &sequence), receives[1]->transfer,
        sequence, receives[1]->buffer, 0, targets_[1].size()));
    PumpUntil([&] {
      return Reached(returned[1], sequence) && Reached(arrived[1], sequence);
    });
    EXPECT_EQ(sources_[1], targets_[1]);
    EXPECT_FALSE(Reached(returned[0]));
    EXPECT_FALSE(Reached(arrived[0]));
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(reader_gate, one, nullptr));
  PumpUntil([&] { return Reached(returned[0]) && Reached(arrived[0]); });
  EXPECT_EQ(sources_[0], targets_[0]);
}

class FourRankChannelTest : public ChannelTest {
 protected:
  int32_t RankCount() const override { return 4; }

  // Callback storage remains alive even if an assertion aborts a test body.
  bool retired_ = false;
};

TEST_F(FourRankChannelTest, DepartureFailsWaitWithoutDirectPeerConnection) {
  expect_failure_ = true;
  auto* send = Prepare(1, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 0,
                       source_[0].data(), source_[0].size(), 3);
  auto* receive = Prepare(3, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE, 0,
                          target_[0].data(), target_[0].size(), 1);
  auto* returned = Semaphore();
  auto* arrived = Semaphore();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_channel_send(queue_, {}, Point(&returned, &one),
                                             send->transfer, one, send->buffer,
                                             0, source_[0].size()));
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(
      queue_, {}, Point(&arrived, &one), receive->transfer, one,
      receive->buffer, 0, target_[0].size()));
  PumpUntil([&] { return Reached(returned) && Reached(arrived); });

  // Rank 3 never prepares its 2->3 edge. Membership 3->1->0->2 must still
  // resolve rank 2's unsignaled producer and release its borrowed buffer.
  const iree_hal_channel_transfer_params_t params = {
      IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 3, 0, payload_pool_, 128};
  iree_hal_channel_transfer_t* pending = nullptr;
  IREE_ASSERT_OK(
      iree_hal_channel_prepare_transfer(channels_[2], &params, &pending));
  transfers_.push_back(pending);
  auto* gate = Semaphore();
  auto* failed = Semaphore();
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(iree_hal_subspan_buffer_create_with_callback(
      Buffer(source_[1].data(), source_[1].size()), 0, 128,
      {[](void* data, iree_hal_buffer_t*) { *static_cast<bool*>(data) = true; },
       &retired_},
      iree_allocator_system(), &view));
  IREE_ASSERT_OK(iree_hal_queue_channel_send(queue_, Point(&gate, &one),
                                             Point(&failed, &one), pending, one,
                                             view, 0, 128));
  iree_hal_buffer_release(view);
  Close(3);
  PumpUntil([&] { return retired_; });
  uint64_t value = 0;
  IREE_EXPECT_NOT_OK(iree_hal_semaphore_query(failed, &value));
}

TEST_F(FourRankChannelTest, ColdDepartureResolvesUnsignaledProducer) {
  expect_failure_ = true;
  auto* send = Prepare(2, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, 0,
                       source_[0].data(), source_[0].size(), 3);
  auto* gate = Semaphore();
  auto* returned = Semaphore();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_channel_send(
      queue_, Point(&gate, &one), Point(&returned, &one), send->transfer, one,
      send->buffer, 0, source_[0].size()));
  Close(3);
  PumpUntil([&] {
    uint64_t value = 0;
    iree_status_t status = iree_hal_semaphore_query(returned, &value);
    const bool failed = !iree_status_is_ok(status);
    iree_status_free(status);
    return failed;
  });
}

TEST_F(ChannelTest, MembershipRejectsMissingAndMalformedContacts) {
  auto options = iree_hal_net_channel_options_default();
  options.rank = 1;
  iree_hal_channel_provider_t* provider = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_net_channel_provider_create(factory_, proactor_, &options,
                                           iree_allocator_system(), &provider));
  EXPECT_EQ(provider, nullptr);
  iree_string_view_t address = {nullptr, 3};
  options.peer_addresses = {1, &address};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_net_channel_provider_create(factory_, proactor_, &options,
                                           iree_allocator_system(), &provider));
  EXPECT_EQ(provider, nullptr);
  address = IREE_SV("127.0.0.1:1");
  options.count = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_net_channel_provider_create(factory_, proactor_, &options,
                                           iree_allocator_system(), &provider));
  EXPECT_EQ(provider, nullptr);
}

class ChannelWindowTest : public ChannelTest {
 protected:
  struct Slot {
    // Stable source backing, filled only after its preceding source return.
    std::array<uint8_t, 257> source = {};
    // Stable target backing, checked before granting its next receive.
    std::array<uint8_t, 257> target = {};
    // Prepared local source; the base fixture owns its public reference.
    PreparedTransfer* send = nullptr;
    // Prepared local target; the base fixture owns its public reference.
    PreparedTransfer* receive = nullptr;
    // Source-return timeline owned by the base fixture.
    iree_hal_semaphore_t* returned = nullptr;
    // Target-arrival timeline owned by the base fixture.
    iree_hal_semaphore_t* arrived = nullptr;
  };

  void PrepareWindow() {
    for (uint32_t rank = 0; rank < slots_.size(); ++rank) {
      for (uint32_t tag = 0; tag < slots_[rank].size(); ++tag) {
        auto& slot = slots_[rank][tag];
        slot.send = Prepare(rank, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND, tag,
                            slot.source.data(), slot.source.size());
        slot.receive = Prepare(rank, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE,
                               tag, slot.target.data(), slot.target.size());
        slot.returned = Semaphore();
        slot.arrived = Semaphore();
      }
    }
  }

  static size_t Length(uint32_t rank, uint32_t tag, uint64_t sequence) {
    return 1 + (sequence * 31 + rank * 73 + tag * 127) % 257;
  }

  iree_status_t Admit(
      uint32_t rank, uint32_t tag, uint64_t sequence,
      iree_hal_semaphore_list_t waits = iree_hal_semaphore_list_empty()) {
    auto& sender = slots_[rank][tag];
    auto& receiver = slots_[1 - rank][tag];
    const size_t length = Length(rank, tag, sequence);
    for (size_t i = 0; i < sender.source.size(); ++i) {
      sender.source[i] = sequence * 13 + rank * 7 + tag * 29 + i * 17;
    }
    receiver.target.fill(0xA5);
    IREE_RETURN_IF_ERROR(iree_hal_queue_channel_recv(
        queue_, {}, Point(&receiver.arrived, &sequence),
        receiver.receive->transfer, sequence, receiver.receive->buffer, 0,
        length));
    return iree_hal_queue_channel_send(
        queue_, waits, Point(&sender.returned, &sequence),
        sender.send->transfer, sequence, sender.send->buffer, 0, length);
  }

  bool Complete(uint32_t rank, uint32_t tag, uint64_t sequence) {
    return Reached(slots_[rank][tag].returned, sequence) &&
           Reached(slots_[1 - rank][tag].arrived, sequence);
  }

  void Check(uint32_t rank, uint32_t tag, uint64_t sequence) {
    const auto& sender = slots_[rank][tag];
    const auto& receiver = slots_[1 - rank][tag];
    const size_t length = Length(rank, tag, sequence);
    for (size_t i = 0; i < receiver.target.size(); ++i) {
      EXPECT_EQ(receiver.target[i], i < length ? sender.source[i] : 0xA5)
          << "rank=" << rank << " tag=" << tag << " sequence=" << sequence
          << " byte=" << i;
    }
  }

  // All borrowed storage survives base TearDown, including assertion failures.
  std::array<std::array<Slot, 4>, 2> slots_;
};

TEST_F(ChannelWindowTest, BidirectionalReuseSharesOneSendBudget) {
  ASSERT_NO_FATAL_FAILURE(PrepareWindow());
  auto* produced = Semaphore();
  uint64_t one = 1;
  IREE_ASSERT_OK(Admit(0, 0, one, Point(&produced, &one)));
  const auto held_source = slots_[0][0].source;
  const auto held_target = slots_[1][0].target;

  // All eight directed edges share one send slot per rank. Hold one producer
  // while both credit and payload traffic for the other seven edges progress.
  for (uint64_t sequence = 1; sequence <= 9; ++sequence) {
    for (uint32_t rank = 0; rank < slots_.size(); ++rank) {
      for (uint32_t tag = 0; tag < slots_[rank].size(); ++tag) {
        if (rank != 0 || tag != 0) {
          IREE_ASSERT_OK(Admit(rank, tag, sequence));
        }
      }
    }
    PumpUntil([&] {
      for (uint32_t rank = 0; rank < slots_.size(); ++rank) {
        for (uint32_t tag = 0; tag < slots_[rank].size(); ++tag) {
          if ((rank != 0 || tag != 0) && !Complete(rank, tag, sequence)) {
            return false;
          }
        }
      }
      return true;
    });
    for (uint32_t rank = 0; rank < slots_.size(); ++rank) {
      for (uint32_t tag = 0; tag < slots_[rank].size(); ++tag) {
        if (rank != 0 || tag != 0) {
          Check(rank, tag, sequence);
        }
      }
    }
    EXPECT_FALSE(Reached(slots_[0][0].returned));
    EXPECT_FALSE(Reached(slots_[1][0].arrived));
    EXPECT_EQ(slots_[0][0].source, held_source);
    EXPECT_EQ(slots_[1][0].target, held_target);
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(produced, one, nullptr));
  PumpUntil([&] { return Complete(0, 0, one); });
  Check(0, 0, one);

  // Rejoin the held edge using an explicit later sequence. All connections,
  // prepared contracts and HAL semaphores are reused.
  for (uint64_t sequence = 10; sequence <= 17; ++sequence) {
    for (uint32_t rank = 0; rank < slots_.size(); ++rank) {
      for (uint32_t tag = 0; tag < slots_[rank].size(); ++tag) {
        IREE_ASSERT_OK(Admit(rank, tag, sequence));
      }
    }
    PumpUntil([&] {
      for (uint32_t rank = 0; rank < slots_.size(); ++rank) {
        for (uint32_t tag = 0; tag < slots_[rank].size(); ++tag) {
          if (!Complete(rank, tag, sequence)) {
            return false;
          }
        }
      }
      return true;
    });
    for (uint32_t rank = 0; rank < slots_.size(); ++rank) {
      for (uint32_t tag = 0; tag < slots_[rank].size(); ++tag) {
        Check(rank, tag, sequence);
      }
    }
  }
}

TEST_F(ChannelWindowTest, PeerFailureJoinsPendingWindow) {
  ASSERT_NO_FATAL_FAILURE(PrepareWindow());
  auto* produced = Semaphore();
  uint64_t one = 1;
  for (uint32_t tag = 0; tag < slots_[0].size(); ++tag) {
    IREE_ASSERT_OK(Admit(0, tag, one, Point(&produced, &one)));
    IREE_ASSERT_OK(Admit(1, tag, one));
  }
  PumpUntil([&] {
    for (uint32_t tag = 0; tag < slots_[0].size(); ++tag) {
      if (!Complete(1, tag, one)) {
        return false;
      }
    }
    return true;
  });
  for (uint32_t tag = 0; tag < slots_[0].size(); ++tag) {
    Check(1, tag, one);
    EXPECT_FALSE(Reached(slots_[0][tag].returned));
    EXPECT_FALSE(Reached(slots_[1][tag].arrived));
  }

  expect_failure_ = true;
  Close(0, iree_make_status(IREE_STATUS_ABORTED, "producer cancelled"));
  // Observe actual peer failure before asking the survivor to deactivate.
  PumpUntil([&] {
    for (const auto& slot : slots_[1]) {
      uint64_t value = 0;
      iree_status_t status = iree_hal_semaphore_query(slot.arrived, &value);
      bool failed = !iree_status_is_ok(status);
      iree_status_free(status);
      if (!failed) {
        return false;
      }
    }
    return true;
  });
  Close(1);
  PumpUntil([&] { return receipts_[0].retired && receipts_[1].retired; });
  for (uint32_t tag = 0; tag < slots_[0].size(); ++tag) {
    uint64_t value = 0;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_hal_semaphore_query(slots_[0][tag].returned, &value));
    IREE_EXPECT_NOT_OK(
        iree_hal_semaphore_query(slots_[1][tag].arrived, &value));
    for (uint8_t byte : slots_[1][tag].target) {
      EXPECT_EQ(byte, 0xA5);
    }
  }
  EXPECT_FALSE(Reached(produced));
}

class ChannelPoolTest : public ChannelTest {
 protected:
  void SetUp() override {
    ChannelTest::SetUp();
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = 4096;
    options.tlsf_options.alignment = 64;
    options.tlsf_options.frontier_capacity = 2;
    options.asan = iree_hal_device_spec_sanitizer(iree_hal_device_spec(device_))
                       ->asan.pool_options;
    iree_hal_slab_cache_options_t cache_options = {.max_count = 2};
    IREE_ASSERT_OK(iree_hal_tlsf_pool_query_backing_request(
        payload_pool_, &options, &cache_options.slab));
    IREE_ASSERT_OK(iree_hal_slab_cache_create(
        payload_pool_, &cache_options, iree_allocator_system(), &cache_));
    IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
        cache_, &options, iree_allocator_system(), &workspace_));
  }

  void TearDown() override {
    // The prepared transfers retain workspace_ until base teardown drains them.
    iree_hal_pool_release(workspace_);
    iree_hal_pool_release(cache_);
    ChannelTest::TearDown();
  }

  iree_hal_channel_transfer_t* Plan(size_t rank,
                                    iree_hal_channel_transfer_mode_t mode,
                                    uint32_t tag = 0) {
    const iree_hal_channel_transfer_params_t params = {
        mode, static_cast<int32_t>(1 - rank), tag, workspace_, 128};
    iree_hal_channel_transfer_t* transfer = nullptr;
    IREE_EXPECT_OK(
        iree_hal_channel_prepare_transfer(channels_[rank], &params, &transfer));
    transfers_.push_back(transfer);
    return transfer;
  }

  void Allocate(iree_hal_semaphore_list_t waits,
                iree_hal_semaphore_list_t signals,
                std::array<iree_hal_buffer_t*, 2>& buffers) {
    const iree_hal_pool_reservation_request_t requests[] = {{256, 64},
                                                            {256, 64}};
    IREE_ASSERT_OK(iree_hal_queue_alloca(queue_, waits, signals, workspace_, 2,
                                         requests, buffers.data()));
  }

  void Release(std::array<iree_hal_buffer_t*, 2>& buffers) {
    auto* released = Semaphore();
    uint64_t one = 1;
    IREE_ASSERT_OK(iree_hal_queue_dealloca(queue_, {}, Point(&released, &one),
                                           buffers.size(), buffers.data()));
    for (auto*& buffer : buffers) {
      iree_hal_buffer_release(buffer);
      buffer = nullptr;
    }
    Await(Point(&released, &one));
    iree_hal_pool_trim(workspace_, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(workspace_, &stats);
    EXPECT_EQ(stats.reservation_count, 0u);
  }

  // Shared physical slab cache; plans never own its reservation ranges.
  iree_hal_pool_t* cache_ = nullptr;
  // Application workspace shared by all execution generations and roles.
  iree_hal_pool_t* workspace_ = nullptr;
};

TEST_F(ChannelPoolTest, QueuedAllocationAndChangingSubrangesReuseOnePlan) {
  auto* send = Plan(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND);
  auto* receive = Plan(1, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE);
  auto* allocated = Semaphore();
  auto* produced = Semaphore();
  auto* initialized = Semaphore();
  auto* returned = Semaphore();
  auto* arrived = Semaphore();
  auto* gate = Semaphore();
  uint64_t one = 1;
  for (uint64_t sequence = 1; sequence <= 8; ++sequence) {
    std::array<iree_hal_buffer_t*, 2> buffers = {};
    Allocate(Point(&gate, &sequence), Point(&allocated, &sequence), buffers);
    const uint8_t pattern = static_cast<uint8_t>(sequence * 17);
    const uint8_t guard = 0xA5;
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queue_, Point(&allocated, &sequence), Point(&produced, &sequence),
        buffers[0], 0, 256, &pattern, 1, nullptr, IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queue_, Point(&allocated, &sequence), Point(&initialized, &sequence),
        buffers[1], 0, 256, &guard, 1, nullptr, IREE_HAL_FILL_FLAG_NONE));
    iree_hal_buffer_t* source = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(buffers[0], 32, 128,
                                           iree_allocator_system(), &source));
    const iree_device_size_t length = sequence * 7;
    const iree_device_size_t target_offset = 32 + sequence;
    IREE_ASSERT_OK(iree_hal_queue_channel_send(
        queue_, Point(&produced, &sequence), Point(&returned, &sequence), send,
        sequence, source, 16, length));
    IREE_ASSERT_OK(iree_hal_queue_channel_recv(
        queue_, Point(&initialized, &sequence), Point(&arrived, &sequence),
        receive, sequence, buffers[1], target_offset, length));
    iree_hal_buffer_release(source);
    // An independent queue operation provides an explicit progress witness
    // while this generation still waits for its allocation dependency.
    auto* independent = Semaphore();
    IREE_ASSERT_OK(iree_hal_queue_barrier(queue_, {}, Point(&independent, &one),
                                          nullptr,
                                          IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    Await(Point(&independent, &one));
    EXPECT_FALSE(Reached(arrived, sequence));
    EXPECT_FALSE(Reached(returned, sequence));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, sequence, nullptr));
    iree_hal_semaphore_t* completed[] = {arrived, returned};
    uint64_t values[] = {sequence, sequence};
    Await({2, completed, values});
    std::array<uint8_t, 256> actual;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(buffers[1], 0, actual.data(), actual.size()));
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(actual[i], i >= target_offset && i < target_offset + length
                               ? pattern
                               : guard);
    }
    Release(buffers);
    // Both plans stay live across idle time with zero workspace reservations.
  }
}

TEST_F(ChannelPoolTest, UnqualifiedRangeRejectionDoesNotConsumeSequence) {
  auto* send = Plan(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND);
  auto* receive = Plan(1, IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE);
  auto* allocated = Semaphore();
  uint64_t one = 1;
  std::array<iree_hal_buffer_t*, 2> buffers = {};
  Allocate({}, Point(&allocated, &one), buffers);
  Await(Point(&allocated, &one));
  auto* returned = Semaphore();
  auto* arrived = Semaphore();
  std::array<uint8_t, 32> external = {};
  iree_hal_buffer_t* unprepared = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      external.size(), iree_make_byte_span(external.data(), external.size()),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &unprepared));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_channel_send(queue_, {}, Point(&returned, &one), send, one,
                                  unprepared, 0, external.size()));
  iree_hal_buffer_release(unprepared);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_channel_send(queue_, {}, Point(&returned, &one), send, one,
                                  buffers[0], 0, 129));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_channel_send(queue_, {}, Point(&returned, &one), send, one,
                                  buffers[0], 250, 32));
  EXPECT_FALSE(Reached(returned));
  const uint8_t pattern = 0x5A;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffers[0], 0, 32, &pattern, 1));
  IREE_ASSERT_OK(iree_hal_queue_channel_recv(queue_, {}, Point(&arrived, &one),
                                             receive, one, buffers[1], 0, 32));
  IREE_ASSERT_OK(iree_hal_queue_channel_send(queue_, {}, Point(&returned, &one),
                                             send, one, buffers[0], 0, 32));
  PumpUntil([&] { return Reached(returned) && Reached(arrived); });
  std::array<uint8_t, 32> actual;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(buffers[1], 0, actual.data(), actual.size()));
  for (auto value : actual) {
    EXPECT_EQ(value, pattern);
  }
  Release(buffers);
}

TEST_F(ChannelPoolTest, PeerCloseRetiresBorrowBeforeWorkspaceReuse) {
  auto* send = Plan(0, IREE_HAL_CHANNEL_TRANSFER_MODE_SEND);
  auto* allocated = Semaphore();
  auto* gate = Semaphore();
  auto* returned = Semaphore();
  uint64_t one = 1;
  std::array<iree_hal_buffer_t*, 2> buffers = {};
  Allocate({}, Point(&allocated, &one), buffers);
  Await(Point(&allocated, &one));
  bool retired = false;
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(iree_hal_subspan_buffer_create_with_callback(
      buffers[0], 32, 128,
      {[](void* user_data, iree_hal_buffer_t*) {
         *static_cast<bool*>(user_data) = true;
       },
       &retired},
      iree_allocator_system(), &view));
  IREE_ASSERT_OK(iree_hal_queue_channel_send(queue_, Point(&gate, &one),
                                             Point(&returned, &one), send, one,
                                             view, 0, 128));
  iree_hal_buffer_release(view);
  EXPECT_FALSE(retired);
  expect_failure_ = true;
  Close(1);
  PumpUntil([&] { return retired; });
  uint64_t value = 0;
  IREE_EXPECT_NOT_OK(iree_hal_semaphore_query(returned, &value));
  Close(0);
  PumpUntil([&] { return receipts_[0].retired && receipts_[1].retired; });
  Release(buffers);
}

}  // namespace
