// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/vulkan/queue.h"

namespace iree::hal::cts {
namespace {

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

// Observes the real queue without changing its commands or synchronization.
class TimestampQueryCapture {
 public:
  struct QueryRange {
    // Native recording that last used this query range.
    VkCommandBuffer commands = VK_NULL_HANDLE;
    // Recording generation, changed whenever the native handle is begun.
    uint64_t generation = 0;
    // Query storage borrowed until the next submission or explicit trim.
    VkQueryPool pool = VK_NULL_HANDLE;
    // First query copied by the observed native command buffer.
    uint32_t first = 0;
    // Number of consecutive queries copied into private resolve storage.
    uint32_t count = 0;
  };

  explicit TimestampQueryCapture(iree_hal_vulkan_queue_t* queue)
      : queue_(queue), original_(queue->syms) {
    active_ = this;
    queue_->syms.vkBeginCommandBuffer = Begin;
    queue_->syms.vkCmdCopyQueryPoolResults = Copy;
    queue_->syms.vkQueueSubmit2 = Submit;
  }

  ~TimestampQueryCapture() {
    // All HAL work has completed. Also join any native submit call returning
    // after its very short device workload completed.
    iree_slim_mutex_lock(&queue_->queue_handle_mutex);
    queue_->syms.vkBeginCommandBuffer = original_.vkBeginCommandBuffer;
    queue_->syms.vkCmdCopyQueryPoolResults =
        original_.vkCmdCopyQueryPoolResults;
    queue_->syms.vkQueueSubmit2 = original_.vkQueueSubmit2;
    active_ = nullptr;
    iree_slim_mutex_unlock(&queue_->queue_handle_mutex);
  }

  QueryRange submitted() {
    std::lock_guard<std::mutex> lock(mutex_);
    // A closing clock operation may have reused this uncached native handle.
    // Only observe query storage that still belongs to the captured recording.
    return Find(submitted_.commands).generation == submitted_.generation
               ? submitted_
               : QueryRange{};
  }

 private:
  struct Recording {
    // Native handle whose recording owns the captured query-copy operation.
    VkCommandBuffer commands = VK_NULL_HANDLE;
    // Generation incremented on each recording into this handle.
    uint64_t generation = 0;
    // Range copied by this recording, empty for non-timestamp commands.
    QueryRange queries;
  };

  Recording& Find(VkCommandBuffer commands) {
    auto it = std::find_if(recordings_.begin(), recordings_.end(),
                           [commands](const Recording& recording) {
                             return recording.commands == commands;
                           });
    if (it != recordings_.end()) {
      return *it;
    }
    recordings_.push_back({commands, 0, {}});
    return recordings_.back();
  }

  static VKAPI_ATTR VkResult VKAPI_CALL
  Begin(VkCommandBuffer commands, const VkCommandBufferBeginInfo* info) {
    auto* capture = active_;
    {
      std::lock_guard<std::mutex> lock(capture->mutex_);
      auto& recording = capture->Find(commands);
      ++recording.generation;
      recording.queries = {};
    }
    return capture->original_.vkBeginCommandBuffer(commands, info);
  }

  static VKAPI_ATTR void VKAPI_CALL Copy(VkCommandBuffer commands,
                                         VkQueryPool pool, uint32_t first,
                                         uint32_t count, VkBuffer target,
                                         VkDeviceSize offset,
                                         VkDeviceSize stride,
                                         VkQueryResultFlags flags) {
    auto* capture = active_;
    {
      std::lock_guard<std::mutex> lock(capture->mutex_);
      auto& recording = capture->Find(commands);
      recording.queries = {commands, recording.generation, pool, first, count};
    }
    capture->original_.vkCmdCopyQueryPoolResults(commands, pool, first, count,
                                                 target, offset, stride, flags);
  }

  static VKAPI_ATTR VkResult VKAPI_CALL Submit(VkQueue queue, uint32_t count,
                                               const VkSubmitInfo2* submissions,
                                               VkFence fence) {
    auto* capture = active_;
    {
      std::lock_guard<std::mutex> lock(capture->mutex_);
      for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t j = 0; j < submissions[i].commandBufferInfoCount; ++j) {
          const auto commands =
              submissions[i].pCommandBufferInfos[j].commandBuffer;
          const auto& queries = capture->Find(commands).queries;
          // Queue clock bounds copy one query; the recording under test
          // contains at least two and is the only multi-query submission.
          if (queries.count > 1) {
            capture->submitted_ = queries;
          }
        }
      }
    }
    return capture->original_.vkQueueSubmit2(queue, count, submissions, fence);
  }

  // The one active fixture owns this queue's native dispatch overrides.
  static TimestampQueryCapture* active_;
  // Borrowed provisioned queue, retained by the CTS device cache.
  iree_hal_vulkan_queue_t* queue_;
  // Original native entry points, restored before fixture destruction.
  iree_hal_vulkan_device_syms_t original_;
  // Protects observation metadata across recording and submission threads.
  std::mutex mutex_;
  // One row per observed native handle, replaced whenever it is rerecorded.
  std::vector<Recording> recordings_;
  // Query range belonging to the last multi-query command actually submitted.
  QueryRange submitted_;
};

TimestampQueryCapture* TimestampQueryCapture::active_ = nullptr;

class TimestampReplayTest : public CtsTestBase<> {
 public:
  void CheckReboundOutput(bool cache_recording);
};

void TimestampReplayTest::CheckReboundOutput(bool cache_recording) {
  auto* queue = reinterpret_cast<iree_hal_vulkan_queue_t*>(
      QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_TRANSFER));
  ASSERT_NE(queue, nullptr);
  ASSERT_GT(queue->timestamp_valid_bits, 0u);
  const uint64_t tick_mask = UINT64_MAX >> (64 - queue->timestamp_valid_bits);
  constexpr uint64_t kGuard = UINT64_C(0xCA7ECA7ECA7ECA7E);
  constexpr uint32_t kReplayCount = 512;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateDevicePool(IREE_HAL_BUFFER_USAGE_TRANSFER, pool.out()));
  Ref<iree_hal_buffer_t> clock_buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(16, 8),
      iree_infinite_timeout(), clock_buffer.out()));
  TimestampQueryCapture capture(queue);

  for (const uint32_t count : {2u, 64u, 65u}) {
    SCOPED_TRACE(count);
    IREE_ASSERT_OK(iree_hal_device_trim(device_));
    iree_hal_vulkan_native_replay_cache_stats_t initial_stats;
    iree_hal_vulkan_queue_sample_native_replay_cache_stats(queue,
                                                           &initial_stats);
    ASSERT_EQ(initial_stats.instance_count, 0u);
    ASSERT_GT(initial_stats.max_instance_count, 0u);
    const uint32_t range_length = (count + 1) * 16;
    Ref<iree_hal_buffer_t> output;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(3 * range_length, 8),
        iree_infinite_timeout(), output.out()));
    std::vector<uint64_t> expected(3 * range_length / 8, kGuard);
    IREE_ASSERT_OK(UploadBufferData(expected.data(), output, 0,
                                    expected.size() * sizeof(uint64_t)));
    Ref<iree_hal_command_buffer_t> commands;
    uint32_t native_comparisons = 0;
    std::vector<uint64_t> native(count * 2);
    for (uint32_t iteration = 0; iteration < kReplayCount; ++iteration) {
      SCOPED_TRACE(iteration);
      if (!commands) {
        IREE_ASSERT_OK(iree_hal_command_buffer_create(
            iree_hal_queue_family(&queue->base),
            cache_recording ? 0 : IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
            IREE_HAL_COMMAND_CATEGORY_TRANSFER, 1, commands.out()));
        IREE_ASSERT_OK(iree_hal_command_buffer_begin(commands));
        for (uint32_t i = 0; i < count; ++i) {
          IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(
              commands, 0, 0,
              iree_hal_make_indirect_buffer_ref(0, 8 + i * 16, 8), 0));
        }
        IREE_ASSERT_OK(iree_hal_command_buffer_end(commands));
      }
      const uint32_t output_offset = (iteration % 3) * range_length;
      const iree_hal_buffer_binding_t binding = {output, output_offset,
                                                 range_length};
      SemaphoreList before(device_, {0}, {1});
      SemaphoreList executed(device_, {0}, {1});
      SemaphoreList after(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_timestamp(&queue->base, {}, before,
                                              clock_buffer, 0, nullptr, 0));
      IREE_ASSERT_OK(iree_hal_queue_execute(&queue->base, before, executed,
                                            commands, {1, &binding}, 0));
      IREE_ASSERT_OK(iree_hal_queue_timestamp(&queue->base, executed, after,
                                              clock_buffer, 8, nullptr, 0));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          after, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      const auto queries = capture.submitted();
      if (queries.pool != VK_NULL_HANDLE) {
        ASSERT_EQ(queries.count, count);
        std::fill(native.begin(), native.end(), kGuard);
        IREE_ASSERT_OK(iree_vkGetQueryPoolResults(
            IREE_VULKAN_DEVICE(&queue->syms), queue->logical_device,
            queries.pool, queries.first, queries.count,
            native.size() * sizeof(uint64_t), native.data(),
            2 * sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
        ++native_comparisons;
      }
      const auto bounds = ReadBufferData<uint64_t>(clock_buffer);
      ASSERT_EQ(bounds.size(), 2u);
      const auto actual = ReadBufferData<uint64_t>(output);
      ASSERT_EQ(actual.size(), expected.size());
      for (uint32_t i = 0; i < count; ++i) {
        SCOPED_TRACE(i);
        const auto value = actual[output_offset / 8 + i * 2 + 1];
        ASSERT_LE((value - bounds[0]) & tick_mask,
                  (bounds[1] - bounds[0]) & tick_mask)
            << "value=" << value << " bounds=[" << bounds[0] << "," << bounds[1]
            << "] native="
            << (queries.pool != VK_NULL_HANDLE ? native[i * 2] : kGuard);
        if (queries.pool != VK_NULL_HANDLE) {
          ASSERT_NE(native[i * 2 + 1], 0u);
          ASSERT_EQ(value, native[i * 2]);
        }
        expected[output_offset / 8 + i * 2 + 1] = value;
      }
      for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_EQ(actual[i], expected[i])
            << "word=" << i << " query_pool=" << queries.pool << " bounds=["
            << bounds[0] << "," << bounds[1] << "]";
      }
      if (!cache_recording) {
        commands.reset();
      }
    }
    iree_hal_vulkan_native_replay_cache_stats_t final_stats;
    iree_hal_vulkan_queue_sample_native_replay_cache_stats(queue, &final_stats);
    std::printf(
        "count=%u executions=%u native_comparisons=%u cache_hits=%llu "
        "publications=%llu one_shot_bypasses=%llu\n",
        count, kReplayCount, native_comparisons,
        static_cast<unsigned long long>(final_stats.hit_count -
                                        initial_stats.hit_count),
        static_cast<unsigned long long>(final_stats.publication_update_count -
                                        initial_stats.publication_update_count),
        static_cast<unsigned long long>(final_stats.one_shot_bypass_count -
                                        initial_stats.one_shot_bypass_count));
    ASSERT_GT(native_comparisons, 0u);
    if (cache_recording) {
      ASSERT_GT(final_stats.hit_count, initial_stats.hit_count);
    } else {
      ASSERT_EQ(final_stats.one_shot_bypass_count -
                    initial_stats.one_shot_bypass_count,
                kReplayCount);
    }
    commands.reset();
    IREE_ASSERT_OK(iree_hal_device_trim(device_));
  }
}

TEST_P(TimestampReplayTest, NativeQueryMatchesPipelinedCachedOutput) {
  CheckReboundOutput(/*cache_recording=*/true);
}

TEST_P(TimestampReplayTest, NativeQueryMatchesPipelinedFreshOutput) {
  CheckReboundOutput(/*cache_recording=*/false);
}

CTS_REGISTER_TEST_SUITE(TimestampReplayTest);

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

}  // namespace
}  // namespace iree::hal::cts
