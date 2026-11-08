// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/driver/vulkan/registration/driver_module.h"

namespace {

enum class Submission { kCopy, kExecute, kRecorded };

// A four-byte value travels through distinct slots. Resetting every slot and
// changing the input between iterations makes missing dependency edges visible.
// Host mappings and verification are outside both timing modes.
class TransferChain {
 public:
  ~TransferChain() {
    iree_status_ignore(Wait());
    iree_hal_command_buffer_release(command_buffer_);
    for (auto* command_buffer : copies_) {
      iree_hal_command_buffer_release(command_buffer);
    }
    iree_hal_buffer_release(buffer_);
    iree_hal_semaphore_release(timeline_);
    iree_hal_pool_release(memory_pool_);
    iree_hal_device_group_release(group_);
    iree_hal_device_release(device_);
    iree_hal_driver_release(driver_);
    iree_async_proactor_pool_release(proactors_);
  }

  iree_status_t Initialize(iree_host_size_t depth) {
    values_.resize(depth + 1);
    iree_status_t status = iree_hal_vulkan_driver_module_register(
        iree_hal_driver_registry_default());
    if (iree_status_is_already_exists(status)) {
      iree_status_free(status);
      status = iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(iree_hal_driver_registry_acquire_driver(
        iree_hal_driver_registry_default(), IREE_SV("vulkan"), &driver_));
    IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactors_));
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    params.event_sink = iree_hal_device_event_sink_stderr();
    IREE_RETURN_IF_ERROR(iree_hal_driver_create_default_device(
        driver_, &params, iree_allocator_system(), &device_));
    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    status = iree_hal_device_group_create_from_device(
        device_, tracker, iree_allocator_system(), &group_);
    iree_async_frontier_tracker_release(tracker);
    IREE_RETURN_IF_ERROR(status);

    const auto* queues =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
      if (queues->families[i].provisioned_queue_count &&
          iree_all_bits_set(queues->families[i].role_flags,
                            IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                                IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
        queue_ = iree_hal_device_queue(device_, i, 0);
        break;
      }
    }
    if (!queue_) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no Vulkan compute/transfer queue");
    }
    const iree_hal_pool_family_access_t family = {
        .family = iree_hal_queue_family(queue_),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family,
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_ALL,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
            },
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_RETURN_IF_ERROR(iree_hal_slab_pool_create(
        group_, scope, &options, iree_allocator_system(), &memory_pool_));
    IREE_RETURN_IF_ERROR(iree_hal_pool_allocate_buffer(
        memory_pool_,
        iree_hal_make_pool_reservation_request(
            values_.size() * sizeof(uint32_t), sizeof(uint32_t)),
        iree_infinite_timeout(), &buffer_));
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &timeline_));

    copies_.resize(depth);
    for (iree_host_size_t i = 0; i < depth && iree_status_is_ok(status); ++i) {
      status = Record(i, 1, &copies_[i]);
    }
    IREE_RETURN_IF_ERROR(status);
    return Record(0, depth, &command_buffer_);
  }

  iree_status_t Record(iree_host_size_t first, iree_host_size_t count,
                       iree_hal_command_buffer_t** out_command_buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue_), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0, out_command_buffer));
    auto* command_buffer = *out_command_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
    const iree_hal_memory_barrier_t memory_barrier = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
    };
    const iree_hal_barrier_t barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .memory_barrier_count = 1,
        .memory_barriers = &memory_barrier,
    };
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = first;
         i < first + count && iree_status_is_ok(status); ++i) {
      if (i != first) {
        status = iree_hal_command_buffer_barrier(command_buffer, &barrier);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_command_buffer_copy_buffer(
            command_buffer,
            iree_hal_make_buffer_ref(buffer_, i * sizeof(uint32_t),
                                     sizeof(uint32_t)),
            iree_hal_make_buffer_ref(buffer_, (i + 1) * sizeof(uint32_t),
                                     sizeof(uint32_t)),
            IREE_HAL_COPY_FLAG_NONE);
      }
    }
    IREE_RETURN_IF_ERROR(status);
    return iree_hal_command_buffer_end(command_buffer);
  }

  iree_status_t Reset() {
    for (auto& value : values_) {
      value = 0;
    }
    values_[0] = ++input_;
    return iree_hal_buffer_map_write(buffer_, 0, values_.data(),
                                     values_.size() * sizeof(uint32_t));
  }

  iree_status_t Submit(Submission submission) {
    const bool recorded = submission == Submission::kRecorded;
    const iree_host_size_t count = recorded ? 1 : values_.size() - 1;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      uint64_t next_value = submitted_value_ + 1;
      const iree_hal_semaphore_list_t waits = {1, &timeline_,
                                               &submitted_value_};
      const iree_hal_semaphore_list_t signals = {1, &timeline_, &next_value};
      if (submission != Submission::kCopy) {
        status = iree_hal_queue_execute(queue_, waits, signals,
                                        recorded ? command_buffer_ : copies_[i],
                                        iree_hal_buffer_binding_table_empty(),
                                        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
      } else {
        status = iree_hal_queue_copy(
            queue_, waits, signals, buffer_, i * sizeof(uint32_t), buffer_,
            (i + 1) * sizeof(uint32_t), sizeof(uint32_t), nullptr,
            IREE_HAL_COPY_FLAG_NONE);
      }
      if (iree_status_is_ok(status)) {
        submitted_value_ = next_value;
      }
    }
    return status;
  }

  iree_status_t Wait() {
    if (!submitted_value_) {
      return iree_ok_status();
    }
    return iree_hal_semaphore_wait(timeline_, submitted_value_,
                                   iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_status_t Verify() {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
        buffer_, 0, values_.data(), values_.size() * sizeof(uint32_t)));
    for (iree_host_size_t i = 0; i < values_.size(); ++i) {
      if (values_[i] != input_) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "slot %zu: expected %u, got %u", i, input_,
                                values_[i]);
      }
    }
    return iree_ok_status();
  }

 private:
  // Device creation and asynchronous host notification owners.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // Native Vulkan driver retained through device destruction.
  iree_hal_driver_t* driver_ = nullptr;
  // Device owning the queue and its native resources.
  iree_hal_device_t* device_ = nullptr;
  // Sealed topology and completion frontier owner.
  iree_hal_device_group_t* group_ = nullptr;
  // Borrowed compute/transfer queue, alive with device_.
  iree_hal_queue_t* queue_ = nullptr;
  // Pool outliving every buffer and terminal operation.
  iree_hal_pool_t* memory_pool_ = nullptr;
  // One resident buffer holding the input and every dependency result.
  iree_hal_buffer_t* buffer_ = nullptr;
  // Reusable single-submission control containing the same copies.
  iree_hal_command_buffer_t* command_buffer_ = nullptr;
  // One reusable command buffer per edge, separating execution from recording.
  std::vector<iree_hal_command_buffer_t*> copies_;
  // Public completion timeline shared by the chain's submissions.
  iree_hal_semaphore_t* timeline_ = nullptr;
  // Last accepted submission, also joined on a partial capture failure.
  uint64_t submitted_value_ = 0;
  // Changes every iteration to expose skipped work and stale reads.
  uint32_t input_ = 0;
  // Host input/reset and result storage, reused outside the timed region.
  std::vector<uint32_t> values_;
};

template <Submission kSubmission, bool kSubmitOnly>
void BM_TransferChain(benchmark::State& state) {
  TransferChain chain;
  iree::Status status = chain.Initialize(state.range(0));
  // Populate native replay caches and activate the queue before measuring the
  // resident dependency chain. The first use includes device setup work.
  if (status.ok()) {
    status = chain.Reset();
  }
  if (status.ok()) {
    status = chain.Submit(kSubmission);
  }
  status = iree_status_join(status.release(), chain.Wait());
  if (status.ok()) {
    status = chain.Verify();
  }
  if (!status.ok()) {
    state.SkipWithError(status.ToString().c_str());
    return;
  }
  for (auto _ : state) {
    state.PauseTiming();
    status = chain.Reset();
    state.ResumeTiming();
    if (status.ok()) {
      status = chain.Submit(kSubmission);
    }
    if (kSubmitOnly) {
      state.PauseTiming();
    }
    status = iree_status_join(status.release(), chain.Wait());
    if (!kSubmitOnly) {
      state.PauseTiming();
    }
    if (status.ok()) {
      status = chain.Verify();
    }
    state.ResumeTiming();
    if (!status.ok()) {
      state.SkipWithError(status.ToString().c_str());
      break;
    }
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kCopy, false)
    ->Name("VulkanTransferChain/CopyCompletion")
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kExecute, false)
    ->Name("VulkanTransferChain/ExecuteCompletion")
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kExecute, true)
    ->Name("VulkanTransferChain/ExecuteSubmit")
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kRecorded, false)
    ->Name("VulkanTransferChain/RecordedCompletion")
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->UseRealTime();

}  // namespace
