// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <atomic>
#include <barrier>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/tooling/device_util.h"

IREE_FLAG(int64_t, queue_family, -1,
          "Transfer queue family ordinal; -1 selects the first "
          "available family.");

namespace {

enum class Submission { kCopy, kExecute, kRecorded, kMixed, kCrossQueue };

// All callers share the production device, queue inventory, and pool. Each
// chain owns its own data and timeline; host enqueue order supplies no edges.
class QueueDevice {
 public:
  ~QueueDevice() {
    iree_hal_pool_release(memory_pool_);
    for (auto* queue : queues_) {
      iree_hal_queue_release(queue);
    }
    iree_hal_device_group_release(group_);
    iree_hal_device_release(device_);
    iree_async_proactor_pool_release(proactors_);
  }

  iree_status_t Initialize(size_t requested_queue_count) {
    const iree_allocator_t allocator = {this, Allocate};
    IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(), allocator,
        &proactors_));
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    params.event_sink = iree_hal_device_event_sink_stderr();
    IREE_RETURN_IF_ERROR(iree_hal_create_device_from_flags(
        iree_hal_available_driver_registry(), IREE_SV("task"), &params,
        allocator, &device_));
    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator, &tracker));
    iree_status_t status = iree_hal_device_group_create_from_device(
        device_, tracker, allocator, &group_);
    iree_async_frontier_tracker_release(tracker);
    IREE_RETURN_IF_ERROR(status);

    const auto* spec =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    if (FLAG_queue_family < -1 ||
        (FLAG_queue_family >= 0 &&
         (uint64_t)FLAG_queue_family >= spec->family_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "queue_family must be -1 or less than %" PRIhsz,
                              spec->family_count);
    }
    std::vector<iree_hal_pool_family_access_t> families;
    for (iree_host_size_t i = 0;
         i < spec->family_count && iree_status_is_ok(status); ++i) {
      const auto& family = spec->families[i];
      const bool can_acquire = iree_any_bit_set(
          family.flags, IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION);
      if ((FLAG_queue_family < 0 || (uint64_t)FLAG_queue_family == i) &&
          (family.provisioned_queue_count || can_acquire) &&
          iree_all_bits_set(family.role_flags,
                            IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
        const auto* queue_family = iree_hal_device_queue_family(device_, i);
        families.push_back({queue_family, IREE_HAL_BUFFER_USAGE_TRANSFER});
        for (uint32_t j = 0; j < family.provisioned_queue_count &&
                             queues_.size() < requested_queue_count;
             ++j) {
          auto* queue = iree_hal_device_queue(device_, i, j);
          iree_hal_queue_retain(queue);
          queues_.push_back(queue);
        }
        // CPU families can provide independent queues over shared executors.
        // Acquire only the inventory this row needs, outside measured work.
        iree_hal_queue_params_t queue_params;
        iree_hal_queue_params_initialize(&queue_params);
        while (can_acquire && queues_.size() < requested_queue_count &&
               iree_status_is_ok(status)) {
          iree_hal_queue_t* queue = nullptr;
          status = iree_hal_queue_acquire(queue_family, &queue_params, &queue);
          if (iree_status_is_ok(status)) {
            queues_.push_back(queue);
          }
        }
        const auto id = iree_hal_device_id(device_);
        label_ = std::string(id.data, id.size) + "/family:" + std::to_string(i);
        break;
      }
    }
    IREE_RETURN_IF_ERROR(status);
    if (queues_.empty()) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "no available transfer queue in the selected families");
    }
    const iree_hal_pool_scope_t scope = {
        .family_count = families.size(),
        .families = families.data(),
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_ALL,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
            },
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_RETURN_IF_ERROR(iree_hal_slab_pool_create(group_, scope, &options,
                                                   allocator, &memory_pool_));
    return iree_ok_status();
  }

  iree_hal_device_t* device() const { return device_; }
  iree_hal_pool_t* pool() const { return memory_pool_; }
  iree_hal_queue_t* queue(size_t index) const { return queues_[index]; }
  size_t queue_count() const { return queues_.size(); }
  const std::string& label() const { return label_; }
  uint64_t allocation_count() const { return allocation_count_.load(); }

 private:
  static iree_status_t Allocate(void* user_data,
                                iree_allocator_command_t command,
                                const void* params, void** inout_pointer) {
    auto* self = static_cast<QueueDevice*>(user_data);
    const bool had_pointer = (command == IREE_ALLOCATOR_COMMAND_FREE ||
                              command == IREE_ALLOCATOR_COMMAND_REALLOC) &&
                             *inout_pointer != nullptr;
    auto allocator = iree_allocator_system();
    IREE_RETURN_IF_ERROR(
        allocator.ctl(allocator.self, command, params, inout_pointer));
    if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_pointer) {
      self->allocation_count_.fetch_add(1, std::memory_order_relaxed);
    }
    return iree_ok_status();
  }

  // Device creation and asynchronous host notification owners.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // Device owning the queues and their native resources.
  iree_hal_device_t* device_ = nullptr;
  // Sealed topology and completion frontier owner.
  iree_hal_device_group_t* group_ = nullptr;
  // Pool outliving every chain buffer and terminal operation.
  iree_hal_pool_t* memory_pool_ = nullptr;
  // Device and selected family identity included with every result.
  std::string label_;
  // Retained provisioned queues followed by any dynamically acquired queues.
  std::vector<iree_hal_queue_t*> queues_;
  // Successful HAL host allocations, including all device worker threads.
  std::atomic<uint64_t> allocation_count_ = 0;
};

// A four-byte value travels through distinct slots. Resetting every slot and
// changing the input between iterations makes missing dependency edges visible.
// Host mappings and verification are outside both timing modes.
class TransferChain {
 public:
  ~TransferChain() {
    IREE_CHECK_OK(Wait());
    iree_hal_command_buffer_release(command_buffer_);
    for (auto* command_buffer : copies_) {
      iree_hal_command_buffer_release(command_buffer);
    }
    iree_hal_buffer_release(buffer_);
    iree_hal_semaphore_release(timeline_);
  }

  iree_status_t Initialize(const QueueDevice& device, size_t queue_index,
                           iree_host_size_t depth, Submission submission) {
    queue_ = device.queue(queue_index);
    if (submission == Submission::kCrossQueue) {
      peer_ = device.queue((queue_index + 1) % device.queue_count());
    }
    values_.resize(depth + 1);
    IREE_RETURN_IF_ERROR(iree_hal_pool_allocate_buffer(
        device.pool(),
        iree_hal_make_pool_reservation_request(
            values_.size() * sizeof(uint32_t), sizeof(uint32_t)),
        iree_infinite_timeout(), &buffer_));
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        device.device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &timeline_));

    if (submission == Submission::kExecute) {
      copies_.resize(depth);
      iree_status_t status = iree_ok_status();
      for (iree_host_size_t i = 0; i < depth && iree_status_is_ok(status);
           ++i) {
        status = Record(i, 1, &copies_[i]);
      }
      return status;
    }
    if (submission == Submission::kRecorded ||
        submission == Submission::kMixed) {
      return Record(0, depth - (submission == Submission::kMixed ? 1 : 0),
                    &command_buffer_);
    }
    return iree_ok_status();
  }

  iree_status_t Record(iree_host_size_t first, iree_host_size_t count,
                       iree_hal_command_buffer_t** out_command_buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue_), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0, out_command_buffer));
    auto* command_buffer = *out_command_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
    // Recorded execution owns payload visibility. Match the conservative
    // boundaries of direct queue_copy, including the final host read.
    const iree_hal_barrier_t acquire = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
    };
    IREE_RETURN_IF_ERROR(
        iree_hal_command_buffer_barrier(command_buffer, &acquire));
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
    const iree_hal_barrier_t release = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    IREE_RETURN_IF_ERROR(
        iree_hal_command_buffer_barrier(command_buffer, &release));
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

  // Exercise both host transfer directions before measuring a resident chain.
  // Retained staging resources must not silently change steady-state costs.
  iree_status_t PrimeStaging() {
    const auto length = values_.size() * sizeof(uint32_t);
    uint64_t next_value = submitted_value_ + 1;
    iree_hal_semaphore_list_t signals = {1, &timeline_, &next_value};
    IREE_RETURN_IF_ERROR(
        iree_hal_queue_upload(queue_, iree_hal_semaphore_list_empty(), signals,
                              values_.data(), buffer_, 0, length, nullptr));
    submitted_value_ = next_value;
    IREE_RETURN_IF_ERROR(Wait());
    std::vector<uint32_t> received(values_.size());
    ++next_value;
    IREE_RETURN_IF_ERROR(iree_hal_queue_download(
        queue_, iree_hal_semaphore_list_empty(), signals, buffer_, 0,
        received.data(), length, nullptr));
    submitted_value_ = next_value;
    IREE_RETURN_IF_ERROR(Wait());
    if (received != values_) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "staged round trip did not preserve input");
    }
    return iree_ok_status();
  }

  iree_status_t Submit(Submission submission) {
    const iree_host_size_t count = submission == Submission::kRecorded ? 1
                                   : submission == Submission::kMixed
                                       ? 2
                                       : values_.size() - 1;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      uint64_t next_value = submitted_value_ + 1;
      const iree_hal_semaphore_list_t waits = {1, &timeline_,
                                               &submitted_value_};
      const iree_hal_semaphore_list_t signals = {1, &timeline_, &next_value};
      if (submission == Submission::kExecute ||
          submission == Submission::kRecorded ||
          (submission == Submission::kMixed && i == 0)) {
        status = iree_hal_queue_execute(
            queue_, waits, signals,
            submission == Submission::kExecute ? copies_[i] : command_buffer_,
            iree_hal_buffer_binding_table_empty(),
            IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
      } else {
        const iree_host_size_t source =
            submission == Submission::kMixed ? values_.size() - 2 : i;
        status = iree_hal_queue_copy(
            peer_ && (i & 1) ? peer_ : queue_, waits, signals, buffer_,
            source * sizeof(uint32_t), buffer_, (source + 1) * sizeof(uint32_t),
            sizeof(uint32_t), nullptr, IREE_HAL_COPY_FLAG_NONE);
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
  // Borrowed queue, alive with the enclosing QueueDevice.
  iree_hal_queue_t* queue_ = nullptr;
  // Optional second borrowed queue for explicit alternating dependency edges.
  iree_hal_queue_t* peer_ = nullptr;
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

template <Submission kSubmission, bool kSubmitOnly, bool kUsedStaging = false>
void BM_TransferChain(benchmark::State& state) {
  QueueDevice device;
  iree::Status status =
      device.Initialize(kSubmission == Submission::kCrossQueue ? 2 : 1);
  if (!status.ok()) {
    state.SkipWithError(status.ToString().c_str());
    return;
  }
  state.SetLabel(device.label());
  if (kSubmission == Submission::kCrossQueue && device.queue_count() < 2) {
    state.SkipWithMessage("two transfer queues are required");
    return;
  }
  TransferChain chain;
  status = chain.Initialize(device, 0, state.range(0), kSubmission);
  // Populate native replay caches and activate the queue before measuring the
  // resident dependency chain. The first use includes device setup work.
  if (status.ok()) {
    status = chain.Reset();
  }
  if (status.ok() && kUsedStaging) {
    status = chain.PrimeStaging();
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
  const auto allocations_before = device.allocation_count();
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
  state.counters["host_allocations"] =
      device.allocation_count() - allocations_before;
}

BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kCopy, false)
    ->Name("TransferChain/CopyCompletion")
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->Arg(512)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kExecute, false)
    ->Name("TransferChain/ExecuteCompletion")
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->Arg(512)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kExecute, true)
    ->Name("TransferChain/ExecuteSubmit")
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->Arg(512)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kExecute, false, true)
    ->Name("TransferChain/ExecuteAfterStaging")
    ->Arg(1)
    ->Arg(32)
    ->Arg(512)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kRecorded, false)
    ->Name("TransferChain/RecordedCompletion")
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->Arg(512)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kMixed, false)
    ->Name("TransferChain/MixedCompletion")
    ->Arg(2)
    ->Arg(33)
    ->Arg(513)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TransferChain, Submission::kCrossQueue, false)
    ->Name("TransferChain/CrossQueueCompletion")
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

enum class Queues { kShared, kSeparate };
enum class Workload { kChain, kMixed, kCompeting };

static void ReportDurations(benchmark::State& state, const std::string& name,
                            std::vector<iree_duration_t>& samples) {
  std::sort(samples.begin(), samples.end());
  state.counters[name + "_p50_ns"] = samples[samples.size() / 2];
  state.counters[name + "_p99_ns"] = samples[samples.size() * 99 / 100];
}

// Persistent callers start each checked batch together. Every caller measures
// its own enqueue and final completion, so a long stream cannot hide a small
// caller's latency behind aggregate throughput. Host reset/checks stay outside
// timing; process CPU time includes the runtime's submission/completion
// workers.
static void BM_Concurrent(benchmark::State& state, Queues placement,
                          Workload workload) {
  const size_t caller_count = state.range(0);
  QueueDevice device;
  iree::Status status =
      device.Initialize(placement == Queues::kSeparate ? caller_count : 1);
  if (!status.ok()) {
    state.SkipWithError(status.ToString().c_str());
    return;
  }
  state.SetLabel(device.label());
  if (placement == Queues::kSeparate && device.queue_count() < caller_count) {
    state.SkipWithMessage("not enough transfer queues in the selected family");
    return;
  }
  struct Caller {
    // Independent values and timeline, shared native execution resources.
    TransferChain chain;
    // Selected public operation shape for every batch.
    Submission submission = Submission::kExecute;
    // Number of value propagation operations checked per batch.
    size_t depth = 0;
    // Host time immediately before submission starts.
    iree_time_t begin = 0;
    // Host time after the last accepted operation returns.
    iree_time_t submitted = 0;
    // Host time after the caller's final public completion wait.
    iree_time_t completed = 0;
    // Per-batch caller enqueue durations, collected outside timing.
    std::vector<iree_duration_t> enqueue_samples;
    // Per-batch caller completion durations, collected outside timing.
    std::vector<iree_duration_t> completion_samples;
    // Application caller retained across warmup and measurement batches.
    std::thread thread;
  };
  std::vector<std::unique_ptr<Caller>> callers;
  size_t operation_count = 0;
  for (size_t i = 0; i < caller_count; ++i) {
    auto caller = std::make_unique<Caller>();
    caller->depth = state.range(1);
    if (workload == Workload::kMixed) {
      caller->submission = Submission::kMixed;
      ++caller->depth;
    } else if (workload == Workload::kCompeting) {
      caller->submission = i == 0 ? Submission::kRecorded : Submission::kCopy;
      if (i != 0) {
        caller->depth = 1;
      }
    }
    status =
        caller->chain.Initialize(device, placement == Queues::kShared ? 0 : i,
                                 caller->depth, caller->submission);
    if (!status.ok()) {
      state.SkipWithError(status.ToString().c_str());
      return;
    }
    operation_count += caller->depth;
    callers.push_back(std::move(caller));
  }
  std::barrier<> phase(caller_count + 1);
  // Published through the start barrier after every previous batch has joined.
  bool stop = false;
  for (auto& caller : callers) {
    caller->thread = std::thread([&, c = caller.get()] {
      while (true) {
        phase.arrive_and_wait();
        if (stop) {
          return;
        }
        c->begin = iree_time_now();
        IREE_CHECK_OK(c->chain.Submit(c->submission));
        c->submitted = iree_time_now();
        IREE_CHECK_OK(c->chain.Wait());
        c->completed = iree_time_now();
        phase.arrive_and_wait();
      }
    });
  }
  for (size_t i = 0; i < 2; ++i) {
    for (auto& caller : callers) {
      IREE_CHECK_OK(caller->chain.Reset());
    }
    phase.arrive_and_wait();
    phase.arrive_and_wait();
    for (auto& caller : callers) {
      IREE_CHECK_OK(caller->chain.Verify());
    }
  }
  const auto allocations_before = device.allocation_count();
  for (auto _ : state) {
    state.PauseTiming();
    for (auto& caller : callers) {
      IREE_CHECK_OK(caller->chain.Reset());
    }
    state.ResumeTiming();
    phase.arrive_and_wait();
    phase.arrive_and_wait();
    state.PauseTiming();
    for (auto& caller : callers) {
      IREE_CHECK_OK(caller->chain.Verify());
      caller->enqueue_samples.push_back(caller->submitted - caller->begin);
      caller->completion_samples.push_back(caller->completed - caller->begin);
    }
    state.ResumeTiming();
  }
  stop = true;
  phase.arrive_and_wait();
  for (auto& caller : callers) {
    caller->thread.join();
  }
  for (size_t i = 0; i < callers.size(); ++i) {
    const auto name = "caller" + std::to_string(i);
    ReportDurations(state, name + "_enqueue", callers[i]->enqueue_samples);
    ReportDurations(state, name + "_completion",
                    callers[i]->completion_samples);
  }
  state.counters["host_allocations"] =
      device.allocation_count() - allocations_before;
  state.counters["callers"] = caller_count;
  state.counters["queues"] = placement == Queues::kShared ? 1 : caller_count;
  state.SetItemsProcessed(state.iterations() * operation_count);
}

#define CONCURRENT_CASE(name, queues, workload)                              \
  BENCHMARK_CAPTURE(BM_Concurrent, name, Queues::queues, Workload::workload) \
      ->Args({1, 8})                                                         \
      ->Args({2, 32})                                                        \
      ->Args({4, 32})                                                        \
      ->Args({2, 512})                                                       \
      ->Args({4, 512})                                                       \
      ->Iterations(50)                                                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime()
CONCURRENT_CASE(SharedChain, kShared, kChain);
CONCURRENT_CASE(SeparateChain, kSeparate, kChain);
CONCURRENT_CASE(SharedMixed, kShared, kMixed);
CONCURRENT_CASE(SeparateMixed, kSeparate, kMixed);
#undef CONCURRENT_CASE

BENCHMARK_CAPTURE(BM_Concurrent, SharedCompeting, Queues::kShared,
                  Workload::kCompeting)
    ->Args({2, 8})
    ->Args({2, 512})
    ->Iterations(100)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_Concurrent, SeparateCompeting, Queues::kSeparate,
                  Workload::kCompeting)
    ->Args({2, 8})
    ->Args({2, 512})
    ->Iterations(100)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

}  // namespace
