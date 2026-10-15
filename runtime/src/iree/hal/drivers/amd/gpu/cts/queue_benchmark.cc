// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstring>
#include <semaphore>
#include <thread>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/gpu/device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/slab_pool.h"

namespace {

enum class Arguments { kBindings, kNative };
enum class Barriers { kDefault, kExplicitEmpty };
enum class Timing { kSubmit, kCompletion, kSampledSubmit };

// The allocator observes complete asynchronous device teardown so each row
// starts after the previous row's workers and native owners have joined. Warm
// dispatches reuse capture blocks; allocation counts report slow-path growth.
class NativeQueue {
 public:
  ~NativeQueue() {
    for (auto& lane : lanes_) {
      iree_hal_semaphore_release(lane.timeline);
      iree_hal_executable_release(lane.executable);
      iree_hal_buffer_release(lane.output);
    }
    iree_hal_pool_release(pool_);
    iree_hal_device_group_release(group_);
    if (device_) {
      iree_hal_device_release(device_);
      reclaimed_.acquire();
    }
    iree_hal_amd_gpu_context_free(context_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_proactor_pool_release(proactors_);
    iree_hal_amd_fabric_release(fabric_);
  }

  iree_status_t Initialize(size_t producer_count = 1) {
    lanes_.resize(producer_count);
    const amdf_api_t* api = nullptr;
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api),
        "amdf_query_api"));
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_create(
        api, AMDF_NATIVE_LIFETIME_INSTANCE, iree_allocator_system(), &fabric_));
    for (uint32_t i = 0; i < fabric_->endpoint_count; ++i) {
      if (fabric_->endpoints[i].engine_kind == AMDF_ENGINE_KIND_GPU) {
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_context_allocate(
            fabric_, &fabric_->endpoints[i].id,
            iree_hal_device_event_sink_stderr(), iree_allocator_system(),
            &context_));
        break;
      }
    }
    if (!context_) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no native AMD GPU endpoint");
    }
    const uint32_t gfx_major = context_->target_info.gfx_ip.major;
    iree_hal_amd_gpu_device_options_t options;
    iree_hal_amd_gpu_device_options_initialize(&options);
    options.queue_family_ordinal = UINT32_MAX;
    for (uint32_t i = 0; i < context_->endpoint_info.queue_family_count; ++i) {
      const auto& family = context_->families[i];
      if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
          (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) &&
          (family.user_queue_capabilities &
           AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER)) {
        options.queue_family_ordinal = i;
        break;
      }
    }
    if (options.queue_family_ordinal == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no PM4 USER host-producer family");
    }
    IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactors_));
    IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker_));
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_device_create(
        context_, IREE_SV("PM4 queue benchmark"), &options, &params,
        iree_allocator_t{this, Allocate}, &device_));
    context_ = nullptr;
    armed_.store(true);
    IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
        device_, tracker_, iree_allocator_system(), &group_));
    queue_ = iree_hal_device_queue(device_, 0, 0);
    const auto* family = iree_hal_queue_family(queue_);
    const iree_hal_pool_family_access_t access = {
        .family = family,
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &access,
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_READ,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
            },
    };
    iree_hal_slab_pool_options_t pool_options;
    iree_hal_slab_pool_options_initialize(&pool_options);
    IREE_RETURN_IF_ERROR(iree_hal_slab_pool_create(
        group_, scope, &pool_options, iree_allocator_system(), &pool_));
    iree_hal_memory_scope_t program_scope;
    IREE_RETURN_IF_ERROR(iree_hal_device_group_resolve_memory_scope(
        group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &program_scope));
    iree_hal_buffer_native_binding_slot_t address_slot;
    IREE_RETURN_IF_ERROR(iree_hal_pool_resolve_binding(
        pool_, program_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
        &address_slot));

    const auto* files = iree_hal_amd_gpu_hip_images_create();
    iree_const_byte_span_t artifact = iree_const_byte_span_empty();
    const char* selector = gfx_major == 11 ? "gfx11_generic" : "gfx12_generic";
    for (size_t i = 0; i < iree_hal_amd_gpu_hip_images_size(); ++i) {
      if (std::strstr(files[i].name, selector)) {
        artifact = iree_make_const_byte_span(files[i].data, files[i].size);
      }
    }
    auto* target =
        &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
             ->targets[0];
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    load.executable_data = artifact;
    for (auto& lane : lanes_) {
      IREE_RETURN_IF_ERROR(iree_hal_pool_allocate_buffer(
          pool_,
          iree_hal_make_pool_reservation_request(2 * sizeof(uint32_t), 4),
          iree_infinite_timeout(), &lane.output));
      lane.address = iree_hal_buffer_native_binding(lane.output, address_slot)
                         .device_address;
      IREE_RETURN_IF_ERROR(
          iree_hal_executable_load(family, target, &load, &lane.executable));
      IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
          lane.executable, IREE_SV("image_transform"), &lane.function));
      IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
          device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_NONE, &lane.timeline));
    }
    return iree_ok_status();
  }

  iree_status_t Submit(size_t producer, Arguments arguments,
                       Barriers barriers) {
    auto& lane = lanes_[producer];
    struct {
      // Native kernarg address of this stream's result word.
      uint64_t output;
      // Dispatch-specific input, also checked in the final result.
      uint32_t value;
    } parameters = {lane.address, static_cast<uint32_t>(lane.epoch + 1)};
    uint64_t before = lane.epoch, after = lane.epoch + 1;
    const auto binding =
        iree_hal_make_buffer_ref(lane.output, 0, sizeof(uint32_t));
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    const iree_hal_barrier_list_t empty = {};
    const iree_hal_queue_barriers_t empty_barriers = {&empty, &empty};
    IREE_RETURN_IF_ERROR(iree_hal_queue_dispatch(
        queue_, {lane.epoch ? 1u : 0u, &lane.timeline, &before},
        {1, &lane.timeline, &after}, lane.executable, lane.function, config,
        arguments == Arguments::kNative
            ? iree_make_const_byte_span(&parameters, 12)
            : iree_make_const_byte_span(&parameters.value, 4),
        arguments == Arguments::kNative
            ? iree_hal_buffer_ref_list_empty()
            : iree_hal_buffer_ref_list_t{1, &binding},
        barriers == Barriers::kDefault ? nullptr : &empty_barriers,
        arguments == Arguments::kNative
            ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS
            : IREE_HAL_DISPATCH_FLAG_NONE));
    lane.expected = parameters.value * 3 + 19;
    lane.epoch = after;
    return iree_ok_status();
  }

  iree_status_t PrepareTransfers() {
    auto& lane = lanes_[0];
    uint64_t after = lane.epoch + 1;
    IREE_RETURN_IF_ERROR(iree_hal_queue_fill(
        queue_, {}, {1, &lane.timeline, &after}, lane.output, 0, 8,
        &lane.host_value, 4, nullptr, IREE_HAL_FILL_FLAG_NONE));
    lane.epoch = after;
    lane.expected = lane.host_value;
    return Complete(0);
  }

  iree_status_t Transfer(iree_hal_transfer_operation_type_t type,
                         Barriers barriers) {
    auto& lane = lanes_[0];
    uint64_t before = lane.epoch, after = lane.epoch + 1;
    const uint32_t value = static_cast<uint32_t>(after);
    iree_hal_transfer_operation_t operation = {.type = type};
    switch (type) {
      case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
        operation.fill = {
            .target_buffer = lane.output,
            .length = 4,
            .pattern = &value,
            .pattern_length = 4,
        };
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
        operation.update = {
            .source_buffer = &value,
            .target_buffer = lane.output,
            .length = 4,
        };
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
        operation.copy = {
            .source_buffer = lane.output,
            .source_offset = 4,
            .target_buffer = lane.output,
            .length = 4,
        };
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
        operation.upload = {
            .source = &lane.host_value,
            .target_buffer = lane.output,
            .length = 4,
        };
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        operation.download = {
            .source_buffer = lane.output,
            .target = &lane.downloaded,
            .length = 4,
        };
        break;
    }
    const iree_hal_barrier_list_t empty = {};
    const iree_hal_queue_barriers_t empty_barriers = {&empty, &empty};
    IREE_RETURN_IF_ERROR(iree_hal_queue_transfer(
        queue_, {1, &lane.timeline, &before}, {1, &lane.timeline, &after}, 1,
        &operation,
        barriers == Barriers::kDefault ? nullptr : &empty_barriers));
    lane.epoch = after;
    lane.expected = type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL ||
                            type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE
                        ? value
                        : lane.host_value;
    return iree_ok_status();
  }

  iree_status_t CheckTransferOutput(iree_hal_transfer_operation_type_t type) {
    IREE_RETURN_IF_ERROR(CheckOutput(0));
    if (type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD &&
        lanes_[0].downloaded != lanes_[0].expected) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "download did not complete before its signal");
    }
    return iree_ok_status();
  }

  iree_status_t Complete(size_t producer) {
    auto& lane = lanes_[producer];
    // Explicit-empty dispatches have no inter-kernel data dependency: each
    // reads immutable globals and overwrites the same output in semaphore
    // order. The final default barrier makes that output visible to the CPU.
    uint64_t before = lane.epoch, after = lane.epoch + 1;
    IREE_RETURN_IF_ERROR(iree_hal_queue_barrier(
        queue_, {1, &lane.timeline, &before}, {1, &lane.timeline, &after},
        nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    lane.epoch = after;
    return iree_hal_semaphore_wait(lane.timeline, lane.epoch,
                                   iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_status_t CheckOutput(size_t producer) {
    const auto& lane = lanes_[producer];
    uint32_t actual = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_map_read(lane.output, 0, &actual, sizeof(actual)));
    if (actual != lane.expected) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "native output mismatch: %u != %u", actual,
                              lane.expected);
    }
    return iree_ok_status();
  }

  uint64_t allocation_count() const { return allocation_count_.load(); }

  // Host retirement supplies an upper bound on first GPU execution, not a
  // calibrated device timestamp. Complete joins this timepoint's callback.
  iree_status_t ObserveFirst(size_t producer,
                             iree_async_semaphore_timepoint_t* timepoint) {
    auto& lane = lanes_[producer];
    return iree_async_semaphore_acquire_timepoint(
        reinterpret_cast<iree_async_semaphore_t*>(lane.timeline),
        lane.epoch + 1, timepoint);
  }

 private:
  static iree_status_t Allocate(void* user_data,
                                iree_allocator_command_t command,
                                const void* params, void** inout_pointer) {
    auto* self = static_cast<NativeQueue*>(user_data);
    const bool had_pointer = *inout_pointer != nullptr;
    auto allocator = iree_allocator_system();
    IREE_RETURN_IF_ERROR(
        allocator.ctl(allocator.self, command, params, inout_pointer));
    if (command == IREE_ALLOCATOR_COMMAND_FREE && had_pointer) {
      if (self->live_count_.fetch_sub(1) == 1 && self->armed_.load()) {
        self->reclaimed_.release();
      }
    } else if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_pointer) {
      self->live_count_.fetch_add(1);
      self->allocation_count_.fetch_add(1);
    }
    return iree_ok_status();
  }

  // Native discovery owner, independently of the transferred context.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Unique constructing owner, transferred to device on success.
  iree_hal_amd_gpu_context_t* context_ = nullptr;
  // Shared process infrastructure passed into device creation.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // Group progress owner, independent of queue-retained references.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Public owner of the actual queue and both execution workers.
  iree_hal_device_t* device_ = nullptr;
  // Sealed scope assignment for the explicit allocation pool.
  iree_hal_device_group_t* group_ = nullptr;
  // Borrowed queue, kept live by device_.
  iree_hal_queue_t* queue_ = nullptr;
  // Explicit host-readable storage for the kernel result.
  iree_hal_pool_t* pool_ = nullptr;
  // Independent caller-owned streams sharing the exact same HAL/native queue.
  // Separate module globals and result buffers prevent inter-stream data races.
  struct alignas(64) Lane {
    // Owned result buffer, independently retained by ordinary binding captures.
    iree_hal_buffer_t* output = nullptr;
    // Immutable native projection used by caller-owned argument-byte rows.
    uint64_t address = 0;
    // Owned module, including this stream's mutable image_zero global.
    iree_hal_executable_t* executable = nullptr;
    // Prepared reflected kernel identity.
    iree_hal_executable_function_t function = {};
    // Explicit dependencies; queue order is not a caller ordering edge.
    iree_hal_semaphore_t* timeline = nullptr;
    // Last accepted value, including final CPU-visibility barriers.
    uint64_t epoch = 0;
    // Expected output from the last dispatch, checked after each burst.
    uint32_t expected = 0;
    // Immutable source for borrowed uploads and the device copy seed.
    uint32_t host_value = 0x13579BDF;
    // Borrowed host destination, read only after terminal download completion.
    uint32_t downloaded = 0;
  };
  // Stable per-producer state; mutable fields do not share cache lines.
  std::vector<Lane> lanes_;
  // Live device allocations, including handles owned by its workers.
  std::atomic<size_t> live_count_ = 0;
  // Total allocations since construction, sampled only outside timing.
  std::atomic<uint64_t> allocation_count_ = 0;
  // Successful construction enables notification of final asynchronous free.
  std::atomic<bool> armed_ = false;
  // Joins final device allocator access before this fixture is destroyed.
  std::binary_semaphore reclaimed_{0};
};

static void BM_Dispatch(benchmark::State& state, Arguments arguments,
                        Barriers barriers, Timing timing) {
  NativeQueue dispatch;
  iree::Status status = dispatch.Initialize();
  if (!status.ok()) {
    state.SkipWithError(status.ToString().c_str());
    return;
  }
  const size_t count = state.range(0);
  for (size_t i = 0; i < count * 2; ++i) {
    IREE_CHECK_OK(dispatch.Submit(0, arguments, barriers));
  }
  IREE_CHECK_OK(dispatch.Complete(0));
  IREE_CHECK_OK(dispatch.CheckOutput(0));
  const uint64_t allocations_before = dispatch.allocation_count();
  std::vector<iree_duration_t> samples;
  if (timing == Timing::kSampledSubmit) {
    samples.reserve(200 * count);
  }
  for (auto _ : state) {
    for (size_t i = 0; i < count; ++i) {
      if (timing == Timing::kSampledSubmit) {
        const iree_time_t begin = iree_time_now();
        IREE_CHECK_OK(dispatch.Submit(0, arguments, barriers));
        samples.push_back(iree_time_now() - begin);
      } else {
        IREE_CHECK_OK(dispatch.Submit(0, arguments, barriers));
      }
    }
    if (timing != Timing::kCompletion) {
      state.PauseTiming();
    }
    IREE_CHECK_OK(dispatch.Complete(0));
    if (timing == Timing::kCompletion) {
      state.PauseTiming();
    }
    IREE_CHECK_OK(dispatch.CheckOutput(0));
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.counters["host_allocations"] =
      dispatch.allocation_count() - allocations_before;
  if (!samples.empty()) {
    std::sort(samples.begin(), samples.end());
    state.counters["call_p50_ns"] = samples[samples.size() / 2];
    state.counters["call_p99_ns"] = samples[samples.size() * 99 / 100];
    state.counters["call_max_ns"] = samples.back();
  }
}

// Fixed counts keep completion waits outside submit timing from turning a
// duration-calibrated run into millions of hidden GPU dispatches. Sampled rows
// include clock-read overhead and are distinct from bulk submission timing.
#define DISPATCH_CASE(name, arguments, barriers, timing)     \
  BENCHMARK_CAPTURE(BM_Dispatch, name, Arguments::arguments, \
                    Barriers::barriers, Timing::timing)      \
      ->Arg(1)                                               \
      ->Arg(32)                                              \
      ->Iterations(200)                                      \
      ->UseRealTime()
DISPATCH_CASE(BindingsDefaultSubmit, kBindings, kDefault, kSubmit);
DISPATCH_CASE(BindingsEmptySubmit, kBindings, kExplicitEmpty, kSubmit);
DISPATCH_CASE(NativeDefaultSubmit, kNative, kDefault, kSubmit);
DISPATCH_CASE(NativeEmptySubmit, kNative, kExplicitEmpty, kSubmit);
DISPATCH_CASE(BindingsDefaultCompletion, kBindings, kDefault, kCompletion);
DISPATCH_CASE(BindingsDefaultLatency, kBindings, kDefault, kSampledSubmit);
#undef DISPATCH_CASE

static void BM_Transfer(benchmark::State& state,
                        iree_hal_transfer_operation_type_t type,
                        Barriers barriers, Timing timing) {
  NativeQueue queue;
  iree::Status status = queue.Initialize();
  if (!status.ok()) {
    state.SkipWithError(status.ToString().c_str());
    return;
  }
  IREE_CHECK_OK(queue.PrepareTransfers());
  const size_t count = state.range(0);
  for (size_t i = 0; i < count * 2; ++i) {
    IREE_CHECK_OK(queue.Transfer(type, barriers));
  }
  IREE_CHECK_OK(queue.Complete(0));
  IREE_CHECK_OK(queue.CheckTransferOutput(type));
  const uint64_t allocations_before = queue.allocation_count();
  for (auto _ : state) {
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(queue.Transfer(type, barriers));
    }
    if (timing == Timing::kSubmit) {
      state.PauseTiming();
    }
    IREE_CHECK_OK(queue.Complete(0));
    if (timing == Timing::kCompletion) {
      state.PauseTiming();
    }
    IREE_CHECK_OK(queue.CheckTransferOutput(type));
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.counters["host_allocations"] =
      queue.allocation_count() - allocations_before;
}

#define TRANSFER_CASE(name, kind, barriers, timing)          \
  BENCHMARK_CAPTURE(BM_Transfer, name,                       \
                    IREE_HAL_TRANSFER_OPERATION_TYPE_##kind, \
                    Barriers::barriers, Timing::timing)      \
      ->Arg(1)                                               \
      ->Arg(32)                                              \
      ->Iterations(200)                                      \
      ->UseRealTime()
TRANSFER_CASE(FillDefaultSubmit, FILL, kDefault, kSubmit);
TRANSFER_CASE(FillEmptySubmit, FILL, kExplicitEmpty, kSubmit);
TRANSFER_CASE(CopyDefaultSubmit, COPY, kDefault, kSubmit);
TRANSFER_CASE(CopyEmptySubmit, COPY, kExplicitEmpty, kSubmit);
TRANSFER_CASE(UpdateDefaultSubmit, UPDATE, kDefault, kSubmit);
TRANSFER_CASE(UpdateEmptySubmit, UPDATE, kExplicitEmpty, kSubmit);
TRANSFER_CASE(UploadDefaultSubmit, UPLOAD, kDefault, kSubmit);
TRANSFER_CASE(DownloadDefaultSubmit, DOWNLOAD, kDefault, kSubmit);
TRANSFER_CASE(UploadDefaultCompletion, UPLOAD, kDefault, kCompletion);
TRANSFER_CASE(DownloadDefaultCompletion, DOWNLOAD, kDefault, kCompletion);
#undef TRANSFER_CASE

// Persistent callers share one actual queue without a caller-side mutex.
// Barrier rendezvous coordinate only benchmark epochs and stay outside each
// caller's submission interval. Overall wall time includes start/join overhead.
// Process CPU time covers callers and both device workers through final drain.
static void BM_ConcurrentDispatch(benchmark::State& state) {
  const size_t producer_count = state.range(0);
  const size_t count = state.range(1);
  NativeQueue dispatch;
  iree::Status status = dispatch.Initialize(producer_count);
  if (!status.ok()) {
    state.SkipWithError(status.ToString().c_str());
    return;
  }
  struct alignas(64) Producer {
    // Host time immediately before this caller's first public dispatch.
    iree_time_t begin = 0;
    // Host time immediately after its last dispatch returned.
    iree_time_t end = 0;
    // Host time after the final visibility barrier and completion wait return.
    iree_time_t completed = 0;
    // First native retirement observed on the existing completion worker.
    std::atomic<iree_time_t> first_retirement = 0;
    // Registration lives until every operation in this burst has completed.
    iree_async_semaphore_timepoint_t first = {};
    // Persistent application caller, independent of both device workers.
    std::thread thread;
  };
  std::vector<Producer> producers(producer_count);
  std::barrier<> phase(producer_count + 1);
  // Published through the preparation barrier after all final drains join.
  bool stop = false;
  for (size_t p = 0; p < producer_count; ++p) {
    producers[p].thread = std::thread([&, p] {
      auto& producer = producers[p];
      for (size_t i = 0; i < count * 2; ++i) {
        IREE_CHECK_OK(
            dispatch.Submit(p, Arguments::kBindings, Barriers::kDefault));
      }
      IREE_CHECK_OK(dispatch.Complete(p));
      IREE_CHECK_OK(dispatch.CheckOutput(p));
      phase.arrive_and_wait();
      for (;;) {
        phase.arrive_and_wait();
        if (stop) {
          break;
        }
        producer.first_retirement.store(0);
        producer.first = {
            .callback =
                [](void* data, iree_async_semaphore_timepoint_t*,
                   iree_status_t status) {
                  IREE_CHECK_OK(status);
                  static_cast<Producer*>(data)->first_retirement.store(
                      iree_time_now(), std::memory_order_release);
                },
            .user_data = &producer,
        };
        IREE_CHECK_OK(dispatch.ObserveFirst(p, &producer.first));
        phase.arrive_and_wait();
        phase.arrive_and_wait();
        producer.begin = iree_time_now();
        for (size_t i = 0; i < count; ++i) {
          IREE_CHECK_OK(
              dispatch.Submit(p, Arguments::kBindings, Barriers::kDefault));
        }
        producer.end = iree_time_now();
        phase.arrive_and_wait();
        phase.arrive_and_wait();
        IREE_CHECK_OK(dispatch.Complete(p));
        producer.completed = iree_time_now();
        IREE_CHECK_OK(dispatch.CheckOutput(p));
        phase.arrive_and_wait();
      }
    });
  }
  phase.arrive_and_wait();
  const uint64_t allocations_before = dispatch.allocation_count();
  iree_duration_t caller_duration = 0;
  iree_duration_t first_retirement_duration = 0;
  iree_duration_t last_acceptance_duration = 0;
  iree_duration_t completion_duration = 0;
  iree_duration_t drain_duration = 0;
  uint64_t pipelined_bursts = 0;
  for (auto _ : state) {
    phase.arrive_and_wait();
    phase.arrive_and_wait();
    const iree_time_t begin = iree_time_now();
    phase.arrive_and_wait();
    phase.arrive_and_wait();
    const iree_time_t accepted = iree_time_now();
    state.SetIterationTime((accepted - begin) * 1e-9);
    phase.arrive_and_wait();
    phase.arrive_and_wait();
    iree_time_t first_retirement = INT64_MAX;
    iree_time_t last_acceptance = 0;
    iree_time_t completed = 0;
    for (auto& producer : producers) {
      caller_duration += producer.end - producer.begin;
      first_retirement =
          std::min(first_retirement, producer.first_retirement.load());
      last_acceptance = std::max(last_acceptance, producer.end);
      completed = std::max(completed, producer.completed);
    }
    first_retirement_duration += first_retirement - begin;
    last_acceptance_duration += last_acceptance - begin;
    completion_duration += completed - begin;
    drain_duration += completed - last_acceptance;
    pipelined_bursts += first_retirement < last_acceptance;
  }
  stop = true;
  phase.arrive_and_wait();
  for (auto& producer : producers) {
    producer.thread.join();
  }
  const double calls = state.iterations() * producer_count * count;
  state.SetItemsProcessed(calls);
  state.counters["caller_mean_ns"] = caller_duration / calls;
  state.counters["first_retirement_ns"] =
      static_cast<double>(first_retirement_duration) / state.iterations();
  state.counters["last_acceptance_ns"] =
      static_cast<double>(last_acceptance_duration) / state.iterations();
  // End-to-end counters include native completion and CPU visibility. They are
  // separate from the benchmark's submit-only interval and caller_mean_ns.
  state.counters["completion_ns"] =
      static_cast<double>(completion_duration) / state.iterations();
  state.counters["drain_ns"] =
      static_cast<double>(drain_duration) / state.iterations();
  state.counters["pipelined_percent"] =
      100.0 * pipelined_bursts / state.iterations();
  state.counters["host_allocations"] =
      dispatch.allocation_count() - allocations_before;
}
BENCHMARK(BM_ConcurrentDispatch)
    ->Args({1, 32})
    ->Args({2, 32})
    ->Args({4, 32})
    ->Args({1, 1000})
    ->Args({2, 1000})
    ->Args({4, 1000})
    ->Iterations(50)
    ->MeasureProcessCPUTime()
    ->UseManualTime();

}  // namespace
