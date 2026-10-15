// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <atomic>
#include <cstring>
#include <semaphore>
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
class NativeDispatch {
 public:
  ~NativeDispatch() {
    iree_hal_semaphore_release(timeline_);
    iree_hal_executable_release(executable_);
    iree_hal_buffer_release(output_);
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

  iree_status_t Initialize() {
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
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
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
    IREE_RETURN_IF_ERROR(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 4),
        iree_infinite_timeout(), &output_));
    iree_hal_memory_scope_t program_scope;
    IREE_RETURN_IF_ERROR(iree_hal_device_group_resolve_memory_scope(
        group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &program_scope));
    iree_hal_buffer_native_binding_slot_t address_slot;
    IREE_RETURN_IF_ERROR(iree_hal_pool_resolve_binding(
        pool_, program_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
        &address_slot));
    address_ =
        iree_hal_buffer_native_binding(output_, address_slot).device_address;

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
    IREE_RETURN_IF_ERROR(
        iree_hal_executable_load(family, target, &load, &executable_));
    IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("image_transform"), &function_));
    return iree_hal_semaphore_create(device_,
                                     IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                     IREE_HAL_SEMAPHORE_FLAG_NONE, &timeline_);
  }

  iree_status_t Submit(Arguments arguments, Barriers barriers) {
    struct {
      uint64_t output;
      uint32_t value;
    } parameters = {address_, static_cast<uint32_t>(epoch_ + 1)};
    uint64_t before = epoch_, after = epoch_ + 1;
    const auto binding = iree_hal_make_buffer_ref(output_, 0, sizeof(uint32_t));
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    const iree_hal_queue_barriers_t empty_barriers = {};
    IREE_RETURN_IF_ERROR(iree_hal_queue_dispatch(
        queue_, {epoch_ ? 1u : 0u, &timeline_, &before},
        {1, &timeline_, &after}, executable_, function_, config,
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
    expected_ = parameters.value * 3 + 19;
    epoch_ = after;
    return iree_ok_status();
  }

  iree_status_t Complete() {
    // Explicit-empty dispatches have no inter-kernel data dependency: each
    // reads immutable globals and overwrites the same output in semaphore
    // order. The final default barrier makes that output visible to the CPU.
    uint64_t before = epoch_, after = epoch_ + 1;
    IREE_RETURN_IF_ERROR(iree_hal_queue_barrier(
        queue_, {1, &timeline_, &before}, {1, &timeline_, &after}, nullptr,
        IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    epoch_ = after;
    return iree_hal_semaphore_wait(timeline_, epoch_, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_status_t CheckOutput() {
    uint32_t actual = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_map_read(output_, 0, &actual, sizeof(actual)));
    if (actual != expected_) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "native output mismatch: %u != %u", actual,
                              expected_);
    }
    return iree_ok_status();
  }

  uint64_t allocation_count() const { return allocation_count_.load(); }

 private:
  static iree_status_t Allocate(void* user_data,
                                iree_allocator_command_t command,
                                const void* params, void** inout_pointer) {
    auto* self = static_cast<NativeDispatch*>(user_data);
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
  // Owned result buffer; native-argument rows retain it outside submissions.
  iree_hal_buffer_t* output_ = nullptr;
  // Immutable GPU projection of output_.
  uint64_t address_ = 0;
  // Exact-family HIP executable with immutable inputs to image_transform.
  iree_hal_executable_t* executable_ = nullptr;
  // Prepared reflected kernel identity.
  iree_hal_executable_function_t function_ = {};
  // Explicit execution ordering; queue order alone is not an ordering edge.
  iree_hal_semaphore_t* timeline_ = nullptr;
  // Last accepted timeline value, including final CPU-visibility barriers.
  uint64_t epoch_ = 0;
  // Expected output from the last dispatch, checked after each burst.
  uint32_t expected_ = 0;
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
  NativeDispatch dispatch;
  iree::Status status = dispatch.Initialize();
  if (!status.ok()) {
    state.SkipWithError(status.ToString().c_str());
    return;
  }
  const size_t count = state.range(0);
  for (size_t i = 0; i < count * 2; ++i) {
    IREE_CHECK_OK(dispatch.Submit(arguments, barriers));
  }
  IREE_CHECK_OK(dispatch.Complete());
  IREE_CHECK_OK(dispatch.CheckOutput());
  const uint64_t allocations_before = dispatch.allocation_count();
  std::vector<iree_duration_t> samples;
  if (timing == Timing::kSampledSubmit) {
    samples.reserve(200 * count);
  }
  for (auto _ : state) {
    for (size_t i = 0; i < count; ++i) {
      if (timing == Timing::kSampledSubmit) {
        const iree_time_t begin = iree_time_now();
        IREE_CHECK_OK(dispatch.Submit(arguments, barriers));
        samples.push_back(iree_time_now() - begin);
      } else {
        IREE_CHECK_OK(dispatch.Submit(arguments, barriers));
      }
    }
    if (timing != Timing::kCompletion) {
      state.PauseTiming();
    }
    IREE_CHECK_OK(dispatch.Complete());
    if (timing == Timing::kCompletion) {
      state.PauseTiming();
    }
    IREE_CHECK_OK(dispatch.CheckOutput());
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

}  // namespace
