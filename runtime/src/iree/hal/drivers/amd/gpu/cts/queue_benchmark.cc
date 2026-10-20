// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/gpu/device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/queue.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/slab_pool.h"
#include "iree/hal/utils/statistics_sink.h"

namespace {

enum class Arguments { kBindings, kNative };
enum class BindingMode { kDirect, kTable };
enum class Program { kDispatch, kTransfers };
enum class Barriers { kDefault, kExplicitEmpty };
enum class Timing { kSubmit, kCompletion, kSampledSubmit };
enum class Capture { kNone, kMetadata, kActive };
enum class Completion { kBarrier, kSubmission };

// KFD's process VM belongs to this fabric throughout the benchmark process.
// Rows still join and release their devices, queues, and execution storage.
struct ProcessFabric {
  ProcessFabric() {
    const amdf_api_t* api = nullptr;
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api),
        "amdf_query_api");
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_fabric_create(api, AMDF_NATIVE_LIFETIME_PROCESS,
                                          iree_allocator_system(), &fabric);
    }
  }
  ~ProcessFabric() {
    iree_hal_amd_fabric_release(fabric);
    iree_status_free(status);
  }
  // One owning reference outlives every benchmark row.
  iree_hal_amd_fabric_t* fabric = nullptr;
  // A construction failure is reported by every row using this owner.
  iree_status_t status = iree_ok_status();
};

static iree_status_t MatchesRunnerSelection(iree_hal_amd_fabric_t* fabric,
                                            const amdf_endpoint_id_t* id,
                                            bool* out_matches) {
  *out_matches = true;
  const char* selected = std::getenv("AMDF_CTS_GPU_NATIVE_IDENTITY");
  if (!selected) {
    return iree_ok_status();
  }
  amdf_endpoint_t* endpoint = nullptr;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      fabric->api->endpoint_open(fabric->instance, id, &endpoint),
      "endpoint_open"));
  amdf_endpoint_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(info),
  };
  const auto status = fabric->api->endpoint_query_info(endpoint, &info);
  const auto close_status = fabric->api->endpoint_close(endpoint);
  IREE_RETURN_IF_ERROR(iree_status_join(
      IREE_HAL_AMD_STATUS_FROM_AMDF(status, "endpoint_query_info"),
      IREE_HAL_AMD_STATUS_FROM_AMDF(close_status, "endpoint_close")));
  char identity[96] = {};
  if (info.native_identity.type ==
      AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE) {
    std::snprintf(identity, sizeof(identity), "linux_device:%u:%u",
                  info.native_identity.value.linux_device.major,
                  info.native_identity.value.linux_device.minor);
  } else if (info.native_identity.type ==
             AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER) {
    std::snprintf(
        identity, sizeof(identity), "windows_adapter:%" PRIx64 ":%u",
        info.native_identity.value.windows_adapter.luid,
        info.native_identity.value.windows_adapter.physical_adapter_index);
  }
  *out_matches = std::strcmp(selected, identity) == 0;
  return iree_ok_status();
}

// Warm dispatches reuse capture blocks; allocation counts report slow-path
// growth. Final device release joins workers and native owners between rows.
class NativeQueue {
 public:
  explicit NativeQueue(
      amdf_queue_command_type_t command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
      bool private_stream = false)
      : command_type_(command_type), private_stream_(private_stream) {}

  ~NativeQueue() {
    for (auto& lane : lanes_) {
      iree_hal_semaphore_release(lane.timeline);
      iree_hal_executable_release(lane.executable);
      iree_hal_buffer_release(lane.output);
    }
    iree_hal_semaphore_release(completion_);
    iree_hal_pool_release(pool_);
    iree_hal_device_release(device_);
    iree_hal_device_group_release(group_);
    iree_hal_amd_gpu_context_free(context_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_proactor_pool_release(proactors_);
    iree_hal_amd_fabric_release(fabric_);
    iree_hal_profile_statistics_sink_release(capture_);
  }

  bool Initialize(benchmark::State& state, size_t producer_count = 1,
                  size_t buffer_length = 2 * sizeof(uint32_t)) {
    iree::Status status = InitializeDevice(producer_count, buffer_length);
    if (!status.ok()) {
      state.SkipWithError(status.ToString().c_str());
      return false;
    }
    if (!device_) {
      state.SkipWithMessage(
          "No native AMD GPU endpoint with the requested packet family");
      return false;
    }
    return true;
  }

  iree_status_t InitializeDevice(size_t producer_count, size_t buffer_length) {
    lanes_.resize(producer_count);
    static ProcessFabric shared;
    IREE_RETURN_IF_ERROR(iree_status_clone(shared.status));
    fabric_ = shared.fabric;
    iree_hal_amd_fabric_retain(fabric_);
    uint32_t family_ordinal = UINT32_MAX;
    for (uint32_t i = 0; i < fabric_->endpoint_count; ++i) {
      if (fabric_->endpoints[i].engine_kind == AMDF_ENGINE_KIND_GPU) {
        bool selected = false;
        IREE_RETURN_IF_ERROR(MatchesRunnerSelection(
            fabric_, &fabric_->endpoints[i].id, &selected));
        if (!selected) {
          continue;
        }
        iree_status_t status = iree_hal_amd_gpu_context_allocate(
            fabric_, &fabric_->endpoints[i].id,
            iree_hal_device_event_sink_stderr(), iree_allocator_system(),
            &context_);
        if (iree_status_is_unimplemented(status) ||
            iree_status_is_unavailable(status)) {
          iree_status_ignore(status);
          continue;
        }
        IREE_RETURN_IF_ERROR(status);
        bool supported = false;
        for (uint32_t family = 0;
             family < context_->endpoint_info.queue_family_count; ++family) {
          const auto& native = context_->families[family];
          if (native.command_type != command_type_) {
            continue;
          }
          supported =
              command_type_ == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                  ? (native.format_features &
                     AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE) &&
                        (native.user_queue_capabilities &
                         AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER)
                  : iree_hal_amd_gpu_pm4_queue_publication_modes(&native) != 0;
          if (supported) {
            family_ordinal = family;
            break;
          }
        }
        if (supported) {
          break;
        }
        iree_hal_amd_gpu_context_free(context_);
        context_ = nullptr;
      }
    }
    if (!context_) {
      return iree_ok_status();
    }
    const auto gfxip = context_->target_info.gfx_ip;
    iree_hal_amd_gpu_device_options_t options;
    iree_hal_amd_gpu_device_options_initialize(&options);
    options.queue_family_ordinal = family_ordinal;
    IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactors_));
    IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker_));
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    auto* context = context_;
    context_ = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_device_create(
        context, IREE_SV("AMD queue benchmark"), &options, &params,
        iree_allocator_t{this, Allocate}, &device_));
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
    const char* selector =
        gfxip.major == 9    ? (gfxip.minor == 4 ? "gfx9_4_generic" : "gfx90a")
        : gfxip.major == 11 ? "gfx11_generic"
                            : "gfx12_generic";
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
          pool_, iree_hal_make_pool_reservation_request(buffer_length, 4),
          iree_infinite_timeout(), &lane.output));
      lane.address = iree_hal_buffer_native_binding(lane.output, address_slot)
                         .device_address;
      IREE_RETURN_IF_ERROR(
          iree_hal_executable_load(family, target, &load, &lane.executable));
      IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
          lane.executable, IREE_SV("image_transform"), &lane.function));
      IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
          device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          private_stream_ ? IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
                                IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER
                          : IREE_HAL_SEMAPHORE_FLAG_NONE,
          &lane.timeline));
    }
    if (private_stream_) {
      IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
          device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_NONE, &completion_));
    }
    return iree_ok_status();
  }

  // The final dispatch signals both its private stream and public completion.
  // A zero grid measures admission/signaling only; it launches no kernel.
  iree_status_t SubmitChain(uint32_t count, bool empty) {
    auto& lane = lanes_[0];
    const auto reference = iree_hal_make_buffer_ref(lane.output, 0, 4);
    const uint32_t value = 1;
    auto config = iree_hal_make_static_dispatch_config(empty ? 0 : 1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    iree_hal_semaphore_t* signal_semaphores[] = {lane.timeline, completion_};
    iree_status_t status = iree_ok_status();
    for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      const bool final = i + 1 == count;
      uint64_t before = lane.epoch;
      uint64_t values[] = {before + 1, completion_value_ + 1};
      status = iree_hal_queue_dispatch(
          queue_, {i ? 1u : 0u, &lane.timeline, &before},
          {final ? 2u : 1u, signal_semaphores, values}, lane.executable,
          lane.function, config,
          iree_make_const_byte_span(&value, sizeof(value)), {1, &reference},
          nullptr, IREE_HAL_DISPATCH_FLAG_NONE);
      if (iree_status_is_ok(status)) {
        ++lane.epoch;
        if (final) {
          ++completion_value_;
        }
      }
    }
    lane.expected = value * 3 + 19;
    return status;
  }

  iree_status_t CompleteChain(
      iree_async_wait_flags_t wait_flags = IREE_ASYNC_WAIT_FLAG_NONE) {
    return iree_hal_semaphore_wait(completion_, completion_value_,
                                   iree_infinite_timeout(), wait_flags);
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

  // Includes public creation, validation, native recording, and finalization.
  // Recorded dispatches use the same kernel and dependency shape as Submit.
  iree_status_t Record(uint32_t count, BindingMode bindings, Program program,
                       iree_hal_command_buffer_t** out_recording,
                       iree_hal_command_buffer_mode_t mode = 0) {
    *out_recording = nullptr;
    const auto& lane = lanes_[0];
    const bool transfers = program == Program::kTransfers;
    const iree_hal_execution_stage_t stage =
        transfers ? IREE_HAL_EXECUTION_STAGE_TRANSFER
                  : IREE_HAL_EXECUTION_STAGE_DISPATCH;
    const iree_hal_access_scope_t write_scope =
        transfers ? IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE
                  : IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
    const iree_hal_access_scope_t read_scope =
        transfers ? IREE_HAL_ACCESS_SCOPE_TRANSFER_READ
                  : IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;
    iree_hal_command_buffer_t* recording = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue_), mode,
        transfers ? IREE_HAL_COMMAND_CATEGORY_TRANSFER
                  : IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        bindings == BindingMode::kTable ? 1 : 0, &recording));
    const auto reference = bindings == BindingMode::kTable
                               ? iree_hal_make_indirect_buffer_ref(0, 0, 4)
                               : iree_hal_make_buffer_ref(lane.output, 0, 4);
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    const iree_hal_memory_barrier_t memory = {
        .source_scope = write_scope,
        .target_scope = read_scope | write_scope,
    };
    const iree_hal_barrier_t dependency = {
        .source_stage_mask = stage,
        .target_stage_mask = stage,
        .memory_barrier_count = 1,
        .memory_barriers = &memory,
    };
    const iree_hal_barrier_t acquire = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        .target_stage_mask = stage,
        .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
    };
    const iree_hal_barrier_t release = {
        .source_stage_mask = stage,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    const uint32_t value = 1;
    iree_status_t status = iree_hal_command_buffer_begin(recording);
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_barrier(recording, &acquire);
    }
    for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      if (i) {
        status = iree_hal_command_buffer_barrier(recording, &dependency);
      }
      if (iree_status_is_ok(status) && transfers) {
        // Alternate captured updates, native fills, and device copies. The
        // first operation initializes both words, and explicit edges protect
        // the copy source when the recording crosses publication boundaries.
        const uint32_t payload[] = {22, 22};
        auto target = reference;
        target.length = sizeof(payload);
        if (i % 3 == 0) {
          status = iree_hal_command_buffer_update_buffer(recording, payload, 0,
                                                         target, 0);
        } else if (i % 3 == 1) {
          status = iree_hal_command_buffer_fill_buffer(
              recording, target, payload, sizeof(payload[0]), 0);
        } else {
          auto source = reference;
          source.offset += sizeof(payload[0]);
          status = iree_hal_command_buffer_copy_buffer(recording, source,
                                                       reference, 0);
        }
      } else if (iree_status_is_ok(status)) {
        status = iree_hal_command_buffer_dispatch(
            recording, lane.executable, lane.function, config,
            iree_make_const_byte_span(&value, sizeof(value)), {1, &reference},
            0);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_barrier(recording, &release);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_end(recording);
    }
    if (iree_status_is_ok(status)) {
      *out_recording = recording;
    } else {
      iree_hal_command_buffer_release(recording);
    }
    return status;
  }

  iree_status_t BeginCapture() {
    IREE_RETURN_IF_ERROR(iree_hal_profile_statistics_sink_create(
        iree_allocator_system(), &capture_));
    const iree_hal_device_profiling_options_t options = {
        .data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA,
        .sink = iree_hal_profile_statistics_sink_base(capture_),
    };
    return iree_hal_device_profiling_begin(device_, &options);
  }

  iree_status_t FlushCapture() {
    IREE_RETURN_IF_ERROR(iree_hal_device_profiling_flush(device_));
    if (iree_hal_profile_statistics_sink_dropped_record_count(capture_)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "benchmark capture dropped dispatch records");
    }
    return iree_ok_status();
  }

  iree_status_t EndCapture() { return iree_hal_device_profiling_end(device_); }

  iree_status_t Execute(iree_hal_command_buffer_t* recording,
                        BindingMode bindings) {
    auto& lane = lanes_[0];
    uint64_t before = lane.epoch, after = lane.epoch + 1;
    const iree_hal_buffer_binding_t binding = {
        .buffer = lane.output,
        .length = 8,
    };
    IREE_RETURN_IF_ERROR(iree_hal_queue_execute(
        queue_, {lane.epoch ? 1u : 0u, &lane.timeline, &before},
        {1, &lane.timeline, &after}, recording,
        {bindings == BindingMode::kTable ? 1u : 0u, &binding}, 0));
    lane.expected = 1 * 3 + 19;
    lane.epoch = after;
    return iree_ok_status();
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

  iree_status_t TransferBytes(iree_hal_transfer_operation_type_t type,
                              iree_byte_span_t bytes) {
    auto& lane = lanes_[0];
    uint64_t before = lane.epoch, after = before + 1;
    iree_hal_transfer_operation_t operation = {.type = type};
    if (type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
      operation.download = {
          .source_buffer = lane.output,
          .target = bytes.data,
          .length = bytes.data_length,
      };
    } else if (type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE) {
      operation.update = {
          .source_buffer = bytes.data,
          .target_buffer = lane.output,
          .length = bytes.data_length,
      };
    } else {
      operation.upload = {
          .source = bytes.data,
          .target_buffer = lane.output,
          .length = bytes.data_length,
      };
    }
    IREE_RETURN_IF_ERROR(iree_hal_queue_transfer(
        queue_, {before ? 1u : 0u, &lane.timeline, &before},
        {1, &lane.timeline, &after}, 1, &operation, nullptr));
    lane.epoch = after;
    return iree_hal_semaphore_wait(lane.timeline, after,
                                   iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_status_t ReadBytes(iree_byte_span_t bytes) {
    return iree_hal_buffer_map_read(lanes_[0].output, 0, bytes.data,
                                    bytes.data_length);
  }

  // The concrete queue advances its causal epoch once per native publication.
  // Query only after completion; this counts real retirement units without
  // adding counters or instrumentation to the production submission path.
  uint64_t completed_publications() const {
    iree_async_single_frontier_t frontier;
    const uint8_t count = iree_async_semaphore_query_frontier(
        reinterpret_cast<iree_async_semaphore_t*>(lanes_[0].timeline),
        iree_async_single_frontier_as_frontier(&frontier), 1);
    IREE_CHECK_OK(count == 1
                      ? iree_ok_status()
                      : iree_make_status(IREE_STATUS_DATA_LOSS,
                                         "missing completed queue frontier"));
    return frontier.entries[0].epoch;
  }

  iree_status_t Complete(size_t producer, iree_async_wait_flags_t wait_flags =
                                              IREE_ASYNC_WAIT_FLAG_NONE) {
    auto& lane = lanes_[producer];
    // Explicit-empty dispatches have no inter-kernel data dependency: each
    // reads immutable globals and overwrites the same output in semaphore
    // order. The final default barrier makes that output visible to the CPU.
    uint64_t before = lane.epoch, after = lane.epoch + 1;
    IREE_RETURN_IF_ERROR(iree_hal_queue_barrier(
        queue_, {1, &lane.timeline, &before}, {1, &lane.timeline, &after},
        nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    lane.epoch = after;
    return Wait(producer, wait_flags);
  }

  // Default dispatches and recorded programs carry their own outgoing
  // visibility boundary. Their signals need no extra submission or doorbell.
  iree_status_t Wait(size_t producer, iree_async_wait_flags_t wait_flags) {
    const auto& lane = lanes_[producer];
    return iree_hal_semaphore_wait(lane.timeline, lane.epoch,
                                   iree_infinite_timeout(), wait_flags);
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
    const bool had_pointer = (command == IREE_ALLOCATOR_COMMAND_FREE ||
                              command == IREE_ALLOCATOR_COMMAND_REALLOC) &&
                             *inout_pointer != nullptr;
    auto allocator = iree_allocator_system();
    IREE_RETURN_IF_ERROR(
        allocator.ctl(allocator.self, command, params, inout_pointer));
    if (command == IREE_ALLOCATOR_COMMAND_FREE && had_pointer) {
      self->live_count_.fetch_sub(1);
    } else if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_pointer) {
      self->live_count_.fetch_add(1);
      self->allocation_count_.fetch_add(1);
    }
    return iree_ok_status();
  }

  // Exact requested packet family, independent of shader ISA.
  amdf_queue_command_type_t command_type_;
  // Chain rows use device-local single-producer dependencies.
  bool private_stream_;
  // Public final notification for a chain, distinct from its private stream.
  iree_hal_semaphore_t* completion_ = nullptr;
  // Last accepted public final notification.
  uint64_t completion_value_ = 0;
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
  // Caller-owned aggregate consumer, flushed outside measured queue intervals.
  iree_hal_profile_statistics_sink_t* capture_ = nullptr;
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
};

static void BM_Dispatch(
    benchmark::State& state, Arguments arguments, Barriers barriers,
    Timing timing,
    amdf_queue_command_type_t command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
    Completion completion = Completion::kBarrier,
    iree_async_wait_flags_t wait_flags = IREE_ASYNC_WAIT_FLAG_NONE) {
  NativeQueue dispatch(command_type);
  if (!dispatch.Initialize(state)) {
    return;
  }
  const size_t count = state.range(0);
  const auto complete = [&] {
    return completion == Completion::kBarrier ? dispatch.Complete(0, wait_flags)
                                              : dispatch.Wait(0, wait_flags);
  };
  for (size_t i = 0; i < count * 2; ++i) {
    IREE_CHECK_OK(dispatch.Submit(0, arguments, barriers));
  }
  IREE_CHECK_OK(complete());
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
    IREE_CHECK_OK(complete());
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
DISPATCH_CASE(BindingsEmptyLatency, kBindings, kExplicitEmpty, kSampledSubmit);
DISPATCH_CASE(NativeDefaultLatency, kNative, kDefault, kSampledSubmit);
DISPATCH_CASE(NativeEmptyLatency, kNative, kExplicitEmpty, kSampledSubmit);
#undef DISPATCH_CASE

BENCHMARK_CAPTURE(BM_Dispatch, AqlBindingsDefaultSubmit, Arguments::kBindings,
                  Barriers::kDefault, Timing::kSubmit,
                  AMDF_QUEUE_COMMAND_TYPE_GPU_AQL)
    ->Arg(1)
    ->Arg(32)
    ->Arg(1000)
    ->Iterations(200)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_Dispatch, AqlBindingsEmptySubmit, Arguments::kBindings,
                  Barriers::kExplicitEmpty, Timing::kSubmit,
                  AMDF_QUEUE_COMMAND_TYPE_GPU_AQL)
    ->Arg(1)
    ->Arg(32)
    ->Arg(1000)
    ->Iterations(200)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_Dispatch, AqlBindingsDefaultCompletion,
                  Arguments::kBindings, Barriers::kDefault, Timing::kCompletion,
                  AMDF_QUEUE_COMMAND_TYPE_GPU_AQL)
    ->Arg(1)
    ->Arg(32)
    ->Arg(1000)
    ->Iterations(200)
    ->UseRealTime();

// Own-submission waits isolate the dispatch's completion boundary; active
// variants distinguish caller wake latency from execution and retirement.
#define AQL_DISPATCH_COMPLETION_CASE(name, completion, wait_flags)           \
  BENCHMARK_CAPTURE(BM_Dispatch, name, Arguments::kBindings,                 \
                    Barriers::kDefault, Timing::kCompletion,                 \
                    AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, Completion::completion, \
                    wait_flags)                                              \
      ->Arg(1)                                                               \
      ->Arg(32)                                                              \
      ->Arg(1000)                                                            \
      ->Iterations(200)                                                      \
      ->UseRealTime()
AQL_DISPATCH_COMPLETION_CASE(AqlBindingsDefaultCompletionActive, kBarrier,
                             IREE_ASYNC_WAIT_FLAG_ACTIVE);
AQL_DISPATCH_COMPLETION_CASE(AqlBindingsDefaultSubmissionCompletion,
                             kSubmission, IREE_ASYNC_WAIT_FLAG_NONE);
AQL_DISPATCH_COMPLETION_CASE(AqlBindingsDefaultSubmissionCompletionActive,
                             kSubmission, IREE_ASYNC_WAIT_FLAG_ACTIVE);
#undef AQL_DISPATCH_COMPLETION_CASE

static void BM_AqlChain(
    benchmark::State& state, bool empty, Timing timing,
    iree_async_wait_flags_t wait_flags = IREE_ASYNC_WAIT_FLAG_NONE) {
  NativeQueue queue(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, true);
  if (!queue.Initialize(state)) {
    return;
  }
  const uint32_t count = state.range(0);
  IREE_CHECK_OK(queue.SubmitChain(count, empty));
  IREE_CHECK_OK(queue.CompleteChain(wait_flags));
  const uint64_t allocations_before = queue.allocation_count();
  for (auto _ : state) {
    IREE_CHECK_OK(queue.SubmitChain(count, empty));
    if (timing == Timing::kSubmit) {
      state.PauseTiming();
    }
    IREE_CHECK_OK(queue.CompleteChain(wait_flags));
    if (timing == Timing::kCompletion) {
      state.PauseTiming();
    }
    if (!empty) {
      IREE_CHECK_OK(queue.CheckOutput(0));
    }
    state.ResumeTiming();
  }
  state.counters["host_allocations"] =
      queue.allocation_count() - allocations_before;
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK_CAPTURE(BM_AqlChain, EmptySubmit, true, Timing::kSubmit)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->Iterations(200)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AqlChain, EmptyCompletion, true, Timing::kCompletion)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->Iterations(200)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AqlChain, DispatchSubmit, false, Timing::kSubmit)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->Iterations(200)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AqlChain, DispatchCompletion, false, Timing::kCompletion)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->Iterations(200)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AqlChain, DispatchCompletionActive, false,
                  Timing::kCompletion, IREE_ASYNC_WAIT_FLAG_ACTIVE)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->Iterations(200)
    ->UseRealTime();

// Completion and destruction are excluded from recording timing. One actual
// invocation verifies the recorded program before measuring its construction.
static void BM_Record(
    benchmark::State& state, BindingMode bindings, Program program,
    amdf_queue_command_type_t command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {
  NativeQueue queue(command_type);
  if (!queue.Initialize(state)) {
    return;
  }
  const uint32_t count = state.range(0);
  iree_hal_command_buffer_t* recording = nullptr;
  IREE_CHECK_OK(queue.Record(count, bindings, program, &recording));
  IREE_CHECK_OK(queue.Execute(recording, bindings));
  IREE_CHECK_OK(queue.Complete(0));
  IREE_CHECK_OK(queue.CheckOutput(0));
  iree_hal_command_buffer_release(recording);
  for (auto _ : state) {
    IREE_CHECK_OK(queue.Record(count, bindings, program, &recording));
    state.PauseTiming();
    iree_hal_command_buffer_release(recording);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK_CAPTURE(BM_Record, Direct, BindingMode::kDirect, Program::kDispatch)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(8000)
    ->Iterations(50)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_Record, Table, BindingMode::kTable, Program::kDispatch)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(8000)
    ->Iterations(50)
    ->UseRealTime();

BENCHMARK_CAPTURE(BM_Record, TransfersDirect, BindingMode::kDirect,
                  Program::kTransfers)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(8000)
    ->Iterations(50)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_Record, TransfersTable, BindingMode::kTable,
                  Program::kTransfers)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(8000)
    ->Iterations(50)
    ->UseRealTime();

BENCHMARK_CAPTURE(BM_Record, AqlDirect, BindingMode::kDirect,
                  Program::kDispatch, AMDF_QUEUE_COMMAND_TYPE_GPU_AQL)
    ->Arg(1)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(5000)
    ->Iterations(50)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_Record, AqlTable, BindingMode::kTable, Program::kDispatch,
                  AMDF_QUEUE_COMMAND_TYPE_GPU_AQL)
    ->Arg(1)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(5000)
    ->Iterations(50)
    ->UseRealTime();

// Each iteration submits a fixed number of command buffers before completing
// through either a separate queue barrier or the submission's own signal.
// Submit timing excludes that boundary; completion timing includes it.
// Sampled rows isolate each API call at the cost of two clock reads. Bursts
// additionally separate the first call after a wait from subsequent calls, so
// idle/wake costs cannot be mistaken for steady submission costs. No row
// measures calibrated time to first workgroup.
static void BM_Execute(
    benchmark::State& state, BindingMode bindings, Timing timing,
    Program program, uint32_t burst_count, Capture capture,
    amdf_queue_command_type_t command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
    Completion completion = Completion::kBarrier,
    iree_async_wait_flags_t wait_flags = IREE_ASYNC_WAIT_FLAG_NONE) {
  NativeQueue queue(command_type);
  if (!queue.Initialize(state)) {
    return;
  }
  const uint32_t count = state.range(0);
  iree_hal_command_buffer_t* recording = nullptr;
  IREE_CHECK_OK(queue.Record(
      count, bindings, program, &recording,
      capture == Capture::kNone
          ? 0
          : IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA));
  if (capture == Capture::kActive) {
    IREE_CHECK_OK(queue.BeginCapture());
  }
  const auto complete = [&] {
    return completion == Completion::kBarrier ? queue.Complete(0, wait_flags)
                                              : queue.Wait(0, wait_flags);
  };
  for (uint32_t i = 0; i < 2; ++i) {
    for (uint32_t j = 0; j < burst_count; ++j) {
      IREE_CHECK_OK(queue.Execute(recording, bindings));
    }
    IREE_CHECK_OK(complete());
    IREE_CHECK_OK(queue.CheckOutput(0));
  }
  if (capture == Capture::kActive) {
    IREE_CHECK_OK(queue.FlushCapture());
  }
  const uint64_t allocations_before = queue.allocation_count();
  std::vector<iree_duration_t> samples;
  std::vector<iree_duration_t> first_samples;
  std::vector<iree_duration_t> following_samples;
  if (timing == Timing::kSampledSubmit) {
    samples.reserve(50 * burst_count);
    if (burst_count > 1) {
      first_samples.reserve(50);
      following_samples.reserve(50 * (burst_count - 1));
    }
  }
  for (auto _ : state) {
    for (uint32_t i = 0; i < burst_count; ++i) {
      if (timing == Timing::kSampledSubmit) {
        const iree_time_t begin = iree_time_now();
        IREE_CHECK_OK(queue.Execute(recording, bindings));
        const iree_duration_t duration = iree_time_now() - begin;
        samples.push_back(duration);
        if (burst_count > 1) {
          (i ? following_samples : first_samples).push_back(duration);
        }
      } else {
        IREE_CHECK_OK(queue.Execute(recording, bindings));
      }
    }
    if (timing != Timing::kCompletion) {
      state.PauseTiming();
    }
    IREE_CHECK_OK(complete());
    if (timing == Timing::kCompletion) {
      state.PauseTiming();
    }
    IREE_CHECK_OK(queue.CheckOutput(0));
    if (capture == Capture::kActive) {
      // Sink export is caller-driven and separate from submission/completion.
      // Flush each bounded batch so this measures retained samples, not drops.
      IREE_CHECK_OK(queue.FlushCapture());
    }
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * count * burst_count);
  state.counters["invocations_per_iteration"] = burst_count;
  state.counters["host_allocations"] =
      queue.allocation_count() - allocations_before;
  if (!samples.empty()) {
    std::sort(samples.begin(), samples.end());
    state.counters["call_p50_ns"] = samples[samples.size() / 2];
    state.counters["call_p99_ns"] = samples[samples.size() * 99 / 100];
    state.counters["call_max_ns"] = samples.back();
  }
  if (!first_samples.empty()) {
    std::sort(first_samples.begin(), first_samples.end());
    std::sort(following_samples.begin(), following_samples.end());
    state.counters["burst_first_p50_ns"] =
        first_samples[first_samples.size() / 2];
    state.counters["burst_following_p50_ns"] =
        following_samples[following_samples.size() / 2];
    state.counters["burst_following_p99_ns"] =
        following_samples[following_samples.size() * 99 / 100];
  }
  iree_hal_command_buffer_release(recording);
  if (capture == Capture::kActive) {
    IREE_CHECK_OK(queue.EndCapture());
  }
}
#define EXECUTE_CASE(name, bindings, timing, program)                        \
  BENCHMARK_CAPTURE(BM_Execute, name, BindingMode::bindings, Timing::timing, \
                    Program::program, 1, Capture::kNone)                     \
      ->Arg(1)                                                               \
      ->Arg(1000)                                                            \
      ->Arg(8000)                                                            \
      ->Iterations(50)                                                       \
      ->UseRealTime()
EXECUTE_CASE(DirectSubmit, kDirect, kSubmit, kDispatch);
EXECUTE_CASE(TableSubmit, kTable, kSubmit, kDispatch);
EXECUTE_CASE(DirectCompletion, kDirect, kCompletion, kDispatch);
EXECUTE_CASE(TableCompletion, kTable, kCompletion, kDispatch);
EXECUTE_CASE(DirectLatency, kDirect, kSampledSubmit, kDispatch);
EXECUTE_CASE(TableLatency, kTable, kSampledSubmit, kDispatch);
EXECUTE_CASE(TransfersDirectSubmit, kDirect, kSubmit, kTransfers);
EXECUTE_CASE(TransfersTableSubmit, kTable, kSubmit, kTransfers);
EXECUTE_CASE(TransfersDirectCompletion, kDirect, kCompletion, kTransfers);
EXECUTE_CASE(TransfersTableCompletion, kTable, kCompletion, kTransfers);
EXECUTE_CASE(TransfersDirectLatency, kDirect, kSampledSubmit, kTransfers);
EXECUTE_CASE(TransfersTableLatency, kTable, kSampledSubmit, kTransfers);
#undef EXECUTE_CASE

// Eight causally chained invocations exercise resident-instance reuse before
// host retirement. Fixed ten-iteration windows bound the largest GPU workload.
#define EXECUTE_BURST_CASE(name, bindings, timing)                           \
  BENCHMARK_CAPTURE(BM_Execute, name, BindingMode::bindings, Timing::timing, \
                    Program::kDispatch, 8, Capture::kNone)                   \
      ->Arg(1)                                                               \
      ->Arg(1000)                                                            \
      ->Arg(8000)                                                            \
      ->Iterations(10)                                                       \
      ->UseRealTime()
EXECUTE_BURST_CASE(DirectBurstSubmit, kDirect, kSubmit);
EXECUTE_BURST_CASE(TableBurstSubmit, kTable, kSubmit);
EXECUTE_BURST_CASE(DirectBurstCompletion, kDirect, kCompletion);
EXECUTE_BURST_CASE(TableBurstCompletion, kTable, kCompletion);
EXECUTE_BURST_CASE(DirectBurstLatency, kDirect, kSampledSubmit);
EXECUTE_BURST_CASE(TableBurstLatency, kTable, kSampledSubmit);
#undef EXECUTE_BURST_CASE

// The same numerical program and memory barriers exercise native AQL packet
// materialization. Direct references and one reused table slot distinguish
// invocation work from the number of recorded dispatches and fixups.
#define AQL_EXECUTE_CASE(name, bindings, timing, burst_count)        \
  BENCHMARK_CAPTURE(BM_Execute, Aql##name, BindingMode::bindings,    \
                    Timing::timing, Program::kDispatch, burst_count, \
                    Capture::kNone, AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) \
      ->Arg(1)                                                       \
      ->Arg(100)                                                     \
      ->Arg(1000)                                                    \
      ->Arg(5000)                                                    \
      ->Iterations(50)                                               \
      ->UseRealTime()
AQL_EXECUTE_CASE(DirectSubmit, kDirect, kSubmit, 1);
AQL_EXECUTE_CASE(TableSubmit, kTable, kSubmit, 1);
AQL_EXECUTE_CASE(DirectCompletion, kDirect, kCompletion, 1);
AQL_EXECUTE_CASE(TableCompletion, kTable, kCompletion, 1);
AQL_EXECUTE_CASE(DirectBurstSubmit, kDirect, kSubmit, 8);
AQL_EXECUTE_CASE(TableBurstSubmit, kTable, kSubmit, 8);
AQL_EXECUTE_CASE(DirectBurstCompletion, kDirect, kCompletion, 8);
AQL_EXECUTE_CASE(TableBurstCompletion, kTable, kCompletion, 8);
#undef AQL_EXECUTE_CASE

// Keep the complete public wait boundary while isolating caller sleep/wakeup
// from native preparation and retirement. ACTIVE is an application policy,
// never an implicit change to the ordinary completion rows above.
#define ACTIVE_EXECUTE_CASE(name, bindings, protocol)               \
  BENCHMARK_CAPTURE(BM_Execute, name, BindingMode::bindings,        \
                    Timing::kCompletion, Program::kDispatch, 1,     \
                    Capture::kNone, protocol, Completion::kBarrier, \
                    IREE_ASYNC_WAIT_FLAG_ACTIVE)                    \
      ->Arg(1)                                                      \
      ->Arg(100)                                                    \
      ->Arg(1000)                                                   \
      ->Arg(5000)                                                   \
      ->Iterations(50)                                              \
      ->UseRealTime()
ACTIVE_EXECUTE_CASE(DirectActiveCompletion, kDirect,
                    AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
ACTIVE_EXECUTE_CASE(TableActiveCompletion, kTable,
                    AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
ACTIVE_EXECUTE_CASE(AqlDirectActiveCompletion, kDirect,
                    AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
ACTIVE_EXECUTE_CASE(AqlTableActiveCompletion, kTable,
                    AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
#undef ACTIVE_EXECUTE_CASE

// These rows time queue_execute followed directly by its semaphore wait.
// Completion rows above retain a separate queue_barrier before waiting, which
// also measures publication of later work while the recording is still live.
#define SUBMISSION_COMPLETION_CASE(name, bindings, protocol, wait_flags) \
  BENCHMARK_CAPTURE(BM_Execute, name, BindingMode::bindings,             \
                    Timing::kCompletion, Program::kDispatch, 1,          \
                    Capture::kNone, protocol, Completion::kSubmission,   \
                    wait_flags)                                          \
      ->Arg(1)                                                           \
      ->Arg(100)                                                         \
      ->Arg(1000)                                                        \
      ->Arg(5000)                                                        \
      ->Iterations(50)                                                   \
      ->UseRealTime()
SUBMISSION_COMPLETION_CASE(DirectSubmissionCompletion, kDirect,
                           AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 0);
SUBMISSION_COMPLETION_CASE(TableSubmissionCompletion, kTable,
                           AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 0);
SUBMISSION_COMPLETION_CASE(AqlDirectSubmissionCompletion, kDirect,
                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, 0);
SUBMISSION_COMPLETION_CASE(AqlTableSubmissionCompletion, kTable,
                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, 0);
SUBMISSION_COMPLETION_CASE(DirectActiveSubmissionCompletion, kDirect,
                           AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                           IREE_ASYNC_WAIT_FLAG_ACTIVE);
SUBMISSION_COMPLETION_CASE(TableActiveSubmissionCompletion, kTable,
                           AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                           IREE_ASYNC_WAIT_FLAG_ACTIVE);
SUBMISSION_COMPLETION_CASE(AqlDirectActiveSubmissionCompletion, kDirect,
                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
                           IREE_ASYNC_WAIT_FLAG_ACTIVE);
SUBMISSION_COMPLETION_CASE(AqlTableActiveSubmissionCompletion, kTable,
                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
                           IREE_ASYNC_WAIT_FLAG_ACTIVE);
#undef SUBMISSION_COMPLETION_CASE

// Identical resident programs distinguish ordinary execution, retained but
// disabled packet regions, and active device samples. The caller upload is
// constant-sized in every row; device work scales with dispatch count.
#define PROFILE_CASE(name, timing, capture, burst_count)                   \
  BENCHMARK_CAPTURE(BM_Execute, name, BindingMode::kTable, Timing::timing, \
                    Program::kDispatch, burst_count, Capture::capture)     \
      ->Arg(1)                                                             \
      ->Arg(128)                                                           \
      ->Arg(1000)                                                          \
      ->Iterations(50)                                                     \
      ->UseRealTime()
PROFILE_CASE(ProfileBaselineSubmit, kSubmit, kNone, 1);
PROFILE_CASE(ProfileBaselineLatency, kSampledSubmit, kNone, 1);
PROFILE_CASE(ProfileBaselineCompletion, kCompletion, kNone, 1);
PROFILE_CASE(ProfileMetadataSubmit, kSubmit, kMetadata, 1);
PROFILE_CASE(ProfileMetadataLatency, kSampledSubmit, kMetadata, 1);
PROFILE_CASE(ProfileMetadataCompletion, kCompletion, kMetadata, 1);
PROFILE_CASE(ProfileCaptureSubmit, kSubmit, kActive, 1);
PROFILE_CASE(ProfileCaptureLatency, kSampledSubmit, kActive, 1);
PROFILE_CASE(ProfileCaptureCompletion, kCompletion, kActive, 1);
// Two invocations separate the first call after completion/export from a
// following call, while keeping all 2000 samples within the capture ring.
PROFILE_CASE(ProfileBaselineBurstLatency, kSampledSubmit, kNone, 2);
PROFILE_CASE(ProfileMetadataBurstLatency, kSampledSubmit, kMetadata, 2);
PROFILE_CASE(ProfileCaptureBurstLatency, kSampledSubmit, kActive, 2);
#undef PROFILE_CASE

static void BM_Transfer(benchmark::State& state,
                        iree_hal_transfer_operation_type_t type,
                        Barriers barriers, Timing timing) {
  NativeQueue queue;
  if (!queue.Initialize(state)) {
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

// A file staging window and a larger borrowed host allocation exercise the
// same upload/download retirement path. Exact byte checks stay outside timing;
// both device workers are included in process CPU time. There is no extra
// queue barrier between the measured operation and its completion wait.
static void BM_BulkTransfer(benchmark::State& state,
                            iree_hal_transfer_operation_type_t type) {
  const size_t length = state.range(0);
  NativeQueue queue;
  if (!queue.Initialize(state, 1, length)) {
    return;
  }
  std::vector<uint8_t> source(length), target(length);
  for (size_t i = 0; i < length; ++i) {
    source[i] = static_cast<uint8_t>((i * 17) ^ (i >> 9));
  }
  const auto source_bytes = iree_make_byte_span(source.data(), source.size());
  const auto target_bytes = iree_make_byte_span(target.data(), target.size());
  const bool download = type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  IREE_CHECK_OK(queue.TransferBytes(IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
                                    source_bytes));
  for (size_t i = 0; i < 2; ++i) {
    IREE_CHECK_OK(
        queue.TransferBytes(type, download ? target_bytes : source_bytes));
  }
  const uint64_t allocations_before = queue.allocation_count();
  const uint64_t publications_before = queue.completed_publications();
  for (auto _ : state) {
    IREE_CHECK_OK(
        queue.TransferBytes(type, download ? target_bytes : source_bytes));
    state.PauseTiming();
    if (!download) {
      IREE_CHECK_OK(queue.ReadBytes(target_bytes));
    }
    IREE_CHECK_OK(
        std::memcmp(source.data(), target.data(), length) == 0
            ? iree_ok_status()
            : iree_make_status(IREE_STATUS_DATA_LOSS,
                               "bulk transfer did not preserve all bytes"));
    std::fill(target.begin(), target.end(), 0xCD);
    state.ResumeTiming();
  }
  state.SetBytesProcessed(state.iterations() * length);
  state.counters["host_allocations"] =
      queue.allocation_count() - allocations_before;
  state.counters["native_publications"] =
      static_cast<double>(queue.completed_publications() -
                          publications_before) /
      state.iterations();
}

#define BULK_TRANSFER_CASE(name, kind)                       \
  BENCHMARK_CAPTURE(BM_BulkTransfer, name,                   \
                    IREE_HAL_TRANSFER_OPERATION_TYPE_##kind) \
      ->Arg(1024 * 1024)                                     \
      ->Arg(32 * 1024 * 1024)                                \
      ->Iterations(8)                                        \
      ->MeasureProcessCPUTime()                              \
      ->UseRealTime()
BULK_TRANSFER_CASE(Upload, UPLOAD);
BULK_TRANSFER_CASE(Update, UPDATE);
BULK_TRANSFER_CASE(Download, DOWNLOAD);
#undef BULK_TRANSFER_CASE

// Persistent callers share one actual queue without a caller-side mutex.
// Barrier rendezvous coordinate only benchmark epochs and stay outside each
// caller's submission interval. Overall wall time includes start/join overhead.
// Process CPU time covers callers and both device workers through final drain.
static void BM_ConcurrentDispatch(benchmark::State& state) {
  const size_t producer_count = state.range(0);
  const size_t count = state.range(1);
  NativeQueue dispatch;
  if (!dispatch.Initialize(state, producer_count)) {
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
