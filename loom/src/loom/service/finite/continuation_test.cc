// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/sync.h"
#include "loom/runtime/driver.h"
#include "loom/runtime/execution/dispatch.h"
#include "loom/runtime/parameter/residency.h"
#include "loom/runtime/program/input.h"
#include "loom/runtime/program/jit.h"
#include "loom/runtime/program/program.h"
#include "loom/runtime/service_schedule.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/runtime/testing/application.h"
#include "loom/runtime/testing/jit.h"
#include "loom/runtime/testing/service_control.h"
#include "loom/runtime/testing/service_schedule.h"
#include "loom/service/finite/batch.h"
#include "loom/service/finite/embedding/protocol.h"
#include "loom/service/finite/generator.h"
#include "loom/service/finite/service.h"

namespace {

class ContinuingSource {
 public:
  using Request = iree_vm_buffer_t;

  iree_status_t Initialize(loom_runtime_device_t* device,
                           loom_runtime_pool_t* pool, const char* source_path) {
    device_ordinal_ =
        iree_hal_device_topology_info(loom_runtime_device_handle(device))
            ->topology_index;
    IREE_RETURN_IF_ERROR(
        loom_runtime_device_create_execution(device, pool, &execution));
    const auto* clock = iree_hal_queue_family_spec(
        loom_runtime_execution_dispatch_family(execution));
    timestamp_bits_ = clock->timestamp_valid_bits;
    iree_unaligned_store_le_u64(&timing_[1], clock->timestamp_frequency_hz);
    IREE_RETURN_IF_ERROR(loom_runtime_residency_allocate(
        loom_runtime_pool_residency_cache(pool), execution, 0, nullptr,
        iree_allocator_system(), &residency_));
    directory_ = std::filesystem::path(source_path).parent_path().string();
    const auto filename =
        std::filesystem::path(source_path).filename().string();
    const iree_string_view_t providers[] = {
        iree_make_cstring_view(filename.c_str())};
    const loom_runtime_sources_t sources = {
        .base_directory = iree_make_cstring_view(directory_.c_str()),
        .count = 1,
        .values = providers,
    };
    IREE_RETURN_IF_ERROR(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &vm_types_));
    IREE_RETURN_IF_ERROR(loom_runtime_input_module_create(
        environment_, nullptr, iree_allocator_system(), &input_module_));
    IREE_RETURN_IF_ERROR(
        loom_service_finite_batch_register_types(environment_));
    const iree_vm_ref_type_table_t* hal_table = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_vm_register_types(environment_, &hal_table));
    iree_hal_vm_types_t hal_types;
    IREE_RETURN_IF_ERROR(iree_hal_vm_types_resolve(hal_table, &hal_types));
    IREE_RETURN_IF_ERROR(loom_service_finite_batch_module_create(
        environment_, &hal_types, 6, iree_allocator_system(), &batch_module_));
    iree_vm_module_t* modules[] = {input_module_, batch_module_};
    const iree_string_view_t roots[] = {IREE_SV("prepare_request"),
                                        IREE_SV("prepare_batch"),
                                        IREE_SV("collect_batch"),
                                        IREE_SV("collect_early_finish"),
                                        IREE_SV("collect_continue_published"),
                                        IREE_SV("collect_over_capacity")};
    IREE_RETURN_IF_ERROR(loom_runtime_program_allocate(
        loom_runtime_host_target_environment(loom_runtime_device_host(device)),
        environment_, &sources, providers[0], IREE_ARRAYSIZE(roots), roots,
        {modules, IREE_ARRAYSIZE(modules)}, &report_, iree_allocator_system(),
        &program_));
    for (size_t i = 0; i < IREE_ARRAYSIZE(roots); ++i) {
      IREE_RETURN_IF_ERROR(
          loom_runtime_program_lookup(program_, roots[i], &functions_[i]));
    }
    IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
        iree_make_byte_span(packet_.data(), sizeof(packet_)), {},
        iree_allocator_system(), &packet_buffer_));
    IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
        iree_make_byte_span(timing_.data(), sizeof(timing_)), {},
        iree_allocator_system(), &timing_buffer_));
    IREE_RETURN_IF_ERROR(loom_runtime_jit_create(
        loom_runtime_device_handle(device),
        loom_runtime_device_dispatch_queue(device),
        loom_runtime_device_host(device), &sources, nullptr, &report_,
        iree_allocator_system(), &jit_));
    loom_runtime::testing::loom_runtime_test_compilation_t compilation;
    compilation.jit = jit_;
    compilation.direct_dispatch = true;
    compilation.root = IREE_SV("advance");
    const loomc_config_options_t config = {
        .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
    };
    compilation.config = &config;
    compilation.Start(
        loom_runtime_host_executor(loom_runtime_device_host(device)));
    IREE_RETURN_IF_ERROR(compilation.Await(&stage_));
    dispatch_ = loom_runtime_jit_stage_dispatch(stage_);
    IREE_RETURN_IF_ERROR(
        loom_runtime_execution_reserve_workspace(execution, 160, 16));
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request(4 * 40, 16);
    return loom_runtime_execution_alloca(execution, 1, &request, &state_);
  }

  iree_status_t Retire() {
    if (!execution) {
      return iree_ok_status();
    }
    iree_hal_buffer_release(std::exchange(state_, nullptr));
    iree_status_t status = loom_runtime_execution_drain(execution);
    ReleaseBatch();
    loom_runtime_residency_free(std::exchange(residency_, nullptr));
    loom_runtime_jit_stage_free(std::exchange(stage_, nullptr));
    loom_runtime_jit_release(std::exchange(jit_, nullptr));
    loom_runtime_execution_release(std::exchange(execution, nullptr));
    iree_vm_buffer_release(std::exchange(packet_buffer_, nullptr));
    iree_vm_buffer_release(std::exchange(timing_buffer_, nullptr));
    loom_runtime_program_free(std::exchange(program_, nullptr));
    iree_vm_module_release(std::exchange(input_module_, nullptr));
    iree_vm_module_release(std::exchange(batch_module_, nullptr));
    iree_vm_environment_free(std::exchange(environment_, nullptr));
    return status;
  }

  loom_service_finite_generator_t generator() {
    static const loom_service_finite_generator_vtable_t vtable = {
        .batch_limit = [](void*) -> uint64_t { return 6; },
        .tokenizer = [](void*) -> const iree_tokenizer_t* { return nullptr; },
        .residency =
            [](void* p) {
              return static_cast<ContinuingSource*>(p)->residency_;
            },
        .execution_update =
            [](void* p) {
              return static_cast<ContinuingSource*>(p)->execution_update;
            },
        .set_observer =
            [](void* p, loom_runtime_wake_callback_t observer) {
              static_cast<ContinuingSource*>(p)->observer_ = observer;
            },
        .is_pending = [](void*) { return false; },
        .query_status = [](void*) { return iree_ok_status(); },
        .prepare_request =
            [](void* p, iree_vm_buffer_t* input,
               loom_service_prepared_request_t* work) {
              return static_cast<ContinuingSource*>(p)->PrepareRequest(input,
                                                                       work);
            },
        .release_request =
            [](void* p, loom_service_prepared_request_t* work) {
              auto* source = static_cast<ContinuingSource*>(p);
              auto* request = static_cast<Request*>(work->bytes);
              source->ReleaseRequest(request);
              *work = {};
            },
        .try_admit_request =
            [](void* p, loom_service_prepared_request_t* work, bool* acquired) {
              auto* source = static_cast<ContinuingSource*>(p);
              ++source->admission_probes;
              const auto slot = iree_unaligned_load_le_u32(
                  iree_vm_buffer_const_data(work->bytes));
              const uint32_t bit = 1u << slot;
              uint32_t occupied = source->admitted_.load();
              for (;;) {
                if (occupied & bit) {
                  *acquired = true;
                  break;
                }
                if (iree_math_count_ones_u32(occupied) >=
                    source->resident_capacity) {
                  *acquired = false;
                  if (source->release_capacity_during_probe) {
                    source->release_capacity_during_probe = false;
                    source->SetCapacity(1);
                  }
                  break;
                }
                if (source->admitted_.compare_exchange_weak(occupied,
                                                            occupied | bit)) {
                  *acquired = true;
                  break;
                }
              }
              return iree_ok_status();
            },
        .capacity_generation =
            [](void* p) {
              return static_cast<ContinuingSource*>(p)
                  ->capacity_generation_.load();
            },
        .prepare_batch =
            [](void* p, size_t count, loom_service_prepared_request_t* work,
               const size_t* rows) {
              return static_cast<ContinuingSource*>(p)->PrepareBatch(
                  count, work, rows);
            },
        .advance_batch =
            [](void* p, bool, bool* complete) {
              auto* source = static_cast<ContinuingSource*>(p);
              *complete = true;
              return loom_runtime_residency_prepare_working_set(
                  source->residency_, 68, source->OutputBytes());
            },
        .device_duration =
            [](void* p, uint64_t* duration_ns) {
              return loom_service_finite_batch_device_duration(
                  static_cast<ContinuingSource*>(p)->batch_module_,
                  duration_ns);
            },
        .submit =
            [](void* p, loom_service_finite_result_t** result,
               loom_runtime_execution_frontier_t* frontier) {
              return static_cast<ContinuingSource*>(p)->Submit(result,
                                                               frontier);
            },
        .collect =
            [](void* p, size_t count, loom_service_prepared_request_t* work,
               const loom_service_finite_result_t* result) {
              auto* source = static_cast<ContinuingSource*>(p);
              iree_vm_variant_t arguments[] = {
                  loom_service_finite_batch_begin(
                      source->batch_module_, LOOM_SERVICE_FINITE_BATCH_COLLECT,
                      count, work, result),
                  iree_vm_buffer_variant_from_ptr_borrowed(
                      &source->vm_types_, source->packet_buffer_)};
              iree_status_t status = iree_vm_invoke(
                  loom_runtime_program_invocation(source->program_),
                  source->functions_[source->collection],
                  iree_vm_variant_span_from_array(arguments),
                  iree_vm_variant_span_empty());
              status =
                  loom_service_finite_batch_end(source->batch_module_, status);
              iree_vm_variant_span_reset(
                  iree_vm_variant_span_from_array(arguments));
              if (iree_status_is_ok(status) &&
                  source->submission == Submission::kMeasureDeviceDuration) {
                const uint64_t begin =
                    iree_unaligned_load_le_u64(&source->timestamps_[0]);
                const uint64_t end =
                    iree_unaligned_load_le_u64(&source->timestamps_[1]);
                const uint64_t mask =
                    UINT64_MAX >> (64 - source->timestamp_bits_);
                iree_unaligned_store_le_u64(&source->timing_[0],
                                            (end - begin) & mask);
                iree_unaligned_store_le_u64(&source->timing_[2],
                                            source->Samples() + 1);
                memcpy(&source->timing_[3], &source->timing_[5], 16);
              }
              ++source->collections;
              return status;
            },
        .complete =
            [](void* p) {
              auto* source = static_cast<ContinuingSource*>(p);
              loom_runtime_residency_release(source->residency_);
              source->ReleaseBatch();
            },
        .release_batch =
            [](void* p) {
              auto* source = static_cast<ContinuingSource*>(p);
              source->ReleaseBatch();
            },
    };
    return {this, &vtable, device_ordinal_};
  }

  iree_status_t PrepareRequest(iree_vm_buffer_t* input,
                               loom_service_prepared_request_t* work) {
    uint32_t used = used_.load();
    uint32_t slot = 0;
    for (;;) {
      slot = 0;
      while (slot < 4 && (used & (1u << slot))) {
        ++slot;
      }
      if (slot == 4) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
      if (used_.compare_exchange_weak(used, used | (1u << slot))) {
        break;
      }
    }
    iree_vm_variant_t arguments[] = {
        iree_vm_buffer_variant_from_ptr_borrowed(&vm_types_, input),
        iree_vm_variant_from_i32(slot)};
    iree_vm_variant_t results[3] = {};
    iree_status_t status =
        iree_vm_invoke(loom_runtime_program_invocation(program_), functions_[0],
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
    Request* request = nullptr;
    int64_t units = 0, tag = 0;
    if (iree_status_is_ok(status)) {
      status = iree_vm_buffer_ptr_from_variant_borrowed(&vm_types_, results[0],
                                                        &request);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_i64_from_variant(results[1], &units);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_i64_from_variant(results[2], &tag);
    }
    if (iree_status_is_ok(status)) {
      iree_vm_buffer_retain(request);
      *work = {
          .bytes = request,
          .units = static_cast<uint64_t>(units),
          .tag = static_cast<uint64_t>(tag),
      };
      ++prepared;
    } else {
      used_.fetch_and(~(1u << slot));
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    return status;
  }

  iree_status_t PrepareBatch(size_t count,
                             loom_service_prepared_request_t* work,
                             const size_t* rows) {
    count_ = count;
    std::vector<uint32_t> ids;
    for (size_t i = 0; i < count; ++i) {
      auto* request = work[i].bytes;
      batch_[i] = request;
      iree_vm_buffer_retain(request);
      iree_const_byte_span_t data;
      IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(request, 16, 4, &data));
      ids.push_back(iree_unaligned_load_le_u32(data.data));
    }
    cohorts.push_back(std::move(ids));
    iree_vm_variant_t arguments[] = {
        loom_service_finite_batch_begin(batch_module_,
                                        LOOM_SERVICE_FINITE_BATCH_PREPARE,
                                        count, work, nullptr),
        iree_vm_buffer_variant_from_ptr_borrowed(&vm_types_, packet_buffer_),
        iree_vm_buffer_variant_from_ptr_borrowed(&vm_types_, timing_buffer_)};
    iree_vm_variant_t plan = {};
    iree_status_t status =
        iree_vm_invoke(loom_runtime_program_invocation(program_), functions_[1],
                       iree_vm_variant_span_from_array(arguments), {&plan, 1});
    status = loom_service_finite_batch_end(batch_module_, status);
    iree_vm_buffer_t* buffer = nullptr;
    if (iree_status_is_ok(status)) {
      status =
          iree_vm_buffer_ptr_from_variant_borrowed(&vm_types_, plan, &buffer);
    }
    if (iree_status_is_ok(status)) {
      status = loom_service_finite_result_create(
          buffer, 16, 16, count, rows, iree_allocator_system(), &result_);
    }
    iree_vm_variant_span_reset({&plan, 1});
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    return status;
  }

  iree_status_t Submit(loom_service_finite_result_t** result,
                       loom_runtime_execution_frontier_t* frontier) {
    const bool measure = submission == Submission::kMeasureDeviceDuration;
    if (measure &&
        (!timestamp_bits_ || !iree_unaligned_load_le_u64(&timing_[1]))) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "source requires an advertised queue clock");
    }
    loom_runtime_residency_cancel_admission(residency_);
    ++submissions;
    uint64_t value = 0;
    if (submissions == 2 && gate) {
      uint64_t one = 1;
      IREE_RETURN_IF_ERROR(
          loom_runtime_execution_barrier(execution, {1, &gate, &one}));
    }
    iree_hal_transfer_operation_t uploads[4] = {};
    size_t upload_count = 0;
    for (size_t i = 0; i < count_; ++i) {
      auto* request = batch_[i];
      if (packet_[2 + i * 4] == 0) {
        iree_const_byte_span_t data;
        IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(request, 16, 40, &data));
        uploads[upload_count++] = {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
            .upload =
                {
                    .source = data.data,
                    .target_buffer = state_,
                    .target_offset = packet_[1 + i * 4] * 40,
                    .length = 40,
                },
        };
        input_bytes += 40;
      }
    }
    if (upload_count) {
      IREE_RETURN_IF_ERROR(loom_runtime_execution_transfer(
          execution, upload_count, uploads, &value));
    }
    iree_hal_buffer_t* buffers[2] = {};
    const iree_hal_pool_reservation_request_t allocations[] = {
        iree_hal_make_pool_reservation_request(sizeof(packet_), 16),
        iree_hal_make_pool_reservation_request(OutputBytes(), 16)};
    IREE_RETURN_IF_ERROR(
        loom_runtime_execution_alloca(execution, 2, allocations, buffers));
    const iree_hal_transfer_operation_t packet = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload =
            {
                .source = packet_.data(),
                .target_buffer = buffers[0],
                .length = sizeof(packet_),
            },
    };
    iree_status_t status =
        loom_runtime_execution_transfer(execution, 1, &packet, &value);
    iree_hal_buffer_binding_t bindings[] = {{buffers[0], 0, sizeof(packet_)},
                                            {state_, 0, 160},
                                            {buffers[1], 0, 16}};
    iree_hal_buffer_ref_t scratch[3] = {};
    if (iree_status_is_ok(status) && measure) {
      status =
          loom_runtime_execution_timestamp(execution, buffers[1], 16, &value);
    }
    if (iree_status_is_ok(status)) {
      status = loom_runtime_dispatch_issue(dispatch_, execution, {3, bindings},
                                           scratch, &value);
    }
    if (iree_status_is_ok(status) && measure) {
      status =
          loom_runtime_execution_timestamp(execution, buffers[1], 24, &value);
    }
    if (iree_status_is_ok(status)) {
      status = loom_runtime_execution_dealloca(execution, buffers[0], &value);
    }
    if (iree_status_is_ok(status) &&
        submission == Submission::kRejectTimestampAfterDispatch) {
      status =
          loom_runtime_execution_timestamp(execution, buffers[1], 3, &value);
    }
    if (iree_status_is_ok(status) && measure) {
      const iree_hal_transfer_operation_t feedback = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = buffers[1],
                  .source_offset = 16,
                  .target = timestamps_.data(),
                  .length = sizeof(timestamps_),
              },
      };
      status = loom_runtime_execution_feedback(execution, 1, &feedback, &value);
    }
    if (iree_status_is_ok(status)) {
      auto* output = iree_hal_buffer_allocated_buffer(buffers[1]);
      iree_hal_buffer_retain(output);
      status = loom_runtime_residency_track_output(residency_, &output);
      if (iree_status_is_ok(status)) {
        loom_service_finite_result_bind(result_, execution, output);
        *result = std::exchange(result_, nullptr);
        *frontier = loom_service_finite_result_ready(*result);
      }
      iree_hal_buffer_release(output);
    }
    for (auto* buffer : buffers) {
      iree_hal_buffer_release(buffer);
    }
    return status;
  }

  enum Collection {
    kProgress = 2,
    kEarlyFinish,
    kContinuePublished,
    kOverCapacity,
  };
  enum class Submission {
    kExecute,
    kMeasureDeviceDuration,
    kRejectTimestampAfterDispatch,
  };
  // Source behavior: ordinary, sampled, or malformed after actual dispatch.
  Submission submission = Submission::kExecute;
  // Optional dependency used to exercise selection at cohort boundaries.
  loom_service_execution_update_t execution_update = {};
  // Authored policy selected to exercise invalid source contracts.
  Collection collection = kProgress;
  // Native timeline belongs to this source; results retain it independently.
  loom_runtime_execution_t* execution = nullptr;
  // Test-controlled dependency on epoch two, not a wall-clock delay.
  iree_hal_semaphore_t* gate = nullptr;
  // Successfully prepared source states, read after CPU synchronization.
  size_t prepared = 0;
  // Released service-owned request references.
  size_t released = 0;
  // Device epochs submitted by the owner.
  size_t submissions = 0;
  // Batched source collection calls, read after CPU synchronization.
  size_t collections = 0;
  // Original payload bytes uploaded once per accepted request.
  size_t input_bytes = 0;
  // Stable original-member identities for each numerical epoch.
  std::vector<std::vector<uint32_t>> cohorts;

  // Dependency capacity is distinct from numerical epoch width and native
  // slots.
  std::atomic<uint32_t> resident_capacity = 4;
  // Selection probes observed after the CPU borrower publishes its result.
  size_t admission_probes = 0;
  // Publishes an external capacity change between a denied probe and parking.
  bool release_capacity_during_probe = false;

  void SetCapacity(uint32_t count) {
    resident_capacity = count;
    ++capacity_generation_;
    if (observer_.wake) {
      observer_.wake(observer_.user_data);
    }
  }

  uint32_t AcceptedPosition(size_t ordinal) {
    iree_const_byte_span_t bytes;
    IREE_CHECK_OK(iree_vm_buffer_map_read(batch_[ordinal], 8, 4, &bytes));
    return iree_unaligned_load_le_u32(bytes.data);
  }

  uint64_t Samples() const { return iree_unaligned_load_le_u64(&timing_[2]); }

  uint64_t LastDurationNs() const {
    uint64_t duration_ns = 0;
    EXPECT_TRUE(iree_math_round_mul_div_u64(
        iree_unaligned_load_le_u64(&timing_[0]), UINT64_C(1000000000),
        iree_unaligned_load_le_u64(&timing_[1]), &duration_ns));
    return duration_ns;
  }

 private:
  uint64_t OutputBytes() const {
    return submission == Submission::kMeasureDeviceDuration ? 32 : 16;
  }
  // Immutable device coordinate captured with this source's execution binding.
  uint64_t device_ordinal_ = 0;
  void ReleaseRequest(Request* request) {
    if (!request) {
      return;
    }
    iree_const_byte_span_t bytes;
    IREE_CHECK_OK(iree_vm_buffer_map_read(request, 0, 4, &bytes));
    const auto slot = iree_unaligned_load_le_u32(bytes.data);
    used_.fetch_and(~(1u << slot));
    if (admitted_.fetch_and(~(1u << slot)) & (1u << slot)) {
      ++capacity_generation_;
      if (observer_.wake) {
        observer_.wake(observer_.user_data);
      }
    }
    ++released;
    iree_vm_buffer_release(request);
  }
  void ReleaseBatch() {
    loom_service_finite_result_release(std::exchange(result_, nullptr));
    for (size_t i = 0; i < count_; ++i) {
      iree_vm_buffer_release(std::exchange(batch_[i], nullptr));
    }
    count_ = 0;
  }
  // Admission identity for this producer and its accepted output.
  loom_runtime_residency_t* residency_ = nullptr;
  // Diagnostics retained through compilation and source retirement.
  loom_runtime_compilation_report_t report_ = {};
  // Device compilation context owning the retained executable.
  loom_runtime_jit_t* jit_ = nullptr;
  // Compiled advance kernel, retained across all requests.
  loom_runtime_jit_stage_t* stage_ = nullptr;
  // Borrowed direct dispatch plan owned by stage_.
  loom_runtime_dispatch_t* dispatch_ = nullptr;
  // Four persistent device slots, each containing input and accumulator.
  iree_hal_buffer_t* state_ = nullptr;
  // Prepared output metadata, transferred to the service at submit.
  loom_service_finite_result_t* result_ = nullptr;
  // Provider base directory retained while source paths are borrowed.
  std::string directory_;
  // Slot ownership shared by CPU preparation and owner-side release.
  std::atomic<uint32_t> used_ = 0;
  // Admission dependency ownership, separate from source request preparation.
  std::atomic<uint32_t> admitted_ = 0;
  // Capacity-return generation shared by CPU probes and owner-side release.
  std::atomic<uint64_t> capacity_generation_ = 0;
  // Owner wake registration enclosing every accepted CPU borrower.
  loom_runtime_wake_callback_t observer_ = {};
  // Accepted source captures retained independently of caller cancellation.
  std::array<Request*, 4> batch_ = {};
  // Source-authored control packet, reused only after accepted work retires.
  std::array<uint32_t, 17> packet_ = {};
  // Feedback destination owned by the accepted epoch through collection.
  std::array<uint64_t, 2> timestamps_ = {};
  // Cold width of the exact dispatch family's capture clock.
  uint32_t timestamp_bits_ = 0;
  // Source history: ticks, frequency, samples, previous shape, prepared shape.
  // The two shapes each contain cohort count and total source units.
  std::array<uint64_t, 7> timing_ = {};
  // Stable original membership of the accepted batch.
  size_t count_ = 0;
  // One model-owned VM instance; request continuations are ordinary data.
  loom_runtime_program_t* program_ = nullptr;
  // Canonical reference namespace outliving all source buffers.
  iree_vm_environment_t* environment_ = nullptr;
  // Native source-input validation capability.
  iree_vm_module_t* input_module_ = nullptr;
  // Reusable source access to the complete accepted cohort.
  iree_vm_module_t* batch_module_ = nullptr;
  // Canonical buffer reference identities.
  iree_vm_ref_types_t vm_types_ = {};
  // Source request construction, work planning, and completion methods.
  std::array<iree_vm_function_t, 6> functions_ = {};
  // Reusable borrowed view over the host upload packet.
  iree_vm_buffer_t* packet_buffer_ = nullptr;
  // CPU source view over collected history, never live device feedback.
  iree_vm_buffer_t* timing_buffer_ = nullptr;
};

class FiniteContinuation : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    auto options = loom_runtime_test_application_options_default();
    loom_runtime_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_runtime_test_application_acquire(
        &options, iree_allocator_system(), &application));
    group_ = loom_runtime_test_application_device_group(application);
    device_ = loom_runtime_test_application_device_at(application, 0);
    auto* pool = loom_runtime_device_group_pool_at(group_, 0);
    const char* source_path = std::getenv("LOOM_FINITE_CONTINUATION_SOURCE");
    ASSERT_NE(source_path, nullptr);
    IREE_ASSERT_OK(source_.Initialize(device_, pool, source_path));
    IREE_ASSERT_OK(loom_runtime_telemetry_stream_allocate(
        2, iree_allocator_system(), &history_));
    ASSERT_NO_FATAL_FAILURE(AllocateService(0));
    loom_runtime_device_group_set_observer(group_, {Wake, this});
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_runtime_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        0, IREE_HAL_SEMAPHORE_FLAG_NONE, &source_.gate));
  }
  void TearDown() override {
    if (source_.gate) {
      ReleaseGate();
    }
    for (auto*& call : calls_) {
      loom_service_finite_call_close(std::exchange(call, nullptr));
    }
    if (service_) {
      driver_.stopping = true;
      Drain([&] {
        return !interface_.runtime.vtable->is_pending(interface_.runtime.self);
      });
      loom_runtime_driver_set_observer(&driver_, {});
      loom_service_finite_service_free(std::exchange(service_, nullptr));
    }
    if (group_) {
      loom_runtime_device_group_set_observer(group_, {});
    }
    IREE_EXPECT_OK(source_.Retire());
    loom_runtime_telemetry_stream_free(history_);
    iree_hal_semaphore_release(source_.gate);
    iree_status_free(driver_.status);
    iree_notification_deinitialize(&notification_);
  }
  static void Wake(void* p) {
    iree_notification_post(&static_cast<FiniteContinuation*>(p)->notification_,
                           IREE_ALL_WAITERS);
  }
  void ReleaseGate() {
    auto* gate = std::exchange(source_.gate, nullptr);
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
    iree_hal_semaphore_release(gate);
  }
  void Drain(const std::function<bool()>& done) {
    IREE_ASSERT_OK(RunUntil(done));
  }
  iree_status_t RunUntil(const std::function<bool()>& done) {
    for (;;) {
      auto ticket = iree_notification_prepare_wait(&notification_);
      driver_.status = iree_status_join(
          driver_.status, loom_runtime_device_group_advance(group_));
      bool progress = false;
      loom_runtime_driver_advance(&driver_, &progress);
      progress |= schedule_.Schedule(&driver_);
      const bool pending = loom_runtime_driver_collect(&driver_, &progress);
      if (done() || !pending || progress) {
        iree_notification_cancel_wait(&notification_);
        if (done()) {
          return iree_status_clone(driver_.status);
        }
        if (!pending) {
          return iree_status_is_ok(driver_.status)
                     ? iree_make_status(
                           IREE_STATUS_FAILED_PRECONDITION,
                           "no accepted work can satisfy test predicate")
                     : iree_status_clone(driver_.status);
        }
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }
  void ProgressUntil(const std::function<bool()>& done) {
    for (;;) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      auto status = loom_runtime_device_group_advance(group_);
      bool progress = false;
      status = iree_status_join(
          status, loom_service_finite_service_progress(service_, 0, &progress));
      const bool complete = done();
      if (!complete && !progress && iree_status_is_ok(status) &&
          !interface_.runtime.vtable->is_pending(interface_.runtime.self)) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "no accepted work can satisfy progress-only predicate");
      }
      if (complete || progress || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
      if (complete) {
        break;
      }
    }
  }
  loom_runtime_service_offer_t Offer() {
    loom_runtime_service_offer_t offer = {};
    EXPECT_EQ(loom_runtime_service_read_offer(scheduling_.binding().control, 0,
                                              &offer),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    return offer;
  }
  void RemoveService() {
    ASSERT_FALSE(
        interface_.runtime.vtable->is_pending(interface_.runtime.self));
    loom_runtime_driver_set_observer(&driver_, {});
    loom_service_finite_service_free(std::exchange(service_, nullptr));
  }
  void AllocateService(iree_host_size_t variant_capacity) {
    scheduling_.Reset();
    IREE_ASSERT_OK(loom_service_finite_service_allocate(
        loom_runtime_device_host(device_), source_.generator(),
        loom_service_embedding_protocol(), IREE_SV("sum"), 4, variant_capacity,
        0, history_, scheduling_.binding(), iree_allocator_system(),
        &service_));
    interface_ = loom_service_finite_service_interface(service_);
    loom_runtime_driver_initialize(1, &interface_.runtime, &driver_);
    loom_runtime_driver_set_observer(&driver_, {Wake, this});
  }
  void Open(size_t index, const char* json,
            loom_service_finite_output_format_t format =
                LOOM_SERVICE_FINITE_OUTPUT_DEVICE) {
    iree_string_builder_t body;
    iree_string_builder_initialize(iree_allocator_system(), &body);
    IREE_ASSERT_OK(iree_string_builder_append_cstring(&body, json));
    IREE_ASSERT_OK(loom_service_finite_service_try_open(service_, &body, format,
                                                        &calls_[index]));
    iree_string_builder_deinitialize(&body);
    ASSERT_NE(calls_[index], nullptr);
  }
  std::vector<float> ReadRows(size_t index) {
    const auto* output = loom_service_finite_call_output(calls_[index]);
    EXPECT_NE(output, nullptr);
    if (!output) {
      return {};
    }
    loom_service_finite_snapshot_t* snapshot = nullptr;
    loom_runtime_execution_frontier_t frontier = {};
    IREE_EXPECT_OK(loom_service_finite_result_download(output->result,
                                                       &snapshot, &frontier));
    if (!snapshot) {
      return {};
    }
    IREE_EXPECT_OK(iree_hal_semaphore_list_wait(
        {2, frontier.semaphores, frontier.values}, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
    std::vector<float> values;
    for (size_t i = 0; i < output->row_count; ++i) {
      const auto bytes =
          loom_service_finite_snapshot_row(snapshot, output->request_index, i);
      EXPECT_EQ(bytes.data_length, sizeof(float));
      values.push_back(iree_unaligned_load_le_f32(bytes.data));
    }
    loom_service_finite_snapshot_release(snapshot);
    return values;
  }
  float Read(size_t index) {
    const auto values = ReadRows(index);
    EXPECT_EQ(values.size(), 1u);
    return values.empty() ? 0 : values[0];
  }
  // Wake publication for the fixture owner, without polling sleeps.
  iree_notification_t notification_ = {};
  // Borrowed application device group, outliving retained results.
  loom_runtime_device_group_t* group_ = nullptr;
  // Placed execution instance selected by the test profile.
  loom_runtime_device_t* device_ = nullptr;
  // Native producer sharing batched source control with registered models.
  ContinuingSource source_;
  // Owned production finite scheduler.
  loom_service_finite_service_t* service_ = nullptr;
  // Composition-owned observations outliving the engine and source borrowers.
  loom_runtime::testing::ServiceControl scheduling_{2};
  // Bounded event history deliberately smaller than one full request lifetime.
  loom_runtime_telemetry_stream_t* history_ = nullptr;
  // Borrowed scheduler, request inspection, and telemetry capabilities.
  loom_service_finite_service_interface_t interface_ = {};
  // Reusable explicit default-policy decisions.
  loom_runtime::testing::ServiceSchedule schedule_;
  // Shared owner driver for the placed service.
  loom_runtime_driver_t driver_ = {};
  // Independent caller interests, possibly outliving the producer.
  std::array<loom_service_finite_call_t*, 3> calls_ = {};
};

TEST_F(FiniteContinuation, RejectsUnrepresentableDevicePlacement) {
  ASSERT_NO_FATAL_FAILURE(RemoveService());
  auto generator = source_.generator();
  generator.device_ordinal = 64;
  loom_service_finite_service_t* rejected = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_service_finite_service_allocate(
          loom_runtime_device_host(device_), generator,
          loom_service_embedding_protocol(), IREE_SV("sum"), 4, 0, 0, history_,
          scheduling_.binding(), iree_allocator_system(), &rejected));
  EXPECT_EQ(rejected, nullptr);
  loom_service_finite_service_free(rejected);
}

TEST_F(FiniteContinuation, LatePeerCancellationAndHeldOutput) {
  Open(0, R"({"model":"sum","input":[1,2,3,4,5,6,7,8]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] { return source_.submissions == 1; }));
  EXPECT_FALSE(loom_service_finite_call_is_complete(calls_[0]));
  Open(2, R"({"model":"sum","input":[40,41,42,43]})");
  Open(1, R"({"model":"sum","input":[20,21]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] { return source_.submissions == 2; }));
  ASSERT_EQ(source_.cohorts.size(), 2u);
  EXPECT_EQ(source_.cohorts[1], (std::vector<uint32_t>{40, 20, 1}));
  EXPECT_EQ(source_.collections, 1u);
  ASSERT_NE(loom_service_finite_call_output(calls_[1]), nullptr);
  EXPECT_EQ(loom_service_finite_call_output(calls_[1])->request_index, 1u);
  EXPECT_EQ(loom_service_finite_call_output(calls_[0]), nullptr);
  loom_service_finite_call_close(std::exchange(calls_[2], nullptr));
  ASSERT_NO_FATAL_FAILURE(ReleaseGate());
  ASSERT_NO_FATAL_FAILURE(Drain([&] {
    return loom_service_finite_call_is_complete(calls_[0]) &&
           loom_service_finite_call_is_complete(calls_[1]);
  }));
  EXPECT_EQ(source_.submissions, 4u);
  EXPECT_EQ(source_.collections, 4u);
  EXPECT_EQ(source_.prepared, 3u);
  EXPECT_EQ(source_.released, 3u);
  EXPECT_EQ(source_.input_bytes, 120u);
  EXPECT_EQ(source_.cohorts[2], (std::vector<uint32_t>{1}));
  EXPECT_EQ(source_.cohorts[3], (std::vector<uint32_t>{1}));
  loom_runtime_driver_set_observer(&driver_, {});
  loom_service_finite_service_free(std::exchange(service_, nullptr));
  IREE_ASSERT_OK(source_.Retire());
  EXPECT_FLOAT_EQ(Read(0), 36.0f);
  EXPECT_FLOAT_EQ(Read(1), 41.0f);
}

TEST_F(FiniteContinuation, CapacityBlockedArrivalCannotBlockContinuingOwner) {
  source_.SetCapacity(1);
  Open(0, R"({"model":"sum","input":[1,2,3,4,5,6,7,8]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] { return source_.submissions == 1; }));
  Open(1, R"({"model":"sum","input":[20,21]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] { return source_.submissions == 2; }));
  EXPECT_EQ(source_.cohorts[1], (std::vector<uint32_t>{1}));
  ASSERT_NO_FATAL_FAILURE(ReleaseGate());
  ASSERT_NO_FATAL_FAILURE(Drain([&] {
    return loom_service_finite_call_is_complete(calls_[0]) &&
           loom_service_finite_call_is_complete(calls_[1]);
  }));
  EXPECT_FLOAT_EQ(Read(0), 36.0f);
  EXPECT_FLOAT_EQ(Read(1), 41.0f);
  EXPECT_EQ(source_.submissions, 5u);
  EXPECT_EQ(source_.prepared, 2u);
}

TEST_F(FiniteContinuation, EmptyCapacitySelectionParksUntilChange) {
  source_.SetCapacity(0);
  ASSERT_NO_FATAL_FAILURE(ReleaseGate());
  Open(0, R"({"model":"sum","input":[20,21]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] {
    return strcmp(interface_.snapshot(interface_.runtime.self).phase, "idle") ==
               0 &&
           source_.prepared == 1;
  }));
  EXPECT_EQ(source_.admission_probes, 1u);
  for (size_t i = 0; i < 16; ++i) {
    bool progress = false;
    IREE_ASSERT_OK(interface_.runtime.vtable->advance(interface_.runtime.self,
                                                      0, &progress));
    EXPECT_FALSE(progress);
  }
  EXPECT_EQ(source_.admission_probes, 1u);
  EXPECT_EQ(source_.submissions, 0u);
  source_.SetCapacity(1);
  ASSERT_NO_FATAL_FAILURE(
      Drain([&] { return loom_service_finite_call_is_complete(calls_[0]); }));
  EXPECT_FLOAT_EQ(Read(0), 41.0f);
}

TEST_F(FiniteContinuation, CapacityReleaseDuringDeniedProbeIsNotLost) {
  source_.SetCapacity(0);
  source_.release_capacity_during_probe = true;
  ASSERT_NO_FATAL_FAILURE(ReleaseGate());
  Open(0, R"({"model":"sum","input":[20,21]})");
  ASSERT_NO_FATAL_FAILURE(
      Drain([&] { return loom_service_finite_call_is_complete(calls_[0]); }));
  EXPECT_EQ(source_.admission_probes, 2u);
  EXPECT_EQ(source_.submissions, 1u);
  EXPECT_FLOAT_EQ(Read(0), 41.0f);
}

TEST_F(FiniteContinuation, MixedJsonAndDeviceWithSourceRejection) {
  ASSERT_NO_FATAL_FAILURE(ReleaseGate());
  Open(0, R"({"model":"sum","input":[1,2,3,4,5,6,7,8]})");
  Open(1, R"({"model":"sum","input":[20,21]})",
       LOOM_SERVICE_FINITE_OUTPUT_JSON);
  Open(2, R"({"model":"sum","input":[1,2,3,4,5,6,7,8,9]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] {
    return loom_service_finite_call_is_complete(calls_[0]) &&
           loom_service_finite_call_is_complete(calls_[1]) &&
           loom_service_finite_call_is_complete(calls_[2]);
  }));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[0]));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[1]));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_service_finite_call_status(calls_[2]));
  EXPECT_FLOAT_EQ(Read(0), 36.0f);
  const auto json = loom_service_finite_call_response(calls_[1]);
  EXPECT_NE(std::string(json.data, json.size).find("\"embedding\":[41]"),
            std::string::npos);
  EXPECT_EQ(interface_.telemetry.counters->input_tokens, 10u);
  EXPECT_EQ(interface_.telemetry.counters->completed, 2u);
  EXPECT_EQ(interface_.telemetry.counters->rejected, 1u);
  EXPECT_EQ(source_.submissions, 4u);
  EXPECT_EQ(source_.prepared, 2u);
  EXPECT_EQ(source_.released, 2u);
}

TEST_F(FiniteContinuation, LogicalRowsSurviveCohortsCancellationAndRemoval) {
  Open(0, R"({"model":"sum","input":[[1,2],[3,4],[5,6],[7,8]]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] { return source_.submissions == 1; }));
  EXPECT_EQ(loom_service_finite_call_output(calls_[0]), nullptr);
  Open(2, R"({"model":"sum","input":[[40,41],[42,43],[44,45]]})",
       LOOM_SERVICE_FINITE_OUTPUT_JSON);
  Open(1, R"({"model":"sum","input":[[20,21],[22,23]]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] { return source_.submissions == 2; }));
  EXPECT_EQ(source_.cohorts[1], (std::vector<uint32_t>{40, 20, 1}));
  EXPECT_EQ(loom_service_finite_call_output(calls_[0]), nullptr);
  EXPECT_EQ(loom_service_finite_call_output(calls_[1]), nullptr);
  loom_service_finite_call_close(std::exchange(calls_[2], nullptr));
  ASSERT_NO_FATAL_FAILURE(ReleaseGate());
  ASSERT_NO_FATAL_FAILURE(
      Drain([&] { return loom_service_finite_call_is_complete(calls_[1]); }));
  EXPECT_FALSE(loom_service_finite_call_is_complete(calls_[0]));
  ASSERT_NO_FATAL_FAILURE(
      Drain([&] { return loom_service_finite_call_is_complete(calls_[0]); }));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[0]));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[1]));
  EXPECT_EQ(source_.prepared, 3u);
  EXPECT_EQ(source_.released, 3u);
  const auto* long_output = loom_service_finite_call_output(calls_[0]);
  ASSERT_NE(long_output, nullptr);
  EXPECT_EQ(loom_service_finite_result_input_tokens(long_output->result,
                                                    long_output->request_index),
            8u);
  const auto* short_output = loom_service_finite_call_output(calls_[1]);
  ASSERT_NE(short_output, nullptr);
  EXPECT_EQ(loom_service_finite_result_input_tokens(
                short_output->result, short_output->request_index),
            4u);
  ASSERT_NO_FATAL_FAILURE(RemoveService());
  IREE_ASSERT_OK(source_.Retire());
  EXPECT_EQ(ReadRows(0), (std::vector<float>{3, 10, 21, 36}));
  EXPECT_EQ(ReadRows(1), (std::vector<float>{41, 86}));
}

TEST_F(FiniteContinuation, LogicalRowsEncodeInOrderWithTotalUsage) {
  ASSERT_NO_FATAL_FAILURE(ReleaseGate());
  Open(0, R"({"model":"sum","input":[[1,2],[3,4],[5,6],[7,8]]})");
  Open(1, R"({"model":"sum","input":[[20,21],[22,23],[24,25]],"dimensions":1})",
       LOOM_SERVICE_FINITE_OUTPUT_JSON);
  Open(2, R"({"model":"sum","input":[[1,2,3],[4,5]]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] {
    return loom_service_finite_call_is_complete(calls_[0]) &&
           loom_service_finite_call_is_complete(calls_[1]) &&
           loom_service_finite_call_is_complete(calls_[2]);
  }));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[0]));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[1]));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_service_finite_call_status(calls_[2]));
  EXPECT_EQ(ReadRows(0), (std::vector<float>{3, 10, 21, 36}));
  const auto response = loom_service_finite_call_response(calls_[1]);
  EXPECT_EQ(std::string(response.data, response.size),
            R"({"object":"list","model":"sum","data":[)"
            R"({"object":"embedding","index":0,"embedding":[41]},)"
            R"({"object":"embedding","index":1,"embedding":[86]},)"
            R"({"object":"embedding","index":2,"embedding":[135]}],)"
            R"("usage":{"prompt_tokens":6,"total_tokens":6}})");
  EXPECT_EQ(interface_.telemetry.counters->input_tokens, 14u);
  EXPECT_EQ(interface_.telemetry.counters->completed, 2u);
  EXPECT_EQ(interface_.telemetry.counters->rejected, 1u);
}

TEST_F(FiniteContinuation, FailedReadinessDoesNotCommitPlannedProgress) {
  ASSERT_NO_FATAL_FAILURE(RemoveService());
  // This source has only its original execution. Management can select it,
  // but replacement compilation is an unsupported external dependency.
  source_.execution_update = {
      .compile =
          [](void*, const loom_runtime_sources_t*,
             loom_runtime_variant_completion_fn_t, void*) {
            return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
          },
      .select = [](void*, void*) { return iree_ok_status(); },
      .dispose = [](void*) {},
  };
  ASSERT_NO_FATAL_FAILURE(AllocateService(1));
  Open(0, R"({"model":"sum","input":[[1,2],[3,4],[5,6],[7,8]]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] { return source_.submissions == 1; }));
  Open(1, R"({"model":"sum","input":[20,21]})");
  ASSERT_NO_FATAL_FAILURE(Drain([&] { return source_.submissions == 2; }));
  ASSERT_EQ(source_.cohorts[1], (std::vector<uint32_t>{20, 1}));
  EXPECT_EQ(source_.AcceptedPosition(0), 0u);
  EXPECT_EQ(source_.AcceptedPosition(1), 2u);
  const auto* control = scheduling_.binding().control;
  loom_runtime_service_operation_t accepted = {};
  ASSERT_EQ(loom_runtime_service_read_operation(control, 0, &accepted),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  ASSERT_NE(accepted.key.serial, 0u);
  const auto configuration = interface_.reconfiguration;
  const auto original = configuration.selected(configuration.self);
  bool selection_complete = false;
  IREE_ASSERT_OK(configuration.select(
      configuration.self, original, original,
      [](void* self, iree_status_t status, loom_runtime_variant_id_t) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
        *static_cast<bool*>(self) = true;
      },
      &selection_complete));
  auto* gate = std::exchange(source_.gate, nullptr);
  iree_hal_semaphore_fail(gate, iree_status_from_code(IREE_STATUS_ABORTED));
  iree_hal_semaphore_release(gate);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED, RunUntil([&] {
        return loom_service_finite_call_is_complete(calls_[0]) &&
               loom_service_finite_call_is_complete(calls_[1]) &&
               !interface_.runtime.vtable->is_pending(interface_.runtime.self);
      }));
  EXPECT_TRUE(selection_complete);
  loom_runtime_service_observation_t observation = {};
  ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.pending_operation_count, 1u);
  EXPECT_EQ(observation.accepted_devices, 1u);
  loom_runtime_service_operation_t failed = {};
  ASSERT_EQ(loom_runtime_service_read_operation(control, 0, &failed),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(failed.key.serial, accepted.key.serial);
  EXPECT_EQ(failed.use_count, 1u);
  EXPECT_EQ(source_.collections, 1u);
  EXPECT_EQ(source_.AcceptedPosition(0), 0u);
  EXPECT_EQ(source_.AcceptedPosition(1), 2u);
  for (size_t i = 0; i < 2; ++i) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                          loom_service_finite_call_status(calls_[i]));
  }
  loom_runtime_driver_set_observer(&driver_, {});
  loom_service_finite_service_free(std::exchange(service_, nullptr));
  // The ready engine can disappear before the source releases failed captures.
  loom_runtime_service_use_t use = {};
  EXPECT_EQ(loom_runtime_service_read_operation_use(
                control, failed.key, failed.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, source_.Retire());
  EXPECT_EQ(source_.released, 2u);
}

TEST_F(FiniteContinuation, DecisionPrefixResumesWithoutReplayingAcceptedWork) {
  ASSERT_NO_FATAL_FAILURE(ReleaseGate());
  ASSERT_NO_FATAL_FAILURE(Open(0, R"({"model":"sum","input":[1,2]})"));
  std::array<uint64_t, 35> storage;
  loom_runtime_service_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_runtime_service_plan_initialize(
      3, iree_make_byte_span(storage.data(), sizeof(storage)), &plan));
  const uint64_t incarnation =
      loom_runtime_service_control_layout(scheduling_.binding().control)
          .incarnation;
  loom_runtime_service_decision_t decision = {
      .service_ordinal = 0,
      .service_incarnation = incarnation + 1,
      .offer = Offer().key,
  };
  ASSERT_TRUE(loom_runtime_service_plan_append(plan, &decision));
  decision.service_incarnation = incarnation;
  ASSERT_TRUE(loom_runtime_service_plan_append(plan, &decision));
  ASSERT_TRUE(loom_runtime_service_plan_append(plan, &decision));

  loom_runtime_service_default_policy_t policy_state = {};
  const auto policy = loom_runtime_service_default_policy(&policy_state);
  loom_runtime_service_schedule_context_t context = {.now_ns = iree_time_now()};
  loom_runtime_service_schedule_budget_t budget = {};
  bool progress = false;
  loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                               &progress);
  EXPECT_FALSE(progress);  // Withheld credit does not poll the pending suffix.
  EXPECT_EQ(context.wake_time_ns, IREE_TIME_INFINITE_FUTURE);
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).applied_count, 0u);
  EXPECT_EQ(source_.prepared, 0u);

  budget.decisions = 1;
  loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                               &progress);
  EXPECT_TRUE(progress);  // Consumed credit yields the retained suffix.
  EXPECT_EQ(context.wake_time_ns, context.now_ns);
  EXPECT_EQ(budget.decisions, 0u);
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).applied_count, 1u);
  EXPECT_FALSE(loom_runtime_service_plan_outcome(plan, 0).accepted);
  EXPECT_EQ(source_.prepared, 0u);

  budget.decisions = 1;
  loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                               &progress);
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).applied_count, 2u);
  const auto preparation = loom_runtime_service_plan_outcome(plan, 1);
  ASSERT_TRUE(preparation.accepted);
  EXPECT_NE(preparation.operation.serial, 0u);
  ASSERT_NO_FATAL_FAILURE(ProgressUntil(
      [&] { return Offer().work == LOOM_RUNTIME_SERVICE_WORK_EXECUTION; }));
  EXPECT_EQ(source_.prepared, 1u);
  EXPECT_EQ(source_.submissions, 0u);
  const auto execution = Offer();

  // Accepted preparation completed at zero new-work budget. The old duplicate
  // now spends its own attempt without starting the newly offered execution.
  budget.decisions = 1;
  loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                               &progress);
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).applied_count, 3u);
  EXPECT_FALSE(loom_runtime_service_plan_outcome(plan, 2).accepted);
  EXPECT_EQ(loom_runtime_service_plan_outcome(plan, 1).operation.serial,
            preparation.operation.serial);
  EXPECT_EQ(Offer().key.revision, execution.key.revision);
  EXPECT_EQ(source_.submissions, 0u);

  // An explicitly empty result ends evaluation even with remaining capacity.
  const loom_runtime_service_policy_t defer = {
      .select =
          [](void*, const loom_runtime_driver_t*,
             loom_runtime_service_schedule_context_t*,
             loom_runtime_service_plan_t* output) {
            loom_runtime_service_plan_reset(output);
          },
  };
  budget = {.evaluations = 4, .decisions = 4};
  loom_runtime_driver_schedule(&driver_, defer, plan, &budget, &context,
                               &progress);
  EXPECT_FALSE(progress);
  EXPECT_EQ(budget.evaluations, 3u);
  EXPECT_EQ(budget.decisions, 4u);
  EXPECT_EQ(context.wake_time_ns, IREE_TIME_INFINITE_FUTURE);
  EXPECT_EQ(source_.submissions, 0u);

  budget = {.evaluations = 1, .decisions = 1};
  loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                               &progress);
  EXPECT_EQ(budget.evaluations, 0u);
  EXPECT_EQ(budget.decisions, 0u);
  ASSERT_TRUE(loom_runtime_service_plan_outcome(plan, 0).accepted);
  ASSERT_NO_FATAL_FAILURE(ProgressUntil(
      [&] { return loom_service_finite_call_is_complete(calls_[0]); }));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[0]));
  EXPECT_FLOAT_EQ(Read(0), 3.0f);
  EXPECT_EQ(source_.prepared, 1u);
  EXPECT_EQ(source_.submissions, 1u);
}

TEST_F(FiniteContinuation, PolicyDeadlineWakesWithoutAnotherRequest) {
  ASSERT_NO_FATAL_FAILURE(ReleaseGate());
  ASSERT_NO_FATAL_FAILURE(Open(0, R"({"model":"sum","input":[20,21]})"));
  std::array<uint64_t, 17> storage;
  loom_runtime_service_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_runtime_service_plan_initialize(
      1, iree_make_byte_span(storage.data(), sizeof(storage)), &plan));
  struct Policy {
    // Native traversal state after the intentional first-request deferral.
    loom_runtime_service_default_policy_t traversal = {};
    // Batching deadline chosen on the first evaluation, never extended.
    iree_time_t deadline_ns = 0;
    // First invocation permitted to emit work.
    iree_time_t granted_ns = 0;
    // Number of empty decisions before the deadline.
    size_t deferrals = 0;
  } state;
  const loom_runtime_service_policy_t policy = {
      .self = &state,
      .select =
          [](void* self, const loom_runtime_driver_t* driver,
             loom_runtime_service_schedule_context_t* context,
             loom_runtime_service_plan_t* output) {
            auto& state = *static_cast<Policy*>(self);
            if (!state.deadline_ns) {
              state.deadline_ns = context->now_ns + iree_make_duration_ms(1);
            }
            if (context->now_ns < state.deadline_ns) {
              loom_runtime_service_plan_reset(output);
              context->wake_time_ns = state.deadline_ns;
              ++state.deferrals;
              return;
            }
            if (!state.granted_ns) {
              state.granted_ns = context->now_ns;
            }
            const auto select =
                loom_runtime_service_default_policy(&state.traversal);
            select.select(select.self, driver, context, output);
          },
  };

  // The embedding owner integrates the returned policy deadline with the same
  // notification that carries real service/resource readiness. No extra
  // request, polling timeout, timer thread, or automatic start supplies a wake.
  bool complete = false;
  while (!complete) {
    const auto ticket = iree_notification_prepare_wait(&notification_);
    auto status = loom_runtime_device_group_advance(group_);
    bool progress = false;
    status = iree_status_join(
        status, loom_service_finite_service_progress(service_, 0, &progress));
    loom_runtime_service_schedule_context_t context = {
        .now_ns = iree_time_now(),
    };
    loom_runtime_service_schedule_budget_t budget = {2, 2};
    bool scheduled = false;
    loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                                 &scheduled);
    status = iree_status_join(status, iree_status_clone(driver_.status));
    complete = loom_service_finite_call_is_complete(calls_[0]);
    if (complete || progress || scheduled || !iree_status_is_ok(status)) {
      iree_notification_cancel_wait(&notification_);
    } else {
      iree_notification_commit_wait(&notification_, ticket, IREE_DURATION_ZERO,
                                    context.wake_time_ns);
    }
    IREE_ASSERT_OK(status);
  }
  EXPECT_GT(state.deferrals, 0u);
  EXPECT_GE(state.granted_ns, state.deadline_ns);
  EXPECT_EQ(source_.prepared, 1u);
  EXPECT_EQ(source_.submissions, 1u);
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[0]));
  EXPECT_FLOAT_EQ(Read(0), 41.0f);
}

TEST_F(FiniteContinuation, FailedStartRetainsItsAcceptedOperation) {
  source_.submission =
      ContinuingSource::Submission::kRejectTimestampAfterDispatch;
  Open(0, R"({"model":"sum","input":[1,2,3,4]})");
  const auto* control = scheduling_.binding().control;
  loom_runtime_service_offer_t offer = {};
  ASSERT_EQ(loom_runtime_service_read_offer(control, 0, &offer),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  loom_runtime_service_operation_key_t accepted;
  IREE_ASSERT_OK(loom_service_finite_service_try_schedule(service_, offer.key,
                                                          0, 0, &accepted));
  ASSERT_NE(accepted.serial, 0u);
  ASSERT_NO_FATAL_FAILURE(ProgressUntil(
      [&] { return Offer().work == LOOM_RUNTIME_SERVICE_WORK_EXECUTION; }));
  offer = Offer();
  const auto before = loom_runtime_execution_frontier(source_.execution);
  std::array<uint64_t, 17> storage;
  loom_runtime_service_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_runtime_service_plan_initialize(
      1, iree_make_byte_span(storage.data(), sizeof(storage)), &plan));
  const loom_runtime_service_decision_t decision = {
      .service_ordinal = 0,
      .service_incarnation =
          loom_runtime_service_control_layout(control).incarnation,
      .offer = offer.key,
  };
  ASSERT_TRUE(loom_runtime_service_plan_append(plan, &decision));
  uint64_t decisions = 1;
  bool progress = false;
  loom_runtime_driver_apply_plan(&driver_, plan, &decisions, &progress);
  EXPECT_TRUE(progress);
  EXPECT_EQ(decisions, 0u);
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).applied_count, 1u);
  const auto outcome = loom_runtime_service_plan_outcome(plan, 0);
  EXPECT_TRUE(outcome.accepted);
  EXPECT_EQ(outcome.status_code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_TRUE(driver_.stopping);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_status_clone(driver_.status));
  accepted = outcome.operation;
  EXPECT_NE(accepted.serial, 0u);
  const auto after = loom_runtime_execution_frontier(source_.execution);
  EXPECT_GT(after.values[0], before.values[0]);
  loom_runtime_service_operation_t operation = {};
  ASSERT_EQ(loom_runtime_service_read_operation(control, 0, &operation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(operation.key.serial, accepted.serial);
  EXPECT_EQ(operation.work, LOOM_RUNTIME_SERVICE_WORK_EXECUTION);
  EXPECT_EQ(operation.use_count, 1u);
  ASSERT_TRUE(loom_service_finite_call_is_complete(calls_[0]));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_service_finite_call_status(calls_[0]));
  ASSERT_NO_FATAL_FAILURE(RemoveService());
  ASSERT_EQ(loom_runtime_service_read_operation(control, 0, &operation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(operation.key.serial, accepted.serial);
  IREE_ASSERT_OK(source_.Retire());
  EXPECT_EQ(source_.released, 1u);
}

TEST_F(FiniteContinuation, DeviceEstimatesFollowCollectedSourceHistory) {
  source_.submission = ContinuingSource::Submission::kMeasureDeviceDuration;
  Open(0, R"({"model":"sum","input":[1,2,3,4,5]})");
  const auto* control = scheduling_.binding().control;
  uint64_t previous_duration_ns = 0;
  for (size_t epoch = 0; epoch < 3; ++epoch) {
    const auto preparation = Offer();
    ASSERT_EQ(preparation.work, LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
    loom_runtime_service_operation_key_t accepted;
    IREE_ASSERT_OK(loom_service_finite_service_try_schedule(
        service_, preparation.key, 0, 0, &accepted));
    ASSERT_NE(accepted.serial, 0u);
    ASSERT_NO_FATAL_FAILURE(ProgressUntil(
        [&] { return Offer().work == LOOM_RUNTIME_SERVICE_WORK_EXECUTION; }));
    const auto execution = Offer();
    if (epoch) {
      // Earlier epochs ran with no history reader. A late reader misses real
      // service events, but source estimates and current control stay intact.
      uint64_t cursor = 0;
      loom_runtime_telemetry_record_t record;
      EXPECT_EQ(loom_runtime_telemetry_stream_read(history_, &cursor, &record),
                LOOM_RUNTIME_TELEMETRY_READ_GAP);
      EXPECT_GT(cursor, 0u);
    }
    loom_runtime_service_choice_t choice = {};
    ASSERT_EQ(
        loom_runtime_service_read_choice(control, execution.key, 0, &choice),
        LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(choice.estimates, 0u);  // Device time is not whole-unit time.
    loom_runtime_service_use_t use = {};
    ASSERT_EQ(loom_runtime_service_read_choice_use(control, execution.key, 0,
                                                   choice.publication, 0, &use),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(source_.Samples(), epoch);
    // The first cohort has no history; the third has only one remaining unit.
    EXPECT_EQ(use.estimates,
              epoch == 1 ? LOOM_RUNTIME_SERVICE_ESTIMATE_DURATION : 0u);
    if (epoch == 1) {
      EXPECT_EQ(use.duration_ns, previous_duration_ns);
    }
    IREE_ASSERT_OK(loom_service_finite_service_try_schedule(
        service_, execution.key, 0, 0, &accepted));
    ASSERT_NE(accepted.serial, 0u);
    if (epoch == 1) {
      // The external dependency prevents the next sample from being complete.
      // Its forecast and accepted resource identity remain independently
      // usable.
      bool progress = false;
      IREE_ASSERT_OK(
          loom_service_finite_service_progress(service_, 0, &progress));
      loom_runtime_service_operation_t operation = {};
      ASSERT_EQ(loom_runtime_service_read_operation(control, accepted.ordinal,
                                                    &operation),
                LOOM_RUNTIME_SERVICE_READ_COMPLETE);
      EXPECT_EQ(operation.waits, LOOM_RUNTIME_SERVICE_WAIT_DEVICE);
      EXPECT_EQ(operation.estimates, 0u);
      IREE_ASSERT_OK(
          loom_service_finite_service_progress(service_, 0, &progress));
      EXPECT_FALSE(progress);
      // An unchanged accepted wait keeps its retained use join readable.
      ASSERT_EQ(loom_runtime_service_read_operation_use(
                    control, operation.key, operation.publication, 0, &use),
                LOOM_RUNTIME_SERVICE_READ_COMPLETE);
      EXPECT_EQ(use.estimates, LOOM_RUNTIME_SERVICE_ESTIMATE_DURATION);
      EXPECT_EQ(use.duration_ns, previous_duration_ns);
      EXPECT_EQ(source_.Samples(), 1u);
      ReleaseGate();
    }
    ASSERT_NO_FATAL_FAILURE(ProgressUntil([&] {
      return epoch == 2 ? loom_service_finite_call_is_complete(calls_[0])
                        : Offer().work == LOOM_RUNTIME_SERVICE_WORK_PREPARATION;
    }));
    EXPECT_EQ(source_.Samples(), epoch + 1);
    previous_duration_ns = source_.LastDurationNs();
  }
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[0]));
  EXPECT_FLOAT_EQ(Read(0), 15.0f);
}

TEST_F(FiniteContinuation, IndependentSamplesSurviveCancellationAndRetirement) {
  loom_runtime::testing::ServiceControl peer_control;
  loom_service_finite_service_t* peer_service = nullptr;
  std::array<loom_runtime_service_t, 2> services = {interface_.runtime, {}};
  auto retire_peer = [&](ContinuingSource* peer) {
    if (source_.gate) {
      ReleaseGate();
    }
    if (peer->gate) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(peer->gate, 1, nullptr));
      iree_hal_semaphore_release(std::exchange(peer->gate, nullptr));
    }
    loom_service_finite_call_close(std::exchange(calls_[1], nullptr));
    driver_.stopping = true;
    Drain([&] {
      for (const auto& service : services) {
        if (service.vtable && service.vtable->is_pending(service.self)) {
          return false;
        }
      }
      return true;
    });
    if (peer_service) {
      services[1].vtable->set_observer(services[1].self, {});
      loom_service_finite_service_free(peer_service);
    }
    IREE_EXPECT_OK(peer->Retire());
    driver_.services = &interface_.runtime;
    driver_.capacity = 1;
    delete peer;
  };
  std::unique_ptr<ContinuingSource, decltype(retire_peer)> peer(
      new ContinuingSource, retire_peer);
  IREE_ASSERT_OK(
      peer->Initialize(device_, loom_runtime_device_group_pool_at(group_, 0),
                       std::getenv("LOOM_FINITE_CONTINUATION_SOURCE")));
  IREE_ASSERT_OK(loom_service_finite_service_allocate(
      loom_runtime_device_host(device_), peer->generator(),
      loom_service_embedding_protocol(), IREE_SV("peer"), 4, 0, 0, nullptr,
      peer_control.binding(), iree_allocator_system(), &peer_service));
  services[1] = loom_service_finite_service_interface(peer_service).runtime;
  driver_.services = services.data();
  driver_.capacity = services.size();
  loom_runtime_driver_set_observer(&driver_, {Wake, this});
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_runtime_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      0, IREE_HAL_SEMAPHORE_FLAG_NONE, &peer->gate));
  source_.submission = ContinuingSource::Submission::kMeasureDeviceDuration;
  peer->submission = ContinuingSource::Submission::kMeasureDeviceDuration;
  Open(0, R"({"model":"sum","input":[1,2,3,4,5]})");
  iree_string_builder_t body;
  iree_string_builder_initialize(iree_allocator_system(), &body);
  IREE_ASSERT_OK(iree_string_builder_append_cstring(
      &body, R"({"model":"peer","input":[10,20,30,40,50]})"));
  auto status = loom_service_finite_service_try_open(
      peer_service, &body, LOOM_SERVICE_FINITE_OUTPUT_DEVICE, &calls_[1]);
  iree_string_builder_deinitialize(&body);
  IREE_ASSERT_OK(status);
  ASSERT_NE(calls_[1], nullptr);
  ASSERT_NO_FATAL_FAILURE(Drain(
      [&] { return source_.submissions == 2 && peer->submissions == 2; }));

  const auto* control = scheduling_.binding().control;
  loom_runtime_service_operation_t operation = {};
  ASSERT_EQ(loom_runtime_service_read_operation(control, 0, &operation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  ASSERT_NE(operation.key.serial, 0u);
  EXPECT_EQ(operation.waits, LOOM_RUNTIME_SERVICE_WAIT_DEVICE);
  const uint64_t duration_ns = source_.LastDurationNs();
  EXPECT_EQ(source_.Samples(), 1u);
  EXPECT_EQ(peer->Samples(), 1u);

  // Cancellation releases caller interest, not the accepted feedback target.
  // The other source has a separate timeline and can finish while it is held.
  loom_service_finite_call_close(std::exchange(calls_[0], nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(peer->gate, 1, nullptr));
  iree_hal_semaphore_release(std::exchange(peer->gate, nullptr));
  ASSERT_NO_FATAL_FAILURE(Drain([&] {
    return loom_service_finite_call_is_complete(calls_[1]) &&
           !services[1].vtable->is_pending(services[1].self);
  }));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[1]));
  EXPECT_EQ(peer->Samples(), 3u);
  services[1].vtable->set_observer(services[1].self, {});
  services[1] = {};
  loom_service_finite_service_free(std::exchange(peer_service, nullptr));
  IREE_ASSERT_OK(peer->Retire());
  EXPECT_FLOAT_EQ(Read(1), 150.0f);
  loom_runtime_service_observation_t observation = {};
  ASSERT_EQ(loom_runtime_service_read_observation(
                peer_control.binding().control, &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.owned_request_count, 0u);
  EXPECT_EQ(observation.pending_operation_count, 0u);
  loom_runtime_service_use_t use = {};
  ASSERT_EQ(loom_runtime_service_read_operation_use(
                control, operation.key, operation.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(use.estimates, LOOM_RUNTIME_SERVICE_ESTIMATE_DURATION);
  EXPECT_EQ(use.duration_ns, duration_ns);
  EXPECT_EQ(source_.Samples(), 1u);
  EXPECT_EQ(source_.LastDurationNs(), duration_ns);

  ReleaseGate();
  ASSERT_NO_FATAL_FAILURE(Drain([&] {
    return !interface_.runtime.vtable->is_pending(interface_.runtime.self);
  }));
  EXPECT_EQ(source_.Samples(), 2u);
  EXPECT_EQ(loom_runtime_service_read_operation_use(
                control, operation.key, operation.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_CHANGED);
}

TEST_F(FiniteContinuation, TerminalOfferRequiresPublishedResult) {
  source_.collection = ContinuingSource::kEarlyFinish;
  Open(0, R"({"model":"sum","input":[1,2,3,4]})");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT, RunUntil([&] {
        return loom_service_finite_call_is_complete(calls_[0]);
      }));
  EXPECT_EQ(source_.collections, 1u);
  EXPECT_EQ(loom_service_finite_call_output(calls_[0]), nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_service_finite_call_status(calls_[0]));
  ASSERT_NO_FATAL_FAILURE(RemoveService());
}

TEST_F(FiniteContinuation, PublishedResultRequiresTerminalOffer) {
  source_.collection = ContinuingSource::kContinuePublished;
  Open(0, R"({"model":"sum","input":[1,2]})");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT, RunUntil([&] {
        return loom_service_finite_call_is_complete(calls_[0]);
      }));
  EXPECT_EQ(source_.collections, 1u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_service_finite_call_status(calls_[0]));
  ASSERT_NO_FATAL_FAILURE(RemoveService());
}

TEST_F(FiniteContinuation, NextOfferRespectsDeclaredCapacity) {
  source_.collection = ContinuingSource::kOverCapacity;
  Open(0, R"({"model":"sum","input":[1,2,3,4]})");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE, RunUntil([&] {
        return loom_service_finite_call_is_complete(calls_[0]);
      }));
  EXPECT_EQ(source_.collections, 1u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_service_finite_call_status(calls_[0]));
  ASSERT_NO_FATAL_FAILURE(RemoveService());
}
}  // namespace
