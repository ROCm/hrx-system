// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL ASAN allocation lifetime CTS coverage.

#include <cstdint>
#include <string>
#include <thread>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/cts/sanitizer/sanitizer_test_util.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace iree::hal::cts {

namespace {

enum AsanAllocationHookSelector : uint32_t {
  kAsanAllocationHookLoad1 = 1,
  kAsanAllocationHookLoad2 = 2,
  kAsanAllocationHookReportLoadN = 3,
  kAsanAllocationHookPoisonRegion = 4,
  kAsanAllocationHookUnpoisonRegion = 5,
};

enum AsanReportExpectationFlagBits : uint32_t {
  kAsanReportExpectationFlagNone = 0u,
  kAsanReportExpectationFlagShadowPoisoned = 1u << 0,
};

using AsanReportExpectationFlags = uint32_t;

constexpr iree_device_size_t kAsanAllocationBufferLength = 64;
constexpr uint32_t kAsanAllocationSentinelLength = 37;
constexpr uint32_t kAsanAllocationBindingEntrypoint = 0;
constexpr uint32_t kAsanAllocationRawAddressEntrypoint = 1;

static iree_hal_buffer_params_t AsanDeviceBufferParams() {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  return params;
}

static iree_hal_buffer_params_t AsanQueueAllocaBufferParams() {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  return params;
}

static iree_status_t DispatchAsanAllocationSelector(
    iree_hal_device_t* device, iree_hal_queue_t* queue,
    iree_hal_executable_t* executable, iree_hal_buffer_ref_list_t bindings,
    uint32_t selector, uint32_t access_length, int32_t address_adjustment) {
  const uint32_t constant_data[] = {
      selector,
      access_length,
      static_cast<uint32_t>(address_adjustment),
  };
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device, {0}, {1});
  IREE_RETURN_IF_ERROR(iree_hal_queue_dispatch(
      queue, empty_wait, dispatch_signal, executable,
      iree_hal_executable_function_from_index(kAsanAllocationBindingEntrypoint),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  return iree_hal_semaphore_list_wait(dispatch_signal, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t DispatchAsanAllocationRawAddress(
    iree_hal_device_t* device, iree_hal_queue_t* queue,
    iree_hal_executable_t* executable, uint64_t address, uint32_t selector,
    uint32_t access_length) {
  const uint32_t constant_data[] = {
      static_cast<uint32_t>(address),
      static_cast<uint32_t>(address >> 32),
      selector,
      access_length,
  };
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device, {0}, {1});
  iree_hal_buffer_ref_list_t empty_bindings = iree_hal_buffer_ref_list_empty();
  IREE_RETURN_IF_ERROR(iree_hal_queue_dispatch(
      queue, empty_wait, dispatch_signal, executable,
      iree_hal_executable_function_from_index(
          kAsanAllocationRawAddressEntrypoint),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, empty_bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  return iree_hal_semaphore_list_wait(dispatch_signal, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t ExportDeviceAddress(iree_hal_buffer_t* buffer,
                                         uint64_t* out_address) {
  *out_address = 0;
  iree_hal_external_buffer_t external_buffer = {};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_export(
      buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  *out_address = external_buffer.handle.device_allocation.ptr;
  return iree_ok_status();
}

}  // namespace

class AsanAllocationTest : public ::testing::TestWithParam<BackendInfo> {
 protected:
  void SetUp() override {
    std::string host_incompatibility_reason;
    if (!IsBackendHostCompatible(GetParam(), &host_incompatibility_reason)) {
      GTEST_SKIP() << "Backend '" << GetParam().name
                   << "' is not compatible with this host: "
                   << host_incompatibility_reason;
    }

    iree_status_t status = asan_device_.Initialize(GetParam(), "asan");
    if (iree_status_is_unavailable(status)) {
      iree::Status unavailable_status(std::move(status));
      GTEST_SKIP() << unavailable_status.ToString();
    }
    IREE_ASSERT_OK(status);

    const iree_hal_device_sanitizer_spec_t* sanitizer =
        iree_hal_device_spec_sanitizer(iree_hal_device_spec(device()));
    if (!iree_all_bits_set(sanitizer->flags,
                           IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN)) {
      GTEST_SKIP() << "Backend '" << GetParam().name
                   << "' does not advertise HAL ASAN";
    }
    if (!iree_hal_asan_pool_options_is_enabled(&sanitizer->asan.pool_options)) {
      GTEST_SKIP() << "Backend '" << GetParam().name
                   << "' was created without HAL ASAN enabled";
    }

    iree_const_byte_span_t executable_data =
        GetParam().executable_data
            ? GetParam().executable_data(
                  iree_make_cstring_view("asan_allocation_test.bin"))
            : iree_const_byte_span_empty();
    if (iree_const_byte_span_is_empty(executable_data)) {
      GTEST_SKIP() << "Backend '" << GetParam().name
                   << "' has no ASAN allocation CTS executable data";
    }

    iree_hal_executable_target_selection_result_t result;
    IREE_ASSERT_OK(SelectBackendExecutableTarget(
        device(), iree_hal_queue_family(queue()), GetParam(), &result));
    if (result.outcome ==
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
      GTEST_SKIP() << "Executable target '"
                   << GetParam().executable_target_family << ":"
                   << GetParam().executable_target_key
                   << "' is not advertised by CTS backend/device '"
                   << GetParam().name << "'";
    }
    ASSERT_EQ(result.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);

    iree_hal_executable_load_params_t load_params;
    iree_hal_executable_load_params_initialize(&load_params);
    load_params.executable_data = executable_data;
    IREE_ASSERT_OK(iree_hal_executable_load(iree_hal_queue_family(queue()),
                                            result.target, &load_params,
                                            executable_.out()));
  }

  void TearDown() override {
    if (device()) {
      IREE_EXPECT_OK(iree_hal_queue_flush(asan_device_.queue()));
    }
  }

  iree_hal_device_t* device() const { return asan_device_.device(); }

  iree_hal_queue_t* queue() const { return asan_device_.queue(); }

  iree_hal_allocator_t* allocator() const { return asan_device_.allocator(); }

  SanitizerDeviceEventRecorder* recorder() const {
    return asan_device_.recorder();
  }

  iree_hal_executable_t* executable() const { return executable_; }

  void ExpectAsanReport(iree_host_size_t expected_count,
                        iree_hal_device_asan_access_kind_t expected_kind,
                        uint64_t expected_access_length,
                        AsanReportExpectationFlags expectation_flags) {
    recorder()->WaitForAsanReportCount(expected_count);
    EXPECT_EQ(recorder()->asan_report_count(), expected_count);
    iree_hal_device_asan_report_t report = recorder()->last_asan_report();
    EXPECT_EQ(report.record_length, sizeof(report));
    EXPECT_EQ(report.abi_version, IREE_HAL_DEVICE_ASAN_REPORT_ABI_VERSION_0);
    EXPECT_EQ(report.access_kind, expected_kind);
    EXPECT_NE(report.fault_address, 0u);
    EXPECT_EQ(report.access_length, expected_access_length);
    EXPECT_NE(report.shadow_address, 0u);
    if (expectation_flags & kAsanReportExpectationFlagShadowPoisoned) {
      EXPECT_NE(report.shadow_value, 0u);
    } else {
      EXPECT_EQ(report.shadow_value, 0u);
    }
  }

  SanitizerCachedBackendDevice asan_device_;
  Ref<iree_hal_executable_t> executable_;
};

TEST_P(AsanAllocationTest, InBoundsAccessStaysQuiet) {
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator(), AsanDeviceBufferParams(), kAsanAllocationBufferLength,
      buffer.out()));

  iree_hal_buffer_ref_t binding_refs[1];
  binding_refs[0] = iree_hal_make_buffer_ref(
      buffer, /*offset=*/0, iree_hal_buffer_byte_length(buffer));
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  recorder()->Reset();
  IREE_ASSERT_OK(DispatchAsanAllocationSelector(
      device(), queue(), executable(), bindings, kAsanAllocationHookLoad2,
      /*access_length=*/2, /*address_adjustment=*/0));
  IREE_ASSERT_OK(DispatchAsanAllocationSelector(
      device(), queue(), executable(), bindings, kAsanAllocationHookReportLoadN,
      kAsanAllocationSentinelLength, /*address_adjustment=*/0));
  ExpectAsanReport(/*expected_count=*/1, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ,
                   kAsanAllocationSentinelLength,
                   kAsanReportExpectationFlagNone);
}

TEST_P(AsanAllocationTest, ScopedSourcesPreserveNativeSanitizerAdvice) {
  const iree_hal_pool_family_access_t family = {
      iree_hal_queue_family(queue()),
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE};
  const auto* sanitizer =
      iree_hal_device_spec_sanitizer(iree_hal_device_spec(device()));
  for (auto host_access :
       {IREE_HAL_MEMORY_ACCESS_NONE, IREE_HAL_MEMORY_ACCESS_READ}) {
    SCOPED_TRACE(host_access);
    iree_hal_pool_scope_t scope = {1, &family, {}};
    scope.host.access = host_access;
    scope.host.modes = host_access ? IREE_HAL_MAPPING_MODE_SCOPED : 0;
    iree_hal_slab_pool_options_t source_options;
    iree_hal_slab_pool_options_initialize(&source_options);
    Ref<iree_hal_pool_t> source;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        asan_device_.device_group(), scope, &source_options,
        iree_allocator_system(), source.out()));
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = 65536;
    options.asan = sanitizer->asan.pool_options;
    Ref<iree_hal_pool_t> pool;
    IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
        source, &options, iree_allocator_system(), pool.out()));
    iree_hal_pool_reservation_request_t request = {};
    request.allocation_size = kAsanAllocationBufferLength;
    Ref<iree_hal_buffer_t> buffer;
    SemaphoreList allocated(device(), {0}, {1});
    IREE_ASSERT_OK(
        iree_hal_queue_alloca(queue(), iree_hal_semaphore_list_empty(),
                              allocated, pool, 1, &request, buffer.out()));
    const uint32_t value = 0x1234ABCD;
    SemaphoreList filled(device(), {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queue(), allocated, filled, buffer, 0, kAsanAllocationBufferLength,
        &value, sizeof(value), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(filled, iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
    const iree_hal_buffer_ref_t binding =
        iree_hal_make_buffer_ref(buffer, 0, kAsanAllocationBufferLength);
    recorder()->Reset();
    IREE_ASSERT_OK(DispatchAsanAllocationSelector(
        device(), queue(), executable(), {1, &binding},
        kAsanAllocationHookLoad2,
        /*access_length=*/2, /*address_adjustment=*/0));
    IREE_ASSERT_OK(DispatchAsanAllocationSelector(
        device(), queue(), executable(), {1, &binding},
        kAsanAllocationHookReportLoadN, kAsanAllocationSentinelLength,
        /*address_adjustment=*/0));
    ExpectAsanReport(
        /*expected_count=*/1, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ,
        kAsanAllocationSentinelLength, kAsanReportExpectationFlagNone);
    uint32_t output = 0;
    SemaphoreList downloaded(device(), {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_download(queue(), filled, downloaded, buffer,
                                           0, &output, sizeof(output),
                                           /*barriers=*/NULL));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        downloaded, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(output, value);
    SemaphoreList deallocated(device(), {0}, {1});
    auto* raw_buffer = buffer.get();
    IREE_ASSERT_OK(iree_hal_queue_dealloca(queue(), downloaded, deallocated, 1,
                                           &raw_buffer));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        deallocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }
}

TEST_P(AsanAllocationTest, AllocationRedzonesReport) {
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator(), AsanDeviceBufferParams(), kAsanAllocationBufferLength,
      buffer.out()));

  iree_hal_buffer_ref_t binding_refs[1];
  binding_refs[0] = iree_hal_make_buffer_ref(
      buffer, /*offset=*/0, iree_hal_buffer_byte_length(buffer));
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  recorder()->Reset();
  IREE_ASSERT_OK(DispatchAsanAllocationSelector(
      device(), queue(), executable(), bindings, kAsanAllocationHookLoad1,
      /*access_length=*/1, /*address_adjustment=*/-1));
  ExpectAsanReport(/*expected_count=*/1, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ,
                   1, kAsanReportExpectationFlagShadowPoisoned);

  iree_hal_buffer_ref_t tail_binding_refs[1];
  tail_binding_refs[0] = iree_hal_make_buffer_ref(
      buffer, kAsanAllocationBufferLength - 1, /*length=*/1);
  iree_hal_buffer_ref_list_t tail_bindings = {
      /*.count=*/IREE_ARRAYSIZE(tail_binding_refs),
      /*.values=*/tail_binding_refs,
  };

  recorder()->Reset();
  IREE_ASSERT_OK(DispatchAsanAllocationSelector(
      device(), queue(), executable(), tail_bindings, kAsanAllocationHookLoad2,
      /*access_length=*/2, /*address_adjustment=*/0));
  ExpectAsanReport(/*expected_count=*/1, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ,
                   2, kAsanReportExpectationFlagShadowPoisoned);
}

TEST_P(AsanAllocationTest, ReleasedAllocatorBufferReports) {
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator(), AsanDeviceBufferParams(), kAsanAllocationBufferLength,
      buffer.out()));

  uint64_t stale_address = 0;
  IREE_ASSERT_OK(ExportDeviceAddress(buffer, &stale_address));
  ASSERT_NE(stale_address, 0u);
  buffer.reset();

  recorder()->Reset();
  IREE_ASSERT_OK(DispatchAsanAllocationRawAddress(
      device(), queue(), executable(), stale_address, kAsanAllocationHookLoad1,
      /*access_length=*/1));
  ExpectAsanReport(/*expected_count=*/1, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ,
                   1, kAsanReportExpectationFlagShadowPoisoned);
}

TEST_P(AsanAllocationTest, QueueDeallocaReleaseReportsAfterSignal) {
  iree_hal_queue_t* queue = iree_hal_device_queue(device(), 0, 0);
  ASSERT_NE(queue, nullptr);
  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
      device(), iree_hal_queue_family(queue), &backend));

  iree_hal_buffer_params_t params = AsanQueueAllocaBufferParams();
  params.queue_family_affinity = iree_hal_make_queue_family_affinity(
      iree_hal_queue_family_ordinal(iree_hal_queue_family(queue)));
  const iree_hal_pool_reservation_request_t request = {
      .params = params,
      .allocation_size = kAsanAllocationBufferLength,
  };
  enum class PoolKind { kNative, kTLSF, kFixedBlock };
  for (auto kind :
       {PoolKind::kNative, PoolKind::kTLSF, PoolKind::kFixedBlock}) {
    SCOPED_TRACE(static_cast<int>(kind));
    Ref<iree_hal_pool_t> backing_pool;
    Ref<iree_hal_pool_t> pool;
    if (kind == PoolKind::kFixedBlock) {
      iree_hal_fixed_block_pool_options_t options = {};
      options.block_size = kAsanAllocationBufferLength;
      options.blocks_per_slab = 2;
      options.asan = backend.asan;
      IREE_ASSERT_OK(CreateFiniteBlockPool(backend, options,
                                           iree_allocator_system(),
                                           backing_pool.out(), pool.out()));
    } else {
      iree_hal_passthrough_pool_options_t options = {};
      if (kind == PoolKind::kNative) {
        options.asan = backend.asan;
      }
      Ref<iree_hal_pool_t> native_pool;
      IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
          options, backend.slab_provider, backend.notification,
          backend.frontier_tracker, backend.maintenance,
          iree_allocator_system(), native_pool.out()));
      if (kind == PoolKind::kNative) {
        pool = std::move(native_pool);
      } else {
        iree_hal_tlsf_pool_options_t tlsf_options = {};
        tlsf_options.tlsf_options.range_length = 65536;
        tlsf_options.asan = backend.asan;
        IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
            native_pool, &tlsf_options, iree_allocator_system(), pool.out()));
      }
    }
    for (uint64_t gate_value : {1u, 0u}) {
      SCOPED_TRACE(gate_value);
      Ref<iree_hal_buffer_t> buffer;
      SemaphoreList empty_wait;
      SemaphoreList alloca_signal(device(), {0}, {1});
      IREE_ASSERT_OK(
          iree_hal_queue_alloca(queue, empty_wait, alloca_signal, pool,
                                /*request_count=*/1, &request, buffer.out()));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          alloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

      uint64_t stale_address = 0;
      IREE_ASSERT_OK(ExportDeviceAddress(buffer, &stale_address));
      ASSERT_NE(stale_address, 0u);

      SemaphoreList dealloca_signal(device(), {0}, {1});
      iree_hal_buffer_t* dealloca_buffer = buffer;
      // Exercise both immediate capture and wait-before-signal deferral. The
      // already-completed allocation remains live until this last-use edge.
      SemaphoreList last_use(device(), {gate_value}, {1});
      IREE_ASSERT_OK(iree_hal_queue_dealloca(queue, last_use, dealloca_signal,
                                             /*buffer_count=*/1,
                                             &dealloca_buffer));
      if (gate_value == 0) {
        EXPECT_FALSE(iree_hal_semaphore_list_poll(dealloca_signal));
        IREE_ASSERT_OK(iree_hal_semaphore_list_signal(last_use, nullptr));
      }
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

      recorder()->Reset();
      IREE_ASSERT_OK(DispatchAsanAllocationRawAddress(
          device(), queue, executable(), stale_address,
          kAsanAllocationHookLoad1,
          /*access_length=*/1));
      ExpectAsanReport(/*expected_count=*/1,
                       IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 1,
                       kAsanReportExpectationFlagShadowPoisoned);
    }
  }
}

TEST_P(AsanAllocationTest, GuardedFiniteAllocaWaitsForInheritedHostUse) {
  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
      device(), iree_hal_queue_family(queue()), &backend));
  const iree_async_axis_t host_axis = iree_async_axis_make(
      iree_async_frontier_tracker_session_epoch(backend.frontier_tracker),
      iree_async_frontier_tracker_machine_index(backend.frontier_tracker),
      IREE_ASYNC_CAUSAL_DOMAIN_HOST, 0x534841444FULL);
  IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
      backend.frontier_tracker, host_axis, nullptr));
  uint64_t host_epoch = 0;
  for (bool use_tlsf : {false, true}) {
    SCOPED_TRACE(use_tlsf ? "TLSF" : "fixed block");
    iree_hal_fixed_block_pool_options_t source_options = {};
    source_options.block_size = 65536;
    source_options.blocks_per_slab = 1;
    source_options.frontier_capacity = 1;
    Ref<iree_hal_pool_t> backing_pool;
    Ref<iree_hal_pool_t> source_pool;
    IREE_ASSERT_OK(
        CreateFiniteBlockPool(backend, source_options, iree_allocator_system(),
                              backing_pool.out(), source_pool.out()));
    iree_hal_pool_reservation_request_t source_request = {
        AsanQueueAllocaBufferParams(), 65536};
    source_request.params.queue_family_affinity =
        iree_hal_make_queue_family_affinity(
            iree_hal_queue_family_ordinal(iree_hal_queue_family(queue())));
    iree_hal_pool_reservation_t source_reservation;
    iree_hal_pool_acquire_info_t source_info;
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        source_pool, 1, &source_request, nullptr,
        IREE_HAL_POOL_RESERVE_FLAG_NONE, &source_reservation, &source_info,
        &result));
    ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_FRESH, result);
    iree_async_single_frontier_t prior_use;
    iree_async_single_frontier_initialize(&prior_use, host_axis, ++host_epoch);
    const auto* frontier =
        iree_async_single_frontier_as_const_frontier(&prior_use);
    iree_hal_pool_release_reservations(source_pool, 1, &source_reservation,
                                       frontier);
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        source_pool, 1, &source_request, nullptr,
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &source_reservation,
        &source_info, &result));
    ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT, result);
    Ref<iree_hal_buffer_t> source;
    IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
        source_pool, 1, &source_request, &source_reservation,
        IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, source.out()));
    Ref<iree_hal_pool_t> pool;
    if (use_tlsf) {
      iree_hal_tlsf_pool_options_t options = {};
      options.asan = backend.asan;
      IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
          source, 0, IREE_HAL_WHOLE_BUFFER, &options, iree_allocator_system(),
          pool.out()));
    } else {
      iree_hal_fixed_block_pool_options_t options = {};
      options.block_size = kAsanAllocationBufferLength;
      options.asan = backend.asan;
      IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
          source, 0, IREE_HAL_WHOLE_BUFFER, &options, iree_allocator_system(),
          pool.out()));
    }
    const iree_hal_pool_reservation_request_t request = {
        source_request.params, kAsanAllocationBufferLength};
    Ref<iree_hal_buffer_t> buffer;
    SemaphoreList allocated(device(), {0}, {1});
    IREE_ASSERT_OK(
        iree_hal_queue_alloca(queue(), iree_hal_semaphore_list_empty(),
                              allocated, pool, 1, &request, buffer.out()));
    // Wait for reservation selection, not elapsed time, before checking the
    // completion edge. The host timeline deliberately remains incomplete.
    iree_hal_pool_stats_t stats = {};
    uint64_t allocated_value = 0;
    iree_status_t allocation_status = iree_ok_status();
    do {
      iree_hal_pool_query_stats(pool, &stats);
      allocation_status =
          iree_hal_semaphore_query(allocated.semaphores[0], &allocated_value);
      std::this_thread::yield();
    } while (stats.wait_count == 0 && allocated_value == 0 &&
             iree_status_is_ok(allocation_status));
    EXPECT_EQ(stats.wait_count, 1u);
    EXPECT_EQ(allocated_value, 0u);
    iree_async_frontier_tracker_advance(backend.frontier_tracker, host_axis,
                                        host_epoch);
    IREE_ASSERT_OK(allocation_status);
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        allocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    iree_hal_pool_query_stats(pool, &stats);
    EXPECT_EQ(stats.reserve_count, 1u);
    EXPECT_EQ(stats.release_count, 0u);
    iree_hal_buffer_ref_t binding =
        iree_hal_make_buffer_ref(buffer, 0, kAsanAllocationBufferLength);
    recorder()->Reset();
    IREE_ASSERT_OK(DispatchAsanAllocationSelector(
        device(), queue(), executable(), {1, &binding},
        kAsanAllocationHookLoad2, 2, 0));
    IREE_ASSERT_OK(DispatchAsanAllocationSelector(
        device(), queue(), executable(), {1, &binding},
        kAsanAllocationHookReportLoadN, kAsanAllocationSentinelLength, 0));
    ExpectAsanReport(1, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ,
                     kAsanAllocationSentinelLength,
                     kAsanReportExpectationFlagNone);
    iree_hal_buffer_t* dealloca_buffer = buffer;
    SemaphoreList released(device(), {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dealloca(queue(),
                                           iree_hal_semaphore_list_empty(),
                                           released, 1, &dealloca_buffer));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        released, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    buffer.reset();
    pool.reset();
    source.reset();
    iree_hal_pool_release_reservations(source_pool, 1, &source_reservation,
                                       nullptr);
  }
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(AsanAllocationTest);

}  // namespace iree::hal::cts
