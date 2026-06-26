// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/stage.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

static iree_hal_device_group_t* CreateMockDeviceGroup() {
  iree_hal_mock_device_options_t device_options;
  iree_hal_mock_device_options_initialize(&device_options);
  device_options.identifier = IREE_SV("id4-smoke-device");

  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(iree_hal_mock_device_create(&device_options,
                                            iree_allocator_system(), &device));

  iree_async_frontier_tracker_options_t tracker_options =
      iree_async_frontier_tracker_options_default();
  iree_async_frontier_tracker_t* frontier_tracker = NULL;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      tracker_options, iree_allocator_system(), &frontier_tracker));

  iree_hal_device_group_t* device_group = NULL;
  IREE_CHECK_OK(iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &device_group));

  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  return device_group;
}

static iree_hal_device_t* CreateLocalSyncDevice() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool));

  iree_hal_allocator_t* device_allocator = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      IREE_SV("id4-local-sync"), iree_allocator_system(),
      iree_allocator_system(), &device_allocator));

  iree_hal_sync_device_params_t sync_params;
  iree_hal_sync_device_params_initialize(&sync_params);
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_sync_device_create(
      IREE_SV("id4-local-sync"), &sync_params, &create_params,
      /*loader_count=*/0, /*loaders=*/nullptr, device_allocator,
      iree_allocator_system(), &device);
  iree_hal_allocator_release(device_allocator);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);
  return device;
}

static iree_hal_device_group_t* CreateLocalSyncDeviceGroup() {
  iree_hal_device_t* device = CreateLocalSyncDevice();

  iree_async_frontier_tracker_options_t tracker_options =
      iree_async_frontier_tracker_options_default();
  iree_async_frontier_tracker_t* frontier_tracker = NULL;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      tracker_options, iree_allocator_system(), &frontier_tracker));

  iree_hal_device_group_t* device_group = NULL;
  IREE_CHECK_OK(iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &device_group));

  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  return device_group;
}

static iree_hal_semaphore_t* CreateSemaphore(iree_hal_device_t* device) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_CHECK_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  return semaphore;
}

typedef struct id4_pipeline_test_diagnostics_log_t {
  // Number of diagnostic events observed.
  iree_host_size_t count;
  // Event keys observed in order.
  std::vector<std::string> keys;
  // Parameter slab events copied from diagnostic payloads.
  std::vector<id4_pipeline_parameter_slab_diagnostic_t> parameter_slabs;
  // Provider scopes copied for parameter slab events.
  std::vector<std::string> parameter_slab_scopes;
  // Parameter keys copied for parameter slab request events.
  std::vector<std::string> parameter_slab_keys;
} id4_pipeline_test_diagnostics_log_t;

typedef struct id4_pipeline_test_parameter_provider_t {
  // Base provider interface.
  iree_io_parameter_provider_t base;
  // True when the provider reports support for the smoke scope.
  bool supports_scope;
  // Number of gather operations observed.
  iree_host_size_t gather_count;
  // Scope passed to the last gather.
  std::string last_scope;
  // Parameter keys observed during gathers.
  std::vector<std::string> keys;
  // Parameter spans observed during gathers.
  std::vector<iree_io_parameter_span_t> spans;
  // Byte length of the last target buffer.
  iree_device_size_t last_target_byte_length;
} id4_pipeline_test_parameter_provider_t;

typedef struct id4_pipeline_test_bundle_payload_t {
  // Value written through the mutable bundle payload accessor.
  int value;
  // Counter incremented when the payload destructor runs.
  int* destroy_count;
} id4_pipeline_test_bundle_payload_t;

static id4_pipeline_test_parameter_provider_t* TestParameterProviderCast(
    iree_io_parameter_provider_t* provider) {
  return reinterpret_cast<id4_pipeline_test_parameter_provider_t*>(provider);
}

static void TestParameterProviderDestroy(
    iree_io_parameter_provider_t* provider) {}

static iree_status_t TestParameterProviderNotify(
    iree_io_parameter_provider_t* provider,
    iree_io_parameter_provider_signal_t signal) {
  return iree_ok_status();
}

static bool TestParameterProviderQuerySupport(
    iree_io_parameter_provider_t* provider, iree_string_view_t scope) {
  id4_pipeline_test_parameter_provider_t* test_provider =
      TestParameterProviderCast(provider);
  return test_provider->supports_scope &&
         iree_string_view_equal(scope, IREE_SV("smoke"));
}

static iree_status_t TestParameterProviderLoad(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_params_t target_params,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator,
    iree_io_parameter_emitter_t emitter) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t TestParameterProviderGather(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_t* target_buffer,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  id4_pipeline_test_parameter_provider_t* test_provider =
      TestParameterProviderCast(provider);
  ++test_provider->gather_count;
  test_provider->last_scope = ToString(source_scope);
  test_provider->last_target_byte_length =
      iree_hal_buffer_byte_length(target_buffer);
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    iree_io_parameter_span_t span;
    memset(&span, 0, sizeof(span));
    IREE_RETURN_IF_ERROR(enumerator.fn(enumerator.user_data, i, &key, &span));
    test_provider->keys.push_back(ToString(key));
    test_provider->spans.push_back(span);
  }
  return iree_hal_device_queue_barrier(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t TestParameterProviderScatter(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_string_view_t target_scope,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static const iree_io_parameter_provider_vtable_t kTestParameterProviderVTable =
    {
        // Releases test provider storage.
        TestParameterProviderDestroy,
        // Handles lifecycle notifications.
        TestParameterProviderNotify,
        // Queries scope support.
        TestParameterProviderQuerySupport,
        // Loads parameters into provider-owned buffers.
        TestParameterProviderLoad,
        // Gathers parameters into caller-owned buffers.
        TestParameterProviderGather,
        // Scatters parameters from caller-owned buffers.
        TestParameterProviderScatter,
};

static void TestParameterProviderInitialize(
    id4_pipeline_test_parameter_provider_t* provider) {
  provider->base = {};
  iree_atomic_ref_count_init(&provider->base.ref_count);
  provider->base.vtable = &kTestParameterProviderVTable;
  provider->supports_scope = true;
  provider->gather_count = 0;
  provider->last_scope.clear();
  provider->keys.clear();
  provider->spans.clear();
  provider->last_target_byte_length = 0;
}

static iree_status_t CreateSplatParameterIndexProvider(
    iree_string_view_t scope, iree_string_view_t key, uint64_t length,
    iree_const_byte_span_t pattern,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "splat parameter key is required");
  }
  if (length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "splat parameter byte length must be nonzero");
  }
  if (pattern.data_length == 0 ||
      pattern.data_length > IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "splat parameter pattern length %" PRIhsz
                            " is outside the supported range",
                            pattern.data_length);
  }

  iree_io_parameter_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_index_create(iree_allocator_system(), &index));
  iree_io_parameter_index_entry_t entry;
  memset(&entry, 0, sizeof(entry));
  entry.key = key;
  entry.length = length;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
  entry.storage.splat.pattern_length =
      static_cast<uint8_t>(pattern.data_length);
  std::memcpy(entry.storage.splat.pattern, pattern.data, pattern.data_length);
  iree_status_t status = iree_io_parameter_index_add(index, &entry);
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        scope, index,
        IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
        iree_allocator_system(), out_provider);
  }
  iree_io_parameter_index_release(index);
  return status;
}

static iree_status_t CaptureDiagnostic(
    void* user_data, const id4_pipeline_diagnostic_event_t* event) {
  id4_pipeline_test_diagnostics_log_t* log =
      static_cast<id4_pipeline_test_diagnostics_log_t*>(user_data);
  ++log->count;
  log->keys.push_back(ToString(event->key));
  if (event->parameter_slab) {
    log->parameter_slabs.push_back(*event->parameter_slab);
    log->parameter_slab_scopes.push_back(
        ToString(event->parameter_slab->scope));
    log->parameter_slab_keys.push_back(
        ToString(event->parameter_slab->parameter_key));
  }
  return iree_ok_status();
}

typedef struct id4_pipeline_smoke_stage_t {
  // Base stage; must be the first field.
  id4_pipeline_stage_t base;
  // True after load has completed.
  bool is_loaded;
  // True after issue has been called at least once.
  bool is_issued;
} id4_pipeline_smoke_stage_t;

static void SmokeStageDestroy(id4_pipeline_stage_t* stage) {
  id4_pipeline_smoke_stage_t* smoke_stage =
      reinterpret_cast<id4_pipeline_smoke_stage_t*>(stage);
  iree_allocator_t host_allocator = stage->services.host_allocator;
  id4_pipeline_stage_deinitialize(stage);
  iree_allocator_free(host_allocator, smoke_stage);
}

static iree_status_t SmokeStageLoad(
    id4_pipeline_stage_t* stage,
    const id4_pipeline_stage_load_options_t* options) {
  id4_pipeline_smoke_stage_t* smoke_stage =
      reinterpret_cast<id4_pipeline_smoke_stage_t*>(stage);
  smoke_stage->is_loaded = true;
  id4_pipeline_diagnostic_event_t event;
  memset(&event, 0, sizeof(event));
  event.kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE;
  event.stage_name = IREE_SV("smoke");
  event.key = IREE_SV("stage.load");
  event.message = IREE_SV("loaded smoke stage");
  return id4_pipeline_diagnostics_emit(options->diagnostics_sink, &event);
}

static iree_status_t SmokeStagePlan(
    id4_pipeline_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  const id4_pipeline_smoke_stage_t* smoke_stage =
      reinterpret_cast<const id4_pipeline_smoke_stage_t*>(stage);
  if (!smoke_stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "smoke stage must be loaded before planning");
  }

  id4_pipeline_parameter_request_t request = id4_pipeline_parameter_request(
      IREE_SV("smoke.weight"),
      id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                  /*buffer_offset=*/0, /*length=*/16));

  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          IREE_SV("smoke"), /*placement_id=*/0, /*binding_slot=*/0,
          options->queue_affinity,
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING |
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ,
          /*byte_length=*/16, /*alignment=*/16, /*request_count=*/1, &request);
  id4_pipeline_parameter_load_step_t load_step =
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("parameters.gather"), IREE_SV("smoke"),
          /*target_slab_index=*/0, /*request_offset=*/0,
          /*request_count=*/1);

  id4_pipeline_device_placement_t placement;
  memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = options->device_index;
  placement.queue_affinity = options->queue_affinity;

  id4_pipeline_plan_create_options_t create_options;
  memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.stage_name = IREE_SV("smoke");
  create_options.device_group = stage->services.device_group;
  create_options.placement_count = 1;
  create_options.placements = &placement;
  create_options.parameter_slab_count = 1;
  create_options.parameter_slabs = &slab;
  create_options.parameter_load_step_count = 1;
  create_options.parameter_load_steps = &load_step;
  create_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_plan_create(&create_options,
                                  stage->services.host_allocator, out_plan);
}

static iree_status_t SmokeStagePrepare(
    id4_pipeline_stage_t* stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  const id4_pipeline_smoke_stage_t* smoke_stage =
      reinterpret_cast<const id4_pipeline_smoke_stage_t*>(stage);
  if (!smoke_stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "smoke stage must be loaded before preparation");
  }
  id4_pipeline_parameter_slab_set_t* parameter_slabs = NULL;
  const iree_host_size_t parameter_slab_count =
      id4_pipeline_plan_parameter_slab_count(plan);
  if (parameter_slab_count != 0 && !options->parameter_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke parameter provider is required");
  }
  iree_status_t status = iree_ok_status();
  if (parameter_slab_count != 0) {
    status = id4_pipeline_plan_load_parameter_slabs(
        plan, options->parameter_provider, options->wait_semaphore_list,
        options->signal_semaphore_list, options->diagnostics_sink,
        stage->services.host_allocator, &parameter_slabs);
  } else if (options->signal_semaphore_list.count != 0) {
    status = iree_hal_semaphore_list_signal(options->signal_semaphore_list,
                                            /*frontier=*/NULL);
  }
  id4_pipeline_bundle_t* bundle = NULL;
  if (iree_status_is_ok(status)) {
    id4_pipeline_bundle_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.plan = plan;
    create_options.parameter_slabs = parameter_slabs;
    create_options.readiness_semaphore_list =
        options->signal_semaphore_list.count != 0
            ? options->signal_semaphore_list
            : iree_hal_semaphore_list_empty();
    status = id4_pipeline_bundle_create(
        &create_options, stage->services.host_allocator, &bundle);
  }
  id4_pipeline_parameter_slab_set_release(parameter_slabs);
  if (iree_status_is_ok(status)) {
    id4_pipeline_diagnostic_event_t event;
    memset(&event, 0, sizeof(event));
    event.kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE;
    event.stage_name = id4_pipeline_plan_stage_name(plan);
    event.key = IREE_SV("stage.prepare");
    event.message = IREE_SV("prepared smoke bundle");
    status = id4_pipeline_diagnostics_emit(options->diagnostics_sink, &event);
  }
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    id4_pipeline_bundle_release(bundle);
  }
  return status;
}

static iree_status_t SmokeStageIssue(
    id4_pipeline_stage_t* stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  id4_pipeline_smoke_stage_t* smoke_stage =
      reinterpret_cast<id4_pipeline_smoke_stage_t*>(stage);
  const id4_pipeline_plan_t* plan = id4_pipeline_bundle_plan(bundle);
  if (!plan) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "smoke bundle has no plan");
  }
  id4_pipeline_diagnostic_event_t event;
  memset(&event, 0, sizeof(event));
  event.kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE;
  event.stage_name = id4_pipeline_plan_stage_name(plan);
  event.key = IREE_SV("stage.issue");
  event.message = IREE_SV("issued smoke bundle");
  IREE_RETURN_IF_ERROR(
      id4_pipeline_diagnostics_emit(options->diagnostics_sink, &event));
  smoke_stage->is_issued = true;
  return iree_hal_semaphore_list_signal(options->signal_semaphore_list,
                                        /*frontier=*/NULL);
}

static const id4_pipeline_stage_vtable_t kSmokeStageVTable = {
    // Destroys the concrete smoke stage.
    SmokeStageDestroy,
    // Loads smoke stage metadata.
    SmokeStageLoad,
    // Plans smoke stage execution.
    SmokeStagePlan,
    // Prepares a smoke stage bundle.
    SmokeStagePrepare,
    // Issues a smoke stage bundle.
    SmokeStageIssue,
};

static iree_status_t SmokeStageCreate(iree_hal_device_group_t* device_group,
                                      id4_pipeline_stage_t** out_stage) {
  *out_stage = NULL;
  id4_pipeline_smoke_stage_t* smoke_stage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(), sizeof(*smoke_stage), (void**)&smoke_stage));
  memset(smoke_stage, 0, sizeof(*smoke_stage));

  id4_pipeline_stage_services_t services;
  memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();
  iree_status_t status = id4_pipeline_stage_initialize(
      &kSmokeStageVTable, &services, &smoke_stage->base);
  if (iree_status_is_ok(status)) {
    *out_stage = &smoke_stage->base;
  } else {
    iree_allocator_free(iree_allocator_system(), smoke_stage);
  }
  return status;
}

static id4_pipeline_diagnostics_sink_t DiagnosticsSink(
    id4_pipeline_test_diagnostics_log_t* log) {
  id4_pipeline_diagnostics_sink_t sink;
  memset(&sink, 0, sizeof(sink));
  sink.emit = CaptureDiagnostic;
  sink.user_data = log;
  return sink;
}

static id4_pipeline_diagnostics_sink_t IgnoreDiagnosticsSink() {
  id4_pipeline_diagnostics_sink_t sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&sink);
  return sink;
}

static void DestroyTestBundlePayload(id4_pipeline_bundle_t* bundle,
                                     void* payload) {
  (void)bundle;
  auto* test_payload =
      static_cast<id4_pipeline_test_bundle_payload_t*>(payload);
  ++*test_payload->destroy_count;
}

TEST(PipelineStage, PlanCopiesPlacementAndParameterSlabMetadata) {
  iree_hal_device_group_t* device_group = CreateMockDeviceGroup();
  id4_pipeline_stage_t* stage = NULL;
  IREE_ASSERT_OK(SmokeStageCreate(device_group, &stage));
  iree_hal_device_group_release(device_group);

  id4_pipeline_test_diagnostics_log_t diagnostics_log = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      DiagnosticsSink(&diagnostics_log);

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = NULL;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  EXPECT_EQ(ToString(id4_pipeline_plan_stage_name(plan)), "smoke");
  EXPECT_EQ(id4_pipeline_plan_placement_count(plan), 1u);
  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(plan, 0);
  ASSERT_NE(placement, nullptr);
  EXPECT_EQ(ToString(placement->role), "default");
  EXPECT_EQ(placement->device_index, 0u);
  EXPECT_EQ(placement->queue_affinity, IREE_HAL_QUEUE_AFFINITY_ANY);

  EXPECT_EQ(id4_pipeline_plan_parameter_slab_count(plan), 1u);
  const id4_pipeline_parameter_slab_plan_t* slab =
      id4_pipeline_plan_parameter_slab_at(plan, 0);
  ASSERT_NE(slab, nullptr);
  EXPECT_EQ(ToString(slab->scope), "smoke");
  EXPECT_EQ(slab->byte_length, 16u);
  EXPECT_EQ(slab->alignment, 16u);
  EXPECT_EQ(slab->target_params.queue_affinity, IREE_HAL_QUEUE_AFFINITY_ANY);
  EXPECT_EQ(slab->target_params.min_alignment, 16u);
  EXPECT_EQ(slab->request_count, 1u);

  id4_pipeline_parameter_slab_enumerator_state_t enumerator_state;
  memset(&enumerator_state, 0, sizeof(enumerator_state));
  enumerator_state.slab = slab;
  enumerator_state.request_offset = 0;
  enumerator_state.request_count = slab->request_count;
  iree_io_parameter_enumerator_t enumerator =
      id4_pipeline_parameter_slab_enumerator(&enumerator_state);
  iree_string_view_t key = iree_string_view_empty();
  iree_io_parameter_span_t span;
  memset(&span, 0, sizeof(span));
  IREE_ASSERT_OK(enumerator.fn(enumerator.user_data, 0, &key, &span));
  EXPECT_EQ(ToString(key), "smoke.weight");
  EXPECT_EQ(span.length, 16u);

  EXPECT_EQ(id4_pipeline_plan_parameter_load_step_count(plan), 1u);
  const id4_pipeline_parameter_load_step_t* load_step =
      id4_pipeline_plan_parameter_load_step_at(plan, 0);
  ASSERT_NE(load_step, nullptr);
  EXPECT_EQ(ToString(load_step->name), "parameters.gather");
  EXPECT_EQ(load_step->kind, ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER);
  EXPECT_EQ(ToString(load_step->source_scope), "smoke");
  EXPECT_EQ(load_step->target_slab_index, 0u);
  EXPECT_EQ(load_step->request_offset, 0u);
  EXPECT_EQ(load_step->request_count, 1u);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(id4_pipeline_plan_format_json(plan, &builder));
  std::string json = ToString(iree_string_builder_view(&builder));
  EXPECT_NE(json.find("\"stage\":\"smoke\""), std::string::npos);
  EXPECT_NE(json.find("\"parameter_slabs\""), std::string::npos);
  EXPECT_NE(json.find("\"parameter_load_steps\""), std::string::npos);
  EXPECT_NE(json.find("\"source_scope\":\"smoke\""), std::string::npos);
  EXPECT_NE(json.find("\"target_params\""), std::string::npos);
  EXPECT_NE(json.find("\"smoke.weight\""), std::string::npos);
  iree_string_builder_deinitialize(&builder);

  EXPECT_EQ(diagnostics_log.count, 3u);
  ASSERT_EQ(diagnostics_log.keys.size(), 3u);
  EXPECT_EQ(diagnostics_log.keys[0], "stage.load");
  EXPECT_EQ(diagnostics_log.keys[1], "plan.create");
  EXPECT_EQ(diagnostics_log.keys[2], "parameter_slab.plan");
  ASSERT_EQ(diagnostics_log.parameter_slabs.size(), 1u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].slab_index, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].request_index,
            IREE_HOST_SIZE_MAX);
  EXPECT_EQ(diagnostics_log.parameter_slab_scopes[0], "smoke");
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].placement_id, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].device_index, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].queue_affinity,
            IREE_HAL_QUEUE_AFFINITY_ANY);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].slab_byte_length, 16u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].slab_alignment, 16u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].request_count, 1u);

  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
}

TEST(PipelineStage, PlanRequiresExplicitParameterLoadSteps) {
  iree_hal_device_group_t* device_group = CreateMockDeviceGroup();
  id4_pipeline_diagnostics_sink_t diagnostics_sink = IgnoreDiagnosticsSink();

  id4_pipeline_parameter_request_t request = id4_pipeline_parameter_request(
      IREE_SV("smoke.weight"),
      id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                  /*buffer_offset=*/0, /*length=*/16));
  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          IREE_SV("smoke"), /*placement_id=*/0, /*binding_slot=*/0,
          IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING |
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ,
          /*byte_length=*/16, /*alignment=*/16, /*request_count=*/1, &request);
  id4_pipeline_device_placement_t placement = {
      // Human-readable placement role.
      /*.role=*/IREE_SV("default"),
      // Device index in the test device group.
      /*.device_index=*/0,
      // Queue affinity used for loading.
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  id4_pipeline_plan_create_options_t create_options;
  memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.stage_name = IREE_SV("smoke");
  create_options.device_group = device_group;
  create_options.placement_count = 1;
  create_options.placements = &placement;
  create_options.parameter_slab_count = 1;
  create_options.parameter_slabs = &slab;
  create_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_plan_create(
                            &create_options, iree_allocator_system(), &plan));
  EXPECT_EQ(plan, nullptr);

  iree_hal_device_group_release(device_group);
}

TEST(PipelineStage, BundlePayloadIsInlineAndDestroyed) {
  iree_hal_device_group_t* device_group = CreateMockDeviceGroup();
  id4_pipeline_diagnostics_sink_t diagnostics_sink = IgnoreDiagnosticsSink();

  id4_pipeline_device_placement_t placement;
  memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = 0;
  placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  id4_pipeline_plan_create_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV("payload");
  plan_options.device_group = device_group;
  plan_options.placement_count = 1;
  plan_options.placements = &placement;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = NULL;
  IREE_ASSERT_OK(
      id4_pipeline_plan_create(&plan_options, iree_allocator_system(), &plan));

  int destroy_count = 0;
  id4_pipeline_bundle_create_options_t bundle_options;
  memset(&bundle_options, 0, sizeof(bundle_options));
  bundle_options.structure_size = sizeof(bundle_options);
  bundle_options.plan = plan;
  bundle_options.payload_size = sizeof(id4_pipeline_test_bundle_payload_t);
  bundle_options.payload_alignment =
      iree_alignof(id4_pipeline_test_bundle_payload_t);
  bundle_options.payload_destroy = DestroyTestBundlePayload;

  id4_pipeline_bundle_t* bundle = NULL;
  IREE_ASSERT_OK(id4_pipeline_bundle_create(&bundle_options,
                                            iree_allocator_system(), &bundle));
  auto* payload = static_cast<id4_pipeline_test_bundle_payload_t*>(
      id4_pipeline_bundle_payload(bundle));
  ASSERT_NE(payload, nullptr);
  payload->value = 42;
  payload->destroy_count = &destroy_count;
  const auto* const_payload =
      static_cast<const id4_pipeline_test_bundle_payload_t*>(
          id4_pipeline_bundle_const_payload(bundle));
  ASSERT_NE(const_payload, nullptr);
  EXPECT_EQ(const_payload->value, 42);

  id4_pipeline_bundle_release(bundle);
  EXPECT_EQ(destroy_count, 1);

  id4_pipeline_plan_release(plan);
  iree_hal_device_group_release(device_group);
}

TEST(PipelineStage, IssueRequiresPlannedBoundaryBindings) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage = NULL;
  IREE_ASSERT_OK(SmokeStageCreate(device_group, &stage));

  id4_pipeline_diagnostics_sink_t diagnostics_sink = IgnoreDiagnosticsSink();

  id4_pipeline_device_placement_t placement;
  memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = 0;
  placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  id4_pipeline_region_plan_t region;
  memset(&region, 0, sizeof(region));
  region.name = IREE_SV("boundary.region");
  region.placement_id = 0;
  region.binding_capacity = 2;
  region.local_binding_slot = 1;

  id4_pipeline_boundary_tensor_plan_t boundary_tensor;
  memset(&boundary_tensor, 0, sizeof(boundary_tensor));
  boundary_tensor.layout.name = IREE_SV("boundary.input");
  boundary_tensor.layout.dtype = ID4_PIPELINE_TENSOR_DTYPE_U32;
  boundary_tensor.layout.shape.rank = 1;
  boundary_tensor.layout.shape.dims[0] = 4;
  boundary_tensor.layout.byte_length = 16;
  boundary_tensor.layout.alignment = 4;
  boundary_tensor.flags = ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED |
                          ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED;
  boundary_tensor.region_id = 0;
  boundary_tensor.placement_id = 0;
  boundary_tensor.binding_slot = 0;

  id4_pipeline_plan_create_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV("boundary");
  plan_options.device_group = device_group;
  plan_options.placement_count = 1;
  plan_options.placements = &placement;
  plan_options.boundary_tensor_count = 1;
  plan_options.boundary_tensors = &boundary_tensor;
  plan_options.region_count = 1;
  plan_options.regions = &region;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = NULL;
  IREE_ASSERT_OK(
      id4_pipeline_plan_create(&plan_options, iree_allocator_system(), &plan));

  id4_pipeline_bundle_create_options_t bundle_options;
  memset(&bundle_options, 0, sizeof(bundle_options));
  bundle_options.structure_size = sizeof(bundle_options);
  bundle_options.plan = plan;

  id4_pipeline_bundle_t* bundle = NULL;
  IREE_ASSERT_OK(id4_pipeline_bundle_create(&bundle_options,
                                            iree_allocator_system(), &bundle));

  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_semaphore_t* issue_semaphore = CreateSemaphore(device);
  uint64_t issue_value = 1;
  iree_hal_semaphore_list_t issue_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&issue_semaphore,
      /*.payload_values=*/&issue_value,
  };

  id4_pipeline_stage_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.signal_semaphore_list = issue_signal_list;
  issue_options.diagnostics_sink = &diagnostics_sink;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_stage_issue(stage, bundle, &issue_options));

  iree_hal_buffer_params_t buffer_params;
  memset(&buffer_params, 0, sizeof(buffer_params));
  buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                       IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                       IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_MAPPING;
  buffer_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  buffer_params.min_alignment = 4;

  iree_hal_buffer_t* boundary_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), buffer_params, 16, &boundary_buffer));
  iree_hal_buffer_binding_t boundary_bindings[1];
  memset(boundary_bindings, 0, sizeof(boundary_bindings));
  boundary_bindings[0].buffer = boundary_buffer;
  boundary_bindings[0].offset = 0;
  boundary_bindings[0].length = 16;

  issue_options.boundary_binding_count = 1;
  issue_options.boundary_bindings = boundary_bindings;
  IREE_ASSERT_OK(id4_pipeline_stage_issue(stage, bundle, &issue_options));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(issue_semaphore, issue_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(boundary_buffer);
  iree_hal_semaphore_release(issue_semaphore);
  id4_pipeline_bundle_release(bundle);
  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

TEST(PipelineStage, PlanRejectsInvalidDevice) {
  iree_hal_device_group_t* device_group = CreateMockDeviceGroup();
  id4_pipeline_stage_t* stage = NULL;
  IREE_ASSERT_OK(SmokeStageCreate(device_group, &stage));
  iree_hal_device_group_release(device_group);

  id4_pipeline_diagnostics_sink_t diagnostics_sink = IgnoreDiagnosticsSink();

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 1;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        id4_pipeline_stage_plan(stage, &plan_options, &plan));
  EXPECT_EQ(plan, nullptr);

  id4_pipeline_stage_release(stage);
}

TEST(PipelineStage, LifecycleRejectsMissingOptions) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage = NULL;
  IREE_ASSERT_OK(SmokeStageCreate(device_group, &stage));
  iree_hal_device_group_release(device_group);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_stage_load(stage, NULL));

  id4_pipeline_stage_load_options_t missing_diagnostics_load_options;
  memset(&missing_diagnostics_load_options, 0,
         sizeof(missing_diagnostics_load_options));
  missing_diagnostics_load_options.structure_size =
      sizeof(missing_diagnostics_load_options);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_stage_load(stage, &missing_diagnostics_load_options));

  id4_pipeline_diagnostics_sink_t diagnostics_sink = IgnoreDiagnosticsSink();

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_plan_t* plan = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_stage_plan(stage, NULL, &plan));

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  id4_pipeline_bundle_t* bundle = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_stage_prepare(stage, plan, NULL, &bundle));

  id4_pipeline_test_parameter_provider_t provider;
  TestParameterProviderInitialize(&provider);
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
  iree_hal_semaphore_t* ready_semaphore = CreateSemaphore(device);
  uint64_t ready_value = 1;
  iree_hal_semaphore_list_t prepare_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&ready_semaphore,
      /*.payload_values=*/&ready_value,
  };

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = &provider.base;
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = prepare_signal_list;
  prepare_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(
      id4_pipeline_stage_prepare(stage, plan, &prepare_options, &bundle));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_stage_issue(stage, bundle, NULL));

  id4_pipeline_bundle_release(bundle);
  iree_hal_semaphore_release(ready_semaphore);
  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
}

TEST(PipelineStage, PrepareAndIssueSmokeBundle) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage = NULL;
  IREE_ASSERT_OK(SmokeStageCreate(device_group, &stage));
  iree_hal_device_group_release(device_group);

  id4_pipeline_test_diagnostics_log_t diagnostics_log = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      DiagnosticsSink(&diagnostics_log);

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = NULL;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  id4_pipeline_test_parameter_provider_t provider;
  TestParameterProviderInitialize(&provider);
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
  iree_hal_semaphore_t* ready_semaphore = CreateSemaphore(device);
  uint64_t ready_value = 1;
  iree_hal_semaphore_list_t prepare_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&ready_semaphore,
      /*.payload_values=*/&ready_value,
  };

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = &provider.base;
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = prepare_signal_list;
  prepare_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_bundle_t* bundle = NULL;
  IREE_ASSERT_OK(
      id4_pipeline_stage_prepare(stage, plan, &prepare_options, &bundle));
  EXPECT_EQ(id4_pipeline_bundle_plan(bundle), plan);
  iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  ASSERT_EQ(readiness_list.count, 1u);
  EXPECT_EQ(readiness_list.semaphores[0], ready_semaphore);
  EXPECT_EQ(readiness_list.payload_values[0], ready_value);

  iree_hal_semaphore_t* issue_semaphore = CreateSemaphore(device);
  uint64_t issue_value = 1;
  iree_hal_semaphore_list_t issue_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&issue_semaphore,
      /*.payload_values=*/&issue_value,
  };

  id4_pipeline_stage_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.wait_semaphore_list = readiness_list;
  issue_options.signal_semaphore_list = issue_signal_list;
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_issue(stage, bundle, &issue_options));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(issue_semaphore, issue_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  ASSERT_EQ(diagnostics_log.keys.size(), 7u);
  EXPECT_EQ(diagnostics_log.keys[0], "stage.load");
  EXPECT_EQ(diagnostics_log.keys[1], "plan.create");
  EXPECT_EQ(diagnostics_log.keys[2], "parameter_slab.plan");
  EXPECT_EQ(diagnostics_log.keys[3], "parameter_slab.load");
  EXPECT_EQ(diagnostics_log.keys[4], "parameter_slab.gather");
  EXPECT_EQ(diagnostics_log.keys[5], "stage.prepare");
  EXPECT_EQ(diagnostics_log.keys[6], "stage.issue");

  id4_pipeline_bundle_release(bundle);
  iree_hal_semaphore_release(issue_semaphore);
  iree_hal_semaphore_release(ready_semaphore);
  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
}

TEST(PipelineStage, PrepareRejectsParameterSlabLoadWithoutSignal) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage = NULL;
  IREE_ASSERT_OK(SmokeStageCreate(device_group, &stage));
  iree_hal_device_group_release(device_group);

  id4_pipeline_diagnostics_sink_t diagnostics_sink = IgnoreDiagnosticsSink();

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = NULL;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  id4_pipeline_test_parameter_provider_t provider;
  TestParameterProviderInitialize(&provider);

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = &provider.base;
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_bundle_t* bundle = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_stage_prepare(stage, plan, &prepare_options, &bundle));
  EXPECT_EQ(bundle, nullptr);

  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
}

TEST(PipelineStage, PrepareEmitsParameterSlabLoadFailureDiagnostic) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage = NULL;
  IREE_ASSERT_OK(SmokeStageCreate(device_group, &stage));
  iree_hal_device_group_release(device_group);

  id4_pipeline_diagnostics_sink_t setup_diagnostics_sink =
      IgnoreDiagnosticsSink();

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &setup_diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &setup_diagnostics_sink;

  id4_pipeline_plan_t* plan = NULL;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  id4_pipeline_test_parameter_provider_t provider;
  TestParameterProviderInitialize(&provider);
  provider.supports_scope = false;

  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
  iree_hal_semaphore_t* ready_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready_semaphore));
  uint64_t ready_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&ready_semaphore,
      /*.payload_values=*/&ready_value,
  };

  id4_pipeline_test_diagnostics_log_t diagnostics_log = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      DiagnosticsSink(&diagnostics_log);

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = &provider.base;
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal_list;
  prepare_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_bundle_t* bundle = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      id4_pipeline_stage_prepare(stage, plan, &prepare_options, &bundle));
  EXPECT_EQ(bundle, nullptr);
  EXPECT_EQ(provider.gather_count, 0u);

  ASSERT_EQ(diagnostics_log.keys.size(), 1u);
  EXPECT_EQ(diagnostics_log.keys[0], "parameter_slab.load.error");
  ASSERT_EQ(diagnostics_log.parameter_slabs.size(), 1u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].slab_index, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].request_index,
            IREE_HOST_SIZE_MAX);
  EXPECT_EQ(diagnostics_log.parameter_slab_scopes[0], "smoke");
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].device_index, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].slab_byte_length, 16u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].request_count, 1u);

  iree_hal_semaphore_release(ready_semaphore);
  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
}

TEST(PipelineStage, PrepareLoadsParameterSlabsWhenProviderIsSupplied) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage = NULL;
  IREE_ASSERT_OK(SmokeStageCreate(device_group, &stage));
  iree_hal_device_group_release(device_group);

  id4_pipeline_diagnostics_sink_t setup_diagnostics_sink =
      IgnoreDiagnosticsSink();

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &setup_diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &setup_diagnostics_sink;

  id4_pipeline_plan_t* plan = NULL;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  id4_pipeline_test_parameter_provider_t provider;
  TestParameterProviderInitialize(&provider);

  id4_pipeline_test_diagnostics_log_t diagnostics_log = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      DiagnosticsSink(&diagnostics_log);

  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
  iree_hal_semaphore_t* ready_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready_semaphore));
  uint64_t ready_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&ready_semaphore,
      /*.payload_values=*/&ready_value,
  };

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = &provider.base;
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal_list;
  prepare_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_bundle_t* bundle = NULL;
  IREE_ASSERT_OK(
      id4_pipeline_stage_prepare(stage, plan, &prepare_options, &bundle));
  iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  ASSERT_EQ(readiness_list.count, 1u);
  EXPECT_EQ(readiness_list.semaphores[0], ready_semaphore);
  EXPECT_EQ(readiness_list.payload_values[0], ready_value);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      readiness_list.semaphores[0], readiness_list.payload_values[0],
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  id4_pipeline_parameter_slab_set_t* slab_set =
      id4_pipeline_bundle_parameter_slabs(bundle);
  ASSERT_NE(slab_set, nullptr);
  EXPECT_EQ(id4_pipeline_parameter_slab_set_count(slab_set), 1u);
  iree_hal_buffer_t* slab_buffer =
      id4_pipeline_parameter_slab_set_buffer_at(slab_set, 0);
  ASSERT_NE(slab_buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_byte_length(slab_buffer), 16u);

  EXPECT_EQ(provider.gather_count, 1u);
  EXPECT_EQ(provider.last_scope, "smoke");
  EXPECT_EQ(provider.last_target_byte_length, 16u);
  ASSERT_EQ(provider.keys.size(), 1u);
  EXPECT_EQ(provider.keys[0], "smoke.weight");
  ASSERT_EQ(provider.spans.size(), 1u);
  EXPECT_EQ(provider.spans[0].parameter_offset, 0u);
  EXPECT_EQ(provider.spans[0].buffer_offset, 0u);
  EXPECT_EQ(provider.spans[0].length, 16u);

  ASSERT_EQ(diagnostics_log.keys.size(), 3u);
  EXPECT_EQ(diagnostics_log.keys[0], "parameter_slab.load");
  EXPECT_EQ(diagnostics_log.keys[1], "parameter_slab.gather");
  EXPECT_EQ(diagnostics_log.keys[2], "stage.prepare");
  ASSERT_EQ(diagnostics_log.parameter_slabs.size(), 2u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].slab_index, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].request_index,
            IREE_HOST_SIZE_MAX);
  EXPECT_EQ(diagnostics_log.parameter_slab_scopes[0], "smoke");
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].device_index, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].slab_byte_length, 16u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[0].request_count, 1u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[1].slab_index, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[1].request_index, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slab_scopes[1], "smoke");
  EXPECT_EQ(diagnostics_log.parameter_slab_keys[1], "smoke.weight");
  EXPECT_EQ(diagnostics_log.parameter_slabs[1].parameter_offset, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[1].buffer_offset, 0u);
  EXPECT_EQ(diagnostics_log.parameter_slabs[1].length, 16u);

  id4_pipeline_bundle_release(bundle);
  iree_hal_semaphore_release(ready_semaphore);
  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
}

TEST(PipelineStage, PrepareLoadsParameterSlabsFromParameterIndexProvider) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage = NULL;
  IREE_ASSERT_OK(SmokeStageCreate(device_group, &stage));
  iree_hal_device_group_release(device_group);

  id4_pipeline_diagnostics_sink_t diagnostics_sink = IgnoreDiagnosticsSink();

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = NULL;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  iree_io_parameter_provider_t* provider = NULL;
  static const uint8_t kWeightPattern[] = {0x11, 0x22, 0x33, 0x44};
  IREE_ASSERT_OK(CreateSplatParameterIndexProvider(
      IREE_SV("smoke"), IREE_SV("smoke.weight"), /*length=*/16,
      iree_make_const_byte_span(kWeightPattern, sizeof(kWeightPattern)),
      &provider));

  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
  iree_hal_semaphore_t* ready_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready_semaphore));
  uint64_t ready_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&ready_semaphore,
      /*.payload_values=*/&ready_value,
  };

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = provider;
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal_list;
  prepare_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_bundle_t* bundle = NULL;
  IREE_ASSERT_OK(
      id4_pipeline_stage_prepare(stage, plan, &prepare_options, &bundle));
  iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  ASSERT_EQ(readiness_list.count, 1u);
  EXPECT_EQ(readiness_list.semaphores[0], ready_semaphore);
  EXPECT_EQ(readiness_list.payload_values[0], ready_value);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(ready_semaphore, ready_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  id4_pipeline_parameter_slab_set_t* slab_set =
      id4_pipeline_bundle_parameter_slabs(bundle);
  ASSERT_NE(slab_set, nullptr);
  iree_hal_buffer_t* slab_buffer =
      id4_pipeline_parameter_slab_set_buffer_at(slab_set, 0);
  ASSERT_NE(slab_buffer, nullptr);
  uint8_t actual[16] = {0};
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(slab_buffer, 0, actual, sizeof(actual)));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(actual); i += 4) {
    EXPECT_EQ(actual[i + 0], 0x11);
    EXPECT_EQ(actual[i + 1], 0x22);
    EXPECT_EQ(actual[i + 2], 0x33);
    EXPECT_EQ(actual[i + 3], 0x44);
  }

  id4_pipeline_bundle_release(bundle);
  iree_hal_semaphore_release(ready_semaphore);
  iree_io_parameter_provider_release(provider);
  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
}

}  // namespace
