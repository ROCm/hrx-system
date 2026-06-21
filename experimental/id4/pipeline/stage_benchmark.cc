// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "experimental/id4/pipeline/plan.h"
#include "iree/async/frontier_tracker.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gbenchmark_harness.h"

namespace {

static iree_hal_device_group_t* CreateMockDeviceGroup() {
  iree_hal_mock_device_options_t device_options;
  iree_hal_mock_device_options_initialize(&device_options);
  device_options.identifier = IREE_SV("id4-benchmark-device");

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

static void BM_PipelinePlanCreateAndFormatJson(benchmark::State& state) {
  iree_hal_device_group_t* device_group = CreateMockDeviceGroup();
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  for (auto _ : state) {
    id4_pipeline_parameter_request_t request = id4_pipeline_parameter_request(
        IREE_SV("benchmark.weight"),
        id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                    /*buffer_offset=*/0, /*length=*/16));

    id4_pipeline_parameter_slab_plan_t slab =
        id4_pipeline_make_device_local_parameter_slab_plan(
            IREE_SV("benchmark"), /*placement_id=*/0, /*binding_slot=*/0,
            IREE_HAL_QUEUE_AFFINITY_ANY,
            IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
            /*byte_length=*/16, /*alignment=*/16, /*request_count=*/1,
            &request);

    id4_pipeline_device_placement_t placement;
    memset(&placement, 0, sizeof(placement));
    placement.role = IREE_SV("benchmark");
    placement.device_index = 0;
    placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

    id4_pipeline_plan_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.stage_name = IREE_SV("benchmark");
    create_options.device_group = device_group;
    create_options.placement_count = 1;
    create_options.placements = &placement;
    create_options.parameter_slab_count = 1;
    create_options.parameter_slabs = &slab;
    create_options.diagnostics_sink = &diagnostics_sink;

    id4_pipeline_plan_t* plan = NULL;
    IREE_CHECK_OK(id4_pipeline_plan_create(&create_options,
                                           iree_allocator_system(), &plan));

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_CHECK_OK(id4_pipeline_plan_format_json(plan, &builder));
    benchmark::DoNotOptimize(iree_string_builder_view(&builder).size);
    iree_string_builder_deinitialize(&builder);

    id4_pipeline_plan_release(plan);
  }

  iree_hal_device_group_release(device_group);
}
BENCHMARK(BM_PipelinePlanCreateAndFormatJson);

}  // namespace
