// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/capture.h"

#include <fstream>
#include <string>
#include <vector>

#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

using ::testing::HasSubstr;

template <typename T, void (*Release)(T*)>
class Ref {
 public:
  Ref() = default;
  Ref(const Ref&) = delete;
  Ref& operator=(const Ref&) = delete;

  ~Ref() { reset(); }

  T* get() const { return value_; }

  T** out() {
    reset();
    return &value_;
  }

  void reset(T* value = nullptr) {
    if (value_) Release(value_);
    value_ = value;
  }

 private:
  // Owned reference released by this wrapper.
  T* value_ = nullptr;
};

static std::string ReadTextFile(const std::string& path) {
  std::ifstream file(path);
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

static std::vector<uint8_t> ReadBinaryFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
                              std::istreambuf_iterator<char>());
}

static iree_status_t CreateSingleBoundaryPlan(
    iree_hal_device_group_t* device_group, id4_pipeline_plan_t** out_plan) {
  id4_pipeline_device_placement_t placement;
  memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("test");
  placement.device_index = 0;
  placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  id4_pipeline_region_plan_t region;
  memset(&region, 0, sizeof(region));
  region.name = IREE_SV("capture.region");
  region.placement_id = 0;
  region.binding_capacity = 2;
  region.local_binding_slot = 1;
  region.local_tensor_alignment = 4;

  id4_pipeline_boundary_tensor_plan_t boundary;
  memset(&boundary, 0, sizeof(boundary));
  boundary.layout.name = IREE_SV("capture.output");
  boundary.layout.dtype = ID4_PIPELINE_TENSOR_DTYPE_F32;
  boundary.layout.shape.rank = 1;
  boundary.layout.shape.dims[0] = 2;
  boundary.layout.byte_length = 8;
  boundary.layout.alignment = 4;
  boundary.flags = ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED |
                   ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED;
  boundary.region_id = 0;
  boundary.placement_id = 0;
  boundary.binding_slot = 0;

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  id4_pipeline_plan_create_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.stage_name = IREE_SV("capture.stage");
  options.device_group = device_group;
  options.placement_count = 1;
  options.placements = &placement;
  options.boundary_tensor_count = 1;
  options.boundary_tensors = &boundary;
  options.region_count = 1;
  options.regions = &region;
  options.diagnostics_sink = &diagnostics_sink;
  return id4_pipeline_plan_create(&options, iree_allocator_system(), out_plan);
}

TEST(FixtureCaptureTest, CapturesExportedBoundaryTensorAfterWait) {
  Ref<iree_hal_device_group_t, iree_hal_device_group_release> device_group;
  device_group.reset(id4::test::CreateLocalSyncDeviceGroup());
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), 0);

  Ref<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(CreateSingleBoundaryPlan(device_group.get(), plan.out()));

  iree_hal_buffer_params_t buffer_params;
  memset(&buffer_params, 0, sizeof(buffer_params));
  buffer_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  buffer_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  Ref<iree_hal_buffer_t, iree_hal_buffer_release> boundary_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), buffer_params, /*allocation_size=*/8,
      boundary_buffer.out()));

  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> fill_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                           fill_semaphore.out()));
  iree_hal_semaphore_t* signal_semaphores[] = {fill_semaphore.get()};
  uint64_t signal_values[] = {1};
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      signal_values,
  };
  const uint32_t one = 0x3F800000u;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal_list, boundary_buffer.get(), /*target_offset=*/0,
      /*length=*/8, &one, sizeof(one), IREE_HAL_FILL_FLAG_NONE));

  iree_hal_buffer_binding_t boundary_binding = {
      // Captured boundary buffer.
      /*.buffer=*/boundary_buffer.get(),
      // Captured boundary base offset.
      /*.offset=*/0,
      // Captured boundary byte length.
      /*.length=*/8,
  };

  iree::testing::TempFilePath capture_root("id4_fixture_capture");
  id4_tooling_capture_execution_options_t capture_options;
  memset(&capture_options, 0, sizeof(capture_options));
  capture_options.structure_size = sizeof(capture_options);
  capture_options.run_id = IREE_SV("unit_run");
  capture_options.output_directory = capture_root.path_view();
  capture_options.plan = plan.get();
  capture_options.device = device;
  capture_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  capture_options.boundary_binding_count = 1;
  capture_options.boundary_bindings = &boundary_binding;
  capture_options.wait_semaphore_list = signal_list;
  capture_options.host_allocator = iree_allocator_system();
  IREE_ASSERT_OK(id4_tooling_capture_execution(&capture_options));

  const std::string manifest =
      ReadTextFile(capture_root.path() + "/manifest.json");
  EXPECT_THAT(manifest, HasSubstr("\"run_id\": \"unit_run\""));
  EXPECT_THAT(manifest, HasSubstr("\"stage\":\"capture.stage\""));
  EXPECT_THAT(manifest, HasSubstr("\"name\":\"capture.output\""));
  EXPECT_THAT(manifest, HasSubstr("\"dtype\":\"f32\""));
  EXPECT_THAT(manifest, HasSubstr("\"shape\":[2]"));

  const std::vector<uint8_t> npy =
      ReadBinaryFile(capture_root.path() + "/boundary_0000.npy");
  ASSERT_GE(npy.size(), 18u);
  EXPECT_EQ(npy[0], 0x93);
  EXPECT_EQ(npy[1], 'N');
  EXPECT_EQ(npy[2], 'U');
  EXPECT_EQ(npy[3], 'M');
  EXPECT_EQ(npy[4], 'P');
  EXPECT_EQ(npy[5], 'Y');
  EXPECT_EQ(npy[npy.size() - 8], 0x00);
  EXPECT_EQ(npy[npy.size() - 7], 0x00);
  EXPECT_EQ(npy[npy.size() - 6], 0x80);
  EXPECT_EQ(npy[npy.size() - 5], 0x3F);
  EXPECT_EQ(npy[npy.size() - 4], 0x00);
  EXPECT_EQ(npy[npy.size() - 3], 0x00);
  EXPECT_EQ(npy[npy.size() - 2], 0x80);
  EXPECT_EQ(npy[npy.size() - 1], 0x3F);
}

}  // namespace
