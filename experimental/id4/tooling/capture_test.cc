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
    iree_hal_device_group_t* device_group, id4_pipeline_tensor_dtype_t dtype,
    uint64_t element_count, iree_device_size_t byte_length,
    iree_device_size_t alignment, id4_pipeline_plan_t** out_plan) {
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
  boundary.layout.dtype = dtype;
  boundary.layout.shape.rank = 1;
  boundary.layout.shape.dims[0] = element_count;
  boundary.layout.byte_length = byte_length;
  boundary.layout.alignment = alignment;
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

static void ExpectFinds(iree_string_view_t value, iree_string_view_t needle) {
  EXPECT_NE(iree_string_view_find(value, needle, 0), IREE_STRING_VIEW_NPOS)
      << "expected to find: " << std::string(needle.data, needle.size);
}

static void CaptureExportedBoundaryTensorAfterWait(
    id4_pipeline_tensor_dtype_t dtype, uint64_t element_count,
    const std::vector<uint8_t>& payload, iree_device_size_t alignment,
    iree_string_view_t expected_dtype_json,
    iree_string_view_t expected_npy_descriptor) {
  Ref<iree_hal_device_group_t, iree_hal_device_group_release> device_group;
  device_group.reset(id4::test::CreateLocalSyncDeviceGroup());
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), 0);

  Ref<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(CreateSingleBoundaryPlan(
      device_group.get(), dtype, element_count,
      static_cast<iree_device_size_t>(payload.size()), alignment, plan.out()));

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
      iree_hal_device_allocator(device), buffer_params,
      static_cast<iree_device_size_t>(payload.size()), boundary_buffer.out()));

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
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal_list, payload.data(), /*source_offset=*/0, boundary_buffer.get(),
      /*target_offset=*/0, static_cast<iree_device_size_t>(payload.size()),
      IREE_HAL_UPDATE_FLAG_NONE));

  iree_hal_buffer_binding_t boundary_binding = {
      // Captured boundary buffer.
      /*.buffer=*/boundary_buffer.get(),
      // Captured boundary base offset.
      /*.offset=*/0,
      // Captured boundary byte length.
      /*.length=*/static_cast<iree_device_size_t>(payload.size()),
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
  const iree_string_view_t manifest_view =
      iree_make_string_view(manifest.data(), manifest.size());
  ExpectFinds(manifest_view, IREE_SV("\"run_id\": \"unit_run\""));
  ExpectFinds(manifest_view, IREE_SV("\"stage\":\"capture.stage\""));
  ExpectFinds(manifest_view, IREE_SV("\"name\":\"capture.output\""));
  ExpectFinds(manifest_view, expected_dtype_json);
  ExpectFinds(manifest_view, IREE_SV("\"shape\":[2]"));

  const std::vector<uint8_t> npy =
      ReadBinaryFile(capture_root.path() + "/boundary_0000.npy");
  ASSERT_GE(npy.size(), 18u);
  const iree_string_view_t npy_view = iree_make_string_view(
      reinterpret_cast<const char*>(npy.data()), npy.size());
  EXPECT_EQ(npy[0], 0x93);
  EXPECT_EQ(npy[1], 'N');
  EXPECT_EQ(npy[2], 'U');
  EXPECT_EQ(npy[3], 'M');
  EXPECT_EQ(npy[4], 'P');
  EXPECT_EQ(npy[5], 'Y');
  ExpectFinds(npy_view, expected_npy_descriptor);
  ASSERT_GE(npy.size(), payload.size());
  for (iree_host_size_t i = 0; i < payload.size(); ++i) {
    EXPECT_EQ(npy[npy.size() - payload.size() + i], payload[i]) << i;
  }
}

TEST(FixtureCaptureTest, CapturesExportedBoundaryTensorAfterWait) {
  CaptureExportedBoundaryTensorAfterWait(
      ID4_PIPELINE_TENSOR_DTYPE_F32, /*element_count=*/2,
      {0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F},
      /*alignment=*/4, IREE_SV("\"dtype\":\"f32\""), IREE_SV("'descr': '<f4'"));
}

TEST(FixtureCaptureTest, CapturesExistingNonF32TensorDtypes) {
  CaptureExportedBoundaryTensorAfterWait(
      ID4_PIPELINE_TENSOR_DTYPE_F16, /*element_count=*/2,
      {0x00, 0x3C, 0x00, 0x40}, /*alignment=*/2, IREE_SV("\"dtype\":\"f16\""),
      IREE_SV("'descr': '<f2'"));
  CaptureExportedBoundaryTensorAfterWait(
      ID4_PIPELINE_TENSOR_DTYPE_BF16, /*element_count=*/2,
      {0x80, 0x3F, 0x00, 0x40}, /*alignment=*/2, IREE_SV("\"dtype\":\"bf16\""),
      IREE_SV("'descr': '<u2'"));
  CaptureExportedBoundaryTensorAfterWait(
      ID4_PIPELINE_TENSOR_DTYPE_U32, /*element_count=*/2,
      {0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00}, /*alignment=*/4,
      IREE_SV("\"dtype\":\"u32\""), IREE_SV("'descr': '<u4'"));
}

}  // namespace
