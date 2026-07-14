// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_layout.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "experimental/id4/pipeline/parameter_window.h"
#include "experimental/id4/pipeline/program_plan.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/io/file_handle.h"
#include "iree/io/memory_stream.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using DeviceGroupPtr =
    std::unique_ptr<iree_hal_device_group_t,
                    decltype(&iree_hal_device_group_release)>;
using PlanPtr =
    std::unique_ptr<id4_pipeline_plan_t, decltype(&id4_pipeline_plan_release)>;
using ProgramPtr = std::unique_ptr<id4_pipeline_program_t,
                                   decltype(&id4_pipeline_program_release)>;
using ParameterIndexPtr =
    std::unique_ptr<iree_io_parameter_index_t,
                    decltype(&iree_io_parameter_index_release)>;
using FileHandlePtr = std::unique_ptr<iree_io_file_handle_t,
                                      decltype(&iree_io_file_handle_release)>;
using StreamPtr =
    std::unique_ptr<iree_io_stream_t, decltype(&iree_io_stream_release)>;
using ParameterProviderPtr =
    std::unique_ptr<iree_io_parameter_provider_t,
                    decltype(&iree_io_parameter_provider_release)>;
using ParameterSlabSetPtr =
    std::unique_ptr<id4_pipeline_parameter_slab_set_t,
                    decltype(&id4_pipeline_parameter_slab_set_release)>;
using HalFilePtr =
    std::unique_ptr<iree_hal_file_t, decltype(&iree_hal_file_release)>;
using SemaphorePtr = std::unique_ptr<iree_hal_semaphore_t,
                                     decltype(&iree_hal_semaphore_release)>;

class ProgramBuilderScope {
 public:
  ProgramBuilderScope() {
    iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                     iree_allocator_system(), &block_pool_);
    id4_pipeline_program_builder_create_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.program_name=*/IREE_SV("test.parameter_layout"),
        /*.block_pool=*/&block_pool_,
    };
    IREE_CHECK_OK(id4_pipeline_program_builder_create(
        &options, iree_allocator_system(), &builder_));
  }

  ~ProgramBuilderScope() {
    id4_pipeline_program_builder_destroy(builder_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  id4_pipeline_program_builder_t* builder() const { return builder_; }

 private:
  iree_arena_block_pool_t block_pool_;
  id4_pipeline_program_builder_t* builder_ = nullptr;
};

static ProgramPtr CreateParameterProgram() {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  const id4_pipeline_program_parameter_source_t embedding_sources[] = {
      {
          /*.source_scope=*/IREE_SV("model"),
          /*.key=*/IREE_SV("embedding.table"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          /*.shape=*/id4_pipeline_program_make_shape_rank2(8, 4),
      },
  };
  const id4_pipeline_program_parameter_source_span_t embedding_spans[] = {
      {
          /*.source_offset=*/3 * 4 * sizeof(uint16_t),
          /*.target_offset=*/0,
          /*.length=*/4 * sizeof(uint16_t),
      },
      {
          /*.source_offset=*/5 * 4 * sizeof(uint16_t),
          /*.target_offset=*/4 * sizeof(uint16_t),
          /*.length=*/4 * sizeof(uint16_t),
      },
  };
  id4_pipeline_program_parameter_options_t embedding_options = {
      /*.structure_size=*/sizeof(embedding_options),
      /*.next=*/nullptr,
      /*.encoding=*/ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      /*.source_count=*/IREE_ARRAYSIZE(embedding_sources),
      /*.sources=*/embedding_sources,
      /*.key=*/IREE_SV("embedding.prompt_rows"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(2, 4),
      /*.source_span_count=*/IREE_ARRAYSIZE(embedding_spans),
      /*.source_spans=*/embedding_spans,
  };
  id4_pipeline_program_tensor_t embedding_rows =
      id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(id4_pipeline_program_parameter(builder, &embedding_options,
                                               &embedding_rows));

  const id4_pipeline_program_parameter_source_t linear_sources[] = {
      {
          /*.source_scope=*/IREE_SV("model"),
          /*.key=*/IREE_SV("linear.weight"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
          /*.shape=*/id4_pipeline_program_make_shape_rank2(16, 16),
      },
  };
  id4_pipeline_program_parameter_options_t linear_options = {
      /*.structure_size=*/sizeof(linear_options),
      /*.next=*/nullptr,
      /*.encoding=*/
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
      /*.source_count=*/IREE_ARRAYSIZE(linear_sources),
      /*.sources=*/linear_sources,
      /*.key=*/IREE_SV("linear.weight"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(16, 16),
  };
  id4_pipeline_program_tensor_t linear_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(
      id4_pipeline_program_parameter(builder, &linear_options, &linear_weight));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  return ProgramPtr(program, id4_pipeline_program_release);
}

static ProgramPtr CreateDirectViewParameterProgram(
    id4_pipeline_program_shape_t source_shape,
    id4_pipeline_program_shape_t execution_shape) {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();
  const id4_pipeline_program_parameter_source_t source = {
      /*.source_scope=*/IREE_SV("model"),
      /*.key=*/IREE_SV("conv.weight.packed"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/source_shape,
  };
  const id4_pipeline_program_parameter_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.encoding=*/ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      /*.source_count=*/1,
      /*.sources=*/&source,
      /*.key=*/source.key,
      /*.dtype=*/source.dtype,
      /*.shape=*/execution_shape,
  };
  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(id4_pipeline_program_parameter(builder, &options, &weight));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  return ProgramPtr(program, id4_pipeline_program_release);
}

static PlanPtr CreateParameterPlan(const id4_pipeline_program_t* program,
                                   iree_hal_device_group_t* device_group) {
  const id4_pipeline_device_placement_t placement = {
      /*.role=*/IREE_SV("default"),
      /*.device_index=*/0,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  const iree_hal_buffer_params_t parameter_params =
      id4_pipeline_parameter_slab_device_local_params(
          IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
              IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
          /*min_alignment=*/16);
  iree_hal_buffer_params_t local_params = {};
  local_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  local_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  local_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  local_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  local_params.min_alignment = 16;

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  id4_pipeline_program_plan_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.flags=*/0,
      /*.stage_name=*/IREE_SV("test.parameter_layout"),
      /*.program=*/program,
      /*.device_group=*/device_group,
      /*.placement_count=*/1,
      /*.placements=*/&placement,
      /*.parameter_scope=*/IREE_SV("model"),
      /*.parameter_slab_placement_id=*/0,
      /*.parameter_slab_binding_slot_base=*/0,
      /*.parameter_slab_target_params=*/parameter_params,
      /*.parameter_slab_alignment=*/16,
      /*.parameter_request_alignment=*/16,
      /*.constant_slab_placement_id=*/0,
      /*.constant_slab_binding_slot=*/1,
      /*.constant_slab_target_params=*/parameter_params,
      /*.constant_slab_alignment=*/16,
      /*.constant_request_alignment=*/16,
      /*.kernel_placement_id=*/0,
      /*.region_placement_id=*/0,
      /*.region_local_slab_params=*/local_params,
      /*.region_local_slab_alignment=*/16,
      /*.region_local_tensor_alignment=*/16,
      /*.region_binding_capacity=*/4,
      /*.region_local_binding_slot=*/2,
      /*.region_shared_binding_slot=*/3,
      /*.region_boundary_binding_slot_base=*/1,
      /*.diagnostic_tap_names=*/iree_string_view_list_empty(),
      /*.diagnostic_tap_binding_slot_base=*/1,
      /*.diagnostics_sink=*/&diagnostics_sink,
  };
  id4_pipeline_plan_t* plan = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_create_plan(
      &options, iree_allocator_system(), &plan));
  return PlanPtr(plan, id4_pipeline_plan_release);
}

static const id4_pipeline_parameter_layout_entry_t FindEntry(
    const id4_pipeline_plan_t* plan, iree_string_view_t key) {
  iree_host_size_t entry_count = 0;
  IREE_CHECK_OK(id4_pipeline_parameter_layout_entry_count(plan, &entry_count));
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    id4_pipeline_parameter_layout_entry_t entry;
    IREE_CHECK_OK(id4_pipeline_parameter_layout_entry_at(plan, i, &entry));
    if (iree_string_view_equal(entry.key, key)) return entry;
  }
  ADD_FAILURE() << "parameter layout entry was not found";
  return {};
}

TEST(ParameterLayoutTest, PreservesDynamicSourcesAndBakesExecutionStorage) {
  DeviceGroupPtr device_group(id4::test::CreateLocalSyncDeviceGroup(),
                              iree_hal_device_group_release);
  ProgramPtr program = CreateParameterProgram();
  PlanPtr plan = CreateParameterPlan(program.get(), device_group.get());

  iree_host_size_t entry_count = 0;
  IREE_ASSERT_OK(
      id4_pipeline_parameter_layout_entry_count(plan.get(), &entry_count));
  ASSERT_EQ(entry_count, 2u);
  id4_pipeline_parameter_layout_statistics_t statistics;
  IREE_ASSERT_OK(
      id4_pipeline_parameter_layout_query_statistics(plan.get(), &statistics));
  EXPECT_EQ(statistics.source_entry_count, 1u);
  EXPECT_EQ(statistics.source_byte_length, 8u * 4u * sizeof(uint16_t));
  EXPECT_EQ(statistics.execution_entry_count, 1u);
  EXPECT_EQ(statistics.execution_byte_length, 16u * 16u);

  const id4_pipeline_parameter_layout_entry_t embedding =
      FindEntry(plan.get(), IREE_SV("embedding.table"));
  EXPECT_EQ(embedding.kind, ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_SOURCE);
  EXPECT_EQ(embedding.encoding, ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT);
  EXPECT_EQ(embedding.dtype, ID4_PIPELINE_TENSOR_DTYPE_BF16);
  ASSERT_EQ(embedding.shape.rank, 2u);
  EXPECT_EQ(embedding.shape.dims[0], 8u);
  EXPECT_EQ(embedding.shape.dims[1], 4u);
  EXPECT_EQ(embedding.byte_length, 8u * 4u * sizeof(uint16_t));
  EXPECT_TRUE(iree_string_view_equal(embedding.source_scope, IREE_SV("model")));
  EXPECT_EQ(embedding.parameter_slab_index, IREE_HOST_SIZE_MAX);

  const id4_pipeline_parameter_layout_entry_t linear =
      FindEntry(plan.get(), IREE_SV("linear.weight"));
  EXPECT_EQ(linear.kind, ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_EXECUTION);
  EXPECT_EQ(linear.encoding,
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE);
  EXPECT_EQ(linear.dtype, ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3);
  EXPECT_EQ(linear.byte_length, 16u * 16u);
  EXPECT_NE(linear.parameter_slab_index, IREE_HOST_SIZE_MAX);
  const id4_pipeline_parameter_slab_plan_t* slab =
      id4_pipeline_plan_parameter_slab_at(plan.get(),
                                          linear.parameter_slab_index);
  ASSERT_NE(slab, nullptr);
  EXPECT_LE(linear.parameter_slab_offset + linear.byte_length,
            slab->byte_length);

  iree_io_parameter_archive_builder_t archive_builder;
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_initialize(
      iree_allocator_system(), &archive_builder));
  IREE_ASSERT_OK(id4_pipeline_parameter_layout_add_archive_entries(
      plan.get(), &archive_builder));
  EXPECT_EQ(iree_io_parameter_index_count(archive_builder.index), entry_count);
  IREE_EXPECT_OK(id4_pipeline_parameter_layout_validate_index(
      plan.get(), archive_builder.index));

  iree_io_parameter_index_t* incompatible_index = nullptr;
  IREE_ASSERT_OK(iree_io_parameter_index_create(iree_allocator_system(),
                                                &incompatible_index));
  ParameterIndexPtr incompatible_index_owner(incompatible_index,
                                             iree_io_parameter_index_release);
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const iree_io_parameter_index_entry_t* source_entry = nullptr;
    IREE_ASSERT_OK(
        iree_io_parameter_index_get(archive_builder.index, i, &source_entry));
    iree_io_parameter_index_entry_t copied_entry = *source_entry;
    std::vector<uint8_t> metadata(
        source_entry->metadata.data,
        source_entry->metadata.data + source_entry->metadata.data_length);
    if (iree_string_view_equal(source_entry->key, IREE_SV("linear.weight"))) {
      ASSERT_FALSE(metadata.empty());
      metadata.back() ^= 1u;
      copied_entry.metadata =
          iree_make_const_byte_span(metadata.data(), metadata.size());
    }
    IREE_ASSERT_OK(
        iree_io_parameter_index_add(incompatible_index, &copied_entry));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        id4_pipeline_parameter_layout_validate_index(
                            plan.get(), incompatible_index));

  iree_io_parameter_archive_builder_deinitialize(&archive_builder);
}

TEST(ParameterLayoutTest, DirectStorageSurvivesConsumerReshape) {
  DeviceGroupPtr device_group(id4::test::CreateLocalSyncDeviceGroup(),
                              iree_hal_device_group_release);
  const id4_pipeline_program_shape_t source_shape =
      id4_pipeline_program_make_shape_rank4(8, 3, 3, 4);
  ProgramPtr convolution_program =
      CreateDirectViewParameterProgram(source_shape, source_shape);
  ProgramPtr matrix_program = CreateDirectViewParameterProgram(
      source_shape, id4_pipeline_program_make_shape_rank2(8, 36));
  PlanPtr convolution_plan =
      CreateParameterPlan(convolution_program.get(), device_group.get());
  PlanPtr matrix_plan =
      CreateParameterPlan(matrix_program.get(), device_group.get());

  const id4_pipeline_parameter_layout_entry_t matrix_entry =
      FindEntry(matrix_plan.get(), IREE_SV("conv.weight.packed"));
  ASSERT_EQ(matrix_entry.shape.rank, source_shape.rank);
  for (uint32_t i = 0; i < source_shape.rank; ++i) {
    EXPECT_EQ(matrix_entry.shape.dims[i], source_shape.dims[i]);
  }

  iree_io_parameter_archive_builder_t archive_builder;
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_initialize(
      iree_allocator_system(), &archive_builder));
  IREE_ASSERT_OK(id4_pipeline_parameter_layout_add_archive_entries(
      convolution_plan.get(), &archive_builder));
  IREE_EXPECT_OK(id4_pipeline_parameter_layout_validate_index(
      matrix_plan.get(), archive_builder.index));
  iree_io_parameter_archive_builder_deinitialize(&archive_builder);
}

TEST(ParameterLayoutTest, ExecutionLayoutWindowGathersBakedStorageDirectly) {
  DeviceGroupPtr device_group(id4::test::CreateLocalSyncDeviceGroup(),
                              iree_hal_device_group_release);
  ProgramPtr program = CreateParameterProgram();
  PlanPtr plan = CreateParameterPlan(program.get(), device_group.get());

  const id4_pipeline_parameter_tensor_plan_t* linear_tensor =
      id4_pipeline_plan_parameter_tensor_at(plan.get(), /*index=*/1);
  ASSERT_NE(linear_tensor, nullptr);
  const uint32_t parameter_tensor_ordinal =
      linear_tensor->program_tensor_ordinal;
  id4_pipeline_parameter_window_create_options_t window_options = {
      /*.structure_size=*/sizeof(window_options),
      /*.next=*/nullptr,
      /*.plan=*/plan.get(),
      /*.parameter_tensor_count=*/1,
      /*.parameter_tensor_ordinals=*/&parameter_tensor_ordinal,
  };
  id4_pipeline_parameter_window_t* window = nullptr;
  IREE_ASSERT_OK(id4_pipeline_parameter_window_create(
      &window_options, iree_allocator_system(), &window));
  std::unique_ptr<id4_pipeline_parameter_window_t,
                  decltype(&id4_pipeline_parameter_window_release)>
      window_owner(window, id4_pipeline_parameter_window_release);

  id4_pipeline_parameter_window_execution_layout_schedule_create_options_t
      schedule_options = {
          /*.structure_size=*/sizeof(schedule_options),
          /*.next=*/nullptr,
          /*.plan=*/plan.get(),
          /*.window=*/window,
          /*.source_scope=*/IREE_SV("baked"),
      };
  id4_pipeline_parameter_window_schedule_t* schedule = nullptr;
  IREE_ASSERT_OK(id4_pipeline_parameter_window_execution_layout_schedule_create(
      &schedule_options, iree_allocator_system(), &schedule));
  std::unique_ptr<id4_pipeline_parameter_window_schedule_t,
                  decltype(&id4_pipeline_parameter_window_schedule_release)>
      schedule_owner(schedule, id4_pipeline_parameter_window_schedule_release);

  ASSERT_EQ(id4_pipeline_parameter_window_schedule_load_count(schedule), 1u);
  const id4_pipeline_parameter_slab_load_t* loads =
      id4_pipeline_parameter_window_schedule_loads(schedule);
  ASSERT_NE(loads, nullptr);
  ASSERT_NE(loads[0].request_table, nullptr);
  ASSERT_EQ(loads[0].request_table->count, 1u);
  const id4_pipeline_parameter_request_t& request =
      loads[0].request_table->values[0];
  EXPECT_TRUE(iree_string_view_equal(request.key, IREE_SV("linear.weight")));
  EXPECT_EQ(request.span.parameter_offset, 0u);
  EXPECT_EQ(request.span.buffer_offset, 0u);
  EXPECT_EQ(request.span.length, 16u * 16u);

  ASSERT_EQ(id4_pipeline_parameter_window_schedule_load_step_count(schedule),
            1u);
  const id4_pipeline_parameter_load_step_t* load_steps =
      id4_pipeline_parameter_window_schedule_load_steps(schedule);
  ASSERT_NE(load_steps, nullptr);
  EXPECT_EQ(load_steps[0].kind, ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER);
  EXPECT_TRUE(
      iree_string_view_equal(load_steps[0].source_scope, IREE_SV("baked")));
  EXPECT_EQ(id4_pipeline_parameter_window_schedule_original_load_group_at(
                schedule, 0),
            IREE_HOST_SIZE_MAX);
}

TEST(ParameterLayoutTest, LoadsBakedStorageWithoutReencoding) {
  DeviceGroupPtr device_group(id4::test::CreateLocalSyncDeviceGroup(),
                              iree_hal_device_group_release);
  ProgramPtr program = CreateParameterProgram();
  PlanPtr plan = CreateParameterPlan(program.get(), device_group.get());

  iree_io_parameter_archive_builder_t archive_builder;
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_initialize(
      iree_allocator_system(), &archive_builder));
  IREE_ASSERT_OK(id4_pipeline_parameter_layout_add_archive_entries(
      plan.get(), &archive_builder));
  std::vector<uint8_t> archive_bytes(
      iree_io_parameter_archive_builder_total_size(&archive_builder));

  iree_io_file_handle_t* archive_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(archive_bytes.data(), archive_bytes.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &archive_handle));
  FileHandlePtr archive_handle_owner(archive_handle,
                                     iree_io_file_handle_release);
  iree_io_stream_t* archive_stream = nullptr;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE,
      iree_make_byte_span(archive_bytes.data(), archive_bytes.size()),
      iree_io_stream_release_callback_null(), iree_allocator_system(),
      &archive_stream));
  StreamPtr archive_stream_owner(archive_stream, iree_io_stream_release);
  iree_io_parameter_index_t* archive_index = nullptr;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &archive_index));
  ParameterIndexPtr archive_index_owner(archive_index,
                                        iree_io_parameter_index_release);
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_write(
      &archive_builder, archive_handle, /*file_offset=*/0, archive_stream,
      archive_index));

  const iree_io_parameter_index_entry_t* embedding_archive_entry = nullptr;
  IREE_ASSERT_OK(iree_io_parameter_index_lookup(
      archive_index, IREE_SV("embedding.table"), &embedding_archive_entry));
  for (uint64_t i = 0; i < embedding_archive_entry->length; ++i) {
    archive_bytes[embedding_archive_entry->storage.file.offset + i] =
        static_cast<uint8_t>(i);
  }
  const iree_io_parameter_index_entry_t* linear_archive_entry = nullptr;
  IREE_ASSERT_OK(iree_io_parameter_index_lookup(
      archive_index, IREE_SV("linear.weight"), &linear_archive_entry));
  for (uint64_t i = 0; i < linear_archive_entry->length; ++i) {
    archive_bytes[linear_archive_entry->storage.file.offset + i] =
        static_cast<uint8_t>(0x80u + i % 0x40u);
  }

  iree_io_parameter_provider_t* archive_provider = nullptr;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("baked"), archive_index,
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), &archive_provider));
  ParameterProviderPtr archive_provider_owner(
      archive_provider, iree_io_parameter_provider_release);

  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group.get(), 0);
  ASSERT_NE(device, nullptr);
  iree_hal_semaphore_t* load_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &load_semaphore));
  SemaphorePtr load_semaphore_owner(load_semaphore, iree_hal_semaphore_release);
  uint64_t load_payload_value = 1;
  const iree_hal_semaphore_list_t load_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&load_semaphore,
      /*.payload_values=*/&load_payload_value,
  };
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  id4_pipeline_parameter_layout_load_options_t load_options = {
      /*.structure_size=*/sizeof(load_options),
      /*.next=*/nullptr,
      /*.index=*/archive_index,
      /*.provider=*/archive_provider,
      /*.scope=*/IREE_SV("baked"),
      /*.wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*.signal_semaphore_list=*/load_signal_list,
      /*.diagnostics_sink=*/&diagnostics_sink,
  };
  id4_pipeline_parameter_slab_set_t* parameter_slabs = nullptr;
  IREE_ASSERT_OK(id4_pipeline_parameter_layout_load(
      plan.get(), &load_options, iree_allocator_system(), &parameter_slabs));
  ParameterSlabSetPtr parameter_slabs_owner(
      parameter_slabs, id4_pipeline_parameter_slab_set_release);
  ASSERT_EQ(id4_pipeline_parameter_slab_set_count(parameter_slabs), 1u);

  iree_hal_buffer_t* slab_buffer =
      id4_pipeline_parameter_slab_set_buffer_at(parameter_slabs, 0);
  ASSERT_NE(slab_buffer, nullptr);
  std::vector<uint8_t> slab_bytes(iree_hal_buffer_byte_length(slab_buffer));
  iree_io_file_handle_t* readback_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(slab_bytes.data(), slab_bytes.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &readback_handle));
  FileHandlePtr readback_handle_owner(readback_handle,
                                      iree_io_file_handle_release);
  iree_hal_file_t* readback_file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_WRITE,
      readback_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &readback_file));
  HalFilePtr readback_file_owner(readback_file, iree_hal_file_release);
  iree_hal_semaphore_t* readback_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &readback_semaphore));
  SemaphorePtr readback_semaphore_owner(readback_semaphore,
                                        iree_hal_semaphore_release);
  uint64_t readback_payload_value = 1;
  const iree_hal_semaphore_list_t readback_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&readback_semaphore,
      /*.payload_values=*/&readback_payload_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_write(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, load_signal_list,
      readback_signal_list, slab_buffer, /*source_offset=*/0, readback_file,
      /*target_offset=*/0, slab_bytes.size(), IREE_HAL_WRITE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      readback_semaphore, readback_payload_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  const id4_pipeline_parameter_request_table_t* requests =
      id4_pipeline_plan_parameter_request_table_at(plan.get(), 0);
  ASSERT_NE(requests, nullptr);
  ASSERT_EQ(requests->count, 3u);
  for (iree_host_size_t i = 0; i < requests->count; ++i) {
    const id4_pipeline_parameter_request_t* request = &requests->values[i];
    const iree_io_parameter_index_entry_t* source_entry = nullptr;
    IREE_ASSERT_OK(iree_io_parameter_index_lookup(archive_index, request->key,
                                                  &source_entry));
    const uint8_t* expected = archive_bytes.data() +
                              source_entry->storage.file.offset +
                              request->span.parameter_offset;
    const uint8_t* actual = slab_bytes.data() + request->span.buffer_offset;
    EXPECT_EQ(std::memcmp(actual, expected, request->span.length), 0)
        << "request index " << i;
  }

  iree_io_parameter_archive_builder_t target_builder;
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_initialize(
      iree_allocator_system(), &target_builder));
  IREE_ASSERT_OK(id4_pipeline_parameter_layout_add_archive_entries(
      plan.get(), &target_builder));
  std::vector<uint8_t> target_bytes(
      iree_io_parameter_archive_builder_total_size(&target_builder));
  iree_io_file_handle_t* target_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(target_bytes.data(), target_bytes.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &target_handle));
  FileHandlePtr target_handle_owner(target_handle, iree_io_file_handle_release);
  iree_io_stream_t* target_stream = nullptr;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE,
      iree_make_byte_span(target_bytes.data(), target_bytes.size()),
      iree_io_stream_release_callback_null(), iree_allocator_system(),
      &target_stream));
  StreamPtr target_stream_owner(target_stream, iree_io_stream_release);
  iree_io_parameter_index_t* target_index = nullptr;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &target_index));
  ParameterIndexPtr target_index_owner(target_index,
                                       iree_io_parameter_index_release);
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_write(
      &target_builder, target_handle, /*file_offset=*/0, target_stream,
      target_index));

  iree_io_parameter_provider_t* source_provider = nullptr;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), archive_index,
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), &source_provider));
  ParameterProviderPtr source_provider_owner(
      source_provider, iree_io_parameter_provider_release);
  iree_io_parameter_provider_t* target_provider = nullptr;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("target"), target_index,
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), &target_provider));
  ParameterProviderPtr target_provider_owner(
      target_provider, iree_io_parameter_provider_release);

  iree_hal_semaphore_t* populate_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &populate_semaphore));
  SemaphorePtr populate_semaphore_owner(populate_semaphore,
                                        iree_hal_semaphore_release);
  uint64_t populate_payload_value = 1;
  const iree_hal_semaphore_list_t populate_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&populate_semaphore,
      /*.payload_values=*/&populate_payload_value,
  };
  id4_pipeline_parameter_layout_populate_options_t populate_options = {
      /*.structure_size=*/sizeof(populate_options),
      /*.next=*/nullptr,
      /*.source_provider=*/source_provider,
      /*.target_index=*/target_index,
      /*.target_provider=*/target_provider,
      /*.target_scope=*/IREE_SV("target"),
      /*.parameter_slabs=*/parameter_slabs,
      /*.staging_chunk_byte_capacity=*/16,
      /*.wait_semaphore_list=*/load_signal_list,
      /*.signal_semaphore_list=*/populate_signal_list,
      /*.diagnostics_sink=*/&diagnostics_sink,
  };
  IREE_ASSERT_OK(id4_pipeline_parameter_layout_populate(
      plan.get(), &populate_options, iree_allocator_system()));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      populate_semaphore, populate_payload_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  iree_host_size_t entry_count = 0;
  IREE_ASSERT_OK(
      id4_pipeline_parameter_layout_entry_count(plan.get(), &entry_count));
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    id4_pipeline_parameter_layout_entry_t layout_entry;
    IREE_ASSERT_OK(
        id4_pipeline_parameter_layout_entry_at(plan.get(), i, &layout_entry));
    const iree_io_parameter_index_entry_t* source_entry = nullptr;
    const iree_io_parameter_index_entry_t* target_entry = nullptr;
    IREE_ASSERT_OK(iree_io_parameter_index_lookup(
        archive_index, layout_entry.key, &source_entry));
    IREE_ASSERT_OK(iree_io_parameter_index_lookup(
        target_index, layout_entry.key, &target_entry));
    EXPECT_EQ(
        std::memcmp(archive_bytes.data() + source_entry->storage.file.offset,
                    target_bytes.data() + target_entry->storage.file.offset,
                    layout_entry.byte_length),
        0)
        << "layout entry index " << i;
  }

  iree_io_parameter_archive_builder_deinitialize(&target_builder);
  iree_io_parameter_archive_builder_deinitialize(&archive_builder);
}

}  // namespace
