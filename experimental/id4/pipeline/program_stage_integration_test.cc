// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "experimental/id4/pipeline/program_stage.h"
#include "experimental/id4/tooling/runtime.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <typename T, void (*Release)(T*)>
class OwningRef {
 public:
  OwningRef() = default;
  OwningRef(const OwningRef&) = delete;
  OwningRef& operator=(const OwningRef&) = delete;

  ~OwningRef() { reset(); }

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

class RuntimeContext {
 public:
  RuntimeContext() { std::memset(&value, 0, sizeof(value)); }
  RuntimeContext(const RuntimeContext&) = delete;
  RuntimeContext& operator=(const RuntimeContext&) = delete;

  ~RuntimeContext() {
    if (initialized) {
      id4_tooling_runtime_context_deinitialize(&value);
    }
  }

  // Runtime context value initialized from standard ID4 runtime flags.
  id4_tooling_runtime_context_t value;
  // True after |value| has been initialized and must be deinitialized.
  bool initialized = false;
};

static RuntimeContext& SharedRuntimeContext() {
  static RuntimeContext context;
  if (!context.initialized) {
    id4_tooling_runtime_context_options_t context_options;
    std::memset(&context_options, 0, sizeof(context_options));
    context_options.structure_size = sizeof(context_options);
    context_options.executable_cache_identifier =
        IREE_SV("id4.program_stage.integration");
    IREE_CHECK_OK(id4_tooling_runtime_context_initialize_from_flags(
        &context_options, iree_allocator_system(), &context.value));
    context.initialized = true;
  }
  return context;
}

class ProgramBuilderScope {
 public:
  ProgramBuilderScope() {
    iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                     iree_allocator_system(), &block_pool_);
    id4_pipeline_program_builder_create_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.program_name=*/IREE_SV("test.two_region_add"),
        /*.block_pool=*/&block_pool_,
    };
    IREE_CHECK_OK(id4_pipeline_program_builder_create(
        &options, iree_allocator_system(), &builder_));
  }

  ~ProgramBuilderScope() {
    id4_pipeline_program_builder_destroy(builder_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  id4_pipeline_program_builder_t* builder() { return builder_; }

 private:
  // Arena block pool backing append-only program builder storage.
  iree_arena_block_pool_t block_pool_;
  // Program builder owned by this scope.
  id4_pipeline_program_builder_t* builder_ = nullptr;
};

static iree_status_t AddSplatParameter(iree_io_parameter_index_t* index,
                                       iree_string_view_t key,
                                       iree_host_size_t count, float value) {
  iree_io_parameter_index_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.key = key;
  entry.length = count * sizeof(value);
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
  entry.storage.splat.pattern_length = sizeof(value);
  std::memcpy(entry.storage.splat.pattern, &value, sizeof(value));
  return iree_io_parameter_index_add(index, &entry);
}

static iree_status_t CreateScopedParameterProvider(
    iree_string_view_t scope, iree_string_view_t key, iree_host_size_t count,
    float value, iree_io_parameter_provider_t** out_provider) {
  *out_provider = nullptr;
  OwningRef<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_RETURN_IF_ERROR(AddSplatParameter(index.get(), key, count, value));
  return iree_io_parameter_index_provider_create(
      scope, index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), out_provider);
}

static iree_status_t CreateTwoScopeParameterProvider(
    iree_io_parameter_provider_t** out_provider) {
  *out_provider = nullptr;
  OwningRef<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      first_provider;
  IREE_RETURN_IF_ERROR(CreateScopedParameterProvider(
      IREE_SV("first"), IREE_SV("first.bias"), /*count=*/6, /*value=*/10.0f,
      first_provider.out()));
  OwningRef<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      second_provider;
  IREE_RETURN_IF_ERROR(CreateScopedParameterProvider(
      IREE_SV("second"), IREE_SV("second.bias"), /*count=*/4,
      /*value=*/100.0f, second_provider.out()));
  const id4_tooling_parameter_provider_set_entry_t entries[] = {
      {
          /*.scope=*/IREE_SV("first"),
          /*.provider=*/first_provider.get(),
      },
      {
          /*.scope=*/IREE_SV("second"),
          /*.provider=*/second_provider.get(),
      },
  };
  return id4_tooling_create_parameter_provider_set(
      IREE_ARRAYSIZE(entries), entries, iree_allocator_system(), out_provider);
}

typedef uint32_t id4_pipeline_test_program_flags_t;

enum id4_pipeline_test_program_flag_bits_t {
  ID4_PIPELINE_TEST_PROGRAM_FLAG_AUTHOR_DIAGNOSTIC_TAPS = 1u << 0,
  ID4_PIPELINE_TEST_PROGRAM_FLAG_DEFER_PARAMETER_LOADS = 1u << 1,
  ID4_PIPELINE_TEST_PROGRAM_FLAG_PREFETCH_DEFERRED_PARAMETER_LOADS = 1u << 2,
  ID4_PIPELINE_TEST_PROGRAM_FLAG_REUSE_PARAMETER_SLABS = 1u << 3,
};

typedef struct id4_pipeline_test_program_diagnostics_t {
  // Parameter load events emitted for load-group submissions.
  std::vector<id4_pipeline_parameter_load_diagnostic_t>
      parameter_load_submissions;
} id4_pipeline_test_program_diagnostics_t;

static iree_status_t CaptureProgramDiagnostics(
    void* user_data, const id4_pipeline_diagnostic_event_t* event) {
  auto* diagnostics =
      static_cast<id4_pipeline_test_program_diagnostics_t*>(user_data);
  if (event->parameter_load &&
      iree_string_view_equal(event->key,
                             IREE_SV("parameter_slab.load_group.submit"))) {
    diagnostics->parameter_load_submissions.push_back(*event->parameter_load);
  }
  return iree_ok_status();
}

static id4_pipeline_diagnostics_sink_t ProgramDiagnosticsSink(
    id4_pipeline_test_program_diagnostics_t* diagnostics) {
  return id4_pipeline_diagnostics_sink_t{
      /*.emit=*/CaptureProgramDiagnostics,
      /*.user_data=*/diagnostics,
  };
}

static id4_pipeline_program_t* CreateTwoRegionAddProgram(
    id4_pipeline_test_program_flags_t flags,
    iree_device_size_t first_parameter_source_offset = 0) {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();
  const id4_pipeline_program_shape_t shape =
      id4_pipeline_program_make_shape_rank1(4);

  id4_pipeline_program_tensor_t input = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t input_options = {
      /*.structure_size=*/sizeof(input_options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      /*.name=*/IREE_SV("boundary.input"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/shape,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &input_options, &input));

  const id4_pipeline_program_parameter_source_t first_source[] = {
      {
          /*.source_scope=*/IREE_SV("first"),
          /*.key=*/IREE_SV("first.bias"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
          /*.shape=*/
          id4_pipeline_program_make_shape_rank1(
              first_parameter_source_offset == 0 ? 6 : 7),
      },
  };
  const id4_pipeline_program_parameter_source_span_t first_source_span[] = {
      {
          /*.source_offset=*/first_parameter_source_offset,
          /*.target_offset=*/0,
          /*.length=*/2 * sizeof(float),
      },
      {
          /*.source_offset=*/first_parameter_source_offset + 4 * sizeof(float),
          /*.target_offset=*/2 * sizeof(float),
          /*.length=*/2 * sizeof(float),
      },
  };
  id4_pipeline_program_parameter_options_t first_options = {
      /*.structure_size=*/sizeof(first_options),
      /*.next=*/nullptr,
      /*.encoding=*/ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      /*.source_count=*/IREE_ARRAYSIZE(first_source),
      /*.sources=*/first_source,
      /*.key=*/IREE_SV("first.bias"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/shape,
      /*.source_span_count=*/IREE_ARRAYSIZE(first_source_span),
      /*.source_spans=*/first_source_span,
  };
  id4_pipeline_program_tensor_t first = id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(
      id4_pipeline_program_parameter(builder, &first_options, &first));

  id4_pipeline_program_tensor_t hidden = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_acquire_tensor_options_t hidden_options = {
      /*.structure_size=*/sizeof(hidden_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("transient.hidden"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/shape,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_acquire_tensor(builder, &hidden_options, &hidden));

  const id4_pipeline_kernel_config_binding_t config_bindings[] = {
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.elementwise.add.element_count"), IREE_SV("4")),
  };
  id4_pipeline_program_dispatch_binding_t first_bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(first),
      id4_pipeline_program_write(hidden),
  };
  id4_pipeline_program_dispatch_loom_options_t first_dispatch_options = {
      /*.structure_size=*/sizeof(first_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("add.first"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("elementwise/add_f32"),
                                   IREE_SV("id4_elementwise_add_f32")),
      /*.config_binding_count=*/IREE_ARRAYSIZE(config_bindings),
      /*.config_bindings=*/config_bindings,
      /*.binding_count=*/IREE_ARRAYSIZE(first_bindings),
      /*.bindings=*/first_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &first_dispatch_options));

  if (iree_all_bits_set(
          flags, ID4_PIPELINE_TEST_PROGRAM_FLAG_AUTHOR_DIAGNOSTIC_TAPS)) {
    id4_pipeline_program_tap_options_t tap_options = {
        /*.structure_size=*/sizeof(tap_options),
        /*.next=*/nullptr,
        /*.name=*/IREE_SV("add.first.hidden"),
        /*.tensor=*/hidden,
    };
    IREE_CHECK_OK(id4_pipeline_program_tap(builder, &tap_options));
  }

  id4_pipeline_program_region_cut_options_t cut_options = {
      /*.structure_size=*/sizeof(cut_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("after.first.add"),
  };
  IREE_CHECK_OK(id4_pipeline_program_region_cut(builder, &cut_options));

  const id4_pipeline_program_parameter_source_t second_source[] = {
      {
          /*.source_scope=*/IREE_SV("second"),
          /*.key=*/IREE_SV("second.bias"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
          /*.shape=*/shape,
      },
  };
  id4_pipeline_program_parameter_options_t second_options = {
      /*.structure_size=*/sizeof(second_options),
      /*.next=*/nullptr,
      /*.encoding=*/ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      /*.source_count=*/IREE_ARRAYSIZE(second_source),
      /*.sources=*/second_source,
      /*.key=*/IREE_SV("second.bias"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/shape,
  };
  id4_pipeline_program_tensor_t second = id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(
      id4_pipeline_program_parameter(builder, &second_options, &second));

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t output_options = {
      /*.structure_size=*/sizeof(output_options),
      /*.next=*/nullptr,
      /*.flags=*/0,
      /*.name=*/IREE_SV("boundary.output"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/shape,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &output_options, &output));

  id4_pipeline_program_dispatch_binding_t second_bindings[] = {
      id4_pipeline_program_read(hidden),
      id4_pipeline_program_read(second),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t second_dispatch_options = {
      /*.structure_size=*/sizeof(second_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("add.second"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("elementwise/add_f32"),
                                   IREE_SV("id4_elementwise_add_f32")),
      /*.config_binding_count=*/IREE_ARRAYSIZE(config_bindings),
      /*.config_bindings=*/config_bindings,
      /*.binding_count=*/IREE_ARRAYSIZE(second_bindings),
      /*.bindings=*/second_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &second_dispatch_options));

  if (iree_all_bits_set(
          flags, ID4_PIPELINE_TEST_PROGRAM_FLAG_AUTHOR_DIAGNOSTIC_TAPS)) {
    id4_pipeline_program_tap_options_t tap_options = {
        /*.structure_size=*/sizeof(tap_options),
        /*.next=*/nullptr,
        /*.name=*/IREE_SV("add.second.output"),
        /*.tensor=*/output,
    };
    IREE_CHECK_OK(id4_pipeline_program_tap(builder, &tap_options));
  }

  id4_pipeline_program_export_options_t export_options = {
      /*.structure_size=*/sizeof(export_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("boundary.output"),
      /*.tensor=*/output,
  };
  IREE_CHECK_OK(id4_pipeline_program_export(builder, &export_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  return program;
}

static iree_status_t CreateTwoRegionAddPlan(
    id4_pipeline_program_t* program, RuntimeContext& context,
    id4_pipeline_stage_plan_flags_t flags,
    iree_string_view_list_t diagnostic_tap_names,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_plan_t** out_plan) {
  id4_pipeline_stage_plan_options_t stage_plan_options;
  std::memset(&stage_plan_options, 0, sizeof(stage_plan_options));
  stage_plan_options.structure_size = sizeof(stage_plan_options);
  stage_plan_options.device_index = 0;
  stage_plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  stage_plan_options.flags = flags;
  stage_plan_options.diagnostic_tap_names = diagnostic_tap_names;
  stage_plan_options.diagnostics_sink = diagnostics_sink;

  id4_pipeline_program_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV("test.two_region_add");
  plan_options.stage_options = &stage_plan_options;
  plan_options.program = program;
  plan_options.device_group = context.value.device_group;
  plan_options.parameter_scope = iree_string_view_empty();
  plan_options.alignment = 16;
  return id4_pipeline_program_stage_create_plan(
      &plan_options, iree_allocator_system(), out_plan);
}

static iree_status_t AllocateDeviceBuffer(iree_hal_device_t* device,
                                          iree_hal_buffer_usage_t usage,
                                          iree_device_size_t byte_length,
                                          iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = usage;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  params.min_alignment = 16;
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_hal_semaphore_list_t OneSemaphoreList(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return iree_hal_semaphore_list_t{
      /*.count=*/1,
      /*.semaphores=*/semaphore,
      /*.payload_values=*/payload_value,
  };
}

static void ExpectFloatValues(iree_hal_device_t* device,
                              iree_hal_buffer_t* buffer,
                              const float expected_values[4]) {
  float values[4] = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device, buffer, /*source_offset=*/0, values, sizeof(values),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_FLOAT_EQ(values[0], expected_values[0]);
  EXPECT_FLOAT_EQ(values[1], expected_values[1]);
  EXPECT_FLOAT_EQ(values[2], expected_values[2]);
  EXPECT_FLOAT_EQ(values[3], expected_values[3]);
}

static void RunTwoRegionAddProgram(id4_pipeline_test_program_flags_t flags) {
  const bool captures_diagnostic_taps = iree_all_bits_set(
      flags, ID4_PIPELINE_TEST_PROGRAM_FLAG_AUTHOR_DIAGNOSTIC_TAPS);
  const bool defers_parameter_loads = iree_all_bits_set(
      flags, ID4_PIPELINE_TEST_PROGRAM_FLAG_DEFER_PARAMETER_LOADS);
  const bool prefetches_deferred_parameter_loads = iree_all_bits_set(
      flags, ID4_PIPELINE_TEST_PROGRAM_FLAG_PREFETCH_DEFERRED_PARAMETER_LOADS);
  const bool reuses_parameter_slabs = iree_all_bits_set(
      flags, ID4_PIPELINE_TEST_PROGRAM_FLAG_REUSE_PARAMETER_SLABS);

  RuntimeContext& context = SharedRuntimeContext();

  id4_pipeline_test_program_diagnostics_t diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      ProgramDiagnosticsSink(&diagnostics);

  OwningRef<id4_pipeline_kernel_library_t, id4_pipeline_kernel_library_release>
      kernel_library;
  IREE_ASSERT_OK(id4_tooling_create_embedded_kernel_library(
      iree_allocator_system(), kernel_library.out()));
  OwningRef<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      parameter_provider;
  IREE_ASSERT_OK(CreateTwoScopeParameterProvider(parameter_provider.out()));
  OwningRef<id4_pipeline_program_t, id4_pipeline_program_release> program;
  program.reset(CreateTwoRegionAddProgram(flags));
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("add.first.hidden"),
      IREE_SV("add.second.output"),
  };
  const id4_pipeline_stage_plan_flags_t stage_plan_flags =
      captures_diagnostic_taps
          ? ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS
          : 0;
  const iree_string_view_list_t stage_diagnostic_tap_names =
      captures_diagnostic_taps
          ? iree_string_view_list_t{
                /*.count=*/IREE_ARRAYSIZE(diagnostic_tap_names),
                /*.values=*/diagnostic_tap_names,
            }
          : iree_string_view_list_empty();
  OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(CreateTwoRegionAddPlan(
      program.get(), context, stage_plan_flags, stage_diagnostic_tap_names,
      &diagnostics_sink, plan.out()));
  ASSERT_EQ(id4_pipeline_plan_region_count(plan.get()), 2u);
  iree_host_size_t load_group_count = 0;
  IREE_ASSERT_OK(id4_pipeline_plan_parameter_load_group_count(
      plan.get(), &load_group_count));
  ASSERT_EQ(load_group_count, 2u);

  iree_hal_device_t* device =
      id4_tooling_runtime_context_primary_device(&context.value);
  OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release> prepare_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  iree_hal_semaphore_t* prepare_signal_semaphore = prepare_semaphore.get();
  uint64_t prepare_signal_value = 1;
  iree_hal_semaphore_list_t prepare_signal_list =
      defers_parameter_loads
          ? iree_hal_semaphore_list_empty()
          : OneSemaphoreList(&prepare_signal_semaphore, &prepare_signal_value);

  id4_pipeline_stage_prepare_options_t stage_prepare_options;
  std::memset(&stage_prepare_options, 0, sizeof(stage_prepare_options));
  stage_prepare_options.structure_size = sizeof(stage_prepare_options);
  stage_prepare_options.flags =
      defers_parameter_loads
          ? ID4_PIPELINE_STAGE_PREPARE_FLAG_DEFER_PARAMETER_LOADS_TO_ISSUE
          : 0;
  stage_prepare_options.parameter_provider = parameter_provider.get();
  stage_prepare_options.kernel_library = kernel_library.get();
  stage_prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  stage_prepare_options.signal_semaphore_list = prepare_signal_list;
  stage_prepare_options.command_buffer_mode = context.value.command_buffer_mode;
  stage_prepare_options.diagnostics_sink = &diagnostics_sink;
  id4_pipeline_program_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.stage_name = IREE_SV("test.two_region_add");
  prepare_options.stage_options = &stage_prepare_options;
  prepare_options.plan = plan.get();
  prepare_options.device_group = context.value.device_group;
  prepare_options.kernel_cache = context.value.kernel_cache;
  prepare_options.executable_cache = context.value.executable_cache;
  OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release> bundle;
  OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release> source_bundle;
  if (reuses_parameter_slabs) {
    IREE_ASSERT_OK(id4_pipeline_program_stage_prepare(
        &prepare_options, iree_allocator_system(), source_bundle.out()));
    id4_pipeline_parameter_slab_set_t* source_parameter_slabs =
        id4_pipeline_bundle_parameter_slabs(source_bundle.get());
    ASSERT_NE(source_parameter_slabs, nullptr);
    if (defers_parameter_loads) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                            id4_pipeline_plan_validate_parameter_slabs(
                                plan.get(), source_parameter_slabs));
    } else {
      IREE_ASSERT_OK(id4_pipeline_plan_validate_parameter_slabs(
          plan.get(), source_parameter_slabs));
    }

    id4_pipeline_stage_prepare_options_t reuse_stage_prepare_options;
    std::memset(&reuse_stage_prepare_options, 0,
                sizeof(reuse_stage_prepare_options));
    reuse_stage_prepare_options.structure_size =
        sizeof(reuse_stage_prepare_options);
    reuse_stage_prepare_options.flags =
        ID4_PIPELINE_STAGE_PREPARE_FLAG_REUSE_PARAMETER_SLABS;
    reuse_stage_prepare_options.parameter_slabs = source_parameter_slabs;
    reuse_stage_prepare_options.kernel_library = kernel_library.get();
    reuse_stage_prepare_options.wait_semaphore_list =
        iree_hal_semaphore_list_empty();
    reuse_stage_prepare_options.signal_semaphore_list =
        iree_hal_semaphore_list_empty();
    reuse_stage_prepare_options.command_buffer_mode =
        context.value.command_buffer_mode;
    reuse_stage_prepare_options.diagnostics_sink = &diagnostics_sink;

    prepare_options.stage_options = &reuse_stage_prepare_options;
    if (defers_parameter_loads) {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          id4_pipeline_program_stage_prepare(
              &prepare_options, iree_allocator_system(), bundle.out()));
      return;
    }
    IREE_ASSERT_OK(id4_pipeline_program_stage_prepare(
        &prepare_options, iree_allocator_system(), bundle.out()));
    EXPECT_EQ(id4_pipeline_bundle_parameter_slabs(bundle.get()),
              source_parameter_slabs);
    source_bundle.reset();
  } else {
    IREE_ASSERT_OK(id4_pipeline_program_stage_prepare(
        &prepare_options, iree_allocator_system(), bundle.out()));
  }
  id4_pipeline_parameter_slab_set_t* parameter_slabs =
      id4_pipeline_bundle_parameter_slabs(bundle.get());
  ASSERT_NE(parameter_slabs, nullptr);
  EXPECT_EQ(id4_pipeline_parameter_slab_set_load_group_count(parameter_slabs),
            2u);
  EXPECT_EQ(id4_pipeline_parameter_slab_set_has_deferred_load_context(
                parameter_slabs),
            defers_parameter_loads);
  EXPECT_EQ(
      id4_pipeline_parameter_slab_set_has_resident_buffers(parameter_slabs),
      !defers_parameter_loads);
  EXPECT_EQ(id4_pipeline_bundle_readiness_semaphore_list(bundle.get()).count,
            (defers_parameter_loads || reuses_parameter_slabs) ? 0u : 1u);

  OwningRef<iree_hal_buffer_t, iree_hal_buffer_release> input_buffer;
  IREE_ASSERT_OK(AllocateDeviceBuffer(
      device,
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      /*byte_length=*/4 * sizeof(float), input_buffer.out()));
  OwningRef<iree_hal_buffer_t, iree_hal_buffer_release> output_buffer;
  IREE_ASSERT_OK(AllocateDeviceBuffer(
      device,
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      /*byte_length=*/4 * sizeof(float), output_buffer.out()));
  OwningRef<iree_hal_buffer_t, iree_hal_buffer_release> first_tap_buffer;
  OwningRef<iree_hal_buffer_t, iree_hal_buffer_release> second_tap_buffer;
  if (captures_diagnostic_taps) {
    IREE_ASSERT_OK(AllocateDeviceBuffer(
        device,
        IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
            IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
        /*byte_length=*/4 * sizeof(float), first_tap_buffer.out()));
    IREE_ASSERT_OK(AllocateDeviceBuffer(
        device,
        IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
            IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
        /*byte_length=*/4 * sizeof(float), second_tap_buffer.out()));
  }

  static const float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device, input_values, input_buffer.get(), /*target_offset=*/0,
      sizeof(input_values), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  iree_hal_buffer_binding_t boundary_bindings[] = {
      {
          /*.buffer=*/input_buffer.get(),
          /*.offset=*/0,
          /*.length=*/sizeof(input_values),
      },
      {
          /*.buffer=*/output_buffer.get(),
          /*.offset=*/0,
          /*.length=*/sizeof(input_values),
      },
  };
  iree_hal_buffer_binding_t diagnostic_tap_bindings[] = {
      {
          /*.buffer=*/first_tap_buffer.get(),
          /*.offset=*/0,
          /*.length=*/sizeof(input_values),
      },
      {
          /*.buffer=*/second_tap_buffer.get(),
          /*.offset=*/0,
          /*.length=*/sizeof(input_values),
      },
  };

  OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release> issue_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));
  iree_hal_semaphore_t* issue_signal_semaphore = issue_semaphore.get();
  uint64_t issue_signal_value = 1;
  iree_hal_semaphore_list_t issue_signal_list =
      OneSemaphoreList(&issue_signal_semaphore, &issue_signal_value);

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = IREE_ARRAYSIZE(boundary_bindings);
  issue_options.boundary_bindings = boundary_bindings;
  issue_options.parameter_load_prefetch_region_distance =
      prefetches_deferred_parameter_loads ? 1 : 0;
  if (captures_diagnostic_taps) {
    issue_options.diagnostic_tap_binding_count =
        IREE_ARRAYSIZE(diagnostic_tap_bindings);
    issue_options.diagnostic_tap_bindings = diagnostic_tap_bindings;
  }
  issue_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  issue_options.signal_semaphore_list = issue_signal_list;
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_program_stage_issue(
      IREE_SV("test.two_region_add"), bundle.get(), &issue_options));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      issue_semaphore.get(), issue_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  if (!defers_parameter_loads) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        prepare_semaphore.get(), prepare_signal_value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
  }

  const float expected_output[] = {111.0f, 112.0f, 113.0f, 114.0f};
  ExpectFloatValues(device, output_buffer.get(), expected_output);
  if (captures_diagnostic_taps) {
    const float expected_first_tap[] = {11.0f, 12.0f, 13.0f, 14.0f};
    ExpectFloatValues(device, first_tap_buffer.get(), expected_first_tap);
    ExpectFloatValues(device, second_tap_buffer.get(), expected_output);
  }
}

TEST(ProgramStageIntegration, IssuesMultiRegionProgramWithParameterGroups) {
  RunTwoRegionAddProgram(/*flags=*/0);
}

TEST(ProgramStageIntegration, IssuesMultiRegionProgramWithDeferredParameters) {
  RunTwoRegionAddProgram(ID4_PIPELINE_TEST_PROGRAM_FLAG_DEFER_PARAMETER_LOADS);
}

TEST(ProgramStageIntegration,
     IssuesMultiRegionProgramWithPrefetchedDeferredParameters) {
  RunTwoRegionAddProgram(
      ID4_PIPELINE_TEST_PROGRAM_FLAG_DEFER_PARAMETER_LOADS |
      ID4_PIPELINE_TEST_PROGRAM_FLAG_PREFETCH_DEFERRED_PARAMETER_LOADS);
}

TEST(ProgramStageIntegration, IssuesMultiRegionProgramWithReusedParameters) {
  RunTwoRegionAddProgram(ID4_PIPELINE_TEST_PROGRAM_FLAG_REUSE_PARAMETER_SLABS);
}

TEST(ProgramStageIntegration, RejectsDeferredParameterSlabReuse) {
  RunTwoRegionAddProgram(ID4_PIPELINE_TEST_PROGRAM_FLAG_DEFER_PARAMETER_LOADS |
                         ID4_PIPELINE_TEST_PROGRAM_FLAG_REUSE_PARAMETER_SLABS);
}

TEST(ProgramStageIntegration, RejectsParameterSlabReuseWhenRequestsDiffer) {
  RuntimeContext& context = SharedRuntimeContext();

  id4_pipeline_test_program_diagnostics_t diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      ProgramDiagnosticsSink(&diagnostics);

  OwningRef<id4_pipeline_kernel_library_t, id4_pipeline_kernel_library_release>
      kernel_library;
  IREE_ASSERT_OK(id4_tooling_create_embedded_kernel_library(
      iree_allocator_system(), kernel_library.out()));
  OwningRef<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      parameter_provider;
  IREE_ASSERT_OK(CreateTwoScopeParameterProvider(parameter_provider.out()));

  OwningRef<id4_pipeline_program_t, id4_pipeline_program_release>
      source_program;
  source_program.reset(CreateTwoRegionAddProgram(/*flags=*/0));
  OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> source_plan;
  IREE_ASSERT_OK(CreateTwoRegionAddPlan(
      source_program.get(), context, /*flags=*/0, iree_string_view_list_empty(),
      &diagnostics_sink, source_plan.out()));

  iree_hal_device_t* device =
      id4_tooling_runtime_context_primary_device(&context.value);
  OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release> prepare_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  iree_hal_semaphore_t* prepare_signal_semaphore = prepare_semaphore.get();
  uint64_t prepare_signal_value = 1;
  iree_hal_semaphore_list_t prepare_signal_list =
      OneSemaphoreList(&prepare_signal_semaphore, &prepare_signal_value);

  id4_pipeline_stage_prepare_options_t stage_prepare_options;
  std::memset(&stage_prepare_options, 0, sizeof(stage_prepare_options));
  stage_prepare_options.structure_size = sizeof(stage_prepare_options);
  stage_prepare_options.parameter_provider = parameter_provider.get();
  stage_prepare_options.kernel_library = kernel_library.get();
  stage_prepare_options.signal_semaphore_list = prepare_signal_list;
  stage_prepare_options.command_buffer_mode = context.value.command_buffer_mode;
  stage_prepare_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_program_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.stage_name = IREE_SV("test.two_region_add");
  prepare_options.stage_options = &stage_prepare_options;
  prepare_options.plan = source_plan.get();
  prepare_options.device_group = context.value.device_group;
  prepare_options.kernel_cache = context.value.kernel_cache;
  prepare_options.executable_cache = context.value.executable_cache;
  OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release> source_bundle;
  IREE_ASSERT_OK(id4_pipeline_program_stage_prepare(
      &prepare_options, iree_allocator_system(), source_bundle.out()));
  id4_pipeline_parameter_slab_set_t* source_parameter_slabs =
      id4_pipeline_bundle_parameter_slabs(source_bundle.get());
  ASSERT_NE(source_parameter_slabs, nullptr);
  IREE_ASSERT_OK(id4_pipeline_plan_validate_parameter_slabs(
      source_plan.get(), source_parameter_slabs));

  OwningRef<id4_pipeline_program_t, id4_pipeline_program_release>
      shifted_program;
  shifted_program.reset(
      CreateTwoRegionAddProgram(/*flags=*/0, /*first_parameter_source_offset=*/
                                sizeof(float)));
  OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> shifted_plan;
  IREE_ASSERT_OK(CreateTwoRegionAddPlan(
      shifted_program.get(), context, /*flags=*/0,
      iree_string_view_list_empty(), &diagnostics_sink, shifted_plan.out()));

  const id4_pipeline_parameter_slab_plan_t* source_slab =
      id4_pipeline_plan_parameter_slab_at(source_plan.get(), /*index=*/0);
  const id4_pipeline_parameter_slab_plan_t* shifted_slab =
      id4_pipeline_plan_parameter_slab_at(shifted_plan.get(), /*index=*/0);
  ASSERT_NE(source_slab, nullptr);
  ASSERT_NE(shifted_slab, nullptr);
  EXPECT_EQ(source_slab->byte_length, shifted_slab->byte_length);
  EXPECT_EQ(source_slab->request_count, shifted_slab->request_count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_plan_validate_parameter_slabs(
                            shifted_plan.get(), source_parameter_slabs));
}

TEST(ProgramStageIntegration, IssuesMultiRegionProgramWithDiagnosticTaps) {
  RunTwoRegionAddProgram(ID4_PIPELINE_TEST_PROGRAM_FLAG_AUTHOR_DIAGNOSTIC_TAPS);
}

}  // namespace
