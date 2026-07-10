// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <vector>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/kernel_library.h"
#include "experimental/id4/pipeline/parameter_slab.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/tooling/runtime.h"
#include "iree/base/internal/math.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

constexpr iree_host_size_t kCompactRhsTileElementCount = 16 * 16;
constexpr iree_host_size_t kCompactRhsTileByteLength =
    kCompactRhsTileElementCount * sizeof(uint16_t);
constexpr iree_host_size_t kCompactFp8RhsTileByteLength =
    kCompactRhsTileElementCount * sizeof(uint8_t);
constexpr iree_host_size_t kFp8RawSweepTileCount = 64;

enum class ParameterLoadIssueMode {
  // Preparation submits parameter load work and signals readiness.
  kEager,
  // Issue submits parameter load groups through a parameter issue context.
  kDeferred,
};

static uint8_t FiniteNonzeroFp8SweepByte(iree_host_size_t ordinal) {
  const uint8_t value = static_cast<uint8_t>(ordinal);
  if ((value & 0x7Fu) == 0) {
    return value == 0 ? 0x01 : 0x81;
  }
  if (value == 0x7Fu) return 0x7E;
  if (value == 0xFFu) return 0xFE;
  return value;
}

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
        IREE_SV("id4.parameter_slab.integration");
    IREE_CHECK_OK(id4_tooling_runtime_context_initialize_from_flags(
        &context_options, iree_allocator_system(), &context.value));
    context.initialized = true;
  }
  return context;
}

class ScopedTempFilePath {
 public:
  explicit ScopedTempFilePath(const char* stem) : path_(stem) {}
  ScopedTempFilePath(const ScopedTempFilePath&) = delete;
  ScopedTempFilePath& operator=(const ScopedTempFilePath&) = delete;

  ~ScopedTempFilePath() {
    if (path_) path_.Remove();
  }

  iree_string_view_t path_view() const { return path_.path_view(); }

 private:
  // Generated path removed when the test scope exits.
  iree::testing::TempFilePath path_;
};

static iree_status_t AddSplatParameter(iree_io_parameter_index_t* index,
                                       iree_string_view_t key, uint64_t length,
                                       iree_const_byte_span_t pattern) {
  if (pattern.data_length == 0 ||
      pattern.data_length > IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "splat parameter pattern length %" PRIhsz
                            " is outside the supported range",
                            pattern.data_length);
  }
  iree_io_parameter_index_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.key = key;
  entry.length = length;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
  entry.storage.splat.pattern_length =
      static_cast<uint8_t>(pattern.data_length);
  std::memcpy(entry.storage.splat.pattern, pattern.data, pattern.data_length);
  return iree_io_parameter_index_add(index, &entry);
}

static iree_status_t AddFileParameter(iree_io_parameter_index_t* index,
                                      iree_string_view_t key, uint64_t length,
                                      iree_io_file_handle_t* file_handle) {
  iree_io_parameter_index_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.key = key;
  entry.length = length;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
  entry.storage.file.handle = file_handle;
  entry.storage.file.offset = 0;
  return iree_io_parameter_index_add(index, &entry);
}

static iree_status_t CreateSparseBf16FileParameterProvider(
    ScopedTempFilePath* temp_path, iree_string_view_t key,
    iree_device_size_t byte_length, iree_host_size_t write_offset_count,
    const iree_device_size_t* write_offsets,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = nullptr;

  OwningRef<iree_io_file_handle_t, iree_io_file_handle_release> file_handle;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_create(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_OVERWRITE | IREE_IO_FILE_MODE_RANDOM_ACCESS |
          IREE_IO_FILE_MODE_ASYNC,
      temp_path->path_view(), byte_length, iree_allocator_system(),
      file_handle.out()));

  OwningRef<iree_io_stream_t, iree_io_stream_release> stream;
  IREE_RETURN_IF_ERROR(iree_io_stream_open(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE,
      file_handle.get(), /*file_offset=*/0, iree_allocator_system(),
      stream.out()));
  const uint16_t bf16_one = 0x3F80u;
  uint16_t values[16];
  for (uint16_t& value : values) {
    value = bf16_one;
  }
  for (iree_host_size_t i = 0; i < write_offset_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_io_stream_seek(
        stream.get(), IREE_IO_STREAM_SEEK_SET, write_offsets[i]));
    IREE_RETURN_IF_ERROR(
        iree_io_stream_write(stream.get(), sizeof(values), values));
  }
  stream.reset();
  IREE_RETURN_IF_ERROR(iree_io_file_handle_flush(file_handle.get()));

  OwningRef<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_RETURN_IF_ERROR(
      AddFileParameter(index.get(), key, byte_length, file_handle.get()));
  return iree_io_parameter_index_provider_create(
      IREE_SV("parameters"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), out_provider);
}

static iree_status_t CreateFileParameterProviderWithContents(
    ScopedTempFilePath* temp_path, iree_string_view_t key,
    iree_const_byte_span_t contents,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = nullptr;

  OwningRef<iree_io_file_handle_t, iree_io_file_handle_release> file_handle;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_create(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_OVERWRITE | IREE_IO_FILE_MODE_RANDOM_ACCESS |
          IREE_IO_FILE_MODE_ASYNC,
      temp_path->path_view(), contents.data_length, iree_allocator_system(),
      file_handle.out()));

  OwningRef<iree_io_stream_t, iree_io_stream_release> stream;
  IREE_RETURN_IF_ERROR(iree_io_stream_open(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE,
      file_handle.get(), /*file_offset=*/0, iree_allocator_system(),
      stream.out()));
  IREE_RETURN_IF_ERROR(
      iree_io_stream_write(stream.get(), contents.data_length, contents.data));
  stream.reset();
  IREE_RETURN_IF_ERROR(iree_io_file_handle_flush(file_handle.get()));

  OwningRef<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_RETURN_IF_ERROR(AddFileParameter(index.get(), key, contents.data_length,
                                        file_handle.get()));
  return iree_io_parameter_index_provider_create(
      IREE_SV("parameters"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), out_provider);
}

static iree_status_t CreateParameterSourceProvider(
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = nullptr;

  OwningRef<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));

  const uint8_t fp8_one = 0x38;
  IREE_RETURN_IF_ERROR(
      AddSplatParameter(index.get(), IREE_SV("weight"), /*length=*/512,
                        iree_make_const_byte_span(&fp8_one, sizeof(fp8_one))));
  IREE_RETURN_IF_ERROR(
      AddSplatParameter(index.get(), IREE_SV("weight.second"), /*length=*/512,
                        iree_make_const_byte_span(&fp8_one, sizeof(fp8_one))));
  const uint32_t f32_two = 0x40000000u;
  IREE_RETURN_IF_ERROR(
      AddSplatParameter(index.get(), IREE_SV("weight_scale"), /*length=*/128,
                        iree_make_const_byte_span(&f32_two, sizeof(f32_two))));
  const uint32_t f32_three = 0x40400000u;
  IREE_RETURN_IF_ERROR(AddSplatParameter(
      index.get(), IREE_SV("weight.second_scale"), /*length=*/128,
      iree_make_const_byte_span(&f32_three, sizeof(f32_three))));

  const uint16_t bf16_one = 0x3F80u;
  IREE_RETURN_IF_ERROR(AddSplatParameter(
      index.get(), IREE_SV("weight.bf16_tile"), /*length=*/512,
      iree_make_const_byte_span(&bf16_one, sizeof(bf16_one))));
  IREE_RETURN_IF_ERROR(
      AddSplatParameter(index.get(), IREE_SV("weight.fp8_tile"), /*length=*/256,
                        iree_make_const_byte_span(&fp8_one, sizeof(fp8_one))));
  char fp8_sweep_key_storage[kFp8RawSweepTileCount][64];
  for (iree_host_size_t i = 0; i < kFp8RawSweepTileCount; ++i) {
    std::snprintf(fp8_sweep_key_storage[i], sizeof(fp8_sweep_key_storage[i]),
                  "weight.fp8_sweep_tile%02" PRIhsz, i);
    uint8_t fp8_sweep_pattern[4];
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(fp8_sweep_pattern); ++j) {
      fp8_sweep_pattern[j] =
          FiniteNonzeroFp8SweepByte(i * IREE_ARRAYSIZE(fp8_sweep_pattern) + j);
    }
    IREE_RETURN_IF_ERROR(AddSplatParameter(
        index.get(), iree_make_cstring_view(fp8_sweep_key_storage[i]),
        /*length=*/256,
        iree_make_const_byte_span(fp8_sweep_pattern,
                                  sizeof(fp8_sweep_pattern))));
  }
  IREE_RETURN_IF_ERROR(AddSplatParameter(
      index.get(), IREE_SV("weight.fp8_tile_scale"), /*length=*/64,
      iree_make_const_byte_span(&f32_two, sizeof(f32_two))));
  const uint32_t f32_one = 0x3F800000u;
  IREE_RETURN_IF_ERROR(AddSplatParameter(
      index.get(), IREE_SV("weight.fp8_sweep_tile_scale"), /*length=*/64,
      iree_make_const_byte_span(&f32_one, sizeof(f32_one))));

  return iree_io_parameter_index_provider_create(
      IREE_SV("parameters"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), out_provider);
}

static iree_status_t PrepareLinearWmmaExecutable(
    RuntimeContext* context, id4_pipeline_kernel_library_t* kernel_library,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_kernel_executable_t** out_executable,
    iree_hal_executable_function_t* out_function) {
  *out_executable = nullptr;
  *out_function = iree_hal_executable_function_invalid();

  const id4_pipeline_kernel_module_t* module = nullptr;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_library_lookup(
      kernel_library, IREE_SV("ideogram4/linear_bf16_f32_wmma"), &module));
  const id4_pipeline_kernel_config_binding_t config_bindings[] = {
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.ideogram4.linear_wmma.token_count"), IREE_SV("16")),
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.ideogram4.linear_wmma.dispatch_token_count"),
          IREE_SV("16")),
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.ideogram4.linear_wmma.input_size"), IREE_SV("16")),
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.ideogram4.linear_wmma.output_size"), IREE_SV("32")),
  };
  id4_pipeline_kernel_cache_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.executable_cache = context->value.executable_cache;
  prepare_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  prepare_options.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_NONE;
  prepare_options.source_identifier = module->source_identifier;
  prepare_options.source_contents = module->source_contents;
  prepare_options.module_path = module->module_path;
  prepare_options.function_name = IREE_SV("id4_ideogram4_linear_bf16_f32_wmma");
  prepare_options.config_binding_count = IREE_ARRAYSIZE(config_bindings);
  prepare_options.config_bindings = config_bindings;
  prepare_options.diagnostics_sink = diagnostics_sink;

  id4_pipeline_kernel_executable_t* executable = nullptr;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_prepare_executable(
      context->value.kernel_cache, &prepare_options, &executable));
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  iree_status_t status = iree_hal_executable_lookup_function_by_name(
      id4_pipeline_kernel_executable_hal_executable(executable),
      prepare_options.function_name, &function);
  if (iree_status_is_ok(status)) {
    *out_executable = executable;
    *out_function = function;
    executable = nullptr;
  }
  id4_pipeline_kernel_executable_release(executable);
  return status;
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

static iree_hal_semaphore_list_t MakeOneSemaphoreList(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  iree_hal_semaphore_list_t list;
  std::memset(&list, 0, sizeof(list));
  list.count = 1;
  list.semaphores = semaphore;
  list.payload_values = payload_value;
  return list;
}

typedef struct FailureDiagnosticCapture {
  // True after observing the parameter load group failure event.
  bool saw_failure_event;
  // True when the failure event names the failed load step.
  bool saw_failure_step_name;
} FailureDiagnosticCapture;

static iree_status_t CaptureFailureDiagnostic(
    void* user_data, const id4_pipeline_diagnostic_event_t* event) {
  auto* capture = static_cast<FailureDiagnosticCapture*>(user_data);
  if (!iree_string_view_equal(event->key,
                              IREE_SV("parameter_slab.load_group.failure"))) {
    return iree_ok_status();
  }
  capture->saw_failure_event = true;
  capture->saw_failure_step_name =
      event->parameter_load &&
      iree_string_view_equal(event->parameter_load->first_load_step_name,
                             IREE_SV("parameters.gather.weight"));
  return iree_ok_status();
}

static void ExpectDecodedFp8Pattern(const uint16_t* encoded_weight,
                                    iree_host_size_t element_offset,
                                    const uint8_t* pattern,
                                    iree_host_size_t pattern_length) {
  for (iree_host_size_t i = 0; i < 256; ++i) {
    const uint8_t fp8_bits = pattern[i % pattern_length];
    const uint16_t expected_bits =
        iree_math_f32_to_bf16(iree_math_f8e4m3fn_to_f32(fp8_bits));
    ASSERT_EQ(encoded_weight[element_offset + i], expected_bits)
        << "element_offset=" << element_offset << " i=" << i << " fp8_bits=0x"
        << std::hex << static_cast<uint32_t>(fp8_bits) << " actual=0x"
        << encoded_weight[element_offset + i] << " expected=0x"
        << expected_bits;
  }
}

static void RunCompactLinearRhsTileEncoding(
    RuntimeContext* context, id4_pipeline_kernel_library_t* kernel_library,
    iree_io_parameter_provider_t* provider,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink);
static void RunCompactFp8LinearRhsTileEncoding(
    RuntimeContext* context, id4_pipeline_kernel_library_t* kernel_library,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink);

TEST(ParameterSlabIntegration, CheckLoadGroupFailuresReportsFailedSemaphore) {
  RuntimeContext& context = SharedRuntimeContext();

  FailureDiagnosticCapture capture;
  std::memset(&capture, 0, sizeof(capture));
  id4_pipeline_diagnostics_sink_t diagnostics_sink = {
      // Captures parameter load failure diagnostic events.
      .emit = CaptureFailureDiagnostic,
      // Mutable capture state owned by this test.
      .user_data = &capture,
  };

  OwningRef<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(CreateParameterSourceProvider(provider.out()));

  id4_pipeline_device_placement_t placement;
  std::memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = 0;
  placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  id4_pipeline_parameter_request_t target_request =
      id4_pipeline_parameter_request(
          IREE_SV("weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0, /*length=*/16));
  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/0, IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
          /*byte_length=*/16, /*alignment=*/16);
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(/*count=*/1, &target_request);
  id4_pipeline_parameter_load_step_t load_step =
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("parameters.gather.weight"), IREE_SV("parameters"),
          /*target_slab_index=*/0, /*request_offset=*/0, /*request_count=*/1);

  id4_pipeline_plan_create_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV("parameter_slab.failure_check");
  plan_options.device_group = context.value.device_group;
  plan_options.diagnostics_sink = &diagnostics_sink;
  plan_options.placement_count = 1;
  plan_options.placements = &placement;
  plan_options.parameter_slab_count = 1;
  plan_options.parameter_slabs = &slab;
  plan_options.parameter_request_tables = &request_table;
  plan_options.parameter_load_step_count = 1;
  plan_options.parameter_load_steps = &load_step;
  OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(id4_pipeline_plan_create(&plan_options,
                                          iree_allocator_system(), plan.out()));

  id4_pipeline_parameter_slab_set_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.provider = provider.get();
  load_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  load_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
  load_options.diagnostics_sink = &diagnostics_sink;
  OwningRef<id4_pipeline_parameter_slab_set_t,
            id4_pipeline_parameter_slab_set_release>
      slab_set;
  IREE_ASSERT_OK(id4_pipeline_plan_prepare_parameter_slabs(
      plan.get(), &load_options, iree_allocator_system(), slab_set.out()));
  ASSERT_EQ(id4_pipeline_parameter_slab_set_load_group_count(slab_set.get()),
            1u);

  iree_hal_semaphore_t* ready_semaphore = nullptr;
  uint64_t ready_value = 0;
  IREE_ASSERT_OK(id4_pipeline_parameter_slab_set_load_group_ready_at(
      slab_set.get(), /*index=*/0, &ready_semaphore, &ready_value));
  ASSERT_NE(ready_semaphore, nullptr);
  EXPECT_EQ(ready_value, 1u);
  iree_hal_semaphore_fail(ready_semaphore,
                          iree_make_status(IREE_STATUS_ABORTED,
                                           "synthetic parameter load failure"));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      id4_pipeline_parameter_slab_set_check_load_group_failures(
          slab_set.get(), plan_options.stage_name, &diagnostics_sink));
  EXPECT_TRUE(capture.saw_failure_event);
  EXPECT_TRUE(capture.saw_failure_step_name);
}

TEST(ParameterSlabIntegration, EncodedFp8WeightsFeedBf16WmmaLinear) {
  RuntimeContext& context = SharedRuntimeContext();

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  OwningRef<id4_pipeline_kernel_library_t, id4_pipeline_kernel_library_release>
      kernel_library;
  IREE_ASSERT_OK(id4_tooling_create_embedded_kernel_library(
      iree_allocator_system(), kernel_library.out()));
  OwningRef<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(CreateParameterSourceProvider(provider.out()));

  id4_pipeline_device_placement_t placement;
  std::memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = 0;
  placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  id4_pipeline_parameter_request_t target_requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV("weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0, /*length=*/1024)),
      id4_pipeline_parameter_request(
          IREE_SV("weight.second"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/1024,
                                      /*length=*/1024)),
  };
  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/0, IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
              IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
              IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
          /*byte_length=*/2048, /*alignment=*/16);
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(IREE_ARRAYSIZE(target_requests),
                                                target_requests);

  id4_pipeline_tensor_shape_t weight_shape;
  std::memset(&weight_shape, 0, sizeof(weight_shape));
  weight_shape.rank = 2;
  weight_shape.dims[0] = 32;
  weight_shape.dims[1] = 16;
  id4_pipeline_tensor_shape_t scale_shape;
  std::memset(&scale_shape, 0, sizeof(scale_shape));
  scale_shape.rank = 1;
  scale_shape.dims[0] = 32;
  const id4_pipeline_parameter_load_source_t first_sources[] = {
      id4_pipeline_parameter_load_source(
          IREE_SV("parameters"), IREE_SV("weight"),
          ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3, weight_shape,
          /*byte_length=*/512),
      id4_pipeline_parameter_load_source(
          IREE_SV("parameters"), IREE_SV("weight_scale"),
          ID4_PIPELINE_TENSOR_DTYPE_F32, scale_shape,
          /*byte_length=*/128),
  };
  const id4_pipeline_parameter_load_source_t second_sources[] = {
      id4_pipeline_parameter_load_source(
          IREE_SV("parameters"), IREE_SV("weight.second"),
          ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3, weight_shape,
          /*byte_length=*/512),
      id4_pipeline_parameter_load_source(
          IREE_SV("parameters"), IREE_SV("weight.second_scale"),
          ID4_PIPELINE_TENSOR_DTYPE_F32, scale_shape,
          /*byte_length=*/128),
  };
  id4_pipeline_parameter_load_step_t load_steps[] = {
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("parameters.encode_fp8.first"), IREE_ARRAYSIZE(first_sources),
          first_sources,
          /*target_slab_index=*/0, /*request_offset=*/0),
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("parameters.encode_fp8.second"),
          IREE_ARRAYSIZE(second_sources), second_sources,
          /*target_slab_index=*/0, /*request_offset=*/1),
  };

  id4_pipeline_plan_create_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV("parameter_slab");
  plan_options.device_group = context.value.device_group;
  plan_options.diagnostics_sink = &diagnostics_sink;
  plan_options.placement_count = 1;
  plan_options.placements = &placement;
  plan_options.parameter_slab_count = 1;
  plan_options.parameter_slabs = &slab;
  plan_options.parameter_request_tables = &request_table;
  plan_options.parameter_load_step_count = IREE_ARRAYSIZE(load_steps);
  plan_options.parameter_load_steps = load_steps;
  OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(id4_pipeline_plan_create(&plan_options,
                                          iree_allocator_system(), plan.out()));

  OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release> prepare_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      id4_tooling_runtime_context_primary_device(&context.value),
      IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  iree_hal_semaphore_t* prepare_signal_semaphore = prepare_semaphore.get();
  uint64_t prepare_signal_value = 1;
  iree_hal_semaphore_list_t prepare_signal_list =
      MakeOneSemaphoreList(&prepare_signal_semaphore, &prepare_signal_value);

  id4_pipeline_parameter_slab_set_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.encoder_staging_chunk_byte_capacity =
      ID4_PIPELINE_PARAMETER_ENCODER_DEFAULT_STAGING_CHUNK_BYTE_CAPACITY;
  load_options.encoder_staging_memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  load_options.provider = provider.get();
  load_options.kernel_library = kernel_library.get();
  load_options.kernel_cache = context.value.kernel_cache;
  load_options.executable_cache = context.value.executable_cache;
  load_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  load_options.signal_semaphore_list = prepare_signal_list;
  load_options.diagnostics_sink = &diagnostics_sink;
  OwningRef<id4_pipeline_parameter_slab_set_t,
            id4_pipeline_parameter_slab_set_release>
      slab_set;
  IREE_ASSERT_OK(id4_pipeline_plan_load_parameter_slabs(
      plan.get(), &load_options, iree_allocator_system(), slab_set.out()));
  ASSERT_EQ(id4_pipeline_parameter_slab_set_load_group_count(slab_set.get()),
            1u);
  iree_hal_semaphore_t* load_group_ready_semaphore = nullptr;
  uint64_t load_group_ready_value = 0;
  IREE_ASSERT_OK(id4_pipeline_parameter_slab_set_load_group_ready_at(
      slab_set.get(), /*index=*/0, &load_group_ready_semaphore,
      &load_group_ready_value));
  ASSERT_NE(load_group_ready_semaphore, nullptr);
  EXPECT_EQ(load_group_ready_value, 1u);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      load_group_ready_semaphore, load_group_ready_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      prepare_semaphore.get(), prepare_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  uint16_t encoded_weight[1024] = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      id4_tooling_runtime_context_primary_device(&context.value),
      id4_pipeline_parameter_slab_set_buffer_at(slab_set.get(), 0),
      /*source_offset=*/0, encoded_weight, sizeof(encoded_weight),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  for (iree_host_size_t i = 0; i < 512; ++i) {
    EXPECT_EQ(encoded_weight[i], 0x4000u);
  }
  for (iree_host_size_t i = 512; i < IREE_ARRAYSIZE(encoded_weight); ++i) {
    EXPECT_EQ(encoded_weight[i], 0x4040u);
  }

  iree_hal_device_t* device =
      id4_tooling_runtime_context_primary_device(&context.value);
  OwningRef<iree_hal_buffer_t, iree_hal_buffer_release> input_buffer;
  IREE_ASSERT_OK(AllocateDeviceBuffer(
      device,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      /*byte_length=*/16 * 16 * sizeof(uint16_t), input_buffer.out()));
  OwningRef<iree_hal_buffer_t, iree_hal_buffer_release> output_buffer;
  IREE_ASSERT_OK(AllocateDeviceBuffer(
      device,
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      /*byte_length=*/16 * 32 * sizeof(float), output_buffer.out()));

  uint16_t input[16 * 16];
  for (uint16_t& value : input) {
    value = 0x3f80u;
  }
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device, input, input_buffer.get(), /*target_offset=*/0, sizeof(input),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  OwningRef<id4_pipeline_kernel_executable_t,
            id4_pipeline_kernel_executable_release>
      linear_executable;
  iree_hal_executable_function_t linear_function =
      iree_hal_executable_function_invalid();
  IREE_ASSERT_OK(PrepareLinearWmmaExecutable(
      &context, kernel_library.get(), &diagnostics_sink,
      linear_executable.out(), &linear_function));

  iree_hal_buffer_ref_t bindings[3];
  bindings[0] = iree_hal_make_buffer_ref(input_buffer.get(), 0,
                                         16 * 16 * sizeof(input[0]));
  bindings[1] = iree_hal_make_buffer_ref(
      id4_pipeline_parameter_slab_set_buffer_at(slab_set.get(), 0), 0, 1024);
  bindings[2] =
      iree_hal_make_buffer_ref(output_buffer.get(), 0, 16 * 32 * sizeof(float));
  iree_hal_buffer_ref_list_t binding_list;
  std::memset(&binding_list, 0, sizeof(binding_list));
  binding_list.count = IREE_ARRAYSIZE(bindings);
  binding_list.values = bindings;

  OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      dispatch_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, dispatch_semaphore.out()));
  iree_hal_semaphore_t* dispatch_signal_semaphore = dispatch_semaphore.get();
  uint64_t dispatch_signal_value = 1;
  iree_hal_semaphore_list_t dispatch_signal_list =
      MakeOneSemaphoreList(&dispatch_signal_semaphore, &dispatch_signal_value);

  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal_list,
      id4_pipeline_kernel_executable_hal_executable(linear_executable.get()),
      linear_function,
      id4_pipeline_kernel_executable_dispatch_config(linear_executable.get()),
      iree_const_byte_span_empty(), binding_list, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      dispatch_semaphore.get(), dispatch_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  float linear_output[16 * 32] = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device, output_buffer.get(), /*source_offset=*/0, linear_output,
      sizeof(linear_output), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  for (float value : linear_output) {
    EXPECT_EQ(value, 32.0f);
  }

  RunCompactLinearRhsTileEncoding(&context, kernel_library.get(),
                                  provider.get(), &diagnostics_sink);
  RunCompactFp8LinearRhsTileEncoding(&context, kernel_library.get(),
                                     &diagnostics_sink);
}

static void RunFileBackedDirectGatherManySmall(
    ParameterLoadIssueMode issue_mode) {
  static constexpr iree_host_size_t kRequestCount = 171;
  static constexpr iree_device_size_t kRequestByteLength = 8192;
  const iree_string_view_t kParameterKey = IREE_SV("token_embedding.weight");

  RuntimeContext& context = SharedRuntimeContext();

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  const iree_device_size_t target_byte_length =
      kRequestCount * kRequestByteLength;
  std::vector<uint8_t> source_data((size_t)kRequestByteLength);
  for (iree_host_size_t i = 0; i < source_data.size(); ++i) {
    source_data[i] = static_cast<uint8_t>((i * 13 + (i >> 5) * 7) & 0xFF);
  }

  ScopedTempFilePath source_path("id4_parameter_slab_direct_gather");
  OwningRef<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(CreateFileParameterProviderWithContents(
      &source_path, kParameterKey,
      iree_make_const_byte_span(source_data.data(), source_data.size()),
      provider.out()));

  id4_pipeline_device_placement_t placement;
  std::memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = 0;
  placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  id4_pipeline_parameter_request_t target_requests[kRequestCount];
  for (iree_host_size_t i = 0; i < kRequestCount; ++i) {
    target_requests[i] = id4_pipeline_parameter_request(
        kParameterKey, id4_pipeline_parameter_span(
                           /*parameter_offset=*/0,
                           /*buffer_offset=*/i * kRequestByteLength,
                           /*length=*/kRequestByteLength));
  }
  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/0, IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
              IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
              IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
          target_byte_length, /*alignment=*/16);
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(kRequestCount, target_requests);
  id4_pipeline_parameter_load_step_t load_step =
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("parameters.gather.token_embedding"), IREE_SV("parameters"),
          /*target_slab_index=*/0, /*request_offset=*/0, kRequestCount);

  id4_pipeline_plan_create_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV("parameter_slab.direct_gather");
  plan_options.device_group = context.value.device_group;
  plan_options.diagnostics_sink = &diagnostics_sink;
  plan_options.placement_count = 1;
  plan_options.placements = &placement;
  plan_options.parameter_slab_count = 1;
  plan_options.parameter_slabs = &slab;
  plan_options.parameter_request_tables = &request_table;
  plan_options.parameter_load_step_count = 1;
  plan_options.parameter_load_steps = &load_step;
  OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(id4_pipeline_plan_create(&plan_options,
                                          iree_allocator_system(), plan.out()));

  iree_hal_device_t* device =
      id4_tooling_runtime_context_primary_device(&context.value);
  OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release> prepare_semaphore;
  const bool uses_deferred_issue =
      issue_mode == ParameterLoadIssueMode::kDeferred;
  if (!uses_deferred_issue) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  }
  iree_hal_semaphore_t* prepare_signal_semaphore =
      uses_deferred_issue ? nullptr : prepare_semaphore.get();
  uint64_t prepare_signal_value = 1;
  iree_hal_semaphore_list_t prepare_signal_list =
      uses_deferred_issue ? iree_hal_semaphore_list_empty()
                          : MakeOneSemaphoreList(&prepare_signal_semaphore,
                                                 &prepare_signal_value);

  id4_pipeline_parameter_slab_set_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.provider = provider.get();
  load_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  load_options.signal_semaphore_list = prepare_signal_list;
  load_options.diagnostics_sink = &diagnostics_sink;

  OwningRef<id4_pipeline_parameter_slab_set_t,
            id4_pipeline_parameter_slab_set_release>
      slab_set;
  if (uses_deferred_issue) {
    IREE_ASSERT_OK(id4_pipeline_plan_prepare_parameter_slabs(
        plan.get(), &load_options, iree_allocator_system(), slab_set.out()));
    OwningRef<id4_pipeline_parameter_slab_issue_context_t,
              id4_pipeline_parameter_slab_issue_context_release>
        issue_context;
    IREE_ASSERT_OK(id4_pipeline_plan_create_parameter_slab_issue_context(
        plan.get(), slab_set.get(), iree_allocator_system(),
        issue_context.out()));
    IREE_ASSERT_OK(id4_pipeline_plan_submit_parameter_load_group(
        plan.get(), issue_context.get(), /*group_index=*/0,
        /*submit_region_id=*/0, &diagnostics_sink));
    iree_hal_semaphore_list_t cleanup_wait_list =
        iree_hal_semaphore_list_empty();
    IREE_ASSERT_OK(id4_pipeline_parameter_slab_issue_context_finish(
        issue_context.get(), &cleanup_wait_list));
    for (iree_host_size_t i = 0; i < cleanup_wait_list.count; ++i) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(
          cleanup_wait_list.semaphores[i], cleanup_wait_list.payload_values[i],
          iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    }
  } else {
    IREE_ASSERT_OK(id4_pipeline_plan_load_parameter_slabs(
        plan.get(), &load_options, iree_allocator_system(), slab_set.out()));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        prepare_semaphore.get(), prepare_signal_value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
  }

  ASSERT_EQ(id4_pipeline_parameter_slab_set_load_group_count(slab_set.get()),
            1u);
  iree_hal_semaphore_t* group_ready_semaphore = nullptr;
  uint64_t group_ready_value = 0;
  IREE_ASSERT_OK(id4_pipeline_parameter_slab_set_load_group_ready_at(
      slab_set.get(), /*index=*/0, &group_ready_semaphore, &group_ready_value));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      group_ready_semaphore, group_ready_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(id4_pipeline_parameter_slab_set_check_load_group_failures(
      slab_set.get(), plan_options.stage_name, &diagnostics_sink));

  std::vector<uint8_t> expected((size_t)target_byte_length);
  for (iree_host_size_t i = 0; i < kRequestCount; ++i) {
    std::memcpy(expected.data() + (size_t)(i * kRequestByteLength),
                source_data.data(), (size_t)kRequestByteLength);
  }
  std::vector<uint8_t> actual((size_t)target_byte_length);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device, id4_pipeline_parameter_slab_set_buffer_at(slab_set.get(), 0),
      /*source_offset=*/0, actual.data(), actual.size(),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_EQ(0, std::memcmp(actual.data(), expected.data(), expected.size()));
}

TEST(ParameterSlabIntegration, FileBackedDirectGatherManySmallEager) {
  RunFileBackedDirectGatherManySmall(ParameterLoadIssueMode::kEager);
}

TEST(ParameterSlabIntegration, FileBackedDirectGatherManySmallDeferredIssue) {
  RunFileBackedDirectGatherManySmall(ParameterLoadIssueMode::kDeferred);
}

static void RunCompactLinearRhsTileEncoding(
    RuntimeContext* context, id4_pipeline_kernel_library_t* kernel_library,
    iree_io_parameter_provider_t* provider,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_device_placement_t placement;
  std::memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = 0;
  placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  constexpr iree_host_size_t kTargetRequestCount = 2 + kFp8RawSweepTileCount;
  id4_pipeline_parameter_request_t target_requests[kTargetRequestCount];
  target_requests[0] = id4_pipeline_parameter_request(
      IREE_SV("weight.bf16_tile"),
      id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                  /*buffer_offset=*/0,
                                  /*length=*/kCompactRhsTileByteLength));
  target_requests[1] = id4_pipeline_parameter_request(
      IREE_SV("weight.fp8_tile"),
      id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                  /*buffer_offset=*/
                                  kCompactRhsTileByteLength,
                                  /*length=*/kCompactRhsTileByteLength));
  char fp8_sweep_key_storage[kFp8RawSweepTileCount][64];
  for (iree_host_size_t i = 0; i < kFp8RawSweepTileCount; ++i) {
    std::snprintf(fp8_sweep_key_storage[i], sizeof(fp8_sweep_key_storage[i]),
                  "weight.fp8_sweep_tile%02" PRIhsz, i);
    target_requests[2 + i] = id4_pipeline_parameter_request(
        iree_make_cstring_view(fp8_sweep_key_storage[i]),
        id4_pipeline_parameter_span(
            /*parameter_offset=*/0,
            /*buffer_offset=*/(2 + i) * kCompactRhsTileByteLength,
            /*length=*/kCompactRhsTileByteLength));
  }
  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/0, IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
              IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
              IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
          /*byte_length=*/kTargetRequestCount * kCompactRhsTileByteLength,
          /*alignment=*/16);
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(kTargetRequestCount,
                                                target_requests);

  id4_pipeline_tensor_shape_t weight_shape;
  std::memset(&weight_shape, 0, sizeof(weight_shape));
  weight_shape.rank = 2;
  weight_shape.dims[0] = 16;
  weight_shape.dims[1] = 16;
  id4_pipeline_tensor_shape_t scale_shape;
  std::memset(&scale_shape, 0, sizeof(scale_shape));
  scale_shape.rank = 1;
  scale_shape.dims[0] = 16;
  const id4_pipeline_parameter_load_source_t bf16_sources[] = {
      id4_pipeline_parameter_load_source(
          IREE_SV("parameters"), IREE_SV("weight.bf16_tile"),
          ID4_PIPELINE_TENSOR_DTYPE_BF16, weight_shape,
          /*byte_length=*/512),
  };
  const id4_pipeline_parameter_load_source_t fp8_sources[] = {
      id4_pipeline_parameter_load_source(
          IREE_SV("parameters"), IREE_SV("weight.fp8_tile"),
          ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3, weight_shape,
          /*byte_length=*/256),
      id4_pipeline_parameter_load_source(
          IREE_SV("parameters"), IREE_SV("weight.fp8_tile_scale"),
          ID4_PIPELINE_TENSOR_DTYPE_F32, scale_shape,
          /*byte_length=*/64),
  };
  id4_pipeline_parameter_load_source_t fp8_sweep_sources[kFp8RawSweepTileCount]
                                                        [2];
  char fp8_sweep_step_name_storage[kFp8RawSweepTileCount][80];
  id4_pipeline_parameter_load_step_t load_steps[kTargetRequestCount];
  load_steps[0] = id4_pipeline_parameter_encode_bf16_linear_rhs_tile_load_step(
      IREE_SV("parameters.encode_bf16_tile"), IREE_ARRAYSIZE(bf16_sources),
      bf16_sources,
      /*target_slab_index=*/0, /*request_offset=*/0);
  load_steps[1] =
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_linear_rhs_tile_load_step(
          IREE_SV("parameters.encode_fp8_tile"), IREE_ARRAYSIZE(fp8_sources),
          fp8_sources,
          /*target_slab_index=*/0, /*request_offset=*/1);
  for (iree_host_size_t i = 0; i < kFp8RawSweepTileCount; ++i) {
    fp8_sweep_sources[i][0] = id4_pipeline_parameter_load_source(
        IREE_SV("parameters"), iree_make_cstring_view(fp8_sweep_key_storage[i]),
        ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3, weight_shape,
        /*byte_length=*/256);
    fp8_sweep_sources[i][1] = id4_pipeline_parameter_load_source(
        IREE_SV("parameters"), IREE_SV("weight.fp8_sweep_tile_scale"),
        ID4_PIPELINE_TENSOR_DTYPE_F32, scale_shape,
        /*byte_length=*/64);
    std::snprintf(fp8_sweep_step_name_storage[i],
                  sizeof(fp8_sweep_step_name_storage[i]),
                  "parameters.encode_fp8_sweep_tile%02" PRIhsz, i);
    load_steps[2 + i] =
        id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_linear_rhs_tile_load_step(
            iree_make_cstring_view(fp8_sweep_step_name_storage[i]),
            IREE_ARRAYSIZE(fp8_sweep_sources[i]), fp8_sweep_sources[i],
            /*target_slab_index=*/0, /*request_offset=*/2 + i);
  }

  id4_pipeline_plan_create_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV("parameter_slab");
  plan_options.device_group = context->value.device_group;
  plan_options.diagnostics_sink = diagnostics_sink;
  plan_options.placement_count = 1;
  plan_options.placements = &placement;
  plan_options.parameter_slab_count = 1;
  plan_options.parameter_slabs = &slab;
  plan_options.parameter_request_tables = &request_table;
  plan_options.parameter_load_step_count = kTargetRequestCount;
  plan_options.parameter_load_steps = load_steps;
  OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(id4_pipeline_plan_create(&plan_options,
                                          iree_allocator_system(), plan.out()));

  OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release> prepare_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      id4_tooling_runtime_context_primary_device(&context->value),
      IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  iree_hal_semaphore_t* prepare_signal_semaphore = prepare_semaphore.get();
  uint64_t prepare_signal_value = 1;
  iree_hal_semaphore_list_t prepare_signal_list =
      MakeOneSemaphoreList(&prepare_signal_semaphore, &prepare_signal_value);

  id4_pipeline_parameter_slab_set_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.encoder_staging_chunk_byte_capacity =
      ID4_PIPELINE_PARAMETER_ENCODER_DEFAULT_STAGING_CHUNK_BYTE_CAPACITY;
  load_options.encoder_staging_memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  load_options.provider = provider;
  load_options.kernel_library = kernel_library;
  load_options.kernel_cache = context->value.kernel_cache;
  load_options.executable_cache = context->value.executable_cache;
  load_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  load_options.signal_semaphore_list = prepare_signal_list;
  load_options.diagnostics_sink = diagnostics_sink;
  OwningRef<id4_pipeline_parameter_slab_set_t,
            id4_pipeline_parameter_slab_set_release>
      slab_set;
  IREE_ASSERT_OK(id4_pipeline_plan_load_parameter_slabs(
      plan.get(), &load_options, iree_allocator_system(), slab_set.out()));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      prepare_semaphore.get(), prepare_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  constexpr iree_host_size_t kEncodedElementCount =
      kTargetRequestCount * kCompactRhsTileElementCount;
  uint16_t encoded_weight[kEncodedElementCount] = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      id4_tooling_runtime_context_primary_device(&context->value),
      id4_pipeline_parameter_slab_set_buffer_at(slab_set.get(), 0),
      /*source_offset=*/0, encoded_weight, sizeof(encoded_weight),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  for (iree_host_size_t i = 0; i < kCompactRhsTileElementCount; ++i) {
    EXPECT_EQ(encoded_weight[i], 0x3F80u);
  }
  for (iree_host_size_t i = kCompactRhsTileElementCount;
       i < 2 * kCompactRhsTileElementCount; ++i) {
    EXPECT_EQ(encoded_weight[i], 0x4000u);
  }
  for (iree_host_size_t i = 0; i < kFp8RawSweepTileCount; ++i) {
    uint8_t fp8_sweep_pattern[4];
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(fp8_sweep_pattern); ++j) {
      fp8_sweep_pattern[j] =
          FiniteNonzeroFp8SweepByte(i * IREE_ARRAYSIZE(fp8_sweep_pattern) + j);
    }
    ExpectDecodedFp8Pattern(
        encoded_weight, (2 + i) * kCompactRhsTileElementCount,
        fp8_sweep_pattern, IREE_ARRAYSIZE(fp8_sweep_pattern));
  }

  id4_pipeline_parameter_slab_set_load_options_t deferred_load_options =
      load_options;
  deferred_load_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
  OwningRef<id4_pipeline_parameter_slab_set_t,
            id4_pipeline_parameter_slab_set_release>
      deferred_slab_set;
  IREE_ASSERT_OK(id4_pipeline_plan_prepare_parameter_slabs(
      plan.get(), &deferred_load_options, iree_allocator_system(),
      deferred_slab_set.out()));
  ASSERT_EQ(
      id4_pipeline_parameter_slab_set_load_group_count(deferred_slab_set.get()),
      1u);

  OwningRef<id4_pipeline_parameter_slab_issue_context_t,
            id4_pipeline_parameter_slab_issue_context_release>
      issue_context;
  IREE_ASSERT_OK(id4_pipeline_plan_create_parameter_slab_issue_context(
      plan.get(), deferred_slab_set.get(), iree_allocator_system(),
      issue_context.out()));
  IREE_ASSERT_OK(id4_pipeline_plan_submit_parameter_load_group(
      plan.get(), issue_context.get(), /*group_index=*/0,
      /*submit_region_id=*/0, diagnostics_sink));
  iree_hal_semaphore_list_t cleanup_wait_list = iree_hal_semaphore_list_empty();
  IREE_ASSERT_OK(id4_pipeline_parameter_slab_issue_context_finish(
      issue_context.get(), &cleanup_wait_list));

  iree_hal_semaphore_t* group_ready_semaphore = nullptr;
  uint64_t group_ready_value = 0;
  IREE_ASSERT_OK(id4_pipeline_parameter_slab_set_load_group_ready_at(
      deferred_slab_set.get(), /*index=*/0, &group_ready_semaphore,
      &group_ready_value));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      group_ready_semaphore, group_ready_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  for (iree_host_size_t i = 0; i < cleanup_wait_list.count; ++i) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        cleanup_wait_list.semaphores[i], cleanup_wait_list.payload_values[i],
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  std::memset(encoded_weight, 0, sizeof(encoded_weight));
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      id4_tooling_runtime_context_primary_device(&context->value),
      id4_pipeline_parameter_slab_set_buffer_at(deferred_slab_set.get(), 0),
      /*source_offset=*/0, encoded_weight, sizeof(encoded_weight),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  for (iree_host_size_t i = 0; i < kCompactRhsTileElementCount; ++i) {
    EXPECT_EQ(encoded_weight[i], 0x3F80u);
  }
  for (iree_host_size_t i = kCompactRhsTileElementCount;
       i < 2 * kCompactRhsTileElementCount; ++i) {
    EXPECT_EQ(encoded_weight[i], 0x4000u);
  }
  for (iree_host_size_t i = 0; i < kFp8RawSweepTileCount; ++i) {
    uint8_t fp8_sweep_pattern[4];
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(fp8_sweep_pattern); ++j) {
      fp8_sweep_pattern[j] =
          FiniteNonzeroFp8SweepByte(i * IREE_ARRAYSIZE(fp8_sweep_pattern) + j);
    }
    ExpectDecodedFp8Pattern(
        encoded_weight, (2 + i) * kCompactRhsTileElementCount,
        fp8_sweep_pattern, IREE_ARRAYSIZE(fp8_sweep_pattern));
  }
}

static void RunCompactFp8LinearRhsTileEncoding(
    RuntimeContext* context, id4_pipeline_kernel_library_t* kernel_library,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  constexpr uint64_t kOutputSize = 32;
  constexpr uint64_t kInputSize = 32;
  constexpr iree_host_size_t kElementCount = kOutputSize * kInputSize;
  constexpr iree_device_size_t kByteLength = kElementCount * sizeof(uint8_t);
  const iree_string_view_t kParameterKey =
      IREE_SV("weight.fp8_compact_rhs_tile");

  std::vector<uint8_t> source_data(kElementCount);
  for (iree_host_size_t i = 0; i < source_data.size(); ++i) {
    source_data[i] = FiniteNonzeroFp8SweepByte(i);
  }

  ScopedTempFilePath source_path("id4_parameter_slab_fp8_rhs_tile");
  OwningRef<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(CreateFileParameterProviderWithContents(
      &source_path, kParameterKey,
      iree_make_const_byte_span(source_data.data(), source_data.size()),
      provider.out()));

  id4_pipeline_device_placement_t placement;
  std::memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = 0;
  placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  id4_pipeline_parameter_request_t target_request =
      id4_pipeline_parameter_request(
          kParameterKey, id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                                     /*buffer_offset=*/0,
                                                     /*length=*/kByteLength));
  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/0, IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
              IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
          kByteLength, /*alignment=*/16);
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(/*count=*/1, &target_request);

  id4_pipeline_tensor_shape_t weight_shape;
  std::memset(&weight_shape, 0, sizeof(weight_shape));
  weight_shape.rank = 2;
  weight_shape.dims[0] = kOutputSize;
  weight_shape.dims[1] = kInputSize;
  const id4_pipeline_parameter_load_source_t source =
      id4_pipeline_parameter_load_source(IREE_SV("parameters"), kParameterKey,
                                         ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3,
                                         weight_shape, kByteLength);
  id4_pipeline_parameter_load_step_t load_step =
      id4_pipeline_parameter_encode_fp8_e4m3_linear_rhs_tile_load_step(
          IREE_SV("parameters.encode_fp8_compact_rhs_tile"),
          /*source_count=*/1, &source,
          /*target_slab_index=*/0, /*request_offset=*/0);

  id4_pipeline_plan_create_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV("parameter_slab.fp8_rhs_tile");
  plan_options.device_group = context->value.device_group;
  plan_options.diagnostics_sink = diagnostics_sink;
  plan_options.placement_count = 1;
  plan_options.placements = &placement;
  plan_options.parameter_slab_count = 1;
  plan_options.parameter_slabs = &slab;
  plan_options.parameter_request_tables = &request_table;
  plan_options.parameter_load_step_count = 1;
  plan_options.parameter_load_steps = &load_step;
  OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(id4_pipeline_plan_create(&plan_options,
                                          iree_allocator_system(), plan.out()));

  iree_hal_device_t* device =
      id4_tooling_runtime_context_primary_device(&context->value);
  OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release> prepare_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  iree_hal_semaphore_t* prepare_signal_semaphore = prepare_semaphore.get();
  uint64_t prepare_signal_value = 1;
  iree_hal_semaphore_list_t prepare_signal_list =
      MakeOneSemaphoreList(&prepare_signal_semaphore, &prepare_signal_value);

  id4_pipeline_parameter_slab_set_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.encoder_staging_chunk_byte_capacity =
      ID4_PIPELINE_PARAMETER_ENCODER_DEFAULT_STAGING_CHUNK_BYTE_CAPACITY;
  load_options.encoder_staging_memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  load_options.provider = provider.get();
  load_options.kernel_library = kernel_library;
  load_options.kernel_cache = context->value.kernel_cache;
  load_options.executable_cache = context->value.executable_cache;
  load_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  load_options.signal_semaphore_list = prepare_signal_list;
  load_options.diagnostics_sink = diagnostics_sink;
  OwningRef<id4_pipeline_parameter_slab_set_t,
            id4_pipeline_parameter_slab_set_release>
      slab_set;
  IREE_ASSERT_OK(id4_pipeline_plan_load_parameter_slabs(
      plan.get(), &load_options, iree_allocator_system(), slab_set.out()));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      prepare_semaphore.get(), prepare_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(id4_pipeline_parameter_slab_set_check_load_group_failures(
      slab_set.get(), plan_options.stage_name, diagnostics_sink));

  std::vector<uint8_t> actual(kByteLength);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device, id4_pipeline_parameter_slab_set_buffer_at(slab_set.get(), 0),
      /*source_offset=*/0, actual.data(), actual.size(),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  std::vector<uint8_t> expected(kByteLength);
  constexpr iree_host_size_t kTileSize = 16;
  const iree_host_size_t input_tile_count = kInputSize / kTileSize;
  const iree_host_size_t output_tile_count = kOutputSize / kTileSize;
  for (iree_host_size_t output_tile = 0; output_tile < output_tile_count;
       ++output_tile) {
    for (iree_host_size_t input_tile = 0; input_tile < input_tile_count;
         ++input_tile) {
      const iree_host_size_t tile = output_tile * input_tile_count + input_tile;
      const iree_host_size_t target_tile_offset =
          tile * kCompactFp8RhsTileByteLength;
      for (iree_host_size_t row = 0; row < kTileSize; ++row) {
        const iree_host_size_t source_row = output_tile * kTileSize + row;
        const iree_host_size_t source_column = input_tile * kTileSize;
        std::memcpy(
            expected.data() + target_tile_offset + row * kTileSize,
            source_data.data() + source_row * kInputSize + source_column,
            kTileSize);
      }
    }
  }
  for (iree_host_size_t i = 0; i < actual.size(); ++i) {
    ASSERT_EQ(actual[i], expected[i]) << "byte offset " << i;
  }
}

static void ExpectBf16OnesAt(iree_hal_device_t* device,
                             iree_hal_buffer_t* buffer,
                             iree_device_size_t source_offset) {
  uint16_t values[16] = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device, buffer, source_offset, values, sizeof(values),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  for (uint16_t value : values) {
    EXPECT_EQ(value, 0x3F80u);
  }
}

static iree_device_size_t RhsTileSourceByteOffsetForTargetByteOffset(
    iree_device_size_t target_byte_offset, uint64_t input_size) {
  const iree_device_size_t target_element_offset =
      target_byte_offset / sizeof(uint16_t);
  const uint64_t input_tile_count = input_size / 16;
  const uint64_t tile = target_element_offset / 256;
  const uint64_t tile_element_offset = target_element_offset % 256;
  const uint64_t lane = tile_element_offset / 16;
  const uint64_t output_tile = tile / input_tile_count;
  const uint64_t input_tile = tile % input_tile_count;
  const uint64_t source_row = output_tile * 16 + lane;
  const uint64_t source_column = input_tile * 16;
  return (source_row * input_size + source_column) * sizeof(uint16_t);
}

typedef uint32_t FileBackedQwenRhsTileEncodingFlags;
enum FileBackedQwenRhsTileEncodingFlagBits : FileBackedQwenRhsTileEncodingFlags {
  FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_NONE = 0u,
  FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_DEFERRED_ISSUE = 1u << 0,
  FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_SPLIT_READINESS_GROUPS = 1u << 1,
  FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_RELEASE_CONTEXT_BEFORE_WAIT = 1u << 2,
};

static void WaitSemaphoreList(iree_hal_semaphore_list_t list) {
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        list.semaphores[i], list.payload_values[i], iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
  }
}

static void RunFileBackedQwenRhsTileEncoding(
    iree_hal_memory_type_t encoder_staging_memory_type,
    iree_host_size_t load_step_count,
    FileBackedQwenRhsTileEncodingFlags flags) {
  constexpr uint64_t kOutputSize = 13824;
  constexpr uint64_t kInputSize = 4608;
  constexpr uint64_t kElementCount = kOutputSize * kInputSize;
  constexpr iree_device_size_t kByteLength = kElementCount * sizeof(uint16_t);
  constexpr iree_host_size_t kMaxLoadStepCount = 5;

  ASSERT_GT(load_step_count, 0u);
  ASSERT_LE(load_step_count, kMaxLoadStepCount);
  const bool use_deferred_issue = iree_all_bits_set(
      flags, FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_DEFERRED_ISSUE);
  const bool split_readiness_groups = iree_all_bits_set(
      flags, FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_SPLIT_READINESS_GROUPS);
  const bool release_context_before_wait = iree_all_bits_set(
      flags,
      FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_RELEASE_CONTEXT_BEFORE_WAIT);

  RuntimeContext& context = SharedRuntimeContext();

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  OwningRef<id4_pipeline_kernel_library_t, id4_pipeline_kernel_library_release>
      kernel_library;
  IREE_ASSERT_OK(id4_tooling_create_embedded_kernel_library(
      iree_allocator_system(), kernel_library.out()));

  ScopedTempFilePath source_path("id4_parameter_slab_qwen_rhs_tile");
  const iree_device_size_t target_check_offsets[] = {
      0,
      kByteLength / 2,
      kByteLength - 16 * sizeof(uint16_t),
  };
  iree_device_size_t source_write_offsets[IREE_ARRAYSIZE(target_check_offsets)];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(target_check_offsets); ++i) {
    source_write_offsets[i] = RhsTileSourceByteOffsetForTargetByteOffset(
        target_check_offsets[i], kInputSize);
  }
  OwningRef<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(CreateSparseBf16FileParameterProvider(
      &source_path, IREE_SV("weight.bf16_qwen_rhs_tile"), kByteLength,
      IREE_ARRAYSIZE(source_write_offsets), source_write_offsets,
      provider.out()));

  id4_pipeline_device_placement_t placement;
  std::memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = 0;
  placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  id4_pipeline_parameter_request_t target_requests[kMaxLoadStepCount];
  for (iree_host_size_t i = 0; i < load_step_count; ++i) {
    target_requests[i] = id4_pipeline_parameter_request(
        IREE_SV("weight.bf16_qwen_rhs_tile"),
        id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                    /*buffer_offset=*/i * kByteLength,
                                    /*length=*/kByteLength));
  }
  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/0, IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
              IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
          /*byte_length=*/load_step_count * kByteLength, /*alignment=*/16);
  id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(load_step_count,
                                                target_requests);

  id4_pipeline_tensor_shape_t weight_shape;
  std::memset(&weight_shape, 0, sizeof(weight_shape));
  weight_shape.rank = 2;
  weight_shape.dims[0] = kOutputSize;
  weight_shape.dims[1] = kInputSize;
  const id4_pipeline_parameter_load_source_t source =
      id4_pipeline_parameter_load_source(
          IREE_SV("parameters"), IREE_SV("weight.bf16_qwen_rhs_tile"),
          ID4_PIPELINE_TENSOR_DTYPE_BF16, weight_shape,
          /*byte_length=*/kByteLength);
  const iree_string_view_t load_step_names[kMaxLoadStepCount] = {
      IREE_SV("parameters.encode_bf16_qwen_rhs_tile.0"),
      IREE_SV("parameters.encode_bf16_qwen_rhs_tile.1"),
      IREE_SV("parameters.encode_bf16_qwen_rhs_tile.2"),
      IREE_SV("parameters.encode_bf16_qwen_rhs_tile.3"),
      IREE_SV("parameters.encode_bf16_qwen_rhs_tile.4"),
  };
  id4_pipeline_parameter_load_step_t load_steps[kMaxLoadStepCount];
  for (iree_host_size_t i = 0; i < load_step_count; ++i) {
    load_steps[i] =
        id4_pipeline_parameter_encode_bf16_linear_rhs_tile_load_step(
            load_step_names[i], /*source_count=*/1, &source,
            /*target_slab_index=*/0, /*request_offset=*/i);
    if (split_readiness_groups) {
      load_steps[i].readiness_group_key = i;
    }
  }

  id4_pipeline_plan_create_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV("parameter_slab.qwen_rhs_tile");
  plan_options.device_group = context.value.device_group;
  plan_options.diagnostics_sink = &diagnostics_sink;
  plan_options.placement_count = 1;
  plan_options.placements = &placement;
  plan_options.parameter_slab_count = 1;
  plan_options.parameter_slabs = &slab;
  plan_options.parameter_request_tables = &request_table;
  plan_options.parameter_load_step_count = load_step_count;
  plan_options.parameter_load_steps = load_steps;
  OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(id4_pipeline_plan_create(&plan_options,
                                          iree_allocator_system(), plan.out()));

  iree_hal_device_t* device =
      id4_tooling_runtime_context_primary_device(&context.value);
  OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release> prepare_semaphore;
  if (!use_deferred_issue) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  }
  iree_hal_semaphore_t* prepare_signal_semaphore =
      use_deferred_issue ? nullptr : prepare_semaphore.get();
  uint64_t prepare_signal_value = 1;
  iree_hal_semaphore_list_t prepare_signal_list =
      use_deferred_issue ? iree_hal_semaphore_list_empty()
                         : MakeOneSemaphoreList(&prepare_signal_semaphore,
                                                &prepare_signal_value);

  id4_pipeline_parameter_slab_set_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.encoder_staging_chunk_byte_capacity =
      ID4_PIPELINE_PARAMETER_ENCODER_DEFAULT_STAGING_CHUNK_BYTE_CAPACITY;
  load_options.encoder_staging_memory_type = encoder_staging_memory_type;
  load_options.provider = provider.get();
  load_options.kernel_library = kernel_library.get();
  load_options.kernel_cache = context.value.kernel_cache;
  load_options.executable_cache = context.value.executable_cache;
  load_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  load_options.signal_semaphore_list = prepare_signal_list;
  load_options.diagnostics_sink = &diagnostics_sink;
  OwningRef<id4_pipeline_parameter_slab_set_t,
            id4_pipeline_parameter_slab_set_release>
      slab_set;
  if (use_deferred_issue) {
    IREE_ASSERT_OK(id4_pipeline_plan_prepare_parameter_slabs(
        plan.get(), &load_options, iree_allocator_system(), slab_set.out()));
    OwningRef<id4_pipeline_parameter_slab_issue_context_t,
              id4_pipeline_parameter_slab_issue_context_release>
        issue_context;
    IREE_ASSERT_OK(id4_pipeline_parameter_slab_issue_context_create(
        slab_set.get(), load_step_count, load_steps, iree_allocator_system(),
        issue_context.out()));
    const iree_host_size_t load_group_count =
        id4_pipeline_parameter_slab_set_load_group_count(slab_set.get());
    ASSERT_EQ(load_group_count,
              split_readiness_groups ? load_step_count : (iree_host_size_t)1);
    for (iree_host_size_t group_index = 0; group_index < load_group_count;
         ++group_index) {
      id4_pipeline_parameter_load_group_context_t group_context = {
          // Plan-local load group ordinal.
          .group_index = group_index,
          // This test has no consumer region graph.
          .first_consumer_region_id = IREE_HOST_SIZE_MAX,
          // This test submits parameter groups directly.
          .submit_region_id = IREE_HOST_SIZE_MAX,
      };
      IREE_ASSERT_OK(
          id4_pipeline_parameter_slab_issue_context_submit_load_group(
              issue_context.get(), load_step_count, load_steps, group_context,
              plan_options.stage_name, &diagnostics_sink));
    }
    iree_hal_semaphore_list_t cleanup_wait_list =
        iree_hal_semaphore_list_empty();
    IREE_ASSERT_OK(id4_pipeline_parameter_slab_issue_context_finish(
        issue_context.get(), &cleanup_wait_list));
    if (release_context_before_wait) {
      OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
          final_semaphore;
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, final_semaphore.out()));
      iree_hal_semaphore_t* final_signal_semaphore = final_semaphore.get();
      uint64_t final_signal_value = 1;
      iree_hal_semaphore_list_t final_signal_list =
          MakeOneSemaphoreList(&final_signal_semaphore, &final_signal_value);
      IREE_ASSERT_OK(iree_hal_device_queue_barrier(
          device, IREE_HAL_QUEUE_AFFINITY_ANY, cleanup_wait_list,
          final_signal_list, IREE_HAL_EXECUTE_FLAG_NONE));
      issue_context.reset();
      IREE_ASSERT_OK(iree_hal_semaphore_wait(
          final_semaphore.get(), final_signal_value, iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE));
    } else {
      WaitSemaphoreList(cleanup_wait_list);
    }
  } else {
    IREE_ASSERT_OK(id4_pipeline_plan_load_parameter_slabs(
        plan.get(), &load_options, iree_allocator_system(), slab_set.out()));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        prepare_semaphore.get(), prepare_signal_value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
  }
  IREE_ASSERT_OK(id4_pipeline_parameter_slab_set_check_load_group_failures(
      slab_set.get(), plan_options.stage_name, &diagnostics_sink));

  iree_hal_buffer_t* target_buffer =
      id4_pipeline_parameter_slab_set_buffer_at(slab_set.get(), 0);
  for (iree_host_size_t step_index = 0; step_index < load_step_count;
       ++step_index) {
    const iree_device_size_t target_base_offset = step_index * kByteLength;
    for (iree_device_size_t target_check_offset : target_check_offsets) {
      ExpectBf16OnesAt(device, target_buffer,
                       target_base_offset + target_check_offset);
    }
  }
}

TEST(ParameterSlabIntegration, FileBackedHostVisibleQwenRhsTileEncoding) {
  RunFileBackedQwenRhsTileEncoding(
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      /*load_step_count=*/1, FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_NONE);
}

TEST(ParameterSlabIntegration, FileBackedHostVisibleQwenQkvRhsTileEncoding) {
  RunFileBackedQwenRhsTileEncoding(
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      /*load_step_count=*/3, FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_NONE);
}

TEST(ParameterSlabIntegration,
     FileBackedHostVisibleQwenFiveChunkRhsTileEncoding) {
  RunFileBackedQwenRhsTileEncoding(
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      /*load_step_count=*/5, FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_NONE);
}

TEST(ParameterSlabIntegration,
     FileBackedHostVisibleQwenSplitIssueRhsTileEncoding) {
  RunFileBackedQwenRhsTileEncoding(
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      /*load_step_count=*/5,
      FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_DEFERRED_ISSUE |
          FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_SPLIT_READINESS_GROUPS |
          FILE_BACKED_QWEN_RHS_TILE_ENCODING_FLAG_RELEASE_CONTEXT_BEFORE_WAIT);
}

}  // namespace
