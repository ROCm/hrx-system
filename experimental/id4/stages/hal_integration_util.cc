// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/hal_integration_util.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <utility>

#include "experimental/id4/kernels/embedded_loom_sources.h"
#include "experimental/id4/pipeline/program.h"
#include "iree/base/internal/json.h"
#include "iree/base/internal/math.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/scope_map.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/parameter_util.h"

namespace id4::test {

static constexpr iree_host_size_t kQueueUpdateChunkLength = 4 * 1024 * 1024;
static constexpr uint8_t kId4TensorMagic[8] = {
    'I', 'D', '4', 'T', 'E', 'N', 'S', 'R',
};

static uint64_t CeilMiB(iree_device_size_t byte_length) {
  return (uint64_t)((byte_length + 1024 * 1024 - 1) / (1024 * 1024));
}

static iree_status_t AddDeviceByteLength(iree_device_size_t addend,
                                         iree_device_size_t* inout_value,
                                         iree_string_view_t name) {
  iree_device_size_t value = 0;
  if (!iree_device_size_checked_add(*inout_value, addend, &value)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%.*s byte length overflow", (int)name.size,
                            name.data);
  }
  *inout_value = value;
  return iree_ok_status();
}

static iree_status_t ProgramDispatchBindingByteLength(
    const id4_pipeline_program_t* program,
    const id4_pipeline_program_dispatch_binding_t* binding,
    id4_pipeline_program_dispatch_binding_flags_t range_flag,
    id4_pipeline_program_tensor_byte_range_t range,
    iree_device_size_t* out_byte_length) {
  *out_byte_length = 0;
  const id4_pipeline_program_tensor_record_t* tensor =
      id4_pipeline_program_tensor_at(program, binding->tensor.ordinal);
  if (!tensor) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program dispatch binding tensor %u is missing",
                            binding->tensor.ordinal);
  }
  *out_byte_length = iree_all_bits_set(binding->flags, range_flag)
                         ? range.length
                         : tensor->byte_length;
  return iree_ok_status();
}

static iree_status_t AccumulateProgramDispatchBindingBytes(
    const id4_pipeline_program_t* program,
    const id4_pipeline_program_dispatch_binding_t* binding,
    id4_pipeline_program_tensor_access_flags_t access,
    id4_pipeline_program_dispatch_binding_flags_t range_flag,
    id4_pipeline_program_tensor_byte_range_t range,
    iree_device_size_t* inout_byte_length) {
  if (!iree_all_bits_set(binding->access, access)) return iree_ok_status();
  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(ProgramDispatchBindingByteLength(
      program, binding, range_flag, range, &byte_length));
  return AddDeviceByteLength(byte_length, inout_byte_length,
                             IREE_SV("program dispatch binding"));
}

static bool IsStreamingRhsEncodeDispatch(
    const id4_pipeline_program_dispatch_loom_op_t* dispatch) {
  return iree_string_view_equal(
             dispatch->kernel.module_path,
             IREE_SV(
                 "parameter/fp8_e4m3_block_scaled_to_bf16_linear_rhs_tile")) ||
         iree_string_view_equal(
             dispatch->kernel.module_path,
             IREE_SV("parameter/fp8_e4m3_scaled_to_bf16_linear_rhs_tile"));
}

iree_status_t AccumulateProgramStreamingRhsEncodeStatistics(
    const id4_pipeline_plan_t* plan,
    ProgramStreamingRhsEncodeStatistics* inout_statistics) {
  const id4_pipeline_program_t* program =
      id4_pipeline_plan_source_program(plan);
  if (!program) return iree_ok_status();
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation ||
        operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
      continue;
    }
    const id4_pipeline_program_dispatch_loom_op_t* dispatch =
        &operation->payload.dispatch_loom;
    if (!IsStreamingRhsEncodeDispatch(dispatch)) continue;

    ++inout_statistics->dispatch_count;
    iree_device_size_t write_byte_length = 0;
    for (iree_host_size_t j = 0; j < dispatch->binding_count; ++j) {
      const id4_pipeline_program_dispatch_binding_t* binding =
          &dispatch->bindings[j];
      IREE_RETURN_IF_ERROR(AccumulateProgramDispatchBindingBytes(
          program, binding, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ,
          ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_READ_RANGE,
          binding->read_range, &inout_statistics->read_byte_length));
      IREE_RETURN_IF_ERROR(AccumulateProgramDispatchBindingBytes(
          program, binding, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE,
          ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_WRITE_RANGE,
          binding->write_range, &write_byte_length));
    }
    IREE_RETURN_IF_ERROR(AddDeviceByteLength(
        write_byte_length, &inout_statistics->write_byte_length,
        IREE_SV("streaming RHS encoder write")));
    inout_statistics->max_write_byte_length =
        iree_max(inout_statistics->max_write_byte_length, write_byte_length);
  }
  return iree_ok_status();
}

static const id4_pipeline_parameter_load_kind_statistics_t*
ParameterLoadKindStatisticsAt(
    const id4_pipeline_parameter_load_kind_statistics_t* statistics,
    id4_pipeline_parameter_load_step_kind_t kind) {
  return &statistics[kind];
}

static iree_status_t AppendParameterLoadKindStatisticsEntry(
    iree_string_builder_t* builder, iree_string_view_t prefix,
    const id4_pipeline_parameter_load_kind_statistics_t* statistics) {
  return iree_string_builder_append_format(
      builder,
      "%.*s_steps=%" PRIhsz ",%.*s_source=%" PRIu64 "MiB,%.*s_target=%" PRIu64
      "MiB",
      (int)prefix.size, prefix.data, statistics->step_count, (int)prefix.size,
      prefix.data, CeilMiB(statistics->source_byte_length), (int)prefix.size,
      prefix.data, CeilMiB(statistics->target_byte_length));
}

iree_status_t AppendParameterLoadKindStatisticsLabel(
    iree_string_builder_t* builder,
    const id4_pipeline_parameter_load_kind_statistics_t* statistics) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, " param_load_kind["));
  IREE_RETURN_IF_ERROR(AppendParameterLoadKindStatisticsEntry(
      builder, IREE_SV("gather"),
      ParameterLoadKindStatisticsAt(
          statistics, ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(AppendParameterLoadKindStatisticsEntry(
      builder, IREE_SV("fp8_bf16"),
      ParameterLoadKindStatisticsAt(
          statistics,
          ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(AppendParameterLoadKindStatisticsEntry(
      builder, IREE_SV("bf16_rhs"),
      ParameterLoadKindStatisticsAt(
          statistics,
          ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(AppendParameterLoadKindStatisticsEntry(
      builder, IREE_SV("fp8_bf16_rhs"),
      ParameterLoadKindStatisticsAt(
          statistics,
          ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(AppendParameterLoadKindStatisticsEntry(
      builder, IREE_SV("fp8_rhs"),
      ParameterLoadKindStatisticsAt(
          statistics,
          ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_LINEAR_RHS_TILE)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(AppendParameterLoadKindStatisticsEntry(
      builder, IREE_SV("fp8_block_bf16_rhs"),
      ParameterLoadKindStatisticsAt(
          statistics,
          ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE)));
  return iree_string_builder_append_cstring(builder, "]");
}

iree_hal_semaphore_list_t SemaphoreListStorage::list() {
  return iree_hal_semaphore_list_t{
      // One semaphore is carried by this stack-backed list.
      /*.count=*/1,
      // Stack-backed semaphore pointer array.
      /*.semaphores=*/&semaphore,
      // Stack-backed payload value array.
      /*.payload_values=*/&payload_value,
  };
}

iree_status_t FixedSemaphoreListStorage::push(iree_hal_semaphore_t* semaphore,
                                              uint64_t payload_value) {
  if (!semaphore) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixed semaphore list edge requires a semaphore");
  }
  if (count == IREE_ARRAYSIZE(semaphores)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixed semaphore list capacity exceeded");
  }
  semaphores[count] = semaphore;
  payload_values[count] = payload_value;
  ++count;
  return iree_ok_status();
}

iree_hal_semaphore_list_t FixedSemaphoreListStorage::list() {
  return iree_hal_semaphore_list_t{
      // Number of semaphore edges in the list.
      /*.count=*/count,
      // Stack-backed semaphore handles.
      /*.semaphores=*/count == 0 ? nullptr : semaphores,
      // Stack-backed payload values.
      /*.payload_values=*/count == 0 ? nullptr : payload_values,
  };
}

static iree_status_t CommandBufferModeFromFlags(
    iree_hal_command_buffer_mode_t* out_mode) {
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  *out_mode = retain_profile_metadata
                  ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA |
                        IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA
                  : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
  return iree_ok_status();
}

BufferBindingSet::~BufferBindingSet() { reset(); }

void BufferBindingSet::reset() {
  if (buffers) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_hal_buffer_release(buffers[i]);
    }
  }
  iree_allocator_free(iree_allocator_system(), buffers);
  iree_allocator_free(iree_allocator_system(), bindings);
  count = 0;
  buffers = nullptr;
  bindings = nullptr;
}

const FixtureTensor* FixtureTensorSet::FindTensor(
    iree_string_view_t name) const {
  for (const FixtureTensor& tensor : tensors) {
    if (iree_string_view_equal(
            iree_make_string_view(tensor.name.data(), tensor.name.size()),
            name)) {
      return &tensor;
    }
  }
  return nullptr;
}

const FixtureTensor* FixtureTensorSet::FindTensor(
    iree_string_view_t role, iree_string_view_t name) const {
  for (const FixtureTensor& tensor : tensors) {
    if (iree_string_view_equal(
            iree_make_string_view(tensor.role.data(), tensor.role.size()),
            role) &&
        iree_string_view_equal(
            iree_make_string_view(tensor.name.data(), tensor.name.size()),
            name)) {
      return &tensor;
    }
  }
  return nullptr;
}

const FixtureTensor* FixtureTensorSet::FindTensor(
    iree_string_view_t role, iree_string_view_t stage,
    iree_string_view_t name) const {
  for (const FixtureTensor& tensor : tensors) {
    if (iree_string_view_equal(
            iree_make_string_view(tensor.role.data(), tensor.role.size()),
            role) &&
        iree_string_view_equal(
            iree_make_string_view(tensor.stage.data(), tensor.stage.size()),
            stage) &&
        iree_string_view_equal(
            iree_make_string_view(tensor.name.data(), tensor.name.size()),
            name)) {
      return &tensor;
    }
  }
  return nullptr;
}

static iree_status_t ReadBinaryFile(const std::string& path,
                                    std::vector<uint8_t>* out_file_contents) {
  out_file_contents->clear();
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "file not found: %s",
                            path.c_str());
  }
  file.seekg(0, std::ios::end);
  const std::streamoff file_size = file.tellg();
  if (file_size < 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to determine file size: %s", path.c_str());
  }
  file.seekg(0, std::ios::beg);
  out_file_contents->resize((size_t)file_size);
  if (file_size != 0) {
    file.read(reinterpret_cast<char*>(out_file_contents->data()), file_size);
    if (file.gcount() != file_size) {
      return iree_make_status(IREE_STATUS_DATA_LOSS, "failed to read file: %s",
                              path.c_str());
    }
  }
  return iree_ok_status();
}

static bool FileExists(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  return file.good();
}

static bool HasPrefix(const std::vector<uint8_t>& contents,
                      const uint8_t* prefix, iree_host_size_t prefix_length) {
  return contents.size() >= prefix_length &&
         std::memcmp(contents.data(), prefix, prefix_length) == 0;
}

static std::string JoinPath(iree_string_view_t directory,
                            const char* file_name) {
  std::string path(directory.data, directory.size);
  if (!path.empty() && path.back() != '/') path.push_back('/');
  path.append(file_name);
  return path;
}

static std::string JoinPath(const std::string& directory,
                            const std::string& file_name) {
  std::string path = directory;
  if (!path.empty() && path.back() != '/') path.push_back('/');
  path.append(file_name);
  return path;
}

static iree_status_t JsonLookupSimpleString(iree_string_view_t object,
                                            iree_string_view_t key,
                                            std::string* out_value) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(object, key, &value));
  out_value->assign(value.data, value.size);
  return iree_ok_status();
}

static iree_status_t ParseFixtureDtype(iree_string_view_t dtype,
                                       id4_pipeline_tensor_dtype_t* out_dtype) {
  if (iree_string_view_equal(dtype, IREE_SV("f32"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_F32;
    return iree_ok_status();
  }
  if (iree_string_view_equal(dtype, IREE_SV("f16"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_F16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(dtype, IREE_SV("bf16"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_BF16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(dtype, IREE_SV("i32"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_I32;
    return iree_ok_status();
  }
  if (iree_string_view_equal(dtype, IREE_SV("u32"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_U32;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported fixture tensor dtype `%.*s`",
                          static_cast<int>(dtype.size), dtype.data);
}

static iree_status_t ParseFixtureShape(iree_string_view_t shape_array,
                                       id4_pipeline_tensor_shape_t* out_shape) {
  *out_shape = id4_pipeline_tensor_shape_t{};
  iree_host_size_t rank = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(shape_array, &rank));
  if (rank > IREE_ARRAYSIZE(out_shape->dims)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixture tensor rank %" PRIhsz
                            " exceeds max rank %zu",
                            rank, IREE_ARRAYSIZE(out_shape->dims));
  }
  out_shape->rank = static_cast<uint32_t>(rank);
  for (iree_host_size_t i = 0; i < rank; ++i) {
    iree_string_view_t dim_value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_array_get(shape_array, i, &dim_value));
    uint64_t dim = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(dim_value, &dim));
    out_shape->dims[i] = dim;
  }
  return iree_ok_status();
}

static iree_status_t ParseFixtureTolerance(iree_string_view_t record,
                                           FixtureTensor* tensor) {
  iree_string_view_t tolerance = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      record, IREE_SV("tolerance"), &tolerance));
  if (iree_string_view_is_empty(tolerance)) return iree_ok_status();

  iree_string_view_t absolute_tolerance = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      tolerance, IREE_SV("atol"), &absolute_tolerance));
  iree_string_view_t relative_tolerance = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      tolerance, IREE_SV("rtol"), &relative_tolerance));
  const bool has_elementwise_tolerance =
      !iree_string_view_is_empty(absolute_tolerance) ||
      !iree_string_view_is_empty(relative_tolerance);

  iree_string_view_t mean_absolute_error = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      tolerance, IREE_SV("mean_abs"), &mean_absolute_error));
  iree_string_view_t p99_absolute_error = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      tolerance, IREE_SV("p99_abs"), &p99_absolute_error));
  iree_string_view_t max_absolute_error = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      tolerance, IREE_SV("max_abs"), &max_absolute_error));
  const bool has_aggregate_tolerance =
      !iree_string_view_is_empty(mean_absolute_error) ||
      !iree_string_view_is_empty(p99_absolute_error) ||
      !iree_string_view_is_empty(max_absolute_error);

  if (has_elementwise_tolerance && has_aggregate_tolerance) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixture tolerance must declare exactly one mode");
  }
  if (has_elementwise_tolerance) {
    if (iree_string_view_is_empty(absolute_tolerance) ||
        iree_string_view_is_empty(relative_tolerance)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture elementwise tolerance requires both atol and rtol");
    }
    IREE_RETURN_IF_ERROR(iree_json_parse_double(
        absolute_tolerance, &tensor->tolerance.absolute_tolerance));
    IREE_RETURN_IF_ERROR(iree_json_parse_double(
        relative_tolerance, &tensor->tolerance.relative_tolerance));
    if (tensor->tolerance.absolute_tolerance < 0.0 ||
        tensor->tolerance.relative_tolerance < 0.0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture elementwise tolerance values must be non-negative");
    }
    tensor->tolerance.mode = FixtureToleranceMode::kElementwise;
    return iree_ok_status();
  }
  if (has_aggregate_tolerance) {
    if (iree_string_view_is_empty(mean_absolute_error) ||
        iree_string_view_is_empty(p99_absolute_error) ||
        iree_string_view_is_empty(max_absolute_error)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture aggregate tolerance requires mean_abs, p99_abs, and "
          "max_abs");
    }
    IREE_RETURN_IF_ERROR(iree_json_parse_double(
        mean_absolute_error, &tensor->tolerance.mean_absolute_error));
    IREE_RETURN_IF_ERROR(iree_json_parse_double(
        p99_absolute_error, &tensor->tolerance.p99_absolute_error));
    IREE_RETURN_IF_ERROR(iree_json_parse_double(
        max_absolute_error, &tensor->tolerance.max_absolute_error));
    if (tensor->tolerance.mean_absolute_error < 0.0 ||
        tensor->tolerance.p99_absolute_error < 0.0 ||
        tensor->tolerance.max_absolute_error < 0.0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture aggregate tolerance values must be non-negative");
    }
    tensor->tolerance.mode = FixtureToleranceMode::kAggregate;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "fixture tolerance must declare a comparison mode");
}

static bool FixtureToleranceIsDeclared(const FixtureTensor& tensor) {
  return tensor.tolerance.mode != FixtureToleranceMode::kNone;
}

static bool FixtureToleranceIsElementwise(const FixtureTensor& tensor) {
  return tensor.tolerance.mode == FixtureToleranceMode::kElementwise;
}

static bool FixtureToleranceIsAggregate(const FixtureTensor& tensor) {
  return tensor.tolerance.mode == FixtureToleranceMode::kAggregate;
}

static iree_status_t VerifyAggregateTolerance(
    const FixtureTensor& expected_tensor, std::vector<float>* absolute_errors) {
  if (absolute_errors->empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "aggregate comparison requires at least one "
                            "tensor element");
  }
  double absolute_error_sum = 0.0;
  float max_absolute_error = 0.0f;
  for (float absolute_error : *absolute_errors) {
    absolute_error_sum += absolute_error;
    max_absolute_error = std::max(max_absolute_error, absolute_error);
  }
  const double mean_absolute_error =
      absolute_error_sum / (double)absolute_errors->size();
  const iree_host_size_t p99_index =
      (absolute_errors->size() * 99 + 99) / 100 - 1;
  std::nth_element(absolute_errors->begin(),
                   absolute_errors->begin() + p99_index,
                   absolute_errors->end());
  const double p99_absolute_error = (*absolute_errors)[p99_index];
  if (mean_absolute_error <= expected_tensor.tolerance.mean_absolute_error &&
      p99_absolute_error <= expected_tensor.tolerance.p99_absolute_error &&
      (double)max_absolute_error <=
          expected_tensor.tolerance.max_absolute_error) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "tensor `%s` aggregate mismatch: mean_abs=%g limit=%g p99_abs=%g "
      "limit=%g max_abs=%g limit=%g",
      expected_tensor.name.c_str(), mean_absolute_error,
      expected_tensor.tolerance.mean_absolute_error, p99_absolute_error,
      expected_tensor.tolerance.p99_absolute_error, (double)max_absolute_error,
      expected_tensor.tolerance.max_absolute_error);
}

static bool ShapeEquals(id4_pipeline_tensor_shape_t lhs,
                        id4_pipeline_tensor_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
}

static iree_status_t ParseFixtureSlice(iree_string_view_t record,
                                       FixtureTensor* tensor) {
  iree_string_view_t slice_array = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      record, IREE_SV("slice"), &slice_array));
  if (iree_string_view_is_empty(slice_array) ||
      iree_string_view_equal(slice_array, IREE_SV("null"))) {
    if (!ShapeEquals(tensor->source_shape, tensor->shape)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%s` omits slice metadata but source shape differs",
          tensor->name.c_str());
    }
    tensor->slice_offsets = id4_pipeline_tensor_shape_t{};
    tensor->slice_offsets.rank = tensor->shape.rank;
    return iree_ok_status();
  }
  iree_host_size_t slice_rank = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(slice_array, &slice_rank));
  if (slice_rank != tensor->shape.rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixture tensor `%s` slice rank %" PRIhsz
                            " does not match tensor rank %u",
                            tensor->name.c_str(), slice_rank,
                            tensor->shape.rank);
  }
  tensor->slice_offsets = id4_pipeline_tensor_shape_t{};
  tensor->slice_offsets.rank = tensor->shape.rank;
  for (iree_host_size_t i = 0; i < slice_rank; ++i) {
    iree_string_view_t slice = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_array_get(slice_array, i, &slice));

    iree_string_view_t start_value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        iree_json_lookup_object_value(slice, IREE_SV("start"), &start_value));
    uint64_t start = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(start_value, &start));

    iree_string_view_t length_value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        iree_json_lookup_object_value(slice, IREE_SV("length"), &length_value));
    uint64_t length = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(length_value, &length));
    if (length != tensor->shape.dims[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%s` slice dimension %" PRIhsz " length %" PRIu64
          " does not match tensor dimension %" PRIu64,
          tensor->name.c_str(), i, length, tensor->shape.dims[i]);
    }
    if (start > tensor->source_shape.dims[i] ||
        length > tensor->source_shape.dims[i] - start) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "fixture tensor `%s` slice dimension %" PRIhsz
                              " [%" PRIu64 ", %" PRIu64
                              ") exceeds source dimension %" PRIu64,
                              tensor->name.c_str(), i, start, start + length,
                              tensor->source_shape.dims[i]);
    }
    tensor->slice_offsets.dims[i] = start;
  }
  return iree_ok_status();
}

static iree_status_t ParseNpyShapeToken(iree_string_view_t token,
                                        uint64_t* out_value) {
  const char* begin = token.data;
  const char* end = token.data + token.size;
  while (begin < end && (*begin == ' ' || *begin == '\t')) ++begin;
  while (end > begin && (end[-1] == ' ' || end[-1] == '\t')) --end;
  if (begin == end) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty NPY shape dimension");
  }
  char* parse_end = nullptr;
  errno = 0;
  unsigned long long parsed = strtoull(begin, &parse_end, 10);
  if (errno != 0 || parse_end != end) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid NPY shape dimension `%.*s`",
                            static_cast<int>(token.size), token.data);
  }
  *out_value = static_cast<uint64_t>(parsed);
  return iree_ok_status();
}

static iree_status_t ParseNpyShape(const std::string& header,
                                   id4_pipeline_tensor_shape_t* out_shape) {
  *out_shape = id4_pipeline_tensor_shape_t{};
  iree_string_view_t header_view =
      iree_make_string_view(header.data(), header.size());
  const iree_string_view_t key = IREE_SV("'shape': (");
  iree_host_size_t shape_begin = iree_string_view_find(header_view, key, 0);
  if (shape_begin == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NPY header is missing shape tuple");
  }
  shape_begin += key.size;
  const iree_host_size_t shape_end =
      iree_string_view_find(header_view, IREE_SV(")"), shape_begin);
  if (shape_end == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NPY header shape tuple is unterminated");
  }
  iree_string_view_t shape_contents = iree_make_string_view(
      header.data() + shape_begin, shape_end - shape_begin);
  while (!iree_string_view_is_empty(shape_contents)) {
    iree_host_size_t comma_index = 0;
    while (comma_index < shape_contents.size &&
           shape_contents.data[comma_index] != ',') {
      ++comma_index;
    }
    iree_string_view_t dim_token =
        iree_make_string_view(shape_contents.data, comma_index);
    if (!iree_string_view_is_empty(iree_string_view_trim(dim_token))) {
      if (out_shape->rank >= IREE_ARRAYSIZE(out_shape->dims)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "NPY tensor rank exceeds max rank %zu",
                                IREE_ARRAYSIZE(out_shape->dims));
      }
      uint64_t dim = 0;
      IREE_RETURN_IF_ERROR(ParseNpyShapeToken(dim_token, &dim));
      out_shape->dims[out_shape->rank++] = dim;
    }
    if (comma_index == shape_contents.size) break;
    shape_contents =
        iree_make_string_view(shape_contents.data + comma_index + 1,
                              shape_contents.size - comma_index - 1);
  }
  return iree_ok_status();
}

static iree_status_t ParseNpyDtype(const std::string& header,
                                   id4_pipeline_tensor_dtype_t* out_dtype) {
  iree_string_view_t header_view =
      iree_make_string_view(header.data(), header.size());
  const iree_string_view_t key = IREE_SV("'descr': '");
  iree_host_size_t dtype_begin = iree_string_view_find(header_view, key, 0);
  if (dtype_begin == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NPY header is missing dtype descriptor");
  }
  dtype_begin += key.size;
  const iree_host_size_t dtype_end =
      iree_string_view_find(header_view, IREE_SV("'"), dtype_begin);
  if (dtype_end == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NPY dtype descriptor is unterminated");
  }
  iree_string_view_t dtype = iree_make_string_view(header.data() + dtype_begin,
                                                   dtype_end - dtype_begin);
  if (iree_string_view_equal(dtype, IREE_SV("<f4")) ||
      iree_string_view_equal(dtype, IREE_SV("|f4"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_F32;
    return iree_ok_status();
  }
  if (iree_string_view_equal(dtype, IREE_SV("<f2")) ||
      iree_string_view_equal(dtype, IREE_SV("|f2"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_F16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(dtype, IREE_SV("<u2")) ||
      iree_string_view_equal(dtype, IREE_SV("|u2"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_BF16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(dtype, IREE_SV("<i4")) ||
      iree_string_view_equal(dtype, IREE_SV("|i4"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_I32;
    return iree_ok_status();
  }
  if (iree_string_view_equal(dtype, IREE_SV("<u4")) ||
      iree_string_view_equal(dtype, IREE_SV("|u4"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_U32;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported NPY dtype descriptor `%.*s`",
                          static_cast<int>(dtype.size), dtype.data);
}

static iree_status_t LoadNpyTensorPayload(
    const std::string& path, id4_pipeline_tensor_dtype_t expected_dtype,
    id4_pipeline_tensor_shape_t expected_shape,
    std::vector<uint8_t>* out_payload) {
  std::vector<uint8_t> file_contents;
  IREE_RETURN_IF_ERROR(ReadBinaryFile(path, &file_contents));
  if (file_contents.size() < 10 || file_contents[0] != 0x93 ||
      file_contents[1] != 'N' || file_contents[2] != 'U' ||
      file_contents[3] != 'M' || file_contents[4] != 'P' ||
      file_contents[5] != 'Y') {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "fixture tensor is not an NPY payload: %s",
                            path.c_str());
  }
  if (file_contents[6] != 1 || file_contents[7] != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixture tensor uses unsupported NPY version %u.%u",
                            file_contents[6], file_contents[7]);
  }
  const iree_host_size_t header_length =
      static_cast<iree_host_size_t>(file_contents[8]) |
      (static_cast<iree_host_size_t>(file_contents[9]) << 8);
  const iree_host_size_t payload_offset = 10 + header_length;
  if (payload_offset > file_contents.size()) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "fixture tensor NPY header overruns file: %s",
                            path.c_str());
  }
  const std::string header(reinterpret_cast<const char*>(&file_contents[10]),
                           header_length);
  id4_pipeline_tensor_dtype_t actual_dtype = ID4_PIPELINE_TENSOR_DTYPE_INVALID;
  IREE_RETURN_IF_ERROR(ParseNpyDtype(header, &actual_dtype));
  if (actual_dtype != expected_dtype) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor NPY dtype does not match manifest for %s",
        path.c_str());
  }
  id4_pipeline_tensor_shape_t actual_shape = {};
  IREE_RETURN_IF_ERROR(ParseNpyShape(header, &actual_shape));
  if (!ShapeEquals(actual_shape, expected_shape)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor NPY shape does not match manifest for %s",
        path.c_str());
  }
  out_payload->assign(file_contents.begin() + payload_offset,
                      file_contents.end());
  return iree_ok_status();
}

static iree_status_t LoadExactTensorPayload(
    const std::string& path, id4_pipeline_tensor_dtype_t expected_dtype,
    id4_pipeline_tensor_shape_t expected_shape,
    std::vector<uint8_t>* out_payload) {
  std::vector<uint8_t> file_contents;
  IREE_RETURN_IF_ERROR(ReadBinaryFile(path, &file_contents));
  if (!HasPrefix(file_contents, kId4TensorMagic,
                 IREE_ARRAYSIZE(kId4TensorMagic))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "fixture tensor is not an ID4 tensor payload: %s",
                            path.c_str());
  }
  if (file_contents.size() < IREE_ARRAYSIZE(kId4TensorMagic) + 4) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "fixture tensor ID4 header is truncated: %s",
                            path.c_str());
  }
  const iree_host_size_t header_length =
      static_cast<iree_host_size_t>(file_contents[8]) |
      (static_cast<iree_host_size_t>(file_contents[9]) << 8) |
      (static_cast<iree_host_size_t>(file_contents[10]) << 16) |
      (static_cast<iree_host_size_t>(file_contents[11]) << 24);
  const iree_host_size_t payload_offset = 12 + header_length;
  if (payload_offset > file_contents.size()) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "fixture tensor ID4 header overruns file: %s",
                            path.c_str());
  }
  iree_string_view_t header = iree_make_string_view(
      reinterpret_cast<const char*>(file_contents.data() + 12), header_length);
  iree_string_view_t version_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(header, IREE_SV("version"),
                                                     &version_value));
  uint64_t version = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(version_value, &version));
  if (version != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported ID4 tensor version %" PRIu64 " for %s",
                            version, path.c_str());
  }
  std::string kind;
  IREE_RETURN_IF_ERROR(JsonLookupSimpleString(header, IREE_SV("kind"), &kind));
  if (kind != "tensor") {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ID4 payload is not a tensor: %s", path.c_str());
  }
  std::string layout;
  IREE_RETURN_IF_ERROR(
      JsonLookupSimpleString(header, IREE_SV("layout"), &layout));
  if (layout != "dense-row-major") {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported ID4 tensor layout `%s` for %s",
                            layout.c_str(), path.c_str());
  }
  std::string dtype;
  IREE_RETURN_IF_ERROR(
      JsonLookupSimpleString(header, IREE_SV("dtype"), &dtype));
  id4_pipeline_tensor_dtype_t actual_dtype = ID4_PIPELINE_TENSOR_DTYPE_INVALID;
  IREE_RETURN_IF_ERROR(ParseFixtureDtype(
      iree_make_string_view(dtype.data(), dtype.size()), &actual_dtype));
  std::string storage_dtype;
  IREE_RETURN_IF_ERROR(
      JsonLookupSimpleString(header, IREE_SV("storage_dtype"), &storage_dtype));
  id4_pipeline_tensor_dtype_t actual_storage_dtype =
      ID4_PIPELINE_TENSOR_DTYPE_INVALID;
  IREE_RETURN_IF_ERROR(ParseFixtureDtype(
      iree_make_string_view(storage_dtype.data(), storage_dtype.size()),
      &actual_storage_dtype));
  if (actual_dtype != expected_dtype ||
      actual_storage_dtype != expected_dtype) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor ID4 dtype does not match manifest for %s",
        path.c_str());
  }
  iree_string_view_t shape_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(header, IREE_SV("shape"), &shape_value));
  id4_pipeline_tensor_shape_t actual_shape = {};
  IREE_RETURN_IF_ERROR(ParseFixtureShape(shape_value, &actual_shape));
  if (!ShapeEquals(actual_shape, expected_shape)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor ID4 shape does not match manifest for %s",
        path.c_str());
  }
  iree_string_view_t storage_shape_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      header, IREE_SV("storage_shape"), &storage_shape_value));
  id4_pipeline_tensor_shape_t actual_storage_shape = {};
  IREE_RETURN_IF_ERROR(
      ParseFixtureShape(storage_shape_value, &actual_storage_shape));
  if (!ShapeEquals(actual_storage_shape, expected_shape)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor ID4 storage shape does not match manifest for %s",
        path.c_str());
  }
  iree_string_view_t byte_length_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      header, IREE_SV("byte_length"), &byte_length_value));
  uint64_t byte_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(byte_length_value, &byte_length));
  if (byte_length != file_contents.size() - payload_offset) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor ID4 byte length does not match payload for %s",
        path.c_str());
  }
  out_payload->assign(file_contents.begin() + payload_offset,
                      file_contents.end());
  return iree_ok_status();
}

static iree_status_t LoadTensorPayload(
    const std::string& path, id4_pipeline_tensor_dtype_t expected_dtype,
    id4_pipeline_tensor_shape_t expected_shape,
    std::vector<uint8_t>* out_payload) {
  std::vector<uint8_t> file_contents;
  IREE_RETURN_IF_ERROR(ReadBinaryFile(path, &file_contents));
  if (HasPrefix(file_contents, kId4TensorMagic,
                IREE_ARRAYSIZE(kId4TensorMagic))) {
    return LoadExactTensorPayload(path, expected_dtype, expected_shape,
                                  out_payload);
  }
  return LoadNpyTensorPayload(path, expected_dtype, expected_shape,
                              out_payload);
}

iree_status_t LoadReferenceTensorPayload(
    iree_string_view_t file_path, id4_pipeline_tensor_dtype_t expected_dtype,
    id4_pipeline_tensor_shape_t expected_shape,
    std::vector<uint8_t>* out_payload) {
  return LoadTensorPayload(std::string(file_path.data, file_path.size),
                           expected_dtype, expected_shape, out_payload);
}

static iree_status_t LoadFixtureTensorRecord(
    iree_string_view_t record, FixtureTensorSet* fixture_tensors) {
  std::string kind;
  IREE_RETURN_IF_ERROR(JsonLookupSimpleString(record, IREE_SV("kind"), &kind));
  if (kind != "tensor") return iree_ok_status();

  FixtureTensor tensor;
  IREE_RETURN_IF_ERROR(
      JsonLookupSimpleString(record, IREE_SV("role"), &tensor.role));
  IREE_RETURN_IF_ERROR(
      JsonLookupSimpleString(record, IREE_SV("stage"), &tensor.stage));
  IREE_RETURN_IF_ERROR(
      JsonLookupSimpleString(record, IREE_SV("name"), &tensor.name));
  IREE_RETURN_IF_ERROR(
      JsonLookupSimpleString(record, IREE_SV("file"), &tensor.file));
  std::string dtype;
  IREE_RETURN_IF_ERROR(
      JsonLookupSimpleString(record, IREE_SV("dtype"), &dtype));
  IREE_RETURN_IF_ERROR(ParseFixtureDtype(
      iree_make_string_view(dtype.data(), dtype.size()), &tensor.dtype));
  iree_string_view_t shape_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(record, IREE_SV("shape"), &shape_value));
  IREE_RETURN_IF_ERROR(ParseFixtureShape(shape_value, &tensor.shape));
  iree_string_view_t source_shape_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      record, IREE_SV("source_shape"), &source_shape_value));
  if (iree_string_view_is_empty(source_shape_value) ||
      iree_string_view_equal(source_shape_value, IREE_SV("null"))) {
    tensor.source_shape = tensor.shape;
  } else {
    IREE_RETURN_IF_ERROR(
        ParseFixtureShape(source_shape_value, &tensor.source_shape));
  }
  if (tensor.source_shape.rank != tensor.shape.rank) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor `%s` source rank %u does not match tensor rank %u",
        tensor.name.c_str(), tensor.source_shape.rank, tensor.shape.rank);
  }
  IREE_RETURN_IF_ERROR(ParseFixtureSlice(record, &tensor));
  IREE_RETURN_IF_ERROR(ParseFixtureTolerance(record, &tensor));
  IREE_RETURN_IF_ERROR(
      LoadTensorPayload(JoinPath(fixture_tensors->directory, tensor.file),
                        tensor.dtype, tensor.shape, &tensor.payload));
  fixture_tensors->tensors.push_back(std::move(tensor));
  return iree_ok_status();
}

static const id4_pipeline_tensor_layout_t* BoundaryLayoutAt(
    iree_host_size_t index, const void* user_data) {
  const id4_pipeline_plan_t* plan =
      static_cast<const id4_pipeline_plan_t*>(user_data);
  const id4_pipeline_boundary_tensor_plan_t* boundary =
      id4_pipeline_plan_boundary_tensor_at(plan, index);
  return boundary ? &boundary->layout : nullptr;
}

static const id4_pipeline_tensor_layout_t* DiagnosticTapLayoutAt(
    iree_host_size_t index, const void* user_data) {
  const id4_pipeline_plan_t* plan =
      static_cast<const id4_pipeline_plan_t*>(user_data);
  const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, index);
  return diagnostic_tap ? &diagnostic_tap->layout : nullptr;
}

static iree_status_t AllocateBufferBindingSet(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t count,
    const id4_pipeline_tensor_layout_t* (*layout_at)(iree_host_size_t index,
                                                     const void* user_data),
    const void* user_data, BufferBindingSet* out_binding_set) {
  IREE_ASSERT_ARGUMENT(out_binding_set);
  out_binding_set->reset();
  if (count == 0) return iree_ok_status();

  iree_status_t status = iree_allocator_malloc_array(
      iree_allocator_system(), count, sizeof(out_binding_set->buffers[0]),
      reinterpret_cast<void**>(&out_binding_set->buffers));
  if (iree_status_is_ok(status)) {
    std::memset(out_binding_set->buffers, 0,
                count * sizeof(out_binding_set->buffers[0]));
    status = iree_allocator_malloc_array(
        iree_allocator_system(), count, sizeof(out_binding_set->bindings[0]),
        reinterpret_cast<void**>(&out_binding_set->bindings));
  }
  if (iree_status_is_ok(status)) {
    std::memset(out_binding_set->bindings, 0,
                count * sizeof(out_binding_set->bindings[0]));
    out_binding_set->count = count;
  }

  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_tensor_layout_t* layout = layout_at(i, user_data);
    if (!layout) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "missing tensor layout %" PRIhsz, i);
      break;
    }
    iree_hal_buffer_params_t params;
    std::memset(&params, 0, sizeof(params));
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                   IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
    params.queue_affinity = queue_affinity;
    params.min_alignment = layout->alignment ? layout->alignment : 1;
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, layout->byte_length,
        &out_binding_set->buffers[i]);
    if (iree_status_is_ok(status)) {
      out_binding_set->bindings[i] = iree_hal_buffer_binding_t{
          // Tensor buffer supplied in plan order.
          /*.buffer=*/out_binding_set->buffers[i],
          // Test buffers are exact standalone allocations.
          /*.offset=*/0,
          // Full planned tensor byte range.
          /*.length=*/layout->byte_length,
      };
    }
  }
  if (!iree_status_is_ok(status)) {
    out_binding_set->reset();
  }
  return status;
}

static iree_status_t CaptureDiagnostics(
    void* user_data, const id4_pipeline_diagnostic_event_t* event) {
  StageDiagnostics* diagnostics = static_cast<StageDiagnostics*>(user_data);
  ++diagnostics->event_count;
  if (event->kind == ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_KERNEL) {
    ++diagnostics->kernel_event_count;
  }
  if (event->timing && event->parameter_load &&
      iree_string_view_equal(event->key,
                             IREE_SV("parameter_slab.load_group.submit"))) {
    const iree_duration_t duration_ns = event->timing->duration_ns;
    ++diagnostics->parameter_load_group_submit_count;
    diagnostics->parameter_load_group_submit_duration_ns += duration_ns;
    diagnostics->parameter_load_group_submit_max_duration_ns = iree_max(
        diagnostics->parameter_load_group_submit_max_duration_ns, duration_ns);
    if (event->parameter_load->submit_execution_ordinal != IREE_HOST_SIZE_MAX &&
        event->parameter_load->first_consumer_execution_ordinal !=
            IREE_HOST_SIZE_MAX &&
        event->parameter_load->submit_execution_ordinal <
            event->parameter_load->first_consumer_execution_ordinal) {
      const iree_host_size_t segment_distance =
          event->parameter_load->first_consumer_execution_ordinal -
          event->parameter_load->submit_execution_ordinal;
      ++diagnostics->parameter_load_group_prefetch_submit_count;
      diagnostics->parameter_load_group_prefetch_segment_distance_sum +=
          segment_distance;
      diagnostics->parameter_load_group_prefetch_segment_distance_max =
          iree_max(
              diagnostics->parameter_load_group_prefetch_segment_distance_max,
              segment_distance);
    }
    if (iree_string_view_equal(event->parameter_load->load_group_kind,
                               IREE_SV("gather"))) {
      ++diagnostics->parameter_load_group_submit_gather_count;
      diagnostics->parameter_load_group_submit_gather_duration_ns +=
          duration_ns;
    } else if (iree_string_view_equal(event->parameter_load->load_group_kind,
                                      IREE_SV("encode"))) {
      ++diagnostics->parameter_load_group_submit_encode_count;
      diagnostics->parameter_load_group_submit_encode_duration_ns +=
          duration_ns;
    }
  }
  if (event->parameter_load &&
      iree_string_view_equal(event->key,
                             IREE_SV("parameter_slab.gather_group"))) {
    const id4_pipeline_parameter_load_diagnostic_t* parameter_load =
        event->parameter_load;
    ++diagnostics->parameter_direct_gather_group_count;
    diagnostics->parameter_direct_gather_request_count +=
        parameter_load->logical_source_count;
    diagnostics->parameter_direct_gather_source_byte_length +=
        parameter_load->source_byte_length;
    diagnostics->parameter_direct_gather_target_byte_length +=
        parameter_load->target_byte_length;
    diagnostics->parameter_direct_gather_max_source_byte_length =
        iree_max(diagnostics->parameter_direct_gather_max_source_byte_length,
                 parameter_load->source_byte_length);
  }
  if (event->parameter_load &&
      iree_string_view_equal(event->key,
                             IREE_SV("parameter_slab.encode_window"))) {
    const id4_pipeline_parameter_load_diagnostic_t* parameter_load =
        event->parameter_load;
    ++diagnostics->parameter_encode_window_count;
    diagnostics->parameter_encode_window_staging_total_byte_length +=
        parameter_load->staging_total_byte_length;
    diagnostics->parameter_encode_window_staging_max_byte_length =
        iree_max(diagnostics->parameter_encode_window_staging_max_byte_length,
                 parameter_load->staging_total_byte_length);
    diagnostics->parameter_encode_window_source_byte_length +=
        parameter_load->source_byte_length;
    diagnostics->parameter_encode_window_target_byte_length +=
        parameter_load->target_byte_length;
    diagnostics->parameter_encode_window_staging_chunk_count +=
        parameter_load->staging_chunk_count;
    diagnostics->parameter_encode_window_logical_source_count +=
        parameter_load->logical_source_count;
    diagnostics->parameter_encode_window_source_gather_batch_count +=
        parameter_load->source_gather_batch_count;
    diagnostics->parameter_encode_window_encoder_dispatch_count +=
        parameter_load->encoder_dispatch_count;
  }
  if (event->parameter_load &&
      iree_string_view_equal(event->key,
                             IREE_SV("parameter_slab.prepare_encode_window"))) {
    const id4_pipeline_parameter_load_diagnostic_t* parameter_load =
        event->parameter_load;
    ++diagnostics->parameter_prepare_encode_window_count;
    diagnostics->parameter_prepare_encode_window_staging_total_byte_length +=
        parameter_load->staging_total_byte_length;
    diagnostics
        ->parameter_prepare_encode_window_staging_max_byte_length = iree_max(
        diagnostics->parameter_prepare_encode_window_staging_max_byte_length,
        parameter_load->staging_total_byte_length);
    diagnostics->parameter_prepare_encode_window_source_byte_length +=
        parameter_load->source_byte_length;
    diagnostics->parameter_prepare_encode_window_target_byte_length +=
        parameter_load->target_byte_length;
    diagnostics->parameter_prepare_encode_window_staging_chunk_count +=
        parameter_load->staging_chunk_count;
    diagnostics->parameter_prepare_encode_window_logical_source_count +=
        parameter_load->logical_source_count;
    diagnostics->parameter_prepare_encode_window_source_gather_batch_count +=
        parameter_load->source_gather_batch_count;
    diagnostics->parameter_prepare_encode_window_encoder_dispatch_count +=
        parameter_load->encoder_dispatch_count;
  }
  if (event->parameter_load &&
      iree_string_view_equal(event->key,
                             IREE_SV("parameter_slab.issue_encode_window"))) {
    const id4_pipeline_parameter_load_diagnostic_t* parameter_load =
        event->parameter_load;
    ++diagnostics->parameter_issue_encode_window_count;
    diagnostics->parameter_issue_encode_window_staging_total_byte_length +=
        parameter_load->staging_total_byte_length;
    diagnostics->parameter_issue_encode_window_staging_max_byte_length =
        iree_max(
            diagnostics->parameter_issue_encode_window_staging_max_byte_length,
            parameter_load->staging_total_byte_length);
    diagnostics->parameter_issue_encode_window_source_byte_length +=
        parameter_load->source_byte_length;
    diagnostics->parameter_issue_encode_window_target_byte_length +=
        parameter_load->target_byte_length;
    diagnostics->parameter_issue_encode_window_staging_chunk_count +=
        parameter_load->staging_chunk_count;
    diagnostics->parameter_issue_encode_window_logical_source_count +=
        parameter_load->logical_source_count;
    diagnostics->parameter_issue_encode_window_source_gather_batch_count +=
        parameter_load->source_gather_batch_count;
    diagnostics->parameter_issue_encode_window_encoder_dispatch_count +=
        parameter_load->encoder_dispatch_count;
  }
  return iree_ok_status();
}

id4_pipeline_diagnostics_sink_t DiagnosticsSink(StageDiagnostics* diagnostics) {
  return (id4_pipeline_diagnostics_sink_t){
      // Callback used to count diagnostics.
      /*.emit=*/CaptureDiagnostics,
      // Caller-owned diagnostics storage.
      /*.user_data=*/diagnostics,
  };
}

static iree_status_t RequireSingleDeviceFlag() {
  iree_string_view_list_t devices = iree_hal_device_flag_list();
  if (devices.count == 1) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "live stage integration tests require exactly one --device= flag; "
      "received %" PRIhsz,
      devices.count);
}

static void ResetLiveStageContext(LiveStageContext* context) {
  context->kernel_cache.reset();
  context->executable_cache.reset();
  context->device.reset();
  context->device_group.reset();
  context->frontier_tracker.reset();
  context->proactor_pool.reset();
  context->command_buffer_mode = IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
}

static iree_status_t InitializeLiveStageContextFromFlags(
    LiveStageContext* out_context) {
  ResetLiveStageContext(out_context);

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  iree_status_t status = iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool);
  if (iree_status_is_ok(status)) {
    out_context->proactor_pool.reset(proactor_pool);
  }

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = out_context->proactor_pool.get();
  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_device_from_flags(
        iree_hal_available_driver_registry(), iree_string_view_empty(),
        &create_params, iree_allocator_system(), &device);
  }
  if (iree_status_is_ok(status)) {
    out_context->device.reset(device);
  }

  iree_async_frontier_tracker_options_t tracker_options =
      iree_async_frontier_tracker_options_default();
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &frontier_tracker);
  }
  if (iree_status_is_ok(status)) {
    out_context->frontier_tracker.reset(frontier_tracker);
  }

  iree_hal_device_group_t* device_group = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        out_context->device.get(), out_context->frontier_tracker.get(),
        iree_allocator_system(), &device_group);
  }
  if (iree_status_is_ok(status)) {
    out_context->device_group.reset(device_group);
  }

  iree_hal_executable_cache_t* executable_cache = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_cache_create(
        out_context->device.get(), IREE_SV("id4.stage"), &executable_cache);
  }
  if (iree_status_is_ok(status)) {
    out_context->executable_cache.reset(executable_cache);
  }

  id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
  memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
  kernel_cache_options.structure_size = sizeof(kernel_cache_options);
  kernel_cache_options.target_processor =
      id4_pipeline_kernel_cache_default_target_processor();
  kernel_cache_options.entry_limit =
      ID4_PIPELINE_KERNEL_CACHE_INTERACTIVE_ENTRY_LIMIT;
  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_create(
        &kernel_cache_options, iree_allocator_system(), &kernel_cache);
  }
  if (iree_status_is_ok(status)) {
    out_context->kernel_cache.reset(kernel_cache);
  }
  if (iree_status_is_ok(status)) {
    status = CommandBufferModeFromFlags(&out_context->command_buffer_mode);
  }
  return status;
}

static void RetainLiveStageContext(const LiveStageContext& source,
                                   LiveStageContext* out_context) {
  ResetLiveStageContext(out_context);
  iree_async_proactor_pool_retain(source.proactor_pool.get());
  out_context->proactor_pool.reset(source.proactor_pool.get());
  iree_async_frontier_tracker_retain(source.frontier_tracker.get());
  out_context->frontier_tracker.reset(source.frontier_tracker.get());
  iree_hal_device_group_retain(source.device_group.get());
  out_context->device_group.reset(source.device_group.get());
  iree_hal_device_retain(source.device.get());
  out_context->device.reset(source.device.get());
  iree_hal_executable_cache_retain(source.executable_cache.get());
  out_context->executable_cache.reset(source.executable_cache.get());
  id4_pipeline_kernel_cache_retain(source.kernel_cache.get());
  out_context->kernel_cache.reset(source.kernel_cache.get());
  out_context->command_buffer_mode = source.command_buffer_mode;
}

iree_status_t CreateLiveStageContextFromFlags(LiveStageContext* out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_RETURN_IF_ERROR(RequireSingleDeviceFlag());

  static std::mutex shared_context_mutex;
  static bool shared_context_initialized = false;
  static LiveStageContext shared_context;

  std::lock_guard<std::mutex> lock(shared_context_mutex);
  if (!shared_context_initialized) {
    iree_status_t status = InitializeLiveStageContextFromFlags(&shared_context);
    if (!iree_status_is_ok(status)) {
      ResetLiveStageContext(&shared_context);
      return status;
    }
    shared_context_initialized = true;
  }
  RetainLiveStageContext(shared_context, out_context);
  return iree_ok_status();
}

iree_status_t CreateEmbeddedKernelLibrary(
    id4_pipeline_kernel_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = nullptr;

  const iree_file_toc_t* toc = id4_kernel_embedded_loom_sources_create();
  const iree_host_size_t file_count = id4_kernel_embedded_loom_sources_size();
  id4_pipeline_kernel_source_file_t* source_files = nullptr;
  iree_status_t status = iree_ok_status();
  if (file_count != 0) {
    status = iree_allocator_malloc_array(
        iree_allocator_system(), file_count, sizeof(source_files[0]),
        reinterpret_cast<void**>(&source_files));
  }
  for (iree_host_size_t i = 0; i < file_count && iree_status_is_ok(status);
       ++i) {
    source_files[i] = id4_pipeline_kernel_source_file_t{
        // Source identifier formatted as <module_path>.loom.
        /*.source_identifier=*/iree_make_cstring_view(toc[i].name),
        // Embedded Loom source payload.
        /*.source_contents=*/
        iree_make_const_byte_span(toc[i].data, toc[i].size),
    };
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_library_create_from_source_files(
        file_count, source_files, iree_allocator_system(), out_library);
  }
  iree_allocator_free(iree_allocator_system(), source_files);
  return status;
}

iree_status_t CreateParameterProviderFromFlags(
    iree_string_view_t scope, iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = nullptr;

  iree_io_scope_map_t scope_map;
  iree_io_scope_map_initialize(iree_allocator_system(), &scope_map);
  iree_status_t status =
      iree_tooling_build_parameter_indices_from_flags(&scope_map);

  iree_io_parameter_index_t* index = nullptr;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < scope_map.count; ++i) {
      iree_io_scope_map_entry_t* entry = scope_map.entries[i];
      if (iree_string_view_equal(entry->scope, scope)) {
        index = entry->index;
        break;
      }
    }
    if (!index) {
      if (iree_string_view_is_empty(scope)) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "required anonymous parameter scope was not "
                                  "loaded; pass --parameters=<file>");
      } else {
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "required parameter scope `%.*s` was not loaded; pass "
            "--parameters=%.*s=<file>",
            static_cast<int>(scope.size), scope.data,
            static_cast<int>(scope.size), scope.data);
      }
    }
  }
  if (iree_status_is_ok(status) && iree_io_parameter_index_count(index) == 0) {
    if (iree_string_view_is_empty(scope)) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "anonymous parameter scope was loaded with no parameters");
    } else {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "parameter scope `%.*s` was loaded with no parameters",
          static_cast<int>(scope.size), scope.data);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        scope, index,
        IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
        iree_allocator_system(), out_provider);
  }
  iree_io_scope_map_deinitialize(&scope_map);
  return status;
}

iree_status_t AllocateBoundaryBindings(iree_hal_device_t* device,
                                       iree_hal_queue_affinity_t queue_affinity,
                                       const id4_pipeline_plan_t* plan,
                                       BufferBindingSet* out_binding_set) {
  return AllocateBufferBindingSet(device, queue_affinity,
                                  id4_pipeline_plan_boundary_tensor_count(plan),
                                  BoundaryLayoutAt, plan, out_binding_set);
}

iree_status_t AllocateDiagnosticTapBindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, BufferBindingSet* out_binding_set) {
  return AllocateBufferBindingSet(device, queue_affinity,
                                  id4_pipeline_plan_diagnostic_tap_count(plan),
                                  DiagnosticTapLayoutAt, plan, out_binding_set);
}

iree_status_t FindBoundaryBinding(const id4_pipeline_plan_t* plan,
                                  const BufferBindingSet& binding_set,
                                  iree_string_view_t name,
                                  iree_hal_buffer_binding_t* out_binding) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      *out_binding = binding_set.bindings[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "boundary tensor `%.*s` not found",
                          static_cast<int>(name.size), name.data);
}

iree_status_t FindDiagnosticTapBinding(const id4_pipeline_plan_t* plan,
                                       const BufferBindingSet& binding_set,
                                       iree_string_view_t name,
                                       iree_hal_buffer_binding_t* out_binding) {
  for (iree_host_size_t i = 0; i < id4_pipeline_plan_diagnostic_tap_count(plan);
       ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (diagnostic_tap && iree_string_view_equal(diagnostic_tap->name, name)) {
      *out_binding = binding_set.bindings[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "diagnostic tap `%.*s` not found",
                          static_cast<int>(name.size), name.data);
}

iree_status_t QueueUpdateBinding(iree_hal_device_t* device,
                                 iree_hal_queue_affinity_t queue_affinity,
                                 const iree_hal_buffer_binding_t* binding,
                                 const void* source_data,
                                 iree_host_size_t source_length,
                                 iree_hal_semaphore_t* semaphore,
                                 uint64_t* inout_payload_value) {
  if (source_length != binding->length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source length %" PRIhsz " does not match binding length %" PRIu64,
        source_length, static_cast<uint64_t>(binding->length));
  }
  iree_host_size_t source_offset = 0;
  while (source_offset < source_length) {
    const iree_host_size_t remaining_length = source_length - source_offset;
    const iree_host_size_t chunk_length =
        remaining_length > kQueueUpdateChunkLength ? kQueueUpdateChunkLength
                                                   : remaining_length;
    iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
    SemaphoreListStorage wait_storage;
    wait_storage.semaphore = semaphore;
    wait_storage.payload_value = *inout_payload_value;
    if (wait_storage.payload_value != 0) {
      wait_list = wait_storage.list();
    }
    SemaphoreListStorage signal_storage;
    signal_storage.semaphore = semaphore;
    signal_storage.payload_value = wait_storage.payload_value + 1;
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_update(
        device, queue_affinity, wait_list, signal_storage.list(), source_data,
        source_offset, binding->buffer, binding->offset + source_offset,
        chunk_length, IREE_HAL_UPDATE_FLAG_NONE));
    *inout_payload_value = signal_storage.payload_value;
    source_offset += chunk_length;
  }
  return iree_ok_status();
}

iree_status_t QueueReadBindingFromHostAllocation(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding, const void* source_data,
    iree_host_size_t source_length, iree_hal_semaphore_t* semaphore,
    uint64_t* inout_payload_value) {
  if (source_length != binding->length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source length %" PRIhsz " does not match binding length %" PRIu64,
        source_length, static_cast<uint64_t>(binding->length));
  }

  iree_io_file_handle_t* handle = nullptr;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(const_cast<void*>(source_data), source_length),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &handle));

  iree_hal_file_t* file = nullptr;
  iree_status_t status =
      iree_hal_file_import(device, queue_affinity, IREE_HAL_MEMORY_ACCESS_READ,
                           handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file);
  iree_io_file_handle_release(handle);

  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  SemaphoreListStorage wait_storage;
  wait_storage.semaphore = semaphore;
  wait_storage.payload_value = *inout_payload_value;
  if (wait_storage.payload_value != 0) {
    wait_list = wait_storage.list();
  }
  SemaphoreListStorage signal_storage;
  signal_storage.semaphore = semaphore;
  signal_storage.payload_value = wait_storage.payload_value + 1;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_read(
        device, queue_affinity, wait_list, signal_storage.list(), file,
        /*source_offset=*/0, binding->buffer, binding->offset, binding->length,
        IREE_HAL_READ_FLAG_NONE);
  }
  iree_hal_file_release(file);
  if (iree_status_is_ok(status)) {
    *inout_payload_value = signal_storage.payload_value;
  }
  return status;
}

iree_status_t QueueFillBinding(iree_hal_device_t* device,
                               iree_hal_queue_affinity_t queue_affinity,
                               const iree_hal_buffer_binding_t* binding,
                               const void* pattern,
                               iree_host_size_t pattern_length,
                               iree_hal_semaphore_t* semaphore,
                               uint64_t* inout_payload_value) {
  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  SemaphoreListStorage wait_storage;
  wait_storage.semaphore = semaphore;
  wait_storage.payload_value = *inout_payload_value;
  if (wait_storage.payload_value != 0) {
    wait_list = wait_storage.list();
  }
  SemaphoreListStorage signal_storage;
  signal_storage.semaphore = semaphore;
  signal_storage.payload_value = wait_storage.payload_value + 1;
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
      device, queue_affinity, wait_list, signal_storage.list(), binding->buffer,
      binding->offset, binding->length, pattern, pattern_length,
      IREE_HAL_FILL_FLAG_NONE));
  *inout_payload_value = signal_storage.payload_value;
  return iree_ok_status();
}

iree_status_t QueueFillBoundaryTensors(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BufferBindingSet& binding_set,
    id4_pipeline_boundary_tensor_flags_t required_flags, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_semaphore_t* fill_semaphore,
    uint64_t* out_fill_value) {
  IREE_ASSERT_ARGUMENT(out_fill_value);
  if (required_flags == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "boundary fill requires at least one selector flag");
  }
  if (!pattern || pattern_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary fill pattern is required");
  }
  for (iree_host_size_t i = 0; i < binding_set.count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary || !iree_all_bits_set(boundary->flags, required_flags)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(QueueFillBinding(
        device, queue_affinity, &binding_set.bindings[i], pattern,
        pattern_length, fill_semaphore, out_fill_value));
  }
  return iree_ok_status();
}

iree_status_t QueueFillDiagnosticTapTensors(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const BufferBindingSet& binding_set, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_semaphore_t* fill_semaphore,
    uint64_t* out_fill_value) {
  IREE_ASSERT_ARGUMENT(out_fill_value);
  if (!pattern || pattern_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic tap fill pattern is required");
  }
  for (iree_host_size_t i = 0; i < binding_set.count; ++i) {
    IREE_RETURN_IF_ERROR(QueueFillBinding(
        device, queue_affinity, &binding_set.bindings[i], pattern,
        pattern_length, fill_semaphore, out_fill_value));
  }
  return iree_ok_status();
}

iree_status_t QueueUpdateBoundaryTensorFromFixture(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BufferBindingSet& binding_set,
    iree_string_view_t boundary_name, const FixtureTensorSet& fixture_tensors,
    iree_string_view_t fixture_name, iree_hal_semaphore_t* update_semaphore,
    uint64_t* inout_update_value) {
  const FixtureTensor* fixture_tensor =
      fixture_tensors.FindTensor(IREE_SV("input"), fixture_name);
  if (!fixture_tensor) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "fixture is missing required `%.*s` input",
        static_cast<int>(fixture_name.size), fixture_name.data);
  }
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary ||
        !iree_string_view_equal(boundary->layout.name, boundary_name)) {
      continue;
    }
    if (i >= binding_set.count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "boundary tensor `%.*s` has no allocated binding",
                              static_cast<int>(boundary_name.size),
                              boundary_name.data);
    }
    if (fixture_tensor->dtype != boundary->layout.dtype) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%.*s` dtype does not match boundary `%.*s` dtype",
          static_cast<int>(fixture_name.size), fixture_name.data,
          static_cast<int>(boundary_name.size), boundary_name.data);
    }
    if (!ShapeEquals(fixture_tensor->shape, boundary->layout.shape)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%.*s` shape does not match boundary `%.*s` shape",
          static_cast<int>(fixture_name.size), fixture_name.data,
          static_cast<int>(boundary_name.size), boundary_name.data);
    }
    if (fixture_tensor->payload.size() != boundary->layout.byte_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%.*s` payload length %zu does not match boundary "
          "`%.*s` byte length %" PRIu64,
          static_cast<int>(fixture_name.size), fixture_name.data,
          fixture_tensor->payload.size(), static_cast<int>(boundary_name.size),
          boundary_name.data,
          static_cast<uint64_t>(boundary->layout.byte_length));
    }
    return QueueUpdateBinding(device, queue_affinity, &binding_set.bindings[i],
                              fixture_tensor->payload.data(),
                              fixture_tensor->payload.size(), update_semaphore,
                              inout_update_value);
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND, "boundary tensor `%.*s` not found",
      static_cast<int>(boundary_name.size), boundary_name.data);
}

iree_status_t ReadBindingToHost(iree_hal_device_t* device,
                                iree_hal_queue_affinity_t queue_affinity,
                                const iree_hal_buffer_binding_t* binding,
                                iree_hal_semaphore_list_t wait_list,
                                std::vector<uint8_t>* out_bytes) {
  out_bytes->assign(static_cast<size_t>(binding->length), 0);

  iree_io_file_handle_t* handle = nullptr;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(out_bytes->data(), out_bytes->size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &handle));

  iree_hal_file_t* file = nullptr;
  iree_status_t status =
      iree_hal_file_import(device, queue_affinity, IREE_HAL_MEMORY_ACCESS_WRITE,
                           handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file);
  iree_io_file_handle_release(handle);

  OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release> read_semaphore;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(device, queue_affinity, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                       read_semaphore.out());
  }
  SemaphoreListStorage read_signal;
  read_signal.semaphore = read_semaphore.get();
  read_signal.payload_value = 1;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_write(
        device, queue_affinity, wait_list, read_signal.list(), binding->buffer,
        binding->offset, file, /*target_offset=*/0, binding->length,
        IREE_HAL_WRITE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(
        read_semaphore.get(), read_signal.payload_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_file_release(file);
  return status;
}

static float LoadF32(const uint8_t* bytes) {
  float value = 0.0f;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

static uint16_t LoadU16(const uint8_t* bytes) {
  uint16_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

static iree_status_t LoadTensorElementAsF32(id4_pipeline_tensor_dtype_t dtype,
                                            const std::vector<uint8_t>& bytes,
                                            iree_host_size_t index,
                                            float* out_value) {
  const iree_device_size_t dtype_byte_length =
      id4_pipeline_tensor_dtype_byte_length(dtype);
  if (dtype_byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "actual tensor dtype is invalid");
  }
  if (index >
      std::numeric_limits<iree_host_size_t>::max() / dtype_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "actual tensor byte offset overflow");
  }
  const iree_host_size_t byte_offset =
      index * static_cast<iree_host_size_t>(dtype_byte_length);
  if (byte_offset > bytes.size() ||
      bytes.size() - byte_offset < dtype_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "actual tensor element index is out of range");
  }
  switch (dtype) {
    case ID4_PIPELINE_TENSOR_DTYPE_F32:
      *out_value = LoadF32(&bytes[byte_offset]);
      return iree_ok_status();
    case ID4_PIPELINE_TENSOR_DTYPE_F16:
      *out_value = iree_math_f16_to_f32(LoadU16(&bytes[byte_offset]));
      return iree_ok_status();
    case ID4_PIPELINE_TENSOR_DTYPE_BF16:
      *out_value = iree_math_bf16_to_f32(LoadU16(&bytes[byte_offset]));
      return iree_ok_status();
    default: {
      iree_string_view_t dtype_name = id4_pipeline_tensor_dtype_format(dtype);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "actual tensor dtype `%.*s` cannot be compared as F32",
          static_cast<int>(dtype_name.size), dtype_name.data);
    }
  }
}

static iree_status_t PushF32AbsoluteError(
    const FixtureTensor& expected_tensor,
    id4_pipeline_tensor_dtype_t actual_dtype, iree_host_size_t expected_index,
    iree_host_size_t actual_index, const std::vector<uint8_t>& actual,
    std::vector<float>* absolute_errors) {
  float actual_value = 0.0f;
  IREE_RETURN_IF_ERROR(LoadTensorElementAsF32(actual_dtype, actual,
                                              actual_index, &actual_value));
  const float expected_value =
      LoadF32(&expected_tensor.payload[expected_index * sizeof(float)]);
  absolute_errors->push_back(std::fabs(actual_value - expected_value));
  return iree_ok_status();
}

static iree_status_t ShapeElementCount(id4_pipeline_tensor_shape_t shape,
                                       iree_host_size_t* out_element_count) {
  IREE_ASSERT_ARGUMENT(out_element_count);
  iree_host_size_t element_count = 1;
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (shape.dims[i] > std::numeric_limits<iree_host_size_t>::max()) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor dimension %u exceeds host size", i);
    }
    iree_host_size_t dim = static_cast<iree_host_size_t>(shape.dims[i]);
    if (dim == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tensor dimension %u is zero", i);
    }
    if (element_count > std::numeric_limits<iree_host_size_t>::max() / dim) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor element count overflow");
    }
    element_count *= dim;
  }
  *out_element_count = element_count;
  return iree_ok_status();
}

static iree_status_t ShapeRowMajorStrides(
    id4_pipeline_tensor_shape_t shape,
    iree_host_size_t out_strides[ID4_PIPELINE_TENSOR_MAX_RANK]) {
  if (shape.rank == 0) return iree_ok_status();
  out_strides[shape.rank - 1] = 1;
  for (uint32_t i = shape.rank - 1; i > 0; --i) {
    if (shape.dims[i] > std::numeric_limits<iree_host_size_t>::max()) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor dimension %u exceeds host size", i);
    }
    iree_host_size_t dim = static_cast<iree_host_size_t>(shape.dims[i]);
    if (dim == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tensor dimension %u is zero", i);
    }
    if (out_strides[i] > std::numeric_limits<iree_host_size_t>::max() / dim) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor stride overflow");
    }
    out_strides[i - 1] = out_strides[i] * dim;
  }
  return iree_ok_status();
}

static iree_status_t CompareF32Element(const FixtureTensor& expected_tensor,
                                       id4_pipeline_tensor_dtype_t actual_dtype,
                                       iree_host_size_t expected_index,
                                       iree_host_size_t actual_index,
                                       const std::vector<uint8_t>& actual) {
  float actual_value = 0.0f;
  IREE_RETURN_IF_ERROR(LoadTensorElementAsF32(actual_dtype, actual,
                                              actual_index, &actual_value));
  const float expected_value =
      LoadF32(&expected_tensor.payload[expected_index * sizeof(float)]);
  const double tolerance = expected_tensor.tolerance.absolute_tolerance +
                           expected_tensor.tolerance.relative_tolerance *
                               std::fabs((double)expected_value);
  const double absolute_error =
      std::fabs((double)actual_value - (double)expected_value);
  if (absolute_error <= tolerance) return iree_ok_status();
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "tensor `%s` mismatch at element %" PRIhsz
                          ": actual=%g expected=%g abs_error=%g tolerance=%g",
                          expected_tensor.name.c_str(), expected_index,
                          (double)actual_value, (double)expected_value,
                          absolute_error, tolerance);
}

static iree_status_t CompareF32DensePayload(
    const FixtureTensor& expected_tensor,
    const id4_pipeline_tensor_layout_t* actual_layout,
    const std::vector<uint8_t>& actual) {
  iree_host_size_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      ShapeElementCount(expected_tensor.shape, &element_count));
  if (FixtureToleranceIsAggregate(expected_tensor)) {
    std::vector<float> absolute_errors;
    absolute_errors.reserve(element_count);
    for (iree_host_size_t i = 0; i < element_count; ++i) {
      IREE_RETURN_IF_ERROR(PushF32AbsoluteError(expected_tensor,
                                                actual_layout->dtype, i, i,
                                                actual, &absolute_errors));
    }
    return VerifyAggregateTolerance(expected_tensor, &absolute_errors);
  }
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    IREE_RETURN_IF_ERROR(
        CompareF32Element(expected_tensor, actual_layout->dtype, i, i, actual));
  }
  return iree_ok_status();
}

static iree_status_t CompareF32SourceSlice(
    const FixtureTensor& expected_tensor,
    const id4_pipeline_tensor_layout_t* actual_layout,
    const std::vector<uint8_t>& actual) {
  iree_host_size_t expected_strides[ID4_PIPELINE_TENSOR_MAX_RANK] = {};
  IREE_RETURN_IF_ERROR(
      ShapeRowMajorStrides(expected_tensor.shape, expected_strides));
  iree_host_size_t actual_strides[ID4_PIPELINE_TENSOR_MAX_RANK] = {};
  IREE_RETURN_IF_ERROR(
      ShapeRowMajorStrides(expected_tensor.source_shape, actual_strides));

  iree_host_size_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      ShapeElementCount(expected_tensor.shape, &element_count));
  std::vector<float> absolute_errors;
  if (FixtureToleranceIsAggregate(expected_tensor)) {
    absolute_errors.reserve(element_count);
  }
  for (iree_host_size_t expected_index = 0; expected_index < element_count;
       ++expected_index) {
    iree_host_size_t remaining = expected_index;
    iree_host_size_t actual_index = 0;
    for (uint32_t dim = 0; dim < expected_tensor.shape.rank; ++dim) {
      const iree_host_size_t coordinate = remaining / expected_strides[dim];
      remaining = remaining % expected_strides[dim];
      actual_index += (static_cast<iree_host_size_t>(
                           expected_tensor.slice_offsets.dims[dim]) +
                       coordinate) *
                      actual_strides[dim];
    }
    if (FixtureToleranceIsAggregate(expected_tensor)) {
      IREE_RETURN_IF_ERROR(PushF32AbsoluteError(
          expected_tensor, actual_layout->dtype, expected_index, actual_index,
          actual, &absolute_errors));
    } else {
      IREE_RETURN_IF_ERROR(
          CompareF32Element(expected_tensor, actual_layout->dtype,
                            expected_index, actual_index, actual));
    }
  }
  if (FixtureToleranceIsAggregate(expected_tensor)) {
    return VerifyAggregateTolerance(expected_tensor, &absolute_errors);
  }
  return iree_ok_status();
}

static iree_status_t CompareRoundedF32ToBf16Element(
    const FixtureTensor& expected_tensor, iree_host_size_t expected_index,
    iree_host_size_t actual_index, const std::vector<uint8_t>& actual) {
  if (actual_index >
      std::numeric_limits<iree_host_size_t>::max() / sizeof(uint16_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "actual tensor byte offset overflow");
  }
  const iree_host_size_t actual_byte_offset = actual_index * sizeof(uint16_t);
  if (actual_byte_offset > actual.size() ||
      actual.size() - actual_byte_offset < sizeof(uint16_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "actual tensor element index is out of range");
  }
  const uint16_t actual_bits = LoadU16(&actual[actual_byte_offset]);
  const float expected_value =
      LoadF32(&expected_tensor.payload[expected_index * sizeof(float)]);
  const uint16_t expected_bits = iree_math_f32_to_bf16(expected_value);
  if (actual_bits == expected_bits) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "tensor `%s` mismatch at element %" PRIhsz
      ": actual_bf16=0x%04x expected_bf16=0x%04x actual=%g expected=%g",
      expected_tensor.name.c_str(), expected_index, (unsigned)actual_bits,
      (unsigned)expected_bits, (double)iree_math_bf16_to_f32(actual_bits),
      (double)iree_math_bf16_to_f32(expected_bits));
}

static iree_status_t CompareRoundedF32ToBf16DensePayload(
    const FixtureTensor& expected_tensor, const std::vector<uint8_t>& actual) {
  iree_host_size_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      ShapeElementCount(expected_tensor.shape, &element_count));
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    IREE_RETURN_IF_ERROR(
        CompareRoundedF32ToBf16Element(expected_tensor, i, i, actual));
  }
  return iree_ok_status();
}

static iree_status_t CompareRoundedF32ToBf16SourceSlice(
    const FixtureTensor& expected_tensor, const std::vector<uint8_t>& actual) {
  iree_host_size_t expected_strides[ID4_PIPELINE_TENSOR_MAX_RANK] = {};
  IREE_RETURN_IF_ERROR(
      ShapeRowMajorStrides(expected_tensor.shape, expected_strides));
  iree_host_size_t actual_strides[ID4_PIPELINE_TENSOR_MAX_RANK] = {};
  IREE_RETURN_IF_ERROR(
      ShapeRowMajorStrides(expected_tensor.source_shape, actual_strides));

  iree_host_size_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      ShapeElementCount(expected_tensor.shape, &element_count));
  for (iree_host_size_t expected_index = 0; expected_index < element_count;
       ++expected_index) {
    iree_host_size_t remaining = expected_index;
    iree_host_size_t actual_index = 0;
    for (uint32_t dim = 0; dim < expected_tensor.shape.rank; ++dim) {
      const iree_host_size_t coordinate = remaining / expected_strides[dim];
      remaining = remaining % expected_strides[dim];
      actual_index += (static_cast<iree_host_size_t>(
                           expected_tensor.slice_offsets.dims[dim]) +
                       coordinate) *
                      actual_strides[dim];
    }
    IREE_RETURN_IF_ERROR(CompareRoundedF32ToBf16Element(
        expected_tensor, expected_index, actual_index, actual));
  }
  return iree_ok_status();
}

iree_status_t CompareBf16BindingWithRoundedF32FixtureTensor(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding,
    iree_hal_semaphore_list_t wait_list,
    const id4_pipeline_tensor_layout_t* actual_layout,
    const FixtureTensor& expected_tensor) {
  if (!actual_layout) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "actual tensor layout is required");
  }
  if (actual_layout->dtype != ID4_PIPELINE_TENSOR_DTYPE_BF16) {
    iree_string_view_t dtype_name =
        id4_pipeline_tensor_dtype_format(actual_layout->dtype);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "actual tensor `%s` dtype `%.*s` is not bf16",
                            expected_tensor.name.c_str(),
                            static_cast<int>(dtype_name.size), dtype_name.data);
  }
  if (expected_tensor.dtype != ID4_PIPELINE_TENSOR_DTYPE_F32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected tensor `%s` must be f32 for rounded BF16 comparison",
        expected_tensor.name.c_str());
  }

  std::vector<uint8_t> actual_bytes;
  IREE_RETURN_IF_ERROR(ReadBindingToHost(device, queue_affinity, binding,
                                         wait_list, &actual_bytes));
  iree_host_size_t actual_element_count = 0;
  IREE_RETURN_IF_ERROR(
      ShapeElementCount(actual_layout->shape, &actual_element_count));
  if (actual_element_count >
      std::numeric_limits<iree_host_size_t>::max() / sizeof(uint16_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "actual tensor `%s` byte length overflow",
                            expected_tensor.name.c_str());
  }
  const iree_host_size_t actual_byte_count =
      actual_element_count * sizeof(uint16_t);
  if (actual_bytes.size() != actual_byte_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "actual tensor `%s` byte length %zu does not match planned byte length "
        "%" PRIhsz,
        expected_tensor.name.c_str(), actual_bytes.size(), actual_byte_count);
  }
  iree_host_size_t expected_element_count = 0;
  IREE_RETURN_IF_ERROR(
      ShapeElementCount(expected_tensor.shape, &expected_element_count));
  if (expected_element_count >
      std::numeric_limits<iree_host_size_t>::max() / sizeof(float)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "expected tensor `%s` byte length overflow",
                            expected_tensor.name.c_str());
  }
  const iree_host_size_t expected_byte_count =
      expected_element_count * sizeof(float);
  if (expected_byte_count != expected_tensor.payload.size()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected tensor `%s` byte length %zu does not match shape byte length "
        "%" PRIhsz,
        expected_tensor.name.c_str(), expected_tensor.payload.size(),
        expected_byte_count);
  }
  if (ShapeEquals(expected_tensor.shape, actual_layout->shape)) {
    return CompareRoundedF32ToBf16DensePayload(expected_tensor, actual_bytes);
  }
  if (ShapeEquals(expected_tensor.source_shape, actual_layout->shape)) {
    return CompareRoundedF32ToBf16SourceSlice(expected_tensor, actual_bytes);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "actual tensor `%s` shape matches neither expected "
                          "shape nor source shape",
                          expected_tensor.name.c_str());
}

iree_status_t CompareBindingWithFixtureTensor(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding,
    iree_hal_semaphore_list_t wait_list,
    const id4_pipeline_tensor_layout_t* actual_layout,
    const FixtureTensor& expected_tensor) {
  if (!actual_layout) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "actual tensor layout is required");
  }
  if (expected_tensor.dtype != ID4_PIPELINE_TENSOR_DTYPE_F32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected tensor `%s` must be f32 for F32 comparison",
        expected_tensor.name.c_str());
  }
  if (!FixtureToleranceIsDeclared(expected_tensor)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected tensor `%s` is missing comparison tolerance",
        expected_tensor.name.c_str());
  }
  if (!FixtureToleranceIsElementwise(expected_tensor) &&
      !FixtureToleranceIsAggregate(expected_tensor)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected tensor `%s` has unsupported comparison tolerance",
        expected_tensor.name.c_str());
  }
  std::vector<uint8_t> actual_bytes;
  IREE_RETURN_IF_ERROR(ReadBindingToHost(device, queue_affinity, binding,
                                         wait_list, &actual_bytes));
  iree_host_size_t expected_element_count = 0;
  IREE_RETURN_IF_ERROR(
      ShapeElementCount(expected_tensor.shape, &expected_element_count));
  if (expected_element_count >
      std::numeric_limits<iree_host_size_t>::max() / sizeof(float)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "expected tensor `%s` byte length overflow",
                            expected_tensor.name.c_str());
  }
  const iree_host_size_t expected_byte_count =
      expected_element_count * sizeof(float);
  if (expected_byte_count != expected_tensor.payload.size()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected tensor `%s` byte length %zu does not match shape byte length "
        "%" PRIhsz,
        expected_tensor.name.c_str(), expected_tensor.payload.size(),
        expected_byte_count);
  }
  const iree_device_size_t actual_dtype_byte_length =
      id4_pipeline_tensor_dtype_byte_length(actual_layout->dtype);
  if (actual_dtype_byte_length == 0) {
    iree_string_view_t dtype_name =
        id4_pipeline_tensor_dtype_format(actual_layout->dtype);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "actual tensor `%s` dtype `%.*s` is invalid",
                            expected_tensor.name.c_str(),
                            static_cast<int>(dtype_name.size), dtype_name.data);
  }
  iree_host_size_t actual_element_count = 0;
  IREE_RETURN_IF_ERROR(
      ShapeElementCount(actual_layout->shape, &actual_element_count));
  if (actual_element_count >
      std::numeric_limits<iree_host_size_t>::max() / actual_dtype_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "actual tensor `%s` byte length overflow",
                            expected_tensor.name.c_str());
  }
  const iree_host_size_t actual_byte_count =
      actual_element_count *
      static_cast<iree_host_size_t>(actual_dtype_byte_length);
  if (actual_bytes.size() != actual_byte_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "actual tensor `%s` byte length %zu does not match planned byte length "
        "%" PRIhsz,
        expected_tensor.name.c_str(), actual_bytes.size(), actual_byte_count);
  }
  if (ShapeEquals(expected_tensor.shape, actual_layout->shape)) {
    return CompareF32DensePayload(expected_tensor, actual_layout, actual_bytes);
  }
  if (ShapeEquals(expected_tensor.source_shape, actual_layout->shape)) {
    return CompareF32SourceSlice(expected_tensor, actual_layout, actual_bytes);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "actual tensor `%s` shape matches neither expected "
                          "shape nor source shape",
                          expected_tensor.name.c_str());
}

iree_status_t CompareF32BindingWithFixtureTensor(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding,
    iree_hal_semaphore_list_t wait_list, const FixtureTensor& expected_tensor) {
  id4_pipeline_tensor_layout_t actual_layout = {};
  actual_layout.name = iree_make_string_view(expected_tensor.name.data(),
                                             expected_tensor.name.size());
  actual_layout.dtype = ID4_PIPELINE_TENSOR_DTYPE_F32;
  actual_layout.shape = expected_tensor.source_shape;
  actual_layout.byte_length = binding->length;
  return CompareBindingWithFixtureTensor(device, queue_affinity, binding,
                                         wait_list, &actual_layout,
                                         expected_tensor);
}

iree_status_t LoadFixtureTensors(iree_string_view_t fixture_directory,
                                 FixtureTensorSet* out_fixture_tensors) {
  out_fixture_tensors->directory.assign(fixture_directory.data,
                                        fixture_directory.size);
  out_fixture_tensors->tensors.clear();

  std::vector<uint8_t> manifest_file;
  IREE_RETURN_IF_ERROR(ReadBinaryFile(
      JoinPath(fixture_directory, "manifest.json"), &manifest_file));
  iree_string_view_t manifest =
      iree_make_string_view(reinterpret_cast<const char*>(manifest_file.data()),
                            manifest_file.size());
  iree_string_view_t records = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(manifest, IREE_SV("records"), &records));
  iree_host_size_t record_count = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(records, &record_count));
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    iree_string_view_t record = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_array_get(records, i, &record));
    IREE_RETURN_IF_ERROR(LoadFixtureTensorRecord(record, out_fixture_tensors));
  }
  return iree_ok_status();
}

iree_status_t InferRank1TensorLengthFromFixture(
    const FixtureTensorSet& fixture_tensors, iree_string_view_t tensor_name,
    id4_pipeline_tensor_dtype_t dtype, uint32_t* out_length) {
  const FixtureTensor* tensor =
      fixture_tensors.FindTensor(IREE_SV("input"), tensor_name);
  if (!tensor) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "fixture is missing required `%.*s` input",
        static_cast<int>(tensor_name.size), tensor_name.data);
  }
  if (tensor->dtype != dtype || tensor->shape.rank != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture `%.*s` input must be rank-1 with the requested dtype",
        static_cast<int>(tensor_name.size), tensor_name.data);
  }
  if (tensor->shape.dims[0] >
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE, "fixture `%.*s` length exceeds uint32_t",
        static_cast<int>(tensor_name.size), tensor_name.data);
  }
  *out_length = static_cast<uint32_t>(tensor->shape.dims[0]);
  return iree_ok_status();
}

iree_status_t QueueUpdateInitializedBoundaryTensorsFromFixture(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BufferBindingSet& binding_set,
    const FixtureTensorSet& fixture_tensors,
    iree_hal_semaphore_t* fill_semaphore, uint64_t* out_fill_value) {
  IREE_ASSERT_ARGUMENT(out_fill_value);
  for (iree_host_size_t i = 0; i < binding_set.count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary ||
        !iree_all_bits_set(boundary->flags,
                           ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED)) {
      continue;
    }
    const FixtureTensor* fixture_tensor =
        fixture_tensors.FindTensor(IREE_SV("input"), boundary->layout.name);
    if (!fixture_tensor) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "fixture is missing initialized boundary tensor `%.*s`",
          static_cast<int>(boundary->layout.name.size),
          boundary->layout.name.data);
    }
    if (fixture_tensor->dtype != boundary->layout.dtype) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%.*s` dtype does not match planned boundary dtype",
          static_cast<int>(boundary->layout.name.size),
          boundary->layout.name.data);
    }
    if (!ShapeEquals(fixture_tensor->shape, boundary->layout.shape)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%.*s` shape does not match planned boundary shape",
          static_cast<int>(boundary->layout.name.size),
          boundary->layout.name.data);
    }
    if (fixture_tensor->payload.size() != boundary->layout.byte_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%.*s` payload length %zu does not match planned "
          "byte length %" PRIu64,
          static_cast<int>(boundary->layout.name.size),
          boundary->layout.name.data, fixture_tensor->payload.size(),
          static_cast<uint64_t>(boundary->layout.byte_length));
    }

    IREE_RETURN_IF_ERROR(QueueUpdateBinding(
        device, queue_affinity, &binding_set.bindings[i],
        fixture_tensor->payload.data(), fixture_tensor->payload.size(),
        fill_semaphore, out_fill_value));
  }
  return iree_ok_status();
}

static iree_status_t VerifyCapturedPayloadWasWritten(
    iree_string_view_t capture_directory, iree_string_view_t file_prefix,
    iree_host_size_t file_ordinal, iree_string_view_t tensor_name,
    uint8_t sentinel) {
  char file_name[48];
  snprintf(file_name, sizeof(file_name), "%.*s_%04" PRIhsz ".id4tensor",
           static_cast<int>(file_prefix.size), file_prefix.data, file_ordinal);
  std::string capture_path = JoinPath(capture_directory, file_name);
  if (!FileExists(capture_path)) {
    snprintf(file_name, sizeof(file_name), "%.*s_%04" PRIhsz ".npy",
             static_cast<int>(file_prefix.size), file_prefix.data,
             file_ordinal);
    capture_path = JoinPath(capture_directory, file_name);
  }
  std::vector<uint8_t> capture;
  IREE_RETURN_IF_ERROR(ReadBinaryFile(capture_path, &capture));
  iree_host_size_t payload_offset = 0;
  if (HasPrefix(capture, kId4TensorMagic, IREE_ARRAYSIZE(kId4TensorMagic))) {
    if (capture.size() < IREE_ARRAYSIZE(kId4TensorMagic) + 4) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "capture for %.*s has a truncated ID4 header",
                              (int)tensor_name.size, tensor_name.data);
    }
    const iree_host_size_t header_length =
        static_cast<iree_host_size_t>(capture[8]) |
        (static_cast<iree_host_size_t>(capture[9]) << 8) |
        (static_cast<iree_host_size_t>(capture[10]) << 16) |
        (static_cast<iree_host_size_t>(capture[11]) << 24);
    payload_offset = 12 + header_length;
  } else if (capture.size() >= 10 && capture[0] == 0x93 && capture[1] == 'N' &&
             capture[2] == 'U' && capture[3] == 'M' && capture[4] == 'P' &&
             capture[5] == 'Y') {
    const iree_host_size_t header_length =
        static_cast<iree_host_size_t>(capture[8]) |
        (static_cast<iree_host_size_t>(capture[9]) << 8);
    payload_offset = 10 + header_length;
  } else {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "capture for %.*s has unsupported tensor payload "
                            "format",
                            (int)tensor_name.size, tensor_name.data);
  }
  if (payload_offset >= capture.size()) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "capture for %.*s has no tensor payload",
                            (int)tensor_name.size, tensor_name.data);
  }
  bool payload_is_sentinel = true;
  for (iree_host_size_t i = payload_offset; i < capture.size(); ++i) {
    if (capture[i] != sentinel) {
      payload_is_sentinel = false;
      break;
    }
  }
  if (payload_is_sentinel) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "captured tensor %.*s was not written",
                            (int)tensor_name.size, tensor_name.data);
  }
  return iree_ok_status();
}

iree_status_t VerifyCapturedExportedBoundaryTensorsWereWritten(
    const id4_pipeline_plan_t* plan, iree_string_view_t capture_directory,
    uint8_t sentinel) {
  bool saw_exported_boundary = false;
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  for (iree_host_size_t i = 0; i < boundary_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary ||
        !iree_all_bits_set(boundary->flags,
                           ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED)) {
      continue;
    }
    saw_exported_boundary = true;
    IREE_RETURN_IF_ERROR(
        VerifyCapturedPayloadWasWritten(capture_directory, IREE_SV("boundary"),
                                        i, boundary->layout.name, sentinel));
  }
  if (!saw_exported_boundary) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stage plan has no exported boundary tensors");
  }
  return iree_ok_status();
}

iree_status_t VerifyCapturedDiagnosticTapTensorsWereWritten(
    const id4_pipeline_plan_t* plan, iree_string_view_t capture_directory,
    uint8_t sentinel) {
  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  for (iree_host_size_t i = 0; i < diagnostic_tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (!diagnostic_tap) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "stage plan is missing diagnostic tap %" PRIhsz,
                              i);
    }
    IREE_RETURN_IF_ERROR(VerifyCapturedPayloadWasWritten(
        capture_directory, IREE_SV("tap"), i, diagnostic_tap->name, sentinel));
  }
  return iree_ok_status();
}

}  // namespace id4::test
