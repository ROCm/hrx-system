// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/tooling/capture.h"
#include "iree/base/internal/json.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

IREE_FLAG(string, id4_capture_dir, "",
          "Directory that receives exported ID4 boundary tensor captures.");
IREE_FLAG(string, id4_fixture_dir, "",
          "Directory containing an ID4 reference fixture manifest and tensor "
          "payloads used to initialize stage inputs.");

namespace {

constexpr uint8_t kExportedBoundarySentinel = 0xA5;

struct BoundaryBindingSet {
  // Number of boundary bindings allocated from the plan.
  iree_host_size_t count = 0;
  // Owned HAL buffers backing each boundary binding.
  iree_hal_buffer_t** buffers = nullptr;
  // Binding table entries in plan boundary tensor order.
  iree_hal_buffer_binding_t* bindings = nullptr;

  ~BoundaryBindingSet() { reset(); }

  void reset() {
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
};

struct DiagnosticTapBindingSet {
  // Number of diagnostic tap bindings allocated from the plan.
  iree_host_size_t count = 0;
  // Owned HAL buffers backing each diagnostic tap binding.
  iree_hal_buffer_t** buffers = nullptr;
  // Binding table entries in plan diagnostic tap order.
  iree_hal_buffer_binding_t* bindings = nullptr;

  ~DiagnosticTapBindingSet() { reset(); }

  void reset() {
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
};

struct FixtureTensor {
  // Stable tensor name from the fixture manifest.
  std::string name;
  // Relative NPY payload path from the fixture manifest.
  std::string file;
  // Tensor dtype declared by the fixture manifest and validated against NPY.
  id4_pipeline_tensor_dtype_t dtype = ID4_PIPELINE_TENSOR_DTYPE_INVALID;
  // Tensor shape declared by the fixture manifest and validated against NPY.
  id4_pipeline_tensor_shape_t shape = {};
  // Raw dense tensor bytes parsed from the NPY payload.
  std::vector<uint8_t> payload;
};

struct FixtureInputSet {
  // Fixture directory containing manifest.json and payload files.
  std::string directory;
  // Input tensor payloads available for boundary initialization.
  std::vector<FixtureTensor> tensors;

  const FixtureTensor* FindTensor(iree_string_view_t name) const {
    for (const FixtureTensor& tensor : tensors) {
      if (iree_string_view_equal(
              iree_make_string_view(tensor.name.data(), tensor.name.size()),
              name)) {
        return &tensor;
      }
    }
    return nullptr;
  }
};

struct SemaphoreListStorage {
  // Semaphore carried by this single-entry list.
  iree_hal_semaphore_t* semaphore = nullptr;
  // Payload value paired with the semaphore.
  uint64_t payload_value = 0;

  iree_hal_semaphore_list_t list() {
    return iree_hal_semaphore_list_t{
        // One semaphore is carried by this stack-backed list.
        /*.count=*/1,
        // Stack-backed semaphore pointer array.
        /*.semaphores=*/&semaphore,
        // Stack-backed payload value array.
        /*.payload_values=*/&payload_value,
    };
  }
};

static iree_status_t CreateQwen3VlStage(
    const id4::test::LiveStageContext& context,
    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  create_options.model = *id4_qwen3_vl_program_ideogram4_model_config();
  return id4_qwen3_vl_stage_create(&create_options, iree_allocator_system(),
                                   out_stage);
}

static iree_status_t AllocateBoundaryBindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, BoundaryBindingSet* out_binding_set) {
  IREE_ASSERT_ARGUMENT(out_binding_set);
  out_binding_set->reset();
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  if (boundary_count == 0) return iree_ok_status();

  iree_status_t status = iree_allocator_malloc_array(
      iree_allocator_system(), boundary_count,
      sizeof(out_binding_set->buffers[0]),
      reinterpret_cast<void**>(&out_binding_set->buffers));
  if (iree_status_is_ok(status)) {
    std::memset(out_binding_set->buffers, 0,
                boundary_count * sizeof(out_binding_set->buffers[0]));
    status = iree_allocator_malloc_array(
        iree_allocator_system(), boundary_count,
        sizeof(out_binding_set->bindings[0]),
        reinterpret_cast<void**>(&out_binding_set->bindings));
  }
  if (iree_status_is_ok(status)) {
    std::memset(out_binding_set->bindings, 0,
                boundary_count * sizeof(out_binding_set->bindings[0]));
    out_binding_set->count = boundary_count;
  }

  for (iree_host_size_t i = 0;
       i < out_binding_set->count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "missing boundary tensor plan %" PRIhsz, i);
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
    params.min_alignment =
        boundary->layout.alignment ? boundary->layout.alignment : 1;
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, boundary->layout.byte_length,
        &out_binding_set->buffers[i]);
    if (iree_status_is_ok(status)) {
      out_binding_set->bindings[i] = iree_hal_buffer_binding_t{
          // Boundary buffer supplied in plan order.
          /*.buffer=*/out_binding_set->buffers[i],
          // Boundary buffers are allocated as exact standalone allocations.
          /*.offset=*/0,
          // Full planned tensor byte range.
          /*.length=*/boundary->layout.byte_length,
      };
    }
  }
  if (!iree_status_is_ok(status)) {
    out_binding_set->reset();
  }
  return status;
}

static iree_status_t AllocateDiagnosticTapBindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, DiagnosticTapBindingSet* out_binding_set) {
  IREE_ASSERT_ARGUMENT(out_binding_set);
  out_binding_set->reset();
  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  if (diagnostic_tap_count == 0) return iree_ok_status();

  iree_status_t status = iree_allocator_malloc_array(
      iree_allocator_system(), diagnostic_tap_count,
      sizeof(out_binding_set->buffers[0]),
      reinterpret_cast<void**>(&out_binding_set->buffers));
  if (iree_status_is_ok(status)) {
    std::memset(out_binding_set->buffers, 0,
                diagnostic_tap_count * sizeof(out_binding_set->buffers[0]));
    status = iree_allocator_malloc_array(
        iree_allocator_system(), diagnostic_tap_count,
        sizeof(out_binding_set->bindings[0]),
        reinterpret_cast<void**>(&out_binding_set->bindings));
  }
  if (iree_status_is_ok(status)) {
    std::memset(out_binding_set->bindings, 0,
                diagnostic_tap_count * sizeof(out_binding_set->bindings[0]));
    out_binding_set->count = diagnostic_tap_count;
  }

  for (iree_host_size_t i = 0;
       i < out_binding_set->count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (!diagnostic_tap) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "missing diagnostic tap plan %" PRIhsz, i);
      break;
    }
    iree_hal_buffer_params_t params;
    std::memset(&params, 0, sizeof(params));
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                   IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
    params.queue_affinity = queue_affinity;
    params.min_alignment =
        diagnostic_tap->layout.alignment ? diagnostic_tap->layout.alignment : 1;
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params,
        diagnostic_tap->layout.byte_length, &out_binding_set->buffers[i]);
    if (iree_status_is_ok(status)) {
      out_binding_set->bindings[i] = iree_hal_buffer_binding_t{
          // Diagnostic tap capture buffer supplied in plan order.
          /*.buffer=*/out_binding_set->buffers[i],
          // Diagnostic tap buffers are exact standalone allocations.
          /*.offset=*/0,
          // Full planned tensor byte range.
          /*.length=*/diagnostic_tap->layout.byte_length,
      };
    }
  }
  if (!iree_status_is_ok(status)) {
    out_binding_set->reset();
  }
  return status;
}

static iree_status_t QueueFillBoundaryTensors(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BoundaryBindingSet& binding_set,
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

    iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
    SemaphoreListStorage wait_storage;
    wait_storage.semaphore = fill_semaphore;
    wait_storage.payload_value = *out_fill_value;
    if (wait_storage.payload_value != 0) {
      wait_list = wait_storage.list();
    }
    SemaphoreListStorage signal_storage;
    signal_storage.semaphore = fill_semaphore;
    signal_storage.payload_value = wait_storage.payload_value + 1;
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
        device, queue_affinity, wait_list, signal_storage.list(),
        binding_set.bindings[i].buffer, binding_set.bindings[i].offset,
        binding_set.bindings[i].length, pattern, pattern_length,
        IREE_HAL_FILL_FLAG_NONE));
    *out_fill_value = signal_storage.payload_value;
  }
  return iree_ok_status();
}

static iree_status_t QueueFillDiagnosticTapTensors(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const DiagnosticTapBindingSet& binding_set, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_semaphore_t* fill_semaphore,
    uint64_t* out_fill_value) {
  IREE_ASSERT_ARGUMENT(out_fill_value);
  if (!pattern || pattern_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic tap fill pattern is required");
  }
  for (iree_host_size_t i = 0; i < binding_set.count; ++i) {
    iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
    SemaphoreListStorage wait_storage;
    wait_storage.semaphore = fill_semaphore;
    wait_storage.payload_value = *out_fill_value;
    if (wait_storage.payload_value != 0) {
      wait_list = wait_storage.list();
    }
    SemaphoreListStorage signal_storage;
    signal_storage.semaphore = fill_semaphore;
    signal_storage.payload_value = wait_storage.payload_value + 1;
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
        device, queue_affinity, wait_list, signal_storage.list(),
        binding_set.bindings[i].buffer, binding_set.bindings[i].offset,
        binding_set.bindings[i].length, pattern, pattern_length,
        IREE_HAL_FILL_FLAG_NONE));
    *out_fill_value = signal_storage.payload_value;
  }
  return iree_ok_status();
}

static iree_status_t ReadBinaryFile(const std::string& path,
                                    std::vector<uint8_t>* out_file_contents) {
  out_file_contents->clear();
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "capture file not found: %s",
                            path.c_str());
  }
  file.seekg(0, std::ios::end);
  const std::streamoff file_size = file.tellg();
  if (file_size < 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to determine capture file size: %s",
                            path.c_str());
  }
  file.seekg(0, std::ios::beg);
  out_file_contents->resize((size_t)file_size);
  if (file_size != 0) {
    file.read(reinterpret_cast<char*>(out_file_contents->data()), file_size);
    if (file.gcount() != file_size) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "failed to read capture file: %s", path.c_str());
    }
  }
  return iree_ok_status();
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
  if (iree_string_view_equal(dtype, IREE_SV("i32"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_I32;
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

static bool ShapeEquals(id4_pipeline_tensor_shape_t lhs,
                        id4_pipeline_tensor_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
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
  const std::string key = "'shape': (";
  size_t shape_begin = header.find(key);
  if (shape_begin == std::string::npos) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NPY header is missing shape tuple");
  }
  shape_begin += key.size();
  const size_t shape_end = header.find(')', shape_begin);
  if (shape_end == std::string::npos) {
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
  const std::string key = "'descr': '";
  size_t dtype_begin = header.find(key);
  if (dtype_begin == std::string::npos) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NPY header is missing dtype descriptor");
  }
  dtype_begin += key.size();
  const size_t dtype_end = header.find('\'', dtype_begin);
  if (dtype_end == std::string::npos) {
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
  if (iree_string_view_equal(dtype, IREE_SV("<i4")) ||
      iree_string_view_equal(dtype, IREE_SV("|i4"))) {
    *out_dtype = ID4_PIPELINE_TENSOR_DTYPE_I32;
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

static iree_status_t LoadFixtureInputRecord(iree_string_view_t record,
                                            FixtureInputSet* fixture_inputs) {
  std::string kind;
  IREE_RETURN_IF_ERROR(JsonLookupSimpleString(record, IREE_SV("kind"), &kind));
  std::string role;
  IREE_RETURN_IF_ERROR(JsonLookupSimpleString(record, IREE_SV("role"), &role));
  if (kind != "tensor" || role != "input") return iree_ok_status();

  FixtureTensor tensor;
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
  IREE_RETURN_IF_ERROR(
      LoadNpyTensorPayload(JoinPath(fixture_inputs->directory, tensor.file),
                           tensor.dtype, tensor.shape, &tensor.payload));
  fixture_inputs->tensors.push_back(std::move(tensor));
  return iree_ok_status();
}

static iree_status_t LoadFixtureInputs(iree_string_view_t fixture_directory,
                                       FixtureInputSet* out_fixture_inputs) {
  out_fixture_inputs->directory.assign(fixture_directory.data,
                                       fixture_directory.size);
  out_fixture_inputs->tensors.clear();

  std::vector<uint8_t> manifest_file;
  IREE_RETURN_IF_ERROR(
      ReadBinaryFile(JoinPath(out_fixture_inputs->directory, "manifest.json"),
                     &manifest_file));
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
    IREE_RETURN_IF_ERROR(LoadFixtureInputRecord(record, out_fixture_inputs));
  }
  return iree_ok_status();
}

static iree_status_t InferTokenCountFromFixture(
    const FixtureInputSet& fixture_inputs, uint32_t* out_token_count) {
  const FixtureTensor* token_ids =
      fixture_inputs.FindTensor(IREE_SV("token_ids"));
  if (!token_ids) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "fixture is missing required token_ids input");
  }
  if (token_ids->dtype != ID4_PIPELINE_TENSOR_DTYPE_I32 ||
      token_ids->shape.rank != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixture token_ids input must be rank-1 i32");
  }
  if (token_ids->shape.dims[0] >
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixture token_ids length exceeds uint32_t");
  }
  *out_token_count = static_cast<uint32_t>(token_ids->shape.dims[0]);
  return iree_ok_status();
}

static iree_status_t QueueUpdateInitializedBoundaryTensorsFromFixture(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BoundaryBindingSet& binding_set,
    const FixtureInputSet& fixture_inputs, iree_hal_semaphore_t* fill_semaphore,
    uint64_t* out_fill_value) {
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
        fixture_inputs.FindTensor(boundary->layout.name);
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

    iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
    SemaphoreListStorage wait_storage;
    wait_storage.semaphore = fill_semaphore;
    wait_storage.payload_value = *out_fill_value;
    if (wait_storage.payload_value != 0) {
      wait_list = wait_storage.list();
    }
    SemaphoreListStorage signal_storage;
    signal_storage.semaphore = fill_semaphore;
    signal_storage.payload_value = wait_storage.payload_value + 1;
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_update(
        device, queue_affinity, wait_list, signal_storage.list(),
        fixture_tensor->payload.data(), /*source_offset=*/0,
        binding_set.bindings[i].buffer, binding_set.bindings[i].offset,
        binding_set.bindings[i].length, IREE_HAL_UPDATE_FLAG_NONE));
    *out_fill_value = signal_storage.payload_value;
  }
  return iree_ok_status();
}

static iree_status_t VerifyCapturedPayloadWasWritten(
    iree_string_view_t capture_directory, iree_string_view_t file_prefix,
    iree_host_size_t file_ordinal, iree_string_view_t tensor_name) {
  char file_name[32];
  snprintf(file_name, sizeof(file_name), "%.*s_%04" PRIhsz ".npy",
           static_cast<int>(file_prefix.size), file_prefix.data, file_ordinal);
  std::vector<uint8_t> npy;
  IREE_RETURN_IF_ERROR(
      ReadBinaryFile(JoinPath(capture_directory, file_name), &npy));
  if (npy.size() < 10 || npy[0] != 0x93 || npy[1] != 'N' || npy[2] != 'U' ||
      npy[3] != 'M' || npy[4] != 'P' || npy[5] != 'Y') {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "capture for %.*s is not an NPY payload",
                            (int)tensor_name.size, tensor_name.data);
  }
  const iree_host_size_t header_length =
      (iree_host_size_t)npy[8] | ((iree_host_size_t)npy[9] << 8);
  const iree_host_size_t payload_offset = 10 + header_length;
  if (payload_offset >= npy.size()) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "capture for %.*s has no tensor payload",
                            (int)tensor_name.size, tensor_name.data);
  }
  bool payload_is_sentinel = true;
  for (iree_host_size_t i = payload_offset; i < npy.size(); ++i) {
    if (npy[i] != kExportedBoundarySentinel) {
      payload_is_sentinel = false;
      break;
    }
  }
  if (payload_is_sentinel) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "exported boundary tensor %.*s was not written",
                            (int)tensor_name.size, tensor_name.data);
  }
  return iree_ok_status();
}

static iree_status_t VerifyCapturedExportedBoundaryTensorsWereWritten(
    const id4_pipeline_plan_t* plan, iree_string_view_t capture_directory) {
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
    IREE_RETURN_IF_ERROR(VerifyCapturedPayloadWasWritten(
        capture_directory, IREE_SV("boundary"), i, boundary->layout.name));
  }
  if (!saw_exported_boundary) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stage plan has no exported boundary tensors");
  }
  return iree_ok_status();
}

static iree_status_t VerifyCapturedDiagnosticTapTensorsWereWritten(
    const id4_pipeline_plan_t* plan, iree_string_view_t capture_directory) {
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
        capture_directory, IREE_SV("tap"), i, diagnostic_tap->name));
  }
  return iree_ok_status();
}

TEST(Qwen3VlStageIntegration, PrepareAndIssueForwardWithDenseParameters) {
  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));

  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      parameter_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      iree_string_view_empty(), parameter_provider.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  IREE_ASSERT_OK(CreateQwen3VlStage(context, stage.out()));

  iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  FixtureInputSet fixture_inputs;
  uint32_t token_count = 64;
  if (!iree_string_view_is_empty(fixture_directory)) {
    IREE_ASSERT_OK(LoadFixtureInputs(fixture_directory, &fixture_inputs));
    IREE_ASSERT_OK(InferTokenCountFromFixture(fixture_inputs, &token_count));
  }

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage.get(), &load_options));

  id4_qwen3_vl_stage_plan_options_t qwen_options;
  std::memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = token_count;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &qwen_options;
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(
      id4_pipeline_stage_plan(stage.get(), &plan_options, plan.out()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  SemaphoreListStorage prepare_signal;
  prepare_signal.semaphore = prepare_semaphore.get();
  prepare_signal.payload_value = 1;

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = parameter_provider.get();
  prepare_options.kernel_library = kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = prepare_signal.list();
  prepare_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_ASSERT_OK(id4_pipeline_stage_prepare(stage.get(), plan.get(),
                                            &prepare_options, bundle.out()));

  BoundaryBindingSet boundary_bindings;
  IREE_ASSERT_OK(AllocateBoundaryBindings(context.device.get(),
                                          IREE_HAL_QUEUE_AFFINITY_ANY,
                                          plan.get(), &boundary_bindings));
  DiagnosticTapBindingSet diagnostic_tap_bindings;
  IREE_ASSERT_OK(AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &diagnostic_tap_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      fill_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, fill_semaphore.out()));
  uint64_t fill_value = 0;
  if (fixture_inputs.tensors.empty()) {
    const uint8_t zero_pattern = 0;
    IREE_ASSERT_OK(QueueFillBoundaryTensors(
        context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
        boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED,
        &zero_pattern, sizeof(zero_pattern), fill_semaphore.get(),
        &fill_value));
  } else {
    IREE_ASSERT_OK(QueueUpdateInitializedBoundaryTensorsFromFixture(
        context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
        boundary_bindings, fixture_inputs, fill_semaphore.get(), &fill_value));
  }
  IREE_ASSERT_OK(QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kExportedBoundarySentinel, sizeof(kExportedBoundarySentinel),
      fill_semaphore.get(), &fill_value));
  IREE_ASSERT_OK(QueueFillDiagnosticTapTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      diagnostic_tap_bindings, &kExportedBoundarySentinel,
      sizeof(kExportedBoundarySentinel), fill_semaphore.get(), &fill_value));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  iree_hal_semaphore_list_t issue_wait_list = iree_hal_semaphore_list_empty();
  SemaphoreListStorage issue_wait;
  if (fill_value != 0) {
    issue_wait.semaphore = fill_semaphore.get();
    issue_wait.payload_value = fill_value;
    issue_wait_list = issue_wait.list();
  }
  SemaphoreListStorage issue_signal;
  issue_signal.semaphore = issue_semaphore.get();
  issue_signal.payload_value = 1;

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  issue_options.wait_semaphore_list = issue_wait_list;
  issue_options.signal_semaphore_list = issue_signal.list();
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(
      id4_pipeline_stage_issue(stage.get(), bundle.get(), &issue_options));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      issue_semaphore.get(), issue_signal.payload_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree::testing::TempFilePath temp_capture_directory;
  iree_string_view_t capture_directory =
      iree_make_cstring_view(FLAG_id4_capture_dir);
  if (iree_string_view_is_empty(capture_directory)) {
    temp_capture_directory =
        iree::testing::TempFilePath("id4_qwen3_vl_capture");
    capture_directory = temp_capture_directory.path_view();
  }

  SemaphoreListStorage capture_wait;
  capture_wait.semaphore = issue_semaphore.get();
  capture_wait.payload_value = issue_signal.payload_value;
  id4_tooling_capture_execution_options_t capture_options;
  std::memset(&capture_options, 0, sizeof(capture_options));
  capture_options.structure_size = sizeof(capture_options);
  capture_options.run_id = IREE_SV("qwen3_vl_forward_integration");
  capture_options.output_directory = capture_directory;
  capture_options.plan = plan.get();
  capture_options.device = context.device.get();
  capture_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  capture_options.boundary_binding_count = boundary_bindings.count;
  capture_options.boundary_bindings = boundary_bindings.bindings;
  capture_options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  capture_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  capture_options.wait_semaphore_list = capture_wait.list();
  capture_options.host_allocator = iree_allocator_system();
  IREE_ASSERT_OK(id4_tooling_capture_execution(&capture_options));
  IREE_ASSERT_OK(VerifyCapturedExportedBoundaryTensorsWereWritten(
      plan.get(), capture_directory));
  IREE_ASSERT_OK(VerifyCapturedDiagnosticTapTensorsWereWritten(
      plan.get(), capture_directory));
}

}  // namespace
