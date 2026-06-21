// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/hal_integration_util.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

#include "experimental/id4/kernels/embedded_loom_sources.h"
#include "iree/base/internal/json.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/scope_map.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/parameter_util.h"

namespace id4::test {

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

static iree_status_t ParseFixtureTolerance(iree_string_view_t record,
                                           FixtureTensor* tensor) {
  iree_string_view_t tolerance = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      record, IREE_SV("tolerance"), &tolerance));
  if (iree_string_view_is_empty(tolerance)) return iree_ok_status();

  iree_string_view_t absolute_tolerance = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(tolerance, IREE_SV("atol"),
                                                     &absolute_tolerance));
  double parsed_absolute_tolerance = 0.0;
  IREE_RETURN_IF_ERROR(
      iree_json_parse_double(absolute_tolerance, &parsed_absolute_tolerance));
  if (parsed_absolute_tolerance < 0.0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixture tolerance atol must be non-negative");
  }

  iree_string_view_t relative_tolerance = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(tolerance, IREE_SV("rtol"),
                                                     &relative_tolerance));
  double parsed_relative_tolerance = 0.0;
  IREE_RETURN_IF_ERROR(
      iree_json_parse_double(relative_tolerance, &parsed_relative_tolerance));
  if (parsed_relative_tolerance < 0.0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixture tolerance rtol must be non-negative");
  }

  tensor->absolute_tolerance = parsed_absolute_tolerance;
  tensor->relative_tolerance = parsed_relative_tolerance;
  tensor->has_tolerance = true;
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

static iree_status_t LoadFixtureTensorRecord(
    iree_string_view_t record, FixtureTensorSet* fixture_tensors) {
  std::string kind;
  IREE_RETURN_IF_ERROR(JsonLookupSimpleString(record, IREE_SV("kind"), &kind));
  if (kind != "tensor") return iree_ok_status();

  FixtureTensor tensor;
  IREE_RETURN_IF_ERROR(
      JsonLookupSimpleString(record, IREE_SV("role"), &tensor.role));
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
  IREE_RETURN_IF_ERROR(ParseFixtureTolerance(record, &tensor));
  IREE_RETURN_IF_ERROR(
      LoadNpyTensorPayload(JoinPath(fixture_tensors->directory, tensor.file),
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

iree_status_t CreateLiveStageContextFromFlags(LiveStageContext* out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_RETURN_IF_ERROR(RequireSingleDeviceFlag());

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
  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_cache_create(
        &kernel_cache_options, iree_allocator_system(), &kernel_cache);
  }
  if (iree_status_is_ok(status)) {
    out_context->kernel_cache.reset(kernel_cache);
  }
  return status;
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
      /*source_offset=*/0, binding->buffer, binding->offset, binding->length,
      IREE_HAL_UPDATE_FLAG_NONE));
  *inout_payload_value = signal_storage.payload_value;
  return iree_ok_status();
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

iree_status_t CompareF32BindingWithFixtureTensor(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding,
    iree_hal_semaphore_list_t wait_list, const FixtureTensor& expected_tensor) {
  if (expected_tensor.dtype != ID4_PIPELINE_TENSOR_DTYPE_F32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected tensor `%s` must be f32 for F32 comparison",
        expected_tensor.name.c_str());
  }
  if (!expected_tensor.has_tolerance) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected tensor `%s` is missing comparison tolerance",
        expected_tensor.name.c_str());
  }
  std::vector<uint8_t> actual_bytes;
  IREE_RETURN_IF_ERROR(ReadBindingToHost(device, queue_affinity, binding,
                                         wait_list, &actual_bytes));
  if (actual_bytes.size() != expected_tensor.payload.size()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected tensor `%s` byte length %zu does not match actual length %zu",
        expected_tensor.name.c_str(), expected_tensor.payload.size(),
        actual_bytes.size());
  }
  if ((actual_bytes.size() % sizeof(float)) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected tensor `%s` byte length is not f32 element-aligned",
        expected_tensor.name.c_str());
  }
  const iree_host_size_t element_count = actual_bytes.size() / sizeof(float);
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    const float actual = LoadF32(&actual_bytes[i * sizeof(float)]);
    const float expected = LoadF32(&expected_tensor.payload[i * sizeof(float)]);
    const double tolerance =
        expected_tensor.absolute_tolerance +
        expected_tensor.relative_tolerance * std::fabs((double)expected);
    const double absolute_error = std::fabs((double)actual - (double)expected);
    if (absolute_error > tolerance) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "tensor `%s` mismatch at element %" PRIhsz
          ": actual=%g expected=%g abs_error=%g tolerance=%g",
          expected_tensor.name.c_str(), i, (double)actual, (double)expected,
          absolute_error, tolerance);
    }
  }
  return iree_ok_status();
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
    if (npy[i] != sentinel) {
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
