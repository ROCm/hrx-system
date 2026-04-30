// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<char> loadVmfb(const char *relative_path) {
  const char *source_dir = std::getenv("HRX_CTS_SOURCE_DIR");
  std::string path =
      std::string((source_dir && source_dir[0]) ? source_dir
                                                : HRX_CTS_SOURCE_DIR) +
      "/" + relative_path;
  std::ifstream file(path, std::ios::binary);
  REQUIRE(file.is_open());
  return std::vector<char>(std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE_METHOD(HrxTestFixture, "Value list i64 round-trip",
                 "[vm][value_list]") {
  hrx_value_list_t list = nullptr;
  REQUIRE_OK(hrx().value_list_create(4, &list));

  REQUIRE_OK(hrx().value_list_push_i64(list, 17));
  REQUIRE_OK(hrx().value_list_push_i64(list, 25));

  size_t size = 0;
  REQUIRE_OK(hrx().value_list_size(list, &size));
  REQUIRE(size == 2);

  int64_t value = 0;
  REQUIRE_OK(hrx().value_list_get_i64(list, 0, &value));
  REQUIRE(value == 17);
  REQUIRE_OK(hrx().value_list_get_i64(list, 1, &value));
  REQUIRE(value == 25);

  hrx().value_list_release(list);
}

TEST_CASE_METHOD(HrxTestFixture, "Fence create/extend/signal", "[vm][fence]") {
  hrx_semaphore_t sem0 = nullptr;
  hrx_semaphore_t sem1 = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &sem0));
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &sem1));

  hrx_fence_t wait = nullptr;
  hrx_fence_t signal = nullptr;
  REQUIRE_OK(hrx().fence_create(2, &wait));
  REQUIRE_OK(hrx().fence_insert(wait, sem0, 1));
  REQUIRE_OK(hrx().fence_create_at(sem1, 3, &signal));
  REQUIRE_OK(hrx().fence_extend(wait, signal));

  REQUIRE_OK(hrx().semaphore_signal(sem0, 1));
  REQUIRE_OK(hrx().fence_signal(signal));
  REQUIRE_OK(hrx().fence_wait(wait, UINT64_MAX));

  hrx().fence_release(signal);
  hrx().fence_release(wait);
  hrx().semaphore_release(sem1);
  hrx().semaphore_release(sem0);
}

TEST_CASE_METHOD(HrxTestFixture, "Buffer view metadata", "[vm][buffer_view]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  hrx_buffer_t buffer = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(
      stream, 6 * sizeof(int32_t),
      HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buffer));

  int64_t shape[] = {2, 3};
  hrx_buffer_view_t view = nullptr;
  REQUIRE_OK(
      hrx().buffer_view_create(buffer, 2, shape, HRX_ELEMENT_TYPE_SINT_32,
                               HRX_ENCODING_TYPE_DENSE_ROW_MAJOR, &view));

  size_t rank = 0;
  REQUIRE_OK(hrx().buffer_view_rank(view, &rank));
  REQUIRE(rank == 2);

  int64_t dim = 0;
  REQUIRE_OK(hrx().buffer_view_dim(view, 0, &dim));
  REQUIRE(dim == 2);
  REQUIRE_OK(hrx().buffer_view_dim(view, 1, &dim));
  REQUIRE(dim == 3);

  hrx_value_list_t list = nullptr;
  REQUIRE_OK(hrx().value_list_create(2, &list));
  REQUIRE_OK(hrx().value_list_push_buffer_view(list, view));
  REQUIRE_OK(hrx().value_list_push_buffer(list, buffer));
  REQUIRE_OK(hrx().value_list_push_null_ref(list));
  hrx().value_list_release(list);

  hrx().buffer_view_release(view);
  hrx().buffer_release(buffer);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Load VMFB and invoke function",
                 "[vm][invoke]") {
  std::vector<char> vmfb = loadVmfb("testdata/add_i64.vmfb");
  REQUIRE(!vmfb.empty());

  hrx_module_t module = nullptr;
  REQUIRE_OK(
      hrx().module_load_vmfb(device_, vmfb.data(), vmfb.size(), &module));

  hrx_function_t function = nullptr;
  REQUIRE_OK(hrx().module_lookup_function(module, "module.add_i64", &function));

  hrx_value_list_t args = nullptr;
  hrx_value_list_t rets = nullptr;
  REQUIRE_OK(hrx().value_list_create(2, &args));
  REQUIRE_OK(hrx().value_list_create(1, &rets));
  REQUIRE_OK(hrx().value_list_push_i64(args, 19));
  REQUIRE_OK(hrx().value_list_push_i64(args, 23));

  REQUIRE_OK(hrx().function_invoke(module, function, args, rets));

  int64_t result = 0;
  REQUIRE_OK(hrx().value_list_get_i64(rets, 0, &result));
  REQUIRE(result == 42);

  hrx().value_list_release(rets);
  hrx().value_list_release(args);
  hrx().function_release(function);
  hrx().module_release(module);
}
