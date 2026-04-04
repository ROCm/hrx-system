// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<char> loadVmfb(const char* relative_path) {
  std::string path = std::string(PYRE_CTS_SOURCE_DIR) + "/" + relative_path;
  std::ifstream file(path, std::ios::binary);
  REQUIRE(file.is_open());
  return std::vector<char>(
      std::istreambuf_iterator<char>(file),
      std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE_METHOD(PyreTestFixture, "Value list i64 round-trip",
                 "[vm][value_list]") {
  pyre_value_list_t list = nullptr;
  REQUIRE_OK(pyre().value_list_create(4, &list));

  REQUIRE_OK(pyre().value_list_push_i64(list, 17));
  REQUIRE_OK(pyre().value_list_push_i64(list, 25));

  size_t size = 0;
  REQUIRE_OK(pyre().value_list_size(list, &size));
  REQUIRE(size == 2);

  int64_t value = 0;
  REQUIRE_OK(pyre().value_list_get_i64(list, 0, &value));
  REQUIRE(value == 17);
  REQUIRE_OK(pyre().value_list_get_i64(list, 1, &value));
  REQUIRE(value == 25);

  pyre().value_list_release(list);
}

TEST_CASE_METHOD(PyreTestFixture, "Fence create/extend/signal",
                 "[vm][fence]") {
  pyre_semaphore_t sem0 = nullptr;
  pyre_semaphore_t sem1 = nullptr;
  REQUIRE_OK(pyre().semaphore_create(device_, 0, &sem0));
  REQUIRE_OK(pyre().semaphore_create(device_, 0, &sem1));

  pyre_fence_t wait = nullptr;
  pyre_fence_t signal = nullptr;
  REQUIRE_OK(pyre().fence_create(2, &wait));
  REQUIRE_OK(pyre().fence_insert(wait, sem0, 1));
  REQUIRE_OK(pyre().fence_create_at(sem1, 3, &signal));
  REQUIRE_OK(pyre().fence_extend(wait, signal));

  REQUIRE_OK(pyre().semaphore_signal(sem0, 1));
  REQUIRE_OK(pyre().fence_signal(signal));
  REQUIRE_OK(pyre().fence_wait(wait, UINT64_MAX));

  pyre().fence_release(signal);
  pyre().fence_release(wait);
  pyre().semaphore_release(sem1);
  pyre().semaphore_release(sem0);
}

TEST_CASE_METHOD(PyreTestFixture, "Buffer view metadata",
                 "[vm][buffer_view]") {
  pyre_stream_t stream = nullptr;
  REQUIRE_OK(pyre().stream_create(device_, 0, &stream));

  pyre_buffer_t buffer = nullptr;
  REQUIRE_OK(pyre().buffer_allocate(
      stream, 6 * sizeof(int32_t),
      PYRE_MEMORY_TYPE_HOST_LOCAL | PYRE_MEMORY_TYPE_DEVICE_VISIBLE,
      PYRE_BUFFER_USAGE_DEFAULT | PYRE_BUFFER_USAGE_MAPPING_SCOPED,
      &buffer));

  int64_t shape[] = {2, 3};
  pyre_buffer_view_t view = nullptr;
  REQUIRE_OK(pyre().buffer_view_create(
      buffer, 2, shape, PYRE_ELEMENT_TYPE_SINT_32,
      PYRE_ENCODING_TYPE_DENSE_ROW_MAJOR, &view));

  size_t rank = 0;
  REQUIRE_OK(pyre().buffer_view_rank(view, &rank));
  REQUIRE(rank == 2);

  int64_t dim = 0;
  REQUIRE_OK(pyre().buffer_view_dim(view, 0, &dim));
  REQUIRE(dim == 2);
  REQUIRE_OK(pyre().buffer_view_dim(view, 1, &dim));
  REQUIRE(dim == 3);

  pyre_value_list_t list = nullptr;
  REQUIRE_OK(pyre().value_list_create(2, &list));
  REQUIRE_OK(pyre().value_list_push_buffer_view(list, view));
  REQUIRE_OK(pyre().value_list_push_buffer(list, buffer));
  REQUIRE_OK(pyre().value_list_push_null_ref(list));
  pyre().value_list_release(list);

  pyre().buffer_view_release(view);
  pyre().buffer_release(buffer);
  pyre().stream_release(stream);
}

TEST_CASE_METHOD(PyreTestFixture, "Load VMFB and invoke function",
                 "[vm][invoke]") {
  std::vector<char> vmfb = loadVmfb("testdata/add_i64.vmfb");
  REQUIRE(!vmfb.empty());

  pyre_module_t module = nullptr;
  REQUIRE_OK(pyre().module_load_vmfb(
      device_, vmfb.data(), vmfb.size(), &module));

  pyre_function_t function = nullptr;
  REQUIRE_OK(pyre().module_lookup_function(module, "module.add_i64",
                                           &function));

  pyre_value_list_t args = nullptr;
  pyre_value_list_t rets = nullptr;
  REQUIRE_OK(pyre().value_list_create(2, &args));
  REQUIRE_OK(pyre().value_list_create(1, &rets));
  REQUIRE_OK(pyre().value_list_push_i64(args, 19));
  REQUIRE_OK(pyre().value_list_push_i64(args, 23));

  REQUIRE_OK(pyre().function_invoke(module, function, args, rets));

  int64_t result = 0;
  REQUIRE_OK(pyre().value_list_get_i64(rets, 0, &result));
  REQUIRE(result == 42);

  pyre().value_list_release(rets);
  pyre().value_list_release(args);
  pyre().function_release(function);
  pyre().module_release(module);
}
