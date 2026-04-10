// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <hrx_runtime_cxx.h>

#include <catch2/catch_test_macros.hpp>
#include <type_traits>

TEST_CASE("hrx_runtime_cxx.h handle wrappers compile and default to null",
          "[cxx_api]") {
  static_assert(std::is_copy_constructible_v<hrx::runtime::device_ptr>);
  static_assert(std::is_copy_assignable_v<hrx::runtime::device_ptr>);
  static_assert(std::is_move_constructible_v<hrx::runtime::device_ptr>);
  static_assert(std::is_move_assignable_v<hrx::runtime::device_ptr>);

  hrx::runtime::device_ptr device;
  hrx::runtime::allocator_ptr allocator;
  hrx::runtime::semaphore_ptr semaphore;
  hrx::runtime::stream_ptr stream;
  hrx::runtime::buffer_ptr buffer;
  hrx::runtime::module_ptr module;
  hrx::runtime::function_ptr function;
  hrx::runtime::value_list_ptr value_list;
  hrx::runtime::fence_ptr fence;
  hrx::runtime::buffer_view_ptr buffer_view;

  CHECK(!device);
  CHECK(!allocator);
  CHECK(!semaphore);
  CHECK(!stream);
  CHECK(!buffer);
  CHECK(!module);
  CHECK(!function);
  CHECK(!value_list);
  CHECK(!fence);
  CHECK(!buffer_view);

  hrx::runtime::device_ptr copied = device;
  hrx::runtime::device_ptr moved = std::move(copied);
  CHECK(!copied);
  CHECK(!moved);
  CHECK(moved.get() == nullptr);
  CHECK(moved.release() == nullptr);
  CHECK(moved.for_output() != nullptr);
}

TEST_CASE("hrx_runtime_cxx.h status formatter handles OK", "[cxx_api]") {
  CHECK(hrx::runtime::format_status(hrx_ok_status()) == "OK");
}
