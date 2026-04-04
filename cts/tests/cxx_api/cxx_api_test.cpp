// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include <pyre_runtime_cxx.h>

#include <catch2/catch_test_macros.hpp>
#include <type_traits>

TEST_CASE("pyre_runtime_cxx.h handle wrappers compile and default to null",
          "[cxx_api]") {
  static_assert(std::is_copy_constructible_v<pyre::runtime::device_ptr>);
  static_assert(std::is_copy_assignable_v<pyre::runtime::device_ptr>);
  static_assert(std::is_move_constructible_v<pyre::runtime::device_ptr>);
  static_assert(std::is_move_assignable_v<pyre::runtime::device_ptr>);

  pyre::runtime::device_ptr device;
  pyre::runtime::allocator_ptr allocator;
  pyre::runtime::semaphore_ptr semaphore;
  pyre::runtime::stream_ptr stream;
  pyre::runtime::buffer_ptr buffer;
  pyre::runtime::module_ptr module;
  pyre::runtime::function_ptr function;
  pyre::runtime::value_list_ptr value_list;
  pyre::runtime::fence_ptr fence;
  pyre::runtime::buffer_view_ptr buffer_view;

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

  pyre::runtime::device_ptr copied = device;
  pyre::runtime::device_ptr moved = std::move(copied);
  CHECK(!copied);
  CHECK(!moved);
  CHECK(moved.get() == nullptr);
  CHECK(moved.release() == nullptr);
  CHECK(moved.for_output() != nullptr);
}

TEST_CASE("pyre_runtime_cxx.h status formatter handles OK", "[cxx_api]") {
  CHECK(pyre::runtime::format_status(pyre_ok_status()) == "OK");
}
