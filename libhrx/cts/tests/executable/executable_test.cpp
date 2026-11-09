// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iterator>
#include <string>

#include "hrx_test_fixture.hpp"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

// Kernels defined by libhrx/cts/tests/executable/executable_kernels.c, the
// source compiled into the HSACO this test loads. Nothing else in that source
// becomes an export, so these names are the loaded executable's whole export
// table.
constexpr const char* kFixtureKernelNames[] = {
    "hrx_noop",
    "hrx_store_output",
    "hrx_transform_padded_parameters",
    "hrx_store_exact_workitem_indices",
    "hrx_report_dispatch_size",
    "hrx_gated_store_output",
    "hrx_transform_nested_pointers",
};

std::string gpu_architecture(hrx_device_t device) {
  std::array<char, 64> arch = {};
  REQUIRE_OK(hrx().device_get_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                       arch.data(), arch.size()));
  return std::string(arch.data());
}

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "executable_load_lookup_dispatch") {
  if (!is_gpu()) {
    SUCCEED("Native executable dispatch is GPU-only");
    return;
  }

  std::string arch = gpu_architecture(device_);
  if (arch.empty() || arch == "unknown") {
    SUCCEED("Skipping native executable CTS: unknown GPU architecture");
    return;
  }

  hrx_cts::AmdgpuExecutableTestImage hsaco =
      hrx_cts::FindAmdgpuExecutableTestImage(arch);
  if (!hsaco.file) {
    SUCCEED("Skipping native executable CTS: no build-time HSACO test asset");
    return;
  }

  hrx_executable_t executable = nullptr;
  REQUIRE_OK(hrx().executable_load_data(device_, hsaco.file->data,
                                        hsaco.file->size, "amdgpu",
                                        hsaco.target_key.c_str(), &executable));
  REQUIRE(executable != nullptr);

  hrx().executable_retain(executable);
  hrx().executable_release(executable);

  size_t export_count = 0;
  REQUIRE_OK(hrx().executable_export_count(executable, &export_count));
  REQUIRE(export_count == std::size(kFixtureKernelNames));

  // A count says nothing about which kernels it counted. Resolving every
  // fixture name to an ordinal of its own pairs with it to say the export table
  // is exactly this set.
  std::array<bool, std::size(kFixtureKernelNames)> ordinal_seen = {};
  for (const char* kernel_name : kFixtureKernelNames) {
    INFO("kernel " << kernel_name);
    uint32_t ordinal = UINT32_MAX;
    REQUIRE_OK(hrx().executable_lookup_export_by_name(executable, kernel_name,
                                                      &ordinal));
    REQUIRE(ordinal < export_count);
    REQUIRE(!ordinal_seen[ordinal]);
    ordinal_seen[ordinal] = true;
  }

  uint32_t noop_ordinal = UINT32_MAX;
  REQUIRE_OK(hrx().executable_lookup_export_by_name(executable, "hrx_noop",
                                                    &noop_ordinal));
  REQUIRE(noop_ordinal < export_count);

  uint32_t store_ordinal = UINT32_MAX;
  REQUIRE_OK(hrx().executable_lookup_export_by_name(
      executable, "hrx_store_output", &store_ordinal));
  REQUIRE(store_ordinal < export_count);
  REQUIRE(store_ordinal != noop_ordinal);

  hrx_executable_export_info_t noop_info = {};
  REQUIRE_OK(
      hrx().executable_export_info(executable, noop_ordinal, &noop_info));
  REQUIRE(noop_info.name != nullptr);
  REQUIRE(std::string(noop_info.name) == "hrx_noop");
  REQUIRE(noop_info.constant_byte_length == 0);
  REQUIRE(noop_info.binding_count == 0);
  REQUIRE(noop_info.workgroup_size[0] >= 1);
  REQUIRE(noop_info.workgroup_size[1] >= 1);
  REQUIRE(noop_info.workgroup_size[2] >= 1);

  hrx_executable_export_info_t store_info = {};
  REQUIRE_OK(
      hrx().executable_export_info(executable, store_ordinal, &store_info));
  REQUIRE(store_info.name != nullptr);
  REQUIRE(std::string(store_info.name) == "hrx_store_output");
  REQUIRE(store_info.constant_byte_length == sizeof(uint32_t));
  REQUIRE(store_info.binding_count == 1);
  REQUIRE(store_info.workgroup_size[0] >= 1);
  REQUIRE(store_info.workgroup_size[1] >= 1);
  REQUIRE(store_info.workgroup_size[2] >= 1);

  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));
  REQUIRE(stream != nullptr);

  hrx_dispatch_config_t noop_config = {
      /* .workgroup_count = */ {1, 1, 1},
      /* .workgroup_size = */
      {
          noop_info.workgroup_size[0],
          noop_info.workgroup_size[1],
          noop_info.workgroup_size[2],
      },
      /* .subgroup_size = */ 0,
  };
  REQUIRE_OK(hrx().stream_dispatch(stream, executable, noop_ordinal,
                                   &noop_config,
                                   /*constants=*/nullptr, /*constants_size=*/0,
                                   /*bindings=*/nullptr, /*binding_count=*/0,
                                   HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS));

  hrx_buffer_t output = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(stream, sizeof(uint32_t),
                                   HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                   HRX_BUFFER_USAGE_DEFAULT, &output));
  REQUIRE(output != nullptr);

  const uint32_t expected = 0xFEED1234u;
  const hrx_buffer_ref_t bindings[] = {{
      /* .buffer = */ output,
      /* .offset = */ 0,
      /* .length = */ sizeof(uint32_t),
  }};
  hrx_dispatch_config_t store_config = {
      /* .workgroup_count = */ {1, 1, 1},
      /* .workgroup_size = */
      {
          store_info.workgroup_size[0],
          store_info.workgroup_size[1],
          store_info.workgroup_size[2],
      },
      /* .subgroup_size = */ 0,
  };
  REQUIRE_OK(hrx().stream_dispatch(
      stream, executable, store_ordinal, &store_config, &expected,
      sizeof(expected), bindings, std::size(bindings), HRX_DISPATCH_FLAG_NONE));
  REQUIRE_OK(hrx().stream_synchronize(stream));

  uint32_t actual = 0;
  REQUIRE_OK(
      hrx().synchronous_d2h(device_, output, 0, &actual, sizeof(actual)));
  REQUIRE(actual == expected);

  hrx().buffer_release(output);
  hrx().stream_release(stream);
  hrx().executable_release(executable);
}

TEST_CASE_METHOD(HrxTestFixture,
                 "graph_kernel_publishes_coherent_host_output") {
  if (!is_gpu()) {
    SKIP("Native executable dispatch is GPU-only");
  }
  const auto hsaco =
      hrx_cts::FindAmdgpuExecutableTestImage(gpu_architecture(device_));
  if (!hsaco.file) {
    SKIP("No build-time HSACO test asset for this GPU");
  }

  hrx_executable_t executable = nullptr;
  REQUIRE_OK(hrx().executable_load_data(device_, hsaco.file->data,
                                        hsaco.file->size, "amdgpu",
                                        hsaco.target_key.c_str(), &executable));
  uint32_t ordinal = 0;
  REQUIRE_OK(hrx().executable_lookup_export_by_name(
      executable, "hrx_store_output", &ordinal));
  hrx_executable_export_info_t info = {};
  REQUIRE_OK(hrx().executable_export_info(executable, ordinal, &info));
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));

  constexpr size_t kWordCount = 256;
  const hrx_buffer_params_t params = {
      HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT |
          HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_MEMORY_ACCESS_ALL,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED |
          HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
      0,
  };
  hrx_buffer_t output = nullptr;
  REQUIRE_OK(
      hrx().allocator_allocate_buffer(hrx().device_allocator(device_), params,
                                      kWordCount * sizeof(uint32_t), &output));
  void* mapped = nullptr;
  REQUIRE_OK(hrx().buffer_map(output, HRX_MAP_READ | HRX_MAP_WRITE, 0,
                              kWordCount * sizeof(uint32_t), &mapped));
  auto* words = static_cast<uint32_t*>(mapped);

  for (size_t word_offset : {size_t{0}, size_t{64}}) {
    CAPTURE(word_offset);
    const uint32_t expected = 0xFEED1234u;
    const hrx_buffer_ref_t binding = {output, word_offset * sizeof(uint32_t),
                                      sizeof(uint32_t)};
    const hrx_graph_kernel_node_attrs_t attrs = {
        executable,
        ordinal,
        {{1, 1, 1},
         {info.workgroup_size[0], info.workgroup_size[1],
          info.workgroup_size[2]},
         0},
        &expected,
        sizeof(expected),
        &binding,
        1,
        HRX_DISPATCH_FLAG_NONE,
    };
    hrx_graph_t graph = nullptr;
    REQUIRE_OK(hrx().graph_create(device_, 0, &graph));
    hrx_graph_node_t node = nullptr;
    REQUIRE_OK(hrx().graph_add_kernel_node(graph, nullptr, 0, &attrs, &node));
    hrx_graph_exec_t exec = nullptr;
    REQUIRE_OK(hrx().graph_instantiate(graph, 0, &exec));

    for (uint32_t replay = 0; replay < 16; ++replay) {
      CAPTURE(replay);
      const uint32_t sentinel = 0x12345678u + replay;
      std::fill_n(words, kWordCount, sentinel);
      REQUIRE_OK(hrx().graph_exec_launch(exec, stream));
      REQUIRE_OK(hrx().stream_synchronize(stream));
      // Read before any transfer or graph teardown can publish cached writes.
      for (size_t i = 0; i < kWordCount; ++i) {
        CAPTURE(i);
        REQUIRE(words[i] == (i == word_offset ? expected : sentinel));
      }
    }
    hrx().graph_exec_release(exec);
    hrx().graph_release(graph);
  }

  REQUIRE_OK(hrx().buffer_unmap(output));
  hrx().buffer_release(output);
  hrx().stream_release(stream);
  hrx().executable_release(executable);
}
