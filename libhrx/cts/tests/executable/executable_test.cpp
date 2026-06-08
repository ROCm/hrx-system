// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <string>

#include "build_tools/amdgpu/target_map.h"
#include "hrx_test_fixture.hpp"
#include "libhrx/cts/amdgpu_executable_test_kernels.h"

namespace {

std::string gpu_architecture(hrx_device_t device) {
  std::array<char, 64> arch = {};
  REQUIRE_OK(hrx().device_get_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                       arch.data(), arch.size()));
  std::string value(arch.data());
  for (char c : value) {
    REQUIRE((std::isalnum(static_cast<unsigned char>(c)) || c == '_'));
  }
  return value;
}

const iree_file_toc_t* find_hsaco_for_target(const std::string& target) {
  char fragment[64] = {};
  if (!iree_amdgpu_target_label_fragment(target.c_str(), fragment,
                                         sizeof(fragment))) {
    return nullptr;
  }
  const std::string filename =
      std::string("hrx_cts_executable_kernel_") + fragment + ".so";
  const iree_file_toc_t* toc = hrx_cts_amdgpu_executable_test_kernels_create();
  for (size_t i = 0; i < hrx_cts_amdgpu_executable_test_kernels_size(); ++i) {
    if (filename == toc[i].name) {
      return &toc[i];
    }
  }
  return nullptr;
}

const iree_file_toc_t* find_hsaco(const std::string& arch) {
  if (const iree_file_toc_t* file = find_hsaco_for_target(arch)) {
    return file;
  }
  const char* code_object_target =
      iree_amdgpu_code_object_target_for_exact(arch.c_str());
  if (code_object_target && arch != code_object_target) {
    return find_hsaco_for_target(code_object_target);
  }
  return nullptr;
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

  const iree_file_toc_t* hsaco = find_hsaco(arch);
  if (!hsaco) {
    SUCCEED("Skipping native executable CTS: no build-time HSACO test asset");
    return;
  }

  hrx_executable_t executable = nullptr;
  REQUIRE_OK(hrx().executable_load_data(device_, hsaco->data, hsaco->size,
                                        nullptr, &executable));
  REQUIRE(executable != nullptr);

  hrx().executable_retain(executable);
  hrx().executable_release(executable);

  size_t export_count = 0;
  REQUIRE_OK(hrx().executable_export_count(executable, &export_count));
  REQUIRE(export_count == 2);

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
  REQUIRE(noop_info.constant_count == 0);
  REQUIRE(noop_info.binding_count == 0);
  REQUIRE(noop_info.workgroup_size[0] >= 1);
  REQUIRE(noop_info.workgroup_size[1] >= 1);
  REQUIRE(noop_info.workgroup_size[2] >= 1);

  hrx_executable_export_info_t store_info = {};
  REQUIRE_OK(
      hrx().executable_export_info(executable, store_ordinal, &store_info));
  REQUIRE(store_info.name != nullptr);
  REQUIRE(std::string(store_info.name) == "hrx_store_output");
  REQUIRE(store_info.constant_count == 1);
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
