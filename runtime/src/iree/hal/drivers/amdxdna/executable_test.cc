// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/executable.h"

#include <cstdint>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/flatcc/building.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"
#include "iree/schemas/pdi_executable_def_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

static std::vector<uint8_t> ToVector(iree_hal_amdxdna_u8_list_t value) {
  return std::vector<uint8_t>(value.data, value.data + value.count);
}

static std::vector<uint32_t> ToVector(iree_hal_amdxdna_u32_list_t value) {
  return std::vector<uint32_t>(value.data, value.data + value.count);
}

struct TestRunDef {
  std::vector<uint32_t> control_code;
  std::vector<uint32_t> data_payload;
  std::vector<uint32_t> patch_table;
};

struct TestEntryPointDef {
  const char* name = nullptr;
  int32_t pdi_index = -1;
  std::vector<TestRunDef> runs;
};

static flatbuffers_uint32_vec_ref_t CreateUInt32Vec(
    flatbuffers_builder_t* builder, const std::vector<uint32_t>& values) {
  return flatbuffers_uint32_vec_create(
      builder, values.empty() ? nullptr : values.data(), values.size());
}

static iree_status_t MakeExecutable(
    const std::vector<std::vector<uint8_t>>& pdis,
    const std::vector<TestEntryPointDef>& entry_points,
    std::vector<uint8_t>* out_executable_data) {
  IREE_ASSERT_ARGUMENT(out_executable_data);
  out_executable_data->clear();

  flatbuffers_builder_t builder;
  if (IREE_UNLIKELY(flatcc_builder_init(&builder) != 0)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to initialize flatbuffer builder");
  }

  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(flatbuffers_failed(
          iree_hal_amdxdna_ExecutableDef_start_as_root(&builder)))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to start PDIX executable flatbuffer");
  }

  std::vector<iree_hal_amdxdna_PdiDef_ref_t> pdi_refs;
  if (iree_status_is_ok(status)) {
    pdi_refs.reserve(pdis.size());
    for (const auto& pdi : pdis) {
      flatbuffers_uint8_vec_ref_t pdi_data_ref = flatbuffers_uint8_vec_create(
          &builder, pdi.empty() ? nullptr : pdi.data(), pdi.size());
      iree_hal_amdxdna_PdiDef_ref_t pdi_ref =
          iree_hal_amdxdna_PdiDef_create(&builder, pdi_data_ref);
      if (!pdi_data_ref || !pdi_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create PDI definition");
        break;
      }
      pdi_refs.push_back(pdi_ref);
    }
  }

  std::vector<iree_hal_amdxdna_EntryPointDef_ref_t> entry_point_refs;
  if (iree_status_is_ok(status)) {
    entry_point_refs.reserve(entry_points.size());
    for (const auto& entry_point : entry_points) {
      std::vector<iree_hal_amdxdna_RunDef_ref_t> run_refs;
      run_refs.reserve(entry_point.runs.size());
      for (const auto& run : entry_point.runs) {
        flatbuffers_uint32_vec_ref_t control_code_ref =
            CreateUInt32Vec(&builder, run.control_code);
        flatbuffers_uint32_vec_ref_t data_payload_ref =
            CreateUInt32Vec(&builder, run.data_payload);
        flatbuffers_uint32_vec_ref_t patch_table_ref =
            CreateUInt32Vec(&builder, run.patch_table);
        iree_hal_amdxdna_RunDef_ref_t run_ref = iree_hal_amdxdna_RunDef_create(
            &builder, control_code_ref, data_payload_ref, patch_table_ref);
        if (!control_code_ref || !data_payload_ref || !patch_table_ref ||
            !run_ref) {
          status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "failed to create run definition");
          break;
        }
        run_refs.push_back(run_ref);
      }
      if (!iree_status_is_ok(status)) break;

      flatbuffers_string_ref_t name_ref =
          flatbuffers_string_create_str(&builder, entry_point.name);
      iree_hal_amdxdna_RunDef_vec_ref_t runs_ref =
          iree_hal_amdxdna_RunDef_vec_create(&builder, run_refs.data(),
                                             run_refs.size());
      iree_hal_amdxdna_EntryPointDef_ref_t entry_point_ref = 0;
      if (name_ref && runs_ref &&
          !flatbuffers_failed(iree_hal_amdxdna_EntryPointDef_start(&builder)) &&
          !flatbuffers_failed(
              iree_hal_amdxdna_EntryPointDef_name_add(&builder, name_ref)) &&
          !flatbuffers_failed(iree_hal_amdxdna_EntryPointDef_pdi_index_add(
              &builder, entry_point.pdi_index)) &&
          !flatbuffers_failed(
              iree_hal_amdxdna_EntryPointDef_runs_add(&builder, runs_ref))) {
        entry_point_ref = iree_hal_amdxdna_EntryPointDef_end(&builder);
      }
      if (!name_ref || !runs_ref || !entry_point_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create entry point definition");
        break;
      }
      entry_point_refs.push_back(entry_point_ref);
    }
  }

  iree_hal_amdxdna_PdiDef_vec_ref_t pdis_ref = 0;
  iree_hal_amdxdna_EntryPointDef_vec_ref_t entry_points_ref = 0;
  if (iree_status_is_ok(status)) {
    pdis_ref = iree_hal_amdxdna_PdiDef_vec_create(&builder, pdi_refs.data(),
                                                  pdi_refs.size());
    entry_points_ref = iree_hal_amdxdna_EntryPointDef_vec_create(
        &builder, entry_point_refs.data(), entry_point_refs.size());
    if (!pdis_ref || !entry_points_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create executable vectors");
    }
  }

  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(
          flatbuffers_failed(
              iree_hal_amdxdna_ExecutableDef_pdis_add(&builder, pdis_ref)) ||
          flatbuffers_failed(iree_hal_amdxdna_ExecutableDef_entry_points_add(
              &builder, entry_points_ref)))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to populate PDIX executable");
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(!iree_hal_amdxdna_ExecutableDef_end_as_root(&builder))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to finish PDIX executable flatbuffer");
  }

  size_t flatbuffer_size = 0;
  void* flatbuffer_data = nullptr;
  if (iree_status_is_ok(status)) {
    flatbuffer_data =
        flatcc_builder_finalize_aligned_buffer(&builder, &flatbuffer_size);
    if (!flatbuffer_data || flatbuffer_size == 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to finalize PDIX executable");
    }
  }

  if (iree_status_is_ok(status)) {
    out_executable_data->assign(
        static_cast<const uint8_t*>(flatbuffer_data),
        static_cast<const uint8_t*>(flatbuffer_data) + flatbuffer_size);
  }

  flatcc_builder_aligned_free(flatbuffer_data);
  flatcc_builder_clear(&builder);
  return status;
}

static void ExpectInvalidExecutable(
    const std::vector<std::vector<uint8_t>>& pdis,
    const std::vector<TestEntryPointDef>& entry_points) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeExecutable(pdis, entry_points, &executable_data));

  char executable_format[64] = {};
  iree_host_size_t inferred_size = 0;
  iree_status_t status = iree_hal_amdxdna_native_executable_infer_format(
      iree_make_const_byte_span(executable_data.data(), executable_data.size()),
      sizeof(executable_format), executable_format, &inferred_size);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
}

TEST(ExecutableTest, ParsesSharedPdiAndRunDefinitions) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeExecutable(
      /*pdis=*/{{0x50, 0x44, 0x49, 0x00}},
      /*entry_points=*/
      {
          {
              /*name=*/"entry0",
              /*pdi_index=*/0,
              /*runs=*/
              {
                  {{10, 11}, {100, 101, 102}, {0, 0, 4}},
                  {{20, 21}, {}, {8, 1, 0, 12, 2, 16}},
              },
          },
          {
              /*name=*/"shared_pdi",
              /*pdi_index=*/0,
              /*runs=*/{{{30}, {}, {}}},
          },
          {
              /*name=*/"reuse_context",
              /*pdi_index=*/-1,
              /*runs=*/{{{40, 41}, {}, {}}},
          },
      },
      &executable_data));

  iree_hal_executable_load_params_t executable_params;
  iree_hal_executable_load_params_initialize(&executable_params);
  executable_params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());

  iree_hal_executable_t* base_executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, &executable_params, iree_allocator_system(),
      &base_executable));
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);

  ASSERT_EQ(executable->entry_point_count, 3u);

  const auto& entry0 = executable->entry_points[0];
  EXPECT_EQ(ToString(entry0.kernel_name), "entry0");
  EXPECT_EQ(ToVector(entry0.pdi),
            std::vector<uint8_t>({0x50, 0x44, 0x49, 0x00}));
  EXPECT_EQ(entry0.xclbin.count, 0u);
  ASSERT_EQ(entry0.asm_inst_runlist_count, 2u);
  ASSERT_EQ(entry0.reconf_data_runlist_count, 1u);
  ASSERT_EQ(entry0.patch_runlist_count, 2u);
  EXPECT_EQ(ToVector(entry0.asm_inst_runlist[0]),
            std::vector<uint32_t>({10, 11}));
  EXPECT_EQ(ToVector(entry0.reconf_data_runlist[0]),
            std::vector<uint32_t>({100, 101, 102}));
  EXPECT_EQ(ToVector(entry0.patch_runlist[0]),
            std::vector<uint32_t>({0, 0, 4}));
  EXPECT_EQ(ToVector(entry0.asm_inst_runlist[1]),
            std::vector<uint32_t>({20, 21}));
  EXPECT_EQ(ToVector(entry0.patch_runlist[1]),
            std::vector<uint32_t>({8, 1, 0, 12, 2, 16}));

  const auto& shared_pdi = executable->entry_points[1];
  EXPECT_EQ(ToString(shared_pdi.kernel_name), "shared_pdi");
  EXPECT_EQ(ToVector(shared_pdi.pdi), ToVector(entry0.pdi));
  EXPECT_EQ(shared_pdi.xclbin.count, 0u);
  ASSERT_EQ(shared_pdi.asm_inst_runlist_count, 1u);
  EXPECT_EQ(shared_pdi.reconf_data_runlist_count, 0u);
  EXPECT_EQ(ToVector(shared_pdi.asm_inst_runlist[0]),
            std::vector<uint32_t>({30}));

  const auto& reuse_context = executable->entry_points[2];
  EXPECT_EQ(ToString(reuse_context.kernel_name), "reuse_context");
  EXPECT_EQ(reuse_context.pdi.count, 0u);
  EXPECT_EQ(reuse_context.xclbin.count, 0u);
  ASSERT_EQ(reuse_context.asm_inst_runlist_count, 1u);
  EXPECT_EQ(reuse_context.reconf_data_runlist_count, 0u);
  EXPECT_EQ(ToVector(reuse_context.asm_inst_runlist[0]),
            std::vector<uint32_t>({40, 41}));

  EXPECT_EQ(iree_hal_executable_function_count(base_executable), 3u);
  iree_hal_executable_function_info_t function_info;
  IREE_ASSERT_OK(iree_hal_executable_function_info(
      base_executable, iree_hal_executable_function_from_index(1),
      &function_info));
  EXPECT_TRUE(
      iree_string_view_equal(function_info.name, IREE_SV("shared_pdi")));
  EXPECT_EQ(function_info.constant_byte_length, 0u);
  EXPECT_EQ(function_info.binding_count, 0u);
  EXPECT_EQ(function_info.parameter_count, 0u);
  EXPECT_EQ(function_info.workgroup_size[0], 1u);
  EXPECT_EQ(function_info.workgroup_size[1], 1u);
  EXPECT_EQ(function_info.workgroup_size[2], 1u);
  iree_hal_executable_function_parameter_t parameters[1];
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      base_executable, iree_hal_executable_function_from_index(1),
      IREE_ARRAYSIZE(parameters), parameters));

  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      base_executable, IREE_SV("reuse_context"), &function));
  EXPECT_EQ(iree_hal_executable_function_index(function), 2u);

  iree_hal_executable_function_t missing_function =
      iree_hal_executable_function_invalid();
  iree_status_t missing_status = iree_hal_executable_lookup_function_by_name(
      base_executable, IREE_SV("missing"), &missing_function);
  EXPECT_EQ(iree_status_code(missing_status), IREE_STATUS_NOT_FOUND);
  iree_status_free(missing_status);
  EXPECT_FALSE(iree_hal_executable_function_is_valid(missing_function));

  iree_hal_executable_release(base_executable);
}

TEST(ExecutableTest, RejectsMalformedExecutableDefinitions) {
  struct InvalidExecutableCase {
    const char* name;
    std::vector<std::vector<uint8_t>> pdis;
    std::vector<TestEntryPointDef> entry_points;
  };
  std::vector<InvalidExecutableCase> cases = {
      {
          "no PDI present",
          {},
          {{"entry", -1, {{{10}, {}, {}}}}},
      },
      {
          "no entry points",
          {{0x50, 0x44, 0x49, 0x00}},
          {},
      },
      {
          "empty PDI bytes",
          {{}},
          {{"entry", 0, {{{10}, {}, {}}}}},
      },
      {
          "empty entry point name",
          {{0x50, 0x44, 0x49, 0x00}},
          {{"", 0, {{{10}, {}, {}}}}},
      },
      {
          "PDI index out of range",
          {{0x50, 0x44, 0x49, 0x00}},
          {{"entry", 1, {{{10}, {}, {}}}}},
      },
      {
          "no entry point references a PDI",
          {{0x50, 0x44, 0x49, 0x00}},
          {{"entry", -1, {{{10}, {}, {}}}}},
      },
      {
          "no runs",
          {{0x50, 0x44, 0x49, 0x00}},
          {{"entry", 0, {}}},
      },
      {
          "empty control code",
          {{0x50, 0x44, 0x49, 0x00}},
          {{"entry", 0, {{{}, {}, {}}}}},
      },
      {
          "malformed patch table",
          {{0x50, 0x44, 0x49, 0x00}},
          {{"entry", 0, {{{10}, {}, {0, 0}}}}},
      },
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    ExpectInvalidExecutable(test_case.pdis, test_case.entry_points);
  }
}

TEST(ExecutableTest, DispatchPlanPrecomputesEntryPointPolicy) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeExecutable(
      /*pdis=*/{{0x50, 0x44, 0x49, 0x00}},
      /*entry_points=*/
      {{
          /*name=*/"entry0",
          /*pdi_index=*/0,
          /*runs=*/
          {
              {{10, 11}, {100, 101, 102}, {0, 0, 4}},
              {{20, 21}, {}, {8, 1, 0, 12, 2, 16}},
          },
      }},
      &executable_data));

  iree_hal_executable_load_params_t executable_params;
  iree_hal_executable_load_params_initialize(&executable_params);
  executable_params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());

  iree_hal_executable_t* base_executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, &executable_params, iree_allocator_system(),
      &base_executable));
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);

  iree_hal_amdxdna_native_c_device_caps_t native_caps = {};
  native_caps.dispatch_models =
      IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_PARTIAL_ELF;

  iree_hal_amdxdna_dispatch_plan_t plan = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_dispatch_plan_initialize(
      &native_caps, base_executable, iree_hal_executable_function_from_index(0),
      &plan));

  EXPECT_EQ(plan.executable, executable);
  EXPECT_EQ(plan.kernel_params, &executable->entry_points[0]);
  EXPECT_EQ(plan.entry_point, 0u);
  EXPECT_EQ(plan.control_code_count, 2u);
  EXPECT_EQ(plan.control_codes, executable->entry_points[0].asm_inst_runlist);
  EXPECT_EQ(plan.patch_table_count, 2u);
  EXPECT_EQ(plan.patch_tables, executable->entry_points[0].patch_runlist);
  EXPECT_EQ(plan.data_payload_count, 1u);
  EXPECT_EQ(plan.data_payloads,
            executable->entry_points[0].reconf_data_runlist);
  EXPECT_EQ(plan.data_payload_run_count, 1u);
  EXPECT_EQ(plan.pdi_span.data_length, 4u);
  EXPECT_EQ(plan.xclbin_span.data_length, 0u);
  EXPECT_TRUE(iree_string_view_equal(plan.kernel_name, IREE_SV("entry0")));
  EXPECT_TRUE(plan.has_host_patch_table);
  EXPECT_TRUE(plan.multi_control_code_or_pdi);
  EXPECT_TRUE(plan.use_chain_accumulation_policy);
  EXPECT_FALSE(plan.use_native_partial_elf_context);

  iree_hal_amdxdna_dispatch_plan_t invalid_plan = {};
  iree_status_t status = iree_hal_amdxdna_dispatch_plan_initialize(
      &native_caps, base_executable, iree_hal_executable_function_from_index(1),
      &invalid_plan);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  iree_status_free(status);

  iree_hal_executable_release(base_executable);
}

}  // namespace
