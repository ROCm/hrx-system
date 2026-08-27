// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/executable.h"

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/flatcc/building.h"
#include "iree/hal/api.h"
#include "iree/schemas/pdi_executable_def_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_status_t MakeMinimalExecutable(
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

  const uint8_t pdi_data[] = {'P', 'D', 'I', 0};
  flatbuffers_uint8_vec_ref_t pdi_data_ref = 0;
  iree_hal_amdxdna_PdiDef_ref_t pdi_ref = 0;
  if (iree_status_is_ok(status)) {
    pdi_data_ref = flatbuffers_uint8_vec_create(&builder, pdi_data,
                                                IREE_ARRAYSIZE(pdi_data));
    pdi_ref = iree_hal_amdxdna_PdiDef_create(&builder, pdi_data_ref);
    if (!pdi_data_ref || !pdi_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create PDI definition");
    }
  }

  const uint32_t control_code[] = {10};
  iree_hal_amdxdna_RunDef_ref_t run_ref = 0;
  if (iree_status_is_ok(status)) {
    flatbuffers_uint32_vec_ref_t control_code_ref =
        flatbuffers_uint32_vec_create(&builder, control_code,
                                      IREE_ARRAYSIZE(control_code));
    flatbuffers_uint32_vec_ref_t data_payload_ref =
        flatbuffers_uint32_vec_create(&builder, nullptr, 0);
    flatbuffers_uint32_vec_ref_t patch_table_ref =
        flatbuffers_uint32_vec_create(&builder, nullptr, 0);
    run_ref = iree_hal_amdxdna_RunDef_create(&builder, control_code_ref,
                                             data_payload_ref, patch_table_ref);
    if (!control_code_ref || !data_payload_ref || !patch_table_ref ||
        !run_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create run definition");
    }
  }

  iree_hal_amdxdna_EntryPointDef_ref_t entry_point_ref = 0;
  if (iree_status_is_ok(status)) {
    flatbuffers_string_ref_t name_ref =
        flatbuffers_string_create_str(&builder, "entry0");
    iree_hal_amdxdna_RunDef_vec_ref_t runs_ref =
        iree_hal_amdxdna_RunDef_vec_create(&builder, &run_ref, 1);
    if (name_ref && runs_ref &&
        !flatbuffers_failed(iree_hal_amdxdna_EntryPointDef_start(&builder)) &&
        !flatbuffers_failed(
            iree_hal_amdxdna_EntryPointDef_name_add(&builder, name_ref)) &&
        !flatbuffers_failed(
            iree_hal_amdxdna_EntryPointDef_pdi_index_add(&builder, 0)) &&
        !flatbuffers_failed(
            iree_hal_amdxdna_EntryPointDef_runs_add(&builder, runs_ref))) {
      entry_point_ref = iree_hal_amdxdna_EntryPointDef_end(&builder);
    }
    if (!name_ref || !runs_ref || !entry_point_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create entry point definition");
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_amdxdna_PdiDef_vec_ref_t pdis_ref =
        iree_hal_amdxdna_PdiDef_vec_create(&builder, &pdi_ref, 1);
    iree_hal_amdxdna_EntryPointDef_vec_ref_t entry_points_ref =
        iree_hal_amdxdna_EntryPointDef_vec_create(&builder, &entry_point_ref,
                                                  1);
    if (!pdis_ref || !entry_points_ref ||
        flatbuffers_failed(
            iree_hal_amdxdna_ExecutableDef_pdis_add(&builder, pdis_ref)) ||
        flatbuffers_failed(iree_hal_amdxdna_ExecutableDef_entry_points_add(
            &builder, entry_points_ref))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to populate PDIX executable");
    }
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

TEST(NopExecutableCacheTest, CanPrepareAmdxdnaFormats) {
  EXPECT_TRUE(iree_hal_amdxdna_executable_format_supported(
      iree_make_cstring_view("amdxdna-pdi-fb")));
  EXPECT_FALSE(iree_hal_amdxdna_executable_format_supported(
      iree_make_cstring_view("FOO?")));
}

TEST(NopExecutableCacheTest, InferFormatRecognizesValidExecutable) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeMinimalExecutable(&executable_data));

  char executable_format[64] = {};
  iree_host_size_t inferred_size = 0;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_infer_format(
      iree_make_const_byte_span(executable_data.data(), executable_data.size()),
      sizeof(executable_format), executable_format, &inferred_size));

  EXPECT_STREQ(executable_format, "amdxdna-pdi-fb");
  EXPECT_EQ(inferred_size, executable_data.size());
}

TEST(NopExecutableCacheTest, PrepareExecutableSucceedsForValidExecutable) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeMinimalExecutable(&executable_data));

  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, &load_params, iree_allocator_system(),
      &executable));
  ASSERT_NE(executable, nullptr);
  iree_hal_executable_release(executable);
}

TEST(NopExecutableCacheTest, PrepareRejectsUnknownFormatBeforeParsing) {
  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data = iree_const_byte_span_empty();

  iree_hal_executable_t* executable = nullptr;
  iree_status_t status = iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, &load_params, iree_allocator_system(),
      &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
  EXPECT_EQ(executable, nullptr);
}

}  // namespace
