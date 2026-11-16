// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/flatcc/building.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"
#include "iree/hal/drivers/amdxdna/executable.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"
#include "iree/hal/drivers/amdxdna/executable_reuse_cache.h"
#include "iree/hal/drivers/amdxdna/program_image.h"
#include "iree/hal/drivers/amdxdna/shim/ert.h"
#include "iree/schemas/amdxdna_xclbin_executable_def_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

static std::vector<uint8_t> ToVector(
    const iree_hal_amdxdna_program_image_t* image) {
  iree_const_byte_span_t contents =
      iree_hal_amdxdna_program_image_contents(image);
  return std::vector<uint8_t>(contents.data,
                              contents.data + contents.data_length);
}

static std::vector<uint32_t> ToVector(iree_hal_amdxdna_u32_list_t value) {
  return std::vector<uint32_t>(value.data, value.data + value.count);
}

struct TestPdiRelocationDef {
  uint32_t offset = 0;
  uint32_t pdi_index = 0;
  int64_t addend = 0;
};

struct TestControlParameterRelocationDef {
  uint32_t offset = 0;
  int64_t addend = 0;
};

struct TestControlParameterDef {
  const char* name = nullptr;
  const char* scalar_type = nullptr;
  uint32_t constant_offset = 0;
  uint32_t state_table_index = 0;
  uint32_t byte_length = 0;
  uint32_t kind = 0;
};

struct TestRunDef {
  std::vector<uint32_t> control_code;
  std::vector<uint32_t> data_payload;
  std::vector<uint32_t> patch_table;
  std::vector<TestPdiRelocationDef> pdi_relocations;
  std::vector<TestControlParameterRelocationDef> control_parameter_relocations;
};

static flatbuffers_uint32_vec_ref_t CreateUInt32Vec(
    flatbuffers_builder_t* builder, const std::vector<uint32_t>& values) {
  return flatbuffers_uint32_vec_create(
      builder, values.empty() ? nullptr : values.data(), values.size());
}

// AXLF / AIE_PARTITION layout constants (mirrored from xclbin_util.cc).
constexpr uint32_t kAiePartitionSection = 32;
constexpr size_t kSectionCountOffset = 0x1C0;
constexpr size_t kSectionTableOffset = 0x1C8;
constexpr size_t kSectionRecordSize = 40;
constexpr size_t kPartitionHeaderSize = 0xC8;
constexpr size_t kPdiArraySizeOffset = 120;
constexpr size_t kPdiArrayOffsetOffset = 124;
constexpr size_t kPdiRecordSize = 0x60;
constexpr size_t kPdiImageSizeOffset = 16;
constexpr size_t kPdiImageOffsetOffset = 20;

static void WriteU32(std::vector<uint8_t>& v, size_t off, uint32_t val) {
  std::memcpy(v.data() + off, &val, sizeof(val));
}
static void WriteU64(std::vector<uint8_t>& v, size_t off, uint64_t val) {
  std::memcpy(v.data() + off, &val, sizeof(val));
}

// Builds a single-section AXLF/xclbin2 container embedding the given PDIs in an
// AIE_PARTITION section (the layout iree_hal_amdxdna_xclbin_extract_pdi reads).
static std::vector<uint8_t> BuildXclbin(
    const std::vector<std::vector<uint8_t>>& pdis) {
  const size_t table_off = kPartitionHeaderSize;
  std::vector<uint8_t> part(table_off + pdis.size() * kPdiRecordSize, 0);
  WriteU32(part, kPdiArraySizeOffset, static_cast<uint32_t>(pdis.size()));
  WriteU32(part, kPdiArrayOffsetOffset, static_cast<uint32_t>(table_off));
  size_t cur = part.size();
  for (size_t i = 0; i < pdis.size(); ++i) {
    const size_t rec = table_off + i * kPdiRecordSize;
    WriteU32(part, rec + kPdiImageSizeOffset,
             static_cast<uint32_t>(pdis[i].size()));
    WriteU32(part, rec + kPdiImageOffsetOffset, static_cast<uint32_t>(cur));
    part.insert(part.end(), pdis[i].begin(), pdis[i].end());
    cur += pdis[i].size();
  }
  const size_t part_off = kSectionTableOffset + kSectionRecordSize;
  std::vector<uint8_t> axlf(part_off, 0);
  std::memcpy(axlf.data(), "xclbin2\0", 8);
  WriteU32(axlf, kSectionCountOffset, 1u);
  WriteU32(axlf, kSectionTableOffset + 0, kAiePartitionSection);
  WriteU64(axlf, kSectionTableOffset + 24, part_off);
  WriteU64(axlf, kSectionTableOffset + 32, part.size());
  axlf.insert(axlf.end(), part.begin(), part.end());
  return axlf;
}

struct TestXadxEntryPointDef {
  const char* name = nullptr;
  int32_t pdi_index = -1;
  int32_t xclbin_index = -1;
  std::vector<TestRunDef> runs;
  std::vector<TestControlParameterDef> control_parameters;
};

static iree_status_t MakeXadxExecutable(
    const std::vector<std::vector<uint8_t>>& xclbins,
    const std::vector<TestXadxEntryPointDef>& entry_points,
    std::vector<uint8_t>* out_executable_data,
    const std::vector<std::vector<uint8_t>>& pdis = {},
    bool include_extensions = true) {
  out_executable_data->clear();
  flatbuffers_builder_t builder;
  if (IREE_UNLIKELY(flatcc_builder_init(&builder) != 0)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to initialize flatbuffer builder");
  }
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(flatbuffers_failed(
          iree_hal_amdxdna_xclbin_ExecutableDef_start_as_root(&builder)))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to start XADX executable flatbuffer");
  }

  std::vector<iree_hal_amdxdna_xclbin_XclbinDef_ref_t> xclbin_refs;
  if (iree_status_is_ok(status)) {
    for (const auto& xclbin : xclbins) {
      flatbuffers_string_ref_t str = flatbuffers_string_create(
          &builder, reinterpret_cast<const char*>(xclbin.data()),
          xclbin.size());
      iree_hal_amdxdna_xclbin_XclbinDef_ref_t ref =
          iree_hal_amdxdna_xclbin_XclbinDef_create(&builder, str);
      if (!str || !ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create XclbinDef");
        break;
      }
      xclbin_refs.push_back(ref);
    }
  }

  std::vector<iree_hal_amdxdna_xclbin_PdiDef_ref_t> pdi_refs;
  if (iree_status_is_ok(status)) {
    for (const auto& pdi : pdis) {
      flatbuffers_uint8_vec_ref_t data_ref =
          flatbuffers_uint8_vec_create(&builder, pdi.data(), pdi.size());
      auto ref = iree_hal_amdxdna_xclbin_PdiDef_create(&builder, data_ref);
      if (!data_ref || !ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create standalone PdiDef");
        break;
      }
      pdi_refs.push_back(ref);
    }
  }

  std::vector<iree_hal_amdxdna_xclbin_EntryPointDef_ref_t> entry_refs;
  if (iree_status_is_ok(status)) {
    for (const auto& ep : entry_points) {
      std::vector<iree_hal_amdxdna_xclbin_RunDef_ref_t> run_refs;
      for (const auto& run : ep.runs) {
        std::vector<iree_hal_amdxdna_xclbin_PdiRelocationDef_ref_t>
            relocation_refs;
        for (const auto& relocation : run.pdi_relocations) {
          relocation_refs.push_back(
              iree_hal_amdxdna_xclbin_PdiRelocationDef_create(
                  &builder, relocation.offset, relocation.pdi_index,
                  relocation.addend));
        }
        auto relocations_ref =
            iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_create(
                &builder, relocation_refs.data(), relocation_refs.size());
        std::vector<iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_ref_t>
            control_parameter_relocation_refs;
        for (const auto& relocation : run.control_parameter_relocations) {
          control_parameter_relocation_refs.push_back(
              iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_create(
                  &builder, relocation.offset, relocation.addend));
        }
        auto control_parameter_relocations_ref =
            iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_create(
                &builder, control_parameter_relocation_refs.data(),
                control_parameter_relocation_refs.size());
        const auto control_code_ref =
            CreateUInt32Vec(&builder, run.control_code);
        const auto data_payload_ref =
            CreateUInt32Vec(&builder, run.data_payload);
        const auto patch_table_ref = CreateUInt32Vec(&builder, run.patch_table);
        iree_hal_amdxdna_xclbin_RunDef_ref_t run_ref = 0;
        if (include_extensions) {
          run_ref = iree_hal_amdxdna_xclbin_RunDef_create(
              &builder, control_code_ref, data_payload_ref, patch_table_ref,
              relocations_ref, control_parameter_relocations_ref);
        } else if (!flatbuffers_failed(
                       iree_hal_amdxdna_xclbin_RunDef_start(&builder)) &&
                   !flatbuffers_failed(
                       iree_hal_amdxdna_xclbin_RunDef_control_code_add(
                           &builder, control_code_ref)) &&
                   !flatbuffers_failed(
                       iree_hal_amdxdna_xclbin_RunDef_data_payload_add(
                           &builder, data_payload_ref)) &&
                   !flatbuffers_failed(
                       iree_hal_amdxdna_xclbin_RunDef_patch_table_add(
                           &builder, patch_table_ref))) {
          run_ref = iree_hal_amdxdna_xclbin_RunDef_end(&builder);
        }
        if (!run_ref) {
          status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "failed to create XADX run");
          break;
        }
        run_refs.push_back(run_ref);
      }
      if (!iree_status_is_ok(status)) break;
      flatbuffers_string_ref_t name_ref =
          flatbuffers_string_create_str(&builder, ep.name);
      iree_hal_amdxdna_xclbin_RunDef_vec_ref_t runs_ref =
          iree_hal_amdxdna_xclbin_RunDef_vec_create(&builder, run_refs.data(),
                                                    run_refs.size());
      std::vector<iree_hal_amdxdna_xclbin_ControlParameterDef_ref_t>
          parameter_refs;
      for (const auto& parameter : ep.control_parameters) {
        auto parameter_name =
            flatbuffers_string_create_str(&builder, parameter.name);
        auto scalar_type =
            flatbuffers_string_create_str(&builder, parameter.scalar_type);
        parameter_refs.push_back(
            iree_hal_amdxdna_xclbin_ControlParameterDef_create(
                &builder, parameter_name, scalar_type,
                parameter.constant_offset, parameter.state_table_index,
                parameter.byte_length, parameter.kind));
      }
      auto parameters_ref =
          iree_hal_amdxdna_xclbin_ControlParameterDef_vec_create(
              &builder, parameter_refs.data(), parameter_refs.size());
      iree_hal_amdxdna_xclbin_EntryPointDef_ref_t ep_ref = 0;
      if (name_ref && runs_ref &&
          !flatbuffers_failed(
              iree_hal_amdxdna_xclbin_EntryPointDef_start(&builder)) &&
          !flatbuffers_failed(iree_hal_amdxdna_xclbin_EntryPointDef_name_add(
              &builder, name_ref)) &&
          !flatbuffers_failed(
              iree_hal_amdxdna_xclbin_EntryPointDef_pdi_index_add(
                  &builder, ep.pdi_index)) &&
          !flatbuffers_failed(
              iree_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_add(
                  &builder, ep.xclbin_index)) &&
          !flatbuffers_failed(iree_hal_amdxdna_xclbin_EntryPointDef_runs_add(
              &builder, runs_ref)) &&
          (!include_extensions ||
           !flatbuffers_failed(
               iree_hal_amdxdna_xclbin_EntryPointDef_control_parameters_add(
                   &builder, parameters_ref)))) {
        ep_ref = iree_hal_amdxdna_xclbin_EntryPointDef_end(&builder);
      }
      if (!ep_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create XADX entry point");
        break;
      }
      entry_refs.push_back(ep_ref);
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_amdxdna_xclbin_XclbinDef_vec_ref_t xclbins_ref =
        iree_hal_amdxdna_xclbin_XclbinDef_vec_create(
            &builder, xclbin_refs.data(), xclbin_refs.size());
    iree_hal_amdxdna_xclbin_PdiDef_vec_ref_t pdis_ref =
        include_extensions ? iree_hal_amdxdna_xclbin_PdiDef_vec_create(
                                 &builder, pdi_refs.data(), pdi_refs.size())
                           : 0;
    iree_hal_amdxdna_xclbin_EntryPointDef_vec_ref_t entries_ref =
        iree_hal_amdxdna_xclbin_EntryPointDef_vec_create(
            &builder, entry_refs.data(), entry_refs.size());
    if (!xclbins_ref || (include_extensions && !pdis_ref) || !entries_ref ||
        flatbuffers_failed(iree_hal_amdxdna_xclbin_ExecutableDef_xclbins_add(
            &builder, xclbins_ref)) ||
        flatbuffers_failed(
            iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_add(
                &builder, entries_ref)) ||
        (include_extensions &&
         flatbuffers_failed(iree_hal_amdxdna_xclbin_ExecutableDef_pdis_add(
             &builder, pdis_ref)))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to populate XADX executable");
    }
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(
          !iree_hal_amdxdna_xclbin_ExecutableDef_end_as_root(&builder))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to finish XADX flatbuffer");
  }

  size_t size = 0;
  void* data = nullptr;
  if (iree_status_is_ok(status)) {
    data = flatcc_builder_finalize_aligned_buffer(&builder, &size);
    if (!data || size == 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to finalize XADX executable");
    }
  }
  if (iree_status_is_ok(status)) {
    out_executable_data->assign(static_cast<const uint8_t*>(data),
                                static_cast<const uint8_t*>(data) + size);
  }
  flatcc_builder_aligned_free(data);
  flatcc_builder_clear(&builder);
  return status;
}

TEST(ExecutableXclbinTest, ParsesXadxXclbinDefinitions) {
  std::vector<uint8_t> pdi0(8, 0xC0);
  std::vector<uint8_t> pdi1(12, 0xC1);
  std::vector<uint8_t> xclbin = BuildXclbin({pdi0, pdi1});

  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeXadxExecutable(
      /*xclbins=*/{xclbin},
      /*entry_points=*/
      {
          {"xadx0", /*pdi_index=*/0, /*xclbin_index=*/0, {{{10, 11}, {}, {}}}},
          {"xadx1", /*pdi_index=*/1, /*xclbin_index=*/0, {{{20}, {}, {}}}},
      },
      &executable_data));

  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());

  iree_hal_executable_t* base_executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, /*image_store=*/nullptr, &params,
      iree_allocator_system(), &base_executable));
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);

  ASSERT_EQ(executable->entry_point_count, 2u);
  const auto& e0 = executable->entry_points[0];
  EXPECT_EQ(ToString(e0.kernel_name), "xadx0");
  EXPECT_EQ(ToVector(e0.pdi_image),
            pdi0);  // extracted from the AIE_PARTITION by index
  EXPECT_EQ(ToVector(e0.xclbin_image),
            xclbin);  // raw AXLF context wrapper retained
  ASSERT_EQ(e0.asm_inst_runlist_count, 1u);
  EXPECT_EQ(ToVector(e0.asm_inst_runlist[0]), std::vector<uint32_t>({10, 11}));
  const auto& e1 = executable->entry_points[1];
  EXPECT_EQ(ToString(e1.kernel_name), "xadx1");
  EXPECT_EQ(ToVector(e1.pdi_image), pdi1);
  ASSERT_EQ(e1.asm_inst_runlist_count, 1u);
  EXPECT_EQ(ToVector(e1.asm_inst_runlist[0]), std::vector<uint32_t>({20}));

  iree_hal_executable_release(base_executable);
}

static iree_hal_executable_t* LoadXadx(
    iree_hal_amdxdna_program_image_store_t* image_store,
    const std::vector<uint8_t>& xclbin, int32_t pdi_index,
    std::vector<uint32_t> control_code) {
  std::vector<uint8_t> executable_data;
  IREE_CHECK_OK(MakeXadxExecutable(
      /*xclbins=*/{xclbin},
      /*entry_points=*/
      {{"shape", pdi_index, /*xclbin_index=*/0, {{control_code, {}, {}}}}},
      &executable_data));
  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());
  iree_hal_executable_t* executable = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, image_store, &params, iree_allocator_system(),
      &executable));
  return executable;
}

TEST(ExecutableXclbinTest, ShapeSpecializedExecutablesShareProgramImages) {
  std::vector<uint8_t> pdi(64, 0xC0);
  std::vector<uint8_t> xclbin = BuildXclbin({pdi});
  iree_hal_amdxdna_program_image_store_t* store = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_program_image_store_create(
      iree_allocator_system(), &store));

  iree_hal_executable_t* first = LoadXadx(store, xclbin, 0, {10, 11});
  iree_hal_executable_t* second = LoadXadx(store, xclbin, 0, {20, 21, 22});
  const auto& first_entry =
      iree_hal_amdxdna_executable_cast(first)->entry_points[0];
  const auto& second_entry =
      iree_hal_amdxdna_executable_cast(second)->entry_points[0];
  ASSERT_NE(first_entry.xclbin_image, nullptr);
  ASSERT_NE(first_entry.pdi_image, nullptr);
  EXPECT_EQ(first_entry.xclbin_image, second_entry.xclbin_image);
  EXPECT_EQ(first_entry.pdi_image, second_entry.pdi_image);
  EXPECT_EQ(ToVector(second_entry.pdi_image), pdi);

  iree_hal_amdxdna_program_image_store_stats_t stats;
  iree_hal_amdxdna_program_image_store_query_stats(store, &stats);
  EXPECT_EQ(stats.image_count, 2u);
  EXPECT_EQ(stats.content_bytes, xclbin.size() + pdi.size());

  // The owner may drop the store first; images keep it alive until the last
  // executable that references them is released.
  iree_hal_amdxdna_program_image_store_retain(store);
  iree_hal_amdxdna_program_image_store_release(store);
  iree_hal_executable_release(first);
  iree_hal_amdxdna_program_image_store_query_stats(store, &stats);
  EXPECT_EQ(stats.image_count, 2u);
  iree_hal_executable_release(second);
  iree_hal_amdxdna_program_image_store_query_stats(store, &stats);
  EXPECT_EQ(stats.image_count, 0u);
  EXPECT_EQ(stats.content_bytes, 0u);
  iree_hal_amdxdna_program_image_store_release(store);
}

TEST(ExecutableXclbinTest, ExecutablesWithoutStoreOwnPrivateImages) {
  std::vector<uint8_t> xclbin = BuildXclbin({std::vector<uint8_t>(16, 0xC0)});
  iree_hal_executable_t* first = LoadXadx(nullptr, xclbin, 0, {10});
  iree_hal_executable_t* second = LoadXadx(nullptr, xclbin, 0, {10});
  const auto& first_entry =
      iree_hal_amdxdna_executable_cast(first)->entry_points[0];
  const auto& second_entry =
      iree_hal_amdxdna_executable_cast(second)->entry_points[0];
  EXPECT_NE(first_entry.xclbin_image, second_entry.xclbin_image);
  EXPECT_EQ(ToVector(first_entry.xclbin_image),
            ToVector(second_entry.xclbin_image));
  iree_hal_executable_release(first);
  iree_hal_executable_release(second);
}

TEST(ExecutableXclbinTest, FootprintSeparatesExclusiveBytesFromImages) {
  std::vector<uint8_t> small_xclbin =
      BuildXclbin({std::vector<uint8_t>(16, 0xC0)});
  std::vector<uint8_t> large_xclbin =
      BuildXclbin({std::vector<uint8_t>(64 * 1024, 0xC1)});
  // A zero transaction header declares no ops, so no patch sites are derived.
  std::vector<uint32_t> large_code(1024, 0);
  iree_hal_executable_t* small = LoadXadx(nullptr, small_xclbin, 0, {1});
  iree_hal_executable_t* large_image = LoadXadx(nullptr, large_xclbin, 0, {1});
  iree_hal_executable_t* large_code_executable =
      LoadXadx(nullptr, small_xclbin, 0, large_code);

  iree_hal_amdxdna_executable_footprint_t small_footprint;
  iree_hal_amdxdna_executable_footprint_t large_image_footprint;
  iree_hal_amdxdna_executable_footprint_t large_code_footprint;
  ASSERT_TRUE(
      iree_hal_amdxdna_executable_query_footprint(small, &small_footprint));
  ASSERT_TRUE(iree_hal_amdxdna_executable_query_footprint(
      large_image, &large_image_footprint));
  ASSERT_TRUE(iree_hal_amdxdna_executable_query_footprint(
      large_code_executable, &large_code_footprint));

  const auto& entry =
      iree_hal_amdxdna_executable_cast(large_image)->entry_points[0];
  ASSERT_EQ(large_image_footprint.image_count, 2u);
  EXPECT_EQ(large_image_footprint.images[0], entry.pdi_image);
  EXPECT_EQ(large_image_footprint.images[1], entry.xclbin_image);
  // Images are reported separately and never charged as exclusive bytes;
  // control code is.
  EXPECT_EQ(large_image_footprint.exclusive_host_bytes,
            small_footprint.exclusive_host_bytes);
  EXPECT_EQ(large_code_footprint.exclusive_host_bytes -
                small_footprint.exclusive_host_bytes,
            (large_code.size() - 1) * sizeof(uint32_t));

  iree_const_byte_span_t control_code =
      iree_hal_amdxdna_executable_single_run_control_code(
          large_code_executable);
  ASSERT_EQ(control_code.data_length, large_code.size() * sizeof(uint32_t));
  EXPECT_EQ(0, memcmp(control_code.data, large_code.data(),
                      control_code.data_length));

  iree_hal_executable_release(small);
  iree_hal_executable_release(large_image);
  iree_hal_executable_release(large_code_executable);
}

class ExecutableReuseCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amdxdna_program_image_store_create(
        iree_allocator_system(), &store_));
    xclbin_ = BuildXclbin({std::vector<uint8_t>(4096, 0xC0)});
  }
  void TearDown() override {
    iree_hal_amdxdna_executable_reuse_cache_destroy(cache_);
    for (iree_hal_executable_t* executable : executables_) {
      iree_hal_executable_release(executable);
    }
    iree_hal_amdxdna_program_image_store_stats_t stats;
    iree_hal_amdxdna_program_image_store_query_stats(store_, &stats);
    EXPECT_EQ(stats.image_count, 0u);
    iree_hal_amdxdna_program_image_store_release(store_);
  }

  void CreateCache(iree_host_size_t max_entries,
                   iree_host_size_t max_host_bytes) {
    iree_hal_amdxdna_executable_reuse_cache_limits_t limits = {max_entries,
                                                               max_host_bytes};
    IREE_ASSERT_OK(iree_hal_amdxdna_executable_reuse_cache_create(
        limits, iree_allocator_system(), &cache_));
  }

  // Loads an executable owned by the fixture.
  iree_hal_executable_t* Load(std::vector<uint32_t> control_code) {
    iree_hal_executable_t* executable =
        LoadXadx(store_, xclbin_, 0, std::move(control_code));
    executables_.push_back(executable);
    return executable;
  }

  // The key aliases |descriptor| and |executable|'s control code.
  static iree_hal_amdxdna_executable_reuse_key_t Key(
      const char* descriptor, iree_hal_executable_t* executable) {
    iree_hal_amdxdna_executable_reuse_key_t key;
    key.descriptor = iree_make_const_byte_span(descriptor, strlen(descriptor));
    key.control_code =
        iree_hal_amdxdna_executable_single_run_control_code(executable);
    return key;
  }

  // Returns the executable cached under |key| without leaking the reference.
  iree_hal_executable_t* Lookup(
      const iree_hal_amdxdna_executable_reuse_key_t& key) {
    iree_hal_executable_t* executable =
        iree_hal_amdxdna_executable_reuse_cache_lookup(cache_, &key);
    iree_hal_executable_release(executable);
    return executable;
  }

  iree_hal_amdxdna_executable_reuse_cache_stats_t Stats() {
    iree_hal_amdxdna_executable_reuse_cache_stats_t stats;
    iree_hal_amdxdna_executable_reuse_cache_query_stats(cache_, &stats);
    return stats;
  }

  iree_hal_amdxdna_program_image_store_t* store_ = nullptr;
  iree_hal_amdxdna_executable_reuse_cache_t* cache_ = nullptr;
  std::vector<uint8_t> xclbin_;
  std::vector<iree_hal_executable_t*> executables_;
};

TEST_F(ExecutableReuseCacheTest, HitsMissesAndLruEviction) {
  CreateCache(/*max_entries=*/2, /*max_host_bytes=*/SIZE_MAX);
  const char* descriptor = "identity";
  iree_hal_executable_t* a = Load({1});
  iree_hal_executable_t* b = Load({2});
  iree_hal_executable_t* c = Load({3});
  auto key_a = Key(descriptor, a);
  auto key_b = Key(descriptor, b);
  auto key_c = Key(descriptor, c);

  EXPECT_EQ(Lookup(key_a), nullptr);
  EXPECT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_a, a));
  EXPECT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_b, b));
  EXPECT_FALSE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_a, a));
  EXPECT_EQ(Lookup(key_a), a);  // b becomes least recently used.
  EXPECT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_c, c));
  EXPECT_EQ(Lookup(key_b), nullptr);
  EXPECT_EQ(Lookup(key_a), a);
  EXPECT_EQ(Lookup(key_c), c);

  auto stats = Stats();
  EXPECT_EQ(stats.entry_count, 2u);
  EXPECT_EQ(stats.hit_count, 3u);
  EXPECT_EQ(stats.miss_count, 2u);
  EXPECT_EQ(stats.eviction_count, 1u);

  iree_hal_amdxdna_executable_reuse_cache_clear(cache_);
  stats = Stats();
  EXPECT_EQ(stats.entry_count, 0u);
  EXPECT_EQ(stats.host_bytes, 0u);
  EXPECT_EQ(stats.image_bytes, 0u);
  EXPECT_EQ(Lookup(key_a), nullptr);
}

TEST_F(ExecutableReuseCacheTest, KeysCompareDescriptorAndControlCodeExactly) {
  CreateCache(/*max_entries=*/8, /*max_host_bytes=*/SIZE_MAX);
  iree_hal_executable_t* a = Load({1, 2});
  iree_hal_executable_t* b = Load({1, 3});
  auto key_a = Key("identity-a", a);
  ASSERT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_a, a));

  EXPECT_EQ(Lookup(Key("identity-b", a)), nullptr);
  EXPECT_EQ(Lookup(Key("identity-a", b)), nullptr);
  EXPECT_EQ(Lookup(Key("identity-a", a)), a);

  // The key's control code must describe the executable being cached.
  auto mismatched = Key("identity-c", a);
  EXPECT_FALSE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &mismatched, b));
}

TEST_F(ExecutableReuseCacheTest, SharedImagesAreChargedOnce) {
  CreateCache(/*max_entries=*/8, /*max_host_bytes=*/SIZE_MAX);
  iree_hal_executable_t* a = Load({1});
  iree_hal_executable_t* b = Load({2});
  auto key_a = Key("identity", a);
  auto key_b = Key("identity", b);
  iree_hal_amdxdna_executable_footprint_t footprint;
  ASSERT_TRUE(iree_hal_amdxdna_executable_query_footprint(a, &footprint));
  ASSERT_EQ(footprint.image_count, 2u);
  const iree_host_size_t image_bytes =
      iree_hal_amdxdna_program_image_host_bytes(footprint.images[0]) +
      iree_hal_amdxdna_program_image_host_bytes(footprint.images[1]);

  ASSERT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_a, a));
  const auto one = Stats();
  EXPECT_EQ(one.image_bytes, image_bytes);
  EXPECT_GT(one.host_bytes - one.image_bytes, footprint.exclusive_host_bytes);

  ASSERT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_b, b));
  const auto two = Stats();
  EXPECT_EQ(two.image_bytes, image_bytes);
  EXPECT_LT(two.host_bytes - one.host_bytes, image_bytes);
}

TEST_F(ExecutableReuseCacheTest, ByteBudgetEvictsAndRejectsOversizedEntries) {
  iree_hal_executable_t* a = Load({1});
  iree_hal_executable_t* b = Load({2});
  auto key_a = Key("identity", a);
  auto key_b = Key("identity", b);

  // Measure one entry's charge with an unbounded cache.
  CreateCache(/*max_entries=*/8, /*max_host_bytes=*/SIZE_MAX);
  ASSERT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_a, a));
  const iree_host_size_t one_entry_bytes = Stats().host_bytes;
  iree_hal_amdxdna_executable_reuse_cache_destroy(cache_);
  cache_ = nullptr;

  CreateCache(/*max_entries=*/8, /*max_host_bytes=*/one_entry_bytes);
  ASSERT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_a, a));
  ASSERT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_b, b));
  EXPECT_EQ(Lookup(key_a), nullptr);
  EXPECT_EQ(Lookup(key_b), b);
  EXPECT_EQ(Stats().entry_count, 1u);
  EXPECT_LE(Stats().host_bytes, one_entry_bytes);
  iree_hal_amdxdna_executable_reuse_cache_destroy(cache_);
  cache_ = nullptr;

  CreateCache(/*max_entries=*/8, /*max_host_bytes=*/one_entry_bytes - 1);
  EXPECT_FALSE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_a, a));
  EXPECT_EQ(Stats().entry_count, 0u);
  EXPECT_EQ(Stats().host_bytes, 0u);
}

TEST_F(ExecutableReuseCacheTest, CacheKeepsExecutablesAliveUntilEvicted) {
  CreateCache(/*max_entries=*/1, /*max_host_bytes=*/SIZE_MAX);
  const std::vector<uint32_t> code_a = {1};
  iree_hal_executable_t* a = LoadXadx(store_, xclbin_, 0, code_a);
  // Own the key bytes: |a| is destroyed when evicted below.
  iree_hal_amdxdna_executable_reuse_key_t key_a;
  key_a.descriptor = iree_make_const_byte_span("identity", 8);
  key_a.control_code = iree_make_const_byte_span(
      code_a.data(), code_a.size() * sizeof(uint32_t));
  ASSERT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_a, a));
  iree_hal_executable_release(a);
  // The cached reference keeps the executable, and with it the shared images,
  // alive after the caller's release.
  EXPECT_EQ(Lookup(key_a), a);
  iree_hal_amdxdna_program_image_store_stats_t image_stats;
  iree_hal_amdxdna_program_image_store_query_stats(store_, &image_stats);
  EXPECT_EQ(image_stats.image_count, 2u);

  iree_hal_executable_t* b = Load({2});
  auto key_b = Key("identity", b);
  ASSERT_TRUE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key_b, b));
  EXPECT_EQ(Lookup(key_a), nullptr);
}

TEST_F(ExecutableReuseCacheTest, RejectsMultiEntryExecutables) {
  CreateCache(/*max_entries=*/8, /*max_host_bytes=*/SIZE_MAX);
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeXadxExecutable(
      /*xclbins=*/{xclbin_},
      /*entry_points=*/
      {
          {"first", /*pdi_index=*/0, /*xclbin_index=*/0, {{{1}, {}, {}}}},
          {"second", /*pdi_index=*/0, /*xclbin_index=*/0, {{{2}, {}, {}}}},
      },
      &executable_data));
  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());
  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, store_, &params, iree_allocator_system(),
      &executable));
  executables_.push_back(executable);

  EXPECT_EQ(iree_hal_amdxdna_executable_single_run_control_code(executable)
                .data_length,
            0u);
  const uint32_t code = 1;
  iree_hal_amdxdna_executable_reuse_key_t key;
  key.descriptor = iree_make_const_byte_span("identity", 8);
  key.control_code = iree_make_const_byte_span(&code, sizeof(code));
  EXPECT_FALSE(
      iree_hal_amdxdna_executable_reuse_cache_insert(cache_, &key, executable));
}

TEST(ExecutableXclbinTest, ParsesLegacyXadxWithoutExtensionFields) {
  std::vector<uint8_t> context_pdi(8, 0xC0);
  std::vector<uint8_t> xclbin = BuildXclbin({context_pdi});
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeXadxExecutable(
      /*xclbins=*/{xclbin},
      /*entry_points=*/
      {{"legacy", /*pdi_index=*/0, /*xclbin_index=*/0, {{{10, 11}, {}, {}}}}},
      &executable_data, /*pdis=*/{}, /*include_extensions=*/false));

  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());
  iree_hal_executable_t* base_executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, /*image_store=*/nullptr, &params,
      iree_allocator_system(), &base_executable));
  auto* executable = iree_hal_amdxdna_executable_cast(base_executable);
  ASSERT_EQ(executable->entry_point_count, 1u);
  EXPECT_EQ(executable->pdi_count, 0u);
  ASSERT_EQ(executable->entry_points[0].pdi_relocation_runlist_count, 1u);
  EXPECT_EQ(executable->entry_points[0].pdi_relocation_runlist[0].count, 0u);
  iree_hal_executable_release(base_executable);
}

TEST(ExecutableXclbinTest, ParsesStandalonePdisAndRelocations) {
  std::vector<uint8_t> context_pdi(8, 0xC0);
  std::vector<uint8_t> xclbin = BuildXclbin({context_pdi});
  std::vector<uint8_t> load_pdi0(12, 0xD0);
  std::vector<uint8_t> load_pdi1(16, 0xD1);

  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeXadxExecutable(
      /*xclbins=*/{xclbin},
      /*entry_points=*/
      {{"fused",
        /*pdi_index=*/0,
        /*xclbin_index=*/0,
        {{{0, 0, 0, 0, 0, 0, 0, 0},
          {},
          {},
          {{/*offset=*/4, /*pdi_index=*/1, /*addend=*/-16}},
          {{/*offset=*/20, /*addend=*/0}}}},
        {{"input_offset", "i32", /*constant_offset=*/0,
          /*state_table_index=*/3, /*byte_length=*/4, /*kind=*/1}}}},
      &executable_data,
      /*pdis=*/{load_pdi0, load_pdi1}));

  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());

  iree_hal_executable_t* base_executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, /*image_store=*/nullptr, &params,
      iree_allocator_system(), &base_executable));
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);

  ASSERT_EQ(executable->pdi_count, 2u);
  EXPECT_EQ(ToVector(executable->pdis[0]), load_pdi0);
  EXPECT_EQ(ToVector(executable->pdis[1]), load_pdi1);
  const auto& entry = executable->entry_points[0];
  ASSERT_EQ(entry.pdi_relocation_runlist_count, 1u);
  ASSERT_EQ(entry.pdi_relocation_runlist[0].count, 1u);
  const auto& relocation = entry.pdi_relocation_runlist[0].data[0];
  EXPECT_EQ(relocation.transaction_offset, 4u);
  EXPECT_EQ(relocation.pdi_ordinal, 1u);
  EXPECT_EQ(relocation.addend, -16);
  ASSERT_EQ(entry.control_parameter_relocation_runlist_count, 1u);
  ASSERT_EQ(entry.control_parameter_relocation_runlist[0].count, 1u);
  EXPECT_EQ(
      entry.control_parameter_relocation_runlist[0].data[0].transaction_offset,
      20u);
  ASSERT_EQ(entry.parameter_count, 1u);
  EXPECT_EQ(entry.constant_byte_length, 4u);
  EXPECT_EQ(ToString(entry.parameters[0].name), "input_offset");
  EXPECT_EQ(entry.parameters[0].offset, 0u);
  EXPECT_EQ(entry.parameters[0].native_abi_offset, 12u);

  iree_hal_amdxdna_native_c_device_caps_t full_elf_caps = {};
  full_elf_caps.dispatch_models =
      IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_FULL_ELF;
  iree_hal_amdxdna_dispatch_plan_t plan = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_dispatch_plan_initialize(
      &full_elf_caps, base_executable,
      iree_hal_executable_function_from_index(0), &plan));
  EXPECT_TRUE(plan.use_native_full_elf);
  EXPECT_FALSE(plan.use_native_partial_elf_context);
  EXPECT_FALSE(plan.use_chain_accumulation_policy);

  iree_hal_amdxdna_native_c_device_caps_t no_full_elf_caps = {};
  no_full_elf_caps.dispatch_models =
      IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU;
  EXPECT_THAT(
      iree_hal_amdxdna_dispatch_plan_initialize(
          &no_full_elf_caps, base_executable,
          iree_hal_executable_function_from_index(0), &plan),
      iree::testing::status::StatusIs(iree::StatusCode::kFailedPrecondition));

  iree_hal_executable_release(base_executable);
}

TEST(ExecutableXclbinTest, LinuxCapsDoNotSelectPartialElfContext) {
  std::vector<uint8_t> pdi(8, 0xC0);
  std::vector<uint8_t> xclbin = BuildXclbin({pdi});

  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeXadxExecutable(
      /*xclbins=*/{xclbin},
      /*entry_points=*/
      {{"xadx0",
        /*pdi_index=*/0,
        /*xclbin_index=*/0,
        {{{10, 11}, {}, {0, 0, 4}}}}},
      &executable_data));

  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());

  iree_hal_executable_t* base_executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, /*image_store=*/nullptr, &params,
      iree_allocator_system(), &base_executable));

  iree_hal_amdxdna_native_c_device_caps_t linux_caps = {};
  linux_caps.dispatch_models =
      IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_CU |
      IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU |
      IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_COMMAND_CHAIN;
  iree_hal_amdxdna_dispatch_plan_t linux_plan = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_dispatch_plan_initialize(
      &linux_caps, base_executable, iree_hal_executable_function_from_index(0),
      &linux_plan));
  EXPECT_FALSE(linux_plan.use_native_partial_elf_context);
  EXPECT_TRUE(linux_plan.use_chain_accumulation_policy);

  iree_hal_amdxdna_native_c_device_caps_t partial_elf_caps = {};
  partial_elf_caps.dispatch_models =
      IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_PARTIAL_ELF;
  iree_hal_amdxdna_dispatch_plan_t partial_elf_plan = {};
  IREE_ASSERT_OK(iree_hal_amdxdna_dispatch_plan_initialize(
      &partial_elf_caps, base_executable,
      iree_hal_executable_function_from_index(0), &partial_elf_plan));
  EXPECT_TRUE(partial_elf_plan.use_native_partial_elf_context);

  iree_hal_executable_release(base_executable);
}

TEST(ExecutableXclbinTest, ErtStartNpuPreemptElfUsesPreemptDataPayload) {
  std::vector<uint32_t> packet(32, 0);
  auto* start = reinterpret_cast<ert_start_kernel_cmd*>(packet.data());
  start->state = ERT_CMD_STATE_NEW;
  start->opcode = ERT_START_NPU_PREEMPT_ELF;
  start->type = ERT_CU;
  start->cu_mask = 1;
  start->count = 1 + sizeof(ert_npu_preempt_data) / sizeof(uint32_t) + 2;

  ert_npu_preempt_data* npu_data = get_ert_npu_elf_data(start);
  ASSERT_NE(npu_data, nullptr);
  npu_data->instruction_buffer = 0x100000000ull;
  npu_data->save_buffer = 0x200000000ull;
  npu_data->restore_buffer = 0x300000000ull;
  npu_data->instruction_buffer_size = 0x4000;
  npu_data->save_buffer_size = 0x1000;
  npu_data->restore_buffer_size = 0x1000;
  npu_data->instruction_prop_count = 0;

  uint32_t* regmap = get_ert_regmap_begin(start);
  regmap[0] = 3;
  regmap[1] = 0;

  EXPECT_TRUE(ert_valid_opcode(reinterpret_cast<ert_packet*>(start)));
  EXPECT_EQ(regmap,
            start->data + sizeof(ert_npu_preempt_data) / sizeof(uint32_t));
  EXPECT_EQ(get_ert_npu_preempt_data(start), nullptr);
}

static void ExpectInvalidXadxExecutable(
    const std::vector<std::vector<uint8_t>>& xclbins,
    const std::vector<TestXadxEntryPointDef>& entry_points) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeXadxExecutable(xclbins, entry_points, &executable_data));
  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());
  iree_hal_executable_t* base_executable = nullptr;
  iree_status_t status = iree_hal_amdxdna_native_executable_create(
      /*native_device=*/nullptr, /*image_store=*/nullptr, &params,
      iree_allocator_system(), &base_executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
}

TEST(ExecutableXclbinTest, RejectsXadxXclbinIndexOutOfRange) {
  std::vector<uint8_t> xclbin = BuildXclbin({std::vector<uint8_t>(8, 0x1)});
  ExpectInvalidXadxExecutable(
      {xclbin},
      {{"entry", /*pdi_index=*/0, /*xclbin_index=*/5, {{{10}, {}, {}}}}});
}

TEST(ExecutableXclbinTest, RejectsXadxPdiIndexWithoutXclbinContext) {
  std::vector<uint8_t> xclbin = BuildXclbin({std::vector<uint8_t>(8, 0x1)});
  ExpectInvalidXadxExecutable(
      {xclbin},
      {{"entry", /*pdi_index=*/0, /*xclbin_index=*/-1, {{{10}, {}, {}}}}});
}

TEST(ExecutableXclbinTest, RejectsOverlappingControlParameterRelocations) {
  std::vector<uint8_t> xclbin = BuildXclbin({std::vector<uint8_t>(8, 0x1)});
  ExpectInvalidXadxExecutable(
      {xclbin},
      {{"entry",
        /*pdi_index=*/0,
        /*xclbin_index=*/0,
        {{{0, 0, 0, 0},
          {},
          {},
          {},
          {{/*offset=*/4, /*addend=*/0}, {/*offset=*/8, /*addend=*/0}}}},
        {{"offset", "i32", /*constant_offset=*/0, /*state_table_index=*/0,
          /*byte_length=*/4, /*kind=*/1}}}});
}

TEST(ExecutableXclbinTest, RejectsDuplicateControlParameterNames) {
  std::vector<uint8_t> xclbin = BuildXclbin({std::vector<uint8_t>(8, 0x1)});
  ExpectInvalidXadxExecutable(
      {xclbin},
      {{"entry",
        /*pdi_index=*/0,
        /*xclbin_index=*/0,
        {{{0, 0}, {}, {}}},
        {{"offset", "i32", /*constant_offset=*/0, /*state_table_index=*/0,
          /*byte_length=*/4, /*kind=*/1},
         {"offset", "i32", /*constant_offset=*/4, /*state_table_index=*/1,
          /*byte_length=*/4, /*kind=*/1}}}});
}

}  // namespace
