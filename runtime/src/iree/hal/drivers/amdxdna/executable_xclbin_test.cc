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
#include "iree/hal/drivers/amdxdna/shim/ert.h"
#include "iree/schemas/amdxdna_xclbin_executable_def_builder.h"
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
};

static iree_status_t MakeXadxExecutable(
    const std::vector<std::vector<uint8_t>>& xclbins,
    const std::vector<TestXadxEntryPointDef>& entry_points,
    std::vector<uint8_t>* out_executable_data) {
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

  std::vector<iree_hal_amdxdna_xclbin_EntryPointDef_ref_t> entry_refs;
  if (iree_status_is_ok(status)) {
    for (const auto& ep : entry_points) {
      std::vector<iree_hal_amdxdna_xclbin_RunDef_ref_t> run_refs;
      for (const auto& run : ep.runs) {
        iree_hal_amdxdna_xclbin_RunDef_ref_t run_ref =
            iree_hal_amdxdna_xclbin_RunDef_create(
                &builder, CreateUInt32Vec(&builder, run.control_code),
                CreateUInt32Vec(&builder, run.data_payload),
                CreateUInt32Vec(&builder, run.patch_table));
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
              &builder, runs_ref))) {
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
    iree_hal_amdxdna_xclbin_EntryPointDef_vec_ref_t entries_ref =
        iree_hal_amdxdna_xclbin_EntryPointDef_vec_create(
            &builder, entry_refs.data(), entry_refs.size());
    if (!xclbins_ref || !entries_ref ||
        flatbuffers_failed(iree_hal_amdxdna_xclbin_ExecutableDef_xclbins_add(
            &builder, xclbins_ref)) ||
        flatbuffers_failed(
            iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_add(
                &builder, entries_ref))) {
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
      /*native_device=*/nullptr, &params, iree_allocator_system(),
      &base_executable));
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);

  ASSERT_EQ(executable->entry_point_count, 2u);
  const auto& e0 = executable->entry_points[0];
  EXPECT_EQ(ToString(e0.kernel_name), "xadx0");
  EXPECT_EQ(ToVector(e0.pdi),
            pdi0);  // extracted from the AIE_PARTITION by index
  EXPECT_EQ(ToVector(e0.xclbin), xclbin);  // raw AXLF context wrapper retained
  ASSERT_EQ(e0.asm_inst_runlist_count, 1u);
  EXPECT_EQ(ToVector(e0.asm_inst_runlist[0]), std::vector<uint32_t>({10, 11}));
  const auto& e1 = executable->entry_points[1];
  EXPECT_EQ(ToString(e1.kernel_name), "xadx1");
  EXPECT_EQ(ToVector(e1.pdi), pdi1);
  ASSERT_EQ(e1.asm_inst_runlist_count, 1u);
  EXPECT_EQ(ToVector(e1.asm_inst_runlist[0]), std::vector<uint32_t>({20}));

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
      /*native_device=*/nullptr, &params, iree_allocator_system(),
      &base_executable));

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
  start->count =
      1 + sizeof(ert_npu_preempt_data) / sizeof(uint32_t) + 2;

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
  EXPECT_EQ(regmap, start->data +
                        sizeof(ert_npu_preempt_data) / sizeof(uint32_t));
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
      /*native_device=*/nullptr, &params, iree_allocator_system(),
      &base_executable);
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

}  // namespace
