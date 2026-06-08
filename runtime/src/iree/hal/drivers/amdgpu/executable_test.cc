// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/base/internal/flatcc/building.h"
#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"
#include "iree/schemas/amdgpu_executable_def_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

extern "C" {
iree_status_t iree_hal_amdgpu_executable_initialize_raw_hsaco_export_infos(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    bool* custom_direct_only_exports,
    iree_hal_executable_function_info_t* export_infos,
    iree_host_size_t* export_parameter_offsets,
    iree_hal_executable_function_parameter_t* export_parameters,
    char* export_name_storage, char* export_parameter_name_storage);

iree_status_t iree_hal_amdgpu_executable_raw_hsaco_custom_kernarg_layout(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    iree_host_size_t* out_explicit_kernarg_size,
    uint16_t* out_implicit_args_offset);
}  // extern "C"

namespace iree::hal::amdgpu {
namespace {

static constexpr uint8_t kElfClass64 = 2;
static constexpr uint8_t kElfData2Lsb = 1;
static constexpr uint8_t kElfVersionCurrent = 1;
static constexpr uint8_t kElfOsAbiAmdgpuHsa = 64;
static constexpr uint8_t kElfAbiVersionV5 = 3;
static constexpr uint16_t kElfMachineAmdgpu = 224;
static constexpr uint32_t kElfMachineGfx942 = 0x04c;
static constexpr uint32_t kElfFeatureXnackOffV4 = 0x200;
static constexpr uint32_t kElfFeatureSrameccOnV4 = 0xc00;

static std::array<uint8_t, 64> MakeElf64AmdgpuHsa(uint8_t abi_version,
                                                  uint16_t machine,
                                                  uint32_t e_flags) {
  std::array<uint8_t, 64> elf = {};
  elf[0] = 0x7f;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = kElfClass64;
  elf[5] = kElfData2Lsb;
  elf[6] = kElfVersionCurrent;
  elf[7] = kElfOsAbiAmdgpuHsa;
  elf[8] = abi_version;
  iree_unaligned_store_le_u16((uint16_t*)&elf[18], machine);
  iree_unaligned_store_le_u32((uint32_t*)&elf[20], kElfVersionCurrent);
  iree_unaligned_store_le_u32((uint32_t*)&elf[48], e_flags);
  iree_unaligned_store_le_u16((uint16_t*)&elf[52], (uint16_t)elf.size());
  return elf;
}

static iree_status_t MakeWrappedAmdgpuExecutable(
    iree_string_view_t metadata_target_id, iree_const_byte_span_t code_object,
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
          iree_hal_amdgpu_ExecutableDef_start_as_root(&builder)))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to start AMDGPU executable flatbuffer");
  }

  flatbuffers_string_ref_t isa_ref = 0;
  flatbuffers_string_ref_t image_ref = 0;
  iree_hal_amdgpu_ModuleDef_ref_t module_ref = 0;
  iree_hal_amdgpu_ModuleDef_vec_ref_t modules_ref = 0;
  if (iree_status_is_ok(status)) {
    isa_ref = flatbuffers_string_create(&builder, metadata_target_id.data,
                                        metadata_target_id.size);
    image_ref = flatbuffers_string_create(
        &builder, (const char*)code_object.data, code_object.data_length);
    if (IREE_UNLIKELY(!isa_ref || !image_ref)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create AMDGPU executable strings");
    }
  }
  if (iree_status_is_ok(status)) {
    module_ref = iree_hal_amdgpu_ModuleDef_create(&builder, image_ref);
    modules_ref = iree_hal_amdgpu_ModuleDef_vec_create(&builder, &module_ref,
                                                       /*len=*/1);
    if (IREE_UNLIKELY(!module_ref || !modules_ref)) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "failed to create AMDGPU executable module vector");
    }
  }
  if (iree_status_is_ok(status)) {
    if (IREE_UNLIKELY(
            flatbuffers_failed(
                iree_hal_amdgpu_ExecutableDef_isa_add(&builder, isa_ref)) ||
            flatbuffers_failed(iree_hal_amdgpu_ExecutableDef_modules_add(
                &builder, modules_ref)))) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "failed to populate AMDGPU executable flatbuffer");
    }
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(!iree_hal_amdgpu_ExecutableDef_end_as_root(&builder))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to finish AMDGPU executable flatbuffer");
  }

  size_t flatbuffer_size = 0;
  void* flatbuffer_data = NULL;
  if (iree_status_is_ok(status)) {
    flatbuffer_data =
        flatcc_builder_finalize_aligned_buffer(&builder, &flatbuffer_size);
    if (IREE_UNLIKELY(!flatbuffer_data || flatbuffer_size == 0)) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "failed to finalize AMDGPU executable flatbuffer");
    }
  }

  if (iree_status_is_ok(status)) {
    iree_flatbuffer_file_header_t header = {};
    memcpy(&header.magic, iree_hal_amdgpu_ExecutableDef_file_identifier,
           sizeof(header.magic));
    header.version = 0;
    header.content_size = flatbuffer_size;

    out_executable_data->resize(sizeof(header) + flatbuffer_size);
    memcpy(out_executable_data->data(), &header, sizeof(header));
    memcpy(out_executable_data->data() + sizeof(header), flatbuffer_data,
           flatbuffer_size);
  }

  flatcc_builder_aligned_free(flatbuffer_data);
  flatcc_builder_clear(&builder);
  return status;
}

static std::string InferExecutableFormat(iree_const_byte_span_t executable_data,
                                         iree_host_size_t* out_inferred_size) {
  char executable_format[64] = {};
  IREE_CHECK_OK(iree_hal_amdgpu_executable_infer_format(
      executable_data, sizeof(executable_format), executable_format,
      iree_allocator_system(), out_inferred_size));
  return std::string(executable_format);
}

static iree_hal_amdgpu_device_kernel_args_t MakeKernelArgs(
    uint32_t kernarg_size) {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
  kernel_args.kernarg_size = kernarg_size;
  kernel_args.kernarg_alignment = 8;
  kernel_args.workgroup_size[0] = 1;
  kernel_args.workgroup_size[1] = 1;
  kernel_args.workgroup_size[2] = 1;
  return kernel_args;
}

static iree_hal_amdgpu_hsaco_metadata_arg_t MakeArg(
    iree_hal_amdgpu_hsaco_metadata_arg_kind_t kind, uint32_t offset,
    uint32_t size, iree_string_view_t name = iree_string_view_empty()) {
  iree_hal_amdgpu_hsaco_metadata_arg_t arg = {};
  arg.name = name;
  arg.kind = kind;
  arg.offset = offset;
  arg.size = size;
  return arg;
}

static iree_hal_amdgpu_hsaco_metadata_kernel_t MakeKernel(
    uint32_t kernarg_size, const iree_hal_amdgpu_hsaco_metadata_arg_t* args,
    iree_host_size_t arg_count) {
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel = {};
  kernel.name = IREE_SV("raw");
  kernel.symbol_name = IREE_SV("raw.kd");
  kernel.reflection_name = IREE_SV("raw");
  kernel.kernarg_segment_size = kernarg_size;
  kernel.kernarg_segment_alignment = 8;
  kernel.arg_count = arg_count;
  kernel.args = args;
  return kernel;
}

TEST(ExecutableTest, InfersRawHsacoTargetIdFromElfFlags) {
  const auto elf = MakeElf64AmdgpuHsa(
      kElfAbiVersionV5, kElfMachineAmdgpu,
      kElfMachineGfx942 | kElfFeatureSrameccOnV4 | kElfFeatureXnackOffV4);

  iree_host_size_t inferred_size = 0;
  EXPECT_EQ(
      InferExecutableFormat(iree_make_const_byte_span(elf.data(), elf.size()),
                            &inferred_size),
      "gfx942:sramecc+:xnack-");
  EXPECT_EQ(inferred_size, elf.size());
}

TEST(ExecutableTest, InfersWrappedFlatbufferTargetIdFromEmbeddedElf) {
  const auto elf = MakeElf64AmdgpuHsa(
      kElfAbiVersionV5, kElfMachineAmdgpu,
      kElfMachineGfx942 | kElfFeatureSrameccOnV4 | kElfFeatureXnackOffV4);
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeWrappedAmdgpuExecutable(
      IREE_SV("gfx1100"), iree_make_const_byte_span(elf.data(), elf.size()),
      &executable_data));

  iree_host_size_t inferred_size = 0;
  EXPECT_EQ(
      InferExecutableFormat(iree_make_const_byte_span(executable_data.data(),
                                                      executable_data.size()),
                            &inferred_size),
      "gfx942:sramecc+:xnack-");
  EXPECT_EQ(inferred_size, executable_data.size());
}

TEST(ExecutableTest, RawHsacoHalAbiMetadataExportsUseNormalDispatch) {
  const iree_hal_amdgpu_hsaco_metadata_arg_t args[] = {
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              /*offset=*/0, /*size=*/8, IREE_SV("input")),
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE, /*offset=*/8,
              /*size=*/4, IREE_SV("scale")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_size=*/12, args, IREE_ARRAYSIZE(args));
  iree_hal_amdgpu_hsaco_metadata_t metadata = {};
  metadata.kernel_count = 1;
  metadata.kernels = &kernel;

  bool custom_direct_only_exports[1] = {};
  iree_hal_executable_function_info_t export_infos[1] = {};
  iree_host_size_t export_parameter_offsets[2] = {};
  iree_hal_executable_function_parameter_t export_parameters[2] = {};
  char export_name_storage[16] = {};
  char export_parameter_name_storage[16] = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_initialize_raw_hsaco_export_infos(
      &metadata, custom_direct_only_exports, export_infos,
      export_parameter_offsets, export_parameters, export_name_storage,
      export_parameter_name_storage));

  EXPECT_FALSE(custom_direct_only_exports[0]);
  EXPECT_EQ(export_infos[0].flags, IREE_HAL_EXECUTABLE_FUNCTION_FLAG_NONE);
  EXPECT_EQ(export_infos[0].parameter_count, 2u);
  EXPECT_EQ(export_infos[0].binding_count, 1u);
  EXPECT_EQ(export_infos[0].constant_count, 1u);
  EXPECT_EQ(export_parameter_offsets[0], 0u);
  EXPECT_EQ(export_parameter_offsets[1], 2u);
  EXPECT_EQ(export_parameters[0].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING);
  EXPECT_EQ(export_parameters[0].offset, 0u);
  EXPECT_EQ(export_parameters[1].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT);
  EXPECT_EQ(export_parameters[1].offset, 0u);
}

TEST(ExecutableTest, RawHsacoNativeMetadataExportsRemainCustomDirectOnly) {
  const iree_hal_amdgpu_hsaco_metadata_arg_t args[] = {
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE, /*offset=*/8,
              /*size=*/4, IREE_SV("x")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_size=*/12, args, IREE_ARRAYSIZE(args));
  iree_hal_amdgpu_hsaco_metadata_t metadata = {};
  metadata.kernel_count = 1;
  metadata.kernels = &kernel;

  bool custom_direct_only_exports[1] = {};
  iree_hal_executable_function_info_t export_infos[1] = {};
  iree_host_size_t export_parameter_offsets[2] = {};
  iree_hal_executable_function_parameter_t export_parameters[1] = {};
  char export_name_storage[16] = {};
  char export_parameter_name_storage[16] = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_initialize_raw_hsaco_export_infos(
      &metadata, custom_direct_only_exports, export_infos,
      export_parameter_offsets, export_parameters, export_name_storage,
      export_parameter_name_storage));

  EXPECT_TRUE(custom_direct_only_exports[0]);
  EXPECT_EQ(export_infos[0].parameter_count, 1u);
  EXPECT_EQ(export_infos[0].constant_count, 3u);
  EXPECT_EQ(export_parameters[0].offset, 8u);
}

TEST(ExecutableTest, RawHsacoCustomKernargLayoutRequiresVisibleExtent) {
  const iree_hal_amdgpu_hsaco_metadata_arg_t args[] = {
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              /*offset=*/0, /*size=*/8),
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              /*offset=*/16, /*size=*/4),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_size=*/20, args, IREE_ARRAYSIZE(args));
  iree_hal_amdgpu_device_kernel_args_t kernel_args =
      MakeKernelArgs(/*kernarg_size=*/20);

  iree_host_size_t explicit_kernarg_size = 0;
  uint16_t implicit_args_offset = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_raw_hsaco_custom_kernarg_layout(
      &kernel, &kernel_args, &explicit_kernarg_size, &implicit_args_offset));
  EXPECT_EQ(explicit_kernarg_size, 20u);
  EXPECT_EQ(implicit_args_offset, UINT16_MAX);
}

TEST(ExecutableTest, RawHsacoCustomKernargLayoutRejectsInterleavedHiddenArgs) {
  const iree_hal_amdgpu_hsaco_metadata_arg_t args[] = {
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              /*offset=*/0, /*size=*/8),
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN,
              /*offset=*/8, /*size=*/8),
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              /*offset=*/16, /*size=*/4),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_size=*/8 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE,
                 args, IREE_ARRAYSIZE(args));
  iree_hal_amdgpu_device_kernel_args_t kernel_args = MakeKernelArgs(
      /*kernarg_size=*/8 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE);

  iree_host_size_t explicit_kernarg_size = 0;
  uint16_t implicit_args_offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_executable_raw_hsaco_custom_kernarg_layout(
          &kernel, &kernel_args, &explicit_kernarg_size,
          &implicit_args_offset));
}

TEST(ExecutableTest, RawHsacoCustomKernargLayoutAcceptsFullImplicitSuffix) {
  const iree_hal_amdgpu_hsaco_metadata_arg_t args[] = {
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              /*offset=*/0, /*size=*/8),
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN,
              /*offset=*/16, /*size=*/8),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_size=*/16 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE,
                 args, IREE_ARRAYSIZE(args));
  iree_hal_amdgpu_device_kernel_args_t kernel_args = MakeKernelArgs(
      /*kernarg_size=*/16 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE);

  iree_host_size_t explicit_kernarg_size = 0;
  uint16_t implicit_args_offset = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_raw_hsaco_custom_kernarg_layout(
      &kernel, &kernel_args, &explicit_kernarg_size, &implicit_args_offset));
  EXPECT_EQ(explicit_kernarg_size, 8u);
  EXPECT_EQ(implicit_args_offset, 16u);
}

TEST(ExecutableTest,
     RawHsacoCustomKernargLayoutRejectsTruncatedImplicitSuffix) {
  const iree_hal_amdgpu_hsaco_metadata_arg_t args[] = {
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              /*offset=*/0, /*size=*/8),
      MakeArg(IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN,
              /*offset=*/16, /*size=*/8),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_size=*/16, args, IREE_ARRAYSIZE(args));
  iree_hal_amdgpu_device_kernel_args_t kernel_args =
      MakeKernelArgs(/*kernarg_size=*/16);

  iree_host_size_t explicit_kernarg_size = 0;
  uint16_t implicit_args_offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_executable_raw_hsaco_custom_kernarg_layout(
          &kernel, &kernel_args, &explicit_kernarg_size,
          &implicit_args_offset));
}

}  // namespace
}  // namespace iree::hal::amdgpu
