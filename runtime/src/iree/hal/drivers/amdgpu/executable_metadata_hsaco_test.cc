// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable_metadata_hsaco.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

static const uint8_t kSourceCodeObjectData[] =
    "amdgcn-amd-amdhsa--gfx942\0"
    "test\0"
    "test.kd\0"
    "scale\0"
    "lhs\0"
    "bias\0"
    "rhs\0"
    "implicit\0"
    "implicit.kd\0"
    "buffer\0"
    "value\0"
    "grid_x\0"
    "direct\0"
    "direct.kd\0"
    "global_buffer\0"
    "by_value\0"
    "hidden_global_offset_x\0";

static iree_const_byte_span_t SourceCodeObjectData() {
  return iree_make_const_byte_span(kSourceCodeObjectData,
                                   sizeof(kSourceCodeObjectData));
}

static std::vector<uint8_t> MakeLoadedCodeObjectData() {
  const iree_const_byte_span_t source_code_object_data = SourceCodeObjectData();
  return std::vector<uint8_t>(
      source_code_object_data.data,
      source_code_object_data.data + source_code_object_data.data_length);
}

static iree_const_byte_span_t LoadedCodeObjectData(
    const std::vector<uint8_t>& loaded_code_object_storage) {
  return iree_make_const_byte_span(loaded_code_object_storage.data(),
                                   loaded_code_object_storage.size());
}

static iree_string_view_t ViewFromCodeObjectData(
    iree_const_byte_span_t code_object_data, const char* value) {
  const uint8_t* data = code_object_data.data;
  const iree_host_size_t data_length = code_object_data.data_length;
  const iree_host_size_t value_length = strlen(value);
  iree_host_size_t offset = 0;
  while (offset < data_length) {
    iree_host_size_t end = offset;
    while (end < data_length && data[end] != 0) ++end;
    if (end - offset == value_length &&
        memcmp(data + offset, value, value_length) == 0) {
      return iree_make_string_view((const char*)data + offset, value_length);
    }
    offset = end + 1;
  }
  ADD_FAILURE() << "test code object string not found: " << value;
  return iree_string_view_empty();
}

static bool StringViewBelongsToCodeObjectData(
    iree_const_byte_span_t code_object_data, iree_string_view_t view) {
  const uintptr_t code_object_begin = (uintptr_t)code_object_data.data;
  const uintptr_t view_begin = (uintptr_t)view.data;
  if (iree_string_view_is_empty(view)) return true;
  if (!view.data || view_begin < code_object_begin) return false;
  const uintptr_t view_offset = view_begin - code_object_begin;
  return view_offset <= code_object_data.data_length &&
         view.size <= code_object_data.data_length - view_offset;
}

static void ExpectRebasedView(iree_const_byte_span_t loaded_code_object_data,
                              iree_string_view_t source_view,
                              iree_string_view_t rebased_view) {
  EXPECT_TRUE(iree_string_view_equal(source_view, rebased_view));
  EXPECT_NE(rebased_view.data, source_view.data);
  EXPECT_TRUE(
      StringViewBelongsToCodeObjectData(loaded_code_object_data, rebased_view));
}

static iree_hal_amdgpu_hsaco_metadata_arg_t MakeArg(
    iree_string_view_t name, uint32_t offset, uint32_t size,
    iree_hal_amdgpu_hsaco_metadata_arg_kind_t kind,
    iree_string_view_t value_kind) {
  return iree_hal_amdgpu_hsaco_metadata_arg_t{
      /*.name=*/name,
      /*.offset=*/offset,
      /*.size=*/size,
      /*.alignment=*/size >= 8 ? 8u : 4u,
      /*.kind=*/kind,
      /*.value_kind=*/value_kind,
  };
}

static iree_hal_amdgpu_hsaco_metadata_kernel_t MakeKernel(
    iree_string_view_t name, iree_string_view_t symbol_name,
    uint32_t kernarg_segment_size,
    const std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t>& args) {
  return iree_hal_amdgpu_hsaco_metadata_kernel_t{
      /*.name=*/name,
      /*.symbol_name=*/symbol_name,
      /*.reflection_name=*/name,
      /*.arg_name_storage_size=*/{},
      /*.kernarg_segment_size=*/kernarg_segment_size,
      /*.kernarg_segment_alignment=*/8,
      /*.group_segment_fixed_size=*/16,
      /*.private_segment_fixed_size=*/32,
      /*.required_workgroup_size=*/{},
      /*.has_required_workgroup_size=*/{},
      /*.arg_count=*/args.size(),
      /*.args=*/args.data(),
  };
}

static iree_hal_amdgpu_executable_metadata_t* AllocateAndPopulate(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_const_byte_span_t loaded_code_object_data) {
  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_CHECK_OK(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
      hsaco_metadata, &counts));
  iree_hal_amdgpu_executable_metadata_t* metadata = nullptr;
  IREE_CHECK_OK(iree_hal_amdgpu_executable_metadata_allocate(
      &counts, iree_allocator_system(), &metadata));
  IREE_CHECK_OK(iree_hal_amdgpu_executable_metadata_populate_from_hsaco(
      hsaco_metadata, loaded_code_object_data, metadata));
  return metadata;
}

TEST(ExecutableMetadataHsacoTest, PopulatesSparseInterleavedKernelLayout) {
  const iree_const_byte_span_t source_code_object_data = SourceCodeObjectData();
  std::vector<uint8_t> loaded_code_object_storage = MakeLoadedCodeObjectData();
  const iree_const_byte_span_t loaded_code_object_data =
      LoadedCodeObjectData(loaded_code_object_storage);
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "scale"), 0, 2,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              ViewFromCodeObjectData(source_code_object_data, "by_value")),
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "lhs"), 8, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              ViewFromCodeObjectData(source_code_object_data, "global_buffer")),
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "bias"), 20, 6,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              ViewFromCodeObjectData(source_code_object_data, "by_value")),
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "rhs"), 32, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              ViewFromCodeObjectData(source_code_object_data, "global_buffer")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(ViewFromCodeObjectData(source_code_object_data, "test"),
                 ViewFromCodeObjectData(source_code_object_data, "test.kd"),
                 /*kernarg_segment_size=*/40, args);
  kernel.has_required_workgroup_size = true;
  kernel.required_workgroup_size[0] = 4;
  kernel.required_workgroup_size[1] = 2;
  kernel.required_workgroup_size[2] = 1;
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      /*.host_allocator=*/{},
      /*.elf_data=*/source_code_object_data,
      /*.message_pack_data=*/{},
      /*.target=*/
      ViewFromCodeObjectData(source_code_object_data,
                             "amdgcn-amd-amdhsa--gfx942"),
      /*.reflection_name_storage_size=*/{},
      /*.arg_name_storage_size=*/{},
      /*.kernel_count=*/1,
      /*.kernels=*/&kernel,
  };

  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
      &hsaco_metadata, &counts));
  EXPECT_EQ(counts.export_count, 1);
  EXPECT_EQ(counts.parameter_count, 4);
  EXPECT_GT(counts.layout_blob_byte_length, 0);

  iree_hal_amdgpu_executable_metadata_t* metadata =
      AllocateAndPopulate(&hsaco_metadata, loaded_code_object_data);
  EXPECT_EQ(metadata->source,
            IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_HSACO_MESSAGEPACK);
  EXPECT_EQ(metadata->export_count, 1);
  ExpectRebasedView(loaded_code_object_data, hsaco_metadata.target,
                    metadata->target);
  EXPECT_EQ(metadata->code_object_data.data, loaded_code_object_data.data);
  EXPECT_EQ(metadata->code_object_data.data_length,
            loaded_code_object_data.data_length);

  const iree_hal_amdgpu_executable_export_t& export_info = metadata->exports[0];
  EXPECT_EQ(export_info.flags, IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_NONE);
  EXPECT_EQ(export_info.workgroup_size[0], 4);
  EXPECT_EQ(export_info.workgroup_size[1], 2);
  EXPECT_EQ(export_info.workgroup_size[2], 1);
  EXPECT_EQ(export_info.fixed_group_segment_size, 16);
  EXPECT_EQ(export_info.fixed_private_segment_size, 32);

  const iree_hal_amdgpu_kernarg_layout_t* layout = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_resolve_layout(
      metadata, export_info.kernarg_layout, &layout));
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->kernarg_byte_length, 40);
  EXPECT_EQ(layout->binding_count, 2);
  EXPECT_EQ(layout->constant_span_count, 2);
  EXPECT_EQ(layout->constant_byte_length, 8);
  EXPECT_FALSE(iree_any_bit_set(
      layout->flags,
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_PACKED_BINDING_PREFIX));
  EXPECT_TRUE(iree_all_bits_set(
      layout->flags,
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_REQUIRES_ZERO_FILL |
          IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_CONTIGUOUS_CONSTANTS));

  const iree_hal_amdgpu_kernarg_binding_slot_t* binding_slots =
      iree_hal_amdgpu_kernarg_layout_binding_slots(layout);
  EXPECT_EQ(binding_slots[0].target_qword_index, 1);
  EXPECT_EQ(binding_slots[1].target_qword_index, 4);

  const iree_hal_amdgpu_kernarg_constant_span_t* constant_spans =
      iree_hal_amdgpu_kernarg_layout_constant_spans(layout);
  EXPECT_EQ(constant_spans[0].target_byte_offset, 0);
  EXPECT_EQ(constant_spans[0].source_byte_offset, 0);
  EXPECT_EQ(constant_spans[0].byte_length, 2);
  EXPECT_EQ(constant_spans[1].target_byte_offset, 20);
  EXPECT_EQ(constant_spans[1].source_byte_offset, 2);
  EXPECT_EQ(constant_spans[1].byte_length, 6);

  ExpectRebasedView(loaded_code_object_data, kernel.reflection_name,
                    metadata->reflection[0].name);
  ExpectRebasedView(loaded_code_object_data, kernel.symbol_name,
                    metadata->reflection[0].symbol_name);
  EXPECT_EQ(metadata->reflection[0].parameter_offset, 0);
  EXPECT_EQ(metadata->reflection[0].parameter_count, 4);
  EXPECT_EQ(metadata->parameters[0].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT);
  ExpectRebasedView(loaded_code_object_data, args[0].name,
                    metadata->parameters[0].name);
  EXPECT_EQ(metadata->parameters[0].offset, 0);
  EXPECT_EQ(metadata->parameters[0].size, 2);
  EXPECT_EQ(metadata->parameters[1].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR);
  ExpectRebasedView(loaded_code_object_data, args[1].name,
                    metadata->parameters[1].name);
  EXPECT_EQ(metadata->parameters[1].offset, 8);
  EXPECT_EQ(metadata->parameters[2].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT);
  ExpectRebasedView(loaded_code_object_data, args[2].name,
                    metadata->parameters[2].name);
  EXPECT_EQ(metadata->parameters[2].offset, 2);
  EXPECT_EQ(metadata->parameters[2].size, 6);
  EXPECT_EQ(metadata->parameters[3].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR);
  ExpectRebasedView(loaded_code_object_data, args[3].name,
                    metadata->parameters[3].name);
  EXPECT_EQ(metadata->parameters[3].offset, 32);

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(ExecutableMetadataHsacoTest, PopulatesImplicitArgsSuffixLayout) {
  const iree_const_byte_span_t source_code_object_data = SourceCodeObjectData();
  std::vector<uint8_t> loaded_code_object_storage = MakeLoadedCodeObjectData();
  const iree_const_byte_span_t loaded_code_object_data =
      LoadedCodeObjectData(loaded_code_object_storage);
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "buffer"), 0, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              ViewFromCodeObjectData(source_code_object_data, "global_buffer")),
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "value"), 8, 4,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              ViewFromCodeObjectData(source_code_object_data, "by_value")),
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "grid_x"), 16, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN,
              ViewFromCodeObjectData(source_code_object_data,
                                     "hidden_global_offset_x")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(ViewFromCodeObjectData(source_code_object_data, "implicit"),
                 ViewFromCodeObjectData(source_code_object_data, "implicit.kd"),
                 16 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE, args);
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      /*.host_allocator=*/{},
      /*.elf_data=*/source_code_object_data,
      /*.message_pack_data=*/{},
      /*.target=*/{},
      /*.reflection_name_storage_size=*/{},
      /*.arg_name_storage_size=*/{},
      /*.kernel_count=*/1,
      /*.kernels=*/&kernel,
  };

  iree_hal_amdgpu_executable_metadata_t* metadata =
      AllocateAndPopulate(&hsaco_metadata, loaded_code_object_data);
  EXPECT_TRUE(iree_any_bit_set(
      metadata->exports[0].flags,
      IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_DISPATCH_WORKGROUP_SIZE));

  const iree_hal_amdgpu_kernarg_layout_t* layout = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_resolve_layout(
      metadata, metadata->exports[0].kernarg_layout, &layout));
  EXPECT_EQ(layout->implicit_args_byte_offset, 16);
  EXPECT_TRUE(iree_all_bits_set(
      layout->flags,
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS |
          IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_REQUIRES_ZERO_FILL));
  EXPECT_EQ(layout->binding_count, 1);
  EXPECT_EQ(layout->constant_byte_length, 4);

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(ExecutableMetadataHsacoTest, PopulatesElfOnlyCustomDirectExport) {
  const iree_const_byte_span_t source_code_object_data = SourceCodeObjectData();
  std::vector<uint8_t> loaded_code_object_storage = MakeLoadedCodeObjectData();
  const iree_const_byte_span_t loaded_code_object_data =
      LoadedCodeObjectData(loaded_code_object_storage);
  iree_hal_amdgpu_hsaco_metadata_elf_kernel_symbol_t symbol = {
      /*.name=*/ViewFromCodeObjectData(source_code_object_data, "direct"),
      /*.symbol_name=*/
      ViewFromCodeObjectData(source_code_object_data, "direct.kd"),
  };
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      /*.host_allocator=*/{},
      /*.elf_data=*/source_code_object_data,
      /*.message_pack_data=*/{},
      /*.target=*/{},
      /*.reflection_name_storage_size=*/{},
      /*.arg_name_storage_size=*/{},
      /*.kernel_count=*/{},
      /*.kernels=*/{},
      /*.elf_kernel_symbol_count=*/1,
      /*.elf_kernel_symbols=*/&symbol,
  };

  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
      &hsaco_metadata, &counts));
  EXPECT_EQ(counts.export_count, 1);
  EXPECT_EQ(counts.parameter_count, 0);
  EXPECT_EQ(counts.layout_blob_byte_length, 0);

  iree_hal_amdgpu_executable_metadata_t* metadata =
      AllocateAndPopulate(&hsaco_metadata, loaded_code_object_data);
  EXPECT_EQ(metadata->source,
            IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_ELF_SYMBOLS);
  ExpectRebasedView(loaded_code_object_data, symbol.name,
                    metadata->reflection[0].name);
  ExpectRebasedView(loaded_code_object_data, symbol.symbol_name,
                    metadata->reflection[0].symbol_name);
  EXPECT_TRUE(iree_all_bits_set(
      metadata->exports[0].flags,
      IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_CUSTOM_DIRECT_ONLY |
          IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_DISPATCH_WORKGROUP_SIZE));
  EXPECT_FALSE(iree_hal_amdgpu_kernarg_layout_ref_is_valid(
      metadata->exports[0].kernarg_layout));

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(ExecutableMetadataHsacoTest, RejectsLoadedCodeObjectStringMismatch) {
  const iree_const_byte_span_t source_code_object_data = SourceCodeObjectData();
  std::vector<uint8_t> loaded_code_object_storage = MakeLoadedCodeObjectData();
  loaded_code_object_storage[0] ^= 1;
  const iree_const_byte_span_t loaded_code_object_data =
      LoadedCodeObjectData(loaded_code_object_storage);
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      /*.host_allocator=*/{},
      /*.elf_data=*/source_code_object_data,
      /*.message_pack_data=*/{},
      /*.target=*/
      ViewFromCodeObjectData(source_code_object_data,
                             "amdgcn-amd-amdhsa--gfx942"),
  };

  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
      &hsaco_metadata, &counts));
  iree_hal_amdgpu_executable_metadata_t* metadata = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_allocate(
      &counts, iree_allocator_system(), &metadata));
  EXPECT_THAT(Status(iree_hal_amdgpu_executable_metadata_populate_from_hsaco(
                  &hsaco_metadata, loaded_code_object_data, metadata)),
              StatusIs(StatusCode::kFailedPrecondition));

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(ExecutableMetadataHsacoTest, RejectsUnsupportedVisibleArgumentKind) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("image"), 0, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_IMAGE, IREE_SV("image")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(IREE_SV("bad"), IREE_SV("bad.kd"), 8, args);
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      /*.host_allocator=*/{},
      /*.elf_data=*/{},
      /*.message_pack_data=*/{},
      /*.target=*/{},
      /*.reflection_name_storage_size=*/{},
      /*.arg_name_storage_size=*/{},
      /*.kernel_count=*/1,
      /*.kernels=*/&kernel,
  };
  iree_hal_amdgpu_executable_metadata_counts_t counts;

  EXPECT_THAT(Status(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
                  &hsaco_metadata, &counts)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(ExecutableMetadataHsacoTest, RejectsMisalignedGlobalBufferArgument) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("buffer"), 4, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(IREE_SV("bad"), IREE_SV("bad.kd"), 16, args);
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      /*.host_allocator=*/{},
      /*.elf_data=*/{},
      /*.message_pack_data=*/{},
      /*.target=*/{},
      /*.reflection_name_storage_size=*/{},
      /*.arg_name_storage_size=*/{},
      /*.kernel_count=*/1,
      /*.kernels=*/&kernel,
  };
  iree_hal_amdgpu_executable_metadata_counts_t counts;

  EXPECT_THAT(Status(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
                  &hsaco_metadata, &counts)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(ExecutableMetadataHsacoTest, RejectsHiddenArgsBeforeVisibleArgsEnd) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("buffer"), 0, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
      MakeArg(IREE_SV("grid_x"), 8, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN,
              IREE_SV("hidden_global_offset_x")),
      MakeArg(IREE_SV("value"), 16, 4,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              IREE_SV("by_value")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(IREE_SV("bad"), IREE_SV("bad.kd"), 24, args);
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      /*.host_allocator=*/{},
      /*.elf_data=*/{},
      /*.message_pack_data=*/{},
      /*.target=*/{},
      /*.reflection_name_storage_size=*/{},
      /*.arg_name_storage_size=*/{},
      /*.kernel_count=*/1,
      /*.kernels=*/&kernel,
  };
  iree_hal_amdgpu_executable_metadata_counts_t counts;

  EXPECT_THAT(Status(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
                  &hsaco_metadata, &counts)),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace iree::hal::amdgpu
