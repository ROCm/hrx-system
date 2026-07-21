// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compares production-resolved kernarg images against goldens whose bytes are
// hand-derived from the native ABI (argument offsets from kernel metadata,
// implicit-field offsets from iree_amdgpu_kernel_implicit_args_t) rather than
// captured from the code under test, so a wrong native-ABI offset, a mislocated
// implicit suffix, or a struct-vs-ABI drift fails here instead of on the GPU.

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/executable_metadata.h"
#include "iree/hal/drivers/amdgpu/executable_metadata_hsaco.h"
#include "iree/hal/drivers/amdgpu/kernarg_layout.h"
#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

// Poison byte written into the destination buffer before emitting so that any
// byte an emitter fails to write is visibly distinct from a real zero.
constexpr uint8_t kPoison = 0xFD;

// Total destination buffer size. Larger than the kernel's reservation so the
// bytes past kernarg_byte_length can be asserted untouched (still poison).
constexpr size_t kBufferSize = 256;

// Null-separated string table backing the synthetic metadata argument names and
// value kinds. Reflection rebases these views into loaded code-object storage,
// so they must live in a stable buffer for the duration of the test.
static const uint8_t kSourceCodeObjectData[] =
    "kernel\0"
    "kernel.kd\0"
    "scale\0"
    "lhs\0"
    "bias\0"
    "rhs\0"
    "grid_x\0"
    "global_buffer\0"
    "by_value\0"
    "hidden_global_offset_x\0";

static iree_const_byte_span_t SourceCodeObjectData() {
  return iree_make_const_byte_span(kSourceCodeObjectData,
                                   sizeof(kSourceCodeObjectData));
}

static std::vector<uint8_t> MakeLoadedCodeObjectData() {
  const iree_const_byte_span_t source = SourceCodeObjectData();
  return std::vector<uint8_t>(source.data, source.data + source.data_length);
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

// |kind| is authoritative at this entry point: reflection switches on the
// already-parsed .kind, so |value_kind| is inert here (value_kind string
// parsing into .kind is exercised upstream in the hsaco_metadata tests).
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
      /*.workgroup_cluster_size=*/{},
      /*.has_workgroup_cluster_size=*/{},
      /*.arg_count=*/args.size(),
      /*.args=*/args.data(),
  };
}

// Writes the populated implicit-args fields into the pre-zeroed golden suffix
// at |expected| using literal native-ABI byte offsets (not the struct itself),
// so a drift between the struct and the documented ABI fails the comparison.
// Every unpopulated byte of the fixed window stays zero.
static void WriteExpectedImplicitArgs(uint8_t* expected,
                                      const uint16_t workgroup_size[3],
                                      const uint32_t workgroup_count[3],
                                      uint32_t dynamic_lds) {
  // hidden_block_count_{x,y,z}: uint32_t at byte offsets 0, 4, 8.
  const uint32_t block_count[3] = {workgroup_count[0], workgroup_count[1],
                                   workgroup_count[2]};
  memcpy(expected + 0, &block_count[0], sizeof(uint32_t));
  memcpy(expected + 4, &block_count[1], sizeof(uint32_t));
  memcpy(expected + 8, &block_count[2], sizeof(uint32_t));
  // hidden_group_size_{x,y,z}: uint16_t at byte offsets 12, 14, 16.
  memcpy(expected + 12, &workgroup_size[0], sizeof(uint16_t));
  memcpy(expected + 14, &workgroup_size[1], sizeof(uint16_t));
  memcpy(expected + 16, &workgroup_size[2], sizeof(uint16_t));
  // hidden_grid_dims: uint16_t at byte offset 64, hardcoded to 3.
  const uint16_t grid_dims = 3;
  memcpy(expected + 64, &grid_dims, sizeof(uint16_t));
  // hidden_printf_buffer / hidden_hostcall_buffer: 8-byte pointers at byte
  // offsets 72 and 80, both null (left as the zeros the caller pre-filled).
  // hidden_dynamic_lds_size: uint32_t at byte offset 120.
  memcpy(expected + 120, &dynamic_lds, sizeof(uint32_t));
}

TEST(KernargGoldenTest, ResolvedKernargImageMatchesGolden) {
  const iree_const_byte_span_t source_code_object_data = SourceCodeObjectData();
  std::vector<uint8_t> loaded_code_object_storage = MakeLoadedCodeObjectData();
  const iree_const_byte_span_t loaded_code_object_data =
      iree_make_const_byte_span(loaded_code_object_storage.data(),
                                loaded_code_object_storage.size());

  // Representative kernel: two by-value constants interleaved with two buffer
  // bindings, followed by a hidden argument that anchors the implicit suffix.
  // Offsets are the native-ABI byte offsets the compiled kernel expects.
  const uint32_t kImplicitArgsOffset = 32;
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "scale"),
              /*offset=*/0, /*size=*/4,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              ViewFromCodeObjectData(source_code_object_data, "by_value")),
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "lhs"),
              /*offset=*/8, /*size=*/8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              ViewFromCodeObjectData(source_code_object_data, "global_buffer")),
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "bias"),
              /*offset=*/16, /*size=*/4,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              ViewFromCodeObjectData(source_code_object_data, "by_value")),
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "rhs"),
              /*offset=*/24, /*size=*/8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              ViewFromCodeObjectData(source_code_object_data, "global_buffer")),
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "grid_x"),
              /*offset=*/kImplicitArgsOffset, /*size=*/8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN,
              ViewFromCodeObjectData(source_code_object_data,
                                     "hidden_global_offset_x")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel = MakeKernel(
      ViewFromCodeObjectData(source_code_object_data, "kernel"),
      ViewFromCodeObjectData(source_code_object_data, "kernel.kd"),
      kImplicitArgsOffset + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE, args);
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

  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
      &hsaco_metadata, &counts));
  iree_hal_amdgpu_executable_metadata_t* metadata = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_allocate(
      &counts, iree_allocator_system(), &metadata));
  IREE_EXPECT_OK(iree_hal_amdgpu_executable_metadata_populate_from_hsaco(
      &hsaco_metadata, loaded_code_object_data, metadata));

  // resolve_layout stays a hard assert: |layout| is dereferenced below, so a
  // failure here cannot safely continue. The post-resolve shape checks are soft
  // so the metadata free at the end of the test is always reached under ASAN.
  const iree_hal_amdgpu_kernarg_layout_t* layout = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_resolve_layout(
      metadata, metadata->exports[0].kernarg_layout, &layout));

  // Confirm the reflected layout matches the intended representative shape
  // before trusting the byte-level golden below.
  EXPECT_EQ(layout->binding_count, 2);
  EXPECT_EQ(layout->constant_span_count, 2);
  EXPECT_EQ(layout->constant_byte_length, 8);
  EXPECT_EQ(layout->implicit_args_byte_offset, kImplicitArgsOffset);
  EXPECT_EQ(layout->kernarg_byte_length,
            kImplicitArgsOffset + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE);

  // Buffer pointers and dense constant bytes fed to the emitter.
  const uint64_t binding_ptrs[2] = {
      0x1111222233334444ull,
      0x5555666677778888ull,
  };
  const uint8_t constants_source[8] = {
      0xA1, 0xA2, 0xA3, 0xA4,  // scale (native offset 0)
      0xB1, 0xB2, 0xB3, 0xB4,  // bias (native offset 16)
  };
  const uint16_t workgroup_size[3] = {4, 5, 6};
  const uint32_t workgroup_count[3] = {7, 8, 9};
  const uint32_t dynamic_lds = 13;

  std::vector<uint8_t> kernargs(kBufferSize);
  memset(kernargs.data(), kPoison, kernargs.size());
  // Explicit args are placed by the production host emplace step.
  iree_hal_amdgpu_kernarg_layout_emplace_explicit_args(
      layout, binding_ptrs,
      iree_make_const_byte_span(constants_source, sizeof(constants_source)),
      kernargs.data());
  // The implicit suffix is populated by the production device implicit-args
  // writer. The cross-emitter equivalence test pins the device, AQL
  // command-buffer, and PM4 command-buffer writers byte-for-byte identical; the
  // host-queue native path is a fourth copy that must match the same fill but
  // is covered here only by inspection. group_size is sourced from
  // kernel_args->workgroup_size, matching how the runtime carries it.
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
  kernel_args.workgroup_size[0] = workgroup_size[0];
  kernel_args.workgroup_size[1] = workgroup_size[1];
  kernel_args.workgroup_size[2] = workgroup_size[2];
  const iree_hal_amdgpu_device_dispatch_kernarg_layout_t implicit_layout = {
      /*.explicit_kernarg_size=*/layout->implicit_args_byte_offset,
      /*.implicit_args_offset=*/layout->implicit_args_byte_offset,
      /*.total_kernarg_size=*/layout->kernarg_byte_length,
      /*.has_implicit_args=*/true,
  };
  iree_hal_amdgpu_device_dispatch_emplace_implicit_args(
      &kernel_args, workgroup_count, dynamic_lds, &implicit_layout,
      kernargs.data());

  // Hand-computed golden. This layout carries REQUIRES_ZERO_FILL (implied by
  // IMPLICIT_ARGS), so emplace_explicit_args zeroes the whole reserved region
  // [0, kernarg_byte_length) before placing bindings and constants; the
  // trailing bytes past the reservation stay poison to prove nothing writes
  // beyond the reservation.
  std::vector<uint8_t> expected(kBufferSize);
  memset(expected.data(), kPoison, expected.size());
  memset(expected.data(), 0, layout->kernarg_byte_length);
  // Constants at their native-ABI offsets.
  memcpy(expected.data() + 0, &constants_source[0], 4);   // scale
  memcpy(expected.data() + 16, &constants_source[4], 4);  // bias
  // Buffer pointers at their native-ABI offsets.
  memcpy(expected.data() + 8, &binding_ptrs[0], 8);
  memcpy(expected.data() + 24, &binding_ptrs[1], 8);
  // Implicit suffix.
  WriteExpectedImplicitArgs(expected.data() + kImplicitArgsOffset,
                            workgroup_size, workgroup_count, dynamic_lds);

  if (memcmp(kernargs.data(), expected.data(), kBufferSize) != 0) {
    for (size_t i = 0; i < kBufferSize; ++i) {
      EXPECT_EQ(kernargs[i], expected[i])
          << "kernarg image diverges from golden at byte " << i;
    }
  }

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(KernargGoldenTest, ByValueOnlyKernelHasNoImplicitSuffix) {
  const iree_const_byte_span_t source_code_object_data = SourceCodeObjectData();
  std::vector<uint8_t> loaded_code_object_storage = MakeLoadedCodeObjectData();
  const iree_const_byte_span_t loaded_code_object_data =
      iree_make_const_byte_span(loaded_code_object_storage.data(),
                                loaded_code_object_storage.size());

  // Kernel with only by-value constants and no bindings and no hidden args, so
  // the reflection must not synthesize an implicit suffix nor extend the
  // reservation past the packed constant bytes.
  const uint32_t kKernargSegmentSize = 8;
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "scale"),
              /*offset=*/0, /*size=*/4,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              ViewFromCodeObjectData(source_code_object_data, "by_value")),
      MakeArg(ViewFromCodeObjectData(source_code_object_data, "bias"),
              /*offset=*/4, /*size=*/4,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              ViewFromCodeObjectData(source_code_object_data, "by_value")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(ViewFromCodeObjectData(source_code_object_data, "kernel"),
                 ViewFromCodeObjectData(source_code_object_data, "kernel.kd"),
                 kKernargSegmentSize, args);
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

  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
      &hsaco_metadata, &counts));
  iree_hal_amdgpu_executable_metadata_t* metadata = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_allocate(
      &counts, iree_allocator_system(), &metadata));
  IREE_EXPECT_OK(iree_hal_amdgpu_executable_metadata_populate_from_hsaco(
      &hsaco_metadata, loaded_code_object_data, metadata));

  const iree_hal_amdgpu_kernarg_layout_t* layout = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_resolve_layout(
      metadata, metadata->exports[0].kernarg_layout, &layout));

  // No hidden args: the implicit suffix must stay absent and the reservation
  // must equal the kernel's own kernarg segment size (not the extended
  // implicit-suffix length).
  EXPECT_EQ(layout->implicit_args_byte_offset,
            IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE);
  EXPECT_EQ(layout->kernarg_byte_length, kKernargSegmentSize);
  EXPECT_EQ(layout->binding_count, 0);
  EXPECT_EQ(layout->constant_span_count, 2);
  EXPECT_EQ(layout->constant_byte_length, kKernargSegmentSize);
  EXPECT_FALSE(iree_any_bit_set(
      layout->flags, IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS));

  const uint8_t constants_source[8] = {
      0xC1, 0xC2, 0xC3, 0xC4,  // scale (native offset 0)
      0xD1, 0xD2, 0xD3, 0xD4,  // bias (native offset 4)
  };
  std::vector<uint8_t> kernargs(kBufferSize);
  memset(kernargs.data(), kPoison, kernargs.size());
  iree_hal_amdgpu_kernarg_layout_emplace_explicit_args(
      layout, /*binding_ptrs=*/nullptr,
      iree_make_const_byte_span(constants_source, sizeof(constants_source)),
      kernargs.data());

  // The two contiguous by-value spans pack into [0, 8); no bindings, no
  // implicit suffix, and no zero-fill, so every byte past the segment stays
  // poison.
  std::vector<uint8_t> expected(kBufferSize);
  memset(expected.data(), kPoison, expected.size());
  memcpy(expected.data() + 0, &constants_source[0], 4);  // scale
  memcpy(expected.data() + 4, &constants_source[4], 4);  // bias

  if (memcmp(kernargs.data(), expected.data(), kBufferSize) != 0) {
    for (size_t i = 0; i < kBufferSize; ++i) {
      EXPECT_EQ(kernargs[i], expected[i])
          << "by-value-only kernarg image diverges from golden at byte " << i;
    }
  }

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

}  // namespace
}  // namespace iree::hal::amdgpu
