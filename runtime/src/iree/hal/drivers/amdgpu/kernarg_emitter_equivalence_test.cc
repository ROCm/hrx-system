// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pins byte-for-byte agreement, across the full reserved kernarg region
// including the undeclared implicit-window bytes, between every custom-direct
// kernarg/implicit-args emitter that must produce identical images for the same
// kernel and arguments: the device path (also used verbatim by the host-queue
// custom-direct path), the AQL command-buffer tail writer, and the PM4
// command-buffer implicit-args writer. Any divergence in implicit-window
// zeroing (e.g. zeroing the struct instead of the reserved suffix) is caught
// here rather than as a poisoned GPU dispatch.

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

// Poison byte written into both destination buffers before emitting so that any
// byte an emitter fails to write is visibly distinct from a real zero.
constexpr uint8_t kPoison = 0xFD;

// Total destination buffer size. Larger than any reservation used below so the
// bytes past total_kernarg_size can be asserted untouched (still poison).
constexpr size_t kBufferSize = 256;

static iree_hal_amdgpu_device_kernel_args_t MakeKernelArgs(
    uint16_t kernarg_size, uint16_t alignment) {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
  kernel_args.kernel_object = 0x1234u;
  kernel_args.setup = 3;
  kernel_args.workgroup_size[0] = 4;
  kernel_args.workgroup_size[1] = 5;
  kernel_args.workgroup_size[2] = 6;
  kernel_args.kernarg_size = kernarg_size;
  kernel_args.kernarg_alignment = alignment;
  return kernel_args;
}

// Runs the device custom-direct pair (copy + implicit) into |out|.
static void EmitDevice(
    const iree_hal_amdgpu_device_kernel_args_t& kernel_args,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t& layout,
    const uint32_t workgroup_count[3], uint32_t dynamic_lds,
    const uint8_t* blob, size_t blob_length, uint8_t* out) {
  iree_hal_amdgpu_device_dispatch_emplace_custom_kernargs(&layout, blob,
                                                          blob_length, out);
  iree_hal_amdgpu_device_dispatch_emplace_implicit_args(
      &kernel_args, workgroup_count, dynamic_lds, &layout, out);
}

// Runs the AQL command-buffer custom-direct tail writer into |out|.
static iree_status_t EmitAql(
    const iree_hal_amdgpu_device_kernel_args_t& kernel_args,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t& layout,
    const uint32_t workgroup_count[3], uint32_t dynamic_lds,
    const uint8_t* blob, size_t blob_length, uint8_t* out) {
  iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
      workgroup_count[0], workgroup_count[1], workgroup_count[2]);
  config.dynamic_workgroup_local_memory = dynamic_lds;
  return iree_hal_amdgpu_aql_command_buffer_emplace_custom_direct_tail(
      &kernel_args, &layout, config,
      iree_make_const_byte_span(blob, blob_length), out);
}

// Runs the PM4 command-buffer implicit-args writer into |out|. PM4 does not
// expose a pure tail writer (its template writer needs live command-buffer
// state), so the shared device explicit copy provides the substrate and the PM4
// implicit writer fills the suffix; this isolates PM4's implicit-window writer,
// which is the byte-level contract under test here. When the layout has no
// implicit suffix PM4 writes nothing past the copy, matching the device path.
static void EmitPm4(
    const iree_hal_amdgpu_device_kernel_args_t& kernel_args,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t& layout,
    const uint32_t workgroup_count[3], uint32_t dynamic_lds,
    const uint8_t* blob, size_t blob_length, uint8_t* out) {
  iree_hal_amdgpu_device_dispatch_emplace_custom_kernargs(&layout, blob,
                                                          blob_length, out);
  if (!layout.has_implicit_args) return;
  iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
      workgroup_count[0], workgroup_count[1], workgroup_count[2]);
  config.dynamic_workgroup_local_memory = dynamic_lds;
  iree_hal_amdgpu_pm4_command_buffer_emplace_implicit_args(
      &kernel_args, config,
      reinterpret_cast<iree_amdgpu_kernel_implicit_args_t*>(
          out + layout.implicit_args_offset));
}

// Emits the same input through every path and asserts full-buffer equality.
// Returns the device buffer so callers can make additional byte-level
// assertions (e.g. that no byte past total_kernarg_size was written).
static std::array<uint8_t, kBufferSize> ExpectEmittersAgree(
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t& layout,
    const uint8_t* blob, size_t blob_length) {
  // The kernarg segment size is a multiple of 16 per the kernel ABI; round the
  // reservation up so the fixture models a real kernel even though the emitters
  // do not read this field.
  const uint16_t kernarg_size =
      static_cast<uint16_t>((layout.total_kernarg_size + 15) & ~size_t{15});
  const iree_hal_amdgpu_device_kernel_args_t kernel_args =
      MakeKernelArgs(kernarg_size, /*alignment=*/16);
  const uint32_t workgroup_count[3] = {7, 8, 9};
  const uint32_t dynamic_lds = 13;

  alignas(64) std::array<uint8_t, kBufferSize> device_buffer;
  alignas(64) std::array<uint8_t, kBufferSize> aql_buffer;
  alignas(64) std::array<uint8_t, kBufferSize> pm4_buffer;
  device_buffer.fill(kPoison);
  aql_buffer.fill(kPoison);
  pm4_buffer.fill(kPoison);

  EmitDevice(kernel_args, layout, workgroup_count, dynamic_lds, blob,
             blob_length, device_buffer.data());
  IREE_EXPECT_OK(EmitAql(kernel_args, layout, workgroup_count, dynamic_lds,
                         blob, blob_length, aql_buffer.data()));
  EmitPm4(kernel_args, layout, workgroup_count, dynamic_lds, blob, blob_length,
          pm4_buffer.data());

  for (size_t i = 0; i < kBufferSize; ++i) {
    EXPECT_EQ(device_buffer[i], aql_buffer[i])
        << "device and AQL emitters diverge at byte " << i;
    EXPECT_EQ(device_buffer[i], pm4_buffer[i])
        << "device and PM4 emitters diverge at byte " << i;
  }
  return device_buffer;
}

TEST(KernargEmitterEquivalenceTest, CustomDirectWithImplicitArgsAgree) {
  // Explicit blob fully fills the region before the implicit suffix.
  constexpr size_t kImplicitArgsOffset = 16;
  const iree_hal_amdgpu_device_dispatch_kernarg_layout_t layout = {
      /*.explicit_kernarg_size=*/kImplicitArgsOffset,
      /*.implicit_args_offset=*/kImplicitArgsOffset,
      /*.total_kernarg_size=*/kImplicitArgsOffset +
          IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE,
      /*.has_implicit_args=*/true,
  };
  const std::array<uint8_t, kImplicitArgsOffset> blob = {
      0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
      0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
  };
  ExpectEmittersAgree(layout, blob.data(), blob.size());
}

TEST(KernargEmitterEquivalenceTest, CustomDirectWithoutImplicitArgsAgree) {
  // No implicit suffix: every path performs a clamped verbatim copy only.
  const iree_hal_amdgpu_device_dispatch_kernarg_layout_t layout = {
      /*.explicit_kernarg_size=*/24,
      /*.implicit_args_offset=*/0,
      /*.total_kernarg_size=*/24,
      /*.has_implicit_args=*/false,
  };
  std::array<uint8_t, 24> blob = {};
  for (size_t i = 0; i < blob.size(); ++i) {
    blob[i] = static_cast<uint8_t>(0x40u + i);
  }
  ExpectEmittersAgree(layout, blob.data(), blob.size());
}

TEST(KernargEmitterEquivalenceTest, CustomDirectShortBlobZeroesGapAgree) {
  // The blob is shorter than the implicit offset so every path must zero the
  // gap between the copied prefix and the implicit suffix identically.
  constexpr size_t kImplicitArgsOffset = 32;
  const iree_hal_amdgpu_device_dispatch_kernarg_layout_t layout = {
      /*.explicit_kernarg_size=*/0,
      /*.implicit_args_offset=*/kImplicitArgsOffset,
      /*.total_kernarg_size=*/kImplicitArgsOffset +
          IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE,
      /*.has_implicit_args=*/true,
  };
  const std::array<uint8_t, 8> blob = {
      0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
  };
  ExpectEmittersAgree(layout, blob.data(), blob.size());
}

TEST(KernargEmitterEquivalenceTest, CustomDirectOverLongBlobClampsAndAgree) {
  // The caller-supplied blob is longer than the reservation. Every emitter must
  // clamp the copy at total_kernarg_size and leave the trailing poison bytes
  // untouched, so an unclamped copy (or a longer memcpy) is caught here.
  constexpr size_t kTotalKernargSize = 40;
  const iree_hal_amdgpu_device_dispatch_kernarg_layout_t layout = {
      /*.explicit_kernarg_size=*/kTotalKernargSize,
      /*.implicit_args_offset=*/0,
      /*.total_kernarg_size=*/kTotalKernargSize,
      /*.has_implicit_args=*/false,
  };
  std::array<uint8_t, 64> blob = {};
  for (size_t i = 0; i < blob.size(); ++i) {
    blob[i] = static_cast<uint8_t>(0x50u + i);
  }
  const std::array<uint8_t, kBufferSize> result =
      ExpectEmittersAgree(layout, blob.data(), blob.size());
  for (size_t i = kTotalKernargSize; i < kBufferSize; ++i) {
    EXPECT_EQ(result[i], kPoison)
        << "over-long blob copy wrote past total_kernarg_size at byte " << i;
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
