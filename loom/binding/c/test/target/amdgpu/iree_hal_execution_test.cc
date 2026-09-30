// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test/target/iree_hal_execution.h"

#include <array>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "test/util.h"

namespace {

constexpr char kSourceText[] = R"(
kernel.def @double_i32_at_byte_offset() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%input: buffer, %output: buffer, %byte_offset: offset) {
  %byte_offset_aligned = index.assume %byte_offset [mul(%byte_offset, 4)] : offset
  %input_aligned = buffer.assume.alignment %input {minimum_alignment = 4} : buffer
  %output_aligned = buffer.assume.alignment %output {minimum_alignment = 4} : buffer
  %input_view = buffer.view %input_aligned[%byte_offset_aligned] : buffer -> view<1xi32>
  %loaded = view.load %input_view[0] : view<1xi32> -> i32
  %doubled = scalar.addi %loaded, %loaded : i32
  %output_view = buffer.view %output_aligned[%byte_offset_aligned] : buffer -> view<1xi32>
  view.store %doubled, %output_view[0] : i32, view<1xi32>
  kernel.return
}
)";

// Parsing resolves every Low representation contract before template selection.
// Include providers only for compiled descriptor sets, with exact target
// witnesses matching the live HAL profile. A generic-family contract need not
// be compiled into a binary that supports only one physical processor.
#define AMDGPU_SPILL_OFFSET_PROVIDER(symbol, processor, representation)        \
  "amdgpu.target<" processor "> @" #symbol                                     \
  "\n"                                                                         \
  "low.func.def target<" representation ">(@" #symbol ") @" #symbol            \
  "_spill(%value: reg<amdgpu.vgpr x2>) -> (reg<amdgpu.vgpr x2>) asm {\n"       \
  "  %storage = storage {byte_alignment = 8, byte_length = 8} : "              \
  "low.storage<private>\n"                                                     \
  "  low.spill %value, %storage : reg<amdgpu.vgpr x2>, low.storage<private>\n" \
  "  %reloaded = low.reload %storage : low.storage<private> -> "               \
  "reg<amdgpu.vgpr x2>\n"                                                      \
  "  return %reloaded\n"                                                       \
  "}\n"                                                                        \
  "template.def<@spill_offset> target(@" #symbol ") @" #symbol                 \
  "_spill_provider(%words: vector<2xi32>) -> (vector<2xi32>) {\n"              \
  "  %reloaded = low.invoke @" #symbol                                         \
  "_spill(%words) : (vector<2xi32>) -> (vector<2xi32>)\n"                      \
  "  template.return %reloaded : vector<2xi32>\n"                              \
  "}\n"

// Keep these concatenated source literals aligned as a target table.
// clang-format off
constexpr char kWideScaledSourceText[] =
    "template.decl @spill_offset(%words: vector<2xi32>) -> (vector<2xi32>)\n"
#if defined(LOOM_AMDGPU_DESCRIPTOR_SET_CDNA3_CORE)
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx940, "gfx940",
                                "amdgpu.cdna3.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx941, "gfx941",
                                "amdgpu.cdna3.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx942, "gfx942",
                                "amdgpu.cdna3.core")
#endif
#if defined(LOOM_AMDGPU_DESCRIPTOR_SET_CDNA4_CORE)
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx950, "gfx950",
                                "amdgpu.cdna4.core")
#endif
#if defined(LOOM_AMDGPU_DESCRIPTOR_SET_RDNA3_CORE)
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1100, "gfx1100",
                                "amdgpu.rdna3.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1101, "gfx1101",
                                "amdgpu.rdna3.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1102, "gfx1102",
                                "amdgpu.rdna3.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1103, "gfx1103",
                                "amdgpu.rdna3.core")
#endif
#if defined(LOOM_AMDGPU_DESCRIPTOR_SET_RDNA3_5_CORE)
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1150, "gfx1150",
                                "amdgpu.rdna3_5.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1151, "gfx1151",
                                "amdgpu.rdna3_5.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1152, "gfx1152",
                                "amdgpu.rdna3_5.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1153, "gfx1153",
                                "amdgpu.rdna3_5.core")
#endif
#if defined(LOOM_AMDGPU_DESCRIPTOR_SET_RDNA4M_CORE)
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1170, "gfx1170",
                                "amdgpu.rdna4m.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1171, "gfx1171",
                                "amdgpu.rdna4m.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1172, "gfx1172",
                                "amdgpu.rdna4m.core")
#endif
#if defined(LOOM_AMDGPU_DESCRIPTOR_SET_RDNA4_CORE)
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1200, "gfx1200",
                                "amdgpu.rdna4.core")
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1201, "gfx1201",
                                "amdgpu.rdna4.core")
#endif
#if defined(LOOM_AMDGPU_DESCRIPTOR_SET_RDNA4_GFX125X_CORE)
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1250, "gfx1250",
                                "amdgpu.rdna4.gfx125x.core")
#endif
#if defined(LOOM_AMDGPU_DESCRIPTOR_SET_RDNA4_GFX1250_A0_CORE)
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1250_a0, "gfx1250-strict",
                                "amdgpu.rdna4.gfx1250_a0.core")
#endif
#if defined(LOOM_AMDGPU_DESCRIPTOR_SET_RDNA4_GFX1251_CORE)
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1251, "gfx1251",
                                "amdgpu.rdna4.gfx1251.core")
#endif
    R"(
kernel.def @double_i32_at_scaled_offset() {
  %unit = index.constant 1 : index
  %lanes = index.constant 2 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%lanes, %unit, %unit) : index
} launch(%input: buffer, %output: buffer, %base: i32) {
  %input_aligned = buffer.assume.alignment %input {minimum_alignment = 4} : buffer
  %output_aligned = buffer.assume.alignment %output {minimum_alignment = 4} : buffer
  %lane = kernel.workitem.id<x> : index
  %lane_bits = index.cast %lane : index to i32
  %wrapped = scalar.addi %base, %lane_bits : i32
  %unsigned = index.cast %wrapped : i32 to offset
  %element = index.cast %unsigned : offset to index
  %stride = index.constant 4 : offset
  %scaled_offset = index.scale %element, %stride : index, offset -> offset
  %offset_bits = index.cast %scaled_offset : offset to i64
  %offset_value = vector.splat %offset_bits : vector<1xi64>
  %offset_words = vector.bitcast %offset_value : vector<1xi64> to vector<2xi32>
  %reloaded_words = template.apply<@spill_offset>(%offset_words) : (vector<2xi32>) -> (vector<2xi32>)
  %reloaded_value = vector.bitcast %reloaded_words : vector<2xi32> to vector<1xi64>
  %reloaded_bits = vector.extract %reloaded_value[0] : vector<1xi64> -> i64
  // The opaque spill provider preserves the scaled unsigned i32 payload.
  %address_bits = scalar.assume %reloaded_bits [range(%reloaded_bits, 0, 17179869180)] : i64
  %byte_offset = index.cast %address_bits : i64 to offset
  %input_view = buffer.view %input_aligned[%byte_offset] : buffer -> view<1xi32>
  %loaded = view.load %input_view[0] : view<1xi32> -> i32
  %doubled = scalar.addi %loaded, %loaded : i32
  %output_view = buffer.view %output_aligned[%byte_offset] : buffer -> view<1xi32>
  view.store %doubled, %output_view[0] : i32, view<1xi32>
  kernel.return
}
)";
// clang-format on

#undef AMDGPU_SPILL_OFFSET_PROVIDER

loomc_status_t CreateAmdgpuTargetEnvironment(
    loomc_allocator_t host_allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_amdgpu(host_allocator,
                                                out_target_environment);
}

loomc_status_t ValidateAmdgpuProfile(loomc_target_profile_t* target_profile,
                                     const char** out_skip_reason) {
  *out_skip_reason = nullptr;
  loomc_amdgpu_target_identity_t identity = {};
  return loomc_amdgpu_target_profile_query_identity(target_profile, &identity);
}

loomc_status_t EmitAmdgpuModule(loomc_target_environment_t* target_environment,
                                loomc_workspace_t* workspace,
                                loomc_module_t* module,
                                loomc_string_view_t artifact_format,
                                loomc_string_view_t artifact_identifier,
                                loomc_result_t** out_result) {
  const loomc_amdgpu_emit_options_t amdgpu_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(amdgpu_options),
      /*.next=*/nullptr,
      /*.runtime_globals=*/LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
  };
  const loomc_emit_options_t emit_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(emit_options),
      /*.next=*/&amdgpu_options,
      /*.artifact_format=*/artifact_format,
      /*.identifier=*/artifact_identifier,
      /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  return loomc_emit_module(target_environment, workspace, module, &emit_options,
                           loomc_allocator_system(), out_result);
}

loomc::testing::target::IreeHalKernelExecutionTarget MakeExecutionTarget(
    const char* source_text, const char* kernel_export_name) {
  static const loomc_iree_hal_profile_provider_t* const profile_providers[] = {
      loomc_amdgpu_iree_hal_profile_provider(),
  };

  loomc::testing::target::IreeHalKernelExecutionTarget target = {};
  target.label = "AMDGPU";
  target.device_uri = IREE_SV("amdgpu");
  target.target_profile_identifier = loomc_make_cstring_view("live-amdgpu");
  target.source_identifier = loomc_make_cstring_view("live_amdgpu.loom");
  target.source_text = loomc_make_cstring_view(source_text);
  target.module_name = loomc_make_cstring_view("live_amdgpu_execution_test");
  target.kernel_export_name = loomc_make_cstring_view(kernel_export_name);
  target.target_pipeline_identifier =
      loomc_make_cstring_view("live-amdgpu-prepared-low");
  target.target_pipeline_kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW;
  target.control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
  target.source_to_low_max_errors = 20;
  target.artifact_format =
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
  target.artifact_identifier = loomc_make_cstring_view("live_amdgpu.hsaco");
  target.executable_target_selection = {
      /*.family=*/IREE_SV("amdgpu"),
      /*.target_key=*/iree_string_view_empty(),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      /*.physical_device_affinity=*/0,
  };
  target.profile_providers = profile_providers;
  target.profile_provider_count = 1;
  target.create_target_environment = CreateAmdgpuTargetEnvironment;
  target.validate_target_profile = ValidateAmdgpuProfile;
  target.emit_module = EmitAmdgpuModule;
  return target;
}

// Backs the first page and the page at 4 GiB with distinct physical memory.
// A truncated address remains mapped and produces an observable wrong result.
class SparseTestBuffer {
 public:
  static constexpr iree_device_size_t kHighPageOffset = uint64_t{1} << 32;

  SparseTestBuffer(iree_hal_allocator_t* allocator,
                   iree_device_size_t page_size)
      : allocator_(allocator), page_size_(page_size) {}
  SparseTestBuffer(const SparseTestBuffer&) = delete;
  SparseTestBuffer& operator=(const SparseTestBuffer&) = delete;

  ~SparseTestBuffer() {
    for (iree_host_size_t i = 0; i < mapped_page_count_; ++i) {
      IREE_EXPECT_OK(iree_hal_allocator_virtual_memory_unmap(
          allocator_, buffer_, i * kHighPageOffset, page_size_));
    }
    for (auto* physical_memory : physical_memory_) {
      if (physical_memory) {
        IREE_EXPECT_OK(iree_hal_allocator_physical_memory_free(
            allocator_, physical_memory));
      }
    }
    if (buffer_) {
      IREE_EXPECT_OK(
          iree_hal_allocator_virtual_memory_release(allocator_, buffer_));
    }
  }

  iree_status_t Initialize(iree_hal_buffer_params_t params) {
    IREE_RETURN_IF_ERROR(iree_hal_allocator_virtual_memory_reserve(
        allocator_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        kHighPageOffset + page_size_, &buffer_));
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
      status = iree_hal_allocator_physical_memory_allocate(
          allocator_, params, page_size_, iree_allocator_system(),
          &physical_memory_[i]);
      if (iree_status_is_ok(status)) {
        status = iree_hal_allocator_virtual_memory_map(
            allocator_, buffer_, i * kHighPageOffset, physical_memory_[i],
            /*physical_offset=*/0, page_size_);
      }
      if (iree_status_is_ok(status)) {
        ++mapped_page_count_;
        status = iree_hal_allocator_virtual_memory_protect(
            allocator_, buffer_, i * kHighPageOffset, page_size_,
            IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
            IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
            IREE_HAL_MEMORY_PROTECTION_READ_WRITE);
      }
    }
    return status;
  }

  iree_hal_buffer_t* get() const { return buffer_; }

 private:
  // Borrowed allocator, which outlives the reservation and physical storage.
  iree_hal_allocator_t* allocator_;
  // Minimum mapping granularity in bytes, queried from the live allocator.
  iree_device_size_t page_size_;
  // Owned virtual reservation, released after all pages have been unmapped.
  iree_hal_buffer_t* buffer_ = nullptr;
  // Owned whole-page allocations, one per mapping as required by ROCr VMM.
  std::array<iree_hal_physical_memory_t*, 2> physical_memory_ = {};
  // Successfully mapped pages requiring teardown, including failed protection.
  iree_host_size_t mapped_page_count_ = 0;
};

void RunSparseByteOffsetExecution(
    const loomc::testing::target::IreeHalKernelExecution& execution,
    iree_const_byte_span_t constants,
    const std::array<int32_t, 4>& expected_high_output) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(execution.device);
  if (!iree_hal_allocator_supports_virtual_memory(allocator)) {
    GTEST_SKIP() << "live allocator does not support sparse virtual memory";
  }
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_device_size_t page_size = 0;
  iree_device_size_t recommended_page_size = 0;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      allocator, params, &page_size, &recommended_page_size));
  ASSERT_GT(page_size, 0u);
  ASSERT_LE(page_size, 2 * 1024 * 1024u);
  ASSERT_EQ(SparseTestBuffer::kHighPageOffset % page_size, 0u);

  SparseTestBuffer input_buffer(allocator, page_size);
  SparseTestBuffer output_buffer(allocator, page_size);
  IREE_ASSERT_OK(input_buffer.Initialize(params));
  IREE_ASSERT_OK(output_buffer.Initialize(params));

  std::array<std::array<int32_t, 4>, 2> input = {
      {{5, 7, 11, 13}, {13, 17, 19, 23}}};
  std::array<std::array<int32_t, 4>, 2> output = {};
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      execution.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  loomc::testing::HandlePtr<iree_hal_semaphore_t, iree_hal_semaphore_release>
      semaphore_ptr(semaphore);
  uint64_t upload_value = 1;
  uint64_t dispatch_value = 2;
  uint64_t download_value = 3;
  iree_hal_semaphore_list_t upload_signal = {1, &semaphore, &upload_value};
  iree_hal_semaphore_list_t dispatch_signal = {1, &semaphore, &dispatch_value};
  iree_hal_semaphore_list_t download_signal = {1, &semaphore, &download_value};

  iree_hal_transfer_operation_t uploads[4] = {};
  iree_hal_transfer_operation_t downloads[2] = {};
  for (iree_host_size_t i = 0; i < 2; ++i) {
    uploads[i * 2].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    uploads[i * 2].upload.source = input[i].data();
    uploads[i * 2].upload.target_buffer = input_buffer.get();
    uploads[i * 2].upload.target_offset = i * SparseTestBuffer::kHighPageOffset;
    uploads[i * 2].upload.length = sizeof(input[i]);
    uploads[i * 2 + 1].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    uploads[i * 2 + 1].upload.source = output[i].data();
    uploads[i * 2 + 1].upload.target_buffer = output_buffer.get();
    uploads[i * 2 + 1].upload.target_offset =
        i * SparseTestBuffer::kHighPageOffset;
    uploads[i * 2 + 1].upload.length = sizeof(output[i]);
    downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    downloads[i].download.source_buffer = output_buffer.get();
    downloads[i].download.source_offset = i * SparseTestBuffer::kHighPageOffset;
    downloads[i].download.target = output[i].data();
    downloads[i].download.length = sizeof(output[i]);
  }

  // Retire the last successfully submitted operation before releasing any host
  // storage or sparse mapping, including when a later submission fails.
  uint64_t completion_value = 0;
  iree_status_t status = iree_hal_queue_transfer(
      execution.transfer_queue, iree_hal_semaphore_list_empty(), upload_signal,
      IREE_ARRAYSIZE(uploads), uploads);
  if (iree_status_is_ok(status)) {
    completion_value = upload_value;
    status = loomc::testing::target::DispatchIreeHalKernel(
        execution, constants, input_buffer.get(), output_buffer.get(),
        upload_signal, dispatch_signal);
  }
  if (iree_status_is_ok(status)) {
    completion_value = dispatch_value;
    status = iree_hal_queue_transfer(execution.transfer_queue, dispatch_signal,
                                     download_signal, IREE_ARRAYSIZE(downloads),
                                     downloads);
  }
  if (iree_status_is_ok(status)) {
    completion_value = download_value;
  }
  if (completion_value) {
    status = iree_status_join(
        status, iree_hal_semaphore_wait(semaphore, completion_value,
                                        iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE));
  }
  IREE_ASSERT_OK(status);
  EXPECT_EQ(output[0], (std::array<int32_t, 4>{0, 0, 0, 0}));
  EXPECT_EQ(output[1], expected_high_output);
}

TEST(LoomcAmdgpuIreeHalExecutionTest,
     SpecializesEmitsAndExecutesOnLiveAmdgpuHalDevice) {
  loomc::testing::target::RunIreeHalKernelExecutionTest(
      MakeExecutionTarget(kSourceText, "double_i32_at_byte_offset"),
      loomc::testing::target::RunIreeHalByteOffsetExecution);
}

TEST(LoomcAmdgpuIreeHalExecutionTest, UniformByteOffsetBeyond4GiB) {
  loomc::testing::target::RunIreeHalKernelExecutionTest(
      MakeExecutionTarget(kSourceText, "double_i32_at_byte_offset"),
      [](const loomc::testing::target::IreeHalKernelExecution& execution) {
        const uint64_t constants[] = {SparseTestBuffer::kHighPageOffset + 4};
        RunSparseByteOffsetExecution(
            execution, iree_make_const_byte_span(constants, sizeof(constants)),
            {0, 34, 0, 0});
      });
}

TEST(LoomcAmdgpuIreeHalExecutionTest, SpilledVaryingScaledOffsetBeyond4GiB) {
  loomc::testing::target::RunIreeHalKernelExecutionTest(
      MakeExecutionTarget(kWideScaledSourceText, "double_i32_at_scaled_offset"),
      [](const loomc::testing::target::IreeHalKernelExecution& execution) {
        const uint32_t constants[] = {0x40000001};
        RunSparseByteOffsetExecution(
            execution, iree_make_const_byte_span(constants, sizeof(constants)),
            {0, 34, 38, 0});
      });
}

}  // namespace
