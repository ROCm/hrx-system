// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "test/target/iree_hal_execution_test_util.h"
#include "test/util.h"

namespace {

constexpr char kSourceText[] = R"(
kernel.def @double_i32_at_byte_offset() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%input: buffer, %output: buffer, %byte_offset: offset) {
  %byte_offset_aligned = index.assume %byte_offset [multiple_of(%byte_offset, 4)] : offset
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
#if defined(LOOM_AMDGPU_DESCRIPTOR_SET_CDNA2_CORE)
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx90a, "gfx90a",
                                "amdgpu.cdna2.core")
#endif
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
    AMDGPU_SPILL_OFFSET_PROVIDER(gfx1250_a0, "gfx1250-a0",
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

loomc::testing::target::IreeHalKernelExecutionTarget MakeExecutionTarget(
    const char* source_text, const char* kernel_export_name) {
  static const loomc_iree_hal_target_provider_t* const target_providers[] = {
      loomc_amdgpu_iree_hal_target_provider(),
  };

  loomc::testing::target::IreeHalKernelExecutionTarget target = {
      .label = "AMDGPU",
      .device_uri = IREE_SV("amdgpu"),
      .target_profile_identifier = loomc_make_cstring_view("live-amdgpu"),
      .source_identifier = loomc_make_cstring_view("live_amdgpu.loom"),
      .source_text = loomc_make_cstring_view(source_text),
      .kernel_export_name = loomc_make_cstring_view(kernel_export_name),
      .target_pipeline_identifier =
          loomc_make_cstring_view("live-amdgpu-prepared-low"),
      .target_pipeline_kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      .source_to_low_max_errors = 20,
      .artifact_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      .artifact_identifier = loomc_make_cstring_view("live_amdgpu.hsaco"),
      .target_providers = target_providers,
      .target_provider_count = 1,
      .create_target_environment = CreateAmdgpuTargetEnvironment,
      .validate_target_profile = ValidateAmdgpuProfile,
  };
  return target;
}

// Backs the first page and the page at 4 GiB with distinct physical pages.
// A truncated address remains mapped and produces an observable wrong result.
class SparseTestBuffer {
 public:
  static constexpr iree_device_size_t kHighPageOffset = uint64_t{1} << 32;

  SparseTestBuffer() = default;
  SparseTestBuffer(const SparseTestBuffer&) = delete;
  SparseTestBuffer& operator=(const SparseTestBuffer&) = delete;

  ~SparseTestBuffer() {
    iree_status_t status = iree_ok_status();
    if (reservation_) {
      const iree_hal_memory_scope_t scope = {
          reservation_->memory.contract->domain, IREE_HAL_MEMORY_SCOPE_ANY};
      status = iree_hal_virtual_buffer_detach(reservation_, scope);
    }
    iree_hal_buffer_release(reservation_);
    if (iree_status_is_ok(status)) {
      for (auto* backing : backing_) {
        iree_hal_buffer_release(backing);
      }
    }
    IREE_EXPECT_OK(status);
  }

  iree_status_t Initialize(iree_hal_device_group_t* group,
                           iree_hal_pool_scope_t scope,
                           iree_hal_pool_t* source_pool,
                           iree_hal_queue_t* queue,
                           iree_device_size_t page_size) {
    const iree_hal_virtual_buffer_options_t options = {
        .byte_length = kHighPageOffset + page_size,
        .mapping_granularity = page_size,
    };
    IREE_RETURN_IF_ERROR(iree_hal_virtual_buffer_create(
        group, scope, &options, iree_allocator_system(), &reservation_));
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < backing_.size() && iree_status_is_ok(status); ++i) {
      status = iree_hal_pool_allocate_buffer(
          source_pool,
          iree_hal_make_pool_reservation_request(page_size, page_size),
          iree_infinite_timeout(), &backing_[i]);
    }
    IREE_RETURN_IF_ERROR(status);
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        iree_hal_queue_family_device(iree_hal_queue_family(queue)),
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &semaphore));
    loomc::testing::HandlePtr<iree_hal_semaphore_t, iree_hal_semaphore_release>
        semaphore_ptr(semaphore);
    uint64_t value = 1;
    const iree_hal_semaphore_list_t signal = {1, &semaphore, &value};
    std::array<iree_hal_memory_mapping_t, 2> mappings;
    for (size_t i = 0; i < mappings.size(); ++i) {
      mappings[i] = {
          .target = reservation_,
          .scope =
              {
                  .domain = iree_hal_device_group_memory_domain(group),
                  .id = IREE_HAL_MEMORY_SCOPE_ANY,
              },
          .target_range =
              iree_hal_make_buffer_range(i * kHighPageOffset, page_size),
          .source = backing_[i],
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      };
    }
    IREE_RETURN_IF_ERROR(iree_hal_queue_map(
        queue, iree_hal_semaphore_list_empty(), signal, mappings.size(),
        mappings.data(), IREE_HAL_QUEUE_MAP_FLAG_NONE));
    return iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_hal_buffer_t* get() const { return reservation_; }

 private:
  // Virtual buffer retaining the address interval until final release.
  iree_hal_buffer_t* reservation_ = nullptr;
  // Separate source allocations preserve distinct low/high contents on devices
  // whose native mappings require whole allocations.
  std::array<iree_hal_buffer_t*, 2> backing_ = {};
};

void RunSparseByteOffsetExecution(
    const loomc::testing::target::IreeHalKernelExecution& execution,
    iree_const_byte_span_t constants,
    const std::array<int32_t, 4>& expected_high_output) {
  std::array<iree_hal_pool_family_access_t, 2> families = {{
      {
          .family = iree_hal_queue_family(execution.transfer_queue),
          .usage =
              IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
      },
      {
          .family = iree_hal_queue_family(execution.dispatch_queue),
          .usage =
              IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
      },
  }};
  const iree_hal_pool_scope_t scope = {
      .family_count = families[0].family == families[1].family ? 1u : 2u,
      .families = families.data(),
  };
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      execution.device_group, scope, iree_allocator_system(), &capabilities));
  if (!capabilities.max_byte_length) {
    GTEST_SKIP() << "live device group does not support sparse virtual memory";
  }
  for (auto& family : families) {
    family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  iree_hal_slab_pool_options_t source_options;
  iree_hal_slab_pool_options_initialize(&source_options);
  iree_hal_pool_t* source_pool = nullptr;
  IREE_ASSERT_OK(
      iree_hal_slab_pool_create(execution.device_group, scope, &source_options,
                                iree_allocator_system(), &source_pool));
  loomc::testing::HandlePtr<iree_hal_pool_t, iree_hal_pool_release>
      source_pool_ptr(source_pool);
  const auto requirements =
      iree_hal_pool_query_mapping_requirements(source_pool);
  const iree_device_size_t page_size =
      iree_max(capabilities.reservation_granularity,
               iree_max(capabilities.mapping_granularity,
                        requirements.minimum_granularity));
  ASSERT_GT(page_size, 0u);
  ASSERT_LE(page_size, 2 * 1024 * 1024u);
  ASSERT_EQ(SparseTestBuffer::kHighPageOffset % page_size, 0u);
  for (auto& family : families) {
    family.interfaces = 0;
  }

  SparseTestBuffer input_buffer;
  SparseTestBuffer output_buffer;
  IREE_ASSERT_OK(input_buffer.Initialize(execution.device_group, scope,
                                         source_pool, execution.transfer_queue,
                                         page_size));
  IREE_ASSERT_OK(output_buffer.Initialize(execution.device_group, scope,
                                          source_pool, execution.transfer_queue,
                                          page_size));

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
      IREE_ARRAYSIZE(uploads), uploads, /*barriers=*/NULL);
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
                                     downloads, /*barriers=*/NULL);
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
