// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/interop/gpu/xdna/recipes/device_fixture.h"
#include "libamdf/cts/interop/gpu/xdna/recipes/pm4_queue.h"
#include "libamdf/cts/xdna/programs/mul_i32.h"
#include "libamdf/cts/xdna/util/executable.h"
#include "libamdf/cts/xdna/util/execution.h"
#include "util/mapped_memory.h"

namespace {

constexpr size_t kBindingByteLength = 64;
constexpr size_t kBindingByteOffset = 64;
constexpr size_t kBindingStorageByteLength = 192;
constexpr size_t kStagingByteLength = 4096;
constexpr size_t kShaderResultByteOffset = 512;
constexpr size_t kArgumentStride = 64;
constexpr size_t kReadbackByteOffset = 1024;
constexpr size_t kCompletionByteOffset = 2048;
constexpr size_t kCompletionByteLength = 64;
constexpr uint32_t kGenerationCount = 8;
static_assert(sizeof(kernels::transform::Arguments) <= kArgumentStride);
constexpr std::array<uint32_t, 16> kValues = {
    0,          1,          2,          3,          7,          31,
    65535,      65536,      0x7fffffff, 0x80000000, 0x80000001, 0xfffffffd,
    0xfffffffe, 0xffffffff, 0x12345678, 0x87654321,
};

enum class GpuOperation { kTransfer, kShader };

void StoreU32(std::span<uint8_t> bytes, size_t offset, uint32_t value) {
  for (uint32_t i = 0; i < sizeof(value); ++i) {
    bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
  }
}

void CheckBytes(std::span<const uint8_t> observed,
                std::span<const uint8_t> expected) {
  const auto mismatch =
      std::mismatch(observed.begin(), observed.end(), expected.begin());
  EXPECT_EQ(mismatch.first, observed.end())
      << "byte " << std::distance(observed.begin(), mismatch.first);
}

class GpuXdnaRecipeTest : public GpuXdnaDeviceFixture {
 protected:
  explicit GpuXdnaRecipeTest(GpuOperation operation = GpuOperation::kTransfer)
      : GpuXdnaDeviceFixture(
            AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL |
            (operation == GpuOperation::kShader ? AMDF_QUEUE_ROLE_COMPUTE : 0)),
        gpu_operation_(operation) {}

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(GpuXdnaDeviceFixture::SetUp());
    if (IsSkipped()) {
      return;
    }
    if (gpu_operation_ == GpuOperation::kShader) {
      shader_.source = kernels::transform::kKernels.Find(gpu_endpoint_info_);
      ASSERT_NE(shader_.source, nullptr)
          << "missing compiled transform kernel for endpoint";
    }
    const iree_file_toc_t* image = nullptr;
    const std::string_view target = xdna_endpoint_info_.target_id;
    if (target == "amd.xdna.strix_halo.17f0_11") {
      image = &amdf_cts_xdna_mul_i32_create()[1];
    } else if (target == "amd.xdna.strix.17f0_10" ||
               target == "amd.xdna.krackan.17f0_20") {
      image = &amdf_cts_xdna_mul_i32_create()[0];
    } else {
      GTEST_SKIP() << "no finite arithmetic fixture for " << target;
    }
    constexpr std::array<amdf_memory_access_t, 3> binding_accesses = {
        AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_READ,
        AMDF_MEMORY_ACCESS_WRITE};
    ASSERT_TRUE(executable_.Initialize(
        {reinterpret_cast<const uint8_t*>(image->data), image->size},
        xdna_endpoint_info_, xdna_device_info_, 1, binding_accesses));
  }

  void TearDown() override {
    // A failed removal retains every backing still reachable by either engine.
    if (!gpu_queue_.Release(api_)) {
      return;
    }
    if (!execution_.Release(api_, xdna_api_)) {
      return;
    }
    if (!shader_.arguments.Release(api_) || !shader_.code.Release(api_)) {
      return;
    }
    if (!staging_.Release(api_)) {
      return;
    }
    for (auto& binding : bindings_) {
      if (!binding.Release(api_)) {
        return;
      }
    }
    for (auto& source : registered_storage_) {
      if (!source.Release(api_)) {
        return;
      }
    }
  }

  void PrepareShader() {
    const auto& source = *shader_.source;
    shader_.program = {
        0,
        source.program.resource1,
        source.program.resource2,
        source.program.resource3,
        source.group_segment_byte_length,
        source.wavefront_size,
        {source.required_workgroup_size[0], source.required_workgroup_size[1],
         source.required_workgroup_size[2]}};
    uint64_t code_address = 0;
    amdf_cache_transition_t code_release = {};
    ASSERT_NO_FATAL_FAILURE(CreateShaderMemory(
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
        pm4_profile_->CodeByteLength(source.executable.byte_length,
                                     source.entry_byte_offset,
                                     source.program.resource3),
        shader_.code, code_address, code_release));
    ASSERT_EQ(code_address % 256, 0u);
    ASSERT_LE(code_address,
              (UINT64_C(1) << 48) - shader_.code.info.byte_length);
    shader_.program.entry_address = code_address + source.entry_byte_offset;
    ASSERT_EQ(shader_.program.entry_address % 256, 0u);
    auto code = shader_.code.bytes();
    std::fill(code.begin(), code.end(), 0);
    std::memcpy(code.data(), source.executable.words,
                source.executable.byte_length);
    ASSERT_EQ(HostTransition(shader_.code, code_release), AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(CreateShaderMemory(
        AMDF_MEMORY_ACCESS_READ, 3 * kArgumentStride, shader_.arguments,
        shader_.argument_address, shader_.argument_release));
    ASSERT_EQ(shader_.argument_address % shader_.source->arguments.alignment,
              0u);
    RecordProperty("gpu_xdna_shader_target", source.target);
    RecordProperty("gpu_xdna_shader_hsaco_sha256", source.hsaco_sha256);
    RecordProperty("gpu_xdna_shader_image_sha256", source.executable.sha256);
    RecordProperty("gpu_xdna_shader_image_bytes",
                   source.executable.byte_length);
    RecordProperty("gpu_xdna_shader_entry_offset", source.entry_byte_offset);
    RecordProperty("gpu_xdna_shader_code_bytes",
                   std::to_string(shader_.code.info.byte_length));
    RecordProperty("gpu_xdna_shader_argument_bytes",
                   std::to_string(shader_.arguments.info.byte_length));
    RecordProperty("gpu_xdna_shader_result_offset", kShaderResultByteOffset);
    RecordProperty("gpu_xdna_shader_dispatches_per_generation", 3);
  }

  void RunRoundTrips(amdf_memory_profile_roles_t role) {
    if (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER &&
        (features_ & AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION) == 0) {
      GTEST_SKIP()
          << "GPU host registration is not advertised for this device lifetime";
    }
    amdf_memory_profile_t profile = {};
    ASSERT_NO_FATAL_FAILURE(FindProfile(accesses_, role, &profile));
    if (profile.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
      GTEST_SKIP() << (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                           ? "joint host registration is not advertised"
                           : "joint allocation is not advertised");
    }
    const auto& geometry = role == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                               ? profile.registration
                               : profile.allocation;
    ASSERT_GT(geometry.byte_length_granularity, 0u);
    amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal = profile.ordinal,
    };
    create.access_count = accesses_.size();
    create.accesses = accesses_.data();
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.byte_length =
        (kBindingStorageByteLength + geometry.byte_length_granularity - 1) /
        geometry.byte_length_granularity * geometry.byte_length_granularity;
    create.minimum_alignment = geometry.minimum_alignment;
    if (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
      create.registered_host_cacheability =
          geometry.registered_host_cacheability;
    }
    std::array<amdf_memory_pair_info_t, kGpuXdnaJointEdges.size()> joint_pairs =
        {};
    ASSERT_NO_FATAL_FAILURE(
        QueryProfilePairs(create, 1, kGpuXdnaJointEdges, joint_pairs));

    const std::span<const amdf_memory_device_access_t> gpu_access(&accesses_[1],
                                                                  1);
    amdf_memory_profile_t staging_profile = {};
    ASSERT_NO_FATAL_FAILURE(FindProfile(
        gpu_access, AMDF_MEMORY_PROFILE_ROLE_CREATE, &staging_profile));
    ASSERT_NE(staging_profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    const uint64_t granularity =
        staging_profile.allocation.byte_length_granularity;
    ASSERT_GT(granularity, 0u);
    amdf_memory_create_info_t staging_create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(staging_create),
        .memory_profile_ordinal = staging_profile.ordinal,
        .access_count = 1,
        .accesses = gpu_access.data(),
    };
    staging_create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    staging_create.byte_length =
        (kStagingByteLength + granularity - 1) / granularity * granularity;
    staging_create.minimum_alignment =
        staging_profile.allocation.minimum_alignment;
    std::array<amdf_memory_pair_info_t, kGpuXdnaStagingEdges.size()>
        staging_pairs = {};
    ASSERT_NO_FATAL_FAILURE(QueryProfilePairs(
        staging_create, 0, kGpuXdnaStagingEdges, staging_pairs));

    std::array<uint64_t, 3> xdna_addresses = {};
    std::array<uint64_t, 3> gpu_addresses = {};
    for (size_t i = 0; i < bindings_.size(); ++i) {
      SCOPED_TRACE(i);
      if (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
        amdf_memory_create_info_t host_create = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
            .structure_size = sizeof(host_create),
            .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
            .byte_length = create.byte_length,
            .minimum_alignment = geometry.registered_host_pointer_alignment,
        };
        ASSERT_NO_FATAL_FAILURE(
            registered_storage_[i].Create(api_, system_scope_, host_create));
        ASSERT_EQ(registered_storage_[i].host.cacheability,
                  create.registered_host_cacheability);
        create.registered_host_pointer = registered_storage_[i].host.pointer;
      }
      ASSERT_NO_FATAL_FAILURE(bindings_[i].Create(api_, system_scope_, create));
      ASSERT_NO_FATAL_FAILURE(CheckAccesses(bindings_[i], accesses_));
      ASSERT_NO_FATAL_FAILURE(
          CheckConcretePairs(bindings_[i], 1, kGpuXdnaJointEdges, joint_pairs));
      ASSERT_EQ(api_->memory_query_address(bindings_[i].memory, 0,
                                           AMDF_MEMORY_ADDRESS_XDNA_DMA,
                                           &xdna_addresses[i]),
                AMDF_STATUS_OK);
      ASSERT_EQ(api_->memory_query_address(bindings_[i].memory, 1,
                                           AMDF_MEMORY_ADDRESS_GPU,
                                           &gpu_addresses[i]),
                AMDF_STATUS_OK);
      ASSERT_NE(gpu_addresses[i], 0u);
      ASSERT_EQ(gpu_addresses[i] % sizeof(uint32_t), 0u);
      xdna_addresses[i] += kBindingByteOffset;
    }
    ASSERT_NO_FATAL_FAILURE(
        staging_.Create(api_, system_scope_, staging_create));
    ASSERT_NO_FATAL_FAILURE(CheckAccesses(staging_, gpu_access));
    ASSERT_NO_FATAL_FAILURE(
        CheckConcretePairs(staging_, 0, kGpuXdnaStagingEdges, staging_pairs));
    ASSERT_EQ(staging_.host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
    ASSERT_LE(staging_.host.cache_line_size, kCompletionByteLength);
    uint64_t staging_address = 0;
    ASSERT_EQ(
        api_->memory_query_address(staging_.memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                   &staging_address),
        AMDF_STATUS_OK);
    ASSERT_NE(staging_address, 0u);
    ASSERT_EQ(staging_address % kCompletionByteLength, 0u);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(staging_.host.pointer) %
                  kCompletionByteLength,
              0u);
    std::fill(staging_.bytes().begin(), staging_.bytes().end(), 0);
    ASSERT_EQ(HostTransition(staging_, staging_pairs[0].release),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(gpu_queue_.Initialize(
        api_, gpu_api_, device_, system_scope_, gpu_family_, publication_mode_,
        reinterpret_cast<uintptr_t>(staging_.host.pointer) +
            kCompletionByteOffset,
        staging_address + kCompletionByteOffset));
    std::vector<uint8_t> image_storage(executable_.allocation_byte_length());
    executable_.Load(image_storage);
    ASSERT_TRUE(executable_.Bind(image_storage, xdna_addresses));
    ASSERT_NO_FATAL_FAILURE(
        execution_.Prepare(api_, xdna_api_, xdna_device_, xdna_family_, 1,
                           executable_.ResolveInvocation(image_storage),
                           executable_.allocation_alignment()));
    if (gpu_operation_ == GpuOperation::kShader) {
      ASSERT_NO_FATAL_FAILURE(PrepareShader());
    }

    std::array<uint32_t, 308> ingress_words = {};
    Pm4CommandWriter ingress(ingress_words.data(), *pm4_profile_);
    ingress.SystemBarrier();
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      if (gpu_operation_ == GpuOperation::kShader && ordinal < 2) {
        ingress.BindCompute(shader_.program, shader_.argument_address +
                                                 ordinal * kArgumentStride);
        ingress.Dispatch(shader_.program, shader_.source->workgroup_size(), 1,
                         1);
        continue;
      }
      for (size_t offset = 0; offset < kBindingByteLength;
           offset += sizeof(uint32_t)) {
        ingress.CopyData32(
            staging_address + ordinal * kBindingByteLength + offset,
            gpu_addresses[ordinal] + kBindingByteOffset + offset);
      }
    }
    ingress.SystemBarrier();
    ASSERT_EQ(ingress.word_count(),
              gpu_operation_ == GpuOperation::kShader ? 178u : 308u);
    std::array<uint32_t, 925> egress_words = {};
    Pm4CommandWriter egress(egress_words.data(), *pm4_profile_);
    egress.SystemBarrier();
    if (gpu_operation_ == GpuOperation::kShader) {
      egress.BindCompute(shader_.program,
                         shader_.argument_address + 2 * kArgumentStride);
      egress.Dispatch(shader_.program, shader_.source->workgroup_size(), 1, 1);
      // Join shader stores before the independent TC/L2 guard readback.
      egress.SystemBarrier();
    }
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      for (size_t offset = 0; offset < kBindingStorageByteLength;
           offset += sizeof(uint32_t)) {
        egress.CopyData32(gpu_addresses[ordinal] + offset,
                          staging_address + kReadbackByteOffset +
                              ordinal * kBindingStorageByteLength + offset);
      }
    }
    egress.SystemBarrier();
    ASSERT_EQ(egress.word_count(),
              gpu_operation_ == GpuOperation::kShader ? 925u : 884u);

    const auto command_bytes = execution_.instructions.bytes();
    const std::vector<uint8_t> original_commands(command_bytes.begin(),
                                                 command_bytes.end());
    std::vector<uint8_t> observed_commands(command_bytes.size());
    const auto code_bytes = shader_.code.bytes();
    std::vector<uint8_t> original_code(code_bytes.size());
    if (!code_bytes.empty()) {
      std::copy(code_bytes.begin(), code_bytes.end(), original_code.begin());
    }
    std::vector<uint8_t> observed_code(code_bytes.size());
    std::vector<uint8_t> expected_arguments(shader_.arguments.bytes().size());
    std::vector<uint8_t> observed_arguments(expected_arguments.size());
    std::vector<uint8_t> expected_staging(staging_.bytes().size());
    std::vector<uint8_t> observed_staging(staging_.bytes().size());
    std::array<std::vector<uint8_t>, 3> expected;
    std::array<std::vector<uint8_t>, 3> observed;
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      expected[ordinal].resize(bindings_[ordinal].bytes().size());
      observed[ordinal].resize(bindings_[ordinal].bytes().size());
    }
    ASSERT_EQ(api_->host_mapping_cache_control(execution_.instructions.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, command_bytes.size()),
              AMDF_STATUS_OK);
    RecordProperty("gpu_xdna_generations", kGenerationCount);
    RecordProperty("gpu_xdna_binding_byte_length", kBindingByteLength);
    RecordProperty("gpu_xdna_binding_byte_offset", kBindingByteOffset);
    RecordProperty("gpu_xdna_guarded_byte_length", kBindingStorageByteLength);
    RecordProperty("gpu_xdna_joint_profile_ordinal", profile.ordinal);
    RecordProperty("gpu_xdna_joint_byte_length",
                   std::to_string(bindings_[0].info.byte_length));
    RecordProperty("gpu_xdna_staging_bytes",
                   std::to_string(staging_.host.byte_length));
    RecordProperty("gpu_xdna_command_bytes",
                   std::to_string(command_bytes.size()));
    RecordProperty("gpu_xdna_readback_byte_offset", kReadbackByteOffset);
    RecordProperty("gpu_xdna_completion_byte_offset", kCompletionByteOffset);
    RecordProperty("gpu_xdna_ingress_words", ingress.word_count());
    RecordProperty("gpu_xdna_egress_words", egress.word_count());
    RecordProperty(
        "gpu_xdna_gpu_operation",
        gpu_operation_ == GpuOperation::kShader ? "shader" : "transfer");

    for (uint32_t generation = 0;
         generation < kGenerationCount && !HasFailure(); ++generation) {
      SCOPED_TRACE(generation);
      const std::array<uint32_t, 3> addends = {2 * generation + 1,
                                               0x80000001u + 2 * generation,
                                               0x12345679u + 2 * generation};
      auto staging = staging_.bytes();
      const uint8_t staging_guard = static_cast<uint8_t>(0x3C ^ generation);
      std::fill(staging.begin(), staging.begin() + kCompletionByteOffset,
                staging_guard);
      std::fill(staging.begin() + kCompletionByteOffset + kCompletionByteLength,
                staging.end(), staging_guard);
      std::fill(expected_staging.begin(), expected_staging.end(),
                staging_guard);
      std::fill_n(expected_staging.begin() + kCompletionByteOffset,
                  kCompletionByteLength, 0);
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        const uint8_t guard =
            static_cast<uint8_t>(0xA5 ^ (generation * 7 + ordinal * 17));
        std::fill(bindings_[ordinal].bytes().begin(),
                  bindings_[ordinal].bytes().end(), guard);
        std::fill(expected[ordinal].begin(), expected[ordinal].end(), guard);
        std::fill_n(staging.begin() + kReadbackByteOffset +
                        ordinal * kBindingStorageByteLength,
                    kBindingStorageByteLength, static_cast<uint8_t>(~guard));
      }
      for (size_t i = 0; i < kValues.size(); ++i) {
        const uint32_t input_lhs = kValues[(i + generation) % kValues.size()];
        const uint32_t input_rhs =
            kValues[(i * 3 + generation + 5) % kValues.size()];
        const uint32_t lhs = gpu_operation_ == GpuOperation::kShader
                                 ? uint64_t{input_lhs} * 3 + addends[0]
                                 : input_lhs;
        const uint32_t rhs = gpu_operation_ == GpuOperation::kShader
                                 ? uint64_t{input_rhs} * 3 + addends[1]
                                 : input_rhs;
        const uint32_t product = uint64_t{lhs} * rhs;
        const std::array<uint32_t, 3> values = {lhs, rhs, product};
        const std::array<uint32_t, 3> source_values = {input_lhs, input_rhs,
                                                       ~product};
        for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
          const size_t source_offset =
              ordinal * kBindingByteLength + i * sizeof(uint32_t);
          const uint32_t source_value = source_values[ordinal];
          StoreU32(staging, source_offset, source_value);
          StoreU32(expected_staging, source_offset, source_value);
          StoreU32(expected[ordinal], kBindingByteOffset + i * sizeof(uint32_t),
                   values[ordinal]);
          StoreU32(staging,
                   kReadbackByteOffset + ordinal * kBindingStorageByteLength +
                       kBindingByteOffset + i * sizeof(uint32_t),
                   ~values[ordinal]);
        }
        if (gpu_operation_ == GpuOperation::kShader) {
          const uint32_t result = uint64_t{product} * 3 + addends[2];
          StoreU32(staging, kShaderResultByteOffset + i * sizeof(uint32_t),
                   ~result);
          StoreU32(expected_staging,
                   kShaderResultByteOffset + i * sizeof(uint32_t), result);
        }
      }
      if (gpu_operation_ == GpuOperation::kShader) {
        // Only the ABI's semantic bytes are copied; every slot's padding and
        // the rest of the rounded allocation remain initialized and checked.
        std::fill(expected_arguments.begin(), expected_arguments.end(), 0);
        const std::array<kernels::transform::Arguments, 3> arguments = {{
            {staging_address, gpu_addresses[0] + kBindingByteOffset,
             kValues.size(), addends[0]},
            {staging_address + kBindingByteLength,
             gpu_addresses[1] + kBindingByteOffset, kValues.size(), addends[1]},
            {gpu_addresses[2] + kBindingByteOffset,
             staging_address + kShaderResultByteOffset, kValues.size(),
             addends[2]},
        }};
        for (size_t ordinal = 0; ordinal < arguments.size(); ++ordinal) {
          std::memcpy(expected_arguments.data() + ordinal * kArgumentStride,
                      &arguments[ordinal],
                      shader_.source->arguments.byte_length);
        }
        std::memcpy(shader_.arguments.bytes().data(), expected_arguments.data(),
                    expected_arguments.size());
        ASSERT_EQ(HostTransition(shader_.arguments, shader_.argument_release),
                  AMDF_STATUS_OK);
      }
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        std::copy_n(expected[ordinal].begin(), kBindingStorageByteLength,
                    expected_staging.begin() + kReadbackByteOffset +
                        ordinal * kBindingStorageByteLength);
        ASSERT_EQ(HostTransition(bindings_[ordinal],
                                 joint_pairs[kGpuXdnaHostToXdna].release),
                  AMDF_STATUS_OK);
        ASSERT_EQ(HostTransition(bindings_[ordinal],
                                 joint_pairs[kGpuXdnaHostToGpu].release),
                  AMDF_STATUS_OK);
      }
      ASSERT_EQ(HostTransition(staging_, staging_pairs[0].release),
                AMDF_STATUS_OK);
      const uint32_t completion_value = generation * 2 + 2;
      if (publication_mode_ == AMDF_QUEUE_PUBLICATION_MODE_USER) {
        StoreU32(expected_staging, kCompletionByteOffset, completion_value);
      }
      ASSERT_TRUE(gpu_queue_.Publish(
          api_, gpu_api_,
          std::span<const uint32_t>(ingress_words).first(ingress.word_count()),
          completion_value - 1));
      ASSERT_TRUE(gpu_queue_.WaitComplete(api_));
      ASSERT_TRUE(gpu_queue_.Retire(api_));
      if (HasFailure()) {
        return;
      }

      // No CPU read or cache operation on joint backing occurs between these
      // device phases. The NPU command joins the finite external payload flow;
      // resident workers and compute DMA retain only tile-local state.
      amdf_xdna_kernel_queue_submission_info_t submit = {
          .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
          .structure_size = sizeof(submit),
          .command_count = 1,
          .commands = &execution_.command,
      };
      uint64_t point = 0;
      ASSERT_EQ(
          xdna_api_->kernel_queue_submit(execution_.queue, &submit, &point),
          AMDF_STATUS_OK);
      ASSERT_EQ(api_->kernel_queue_wait(execution_.queue, point,
                                        AMDF_TIMEOUT_INFINITE, 0),
                AMDF_STATUS_OK);
      ASSERT_TRUE(gpu_queue_.Publish(
          api_, gpu_api_,
          std::span<const uint32_t>(egress_words).first(egress.word_count()),
          completion_value));
      ASSERT_TRUE(gpu_queue_.WaitComplete(api_));

      // Acquire and capture the complete GPU readback owner first. Only then
      // inspect rounded joint allocations, using their actual last writer:
      // GPU inputs and NPU output. This later maintenance cannot assist the
      // already captured device-to-device readback.
      const auto staging_status =
          HostTransition(staging_, staging_pairs[1].acquire);
      if (amdf_status_is_ok(staging_status)) {
        std::copy(staging.begin(), staging.end(), observed_staging.begin());
      }
      std::array<amdf_status_t, 3> observation_status = {};
      for (size_t ordinal : {2u, 0u, 1u}) {
        observation_status[ordinal] = HostTransition(
            bindings_[ordinal],
            joint_pairs[ordinal == 2 ? kGpuXdnaXdnaToHost : kGpuXdnaGpuToHost]
                .acquire);
        if (amdf_status_is_ok(observation_status[ordinal])) {
          const auto bytes = bindings_[ordinal].bytes();
          std::copy(bytes.begin(), bytes.end(), observed[ordinal].begin());
        }
      }
      const auto command_status = api_->host_mapping_cache_control(
          execution_.instructions.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
          0, command_bytes.size());
      if (amdf_status_is_ok(command_status)) {
        std::copy(command_bytes.begin(), command_bytes.end(),
                  observed_commands.begin());
      }
      amdf_status_t code_status = AMDF_STATUS_OK;
      amdf_status_t argument_status = AMDF_STATUS_OK;
      if (gpu_operation_ == GpuOperation::kShader) {
        code_status =
            HostTransition(shader_.code, shader_.code.host.invalidate);
        if (amdf_status_is_ok(code_status)) {
          std::copy(code_bytes.begin(), code_bytes.end(),
                    observed_code.begin());
        }
        argument_status = HostTransition(shader_.arguments,
                                         shader_.arguments.host.invalidate);
        if (amdf_status_is_ok(argument_status)) {
          const auto argument_bytes = shader_.arguments.bytes();
          std::copy(argument_bytes.begin(), argument_bytes.end(),
                    observed_arguments.begin());
        }
      }
      EXPECT_EQ(staging_status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(staging_status)) {
        CheckBytes(observed_staging, expected_staging);
      }
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        SCOPED_TRACE(ordinal);
        EXPECT_EQ(observation_status[ordinal], AMDF_STATUS_OK);
        if (amdf_status_is_ok(observation_status[ordinal])) {
          CheckBytes(observed[ordinal], expected[ordinal]);
        }
      }
      EXPECT_EQ(command_status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(command_status)) {
        CheckBytes(observed_commands, original_commands);
      }
      EXPECT_EQ(code_status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(code_status)) {
        CheckBytes(observed_code, original_code);
      }
      EXPECT_EQ(argument_status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(argument_status)) {
        CheckBytes(observed_arguments, expected_arguments);
      }
      amdf_kernel_queue_status_t status = {
          .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
          .structure_size = sizeof(status),
      };
      const auto query_status =
          api_->kernel_queue_query_status(execution_.queue, &status);
      EXPECT_EQ(query_status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(query_status)) {
        EXPECT_EQ(status.retired_submission, point);
        EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
      }
      // Diagnostics never bypass command-storage retirement or permit a new
      // generation after a failed observation.
      EXPECT_TRUE(gpu_queue_.Retire(api_));
    }
  }

  // Fixed case workload, selected before endpoint and queue admission.
  const GpuOperation gpu_operation_;
  // Immutable arithmetic program borrowing build-generated Loom output.
  XdnaExecutable executable_;
  // NPU context, immutable bound commands and checked native queue.
  XdnaExecution execution_;
  // GPU queue and its command storage, removed before reachable backing.
  Pm4RecipeQueue gpu_queue_;
  // CPU source owners retained until every native registration is detached.
  std::array<CtsMappedMemory, 3> registered_storage_;
  // Three joint allocations retained through both engines' queue removal.
  std::array<CtsMappedMemory, 3> bindings_;
  // GPU-only inputs, readback and a separately aligned coherent marker line.
  CtsMappedMemory staging_;
  // Optional shader state retained through removal of its GPU borrower.
  struct {
    // Exact compiled source selected before native endpoint activation.
    const kernels::Kernel* source = nullptr;
    // Compiled entry and resource configuration for all three dispatches.
    Pm4ComputeProgram program = {};
    // Immutable full image plus the declared instruction fetch extent.
    CtsMappedMemory code;
    // Three kernarg slots rewritten only after final egress retirement.
    CtsMappedMemory arguments;
    // GPU address of the first kernarg slot, independent of its host mapping.
    uint64_t argument_address = 0;
    // Exact HOST-to-GPU publication recipe for the argument allocation.
    amdf_cache_transition_t argument_release = {};
  } shader_;
};

TEST_F(GpuXdnaRecipeTest, AllocatedRoundTrip) {
  RunRoundTrips(AMDF_MEMORY_PROFILE_ROLE_CREATE);
}

TEST_F(GpuXdnaRecipeTest, RegisteredRoundTrip) {
  RunRoundTrips(AMDF_MEMORY_PROFILE_ROLE_REGISTER);
}

class GpuXdnaShaderRecipeTest : public GpuXdnaRecipeTest {
 protected:
  GpuXdnaShaderRecipeTest() : GpuXdnaRecipeTest(GpuOperation::kShader) {}
};

TEST_F(GpuXdnaShaderRecipeTest, AllocatedRoundTrip) {
  RunRoundTrips(AMDF_MEMORY_PROFILE_ROLE_CREATE);
}

TEST_F(GpuXdnaShaderRecipeTest, RegisteredRoundTrip) {
  RunRoundTrips(AMDF_MEMORY_PROFILE_ROLE_REGISTER);
}

}  // namespace
