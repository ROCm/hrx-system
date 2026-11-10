// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/image/hsaco_metadata.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

using Bytes = std::vector<uint8_t>;
using Fields = std::vector<std::pair<std::string, Bytes>>;

Bytes Integer(uint64_t value) {
  if (value < 128) {
    return {static_cast<uint8_t>(value)};
  }
  Bytes result = {0xCF};
  for (int shift = 56; shift >= 0; shift -= 8) {
    result.push_back(static_cast<uint8_t>(value >> shift));
  }
  return result;
}

Bytes String(const std::string& value) {
  Bytes result = {0xDA, static_cast<uint8_t>(value.size() >> 8),
                  static_cast<uint8_t>(value.size())};
  result.insert(result.end(), value.begin(), value.end());
  return result;
}

Bytes Array(const std::vector<Bytes>& values) {
  Bytes result = {0xDC, static_cast<uint8_t>(values.size() >> 8),
                  static_cast<uint8_t>(values.size())};
  for (const auto& value : values) {
    result.insert(result.end(), value.begin(), value.end());
  }
  return result;
}

Bytes Map(const Fields& fields) {
  Bytes result = {0xDE, static_cast<uint8_t>(fields.size() >> 8),
                  static_cast<uint8_t>(fields.size())};
  for (const auto& [name, value] : fields) {
    auto key = String(name);
    result.insert(result.end(), key.begin(), key.end());
    result.insert(result.end(), value.begin(), value.end());
  }
  return result;
}

void Set(Fields& fields, const std::string& name, Bytes value) {
  auto field =
      std::find_if(fields.begin(), fields.end(),
                   [&](const auto& field) { return field.first == name; });
  if (field == fields.end()) {
    fields.emplace_back(name, std::move(value));
  } else {
    field->second = std::move(value);
  }
}

void Erase(Fields& fields, const std::string& name) {
  fields.erase(
      std::remove_if(fields.begin(), fields.end(),
                     [&](const auto& field) { return field.first == name; }),
      fields.end());
}

Fields Kernel() {
  return {{".name", String("transform")},
          {".symbol", String("transform.kd")},
          {".kernarg_segment_size", Integer(16)},
          {".kernarg_segment_align", Integer(8)},
          {".group_segment_fixed_size", Integer(512)},
          {".private_segment_fixed_size", Integer(32)},
          {".wavefront_size", Integer(64)},
          {".sgpr_count", Integer(24)},
          {".vgpr_count", Integer(40)},
          {".max_flat_workgroup_size", Integer(256)}};
}

Fields Argument(const char* kind, uint32_t offset = 0, uint32_t size = 8) {
  return {{".offset", Integer(offset)},
          {".size", Integer(size)},
          {".value_kind", String(kind)}};
}

Fields Module(Fields kernel = Kernel(), uint32_t minor_version = 2) {
  Fields fields = {
      {"amdhsa.version", Array({Integer(1), Integer(minor_version)})},
      {"amdhsa.kernels", Array({Map(kernel)})}};
  if (minor_version) {
    fields.emplace_back("amdhsa.target", String("amdgcn-amd-amdhsa--gfx942"));
  }
  return fields;
}

std::string ToString(iree_string_view_t value) {
  return std::string(value.data ? value.data : "", value.size);
}

class HsacoMetadataTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_hal_amd_gpu_metadata_deinitialize(&metadata_);
  }

  iree_status_t Parse(Bytes payload) {
    return ParsePayloads({std::move(payload)});
  }

  iree_status_t ParsePayloads(std::vector<Bytes> payloads) {
    iree_hal_amd_gpu_metadata_deinitialize(&metadata_);
    payloads_ = std::move(payloads);
    std::vector<iree_const_byte_span_t> spans;
    for (const auto& payload : payloads_) {
      spans.push_back(
          iree_make_const_byte_span(payload.data(), payload.size()));
    }
    return iree_hal_amd_gpu_hsaco_metadata_initialize(
        spans.size(), spans.data(), iree_allocator_system(), &metadata_);
  }

  // Owns borrowed payload bytes through metadata deinitialization.
  std::vector<Bytes> payloads_;
  // Owns decoded records and borrows strings from payloads_.
  iree_hal_amd_gpu_metadata_t metadata_ = {};
};

TEST_F(HsacoMetadataTest, AdmitsEveryCurrentMetadataVersionWithoutArguments) {
  for (uint32_t minor : {0u, 1u, 2u}) {
    SCOPED_TRACE(minor);
    IREE_ASSERT_OK(Parse(Map(Module(Kernel(), minor))));
    EXPECT_EQ(metadata_.version[0], 1u);
    EXPECT_EQ(metadata_.version[1], minor);
    ASSERT_EQ(metadata_.kernel_count, 1u);
    EXPECT_EQ(metadata_.argument_count, 0u);
    EXPECT_EQ(metadata_.kernels[0].arguments, nullptr);
    EXPECT_FALSE(metadata_.kernels[0].has_argument_metadata);
    EXPECT_EQ(ToString(metadata_.kernels[0].reflection_name), "transform");
    EXPECT_EQ(metadata_.kernels[0].wavefront_size, 64u);
  }
}

TEST_F(HsacoMetadataTest, DistinguishesAbsentAndEmptyArgumentReflection) {
  auto kernel = Kernel();
  IREE_ASSERT_OK(Parse(Map(Module(kernel))));
  EXPECT_FALSE(metadata_.kernels[0].has_argument_metadata);
  Set(kernel, ".args", Array({}));
  IREE_ASSERT_OK(Parse(Map(Module(kernel))));
  EXPECT_TRUE(metadata_.kernels[0].has_argument_metadata);
  EXPECT_EQ(metadata_.kernels[0].argument_count, 0u);
}

TEST_F(HsacoMetadataTest, RetainsCompleteKernelResourceAndSourceMetadata) {
  auto kernel = Kernel();
  kernel.insert(
      kernel.end(),
      {
          {".language", String("OpenCL C")},
          {".language_version", Array({Integer(2), Integer(0)})},
          {".vec_type_hint", String("float4")},
          {".device_enqueue_symbol", String("enqueue_handle")},
          {".kind", String("init")},
          {".agpr_count", Integer(16)},
          {".sgpr_spill_count", Integer(3)},
          {".vgpr_spill_count", Integer(7)},
          {".reqd_workgroup_size", Array({Integer(8), Integer(4), Integer(2)})},
          {".workgroup_size_hint", Array({Integer(4), Integer(4), Integer(4)})},
          {".max_num_work_groups_x", Integer(1024)},
          {".max_num_work_groups_y", Integer(128)},
          {".max_num_work_groups_z", Integer(64)},
          {".uniform_work_group_size", Integer(1)},
          {".uses_dynamic_stack", {0xC3}},
          {".workgroup_processor_mode", {0xC2}},
          {".cluster_dims", Array({Integer(2), Integer(1), Integer(1)})},
      });
  IREE_ASSERT_OK(Parse(Map(Module(kernel))));
  const auto& result = metadata_.kernels[0];
  EXPECT_EQ(ToString(result.source.language), "OpenCL C");
  EXPECT_EQ(result.source.language_version[0], 2u);
  EXPECT_EQ(result.source.language_version[1], 0u);
  EXPECT_EQ(ToString(result.source.vector_type_hint), "float4");
  EXPECT_EQ(ToString(result.device_enqueue_symbol), "enqueue_handle");
  EXPECT_EQ(result.kind, IREE_HAL_AMD_GPU_METADATA_KERNEL_KIND_INITIALIZER);
  EXPECT_EQ(result.kernarg_segment_size, 16u);
  EXPECT_EQ(result.kernarg_segment_alignment, 8u);
  EXPECT_EQ(result.group_segment_fixed_size, 512u);
  EXPECT_EQ(result.private_segment_fixed_size, 32u);
  EXPECT_EQ(result.sgpr_count, 24u);
  EXPECT_EQ(result.vgpr_count, 40u);
  EXPECT_EQ(result.agpr_count, 16u);
  EXPECT_EQ(result.sgpr_spill_count, 3u);
  EXPECT_EQ(result.vgpr_spill_count, 7u);
  EXPECT_TRUE(result.has_required_workgroup_size);
  EXPECT_EQ(result.required_workgroup_size[0], 8u);
  EXPECT_EQ(result.required_workgroup_size[1], 4u);
  EXPECT_EQ(result.required_workgroup_size[2], 2u);
  EXPECT_EQ(result.workgroup_size_hint[0], 4u);
  EXPECT_EQ(result.max_workgroup_count[0], 1024u);
  EXPECT_EQ(result.max_workgroup_count[1], 128u);
  EXPECT_EQ(result.max_workgroup_count[2], 64u);
  EXPECT_TRUE(result.uniform_workgroup_size);
  EXPECT_TRUE(result.uses_dynamic_stack);
  EXPECT_FALSE(result.workgroup_processor_mode);
  EXPECT_TRUE(result.has_workgroup_processor_mode);
  EXPECT_TRUE(result.has_workgroup_cluster_size);
  EXPECT_EQ(result.workgroup_cluster_size[0], 2u);
  Set(kernel, ".kind", String("fini"));
  IREE_ASSERT_OK(Parse(Map(Module(kernel))));
  EXPECT_EQ(metadata_.kernels[0].kind,
            IREE_HAL_AMD_GPU_METADATA_KERNEL_KIND_FINALIZER);
}

TEST_F(HsacoMetadataTest, RetainsArgumentQualifiersAndBothAccessDeclarations) {
  auto argument = Argument("global_buffer");
  argument.insert(argument.end(), {
                                      {".name", String("input")},
                                      {".type_name", String("const float *")},
                                      {".value_type", String("F32")},
                                      {".alignment", Integer(8)},
                                      {".address_space", String("global")},
                                      {".access", String("read_write")},
                                      {".actual_access", String("read_only")},
                                      {".is_const", {0xC3}},
                                      {".is_restrict", {0xC3}},
                                      {".is_volatile", {0xC3}},
                                  });
  auto local = Argument("dynamic_shared_pointer", 8, 4);
  local.emplace_back(".address_space", String("local"));
  local.emplace_back(".pointee_align", Integer(16));
  auto pipe = Argument("pipe", 12, 4);
  pipe.emplace_back(".is_pipe", Bytes{0xC3});
  auto kernel = Kernel();
  Set(kernel, ".args", Array({Map(argument), Map(local), Map(pipe)}));
  IREE_ASSERT_OK(Parse(Map(Module(kernel))));
  ASSERT_EQ(metadata_.argument_count, 3u);
  const auto& result = metadata_.arguments[0];
  EXPECT_EQ(ToString(result.name), "input");
  EXPECT_EQ(ToString(result.type_name), "const float *");
  EXPECT_EQ(ToString(result.value_type), "F32");
  EXPECT_EQ(result.alignment, 8u);
  EXPECT_EQ(result.address_space,
            IREE_HAL_AMD_GPU_METADATA_ADDRESS_SPACE_GLOBAL);
  EXPECT_EQ(result.access, IREE_HAL_AMD_GPU_METADATA_ACCESS_READ_WRITE);
  EXPECT_EQ(result.actual_access, IREE_HAL_AMD_GPU_METADATA_ACCESS_READ_ONLY);
  EXPECT_EQ(result.effective_access,
            IREE_HAL_AMD_GPU_METADATA_ACCESS_READ_ONLY);
  EXPECT_TRUE(result.qualifiers.is_const);
  EXPECT_TRUE(result.qualifiers.is_restrict);
  EXPECT_TRUE(result.qualifiers.is_volatile);
  EXPECT_EQ(metadata_.arguments[1].pointee_alignment, 16u);
  EXPECT_EQ(metadata_.arguments[1].address_space,
            IREE_HAL_AMD_GPU_METADATA_ADDRESS_SPACE_LOCAL);
  EXPECT_TRUE(metadata_.arguments[2].qualifiers.is_pipe);
  EXPECT_EQ(metadata_.argument_name_storage_size, 5u);
}

TEST_F(HsacoMetadataTest, ClassifiesEveryExplicitAndHiddenArgumentKind) {
  const std::pair<const char*, iree_hal_amd_gpu_metadata_argument_kind_t>
      cases[] = {
#define KIND(name, value) \
  {name, IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_##value}
          KIND("by_value", BY_VALUE),
          KIND("global_buffer", GLOBAL_BUFFER),
          KIND("dynamic_shared_pointer", DYNAMIC_SHARED_POINTER),
          KIND("image", IMAGE),
          KIND("sampler", SAMPLER),
          KIND("pipe", PIPE),
          KIND("queue", QUEUE),
          KIND("hidden_none", HIDDEN_NONE),
          KIND("hidden_global_offset_x", HIDDEN_GLOBAL_OFFSET_X),
          KIND("hidden_global_offset_y", HIDDEN_GLOBAL_OFFSET_Y),
          KIND("hidden_global_offset_z", HIDDEN_GLOBAL_OFFSET_Z),
          KIND("hidden_printf_buffer", HIDDEN_PRINTF_BUFFER),
          KIND("hidden_hostcall_buffer", HIDDEN_HOSTCALL_BUFFER),
          KIND("hidden_default_queue", HIDDEN_DEFAULT_QUEUE),
          KIND("hidden_completion_action", HIDDEN_COMPLETION_ACTION),
          KIND("hidden_multigrid_sync_arg", HIDDEN_MULTIGRID_SYNC_ARG),
          KIND("hidden_block_count_x", HIDDEN_BLOCK_COUNT_X),
          KIND("hidden_block_count_y", HIDDEN_BLOCK_COUNT_Y),
          KIND("hidden_block_count_z", HIDDEN_BLOCK_COUNT_Z),
          KIND("hidden_group_size_x", HIDDEN_GROUP_SIZE_X),
          KIND("hidden_group_size_y", HIDDEN_GROUP_SIZE_Y),
          KIND("hidden_group_size_z", HIDDEN_GROUP_SIZE_Z),
          KIND("hidden_remainder_x", HIDDEN_REMAINDER_X),
          KIND("hidden_remainder_y", HIDDEN_REMAINDER_Y),
          KIND("hidden_remainder_z", HIDDEN_REMAINDER_Z),
          KIND("hidden_grid_dims", HIDDEN_GRID_DIMS),
          KIND("hidden_heap_v1", HIDDEN_HEAP_V1),
          KIND("hidden_dynamic_lds_size", HIDDEN_DYNAMIC_LDS_SIZE),
          KIND("hidden_private_base", HIDDEN_PRIVATE_BASE),
          KIND("hidden_shared_base", HIDDEN_SHARED_BASE),
          KIND("hidden_queue_ptr", HIDDEN_QUEUE_PTR),
          KIND("hidden_future_service", UNKNOWN),
          KIND("future_explicit_value", UNKNOWN),
#undef KIND
      };
  for (const auto& [name, kind] : cases) {
    SCOPED_TRACE(name);
    auto kernel = Kernel();
    Set(kernel, ".args", Array({Map(Argument(name))}));
    IREE_ASSERT_OK(Parse(Map(Module(kernel))));
    ASSERT_EQ(metadata_.argument_count, 1u);
    EXPECT_EQ(metadata_.arguments[0].kind, kind);
    EXPECT_EQ(ToString(metadata_.arguments[0].value_kind), name);
  }
}

TEST_F(HsacoMetadataTest, DerivesConservativeAccessWhenQualifiersAreAbsent) {
  for (bool is_const : {false, true}) {
    auto argument = Argument("global_buffer");
    argument.emplace_back(".is_const",
                          Bytes{static_cast<uint8_t>(is_const ? 0xC3 : 0xC2)});
    auto kernel = Kernel();
    Set(kernel, ".args", Array({Map(argument)}));
    IREE_ASSERT_OK(Parse(Map(Module(kernel))));
    EXPECT_EQ(metadata_.arguments[0].effective_access,
              is_const ? IREE_HAL_AMD_GPU_METADATA_ACCESS_READ_ONLY
                       : IREE_HAL_AMD_GPU_METADATA_ACCESS_READ_WRITE);
  }
}

TEST_F(HsacoMetadataTest,
       DecodesAndSortsPrintfRecordsWithoutSplittingFormatColons) {
  auto module = Module();
  module.emplace_back("amdhsa.printf", Array({String("42:2:4:8:value: %u %f\n"),
                                              String("3:0:ready:")}));
  IREE_ASSERT_OK(Parse(Map(module)));
  ASSERT_EQ(metadata_.printf_count, 2u);
  EXPECT_EQ(metadata_.printfs[0].id, 3u);
  EXPECT_EQ(metadata_.printfs[0].argument_count, 0u);
  EXPECT_EQ(ToString(metadata_.printfs[0].format), "ready:");
  const auto& record = metadata_.printfs[1];
  EXPECT_EQ(record.id, 42u);
  ASSERT_EQ(record.argument_count, 2u);
  EXPECT_EQ(record.argument_sizes[0], 4u);
  EXPECT_EQ(record.argument_sizes[1], 8u);
  EXPECT_EQ(ToString(record.format), "value: %u %f\n");
}

TEST_F(HsacoMetadataTest, MergesNotesAndRejectsConflictingModuleDeclarations) {
  auto first = Module();
  first.emplace_back("amdhsa.printf", Array({String("1:1:4:%u")}));
  auto second = Module();
  second.emplace_back("amdhsa.printf", Array({String("2:0:done")}));
  IREE_ASSERT_OK(ParsePayloads({Map(first), Map(second)}));
  EXPECT_EQ(metadata_.kernel_count, 2u);
  EXPECT_EQ(metadata_.printf_count, 2u);
  Set(second, "amdhsa.target", String("amdgcn-amd-amdhsa--gfx1100"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ParsePayloads({Map(first), Map(second)}));
  EXPECT_EQ(metadata_.storage, nullptr);
  second = Module(Kernel(), 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ParsePayloads({Map(first), Map(second)}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ParsePayloads({Map(first), Map(first)}));
}

TEST_F(HsacoMetadataTest, MergesLinkedMetadataWithOptionalTarget) {
  auto first = Module();
  Erase(first, "amdhsa.target");
  auto second_kernel = Kernel();
  Set(second_kernel, ".name", String("second"));
  Set(second_kernel, ".symbol", String("second.kd"));
  auto second = Module(second_kernel);
  IREE_ASSERT_OK(ParsePayloads({Map(first), Map(second)}));
  EXPECT_EQ(ToString(metadata_.target), "amdgcn-amd-amdhsa--gfx942");
  EXPECT_EQ(metadata_.kernel_count, 2u);

  Erase(second, "amdhsa.target");
  IREE_ASSERT_OK(ParsePayloads({Map(first), Map(second)}));
  EXPECT_TRUE(iree_string_view_is_empty(metadata_.target));
  EXPECT_EQ(metadata_.kernel_count, 2u);
}

TEST_F(HsacoMetadataTest, RejectsMalformedPrintfLayout) {
  for (const char* value : {"", "1:", "1:-1:text", "1:1:0:text", "1:2:4:text",
                            "4294967296:0:text", "1:1:4294967296:text"}) {
    SCOPED_TRACE(value);
    auto module = Module();
    module.emplace_back("amdhsa.printf", Array({String(value)}));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(Map(module)));
    EXPECT_EQ(metadata_.storage, nullptr);
  }
}

TEST_F(HsacoMetadataTest, RejectsUnsupportedVersions) {
  auto module = Module();
  Set(module, "amdhsa.version", Array({Integer(2), Integer(0)}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Parse(Map(module)));
  Set(module, "amdhsa.version", Array({Integer(1), Integer(3)}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Parse(Map(module)));
}

TEST_F(HsacoMetadataTest, ChecksRequiredFieldsIndependently) {
  const auto valid = Kernel();
  for (size_t i = 0; i < valid.size(); ++i) {
    SCOPED_TRACE(valid[i].first);
    auto kernel = valid;
    kernel.erase(kernel.begin() + i);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Parse(Map(Module(kernel))));
  }
}

TEST_F(HsacoMetadataTest, PreservesClusterDimensionsBeforeDeviceAdmission) {
  for (uint32_t dimension : {1u, 2u, 256u, UINT32_MAX}) {
    auto kernel = Kernel();
    Set(kernel, ".cluster_dims",
        Array({Integer(1), Integer(dimension), Integer(1)}));
    IREE_ASSERT_OK(Parse(Map(Module(kernel))));
    EXPECT_EQ(metadata_.kernels[0].workgroup_cluster_size[1], dimension);
    EXPECT_EQ(metadata_.kernels[0].has_workgroup_cluster_size, dimension > 1);
  }
}

TEST_F(HsacoMetadataTest, AcceptsLlvmBooleanAndWorkgroupLimitEncodings) {
  for (bool value : {false, true}) {
    for (Bytes encoding :
         {Bytes{static_cast<uint8_t>(value ? 0xC3 : 0xC2)}, Integer(value),
          Bytes{0xCC, static_cast<uint8_t>(value)}}) {
      auto kernel = Kernel();
      Set(kernel, ".workgroup_processor_mode", encoding);
      Set(kernel, ".uses_dynamic_stack", encoding);
      IREE_ASSERT_OK(Parse(Map(Module(kernel))));
      EXPECT_EQ(metadata_.kernels[0].workgroup_processor_mode, value);
      EXPECT_EQ(metadata_.kernels[0].uses_dynamic_stack, value);
    }
  }
  auto kernel = Kernel();
  Set(kernel, ".max_num_workgroups_x", Integer(17));
  Set(kernel, ".max_num_workgroups_y", Integer(23));
  Set(kernel, ".max_num_workgroups_z", Integer(29));
  Set(kernel, ".gfx1250_revision", String("B0"));
  IREE_ASSERT_OK(Parse(Map(Module(kernel))));
  EXPECT_EQ(metadata_.kernels[0].max_workgroup_count[0], 17u);
  EXPECT_EQ(metadata_.kernels[0].max_workgroup_count[1], 23u);
  EXPECT_EQ(metadata_.kernels[0].max_workgroup_count[2], 29u);
  EXPECT_EQ(ToString(metadata_.kernels[0].gfx1250_revision), "B0");
  Set(kernel, ".max_num_work_groups_x", Integer(17));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(Map(Module(kernel))));
}

TEST_F(HsacoMetadataTest,
       ChecksWorkgroupConstraintsAndTreatsZeroShapeAsUnspecified) {
  auto kernel = Kernel();
  Set(kernel, ".reqd_workgroup_size",
      Array({Integer(0), Integer(0), Integer(0)}));
  IREE_ASSERT_OK(Parse(Map(Module(kernel))));
  EXPECT_FALSE(metadata_.kernels[0].has_required_workgroup_size);
  for (const auto& shape : {Array({Integer(0), Integer(1), Integer(1)}),
                            Array({Integer(32), Integer(32), Integer(1)}),
                            Array({Integer(UINT32_MAX), Integer(UINT32_MAX),
                                   Integer(UINT32_MAX)})}) {
    Set(kernel, ".reqd_workgroup_size", shape);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Parse(Map(Module(kernel))));
  }
}

TEST_F(HsacoMetadataTest, RejectsDuplicateFieldsIncludingAlignmentAliases) {
  auto kernel = Kernel();
  kernel.emplace_back(".vgpr_count", Integer(42));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(Map(Module(kernel))));
  auto argument = Argument("by_value");
  argument.emplace_back(".align", Integer(8));
  argument.emplace_back(".alignment", Integer(8));
  kernel = Kernel();
  Set(kernel, ".args", Array({Map(argument)}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(Map(Module(kernel))));
  kernel = Kernel();
  kernel.emplace_back(".args", Array({}));
  kernel.emplace_back(".args", Array({}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(Map(Module(kernel))));
}

TEST_F(HsacoMetadataTest, RejectsInvalidArgumentRangesAndQualifiers) {
  const std::vector<Fields> cases = {
      Argument("by_value", 12, 8),
      Argument("by_value", UINT32_MAX, 8),
      {{".offset", Integer(0)}, {".size", Integer(8)}},
  };
  for (const auto& argument : cases) {
    auto kernel = Kernel();
    Set(kernel, ".args", Array({Map(argument)}));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Parse(Map(Module(kernel))));
  }
  for (const auto& [name, value] : Fields{{".address_space", String("unknown")},
                                          {".access", String("unknown")},
                                          {".pointee_align", Integer(3)},
                                          {".is_const", Integer(2)}}) {
    auto argument = Argument("global_buffer");
    argument.emplace_back(name, value);
    auto kernel = Kernel();
    Set(kernel, ".args", Array({Map(argument)}));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Parse(Map(Module(kernel))));
  }
}

TEST_F(HsacoMetadataTest, AcceptsEveryNonnegativeIntegerEncoding) {
  const std::vector<Bytes> encodings = {
      {16},
      {0xCC, 16},
      {0xCD, 0, 16},
      {0xCE, 0, 0, 0, 16},
      {0xCF, 0, 0, 0, 0, 0, 0, 0, 16},
      {0xD0, 16},
      {0xD1, 0, 16},
      {0xD2, 0, 0, 0, 16},
      {0xD3, 0, 0, 0, 0, 0, 0, 0, 16},
  };
  for (const auto& encoding : encodings) {
    auto kernel = Kernel();
    Set(kernel, ".kernarg_segment_size", encoding);
    IREE_ASSERT_OK(Parse(Map(Module(kernel))));
    EXPECT_EQ(metadata_.kernels[0].kernarg_segment_size, 16u);
  }
  for (const auto& encoding :
       std::vector<Bytes>{{0xFF}, {0xD0, 0xFF}, {0xC3}}) {
    auto kernel = Kernel();
    Set(kernel, ".kernarg_segment_size", encoding);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Parse(Map(Module(kernel))));
  }
  auto kernel = Kernel();
  Set(kernel, ".kernarg_segment_size", Integer(UINT64_C(1) << 32));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Parse(Map(Module(kernel))));
}

TEST_F(HsacoMetadataTest, RejectsEveryTruncatedPrefixAndTrailingBytes) {
  auto payload = Map(Module());
  for (size_t length = 0; length < payload.size(); ++length) {
    SCOPED_TRACE(length);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Parse(Bytes(payload.begin(), payload.begin() + length)));
    EXPECT_EQ(metadata_.storage, nullptr);
  }
  payload.push_back(0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(payload));
}

TEST_F(HsacoMetadataTest, SkipsUnknownDescriptiveFields) {
  auto module = Module();
  module.emplace_back("future",
                      Map({{"nested", Array({String("description"), Bytes{0xC0},
                                             Bytes{0xC4, 2, 1, 2}})}}));
  IREE_ASSERT_OK(Parse(Map(module)));
  EXPECT_EQ(metadata_.kernel_count, 1u);
}

TEST_F(HsacoMetadataTest, AllocationFailureLeavesNoOwnedState) {
  auto payload = Map(Module());
  auto span = iree_make_const_byte_span(payload.data(), payload.size());
  iree_allocator_t allocator = {
      .self = nullptr,
      .ctl = [](void*, iree_allocator_command_t command, const void*,
                void**) -> iree_status_t {
        EXPECT_EQ(command, IREE_ALLOCATOR_COMMAND_CALLOC);
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "allocation refused by test allocator");
      },
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amd_gpu_hsaco_metadata_initialize(
                            1, &span, allocator, &metadata_));
  EXPECT_EQ(metadata_.storage, nullptr);
}

}  // namespace
}  // namespace iree::hal::amd::gpu
