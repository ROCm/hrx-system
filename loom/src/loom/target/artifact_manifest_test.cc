// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_manifest.h"

#include <stdint.h>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static iree_string_view_t FormatManifest(
    const loom_target_artifact_manifest_t* manifest,
    loom_target_artifact_manifest_mode_t mode, iree_string_builder_t* builder) {
  iree_string_builder_initialize(iree_allocator_system(), builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_target_artifact_manifest_format_options_t options = {};
  loom_target_artifact_manifest_format_options_initialize(&options);
  options.mode = mode;
  IREE_EXPECT_OK(
      loom_target_artifact_manifest_format_json(manifest, &options, &stream));
  return iree_string_builder_view(builder);
}

TEST(ArtifactManifestTest, ParsesModes) {
  loom_target_artifact_manifest_mode_t mode =
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE;
  IREE_EXPECT_OK(
      loom_target_artifact_manifest_mode_parse(IREE_SV("summary"), &mode));
  EXPECT_EQ(mode, LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY);
  IREE_EXPECT_OK(
      loom_target_artifact_manifest_mode_parse(IREE_SV("details"), &mode));
  EXPECT_EQ(mode, LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS);
  IREE_EXPECT_OK(
      loom_target_artifact_manifest_mode_parse(IREE_SV("analysis"), &mode));
  EXPECT_EQ(mode, LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS);
  IREE_EXPECT_OK(loom_target_artifact_manifest_mode_parse(IREE_SV(""), &mode));
  EXPECT_EQ(mode, LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE);
}

TEST(ArtifactManifestTest, FormatsMinimalSummaryWithoutEmptyFields) {
  loom_target_artifact_manifest_t manifest = {};
  manifest.artifact.format = IREE_SVL("spirv-binary");

  iree_string_builder_t builder;
  iree_string_view_t output = FormatManifest(
      &manifest, LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"summary\","
                      "\"artifact\":{\"format\":\"spirv-binary\"}}")));
  EXPECT_EQ(iree_string_view_find(output, IREE_SV("null"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(output, IREE_SV("[]"), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

TEST(ArtifactManifestTest, FormatsSummaryFacts) {
  const iree_string_view_t target_names[] = {IREE_SVL("gfx1100")};
  const iree_string_view_t feature_names[] = {IREE_SVL("xnack-"),
                                              IREE_SVL("sramecc+")};
  const iree_string_view_t used_global_names[] = {IREE_SVL("q4_table")};

  loom_target_artifact_manifest_parameter_t parameters[2] = {};
  parameters[0].name = IREE_SVL("input");
  parameters[0].kind = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_POINTER;
  parameters[0].type = IREE_SVL("*f16");
  parameters[0].flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_OFFSET |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_LENGTH |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_ALIGNMENT;
  parameters[0].index = 0;
  parameters[0].byte_offset = 0;
  parameters[0].byte_length = 8;
  parameters[0].byte_alignment = 8;
  parameters[1].kind = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_CONSTANT;
  parameters[1].flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_LENGTH;
  parameters[1].index = 1;
  parameters[1].byte_length = 4;

  loom_target_artifact_manifest_target_t targets[1] = {};
  targets[0].name = IREE_SVL("gfx1100");
  targets[0].family = IREE_SVL("amdgpu");
  targets[0].processor = IREE_SVL("gfx1100");
  targets[0].triple = IREE_SVL("amdgcn-amd-amdhsa");
  targets[0].profile = IREE_SVL("code-object-v5");
  targets[0].code_object_target = IREE_SVL("amdgcn-amd-amdhsa--gfx1100");
  targets[0].feature_names = feature_names;
  targets[0].feature_name_count = IREE_ARRAYSIZE(feature_names);

  loom_target_artifact_manifest_function_t functions[1] = {};
  functions[0].name = IREE_SVL("moe_dispatch");
  functions[0].source_name = IREE_SVL("moe_dispatch");
  functions[0].target_names = target_names;
  functions[0].target_name_count = IREE_ARRAYSIZE(target_names);
  functions[0].interface.flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_PARAMETER_COUNT |
      LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_BINDING_COUNT |
      LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_CONSTANT_BYTE_LENGTH;
  functions[0].interface.parameter_count = 3;
  functions[0].interface.binding_count = 2;
  functions[0].interface.constant_byte_length = 16;
  functions[0].interface.parameters = parameters;
  functions[0].interface.parameter_detail_count = IREE_ARRAYSIZE(parameters);
  functions[0].execution.flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_WORKGROUP_SIZE |
      LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_SUBGROUP_SIZE;
  functions[0].execution.workgroup_size[0] = 256;
  functions[0].execution.workgroup_size[1] = 1;
  functions[0].execution.workgroup_size[2] = 1;
  functions[0].execution.subgroup_size = 32;
  functions[0].used_global_names = used_global_names;
  functions[0].used_global_name_count = IREE_ARRAYSIZE(used_global_names);

  loom_target_artifact_manifest_global_t globals[1] = {};
  globals[0].name = IREE_SVL("q4_table");
  globals[0].source_name = IREE_SVL("q4_decode_table");
  globals[0].type = IREE_SVL("u8[32]");
  globals[0].target_names = target_names;
  globals[0].target_name_count = IREE_ARRAYSIZE(target_names);
  globals[0].flags = LOOM_TARGET_ARTIFACT_MANIFEST_GLOBAL_FLAG_BYTE_LENGTH |
                     LOOM_TARGET_ARTIFACT_MANIFEST_GLOBAL_FLAG_BYTE_ALIGNMENT;
  globals[0].byte_length = 32;
  globals[0].byte_alignment = 4;

  loom_target_artifact_manifest_t manifest = {};
  manifest.artifact.format = IREE_SVL("amdgpu-hsaco");
  manifest.artifact.name = IREE_SVL("moe");
  manifest.artifact.flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_ARTIFACT_FLAG_BYTE_LENGTH;
  manifest.artifact.byte_length = 8192;
  manifest.targets = targets;
  manifest.target_count = IREE_ARRAYSIZE(targets);
  manifest.functions = functions;
  manifest.function_count = IREE_ARRAYSIZE(functions);
  manifest.globals = globals;
  manifest.global_count = IREE_ARRAYSIZE(globals);

  iree_string_builder_t builder;
  iree_string_view_t output = FormatManifest(
      &manifest, LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"summary\","
                      "\"artifact\":{\"format\":\"amdgpu-hsaco\","
                      "\"name\":\"moe\","
                      "\"byte_length\":8192},"
                      "\"targets\":[{\"name\":\"gfx1100\","
                      "\"family\":\"amdgpu\","
                      "\"processor\":\"gfx1100\","
                      "\"triple\":\"amdgcn-amd-amdhsa\","
                      "\"profile\":\"code-object-v5\","
                      "\"code_object_target\":\"amdgcn-amd-amdhsa--gfx1100\","
                      "\"features\":[\"xnack-\",\"sramecc+\"]}],"
                      "\"functions\":[{\"name\":\"moe_dispatch\","
                      "\"targets\":[\"gfx1100\"],"
                      "\"interface\":{\"parameter_count\":3,"
                      "\"binding_count\":2,"
                      "\"constant_byte_length\":16},"
                      "\"execution\":{\"workgroup_size\":[256,1,1],"
                      "\"subgroup_size\":32}}],"
                      "\"globals\":[{\"name\":\"q4_table\","
                      "\"source\":\"q4_decode_table\","
                      "\"type\":\"u8[32]\","
                      "\"targets\":[\"gfx1100\"],"
                      "\"byte_length\":32,"
                      "\"byte_alignment\":4}]}")));
  EXPECT_EQ(iree_string_view_find(output, IREE_SV("parameters"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(output, IREE_SV("uses_globals"), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

TEST(ArtifactManifestTest, FormatsDetailsFacts) {
  const iree_string_view_t target_names[] = {IREE_SVL("gfx942")};
  const iree_string_view_t used_global_names[] = {IREE_SVL("scale_table")};

  loom_target_artifact_manifest_target_t targets[1] = {};
  targets[0].name = IREE_SVL("gfx942");
  targets[0].family = IREE_SVL("amdgpu");
  targets[0].processor = IREE_SVL("gfx942");
  targets[0].flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_DEFAULT_POINTER_BITWIDTH |
      LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_INDEX_BITWIDTH |
      LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_OFFSET_BITWIDTH |
      LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_SUBGROUP_SIZE |
      LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MEMORY_SPACES;
  targets[0].default_pointer_bitwidth = 64;
  targets[0].index_bitwidth = 32;
  targets[0].offset_bitwidth = 64;
  targets[0].subgroup_size = 64;
  targets[0].memory_spaces.generic = 0;
  targets[0].memory_spaces.global = 1;
  targets[0].memory_spaces.workgroup = 3;
  targets[0].memory_spaces.constant = 4;
  targets[0].memory_spaces.private_memory = 5;
  targets[0].memory_spaces.host = UINT32_MAX;
  targets[0].memory_spaces.descriptor = 7;

  loom_target_artifact_manifest_parameter_t parameters[2] = {};
  parameters[0].name = IREE_SVL("weights");
  parameters[0].kind = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_POINTER;
  parameters[0].type = IREE_SVL("*u8");
  parameters[0].flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_OFFSET |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_LENGTH |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_ALIGNMENT;
  parameters[0].index = 0;
  parameters[0].byte_offset = 0;
  parameters[0].byte_length = 8;
  parameters[0].byte_alignment = 8;
  parameters[1].kind = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_CONSTANT;
  parameters[1].flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_LENGTH;
  parameters[1].index = 1;
  parameters[1].byte_length = 4;

  loom_target_artifact_manifest_function_t functions[1] = {};
  functions[0].name = IREE_SVL("gemv_q4");
  functions[0].source_name = IREE_SVL("gemv_q4_template");
  functions[0].target_names = target_names;
  functions[0].target_name_count = IREE_ARRAYSIZE(target_names);
  functions[0].interface.parameters = parameters;
  functions[0].interface.parameter_detail_count = IREE_ARRAYSIZE(parameters);
  functions[0].used_global_names = used_global_names;
  functions[0].used_global_name_count = IREE_ARRAYSIZE(used_global_names);

  loom_target_artifact_manifest_t manifest = {};
  manifest.artifact.format = IREE_SVL("amdgpu-hsaco");
  manifest.targets = targets;
  manifest.target_count = IREE_ARRAYSIZE(targets);
  manifest.functions = functions;
  manifest.function_count = IREE_ARRAYSIZE(functions);

  iree_string_builder_t builder;
  iree_string_view_t output = FormatManifest(
      &manifest, LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"details\","
                      "\"artifact\":{\"format\":\"amdgpu-hsaco\"},"
                      "\"targets\":[{\"name\":\"gfx942\","
                      "\"family\":\"amdgpu\","
                      "\"processor\":\"gfx942\","
                      "\"default_pointer_bitwidth\":64,"
                      "\"index_bitwidth\":32,"
                      "\"offset_bitwidth\":64,"
                      "\"subgroup_size\":64,"
                      "\"address_spaces\":{\"generic\":0,"
                      "\"global\":1,"
                      "\"workgroup\":3,"
                      "\"constant\":4,"
                      "\"private\":5,"
                      "\"descriptor\":7}}],"
                      "\"functions\":[{\"name\":\"gemv_q4\","
                      "\"source\":\"gemv_q4_template\","
                      "\"targets\":[\"gfx942\"],"
                      "\"interface\":{\"parameters\":["
                      "{\"name\":\"weights\","
                      "\"kind\":\"pointer\","
                      "\"type\":\"*u8\","
                      "\"index\":0,"
                      "\"byte_offset\":0,"
                      "\"byte_length\":8,"
                      "\"byte_alignment\":8},"
                      "{\"kind\":\"constant\","
                      "\"index\":1,"
                      "\"byte_length\":4}]},"
                      "\"uses_globals\":[\"scale_table\"]}]}")));
  iree_string_builder_deinitialize(&builder);
}

TEST(ArtifactManifestTest, EmitsFlaggedZeroFacts) {
  loom_target_artifact_manifest_function_t functions[1] = {};
  functions[0].name = IREE_SVL("zero_budget_kernel");
  functions[0].interface.flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_PARAMETER_COUNT |
      LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_BINDING_COUNT |
      LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_CONSTANT_BYTE_LENGTH;

  loom_target_artifact_manifest_global_t globals[1] = {};
  globals[0].name = IREE_SVL("empty_table");
  globals[0].flags = LOOM_TARGET_ARTIFACT_MANIFEST_GLOBAL_FLAG_BYTE_LENGTH;

  loom_target_artifact_manifest_t manifest = {};
  manifest.artifact.format = IREE_SVL("amdgpu-hsaco");
  manifest.artifact.flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_ARTIFACT_FLAG_BYTE_LENGTH;
  manifest.functions = functions;
  manifest.function_count = IREE_ARRAYSIZE(functions);
  manifest.globals = globals;
  manifest.global_count = IREE_ARRAYSIZE(globals);

  iree_string_builder_t builder;
  iree_string_view_t output = FormatManifest(
      &manifest, LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"summary\","
                      "\"artifact\":{\"format\":\"amdgpu-hsaco\","
                      "\"byte_length\":0},"
                      "\"functions\":[{\"name\":\"zero_budget_kernel\","
                      "\"interface\":{\"parameter_count\":0,"
                      "\"binding_count\":0,"
                      "\"constant_byte_length\":0}}],"
                      "\"globals\":[{\"name\":\"empty_table\","
                      "\"byte_length\":0}]}")));
  iree_string_builder_deinitialize(&builder);
}

TEST(ArtifactManifestTest, RejectsUndeclaredTargetReferences) {
  const iree_string_view_t target_names[] = {IREE_SVL("gfx999")};

  loom_target_artifact_manifest_target_t targets[1] = {};
  targets[0].name = IREE_SVL("gfx942");

  loom_target_artifact_manifest_function_t functions[1] = {};
  functions[0].name = IREE_SVL("gemv_q4");
  functions[0].target_names = target_names;
  functions[0].target_name_count = IREE_ARRAYSIZE(target_names);

  loom_target_artifact_manifest_t manifest = {};
  manifest.artifact.format = IREE_SVL("amdgpu-hsaco");
  manifest.targets = targets;
  manifest.target_count = IREE_ARRAYSIZE(targets);
  manifest.functions = functions;
  manifest.function_count = IREE_ARRAYSIZE(functions);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_artifact_manifest_format_options_t options = {
      /*.mode=*/LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_artifact_manifest_format_json(&manifest, &options, &stream));
  iree_string_builder_deinitialize(&builder);
}

TEST(ArtifactManifestTest, NoneModeWritesNothing) {
  loom_target_artifact_manifest_t manifest = {};
  manifest.artifact.format = IREE_SVL("amdgpu-hsaco");

  iree_string_builder_t builder;
  iree_string_view_t output = FormatManifest(
      &manifest, LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE, &builder);
  EXPECT_TRUE(iree_string_view_is_empty(output));
  iree_string_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace loom
