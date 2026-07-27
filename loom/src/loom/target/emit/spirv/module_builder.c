// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_builder.h"

#include "loom/target/emit/spirv/binary_format.h"

void loom_spirv_module_binary_deinitialize(loom_spirv_module_binary_t* module,
                                           iree_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(module);
  iree_allocator_free(allocator, module->words);
  *module = (loom_spirv_module_binary_t){0};
}

static iree_status_t loom_spirv_module_builder_validate_target(
    const loom_target_bundle_t* target) {
  if (target->snapshot->codegen_format != LOOM_TARGET_CODEGEN_FORMAT_SPIRV) {
    iree_string_view_t actual =
        loom_target_codegen_format_name(target->snapshot->codegen_format);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "target '%.*s' uses codegen format '%.*s', "
                            "expected 'spirv'",
                            (int)target->name.size, target->name.data,
                            (int)actual.size, actual.data);
  }
  if (target->snapshot->artifact_format !=
      LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY) {
    iree_string_view_t actual =
        loom_target_artifact_format_name(target->snapshot->artifact_format);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "target '%.*s' emits artifact format '%.*s', "
                            "expected 'spirv_binary'",
                            (int)target->name.size, target->name.data,
                            (int)actual.size, actual.data);
  }
  if (target->export_plan->abi_kind != LOOM_TARGET_ABI_SHADER_ENTRY_POINT &&
      target->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    iree_string_view_t actual =
        loom_target_abi_kind_name(target->export_plan->abi_kind);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "target '%.*s' uses export ABI '%.*s', expected "
                            "'shader_entry_point' or 'hal_kernel'",
                            (int)target->name.size, target->name.data,
                            (int)actual.size, actual.data);
  }
  return iree_ok_status();
}

static bool loom_spirv_module_builder_uses_raw_bda_abi(
    const loom_spirv_module_builder_t* builder) {
  return builder->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL;
}

static bool loom_spirv_module_builder_uses_vulkan_memory_model(
    const loom_spirv_module_builder_t* builder) {
  if (!loom_spirv_module_builder_uses_raw_bda_abi(builder)) {
    return true;
  }
  return loom_spirv_feature_set_has_atom(
      &builder->feature_set, LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR);
}

static bool loom_spirv_module_builder_should_emit_capability(
    const loom_spirv_module_builder_t* builder, uint32_t capability) {
  if (!loom_spirv_module_builder_uses_vulkan_memory_model(builder) &&
      capability == LOOM_SPIRV_CAPABILITY_VULKAN_MEMORY_MODEL) {
    return false;
  }
  return true;
}

static bool loom_spirv_module_builder_should_emit_extension(
    const loom_spirv_module_builder_t* builder, iree_string_view_t extension) {
  if (!loom_spirv_module_builder_uses_vulkan_memory_model(builder) &&
      iree_string_view_equal(extension,
                             IREE_SV("SPV_KHR_vulkan_memory_model"))) {
    return false;
  }
  return true;
}

static void loom_spirv_module_builder_select_memory_model(
    const loom_spirv_module_builder_t* builder, uint32_t* out_addressing_model,
    uint32_t* out_memory_model) {
  if (loom_spirv_module_builder_uses_raw_bda_abi(builder)) {
    *out_addressing_model =
        LOOM_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64;
    *out_memory_model =
        loom_spirv_module_builder_uses_vulkan_memory_model(builder)
            ? LOOM_SPIRV_MEMORY_MODEL_VULKAN
            : LOOM_SPIRV_MEMORY_MODEL_GLSL450;
    return;
  }
  *out_addressing_model = builder->feature_set.addressing_model;
  *out_memory_model = builder->feature_set.memory_model;
}

static loom_spirv_feature_bits_t loom_spirv_module_builder_base_feature_bits(
    const loom_target_bundle_t* target) {
  if (target->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL) {
    return LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA;
  }
  return (loom_spirv_feature_bits_t)target->config->contract_feature_bits;
}

static iree_status_t loom_spirv_module_builder_emit_feature_preamble(
    loom_spirv_module_builder_t* builder) {
  if (!loom_spirv_feature_set_has_atom(&builder->feature_set,
                                       LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V shader module emission requires the Vulkan shader feature");
  }

  loom_spirv_binary_writer_t* capability_writer =
      &builder->sections[LOOM_SPIRV_MODULE_SECTION_CAPABILITY];
  for (uint8_t i = 0; i < builder->feature_set.capability_count; ++i) {
    if (!loom_spirv_module_builder_should_emit_capability(
            builder, builder->feature_set.capabilities[i])) {
      continue;
    }
    const uint32_t operands[] = {builder->feature_set.capabilities[i]};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        capability_writer, LOOM_SPIRV_OP_CAPABILITY, operands,
        IREE_ARRAYSIZE(operands)));
  }

  loom_spirv_binary_writer_t* extension_writer =
      &builder->sections[LOOM_SPIRV_MODULE_SECTION_EXTENSION];
  for (uint8_t i = 0; i < builder->feature_set.extension_count; ++i) {
    if (!loom_spirv_module_builder_should_emit_extension(
            builder, builder->feature_set.extension_names[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_string_instruction(
        extension_writer, LOOM_SPIRV_OP_EXTENSION, NULL, 0,
        builder->feature_set.extension_names[i], NULL, 0));
  }

  loom_spirv_binary_writer_t* memory_model_writer =
      &builder->sections[LOOM_SPIRV_MODULE_SECTION_MEMORY_MODEL];
  uint32_t addressing_model = LOOM_SPIRV_ADDRESSING_MODEL_LOGICAL;
  uint32_t memory_model = LOOM_SPIRV_MEMORY_MODEL_GLSL450;
  loom_spirv_module_builder_select_memory_model(builder, &addressing_model,
                                                &memory_model);
  const uint32_t memory_model_operands[] = {
      addressing_model,
      memory_model,
  };
  return loom_spirv_binary_write_instruction(
      memory_model_writer, LOOM_SPIRV_OP_MEMORY_MODEL, memory_model_operands,
      IREE_ARRAYSIZE(memory_model_operands));
}

iree_status_t loom_spirv_module_builder_initialize(
    const loom_target_bundle_t* target, iree_allocator_t allocator,
    loom_spirv_module_builder_t* out_builder) {
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_builder);

  IREE_RETURN_IF_ERROR(loom_spirv_module_builder_validate_target(target));

  *out_builder = (loom_spirv_module_builder_t){
      .allocator = allocator,
      .target_name = target->name,
      .abi_kind = target->export_plan->abi_kind,
      .required_feature_bits =
          loom_spirv_module_builder_base_feature_bits(target),
      .id_bound = 1,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(out_builder->sections); ++i) {
    loom_spirv_binary_writer_initialize(allocator, &out_builder->sections[i]);
  }
  return iree_ok_status();
}

void loom_spirv_module_builder_deinitialize(
    loom_spirv_module_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(builder->sections); ++i) {
    loom_spirv_binary_writer_deinitialize(&builder->sections[i]);
  }
  *builder = (loom_spirv_module_builder_t){0};
}

void loom_spirv_module_builder_require_feature_bits(
    loom_spirv_module_builder_t* builder,
    loom_spirv_feature_bits_t feature_bits) {
  IREE_ASSERT_ARGUMENT(builder);
  builder->required_feature_bits |= feature_bits;
}

loom_spirv_binary_writer_t* loom_spirv_module_builder_section(
    loom_spirv_module_builder_t* builder, loom_spirv_module_section_t section) {
  IREE_ASSERT_ARGUMENT(builder);
  const uint32_t section_index = (uint32_t)section;
  IREE_ASSERT(section_index < LOOM_SPIRV_MODULE_SECTION_COUNT);
  return &builder->sections[section_index];
}

uint32_t loom_spirv_module_builder_allocate_id(
    loom_spirv_module_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT(builder->id_bound != UINT32_MAX);
  return builder->id_bound++;
}

void loom_spirv_module_builder_require_id_bound(
    loom_spirv_module_builder_t* builder, uint32_t id_bound) {
  IREE_ASSERT_ARGUMENT(builder);
  builder->id_bound = iree_max(builder->id_bound, id_bound);
}

static iree_status_t loom_spirv_module_builder_write_header(
    const loom_spirv_module_builder_t* builder,
    loom_spirv_binary_writer_t* module_writer) {
  const uint32_t header[] = {
      LOOM_SPIRV_MAGIC_NUMBER,    builder->feature_set.minimum_spirv_version,
      LOOM_SPIRV_GENERATOR_LOOM,  builder->id_bound,
      LOOM_SPIRV_SCHEMA_RESERVED,
  };
  return loom_spirv_binary_write_words(module_writer, header,
                                       IREE_ARRAYSIZE(header));
}

iree_status_t loom_spirv_module_builder_finalize(
    loom_spirv_module_builder_t* builder,
    loom_spirv_module_binary_t* out_module) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_module);

  *out_module = (loom_spirv_module_binary_t){0};
  IREE_RETURN_IF_ERROR(loom_spirv_feature_set_prepare(
      builder->target_name, builder->required_feature_bits,
      &builder->feature_set));
  IREE_RETURN_IF_ERROR(
      loom_spirv_module_builder_emit_feature_preamble(builder));

  loom_spirv_binary_writer_t module_writer;
  loom_spirv_binary_writer_initialize(builder->allocator, &module_writer);
  iree_status_t status =
      loom_spirv_module_builder_write_header(builder, &module_writer);
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(builder->sections) && iree_status_is_ok(status);
       ++i) {
    status = loom_spirv_binary_write_words(&module_writer,
                                           builder->sections[i].words,
                                           builder->sections[i].word_count);
  }
  if (iree_status_is_ok(status)) {
    loom_spirv_binary_writer_steal_words(&module_writer, &out_module->words,
                                         &out_module->word_count);
  }
  loom_spirv_binary_writer_deinitialize(&module_writer);
  return status;
}
