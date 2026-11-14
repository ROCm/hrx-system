// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_builder.h"

#include "loom/target/artifact_diagnostics.h"
#include "loom/target/emit/spirv/binary_format.h"

void loom_spirv_module_binary_deinitialize(loom_spirv_module_binary_t* module,
                                           iree_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(module);
  iree_allocator_free(allocator, module->words);
  *module = (loom_spirv_module_binary_t){0};
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
             &builder->feature_set,
             LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR) ||
         loom_spirv_feature_set_has_atom(
             &builder->feature_set,
             LOOM_SPIRV_FEATURE_ATOM_VULKAN_MEMORY_MODEL_DEVICE_SCOPE);
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
    return LOOM_SPIRV_FEATURE_MODULE_VULKAN_1_3_BDA_BASELINE;
  }

  // Shader entry-point modules inherit their target contract features. Features
  // used only by particular operations or entry-point execution modes remain
  // demand-driven through their descriptors so unrelated modules do not declare
  // unused capabilities and extensions.
  const loom_spirv_feature_bits_t demand_driven_feature_bits =
      LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM |
      LOOM_SPIRV_FEATURE_FLOAT32_DENORM_PRESERVE |
      LOOM_SPIRV_FEATURE_FLOAT16_64_DENORM_PRESERVE |
      LOOM_SPIRV_FEATURE_FLOAT16_64_ROUNDING_RTE |
      LOOM_SPIRV_FEATURE_FLOAT32_ROUNDING_RTE |
      LOOM_SPIRV_FEATURE_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE |
      LOOM_SPIRV_FEATURE_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE |
      LOOM_SPIRV_FEATURE_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE |
      LOOM_SPIRV_FEATURE_FMA_FLOAT16 | LOOM_SPIRV_FEATURE_FMA_FLOAT32 |
      LOOM_SPIRV_FEATURE_FMA_FLOAT64 | LOOM_SPIRV_FEATURE_FLOAT_CONTROLS2 |
      LOOM_SPIRV_FEATURE_FLOAT16_DENORM_PRESERVE |
      LOOM_SPIRV_FEATURE_FLOAT64_DENORM_PRESERVE |
      LOOM_SPIRV_FEATURE_FLOAT16_ROUNDING_RTE |
      LOOM_SPIRV_FEATURE_FLOAT64_ROUNDING_RTE |
      LOOM_SPIRV_FEATURE_FLOAT_ALL_DENORM_PRESERVE |
      LOOM_SPIRV_FEATURE_FLOAT_ALL_ROUNDING_RTE;
  return (loom_spirv_feature_bits_t)target->config->contract_feature_bits &
         ~demand_driven_feature_bits;
}

static iree_status_t loom_spirv_module_builder_emit_feature_preamble(
    loom_spirv_module_builder_t* builder) {
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

void loom_spirv_module_builder_initialize(
    const loom_target_bundle_t* target, iree_allocator_t allocator,
    loom_spirv_module_builder_t* out_builder) {
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_builder);

  *out_builder = (loom_spirv_module_builder_t){
      .allocator = allocator,
      .abi_kind = target->export_plan->abi_kind,
      .required_feature_bits =
          loom_spirv_module_builder_base_feature_bits(target),
      .id_bound = 1,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(out_builder->sections); ++i) {
    loom_spirv_binary_writer_initialize(allocator, &out_builder->sections[i]);
  }
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

iree_status_t loom_spirv_module_builder_import_extended_instruction_set(
    loom_spirv_module_builder_t* builder,
    loom_spirv_extended_instruction_set_t instruction_set,
    uint32_t* out_result_id) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_result_id);
  IREE_ASSERT(instruction_set > LOOM_SPIRV_EXTENDED_INSTRUCTION_SET_UNKNOWN &&
              instruction_set < LOOM_SPIRV_EXTENDED_INSTRUCTION_SET_COUNT);

  uint32_t* result_id = &builder->extended_instruction_set_ids[instruction_set];
  if (*result_id == 0) {
    static const iree_string_view_t kImportNames[] = {
        [LOOM_SPIRV_EXTENDED_INSTRUCTION_SET_GLSL_STD_450] =
            IREE_SVL("GLSL.std.450"),
    };
    const uint32_t new_result_id =
        loom_spirv_module_builder_allocate_id(builder);
    const uint32_t prefix_operands[] = {new_result_id};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_string_instruction(
        loom_spirv_module_builder_section(
            builder, LOOM_SPIRV_MODULE_SECTION_EXTENDED_INSTRUCTION_IMPORT),
        LOOM_SPIRV_OP_EXT_INST_IMPORT, prefix_operands,
        IREE_ARRAYSIZE(prefix_operands), kImportNames[instruction_set], NULL,
        0));
    *result_id = new_result_id;
  }
  *out_result_id = *result_id;
  return iree_ok_status();
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

iree_status_t loom_spirv_module_builder_emit_string(
    loom_spirv_module_builder_t* builder, loom_spirv_module_section_t section,
    uint16_t opcode, const uint32_t* prefix_operands,
    iree_host_size_t prefix_operand_count, iree_string_view_t string,
    const uint32_t* suffix_operands, iree_host_size_t suffix_operand_count,
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_op_t* source_op,
    bool* accepted) {
  // The header and fixed operands leave the remaining words for a NUL-padded
  // string. All fixed operand spans are bounded by their instruction schema.
  const iree_host_size_t maximum_length =
      (UINT16_MAX - 1u - prefix_operand_count - suffix_operand_count) * 4u - 1u;
  if (string.size > maximum_length) {
    *accepted = false;
    return loom_target_artifact_emit_limit(
        diagnostic_emitter, source_op, IREE_SV("SPIR-V"),
        IREE_SV("instruction string bytes"), string.size, maximum_length);
  }
  return loom_spirv_binary_write_string_instruction(
      loom_spirv_module_builder_section(builder, section), opcode,
      prefix_operands, prefix_operand_count, string, suffix_operands,
      suffix_operand_count);
}

iree_status_t loom_spirv_module_builder_finalize(
    loom_spirv_module_builder_t* builder,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_accepted,
    loom_spirv_module_binary_t* out_module) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_module);

  *out_module = (loom_spirv_module_binary_t){0};
  *out_accepted = false;
  loom_spirv_feature_set_prepare(builder->required_feature_bits,
                                 &builder->feature_set);
  if (!loom_spirv_feature_set_has_atom(&builder->feature_set,
                                       LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER)) {
    return loom_target_artifact_emit_requirement(
        diagnostic_emitter, NULL, IREE_SV("SPIR-V"),
        IREE_SV("the Vulkan shader feature"));
  }
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
    *out_accepted = true;
    loom_spirv_binary_writer_steal_words(&module_writer, &out_module->words,
                                         &out_module->word_count);
  }
  loom_spirv_binary_writer_deinitialize(&module_writer);
  return status;
}
