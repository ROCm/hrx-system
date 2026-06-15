// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/testing/text_asm_test_util.h"

#include "loom/codegen/low/text_asm.h"

namespace loom::testing {

LowTextAsmTypeInferenceHarness::~LowTextAsmTypeInferenceHarness() {
  Deinitialize();
}

iree_status_t LowTextAsmTypeInferenceHarness::Initialize(
    loom_low_descriptor_set_provider_t descriptor_set_provider) {
  if (module_ || context_initialized_ || block_pool_initialized_) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low asm type inference harness is already "
                            "initialized");
  }
  if (!descriptor_set_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "descriptor set provider must not be NULL");
  }

  descriptor_set_provider_ = descriptor_set_provider;
  descriptor_registry_ = {};
  descriptor_registry_.descriptor_set_providers = &descriptor_set_provider_;
  descriptor_registry_.descriptor_set_provider_count = 1;

  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool_);
  block_pool_initialized_ = true;
  loom_context_initialize(iree_allocator_system(), &context_);
  context_initialized_ = true;

  iree_status_t status = loom_context_finalize(&context_);
  if (iree_status_is_ok(status)) {
    status = loom_module_allocate(
        &context_, IREE_SV("low_asm_type_inference_test"), &block_pool_,
        nullptr, iree_allocator_system(), &module_);
  }
  if (iree_status_is_ok(status)) {
    loom_low_descriptor_text_asm_environment_initialize(&descriptor_registry_,
                                                        &environment_);
  } else {
    Deinitialize();
  }
  return status;
}

void LowTextAsmTypeInferenceHarness::Deinitialize() {
  if (module_) {
    loom_module_free(module_);
    module_ = nullptr;
  }
  environment_ = {};
  descriptor_registry_ = {};
  descriptor_set_provider_ = nullptr;
  if (context_initialized_) {
    loom_context_deinitialize(&context_);
    context_ = {};
    context_initialized_ = false;
  }
  if (block_pool_initialized_) {
    iree_arena_block_pool_deinitialize(&block_pool_);
    block_pool_ = {};
    block_pool_initialized_ = false;
  }
}

iree_status_t LowTextAsmTypeInferenceHarness::LookupPacket(
    iree_string_view_t descriptor_set_key, iree_string_view_t mnemonic,
    loom_text_low_asm_packet_descriptor_t* out_packet) const {
  const loom_text_low_asm_descriptor_set_t* descriptor_set = nullptr;
  IREE_RETURN_IF_ERROR(environment_.vtable->lookup_descriptor_set(
      environment_.state, descriptor_set_key, &descriptor_set));
  return environment_.vtable->lookup_packet(environment_.state, descriptor_set,
                                            mnemonic, out_packet);
}

iree_status_t LowTextAsmTypeInferenceHarness::MakeRegisterType(
    iree_string_view_t reg_class_name, uint16_t unit_count,
    loom_type_t* out_type) const {
  const loom_low_descriptor_set_t* descriptor_set = descriptor_set_provider_();
  const loom_text_low_asm_descriptor_set_t* descriptor_set_handle = nullptr;
  IREE_RETURN_IF_ERROR(environment_.vtable->lookup_descriptor_set(
      environment_.state,
      loom_low_descriptor_set_string(descriptor_set,
                                     descriptor_set->key_string_offset),
      &descriptor_set_handle));
  bool found = false;
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(environment_.vtable->resolve_register_type(
      environment_.state, descriptor_set_handle, reg_class_name, unit_count,
      &type, &found));
  if (!found) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "register class '%.*s' was not found",
                            (int)reg_class_name.size, reg_class_name.data);
  }
  return loom_module_intern_type(module_, type, out_type);
}

iree_status_t LowTextAsmTypeInferenceHarness::DefineRegisterValue(
    iree_string_view_t reg_class_name, uint16_t unit_count,
    loom_value_id_t* out_value_id) const {
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(MakeRegisterType(reg_class_name, unit_count, &type));
  return loom_module_define_value(module_, type, out_value_id);
}

iree_status_t LowTextAsmTypeInferenceHarness::InferResultType(
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_type_t* out_type,
    iree_string_view_t* out_diagnostic_detail) const {
  return environment_.vtable->infer_result_type(
      environment_.state, packet, operands, operand_count, result_index,
      module_, out_type, out_diagnostic_detail);
}

iree_status_t LowTextAsmTypeInferenceHarness::ValidateResultType(
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_type_t type,
    iree_string_view_t* out_diagnostic_detail) const {
  return environment_.vtable->validate_result_type(
      environment_.state, packet, operands, operand_count, result_index,
      module_, type, out_diagnostic_detail);
}

bool LowTextAsmTypeInferenceHarness::RegisterTypeEquals(
    loom_type_t type, iree_string_view_t reg_class_name,
    uint32_t unit_count) const {
  if (!loom_type_is_register(type)) return false;
  const loom_low_descriptor_set_t* descriptor_set = descriptor_set_provider_();
  const loom_text_low_asm_descriptor_set_t* descriptor_set_handle = nullptr;
  if (!iree_status_is_ok(environment_.vtable->lookup_descriptor_set(
          environment_.state,
          loom_low_descriptor_set_string(descriptor_set,
                                         descriptor_set->key_string_offset),
          &descriptor_set_handle))) {
    return false;
  }
  iree_string_view_t actual_class_name = iree_string_view_empty();
  uint32_t actual_unit_count = 0;
  bool found = false;
  if (!iree_status_is_ok(environment_.vtable->describe_register_type(
          environment_.state, descriptor_set_handle, type, &actual_class_name,
          &actual_unit_count, &found))) {
    return false;
  }
  return found && actual_unit_count == unit_count &&
         iree_string_view_equal(actual_class_name, reg_class_name);
}

}  // namespace loom::testing
