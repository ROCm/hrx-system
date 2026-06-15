// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test helpers for descriptor-backed low asm type inference.

#ifndef LOOM_CODEGEN_LOW_TESTING_TEXT_ASM_TEST_UTIL_H_
#define LOOM_CODEGEN_LOW_TESTING_TEXT_ASM_TEST_UTIL_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/format/text/low_asm.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom::testing {

class LowTextAsmTypeInferenceHarness {
 public:
  LowTextAsmTypeInferenceHarness() = default;
  LowTextAsmTypeInferenceHarness(const LowTextAsmTypeInferenceHarness&) =
      delete;
  LowTextAsmTypeInferenceHarness& operator=(
      const LowTextAsmTypeInferenceHarness&) = delete;

  ~LowTextAsmTypeInferenceHarness();

  // Initializes a minimal module and low asm environment using one descriptor
  // set provider.
  iree_status_t Initialize(
      loom_low_descriptor_set_provider_t descriptor_set_provider);

  // Releases all module, context, and arena state. Safe to call repeatedly.
  void Deinitialize();

  loom_module_t* module() const { return module_; }

  // Looks up an asm packet by descriptor-set key and mnemonic.
  iree_status_t LookupPacket(
      iree_string_view_t descriptor_set_key, iree_string_view_t mnemonic,
      loom_text_low_asm_packet_descriptor_t* out_packet) const;

  // Interns a canonical register type in the harness module.
  iree_status_t MakeRegisterType(iree_string_view_t reg_class_name,
                                 uint16_t unit_count,
                                 loom_type_t* out_type) const;

  // Defines a fresh SSA value with a canonical register type.
  iree_status_t DefineRegisterValue(iree_string_view_t reg_class_name,
                                    uint16_t unit_count,
                                    loom_value_id_t* out_value_id) const;

  // Runs descriptor-backed result type inference for one packet result.
  iree_status_t InferResultType(
      const loom_text_low_asm_packet_descriptor_t* packet,
      const loom_value_id_t* operands, iree_host_size_t operand_count,
      uint16_t result_index, loom_type_t* out_type,
      iree_string_view_t* out_diagnostic_detail) const;

  // Runs descriptor-backed validation for one explicit packet result type.
  iree_status_t ValidateResultType(
      const loom_text_low_asm_packet_descriptor_t* packet,
      const loom_value_id_t* operands, iree_host_size_t operand_count,
      uint16_t result_index, loom_type_t type,
      iree_string_view_t* out_diagnostic_detail) const;

  // Returns true when |type| is a register of |reg_class_name| and
  // |unit_count| allocation units in the harness module.
  bool RegisterTypeEquals(loom_type_t type, iree_string_view_t reg_class_name,
                          uint32_t unit_count) const;

 private:
  // Arena block pool backing the temporary module.
  iree_arena_block_pool_t block_pool_ = {};
  // Minimal context used by the temporary module.
  loom_context_t context_ = {};
  // Single descriptor-set provider borrowed from the target package.
  loom_low_descriptor_set_provider_t descriptor_set_provider_ = nullptr;
  // Registry view exposing |descriptor_set_provider_|.
  loom_low_descriptor_registry_t descriptor_registry_ = {};
  // Parser-facing low asm environment backed by |descriptor_registry_|.
  loom_text_low_asm_environment_t environment_ = {};
  // Temporary module whose string/type/value tables drive inference.
  loom_module_t* module_ = nullptr;
  // True after |block_pool_| has been initialized.
  bool block_pool_initialized_ = false;
  // True after |context_| has been initialized.
  bool context_initialized_ = false;
};

}  // namespace loom::testing

#endif  // LOOM_CODEGEN_LOW_TESTING_TEXT_ASM_TEST_UTIL_H_
