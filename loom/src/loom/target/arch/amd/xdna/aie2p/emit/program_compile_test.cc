// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_compile.h"

#include <algorithm>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"
#include "loom/target/arch/amd/xdna/aie2p/provider.h"
#include "loom/target/emit/native/object.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::loom::testing::ModulePtr;

constexpr char kDirectHelperSource[] = R"(
aie2p.target<core> @target

low.func.def target<amd.xdna.aie2p.core>(@target) @add_pair(%lhs: reg<aie2p.er>, %rhs: reg<aie2p.er>) -> (reg<aie2p.er>) asm {
  %sum = add.rr %lhs, %rhs
  return %sum
}

low.func.def target<amd.xdna.aie2p.core>(@target) abi(object_function) @entry() -> (reg<aie2p.er>) asm {
  %lhs = mov.i32 7
  %rhs = mov.i32 9
  %sum = low.func.call @add_pair(%lhs, %rhs) : (reg<aie2p.er>, reg<aie2p.er>) -> (reg<aie2p.er>)
  %result = add %sum, 1
  low.br ^finish(%result: reg<aie2p.er>)
^return(%final_result: reg<aie2p.er>):
  return %final_result
^finish(%finished_result: reg<aie2p.er>):
  low.br ^return(%finished_result: reg<aie2p.er>)
}
)";

constexpr char kPointerHelperSource[] = R"(
aie2p.target<core> @target

low.func.def target<amd.xdna.aie2p.core>(@target) @sum_streams(%first_base: reg<aie2p.ep>, %second_base: reg<aie2p.ep>, %vector_output: reg<aie2p.ep>, %scalar_output: reg<aie2p.ep>) asm {
  %first_pointer = copy %first_base : reg<aie2p.ep> -> reg<aie2p.eps>
  %first_low = vlda.load-fifo.low512 %first_base, 0
  %first_fifo = vlda.load-fifo.high512 %first_base, %first_low, 64
  %first_position = mova.fifo.position 0
  %second_pointer = copy %second_base : reg<aie2p.ep> -> reg<aie2p.eps>
  %second_low = vlda.load-fifo.low512 %second_base, 0
  %second_fifo = vlda.load-fifo.high512 %second_base, %second_low, 64
  %second_position = mova.fifo.position 0
  %first, %first_pointer1, %first_fifo1, %first_position1 = vlda.pop.512.i32x16 %first_pointer, %first_fifo, %first_position
  %second, %second_pointer1, %second_fifo1, %second_position1 = vldb.pop.512.i32x16 %second_pointer, %second_fifo, %second_position
  %combined = vadd.32 %first, %second
  vst.512.i32x16 %combined, %vector_output, 0
  %complete = mov.i32 1
  st %complete, %scalar_output, 0
  return
}

low.func.def target<amd.xdna.aie2p.core>(@target) abi(object_function) @entry() asm {
  %first_base = mov.local-address 458752
  %second_base = mov.local-address 458880
  %vector_output = mov.local-address 459008
  %scalar_output = mov.local-address 459072
  low.func.call @sum_streams(%first_base, %second_base, %vector_output, %scalar_output) : (reg<aie2p.ep>, reg<aie2p.ep>, reg<aie2p.ep>, reg<aie2p.ep>)
  return
}
)";

iree_status_t InitializeXdnaContext(loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  iree_status_t status = loom_op_registry_register_all_dialects(context);
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_target_provider.register_context(context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(context);
  }
  return status;
}

struct DecodedProgramBundle {
  iree_host_size_t byte_offset;
  iree_host_size_t byte_length;
  loom_aie2p_decoded_bundle_t bundle;
};

iree_status_t DecodeProgram(iree_const_byte_span_t code,
                            std::vector<DecodedProgramBundle>* out_bundles) {
  out_bundles->clear();
  for (iree_host_size_t offset = 0; offset < code.data_length;) {
    DecodedProgramBundle decoded = {.byte_offset = offset};
    const iree_host_size_t remaining = code.data_length - offset;
    IREE_RETURN_IF_ERROR(loom_aie2p_encoding_decode_bundle_prefix(
        iree_make_const_byte_span(
            code.data + offset,
            iree_min(remaining,
                     (iree_host_size_t)LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE)),
        &decoded.bundle, &decoded.byte_length));
    out_bundles->push_back(decoded);
    offset += decoded.byte_length;
  }
  return iree_ok_status();
}

bool SlotMayBeInstruction(const loom_aie2p_encoded_slot_t& slot,
                          loom_aie2p_instruction_id_t instruction) {
  loom_aie2p_instruction_id_t candidates[16];
  const iree_host_size_t count =
      loom_aie2p_encoding_query_instruction_candidates(
          slot.slot, slot.value, IREE_ARRAYSIZE(candidates), candidates);
  if (count > IREE_ARRAYSIZE(candidates)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (candidates[i] == instruction) {
      return true;
    }
  }
  return false;
}

bool BundleHasInstruction(const DecodedProgramBundle& bundle,
                          loom_aie2p_instruction_id_t instruction) {
  for (uint8_t i = 0; i < bundle.bundle.slot_count; ++i) {
    if (SlotMayBeInstruction(bundle.bundle.slots[i], instruction)) {
      return true;
    }
  }
  return false;
}

const loom_aie2p_encoded_slot_t* FindInstructionSlot(
    const DecodedProgramBundle& bundle,
    loom_aie2p_instruction_id_t instruction) {
  for (uint8_t i = 0; i < bundle.bundle.slot_count; ++i) {
    if (SlotMayBeInstruction(bundle.bundle.slots[i], instruction)) {
      return &bundle.bundle.slots[i];
    }
  }
  return nullptr;
}

bool BundleHasUsefulInstruction(const DecodedProgramBundle& bundle) {
  for (uint8_t i = 0; i < bundle.bundle.slot_count; ++i) {
    const loom_aie2p_encoded_slot_t& slot = bundle.bundle.slots[i];
    loom_aie2p_instruction_id_t candidates[16];
    const iree_host_size_t count =
        loom_aie2p_encoding_query_instruction_candidates(
            slot.slot, slot.value, IREE_ARRAYSIZE(candidates), candidates);
    if (count > IREE_ARRAYSIZE(candidates)) {
      return false;
    }
    for (iree_host_size_t j = 0; j < count; ++j) {
      loom_aie2p_instruction_info_t info;
      if (!loom_aie2p_encoding_query_instruction_info(candidates[j], &info)) {
        return false;
      }
      if (!iree_string_view_starts_with(info.name, IREE_SV("NOP"))) {
        return true;
      }
    }
  }
  return false;
}

loom_aie2p_encoded_slot_t EncodeMove(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_aie2p_physical_register_id_t source,
    loom_aie2p_physical_register_id_t destination) {
  const uint32_t descriptor_ordinal =
      loom_aie2p_descriptor_select_move(source, destination);
  EXPECT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const loom_low_allocation_assignment_t destination_assignment = {
      .unit_count = 1,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = destination,
      .location_count = 1,
  };
  const loom_low_allocation_assignment_t source_assignment = {
      .unit_count = 1,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = source,
      .location_count = 1,
  };
  const loom_low_allocation_assignment_t* assignments[] = {
      &destination_assignment,
      &source_assignment,
  };
  return loom_aie2p_descriptor_encode(descriptor_set, descriptor_ordinal,
                                      assignments, nullptr);
}

bool BundleHasEncodedSlot(const DecodedProgramBundle& bundle,
                          loom_aie2p_encoded_slot_t expected) {
  for (uint8_t i = 0; i < bundle.bundle.slot_count; ++i) {
    const loom_aie2p_encoded_slot_t& actual = bundle.bundle.slots[i];
    if (actual.slot == expected.slot && actual.value == expected.value) {
      return true;
    }
  }
  return false;
}

class Aie2pProgramCompileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &module_block_pool_);
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &program_block_pool_);
    iree_arena_initialize(&program_block_pool_, &program_arena_);
    IREE_ASSERT_OK(InitializeXdnaContext(&context_));
    loom_aie2p_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&program_arena_);
    iree_arena_block_pool_deinitialize(&program_block_pool_);
    iree_arena_block_pool_deinitialize(&module_block_pool_);
  }

  ModulePtr ParseModule(iree_string_view_t source) {
    loom_text_parse_options_t options = {
        .diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr},
        .max_errors = 20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("program_compile_test.loom"),
                                  &context_, &module_block_pool_, &options,
                                  &module));
    return ModulePtr(module);
  }

  loom_op_t* FindFunction(loom_module_t* module, iree_string_view_t name) {
    for (loom_symbol_id_t i = 0; i < module->symbols.count; ++i) {
      const loom_symbol_t* symbol = &module->symbols.entries[i];
      if (iree_string_view_equal(
              loom_string_table_get(&module->strings, symbol->name_id), name)) {
        return symbol->defining_op;
      }
    }
    return nullptr;
  }

  iree_arena_block_pool_t module_block_pool_;
  iree_arena_block_pool_t program_block_pool_;
  iree_arena_allocator_t program_arena_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(Aie2pProgramCompileTest,
       LinksDirectHelperAndPreservesIncomingLinkRegister) {
  ModulePtr module = ParseModule(IREE_SV(kDirectHelperSource));
  loom_op_t* entry = FindFunction(module.get(), IREE_SV("entry"));
  ASSERT_NE(entry, nullptr);
  loom_block_t* entry_block =
      loom_region_entry_block(loom_low_func_def_body(entry));
  ASSERT_GE(entry_block->op_count, 2u);
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {
          .value_id = loom_op_const_results(loom_block_op(entry_block, 0))[0],
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_base =
              loom_aie2p_machine_find_physical_register(IREE_SV("r8")),
          .location_count = 1,
      },
      {
          .value_id = loom_op_const_results(loom_block_op(entry_block, 1))[0],
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_base =
              loom_aie2p_machine_find_physical_register(IREE_SV("r9")),
          .location_count = 1,
      },
  };
  const loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = &low_registry_.registry,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = IREE_ARRAYSIZE(fixed_values),
  };
  bool compiled = false;
  loom_aie2p_leaf_contribution_t contribution = {};
  IREE_ASSERT_OK(loom_aie2p_program_compile(module.get(), entry, &options,
                                            &program_arena_, &compiled,
                                            &contribution));
  ASSERT_TRUE(compiled);
  ASSERT_EQ(contribution.object.section_count, 2u);
  ASSERT_EQ(contribution.object.fixup_count, 3u);
  EXPECT_LE(contribution.realization.code.byte_length,
            LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE);
  EXPECT_EQ(contribution.realization.spill.byte_length, 0u);

  const loom_native_object_fixup_t* call_fixup = nullptr;
  iree_host_size_t call_fixup_index = IREE_HOST_SIZE_MAX;
  iree_host_size_t branch_fixup_count = 0;
  for (iree_host_size_t i = 0; i < contribution.object.fixup_count; ++i) {
    const loom_native_object_fixup_t* fixup = &contribution.object.fixups[i];
    if (fixup->relocation_kind ==
        LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_CALL_ABSOLUTE) {
      ASSERT_EQ(call_fixup, nullptr);
      call_fixup = fixup;
      call_fixup_index = i;
    } else if (fixup->relocation_kind ==
               LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE) {
      ++branch_fixup_count;
    }
  }
  ASSERT_NE(call_fixup, nullptr);
  EXPECT_EQ(branch_fixup_count, 2u);
  ASSERT_LT(call_fixup->target_symbol_index, contribution.object.symbol_count);
  const loom_native_object_symbol_t& helper_symbol =
      contribution.object.symbols[call_fixup->target_symbol_index];
  EXPECT_TRUE(iree_string_view_equal(helper_symbol.name, IREE_SV("add_pair")));
  EXPECT_NE(helper_symbol.section_contribution_index, IREE_HOST_SIZE_MAX);

  const loom_native_object_symbol_t& entry_symbol =
      contribution.object.symbols[contribution.realization.entry_symbol_index];
  const iree_const_byte_span_t entry_code =
      contribution.object.sections[entry_symbol.section_contribution_index]
          .contents;
  const iree_const_byte_span_t helper_code =
      contribution.object.sections[helper_symbol.section_contribution_index]
          .contents;
  std::vector<DecodedProgramBundle> entry_bundles;
  std::vector<DecodedProgramBundle> helper_bundles;
  IREE_ASSERT_OK(DecodeProgram(entry_code, &entry_bundles));
  IREE_ASSERT_OK(DecodeProgram(helper_code, &helper_bundles));

  const loom_aie2p_instruction_id_t call_instruction =
      loom_aie2p_encoding_find_instruction(IREE_SV("JL_lng"));
  const loom_aie2p_instruction_id_t return_instruction =
      loom_aie2p_encoding_find_instruction(IREE_SV("RET"));
  ASSERT_NE(call_instruction, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  ASSERT_NE(return_instruction, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  const loom_aie2p_physical_register_id_t link_register =
      loom_aie2p_machine_find_physical_register(IREE_SV("lr"));
  const loom_aie2p_physical_register_id_t shadow_register =
      loom_aie2p_machine_find_physical_register(IREE_SV("r23"));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const loom_aie2p_encoded_slot_t save_link =
      EncodeMove(descriptor_set, link_register, shadow_register);
  const loom_aie2p_encoded_slot_t restore_link =
      EncodeMove(descriptor_set, shadow_register, link_register);

  iree_host_size_t save_bundle = IREE_HOST_SIZE_MAX;
  iree_host_size_t call_bundle = IREE_HOST_SIZE_MAX;
  iree_host_size_t restore_bundle = IREE_HOST_SIZE_MAX;
  iree_host_size_t return_bundle = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < entry_bundles.size(); ++i) {
    const DecodedProgramBundle& bundle = entry_bundles[i];
    if (BundleHasInstruction(bundle, call_instruction)) {
      call_bundle = i;
    }
    if (BundleHasInstruction(bundle, return_instruction)) {
      return_bundle = i;
    }
    if (BundleHasEncodedSlot(bundle, save_link)) {
      save_bundle = i;
    }
    if (BundleHasEncodedSlot(bundle, restore_link)) {
      restore_bundle = i;
    }
  }
  ASSERT_NE(save_bundle, IREE_HOST_SIZE_MAX);
  ASSERT_NE(call_bundle, IREE_HOST_SIZE_MAX);
  ASSERT_NE(restore_bundle, IREE_HOST_SIZE_MAX);
  ASSERT_NE(return_bundle, IREE_HOST_SIZE_MAX);
  EXPECT_LT(save_bundle, call_bundle);
  EXPECT_LT(call_bundle, restore_bundle);
  EXPECT_LT(restore_bundle, return_bundle);

  ASSERT_GE(entry_bundles.size(), call_bundle + 6u);
  iree_host_size_t useful_delay_slot_count = 0;
  for (iree_host_size_t i = call_bundle + 1; i <= call_bundle + 5; ++i) {
    useful_delay_slot_count += BundleHasUsefulInstruction(entry_bundles[i]);
  }
  EXPECT_EQ(useful_delay_slot_count, 2u);
  ASSERT_GT(entry_bundles.size(), call_bundle + 6u);
  EXPECT_TRUE(BundleHasUsefulInstruction(entry_bundles[call_bundle + 6u]));
  EXPECT_TRUE(std::any_of(helper_bundles.begin(), helper_bundles.end(),
                          [return_instruction](const auto& bundle) {
                            return BundleHasInstruction(bundle,
                                                        return_instruction);
                          }));

  const loom_aie2p_tile_link_layout_t layout = {
      .program_address = 0x2000,
      .program_owner_offset = 0,
      .program_byte_capacity = LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE,
  };
  loom_aie2p_linked_tile_t linked_tile = {};
  IREE_ASSERT_OK(loom_aie2p_tile_link(&contribution, &layout, &program_arena_,
                                      &linked_tile));
  ASSERT_EQ(linked_tile.assembly.section_count, 1u);

  std::vector<loom_native_object_symbol_layout_t> symbol_layouts(
      contribution.object.symbol_count);
  IREE_ASSERT_OK(loom_native_object_resolve_symbol_layouts(
      contribution.object.symbols, contribution.object.symbol_count,
      linked_tile.assembly.contribution_layouts,
      linked_tile.assembly.contribution_layout_count, symbol_layouts.data()));
  std::vector<loom_native_object_fixup_layout_t> fixup_layouts(
      contribution.object.fixup_count);
  IREE_ASSERT_OK(loom_native_object_resolve_fixup_layouts(
      contribution.object.fixups, contribution.object.fixup_count,
      contribution.object.symbol_count,
      linked_tile.assembly.contribution_layouts,
      linked_tile.assembly.contribution_layout_count, fixup_layouts.data()));
  const loom_native_object_fixup_layout_t& fixup_layout =
      fixup_layouts[call_fixup_index];
  ASSERT_EQ(fixup_layout.section_index, 0u);
  const loom_native_section_t& linked_code = linked_tile.assembly.sections[0];
  loom_aie2p_decoded_bundle_t relocated_call_bundle;
  iree_host_size_t relocated_call_length = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_decode_bundle_prefix(
      iree_make_const_byte_span(
          linked_code.contents.data + fixup_layout.section_offset,
          iree_min(
              linked_code.contents.data_length - fixup_layout.section_offset,
              (iree_host_size_t)LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE)),
      &relocated_call_bundle, &relocated_call_length));
  const DecodedProgramBundle relocated_call = {
      .byte_offset = fixup_layout.section_offset,
      .byte_length = relocated_call_length,
      .bundle = relocated_call_bundle,
  };
  const loom_aie2p_encoded_slot_t* call_slot =
      FindInstructionSlot(relocated_call, call_instruction);
  ASSERT_NE(call_slot, nullptr);
  loom_aie2p_encoding_field_value_t call_fields[1];
  iree_host_size_t call_field_count = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_unpack_instruction(
      call_instruction, call_slot->value, IREE_ARRAYSIZE(call_fields),
      call_fields, &call_field_count));
  ASSERT_EQ(call_field_count, 1u);
  const uint64_t helper_address =
      linked_code.address +
      symbol_layouts[call_fixup->target_symbol_index].section_offset;
  EXPECT_EQ(call_fields[0].value, helper_address);
}

TEST_F(Aie2pProgramCompileTest,
       CompilesFourPointerHelperUsingBothLoadFifoTuples) {
  ModulePtr module = ParseModule(IREE_SV(kPointerHelperSource));
  loom_op_t* entry = FindFunction(module.get(), IREE_SV("entry"));
  ASSERT_NE(entry, nullptr);
  const loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = &low_registry_.registry,
  };
  bool compiled = false;
  loom_aie2p_leaf_contribution_t contribution = {};
  IREE_ASSERT_OK(loom_aie2p_program_compile(module.get(), entry, &options,
                                            &program_arena_, &compiled,
                                            &contribution));
  ASSERT_TRUE(compiled);
  ASSERT_EQ(contribution.object.section_count, 2u);
  EXPECT_EQ(contribution.realization.spill.byte_length, 0u);

  const loom_native_object_symbol_t* helper_symbol = nullptr;
  for (iree_host_size_t i = 0; i < contribution.object.symbol_count; ++i) {
    const loom_native_object_symbol_t* symbol = &contribution.object.symbols[i];
    if (iree_string_view_equal(symbol->name, IREE_SV("sum_streams")) &&
        symbol->section_contribution_index != IREE_HOST_SIZE_MAX) {
      helper_symbol = symbol;
      break;
    }
  }
  ASSERT_NE(helper_symbol, nullptr);
  const iree_const_byte_span_t helper_code =
      contribution.object.sections[helper_symbol->section_contribution_index]
          .contents;
  std::vector<DecodedProgramBundle> helper_bundles;
  IREE_ASSERT_OK(DecodeProgram(helper_code, &helper_bundles));
  const loom_aie2p_instruction_id_t first_fifo_pop =
      loom_aie2p_encoding_find_instruction(IREE_SV("VLDA_POP_512_normal_pop"));
  const loom_aie2p_instruction_id_t second_fifo_pop =
      loom_aie2p_encoding_find_instruction(IREE_SV("VLDB_POP_512_normal_pop"));
  ASSERT_NE(first_fifo_pop, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  ASSERT_NE(second_fifo_pop, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  EXPECT_TRUE(std::any_of(helper_bundles.begin(), helper_bundles.end(),
                          [first_fifo_pop](const auto& bundle) {
                            return BundleHasInstruction(bundle, first_fifo_pop);
                          }));
  EXPECT_TRUE(std::any_of(helper_bundles.begin(), helper_bundles.end(),
                          [second_fifo_pop](const auto& bundle) {
                            return BundleHasInstruction(bundle,
                                                        second_fifo_pop);
                          }));
}

}  // namespace
}  // namespace loom
