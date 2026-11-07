// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_compile.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/function_requirements.h"
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
#include "loom/target/reporting/report.h"
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

constexpr char kRegisterDeadlineCallSource[] = R"(
aie2p.target<core> @target

low.func.def target<amd.xdna.aie2p.core>(@target) @add_five(%a: reg<aie2p.er>, %b: reg<aie2p.er>, %c: reg<aie2p.er>, %d: reg<aie2p.er>, %e: reg<aie2p.er>) -> (reg<aie2p.er>) asm {
  %ab = add.rr %a, %b
  %abc = add.rr %ab, %c
  %abcd = add.rr %abc, %d
  %abcde = add.rr %abcd, %e
  return %abcde
}

low.func.def target<amd.xdna.aie2p.core>(@target) abi(object_function) @entry(%input: reg<aie2p.ep>, %a: reg<aie2p.er>, %b: reg<aie2p.er>, %c: reg<aie2p.er>, %d: reg<aie2p.er>, %e: reg<aie2p.er>) -> (reg<aie2p.er>) asm {
  %dead = lda %input, 0
  %sum = low.func.call pure @add_five(%a, %b, %c, %d, %e) : (reg<aie2p.er>, reg<aie2p.er>, reg<aie2p.er>, reg<aie2p.er>, reg<aie2p.er>) -> (reg<aie2p.er>)
  return %sum
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

constexpr char kArgumentTransportClobberSource[] = R"(
aie2p.target<core> @target

low.func.def target<amd.xdna.aie2p.core>(@target) @add_pair(%lhs: reg<aie2p.er>, %rhs: reg<aie2p.er>, %transport_only: reg<aie2p.er>) -> (reg<aie2p.er>) asm {
  %sum = add.rr %lhs, %rhs
  return %sum
}

low.func.def target<amd.xdna.aie2p.core>(@target) abi(object_function) @entry() -> (reg<aie2p.er>) asm {
  %lhs = mov.i32 7
  %rhs = mov.i32 9
  %count = mov.i32 16
  %decrement = mov.i32 -1
  low.br ^loop(%count: reg<aie2p.er>)
^loop(%remaining: reg<aie2p.er>):
  %next = add.rr %remaining, %decrement
  %sum = low.func.call @add_pair(%lhs, %rhs, %count) : (reg<aie2p.er>, reg<aie2p.er>, reg<aie2p.er>) -> (reg<aie2p.er>)
  low.cond_br %next, ^again, ^done : reg<aie2p.er>
^again:
  low.br ^loop(%next: reg<aie2p.er>)
^done:
  return %sum
}
)";

constexpr char kCallThenBranchSource[] = R"(
aie2p.target<core> @target

low.func.def target<amd.xdna.aie2p.core>(@target) @return_immediately() asm {
  return
}

low.func.def target<amd.xdna.aie2p.core>(@target) abi(object_function) @entry() asm {
  %condition = mov.i32 1
  low.func.call @return_immediately() : ()
  low.cond_br %condition, ^true, ^false : reg<aie2p.er>
^true:
  return
^false:
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
  const loom_aie2p_move_route_id_t route_id =
      loom_aie2p_descriptor_resolve_move_route(source, destination);
  EXPECT_NE(route_id, LOOM_AIE2P_MOVE_ROUTE_ID_INVALID);
  EXPECT_EQ(loom_aie2p_descriptor_move_route_step_count(route_id), 1);
  return loom_aie2p_descriptor_encode_move_step(
      descriptor_set,
      loom_aie2p_descriptor_move_route_step(route_id,
                                            loom_aie2p_register_move_t{
                                                .source = source,
                                                .destination = destination,
                                            },
                                            0));
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

  iree_status_t CompileProgram(
      loom_module_t* module, loom_op_t* entry,
      const loom_aie2p_leaf_compile_options_t* options,
      loom_aie2p_program_compiler_t* compiler, bool* out_compiled,
      loom_aie2p_leaf_contribution_t* out_contribution) {
    if (compiler == nullptr) {
      IREE_RETURN_IF_ERROR(loom_aie2p_program_compiler_allocate(
          module, &program_arena_, options->compile_report, &compiler));
    }
    iree_arena_allocator_t requirements_arena;
    iree_arena_initialize(&program_block_pool_, &requirements_arena);
    loom_low_function_requirements_t requirements = {};
    iree_status_t status = loom_low_function_requirements_build(
        module, loom_aie2p_core_descriptor_set(), loom_low_func_def_body(entry),
        &requirements_arena, &requirements);
    if (iree_status_is_ok(status)) {
      std::vector<loom_aie2p_program_helper_t> direct_helpers(
          requirements.direct_callees.count);
      for (iree_host_size_t i = 0; i < direct_helpers.size(); ++i) {
        direct_helpers[i].function =
            requirements.direct_callees.values[i].symbol;
      }
      status = loom_aie2p_program_compile(compiler, entry,
                                          requirements.return_count != 0,
                                          loom_aie2p_program_helper_list_t{
                                              .values = direct_helpers.data(),
                                              .count = direct_helpers.size(),
                                          },
                                          &requirements.read_only_data, options,
                                          out_compiled, out_contribution);
    }
    iree_arena_deinitialize(&requirements_arena);
    return status;
  }

  iree_status_t CompileProgram(
      loom_module_t* module, loom_op_t* entry,
      const loom_aie2p_leaf_compile_options_t* options, bool* out_compiled,
      loom_aie2p_leaf_contribution_t* out_contribution) {
    return CompileProgram(module, entry, options, /*compiler=*/nullptr,
                          out_compiled, out_contribution);
  }

  iree_arena_block_pool_t module_block_pool_;
  iree_arena_block_pool_t program_block_pool_;
  iree_arena_allocator_t program_arena_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(Aie2pProgramCompileTest, RetainsHelperObjectsAcrossProgramCompilations) {
  const iree_string_view_t sources[] = {
      IREE_SV(kDirectHelperSource),
      IREE_SV(kArgumentTransportClobberSource),
      IREE_SV(kPointerHelperSource),
  };
  const loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = &low_registry_.registry,
  };
  for (iree_string_view_t source : sources) {
    ModulePtr module = ParseModule(source);
    loom_op_t* entry = FindFunction(module.get(), IREE_SV("entry"));
    ASSERT_NE(entry, nullptr);
    loom_aie2p_program_compiler_t* compiler = nullptr;
    IREE_ASSERT_OK(loom_aie2p_program_compiler_allocate(
        module.get(), &program_arena_, /*report_template=*/nullptr, &compiler));
    const uint8_t* helper_contents = nullptr;
    const uint8_t* entry_contents = nullptr;
    for (int program_index = 0; program_index < 3; ++program_index) {
      bool compiled = false;
      loom_aie2p_leaf_contribution_t contribution = {};
      IREE_ASSERT_OK(CompileProgram(module.get(), entry, &options, compiler,
                                    &compiled, &contribution));
      ASSERT_TRUE(compiled);
      ASSERT_EQ(contribution.object.section_count, 2u);
      // Each caller is allocated independently. Its native call still borrows
      // the same helper code and ABI contract after earlier frame scratch has
      // been released, including pointer carriers and transport-only clobbers.
      EXPECT_NE(contribution.object.sections[0].contents.data, entry_contents);
      entry_contents = contribution.object.sections[0].contents.data;
      const uint8_t* current_helper =
          contribution.object.sections[1].contents.data;
      if (program_index == 0) {
        helper_contents = current_helper;
      } else {
        EXPECT_EQ(current_helper, helper_contents);
      }
    }

    // Equal symbol ordinals in a different module cannot reuse the first
    // module's native object or register contract.
    ModulePtr other_module = ParseModule(source);
    loom_op_t* other_entry = FindFunction(other_module.get(), IREE_SV("entry"));
    ASSERT_NE(other_entry, nullptr);
    bool compiled = false;
    loom_aie2p_leaf_contribution_t other_contribution = {};
    IREE_ASSERT_OK(CompileProgram(other_module.get(), other_entry, &options,
                                  &compiled, &other_contribution));
    ASSERT_TRUE(compiled);
    ASSERT_EQ(other_contribution.object.section_count, 2u);
    EXPECT_NE(other_contribution.object.sections[1].contents.data,
              helper_contents);
  }
}

TEST_F(Aie2pProgramCompileTest, ReportsCompleteProgramIncludingRetainedHelper) {
  ModulePtr module = ParseModule(IREE_SV(kPointerHelperSource));
  loom_op_t* entry = FindFunction(module.get(), IREE_SV("entry"));
  loom_op_t* helper = FindFunction(module.get(), IREE_SV("sum_streams"));
  ASSERT_NE(entry, nullptr);
  ASSERT_NE(helper, nullptr);

  loom_target_compile_report_t report_template;
  loom_target_compile_report_initialize(iree_allocator_system(),
                                        &report_template);
  loom_aie2p_program_compiler_t* compiler = nullptr;
  IREE_ASSERT_OK(loom_aie2p_program_compiler_allocate(
      module.get(), &program_arena_, &report_template, &compiler));
  loom_target_compile_report_deinitialize(&report_template);

  for (int program_index = 0; program_index < 2; ++program_index) {
    loom_target_compile_report_t report;
    loom_target_compile_report_initialize(iree_allocator_system(), &report);
    const loom_aie2p_leaf_compile_options_t options = {
        .descriptor_registry = &low_registry_.registry,
        .compile_report = &report,
    };
    bool compiled = false;
    loom_aie2p_leaf_contribution_t contribution = {};
    IREE_ASSERT_OK(CompileProgram(module.get(), entry, &options, compiler,
                                  &compiled, &contribution));
    ASSERT_TRUE(compiled);
    ASSERT_EQ(contribution.object.section_count, 2u);

    uint64_t semantic_code_bytes = 0;
    for (iree_host_size_t i = 0; i < contribution.object.section_count; ++i) {
      semantic_code_bytes +=
          contribution.object.sections[i].contents.data_length;
    }
    EXPECT_EQ(report.emitted_code_byte_count, semantic_code_bytes);
    EXPECT_EQ(report.emitted_code_storage_byte_count,
              contribution.realization.code.byte_length);
    EXPECT_EQ(report.emission_breakdown.body_instruction_count,
              report.emitted_instruction_count);
    EXPECT_EQ(report.static_instruction_mix.vector_alu_count, 1u);
    EXPECT_EQ(report.dynamic_instruction_mix.vector_alu_count, 1u);

    const loom_target_compile_report_call_edge_t* edge =
        loom_target_compile_report_find_call_edge(
            &report, loom_low_func_def_callee(entry),
            loom_low_func_def_callee(helper));
    ASSERT_NE(edge, nullptr);
    EXPECT_EQ(edge->callsite_count, 1u);
    EXPECT_TRUE(iree_all_bits_set(
        edge->flags, LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT));
    EXPECT_EQ(edge->dynamic_call_count, 1u);
    loom_target_compile_report_deinitialize(&report);
  }
}

TEST_F(Aie2pProgramCompileTest, FillsCallDelayWindowAfterRegisterDeadline) {
  ModulePtr module = ParseModule(IREE_SV(kRegisterDeadlineCallSource));
  loom_op_t* entry = FindFunction(module.get(), IREE_SV("entry"));
  ASSERT_NE(entry, nullptr);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments = loom_func_like_arg_ids(
      loom_func_like_cast(module.get(), entry), &argument_count);
  ASSERT_EQ(argument_count, 6u);
  const char* fixed_register_names[] = {"r18", "r19", "r20", "r21", "r22"};
  loom_low_allocation_fixed_value_t fixed_values[6];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fixed_register_names); ++i) {
    fixed_values[i] = loom_low_allocation_fixed_value_t{
        .value_id = arguments[i + 1],
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = loom_aie2p_machine_find_physical_register(
            iree_make_cstring_view(fixed_register_names[i])),
        .location_count = 1,
    };
  }
  loom_block_t* entry_block =
      loom_region_entry_block(loom_low_func_def_body(entry));
  ASSERT_NE(entry_block, nullptr);
  fixed_values[5] = loom_low_allocation_fixed_value_t{
      .value_id = loom_op_results(loom_block_op(entry_block, 0))[0],
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = loom_aie2p_machine_find_physical_register(IREE_SV("r0")),
      .location_count = 1,
  };
  const loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = &low_registry_.registry,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = IREE_ARRAYSIZE(fixed_values),
  };
  bool compiled = false;
  loom_aie2p_leaf_contribution_t contribution = {};
  IREE_ASSERT_OK(
      CompileProgram(module.get(), entry, &options, &compiled, &contribution));
  ASSERT_TRUE(compiled);

  const loom_native_object_symbol_t& entry_symbol =
      contribution.object.symbols[contribution.realization.entry_symbol_index];
  std::vector<DecodedProgramBundle> bundles;
  IREE_ASSERT_OK(DecodeProgram(
      contribution.object.sections[entry_symbol.section_contribution_index]
          .contents,
      &bundles));
  const loom_aie2p_instruction_id_t call =
      loom_aie2p_encoding_find_instruction(IREE_SV("JL_lng"));
  const loom_aie2p_instruction_id_t load =
      loom_aie2p_encoding_find_instruction(IREE_SV("LDA_dms_lda_idx_imm"));
  iree_host_size_t call_bundle = IREE_HOST_SIZE_MAX;
  iree_host_size_t load_bundle = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < bundles.size(); ++i) {
    if (BundleHasInstruction(bundles[i], call)) {
      call_bundle = i;
    }
    if (BundleHasInstruction(bundles[i], load)) {
      load_bundle = i;
    }
  }
  ASSERT_NE(call_bundle, IREE_HOST_SIZE_MAX);
  ASSERT_NE(load_bundle, IREE_HOST_SIZE_MAX);
  EXPECT_LT(load_bundle, call_bundle);
  ASSERT_GE(bundles.size(), call_bundle + 6u);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fixed_register_names); ++i) {
    const loom_aie2p_physical_register_id_t source =
        loom_aie2p_machine_find_physical_register(
            iree_make_cstring_view(fixed_register_names[i]));
    const loom_aie2p_physical_register_id_t destination =
        loom_low_descriptor_set_physical_register_candidate(
            descriptor_set, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, (uint16_t)i);
    EXPECT_TRUE(
        BundleHasEncodedSlot(bundles[call_bundle + i + 1u],
                             EncodeMove(descriptor_set, source, destination)));
  }
}

TEST_F(Aie2pProgramCompileTest, ReusesOnlyIdenticalAdjacentPhysicalConstants) {
  struct TestCase {
    // Immediate written by the second materialization.
    uint32_t second_constant;
    // Physical destination of the second materialization.
    const char* second_register;
    // Number of distinct writes that native emission must preserve.
    size_t expected_write_count;
  };
  const TestCase cases[] = {{780, "r0", 1}, {781, "r0", 2}, {780, "r1", 2}};
  const loom_aie2p_instruction_id_t move =
      loom_aie2p_encoding_find_instruction(IREE_SV("MOVA"));
  ASSERT_NE(move, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  const auto destination_field = loom_aie2p_encoding_find_field(IREE_SV("dst"));
  const auto immediate_field = loom_aie2p_encoding_find_field(IREE_SV("i"));
  const auto destination_adapter =
      loom_aie2p_machine_find_register_adapter(IREE_SV("OP_mLdaCg"));
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.second_constant);
    SCOPED_TRACE(test_case.second_register);
    std::string source = R"(
aie2p.target<core> @target
low.func.def target<amd.xdna.aie2p.core>(@target) @entry(%input: reg<aie2p.ep>, %accumulator: reg<aie2p.mbms x4>, %lhs: reg<aie2p.mexa>, %rhs: reg<aie2p.mexa>) -> (reg<aie2p.mbms x4>, reg<aie2p.vec256 x2>) asm {
  %first_control = mova.i32 780
  %loaded = vlda.512.i32x16 %input, 0
  %first = mma.bfp16ebs8.m8n8k8 %accumulator, %lhs, %rhs, %first_control
  %second_control = mova.i32 SECOND_CONSTANT
  %second = mma.bfp16ebs8.m8n8k8 %first, %lhs, %rhs, %second_control
  return %second, %loaded
}
)";
    source.replace(source.find("SECOND_CONSTANT"), 15,
                   std::to_string(test_case.second_constant));
    ModulePtr module =
        ParseModule(iree_make_string_view(source.data(), source.size()));
    loom_op_t* entry = FindFunction(module.get(), IREE_SV("entry"));
    ASSERT_NE(entry, nullptr);
    loom_block_t* block =
        loom_region_entry_block(loom_low_func_def_body(entry));
    const auto first_register =
        loom_aie2p_machine_find_physical_register(IREE_SV("r0"));
    const auto second_register = loom_aie2p_machine_find_physical_register(
        iree_make_cstring_view(test_case.second_register));
    const loom_low_allocation_fixed_value_t fixed_values[] = {
        {
            .value_id = loom_op_const_results(loom_block_op(block, 0))[0],
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = first_register,
            .location_count = 1,
        },
        {
            .value_id = loom_op_const_results(loom_block_op(block, 3))[0],
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = second_register,
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
    IREE_ASSERT_OK(CompileProgram(module.get(), entry, &options, &compiled,
                                  &contribution));
    ASSERT_TRUE(compiled);
    const auto& symbol =
        contribution.object
            .symbols[contribution.realization.entry_symbol_index];
    std::vector<DecodedProgramBundle> bundles;
    IREE_ASSERT_OK(DecodeProgram(
        contribution.object.sections[symbol.section_contribution_index]
            .contents,
        &bundles));
    std::vector<std::pair<uint64_t, uint64_t>> writes;
    for (const auto& bundle : bundles) {
      const auto* slot = FindInstructionSlot(bundle, move);
      if (!slot) {
        continue;
      }
      loom_aie2p_encoding_field_value_t fields[2];
      iree_host_size_t field_count = 0;
      IREE_ASSERT_OK(loom_aie2p_encoding_unpack_instruction(
          move, slot->value, IREE_ARRAYSIZE(fields), fields, &field_count));
      ASSERT_EQ(field_count, 2u);
      ASSERT_EQ(fields[0].field_id, destination_field);
      ASSERT_EQ(fields[1].field_id, immediate_field);
      writes.emplace_back(fields[0].value, fields[1].value);
    }
    ASSERT_EQ(writes.size(), test_case.expected_write_count);
    uint8_t first_destination;
    ASSERT_TRUE(loom_aie2p_machine_encode_register(
        destination_adapter, first_register, &first_destination));
    EXPECT_EQ(writes[0].first, first_destination);
    EXPECT_EQ(writes[0].second, 780u);
    if (test_case.expected_write_count == 2) {
      uint8_t second_destination;
      ASSERT_TRUE(loom_aie2p_machine_encode_register(
          destination_adapter, second_register, &second_destination));
      EXPECT_EQ(writes[1].first, second_destination);
      EXPECT_EQ(writes[1].second, test_case.second_constant);
    }
  }
}

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
  IREE_ASSERT_OK(
      CompileProgram(module.get(), entry, &options, &compiled, &contribution));
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
  EXPECT_EQ(entry_bundles[call_bundle + 6u].byte_offset % 16u, 0u);
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
       PreservesLoopCarriedValueAcrossArgumentTransport) {
  ModulePtr module = ParseModule(IREE_SV(kArgumentTransportClobberSource));
  loom_op_t* entry = FindFunction(module.get(), IREE_SV("entry"));
  ASSERT_NE(entry, nullptr);
  loom_block_t* entry_block =
      loom_region_entry_block(loom_low_func_def_body(entry));
  ASSERT_GE(entry_block->op_count, 4u);
  // Pin the call inputs outside r0-r2 so only the call contract can prevent
  // the loop-carried value from overlapping those transport destinations.
  const char* fixed_register_names[] = {"r18", "r19", "r21", "r20"};
  loom_low_allocation_fixed_value_t fixed_values[4];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fixed_values); ++i) {
    fixed_values[i] = loom_low_allocation_fixed_value_t{
        .value_id = loom_op_const_results(loom_block_op(entry_block, i))[0],
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = loom_aie2p_machine_find_physical_register(
            iree_make_cstring_view(fixed_register_names[i])),
        .location_count = 1,
    };
  }
  const loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = &low_registry_.registry,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = IREE_ARRAYSIZE(fixed_values),
  };
  bool compiled = false;
  loom_aie2p_leaf_contribution_t contribution = {};
  IREE_ASSERT_OK(
      CompileProgram(module.get(), entry, &options, &compiled, &contribution));
  ASSERT_TRUE(compiled);
  EXPECT_EQ(contribution.realization.spill.byte_length, 0u);

  const loom_native_object_symbol_t& entry_symbol =
      contribution.object.symbols[contribution.realization.entry_symbol_index];
  const iree_const_byte_span_t entry_code =
      contribution.object.sections[entry_symbol.section_contribution_index]
          .contents;
  std::vector<DecodedProgramBundle> entry_bundles;
  IREE_ASSERT_OK(DecodeProgram(entry_code, &entry_bundles));

  const loom_aie2p_instruction_id_t call_instruction =
      loom_aie2p_encoding_find_instruction(IREE_SV("JL_lng"));
  const loom_aie2p_instruction_id_t add_instruction =
      loom_aie2p_encoding_find_instruction(IREE_SV("ADD_alu_r_rr"));
  ASSERT_NE(call_instruction, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  ASSERT_NE(add_instruction, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  const loom_aie2p_encoding_field_id_t destination_field =
      loom_aie2p_encoding_find_field(IREE_SV("d0"));
  ASSERT_NE(destination_field, LOOM_AIE2P_ENCODING_FIELD_ID_INVALID);

  const loom_aie2p_encoded_slot_t* next_add = nullptr;
  iree_host_size_t call_count = 0;
  for (const DecodedProgramBundle& bundle : entry_bundles) {
    call_count += BundleHasInstruction(bundle, call_instruction);
    const loom_aie2p_encoded_slot_t* slot =
        FindInstructionSlot(bundle, add_instruction);
    if (slot != nullptr) {
      ASSERT_EQ(next_add, nullptr);
      next_add = slot;
    }
  }
  EXPECT_EQ(call_count, 1u);
  ASSERT_NE(next_add, nullptr);

  loom_aie2p_encoding_field_value_t add_fields[3];
  iree_host_size_t add_field_count = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_unpack_instruction(
      add_instruction, next_add->value, IREE_ARRAYSIZE(add_fields), add_fields,
      &add_field_count));
  ASSERT_EQ(add_field_count, IREE_ARRAYSIZE(add_fields));
  const loom_aie2p_encoding_field_value_t* destination = nullptr;
  for (const loom_aie2p_encoding_field_value_t& field : add_fields) {
    if (field.field_id == destination_field) {
      destination = &field;
      break;
    }
  }
  ASSERT_NE(destination, nullptr);

  const iree_string_view_t argument_registers[] = {
      IREE_SV("r0"),
      IREE_SV("r1"),
      IREE_SV("r2"),
  };
  for (iree_string_view_t register_name : argument_registers) {
    loom_aie2p_physical_register_info_t register_info;
    ASSERT_TRUE(loom_aie2p_machine_query_physical_register(
        loom_aie2p_machine_find_physical_register(register_name),
        &register_info));
    EXPECT_NE(destination->value, register_info.hardware_encoding);
  }
}

TEST_F(Aie2pProgramCompileTest, SeparatesReturnedCallFromFollowingBranch) {
  ModulePtr module = ParseModule(IREE_SV(kCallThenBranchSource));
  loom_op_t* entry = FindFunction(module.get(), IREE_SV("entry"));
  ASSERT_NE(entry, nullptr);
  const loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = &low_registry_.registry,
  };
  bool compiled = false;
  loom_aie2p_leaf_contribution_t contribution = {};
  IREE_ASSERT_OK(
      CompileProgram(module.get(), entry, &options, &compiled, &contribution));
  ASSERT_TRUE(compiled);

  const loom_native_object_symbol_t& entry_symbol =
      contribution.object.symbols[contribution.realization.entry_symbol_index];
  const iree_const_byte_span_t entry_code =
      contribution.object.sections[entry_symbol.section_contribution_index]
          .contents;
  std::vector<DecodedProgramBundle> entry_bundles;
  IREE_ASSERT_OK(DecodeProgram(entry_code, &entry_bundles));

  const loom_aie2p_instruction_id_t call_instruction =
      loom_aie2p_encoding_find_instruction(IREE_SV("JL_lng"));
  const loom_aie2p_instruction_id_t return_instruction =
      loom_aie2p_encoding_find_instruction(IREE_SV("RET"));
  const loom_aie2p_instruction_id_t branch_instructions[] = {
      loom_aie2p_encoding_find_instruction(IREE_SV("J_lng")),
      loom_aie2p_encoding_find_instruction(IREE_SV("JZ")),
      loom_aie2p_encoding_find_instruction(IREE_SV("JNZ")),
  };
  ASSERT_NE(call_instruction, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  ASSERT_NE(return_instruction, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  for (loom_aie2p_instruction_id_t branch_instruction : branch_instructions) {
    ASSERT_NE(branch_instruction, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  }

  iree_host_size_t call_bundle = IREE_HOST_SIZE_MAX;
  iree_host_size_t branch_bundle = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < entry_bundles.size(); ++i) {
    if (BundleHasInstruction(entry_bundles[i], call_instruction)) {
      ASSERT_EQ(call_bundle, IREE_HOST_SIZE_MAX);
      call_bundle = i;
    }
    for (loom_aie2p_instruction_id_t branch_instruction : branch_instructions) {
      if (BundleHasInstruction(entry_bundles[i], branch_instruction)) {
        ASSERT_EQ(branch_bundle, IREE_HOST_SIZE_MAX);
        branch_bundle = i;
      }
    }
  }
  ASSERT_NE(call_bundle, IREE_HOST_SIZE_MAX);
  ASSERT_NE(branch_bundle, IREE_HOST_SIZE_MAX);

  // JL has five delay positions, which may contain caller work. The next two
  // issue positions contain no structural control, establishing the complete
  // post-return gap required by the hardware before the branch.
  ASSERT_EQ(branch_bundle, call_bundle + 8u);
  for (iree_host_size_t i = branch_bundle - 2u; i < branch_bundle; ++i) {
    EXPECT_FALSE(BundleHasInstruction(entry_bundles[i], call_instruction));
    EXPECT_FALSE(BundleHasInstruction(entry_bundles[i], return_instruction));
    for (loom_aie2p_instruction_id_t branch_instruction : branch_instructions) {
      EXPECT_FALSE(BundleHasInstruction(entry_bundles[i], branch_instruction));
    }
  }
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
  IREE_ASSERT_OK(
      CompileProgram(module.get(), entry, &options, &compiled, &contribution));
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
  ASSERT_FALSE(helper_bundles.empty());

  // The first pointer argument arrives in p2, while the two-FIFO body assigns
  // it to p0. The helper prologue must perform that ABI transport before the
  // first body instruction uses p0.
  const loom_aie2p_instruction_id_t pointer_move =
      loom_aie2p_encoding_find_instruction(IREE_SV("MOVS"));
  ASSERT_NE(pointer_move, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  const loom_aie2p_encoded_slot_t* entry_move =
      FindInstructionSlot(helper_bundles.front(), pointer_move);
  ASSERT_NE(entry_move, nullptr);
  const loom_aie2p_physical_register_id_t p0 =
      loom_aie2p_machine_find_physical_register(IREE_SV("p0"));
  const loom_aie2p_physical_register_id_t p2 =
      loom_aie2p_machine_find_physical_register(IREE_SV("p2"));
  ASSERT_NE(p0, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
  ASSERT_NE(p2, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
  const loom_aie2p_encoded_slot_t expected_entry_move =
      EncodeMove(loom_aie2p_core_descriptor_set(), p2, p0);
  EXPECT_EQ(entry_move->slot, expected_entry_move.slot);
  EXPECT_EQ(entry_move->value, expected_entry_move.value);

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
