// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks AMDGPU packet planning over immutable production emission frames.
//
// Fixture parsing, lowering, verification, scheduling, and allocation happen
// before the timed loop. Each iteration resets only the transient plan arena
// and rebuilds either the wait plan or the complete target-owned packet plan.

#include <inttypes.h>

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/planning/occupancy.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/arch/amdgpu/planning/storage_lease.h"
#include "loom/target/arch/amdgpu/planning/vopd_plan.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/emit/native/amdgpu/packet_plan_attention_bf16.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/provider.h"
#include "loom/tooling/compile/pipeline.h"

namespace {

constexpr iree_host_size_t kArenaBlockSize = 128 * 1024;
constexpr uint64_t kFnvOffsetBasis = UINT64_C(1469598103934665603);
constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);

enum class PlanComponent {
  kWait,
  kComplete,
};

enum class FixtureInputStage {
  kGeneratedLow,
  kAuthoredSource,
};

struct GeneratedMatrixShape {
  // Number of matrix phases separated by CFG edges.
  uint32_t phase_count;
  // Total synthesized VALU packets in each matrix phase.
  uint32_t valu_packets_per_phase;
  // VALU packets retained in the dependent prefix of each phase.
  uint32_t dependent_valu_packets_per_phase;
};

struct FrameShape {
  // Number of scheduled CFG blocks.
  iree_host_size_t block_count;
  // Number of scheduled packets.
  iree_host_size_t packet_count;
  // Number of physical allocation assignments.
  iree_host_size_t assignment_count;
  // Number of matrix-class packets.
  iree_host_size_t matrix_packet_count;
  // Number of storage lease records.
  iree_host_size_t storage_lease_count;
  // Number of physical register units covered by storage leases.
  iree_host_size_t storage_lease_unit_count;
  // Number of scheduled storage release actions.
  iree_host_size_t storage_release_count;
};

struct FixtureSpec {
  // Stable benchmark row name.
  const char* name;
  // Compilation stage represented by the fixture input.
  FixtureInputStage input_stage;
  // Generated target-low topology; empty for authored source fixtures.
  GeneratedMatrixShape generated_matrix;
};

constexpr FixtureSpec kMemoryControl = {
    /*.name=*/"memory_control",
    /*.input_stage=*/FixtureInputStage::kGeneratedLow,
    /*.generated_matrix=*/{},
};

// A straight-line matrix workload that isolates the common single-block
// coexecution path without CFG frontier propagation.
constexpr FixtureSpec kMatrixSingleBlockCanary = {
    /*.name=*/"matrix_single_block_canary",
    /*.input_stage=*/FixtureInputStage::kGeneratedLow,
    /*.generated_matrix=*/
    {
        /*.phase_count=*/1,
        /*.valu_packets_per_phase=*/4,
        /*.dependent_valu_packets_per_phase=*/4,
    },
};

constexpr FixtureSpec kMatrixDependencyCanary = {
    /*.name=*/"matrix_dependency_canary",
    /*.input_stage=*/FixtureInputStage::kGeneratedLow,
    /*.generated_matrix=*/
    {
        /*.phase_count=*/2,
        /*.valu_packets_per_phase=*/4,
        /*.dependent_valu_packets_per_phase=*/4,
    },
};

// Legal adversarial topology for storage-lease overlap planning. Every address
// packet remains live through a balanced reduction feeding physical result
// writes, producing wide simultaneous lease pressure across 17 matrix phases.
// Its scale resembles a large unrolled production kernel, but its deliberately
// broad live topology makes this a planner stress case rather than a workload
// performance proxy.
constexpr FixtureSpec kMatrixLeaseOverlapStress = {
    /*.name=*/"matrix_lease_overlap_stress",
    /*.input_stage=*/FixtureInputStage::kGeneratedLow,
    /*.generated_matrix=*/
    {
        /*.phase_count=*/17,
        /*.valu_packets_per_phase=*/208,
        /*.dependent_valu_packets_per_phase=*/64,
    },
};

// Fixed-shape 4096-token, 18-head, 256-wide online attention workload. The
// authored source runs through the shared prepared-low pipeline before the
// immutable planner frame is built, keeping the benchmark coupled to the
// shipping compilation boundary rather than a retained low-IR snapshot.
constexpr FixtureSpec kAttentionBf16 = {
    /*.name=*/"attention_bf16",
    /*.input_stage=*/FixtureInputStage::kAuthoredSource,
    /*.generated_matrix=*/{},
};

static void AbortOnError(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
}

static loom_op_t* FindFirstLowFunction(loom_module_t* module) {
  loom_op_t* op = nullptr;
  loom_block_for_each_op(loom_module_block(module), op) {
    if (loom_low_function_def_isa(op)) {
      return op;
    }
  }
  return nullptr;
}

static uint64_t HashValue(uint64_t hash, uint64_t value) {
  for (uint32_t i = 0; i < 8; ++i) {
    hash ^= static_cast<uint8_t>(value >> (i * 8));
    hash *= kFnvPrime;
  }
  return hash;
}

struct FrameAnalysis {
  // Comparable cardinalities extracted during signature construction.
  FrameShape shape;
  // Hash over schedule descriptors and physical allocation assignments.
  uint64_t signature;
};

static FrameAnalysis AnalyzeFrame(const loom_low_emission_frame_t& frame) {
  FrameAnalysis analysis = {
      /*.shape=*/
      {
          /*.block_count=*/frame.schedule.block_count,
          /*.packet_count=*/frame.schedule.scheduled_node_count,
          /*.assignment_count=*/frame.allocation.assignment_count,
          /*.matrix_packet_count=*/0,
          /*.storage_lease_count=*/
          frame.allocation.storage_leases.record_count,
          /*.storage_lease_unit_count=*/0,
          /*.storage_release_count=*/
          frame.allocation.storage_release_action_count,
      },
      /*.signature=*/HashValue(kFnvOffsetBasis, frame.schedule.block_count),
  };
  analysis.signature =
      HashValue(analysis.signature, frame.schedule.scheduled_node_count);
  for (iree_host_size_t i = 0; i < frame.schedule.scheduled_node_count; ++i) {
    const uint32_t node_index = frame.schedule.scheduled_node_indices[i];
    const loom_low_schedule_node_t& node = frame.schedule.nodes[node_index];
    analysis.signature = HashValue(analysis.signature, node_index);
    analysis.signature =
        HashValue(analysis.signature,
                  node.descriptor == nullptr ? 0 : node.descriptor->stable_id);
    if (node.descriptor != nullptr &&
        iree_any_bit_set(
            loom_low_descriptor_set_descriptor_view(
                frame.schedule.target.descriptor_set, node.descriptor)
                ->instruction_class_flags,
            LOOM_LOW_INSTRUCTION_CLASS_FLAG_MATRIX)) {
      ++analysis.shape.matrix_packet_count;
    }
  }
  analysis.signature =
      HashValue(analysis.signature, frame.allocation.assignment_count);
  for (iree_host_size_t i = 0; i < frame.allocation.assignment_count; ++i) {
    const loom_low_allocation_assignment_t& assignment =
        frame.allocation.assignments[i];
    analysis.signature = HashValue(analysis.signature, assignment.value_id);
    analysis.signature =
        HashValue(analysis.signature, assignment.descriptor_reg_class_id);
    analysis.signature = HashValue(analysis.signature, assignment.start_point);
    analysis.signature = HashValue(analysis.signature, assignment.end_point);
    analysis.signature = HashValue(analysis.signature, assignment.unit_count);
    analysis.signature =
        HashValue(analysis.signature, assignment.location_kind);
    analysis.signature =
        HashValue(analysis.signature, assignment.location_base);
    analysis.signature =
        HashValue(analysis.signature, assignment.location_count);
  }
  for (iree_host_size_t i = 0;
       i < frame.allocation.storage_lease_instance_count; ++i) {
    analysis.shape.storage_lease_unit_count +=
        frame.allocation.storage_lease_instances[i].location_count;
  }
  return analysis;
}

static std::string BuildMemoryControlSource() {
  return R"(
amdgpu.target<gfx1250> @target

low.kernel.def target<amdgpu.rdna4.gfx125x.core>(@target) abi_layout({constant_count = 0, direct_arg_count = 0, direct_arg_names = {}, direct_arg_offsets = [], direct_arg_parameter_indices = [], direct_arg_sizes = [], parameter_count = 3, resource_count = 3, resource_offsets = [0, 8, 16], resource_parameter_indices = [0, 1, 2], uses_kernarg_segment_ptr = true}) export("memory_control") workgroup_size(32, 1, 1) workgroup_count(1, 1, 1) @memory_control() {
  %lane = low.live_in<amdgpu.workitem_id.x> : reg<amdgpu.vgpr>
  %lhs_view = low.resource<hal_binding> {extent = 128, index = 0, source_type = hal.buffer} : reg<amdgpu.sgpr x2>
  %rhs_view = low.resource<hal_binding> {extent = 128, index = 1, source_type = hal.buffer} : reg<amdgpu.sgpr x2>
  %output_view = low.resource<hal_binding> {extent = 128, index = 2, source_type = hal.buffer} : reg<amdgpu.sgpr x2>
  %address = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%lane) {imm32 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %lhs = low.op<amdgpu.global_load_b32_saddr>(%address, %lhs_view) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr>
  %rhs = low.op<amdgpu.global_load_b32_saddr>(%address, %rhs_view) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr>
  %result = low.op<amdgpu.v_add_f32>(%lhs, %rhs) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  low.op<amdgpu.global_store_b32_saddr>(%address, %result, %output_view) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>)
  low.return
}
)";
}

static std::string BuildMatrixSource(const FixtureSpec& spec) {
  std::ostringstream source;
  source << R"(
amdgpu.target<gfx1250> @target

low.kernel.def target<amdgpu.rdna4.gfx125x.core>(@target) abi_layout({constant_count = 0, direct_arg_count = 0, direct_arg_names = {}, direct_arg_offsets = [], direct_arg_parameter_indices = [], direct_arg_sizes = [], parameter_count = 3, resource_count = 3, resource_offsets = [0, 8, 16], resource_parameter_indices = [0, 1, 2], uses_kernarg_segment_ptr = true}) export(")"
         << spec.name
         << R"(") workgroup_size(32, 1, 1) workgroup_count(1, 1, 1) @)"
         << spec.name << R"(() {
  %lane = low.live_in<amdgpu.workitem_id.x> : reg<amdgpu.vgpr>
  %input_view = low.resource<hal_binding> {extent = 4096, index = 0, source_type = hal.buffer} : reg<amdgpu.sgpr x2>
  %weight_view = low.resource<hal_binding> {extent = 4096, index = 1, source_type = hal.buffer} : reg<amdgpu.sgpr x2>
  %output_view = low.resource<hal_binding> {extent = 4096, index = 2, source_type = hal.buffer} : reg<amdgpu.sgpr x2>
  %zero0 = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %zero1 = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %zero2 = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %zero3 = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %zero4 = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %zero5 = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %zero6 = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %zero7 = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %initial_accumulator = low.concat(%zero0, %zero1, %zero2, %zero3, %zero4, %zero5, %zero6, %zero7) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x8>
  %lane_column = low.op<amdgpu.v_and_b32.src0_inline>(%lane) {imm32 = 15} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %row_address = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%lane_column) {imm32 = 5} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
)";

  std::string accumulator = "%initial_accumulator";
  for (uint32_t phase = 0; phase < spec.generated_matrix.phase_count; ++phase) {
    if (phase != 0) {
      source << "  low.br ^_bb" << phase << "(" << accumulator
             << ": reg<amdgpu.vgpr x8>)\n"
             << "^_bb" << phase << "(%incoming_accumulator" << phase
             << ": reg<amdgpu.vgpr x8>):\n";
      accumulator = "%incoming_accumulator" + std::to_string(phase);
    }

    const uint32_t byte_offset = phase * 32;
    source << "  %lhs_a" << phase
           << " = low.op<amdgpu.global_load_b128_saddr>(%row_address, "
              "%input_view) {offset = 0} : (reg<amdgpu.vgpr>, "
              "reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>\n"
           << "  %lhs_b" << phase
           << " = low.op<amdgpu.global_load_b128_saddr>(%row_address, "
              "%input_view) {offset = 16} : (reg<amdgpu.vgpr>, "
              "reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>\n"
           << "  %lhs" << phase << " = low.concat(%lhs_a" << phase << ", %lhs_b"
           << phase
           << ") : (reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x4>) -> "
              "reg<amdgpu.vgpr x8>\n"
           << "  %rhs_a" << phase
           << " = low.op<amdgpu.global_load_b128_saddr>(%row_address, "
              "%weight_view) {offset = "
           << byte_offset
           << "} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> "
              "reg<amdgpu.vgpr x4>\n"
           << "  %rhs_b" << phase
           << " = low.op<amdgpu.global_load_b128_saddr>(%row_address, "
              "%weight_view) {offset = "
           << byte_offset + 16
           << "} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> "
              "reg<amdgpu.vgpr x4>\n"
           << "  %rhs" << phase << " = low.concat(%rhs_a" << phase << ", %rhs_b"
           << phase
           << ") : (reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x4>) -> "
              "reg<amdgpu.vgpr x8>\n"
           << "  %accumulator_copy" << phase << " = low.copy " << accumulator
           << " : reg<amdgpu.vgpr x8> -> reg<amdgpu.vgpr x8>\n"
           << "  %accumulator" << phase
           << " = low.op<amdgpu.v_wmma_f32_16x16x32_bf16>(%lhs" << phase
           << ", %rhs" << phase << ", %accumulator_copy" << phase
           << ") {matrix_a_reuse = 0, matrix_b_reuse = 0} : "
              "(reg<amdgpu.vgpr x8>, reg<amdgpu.vgpr x8>, "
              "reg<amdgpu.vgpr x8>) -> %accumulator_copy"
           << phase << " as reg<amdgpu.vgpr x8>\n";
    accumulator = "%accumulator" + std::to_string(phase);

    std::string address = "%row_address";
    for (uint32_t i = 0;
         i < spec.generated_matrix.dependent_valu_packets_per_phase; ++i) {
      const std::string next_address =
          "%address" + std::to_string(phase) + "_chain" + std::to_string(i);
      source << "  " << next_address << " = low.op<amdgpu.v_add_u32>("
             << address
             << ", %lane) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
                "reg<amdgpu.vgpr>\n";
      address = next_address;
    }

    // Keep every scaling packet semantically live. Independent leaves expose
    // width while the balanced reduction bounds dependency depth; the final
    // address feeds both stores.
    const uint32_t independent_leaf_count =
        (spec.generated_matrix.valu_packets_per_phase -
         spec.generated_matrix.dependent_valu_packets_per_phase) /
        2;
    std::vector<std::string> address_values = {address};
    for (uint32_t i = 0; i < independent_leaf_count; ++i) {
      const std::string leaf_address =
          "%address" + std::to_string(phase) + "_leaf" + std::to_string(i);
      source << "  " << leaf_address
             << " = low.op<amdgpu.v_add_u32>(%row_address, %lane) : "
                "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
                "reg<amdgpu.vgpr>\n";
      address_values.push_back(leaf_address);
    }
    uint32_t reduction_index = 0;
    while (address_values.size() > 1) {
      std::vector<std::string> next_values;
      next_values.reserve((address_values.size() + 1) / 2);
      for (iree_host_size_t i = 0; i < address_values.size(); i += 2) {
        if (i + 1 == address_values.size()) {
          next_values.push_back(address_values[i]);
          continue;
        }
        const std::string reduced_address = "%address" + std::to_string(phase) +
                                            "_reduce" +
                                            std::to_string(reduction_index++);
        source << "  " << reduced_address << " = low.op<amdgpu.v_add_u32>("
               << address_values[i] << ", " << address_values[i + 1]
               << ") : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
                  "reg<amdgpu.vgpr>\n";
        next_values.push_back(reduced_address);
      }
      address_values = std::move(next_values);
    }
    address = address_values.front();

    source << "  %result_low" << phase << " = low.slice " << accumulator
           << "[0] : reg<amdgpu.vgpr x8> -> reg<amdgpu.vgpr x4>\n"
           << "  low.op<amdgpu.global_store_b128_saddr>(" << address
           << ", %result_low" << phase
           << ", %output_view) {offset = " << byte_offset
           << "} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>, "
              "reg<amdgpu.sgpr x2>)\n"
           << "  %result_high" << phase << " = low.slice " << accumulator
           << "[4] : reg<amdgpu.vgpr x8> -> reg<amdgpu.vgpr x4>\n"
           << "  low.op<amdgpu.global_store_b128_saddr>(" << address
           << ", %result_high" << phase
           << ", %output_view) {offset = " << byte_offset + 16
           << "} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>, "
              "reg<amdgpu.sgpr x2>)\n";
  }
  source << "  low.return\n}\n";
  return source.str();
}

static std::string BuildGeneratedLowSource(const FixtureSpec& spec) {
  return spec.generated_matrix.phase_count == 0 ? BuildMemoryControlSource()
                                                : BuildMatrixSource(spec);
}

static iree_string_view_t AttentionBf16Source() {
  const iree_file_toc_t* files = packet_plan_attention_bf16_create();
  return iree_make_string_view(reinterpret_cast<const char*>(files[0].data),
                               files[0].size);
}

struct PlanMetrics {
  // Arena bytes occupied by plan allocations.
  iree_host_size_t plan_used_bytes = 0;
  // Arena bytes retained from the block pool.
  iree_host_size_t plan_owned_bytes = 0;
  // Wait actions produced by the selected plan component.
  iree_host_size_t wait_action_count = 0;
  // Hazard records retained across all selected plan components.
  iree_host_size_t hazard_record_count = 0;
  // Progress records retained across all selected plan components.
  iree_host_size_t progress_record_count = 0;
  // Materialized wait packets in a complete packet plan.
  iree_host_size_t wait_packet_count = 0;
  // Wait-counter states in a complete packet plan.
  iree_host_size_t wait_state_count = 0;
  // VOPD pairs selected in a complete packet plan.
  iree_host_size_t vopd_pair_count = 0;
};

class PacketPlanFixture {
 public:
  explicit PacketPlanFixture(const FixtureSpec& spec) {
    iree_arena_block_pool_initialize(kArenaBlockSize, iree_allocator_system(),
                                     &module_block_pool_);
    iree_arena_block_pool_initialize(kArenaBlockSize, iree_allocator_system(),
                                     &frame_block_pool_);
    iree_arena_block_pool_initialize(kArenaBlockSize, iree_allocator_system(),
                                     &plan_block_pool_);
    iree_arena_initialize(&frame_block_pool_, &frame_arena_);
    iree_arena_initialize(&plan_block_pool_, &plan_arena_);
    iree_arena_initialize(&plan_block_pool_, &transient_arena_);

    AbortOnError(loom_target_environment_initialize(
        &loom_amdgpu_target_provider_set, &target_environment_));
    loom_context_initialize(iree_allocator_system(), &context_);
    AbortOnError(loom_op_registry_register_all_dialects(&context_));
    AbortOnError(loom_target_environment_register_context(&target_environment_,
                                                          &context_));
    AbortOnError(loom_context_finalize(&context_));
    AbortOnError(loom_target_environment_initialize_low_descriptor_registry(
        &target_environment_, &target_registry_));

    std::string generated_source;
    iree_string_view_t source = iree_string_view_empty();
    if (spec.input_stage == FixtureInputStage::kAuthoredSource) {
      source = AttentionBf16Source();
    } else {
      generated_source = BuildGeneratedLowSource(spec);
      source = iree_make_string_view(generated_source.data(),
                                     generated_source.size());
    }
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &target_registry_.registry, &parse_options.low_asm_environment);
    AbortOnError(loom_text_parse(source, iree_make_cstring_view(spec.name),
                                 &context_, &module_block_pool_, &parse_options,
                                 &module_));
    if (module_ == nullptr) {
      std::abort();
    }

    if (spec.input_stage == FixtureInputStage::kAuthoredSource) {
      loom_compile_pipeline_options_t pipeline_options = {};
      loom_compile_pipeline_options_initialize(&pipeline_options);
      pipeline_options.target_environment = &target_environment_;
      pipeline_options.low_descriptor_registry = &target_registry_;
      pipeline_options.diagnostic_sink = {
          /*.fn=*/loom_diagnostic_stderr_sink,
          /*.user_data=*/nullptr,
      };
      pipeline_options.max_errors = 20;
      loom_compile_pipeline_result_t pipeline_result = {};
      iree_status_t status = loom_compile_run_pipeline(
          module_, &pipeline_options, &module_block_pool_, &pipeline_result);
      const uint32_t error_count = pipeline_result.pass.error_count;
      loom_compile_pipeline_result_deinitialize(&pipeline_result);
      AbortOnError(status);
      if (error_count != 0) {
        std::abort();
      }
    }

    loom_op_t* low_function = FindFirstLowFunction(module_);
    if (low_function == nullptr) {
      std::abort();
    }
    loom_low_verify_options_t verify_options = {};
    verify_options.descriptor_registry = &target_registry_.registry;
    verify_options.max_errors = 20;
    loom_low_verify_result_t verify_result = {};
    AbortOnError(
        loom_low_verify_module(module_, &verify_options, &verify_result));
    if (verify_result.error_count != 0) {
      std::abort();
    }

    loom_symbol_fact_table_t symbol_facts = {};
    loom_symbol_fact_table_initialize(&symbol_facts, &frame_arena_);
    loom_low_resolved_target_t resolved_target = {};
    AbortOnError(loom_low_resolve_function_target(
        module_, &symbol_facts, low_function,
        /*function_target_facts=*/nullptr, &target_registry_.registry,
        /*emitter=*/{}, &resolved_target));
    if (resolved_target.descriptor_set == nullptr) {
      std::abort();
    }

    loom_low_schedule_pair_affinity_list_t pair_affinities =
        loom_low_schedule_pair_affinity_list_empty();
    AbortOnError(loom_amdgpu_vopd_build_schedule_pair_affinities(
        &resolved_target, &frame_arena_, &pair_affinities));
    const loom_low_schedule_structural_state_read_list_t
        structural_state_reads =
            loom_amdgpu_descriptor_structural_state_reads();

    loom_amdgpu_hal_kernel_abi_verify_result_t abi_verify_result = {};
    AbortOnError(loom_amdgpu_hal_kernel_abi_verify_low(
        module_, low_function, resolved_target.descriptor_set,
        /*max_errors=*/20, /*emitter=*/{}, &abi_verify_result, &frame_arena_));
    if (abi_verify_result.error_count != 0) {
      std::abort();
    }
    loom_low_storage_lease_provider_t storage_lease_provider = {};
    loom_amdgpu_storage_lease_provider(&storage_lease_provider);
    loom_low_emission_frame_options_t frame_options = {};
    frame_options.descriptor_registry = &target_registry_.registry;
    frame_options.memory_access_table = loom_low_memory_access_table_empty();
    frame_options.residency_model =
        loom_amdgpu_occupancy_residency_model(&resolved_target);
    frame_options.schedule_pair_affinities = pair_affinities;
    frame_options.schedule_structural_state_reads = structural_state_reads;
    frame_options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
    frame_options.allocation_fixed_values = abi_verify_result.fixed_values;
    frame_options.allocation_fixed_value_count =
        abi_verify_result.fixed_value_count;
    frame_options.storage_lease_provider = &storage_lease_provider;
    AbortOnError(loom_low_emission_frame_build(
        module_, low_function, &frame_options, &frame_arena_, &frame_));
    if (frame_.schedule.error_count != 0 ||
        frame_.allocation.error_count != 0 ||
        frame_.allocation.spill_plan_count != 0) {
      std::abort();
    }

    analysis_ = AnalyzeFrame(frame_);
  }

  ~PacketPlanFixture() {
    iree_arena_deinitialize(&transient_arena_);
    iree_arena_deinitialize(&plan_arena_);
    iree_arena_deinitialize(&frame_arena_);
    if (module_ != nullptr) {
      loom_module_free(module_);
    }
    loom_context_deinitialize(&context_);
    loom_target_environment_deinitialize(&target_environment_);
    iree_arena_block_pool_deinitialize(&plan_block_pool_);
    iree_arena_block_pool_deinitialize(&frame_block_pool_);
    iree_arena_block_pool_deinitialize(&module_block_pool_);
  }

  PacketPlanFixture(const PacketPlanFixture&) = delete;
  PacketPlanFixture& operator=(const PacketPlanFixture&) = delete;

  const loom_low_emission_frame_t& frame() const { return frame_; }
  iree_arena_allocator_t* plan_arena() { return &plan_arena_; }
  iree_arena_allocator_t* transient_arena() { return &transient_arena_; }
  iree_host_size_t frame_arena_used_bytes() const {
    return frame_arena_.used_allocation_size;
  }
  const FrameAnalysis& analysis() const { return analysis_; }

 private:
  iree_arena_block_pool_t module_block_pool_ = {};
  iree_arena_block_pool_t frame_block_pool_ = {};
  iree_arena_block_pool_t plan_block_pool_ = {};
  iree_arena_allocator_t frame_arena_ = {};
  iree_arena_allocator_t plan_arena_ = {};
  // Scratch storage discarded after each direct wait-plan build.
  iree_arena_allocator_t transient_arena_ = {};
  loom_target_environment_t target_environment_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t target_registry_ = {};
  loom_module_t* module_ = nullptr;
  loom_low_emission_frame_t frame_ = {};
  FrameAnalysis analysis_ = {};
};

static PlanMetrics BuildReferencePlan(PacketPlanFixture& fixture,
                                      PlanComponent component) {
  iree_arena_reset(fixture.plan_arena());
  PlanMetrics metrics = {};
  if (component == PlanComponent::kWait) {
    iree_arena_reset(fixture.transient_arena());
    loom_amdgpu_wait_plan_t plan = {};
    AbortOnError(loom_amdgpu_wait_plan_build(
        &fixture.frame().schedule, &fixture.frame().allocation,
        fixture.plan_arena(), fixture.transient_arena(), &plan));
    metrics.wait_action_count = plan.action_count;
    metrics.hazard_record_count = plan.hazard_plan.record_count;
    metrics.progress_record_count = plan.progress.record_count;
  } else {
    loom_amdgpu_packet_plan_t plan = {};
    AbortOnError(loom_amdgpu_packet_plan_build(&fixture.frame().schedule,
                                               &fixture.frame().allocation,
                                               fixture.plan_arena(), &plan));
    metrics.wait_action_count = plan.wait_plan.action_count;
    metrics.hazard_record_count = plan.wait_plan.hazard_plan.record_count +
                                  plan.wait_states.hazard_plan.record_count;
    metrics.progress_record_count = plan.wait_plan.progress.record_count +
                                    plan.wait_states.progress.record_count;
    metrics.wait_packet_count = plan.wait_packets.packet_count;
    metrics.wait_state_count = plan.wait_states.state_count;
    metrics.vopd_pair_count = plan.vopd_plan.pair_count;
  }
  metrics.plan_used_bytes = fixture.plan_arena()->used_allocation_size;
  metrics.plan_owned_bytes = fixture.plan_arena()->total_allocation_size;
  return metrics;
}

static void RecordMetrics(benchmark::State& state,
                          const PacketPlanFixture& fixture,
                          const PlanMetrics& metrics) {
  const FrameAnalysis& analysis = fixture.analysis();
  state.counters["assignments"] =
      static_cast<double>(analysis.shape.assignment_count);
  state.counters["blocks"] = static_cast<double>(analysis.shape.block_count);
  state.counters["frame_arena_bytes"] =
      static_cast<double>(fixture.frame_arena_used_bytes());
  state.counters["planned_hazards"] =
      static_cast<double>(metrics.hazard_record_count);
  state.counters["matrix_packets"] =
      static_cast<double>(analysis.shape.matrix_packet_count);
  state.counters["packets"] = static_cast<double>(analysis.shape.packet_count);
  state.counters["plan_arena_owned_bytes"] =
      static_cast<double>(metrics.plan_owned_bytes);
  state.counters["plan_arena_used_bytes"] =
      static_cast<double>(metrics.plan_used_bytes);
  state.counters["progress_events"] =
      static_cast<double>(metrics.progress_record_count);
  state.counters["storage_leases"] =
      static_cast<double>(analysis.shape.storage_lease_count);
  state.counters["storage_lease_units"] =
      static_cast<double>(analysis.shape.storage_lease_unit_count);
  state.counters["storage_releases"] =
      static_cast<double>(analysis.shape.storage_release_count);
  state.counters["vopd_pairs"] = static_cast<double>(metrics.vopd_pair_count);
  state.counters["wait_actions"] =
      static_cast<double>(metrics.wait_action_count);
  state.counters["wait_packets"] =
      static_cast<double>(metrics.wait_packet_count);
  state.counters["wait_states"] = static_cast<double>(metrics.wait_state_count);
  state.counters["packets_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * analysis.shape.packet_count,
      benchmark::Counter::kIsRate);
  state.counters["plans_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);

  char label[64] = {};
  std::snprintf(label, sizeof(label), "frame_signature=%016" PRIx64,
                analysis.signature);
  state.SetLabel(label);
}

static void BenchmarkPlan(benchmark::State& state, const FixtureSpec& spec,
                          PlanComponent component) {
  PacketPlanFixture fixture(spec);
  const PlanMetrics reference_metrics = BuildReferencePlan(fixture, component);
  for (auto _ : state) {
    iree_arena_reset(fixture.plan_arena());
    if (component == PlanComponent::kWait) {
      iree_arena_reset(fixture.transient_arena());
      loom_amdgpu_wait_plan_t plan = {};
      AbortOnError(loom_amdgpu_wait_plan_build(
          &fixture.frame().schedule, &fixture.frame().allocation,
          fixture.plan_arena(), fixture.transient_arena(), &plan));
      benchmark::DoNotOptimize(plan);
    } else {
      loom_amdgpu_packet_plan_t plan = {};
      AbortOnError(loom_amdgpu_packet_plan_build(&fixture.frame().schedule,
                                                 &fixture.frame().allocation,
                                                 fixture.plan_arena(), &plan));
      benchmark::DoNotOptimize(plan);
    }
  }
  RecordMetrics(state, fixture, reference_metrics);
}

static void BM_WaitPlan_MemoryControl(benchmark::State& state) {
  BenchmarkPlan(state, kMemoryControl, PlanComponent::kWait);
}
BENCHMARK(BM_WaitPlan_MemoryControl)
    ->Iterations(2000)
    ->Unit(benchmark::kNanosecond);

static void BM_PacketPlan_MemoryControl(benchmark::State& state) {
  BenchmarkPlan(state, kMemoryControl, PlanComponent::kComplete);
}
BENCHMARK(BM_PacketPlan_MemoryControl)
    ->Iterations(2000)
    ->Unit(benchmark::kNanosecond);

static void BM_WaitPlan_MatrixSingleBlockCanary(benchmark::State& state) {
  BenchmarkPlan(state, kMatrixSingleBlockCanary, PlanComponent::kWait);
}
BENCHMARK(BM_WaitPlan_MatrixSingleBlockCanary)
    ->Iterations(1000)
    ->Unit(benchmark::kNanosecond);

static void BM_PacketPlan_MatrixSingleBlockCanary(benchmark::State& state) {
  BenchmarkPlan(state, kMatrixSingleBlockCanary, PlanComponent::kComplete);
}
BENCHMARK(BM_PacketPlan_MatrixSingleBlockCanary)
    ->Iterations(1000)
    ->Unit(benchmark::kNanosecond);

static void BM_WaitPlan_MatrixDependencyCanary(benchmark::State& state) {
  BenchmarkPlan(state, kMatrixDependencyCanary, PlanComponent::kWait);
}
BENCHMARK(BM_WaitPlan_MatrixDependencyCanary)
    ->Iterations(500)
    ->Unit(benchmark::kNanosecond);

static void BM_PacketPlan_MatrixDependencyCanary(benchmark::State& state) {
  BenchmarkPlan(state, kMatrixDependencyCanary, PlanComponent::kComplete);
}
BENCHMARK(BM_PacketPlan_MatrixDependencyCanary)
    ->Iterations(500)
    ->Unit(benchmark::kNanosecond);

static void BM_WaitPlan_MatrixLeaseOverlapStress(benchmark::State& state) {
  BenchmarkPlan(state, kMatrixLeaseOverlapStress, PlanComponent::kWait);
}
BENCHMARK(BM_WaitPlan_MatrixLeaseOverlapStress)
    ->Iterations(10)
    ->Unit(benchmark::kNanosecond);

static void BM_PacketPlan_MatrixLeaseOverlapStress(benchmark::State& state) {
  BenchmarkPlan(state, kMatrixLeaseOverlapStress, PlanComponent::kComplete);
}
BENCHMARK(BM_PacketPlan_MatrixLeaseOverlapStress)
    ->Iterations(10)
    ->Unit(benchmark::kNanosecond);

static void BM_WaitPlan_AttentionBf16(benchmark::State& state) {
  BenchmarkPlan(state, kAttentionBf16, PlanComponent::kWait);
}
BENCHMARK(BM_WaitPlan_AttentionBf16)
    ->Iterations(10)
    ->Unit(benchmark::kNanosecond);

static void BM_PacketPlan_AttentionBf16(benchmark::State& state) {
  BenchmarkPlan(state, kAttentionBf16, PlanComponent::kComplete);
}
BENCHMARK(BM_PacketPlan_AttentionBf16)
    ->Iterations(10)
    ->Unit(benchmark::kNanosecond);

}  // namespace
