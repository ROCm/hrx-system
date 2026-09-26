// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scaling and memory benchmarks for verified Low programs. Chain depth,
// independent components, register width and shared-source fan-out exercise
// distinct allocation costs. Timing uses the ordinary system allocator; a
// separate untimed pass observes requested live memory and allocation traffic.

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/low_registry.h"

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

struct AllocationObserver {
  // Requested bytes currently owned through the observed allocator.
  uint64_t live_bytes = 0;
  // Highest live-byte count since the last measurement boundary.
  uint64_t peak_bytes = 0;
  // Cumulative requested sizes of successful allocation/reallocation commands.
  uint64_t requested_bytes = 0;
  // Number of successful allocation/reallocation commands.
  uint64_t allocation_count = 0;
  // Outstanding allocation sizes; observer storage uses an unobserved
  // allocator.
  std::unordered_map<void*, size_t> sizes;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* parameters, void** pointer) {
    auto& observer = *static_cast<AllocationObserver*>(self);
    const auto delegate = iree_allocator_system();
    void* old_pointer = nullptr;
    size_t old_size = 0;
    if (command == IREE_ALLOCATOR_COMMAND_FREE ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      old_pointer = *pointer;
      if (old_pointer != nullptr) {
        old_size = observer.sizes.at(old_pointer);
      }
    }
    const iree_status_t status =
        delegate.ctl(delegate.self, command, parameters, pointer);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    if (command == IREE_ALLOCATOR_COMMAND_FREE ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      observer.live_bytes -= old_size;
      if (old_pointer != nullptr) {
        observer.sizes.erase(old_pointer);
      }
    }
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      const size_t size =
          static_cast<const iree_allocator_alloc_params_t*>(parameters)
              ->byte_length;
      observer.sizes.emplace(*pointer, size);
      observer.live_bytes += size;
      observer.peak_bytes = std::max(observer.peak_bytes, observer.live_bytes);
      observer.requested_bytes += size;
      ++observer.allocation_count;
    }
    return status;
  }

  iree_allocator_t allocator() { return {this, Control}; }
};

enum class Shape { kLinear, kLoop, kLoopRelocation, kBranch, kTied, kFanout };
enum class Phase { kModel, kLiveness, kPlacement, kAllocation };

std::string MakeSource(uint32_t chain_length, uint32_t component_count,
                       uint32_t width, Shape shape) {
  const std::string type = "reg<test.i32 x" + std::to_string(width) + ">";
  if (shape == Shape::kFanout) {
    const uint32_t count = chain_length * component_count;
    std::string source =
        "test.target<low_core> @target\n"
        "low.func.def target<test.low.core>(@target) @kernel(";
    for (uint32_t i = 0; i < component_count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "%seed" + std::to_string(i) + ": " + type;
    }
    source += ") -> (";
    for (uint32_t i = 0; i < count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += type;
    }
    source += ") asm {\n";
    for (uint32_t i = 0; i < count; ++i) {
      source += "  %copy" + std::to_string(i) + " = copy %seed" +
                std::to_string(i / chain_length) + " : " + type + " -> " +
                type + "\n";
    }
    for (uint32_t i = 0; i < count; ++i) {
      source += "  %next" + std::to_string(i) + " = " +
                (width == 1 ? "test.tied.any " : "test.early_tied.v4i32 ") +
                "%copy" + std::to_string(i) + "\n";
    }
    source += "  return ";
    for (uint32_t i = 0; i < count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "%next" + std::to_string(i);
    }
    return source + "\n}\n";
  }
  if (shape == Shape::kLoopRelocation) {
    std::string source =
        "test.target<low_core> @target\n"
        "low.func.def target<test.low.core>(@target) @kernel("
        "%condition: reg<test.i32>";
    for (uint32_t i = 0; i < component_count; ++i) {
      source += ", %seed" + std::to_string(i) + ": reg<test.i32>";
    }
    source += ", %lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (";
    for (uint32_t i = 0; i < component_count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "reg<test.i32>";
    }
    source += ") asm {\n  low.br ^loop(";
    for (uint32_t i = 0; i < component_count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "%seed" + std::to_string(i) + ": reg<test.i32>";
    }
    source += ")\n^loop(";
    for (uint32_t i = 0; i < component_count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "%state" + std::to_string(i) + ": reg<test.i32>";
    }
    source +=
        "):\n  low.cond_br %condition, ^body, ^exit : reg<test.i32>\n"
        "^body:\n";
    if (chain_length == 1) {
      source +=
          "  %early0 = test.add.i32 %lhs, %rhs\n"
          "  %early1 = test.mul.i32 %early0, %rhs\n"
          "  %early2 = test.add.i32 %early1, %rhs\n"
          "  %acc0 = test.add.i32 %state0, %early2\n";
      for (uint32_t i = 1; i < component_count; ++i) {
        source += "  %acc" + std::to_string(i) + " = test.add.i32 %acc" +
                  std::to_string(i - 1) + ", %state" + std::to_string(i) + "\n";
      }
    } else {
      Require(chain_length == component_count && component_count >= 2,
              "Dense relocation requires one color per component");
      for (uint32_t i = 0; i < component_count; ++i) {
        source += "  %early" + std::to_string(i) +
                  (i % 2 == 0 ? " = test.add.i32 %lhs, %rhs\n"
                              : " = test.mul.i32 %lhs, %rhs\n");
      }
      source += "  %early_acc0 = test.add.i32 %early0, %early1\n";
      for (uint32_t i = 2; i < component_count; ++i) {
        source += "  %early_acc" + std::to_string(i - 1) +
                  " = test.add.i32 %early_acc" + std::to_string(i - 2) +
                  ", %early" + std::to_string(i) + "\n";
      }
    }
    for (uint32_t i = 0; i < component_count; ++i) {
      source += "  %next" + std::to_string(i);
      if (chain_length == 1) {
        source += i % 2 == 0 ? " = test.mul.i32 %lhs, %rhs\n"
                             : " = test.add.i32 %lhs, %rhs\n";
      } else {
        source += " = test.add.i32 %state" + std::to_string(i) + ", %rhs\n";
      }
    }
    if (chain_length == 1) {
      source += "  %sink = test.mul.i32 %acc" +
                std::to_string(component_count - 1) + ", %rhs\n";
    } else {
      source += "  %sink = test.mul.i32 %early_acc" +
                std::to_string(component_count - 2) + ", %rhs\n";
    }
    source += "  low.br ^loop(";
    for (uint32_t i = 0; i < component_count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "%next" + std::to_string(i) + ": reg<test.i32>";
    }
    source += ")\n^exit:\n  return ";
    for (uint32_t i = 0; i < component_count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "%state" + std::to_string(i);
    }
    return source + "\n}\n";
  }
  const bool has_loop = shape == Shape::kLoop || shape == Shape::kBranch;
  std::string source =
      "test.target<low_core> @target\n"
      "low.func.def target<test.low.core>(@target) @kernel("
      "%condition: reg<test.i32>";
  for (uint32_t i = 0; i < component_count; ++i) {
    source += ", %seed" + std::to_string(i) + ": " + type;
  }
  source += ") -> (";
  for (uint32_t i = 0; i < component_count; ++i) {
    if (i != 0) {
      source += ", ";
    }
    source += type;
  }
  source += ") asm {\n";
  auto payload = [&](const std::string& prefix) {
    std::string text;
    for (uint32_t i = 0; i < component_count; ++i) {
      if (i != 0) {
        text += ", ";
      }
      text += "%" + prefix + std::to_string(i) + ": " + type;
    }
    return text;
  };
  if (has_loop) {
    source +=
        "  low.br ^loop(" + payload("seed") + ")\n^loop(" + payload("state") +
        "):\n"
        "  low.cond_br %condition, ^body, ^exit : reg<test.i32>\n^body:\n";
    if (shape == Shape::kBranch) {
      source += "  low.cond_br %condition, ^left, ^right : reg<test.i32>\n";
    }
  }
  const uint32_t branches = shape == Shape::kBranch ? 2 : 1;
  std::string result_names;
  for (uint32_t branch = 0; branch < branches; ++branch) {
    std::string result_payload;
    result_names.clear();
    if (shape == Shape::kBranch) {
      source += branch == 0 ? "^left:\n" : "^right:\n";
    }
    for (uint32_t component = 0; component < component_count; ++component) {
      std::string value = std::string(has_loop ? "%state" : "%seed") +
                          std::to_string(component);
      const std::string prefix =
          "b" + std::to_string(branch) + "c" + std::to_string(component) + "_";
      for (uint32_t i = 0; i < chain_length; ++i) {
        const std::string copy = "%" + prefix + "copy" + std::to_string(i);
        const std::string next = "%" + prefix + "next" + std::to_string(i);
        if (shape != Shape::kTied) {
          source += "  " + copy + " = copy " + value + " : " + type + " -> " +
                    type + "\n";
        }
        source += "  " + next + " = " +
                  (width == 1 ? "test.tied.any " : "test.early_tied.v4i32 ") +
                  (shape == Shape::kTied ? value : copy) + "\n";
        value = next;
      }
      if (component != 0) {
        result_payload += ", ";
        result_names += ", ";
      }
      result_payload += value + ": " + type;
      result_names += value;
    }
    if (has_loop) {
      source += "  low.br ^loop(" + result_payload + ")\n";
    }
  }
  if (has_loop) {
    source += "^exit:\n";
    result_names.clear();
    for (uint32_t i = 0; i < component_count; ++i) {
      if (i != 0) {
        result_names += ", ";
      }
      result_names += "%state" + std::to_string(i);
    }
  }
  source += "  return " + result_names + "\n}\n";
  return source;
}

struct RunResult {
  // Requested bytes in the result arena immediately before destruction.
  uint64_t used_bytes = 0;
  // Bytes owned by that arena, including block slack and oversized storage.
  uint64_t owned_bytes = 0;
  // Function-local value count processed by the selected phase.
  uint32_t value_count = 0;
  // Copy decisions produced by allocation; zero for analysis-only phases.
  uint32_t copy_count = 0;
  // Copies requiring real moves; live fan-out copies must remain independent.
  uint32_t materialized_copy_count = 0;
  // Physical backedge moves remaining after loop-edge relocation.
  uint64_t backedge_move_count = 0;
};

class AllocationBenchmark {
 public:
  AllocationBenchmark(const std::string& source, Phase phase, Shape shape,
                      uint32_t chain_length, uint32_t component_count,
                      iree_allocator_t allocator)
      : phase_(phase) {
    iree_arena_block_pool_initialize(128 * 1024, allocator, &source_pool_);
    iree_arena_block_pool_initialize(128 * 1024, allocator, &analysis_pool_);
    iree_arena_initialize(&analysis_pool_, &base_arena_);
    loom_context_initialize(allocator, &context_);
    Register(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    Register(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    Register(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    Register(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_CHECK_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    loom_text_parse_options_t parse_options = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &parse_options.low_asm_environment);
    IREE_CHECK_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("allocation_benchmark.loom"), &context_,
                        &source_pool_, &parse_options, &module_));
    Require(module_ != nullptr, "Parsing failed");
    loom_low_verify_options_t verify_options = {};
    verify_options.descriptor_registry = &registry_.registry;
    auto scratch = loom_low_verify_scratch_for_module(module_);
    loom_low_verify_result_t verified = {};
    IREE_CHECK_OK(
        loom_low_verify_module(module_, &verify_options, &scratch, &verified));
    Require(verified.error_count == 0, "Low verification failed");
    auto name = loom_module_lookup_string(module_, IREE_SV("kernel"));
    auto symbol = loom_module_find_symbol(module_, name);
    Require(symbol != LOOM_SYMBOL_ID_INVALID, "Kernel symbol missing");
    function_ = module_->symbols.entries[symbol].defining_op;
    if (shape == Shape::kLoopRelocation) {
      const loom_region_t* body = loom_low_func_def_body(function_);
      Require(body->block_count == 4, "Relocation loop shape changed");
      backedge_terminator_ =
          loom_block_const_last_op(loom_region_const_block(body, 2));
      Require(loom_low_br_isa(backedge_terminator_),
              "Relocation loop backedge missing");
    }
    if (shape == Shape::kLoopRelocation && chain_length != 1) {
      fixed_values_.resize(component_count);
      std::vector<uint8_t> found(component_count);
      for (loom_value_id_t value_id = 0; value_id < module_->values.count;
           ++value_id) {
        const iree_string_view_t name =
            loom_module_value_name(module_, value_id);
        if (name.size <= 4 || std::memcmp(name.data, "next", 4) != 0) {
          continue;
        }
        uint32_t next_index = 0;
        const auto parsed =
            std::from_chars(name.data + 4, name.data + name.size, next_index);
        if (parsed.ec != std::errc{} || parsed.ptr != name.data + name.size ||
            next_index >= component_count) {
          continue;
        }
        auto& fixed_value = fixed_values_[next_index];
        fixed_value.value_id = value_id;
        fixed_value.location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
        fixed_value.location_base =
            next_index < 2 ? next_index : component_count + 1 + next_index;
        fixed_value.location_count = 1;
        found[next_index] = 1;
      }
      Require(std::all_of(found.begin(), found.end(),
                          [](uint8_t value) { return value != 0; }),
              "Dense relocation source values missing");
    }
    if (phase_ != Phase::kModel) {
      InitializeModel(&base_arena_, &model_);
    }
    if (phase_ == Phase::kPlacement) {
      IREE_CHECK_OK(loom_liveness_analyze_local_value_domain_with_dataflow(
          &model_.value_domain, &model_.liveness_dataflow,
          loom_liveness_order_empty(), &base_arena_, &liveness_));
    }
  }

  AllocationBenchmark(const AllocationBenchmark&) = delete;
  AllocationBenchmark& operator=(const AllocationBenchmark&) = delete;

  ~AllocationBenchmark() {
    if (phase_ != Phase::kModel) {
      loom_low_function_model_deinitialize(&model_);
    }
    iree_arena_deinitialize(&base_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&analysis_pool_);
    iree_arena_block_pool_deinitialize(&source_pool_);
  }

  RunResult Run() {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&analysis_pool_, &arena);
    RunResult result;
    if (phase_ == Phase::kModel) {
      loom_low_function_model_t model = {};
      InitializeModel(&arena, &model);
      result.value_count = model.value_domain.value_count;
      loom_low_function_model_deinitialize(&model);
    } else if (phase_ == Phase::kLiveness) {
      loom_liveness_analysis_t liveness = {};
      IREE_CHECK_OK(loom_liveness_analyze_local_value_domain_with_dataflow(
          &model_.value_domain, &model_.liveness_dataflow,
          loom_liveness_order_empty(), &arena, &liveness));
      result.value_count = liveness.value_count;
      benchmark::DoNotOptimize(liveness.intervals);
    } else if (phase_ == Phase::kPlacement) {
      loom_low_placement_table_t placement = {};
      IREE_CHECK_OK(loom_low_placement_analyze_region(
          module_, model_.body, model_.target.descriptor_set,
          &model_.value_domain, &liveness_,
          loom_low_placement_pair_use_list_empty(), &arena, &placement));
      result.value_count = placement.value_count;
      benchmark::DoNotOptimize(placement.relations);
    } else {
      loom_low_allocation_options_t options = {};
      options.fixed_values = fixed_values_.data();
      options.fixed_value_count = fixed_values_.size();
      loom_low_allocation_table_t allocation = {};
      IREE_CHECK_OK(
          loom_low_allocate_function(&model_, &options, &arena, &allocation));
      Require(allocation.error_count == 0, "Allocation failed");
      Require(allocation.spill_count == 0, "Unexpected spill");
      result.value_count = model_.value_domain.value_count;
      result.copy_count = allocation.copy_decision_count;
      result.materialized_copy_count = allocation.materialized_copy_count;
      if (backedge_terminator_ != nullptr) {
        result.backedge_move_count = UINT64_MAX;
        for (iree_host_size_t i = 0; i < allocation.edge_copy_group_count;
             ++i) {
          const auto& group = allocation.edge_copy_groups[i];
          if (group.terminator_op == backedge_terminator_) {
            result.backedge_move_count = group.move_group.moves.count;
            break;
          }
        }
      }
      benchmark::DoNotOptimize(allocation.assignments);
    }
    result.used_bytes = arena.used_allocation_size;
    result.owned_bytes = arena.total_allocation_size;
    iree_arena_deinitialize(&arena);
    return result;
  }

 private:
  using Vtables = const loom_op_vtable_t* const* (*)(iree_host_size_t*);
  void Register(loom_dialect_id_t dialect, Vtables function) {
    iree_host_size_t count = 0;
    const auto tables = function(&count);
    IREE_CHECK_OK(loom_context_register_dialect(&context_, dialect, tables,
                                                static_cast<uint16_t>(count)));
  }
  void InitializeModel(iree_arena_allocator_t* arena,
                       loom_low_function_model_t* model) {
    IREE_CHECK_OK(loom_low_function_model_initialize(
        module_, function_, nullptr, &registry_.registry, {},
        LOOM_LOW_FUNCTION_MODEL_FLAG_REGION_TREE, arena, model));
    Require(model->error_count == 0, "Function model failed");
  }

  // Shipping compiler boundary measured by each Run invocation.
  Phase phase_;
  // Pool retaining the parsed input across all phase invocations.
  iree_arena_block_pool_t source_pool_;
  // Pool shared by retained analyses and resettable per-invocation arenas.
  iree_arena_block_pool_t analysis_pool_;
  // Arena retaining the selected phase's prerequisite analyses.
  iree_arena_allocator_t base_arena_;
  // Dialect context used to parse and verify the authored Low programs.
  loom_context_t context_;
  // Immutable parsed and verified input module.
  loom_module_t* module_ = nullptr;
  // Function under test, borrowed from the input module.
  const loom_op_t* function_ = nullptr;
  // Target-independent machine descriptors used by verification/allocation.
  loom_target_low_descriptor_registry_t registry_ = {};
  // Retained function model when model construction is outside the timed phase.
  loom_low_function_model_t model_ = {};
  // Retained semantic liveness for placement-only measurements.
  loom_liveness_analysis_t liveness_ = {};
  // Dense relocation source colors fixed outside the header destination set.
  std::vector<loom_low_allocation_fixed_value_t> fixed_values_;
  // Generated loop backedge used to validate final edge-copy materialization.
  const loom_op_t* backedge_terminator_ = nullptr;
};

struct AllocationTraffic {
  // Cumulative requested sizes of allocation/reallocation commands.
  uint64_t bytes = 0;
  // Number of allocation/reallocation commands.
  uint64_t count = 0;
};

// System-allocation requests observed at the caller's allocator boundary.
// Libc overhead and transient storage inside realloc are not observable here.
struct MemoryObservation {
  // Live input, context, retained analyses and pooled setup storage.
  uint64_t setup_live_bytes = 0;
  // Peak requested memory during the first phase invocation.
  struct {
    // Absolute peak live requested bytes, including retained setup storage.
    uint64_t live_bytes = 0;
    // Additional peak live requested bytes beyond setup_live_bytes.
    uint64_t incremental_bytes = 0;
  } peak;
  // Allocation traffic during the first phase invocation.
  AllocationTraffic cold;
  // Allocation traffic on the second invocation using retained pools.
  AllocationTraffic warm;
  // Additional pooled bytes retained after the first result arena is destroyed.
  uint64_t retained_pool_bytes = 0;
};

MemoryObservation ObserveMemory(const std::string& source, Phase phase,
                                Shape shape, uint32_t chain_length,
                                uint32_t component_count) {
  AllocationObserver observer;
  MemoryObservation observation;
  {
    AllocationBenchmark observed(source, phase, shape, chain_length,
                                 component_count, observer.allocator());
    const auto before_live = observer.live_bytes;
    observation.setup_live_bytes = before_live;
    const auto before_requested = observer.requested_bytes;
    const auto before_count = observer.allocation_count;
    observer.peak_bytes = observer.live_bytes;
    observed.Run();
    observation.peak.incremental_bytes = observer.peak_bytes - before_live;
    observation.peak.live_bytes = observer.peak_bytes;
    observation.cold.bytes = observer.requested_bytes - before_requested;
    observation.cold.count = observer.allocation_count - before_count;
    observation.retained_pool_bytes = observer.live_bytes - before_live;
    const auto warm_before_requested = observer.requested_bytes;
    const auto warm_before_count = observer.allocation_count;
    observed.Run();
    observation.warm.bytes = observer.requested_bytes - warm_before_requested;
    observation.warm.count = observer.allocation_count - warm_before_count;
  }
  Require(observer.live_bytes == 0, "Observed allocation leak");
  Require(observer.sizes.empty(), "Observed zero-sized allocation leak");
  return observation;
}

void RunBenchmark(benchmark::State& state, Shape shape, Phase phase) {
  const uint32_t chain_length = static_cast<uint32_t>(state.range(0));
  const uint32_t component_count = static_cast<uint32_t>(state.range(1));
  const uint32_t width = static_cast<uint32_t>(state.range(2));
  const auto source = MakeSource(chain_length, component_count, width, shape);
  const auto memory =
      ObserveMemory(source, phase, shape, chain_length, component_count);
  AllocationBenchmark fixture(source, phase, shape, chain_length,
                              component_count, iree_allocator_system());
  RunResult result = fixture.Run();
  for (auto _ : state) {
    result = fixture.Run();
    benchmark::DoNotOptimize(result);
  }
  if (phase == Phase::kAllocation) {
    const uint32_t expected_copy_count =
        shape == Shape::kTied || shape == Shape::kLoopRelocation
            ? 0
            : chain_length * component_count *
                  (shape == Shape::kBranch ? 2 : 1);
    Require(result.copy_count == expected_copy_count, "Copy decisions missing");
    if (shape == Shape::kLoopRelocation) {
      Require(result.backedge_move_count == 0,
              "Loop-edge relocation left branch copies");
    }
    if (shape == Shape::kFanout) {
      Require(result.materialized_copy_count >=
                  expected_copy_count - component_count,
              "Independently live copies were incorrectly coalesced");
    }
  }
  state.counters["value_count"] = result.value_count;
  state.counters["copy_count"] = result.copy_count;
  state.counters["materialized_copy_count"] = result.materialized_copy_count;
  state.counters["backedge_move_count"] = result.backedge_move_count;
  state.counters["arena_used_bytes"] = result.used_bytes;
  state.counters["arena_owned_bytes"] = result.owned_bytes;
  state.counters["setup_live_requested_bytes"] = memory.setup_live_bytes;
  state.counters["cold_peak_incremental_bytes"] = memory.peak.incremental_bytes;
  state.counters["cold_peak_live_requested_bytes"] = memory.peak.live_bytes;
  state.counters["cold_allocation_bytes"] = memory.cold.bytes;
  state.counters["cold_allocation_count"] = memory.cold.count;
  state.counters["cold_retained_pool_bytes"] = memory.retained_pool_bytes;
  state.counters["warm_allocation_bytes"] = memory.warm.bytes;
  state.counters["warm_allocation_count"] = memory.warm.count;
  state.SetItemsProcessed(state.iterations() * result.value_count);
}

[[maybe_unused]] const bool kBenchmarksRegistered = [] {
  for (auto shape : {Shape::kLinear, Shape::kLoop, Shape::kLoopRelocation,
                     Shape::kBranch, Shape::kTied, Shape::kFanout}) {
    for (auto phase : {Phase::kModel, Phase::kLiveness, Phase::kPlacement,
                       Phase::kAllocation}) {
      if (shape == Shape::kLoopRelocation && phase != Phase::kAllocation) {
        continue;
      }
      const std::string name =
          "LowAllocation/" +
          std::string(shape == Shape::kLinear           ? "linear/"
                      : shape == Shape::kLoop           ? "loop/"
                      : shape == Shape::kLoopRelocation ? "loop_relocation/"
                      : shape == Shape::kBranch         ? "branch/"
                      : shape == Shape::kTied           ? "tied/"
                                                        : "fanout/") +
          (phase == Phase::kModel       ? "model"
           : phase == Phase::kLiveness  ? "liveness"
           : phase == Phase::kPlacement ? "placement"
                                        : "allocation");
      auto* registration = benchmark::RegisterBenchmark(
          name.c_str(),
          [=](benchmark::State& state) { RunBenchmark(state, shape, phase); });
      registration->ArgNames({"length", "components", "width"});
      if (shape == Shape::kLoopRelocation) {
        for (int64_t components : {32, 64, 128, 256, 512, 1024}) {
          registration->Args({1, components, 1});
        }
        for (int64_t components : {8, 16, 32, 64, 128, 256}) {
          registration->Args({components, components, 1});
        }
        continue;
      }
      for (int64_t count : {8, 16, 32, 64, 128, 256, 512, 1024, 2048}) {
        registration->Args({count, 1, 1});
      }
      for (int64_t components : {2, 4, 8, 16}) {
        registration->Args({64, components, 1});
      }
      for (int64_t count : {64, 256, 1024}) {
        registration->Args({count, 1, 4});
      }
    }
  }
  return true;
}();

}  // namespace
