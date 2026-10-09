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
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation/placement.h"
#include "loom/codegen/low/allocation/unit_liveness_builder.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/schedule/run.h"
#include "loom/codegen/low/storage_lease.h"
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
#include "loom/verify/verify.h"

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

bool ParseIndexedName(iree_string_view_t name, const char* prefix,
                      uint32_t index_limit, uint32_t* out_index) {
  const iree_host_size_t prefix_length = std::strlen(prefix);
  if (name.size <= prefix_length ||
      std::memcmp(name.data, prefix, prefix_length) != 0) {
    return false;
  }
  uint32_t index = 0;
  const auto parsed =
      std::from_chars(name.data + prefix_length, name.data + name.size, index);
  if (parsed.ec != std::errc{} || parsed.ptr != name.data + name.size ||
      index >= index_limit) {
    return false;
  }
  *out_index = index;
  return true;
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

enum class Shape {
  kLinear,
  kLoop,
  kLoopRelocation,
  kLoopRelocationQuery,
  kMoveScratch,
  kBranch,
  kTied,
  kFanout,
  kFutureFixed,
  kReservedPrefix,
  kLeasedPrefix,
  kLeasedAliasTree,
};
enum class Phase {
  kModel,
  kLiveness,
  kPlacement,
  kUnitLiveness,
  kAllocation,
};

std::string MakeSource(uint32_t chain_length, uint32_t component_count,
                       uint32_t width, Shape shape) {
  const std::string type = "reg<test.i32 x" + std::to_string(width) + ">";
  if (shape == Shape::kMoveScratch) {
    std::string source =
        "test.target<low_core> @target\n"
        "low.func.def target<test.low.core>(@target) @kernel() -> "
        "(reg<test.i32 x2>, reg<test.i32>) asm {\n"
        "  %cycle_lhs = test.const.i32 23\n"
        "  %cycle_rhs = test.const.i32 17\n";
    for (uint32_t i = 0; i < component_count; ++i) {
      source += "  %blocker" + std::to_string(i) + " = test.const.i32 " +
                std::to_string(i) + "\n";
    }
    source +=
        "  %cycle_pair = concat(%cycle_lhs, %cycle_rhs) : "
        "(reg<test.i32>, reg<test.i32>) -> reg<test.i32 x2>\n"
        "  %sum1 = test.add.i32 %blocker0, %blocker1\n";
    for (uint32_t i = 2; i < component_count; ++i) {
      source += "  %sum" + std::to_string(i) + " = test.add.i32 %sum" +
                std::to_string(i - 1) + ", %blocker" + std::to_string(i) + "\n";
    }
    return source + "  return %cycle_pair, %sum" +
           std::to_string(component_count - 1) + "\n}\n";
  }
  if (shape == Shape::kFanout || shape == Shape::kReservedPrefix) {
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
  if (shape == Shape::kFutureFixed) {
    Require(width == 1, "Future-fixed shape requires scalar registers");
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
    source += ", reg<test.i32 x" + std::to_string(count) + ">";
    source += ") asm {\n";
    for (uint32_t i = 0; i < count; ++i) {
      source += "  %copy" + std::to_string(i) + " = copy %seed" +
                std::to_string(i / chain_length) + " : " + type + " -> " +
                type + "\n";
    }
    for (uint32_t i = 0; i < count; ++i) {
      source += "  %fixed" + std::to_string(i) + " = test.const.i32 " +
                std::to_string(i) + "\n";
    }
    for (uint32_t i = 0; i < count; ++i) {
      source += "  %next" + std::to_string(i) + " = test.tied.any %copy" +
                std::to_string(i) + "\n";
    }
    source += "  %fixed_pack = concat(";
    for (uint32_t i = 0; i < count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "%fixed" + std::to_string(i);
    }
    source += ") : (";
    for (uint32_t i = 0; i < count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += type;
    }
    source += ") -> reg<test.i32 x" + std::to_string(count) + ">\n";
    source += "  return ";
    for (uint32_t i = 0; i < count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "%next" + std::to_string(i);
    }
    source += ", %fixed_pack";
    return source + "\n}\n";
  }
  if (shape == Shape::kLeasedPrefix) {
    Require(width == 1, "Leased-prefix shape requires scalar registers");
    const uint32_t count = chain_length * component_count;
    std::string source =
        "test.target<low_core> @target\n"
        "low.func.def schedule(locked) target<test.low.core>(@target) "
        "@kernel() -> (reg<test.i32>) asm {\n";
    for (uint32_t i = 0; i < count; ++i) {
      source += "  %leased" + std::to_string(i) + " = test.const.issued.i32 " +
                std::to_string(i) + "\n";
      source += "  test.leased.consume.i32 %leased" + std::to_string(i) + "\n";
    }
    for (uint32_t i = 0; i < count; ++i) {
      source += "  %result" + std::to_string(i) + " = test.const.issued.i32 " +
                std::to_string(i) + "\n";
    }
    return source + "  return %result" + std::to_string(count - 1u) + "\n}\n";
  }
  if (shape == Shape::kLeasedAliasTree) {
    Require(width == 1, "Leased-alias shape requires scalar registers");
    const uint32_t count = chain_length * component_count;
    Require(count >= 2 && (count & (count - 1u)) == 0,
            "Leased-alias shape requires a power-of-two leaf count");
    std::string source =
        "test.target<low_core> @target\n"
        "low.func.def schedule(locked) target<test.low.core>(@target) "
        "@kernel() -> (reg<test.i32 x" +
        std::to_string(count) + ">) asm {\n";
    std::vector<std::string> values;
    values.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const std::string value = "%leased" + std::to_string(i);
      source +=
          "  " + value + " = test.const.issued.i32 " + std::to_string(i) + "\n";
      source += "  test.leased.consume.i32 " + value + "\n";
      values.push_back(value);
    }
    uint32_t level = 0;
    uint32_t result_width = 1;
    while (values.size() > 1) {
      std::vector<std::string> next_values;
      next_values.reserve(values.size() / 2);
      for (uint32_t i = 0; i < values.size(); i += 2) {
        const std::string result =
            "%alias" + std::to_string(level) + "_" + std::to_string(i / 2);
        source += "  " + result + " = concat(" + values[i] + ", " +
                  values[i + 1] + ") : (reg<test.i32 x" +
                  std::to_string(result_width) + ">, reg<test.i32 x" +
                  std::to_string(result_width) + ">) -> reg<test.i32 x" +
                  std::to_string(result_width * 2u) + ">\n";
        next_values.push_back(result);
      }
      values = std::move(next_values);
      result_width *= 2u;
      ++level;
    }
    return source + "  return " + values.front() + "\n}\n";
  }
  if (shape == Shape::kLoopRelocation || shape == Shape::kLoopRelocationQuery) {
    const bool is_query_scaling = shape == Shape::kLoopRelocationQuery;
    const uint32_t unrelated_count = is_query_scaling ? chain_length : 0;
    const bool is_sparse_relocation = is_query_scaling || chain_length == 1;
    std::string source =
        "test.target<low_core> @target\n"
        "low.func.def target<test.low.core>(@target) @kernel("
        "%condition: reg<test.i32>";
    for (uint32_t i = 0; i < component_count; ++i) {
      source += ", %seed" + std::to_string(i) + ": reg<test.i32>";
    }
    source += ", %lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (";
    const uint32_t result_count =
        component_count + (unrelated_count == 0 ? 0 : 1);
    for (uint32_t i = 0; i < result_count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "reg<test.i32>";
    }
    source += ") asm {\n";
    for (uint32_t i = 0; i < unrelated_count; ++i) {
      source += "  %unrelated" + std::to_string(i) + " = test.const.i32 " +
                std::to_string(i) + "\n";
    }
    source += "  low.br ^loop(";
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
    if (is_sparse_relocation) {
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
      if (is_sparse_relocation) {
        source += i % 2 == 0 ? " = test.mul.i32 %lhs, %rhs\n"
                             : " = test.add.i32 %lhs, %rhs\n";
      } else {
        source += " = test.add.i32 %state" + std::to_string(i) + ", %rhs\n";
      }
    }
    if (is_sparse_relocation) {
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
    source += ")\n^exit:\n";
    if (unrelated_count > 1) {
      source += "  %fold0 = test.add.i32 %unrelated0, %unrelated1\n";
      for (uint32_t i = 2; i < unrelated_count; ++i) {
        source += "  %fold" + std::to_string(i - 1) + " = test.add.i32 %fold" +
                  std::to_string(i - 2) + ", %unrelated" + std::to_string(i) +
                  "\n";
      }
    }
    source += "  return ";
    for (uint32_t i = 0; i < component_count; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "%state" + std::to_string(i);
    }
    if (unrelated_count != 0) {
      source += unrelated_count == 1
                    ? ", %unrelated0"
                    : ", %fold" + std::to_string(unrelated_count - 2);
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
  // Final moves used to sequence the forced cyclic group.
  uint64_t scratch_group_move_count = 0;
  // Cycle-scratch writes in the forced cyclic group.
  uint64_t scratch_move_count = 0;
  // First cycle-scratch location selected for the forced cyclic group.
  uint64_t scratch_location = UINT64_MAX;
  // Materialized asynchronous storage leases.
  uint64_t storage_lease_count = 0;
  // Allocator-requested early release actions.
  uint64_t storage_release_action_count = 0;
  // Largest assigned target-ID endpoint.
  uint64_t assigned_target_id_extent = 0;
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
    loom_verify_result_t structure = {};
    IREE_CHECK_OK(loom_verify_module(module_, nullptr, &structure));
    Require(structure.error_count == 0, "Generic verification failed");
    loom_low_verify_options_t low_verify_options = {.descriptor_registry =
                                                        &registry_.registry};
    auto scratch = loom_low_verify_scratch_for_module(module_);
    loom_low_verify_result_t low_verified = {};
    IREE_CHECK_OK(loom_low_verify_module(module_, &low_verify_options, &scratch,
                                         &low_verified));
    Require(low_verified.error_count == 0, "Low verification failed");
    auto name = loom_module_lookup_string(module_, IREE_SV("kernel"));
    auto symbol = loom_module_find_symbol(module_, name);
    Require(symbol != LOOM_SYMBOL_ID_INVALID, "Kernel symbol missing");
    function_ = module_->symbols.entries[symbol].defining_op;
    if (shape == Shape::kLoopRelocation ||
        shape == Shape::kLoopRelocationQuery) {
      const loom_region_t* body = loom_low_func_def_body(function_);
      Require(body->block_count == 4, "Relocation loop shape changed");
      backedge_terminator_ =
          loom_block_const_last_op(loom_region_const_block(body, 2));
      Require(loom_low_br_isa(backedge_terminator_),
              "Relocation loop backedge missing");
    }
    if (shape == Shape::kLoopRelocationQuery) {
      fixed_values_.reserve(chain_length * 2u - 1u);
      for (loom_value_id_t value_id = 0; value_id < module_->values.count;
           ++value_id) {
        const iree_string_view_t name =
            loom_module_value_name(module_, value_id);
        uint32_t index = 0;
        uint32_t location = 0;
        if (ParseIndexedName(name, "unrelated", chain_length, &index)) {
          location = 1024u + index;
        } else if (ParseIndexedName(name, "fold", chain_length - 1u, &index)) {
          location = 1024u + chain_length + index;
        } else {
          continue;
        }
        fixed_values_.push_back({
            value_id,
            LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
            location,
            1,
        });
      }
      Require(fixed_values_.size() == chain_length * 2u - 1u,
              "Unrelated relocation values missing");
    }
    if (shape == Shape::kMoveScratch) {
      has_scratch_cycle_ = true;
      expected_scratch_location_ = component_count + 2;
      fixed_values_.reserve(component_count + 3);
      for (loom_value_id_t value_id = 0; value_id < module_->values.count;
           ++value_id) {
        const iree_string_view_t name =
            loom_module_value_name(module_, value_id);
        uint32_t location = UINT32_MAX;
        auto name_equals = [&](const char* expected) {
          const iree_host_size_t length = std::strlen(expected);
          return name.size == length &&
                 std::memcmp(name.data, expected, length) == 0;
        };
        uint32_t location_count = 1;
        if (name_equals("cycle_lhs")) {
          location = 1;
        } else if (name_equals("cycle_rhs")) {
          location = 0;
        } else if (name_equals("cycle_pair")) {
          location = 0;
          location_count = 2;
        } else {
          const char* prefix = "blocker";
          const iree_host_size_t prefix_length = std::strlen(prefix);
          if (name.size > prefix_length &&
              std::memcmp(name.data, prefix, prefix_length) == 0) {
            uint32_t index = 0;
            const auto parsed = std::from_chars(name.data + prefix_length,
                                                name.data + name.size, index);
            if (parsed.ec == std::errc{} &&
                parsed.ptr == name.data + name.size &&
                index < component_count) {
              location = index + 2;
            }
          }
        }
        if (location == UINT32_MAX) {
          continue;
        }
        fixed_values_.push_back({
            value_id,
            LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
            location,
            location_count,
        });
      }
      Require(fixed_values_.size() == component_count + 3,
              "Scratch group fixed values missing");
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
    if (shape == Shape::kFutureFixed) {
      const uint32_t count = chain_length * component_count;
      fixed_values_.resize(count);
      std::vector<uint8_t> found(count);
      for (loom_value_id_t value_id = 0; value_id < module_->values.count;
           ++value_id) {
        const iree_string_view_t name =
            loom_module_value_name(module_, value_id);
        constexpr iree_host_size_t kPrefixLength = sizeof("fixed") - 1;
        if (name.size <= kPrefixLength ||
            std::memcmp(name.data, "fixed", kPrefixLength) != 0) {
          continue;
        }
        uint32_t fixed_index = 0;
        const auto parsed = std::from_chars(name.data + kPrefixLength,
                                            name.data + name.size, fixed_index);
        if (parsed.ec != std::errc{} || parsed.ptr != name.data + name.size ||
            fixed_index >= count) {
          continue;
        }
        fixed_values_[fixed_index] = {
            value_id,
            LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
            fixed_index,
            1,
        };
        found[fixed_index] = 1;
      }
      Require(std::all_of(found.begin(), found.end(),
                          [](uint8_t value) { return value != 0; }),
              "Future-fixed source values missing");
    }
    if (shape == Shape::kReservedPrefix) {
      reserved_ranges_.push_back({
          IREE_SV("test.i32"),
          LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
          /*location_base=*/0,
          /*location_count=*/chain_length * component_count,
      });
    }
    if (phase_ != Phase::kModel) {
      InitializeModel(&base_arena_, &model_);
    }
    if (shape == Shape::kLeasedPrefix || shape == Shape::kLeasedAliasTree) {
      loom_low_schedule_options_t schedule_options = {
          .flags = LOOM_LOW_SCHEDULE_FLAG_RETAIN_VALUE_PRODUCER_NODES};
      IREE_CHECK_OK(loom_low_schedule_function(&model_, &schedule_options,
                                               &base_arena_, &schedule_));
      Require(schedule_.error_count == 0, "Scheduling failed");
      const loom_low_storage_lease_provider_t provider = {
          .user_data = {},
          .query = loom_low_storage_lease_query_descriptor_rows,
      };
      IREE_CHECK_OK(loom_low_storage_lease_build(
          &schedule_, &provider, &base_arena_, &storage_leases_));
      Require(storage_leases_.record_count == chain_length * component_count,
              "Leased-prefix records missing");
    }
    if (phase_ == Phase::kPlacement || phase_ == Phase::kUnitLiveness) {
      IREE_CHECK_OK(loom_liveness_analyze_local_value_domain_with_dataflow(
          &model_.context.value_domain, &model_.liveness_dataflow,
          loom_liveness_order_empty(), &base_arena_, &liveness_));
    }
    if (phase_ == Phase::kUnitLiveness) {
      loom_low_placement_preference_index_t preferences = {};
      loom_low_allocation_target_constraints_t constraints = {};
      IREE_CHECK_OK(loom_low_allocation_target_constraints_initialize(
          module_, function_, &model_.context.target, nullptr, 0,
          reserved_ranges_.data(), reserved_ranges_.size(), {}, &base_arena_,
          &constraints));
      IREE_CHECK_OK(loom_low_allocation_placement_build(
          &constraints, model_.context.body, &model_.context.value_domain,
          model_.context.storage_origins, &liveness_, fixed_values_.data(),
          fixed_values_.size(), loom_low_placement_pair_use_list_empty(), {},
          &base_arena_, &base_arena_, &placement_, &preferences));
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
      result.value_count = model.context.value_domain.value_count;
      loom_low_function_model_deinitialize(&model);
    } else if (phase_ == Phase::kLiveness) {
      loom_liveness_analysis_t liveness = {};
      IREE_CHECK_OK(loom_liveness_analyze_local_value_domain_with_dataflow(
          &model_.context.value_domain, &model_.liveness_dataflow,
          loom_liveness_order_empty(), &arena, &liveness));
      result.value_count = liveness.value_count;
      benchmark::DoNotOptimize(liveness.intervals);
    } else if (phase_ == Phase::kPlacement) {
      loom_low_placement_table_t placement = {};
      loom_low_placement_preference_index_t preferences = {};
      loom_low_allocation_target_constraints_t constraints = {};
      IREE_CHECK_OK(loom_low_allocation_target_constraints_initialize(
          module_, function_, &model_.context.target, nullptr, 0,
          reserved_ranges_.data(), reserved_ranges_.size(), {}, &arena,
          &constraints));
      IREE_CHECK_OK(loom_low_allocation_placement_build(
          &constraints, model_.context.body, &model_.context.value_domain,
          model_.context.storage_origins, &liveness_, fixed_values_.data(),
          fixed_values_.size(), loom_low_placement_pair_use_list_empty(), {},
          &arena, &arena, &placement, &preferences));
      result.value_count = placement.value_count;
      benchmark::DoNotOptimize(placement.relations);
    } else if (phase_ == Phase::kUnitLiveness) {
      iree_arena_allocator_t decision_arena;
      iree_arena_initialize(&analysis_pool_, &decision_arena);
      loom_low_allocation_unit_liveness_t unit_liveness = {};
      IREE_CHECK_OK(loom_low_allocation_unit_liveness_initialize(
          &model_.context.target, &placement_, &model_.context.value_domain,
          &liveness_, &model_.cfg_graph, {}, &arena, &decision_arena,
          &unit_liveness));
      result.value_count = liveness_.value_count;
      benchmark::DoNotOptimize(unit_liveness.end_points);
      iree_arena_deinitialize(&decision_arena);
    } else {
      loom_low_allocation_options_t options = {};
      options.fixed_values = fixed_values_.data();
      options.fixed_value_count = fixed_values_.size();
      options.reserved_ranges = reserved_ranges_.data();
      options.reserved_range_count = reserved_ranges_.size();
      if (storage_leases_.record_count != 0) {
        options.schedule = &schedule_;
        options.storage_leases = storage_leases_;
      }
      loom_low_allocation_table_t allocation = {};
      IREE_CHECK_OK(
          loom_low_allocate_function(&model_, &options, &arena, &allocation));
      Require(allocation.error_count == 0, "Allocation failed");
      Require(allocation.spill_count == 0, "Unexpected spill");
      result.value_count = model_.context.value_domain.value_count;
      result.copy_count = allocation.copy_decision_count;
      result.materialized_copy_count = allocation.materialized_copy_count;
      result.storage_lease_count = allocation.storage_lease_instance_count;
      result.storage_release_action_count =
          allocation.storage_release_action_count;
      for (iree_host_size_t i = 0; i < allocation.assignment_count; ++i) {
        const auto& assignment = allocation.assignments[i];
        if (assignment.location_kind ==
            LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID) {
          result.assigned_target_id_extent =
              std::max(result.assigned_target_id_extent,
                       static_cast<uint64_t>(assignment.location_base) +
                           assignment.location_count);
        }
      }
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
      if (has_scratch_cycle_) {
        const loom_low_move_group_t* scratch_group = nullptr;
        for (iree_host_size_t i = 0; i < allocation.packet_move_group_count;
             ++i) {
          const auto& group = allocation.packet_move_groups[i].move_group;
          if (group.scratch_move_index_count != 0) {
            Require(scratch_group == nullptr,
                    "Multiple scratch groups in scaling witness");
            scratch_group = &group;
          }
        }
        Require(scratch_group != nullptr, "Scratch move group missing");
        result.scratch_group_move_count = scratch_group->moves.count;
        result.scratch_move_count = scratch_group->scratch_move_index_count;
        Require(scratch_group->scratch_move_index_count == 1,
                "Scratch group cycle missing");
        const iree_host_size_t scratch_move_index =
            allocation
                .scratch_move_indices[scratch_group->scratch_move_index_start];
        Require(scratch_move_index >= scratch_group->moves.start &&
                    scratch_move_index <
                        scratch_group->moves.start + scratch_group->moves.count,
                "Scratch move index outside move group");
        result.scratch_location =
            allocation.moves[scratch_move_index].destination.location;
        Require(result.scratch_location == expected_scratch_location_,
                "Scratch group did not choose the first free location");

        uint32_t lhs = 23;
        uint32_t rhs = 17;
        uint32_t scratch = 0;
        auto value_at = [&](uint32_t location) -> uint32_t& {
          if (location == 0) {
            return lhs;
          }
          if (location == 1) {
            return rhs;
          }
          Require(location == result.scratch_location,
                  "Scratch group touched an unexpected location");
          return scratch;
        };
        const loom_low_move_t* moves =
            allocation.moves + scratch_group->moves.start;
        for (iree_host_size_t i = 0; i < scratch_group->moves.count; ++i) {
          value_at(moves[i].destination.location) =
              value_at(moves[i].source.location);
        }
        Require(lhs == 17 && rhs == 23,
                "Scratch group sequence did not preserve the swap");
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
    Require(model->context.error_count == 0, "Function model failed");
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
  // Retained semantic liveness for placement and unit-liveness measurements.
  loom_liveness_analysis_t liveness_ = {};
  // Retained placement facts for unit-liveness-only measurements.
  loom_low_placement_table_t placement_ = {};
  // Retained locked schedule used by the descriptor-driven lease witness.
  loom_low_schedule_table_t schedule_ = {};
  // Descriptor-provided storage leases over |schedule_|.
  loom_low_storage_lease_table_t storage_leases_ = {};
  // Dense relocation source colors fixed outside the header destination set.
  std::vector<loom_low_allocation_fixed_value_t> fixed_values_;
  // Whole-function reservation used by the prefix-scaling witness.
  std::vector<loom_low_allocation_reserved_range_t> reserved_ranges_;
  // Generated loop backedge used to validate final edge-copy materialization.
  const loom_op_t* backedge_terminator_ = nullptr;
  // Whether the generated packet-local cycle must use scratch storage.
  bool has_scratch_cycle_ = false;
  // Lowest location not occupied by the cyclic edge's live source values.
  uint32_t expected_scratch_location_ = UINT32_MAX;
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
        shape == Shape::kTied || shape == Shape::kLoopRelocation ||
                shape == Shape::kLoopRelocationQuery ||
                shape == Shape::kMoveScratch || shape == Shape::kLeasedPrefix ||
                shape == Shape::kLeasedAliasTree
            ? 0
            : chain_length * component_count *
                  (shape == Shape::kBranch ? 2 : 1);
    Require(result.copy_count == expected_copy_count, "Copy decisions missing");
    if (shape == Shape::kLoopRelocation ||
        shape == Shape::kLoopRelocationQuery) {
      Require(result.backedge_move_count == 0,
              "Loop-edge relocation left branch copies");
    }
    if (shape == Shape::kFanout || shape == Shape::kFutureFixed ||
        shape == Shape::kReservedPrefix) {
      Require(result.materialized_copy_count >=
                  expected_copy_count - component_count,
              "Independently live copies were incorrectly coalesced");
    }
    if (shape == Shape::kLeasedPrefix) {
      const uint64_t count = chain_length * component_count;
      Require(result.storage_lease_count == count,
              "Storage leases were not materialized");
      Require(result.storage_release_action_count == 0,
              "Leased prefix requested an unexpected early release");
      Require(result.assigned_target_id_extent == count + 1u,
              "Leased prefix did not retain its physical locations");
    }
    if (shape == Shape::kLeasedAliasTree) {
      const uint64_t count = chain_length * component_count;
      Require(result.storage_lease_count == count,
              "Leased-alias storage leases were not materialized");
      Require(result.storage_release_action_count == 0,
              "Identity aliases requested artificial early releases");
      Require(result.assigned_target_id_extent == count,
              "Leased-alias tree did not retain its leaf storage");
    }
  }
  state.counters["value_count"] = result.value_count;
  state.counters["copy_count"] = result.copy_count;
  state.counters["materialized_copy_count"] = result.materialized_copy_count;
  state.counters["backedge_move_count"] = result.backedge_move_count;
  state.counters["scratch_group_move_count"] = result.scratch_group_move_count;
  state.counters["scratch_move_count"] = result.scratch_move_count;
  state.counters["scratch_location"] = result.scratch_location;
  state.counters["storage_lease_count"] = result.storage_lease_count;
  state.counters["storage_release_action_count"] =
      result.storage_release_action_count;
  state.counters["assigned_target_id_extent"] =
      result.assigned_target_id_extent;
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
                     Shape::kLoopRelocationQuery, Shape::kMoveScratch,
                     Shape::kBranch, Shape::kTied, Shape::kFanout,
                     Shape::kFutureFixed, Shape::kReservedPrefix,
                     Shape::kLeasedPrefix, Shape::kLeasedAliasTree}) {
    for (auto phase : {Phase::kModel, Phase::kLiveness, Phase::kPlacement,
                       Phase::kUnitLiveness, Phase::kAllocation}) {
      if (shape == Shape::kLoopRelocation && phase != Phase::kAllocation) {
        continue;
      }
      if (shape == Shape::kLoopRelocationQuery && phase != Phase::kAllocation) {
        continue;
      }
      if (shape == Shape::kMoveScratch && phase != Phase::kAllocation) {
        continue;
      }
      if (shape == Shape::kFutureFixed && phase != Phase::kAllocation) {
        continue;
      }
      if (shape == Shape::kReservedPrefix && phase != Phase::kAllocation) {
        continue;
      }
      if ((shape == Shape::kLeasedPrefix || shape == Shape::kLeasedAliasTree) &&
          phase != Phase::kAllocation) {
        continue;
      }
      const std::string name =
          "LowAllocation/" +
          std::string(shape == Shape::kLinear           ? "linear/"
                      : shape == Shape::kLoop           ? "loop/"
                      : shape == Shape::kLoopRelocation ? "loop_relocation/"
                      : shape == Shape::kLoopRelocationQuery
                          ? "loop_relocation_query/"
                      : shape == Shape::kMoveScratch    ? "move_scratch/"
                      : shape == Shape::kBranch         ? "branch/"
                      : shape == Shape::kTied           ? "tied/"
                      : shape == Shape::kFanout         ? "fanout/"
                      : shape == Shape::kFutureFixed    ? "future_fixed/"
                      : shape == Shape::kReservedPrefix ? "reserved_prefix/"
                      : shape == Shape::kLeasedPrefix ? "leased_prefix/"
                                                      : "leased_alias_tree/") +
          (phase == Phase::kModel          ? "model"
           : phase == Phase::kLiveness     ? "liveness"
           : phase == Phase::kPlacement    ? "placement"
           : phase == Phase::kUnitLiveness ? "unit_liveness"
                                           : "allocation");
      auto* registration = benchmark::RegisterBenchmark(
          name.c_str(),
          [=](benchmark::State& state) { RunBenchmark(state, shape, phase); });
      if (shape == Shape::kLoopRelocationQuery) {
        registration->ArgNames({"unrelated", "candidates", "width"});
      } else {
        registration->ArgNames({"length", "components", "width"});
      }
      if (shape == Shape::kLoopRelocation) {
        for (int64_t components : {32, 64, 128, 256, 512, 1024}) {
          registration->Args({1, components, 1});
        }
        for (int64_t components : {8, 16, 32, 64, 128, 256}) {
          registration->Args({components, components, 1});
        }
        continue;
      }
      if (shape == Shape::kLoopRelocationQuery) {
        for (int64_t unrelated : {32, 64, 128, 256, 512, 1024, 2048, 4096}) {
          registration->Args({unrelated, 4, 1});
        }
        for (int64_t candidates : {1, 2, 8, 16, 32, 64}) {
          registration->Args({4096, candidates, 1});
        }
        continue;
      }
      if (shape == Shape::kMoveScratch) {
        for (int64_t blockers : {32, 64, 128, 256, 512, 1024, 2048}) {
          registration->Args({1, blockers, 1});
        }
        continue;
      }
      if (shape == Shape::kFutureFixed) {
        for (int64_t count : {32, 64, 128, 256, 512, 1024, 2048}) {
          registration->Args({count, 1, 1});
        }
        continue;
      }
      if (shape == Shape::kReservedPrefix) {
        for (int64_t count : {32, 64, 128, 256, 512, 1024, 2048}) {
          registration->Args({count, 1, 1});
        }
        continue;
      }
      if (shape == Shape::kLeasedPrefix || shape == Shape::kLeasedAliasTree) {
        for (int64_t count : {32, 64, 128, 256, 512, 1024, 2048}) {
          registration->Args({count, 1, 1});
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
