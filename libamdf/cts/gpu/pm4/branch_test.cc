// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"

namespace {

struct ComparisonCase {
  // Stable comparison name in failure traces and result properties.
  const char* name;
  // Native function applied to the masked memory operand.
  Pm4MemoryComparison comparison;
};

constexpr std::array<ComparisonCase, 7> kComparisons = {{
    {"Always", Pm4MemoryComparison::kAlways},
    {"Less", Pm4MemoryComparison::kLess},
    {"LessOrEqual", Pm4MemoryComparison::kLessOrEqual},
    {"Equal", Pm4MemoryComparison::kEqual},
    {"NotEqual", Pm4MemoryComparison::kNotEqual},
    {"GreaterOrEqual", Pm4MemoryComparison::kGreaterOrEqual},
    {"Greater", Pm4MemoryComparison::kGreater},
}};

struct OperandCase {
  // Stable operand partition name in failure traces and result properties.
  const char* name;
  // Significant bits of the memory operand; ignored bits change each epoch.
  uint64_t mask;
  // Pre-masked unsigned reference: reference & ~mask is zero.
  uint64_t reference;
  // Spacing of significant values; zero for the constant zero-mask case.
  uint64_t step;
};

// Pre-masked references give the same decisions on engines that mask memory
// alone and engines that mask both operands. Ignored memory bits still vary.
constexpr std::array<OperandCase, 6> kOperands = {{
    {"FullCarry", UINT64_MAX, UINT64_C(0x0000000100000000), 1},
    {"FullSign", UINT64_MAX, UINT64_C(0x8000000000000000), 1},
    {"LowWord", UINT64_C(0x00000000ffffffff), UINT64_C(0x0000000080000000), 1},
    {"HighWord", UINT64_C(0xffffffff00000000), UINT64_C(0x8000000000000000),
     UINT64_C(0x0000000100000000)},
    {"SparseHalves", UINT64_C(0x00ffff0000ffff00), UINT64_C(0x0040000000400000),
     UINT64_C(0x0000010000000100)},
    {"ZeroMask", 0, 0, 0},
}};

bool SelectsPass(Pm4MemoryComparison comparison, uint64_t value,
                 uint64_t reference) {
  return comparison == Pm4MemoryComparison::kAlways ||
         (comparison == Pm4MemoryComparison::kLess && value < reference) ||
         (comparison == Pm4MemoryComparison::kLessOrEqual &&
          value <= reference) ||
         (comparison == Pm4MemoryComparison::kEqual && value == reference) ||
         (comparison == Pm4MemoryComparison::kNotEqual && value != reference) ||
         (comparison == Pm4MemoryComparison::kGreaterOrEqual &&
          value >= reference) ||
         (comparison == Pm4MemoryComparison::kGreater && value > reference);
}

void FillGuardWords(std::span<uint32_t> words, uint32_t seed) {
  for (size_t word = 0; word < words.size(); ++word) {
    words[word] = seed ^ (static_cast<uint32_t>(word) * 0x03050709u);
  }
}

void ExpectWords(std::span<const uint32_t> observed,
                 std::span<const uint32_t> expected, const char* owner) {
  SCOPED_TRACE(owner);
  ASSERT_EQ(observed.size(), expected.size());
  for (size_t word = 0; word < expected.size(); ++word) {
    EXPECT_EQ(observed[word], expected[word]) << "word=" << word;
  }
}

// DWORD offsets of the compared QWORD. Offsets 16 and 17 distinguish
// eight-byte alignment from PAL's four-byte exception.
constexpr std::array<uint32_t, 2> kPredicateWords = {16, 17};

std::string BranchCaseName(const ComparisonCase& comparison,
                           const OperandCase& operand,
                           uint32_t predicate_word) {
  return std::string(comparison.name) + operand.name +
         (predicate_word == 16 ? "Aligned8" : "Aligned4");
}

class Pm4BranchTest : public Pm4DispatchTest {};

TEST_F(Pm4BranchTest, ShaderOperandMatrixSelectsRetainedGraphs) {
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);
  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kPageWordCount = kPageByteLength / sizeof(uint32_t);
  constexpr uint32_t kGridSize = 64;
  constexpr uint32_t kPayloadWord = 64;
  constexpr uint32_t kArgumentStride = 16;
  constexpr uint32_t kEpochArgumentWord = 48;
  constexpr uint32_t kEpochCount = 5;
  constexpr uint32_t kBlockStride = 256;
  constexpr uint32_t kPublicationCompletionWord = 1;
  constexpr uint32_t kCommandWordsPerPublication = 8;
  constexpr size_t kCaseCount =
      kComparisons.size() * kOperands.size() * kPredicateWords.size();
  constexpr size_t kExecutionCount = kCaseCount * kEpochCount;
  constexpr size_t kPublicationCount = kCaseCount;
  constexpr size_t kExecutionsPerPublication = kEpochCount;
  constexpr size_t kRequiredCommandWords =
      kPublicationCount * kCommandWordsPerPublication;
  constexpr uint64_t kBatchByteLength =
      uint64_t{kExecutionCount} * kPageByteLength;
  constexpr uint64_t kCommandByteLength = 4096;
  constexpr std::array<uint32_t, 2> kOutputWords = {64, 128};
  constexpr std::array<uint32_t, 2> kSelectedWords = {32, 48};
  static_assert(4 * kBlockStride == kPageWordCount);
  static_assert(kExecutionCount ==
                kPublicationCount * kExecutionsPerPublication);
  static_assert(kRequiredCommandWords * sizeof(uint32_t) < kCommandByteLength);

  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* control = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* graph = nullptr;
  GpuMemory* code = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kBatchByteLength, &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kBatchByteLength, &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kBatchByteLength, &control));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kBatchByteLength, &arguments));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                   kBatchByteLength, &graph));
  for (const auto* memory : {input, output, control, arguments, graph}) {
    ASSERT_EQ(memory->info.byte_length, kBatchByteLength);
  }
  ASSERT_EQ(kernel.arguments.byte_length, 24u);
  ASSERT_EQ(kernel.workgroup_size(), kGridSize);
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  ASSERT_EQ(kPageByteLength % kernel.arguments.alignment, 0u);
  ASSERT_EQ(kArgumentStride * sizeof(uint32_t) % kernel.arguments.alignment,
            0u);
  ASSERT_EQ(graph->device_address % kPageByteLength, 0u);
  ASSERT_LE(graph->device_address,
            (UINT64_C(1) << 48) - graph->info.byte_length);
  Pm4ComputeProgram program = {
      0,
      kernel.program.resource1,
      kernel.program.resource2,
      kernel.program.resource3,
      kernel.group_segment_byte_length,
      kernel.wavefront_size,
      {kernel.workgroup_size(), 1, 1},
  };
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(kernel.executable,
                                         kernel.entry_byte_offset, &program,
                                         "pm4_branch", &code));

  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(kCommandByteLength, &queue));
  const bool primary_ring =
      queue->publication_mode() == AMDF_QUEUE_PUBLICATION_MODE_USER;
  const size_t command_capacity = primary_ring ? queue->words().size() : 0;
  if (primary_ring) {
    ASSERT_GT(command_capacity, kRequiredCommandWords);
  }
  const size_t batch_word_count = kBatchByteLength / sizeof(uint32_t);
  std::vector<uint32_t> expected_input(batch_word_count);
  std::vector<uint32_t> expected_output(batch_word_count);
  std::vector<uint32_t> expected_control(batch_word_count);
  std::vector<uint32_t> expected_arguments(batch_word_count);
  std::vector<uint32_t> expected_graph(batch_word_count, 0);
  std::vector<uint32_t> expected_code(code->info.byte_length / sizeof(uint32_t),
                                      0);
  std::vector<uint32_t> expected_commands(command_capacity, 0);
  std::memcpy(expected_code.data(), kernel.executable.words,
              kernel.executable.byte_length);

  struct ExecutionPlan {
    // Comparison encoded by this execution's terminal branch.
    const ComparisonCase* comparison;
    // Operand partition sampled by this execution's terminal branch.
    const OperandCase* operand;
    // DWORD offset of this execution's QWORD predicate.
    uint32_t predicate_word;
    // Epoch within the five-transition semantic case.
    uint32_t epoch;
    // Page-local kernel argument storage address.
    uint64_t argument_address;
    // Page-local predicate, marker, and completion storage address.
    uint64_t control_address;
    // Page-local four-block immutable command graph address.
    uint64_t graph_address;
  };

  size_t execution_index = 0;
  std::array<ExecutionPlan, kExecutionCount> execution_plans = {};
  std::array<Pm4IndirectBuffer, kExecutionCount> roots = {};
  std::array<size_t, kPublicationCount> publication_frontiers = {};
  std::array<uintptr_t, kPublicationCount> publication_completion_addresses =
      {};
  std::array<uint32_t, kPublicationCount> publication_completion_values = {};
  for (size_t comparison_index = 0; comparison_index < kComparisons.size();
       ++comparison_index) {
    const auto& comparison = kComparisons[comparison_index];
    for (const auto& operand : kOperands) {
      for (const uint32_t predicate_word : kPredicateWords) {
        const std::string case_name =
            BranchCaseName(comparison, operand, predicate_word);
        const std::string property_prefix = "pm4_branch_" + case_name;
        const size_t publication_index = execution_index / kEpochCount;
        ASSERT_EQ(execution_index, publication_index * kEpochCount);
        std::array<uint32_t, kPageWordCount> control_state;
        FillGuardWords(control_state, 0x68d329b7u);
        control_state[0] = 0;
        for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
          SCOPED_TRACE(case_name);
          SCOPED_TRACE("epoch=" + std::to_string(epoch));
          const size_t page_word_offset = execution_index * kPageWordCount;
          const uint64_t page_byte_offset =
              uint64_t{execution_index} * kPageByteLength;
          std::span<uint32_t> input_page(
              expected_input.data() + page_word_offset, kPageWordCount);
          std::span<uint32_t> output_page(
              expected_output.data() + page_word_offset, kPageWordCount);
          std::span<uint32_t> control_page(
              expected_control.data() + page_word_offset, kPageWordCount);
          std::span<uint32_t> argument_page(
              expected_arguments.data() + page_word_offset, kPageWordCount);
          const uint64_t input_address =
              input->device_address + page_byte_offset;
          const uint64_t output_address =
              output->device_address + page_byte_offset;
          const uint64_t control_address =
              control->device_address + page_byte_offset;
          const uint64_t argument_address =
              arguments->device_address + page_byte_offset;
          const uint64_t graph_address =
              graph->device_address + page_byte_offset;
          const uint64_t operand_address =
              control_address + predicate_word * sizeof(uint32_t);
          const uint32_t relation = epoch % 3;
          const uint64_t masked_value =
              (relation == 0   ? operand.reference - operand.step
               : relation == 1 ? operand.reference
                               : operand.reference + operand.step) &
              operand.mask;
          const uint64_t ignored =
              (UINT64_C(0x9d5713eca68420bf) ^
               (uint64_t{epoch} * UINT64_C(0x01030507090b0d0f))) &
              ~operand.mask;
          const uint64_t value = masked_value | ignored;
          const uint32_t selected_arm =
              SelectsPass(comparison.comparison, value & operand.mask,
                          operand.reference)
                  ? 0
                  : 1;

          FillGuardWords(input_page, 0x759bf13du ^ epoch);
          input_page[16] = static_cast<uint32_t>(
              uint64_t{static_cast<uint32_t>(value)} * 0xaaaaaaab);
          input_page[17] = static_cast<uint32_t>((value >> 32) * 0xaaaaaaab);
          for (uint32_t word = 0; word < kGridSize; ++word) {
            input_page[kPayloadWord + word] =
                0xfffffff0u + word * 0x01030507u + epoch;
          }
          std::memcpy(
              static_cast<uint8_t*>(input->host.pointer) + page_byte_offset,
              input_page.data(), kPageByteLength);

          std::array<uint32_t, kPageWordCount> initial_output;
          FillGuardWords(initial_output, 0x4e90b725u ^ epoch);
          std::copy(initial_output.begin(), initial_output.end(),
                    output_page.begin());
          std::fill(argument_page.begin(), argument_page.end(), 0u);
          const kernels::transform::Arguments producer = {
              input_address + 16 * sizeof(uint32_t),
              operand_address,
              2,
              0,
          };
          std::memcpy(argument_page.data(), &producer,
                      kernel.arguments.byte_length);
          for (uint32_t arm = 0; arm < kOutputWords.size(); ++arm) {
            const uint32_t count = 61 - arm - epoch % 3;
            const uint32_t addend = 7 + arm * 0x80000000u + epoch * 0x01010101u;
            for (uint32_t word = 0; word < kGridSize; ++word) {
              const uint32_t result = static_cast<uint32_t>(
                  uint64_t{input_page[kPayloadWord + word]} * 3 + addend);
              initial_output[kOutputWords[arm] + word] = ~result;
              output_page[kOutputWords[arm] + word] =
                  arm == selected_arm && word < count ? result : ~result;
            }
            const kernels::transform::Arguments consumer = {
                input_address + kPayloadWord * sizeof(uint32_t),
                output_address + kOutputWords[arm] * sizeof(uint32_t),
                count,
                addend,
            };
            std::memcpy(argument_page.data() + (arm + 1) * kArgumentStride,
                        &consumer, kernel.arguments.byte_length);
          }
          argument_page[kEpochArgumentWord] = epoch + 1;
          std::memcpy(
              static_cast<uint8_t*>(output->host.pointer) + page_byte_offset,
              initial_output.data(), kPageByteLength);
          std::memcpy(
              static_cast<uint8_t*>(arguments->host.pointer) + page_byte_offset,
              argument_page.data(), kPageByteLength);

          // Preseed this execution with the exact state produced by the prior
          // epoch, then preserve the state expected after this transition.
          std::memcpy(
              static_cast<uint8_t*>(control->host.pointer) + page_byte_offset,
              control_state.data(), kPageByteLength);
          control_state[0] = epoch + 1;
          control_state[predicate_word] = static_cast<uint32_t>(value);
          control_state[predicate_word + 1] =
              static_cast<uint32_t>(value >> 32);
          control_state[kSelectedWords[selected_arm]] = epoch + 1;
          std::copy(control_state.begin(), control_state.end(),
                    control_page.begin());

          execution_plans[execution_index] = {
              &comparison,      &operand,        predicate_word, epoch,
              argument_address, control_address, graph_address};

          if (epoch == 0) {
            RecordProperty(property_prefix + "_comparison", comparison.name);
            RecordProperty(property_prefix + "_mask",
                           std::to_string(operand.mask));
            RecordProperty(property_prefix + "_reference",
                           std::to_string(operand.reference));
            RecordProperty(property_prefix + "_operand_byte_offset",
                           predicate_word * sizeof(uint32_t));
          }
          const std::string epoch_prefix =
              property_prefix + "_epoch_" + std::to_string(epoch);
          RecordProperty(epoch_prefix + "_operand", std::to_string(value));
          RecordProperty(epoch_prefix + "_selected_arm", selected_arm);
          RecordProperty(epoch_prefix + "_completion", epoch + 1);
          publication_completion_addresses[publication_index] =
              reinterpret_cast<uintptr_t>(
                  static_cast<uint8_t*>(control->host.pointer) +
                  page_byte_offset) +
              kPublicationCompletionWord * sizeof(uint32_t);
          publication_completion_values[publication_index] =
              static_cast<uint32_t>(publication_index + 1);
          ++execution_index;
        }
        expected_control[(execution_index - 1) * kPageWordCount +
                         kPublicationCompletionWord] =
            publication_completion_values[publication_index];
        RecordProperty(property_prefix + "_completed_epochs", kEpochCount);
      }
    }
  }
  ASSERT_EQ(execution_index, kExecutionCount);

  // Build backward so each nonterminal join can continue at the same IB level
  // through the already-complete next epoch. The final join publishes every
  // payload in the semantic case before its completion marker becomes visible.
  for (size_t reverse_index = kExecutionCount; reverse_index > 0;
       --reverse_index) {
    const size_t plan_index = reverse_index - 1;
    const auto& plan = execution_plans[plan_index];
    const size_t page_word_offset = plan_index * kPageWordCount;
    std::span<uint32_t> graph_page(expected_graph.data() + page_word_offset,
                                   kPageWordCount);
    const uint64_t epoch_address =
        plan.argument_address + kEpochArgumentWord * sizeof(uint32_t);
    const uint64_t operand_address =
        plan.control_address + plan.predicate_word * sizeof(uint32_t);

    Pm4CommandWriter continuation(graph_page.data() + 3 * kBlockStride,
                                  *pm4_profile_);
    continuation.SystemBarrier();
    continuation.CopyData32(epoch_address, plan.control_address);
    const bool has_next_epoch = plan.epoch + 1 < kEpochCount;
    if (has_next_epoch) {
      continuation.PadToEightWords(4);
      continuation.ChainIndirectBuffer(roots[plan_index + 1]);
    } else {
      const size_t publication_index = plan_index / kEpochCount;
      continuation.ReleaseSystem32(
          plan.control_address + kPublicationCompletionWord * sizeof(uint32_t),
          publication_completion_values[publication_index]);
      continuation.PadToEightWords();
    }
    ASSERT_EQ(continuation.word_count() % 8, 0u);
    ASSERT_LT(continuation.word_count(), kBlockStride);
    const Pm4IndirectBuffer join = {
        plan.graph_address + 3 * kBlockStride * sizeof(uint32_t),
        static_cast<uint32_t>(continuation.word_count()),
    };
    std::array<Pm4IndirectBuffer, 2> arms = {};
    for (uint32_t arm = 0; arm < arms.size(); ++arm) {
      Pm4CommandWriter commands(graph_page.data() + (arm + 1) * kBlockStride,
                                *pm4_profile_);
      commands.BindCompute(program,
                           plan.argument_address +
                               (arm + 1) * kArgumentStride * sizeof(uint32_t),
                           family_.format_features);
      commands.Dispatch(program, kGridSize, 1, 1);
      commands.CopyData32(
          epoch_address,
          plan.control_address + kSelectedWords[arm] * sizeof(uint32_t));
      commands.PadToEightWords(4);
      commands.ChainIndirectBuffer(join);
      ASSERT_EQ(commands.word_count() % 8, 0u);
      ASSERT_LT(commands.word_count(), kBlockStride);
      arms[arm] = {
          plan.graph_address + (arm + 1) * kBlockStride * sizeof(uint32_t),
          static_cast<uint32_t>(commands.word_count())};
    }
    Pm4CommandWriter root(graph_page.data(), *pm4_profile_);
    root.SystemBarrier();
    root.BindCompute(program, plan.argument_address, family_.format_features);
    root.Dispatch(program, kGridSize, 1, 1);
    root.SystemBarrier();
    root.PadToEightWords(14);
    root.BranchIndirectBuffer(operand_address, plan.operand->reference,
                              plan.operand->mask, plan.comparison->comparison,
                              arms[0], arms[1]);
    ASSERT_EQ(root.word_count() % 8, 0u);
    ASSERT_LT(root.word_count(), kBlockStride);
    roots[plan_index] = {plan.graph_address,
                         static_cast<uint32_t>(root.word_count())};

    if (plan.epoch == 0) {
      const std::string property_prefix =
          "pm4_branch_" +
          BranchCaseName(*plan.comparison, *plan.operand, plan.predicate_word);
      RecordProperty(property_prefix + "_root_word_count",
                     std::to_string(root.word_count()));
      RecordProperty(property_prefix + "_pass_word_count", arms[0].word_count);
      RecordProperty(property_prefix + "_fail_word_count", arms[1].word_count);
      RecordProperty(property_prefix + "_join_word_count", join.word_count);
    }
  }

  size_t command_word_count = 0;
  if (primary_ring) {
    Pm4CommandWriter stream(expected_commands.data(), *pm4_profile_);
    for (size_t publication_index = 0; publication_index < kPublicationCount;
         ++publication_index) {
      const auto& root = roots[publication_index * kExecutionsPerPublication];
      stream.CallIndirectBuffer(root.address, root.word_count);
      stream.PadToEightWords();
      publication_frontiers[publication_index] = stream.word_count();
      ASSERT_EQ(stream.word_count(),
                (publication_index + 1) * kCommandWordsPerPublication);
    }
    command_word_count = stream.word_count();
    ASSERT_EQ(command_word_count, kRequiredCommandWords);
    std::memcpy(queue->words().data(), expected_commands.data(),
                queue->words().size_bytes());
  }
  std::memcpy(graph->host.pointer, expected_graph.data(),
              graph->info.byte_length);

  RecordProperty("pm4_branch_entry",
                 primary_ring ? "primary_ring_case_call" : "kernel_graph_root");
  RecordProperty("pm4_branch_case_count", kCaseCount);
  RecordProperty("pm4_branch_epochs_per_case", kEpochCount);
  RecordProperty("pm4_branch_total_execution_count", kExecutionCount);
  RecordProperty("pm4_branch_publication_count", kPublicationCount);
  RecordProperty("pm4_branch_publication_completion_byte_offset",
                 kPublicationCompletionWord * sizeof(uint32_t));
  RecordProperty("pm4_branch_command_capacity_dwords",
                 std::to_string(command_capacity));
  RecordProperty("pm4_branch_command_word_count",
                 std::to_string(command_word_count));
  RecordProperty(
      "pm4_branch_observed_byte_length",
      std::to_string(input->info.byte_length + output->info.byte_length +
                     control->info.byte_length + arguments->info.byte_length +
                     graph->info.byte_length + code->info.byte_length +
                     command_capacity * sizeof(uint32_t)));

  std::vector<uint32_t> observed_output(batch_word_count);
  std::vector<uint32_t> observed_input(batch_word_count);
  std::vector<uint32_t> observed_control(batch_word_count);
  std::vector<uint32_t> observed_arguments(batch_word_count);
  std::vector<uint32_t> observed_graph(batch_word_count);
  std::vector<uint32_t> observed_code(expected_code.size());
  std::vector<uint32_t> observed_commands(command_capacity);
  for (size_t publication_index = 0; publication_index < kPublicationCount;
       ++publication_index) {
    SCOPED_TRACE("publication=" + std::to_string(publication_index));
    const size_t first_execution =
        publication_index * kExecutionsPerPublication;
    const auto& first_plan = execution_plans[first_execution];
    SCOPED_TRACE(BranchCaseName(*first_plan.comparison, *first_plan.operand,
                                first_plan.predicate_word));
    if (primary_ring) {
      ASSERT_NO_FATAL_FAILURE(queue->Publish(
          api_, gpu_api_, publication_frontiers[publication_index]));
    } else {
      // KERNEL consumes the graph root itself. A ring-style call inside that
      // submission would incorrectly request compute IB2 nesting.
      const auto& root = roots[first_execution];
      const amdf_gpu_kernel_command_t command = {
          .memory = graph->memory,
          .byte_offset = uint64_t{first_execution} * kPageByteLength,
          .byte_length = root.word_count * sizeof(uint32_t),
      };
      ASSERT_NO_FATAL_FAILURE(queue->Submit(gpu_api_, command));
    }
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        publication_completion_addresses[publication_index],
        publication_completion_values[publication_index]));

    // Snapshot this complete semantic case before native retirement can supply
    // any payload visibility missing from the command stream.
    const size_t page_word_offset = first_execution * kPageWordCount;
    const size_t page_word_count = kExecutionsPerPublication * kPageWordCount;
    const size_t page_byte_offset = page_word_offset * sizeof(uint32_t);
    const size_t page_byte_length = page_word_count * sizeof(uint32_t);
    std::memcpy(observed_output.data() + page_word_offset,
                static_cast<uint8_t*>(output->host.pointer) + page_byte_offset,
                page_byte_length);
    std::memcpy(observed_input.data() + page_word_offset,
                static_cast<uint8_t*>(input->host.pointer) + page_byte_offset,
                page_byte_length);
    std::memcpy(observed_control.data() + page_word_offset,
                static_cast<uint8_t*>(control->host.pointer) + page_byte_offset,
                page_byte_length);
    std::memcpy(
        observed_arguments.data() + page_word_offset,
        static_cast<uint8_t*>(arguments->host.pointer) + page_byte_offset,
        page_byte_length);
    std::memcpy(observed_graph.data() + page_word_offset,
                static_cast<uint8_t*>(graph->host.pointer) + page_byte_offset,
                page_byte_length);
    if (primary_ring) {
      const size_t first_command_word =
          publication_index == 0 ? 0
                                 : publication_frontiers[publication_index - 1];
      const size_t publication_command_word_count =
          publication_frontiers[publication_index] - first_command_word;
      std::memcpy(observed_commands.data() + first_command_word,
                  queue->words().data() + first_command_word,
                  publication_command_word_count * sizeof(uint32_t));
    }
    if (publication_index + 1 == kPublicationCount) {
      std::memcpy(observed_code.data(), code->host.pointer,
                  code->info.byte_length);
      if (primary_ring) {
        const size_t trailing_command_word_count =
            command_capacity - publication_frontiers[publication_index];
        std::memcpy(
            observed_commands.data() + publication_frontiers[publication_index],
            queue->words().data() + publication_frontiers[publication_index],
            trailing_command_word_count * sizeof(uint32_t));
      }
    }
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
  }

  execution_index = 0;
  for (const auto& comparison : kComparisons) {
    for (const auto& operand : kOperands) {
      for (const uint32_t predicate_word : kPredicateWords) {
        const std::string case_name =
            BranchCaseName(comparison, operand, predicate_word);
        SCOPED_TRACE(case_name);
        const size_t publication_index = execution_index / kEpochCount;
        for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
          SCOPED_TRACE("epoch=" + std::to_string(epoch));
          const size_t page_word_offset = execution_index * kPageWordCount;
          ExpectWords(std::span(observed_output)
                          .subspan(page_word_offset, kPageWordCount),
                      std::span(expected_output)
                          .subspan(page_word_offset, kPageWordCount),
                      "selected and unselected output");
          ExpectWords(std::span(observed_input)
                          .subspan(page_word_offset, kPageWordCount),
                      std::span(expected_input)
                          .subspan(page_word_offset, kPageWordCount),
                      "input");
          ExpectWords(std::span(observed_control)
                          .subspan(page_word_offset, kPageWordCount),
                      std::span(expected_control)
                          .subspan(page_word_offset, kPageWordCount),
                      "control");
          ExpectWords(std::span(observed_arguments)
                          .subspan(page_word_offset, kPageWordCount),
                      std::span(expected_arguments)
                          .subspan(page_word_offset, kPageWordCount),
                      "arguments");
          ExpectWords(std::span(observed_graph)
                          .subspan(page_word_offset, kPageWordCount),
                      std::span(expected_graph)
                          .subspan(page_word_offset, kPageWordCount),
                      "immutable graph");
          ++execution_index;
        }
        if (primary_ring) {
          const size_t command_word_offset =
              publication_index * kCommandWordsPerPublication;
          ExpectWords(
              std::span(observed_commands)
                  .subspan(command_word_offset, kCommandWordsPerPublication),
              std::span(expected_commands)
                  .subspan(command_word_offset, kCommandWordsPerPublication),
              "case graph entry");
        }
      }
    }
  }
  ExpectWords(observed_code, expected_code, "code");
  if (primary_ring) {
    ExpectWords(observed_commands, expected_commands, "finite primary ring");
  }
  if (HasFailure()) {
    return;
  }
  RecordProperty("pm4_branch_completed_cases", kCaseCount);
  RecordProperty("pm4_branch_completed_executions", kExecutionCount);
}

}  // namespace
