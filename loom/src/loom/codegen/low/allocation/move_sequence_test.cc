// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/move_sequence.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

const loom_low_descriptor_set_t* IndependentDescriptorSet() {
  static const loom_low_reg_class_t kRegClasses[3] = {};
  static const loom_low_descriptor_set_t kDescriptorSet = [] {
    loom_low_descriptor_set_t descriptor_set = {};
    descriptor_set.reg_classes = kRegClasses;
    descriptor_set.reg_class_count = IREE_ARRAYSIZE(kRegClasses);
    return descriptor_set;
  }();
  return &kDescriptorSet;
}

const loom_low_descriptor_set_t* AliasDescriptorSet() {
  static const loom_low_reg_class_t kRegClasses[] = {
      {
          .name_string_ref = {},
          .target_bank_id = {},
          .flags = {},
          .alloc_unit_bits = 32,
          .allocatable_count = {},
          .fixed_location_base = {},
          .fixed_location_count = {},
          .physical_register_candidate_start = {},
          .candidate_lookup = {},
          .alias_set_id = 1,
      },
      {
          .name_string_ref = {},
          .target_bank_id = {},
          .flags = {},
          .alloc_unit_bits = 64,
          .allocatable_count = {},
          .fixed_location_base = {},
          .fixed_location_count = {},
          .physical_register_candidate_start = {},
          .candidate_lookup = {},
          .alias_set_id = 1,
      },
      {},
  };
  static const loom_low_descriptor_set_t kDescriptorSet = [] {
    loom_low_descriptor_set_t descriptor_set = *IndependentDescriptorSet();
    descriptor_set.reg_classes = kRegClasses;
    return descriptor_set;
  }();
  return &kDescriptorSet;
}

const loom_low_descriptor_set_t* ExplicitDescriptorSet() {
  static const loom_low_reg_class_t kRegClasses[] = {
      {
          /*.name_string_ref=*/{},
          /*.target_bank_id=*/{},
          /*.flags=*/LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
              LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS,
      },
      {
          /*.name_string_ref=*/{},
          /*.target_bank_id=*/{},
          /*.flags=*/LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
              LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS,
      },
  };
  static const uint16_t kAtomicUnits[] = {0, 1, 0, 2, 3, 4, 5};
  static const loom_low_physical_register_t kPhysicalRegisters[] = {
      {.name_string_ref = {}, .atomic_unit_start = 0, .atomic_unit_count = 2},
      {.name_string_ref = {}, .atomic_unit_start = 2, .atomic_unit_count = 1},
      {.name_string_ref = {}, .atomic_unit_start = 3, .atomic_unit_count = 2},
      {.name_string_ref = {}, .atomic_unit_start = 5, .atomic_unit_count = 2},
  };
  static const loom_low_descriptor_set_t kDescriptorSet = [] {
    loom_low_descriptor_set_t descriptor_set = {};
    descriptor_set.reg_classes = kRegClasses;
    descriptor_set.reg_class_count = IREE_ARRAYSIZE(kRegClasses);
    descriptor_set.physical_registers = kPhysicalRegisters;
    descriptor_set.physical_register_count = IREE_ARRAYSIZE(kPhysicalRegisters);
    descriptor_set.physical_register_atomic_units = kAtomicUnits;
    descriptor_set.physical_register_atomic_unit_count =
        IREE_ARRAYSIZE(kAtomicUnits);
    descriptor_set.physical_register_unit_count = 6;
    return descriptor_set;
  }();
  return &kDescriptorSet;
}

loom_low_move_location_t Location(uint32_t ordinal,
                                  uint16_t register_class_id = 0) {
  return loom_low_move_location_t{
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .descriptor_reg_class_id = register_class_id,
      .location = ordinal,
  };
}

loom_low_move_t Move(uint32_t destination, uint32_t source,
                     uint16_t register_class_id = 0) {
  return loom_low_move_t{
      .destination = Location(destination, register_class_id),
      .source = Location(source, register_class_id),
  };
}

loom_low_move_t MoveBetween(uint32_t destination,
                            uint16_t destination_register_class_id,
                            uint32_t source,
                            uint16_t source_register_class_id) {
  return loom_low_move_t{
      .destination = Location(destination, destination_register_class_id),
      .source = Location(source, source_register_class_id),
  };
}

std::string MoveString(const loom_low_move_t& move) {
  return std::to_string(move.destination.descriptor_reg_class_id) + ":" +
         std::to_string(move.destination.location) + "<-" +
         std::to_string(move.source.location);
}

class TestArena {
 public:
  TestArena() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  ~TestArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t arena_ = {};
};

struct TemporaryResolver {
  // Candidate scratch units supplied by the owning allocation.
  const loom_low_move_location_t* locations = nullptr;
  // Number of candidate scratch units.
  iree_host_size_t count = 0;
  // Optional location whose move-group occupancy is observed by the callback.
  const loom_low_move_location_t* occupancy_probe = nullptr;
  // Result of the most recent optional occupancy probe.
  bool occupancy_probe_result = false;
  // Number of resolver calls, used to check reuse across independent cycles.
  unsigned call_count = 0;
};

iree_status_t ResolveTemporary(
    void* user_data, const loom_low_move_location_t* storage_class,
    const loom_low_move_sequence_location_set_t* occupied_locations,
    loom_low_move_location_t* out_temporary, bool* out_resolved) {
  auto* resolver = static_cast<TemporaryResolver*>(user_data);
  ++resolver->call_count;
  if (resolver->occupancy_probe != nullptr) {
    resolver->occupancy_probe_result =
        loom_low_move_sequence_location_set_contains(occupied_locations,
                                                     resolver->occupancy_probe);
  }
  *out_resolved = false;
  for (iree_host_size_t i = 0; i < resolver->count; ++i) {
    const loom_low_move_location_t* location = &resolver->locations[i];
    if ((location->location_kind == storage_class->location_kind ||
         location->location_kind ==
             LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE) &&
        location->descriptor_reg_class_id ==
            storage_class->descriptor_reg_class_id &&
        !loom_low_move_sequence_location_set_contains(occupied_locations,
                                                      location)) {
      *out_temporary = *location;
      *out_resolved = true;
      break;
    }
  }
  return iree_ok_status();
}

std::vector<std::string> ResolveMoves(
    const loom_low_move_t* input_moves, iree_host_size_t move_count,
    const loom_low_move_location_t* temporaries,
    iree_host_size_t temporary_count,
    const loom_low_descriptor_set_t* descriptor_set =
        IndependentDescriptorSet(),
    loom_low_move_sequence_input_flags_t* out_input_flags = nullptr) {
  TestArena arena;
  loom_low_move_sequence_scratch_t scratch;
  IREE_EXPECT_OK(loom_low_move_sequence_scratch_initialize(
      arena.arena(), move_count, &scratch));
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    scratch.moves[i] = input_moves[i];
  }
  TemporaryResolver resolver = {
      temporaries,
      temporary_count,
  };
  const loom_low_move_sequence_options_t options = {
      descriptor_set,
      {
          ResolveTemporary,
          &resolver,
      },
  };
  std::vector<loom_low_move_t> resolved_moves(move_count * 2);
  iree_host_size_t resolved_move_count = 0;
  bool complete = false;
  IREE_EXPECT_OK(loom_low_move_sequence_resolve(
      &scratch, move_count, &options, resolved_moves.size(),
      resolved_moves.data(), &resolved_move_count, out_input_flags, &complete));
  EXPECT_TRUE(complete);
  std::vector<std::string> result;
  for (iree_host_size_t i = 0; i < resolved_move_count; ++i) {
    result.push_back(MoveString(resolved_moves[i]));
  }
  return result;
}

TEST(LowMoveSequenceTest, SkipsIdentityMoves) {
  const loom_low_move_t moves[] = {
      Move(0, 0),
      Move(1, 1),
  };

  EXPECT_TRUE(ResolveMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0).empty());
}

TEST(LowMoveSequenceTest, SkipsAliasIdentityMoves) {
  const loom_low_move_t moves[] = {
      MoveBetween(0, 1, 0, 0),
  };

  EXPECT_TRUE(ResolveMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0,
                           AliasDescriptorSet())
                  .empty());
}

TEST(LowMoveSequenceTest, ReportsIdentityAndActiveInputs) {
  const loom_low_move_t moves[] = {
      Move(0, 0),
      Move(1, 2),
  };
  loom_low_move_sequence_input_flags_t input_flags = 0;

  ResolveMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0,
               IndependentDescriptorSet(), &input_flags);

  EXPECT_TRUE(iree_all_bits_set(input_flags,
                                LOOM_LOW_MOVE_SEQUENCE_INPUT_FLAG_IDENTITY |
                                    LOOM_LOW_MOVE_SEQUENCE_INPUT_FLAG_ACTIVE));
}

TEST(LowMoveSequenceTest, EmitsIndependentMovesInInputOrder) {
  const loom_low_move_t moves[] = {
      Move(4, 0),
      Move(5, 1),
  };

  EXPECT_THAT(ResolveMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0),
              ::testing::ElementsAre("0:4<-0", "0:5<-1"));
}

TEST(LowMoveSequenceTest, ReordersForwardClobberingShift) {
  const loom_low_move_t moves[] = {
      Move(1, 0),
      Move(2, 1),
  };

  EXPECT_THAT(ResolveMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0),
              ::testing::ElementsAre("0:2<-1", "0:1<-0"));
}

TEST(LowMoveSequenceTest, ReordersAliasClobberingShift) {
  const loom_low_move_t moves[] = {
      MoveBetween(1, 1, 0, 1),
      MoveBetween(2, 0, 1, 0),
  };

  EXPECT_THAT(ResolveMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0,
                           AliasDescriptorSet()),
              ::testing::ElementsAre("0:2<-1", "1:1<-0"));
}

TEST(LowMoveSequenceTest, KeepsBackwardShiftInInputOrder) {
  const loom_low_move_t moves[] = {
      Move(0, 1),
      Move(1, 2),
  };

  EXPECT_THAT(ResolveMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0),
              ::testing::ElementsAre("0:0<-1", "0:1<-2"));
}

TEST(LowMoveSequenceTest, UsesTemporaryForCycle) {
  const loom_low_move_t moves[] = {
      Move(0, 1),
      Move(1, 0),
  };
  const loom_low_move_location_t temporary = Location(9);

  EXPECT_THAT(ResolveMoves(moves, IREE_ARRAYSIZE(moves), &temporary, 1),
              ::testing::ElementsAre("0:9<-1", "0:1<-0", "0:0<-9"));
}

TEST(LowMoveSequenceTest, ReusesMemoryTemporaryAcrossCycles) {
  TestArena arena;
  loom_low_move_sequence_scratch_t scratch = {};
  IREE_ASSERT_OK(
      loom_low_move_sequence_scratch_initialize(arena.arena(), 5, &scratch));
  scratch.moves[0] = Move(0, 1);
  scratch.moves[1] = Move(1, 0);
  scratch.moves[2] = Move(2, 3);
  scratch.moves[3] = Move(3, 2);
  scratch.moves[4] = Move(4, 4);
  auto temporary = Location(0);
  temporary.location_kind = LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE;
  TemporaryResolver resolver = {&temporary, 1};
  const loom_low_move_sequence_options_t options = {
      IndependentDescriptorSet(), {ResolveTemporary, &resolver}};
  loom_low_move_t output[6] = {};
  iree_host_size_t count = 0;
  bool complete = false;
  IREE_ASSERT_OK(loom_low_move_sequence_resolve(
      &scratch, 5, &options, IREE_ARRAYSIZE(output), output, &count,
      /*out_input_flags=*/nullptr, &complete));
  ASSERT_TRUE(complete);
  ASSERT_EQ(count, 6u);
  EXPECT_EQ(resolver.call_count, 1u);
  uint64_t registers[] = {11, 22, 33, 44, 55};
  uint64_t cell = 0;
  for (const auto& move : output) {
    const uint64_t value =
        move.source.location_kind == LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE
            ? cell
            : registers[move.source.location];
    if (move.destination.location_kind ==
        LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE) {
      cell = value;
    } else {
      registers[move.destination.location] = value;
    }
  }
  EXPECT_THAT(registers, ::testing::ElementsAre(22, 11, 44, 33, 55));
}

TEST(LowMoveSequenceTest, IdentityDestinationCannotBecomeCycleScratch) {
  const loom_low_move_location_t temporaries[] = {Location(2), Location(3)};
  const loom_low_descriptor_set_t* descriptor_sets[] = {
      IndependentDescriptorSet(), AliasDescriptorSet()};
  for (const auto* descriptor_set : descriptor_sets) {
    const loom_low_move_t moves[] = {
        Move(2, 2, descriptor_set == AliasDescriptorSet() ? 1 : 0),
        Move(0, 1),
        Move(1, 0),
    };
    EXPECT_THAT(ResolveMoves(moves, IREE_ARRAYSIZE(moves), temporaries,
                             IREE_ARRAYSIZE(temporaries), descriptor_set),
                ::testing::ElementsAre("0:3<-1", "0:1<-0", "0:0<-3"));
  }
}

TEST(LowMoveSequenceTest, IdentityExcludesAliasedPhysicalScratch) {
  TestArena arena;
  loom_low_move_sequence_scratch_t scratch = {};
  IREE_ASSERT_OK(
      loom_low_move_sequence_scratch_initialize(arena.arena(), 3, &scratch));
  scratch.moves[0] = Move(0, 0);
  scratch.moves[1] = Move(2, 3);
  scratch.moves[2] = Move(3, 2);
  // View 1 overlaps the low atomic unit of the identity's wider view 0.
  const loom_low_move_location_t temporary = Location(1);
  TemporaryResolver resolver = {&temporary, 1};
  const loom_low_move_sequence_options_t options = {
      ExplicitDescriptorSet(), {ResolveTemporary, &resolver}};
  loom_low_move_t output[4] = {};
  iree_host_size_t output_count = 0;
  bool complete = false;
  IREE_ASSERT_OK(loom_low_move_sequence_resolve(
      &scratch, 3, &options, IREE_ARRAYSIZE(output), output, &output_count,
      /*out_input_flags=*/nullptr, &complete));
  EXPECT_FALSE(complete);
}

TEST(LowMoveSequenceTest, AliasedCyclePreservesTransferWidths) {
  for (unsigned order = 0; order < 2; ++order) {
    SCOPED_TRACE(order);
    TestArena arena;
    loom_low_move_sequence_scratch_t scratch = {};
    IREE_ASSERT_OK(
        loom_low_move_sequence_scratch_initialize(arena.arena(), 2, &scratch));
    // Class 0 reads/writes the low 32 bits, and class 1 reads/writes 64 bits.
    // Both views name the same physical registers, as EAX/RAX do on x86.
    scratch.moves[order] = Move(1, 0, 0);
    scratch.moves[order ^ 1u] = Move(0, 1, 1);
    const loom_low_move_location_t temporaries[] = {Location(2, 0),
                                                    Location(2, 1)};
    TemporaryResolver resolver = {temporaries, IREE_ARRAYSIZE(temporaries)};
    const loom_low_move_sequence_options_t options = {
        AliasDescriptorSet(), {ResolveTemporary, &resolver}};
    loom_low_move_t output[3] = {};
    iree_host_size_t count = 0;
    bool complete = false;
    IREE_ASSERT_OK(loom_low_move_sequence_resolve(
        &scratch, 2, &options, IREE_ARRAYSIZE(output), output, &count,
        /*out_input_flags=*/nullptr, &complete));
    ASSERT_TRUE(complete);
    ASSERT_EQ(count, IREE_ARRAYSIZE(output));
    uint64_t registers[] = {UINT64_C(0x0123456789abcdef),
                            UINT64_C(0xfedcba9876543210), 0};
    for (iree_host_size_t i = 0; i < count; ++i) {
      const auto& move = output[i];
      uint64_t value = registers[move.source.location];
      if (move.source.descriptor_reg_class_id == 0 ||
          move.destination.descriptor_reg_class_id == 0) {
        value = static_cast<uint32_t>(value);
      }
      registers[move.destination.location] = value;
    }
    EXPECT_EQ(registers[0], UINT64_C(0xfedcba9876543210));
    EXPECT_EQ(registers[1], UINT64_C(0x89abcdef));
  }
}

TEST(LowMoveSequenceTest, TracksExplicitAtomicAliasesInLocationSet) {
  TestArena arena;
  loom_low_move_sequence_scratch_t scratch = {};
  IREE_ASSERT_OK(
      loom_low_move_sequence_scratch_initialize(arena.arena(), 2, &scratch));
  scratch.moves[0] = Move(0, 2);
  scratch.moves[1] = Move(2, 0);
  const loom_low_move_location_t temporary = Location(3);
  const loom_low_move_location_t alias_probe = Location(1, 1);
  TemporaryResolver resolver = {
      .locations = &temporary,
      .count = 1,
      .occupancy_probe = &alias_probe,
  };
  const loom_low_move_sequence_options_t options = {
      .descriptor_set = ExplicitDescriptorSet(),
      .resolve_temporary = {ResolveTemporary, &resolver},
  };
  loom_low_move_t output[3] = {};
  iree_host_size_t output_count = 0;
  bool complete = false;

  IREE_ASSERT_OK(loom_low_move_sequence_resolve(
      &scratch, 2, &options, IREE_ARRAYSIZE(output), output, &output_count,
      /*out_input_flags=*/nullptr, &complete));

  ASSERT_TRUE(complete);
  EXPECT_TRUE(resolver.occupancy_probe_result);
  ASSERT_EQ(output_count, IREE_ARRAYSIZE(output));
  EXPECT_EQ(MoveString(output[0]), "0:3<-2");
  EXPECT_EQ(MoveString(output[1]), "0:2<-0");
  EXPECT_EQ(MoveString(output[2]), "0:0<-3");

  scratch.moves[0] = Move(2, 3);
  scratch.moves[1] = Move(3, 2);
  const loom_low_move_location_t second_temporary = Location(0);
  resolver.locations = &second_temporary;
  resolver.occupancy_probe_result = true;
  IREE_ASSERT_OK(loom_low_move_sequence_resolve(
      &scratch, 2, &options, IREE_ARRAYSIZE(output), output, &output_count,
      /*out_input_flags=*/nullptr, &complete));
  ASSERT_TRUE(complete);
  EXPECT_FALSE(resolver.occupancy_probe_result);
}

TEST(LowMoveSequenceTest, UsesMatchingTemporaryForMixedClassCycles) {
  const loom_low_move_t moves[] = {
      Move(0, 1, 0),
      Move(1, 0, 0),
      Move(4, 5, 1),
      Move(5, 4, 1),
  };
  const loom_low_move_location_t temporaries[] = {
      Location(9, 0),
      Location(11, 1),
  };

  EXPECT_THAT(ResolveMoves(moves, IREE_ARRAYSIZE(moves), temporaries,
                           IREE_ARRAYSIZE(temporaries)),
              ::testing::ElementsAre("0:9<-1", "0:1<-0", "0:0<-9", "1:11<-5",
                                     "1:5<-4", "1:4<-11"));
  // Aliased classes still require their own encoding-compatible scratch
  // locations; sharing storage alone does not make their move forms equal.
  EXPECT_THAT(ResolveMoves(moves, IREE_ARRAYSIZE(moves), temporaries,
                           IREE_ARRAYSIZE(temporaries), AliasDescriptorSet()),
              ::testing::ElementsAre("0:9<-1", "0:1<-0", "0:0<-9", "1:11<-5",
                                     "1:5<-4", "1:4<-11"));
}

TEST(LowMoveSequenceTest, ReusesBoundedSolverStorageAcrossIncreasingGroups) {
  TestArena arena;
  constexpr uint32_t kCapacity = 128;
  loom_low_move_sequence_scratch_t scratch = {};
  IREE_ASSERT_OK(loom_low_move_sequence_scratch_initialize(
      arena.arena(), kCapacity, &scratch));
  loom_low_move_sequence_options_t options = {.descriptor_set =
                                                  IndependentDescriptorSet()};
  loom_low_move_t output[kCapacity] = {};
  iree_host_size_t storage_bytes = arena.arena()->used_allocation_size;
  for (uint32_t count = 0; count <= kCapacity; ++count) {
    SCOPED_TRACE(count);
    for (uint32_t i = 0; i < count; ++i) {
      scratch.moves[i] = Move(kCapacity + i, i);
    }
    iree_host_size_t output_count = 0;
    bool complete = false;
    IREE_ASSERT_OK(loom_low_move_sequence_resolve(
        &scratch, count, &options, kCapacity, output, &output_count,
        /*out_input_flags=*/nullptr, &complete));
    ASSERT_TRUE(complete);
    ASSERT_EQ(output_count, count);
    for (uint32_t i = 0; i < count; ++i) {
      EXPECT_EQ(output[i].destination.location, kCapacity + i);
      EXPECT_EQ(output[i].source.location, i);
    }
    if (count < 2) {
      EXPECT_EQ(scratch.nodes, nullptr);
    }
    if (count == 2) {
      storage_bytes = arena.arena()->used_allocation_size;
    }
    EXPECT_EQ(arena.arena()->used_allocation_size, storage_bytes);
    EXPECT_EQ(scratch.temporaries, nullptr);
  }
}

TEST(LowMoveSequenceTest, ReusesBoundedCycleStorageAcrossClassesAndGroups) {
  TestArena arena;
  constexpr uint32_t kCapacity = 128;
  loom_low_move_sequence_scratch_t scratch = {};
  IREE_ASSERT_OK(loom_low_move_sequence_scratch_initialize(
      arena.arena(), kCapacity, &scratch));
  const loom_low_move_location_t temporaries[] = {
      Location(kCapacity, 0), Location(kCapacity, 1), Location(kCapacity, 2)};
  TemporaryResolver resolver = {temporaries, IREE_ARRAYSIZE(temporaries)};
  loom_low_move_sequence_options_t options = {
      .descriptor_set = IndependentDescriptorSet(),
      .resolve_temporary = {ResolveTemporary, &resolver}};
  loom_low_move_t output[kCapacity + kCapacity / 2] = {};
  iree_host_size_t storage_bytes = 0;
  for (uint32_t count = 2; count <= kCapacity; count += 2) {
    SCOPED_TRACE(count);
    for (uint32_t i = 0; i < count; ++i) {
      scratch.moves[i] = Move(i, i ^ 1u, (i / 2) % 3);
    }
    iree_host_size_t output_count = 0;
    bool complete = false;
    IREE_ASSERT_OK(loom_low_move_sequence_resolve(
        &scratch, count, &options, IREE_ARRAYSIZE(output), output,
        &output_count, /*out_input_flags=*/nullptr, &complete));
    ASSERT_TRUE(complete);
    ASSERT_EQ(output_count, count + count / 2);
    uint32_t values[3][kCapacity + 1];
    for (uint32_t class_id = 0; class_id < 3; ++class_id) {
      for (uint32_t i = 0; i <= kCapacity; ++i) {
        values[class_id][i] = class_id * (kCapacity + 1) + i;
      }
    }
    for (iree_host_size_t i = 0; i < output_count; ++i) {
      const auto& move = output[i];
      values[move.destination.descriptor_reg_class_id][move.destination
                                                           .location] =
          values[move.source.descriptor_reg_class_id][move.source.location];
    }
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t class_id = (i / 2) % 3;
      EXPECT_EQ(values[class_id][i], class_id * (kCapacity + 1) + (i ^ 1u));
    }
    if (count == 2) {
      storage_bytes = arena.arena()->used_allocation_size;
    }
    EXPECT_EQ(arena.arena()->used_allocation_size, storage_bytes);
  }
}

}  // namespace
}  // namespace loom
