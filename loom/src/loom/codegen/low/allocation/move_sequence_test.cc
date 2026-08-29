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
#include "loom/codegen/low/allocation/storage.h"

namespace loom {
namespace {

loom_liveness_value_class_t ValueClass(uint16_t register_class_id) {
  return loom_liveness_value_class_t{
      /*.type_kind=*/LOOM_TYPE_REGISTER,
      /*.element_type=*/{},
      /*.register_descriptor_set_stable_id=*/{},
      /*.register_class_id=*/register_class_id,
  };
}

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
          /*.name_string_offset=*/{},
          /*.target_bank_id=*/{},
          /*.flags=*/{},
          /*.alloc_unit_bits=*/{},
          /*.allocatable_count=*/{},
          /*.fixed_location_base=*/{},
          /*.fixed_location_count=*/{},
          /*.physical_register_candidate_start=*/{},
          /*.alias_set_id=*/1,
      },
      {
          /*.name_string_offset=*/{},
          /*.target_bank_id=*/{},
          /*.flags=*/{},
          /*.alloc_unit_bits=*/{},
          /*.allocatable_count=*/{},
          /*.fixed_location_base=*/{},
          /*.fixed_location_count=*/{},
          /*.physical_register_candidate_start=*/{},
          /*.alias_set_id=*/1,
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

loom_low_move_location_t Location(uint32_t ordinal,
                                  uint16_t register_class_id = 0) {
  return loom_low_move_location_t{
      /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      /*.value_class=*/ValueClass(register_class_id),
      /*.descriptor_reg_class_id=*/register_class_id,
      /*.location=*/ordinal,
  };
}

loom_low_move_t Move(uint32_t destination, uint32_t source,
                     uint16_t register_class_id = 0) {
  return loom_low_move_t{
      /*.destination=*/Location(destination, register_class_id),
      /*.source=*/Location(source, register_class_id),
  };
}

loom_low_move_t MoveBetween(uint32_t destination,
                            uint16_t destination_register_class_id,
                            uint32_t source,
                            uint16_t source_register_class_id) {
  return loom_low_move_t{
      /*.destination=*/Location(destination, destination_register_class_id),
      /*.source=*/Location(source, source_register_class_id),
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
  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  const loom_low_move_location_t* locations = nullptr;
  iree_host_size_t count = 0;
};

iree_status_t ResolveTemporary(void* user_data,
                               const loom_low_move_location_t* storage_class,
                               const loom_low_move_t* moves,
                               iree_host_size_t move_count,
                               loom_low_move_location_t* out_temporary,
                               bool* out_resolved) {
  (void)moves;
  (void)move_count;
  auto* resolver = static_cast<TemporaryResolver*>(user_data);
  *out_resolved = false;
  for (iree_host_size_t i = 0; i < resolver->count; ++i) {
    const loom_low_move_location_t* location = &resolver->locations[i];
    if (location->location_kind == storage_class->location_kind &&
        loom_low_allocation_storage_reg_classes_share(
            resolver->descriptor_set, location->descriptor_reg_class_id,
            storage_class->descriptor_reg_class_id) &&
        loom_liveness_value_class_equal(location->value_class,
                                        storage_class->value_class)) {
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
        IndependentDescriptorSet()) {
  TestArena arena;
  loom_low_move_sequence_scratch_t scratch;
  IREE_EXPECT_OK(loom_low_move_sequence_scratch_initialize(
      arena.arena(), move_count, &scratch));
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    scratch.moves[i] = input_moves[i];
  }
  TemporaryResolver resolver = {
      descriptor_set,
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
      resolved_moves.data(), &resolved_move_count, &complete));
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
              ::testing::ElementsAre("0:9<-0", "0:0<-1", "0:1<-9"));
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
              ::testing::ElementsAre("0:9<-0", "0:0<-1", "0:1<-9", "1:11<-4",
                                     "1:4<-5", "1:5<-11"));
}

}  // namespace
}  // namespace loom
