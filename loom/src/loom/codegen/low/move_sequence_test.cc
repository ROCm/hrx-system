// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/move_sequence.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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

const loom_low_descriptor_set_t* AliasDescriptorSet() {
  static const loom_low_reg_class_t kRegClasses[] = {
      {
          /*.name_string_offset=*/{},
          /*.target_bank_id=*/{},
          /*.flags=*/{},
          /*.alloc_unit_bits=*/{},
          /*.allocatable_count=*/{},
          /*.alias_set_id=*/1,
      },
      {
          /*.name_string_offset=*/{},
          /*.target_bank_id=*/{},
          /*.flags=*/{},
          /*.alloc_unit_bits=*/{},
          /*.allocatable_count=*/{},
          /*.alias_set_id=*/1,
      },
      {
          /*.name_string_offset=*/{},
          /*.target_bank_id=*/{},
          /*.flags=*/{},
          /*.alloc_unit_bits=*/{},
          /*.allocatable_count=*/{},
          /*.alias_set_id=*/0,
      },
  };
  static const loom_low_descriptor_set_t kDescriptorSet = {
      /*.abi_version=*/{},
      /*.generator_version=*/{},
      /*.stable_id=*/{},
      /*.target_stable_id=*/{},
      /*.descriptor_set_ordinal=*/{},
      /*.key_string_offset=*/{},
      /*.target_key_string_offset=*/{},
      /*.feature_key_string_offset=*/{},
      /*.string_table=*/{},
      /*.descriptors=*/{},
      /*.descriptor_count=*/{},
      /*.descriptor_refs=*/{},
      /*.descriptor_ref_count=*/{},
      /*.asm_forms=*/{},
      /*.asm_form_count=*/{},
      /*.asm_operand_indices=*/{},
      /*.asm_operand_index_count=*/{},
      /*.asm_immediates=*/{},
      /*.asm_immediate_count=*/{},
      /*.operands=*/{},
      /*.operand_count=*/{},
      /*.immediates=*/{},
      /*.immediate_count=*/{},
      /*.immediate_encoding_slices=*/{},
      /*.immediate_encoding_slice_count=*/{},
      /*.enum_domains=*/{},
      /*.enum_domain_count=*/{},
      /*.enum_values=*/{},
      /*.enum_value_count=*/{},
      /*.effects=*/{},
      /*.effect_count=*/{},
      /*.constraints=*/{},
      /*.constraint_count=*/{},
      /*.storage_leases=*/{},
      /*.storage_lease_count=*/{},
      /*.operand_forms=*/{},
      /*.operand_form_count=*/{},
      /*.operand_form_matches=*/{},
      /*.operand_form_match_count=*/{},
      /*.operand_form_operand_indices=*/{},
      /*.operand_form_operand_index_count=*/{},
      /*.reg_classes=*/kRegClasses,
      /*.reg_class_count=*/IREE_ARRAYSIZE(kRegClasses),
  };
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

std::string MoveString(const loom_low_move_location_t* destination,
                       const loom_low_move_location_t* source) {
  return std::to_string(destination->descriptor_reg_class_id) + ":" +
         std::to_string(destination->location) + "<-" +
         std::to_string(source->location);
}

iree_status_t RecordMove(void* user_data,
                         const loom_low_move_location_t* destination,
                         const loom_low_move_location_t* source) {
  auto* moves = static_cast<std::vector<std::string>*>(user_data);
  moves->push_back(MoveString(destination, source));
  return iree_ok_status();
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
  // Block pool backing |arena_|.
  iree_arena_block_pool_t block_pool_ = {};
  // Arena used for one test's move-sequencing scratch.
  iree_arena_allocator_t arena_ = {};
};

std::vector<std::string> EmitMoves(
    const loom_low_move_t* input_moves, iree_host_size_t move_count,
    const loom_low_move_location_t* temporaries,
    iree_host_size_t temporary_count,
    const loom_low_descriptor_set_t* descriptor_set = nullptr) {
  TestArena arena;
  loom_low_move_sequence_scratch_t scratch;
  loom_low_move_sequence_scratch_initialize(arena.arena(), &scratch);
  loom_low_move_t* moves = nullptr;
  IREE_EXPECT_OK(loom_low_move_sequence_scratch_reserve_moves(
      &scratch, move_count, &moves));
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    moves[i] = input_moves[i];
  }
  loom_low_move_location_t* temporary_storage = nullptr;
  IREE_EXPECT_OK(loom_low_move_sequence_scratch_reserve_temporaries(
      &scratch, temporary_count, &temporary_storage));
  for (iree_host_size_t i = 0; i < temporary_count; ++i) {
    temporary_storage[i] = temporaries[i];
  }
  std::vector<std::string> emitted_moves;
  loom_low_move_sequence_options_t options = {
      /*.descriptor_set=*/descriptor_set,
      /*.temporary_locations=*/temporary_storage,
      /*.temporary_location_count=*/temporary_count,
      /*.emit_move=*/
      {
          /*.fn=*/RecordMove,
          /*.user_data=*/&emitted_moves,
      },
  };
  IREE_EXPECT_OK(loom_low_move_sequence_emit(&scratch, move_count, &options));
  return emitted_moves;
}

TEST(LowMoveSequenceTest, SkipsIdentityMoves) {
  loom_low_move_t moves[] = {
      Move(0, 0),
      Move(1, 1),
  };

  EXPECT_TRUE(EmitMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0).empty());
}

TEST(LowMoveSequenceTest, SkipsAliasIdentityMoves) {
  loom_low_move_t moves[] = {
      MoveBetween(0, 1, 0, 0),
  };

  EXPECT_TRUE(
      EmitMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0, AliasDescriptorSet())
          .empty());
}

TEST(LowMoveSequenceTest, EmitsIndependentMovesInInputOrder) {
  loom_low_move_t moves[] = {
      Move(4, 0),
      Move(5, 1),
  };

  EXPECT_THAT(EmitMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0),
              ::testing::ElementsAre("0:4<-0", "0:5<-1"));
}

TEST(LowMoveSequenceTest, ReordersForwardClobberingShift) {
  loom_low_move_t moves[] = {
      Move(1, 0),
      Move(2, 1),
  };

  EXPECT_THAT(EmitMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0),
              ::testing::ElementsAre("0:2<-1", "0:1<-0"));
}

TEST(LowMoveSequenceTest, ReordersAliasClobberingShift) {
  loom_low_move_t moves[] = {
      MoveBetween(1, 1, 0, 1),
      MoveBetween(2, 0, 1, 0),
  };

  EXPECT_THAT(
      EmitMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0, AliasDescriptorSet()),
      ::testing::ElementsAre("0:2<-1", "1:1<-0"));
}

TEST(LowMoveSequenceTest, KeepsBackwardShiftInInputOrder) {
  loom_low_move_t moves[] = {
      Move(0, 1),
      Move(1, 2),
  };

  EXPECT_THAT(EmitMoves(moves, IREE_ARRAYSIZE(moves), nullptr, 0),
              ::testing::ElementsAre("0:0<-1", "0:1<-2"));
}

TEST(LowMoveSequenceTest, UsesTemporaryForCycle) {
  loom_low_move_t moves[] = {
      Move(0, 1),
      Move(1, 0),
  };
  const loom_low_move_location_t temporary = Location(9);

  EXPECT_THAT(EmitMoves(moves, IREE_ARRAYSIZE(moves), &temporary, 1),
              ::testing::ElementsAre("0:9<-0", "0:0<-1", "0:1<-9"));
}

TEST(LowMoveSequenceTest, UsesMatchingTemporaryForMixedClassCycles) {
  loom_low_move_t moves[] = {
      Move(0, 1, 0),
      Move(1, 0, 0),
      Move(4, 5, 1),
      Move(5, 4, 1),
  };
  const loom_low_move_location_t temporaries[] = {
      Location(9, 0),
      Location(11, 1),
  };

  EXPECT_THAT(EmitMoves(moves, IREE_ARRAYSIZE(moves), temporaries,
                        IREE_ARRAYSIZE(temporaries)),
              ::testing::ElementsAre("0:9<-0", "0:0<-1", "0:1<-9", "1:11<-4",
                                     "1:4<-5", "1:5<-11"));
}

TEST(LowMoveSequenceTest, RejectsCycleWithoutTemporary) {
  loom_low_move_t moves[] = {
      Move(0, 1),
      Move(1, 0),
  };
  std::vector<std::string> emitted_moves;
  loom_low_move_sequence_options_t options = {
      /*.descriptor_set=*/{},
      /*.temporary_locations=*/{},
      /*.temporary_location_count=*/{},
      /*.emit_move=*/
      {
          /*.fn=*/RecordMove,
          /*.user_data=*/&emitted_moves,
      },
  };
  TestArena arena;
  loom_low_move_sequence_scratch_t scratch;
  loom_low_move_sequence_scratch_initialize(arena.arena(), &scratch);
  loom_low_move_t* scratch_moves = nullptr;
  IREE_ASSERT_OK(loom_low_move_sequence_scratch_reserve_moves(
      &scratch, IREE_ARRAYSIZE(moves), &scratch_moves));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(moves); ++i) {
    scratch_moves[i] = moves[i];
  }

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_move_sequence_emit(&scratch, IREE_ARRAYSIZE(moves), &options));
  EXPECT_TRUE(emitted_moves.empty());
}

TEST(LowMoveSequenceTest, RejectsAliasDuplicateDestinations) {
  loom_low_move_t moves[] = {
      MoveBetween(0, 0, 2, 0),
      MoveBetween(0, 1, 3, 1),
  };
  std::vector<std::string> emitted_moves;
  loom_low_move_sequence_options_t options = {
      /*.descriptor_set=*/AliasDescriptorSet(),
      /*.temporary_locations=*/{},
      /*.temporary_location_count=*/{},
      /*.emit_move=*/
      {
          /*.fn=*/RecordMove,
          /*.user_data=*/&emitted_moves,
      },
  };
  TestArena arena;
  loom_low_move_sequence_scratch_t scratch;
  loom_low_move_sequence_scratch_initialize(arena.arena(), &scratch);
  loom_low_move_t* scratch_moves = nullptr;
  IREE_ASSERT_OK(loom_low_move_sequence_scratch_reserve_moves(
      &scratch, IREE_ARRAYSIZE(moves), &scratch_moves));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(moves); ++i) {
    scratch_moves[i] = moves[i];
  }

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_move_sequence_emit(&scratch, IREE_ARRAYSIZE(moves), &options));
  EXPECT_TRUE(emitted_moves.empty());
}

TEST(LowMoveSequenceTest, RejectsDuplicateDestinations) {
  loom_low_move_t moves[] = {
      Move(0, 1),
      Move(0, 2),
  };
  std::vector<std::string> emitted_moves;
  loom_low_move_sequence_options_t options = {
      /*.descriptor_set=*/{},
      /*.temporary_locations=*/{},
      /*.temporary_location_count=*/{},
      /*.emit_move=*/
      {
          /*.fn=*/RecordMove,
          /*.user_data=*/&emitted_moves,
      },
  };
  TestArena arena;
  loom_low_move_sequence_scratch_t scratch;
  loom_low_move_sequence_scratch_initialize(arena.arena(), &scratch);
  loom_low_move_t* scratch_moves = nullptr;
  IREE_ASSERT_OK(loom_low_move_sequence_scratch_reserve_moves(
      &scratch, IREE_ARRAYSIZE(moves), &scratch_moves));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(moves); ++i) {
    scratch_moves[i] = moves[i];
  }

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_move_sequence_emit(&scratch, IREE_ARRAYSIZE(moves), &options));
  EXPECT_TRUE(emitted_moves.empty());
}

}  // namespace
}  // namespace loom
