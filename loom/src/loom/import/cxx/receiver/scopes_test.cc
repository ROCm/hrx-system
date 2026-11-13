// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/receiver/scopes.h"

#include <cxx/symbols.h>
#include <cxx/views/symbol_chain.h>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

class ReceiverScopesTest : public ::testing::Test {
 protected:
  static loom_cxx_import_options_t options() {
    loom_cxx_import_options_t result;
    loom_cxx_import_options_initialize(&result);
    return result;
  }

  ReceiverScopesTest()
      : source_(IREE_SV("struct Record { unsigned first, second; };"
                        "struct Nested { Record state; };"),
                IREE_SV("receivers.cxx"), options()),
        types_(source_.unit(), source_.diagnostics()),
        scopes_(values_) {}

  void SetUp() override {
    auto symbols = source_.unit().globalScope()->find("Record");
    ASSERT_NE(symbols.begin(), symbols.end());
    record_ = types_.record((*symbols.begin())->type(), nullptr);
    ASSERT_NE(record_, nullptr);
    ASSERT_EQ(record_->members.size(), 2u);
    first_ = record_->members[0].source;
    second_ = record_->members[1].source;
  }

  // Real source declarations shared by nested receiver instances.
  Source source_;
  // Canonical value schema admitted by the production type owner.
  Types types_;
  // Common SSA bindings observed by receiver and lexical expression updates.
  ValueBindings values_;
  // Scope transitions under test, borrowing the common value environment.
  ReceiverScopes scopes_;
  // Admitted complete record used for value and addressable receivers.
  const RecordPartition* record_ = nullptr;
  // First direct subobject declaration, reused by recursive current objects.
  cxx::Symbol* first_ = nullptr;
  // Second direct subobject, used to retain partial initialization state.
  cxx::Symbol* second_ = nullptr;
};

TEST_F(ReceiverScopesTest, NestedObjectsRestoreValuesAndCompletion) {
  EXPECT_EQ(scopes_.current(), nullptr);
  scopes_.enter(*record_, std::nullopt, record_->source->type());
  EXPECT_FALSE(scopes_.suspend());
  EXPECT_FALSE(scopes_.is_complete(first_));
  scopes_.publish(first_, Value(1));
  scopes_.publish(second_, Value(2));

  scopes_.enter(*record_, std::nullopt, record_->source->type());
  EXPECT_FALSE(scopes_.is_complete(first_));
  EXPECT_FALSE(scopes_.is_complete(second_));
  EXPECT_FALSE(values_.contains(first_));
  scopes_.publish(first_, Value(3));
  EXPECT_EQ(values_.at(first_).ssa(), 3u);
  scopes_.leave();

  EXPECT_TRUE(scopes_.is_complete(first_));
  EXPECT_TRUE(scopes_.is_complete(second_));
  EXPECT_EQ(values_.at(first_).ssa(), 1u);
  EXPECT_EQ(values_.at(second_).ssa(), 2u);
  EXPECT_EQ(scopes_.find(first_), scopes_.current());
  scopes_.leave();
  EXPECT_EQ(scopes_.current(), nullptr);
  EXPECT_EQ(scopes_.find(first_), nullptr);
  EXPECT_FALSE(values_.contains(first_));
  EXPECT_FALSE(values_.contains(second_));
}

TEST_F(ReceiverScopesTest, ExplicitClausesKeepLexicalWritesWhenResumed) {
  scopes_.enter(*record_, std::nullopt, record_->source->type());
  scopes_.publish(first_, Value(1));
  scopes_.publish(second_, Value(2));
  scopes_.enter(*record_, std::nullopt, record_->source->type());
  scopes_.publish(first_, Value(3));

  ASSERT_TRUE(scopes_.suspend());
  EXPECT_EQ(values_.at(first_).ssa(), 1u);
  EXPECT_TRUE(scopes_.is_complete(second_));
  scopes_.publish(first_, Value(4));
  // An explicit clause may construct another object before resuming its
  // original destination. Its temporary receiver restores the lexical write.
  scopes_.enter(*record_, std::nullopt, record_->source->type());
  scopes_.publish(first_, Value(5));
  scopes_.publish(second_, Value(6));
  scopes_.leave();
  EXPECT_EQ(values_.at(first_).ssa(), 4u);
  scopes_.resume();

  EXPECT_EQ(values_.at(first_).ssa(), 3u);
  EXPECT_TRUE(scopes_.is_complete(first_));
  EXPECT_FALSE(scopes_.is_complete(second_));
  EXPECT_FALSE(values_.contains(second_));
  scopes_.publish(second_, Value(7));
  scopes_.leave();
  EXPECT_EQ(values_.at(first_).ssa(), 4u);
  EXPECT_EQ(values_.at(second_).ssa(), 2u);
  scopes_.leave();
}

TEST_F(ReceiverScopesTest, StorageReceiversRetainIdentityWithoutValueBindings) {
  auto* observed_type =
      source_.unit().typeTraits().add_volatile(record_->source->type());
  const StorageProjection destination = {{11, 12}, 4, true};
  scopes_.enter(*record_, destination, observed_type);
  scopes_.mark_complete(first_);
  EXPECT_TRUE(scopes_.is_complete(first_));
  EXPECT_FALSE(scopes_.is_complete(second_));
  EXPECT_EQ(scopes_.current()->object_type, observed_type);
  EXPECT_EQ(scopes_.find(first_)->storage->pointer.root, 11u);
  EXPECT_EQ(scopes_.find(first_)->storage->pointer.byte_offset, 12u);
  EXPECT_EQ(scopes_.find(first_)->storage->alignment, 4u);
  EXPECT_TRUE(scopes_.find(first_)->storage->pointer_width_constrained);
  EXPECT_FALSE(values_.contains(first_));

  scopes_.enter(*record_, std::nullopt, record_->source->type());
  scopes_.publish(first_, Value(1));
  ASSERT_TRUE(scopes_.suspend());
  scopes_.mark_complete(second_);
  scopes_.resume();
  EXPECT_FALSE(scopes_.is_complete(second_));
  scopes_.leave();
  EXPECT_TRUE(scopes_.is_complete(second_));
  EXPECT_FALSE(values_.contains(first_));

  values_.clear();
  scopes_.clear();
  EXPECT_EQ(scopes_.current(), nullptr);
  EXPECT_EQ(scopes_.find(first_), nullptr);
}

TEST_F(ReceiverScopesTest, PartialNestedFieldsSurviveSuspensionAndCompletion) {
  auto* nested = types_.record(
      (*source_.unit().globalScope()->find("Nested").begin())->type(), nullptr);
  ASSERT_NE(nested, nullptr);
  auto* state = nested->members[0].source;
  ValueArena arena;

  scopes_.enter(*nested, std::nullopt, nested->source->type());
  values_.declare(state, *record_);
  values_.replace(state, 0, Value(11), arena);
  scopes_.enter(*nested, std::nullopt, nested->source->type());
  EXPECT_FALSE(values_.contains(state));
  values_.declare(state, *record_);
  values_.replace(state, 1, Value(22), arena);

  ASSERT_TRUE(scopes_.suspend());
  EXPECT_EQ(values_.project(state, kSSAPartition, 0, arena)->ssa(), 11u);
  EXPECT_FALSE(values_.project(state, kSSAPartition, 1, arena));
  values_.replace(state, 0, Value(33), arena);
  scopes_.resume();
  EXPECT_FALSE(values_.project(state, kSSAPartition, 0, arena));
  EXPECT_EQ(values_.project(state, kSSAPartition, 1, arena)->ssa(), 22u);
  EXPECT_FALSE(scopes_.is_complete(state));
  values_.replace(state, 0, Value(44), arena);
  scopes_.publish(state, values_.at(state));
  EXPECT_TRUE(scopes_.is_complete(state));

  scopes_.leave();
  EXPECT_FALSE(scopes_.is_complete(state));
  EXPECT_FALSE(values_.initialized(state));
  EXPECT_EQ(values_.project(state, kSSAPartition, 0, arena)->ssa(), 33u);
  EXPECT_FALSE(values_.project(state, kSSAPartition, 1, arena));
  scopes_.leave();
  EXPECT_FALSE(values_.contains(state));
}

}  // namespace
}  // namespace loom::cxx_import
