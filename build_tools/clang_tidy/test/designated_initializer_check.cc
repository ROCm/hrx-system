// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "designated_initializer_check.h"

struct Config {
  int ordinal;
  const char* name;
  unsigned flags;
};

struct InnerConfig {
  int x;
  int y;
};

struct OuterConfig {
  InnerConfig inner;
  int tail;
};

union Choice {
  int integer;
  float real;
};

struct Numbers {
  int first;
  int second;
};

struct WithDefault {
  int first = 1;
  int second;
};

struct NontrivialMember {
  NontrivialMember() : value(0) {}
  NontrivialMember(int new_value) : value(new_value) {}
  NontrivialMember& operator=(int new_value) {
    value = new_value;
    return *this;
  }
  int value;
};

struct NontrivialAggregate {
  NontrivialMember member;
};

struct CopyableMember {
  CopyableMember() = default;
  CopyableMember(const CopyableMember&) = default;
};

struct CopyableAggregate {
  CopyableMember member;
};

struct WithAnonymous {
  union {
    int integer;
    float real;
  };
  int tail;
};

struct BaseConfig {
  int base;
};

struct DerivedConfig : BaseConfig {
  int member;
};

struct NarrowConfig {
  unsigned char value;
};

#define MAKE_CONFIG(value) \
  Config { /*.ordinal=*/ value }

#define CONFIG_VALUE(value) value

#define CONFIG_ORDINAL 41

#define FORWARD_CONFIG(...) __VA_ARGS__

#define EMPTY_BRACES \
  {                  \
  }

Config labeled_config = {
    /*.ordinal=*/1,
    /*.name=*/"device",
    /*.flags=*/2,
};

OuterConfig nested_config = {
    /*.inner=*/{
        /*.x=*/3,
        /*.y=*/4,
    },
    /*.tail=*/5,
};

Config already_sparse = {
    .ordinal = 6,
    .flags = 7,
};

Config mixed_labels = {
    /*.ordinal=*/7,
    "mixed",
    /*.flags=*/8,
};

Choice labeled_union = {/*.integer=*/9};
Config empty_braced_label = {/*.ordinal=*/{}};

// The comment names a different field than positional initialization selects.
Config stale_label = {/*.name=*/10};
Choice stale_union_label = {/*.real=*/11};

WithAnonymous anonymous_label = {/*.integer=*/12, /*.tail=*/13};
WithAnonymous empty_braced_anonymous_label = {/*.integer=*/{}};
WithAnonymous stale_anonymous_label = {/*.real=*/14, /*.tail=*/15};
WithAnonymous braced_anonymous_label = {/*.integer=*/{101}};

DerivedConfig base_label = {/*.base=*/{16}, /*.member=*/17};
OuterConfig brace_elided = {/*.inner=*/18, 19, 20};

Config macro_config = MAKE_CONFIG(21);

Config macro_value_config = {
    /*.ordinal=*/CONFIG_VALUE(22),
    /*.name=*/"macro-value",
    /*.flags=*/CONFIG_VALUE(23),
};

Config macro_argument_config = FORWARD_CONFIG(Config{
    /*.ordinal=*/24,
    /*.name=*/"macro-argument",
    /*.flags=*/25,
});

CopyableMember copyable_source;
CopyableAggregate copyable_config = {/*.member=*/copyable_source};

void ConsumeConfig(int ordinal, const char* name);

struct ConstructedConfig {
  ConstructedConfig(int ordinal, const char* name);
};

void LabelCallArguments() {
  FORWARD_CONFIG(ConsumeConfig(/*.ordinal=*/26, /*.name=*/"call"));
  ConstructedConfig config(/*.ordinal=*/27, /*.name=*/"constructor");
  (void)config;
}

void Observe(const Numbers&);
int Next();

void FoldSetupBlocks() {
  Numbers configured = {};
  configured.first = 20;
  configured.second = 21;
  Observe(configured);

  Config sparse = {};
  sparse.flags = 22;
  (void)sparse;

  Numbers evaluated_in_order = {};
  evaluated_in_order.first = Next();
  evaluated_in_order.second = Next();
  Observe(evaluated_in_order);

  Numbers macro_values = {};
  macro_values.first = CONFIG_VALUE(40);
  macro_values.second = CONFIG_ORDINAL;
  Observe(macro_values);

  Numbers sized = {};
  sized.first = sizeof(sized);
  Observe(sized);
}

void PreserveUnsafeSetupBlocks(bool condition) {
  Numbers reordered = {};
  reordered.second = 23;
  reordered.first = 24;

  Numbers self_referencing = {};
  self_referencing.first = 25;
  self_referencing.second = self_referencing.first;

  Numbers aliased = {};
  int* alias = &aliased.first;
  aliased.first = 26;
  *alias = 27;

  Numbers observed = {};
  observed.first = 28;
  Observe(observed);
  observed.second = 29;

  Numbers conditional = {};
  if (condition) {
    conditional.first = 30;
  }

  Choice union_setup = {};
  union_setup.integer = 31;

  WithDefault defaulted = {};
  defaulted.second = 32;

  NontrivialAggregate nontrivial = {};
  nontrivial.member = 33;

  NarrowConfig narrowing = {};
  narrowing.value = -1;

  Numbers macro_initialized = EMPTY_BRACES;
  macro_initialized.first = 34;

  // This setup intentionally remains assignment-based for an external API.
  Numbers suppressed = {};  // NOLINT(iree-cpp-designated-initializer)
  suppressed.first = 35;

  Numbers commented_initializer = {/* Preserve the explicit zero setup. */};
  commented_initializer.first = 36;

  Numbers comment_before_assignment = {};
  // Preserve the explanation attached to the assignment.
  comment_before_assignment.first = 37;

  Numbers comment_between_assignments = {};
  comment_between_assignments.first = 38;
  // Preserve the explanation attached to the second assignment.
  comment_between_assignments.second = 39;

  (void)reordered;
  (void)self_referencing;
  (void)aliased;
  (void)conditional;
  (void)union_setup;
  (void)defaulted;
  (void)nontrivial;
  (void)narrowing;
  (void)macro_initialized;
  (void)suppressed;
  (void)commented_initializer;
  (void)comment_before_assignment;
  (void)comment_between_assignments;
}
