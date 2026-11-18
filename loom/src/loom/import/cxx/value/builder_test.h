// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_BUILDER_TEST_H_
#define LOOM_IMPORT_CXX_VALUE_BUILDER_TEST_H_

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/types.h>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/scalar.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"

namespace loom::cxx_import {

// Native builder fixture with an independently configured frontend. Components
// are constructed locally by each test against the same production interfaces.
class ValueBuilderTest : public ::testing::Test {
 protected:
  static loom_cxx_import_options_t options() {
    loom_cxx_import_options_t result;
    loom_cxx_import_options_initialize(&result);
    return result;
  }
  void SetUp() override {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("values"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }
  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  loom_op_t* producer(loom_value_id_t value) {
    return loom_value_def_op(loom_module_value(module_, value));
  }
  // Source frontend owns type interning and diagnostic provenance.
  Source source_{IREE_SV("int entry();"), IREE_SV("values.cpp"), options()};
  // Type projection shared with the tested builder.
  Types types_{source_.unit(), source_.diagnostics()};
  // Pool outlives the module.
  iree_arena_block_pool_t pool_ = {};
  // Finalized dialect context outlives the module.
  loom_context_t context_ = {};
  // Test-owned output module.
  loom_module_t* module_ = nullptr;
  // Insertion point used by the tested builder.
  loom_builder_t builder_ = {};
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_BUILDER_TEST_H_
