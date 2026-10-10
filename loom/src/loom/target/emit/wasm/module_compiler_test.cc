// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/module_compiler.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/selection.h"

namespace {

TEST(WasmModuleCompilerTest, ComposesTargetAndCanonicalEmitter) {
  loom_target_environment_t environment;
  IREE_ASSERT_OK(loom_target_environment_initialize(
      &loom_wasm_compiler_provider_set, &environment));

  const loom_target_specification_t specification = {
      .family = IREE_SVL("wasm"),
      .selector = IREE_SVL("simd128"),
  };
  const loom_target_profile_t* profile = nullptr;
  IREE_ASSERT_OK(loom_target_environment_select_profile(
      &environment, &specification, &profile));
  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(loom_target_environment_lookup_canonical_module_emitter(
                &environment, profile->type->fact_type),
            &loom_wasm_module_emitter);

  loom_target_environment_deinitialize(&environment);
}

}  // namespace
