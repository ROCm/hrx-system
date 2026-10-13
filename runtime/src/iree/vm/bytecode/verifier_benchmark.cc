// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "iree/testing/benchmark.h"
#include "iree/vm/bytecode/module_fixture.h"
#include "iree/vm/bytecode/verifier.h"
#include "iree/vm/bytecode/verifier_testdata.h"

namespace {

iree_status_t InitializeFixture(
    iree_vm_bytecode_module_fixture_t* out_fixture) {
  const iree_file_toc_t* files = iree_vm_bytecode_verifier_testdata_create();
  return iree_vm_bytecode_module_fixture_initialize(
      iree_make_string_view(reinterpret_cast<const char*>(files[0].data),
                            files[0].size),
      iree_allocator_system(), out_fixture);
}

IREE_BENCHMARK_FN(BM_VerifyInstructions) {
  iree_vm_bytecode_module_fixture_t fixture = {};
  iree_vm_bytecode_module_plan_t plan = {};
  iree_status_t status = InitializeFixture(&fixture);
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_verify_module_structure(
        iree_const_cast_byte_span(fixture.contents), iree_allocator_system(),
        &plan);
  }
  std::vector<uint32_t> block_offsets;
  if (iree_status_is_ok(status)) {
    block_offsets.resize(plan.layout.functions.maximum_block_count);
  }

  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    status = iree_vm_bytecode_verify_module_instructions(&plan,
                                                         block_offsets.data());
  }
  iree_vm_bytecode_module_fixture_deinitialize(&fixture);
  return status;
}
IREE_BENCHMARK_REGISTER(BM_VerifyInstructions);

IREE_BENCHMARK_FN(BM_MapModule) {
  iree_vm_bytecode_module_fixture_t fixture = {};
  iree_status_t status = InitializeFixture(&fixture);
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_vm_bytecode_module_plan_t plan;
    status = iree_vm_bytecode_module_plan_build(
        iree_const_cast_byte_span(fixture.contents), &plan);
  }
  iree_vm_bytecode_module_fixture_deinitialize(&fixture);
  return status;
}
IREE_BENCHMARK_REGISTER(BM_MapModule);

IREE_BENCHMARK_FN(BM_VerifyLayout) {
  iree_vm_bytecode_module_fixture_t fixture = {};
  iree_vm_bytecode_module_plan_t plan;
  iree_status_t status = InitializeFixture(&fixture);
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_plan_build(
        iree_const_cast_byte_span(fixture.contents), &plan);
  }
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    status =
        iree_vm_bytecode_verify_module_layout(&plan, iree_allocator_system());
  }
  iree_vm_bytecode_module_fixture_deinitialize(&fixture);
  return status;
}
IREE_BENCHMARK_REGISTER(BM_VerifyLayout);

IREE_BENCHMARK_FN(BM_VerifyModule) {
  iree_vm_bytecode_module_fixture_t fixture = {};
  iree_status_t status = InitializeFixture(&fixture);
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_vm_bytecode_module_plan_t plan = {};
    status = iree_vm_bytecode_verify_module(
        iree_const_cast_byte_span(fixture.contents), iree_allocator_system(),
        &plan);
  }
  iree_vm_bytecode_module_fixture_deinitialize(&fixture);
  return status;
}
IREE_BENCHMARK_REGISTER(BM_VerifyModule);

}  // namespace
