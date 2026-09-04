// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "iree/testing/benchmark.h"
#include "iree/vm/bytecode/verifier.h"
#include "iree/vm/bytecode/verifier_testdata.h"

namespace {

iree_const_byte_span_t GetFixtureContents() {
  const iree_file_toc_t* files = iree_vm_bytecode_verifier_testdata_create();
  return iree_make_const_byte_span(files[0].data, files[0].size);
}

IREE_BENCHMARK_FN(BM_VerifyInstructions) {
  iree_vm_bytecode_module_plan_t plan = {};
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_verify_module_structure(GetFixtureContents(), &plan));
  std::vector<uint32_t> block_offsets(plan.maximum_block_count);

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    status = iree_vm_bytecode_verify_module_instructions(&plan,
                                                         block_offsets.data());
    iree_optimization_barrier(plan.required_atomic_carrier_bits);
  }
  return status;
}
IREE_BENCHMARK_REGISTER(BM_VerifyInstructions);

IREE_BENCHMARK_FN(BM_VerifyModule) {
  iree_vm_bytecode_module_plan_t initial_plan = {};
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_module_structure(
      GetFixtureContents(), &initial_plan));
  std::vector<uint32_t> block_offsets(initial_plan.maximum_block_count);

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_vm_bytecode_module_plan_t plan = {};
    status =
        iree_vm_bytecode_verify_module_structure(GetFixtureContents(), &plan);
    if (iree_status_is_ok(status)) {
      status = iree_vm_bytecode_verify_module_instructions(
          &plan, block_offsets.data());
    }
    iree_optimization_barrier(plan.required_atomic_carrier_bits);
  }
  return status;
}
IREE_BENCHMARK_REGISTER(BM_VerifyModule);

}  // namespace
