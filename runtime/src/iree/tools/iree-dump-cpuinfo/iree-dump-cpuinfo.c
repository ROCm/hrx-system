// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/base/tooling/flags.h"
#include "iree/task/api.h"

IREE_FLAG(bool, dump_task_topologies, false,
          "Dumps the flag-specified topology used for creating task "
          "executors.");

static void iree_dump_cpuinfo_print_cpu_features(void) {
  const uint64_t* cpu_data = iree_cpu_data_fields();

#define IREE_CPU_FEATURE_BIT(arch, field_index, bit_pos, bit_name, llvm_name) \
  if (IREE_ARCH_ENUM == IREE_ARCH_ENUM_##arch) {                              \
    bool result = (cpu_data[field_index] & (1ull << bit_pos)) != 0;           \
    printf("%-20s %d\n", llvm_name, (int)result);                             \
  }
#include "iree/schemas/cpu_feature_bits.inl"
#undef IREE_CPU_FEATURE_BIT
}

int main(int argc, char* argv[]) {
  iree_cpu_initialize(iree_allocator_system());

  iree_flags_set_usage(
      "iree-dump-cpuinfo",
      "Dumps host CPU features and task executor topologies.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (FLAG_dump_task_topologies) {
    status = iree_task_topologies_print_from_flags();
  } else {
    iree_dump_cpuinfo_print_cpu_features();
  }

  fflush(stdout);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
