// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/abi/task/state_layout.h"

#include "iree/hal/driver/task/executable/library/abi.h"

const loom_task_builtin_info_t loom_task_builtins[LOOM_TASK_BUILTIN_COUNT_] = {
    {IREE_SVL("task.workgroup_id_x"), 2, 0, 4},
    {IREE_SVL("task.workgroup_id_y"), 2, 4, 4},
    {IREE_SVL("task.workgroup_id_z"), 2, 8, 2},
    {IREE_SVL("task.workgroup_count_x"), 1, 12, 4},
    {IREE_SVL("task.workgroup_count_y"), 1, 16, 4},
    {IREE_SVL("task.workgroup_count_z"), 1, 20, 2},
};

// Check the target layout against the versioned schema on matching hosts.
#if defined(IREE_PTR_SIZE_64)
static_assert(sizeof(iree_hal_executable_library_v0_t) ==
                  LOOM_TASK_ABI64_LIBRARY_SIZE,
              "update the 64-bit task library layout adapter");
static_assert(offsetof(iree_hal_executable_library_v0_t, exports) ==
                  LOOM_TASK_ABI64_LIBRARY_EXPORTS,
              "update the 64-bit task export table offset");
static_assert(sizeof(iree_hal_executable_library_header_t) ==
                  LOOM_TASK_ABI64_HEADER_SIZE,
              "update the 64-bit task library header adapter");
static_assert(offsetof(iree_hal_executable_library_header_t, name) ==
                  LOOM_TASK_ABI64_HEADER_NAME,
              "update the 64-bit task library name offset");
static_assert(offsetof(iree_hal_executable_export_table_v0_t, ptrs) ==
                      LOOM_TASK_ABI64_EXPORT_POINTERS &&
                  offsetof(iree_hal_executable_export_table_v0_t, attrs) ==
                      LOOM_TASK_ABI64_EXPORT_ATTRIBUTES &&
                  offsetof(iree_hal_executable_export_table_v0_t, params) ==
                      LOOM_TASK_ABI64_EXPORT_PARAMETERS &&
                  offsetof(iree_hal_executable_export_table_v0_t, names) ==
                      LOOM_TASK_ABI64_EXPORT_NAMES,
              "update the 64-bit task export table adapter");
static_assert(sizeof(iree_hal_executable_dispatch_attrs_v0_t) ==
                  LOOM_TASK_ABI64_ATTRIBUTES_SIZE,
              "update the dispatch attribute adapter");
static_assert(offsetof(iree_hal_executable_dispatch_state_v0_t, constants) ==
                  LOOM_TASK_ABI64_CONSTANTS_OFFSET,
              "task constant span ABI changed");
static_assert(offsetof(iree_hal_executable_dispatch_state_v0_t, binding_ptrs) ==
                  LOOM_TASK_ABI64_BINDINGS_OFFSET,
              "task binding table ABI changed");

#endif  // IREE_PTR_SIZE_64
static_assert(sizeof(iree_hal_executable_dispatch_parameter_v0_t) ==
                  LOOM_TASK_ABI64_PARAMETER_SIZE,
              "update the parameter adapter");
