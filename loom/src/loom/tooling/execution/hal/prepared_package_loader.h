// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loader for prepared HAL testbench packages.

#ifndef LOOM_TOOLING_EXECUTION_HAL_PREPARED_PACKAGE_LOADER_H_
#define LOOM_TOOLING_EXECUTION_HAL_PREPARED_PACKAGE_LOADER_H_

#include "loom/tooling/execution/hal/scenario_profile.h"
#include "loomc/loomc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_testbench_prepared_package_t
    loom_run_hal_testbench_prepared_package_t;

// Opens and validates one prepared package directory.
//
// The package owns deserialized prepared modules, launch-config bytes, and the
// exact target profile selected by its manifest. Payload paths that are shared
// by multiple subjects are loaded once. |context| and |workspace| must outlive
// the returned package.
iree_status_t loom_run_hal_testbench_prepared_package_open(
    iree_string_view_t package_directory,
    loomc_target_environment_t* target_environment, loomc_context_t* context,
    loomc_workspace_t* workspace, iree_allocator_t host_allocator,
    loom_run_hal_testbench_prepared_package_t** out_package);

// Releases all storage and compiler handles owned by |package|.
void loom_run_hal_testbench_prepared_package_release(
    loom_run_hal_testbench_prepared_package_t* package);

// Returns the package-owned path of the untouched authored Loom bytecode.
iree_string_view_t loom_run_hal_testbench_prepared_package_authored_module_path(
    const loom_run_hal_testbench_prepared_package_t* package);

// Returns the exact canonical target-profile selector from the manifest.
iree_string_view_t loom_run_hal_testbench_prepared_package_target_profile(
    const loom_run_hal_testbench_prepared_package_t* package);

// Returns a resolver borrowing |package| until scenario product teardown.
loom_run_hal_testbench_prepared_actual_resolver_t
loom_run_hal_testbench_prepared_package_resolver(
    loom_run_hal_testbench_prepared_package_t* package);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_PREPARED_PACKAGE_LOADER_H_
