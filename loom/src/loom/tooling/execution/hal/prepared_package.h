// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Strict manifest for prepared HAL testbench actuals.

#ifndef LOOM_TOOLING_EXECUTION_HAL_PREPARED_PACKAGE_H_
#define LOOM_TOOLING_EXECUTION_HAL_PREPARED_PACKAGE_H_

#include "iree/base/api.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// One root-selected direct-kernel product in a prepared package.
typedef struct loom_run_hal_testbench_prepared_package_product_t {
  // Unique module-local subject name in the authored module.
  iree_string_view_t subject;
  // Package-relative prepared-low Loom bytecode path.
  iree_string_view_t prepared_module_path;
  // Package-relative launch-config Loom bytecode path.
  iree_string_view_t launch_config_path;
} loom_run_hal_testbench_prepared_package_product_t;

// Parsed `loom.testbench.prepared_actuals.v0` package manifest.
//
// V0 represents one unconfigured direct-kernel product set for one exact target
// profile. Strings and product rows are owned by |storage| and remain valid
// until deinitialization.
typedef struct loom_run_hal_testbench_prepared_package_manifest_t {
  // Allocator owning |products| and |storage|.
  iree_allocator_t host_allocator;
  // Product rows in manifest order.
  loom_run_hal_testbench_prepared_package_product_t* products;
  // Number of entries in |products|.
  iree_host_size_t product_count;
  // Decoded string storage referenced by this manifest and its products.
  void* storage;
  // Package-relative untouched authored Loom bytecode path.
  iree_string_view_t authored_module_path;
  // Exact canonical target-profile selector used by the producer.
  iree_string_view_t target_profile;
  // Exact public executable artifact format selected by the producer.
  iree_string_view_t artifact_format;
} loom_run_hal_testbench_prepared_package_manifest_t;

// Parses and strictly validates one v0 prepared-package manifest.
//
// Unknown, duplicate, missing, and wrongly typed fields are rejected. All file
// paths must be portable relative paths contained by the package. Products may
// share prepared modules and launch programs, but those two semantic roles and
// the authored module must never alias each other.
iree_status_t loom_run_hal_testbench_prepared_package_manifest_parse(
    iree_string_view_t json, iree_allocator_t host_allocator,
    loom_run_hal_testbench_prepared_package_manifest_t* out_manifest);

// Releases all storage owned by |manifest|.
void loom_run_hal_testbench_prepared_package_manifest_deinitialize(
    loom_run_hal_testbench_prepared_package_manifest_t* manifest);

// Writes one validated v0 prepared-package manifest as canonical compact JSON.
iree_status_t loom_run_hal_testbench_prepared_package_manifest_write(
    const loom_run_hal_testbench_prepared_package_manifest_t* manifest,
    loom_output_stream_t* stream);

// Finds the product for |subject| or returns NULL when absent.
const loom_run_hal_testbench_prepared_package_product_t*
loom_run_hal_testbench_prepared_package_manifest_find(
    const loom_run_hal_testbench_prepared_package_manifest_t* manifest,
    iree_string_view_t subject);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_PREPARED_PACKAGE_H_
