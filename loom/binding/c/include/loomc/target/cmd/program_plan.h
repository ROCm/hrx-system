// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_PROGRAM_PLAN_H_
#define LOOMC_TARGET_CMD_PROGRAM_PLAN_H_

#include "loomc/program_plan.h"

/// @file
/// Command-program projections over generic production plans.
///
/// A command plan contains one portable multi-root package unit, an optional
/// shared launch-config unit, and independently compilable executable units.
/// The generic plan API remains unaware of those roles; this target-specific
/// projection supplies the associations an application needs to compile cache
/// misses and form each root's live executable table.

#ifdef __cplusplus
extern "C" {
#endif

/// Portable single-root command-program artifact format.
#define LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM "loom-cmd"

/// Portable multi-root command-program package artifact format.
#define LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE "loom-cmd-package"

/// One root-local executable-slot requirement.
///
/// Exactly one source identifies the executable. An internal requirement has a
/// valid `unit` and an empty `import_name`. An external requirement has an
/// invalid `unit` and a non-empty application import name.
typedef struct loomc_cmd_program_plan_executable_requirement_t {
  /// Plan unit producing the executable, or an invalid token for an import.
  loomc_program_plan_unit_t unit;

  /// Application import name, or empty for an internal plan unit.
  loomc_string_view_t import_name;
} loomc_cmd_program_plan_executable_requirement_t;

/// Immutable command-specific metadata for one selected root.
///
/// @lifetime
/// `executable_requirements` and all string views borrow from the plan and
/// remain valid until its final release.
typedef struct loomc_cmd_program_plan_root_info_t {
  /// Unit producing the portable package containing every selected root.
  loomc_program_plan_unit_t package_unit;

  /// Shared command launch-config unit required by this root.
  ///
  /// The token is invalid when every dispatch in this root has direct counts.
  /// When valid, the matching function is looked up by the canonical root name
  /// returned from `loomc_program_plan_root_info`.
  loomc_program_plan_unit_t launch_config_unit;

  /// Requirements in the root-local executable-slot order used by the package.
  const loomc_cmd_program_plan_executable_requirement_t*
      executable_requirements;

  /// Number of entries in `executable_requirements`.
  loomc_host_size_t executable_requirement_count;
} loomc_cmd_program_plan_root_info_t;

/// Returns the command-specific projection for one selected root.
///
/// The operation fails with `LOOMC_STATUS_INVALID_ARGUMENT` for a malformed
/// root token and `LOOMC_STATUS_FAILED_PRECONDITION` when `program_plan` was
/// prepared by another program family.
///
/// Internal executable requirements identify independently compilable units.
/// After compiling or finding each unit in an application cache, place its
/// loaded executable in the same root-local order when materializing the
/// matching package export. The command package independently names each entry
/// within those possibly multi-export executables.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_plan_root_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_root_t root,
    loomc_cmd_program_plan_root_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_PROGRAM_PLAN_H_
