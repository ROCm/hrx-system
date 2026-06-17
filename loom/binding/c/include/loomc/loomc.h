// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LOOMC_H_
#define LOOMC_LOOMC_H_

/// @file
/// Umbrella include for the public Loom C API.
///
/// This header re-exports the focused leaf headers. Unique API declarations
/// live in those leaf headers so each concept has local documentation and
/// tests.

#include "loomc/artifact.h"
#include "loomc/artifact_manifest.h"
#include "loomc/base.h"
#include "loomc/compile.h"
#include "loomc/compile_report.h"
#include "loomc/config.h"
#include "loomc/context.h"
#include "loomc/diagnostic.h"
#include "loomc/emit.h"
#include "loomc/link.h"
#include "loomc/link_index.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/sanitizer.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target.h"
#include "loomc/workspace.h"

#endif  // LOOMC_LOOMC_H_
