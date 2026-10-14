// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-line, file, and reporting adapters for Loom configuration.

#ifndef LOOM_TOOLING_CONFIG_CONFIG_H_
#define LOOM_TOOLING_CONFIG_CONFIG_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/tooling/config/input.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes a stable JSON description of config symbols in |module|.
//
// The report includes unresolved config.decl symbols and resolved config.def
// defaults or overrides. Textual type/default fields are display and
// round-trip strings; the constraints array is the structured contract.
iree_status_t loom_tooling_config_format_schema_json(
    const loom_module_t* module, loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_CONFIG_CONFIG_H_
