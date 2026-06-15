// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_LOCATIONS_H_
#define LOOM_FORMAT_TEXT_PARSER_LOCATIONS_H_

#include "iree/base/api.h"
#include "loom/format/text/parser/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses an optional `loc(...)` annotation.
iree_status_t loom_parse_optional_op_location(
    loom_parser_t* parser, loom_location_id_t fallback_location,
    loom_location_id_t* out_location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_LOCATIONS_H_
