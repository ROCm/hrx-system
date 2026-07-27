// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_PIPELINE_H_
#define LOOM_FORMAT_TEXT_PARSER_PIPELINE_H_

#include "iree/base/api.h"
#include "loom/format/text/parser/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses the pass pipeline friendly region syntax:
// `pipeline { canonicalize(options...) ... }`.
iree_status_t loom_parse_pipeline_prefixed_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_PIPELINE_H_
