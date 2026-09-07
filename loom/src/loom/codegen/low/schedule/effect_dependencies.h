// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Effect ordering and timing dependency construction.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_EFFECT_DEPENDENCIES_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_EFFECT_DEPENDENCIES_H_

#include "loom/codegen/low/schedule/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when any descriptor effect forms an ordered frontier.
bool loom_low_schedule_descriptor_has_ordered_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Adds effect ordering and target timing dependencies within and across CFG
// blocks.
iree_status_t loom_low_schedule_build_effect_dependencies(
    loom_low_schedule_build_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_EFFECT_DEPENDENCIES_H_
