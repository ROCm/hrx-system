// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SRC_LOOM_OPS_COMMAND_COMPOSITION_H_
#define LOOM_SRC_LOOM_OPS_COMMAND_COMPOSITION_H_

#include "loom/ops/command/ops.h"
#include "loom/ops/kernel/ops.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Ordering rule applied to direct children of a command composition.
typedef uint8_t loom_command_composition_mode_t;
enum loom_command_composition_mode_e {
  LOOM_COMMAND_COMPOSITION_MODE_NONE = 0,
  // Each child begins after its preceding sibling completes.
  LOOM_COMMAND_COMPOSITION_MODE_SERIAL = 1,
  // Children have no cross-sibling dependency and join at region exit.
  LOOM_COMMAND_COMPOSITION_MODE_CONCURRENT = 2,
  LOOM_COMMAND_COMPOSITION_MODE_COUNT_ = 3,
};

// Returns true when |op| is a serial nested schedule. Hot command-stream
// consumers use this narrow predicate so their other operation-kind decisions
// can still fold into one dispatch table.
static inline bool loom_command_composition_is_serial_schedule(
    const loom_op_t* op) {
  if (op == NULL) {
    return false;
  }
  return op->kind == LOOM_OP_COMMAND_SERIAL ||
         op->kind == LOOM_OP_KERNEL_LAUNCH_SERIAL;
}

// Returns true when |op| is a concurrent nested schedule.
static inline bool loom_command_composition_is_concurrent_schedule(
    const loom_op_t* op) {
  if (op == NULL) {
    return false;
  }
  return op->kind == LOOM_OP_COMMAND_CONCURRENT ||
         op->kind == LOOM_OP_KERNEL_LAUNCH_CONCURRENT;
}

// Returns true when |op| is the serial schedule of a command program rather
// than a source kernel-launch schedule.
static inline bool loom_command_composition_is_serial_program_schedule(
    const loom_op_t* op) {
  return op != NULL && op->kind == LOOM_OP_COMMAND_SERIAL;
}

// Returns the nested-schedule mode of |op|, or NONE when |op| is not a
// command-program or source kernel-launch schedule. Consumers walking command
// streams use this narrow query so ordinary leaf commands do not pay to
// compute composition facts they do not consume.
static inline loom_command_composition_mode_t
loom_command_composition_schedule_mode(const loom_op_t* op) {
  if (loom_command_composition_is_serial_schedule(op)) {
    return LOOM_COMMAND_COMPOSITION_MODE_SERIAL;
  }
  return loom_command_composition_is_concurrent_schedule(op)
             ? LOOM_COMMAND_COMPOSITION_MODE_CONCURRENT
             : LOOM_COMMAND_COMPOSITION_MODE_NONE;
}

// Returns true when |op| is a source kernel-launch schedule.
static inline bool loom_command_composition_is_kernel_launch_schedule(
    const loom_op_t* op) {
  return op != NULL && (op->kind == LOOM_OP_KERNEL_LAUNCH_SERIAL ||
                        op->kind == LOOM_OP_KERNEL_LAUNCH_CONCURRENT);
}

// Returns true when |op| composes its direct children serially. Launch-count
// analysis asks this narrow question for every issued command.
static inline bool loom_command_composition_is_serial(const loom_op_t* op) {
  if (op == NULL) {
    return false;
  }
  return op->kind == LOOM_OP_COMMAND_DEF ||
         op->kind == LOOM_OP_COMMAND_SERIAL ||
         op->kind == LOOM_OP_KERNEL_LAUNCH_SERIAL;
}

// Returns the direct-child ordering mode of any composition operation,
// including a command-program root.
static inline loom_command_composition_mode_t loom_command_composition_mode(
    const loom_op_t* op) {
  if (loom_command_composition_is_serial(op)) {
    return LOOM_COMMAND_COMPOSITION_MODE_SERIAL;
  }
  return loom_command_composition_is_concurrent_schedule(op)
             ? LOOM_COMMAND_COMPOSITION_MODE_CONCURRENT
             : LOOM_COMMAND_COMPOSITION_MODE_NONE;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_SRC_LOOM_OPS_COMMAND_COMPOSITION_H_
