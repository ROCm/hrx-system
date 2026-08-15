// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable target-neutral portable command programs.

#ifndef LOOM_TARGET_ARCH_CMD_PROGRAM_H_
#define LOOM_TARGET_ARCH_CMD_PROGRAM_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Number of u32 dimensions in one serialized workgroup-count tuple.
  LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT = 3,
  // Byte length of one serialized workgroup-count tuple.
  LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_BYTE_LENGTH =
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT * sizeof(uint32_t),
  // Required byte alignment of one serialized workgroup-count tuple.
  LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_ALIGNMENT = 4,
};

// Materialization role of a serialized buffer root.
typedef enum loom_cmd_program_buffer_role_e {
  // A concrete buffer is fixed while the live program remains materialized.
  LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED = 1,
  // A stable binding slot receives its concrete buffer at issue time.
  LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE = 2,
} loom_cmd_program_buffer_role_t;

// One resolved range within a fixed or rebindable buffer root.
typedef struct loom_cmd_program_buffer_ref_t {
  // Root materialization role.
  loom_cmd_program_buffer_role_t role;
  // Dense root index in the table selected by |role|.
  uint32_t root_index;
  // Root-relative byte offset.
  uint64_t byte_offset;
  // Byte length, or UINT64_MAX for the remaining root range.
  uint64_t byte_length;
} loom_cmd_program_buffer_ref_t;

// Kind of one logical kernel argument in an executable-entry schema.
typedef enum loom_cmd_program_argument_kind_e {
  // A direct fixed or rebindable root, byte offset, and byte length tuple.
  LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER = 1,
  // An exact 8-bit scalar payload.
  LOOM_CMD_PROGRAM_ARGUMENT_KIND_B8 = 2,
  // An exact 16-bit scalar payload.
  LOOM_CMD_PROGRAM_ARGUMENT_KIND_B16 = 3,
  // An exact 32-bit scalar payload.
  LOOM_CMD_PROGRAM_ARGUMENT_KIND_B32 = 4,
  // An exact 64-bit scalar payload.
  LOOM_CMD_PROGRAM_ARGUMENT_KIND_B64 = 5,
} loom_cmd_program_argument_kind_t;

// Logical argument schema for one executable-entry requirement.
//
// The schema identifies a program-local entry and describes every tagless
// dispatch payload targeting it. It does not encode native offsets, alignment,
// or padding. Materializers combine the logical kinds with implementation
// reflection when recording each dispatch.
typedef struct loom_cmd_program_entry_schema_t {
  // Dense program-local executable entry requirement index.
  uint32_t entry_index;
  // First kind byte in the flattened schema-kind table.
  uint32_t kind_offset;
  // Number of logical arguments in the schema.
  uint32_t argument_count;
  // Exact bytes consumed by one payload matching the schema.
  uint32_t argument_byte_length;
} loom_cmd_program_entry_schema_t;

// Bit set on command kinds that perform a full execution barrier before their
// payload. Every supported barrier kind has its own named enum value below.
#define LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_BIT (1u << 8)

// Kind of one command recorded by a portable command program.
typedef enum loom_cmd_program_command_kind_e {
  // Fill one buffer range with a repeated scalar pattern.
  LOOM_CMD_PROGRAM_COMMAND_KIND_FILL = 1,
  // Copy one buffer range into another.
  LOOM_CMD_PROGRAM_COMMAND_KIND_COPY = 2,
  // Dispatch with exact workgroup counts embedded in the program.
  LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT = 3,
  // Dispatch with indirect counts stable before command execution begins.
  LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC = 4,
  // Dispatch with indirect counts produced within the command program.
  LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC = 5,
  // Order all earlier commands before all later commands.
  LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION = 6,
  // Perform a full execution barrier, then fill one buffer range.
  LOOM_CMD_PROGRAM_COMMAND_KIND_FILL_BARRIER =
      LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_BIT |
      LOOM_CMD_PROGRAM_COMMAND_KIND_FILL,
  // Perform a full execution barrier, then copy one buffer range.
  LOOM_CMD_PROGRAM_COMMAND_KIND_COPY_BARRIER =
      LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_BIT |
      LOOM_CMD_PROGRAM_COMMAND_KIND_COPY,
  // Perform a full execution barrier, then dispatch with exact counts.
  LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT_BARRIER =
      LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_BIT |
      LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT,
  // Perform a full execution barrier, then dispatch with stable indirect
  // counts.
  LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC_BARRIER =
      LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_BIT |
      LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC,
  // Perform a full execution barrier, then dispatch with dynamically produced
  // indirect counts.
  LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC_BARRIER =
      LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_BIT |
      LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC,
} loom_cmd_program_command_kind_t;

// Returns true when |kind| joins all prior execution before its payload.
static inline bool loom_cmd_program_command_kind_has_barrier(
    loom_cmd_program_command_kind_t kind) {
  return ((uint32_t)kind & LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_BIT) != 0;
}

// Returns the payload command kind without its preceding barrier semantic.
static inline loom_cmd_program_command_kind_t
loom_cmd_program_command_kind_base(loom_cmd_program_command_kind_t kind) {
  const uint32_t base_kind =
      (uint32_t)kind & ~LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_BIT;
  return (loom_cmd_program_command_kind_t)base_kind;
}

// Returns true when |kind| opens a new full-execution barrier wave.
//
// This includes both standalone barriers and payload commands carrying a
// folded leading barrier.
static inline bool loom_cmd_program_command_kind_begins_barrier_wave(
    loom_cmd_program_command_kind_t kind) {
  return loom_cmd_program_command_kind_has_barrier(kind) ||
         loom_cmd_program_command_kind_base(kind) ==
             LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION;
}

// One decoded portable command.
typedef struct loom_cmd_program_command_t {
  // Command payload selector.
  loom_cmd_program_command_kind_t kind;
  // First byte in the flattened tagless argument payload storage.
  uint32_t argument_offset;
  // Executable-entry schema describing the tagless argument payload.
  uint32_t argument_schema_index;
  // Command-specific payload.
  union {
    // Payload for either fill command kind.
    struct {
      // Target buffer-reference table index.
      uint32_t target_buffer_ref;
      // Repeated fill pattern bits.
      uint32_t pattern;
      // Number of low pattern bytes to repeat.
      uint32_t pattern_length;
    } fill;
    // Payload for either copy command kind.
    struct {
      // Source buffer-reference table index.
      uint32_t source_buffer_ref;
      // Target buffer-reference table index.
      uint32_t target_buffer_ref;
    } copy;
    // Payload for either direct dispatch command kind.
    struct {
      // Dense executable requirement index.
      uint32_t executable_index;
      // Dense program entry requirement index.
      uint32_t entry_index;
      // Exact X workgroup count.
      uint32_t workgroup_count_x;
      // Exact Y workgroup count.
      uint32_t workgroup_count_y;
      // Exact Z workgroup count.
      uint32_t workgroup_count_z;
    } dispatch_direct;
    // Payload for any indirect dispatch command kind.
    struct {
      // Dense executable requirement index.
      uint32_t executable_index;
      // Dense program entry requirement index.
      uint32_t entry_index;
      // Buffer-reference table index containing XYZ workgroup counts.
      uint32_t workgroup_count_buffer_ref;
    } dispatch_indirect;
  } payload;
} loom_cmd_program_command_t;

// Aggregate issue-time storage required by command-program allocations.
typedef struct loom_cmd_program_transient_requirement_t {
  // Dense rebindable root index, or UINT32_MAX when no slab is required.
  uint32_t binding_index;
  // Minimum byte length of the supplied transient slab.
  uint64_t required_byte_length;
  // Minimum required alignment of the supplied transient slab.
  uint64_t minimum_alignment;
} loom_cmd_program_transient_requirement_t;

// Host-produced workgroup-count storage required by static indirect dispatches.
typedef struct loom_cmd_program_launch_count_requirement_t {
  // Dense rebindable root index, or UINT32_MAX when no table is required.
  uint32_t binding_index;
  // Minimum byte length of the supplied workgroup-count table.
  uint64_t required_byte_length;
  // Minimum required alignment of the supplied workgroup-count table.
  uint64_t minimum_alignment;
} loom_cmd_program_launch_count_requirement_t;

// External resource counts required to materialize one command program.
typedef struct loom_cmd_program_requirements_t {
  // Number of fixed buffer roots supplied during materialization.
  uint32_t fixed_buffer_count;
  // Number of rebindable buffer roots supplied during issue.
  uint32_t rebindable_binding_count;
  // Number of loaded executables supplied during materialization.
  uint32_t executable_count;
  // Number of program-local executable entry tokens.
  uint32_t entry_count;
  // Aggregate issue-time storage required by command-program allocations.
  loom_cmd_program_transient_requirement_t transient;
  // Host-produced workgroup-count storage for static indirect dispatches.
  loom_cmd_program_launch_count_requirement_t launch_counts;
} loom_cmd_program_requirements_t;

// Aggregate storage requirement for one fixed parameter-buffer root.
typedef struct loom_cmd_program_parameter_root_t {
  // Dense fixed-buffer table index populated by this root.
  uint32_t fixed_buffer_index;
  // Minimum byte length required by all parameters assigned to the root.
  uint64_t required_byte_length;
  // Minimum required alignment of the supplied fixed-buffer range.
  uint64_t minimum_alignment;
} loom_cmd_program_parameter_root_t;

// One concrete immutable parameter placed in a fixed buffer root.
typedef struct loom_cmd_program_parameter_t {
  // Fully substituted parameter key borrowed from program storage.
  iree_string_view_t key;
  // Dense fixed-buffer table index containing the parameter.
  uint32_t fixed_buffer_index;
  // Root-relative byte offset of the parameter payload.
  uint64_t byte_offset;
  // Exact byte length of the parameter payload.
  uint64_t byte_length;
  // Minimum required alignment of the placed parameter payload.
  uint64_t minimum_alignment;
} loom_cmd_program_parameter_t;

// Borrowed table within a parsed command-program artifact.
typedef struct loom_cmd_program_table_t {
  // First byte of the canonical fixed-size records.
  const uint8_t* data;
  // Number of records in the table.
  uint32_t count;
} loom_cmd_program_table_t;

// Validated zero-allocation view of one serialized command program.
//
// All storage is borrowed from |data| passed to loom_cmd_program_parse and
// must remain live while the view is used. Fields are read-only after parsing.
typedef struct loom_cmd_program_t {
  // Complete canonical artifact storage.
  iree_const_byte_span_t storage;
  // External resources required by the program.
  loom_cmd_program_requirements_t requirements;
  // Buffer ranges used by fills, copies, and indirect launch counts.
  loom_cmd_program_table_t buffer_refs;
  // Executable-entry logical argument schemas.
  loom_cmd_program_table_t entry_schemas;
  // Flattened one-byte logical argument kinds referenced by entry schemas.
  loom_cmd_program_table_t entry_schema_kinds;
  // Tagless dispatch argument payloads in command traversal order.
  iree_const_byte_span_t argument_data;
  // Ordered command table.
  loom_cmd_program_table_t commands;
  // Fixed parameter-buffer roots in canonical ascending root order.
  loom_cmd_program_table_t parameter_roots;
  // Concrete immutable parameter requirements.
  loom_cmd_program_table_t parameters;
  // Concatenated parameter-key bytes referenced by |parameters|.
  iree_const_byte_span_t parameter_keys;
} loom_cmd_program_t;

// Half-open range within the canonical command table.
typedef struct loom_cmd_program_command_range_t {
  // Zero-based first command ordinal.
  uint32_t first_command;
  // Number of commands in the range.
  uint32_t command_count;
} loom_cmd_program_command_range_t;

// One non-empty command range bounded by full execution barriers.
typedef struct loom_cmd_program_barrier_wave_t {
  // Canonical barrier-wave ordinal used by recorded operation metadata.
  uint32_t ordinal;
  // Contiguous canonical commands belonging to the wave.
  loom_cmd_program_command_range_t commands;
} loom_cmd_program_barrier_wave_t;

// Single-pass cursor over the barrier waves in one parsed program.
typedef struct loom_cmd_program_barrier_wave_iterator_t {
  // Parsed program borrowed for the complete iteration.
  const loom_cmd_program_t* program;
  // First canonical command not yet returned.
  uint32_t next_command;
  // Barrier-wave ordinal active before |next_command| is inspected.
  uint32_t barrier_wave_ordinal;
} loom_cmd_program_barrier_wave_iterator_t;

// Parses and validates one complete command-program artifact.
//
// This is the untrusted byte boundary. Successful parsing guarantees that all
// table ranges, enum values, indices, slices, and reserved fields satisfy the
// canonical format. |out_program| borrows |data| without allocating.
iree_status_t loom_cmd_program_parse(iree_const_byte_span_t data,
                                     loom_cmd_program_t* out_program);

// Binds a previously validated canonical command-program artifact.
//
// This is an infallible trusted-data operation for package parsers and other
// producers that have already established the complete byte-format contract.
// The returned view borrows |data| without allocating.
void loom_cmd_program_bind_verified(iree_const_byte_span_t data,
                                    loom_cmd_program_t* out_program);

// Returns one validated buffer-reference table entry.
loom_cmd_program_buffer_ref_t loom_cmd_program_buffer_ref_at(
    const loom_cmd_program_t* program, uint32_t index);

// Returns one validated executable-entry logical argument schema.
loom_cmd_program_entry_schema_t loom_cmd_program_entry_schema_at(
    const loom_cmd_program_t* program, uint32_t index);

// Returns one logical argument kind from a validated entry schema.
loom_cmd_program_argument_kind_t loom_cmd_program_entry_schema_kind_at(
    const loom_cmd_program_t* program,
    const loom_cmd_program_entry_schema_t* schema, uint32_t argument_index);

// Returns the validated tagless argument payload for |command|.
iree_const_byte_span_t loom_cmd_program_command_argument_data(
    const loom_cmd_program_t* program,
    const loom_cmd_program_command_t* command);

// Returns one validated command table entry.
loom_cmd_program_command_t loom_cmd_program_command_at(
    const loom_cmd_program_t* program, uint32_t index);

// Returns the complete canonical command range in |program|.
loom_cmd_program_command_range_t loom_cmd_program_command_range_all(
    const loom_cmd_program_t* program);

// Initializes |iterator| before a single forward traversal of |program|.
void loom_cmd_program_barrier_wave_iterator_initialize(
    const loom_cmd_program_t* program,
    loom_cmd_program_barrier_wave_iterator_t* iterator);

// Returns the next non-empty barrier wave and advances |iterator|.
//
// Returns false after every canonical command has been returned. Each command
// appears in exactly one range, and concatenating the returned ranges restores
// the original command traversal.
bool loom_cmd_program_barrier_wave_iterator_next(
    loom_cmd_program_barrier_wave_iterator_t* iterator,
    loom_cmd_program_barrier_wave_t* out_wave);

// Returns one validated fixed parameter-buffer root requirement.
loom_cmd_program_parameter_root_t loom_cmd_program_parameter_root_at(
    const loom_cmd_program_t* program, uint32_t index);

// Returns one validated concrete parameter requirement.
loom_cmd_program_parameter_t loom_cmd_program_parameter_at(
    const loom_cmd_program_t* program, uint32_t index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_PROGRAM_H_
