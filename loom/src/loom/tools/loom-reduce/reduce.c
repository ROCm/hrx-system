// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-reduce/reduce.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/error/source.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ops/op_defs.h"
#include "loom/util/json.h"

//===----------------------------------------------------------------------===//
// Text helpers
//===----------------------------------------------------------------------===//

static bool loom_reduce_is_name_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool loom_reduce_is_name_char(char c) {
  return loom_reduce_is_name_start(c) || (c >= '0' && c <= '9') || c == '.' ||
         c == '$';
}

static bool loom_reduce_is_value_char(char c) {
  return loom_reduce_is_name_char(c) || c == '#';
}

static bool loom_reduce_starts_with(iree_string_view_t text,
                                    const char* prefix) {
  return iree_string_view_starts_with(text, iree_make_cstring_view(prefix));
}

static iree_string_view_t loom_reduce_trim(iree_string_view_t text) {
  return iree_string_view_trim(text);
}

// Returns the length of the name starting at |text|[0].
static iree_host_size_t loom_reduce_name_length(iree_string_view_t text) {
  iree_host_size_t length = 0;
  while (length < text.size && loom_reduce_is_name_char(text.data[length])) {
    ++length;
  }
  return length;
}

// Returns true when |text| at |offset| spells |symbol| followed by a
// non-name character.
static bool loom_reduce_symbol_at(iree_string_view_t text,
                                  iree_host_size_t offset,
                                  iree_string_view_t symbol) {
  if (offset + symbol.size > text.size) {
    return false;
  }
  if (memcmp(text.data + offset, symbol.data, symbol.size) != 0) {
    return false;
  }
  const iree_host_size_t end = offset + symbol.size;
  return end == text.size || !loom_reduce_is_name_char(text.data[end]);
}

// Returns the next line of |*remaining| (without '\n') and advances it.
static iree_string_view_t loom_reduce_next_line(iree_string_view_t* remaining) {
  iree_host_size_t end = 0;
  while (end < remaining->size && remaining->data[end] != '\n') {
    ++end;
  }
  iree_string_view_t line = iree_make_string_view(remaining->data, end);
  if (end < remaining->size) {
    ++end;
  }
  *remaining = iree_string_view_substr(*remaining, end, IREE_HOST_SIZE_MAX);
  return line;
}

static uint64_t loom_reduce_fnv1a(iree_string_view_t text) {
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  for (iree_host_size_t i = 0; i < text.size; ++i) {
    hash = (hash ^ (uint8_t)text.data[i]) * UINT64_C(0x100000001b3);
  }
  return hash;
}

//===----------------------------------------------------------------------===//
// Plan state
//===----------------------------------------------------------------------===//

typedef enum loom_reduce_job_status_e {
  LOOM_REDUCE_JOB_PENDING = 0,
  LOOM_REDUCE_JOB_APPLIED,
  LOOM_REDUCE_JOB_SKIPPED,
  LOOM_REDUCE_JOB_REVERTED,
} loom_reduce_job_status_t;

typedef struct loom_reduce_source_t {
  // Path the module was loaded under; empty for a created motif file.
  iree_string_view_t load_path;
  // Root-relative path with '/' separators.
  iree_string_view_t relative_path;
  // Original text; empty for a created motif file.
  iree_string_view_t text;
  // Verified native module of |text|; NULL for a created motif file.
  const loom_module_t* module;
  // Printer for declarations taken from |module|, valid when |printer_ready|.
  loom_text_printer_t printer;
  bool printer_ready;
  // Byte offset of each line start.
  iree_host_size_t* line_starts;
  uint32_t line_count;
  // Text after the most recent render, valid when |rendered| is true.
  iree_string_builder_t output;
  bool rendered;
} loom_reduce_source_t;

typedef struct loom_reduce_copy_t {
  uint32_t source;
  iree_host_size_t start;
  iree_host_size_t end;
  uint32_t start_column;
  const loom_analyze_occurrence_t* occurrence;
  // Existing motifs: occurrence input index per motif parameter and output
  // index per motif result.
  const uint32_t* argument_order;
  const uint32_t* result_order;
} loom_reduce_copy_t;

typedef struct loom_reduce_job_t {
  bool existing;
  loom_reduce_job_status_t status;
  const char* reason;
  iree_string_view_t name;
  uint64_t fingerprint;
  uint32_t op_count;
  loom_reduce_copy_t* copies;
  uint32_t copy_count;
  // Source hosting a new motif, or UINT32_MAX for an existing motif.
  uint32_t motif_source;
  // Existing motifs: the source defining the called function after the
  // rewrite. When the definition moves there from |definition_source| (a
  // relocation), [definition_start, definition_end) is its original text.
  uint32_t host_source;
  uint32_t definition_source;
  bool relocate;
  iree_host_size_t definition_start;
  iree_host_size_t definition_end;
  // Existing motifs: `func.decl @name(...)` for files calling it elsewhere.
  iree_string_view_t declaration;
  // Existing motifs: parameter and result counts of the called definition.
  uint32_t parameter_count;
  uint32_t result_count;
  uint64_t removed_op_count;
} loom_reduce_job_t;

typedef struct loom_reduce_skip_t {
  uint64_t fingerprint;
  const char* reason;
} loom_reduce_skip_t;

struct loom_reduce_plan_t {
  iree_allocator_t allocator;
  loom_reduce_options_t options;
  iree_arena_block_pool_t block_pool;
  iree_arena_allocator_t arena;
  loom_reduce_source_t* sources;
  iree_host_size_t source_count;
  iree_host_size_t source_capacity;
  loom_reduce_job_t* jobs;
  iree_host_size_t job_count;
  iree_host_size_t job_capacity;
  loom_reduce_skip_t* skips;
  iree_host_size_t skip_count;
  iree_host_size_t skip_capacity;
  // Findings repeating an existing function that got a new motif instead,
  // with the reason reuse was impossible.
  loom_reduce_skip_t* fallbacks;
  iree_host_size_t fallback_count;
  iree_host_size_t fallback_capacity;
  bool applied;
};

static iree_status_t loom_reduce_arena_copy(loom_reduce_plan_t* plan,
                                            iree_string_view_t text,
                                            iree_string_view_t* out_copy) {
  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&plan->arena, text.size + 1, (void**)&data));
  if (text.size) {
    memcpy(data, text.data, text.size);
  }
  data[text.size] = 0;
  *out_copy = iree_make_string_view(data, text.size);
  return iree_ok_status();
}

static iree_status_t loom_reduce_arena_format(loom_reduce_plan_t* plan,
                                              iree_string_view_t* out_text,
                                              const char* format, ...) {
  va_list args;
  va_start(args, format);
  char buffer[512];
  const int length = vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  if (length < 0 || (size_t)length >= sizeof(buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "formatted text too long");
  }
  return loom_reduce_arena_copy(
      plan, iree_make_string_view(buffer, (iree_host_size_t)length), out_text);
}

static iree_status_t loom_reduce_index_lines(loom_reduce_plan_t* plan,
                                             loom_reduce_source_t* source) {
  uint32_t count = 1;
  for (iree_host_size_t i = 0; i < source->text.size; ++i) {
    if (source->text.data[i] == '\n') {
      ++count;
    }
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&plan->arena,
                                           count * sizeof(iree_host_size_t),
                                           (void**)&source->line_starts));
  source->line_starts[0] = 0;
  uint32_t line = 1;
  for (iree_host_size_t i = 0; i < source->text.size; ++i) {
    if (source->text.data[i] == '\n') {
      source->line_starts[line++] = i + 1;
    }
  }
  source->line_count = count;
  return iree_ok_status();
}

static iree_status_t loom_reduce_add_source(loom_reduce_plan_t* plan,
                                            iree_string_view_t load_path,
                                            iree_string_view_t relative_path,
                                            iree_string_view_t text,
                                            const loom_module_t* module,
                                            uint32_t* out_index) {
  if (plan->source_count == plan->source_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        plan->allocator, plan->source_count + 1, sizeof(loom_reduce_source_t),
        &plan->source_capacity, (void**)&plan->sources));
  }
  loom_reduce_source_t* source = &plan->sources[plan->source_count];
  memset(source, 0, sizeof(*source));
  source->load_path = load_path;
  source->relative_path = relative_path;
  source->text = text;
  source->module = module;
  iree_string_builder_initialize(plan->allocator, &source->output);
  IREE_RETURN_IF_ERROR(loom_reduce_index_lines(plan, source));
  *out_index = (uint32_t)plan->source_count++;
  return iree_ok_status();
}

static uint32_t loom_reduce_find_loaded(const loom_reduce_plan_t* plan,
                                        iree_string_view_t load_path) {
  for (iree_host_size_t i = 0; i < plan->source_count; ++i) {
    if (iree_string_view_equal(plan->sources[i].load_path, load_path)) {
      return (uint32_t)i;
    }
  }
  return UINT32_MAX;
}

static uint32_t loom_reduce_find_relative(const loom_reduce_plan_t* plan,
                                          iree_string_view_t relative_path) {
  for (iree_host_size_t i = 0; i < plan->source_count; ++i) {
    if (iree_string_view_equal(plan->sources[i].relative_path, relative_path)) {
      return (uint32_t)i;
    }
  }
  return UINT32_MAX;
}

// Converts a one-based line and Unicode code-point column to a byte offset.
static bool loom_reduce_offset(const loom_reduce_source_t* source,
                               uint32_t line, uint32_t column,
                               iree_host_size_t* out_offset) {
  if (line < 1 || line > source->line_count || column < 1) {
    return false;
  }
  iree_host_size_t offset = source->line_starts[line - 1];
  for (uint32_t i = 1; i < column; ++i) {
    if (offset >= source->text.size) {
      return false;
    }
    const uint8_t lead = (uint8_t)source->text.data[offset];
    offset += lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
  }
  *out_offset = offset > source->text.size ? source->text.size : offset;
  return true;
}

static iree_string_view_t loom_reduce_copy_body(
    const loom_reduce_plan_t* plan, const loom_reduce_copy_t* copy) {
  const loom_reduce_source_t* source = &plan->sources[copy->source];
  return iree_make_string_view(source->text.data + copy->start,
                               copy->end - copy->start);
}

static iree_status_t loom_reduce_skip(loom_reduce_plan_t* plan,
                                      uint64_t fingerprint,
                                      const char* reason) {
  if (plan->skip_count == plan->skip_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        plan->allocator, plan->skip_count + 1, sizeof(loom_reduce_skip_t),
        &plan->skip_capacity, (void**)&plan->skips));
  }
  plan->skips[plan->skip_count++] = (loom_reduce_skip_t){fingerprint, reason};
  return iree_ok_status();
}

static iree_status_t loom_reduce_new_job(loom_reduce_plan_t* plan,
                                         loom_reduce_job_t** out_job) {
  if (plan->job_count == plan->job_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        plan->allocator, plan->job_count + 1, sizeof(loom_reduce_job_t),
        &plan->job_capacity, (void**)&plan->jobs));
  }
  loom_reduce_job_t* job = &plan->jobs[plan->job_count++];
  memset(job, 0, sizeof(*job));
  job->motif_source = UINT32_MAX;
  job->host_source = UINT32_MAX;
  job->definition_source = UINT32_MAX;
  *out_job = job;
  return iree_ok_status();
}

static bool loom_reduce_init_copy(const loom_reduce_plan_t* plan,
                                  const loom_analyze_occurrence_t* occurrence,
                                  loom_reduce_copy_t* out_copy) {
  memset(out_copy, 0, sizeof(*out_copy));
  out_copy->source = loom_reduce_find_loaded(plan, occurrence->file);
  if (out_copy->source == UINT32_MAX) {
    return false;
  }
  const loom_reduce_source_t* source = &plan->sources[out_copy->source];
  out_copy->start_column = occurrence->start_column;
  out_copy->occurrence = occurrence;
  return loom_reduce_offset(source, occurrence->start_line,
                            occurrence->start_column, &out_copy->start) &&
         loom_reduce_offset(source, occurrence->end_line,
                            occurrence->end_column, &out_copy->end) &&
         out_copy->end > out_copy->start;
}

static uint64_t loom_reduce_removed_ops(const loom_reduce_job_t* job) {
  const uint64_t count = job->copy_count;
  const uint64_t ops = job->op_count;
  // An existing body already exists; each copy becomes one call.
  if (job->existing) {
    return count * (ops - 1);
  }
  // Each copy becomes one call; one motif body and its func.return are added.
  const uint64_t added = ops + 1 + count;
  return count * ops > added ? count * ops - added : 0;
}

//===----------------------------------------------------------------------===//
// Job construction
//===----------------------------------------------------------------------===//

// Returns the link group of |relative_path|: the file itself or its first
// directory.
static iree_string_view_t loom_reduce_group(const loom_reduce_plan_t* plan,
                                            iree_string_view_t relative_path) {
  if (plan->options.scope == LOOM_REDUCE_SCOPE_FILE) {
    return relative_path;
  }
  const iree_host_size_t slash =
      iree_string_view_find_char(relative_path, '/', 0);
  if (slash == IREE_STRING_VIEW_NPOS) {
    return iree_string_view_empty();
  }
  return iree_make_string_view(relative_path.data, slash);
}

// Returns the source of |group|'s shared motif file, adding an empty one when
// the root has none yet. `{group}` in the configured name becomes the group
// directory ("root" for files directly under the root), so motif files of
// different groups never share a name when their groups are combined later.
static iree_status_t loom_reduce_group_motif_source(loom_reduce_plan_t* plan,
                                                    iree_string_view_t group,
                                                    uint32_t* out_source) {
  iree_string_builder_t path;
  iree_string_builder_initialize(plan->allocator, &path);
  iree_status_t status = iree_ok_status();
  if (!iree_string_view_is_empty(group)) {
    status = iree_string_builder_append_format(&path, "%.*s/", (int)group.size,
                                               group.data);
  }
  const iree_string_view_t placeholder = IREE_SV("{group}");
  iree_string_view_t name = plan->options.motif_file_name;
  while (iree_status_is_ok(status) && !iree_string_view_is_empty(name)) {
    const iree_host_size_t at = iree_string_view_find_char(name, '{', 0);
    const bool is_placeholder =
        at != IREE_STRING_VIEW_NPOS &&
        iree_string_view_starts_with(
            iree_string_view_substr(name, at, IREE_HOST_SIZE_MAX), placeholder);
    const iree_host_size_t literal =
        at == IREE_STRING_VIEW_NPOS ? name.size : at + (is_placeholder ? 0 : 1);
    status = iree_string_builder_append_string(
        &path, iree_make_string_view(name.data, literal));
    name = iree_string_view_substr(name, literal, IREE_HOST_SIZE_MAX);
    if (iree_status_is_ok(status) && is_placeholder) {
      status = iree_string_builder_append_string(
          &path, iree_string_view_is_empty(group) ? IREE_SV("root") : group);
      name =
          iree_string_view_substr(name, placeholder.size, IREE_HOST_SIZE_MAX);
    }
  }
  iree_string_view_t relative = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = loom_reduce_arena_copy(plan, iree_string_builder_view(&path),
                                    &relative);
  }
  iree_string_builder_deinitialize(&path);
  IREE_RETURN_IF_ERROR(status);
  *out_source = loom_reduce_find_relative(plan, relative);
  if (*out_source == UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_reduce_add_source(
        plan, iree_string_view_empty(), relative, iree_string_view_empty(),
        /*module=*/NULL, out_source));
  }
  return iree_ok_status();
}

static iree_status_t loom_reduce_add_duplicate_jobs(
    loom_reduce_plan_t* plan, const loom_analyze_finding_t* finding) {
  // Group callable copies; each group with two or more copies is one job.
  bool* grouped = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &plan->arena, finding->occurrence_count * sizeof(bool),
      (void**)&grouped));
  memset(grouped, 0, finding->occurrence_count * sizeof(bool));
  for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
    const loom_analyze_occurrence_t* first = &finding->occurrences[i];
    if (grouped[i] || first->in_launch_config) {
      continue;
    }
    const uint32_t first_source = loom_reduce_find_loaded(plan, first->file);
    if (first_source == UINT32_MAX) {
      continue;
    }
    const iree_string_view_t group =
        loom_reduce_group(plan, plan->sources[first_source].relative_path);
    uint32_t count = 0;
    for (uint32_t j = i; j < finding->occurrence_count; ++j) {
      const loom_analyze_occurrence_t* other = &finding->occurrences[j];
      if (grouped[j] || other->in_launch_config) {
        continue;
      }
      const uint32_t source = loom_reduce_find_loaded(plan, other->file);
      if (source == UINT32_MAX ||
          !iree_string_view_equal(
              loom_reduce_group(plan, plan->sources[source].relative_path),
              group)) {
        continue;
      }
      ++count;
    }
    if (count < 2) {
      grouped[i] = true;
      continue;
    }

    loom_reduce_job_t* job = NULL;
    IREE_RETURN_IF_ERROR(loom_reduce_new_job(plan, &job));
    job->fingerprint = finding->fingerprint;
    job->op_count = finding->op_count;
    IREE_RETURN_IF_ERROR(loom_reduce_arena_format(
        plan, &job->name, "loom_reduce_%012" PRIx64 "_%06" PRIx64,
        finding->fingerprint >> 16, loom_reduce_fnv1a(group) & 0xFFFFFF));
    IREE_RETURN_IF_ERROR(iree_arena_allocate(&plan->arena,
                                             count * sizeof(loom_reduce_copy_t),
                                             (void**)&job->copies));
    for (uint32_t j = i; j < finding->occurrence_count; ++j) {
      const loom_analyze_occurrence_t* other = &finding->occurrences[j];
      if (grouped[j] || other->in_launch_config) {
        continue;
      }
      const uint32_t source = loom_reduce_find_loaded(plan, other->file);
      if (source == UINT32_MAX ||
          !iree_string_view_equal(
              loom_reduce_group(plan, plan->sources[source].relative_path),
              group)) {
        continue;
      }
      grouped[j] = true;
      loom_reduce_copy_t* copy = &job->copies[job->copy_count];
      if (!loom_reduce_init_copy(plan, other, copy)) {
        job->status = LOOM_REDUCE_JOB_SKIPPED;
        job->reason = "a copy's source range does not resolve";
        continue;
      }
      ++job->copy_count;
    }

    // The motif lives beside its copies or in the group's motif file.
    if (plan->options.scope == LOOM_REDUCE_SCOPE_FILE) {
      job->motif_source = job->copies[0].source;
    } else {
      IREE_RETURN_IF_ERROR(
          loom_reduce_group_motif_source(plan, group, &job->motif_source));
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Module facts
//===----------------------------------------------------------------------===//

static iree_string_view_t loom_reduce_symbol_name(const loom_module_t* module,
                                                  uint32_t symbol_id) {
  if (symbol_id >= module->symbols.count) {
    return iree_string_view_empty();
  }
  const loom_string_id_t name_id = module->symbols.entries[symbol_id].name_id;
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, name_id);
}

// Returns the operation defining |name| in |module|, or NULL.
static const loom_op_t* loom_reduce_symbol_op(const loom_module_t* module,
                                              iree_string_view_t name) {
  if (!module) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (iree_string_view_equal(loom_reduce_symbol_name(module, (uint32_t)i),
                               name)) {
      return module->symbols.entries[i].defining_op;
    }
  }
  return NULL;
}

// Returns the symbol |op| defines in |module|, or an empty view.
static iree_string_view_t loom_reduce_defined_symbol(
    const loom_module_t* module, const loom_op_t* op) {
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (module->symbols.entries[i].defining_op == op) {
      return loom_reduce_symbol_name(module, (uint32_t)i);
    }
  }
  return iree_string_view_empty();
}

// Calls |fn| with the name of every symbol |op|'s attributes reference.
static iree_status_t loom_reduce_for_each_symbol_ref(
    const loom_module_t* module, const loom_op_t* op,
    iree_status_t (*fn)(void* user_data, iree_string_view_t name),
    void* user_data) {
  const loom_attribute_t* attributes = loom_op_attrs(op);
  for (uint8_t a = 0; a < op->attribute_count; ++a) {
    const loom_attribute_t* attribute = &attributes[a];
    if (attribute->kind == LOOM_ATTR_SYMBOL) {
      if (loom_symbol_ref_is_valid(attribute->symbol)) {
        IREE_RETURN_IF_ERROR(
            fn(user_data,
               loom_reduce_symbol_name(module, attribute->symbol.symbol_id)));
      }
    } else if (attribute->kind == LOOM_ATTR_SYMBOL_ARRAY ||
               attribute->kind == LOOM_ATTR_SYMBOL_SET) {
      for (uint16_t i = 0; i < attribute->count; ++i) {
        if (loom_symbol_ref_is_valid(attribute->symbol_refs[i])) {
          IREE_RETURN_IF_ERROR(
              fn(user_data, loom_reduce_symbol_name(
                                module, attribute->symbol_refs[i].symbol_id)));
        }
      }
    }
  }
  return iree_ok_status();
}

// Prints |op| of |source|'s module with regions elided and comments removed,
// joined onto one line, into |builder|.
static iree_status_t loom_reduce_print_header(loom_reduce_source_t* source,
                                              const loom_op_t* op,
                                              iree_allocator_t allocator,
                                              iree_string_builder_t* builder) {
  if (!source->printer_ready) {
    loom_text_printer_initialize(source->module, &source->printer);
    source->printer_ready = true;
  }
  iree_string_builder_t printed;
  iree_string_builder_initialize(allocator, &printed);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&printed, &stream);
  loom_text_print_options_t options;
  memset(&options, 0, sizeof(options));
  options.flags = LOOM_TEXT_PRINT_SKIP_REGIONS;
  iree_status_t status = loom_text_printer_write_operation(&source->printer, op,
                                                           &stream, &options);
  iree_string_view_t remaining = iree_string_builder_view(&printed);
  bool first = true;
  while (iree_status_is_ok(status) && !iree_string_view_is_empty(remaining)) {
    const iree_string_view_t line =
        loom_reduce_trim(loom_reduce_next_line(&remaining));
    if (iree_string_view_is_empty(line) ||
        loom_reduce_starts_with(line, "//")) {
      continue;
    }
    status = iree_string_builder_append_format(
        builder, "%s%.*s", first ? "" : " ", (int)line.size, line.data);
    first = false;
  }
  iree_string_builder_deinitialize(&printed);
  return status;
}

#define LOOM_REDUCE_MAX_BINDINGS 128

static uint32_t loom_reduce_find_binding(const loom_analyze_binding_t* bindings,
                                         uint32_t count,
                                         iree_string_view_t name) {
  for (uint32_t i = 0; i < count; ++i) {
    if (iree_string_view_equal(bindings[i].name, name)) {
      return i;
    }
  }
  return UINT32_MAX;
}

static iree_status_t loom_reduce_find_declaration(
    loom_reduce_plan_t* plan, loom_reduce_source_t* source,
    iree_string_view_t symbol, iree_string_builder_t* builder, bool* out_found);

// Records why an existing function cannot be reused, then gives its copies a
// new motif instead.
static iree_status_t loom_reduce_fall_back(
    loom_reduce_plan_t* plan, const loom_analyze_finding_t* finding,
    const char* reason) {
  if (plan->fallback_count == plan->fallback_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        plan->allocator, plan->fallback_count + 1, sizeof(loom_reduce_skip_t),
        &plan->fallback_capacity, (void**)&plan->fallbacks));
  }
  plan->fallbacks[plan->fallback_count++] =
      (loom_reduce_skip_t){finding->fingerprint, reason};
  return loom_reduce_add_duplicate_jobs(plan, finding);
}

static iree_status_t loom_reduce_add_existing_job(
    loom_reduce_plan_t* plan, const loom_analyze_finding_t* finding) {
  const loom_analyze_occurrence_t* definition = NULL;
  for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
    if (finding->occurrences[i].is_motif_definition) {
      definition = &finding->occurrences[i];
    }
  }
  const uint32_t definition_source =
      definition ? loom_reduce_find_loaded(plan, definition->file) : UINT32_MAX;
  if (definition_source == UINT32_MAX) {
    return loom_reduce_fall_back(plan, finding,
                                 "the existing motif's source is unavailable");
  }
  const loom_reduce_source_t* source = &plan->sources[definition_source];

  // An existing function is reused rather than duplicated. Copies in its own
  // file call it in place. Copies elsewhere in its group call it in the
  // group's motif file: either it already lives there, or its definition
  // moves there unchanged, since linking its original file elsewhere could
  // bring conflicting definitions of other symbols with it. A new motif is
  // made only when reuse is impossible (template providers, multi-block
  // bodies, names defined more than once in the group, other groups).
  bool cross_file = false;
  const iree_string_view_t group =
      loom_reduce_group(plan, source->relative_path);
  for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
    const loom_analyze_occurrence_t* occurrence = &finding->occurrences[i];
    if (occurrence->is_motif_definition || occurrence->in_launch_config) {
      continue;
    }
    const uint32_t copy_source =
        loom_reduce_find_loaded(plan, occurrence->file);
    if (copy_source == definition_source) {
      continue;
    }
    if (copy_source == UINT32_MAX ||
        plan->options.scope == LOOM_REDUCE_SCOPE_FILE ||
        !iree_string_view_equal(
            loom_reduce_group(plan, plan->sources[copy_source].relative_path),
            group)) {
      return loom_reduce_fall_back(
          plan, finding,
          "copies lie outside the existing motif's group, or file scope cannot "
          "call into another file");
    }
    cross_file = true;
  }
  uint32_t host_source = definition_source;
  if (cross_file) {
    IREE_RETURN_IF_ERROR(
        loom_reduce_group_motif_source(plan, group, &host_source));
  }

  // The definition's entry block arguments are the parameters and its
  // func.return operands the results.
  const loom_op_t* definition_op =
      loom_reduce_symbol_op(source->module, finding->motif_symbol);
  if (!definition_op ||
      !iree_string_view_equal(loom_op_name(source->module, definition_op),
                              IREE_SV("func.def"))) {
    return loom_reduce_fall_back(plan, finding,
                                 "the existing motif is not a func.def; "
                                 "template providers are not called directly");
  }
  loom_region_t* body = loom_func_like_body(
      loom_func_like_cast(source->module, (loom_op_t*)definition_op));
  const loom_block_t* entry =
      body && body->block_count == 1 ? body->blocks[0] : NULL;
  const loom_op_t* terminator = entry ? entry->last_op : NULL;
  if (!terminator ||
      !iree_string_view_equal(loom_op_name(source->module, terminator),
                              IREE_SV("func.return"))) {
    return loom_reduce_fall_back(
        plan, finding,
        "the existing motif is not a single block ending in func.return");
  }
  if (entry->arg_count > LOOM_REDUCE_MAX_BINDINGS ||
      terminator->operand_count > LOOM_REDUCE_MAX_BINDINGS) {
    return loom_reduce_fall_back(plan, finding,
                                 "the existing motif has too many parameters");
  }
  iree_string_view_t parameters[LOOM_REDUCE_MAX_BINDINGS];
  const uint32_t parameter_count = entry->arg_count;
  for (uint32_t k = 0; k < parameter_count; ++k) {
    parameters[k] = loom_module_value_name(source->module, entry->arg_ids[k]);
  }
  iree_string_view_t returned[LOOM_REDUCE_MAX_BINDINGS];
  const uint32_t returned_count = terminator->operand_count;
  const loom_value_id_t* returned_values = loom_op_const_operands(terminator);
  for (uint32_t k = 0; k < returned_count; ++k) {
    returned[k] = loom_module_value_name(source->module, returned_values[k]);
  }

  uint32_t argument_order[LOOM_REDUCE_MAX_BINDINGS];
  for (uint32_t k = 0; k < parameter_count; ++k) {
    argument_order[k] = loom_reduce_find_binding(
        definition->inputs, definition->input_count, parameters[k]);
    if (argument_order[k] == UINT32_MAX) {
      return loom_reduce_fall_back(
          plan, finding,
          "the existing motif has a parameter the copies do not supply");
    }
  }
  uint32_t result_order[LOOM_REDUCE_MAX_BINDINGS];
  for (uint32_t k = 0; k < returned_count; ++k) {
    result_order[k] = loom_reduce_find_binding(
        definition->outputs, definition->output_count, returned[k]);
    if (result_order[k] == UINT32_MAX) {
      return loom_reduce_fall_back(
          plan, finding,
          "the existing motif returns different values than the copies use");
    }
  }
  if (returned_count != definition->output_count) {
    return loom_reduce_fall_back(
        plan, finding,
        "the existing motif returns different values than the copies use");
  }

  // Moving the definition needs its name defined only once in the group
  // and a source range covering exactly the func.def.
  const bool relocate = host_source != definition_source;
  iree_host_size_t definition_start = 0;
  iree_host_size_t definition_end = 0;
  if (relocate) {
    for (iree_host_size_t i = 0; i < plan->source_count; ++i) {
      const loom_reduce_source_t* other = &plan->sources[i];
      if (i == definition_source || !other->module ||
          !iree_string_view_equal(loom_reduce_group(plan, other->relative_path),
                                  group)) {
        continue;
      }
      const loom_op_t* op =
          loom_reduce_symbol_op(other->module, finding->motif_symbol);
      if (op && !iree_string_view_ends_with(loom_op_name(other->module, op),
                                            IREE_SV(".decl"))) {
        return loom_reduce_fall_back(
            plan, finding,
            "the existing motif's name is defined more than once in its group");
      }
    }
    loom_source_range_t range;
    const loom_source_resolver_t resolver = {NULL, NULL};
    if (!loom_source_resolve(resolver, source->module, definition_op->location,
                             &range) ||
        !loom_reduce_offset(source, range.start_line, range.start_column,
                            &definition_start) ||
        !loom_reduce_offset(source, range.end_line, range.end_column,
                            &definition_end) ||
        definition_end <= definition_start) {
      return loom_reduce_fall_back(
          plan, finding, "the existing motif's source range does not resolve");
    }
    const iree_string_view_t text = loom_reduce_trim(
        iree_make_string_view(source->text.data + definition_start,
                              definition_end - definition_start));
    if (!loom_reduce_starts_with(text, "func.def") ||
        !iree_string_view_ends_with(text, IREE_SV("}"))) {
      return loom_reduce_fall_back(
          plan, finding,
          "the existing motif's source range is not exactly its func.def");
    }
  }
  iree_string_builder_t declaration;
  iree_string_builder_initialize(plan->allocator, &declaration);
  bool declared = false;
  iree_status_t status = loom_reduce_find_declaration(
      plan, &plan->sources[definition_source], finding->motif_symbol,
      &declaration, &declared);
  iree_string_view_t declaration_text = iree_string_view_empty();
  if (iree_status_is_ok(status) && declared) {
    iree_string_view_t text = iree_string_builder_view(&declaration);
    while (text.size && text.data[text.size - 1] == '\n') {
      --text.size;
    }
    status = loom_reduce_arena_copy(plan, text, &declaration_text);
  }
  iree_string_builder_deinitialize(&declaration);
  IREE_RETURN_IF_ERROR(status);
  if (!declared) {
    return loom_reduce_fall_back(
        plan, finding, "the existing motif has no declarable signature");
  }

  uint32_t count = 0;
  for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
    const loom_analyze_occurrence_t* occurrence = &finding->occurrences[i];
    if (!occurrence->is_motif_definition && !occurrence->in_launch_config) {
      ++count;
    }
  }
  if (count == 0) {
    return loom_reduce_fall_back(plan, finding,
                                 "the existing motif has no callable copies");
  }

  loom_reduce_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(loom_reduce_new_job(plan, &job));
  job->existing = true;
  job->fingerprint = finding->fingerprint;
  job->op_count = finding->op_count;
  job->parameter_count = parameter_count;
  job->result_count = returned_count;
  job->host_source = host_source;
  job->definition_source = definition_source;
  job->relocate = relocate;
  job->definition_start = definition_start;
  job->definition_end = definition_end;
  job->declaration = declaration_text;
  IREE_RETURN_IF_ERROR(
      loom_reduce_arena_copy(plan, finding->motif_symbol, &job->name));
  uint32_t* arguments = NULL;
  uint32_t* results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &plan->arena, (parameter_count + 1) * sizeof(uint32_t),
      (void**)&arguments));
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &plan->arena, (returned_count + 1) * sizeof(uint32_t), (void**)&results));
  memcpy(arguments, argument_order, parameter_count * sizeof(uint32_t));
  memcpy(results, result_order, returned_count * sizeof(uint32_t));
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &plan->arena, count * sizeof(loom_reduce_copy_t), (void**)&job->copies));
  for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
    const loom_analyze_occurrence_t* occurrence = &finding->occurrences[i];
    if (occurrence->is_motif_definition || occurrence->in_launch_config) {
      continue;
    }
    loom_reduce_copy_t* copy = &job->copies[job->copy_count];
    if (!loom_reduce_init_copy(plan, occurrence, copy)) {
      continue;
    }
    copy->argument_order = arguments;
    copy->result_order = results;
    ++job->copy_count;
  }
  if (job->copy_count == 0) {
    job->status = LOOM_REDUCE_JOB_SKIPPED;
    job->reason = "a copy's source range does not resolve";
  }
  return iree_ok_status();
}

iree_status_t loom_reduce_plan_create(const loom_reduce_options_t* options,
                                      const loom_reduce_input_t* inputs,
                                      iree_host_size_t input_count,
                                      const loom_analyze_result_t* result,
                                      iree_allocator_t allocator,
                                      loom_reduce_plan_t** out_plan) {
  *out_plan = NULL;
  loom_reduce_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*plan), (void**)&plan));
  plan->allocator = allocator;
  plan->options = *options;
  iree_arena_block_pool_initialize(64 * 1024, allocator, &plan->block_pool);
  iree_arena_initialize(&plan->block_pool, &plan->arena);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
       ++i) {
    uint32_t index = 0;
    status = loom_reduce_add_source(plan, inputs[i].load_path,
                                    inputs[i].relative_path, inputs[i].text,
                                    inputs[i].module, &index);
  }
  for (iree_host_size_t f = 0;
       f < result->finding_count && iree_status_is_ok(status); ++f) {
    const loom_analyze_finding_t* finding = &result->findings[f];
    if (!finding->selected) {
      continue;
    }
    if (finding->parameterized_by != LOOM_ANALYZE_PARAMETERIZE_NONE) {
      status = loom_reduce_skip(plan, finding->fingerprint,
                                "copies differ in constants or types");
    } else if (finding->rule == LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF) {
      status = loom_reduce_add_existing_job(plan, finding);
    } else {
      status = loom_reduce_add_duplicate_jobs(plan, finding);
    }
  }
  for (iree_host_size_t j = 0; j < plan->job_count; ++j) {
    plan->jobs[j].removed_op_count = loom_reduce_removed_ops(&plan->jobs[j]);
  }
  if (!iree_status_is_ok(status)) {
    loom_reduce_plan_free(plan);
    return status;
  }
  *out_plan = plan;
  return iree_ok_status();
}

void loom_reduce_plan_free(loom_reduce_plan_t* plan) {
  if (!plan) {
    return;
  }
  for (iree_host_size_t i = 0; i < plan->source_count; ++i) {
    iree_string_builder_deinitialize(&plan->sources[i].output);
    if (plan->sources[i].printer_ready) {
      loom_text_printer_deinitialize(&plan->sources[i].printer);
    }
  }
  iree_allocator_free(plan->allocator, plan->sources);
  iree_allocator_free(plan->allocator, plan->jobs);
  iree_allocator_free(plan->allocator, plan->skips);
  iree_allocator_free(plan->allocator, plan->fallbacks);
  iree_arena_deinitialize(&plan->arena);
  iree_arena_block_pool_deinitialize(&plan->block_pool);
  iree_allocator_free(plan->allocator, plan);
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

// A small list of unique strings.
typedef struct loom_reduce_strings_t {
  iree_string_view_t* items;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_allocator_t allocator;
} loom_reduce_strings_t;

static void loom_reduce_strings_initialize(iree_allocator_t allocator,
                                           loom_reduce_strings_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  out_list->allocator = allocator;
}

static void loom_reduce_strings_deinitialize(loom_reduce_strings_t* list) {
  iree_allocator_free(list->allocator, list->items);
  memset(list, 0, sizeof(*list));
}

static bool loom_reduce_strings_contains(const loom_reduce_strings_t* list,
                                         iree_string_view_t value) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    if (iree_string_view_equal(list->items[i], value)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_reduce_strings_add(loom_reduce_strings_t* list,
                                             iree_string_view_t value) {
  if (loom_reduce_strings_contains(list, value)) {
    return iree_ok_status();
  }
  if (list->count == list->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        list->allocator, list->count + 1, sizeof(iree_string_view_t),
        &list->capacity, (void**)&list->items));
  }
  list->items[list->count++] = value;
  return iree_ok_status();
}

static int loom_reduce_compare_views(const void* lhs, const void* rhs) {
  return iree_string_view_compare(*(const iree_string_view_t*)lhs,
                                  *(const iree_string_view_t*)rhs);
}

// Collects the unique names following |sigil| in |text|.
static iree_status_t loom_reduce_collect_references(
    iree_string_view_t text, char sigil, loom_reduce_strings_t* out_names) {
  for (iree_host_size_t i = 0; i + 1 < text.size; ++i) {
    if (text.data[i] != sigil || !loom_reduce_is_name_start(text.data[i + 1])) {
      continue;
    }
    const iree_string_view_t name = iree_make_string_view(
        text.data + i + 1, loom_reduce_name_length(iree_string_view_substr(
                               text, i + 1, IREE_HOST_SIZE_MAX)));
    IREE_RETURN_IF_ERROR(loom_reduce_strings_add(out_names, name));
    i += name.size;
  }
  if (out_names->count > 1) {
    qsort(out_names->items, out_names->count, sizeof(iree_string_view_t),
          loom_reduce_compare_views);
  }
  return iree_ok_status();
}

// Appends a declaration of |symbol| taken from |source|'s module: declaration
// ops print as they are, a func.def header becomes a func.decl, a target
// becomes a target.decl, and config declarations keep only their type. Their
// `where` constraints stay with their source rather than being imposed on
// every kernel linking a shared motif file.
static iree_status_t loom_reduce_find_declaration(
    loom_reduce_plan_t* plan, loom_reduce_source_t* source,
    iree_string_view_t symbol, iree_string_builder_t* builder,
    bool* out_found) {
  *out_found = false;
  const loom_op_t* op = loom_reduce_symbol_op(source->module, symbol);
  if (!op) {
    return iree_ok_status();
  }
  const iree_string_view_t name = loom_op_name(source->module, op);
  iree_string_builder_t header;
  iree_string_builder_initialize(plan->allocator, &header);
  iree_status_t status =
      loom_reduce_print_header(source, op, plan->allocator, &header);
  const iree_string_view_t text = iree_string_builder_view(&header);
  const iree_host_size_t at = iree_string_view_find_char(text, '@', 0);
  if (!iree_status_is_ok(status) || at == IREE_STRING_VIEW_NPOS) {
    // Not expressible as a declaration.
  } else if (iree_string_view_equal(name, IREE_SV("config.decl")) ||
             iree_string_view_equal(name, IREE_SV("config.def"))) {
    // decl: `@x : [%value :] TYPE [where ...]`; def: `@x = VALUE : TYPE`.
    iree_host_size_t colon = IREE_STRING_VIEW_NPOS;
    if (iree_string_view_equal(name, IREE_SV("config.decl"))) {
      colon = iree_string_view_find_char(text, ':', at);
    } else {
      for (iree_host_size_t i = text.size; i-- > at;) {
        if (text.data[i] == ':') {
          colon = i;
          break;
        }
      }
    }
    iree_string_view_t type = colon == IREE_STRING_VIEW_NPOS
                                  ? iree_string_view_empty()
                                  : loom_reduce_trim(iree_string_view_substr(
                                        text, colon + 1, IREE_HOST_SIZE_MAX));
    if (iree_string_view_consume_prefix(&type, IREE_SV("%"))) {
      type = loom_reduce_trim(iree_string_view_substr(
          type, loom_reduce_name_length(type), IREE_HOST_SIZE_MAX));
      iree_string_view_consume_prefix(&type, IREE_SV(":"));
      type = loom_reduce_trim(type);
    }
    iree_host_size_t end = 0;
    while (end < type.size && type.data[end] != ' ') {
      ++end;
    }
    if (end) {
      *out_found = true;
      status = iree_string_builder_append_format(
          builder, "config.decl @%.*s : %.*s\n", (int)symbol.size, symbol.data,
          (int)end, type.data);
    }
  } else if (iree_string_view_ends_with(name, IREE_SV(".decl"))) {
    *out_found = true;
    status = iree_string_builder_append_format(builder, "%.*s\n",
                                               (int)text.size, text.data);
  } else if (iree_string_view_equal(name, IREE_SV("func.def"))) {
    // `func.def [attrs] @x(sig) -> (results) {...}` -> `func.decl @x(...)...`.
    iree_host_size_t brace = IREE_STRING_VIEW_NPOS;
    for (iree_host_size_t i = text.size; i-- > at;) {
      if (text.data[i] == '{') {
        brace = i;
        break;
      }
    }
    if (brace != IREE_STRING_VIEW_NPOS) {
      const iree_string_view_t signature =
          loom_reduce_trim(iree_make_string_view(text.data + at, brace - at));
      *out_found = true;
      status = iree_string_builder_append_format(
          builder, "func.decl %.*s\n", (int)signature.size, signature.data);
    }
  } else if (iree_string_view_ends_with(name, IREE_SV(".target"))) {
    *out_found = true;
    status = iree_string_builder_append_format(builder, "target.decl @%.*s\n",
                                               (int)symbol.size, symbol.data);
  }
  iree_string_builder_deinitialize(&header);
  return status;
}

// Appends the `#alias = ...` line defining |alias| in |source|, if any.
static iree_status_t loom_reduce_find_alias(const loom_reduce_source_t* source,
                                            iree_string_view_t alias,
                                            iree_string_builder_t* builder) {
  iree_string_view_t remaining = source->text;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line =
        loom_reduce_trim(loom_reduce_next_line(&remaining));
    if (!iree_string_view_consume_prefix(&line, IREE_SV("#")) ||
        !loom_reduce_symbol_at(line, 0, alias)) {
      continue;
    }
    const iree_string_view_t rest = loom_reduce_trim(
        iree_string_view_substr(line, alias.size, IREE_HOST_SIZE_MAX));
    if (!loom_reduce_starts_with(rest, "=")) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "#%.*s\n", (int)line.size, line.data));
    return iree_ok_status();
  }
  return iree_ok_status();
}

// Appends the declarations and aliases |body| references, taken from
// |source|, skipping |exclude| (the function the body defines, if any). Sets
// |*out_complete| false when a referenced symbol has no declaration there.
static iree_status_t loom_reduce_body_declarations(
    loom_reduce_plan_t* plan, loom_reduce_source_t* source,
    iree_string_view_t body, iree_string_view_t exclude,
    iree_string_builder_t* aliases, iree_string_builder_t* declarations,
    bool* out_complete) {
  *out_complete = true;
  loom_reduce_strings_t names;
  loom_reduce_strings_initialize(plan->allocator, &names);
  iree_status_t status = loom_reduce_collect_references(body, '@', &names);
  for (iree_host_size_t i = 0; i < names.count && iree_status_is_ok(status);
       ++i) {
    if (iree_string_view_equal(names.items[i], exclude)) {
      continue;
    }
    bool found = false;
    status = loom_reduce_find_declaration(plan, source, names.items[i],
                                          declarations, &found);
    if (!found) {
      *out_complete = false;
    }
  }
  names.count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_reduce_collect_references(body, '#', &names);
  }
  for (iree_host_size_t i = 0; i < names.count && iree_status_is_ok(status);
       ++i) {
    status = loom_reduce_find_alias(source, names.items[i], aliases);
  }
  loom_reduce_strings_deinitialize(&names);
  return status;
}

// Appends the declarations and aliases a new motif's body references, taken
// from the source of its first copy.
static iree_status_t loom_reduce_required_declarations(
    loom_reduce_plan_t* plan, const loom_reduce_job_t* job,
    iree_string_builder_t* aliases, iree_string_builder_t* declarations,
    bool* out_complete) {
  const loom_reduce_copy_t* template = &job->copies[0];
  return loom_reduce_body_declarations(plan, &plan->sources[template->source],
                                       loom_reduce_copy_body(plan, template),
                                       iree_string_view_empty(), aliases,
                                       declarations, out_complete);
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

// Returns true when |line| (indentation removed) starts with a terminator op.
static bool loom_reduce_is_terminator(iree_string_view_t line) {
  line = loom_reduce_trim(line);
  if (loom_reduce_starts_with(line, "%")) {
    const iree_host_size_t equals = iree_string_view_find_char(line, '=', 0);
    if (equals == IREE_STRING_VIEW_NPOS) {
      return false;
    }
    line = loom_reduce_trim(
        iree_string_view_substr(line, equals + 1, IREE_HOST_SIZE_MAX));
  }
  iree_host_size_t end = 0;
  while (end < line.size && line.data[end] != ' ' && line.data[end] != '<' &&
         line.data[end] != '(') {
    ++end;
  }
  const iree_string_view_t op = iree_make_string_view(line.data, end);
  const iree_host_size_t dot = iree_string_view_find_char(op, '.', 0);
  if (dot == IREE_STRING_VIEW_NPOS) {
    return false;
  }
  const iree_string_view_t name =
      iree_string_view_substr(op, dot + 1, IREE_HOST_SIZE_MAX);
  static const char* kTerminators[] = {"return", "yield",   "launch.config",
                                       "br",     "cond_br", "exit"};
  for (size_t i = 0; i < IREE_ARRAYSIZE(kTerminators); ++i) {
    if (iree_string_view_equal(name, iree_make_cstring_view(kTerminators[i]))) {
      return true;
    }
  }
  return false;
}

// Returns true when every `%name` in |type| names an entry of |names|.
static bool loom_reduce_type_references_known(
    iree_string_view_t type, const loom_analyze_binding_t* names,
    uint32_t name_count) {
  for (iree_host_size_t i = 0; i + 1 < type.size; ++i) {
    if (type.data[i] != '%') {
      continue;
    }
    iree_host_size_t end = i + 1;
    while (end < type.size && loom_reduce_is_value_char(type.data[end])) {
      ++end;
    }
    const iree_string_view_t name =
        iree_make_string_view(type.data + i + 1, end - i - 1);
    if (loom_reduce_find_binding(names, name_count, name) == UINT32_MAX) {
      return false;
    }
    i = end - 1;
  }
  return true;
}

static iree_status_t loom_reduce_validate(loom_reduce_plan_t* plan,
                                          loom_reduce_job_t* job,
                                          const char** out_reason) {
  *out_reason = NULL;
  if (job->removed_op_count == 0) {
    // Short sequences with few copies cost more as calls than they save.
    *out_reason = "the motif would not remove operations";
    return iree_ok_status();
  }
  const loom_reduce_copy_t* template = &job->copies[0];
  const iree_string_view_t body = loom_reduce_copy_body(plan, template);
  if (!job->existing) {
    for (iree_host_size_t i = 0; i + 13 <= body.size; ++i) {
      if (memcmp(body.data + i, "buffer.alloca", 13) == 0) {
        *out_reason =
            "moving buffer.alloca into a motif would allocate once per call";
        return iree_ok_status();
      }
    }
  }
  // Only the last top-level operation can be a terminator; check every
  // top-level line, as loom-analyze excludes block terminators already.
  const uint32_t base_indent = template->start_column - 1;
  iree_string_view_t remaining = body;
  bool first = true;
  while (!iree_string_view_is_empty(remaining)) {
    const iree_string_view_t line = loom_reduce_next_line(&remaining);
    iree_host_size_t indent = 0;
    while (indent < line.size && line.data[indent] == ' ') {
      ++indent;
    }
    if ((first || indent == base_indent) && loom_reduce_is_terminator(line)) {
      *out_reason = "sequence contains a block terminator";
      return iree_ok_status();
    }
    first = false;
  }
  for (uint32_t c = 0; c < job->copy_count; ++c) {
    const loom_analyze_occurrence_t* occurrence = job->copies[c].occurrence;
    if (occurrence->input_count != template->occurrence->input_count ||
        occurrence->output_count != template->occurrence->output_count) {
      *out_reason = "copies disagree on their boundary values";
      return iree_ok_status();
    }
    for (uint32_t i = 0; i < occurrence->input_count; ++i) {
      if (iree_string_view_is_empty(occurrence->inputs[i].name)) {
        *out_reason = "a boundary value has no source name";
        return iree_ok_status();
      }
    }
    for (uint32_t i = 0; i < occurrence->output_count; ++i) {
      if (iree_string_view_is_empty(occurrence->outputs[i].name)) {
        *out_reason = "a boundary value has no source name";
        return iree_ok_status();
      }
    }
  }
  if (job->existing) {
    return iree_ok_status();
  }

  // A parameter type may only mention earlier parameters; result types may
  // mention parameters.
  const loom_analyze_occurrence_t* occurrence = template->occurrence;
  for (uint32_t i = 0; i < occurrence->input_count; ++i) {
    if (!loom_reduce_type_references_known(occurrence->inputs[i].type,
                                           occurrence->inputs, i)) {
      *out_reason =
          "a parameter type depends on a value that is not an earlier "
          "parameter";
      return iree_ok_status();
    }
  }
  for (uint32_t i = 0; i < occurrence->output_count; ++i) {
    if (!loom_reduce_type_references_known(occurrence->outputs[i].type,
                                           occurrence->inputs,
                                           occurrence->input_count)) {
      *out_reason =
          "a result type depends on a value defined inside the sequence";
      return iree_ok_status();
    }
  }
  if (job->motif_source != template->source) {
    iree_string_builder_t aliases, declarations;
    iree_string_builder_initialize(plan->allocator, &aliases);
    iree_string_builder_initialize(plan->allocator, &declarations);
    bool complete = false;
    iree_status_t status = loom_reduce_required_declarations(
        plan, job, &aliases, &declarations, &complete);
    iree_string_builder_deinitialize(&aliases);
    iree_string_builder_deinitialize(&declarations);
    IREE_RETURN_IF_ERROR(status);
    if (!complete) {
      *out_reason =
          "a symbol the motif references has no declaration in its source";
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Call cycles
//===----------------------------------------------------------------------===//

typedef struct loom_reduce_node_t {
  iree_string_view_t name;
  uint32_t* edges;
  uint32_t edge_count;
  iree_host_size_t edge_capacity;
} loom_reduce_node_t;

typedef struct loom_reduce_graph_t {
  iree_allocator_t allocator;
  loom_reduce_node_t* nodes;
  iree_host_size_t node_count;
  iree_host_size_t node_capacity;
  // Open-addressed name index storing node index + 1.
  uint32_t* slots;
  iree_host_size_t slot_count;
  uint32_t* visited;
  uint32_t visit_stamp;
} loom_reduce_graph_t;

static void loom_reduce_graph_deinitialize(loom_reduce_graph_t* graph) {
  for (iree_host_size_t i = 0; i < graph->node_count; ++i) {
    iree_allocator_free(graph->allocator, graph->nodes[i].edges);
  }
  iree_allocator_free(graph->allocator, graph->nodes);
  iree_allocator_free(graph->allocator, graph->slots);
  iree_allocator_free(graph->allocator, graph->visited);
}

static iree_status_t loom_reduce_graph_rehash(loom_reduce_graph_t* graph,
                                              iree_host_size_t slot_count) {
  uint32_t* slots = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      graph->allocator, slot_count, sizeof(uint32_t), (void**)&slots));
  for (iree_host_size_t i = 0; i < graph->node_count; ++i) {
    iree_host_size_t slot =
        loom_reduce_fnv1a(graph->nodes[i].name) & (slot_count - 1);
    while (slots[slot]) {
      slot = (slot + 1) & (slot_count - 1);
    }
    slots[slot] = (uint32_t)i + 1;
  }
  iree_allocator_free(graph->allocator, graph->slots);
  graph->slots = slots;
  graph->slot_count = slot_count;
  return iree_ok_status();
}

static iree_status_t loom_reduce_graph_intern(loom_reduce_graph_t* graph,
                                              iree_string_view_t name,
                                              uint32_t* out_node) {
  if ((graph->node_count + 1) * 2 > graph->slot_count) {
    IREE_RETURN_IF_ERROR(loom_reduce_graph_rehash(
        graph, graph->slot_count ? graph->slot_count * 2 : 1024));
  }
  iree_host_size_t slot = loom_reduce_fnv1a(name) & (graph->slot_count - 1);
  while (graph->slots[slot]) {
    const uint32_t node = graph->slots[slot] - 1;
    if (iree_string_view_equal(graph->nodes[node].name, name)) {
      *out_node = node;
      return iree_ok_status();
    }
    slot = (slot + 1) & (graph->slot_count - 1);
  }
  if (graph->node_count == graph->node_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        graph->allocator, graph->node_count + 1, sizeof(loom_reduce_node_t),
        &graph->node_capacity, (void**)&graph->nodes));
  }
  loom_reduce_node_t* node = &graph->nodes[graph->node_count];
  memset(node, 0, sizeof(*node));
  node->name = name;
  graph->slots[slot] = (uint32_t)graph->node_count + 1;
  *out_node = (uint32_t)graph->node_count++;
  return iree_ok_status();
}

static iree_status_t loom_reduce_graph_add_edge(loom_reduce_graph_t* graph,
                                                uint32_t from, uint32_t to) {
  loom_reduce_node_t* node = &graph->nodes[from];
  if (node->edge_count == node->edge_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        graph->allocator, node->edge_count + 1, sizeof(uint32_t),
        &node->edge_capacity, (void**)&node->edges));
  }
  node->edges[node->edge_count++] = to;
  return iree_ok_status();
}

typedef struct loom_reduce_edge_context_t {
  loom_reduce_graph_t* graph;
  uint32_t node;
  // True: |node| references the symbol (a call). False: the symbol reaches
  // |node|, as a template family reaches its providers.
  bool outgoing;
} loom_reduce_edge_context_t;

static iree_status_t loom_reduce_add_symbol_edge(void* user_data,
                                                 iree_string_view_t name) {
  loom_reduce_edge_context_t* context = (loom_reduce_edge_context_t*)user_data;
  if (iree_string_view_is_empty(name)) {
    return iree_ok_status();
  }
  uint32_t other = 0;
  IREE_RETURN_IF_ERROR(loom_reduce_graph_intern(context->graph, name, &other));
  return context->outgoing
             ? loom_reduce_graph_add_edge(context->graph, context->node, other)
             : loom_reduce_graph_add_edge(context->graph, other, context->node);
}

static iree_status_t loom_reduce_graph_add_regions(
    const loom_module_t* module, const loom_op_t* op,
    loom_reduce_edge_context_t* context) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t r = 0; r < op->region_count; ++r) {
    if (!regions[r]) {
      continue;
    }
    for (uint16_t b = 0; b < regions[r]->block_count; ++b) {
      const loom_block_t* block = regions[r]->blocks[b];
      for (const loom_op_t* child = block ? block->first_op : NULL; child;
           child = child->next_op) {
        IREE_RETURN_IF_ERROR(loom_reduce_for_each_symbol_ref(
            module, child, loom_reduce_add_symbol_edge, context));
        IREE_RETURN_IF_ERROR(
            loom_reduce_graph_add_regions(module, child, context));
      }
    }
  }
  return iree_ok_status();
}

// Adds edges from every top-level definition of |module| to the symbols its
// body references, and from the symbols its own attributes name (template
// family, target) to the definition.
static iree_status_t loom_reduce_graph_add_module(loom_reduce_graph_t* graph,
                                                  const loom_module_t* module) {
  const loom_region_t* body = module->body;
  for (uint16_t b = 0; body && b < body->block_count; ++b) {
    const loom_block_t* block = body->blocks[b];
    for (const loom_op_t* op = block ? block->first_op : NULL; op;
         op = op->next_op) {
      const iree_string_view_t name = loom_reduce_defined_symbol(module, op);
      if (iree_string_view_is_empty(name)) {
        continue;
      }
      loom_reduce_edge_context_t context = {graph, 0, false};
      IREE_RETURN_IF_ERROR(
          loom_reduce_graph_intern(graph, name, &context.node));
      IREE_RETURN_IF_ERROR(loom_reduce_for_each_symbol_ref(
          module, op, loom_reduce_add_symbol_edge, &context));
      context.outgoing = true;
      IREE_RETURN_IF_ERROR(loom_reduce_graph_add_regions(module, op, &context));
    }
  }
  return iree_ok_status();
}

// Returns true when |target| is reachable from any node in |starts|.
static iree_status_t loom_reduce_graph_reaches(loom_reduce_graph_t* graph,
                                               const uint32_t* starts,
                                               iree_host_size_t start_count,
                                               uint32_t target,
                                               bool* out_reaches) {
  *out_reaches = false;
  if (graph->visited == NULL || graph->node_count > graph->node_capacity) {
    return iree_make_status(IREE_STATUS_INTERNAL, "graph not prepared");
  }
  if (++graph->visit_stamp == 0) {
    memset(graph->visited, 0, graph->node_capacity * sizeof(uint32_t));
    graph->visit_stamp = 1;
  }
  uint32_t* stack = NULL;
  iree_host_size_t stack_count = 0;
  iree_host_size_t stack_capacity = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < start_count && iree_status_is_ok(status);
       ++i) {
    if (stack_count == stack_capacity) {
      status = iree_allocator_grow_array(graph->allocator, stack_count + 1,
                                         sizeof(uint32_t), &stack_capacity,
                                         (void**)&stack);
    }
    if (iree_status_is_ok(status)) {
      stack[stack_count++] = starts[i];
    }
  }
  while (iree_status_is_ok(status) && stack_count > 0 && !*out_reaches) {
    const uint32_t node = stack[--stack_count];
    if (node == target) {
      *out_reaches = true;
      break;
    }
    if (graph->visited[node] == graph->visit_stamp) {
      continue;
    }
    graph->visited[node] = graph->visit_stamp;
    const loom_reduce_node_t* entry = &graph->nodes[node];
    for (uint32_t e = 0; e < entry->edge_count && iree_status_is_ok(status);
         ++e) {
      if (stack_count == stack_capacity) {
        status = iree_allocator_grow_array(graph->allocator, stack_count + 1,
                                           sizeof(uint32_t), &stack_capacity,
                                           (void**)&stack);
      }
      if (iree_status_is_ok(status)) {
        stack[stack_count++] = entry->edges[e];
      }
    }
  }
  iree_allocator_free(graph->allocator, stack);
  return status;
}

static iree_status_t loom_reduce_graph_prepare_visits(
    loom_reduce_graph_t* graph) {
  iree_allocator_free(graph->allocator, graph->visited);
  graph->visited = NULL;
  graph->visit_stamp = 0;
  return iree_allocator_malloc_array(graph->allocator, graph->node_capacity,
                                     sizeof(uint32_t), (void**)&graph->visited);
}

static const loom_reduce_job_t* loom_reduce_sort_jobs;

static int loom_reduce_compare_savings(const void* lhs, const void* rhs) {
  const uint32_t a = *(const uint32_t*)lhs;
  const uint32_t b = *(const uint32_t*)rhs;
  const uint64_t a_ops = loom_reduce_sort_jobs[a].removed_op_count;
  const uint64_t b_ops = loom_reduce_sort_jobs[b].removed_op_count;
  if (a_ops != b_ops) {
    return a_ops > b_ops ? -1 : 1;
  }
  return a < b ? -1 : (a > b ? 1 : 0);
}

// Sorts |indices| by decreasing removed operations, then by job order.
static void loom_reduce_sort_by_savings(const loom_reduce_plan_t* plan,
                                        uint32_t* indices,
                                        iree_host_size_t count) {
  loom_reduce_sort_jobs = plan->jobs;
  qsort(indices, count, sizeof(uint32_t), loom_reduce_compare_savings);
  loom_reduce_sort_jobs = NULL;
}

// Skips new motifs whose body could reach a function that will call them.
// Inline motifs must not recurse: a motif calls what its body references and
// every function holding a copy calls the motif. Jobs are admitted largest
// saving first while the combined graph stays acyclic.
static iree_status_t loom_reduce_reject_call_cycles(loom_reduce_plan_t* plan,
                                                    uint32_t* jobs,
                                                    iree_host_size_t count) {
  loom_reduce_graph_t graph;
  memset(&graph, 0, sizeof(graph));
  graph.allocator = plan->allocator;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < plan->source_count && iree_status_is_ok(status); ++i) {
    if (plan->sources[i].module) {
      status = loom_reduce_graph_add_module(&graph, plan->sources[i].module);
    }
  }
  loom_reduce_strings_t names;
  loom_reduce_strings_initialize(plan->allocator, &names);
  uint32_t* starts = NULL;
  iree_host_size_t start_capacity = 0;
  loom_reduce_sort_by_savings(plan, jobs, count);
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    loom_reduce_job_t* job = &plan->jobs[jobs[i]];
    if (job->existing) {
      continue;
    }
    names.count = 0;
    status = loom_reduce_collect_references(
        loom_reduce_copy_body(plan, &job->copies[0]), '@', &names);
    if (iree_status_is_ok(status) && names.count > start_capacity) {
      status = iree_allocator_grow_array(plan->allocator, names.count,
                                         sizeof(uint32_t), &start_capacity,
                                         (void**)&starts);
    }
    uint32_t motif = 0;
    if (iree_status_is_ok(status)) {
      status = loom_reduce_graph_intern(&graph, job->name, &motif);
    }
    for (iree_host_size_t n = 0; n < names.count && iree_status_is_ok(status);
         ++n) {
      status = loom_reduce_graph_intern(&graph, names.items[n], &starts[n]);
    }
    // Callers gain an edge to the motif; remember how many to undo.
    uint32_t caller_nodes[64];
    uint32_t caller_count = 0;
    for (uint32_t c = 0; c < job->copy_count && iree_status_is_ok(status);
         ++c) {
      const iree_string_view_t symbol = job->copies[c].occurrence->symbol;
      if (iree_string_view_is_empty(symbol) || caller_count == 64) {
        continue;
      }
      uint32_t caller = 0;
      status = loom_reduce_graph_intern(&graph, symbol, &caller);
      if (iree_status_is_ok(status)) {
        status = loom_reduce_graph_add_edge(&graph, caller, motif);
        caller_nodes[caller_count++] = caller;
      }
    }
    for (iree_host_size_t n = 0; n < names.count && iree_status_is_ok(status);
         ++n) {
      status = loom_reduce_graph_add_edge(&graph, motif, starts[n]);
    }
    if (iree_status_is_ok(status)) {
      status = loom_reduce_graph_prepare_visits(&graph);
    }
    bool cycle = false;
    if (iree_status_is_ok(status)) {
      status =
          loom_reduce_graph_reaches(&graph, starts, names.count, motif, &cycle);
    }
    if (iree_status_is_ok(status) && cycle) {
      graph.nodes[motif].edge_count = 0;
      for (uint32_t c = caller_count; c-- > 0;) {
        --graph.nodes[caller_nodes[c]].edge_count;
      }
      job->status = LOOM_REDUCE_JOB_SKIPPED;
      job->reason = "the motif would be part of a call cycle";
    }
  }
  iree_allocator_free(plan->allocator, starts);
  loom_reduce_strings_deinitialize(&names);
  loom_reduce_graph_deinitialize(&graph);
  return status;
}

//===----------------------------------------------------------------------===//
// Rendering
//===----------------------------------------------------------------------===//

static iree_status_t loom_reduce_append_types(
    iree_string_builder_t* builder, const loom_analyze_binding_t* bindings,
    const uint32_t* order, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    const loom_analyze_binding_t* binding = &bindings[order ? order[i] : i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%s%.*s", i ? ", " : "", (int)binding->type.size,
        binding->type.data));
  }
  return iree_ok_status();
}

static iree_status_t loom_reduce_append_names(
    iree_string_builder_t* builder, const loom_analyze_binding_t* bindings,
    const uint32_t* order, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    const loom_analyze_binding_t* binding = &bindings[order ? order[i] : i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%s%%%.*s", i ? ", " : "", (int)binding->name.size,
        binding->name.data));
  }
  return iree_ok_status();
}

static iree_status_t loom_reduce_append_call(iree_string_builder_t* builder,
                                             const loom_reduce_job_t* job,
                                             const loom_reduce_copy_t* copy) {
  const loom_analyze_occurrence_t* occurrence = copy->occurrence;
  const uint32_t argument_count =
      job->existing ? job->parameter_count : occurrence->input_count;
  const uint32_t result_count =
      job->existing ? job->result_count : occurrence->output_count;
  if (result_count > 0) {
    IREE_RETURN_IF_ERROR(loom_reduce_append_names(
        builder, occurrence->outputs, copy->result_order, result_count));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " = "));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "func.call @%.*s(", (int)job->name.size, job->name.data));
  IREE_RETURN_IF_ERROR(loom_reduce_append_names(
      builder, occurrence->inputs, copy->argument_order, argument_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ") : ("));
  IREE_RETURN_IF_ERROR(loom_reduce_append_types(
      builder, occurrence->inputs, copy->argument_order, argument_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ")"));
  if (result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " -> ("));
    IREE_RETURN_IF_ERROR(loom_reduce_append_types(
        builder, occurrence->outputs, copy->result_order, result_count));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ")"));
  }
  return iree_ok_status();
}

// Appends `@name(%a: T, ...) -> (R, ...)` for a new motif.
static iree_status_t loom_reduce_append_signature(
    iree_string_builder_t* builder, const loom_reduce_job_t* job) {
  const loom_analyze_occurrence_t* occurrence = job->copies[0].occurrence;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "@%.*s(", (int)job->name.size, job->name.data));
  for (uint32_t i = 0; i < occurrence->input_count; ++i) {
    const loom_analyze_binding_t* input = &occurrence->inputs[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%s%%%.*s: %.*s", i ? ", " : "", (int)input->name.size,
        input->name.data, (int)input->type.size, input->type.data));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ")"));
  if (occurrence->output_count > 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " -> ("));
    IREE_RETURN_IF_ERROR(loom_reduce_append_types(
        builder, occurrence->outputs, NULL, occurrence->output_count));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ")"));
  }
  return iree_ok_status();
}

static iree_status_t loom_reduce_append_motif(loom_reduce_plan_t* plan,
                                              iree_string_builder_t* builder,
                                              const loom_reduce_job_t* job) {
  const loom_reduce_copy_t* template = &job->copies[0];
  const loom_analyze_occurrence_t* occurrence = template->occurrence;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "// Motif extracted by loom-reduce from %u copies (%u ops each).\n"
      "func.def inline ",
      job->copy_count, job->op_count));
  IREE_RETURN_IF_ERROR(loom_reduce_append_signature(builder, job));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " {\n"));
  const uint32_t base_indent = template->start_column - 1;
  iree_string_view_t remaining = loom_reduce_copy_body(plan, template);
  bool first = true;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_reduce_next_line(&remaining);
    if (!first) {
      iree_host_size_t indent = 0;
      while (indent < line.size && line.data[indent] == ' ') {
        ++indent;
      }
      line = iree_string_view_substr(
          line, indent >= base_indent ? base_indent : indent,
          IREE_HOST_SIZE_MAX);
    }
    first = false;
    if (iree_string_view_is_empty(line)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "  %.*s\n", (int)line.size, line.data));
    }
  }
  if (occurrence->output_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "  func.return "));
    IREE_RETURN_IF_ERROR(loom_reduce_append_names(
        builder, occurrence->outputs, NULL, occurrence->output_count));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " : "));
    IREE_RETURN_IF_ERROR(loom_reduce_append_types(
        builder, occurrence->outputs, NULL, occurrence->output_count));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n}\n"));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "  func.return\n}\n"));
  }
  return iree_ok_status();
}

typedef struct loom_reduce_edit_t {
  uint32_t source;
  iree_host_size_t start;
  iree_host_size_t end;
  const loom_reduce_job_t* job;
  const loom_reduce_copy_t* copy;
} loom_reduce_edit_t;

static int loom_reduce_compare_edits(const void* lhs, const void* rhs) {
  const loom_reduce_edit_t* a = (const loom_reduce_edit_t*)lhs;
  const loom_reduce_edit_t* b = (const loom_reduce_edit_t*)rhs;
  if (a->source != b->source) {
    return a->source < b->source ? -1 : 1;
  }
  if (a->start != b->start) {
    return a->start < b->start ? -1 : 1;
  }
  return 0;
}

// Returns the length of the leading comment and blank-line block of |text|.
static iree_host_size_t loom_reduce_preamble_length(iree_string_view_t text) {
  iree_host_size_t offset = 0;
  while (offset < text.size) {
    iree_string_view_t rest =
        iree_string_view_substr(text, offset, IREE_HOST_SIZE_MAX);
    const iree_host_size_t start = offset;
    const iree_string_view_t line = loom_reduce_next_line(&rest);
    if (!loom_reduce_starts_with(line, "//") &&
        !iree_string_view_is_empty(loom_reduce_trim(line))) {
      return start;
    }
    offset = (iree_host_size_t)(rest.data - text.data);
    if (rest.size == 0) {
      offset = text.size;
    }
  }
  return text.size;
}

// Returns the symbol a `<dialect>.decl @name...` line declares.
static iree_string_view_t loom_reduce_declared_symbol(iree_string_view_t line) {
  const iree_host_size_t at = iree_string_view_find_char(line, '@', 0);
  if (at == IREE_STRING_VIEW_NPOS) {
    return iree_string_view_empty();
  }
  const iree_string_view_t rest =
      iree_string_view_substr(line, at + 1, IREE_HOST_SIZE_MAX);
  return iree_make_string_view(rest.data, loom_reduce_name_length(rest));
}

// Returns true when job |job| defines |name| in |source| after the rewrite.
static bool loom_reduce_job_defines(const loom_reduce_job_t* job,
                                    uint32_t source, iree_string_view_t name) {
  if (!iree_string_view_equal(job->name, name)) {
    return false;
  }
  return job->existing ? job->host_source == source
                       : job->motif_source == source;
}

// Renders every file the jobs at |active| rewrite or create.
static iree_status_t loom_reduce_render(loom_reduce_plan_t* plan,
                                        const uint32_t* active,
                                        iree_host_size_t active_count) {
  iree_allocator_t allocator = plan->allocator;
  for (iree_host_size_t i = 0; i < plan->source_count; ++i) {
    iree_string_builder_reset(&plan->sources[i].output);
    plan->sources[i].rendered = false;
  }

  // 1. Replace copies with calls; a moving definition becomes its declaration.
  iree_host_size_t edit_count = 0;
  for (iree_host_size_t i = 0; i < active_count; ++i) {
    edit_count += plan->jobs[active[i]].copy_count + 1;
  }
  loom_reduce_edit_t* edits = NULL;
  if (edit_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, edit_count, sizeof(*edits), (void**)&edits));
  }
  edit_count = 0;
  for (iree_host_size_t i = 0; i < active_count; ++i) {
    const loom_reduce_job_t* job = &plan->jobs[active[i]];
    for (uint32_t c = 0; c < job->copy_count; ++c) {
      const loom_reduce_copy_t* copy = &job->copies[c];
      edits[edit_count++] =
          (loom_reduce_edit_t){copy->source, copy->start, copy->end, job, copy};
    }
    if (job->relocate) {
      edits[edit_count++] =
          (loom_reduce_edit_t){job->definition_source, job->definition_start,
                               job->definition_end, job, NULL};
    }
  }
  if (edit_count > 1) {
    qsort(edits, edit_count, sizeof(*edits), loom_reduce_compare_edits);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t e = 0; e < edit_count && iree_status_is_ok(status);) {
    loom_reduce_source_t* source = &plan->sources[edits[e].source];
    iree_host_size_t cursor = 0;
    for (; e < edit_count && iree_status_is_ok(status) &&
           &plan->sources[edits[e].source] == source;
         ++e) {
      if (edits[e].start < cursor) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL, "overlapping rewrites in %.*s",
            (int)source->relative_path.size, source->relative_path.data);
        break;
      }
      status = iree_string_builder_append_string(
          &source->output, iree_make_string_view(source->text.data + cursor,
                                                 edits[e].start - cursor));
      if (iree_status_is_ok(status)) {
        status = edits[e].copy
                     ? loom_reduce_append_call(&source->output, edits[e].job,
                                               edits[e].copy)
                     : iree_string_builder_append_string(
                           &source->output, edits[e].job->declaration);
      }
      cursor = edits[e].end;
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(
          &source->output,
          iree_string_view_substr(source->text, cursor, IREE_HOST_SIZE_MAX));
      source->rendered = true;
    }
  }
  iree_allocator_free(allocator, edits);
  IREE_RETURN_IF_ERROR(status);

  // 2. Declarations in calling files for functions defined elsewhere.
  iree_string_builder_t scratch;
  iree_string_builder_initialize(allocator, &scratch);
  for (iree_host_size_t s = 0;
       s < plan->source_count && iree_status_is_ok(status); ++s) {
    loom_reduce_source_t* source = &plan->sources[s];
    if (!source->rendered) {
      continue;
    }
    iree_string_builder_reset(&scratch);
    loom_reduce_strings_t seen;
    loom_reduce_strings_initialize(allocator, &seen);
    for (iree_host_size_t i = 0; i < active_count && iree_status_is_ok(status);
         ++i) {
      const loom_reduce_job_t* job = &plan->jobs[active[i]];
      const uint32_t host =
          job->existing ? job->host_source : job->motif_source;
      if (host == s || (job->relocate && job->definition_source == s) ||
          loom_reduce_strings_contains(&seen, job->name)) {
        continue;
      }
      bool calls = false;
      for (uint32_t c = 0; c < job->copy_count && !calls; ++c) {
        calls = job->copies[c].source == s;
      }
      // A file that already declares or defines the function needs nothing.
      if (!calls || (job->existing && source->module &&
                     loom_reduce_symbol_op(source->module, job->name))) {
        continue;
      }
      status = loom_reduce_strings_add(&seen, job->name);
      if (iree_status_is_ok(status) && job->existing) {
        status = iree_string_builder_append_format(&scratch, "%.*s\n",
                                                   (int)job->declaration.size,
                                                   job->declaration.data);
      } else if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(&scratch, "func.decl ");
        if (iree_status_is_ok(status)) {
          status = loom_reduce_append_signature(&scratch, job);
        }
        if (iree_status_is_ok(status)) {
          status = iree_string_builder_append_cstring(&scratch, "\n");
        }
      }
    }
    loom_reduce_strings_deinitialize(&seen);
    if (!iree_status_is_ok(status) || iree_string_builder_size(&scratch) == 0) {
      continue;
    }
    // Rebuild the output with the declarations after its leading comments.
    iree_string_builder_t combined;
    iree_string_builder_initialize(allocator, &combined);
    const iree_string_view_t text = iree_string_builder_view(&source->output);
    const iree_host_size_t preamble = loom_reduce_preamble_length(text);
    status = iree_string_builder_append_string(
        &combined, iree_make_string_view(text.data, preamble));
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(
          &combined, "// Declarations added by loom-reduce.\n");
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(
          &combined, iree_string_builder_view(&scratch));
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_format(&combined, "\n%.*s",
                                                 (int)(text.size - preamble),
                                                 text.data + preamble);
    }
    if (iree_status_is_ok(status)) {
      iree_string_builder_reset(&source->output);
      status = iree_string_builder_append_string(
          &source->output, iree_string_builder_view(&combined));
    }
    iree_string_builder_deinitialize(&combined);
  }

  // 3. Hosted definitions: new motifs and functions moved into a motif file,
  //    with only the declarations their bodies need that the file lacks.
  for (iree_host_size_t s = 0;
       s < plan->source_count && iree_status_is_ok(status); ++s) {
    loom_reduce_source_t* source = &plan->sources[s];
    bool hosts = false;
    for (iree_host_size_t i = 0; i < active_count && !hosts; ++i) {
      const loom_reduce_job_t* job = &plan->jobs[active[i]];
      hosts = job->existing ? (job->relocate && job->host_source == s)
                            : job->motif_source == s;
    }
    if (!hosts) {
      continue;
    }
    const bool created = iree_string_view_is_empty(source->text);

    iree_string_builder_t aliases, declarations;
    iree_string_builder_initialize(allocator, &aliases);
    iree_string_builder_initialize(allocator, &declarations);
    for (iree_host_size_t i = 0; i < active_count && iree_status_is_ok(status);
         ++i) {
      const loom_reduce_job_t* job = &plan->jobs[active[i]];
      bool complete = false;
      if (!job->existing && job->motif_source == s) {
        bool separate = true;
        for (uint32_t c = 0; c < job->copy_count; ++c) {
          if (job->copies[c].source == s) {
            separate = false;
          }
        }
        if (separate) {
          status = loom_reduce_required_declarations(plan, job, &aliases,
                                                     &declarations, &complete);
        }
      } else if (job->relocate && job->host_source == s) {
        loom_reduce_source_t* definition =
            &plan->sources[job->definition_source];
        status = loom_reduce_body_declarations(
            plan, definition,
            iree_make_string_view(definition->text.data + job->definition_start,
                                  job->definition_end - job->definition_start),
            job->name, &aliases, &declarations, &complete);
      }
    }
    iree_string_builder_reset(&scratch);
    if (iree_status_is_ok(status)) {
      iree_string_view_t text = source->rendered
                                    ? iree_string_builder_view(&source->output)
                                    : source->text;
      while (text.size && text.data[text.size - 1] == '\n') {
        --text.size;
      }
      status = iree_string_builder_append_string(&scratch, text);
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(
            &scratch,
            created ? "// Motifs extracted by loom-reduce. Do not edit by "
                      "hand: rerun\n// loom-analyze and loom-reduce on the "
                      "original sources instead.\n\n"
                    : "\n\n// Motifs extracted by loom-reduce.\n");
      }
    }
    loom_reduce_strings_t unique;
    loom_reduce_strings_initialize(allocator, &unique);
    iree_string_builder_t* blocks[2] = {&aliases, &declarations};
    for (int b = 0; b < 2 && iree_status_is_ok(status); ++b) {
      iree_string_view_t remaining = iree_string_builder_view(blocks[b]);
      while (!iree_string_view_is_empty(remaining) &&
             iree_status_is_ok(status)) {
        const iree_string_view_t line = loom_reduce_next_line(&remaining);
        if (iree_string_view_is_empty(line) ||
            loom_reduce_strings_contains(&unique, line)) {
          continue;
        }
        // Skip what the file already declares, defines, or will define.
        bool present = false;
        if (b == 0) {
          // An alias line already spelled in the file.
          for (iree_host_size_t k = 0;
               !created && !present && k + line.size <= source->text.size;
               ++k) {
            present = memcmp(source->text.data + k, line.data, line.size) == 0;
          }
        } else {
          const iree_string_view_t symbol = loom_reduce_declared_symbol(line);
          present = source->module &&
                    loom_reduce_symbol_op(source->module, symbol) != NULL;
          for (iree_host_size_t i = 0; i < active_count && !present; ++i) {
            present = loom_reduce_job_defines(&plan->jobs[active[i]],
                                              (uint32_t)s, symbol);
          }
        }
        status = loom_reduce_strings_add(&unique, line);
        if (iree_status_is_ok(status) && !present) {
          status = iree_string_builder_append_format(&scratch, "%.*s\n",
                                                     (int)line.size, line.data);
        }
      }
    }
    loom_reduce_strings_deinitialize(&unique);
    iree_string_builder_deinitialize(&aliases);
    iree_string_builder_deinitialize(&declarations);

    // Moved functions keep their text; new motifs follow.
    for (iree_host_size_t i = 0; i < active_count && iree_status_is_ok(status);
         ++i) {
      const loom_reduce_job_t* job = &plan->jobs[active[i]];
      if (job->relocate && job->host_source == s) {
        const loom_reduce_source_t* definition =
            &plan->sources[job->definition_source];
        status = iree_string_builder_append_format(
            &scratch, "\n%.*s\n",
            (int)(job->definition_end - job->definition_start),
            definition->text.data + job->definition_start);
      }
    }
    for (iree_host_size_t i = 0; i < active_count && iree_status_is_ok(status);
         ++i) {
      const loom_reduce_job_t* job = &plan->jobs[active[i]];
      if (job->existing || job->motif_source != s) {
        continue;
      }
      status = iree_string_builder_append_cstring(&scratch, "\n");
      if (iree_status_is_ok(status)) {
        status = loom_reduce_append_motif(plan, &scratch, job);
      }
    }
    if (iree_status_is_ok(status)) {
      iree_string_builder_reset(&source->output);
      status = iree_string_builder_append_string(
          &source->output, iree_string_builder_view(&scratch));
    }
    source->rendered = true;
  }
  iree_string_builder_deinitialize(&scratch);
  return status;
}

//===----------------------------------------------------------------------===//
// Apply
//===----------------------------------------------------------------------===//

static bool loom_reduce_job_touches(const loom_reduce_job_t* job,
                                    uint32_t source) {
  if (!job->existing && job->motif_source == source) {
    return true;
  }
  if (job->existing && job->host_source == source) {
    return true;
  }
  if (job->relocate && job->definition_source == source) {
    return true;
  }
  for (uint32_t c = 0; c < job->copy_count; ++c) {
    if (job->copies[c].source == source) {
      return true;
    }
  }
  return false;
}

// Verifies the rendered files in |sources| (all rendered files when NULL);
// marks failures in |failing|.
static iree_status_t loom_reduce_verify(loom_reduce_plan_t* plan,
                                        loom_reduce_verifier_t verifier,
                                        const loom_reduce_job_t* only_job,
                                        bool* failing, bool* out_any_failed) {
  *out_any_failed = false;
  for (iree_host_size_t s = 0; s < plan->source_count; ++s) {
    loom_reduce_source_t* source = &plan->sources[s];
    if (!source->rendered) {
      continue;
    }
    if (only_job && !loom_reduce_job_touches(only_job, (uint32_t)s)) {
      continue;
    }
    bool accepted = false;
    IREE_RETURN_IF_ERROR(verifier.fn(verifier.user_data, source->relative_path,
                                     iree_string_builder_view(&source->output),
                                     &accepted));
    if (!accepted) {
      if (failing) {
        failing[s] = true;
      }
      *out_any_failed = true;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_reduce_plan_apply(loom_reduce_plan_t* plan,
                                     loom_reduce_verifier_t verifier) {
  if (plan->applied) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "the plan was already applied");
  }
  plan->applied = true;
  iree_allocator_t allocator = plan->allocator;
  uint32_t* active = NULL;
  uint32_t* suspects = NULL;
  bool* failing = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, plan->job_count + 1, sizeof(uint32_t), (void**)&active);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, plan->job_count + 1,
                                         sizeof(uint32_t), (void**)&suspects);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, plan->source_count + 1,
                                         sizeof(bool), (void**)&failing);
  }

  iree_host_size_t active_count = 0;
  for (iree_host_size_t j = 0; j < plan->job_count && iree_status_is_ok(status);
       ++j) {
    loom_reduce_job_t* job = &plan->jobs[j];
    if (job->status != LOOM_REDUCE_JOB_PENDING) {
      continue;
    }
    const char* reason = NULL;
    status = loom_reduce_validate(plan, job, &reason);
    if (reason) {
      job->status = LOOM_REDUCE_JOB_SKIPPED;
      job->reason = reason;
    } else {
      active[active_count++] = (uint32_t)j;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_reduce_reject_call_cycles(plan, active, active_count);
  }
  // A function cannot move while another rewrite edits inside its body; the
  // move happens in a later pass.
  for (iree_host_size_t i = 0; i < active_count; ++i) {
    loom_reduce_job_t* job = &plan->jobs[active[i]];
    if (!job->relocate || job->status != LOOM_REDUCE_JOB_PENDING) {
      continue;
    }
    for (iree_host_size_t k = 0; k < active_count; ++k) {
      const loom_reduce_job_t* other = &plan->jobs[active[k]];
      if (k == i || other->status != LOOM_REDUCE_JOB_PENDING) {
        continue;
      }
      for (uint32_t c = 0; c < other->copy_count; ++c) {
        const loom_reduce_copy_t* copy = &other->copies[c];
        if (copy->source == job->definition_source &&
            copy->start < job->definition_end &&
            copy->end > job->definition_start) {
          job->status = LOOM_REDUCE_JOB_SKIPPED;
          job->reason =
              "the existing motif's body is itself being reduced; rerun to "
              "reuse it";
          break;
        }
      }
      if (job->status != LOOM_REDUCE_JOB_PENDING) {
        break;
      }
    }
  }
  // Drop the jobs the cycle and relocation checks skipped.
  iree_host_size_t kept_count = 0;
  for (iree_host_size_t i = 0; i < active_count; ++i) {
    if (plan->jobs[active[i]].status == LOOM_REDUCE_JOB_PENDING) {
      active[kept_count++] = active[i];
    }
  }
  active_count = kept_count;

  if (iree_status_is_ok(status)) {
    status = loom_reduce_render(plan, active, active_count);
  }
  bool any_failed = false;
  if (iree_status_is_ok(status) && verifier.fn) {
    status = loom_reduce_verify(plan, verifier, NULL, failing, &any_failed);
  }
  if (iree_status_is_ok(status) && any_failed) {
    // Withdraw every job touching a failing file, then re-admit them one at a
    // time, largest saving first.
    iree_host_size_t suspect_count = 0;
    kept_count = 0;
    for (iree_host_size_t i = 0; i < active_count; ++i) {
      const loom_reduce_job_t* job = &plan->jobs[active[i]];
      bool touches = false;
      for (iree_host_size_t s = 0; s < plan->source_count && !touches; ++s) {
        touches = failing[s] && loom_reduce_job_touches(job, (uint32_t)s);
      }
      if (touches) {
        suspects[suspect_count++] = active[i];
      } else {
        active[kept_count++] = active[i];
      }
    }
    active_count = kept_count;
    loom_reduce_sort_by_savings(plan, suspects, suspect_count);
    for (iree_host_size_t i = 0; i < suspect_count && iree_status_is_ok(status);
         ++i) {
      loom_reduce_job_t* job = &plan->jobs[suspects[i]];
      active[active_count] = suspects[i];
      status = loom_reduce_render(plan, active, active_count + 1);
      bool failed = false;
      if (iree_status_is_ok(status)) {
        status = loom_reduce_verify(plan, verifier, job, NULL, &failed);
      }
      if (failed) {
        job->status = LOOM_REDUCE_JOB_REVERTED;
        job->reason = "rewritten file failed verification";
      } else {
        ++active_count;
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_reduce_render(plan, active, active_count);
    }
  }
  for (iree_host_size_t i = 0; i < active_count; ++i) {
    plan->jobs[active[i]].status = LOOM_REDUCE_JOB_APPLIED;
  }
  iree_allocator_free(allocator, failing);
  iree_allocator_free(allocator, suspects);
  iree_allocator_free(allocator, active);
  return status;
}

//===----------------------------------------------------------------------===//
// Results
//===----------------------------------------------------------------------===//

iree_host_size_t loom_reduce_plan_output_count(const loom_reduce_plan_t* plan) {
  iree_host_size_t count = 0;
  for (iree_host_size_t s = 0; s < plan->source_count; ++s) {
    if (plan->sources[s].rendered) {
      ++count;
    }
  }
  return count;
}

void loom_reduce_plan_output(const loom_reduce_plan_t* plan,
                             iree_host_size_t index,
                             iree_string_view_t* out_relative_path,
                             iree_string_view_t* out_text) {
  *out_relative_path = iree_string_view_empty();
  *out_text = iree_string_view_empty();
  for (iree_host_size_t s = 0; s < plan->source_count; ++s) {
    if (!plan->sources[s].rendered) {
      continue;
    }
    if (index-- == 0) {
      *out_relative_path = plan->sources[s].relative_path;
      *out_text = iree_string_builder_view(&plan->sources[s].output);
      return;
    }
  }
}

// Returns true when |source| calls a motif defined in |motif_source|.
static bool loom_reduce_links_to(const loom_reduce_plan_t* plan,
                                 uint32_t source, uint32_t motif_source) {
  for (iree_host_size_t j = 0; j < plan->job_count; ++j) {
    const loom_reduce_job_t* job = &plan->jobs[j];
    if (job->status != LOOM_REDUCE_JOB_APPLIED || motif_source == source) {
      continue;
    }
    if (job->existing) {
      if (job->host_source != motif_source) {
        continue;
      }
      if (job->relocate && job->definition_source == source) {
        return true;
      }
    } else if (job->motif_source != motif_source) {
      continue;
    }
    for (uint32_t c = 0; c < job->copy_count; ++c) {
      if (job->copies[c].source == source) {
        return true;
      }
    }
  }
  return false;
}

void loom_reduce_plan_summary(const loom_reduce_plan_t* plan,
                              loom_reduce_summary_t* out_summary) {
  memset(out_summary, 0, sizeof(*out_summary));
  out_summary->skipped_count = (uint32_t)plan->skip_count;
  out_summary->reuse_fallback_count = (uint32_t)plan->fallback_count;
  for (iree_host_size_t j = 0; j < plan->job_count; ++j) {
    const loom_reduce_job_t* job = &plan->jobs[j];
    switch (job->status) {
      case LOOM_REDUCE_JOB_APPLIED:
        ++out_summary->motif_count;
        if (job->existing) {
          ++out_summary->reused_count;
        }
        out_summary->replaced_copy_count += job->copy_count;
        out_summary->removed_op_count += job->removed_op_count;
        break;
      case LOOM_REDUCE_JOB_SKIPPED:
        ++out_summary->skipped_count;
        break;
      case LOOM_REDUCE_JOB_REVERTED:
        ++out_summary->reverted_count;
        break;
      default:
        break;
    }
  }
  out_summary->changed_file_count =
      (uint32_t)loom_reduce_plan_output_count(plan);
  for (iree_host_size_t s = 0; s < plan->source_count; ++s) {
    for (iree_host_size_t m = 0; m < plan->source_count; ++m) {
      if (loom_reduce_links_to(plan, (uint32_t)s, (uint32_t)m)) {
        ++out_summary->linked_file_count;
        break;
      }
    }
  }
}

static const char* loom_reduce_status_name(loom_reduce_job_status_t status) {
  switch (status) {
    case LOOM_REDUCE_JOB_APPLIED:
      return "applied";
    case LOOM_REDUCE_JOB_SKIPPED:
      return "skipped";
    case LOOM_REDUCE_JOB_REVERTED:
      return "reverted";
    default:
      return "pending";
  }
}

static iree_status_t loom_reduce_write_fingerprint(
    loom_json_object_writer_t* object, uint64_t fingerprint) {
  char buffer[17];
  snprintf(buffer, sizeof(buffer), "%016" PRIx64, fingerprint);
  return loom_json_object_write_string_field(object, IREE_SV("fingerprint"),
                                             iree_make_cstring_view(buffer));
}

iree_status_t loom_reduce_plan_write_report(const loom_reduce_plan_t* plan,
                                            uint64_t op_count,
                                            loom_output_stream_t* stream) {
  loom_reduce_summary_t summary;
  loom_reduce_plan_summary(plan, &summary);
  loom_json_object_writer_t root;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &root));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("tool"), IREE_SV("loom-reduce")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("scope"),
      plan->options.scope == LOOM_REDUCE_SCOPE_FILE ? IREE_SV("file")
                                                    : IREE_SV("corpus")));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&root, IREE_SV("summary")));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("ops_before"), op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("removed_ops"), summary.removed_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("motifs"), summary.motif_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("replaced_copies"), summary.replaced_copy_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("changed_files"), summary.changed_file_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("skipped"), summary.skipped_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("reverted"), summary.reverted_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("reused"), summary.reused_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("reuse_fallbacks"), summary.reuse_fallback_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));

  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&root, IREE_SV("jobs")));
  loom_json_array_writer_t jobs;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &jobs));
  for (iree_host_size_t j = 0; j < plan->job_count; ++j) {
    const loom_reduce_job_t* job = &plan->jobs[j];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&jobs));
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("motif"), job->name));
    const uint32_t host = job->existing ? job->host_source : job->motif_source;
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("file"),
        job->copy_count || !job->existing ? plan->sources[host].relative_path
                                          : iree_string_view_empty()));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("existing"), job->existing));
    IREE_RETURN_IF_ERROR(
        loom_reduce_write_fingerprint(&object, job->fingerprint));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("op_count"), job->op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("copies"), job->copy_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("removed_ops"),
        job->status == LOOM_REDUCE_JOB_APPLIED ? job->removed_op_count : 0));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("status"),
        iree_make_cstring_view(loom_reduce_status_name(job->status))));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("reason"),
        iree_make_cstring_view(job->reason ? job->reason : "")));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&jobs));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("skipped_findings")));
  loom_json_array_writer_t skips;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &skips));
  for (iree_host_size_t i = 0; i < plan->skip_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&skips));
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(
        loom_reduce_write_fingerprint(&object, plan->skips[i].fingerprint));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("reason"),
        iree_make_cstring_view(plan->skips[i].reason)));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&skips));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("reuse_fallbacks")));
  loom_json_array_writer_t fallbacks;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &fallbacks));
  for (iree_host_size_t i = 0; i < plan->fallback_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&fallbacks));
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(
        loom_reduce_write_fingerprint(&object, plan->fallbacks[i].fingerprint));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("reason"),
        iree_make_cstring_view(plan->fallbacks[i].reason)));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&fallbacks));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("changed_files")));
  loom_json_array_writer_t files;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &files));
  for (iree_host_size_t s = 0; s < plan->source_count; ++s) {
    if (!plan->sources[s].rendered) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &files, plan->sources[s].relative_path));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&files));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&root));
  return loom_output_stream_write_char(stream, '\n');
}

iree_status_t loom_reduce_plan_write_links(const loom_reduce_plan_t* plan,
                                           loom_output_stream_t* stream) {
  loom_json_object_writer_t root;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &root));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("requires")));
  loom_json_object_writer_t requires;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &requires));
  for (iree_host_size_t s = 0; s < plan->source_count; ++s) {
    bool any = false;
    for (iree_host_size_t m = 0; m < plan->source_count && !any; ++m) {
      any = loom_reduce_links_to(plan, (uint32_t)s, (uint32_t)m);
    }
    if (!any) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &requires, plan->sources[s].relative_path));
    loom_json_array_writer_t list;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &list));
    for (iree_host_size_t m = 0; m < plan->source_count; ++m) {
      if (loom_reduce_links_to(plan, (uint32_t)s, (uint32_t)m)) {
        IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
            &list, plan->sources[m].relative_path));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&list));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&requires));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&root));
  return loom_output_stream_write_char(stream, '\n');
}
