// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-analyze/clone_analysis.h"

#include <stdlib.h>
#include <string.h>

#include "loom/error/source.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/type_dependencies.h"
#include "loom/ir/type_refs.h"
#include "loom/ir/value_refs.h"
#include "loom/ops/op_defs.h"
#include "loom/util/stream.h"

//===----------------------------------------------------------------------===//
// Rules and options
//===----------------------------------------------------------------------===//

iree_string_view_t loom_analyze_rule_id(loom_analyze_rule_t rule) {
  switch (rule) {
    case LOOM_ANALYZE_RULE_DUPLICATE_SEQUENCE:
      return IREE_SV("loom-analyze/duplicate-sequence");
    case LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF:
      return IREE_SV("loom-analyze/reimplemented-motif");
    default:
      return IREE_SV("loom-analyze/unknown");
  }
}

iree_string_view_t loom_analyze_rule_description(loom_analyze_rule_t rule) {
  switch (rule) {
    case LOOM_ANALYZE_RULE_DUPLICATE_SEQUENCE:
      return IREE_SV(
          "An operation sequence is written more than once; factor it into a "
          "func.def inline or template.def motif.");
    case LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF:
      return IREE_SV(
          "An operation sequence repeats the body of an existing motif; call "
          "the motif instead of copying it.");
    default:
      return IREE_SV("Unknown rule.");
  }
}

void loom_analyze_options_initialize(loom_analyze_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->min_op_count = 6;
  out_options->min_occurrence_count = 2;
  out_options->parameterization = LOOM_ANALYZE_PARAMETERIZE_NONE;
}

iree_status_t loom_analyze_parse_parameterization(
    iree_string_view_t value, loom_analyze_parameterization_t* out_value) {
  loom_analyze_parameterization_t result = LOOM_ANALYZE_PARAMETERIZE_NONE;
  iree_string_view_t remaining = value;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t item = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &item, &remaining);
    item = iree_string_view_trim(item);
    if (iree_string_view_equal(item, IREE_SV("none"))) {
      continue;
    } else if (iree_string_view_equal(item, IREE_SV("constants"))) {
      result |= LOOM_ANALYZE_PARAMETERIZE_CONSTANTS;
    } else if (iree_string_view_equal(item, IREE_SV("types"))) {
      result |= LOOM_ANALYZE_PARAMETERIZE_TYPES;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown --parameterize '%.*s'; expected a "
                              "comma list of none, constants, types",
                              (int)item.size, item.data);
    }
  }
  *out_value = result;
  return iree_ok_status();
}

iree_string_view_t loom_analyze_parameterization_name(
    loom_analyze_parameterization_t value) {
  switch (value & LOOM_ANALYZE_PARAMETERIZE_ALL_) {
    case LOOM_ANALYZE_PARAMETERIZE_CONSTANTS:
      return IREE_SV("constants");
    case LOOM_ANALYZE_PARAMETERIZE_TYPES:
      return IREE_SV("types");
    case LOOM_ANALYZE_PARAMETERIZE_CONSTANTS | LOOM_ANALYZE_PARAMETERIZE_TYPES:
      return IREE_SV("constants,types");
    default:
      return iree_string_view_empty();
  }
}

//===----------------------------------------------------------------------===//
// Stable hashing
//===----------------------------------------------------------------------===//

// FNV-1a 64. Fingerprints are written to SARIF and must be stable across
// processes and platforms, so the process-local structural hash is not used.
#define LOOM_ANALYZE_FNV_OFFSET UINT64_C(0xcbf29ce484222325)
#define LOOM_ANALYZE_FNV_PRIME UINT64_C(0x100000001b3)

static inline uint64_t loom_analyze_hash_byte(uint64_t hash, uint8_t byte) {
  return (hash ^ byte) * LOOM_ANALYZE_FNV_PRIME;
}

static inline uint64_t loom_analyze_hash_u64(uint64_t hash, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    hash = loom_analyze_hash_byte(hash, (uint8_t)(value >> (i * 8)));
  }
  return hash;
}

static inline bool loom_analyze_is_name_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '$' || c == '#';
}

static inline bool loom_analyze_is_digit(char c) {
  return c >= '0' && c <= '9';
}

// Returns the offset of the last ':' outside quoted strings, or |text.size|.
// In a printed constant this separates the literal from its result type.
static iree_host_size_t loom_analyze_find_type_separator(
    iree_string_view_t text) {
  iree_host_size_t separator = text.size;
  bool in_string = false;
  for (iree_host_size_t i = 0; i < text.size; ++i) {
    const char c = text.data[i];
    if (in_string) {
      if (c == '\\') {
        ++i;
      } else if (c == '"') {
        in_string = false;
      }
    } else if (c == '"') {
      in_string = true;
    } else if (c == ':') {
      separator = i;
    }
  }
  return separator;
}

// Returns true for scalar element type spellings: i8, si32, ui16, bf16, f32,
// f8E4M3FN, ...
static bool loom_analyze_is_element_type(iree_string_view_t token) {
  iree_host_size_t i = 0;
  if (iree_string_view_equal(token, IREE_SV("bf16"))) {
    return true;
  }
  if (token.size >= 2 && token.data[0] == 'f' &&
      loom_analyze_is_digit(token.data[1])) {
    for (i = 2; i < token.size; ++i) {
      const char c = token.data[i];
      if (!loom_analyze_is_digit(c) && !(c >= 'a' && c <= 'z') &&
          !(c >= 'A' && c <= 'Z')) {
        return false;
      }
    }
    return true;
  }
  if (token.size > 0 && (token.data[0] == 's' || token.data[0] == 'u')) {
    i = 1;
  }
  if (i >= token.size || token.data[i] != 'i' || i + 1 >= token.size) {
    return false;
  }
  for (++i; i < token.size; ++i) {
    if (!loom_analyze_is_digit(token.data[i])) {
      return false;
    }
  }
  return true;
}

// Returns true when |token| spells an element type, either alone (f32) or as
// the suffix of a shape (4x8xf32, ?xi16). |out_keep| receives the length of
// the shape prefix that still compares exactly.
static bool loom_analyze_split_type_token(iree_string_view_t token,
                                          iree_host_size_t* out_keep) {
  *out_keep = 0;
  if (loom_analyze_is_element_type(token)) {
    return true;
  }
  iree_host_size_t split = IREE_STRING_VIEW_NPOS;
  for (iree_host_size_t i = 0; i < token.size; ++i) {
    if (token.data[i] == 'x') {
      split = i;
    }
  }
  if (split == IREE_STRING_VIEW_NPOS || split == 0) {
    return false;
  }
  for (iree_host_size_t i = 0; i < split; ++i) {
    const char c = token.data[i];
    if (!loom_analyze_is_digit(c) && c != '?' && c != 'x') {
      return false;
    }
  }
  if (!loom_analyze_is_element_type(
          iree_string_view_substr(token, split + 1, IREE_HOST_SIZE_MAX))) {
    return false;
  }
  *out_keep = split + 1;
  return true;
}

// Hashes printed IR after removing everything that does not carry program
// content: SSA value names (%x), block labels (^bb), comments, and layout
// whitespace. With |strip_numbers|, numeric literals before the result type
// separator are also removed so constants differing only in value hash
// equally while their types (including shapes such as vector<4xi32>) still
// compare exactly. With |strip_types|, element types become one placeholder
// while shapes and type constructors remain.
static uint64_t loom_analyze_hash_normalized_text(iree_string_view_t text,
                                                  bool strip_numbers,
                                                  bool strip_types) {
  uint64_t hash = LOOM_ANALYZE_FNV_OFFSET;
  const iree_host_size_t strip_limit =
      strip_numbers ? loom_analyze_find_type_separator(text) : 0;
  bool pending_space = false;
  bool emitted_any = false;
  char previous = ' ';
  iree_host_size_t i = 0;
  while (i < text.size) {
    const char c = text.data[i];
    if (c == '"') {
      // Quoted strings are content; copy them verbatim including escapes.
      if (pending_space && emitted_any) {
        hash = loom_analyze_hash_byte(hash, ' ');
      }
      pending_space = false;
      hash = loom_analyze_hash_byte(hash, (uint8_t)c);
      ++i;
      while (i < text.size && text.data[i] != '"') {
        if (text.data[i] == '\\' && i + 1 < text.size) {
          hash = loom_analyze_hash_byte(hash, (uint8_t)text.data[i]);
          ++i;
        }
        hash = loom_analyze_hash_byte(hash, (uint8_t)text.data[i]);
        ++i;
      }
      if (i < text.size) {
        hash = loom_analyze_hash_byte(hash, '"');
        ++i;
      }
      emitted_any = true;
      previous = '"';
      continue;
    }
    if (c == '/' && i + 1 < text.size && text.data[i + 1] == '/') {
      while (i < text.size && text.data[i] != '\n') {
        ++i;
      }
      pending_space = true;
      previous = ' ';
      continue;
    }
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      pending_space = true;
      previous = ' ';
      ++i;
      continue;
    }
    if (strip_types && loom_analyze_is_name_char(c) && c != '#' && c != '$' &&
        !loom_analyze_is_name_char(previous)) {
      iree_host_size_t end = i;
      while (end < text.size && loom_analyze_is_name_char(text.data[end])) {
        ++end;
      }
      iree_host_size_t keep = 0;
      if (loom_analyze_split_type_token(
              iree_make_string_view(text.data + i, end - i), &keep)) {
        if (pending_space && emitted_any) {
          hash = loom_analyze_hash_byte(hash, ' ');
        }
        for (iree_host_size_t k = 0; k < keep; ++k) {
          hash = loom_analyze_hash_byte(hash, (uint8_t)text.data[i + k]);
        }
        hash = loom_analyze_hash_byte(hash, 'T');
        pending_space = false;
        emitted_any = true;
        previous = 'T';
        i = end;
        continue;
      }
    }
    char emitted = c;
    if (c == '%' || c == '^') {
      // SSA names and block labels: keep the sigil, drop the spelling.
      ++i;
      while (i < text.size && loom_analyze_is_name_char(text.data[i])) {
        ++i;
      }
    } else if (i < strip_limit && !loom_analyze_is_name_char(previous) &&
               (loom_analyze_is_digit(c) ||
                (c == '-' && i + 1 < text.size &&
                 loom_analyze_is_digit(text.data[i + 1])))) {
      // Numeric literal: decimal, float, exponent, or hex spelling.
      emitted = '?';
      ++i;
      while (i < text.size) {
        const char n = text.data[i];
        const char p = text.data[i - 1];
        if (loom_analyze_is_name_char(n) && n != '#' && n != '$') {
          ++i;
        } else if ((n == '-' || n == '+') && (p == 'e' || p == 'E')) {
          ++i;
        } else {
          break;
        }
      }
    } else {
      ++i;
    }
    if (pending_space && emitted_any) {
      hash = loom_analyze_hash_byte(hash, ' ');
    }
    pending_space = false;
    hash = loom_analyze_hash_byte(hash, (uint8_t)emitted);
    emitted_any = true;
    previous = emitted == '?' ? '0' : emitted;
  }
  return hash;
}

//===----------------------------------------------------------------------===//
// Analyzer state
//===----------------------------------------------------------------------===//

// One indexed operation inside a definition, in module pre-order. Nested
// operations of an op occupy the indices immediately after it, so any
// contiguous run of sibling operations is also a contiguous index range.
typedef struct loom_analyze_op_t {
  // Borrowed operation.
  const loom_op_t* op;
  // Index of the owning module.
  uint32_t module_index;
  // Index one past the last operation nested under this one.
  uint32_t subtree_end;
  // Index into blocks of the first block nested under this operation.
  uint32_t block_cursor;
  // Hash of the operation's normalized printed form, regions elided, for each
  // combination of loom_analyze_parameterization_t relaxations. Index 0 is
  // exact; constant-literal stripping applies only to `*.constant` ops.
  uint64_t hashes[LOOM_ANALYZE_PARAMETERIZE_ALL_ + 1];
} loom_analyze_op_t;

// One indexed block, in the same pre-order as operations.
typedef struct loom_analyze_block_t {
  // Hash of the argument count and normalized argument types: [0] exact,
  // [1] with element types relaxed.
  uint64_t argument_hashes[2];
} loom_analyze_block_t;

// One candidate operation sequence: the operations of one nested block.
typedef struct loom_analyze_sequence_t {
  // Index of the owning module.
  uint32_t module_index;
  // Offset into sequence_ops of the block's operation indices.
  uint32_t op_offset;
  // Number of candidate operations: the block's operations except a
  // trailing terminator.
  uint32_t length;
  // Body length excluding the terminator when this block is the entry block
  // of a top-level definition body; zero otherwise.
  uint32_t motif_length;
  // True when the block lies inside a kernel.def launch-config region.
  bool in_launch_config;
  // Symbol of the enclosing top-level definition, or empty.
  iree_string_view_t symbol;
} loom_analyze_sequence_t;

typedef struct loom_analyze_module_t {
  // Borrowed module.
  const loom_module_t* module;
  // Per-value window tags, valid when value_stamps matches generation.
  uint64_t* value_tags;
  uint32_t* value_stamps;
  // Number of entries in value_tags and value_stamps.
  iree_host_size_t value_count;
  // Current encoding generation; bumped for every encoded window.
  uint32_t generation;
} loom_analyze_module_t;

// A candidate window over one sequence.
typedef struct loom_analyze_candidate_t {
  uint64_t hash;
  uint32_t sequence_index;
  uint32_t start;
  uint32_t length;
} loom_analyze_candidate_t;

// One occurrence of a clone class.
typedef struct loom_analyze_class_occurrence_t {
  uint32_t sequence_index;
  uint32_t start;
  // Global operation index range covered, including nested operations.
  uint32_t op_begin;
  uint32_t op_end;
} loom_analyze_class_occurrence_t;

typedef struct loom_analyze_class_t {
  uint64_t fingerprint;
  uint32_t length;
  uint32_t op_count;
  uint32_t occurrence_offset;
  uint32_t occurrence_count;
  uint32_t input_count;
  uint32_t output_count;
  // Relaxations the copies actually needed to match.
  loom_analyze_parameterization_t parameterized_by;
  // Top-level constants differing between copies, and whether a constant
  // nested in a region differs.
  uint32_t call_site_constant_count;
  bool nested_constant_differs;
} loom_analyze_class_t;

// Growable token buffer used to compare canonical encodings exactly.
typedef struct loom_analyze_tokens_t {
  uint64_t* data;
  iree_host_size_t count;
  iree_host_size_t capacity;
} loom_analyze_tokens_t;

#define LOOM_ANALYZE_DEFINE_VECTOR(name, type)                              \
  typedef struct name##_vector_t {                                          \
    type* data;                                                             \
    iree_host_size_t count;                                                 \
    iree_host_size_t capacity;                                              \
  } name##_vector_t;                                                        \
  static iree_status_t name##_vector_reserve(iree_allocator_t allocator,    \
                                             name##_vector_t* vector,       \
                                             iree_host_size_t extra) {      \
    if (vector->count + extra <= vector->capacity) return iree_ok_status(); \
    return iree_allocator_grow_array(allocator, vector->count + extra,      \
                                     sizeof(type), &vector->capacity,       \
                                     (void**)&vector->data);                \
  }                                                                         \
  static void name##_vector_free(iree_allocator_t allocator,                \
                                 name##_vector_t* vector) {                 \
    iree_allocator_free(allocator, vector->data);                           \
    memset(vector, 0, sizeof(*vector));                                     \
  }

LOOM_ANALYZE_DEFINE_VECTOR(loom_analyze_op, loom_analyze_op_t)
LOOM_ANALYZE_DEFINE_VECTOR(loom_analyze_block, loom_analyze_block_t)
LOOM_ANALYZE_DEFINE_VECTOR(loom_analyze_sequence, loom_analyze_sequence_t)
LOOM_ANALYZE_DEFINE_VECTOR(loom_analyze_module, loom_analyze_module_t)
LOOM_ANALYZE_DEFINE_VECTOR(loom_analyze_index, uint32_t)
LOOM_ANALYZE_DEFINE_VECTOR(loom_analyze_candidate, loom_analyze_candidate_t)
LOOM_ANALYZE_DEFINE_VECTOR(loom_analyze_class, loom_analyze_class_t)
LOOM_ANALYZE_DEFINE_VECTOR(loom_analyze_class_occurrence,
                           loom_analyze_class_occurrence_t)
LOOM_ANALYZE_DEFINE_VECTOR(loom_analyze_string, char*)

struct loom_analyze_analyzer_t {
  iree_allocator_t allocator;
  loom_analyze_options_t options;
  loom_analyze_module_vector_t modules;
  loom_analyze_op_vector_t ops;
  loom_analyze_block_vector_t blocks;
  loom_analyze_sequence_vector_t sequences;
  // Operation indices of every sequence, addressed by op_offset.
  loom_analyze_index_vector_t sequence_ops;
  // Scratch text for printing operations and types.
  iree_string_builder_t scratch;
  // Scratch encodings for exact comparison.
  loom_analyze_tokens_t tokens_a;
  loom_analyze_tokens_t tokens_b;
  // Distinct external inputs of the most recently encoded window.
  uint32_t last_window_input_count;
  // Hash variant used by window encoding; the configured relaxations except
  // while classifying which relaxations a finding needed.
  uint32_t encode_variant;
  // Owned strings referenced by results, such as printed binding types.
  loom_analyze_string_vector_t strings;
  // Results.
  bool has_run;
  loom_analyze_result_t result;
};

iree_status_t loom_analyze_analyzer_create(
    const loom_analyze_options_t* options, iree_allocator_t allocator,
    loom_analyze_analyzer_t** out_analyzer) {
  *out_analyzer = NULL;
  if (options->min_op_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "minimum clone size must be at least one op");
  }
  if (options->min_occurrence_count < 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "minimum occurrence count must be at least 2");
  }
  loom_analyze_analyzer_t* analyzer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*analyzer), (void**)&analyzer));
  analyzer->allocator = allocator;
  analyzer->options = *options;
  analyzer->encode_variant =
      options->parameterization & LOOM_ANALYZE_PARAMETERIZE_ALL_;
  iree_string_builder_initialize(allocator, &analyzer->scratch);
  *out_analyzer = analyzer;
  return iree_ok_status();
}

void loom_analyze_analyzer_free(loom_analyze_analyzer_t* analyzer) {
  if (!analyzer) {
    return;
  }
  iree_allocator_t allocator = analyzer->allocator;
  for (iree_host_size_t i = 0; i < analyzer->modules.count; ++i) {
    iree_allocator_free(allocator, analyzer->modules.data[i].value_tags);
    iree_allocator_free(allocator, analyzer->modules.data[i].value_stamps);
  }
  for (iree_host_size_t i = 0; i < analyzer->result.finding_count; ++i) {
    loom_analyze_finding_t* finding = &analyzer->result.findings[i];
    for (uint32_t j = 0; finding->occurrences && j < finding->occurrence_count;
         ++j) {
      // Inputs and outputs share one allocation rooted at inputs.
      iree_allocator_free(allocator, finding->occurrences[j].inputs);
    }
    iree_allocator_free(allocator, finding->occurrences);
  }
  iree_allocator_free(allocator, analyzer->result.findings);
  for (iree_host_size_t i = 0; i < analyzer->strings.count; ++i) {
    iree_allocator_free(allocator, analyzer->strings.data[i]);
  }
  loom_analyze_string_vector_free(allocator, &analyzer->strings);
  loom_analyze_module_vector_free(allocator, &analyzer->modules);
  loom_analyze_op_vector_free(allocator, &analyzer->ops);
  loom_analyze_block_vector_free(allocator, &analyzer->blocks);
  loom_analyze_sequence_vector_free(allocator, &analyzer->sequences);
  loom_analyze_index_vector_free(allocator, &analyzer->sequence_ops);
  iree_allocator_free(allocator, analyzer->tokens_a.data);
  iree_allocator_free(allocator, analyzer->tokens_b.data);
  iree_string_builder_deinitialize(&analyzer->scratch);
  iree_allocator_free(allocator, analyzer);
}

//===----------------------------------------------------------------------===//
// Indexing
//===----------------------------------------------------------------------===//

typedef struct loom_analyze_index_context_t {
  loom_analyze_analyzer_t* analyzer;
  const loom_module_t* module;
  uint32_t module_index;
  loom_text_printer_t* printer;
  // Symbol of the enclosing top-level definition.
  iree_string_view_t symbol;
  // True below a kernel.def launch-config region.
  bool in_launch_config;
} loom_analyze_index_context_t;

// Prints |op| without regions and hashes its normalized text. Operations that
// fail to print (for example Low regions without an asm environment) fall back
// to their operation name and arity, which still separates distinct ops.
static void loom_analyze_hash_operation(loom_analyze_index_context_t* context,
                                        const loom_op_t* op,
                                        uint64_t* out_hashes) {
  iree_string_builder_t* scratch = &context->analyzer->scratch;
  iree_string_builder_reset(scratch);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(scratch, &stream);
  loom_text_print_options_t print_options;
  memset(&print_options, 0, sizeof(print_options));
  print_options.flags = LOOM_TEXT_PRINT_SKIP_REGIONS;
  const iree_string_view_t name = loom_op_name(context->module, op);
  iree_status_t status = loom_text_printer_write_operation(
      context->printer, op, &stream, &print_options);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    uint64_t hash = loom_analyze_hash_normalized_text(name, false, false);
    hash = loom_analyze_hash_u64(hash, op->operand_count);
    hash = loom_analyze_hash_u64(hash, op->result_count);
    hash = loom_analyze_hash_u64(hash, op->region_count);
    for (uint32_t v = 0; v <= LOOM_ANALYZE_PARAMETERIZE_ALL_; ++v) {
      out_hashes[v] = hash;
    }
    return;
  }
  const iree_string_view_t text = iree_string_builder_view(scratch);
  const bool is_constant =
      iree_string_view_ends_with(name, IREE_SV(".constant"));
  for (uint32_t v = 0; v <= LOOM_ANALYZE_PARAMETERIZE_ALL_; ++v) {
    out_hashes[v] = loom_analyze_hash_normalized_text(
        text,
        /*strip_numbers=*/is_constant &&
            (v & LOOM_ANALYZE_PARAMETERIZE_CONSTANTS) != 0,
        /*strip_types=*/(v & LOOM_ANALYZE_PARAMETERIZE_TYPES) != 0);
  }
}

// Hashes the argument count and normalized argument types, once exactly and
// once with element types relaxed.
static void loom_analyze_hash_block_arguments(
    loom_analyze_index_context_t* context, const loom_block_t* block,
    uint64_t out_hashes[2]) {
  out_hashes[0] =
      loom_analyze_hash_u64(LOOM_ANALYZE_FNV_OFFSET, block->arg_count);
  out_hashes[1] = out_hashes[0];
  iree_string_builder_t* scratch = &context->analyzer->scratch;
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(context->module, block->arg_ids[i]);
    iree_string_builder_reset(scratch);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(scratch, &stream);
    iree_status_t status =
        loom_text_printer_write_type(context->printer, type, &stream, NULL);
    for (int relaxed = 0; relaxed < 2; ++relaxed) {
      uint64_t type_hash = 0;
      if (iree_status_is_ok(status)) {
        type_hash = loom_analyze_hash_normalized_text(
            iree_string_builder_view(scratch), /*strip_numbers=*/false,
            /*strip_types=*/relaxed != 0);
      }
      out_hashes[relaxed] =
          loom_analyze_hash_u64(out_hashes[relaxed], type_hash);
    }
    iree_status_free(status);
  }
}

static iree_status_t loom_analyze_index_operation(
    loom_analyze_index_context_t* context, const loom_op_t* op);

// Indexes every block of every region of |op| as candidate sequences.
// |is_top_level| marks module-level definitions, whose entry body block is
// recorded as a motif body when |op| is a func.def or template.def.
static iree_status_t loom_analyze_index_regions(
    loom_analyze_index_context_t* context, const loom_op_t* op,
    bool is_top_level) {
  loom_analyze_analyzer_t* analyzer = context->analyzer;
  iree_allocator_t allocator = analyzer->allocator;
  const bool is_kernel_def = iree_string_view_equal(
      loom_op_name(context->module, op), IREE_SV("kernel.def"));
  // Only callable definitions are motifs: func.def through func.call and
  // template.def through template.apply. Kernel bodies are entry points.
  loom_region_t* motif_region = NULL;
  if (is_top_level) {
    const iree_string_view_t name = loom_op_name(context->module, op);
    if (iree_string_view_equal(name, IREE_SV("func.def")) ||
        iree_string_view_equal(name, IREE_SV("template.def"))) {
      loom_func_like_t func =
          loom_func_like_cast(context->module, (loom_op_t*)op);
      if (loom_func_like_isa(func)) {
        motif_region = loom_func_like_body(func);
      }
    }
  }
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t r = 0; r < op->region_count; ++r) {
    loom_region_t* region = regions[r];
    if (!region) {
      continue;
    }
    const bool saved_launch_config = context->in_launch_config;
    // kernel.def region 0 is the launch-config body; region 1 is the kernel.
    if (is_kernel_def && r == 0) {
      context->in_launch_config = true;
    }
    for (uint16_t b = 0; b < region->block_count; ++b) {
      const loom_block_t* block = region->blocks[b];
      if (!block) {
        continue;
      }

      IREE_RETURN_IF_ERROR(
          loom_analyze_block_vector_reserve(allocator, &analyzer->blocks, 1));
      loom_analyze_hash_block_arguments(
          context, block,
          analyzer->blocks.data[analyzer->blocks.count++].argument_hashes);

      uint32_t length = 0;
      for (const loom_op_t* child = block->first_op; child;
           child = child->next_op) {
        ++length;
      }
      IREE_RETURN_IF_ERROR(loom_analyze_sequence_vector_reserve(
          allocator, &analyzer->sequences, 1));
      IREE_RETURN_IF_ERROR(loom_analyze_index_vector_reserve(
          allocator, &analyzer->sequence_ops, length));
      const uint32_t op_offset = (uint32_t)analyzer->sequence_ops.count;
      analyzer->sequence_ops.count += length;
      loom_analyze_sequence_t* sequence =
          &analyzer->sequences.data[analyzer->sequences.count++];
      memset(sequence, 0, sizeof(*sequence));
      sequence->module_index = context->module_index;
      sequence->op_offset = op_offset;
      // A block terminator never joins a candidate window: it cannot move
      // into a motif, and implicit terminators (an elided scf.yield) have a
      // source range reaching the region's closing brace. Its operation is
      // still indexed below so nested pre-order numbering stays dense.
      const bool ends_in_terminator =
          length > 0 &&
          iree_any_bit_set(block->last_op->traits, LOOM_TRAIT_TERMINATOR);
      sequence->length = ends_in_terminator ? length - 1 : length;
      sequence->in_launch_config = context->in_launch_config;
      sequence->symbol = context->symbol;
      if (motif_region == region && b == 0) {
        sequence->motif_length = sequence->length;
      }

      uint32_t position = 0;
      for (const loom_op_t* child = block->first_op; child;
           child = child->next_op) {
        // Reservations for nested blocks may move sequence_ops; re-index.
        analyzer->sequence_ops.data[op_offset + position++] =
            (uint32_t)analyzer->ops.count;
        IREE_RETURN_IF_ERROR(loom_analyze_index_operation(context, child));
      }
    }
    context->in_launch_config = saved_launch_config;
  }
  return iree_ok_status();
}

static iree_status_t loom_analyze_index_operation(
    loom_analyze_index_context_t* context, const loom_op_t* op) {
  loom_analyze_analyzer_t* analyzer = context->analyzer;
  IREE_RETURN_IF_ERROR(
      loom_analyze_op_vector_reserve(analyzer->allocator, &analyzer->ops, 1));
  const uint32_t op_index = (uint32_t)analyzer->ops.count++;
  loom_analyze_op_t* entry = &analyzer->ops.data[op_index];
  memset(entry, 0, sizeof(*entry));
  entry->op = op;
  entry->module_index = context->module_index;
  entry->block_cursor = (uint32_t)analyzer->blocks.count;
  uint64_t hashes[LOOM_ANALYZE_PARAMETERIZE_ALL_ + 1];
  loom_analyze_hash_operation(context, op, hashes);
  memcpy(analyzer->ops.data[op_index].hashes, hashes, sizeof(hashes));
  IREE_RETURN_IF_ERROR(
      loom_analyze_index_regions(context, op, /*is_top_level=*/false));
  // Recursion may have grown the vector; address the entry by index.
  analyzer->ops.data[op_index].subtree_end = (uint32_t)analyzer->ops.count;
  return iree_ok_status();
}

static iree_string_view_t loom_analyze_symbol_name(const loom_module_t* module,
                                                   const loom_op_t* op) {
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (symbol->defining_op != op) {
      continue;
    }
    if (symbol->name_id == LOOM_STRING_ID_INVALID ||
        (iree_host_size_t)symbol->name_id >= module->strings.count) {
      break;
    }
    return loom_string_table_get(&module->strings, symbol->name_id);
  }
  return iree_string_view_empty();
}

iree_status_t loom_analyze_analyzer_add_module(
    loom_analyze_analyzer_t* analyzer, const loom_module_t* module) {
  if (analyzer->has_run) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "modules must be added before the analysis runs");
  }
  iree_allocator_t allocator = analyzer->allocator;
  IREE_RETURN_IF_ERROR(
      loom_analyze_module_vector_reserve(allocator, &analyzer->modules, 1));
  const uint32_t module_index = (uint32_t)analyzer->modules.count++;
  loom_analyze_module_t* entry = &analyzer->modules.data[module_index];
  memset(entry, 0, sizeof(*entry));
  entry->module = module;
  entry->value_count = module->values.count;
  if (entry->value_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, entry->value_count, sizeof(uint64_t),
        (void**)&entry->value_tags));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, entry->value_count, sizeof(uint32_t),
        (void**)&entry->value_stamps));
  }

  loom_text_printer_t printer;
  loom_text_printer_initialize(module, &printer);
  loom_analyze_index_context_t context = {
      .analyzer = analyzer,
      .module = module,
      .module_index = module_index,
      .printer = &printer,
  };
  iree_status_t status = iree_ok_status();
  loom_region_t* body = module->body;
  for (uint16_t b = 0;
       body && b < body->block_count && iree_status_is_ok(status); ++b) {
    const loom_block_t* block = body->blocks[b];
    if (!block) {
      continue;
    }
    for (const loom_op_t* op = block->first_op; op && iree_status_is_ok(status);
         op = op->next_op) {
      context.symbol = loom_analyze_symbol_name(module, op);
      context.in_launch_config = false;
      status = loom_analyze_index_regions(&context, op, /*is_top_level=*/true);
    }
  }
  loom_text_printer_deinitialize(&printer);
  return status;
}

//===----------------------------------------------------------------------===//
// Canonical window encoding
//===----------------------------------------------------------------------===//

// Token kinds occupy the top byte; payloads use the remaining bits.
#define LOOM_ANALYZE_TOKEN_OP (UINT64_C(0xF1) << 56)
#define LOOM_ANALYZE_TOKEN_DEFS (UINT64_C(0xF2) << 56)
#define LOOM_ANALYZE_TOKEN_REGION (UINT64_C(0xF3) << 56)
#define LOOM_ANALYZE_TOKEN_BLOCK (UINT64_C(0xF4) << 56)
#define LOOM_ANALYZE_TOKEN_END_BLOCK (UINT64_C(0xF5) << 56)
#define LOOM_ANALYZE_TOKEN_END_OP (UINT64_C(0xF6) << 56)
#define LOOM_ANALYZE_TAG_DEFINED (UINT64_C(0xE1) << 56)
#define LOOM_ANALYZE_TAG_EXTERNAL (UINT64_C(0xE2) << 56)
#define LOOM_ANALYZE_TAG_UNKNOWN (UINT64_C(0xE3) << 56)

#define LOOM_ANALYZE_MAX_SORTED_OPERANDS 16

typedef struct loom_analyze_encoder_t {
  loom_analyze_analyzer_t* analyzer;
  loom_analyze_module_t* module_entry;
  const loom_module_t* module;
  // Index into loom_analyze_op_t::hashes selecting the active relaxations.
  uint32_t variant;
  uint32_t generation;
  uint64_t next_defined;
  uint64_t next_external;
  uint64_t hash;
  // Optional exact token record.
  loom_analyze_tokens_t* tokens;
} loom_analyze_encoder_t;

static iree_status_t loom_analyze_emit(loom_analyze_encoder_t* encoder,
                                       uint64_t token) {
  encoder->hash = loom_analyze_hash_u64(encoder->hash, token);
  loom_analyze_tokens_t* tokens = encoder->tokens;
  if (!tokens) {
    return iree_ok_status();
  }
  if (tokens->count == tokens->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        encoder->analyzer->allocator, tokens->count + 1, sizeof(uint64_t),
        &tokens->capacity, (void**)&tokens->data));
  }
  tokens->data[tokens->count++] = token;
  return iree_ok_status();
}

// Returns the window-relative tag of |value_id|, assigning the next external
// identity on first sight.
static uint64_t loom_analyze_value_tag(loom_analyze_encoder_t* encoder,
                                       loom_value_id_t value_id) {
  loom_analyze_module_t* entry = encoder->module_entry;
  if ((iree_host_size_t)value_id >= entry->value_count) {
    return LOOM_ANALYZE_TAG_UNKNOWN;
  }
  if (entry->value_stamps[value_id] == encoder->generation) {
    return entry->value_tags[value_id];
  }
  const uint64_t tag = LOOM_ANALYZE_TAG_EXTERNAL | encoder->next_external++;
  entry->value_stamps[value_id] = encoder->generation;
  entry->value_tags[value_id] = tag;
  return tag;
}

static void loom_analyze_define_value(loom_analyze_encoder_t* encoder,
                                      loom_value_id_t value_id) {
  loom_analyze_module_t* entry = encoder->module_entry;
  if ((iree_host_size_t)value_id >= entry->value_count) {
    return;
  }
  entry->value_stamps[value_id] = encoder->generation;
  entry->value_tags[value_id] =
      LOOM_ANALYZE_TAG_DEFINED | encoder->next_defined++;
}

static iree_status_t loom_analyze_emit_value_ref(loom_value_id_t value_id,
                                                 void* user_data) {
  loom_analyze_encoder_t* encoder = (loom_analyze_encoder_t*)user_data;
  return loom_analyze_emit(encoder, loom_analyze_value_tag(encoder, value_id));
}

static int loom_analyze_compare_u64(const void* lhs, const void* rhs) {
  const uint64_t a = *(const uint64_t*)lhs;
  const uint64_t b = *(const uint64_t*)rhs;
  return a < b ? -1 : (a > b ? 1 : 0);
}

static iree_status_t loom_analyze_encode_operation(
    loom_analyze_encoder_t* encoder, uint32_t* op_cursor,
    uint32_t* block_cursor) {
  loom_analyze_analyzer_t* analyzer = encoder->analyzer;
  const loom_analyze_op_t* entry = &analyzer->ops.data[(*op_cursor)++];
  const loom_op_t* op = entry->op;
  const loom_module_t* module = encoder->module;

  IREE_RETURN_IF_ERROR(loom_analyze_emit(encoder, LOOM_ANALYZE_TOKEN_OP));
  IREE_RETURN_IF_ERROR(
      loom_analyze_emit(encoder, entry->hashes[encoder->variant]));

  // SSA values embedded in operand types (dynamic dims such as
  // view<[%n]xf32>) come first: the printed op spells them, so they are inputs
  // of the sequence, and a motif parameter's type may only mention earlier
  // parameters.
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
        module, loom_module_value_type(module, operands[i]),
        loom_analyze_emit_value_ref, encoder));
  }

  // Operands. Commutative operands compare as a multiset.
  if (iree_any_bit_set(op->traits, LOOM_TRAIT_COMMUTATIVE) &&
      op->operand_count <= LOOM_ANALYZE_MAX_SORTED_OPERANDS) {
    uint64_t tags[LOOM_ANALYZE_MAX_SORTED_OPERANDS];
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      tags[i] = loom_analyze_value_tag(encoder, operands[i]);
    }
    qsort(tags, op->operand_count, sizeof(tags[0]), loom_analyze_compare_u64);
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_analyze_emit(encoder, tags[i]));
    }
  } else {
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_analyze_emit(
          encoder, loom_analyze_value_tag(encoder, operands[i])));
    }
  }

  // SSA values embedded in result types and attributes, in structural order.
  const loom_value_id_t* results = loom_op_results((loom_op_t*)op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
        module, loom_module_value_type(module, results[i]),
        loom_analyze_emit_value_ref, encoder));
  }
  const loom_attribute_t* attributes = loom_op_attrs((loom_op_t*)op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_walk_attribute_value_refs(
        module, attributes[i], loom_analyze_emit_value_ref, encoder));
  }

  IREE_RETURN_IF_ERROR(loom_analyze_emit(encoder, LOOM_ANALYZE_TOKEN_DEFS));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      loom_analyze_define_value(encoder, results[i]);
    }
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t r = 0; r < op->region_count; ++r) {
    const loom_region_t* region = regions[r];
    if (!region) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_analyze_emit(
        encoder, LOOM_ANALYZE_TOKEN_REGION | region->block_count));
    for (uint16_t b = 0; b < region->block_count; ++b) {
      const loom_block_t* block = region->blocks[b];
      if (!block) {
        continue;
      }
      const loom_analyze_block_t* block_entry =
          &analyzer->blocks.data[(*block_cursor)++];
      IREE_RETURN_IF_ERROR(
          loom_analyze_emit(encoder, LOOM_ANALYZE_TOKEN_BLOCK));
      IREE_RETURN_IF_ERROR(loom_analyze_emit(
          encoder,
          block_entry
              ->argument_hashes[(encoder->variant &
                                 LOOM_ANALYZE_PARAMETERIZE_TYPES) != 0]));
      for (uint16_t a = 0; a < block->arg_count; ++a) {
        const loom_value_id_t argument = block->arg_ids[a];
        IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
            module, loom_module_value_type(module, argument),
            loom_analyze_emit_value_ref, encoder));
        loom_analyze_define_value(encoder, argument);
      }
      for (const loom_op_t* child = block->first_op; child;
           child = child->next_op) {
        IREE_ASSERT(analyzer->ops.data[*op_cursor].op == child);
        IREE_RETURN_IF_ERROR(
            loom_analyze_encode_operation(encoder, op_cursor, block_cursor));
      }
      IREE_RETURN_IF_ERROR(
          loom_analyze_emit(encoder, LOOM_ANALYZE_TOKEN_END_BLOCK));
    }
  }
  return loom_analyze_emit(encoder, LOOM_ANALYZE_TOKEN_END_OP);
}

// Encodes |length| sibling operations of |sequence_index| starting at |start|.
// Returns the canonical hash, and records exact tokens when |tokens| is set.
static iree_status_t loom_analyze_encode_window(
    loom_analyze_analyzer_t* analyzer, uint32_t sequence_index, uint32_t start,
    uint32_t length, loom_analyze_tokens_t* tokens, uint64_t* out_hash) {
  const loom_analyze_sequence_t* sequence =
      &analyzer->sequences.data[sequence_index];
  loom_analyze_module_t* module_entry =
      &analyzer->modules.data[sequence->module_index];
  loom_analyze_encoder_t encoder = {
      .analyzer = analyzer,
      .module_entry = module_entry,
      .module = module_entry->module,
      .variant = analyzer->encode_variant,
      .generation = ++module_entry->generation,
      .hash = LOOM_ANALYZE_FNV_OFFSET,
      .tokens = tokens,
  };
  if (tokens) {
    tokens->count = 0;
  }
  uint32_t op_cursor = analyzer->sequence_ops.data[sequence->op_offset + start];
  uint32_t block_cursor = analyzer->ops.data[op_cursor].block_cursor;
  for (uint32_t i = 0; i < length; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_analyze_encode_operation(&encoder, &op_cursor, &block_cursor));
  }
  analyzer->last_window_input_count = (uint32_t)encoder.next_external;
  *out_hash = encoder.hash;
  return iree_ok_status();
}

static bool loom_analyze_tokens_equal(const loom_analyze_tokens_t* a,
                                      const loom_analyze_tokens_t* b) {
  return a->count == b->count &&
         (a->count == 0 ||
          memcmp(a->data, b->data, a->count * sizeof(uint64_t)) == 0);
}

// Compares the windows of two occurrences exactly.
static iree_status_t loom_analyze_windows_equal(
    loom_analyze_analyzer_t* analyzer, uint32_t sequence_a, uint32_t start_a,
    uint32_t sequence_b, uint32_t start_b, uint32_t length, bool* out_equal) {
  uint64_t hash_a = 0;
  uint64_t hash_b = 0;
  IREE_RETURN_IF_ERROR(loom_analyze_encode_window(
      analyzer, sequence_a, start_a, length, &analyzer->tokens_a, &hash_a));
  IREE_RETURN_IF_ERROR(loom_analyze_encode_window(
      analyzer, sequence_b, start_b, length, &analyzer->tokens_b, &hash_b));
  *out_equal =
      hash_a == hash_b &&
      loom_analyze_tokens_equal(&analyzer->tokens_a, &analyzer->tokens_b);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Clone detection
//===----------------------------------------------------------------------===//

static uint32_t loom_analyze_op_size(const loom_analyze_analyzer_t* analyzer,
                                     uint32_t op_index) {
  return analyzer->ops.data[op_index].subtree_end - op_index;
}

static uint32_t loom_analyze_sequence_op(
    const loom_analyze_analyzer_t* analyzer,
    const loom_analyze_sequence_t* sequence, uint32_t position) {
  return analyzer->sequence_ops.data[sequence->op_offset + position];
}

static int loom_analyze_compare_candidates(const void* lhs, const void* rhs) {
  const loom_analyze_candidate_t* a = (const loom_analyze_candidate_t*)lhs;
  const loom_analyze_candidate_t* b = (const loom_analyze_candidate_t*)rhs;
  if (a->hash != b->hash) {
    return a->hash < b->hash ? -1 : 1;
  }
  if (a->sequence_index != b->sequence_index) {
    return a->sequence_index < b->sequence_index ? -1 : 1;
  }
  if (a->start != b->start) {
    return a->start < b->start ? -1 : 1;
  }
  return 0;
}

// Emits one candidate per start position: the shortest window there whose
// total operation count reaches the minimum clone size.
static iree_status_t loom_analyze_collect_candidates(
    loom_analyze_analyzer_t* analyzer,
    loom_analyze_candidate_vector_t* candidates) {
  const uint32_t min_ops = analyzer->options.min_op_count;
  for (uint32_t s = 0; s < analyzer->sequences.count; ++s) {
    const loom_analyze_sequence_t* sequence = &analyzer->sequences.data[s];
    uint32_t end = 0;
    uint32_t window_ops = 0;
    for (uint32_t start = 0; start < sequence->length; ++start) {
      if (end < start) {
        end = start;
        window_ops = 0;
      }
      while (end < sequence->length && window_ops < min_ops) {
        window_ops += loom_analyze_op_size(
            analyzer, loom_analyze_sequence_op(analyzer, sequence, end));
        ++end;
      }
      if (window_ops < min_ops) {
        break;
      }
      IREE_RETURN_IF_ERROR(loom_analyze_candidate_vector_reserve(
          analyzer->allocator, candidates, 1));
      loom_analyze_candidate_t* candidate =
          &candidates->data[candidates->count++];
      candidate->sequence_index = s;
      candidate->start = start;
      candidate->length = end - start;
      IREE_RETURN_IF_ERROR(loom_analyze_encode_window(
          analyzer, s, start, candidate->length, NULL, &candidate->hash));
      window_ops -= loom_analyze_op_size(
          analyzer, loom_analyze_sequence_op(analyzer, sequence, start));
    }
  }
  return iree_ok_status();
}

typedef struct loom_analyze_detection_t {
  loom_analyze_class_vector_t classes;
  loom_analyze_class_occurrence_vector_t occurrences;
  // Scratch membership for one hash run.
  loom_analyze_index_vector_t pending;
  loom_analyze_index_vector_t group;
} loom_analyze_detection_t;

static void loom_analyze_detection_free(iree_allocator_t allocator,
                                        loom_analyze_detection_t* detection) {
  loom_analyze_class_vector_free(allocator, &detection->classes);
  loom_analyze_class_occurrence_vector_free(allocator, &detection->occurrences);
  loom_analyze_index_vector_free(allocator, &detection->pending);
  loom_analyze_index_vector_free(allocator, &detection->group);
}

// Returns true when |user|, lifted to its ancestor in |block|, follows |last|
// or lies outside |block|.
static bool loom_analyze_used_after(const loom_op_t* user,
                                    const loom_block_t* block,
                                    const loom_op_t* last) {
  while (user && user->parent_block != block) {
    user = user->parent_op;
  }
  return !user || user->block_ordinal > last->block_ordinal;
}

// Returns the operation owning |value_id|: its defining operation, or the
// operation whose region holds the block it is an argument of.
static const loom_op_t* loom_analyze_value_owner(const loom_module_t* module,
                                                 loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    const loom_block_t* block = loom_value_def_block(value);
    return block && block->parent_region ? block->parent_region->parent_op
                                         : NULL;
  }
  return loom_value_owner_op(value);
}

// Counts values defined by the window's top-level operations that are used
// after the window: the results a motif replacing the window must return.
// Nested values cannot escape their regions except through the owning op's
// results, which are top-level values themselves.
// When |out_values| is non-NULL, the first |capacity| escaping values are
// stored in definition order. Returns the total count.
static uint32_t loom_analyze_window_outputs(
    const loom_analyze_analyzer_t* analyzer, uint32_t sequence_index,
    uint32_t start, uint32_t length, loom_value_id_t* out_values,
    uint32_t capacity) {
  const loom_analyze_sequence_t* sequence =
      &analyzer->sequences.data[sequence_index];
  const loom_module_t* module =
      analyzer->modules.data[sequence->module_index].module;
  const loom_op_t* last = analyzer->ops
                              .data[loom_analyze_sequence_op(
                                  analyzer, sequence, start + length - 1)]
                              .op;
  const loom_block_t* block = last->parent_block;
  uint32_t count = 0;
  for (uint32_t i = 0; i < length; ++i) {
    const loom_op_t* op =
        analyzer->ops
            .data[loom_analyze_sequence_op(analyzer, sequence, start + i)]
            .op;
    const loom_value_id_t* results = loom_op_results(op);
    for (uint16_t r = 0; r < op->result_count; ++r) {
      if (results[r] == LOOM_VALUE_ID_INVALID) {
        continue;
      }
      const loom_value_t* value = loom_module_value(module, results[r]);
      const loom_use_t* uses = loom_value_uses(value);
      bool escapes = false;
      for (uint32_t u = 0; u < value->use_count && !escapes; ++u) {
        escapes =
            loom_analyze_used_after(loom_use_user_op(uses[u]), block, last);
      }
      // Later types (view<[%n]xf32>) and attributes may name the value too.
      loom_type_use_iterator_t iterator;
      if (!escapes) {
        loom_type_users_begin(&module->type_uses, results[r], &iterator);
        for (loom_value_id_t carrier = loom_type_users_next(&iterator);
             !escapes && carrier != LOOM_VALUE_ID_INVALID;
             carrier = loom_type_users_next(&iterator)) {
          escapes = loom_analyze_used_after(
              loom_analyze_value_owner(module, carrier), block, last);
        }
      }
      if (!escapes) {
        loom_attribute_users_begin(&module->type_uses, results[r], &iterator);
        for (loom_attribute_user_t user = loom_attribute_users_next(&iterator);
             !escapes && user.op; user = loom_attribute_users_next(&iterator)) {
          escapes = loom_analyze_used_after(user.op, block, last);
        }
      }
      if (escapes) {
        if (out_values && count < capacity) {
          out_values[count] = results[r];
        }
        ++count;
      }
    }
  }
  return count;
}

// Returns the longest length every occurrence of |group| can span without
// leaving its block or running into the next occurrence in the same block.
static uint32_t loom_analyze_group_max_length(
    const loom_analyze_analyzer_t* analyzer,
    const loom_analyze_candidate_t* candidates, const uint32_t* group,
    uint32_t count) {
  uint32_t max_length = UINT32_MAX;
  for (uint32_t i = 0; i < count; ++i) {
    const loom_analyze_candidate_t* occurrence = &candidates[group[i]];
    const loom_analyze_sequence_t* sequence =
        &analyzer->sequences.data[occurrence->sequence_index];
    uint32_t limit = sequence->length - occurrence->start;
    if (i + 1 < count) {
      const loom_analyze_candidate_t* next = &candidates[group[i + 1]];
      if (next->sequence_index == occurrence->sequence_index &&
          next->start - occurrence->start < limit) {
        limit = next->start - occurrence->start;
      }
    }
    if (limit < max_length) {
      max_length = limit;
    }
  }
  return max_length;
}

// Returns whether every occurrence of |group| matches the first at |length|.
static iree_status_t loom_analyze_group_equal_at(
    loom_analyze_analyzer_t* analyzer,
    const loom_analyze_candidate_t* candidates, const uint32_t* group,
    uint32_t count, uint32_t length, bool* out_equal) {
  const loom_analyze_candidate_t* representative = &candidates[group[0]];
  *out_equal = true;
  for (uint32_t i = 1; i < count && *out_equal; ++i) {
    const loom_analyze_candidate_t* other = &candidates[group[i]];
    IREE_RETURN_IF_ERROR(loom_analyze_windows_equal(
        analyzer, representative->sequence_index, representative->start,
        other->sequence_index, other->start, length, out_equal));
  }
  return iree_ok_status();
}

// Turns one exact-equivalence group into a maximal clone class.
static iree_status_t loom_analyze_record_group(
    loom_analyze_analyzer_t* analyzer,
    const loom_analyze_candidate_t* candidates,
    loom_analyze_detection_t* detection) {
  iree_allocator_t allocator = analyzer->allocator;
  uint32_t* group = detection->group.data;
  // Drop occurrences overlapping an earlier kept one in the same block.
  // Candidates are sorted by sequence and start within a hash run.
  uint32_t kept = 0;
  for (uint32_t i = 0; i < detection->group.count; ++i) {
    const loom_analyze_candidate_t* candidate = &candidates[group[i]];
    if (kept > 0) {
      const loom_analyze_candidate_t* previous = &candidates[group[kept - 1]];
      if (previous->sequence_index == candidate->sequence_index &&
          previous->start + previous->length > candidate->start) {
        continue;
      }
    }
    group[kept++] = group[i];
  }
  if (kept < analyzer->options.min_occurrence_count) {
    return iree_ok_status();
  }

  // Extend all occurrences forward to the longest common length. A window's
  // encoding is a prefix of every longer window's encoding at the same start,
  // so equality is prefix-closed: gallop to bracket the first mismatch, then
  // binary search. This keeps long copies at O(L log L) encoding work.
  const loom_analyze_candidate_t* representative = &candidates[group[0]];
  const uint32_t max_length =
      loom_analyze_group_max_length(analyzer, candidates, group, kept);
  uint32_t good = representative->length;
  uint32_t bad = max_length + 1;
  uint32_t step = 1;
  while (good < max_length) {
    const uint32_t probe = max_length - good < step ? max_length : good + step;
    bool equal = false;
    IREE_RETURN_IF_ERROR(loom_analyze_group_equal_at(
        analyzer, candidates, group, kept, probe, &equal));
    if (!equal) {
      bad = probe;
      break;
    }
    good = probe;
    step *= 2;
  }
  while (bad - good > 1) {
    const uint32_t middle = good + (bad - good) / 2;
    bool equal = false;
    IREE_RETURN_IF_ERROR(loom_analyze_group_equal_at(
        analyzer, candidates, group, kept, middle, &equal));
    if (equal) {
      good = middle;
    } else {
      bad = middle;
    }
  }
  const uint32_t length = good;

  uint64_t fingerprint = 0;
  IREE_RETURN_IF_ERROR(loom_analyze_encode_window(
      analyzer, representative->sequence_index, representative->start, length,
      NULL, &fingerprint));
  const uint32_t input_count = analyzer->last_window_input_count;
  const uint32_t output_count = loom_analyze_window_outputs(
      analyzer, representative->sequence_index, representative->start, length,
      /*out_values=*/NULL, /*capacity=*/0);

  // Classify the weakest relaxation under which every copy still matches, so
  // exact clones found with relaxations enabled stay applicable.
  const uint32_t configured = analyzer->encode_variant;
  loom_analyze_parameterization_t parameterized_by = configured;
  if (configured != LOOM_ANALYZE_PARAMETERIZE_NONE) {
    static const uint32_t kWeakerVariants[] = {
        LOOM_ANALYZE_PARAMETERIZE_NONE,
        LOOM_ANALYZE_PARAMETERIZE_CONSTANTS,
        LOOM_ANALYZE_PARAMETERIZE_TYPES,
    };
    iree_status_t status = iree_ok_status();
    for (size_t v = 0; v < IREE_ARRAYSIZE(kWeakerVariants); ++v) {
      const uint32_t variant = kWeakerVariants[v];
      if (variant == configured || (variant & ~configured) != 0) {
        continue;
      }
      analyzer->encode_variant = variant;
      bool equal = false;
      status = loom_analyze_group_equal_at(analyzer, candidates, group, kept,
                                           length, &equal);
      if (!iree_status_is_ok(status) || equal) {
        parameterized_by = variant;
        break;
      }
    }
    analyzer->encode_variant = configured;
    IREE_RETURN_IF_ERROR(status);
  }

  IREE_RETURN_IF_ERROR(
      loom_analyze_class_vector_reserve(allocator, &detection->classes, 1));
  IREE_RETURN_IF_ERROR(loom_analyze_class_occurrence_vector_reserve(
      allocator, &detection->occurrences, kept));
  loom_analyze_class_t* clone_class =
      &detection->classes.data[detection->classes.count++];
  clone_class->fingerprint = fingerprint;
  clone_class->length = length;
  clone_class->occurrence_offset = (uint32_t)detection->occurrences.count;
  clone_class->occurrence_count = kept;
  clone_class->input_count = input_count;
  clone_class->output_count = output_count;
  clone_class->parameterized_by = parameterized_by;
  clone_class->op_count = 0;
  for (uint32_t i = 0; i < kept; ++i) {
    const loom_analyze_candidate_t* candidate = &candidates[group[i]];
    const loom_analyze_sequence_t* sequence =
        &analyzer->sequences.data[candidate->sequence_index];
    loom_analyze_class_occurrence_t* occurrence =
        &detection->occurrences.data[detection->occurrences.count++];
    occurrence->sequence_index = candidate->sequence_index;
    occurrence->start = candidate->start;
    occurrence->op_begin =
        loom_analyze_sequence_op(analyzer, sequence, candidate->start);
    const uint32_t last_op = loom_analyze_sequence_op(
        analyzer, sequence, candidate->start + length - 1);
    occurrence->op_end = analyzer->ops.data[last_op].subtree_end;
    if (i == 0) {
      clone_class->op_count = occurrence->op_end - occurrence->op_begin;
    }
  }

  // Copies matching only up to constants differ in constant literals; compare
  // the exact hash of each operation, in the shared pre-order, against the
  // first copy.
  clone_class->call_site_constant_count = 0;
  clone_class->nested_constant_differs = false;
  if (parameterized_by == LOOM_ANALYZE_PARAMETERIZE_CONSTANTS) {
    const loom_analyze_class_occurrence_t* first =
        &detection->occurrences.data[clone_class->occurrence_offset];
    const loom_block_t* block =
        analyzer->ops.data[first->op_begin].op->parent_block;
    for (uint32_t k = 0; k < clone_class->op_count; ++k) {
      const loom_analyze_op_t* reference =
          &analyzer->ops.data[first->op_begin + k];
      bool differs = false;
      for (uint32_t i = 1; i < kept && !differs; ++i) {
        const loom_analyze_class_occurrence_t* other =
            &detection->occurrences.data[clone_class->occurrence_offset + i];
        differs = analyzer->ops.data[other->op_begin + k].hashes[0] !=
                  reference->hashes[0];
      }
      if (!differs) {
        continue;
      }
      if (reference->op->parent_block == block) {
        ++clone_class->call_site_constant_count;
      } else {
        clone_class->nested_constant_differs = true;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_analyze_detect_classes(
    loom_analyze_analyzer_t* analyzer,
    loom_analyze_candidate_vector_t* candidates,
    loom_analyze_detection_t* detection) {
  iree_allocator_t allocator = analyzer->allocator;
  if (candidates->count > 1) {
    qsort(candidates->data, candidates->count, sizeof(candidates->data[0]),
          loom_analyze_compare_candidates);
  }
  const uint32_t min_occurrences = analyzer->options.min_occurrence_count;
  iree_host_size_t run_begin = 0;
  while (run_begin < candidates->count) {
    iree_host_size_t run_end = run_begin + 1;
    while (run_end < candidates->count &&
           candidates->data[run_end].hash == candidates->data[run_begin].hash) {
      ++run_end;
    }
    if (run_end - run_begin >= min_occurrences) {
      // Partition the hash run into exact-equivalence groups.
      detection->pending.count = 0;
      IREE_RETURN_IF_ERROR(loom_analyze_index_vector_reserve(
          allocator, &detection->pending, run_end - run_begin));
      for (iree_host_size_t i = run_begin; i < run_end; ++i) {
        detection->pending.data[detection->pending.count++] = (uint32_t)i;
      }
      while (detection->pending.count >= min_occurrences) {
        detection->group.count = 0;
        IREE_RETURN_IF_ERROR(loom_analyze_index_vector_reserve(
            allocator, &detection->group, detection->pending.count));
        const uint32_t representative = detection->pending.data[0];
        const loom_analyze_candidate_t* rep = &candidates->data[representative];
        detection->group.data[detection->group.count++] = representative;
        iree_host_size_t remaining = 0;
        for (iree_host_size_t i = 1; i < detection->pending.count; ++i) {
          const uint32_t index = detection->pending.data[i];
          const loom_analyze_candidate_t* other = &candidates->data[index];
          bool equal = other->length == rep->length;
          if (equal) {
            IREE_RETURN_IF_ERROR(loom_analyze_windows_equal(
                analyzer, rep->sequence_index, rep->start,
                other->sequence_index, other->start, rep->length, &equal));
          }
          if (equal) {
            detection->group.data[detection->group.count++] = index;
          } else {
            detection->pending.data[remaining++] = index;
          }
        }
        detection->pending.count = remaining;
        if (detection->group.count >= min_occurrences) {
          IREE_RETURN_IF_ERROR(
              loom_analyze_record_group(analyzer, candidates->data, detection));
        }
      }
    }
    run_begin = run_end;
  }
  return iree_ok_status();
}

// Class table consulted by loom_analyze_compare_class_size during one qsort
// call. The analyzer is single-threaded and runs once, so a file-scope binding
// keeps the comparator within the C standard qsort contract.
static const loom_analyze_class_t* loom_analyze_sort_classes;

static int loom_analyze_compare_class_size(const void* lhs, const void* rhs) {
  const loom_analyze_class_t* a =
      &loom_analyze_sort_classes[*(const uint32_t*)lhs];
  const loom_analyze_class_t* b =
      &loom_analyze_sort_classes[*(const uint32_t*)rhs];
  if (a->op_count != b->op_count) {
    return a->op_count > b->op_count ? -1 : 1;
  }
  if (a->occurrence_count != b->occurrence_count) {
    return a->occurrence_count > b->occurrence_count ? -1 : 1;
  }
  if (a->occurrence_offset != b->occurrence_offset) {
    return a->occurrence_offset < b->occurrence_offset ? -1 : 1;
  }
  return 0;
}

// Returns true when every occurrence of |inner| lies within an occurrence of
// |outer|. Global operation ranges never span modules.
static bool loom_analyze_class_contained(
    const loom_analyze_detection_t* detection,
    const loom_analyze_class_t* inner, const loom_analyze_class_t* outer) {
  const loom_analyze_class_occurrence_t* occurrences =
      detection->occurrences.data;
  for (uint32_t i = 0; i < inner->occurrence_count; ++i) {
    const loom_analyze_class_occurrence_t* a =
        &occurrences[inner->occurrence_offset + i];
    bool contained = false;
    for (uint32_t j = 0; j < outer->occurrence_count && !contained; ++j) {
      const loom_analyze_class_occurrence_t* b =
          &occurrences[outer->occurrence_offset + j];
      contained = b->op_begin <= a->op_begin && a->op_end <= b->op_end;
    }
    if (!contained) {
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Results
//===----------------------------------------------------------------------===//

static void loom_analyze_resolve_occurrence(
    const loom_analyze_analyzer_t* analyzer,
    const loom_analyze_class_occurrence_t* source, uint32_t length,
    loom_analyze_occurrence_t* out_occurrence) {
  memset(out_occurrence, 0, sizeof(*out_occurrence));
  const loom_analyze_sequence_t* sequence =
      &analyzer->sequences.data[source->sequence_index];
  const loom_module_t* module =
      analyzer->modules.data[sequence->module_index].module;
  const loom_op_t* first =
      analyzer->ops
          .data[loom_analyze_sequence_op(analyzer, sequence, source->start)]
          .op;
  const loom_op_t* last =
      analyzer->ops
          .data[loom_analyze_sequence_op(analyzer, sequence,
                                         source->start + length - 1)]
          .op;
  const loom_source_resolver_t resolver = {NULL, NULL};
  loom_source_range_t range;
  if (loom_source_resolve(resolver, module, first->location, &range)) {
    out_occurrence->file = range.filename;
    out_occurrence->start_line = range.start_line;
    out_occurrence->start_column = range.start_column;
    out_occurrence->end_line = range.start_line;
  }
  if (loom_source_resolve(resolver, module, last->location, &range)) {
    if (iree_string_view_is_empty(out_occurrence->file)) {
      out_occurrence->file = range.filename;
    }
    if (range.end_line >= range.start_line && range.end_line > 0) {
      out_occurrence->end_line = range.end_line;
      out_occurrence->end_column = range.end_column;
    } else {
      out_occurrence->end_line = range.start_line;
    }
  }
  out_occurrence->symbol = sequence->symbol;
  out_occurrence->in_launch_config = sequence->in_launch_config;
}

// Prints |type| into an analyzer-owned string.
static iree_status_t loom_analyze_print_type(loom_analyze_analyzer_t* analyzer,
                                             loom_text_printer_t* printer,
                                             loom_type_t type,
                                             iree_string_view_t* out_text) {
  *out_text = iree_string_view_empty();
  iree_string_builder_t* scratch = &analyzer->scratch;
  iree_string_builder_reset(scratch);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(scratch, &stream);
  IREE_RETURN_IF_ERROR(
      loom_text_printer_write_type(printer, type, &stream, NULL));
  const iree_string_view_t text = iree_string_builder_view(scratch);
  IREE_RETURN_IF_ERROR(loom_analyze_string_vector_reserve(
      analyzer->allocator, &analyzer->strings, 1));
  char* copy = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(analyzer->allocator, text.size + 1, (void**)&copy));
  if (text.size > 0) {
    memcpy(copy, text.data, text.size);
  }
  copy[text.size] = 0;
  analyzer->strings.data[analyzer->strings.count++] = copy;
  *out_text = iree_make_string_view(copy, text.size);
  return iree_ok_status();
}

#define LOOM_ANALYZE_TAG_KIND_MASK (UINT64_C(0xFF) << 56)

// Returns the definition ordinal of a value defined inside the most recently
// encoded window of |module_entry|, or UINT32_MAX.
static uint32_t loom_analyze_defined_ordinal(
    const loom_analyze_module_t* module_entry, loom_value_id_t value_id) {
  if ((iree_host_size_t)value_id >= module_entry->value_count ||
      module_entry->value_stamps[value_id] != module_entry->generation) {
    return UINT32_MAX;
  }
  const uint64_t tag = module_entry->value_tags[value_id];
  if ((tag & LOOM_ANALYZE_TAG_KIND_MASK) != LOOM_ANALYZE_TAG_DEFINED) {
    return UINT32_MAX;
  }
  return (uint32_t)(tag & ~LOOM_ANALYZE_TAG_KIND_MASK);
}

static int loom_analyze_compare_u32(const void* lhs, const void* rhs) {
  const uint32_t a = *(const uint32_t*)lhs;
  const uint32_t b = *(const uint32_t*)rhs;
  return a < b ? -1 : (a > b ? 1 : 0);
}

// Collects the motif results of |clone_class|: the definition ordinals of
// values used after the sequence in any occurrence, sorted. Equal encodings
// give every occurrence the same ordinal for the same value, so one motif
// signature serves all copies even when each copy uses a different subset.
static iree_status_t loom_analyze_collect_output_ordinals(
    loom_analyze_analyzer_t* analyzer,
    const loom_analyze_detection_t* detection,
    const loom_analyze_class_t* clone_class,
    loom_analyze_index_vector_t* out_ordinals) {
  iree_allocator_t allocator = analyzer->allocator;
  out_ordinals->count = 0;
  loom_value_id_t* escaping = NULL;
  iree_host_size_t escaping_capacity = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < clone_class->occurrence_count && iree_status_is_ok(status); ++i) {
    const loom_analyze_class_occurrence_t* source =
        &detection->occurrences.data[clone_class->occurrence_offset + i];
    uint64_t hash = 0;
    status = loom_analyze_encode_window(analyzer, source->sequence_index,
                                        source->start, clone_class->length,
                                        NULL, &hash);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const uint32_t count = loom_analyze_window_outputs(
        analyzer, source->sequence_index, source->start, clone_class->length,
        NULL, 0);
    if (count > escaping_capacity) {
      status =
          iree_allocator_grow_array(allocator, count, sizeof(loom_value_id_t),
                                    &escaping_capacity, (void**)&escaping);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    loom_analyze_window_outputs(analyzer, source->sequence_index, source->start,
                                clone_class->length, escaping, count);
    const loom_analyze_module_t* module_entry =
        &analyzer->modules.data[analyzer->sequences.data[source->sequence_index]
                                    .module_index];
    for (uint32_t j = 0; j < count && iree_status_is_ok(status); ++j) {
      const uint32_t ordinal =
          loom_analyze_defined_ordinal(module_entry, escaping[j]);
      if (ordinal == UINT32_MAX) {
        continue;
      }
      bool present = false;
      for (iree_host_size_t k = 0; k < out_ordinals->count && !present; ++k) {
        present = out_ordinals->data[k] == ordinal;
      }
      if (present) {
        continue;
      }
      status = loom_analyze_index_vector_reserve(allocator, out_ordinals, 1);
      if (iree_status_is_ok(status)) {
        out_ordinals->data[out_ordinals->count++] = ordinal;
      }
    }
  }
  iree_allocator_free(allocator, escaping);
  if (iree_status_is_ok(status) && out_ordinals->count > 1) {
    qsort(out_ordinals->data, out_ordinals->count, sizeof(uint32_t),
          loom_analyze_compare_u32);
  }
  return status;
}

// Records the values crossing one occurrence's boundary: its inputs in
// canonical parameter order and the values at |output_ordinals| (the class's
// motif results) in definition order. Re-encoding the window leaves each
// boundary value stamped with the current generation and tagged with its
// parameter index or definition ordinal.
static iree_status_t loom_analyze_bind_occurrence(
    loom_analyze_analyzer_t* analyzer, loom_text_printer_t* printers,
    const loom_analyze_class_occurrence_t* source, uint32_t length,
    const loom_analyze_index_vector_t* output_ordinals,
    loom_analyze_occurrence_t* occurrence) {
  iree_allocator_t allocator = analyzer->allocator;
  const loom_analyze_sequence_t* sequence =
      &analyzer->sequences.data[source->sequence_index];
  const loom_analyze_module_t* module_entry =
      &analyzer->modules.data[sequence->module_index];
  const loom_module_t* module = module_entry->module;
  loom_text_printer_t* printer = &printers[sequence->module_index];

  uint64_t hash = 0;
  IREE_RETURN_IF_ERROR(loom_analyze_encode_window(
      analyzer, source->sequence_index, source->start, length, NULL, &hash));
  const uint32_t input_count = analyzer->last_window_input_count;
  const uint32_t generation = module_entry->generation;
  const uint32_t output_count = (uint32_t)output_ordinals->count;
  const uint32_t total = input_count + output_count;
  if (total == 0) {
    return iree_ok_status();
  }

  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, total, sizeof(loom_value_id_t), (void**)&values));
  for (uint32_t i = 0; i < total; ++i) {
    values[i] = LOOM_VALUE_ID_INVALID;
  }
  for (iree_host_size_t v = 0; v < module_entry->value_count; ++v) {
    if (module_entry->value_stamps[v] != generation) {
      continue;
    }
    const uint64_t tag = module_entry->value_tags[v];
    const uint64_t kind = tag & LOOM_ANALYZE_TAG_KIND_MASK;
    const uint64_t payload = tag & ~LOOM_ANALYZE_TAG_KIND_MASK;
    if (kind == LOOM_ANALYZE_TAG_EXTERNAL && payload < input_count) {
      values[payload] = (loom_value_id_t)v;
    } else if (kind == LOOM_ANALYZE_TAG_DEFINED) {
      const uint32_t* found = (const uint32_t*)bsearch(
          &(uint32_t){(uint32_t)payload}, output_ordinals->data, output_count,
          sizeof(uint32_t), loom_analyze_compare_u32);
      if (found) {
        values[input_count + (found - output_ordinals->data)] =
            (loom_value_id_t)v;
      }
    }
  }

  loom_analyze_binding_t* bindings = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, total, sizeof(loom_analyze_binding_t), (void**)&bindings);
  for (uint32_t i = 0; i < total && iree_status_is_ok(status); ++i) {
    if (values[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    bindings[i].name = loom_module_value_name(module, values[i]);
    status = loom_analyze_print_type(analyzer, printer,
                                     loom_module_value_type(module, values[i]),
                                     &bindings[i].type);
  }
  iree_allocator_free(allocator, values);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, bindings);
    return status;
  }
  occurrence->inputs = bindings;
  occurrence->input_count = input_count;
  occurrence->outputs = bindings + input_count;
  occurrence->output_count = output_count;
  return iree_ok_status();
}

static int loom_analyze_compare_locations(const loom_analyze_occurrence_t* a,
                                          const loom_analyze_occurrence_t* b) {
  const int file_order = iree_string_view_compare(a->file, b->file);
  if (file_order != 0) {
    return file_order;
  }
  if (a->start_line != b->start_line) {
    return a->start_line < b->start_line ? -1 : 1;
  }
  if (a->start_column != b->start_column) {
    return a->start_column < b->start_column ? -1 : 1;
  }
  return 0;
}

static int loom_analyze_compare_occurrences(const void* lhs, const void* rhs) {
  return loom_analyze_compare_locations((const loom_analyze_occurrence_t*)lhs,
                                        (const loom_analyze_occurrence_t*)rhs);
}

static int loom_analyze_compare_findings(const void* lhs, const void* rhs) {
  const loom_analyze_finding_t* a = (const loom_analyze_finding_t*)lhs;
  const loom_analyze_finding_t* b = (const loom_analyze_finding_t*)rhs;
  const int order =
      loom_analyze_compare_locations(&a->occurrences[0], &b->occurrences[0]);
  if (order != 0) {
    return order;
  }
  if (a->op_count != b->op_count) {
    return a->op_count > b->op_count ? -1 : 1;
  }
  if (a->fingerprint != b->fingerprint) {
    return a->fingerprint < b->fingerprint ? -1 : 1;
  }
  return 0;
}

// Findings consulted by loom_analyze_compare_savings during one qsort call.
static const loom_analyze_finding_t* loom_analyze_sort_findings;

static int loom_analyze_compare_savings(const void* lhs, const void* rhs) {
  const uint32_t a_index = *(const uint32_t*)lhs;
  const uint32_t b_index = *(const uint32_t*)rhs;
  const loom_analyze_finding_t* a = &loom_analyze_sort_findings[a_index];
  const loom_analyze_finding_t* b = &loom_analyze_sort_findings[b_index];
  if (a->removable_op_count != b->removable_op_count) {
    return a->removable_op_count > b->removable_op_count ? -1 : 1;
  }
  return a_index < b_index ? -1 : (a_index > b_index ? 1 : 0);
}

static int loom_analyze_compare_type_generic_savings(const void* lhs,
                                                     const void* rhs) {
  const uint32_t a_index = *(const uint32_t*)lhs;
  const uint32_t b_index = *(const uint32_t*)rhs;
  const loom_analyze_finding_t* a = &loom_analyze_sort_findings[a_index];
  const loom_analyze_finding_t* b = &loom_analyze_sort_findings[b_index];
  if (a->type_generic_removable_op_count !=
      b->type_generic_removable_op_count) {
    return a->type_generic_removable_op_count >
                   b->type_generic_removable_op_count
               ? -1
               : 1;
  }
  return a_index < b_index ? -1 : (a_index > b_index ? 1 : 0);
}

// Claims the operations of every callable copy of |clone_class| when none is
// already claimed. Returns false and claims nothing on overlap.
static bool loom_analyze_claim_callable_ops(
    const loom_analyze_analyzer_t* analyzer,
    const loom_analyze_detection_t* detection,
    const loom_analyze_class_t* clone_class, uint8_t* claimed) {
  const loom_analyze_class_occurrence_t* occurrences =
      &detection->occurrences.data[clone_class->occurrence_offset];
  for (uint32_t j = 0; j < clone_class->occurrence_count; ++j) {
    if (analyzer->sequences.data[occurrences[j].sequence_index]
            .in_launch_config) {
      continue;
    }
    for (uint32_t op = occurrences[j].op_begin; op < occurrences[j].op_end;
         ++op) {
      if (claimed[op]) {
        return false;
      }
    }
  }
  for (uint32_t j = 0; j < clone_class->occurrence_count; ++j) {
    if (analyzer->sequences.data[occurrences[j].sequence_index]
            .in_launch_config) {
      continue;
    }
    memset(&claimed[occurrences[j].op_begin], 1,
           occurrences[j].op_end - occurrences[j].op_begin);
  }
  return true;
}

// Selects the refactor plan: largest savings first, a finding joins the plan
// only when none of its callable copies touches an operation already claimed
// by a selected finding. Overlapping alternatives, such as the phase-shifted
// findings an unrolled loop produces, therefore contribute once. Findings must
// still be index-aligned with |kept_classes|.
static iree_status_t loom_analyze_select_plan(
    loom_analyze_analyzer_t* analyzer,
    const loom_analyze_detection_t* detection, const uint32_t* kept_classes) {
  iree_allocator_t allocator = analyzer->allocator;
  loom_analyze_result_t* result = &analyzer->result;
  const uint32_t finding_count = (uint32_t)result->finding_count;
  if (finding_count == 0 || analyzer->ops.count == 0) {
    return iree_ok_status();
  }
  uint32_t* order = NULL;
  uint8_t* claimed = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, finding_count, sizeof(uint32_t), (void**)&order);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, analyzer->ops.count,
                                         sizeof(uint8_t), (void**)&claimed);
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < finding_count; ++i) {
      order[i] = i;
    }
    loom_analyze_sort_findings = result->findings;
    qsort(order, finding_count, sizeof(order[0]), loom_analyze_compare_savings);
    loom_analyze_sort_findings = NULL;
    for (uint32_t i = 0; i < finding_count; ++i) {
      loom_analyze_finding_t* finding = &result->findings[order[i]];
      if (finding->removable_op_count == 0) {
        continue;
      }
      const loom_analyze_class_t* clone_class =
          &detection->classes.data[kept_classes[order[i]]];
      if (!loom_analyze_claim_callable_ops(analyzer, detection, clone_class,
                                           claimed)) {
        continue;
      }
      finding->selected = true;
      ++result->summary.selected_count;
      result->summary.removable_op_count += finding->removable_op_count;
    }

    // Second pass: what type-generic motifs would add on top of the plan.
    loom_analyze_sort_findings = result->findings;
    qsort(order, finding_count, sizeof(order[0]),
          loom_analyze_compare_type_generic_savings);
    loom_analyze_sort_findings = NULL;
    for (uint32_t i = 0; i < finding_count; ++i) {
      loom_analyze_finding_t* finding = &result->findings[order[i]];
      if (finding->applicable ||
          finding->type_generic_removable_op_count == 0) {
        continue;
      }
      const loom_analyze_class_t* clone_class =
          &detection->classes.data[kept_classes[order[i]]];
      if (loom_analyze_claim_callable_ops(analyzer, detection, clone_class,
                                          claimed)) {
        result->summary.type_generic_removable_op_count +=
            finding->type_generic_removable_op_count;
      }
    }
  }
  iree_allocator_free(allocator, claimed);
  iree_allocator_free(allocator, order);
  return status;
}

static iree_status_t loom_analyze_build_findings(
    loom_analyze_analyzer_t* analyzer,
    const loom_analyze_detection_t* detection, const uint32_t* kept_classes,
    uint32_t kept_count) {
  iree_allocator_t allocator = analyzer->allocator;
  loom_analyze_result_t* result = &analyzer->result;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, kept_count, sizeof(loom_analyze_finding_t),
        (void**)&result->findings));
  }
  // One printer per module spells binding types with module-local names.
  loom_text_printer_t* printers = NULL;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, analyzer->modules.count, sizeof(loom_text_printer_t),
        (void**)&printers));
    for (iree_host_size_t m = 0; m < analyzer->modules.count; ++m) {
      loom_text_printer_initialize(analyzer->modules.data[m].module,
                                   &printers[m]);
    }
  }
  iree_status_t status = iree_ok_status();
  loom_analyze_index_vector_t output_ordinals = {0};
  for (uint32_t k = 0; k < kept_count && iree_status_is_ok(status); ++k) {
    const loom_analyze_class_t* clone_class =
        &detection->classes.data[kept_classes[k]];
    loom_analyze_finding_t* finding =
        &result->findings[result->finding_count++];
    finding->rule = LOOM_ANALYZE_RULE_DUPLICATE_SEQUENCE;
    finding->op_count = clone_class->op_count;
    finding->fingerprint = clone_class->fingerprint;
    finding->parameterized_by = clone_class->parameterized_by;
    // Loom motifs cannot be generic over element types.
    finding->applicable =
        (clone_class->parameterized_by & LOOM_ANALYZE_PARAMETERIZE_TYPES) == 0;
    finding->occurrence_count = clone_class->occurrence_count;
    status = iree_allocator_malloc_array(
        allocator, clone_class->occurrence_count,
        sizeof(loom_analyze_occurrence_t), (void**)&finding->occurrences);
    if (!iree_status_is_ok(status)) {
      finding->occurrence_count = 0;
      break;
    }

    status = loom_analyze_collect_output_ordinals(
        analyzer, detection, clone_class, &output_ordinals);
    if (!iree_status_is_ok(status)) {
      break;
    }

    // An occurrence covering a whole definition body makes the other copies
    // reimplementations of that motif.
    uint32_t motif_index = UINT32_MAX;
    for (uint32_t i = 0;
         i < clone_class->occurrence_count && iree_status_is_ok(status); ++i) {
      const loom_analyze_class_occurrence_t* source =
          &detection->occurrences.data[clone_class->occurrence_offset + i];
      loom_analyze_resolve_occurrence(analyzer, source, clone_class->length,
                                      &finding->occurrences[i]);
      status = loom_analyze_bind_occurrence(
          analyzer, printers, source, clone_class->length, &output_ordinals,
          &finding->occurrences[i]);
      const loom_analyze_sequence_t* sequence =
          &analyzer->sequences.data[source->sequence_index];
      if (motif_index == UINT32_MAX && sequence->motif_length > 0 &&
          source->start == 0 && clone_class->length == sequence->motif_length) {
        motif_index = i;
      }
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (motif_index != UINT32_MAX) {
      finding->rule = LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF;
      finding->occurrences[motif_index].is_motif_definition = true;
      finding->motif_symbol = finding->occurrences[motif_index].symbol;
    }
    finding->input_count = clone_class->input_count;
    finding->call_site_constant_count = clone_class->call_site_constant_count;
    finding->output_count = (uint32_t)output_ordinals.count;
    // Each replaced callable copy leaves one call behind.
    const uint64_t ops = finding->op_count;
    uint64_t callable_copies = 0;
    for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
      const loom_analyze_occurrence_t* occurrence = &finding->occurrences[i];
      if (!occurrence->in_launch_config && !occurrence->is_motif_definition) {
        ++callable_copies;
      }
    }
    uint64_t removable = 0;
    if (finding->rule == LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF) {
      removable = callable_copies * (ops - 1);
    } else if (!clone_class->nested_constant_differs) {
      // Call-site constants stay in every copy and leave the motif body.
      const uint64_t kept_ops = clone_class->call_site_constant_count;
      const uint64_t before = callable_copies * ops;
      const uint64_t after =
          (ops - kept_ops) + callable_copies * (kept_ops + 1);
      removable = callable_copies >= 2 && before > after ? before - after : 0;
    }
    finding->type_generic_removable_op_count = removable;
    finding->removable_op_count = finding->applicable ? removable : 0;
    qsort(finding->occurrences, finding->occurrence_count,
          sizeof(finding->occurrences[0]), loom_analyze_compare_occurrences);
  }
  loom_analyze_index_vector_free(allocator, &output_ordinals);
  if (printers) {
    for (iree_host_size_t m = 0; m < analyzer->modules.count; ++m) {
      loom_text_printer_deinitialize(&printers[m]);
    }
    iree_allocator_free(allocator, printers);
  }
  if (iree_status_is_ok(status)) {
    status = loom_analyze_select_plan(analyzer, detection, kept_classes);
  }
  if (iree_status_is_ok(status) && result->finding_count > 1) {
    qsort(result->findings, result->finding_count, sizeof(result->findings[0]),
          loom_analyze_compare_findings);
  }
  return status;
}

iree_status_t loom_analyze_analyzer_run(
    loom_analyze_analyzer_t* analyzer,
    const loom_analyze_result_t** out_result) {
  *out_result = NULL;
  if (analyzer->has_run) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "the analysis has already run");
  }
  analyzer->has_run = true;
  iree_allocator_t allocator = analyzer->allocator;
  loom_analyze_result_t* result = &analyzer->result;
  result->summary.module_count = (uint32_t)analyzer->modules.count;
  result->summary.op_count = analyzer->ops.count;

  loom_analyze_candidate_vector_t candidates = {0};
  loom_analyze_detection_t detection;
  memset(&detection, 0, sizeof(detection));
  uint32_t* order = NULL;
  uint32_t* kept = NULL;
  uint8_t* covered = NULL;
  iree_status_t status = loom_analyze_collect_candidates(analyzer, &candidates);
  if (iree_status_is_ok(status)) {
    status = loom_analyze_detect_classes(analyzer, &candidates, &detection);
  }

  // Keep classes not subsumed by a larger kept class.
  const uint32_t class_count = (uint32_t)detection.classes.count;
  uint32_t kept_count = 0;
  if (iree_status_is_ok(status) && class_count > 0) {
    status = iree_allocator_malloc_array(allocator, class_count,
                                         sizeof(uint32_t), (void**)&order);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(allocator, class_count,
                                           sizeof(uint32_t), (void**)&kept);
    }
    if (iree_status_is_ok(status)) {
      for (uint32_t i = 0; i < class_count; ++i) {
        order[i] = i;
      }
      loom_analyze_sort_classes = detection.classes.data;
      qsort(order, class_count, sizeof(order[0]),
            loom_analyze_compare_class_size);
      loom_analyze_sort_classes = NULL;
      for (uint32_t i = 0; i < class_count; ++i) {
        const loom_analyze_class_t* candidate =
            &detection.classes.data[order[i]];
        bool subsumed = false;
        for (uint32_t j = 0; j < kept_count && !subsumed; ++j) {
          subsumed = loom_analyze_class_contained(
              &detection, candidate, &detection.classes.data[kept[j]]);
        }
        if (!subsumed) {
          kept[kept_count++] = order[i];
        }
      }
    }
  }

  // Count distinct operations covered by reported occurrences.
  if (iree_status_is_ok(status) && kept_count > 0 && analyzer->ops.count > 0) {
    status = iree_allocator_malloc_array(allocator, analyzer->ops.count,
                                         sizeof(uint8_t), (void**)&covered);
    if (iree_status_is_ok(status)) {
      for (uint32_t k = 0; k < kept_count; ++k) {
        const loom_analyze_class_t* clone_class =
            &detection.classes.data[kept[k]];
        for (uint32_t i = 0; i < clone_class->occurrence_count; ++i) {
          const loom_analyze_class_occurrence_t* occurrence =
              &detection.occurrences.data[clone_class->occurrence_offset + i];
          for (uint32_t op = occurrence->op_begin; op < occurrence->op_end;
               ++op) {
            if (!covered[op]) {
              covered[op] = 1;
              ++result->summary.duplicated_op_count;
            }
          }
        }
      }
    }
  }

  if (iree_status_is_ok(status)) {
    status =
        loom_analyze_build_findings(analyzer, &detection, kept, kept_count);
  }

  iree_allocator_free(allocator, covered);
  iree_allocator_free(allocator, kept);
  iree_allocator_free(allocator, order);
  loom_analyze_detection_free(allocator, &detection);
  loom_analyze_candidate_vector_free(allocator, &candidates);
  if (iree_status_is_ok(status)) {
    *out_result = result;
  }
  return status;
}
