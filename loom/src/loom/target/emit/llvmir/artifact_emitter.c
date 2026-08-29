// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/artifact_emitter.h"

#include "iree/io/vec_stream.h"
#include "loom/target/emit/llvmir/bitcode_writer.h"
#include "loom/target/emit/llvmir/module_emitter.h"
#include "loom/target/emit/llvmir/text_writer.h"
#include "loom/target/emit/llvmir/verify.h"
#include "loom/util/stream.h"

typedef enum loom_llvmir_artifact_emitter_format_e {
  LOOM_LLVMIR_ARTIFACT_EMITTER_FORMAT_TEXT = 0,
  LOOM_LLVMIR_ARTIFACT_EMITTER_FORMAT_BITCODE = 1,
} loom_llvmir_artifact_emitter_format_t;

typedef struct loom_llvmir_artifact_emitter_option_prefix_t {
  // Option descriptor type.
  uint32_t type;
  // Size of this structure in bytes.
  iree_host_size_t structure_size;
  // Next emitter option descriptor.
  const void* next;
} loom_llvmir_artifact_emitter_option_prefix_t;

void loom_llvmir_artifact_emitter_options_initialize(
    loom_llvmir_artifact_emitter_options_t* out_options) {
  *out_options = (loom_llvmir_artifact_emitter_options_t){
      .type = LOOM_LLVMIR_ARTIFACT_EMITTER_OPTION_TYPE_OPTIONS,
      .structure_size = sizeof(*out_options),
  };
}

static iree_status_t loom_llvmir_artifact_emitter_options_resolve(
    const void* option_chain,
    loom_llvmir_emit_low_module_options_t* out_options) {
  loom_llvmir_emit_low_module_options_initialize(out_options);
  const void* node = option_chain;
  while (node != NULL) {
    const loom_llvmir_artifact_emitter_option_prefix_t* prefix =
        (const loom_llvmir_artifact_emitter_option_prefix_t*)node;
    if (prefix->type == LOOM_LLVMIR_ARTIFACT_EMITTER_OPTION_TYPE_OPTIONS) {
      if (prefix->structure_size != 0 &&
          prefix->structure_size <
              sizeof(loom_llvmir_artifact_emitter_options_t)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "LLVMIR artifact emitter options structure_size is too small");
      }
      const loom_llvmir_artifact_emitter_options_t* options =
          (const loom_llvmir_artifact_emitter_options_t*)node;
      out_options->target_profile_registry = options->target_profile_registry;
      node = options->next;
      continue;
    }
    node = prefix->next;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_artifact_write_text(
    const loom_llvmir_module_t* module, iree_allocator_t allocator,
    iree_byte_sequence_t** out_contents) {
  *out_contents = NULL;
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status = loom_llvmir_text_write_module(module, &stream);
  iree_byte_span_t contents = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    const iree_host_size_t contents_length = iree_string_builder_size(&builder);
    contents = iree_make_byte_span(iree_string_builder_take_storage(&builder),
                                   contents_length);
    status = iree_byte_sequence_create_from_span_move(&contents, allocator,
                                                      out_contents);
  }
  iree_allocator_free(allocator, contents.data);
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_llvmir_artifact_write_bitcode(
    const loom_llvmir_module_t* module, iree_allocator_t allocator,
    iree_byte_sequence_t** out_contents) {
  *out_contents = NULL;
  iree_io_stream_t* stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      32 * 1024, allocator, &stream);

  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_module(module, stream);
  }

  if (iree_status_is_ok(status)) {
    const iree_io_stream_pos_t stream_length = iree_io_stream_length(stream);
    if (stream_length <= 0) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "LLVM bitcode output length is invalid");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_move_contents(stream, out_contents);
  }

  iree_io_stream_release(stream);
  return status;
}

static iree_status_t loom_llvmir_artifact_emit(
    const loom_target_emit_request_t* request,
    loom_llvmir_artifact_emitter_format_t format,
    loom_target_artifact_format_t artifact_format,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LLVMIR target artifact emitters do not produce artifact manifests");
  }

  loom_llvmir_module_t* module = NULL;
  loom_llvmir_emit_low_module_options_t emit_options = {0};
  IREE_RETURN_IF_ERROR(loom_llvmir_artifact_emitter_options_resolve(
      request->option_chain, &emit_options));
  emit_options.function_versions = request->function_versions;
  iree_status_t status = loom_llvmir_emit_low_module(
      request->module, request->low_descriptor_registry,
      request->diagnostic_emitter, request->scratch_arena, &emit_options,
      &module, request->allocator);
  if (iree_status_is_ok(status) && module == NULL) {
    return iree_ok_status();
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_verify_module(module);
  }

  iree_byte_sequence_t* contents = NULL;
  if (iree_status_is_ok(status)) {
    switch (format) {
      case LOOM_LLVMIR_ARTIFACT_EMITTER_FORMAT_TEXT:
        status = loom_llvmir_artifact_write_text(module, request->allocator,
                                                 &contents);
        break;
      case LOOM_LLVMIR_ARTIFACT_EMITTER_FORMAT_BITCODE:
        status = loom_llvmir_artifact_write_bitcode(module, request->allocator,
                                                    &contents);
        break;
    }
  }
  if (iree_status_is_ok(status)) {
    out_artifact->target_artifact_format = artifact_format;
    out_artifact->contents = contents;
    contents = NULL;
  }

  iree_byte_sequence_release(contents);
  loom_llvmir_module_free(module);
  return status;
}

static iree_status_t loom_llvmir_text_artifact_emit(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  return loom_llvmir_artifact_emit(
      request, LOOM_LLVMIR_ARTIFACT_EMITTER_FORMAT_TEXT,
      LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_TEXT, out_artifact);
}

static iree_status_t loom_llvmir_bitcode_artifact_emit(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  return loom_llvmir_artifact_emit(
      request, LOOM_LLVMIR_ARTIFACT_EMITTER_FORMAT_BITCODE,
      LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_BITCODE, out_artifact);
}

static const loom_target_emitter_t loom_llvmir_text_artifact_emitter = {
    .name = IREE_SVL("llvmir-text"),
    .public_artifact_format = IREE_SVL("llvmir-text"),
    .default_identifier = IREE_SVL("module.ll"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_TEXT,
    .emit = loom_llvmir_text_artifact_emit,
};

static const loom_target_emitter_t loom_llvmir_bitcode_artifact_emitter = {
    .name = IREE_SVL("llvmir-bitcode"),
    .public_artifact_format = IREE_SVL("llvmir-bitcode"),
    .default_identifier = IREE_SVL("module.bc"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_BITCODE,
    .emit = loom_llvmir_bitcode_artifact_emit,
};

static const loom_target_emitter_t* const kLoomLlvmirArtifactEmitters[] = {
    &loom_llvmir_text_artifact_emitter,
    &loom_llvmir_bitcode_artifact_emitter,
};

const loom_target_provider_t loom_llvmir_artifact_emitter_provider = {
    .emitter_list =
        {
            .values = kLoomLlvmirArtifactEmitters,
            .count = IREE_ARRAYSIZE(kLoomLlvmirArtifactEmitters),
        },
};
