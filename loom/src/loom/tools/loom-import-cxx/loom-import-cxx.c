// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/printer.h"
#include "loom/import/cxx/import.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/tooling.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/io/file.h"
#include "loom/verify/verify.h"

IREE_FLAG(string, output, "-", "Output path, or '-' for stdout.");
IREE_FLAG(string, to, "text", "Output format: text or bc.");
IREE_FLAG(string, std, "c++26", "Source standard: c23, c++14/17/20/23/26.");
IREE_FLAG(string, triple, "",
          "Source ABI triple; does not select an output target.");
IREE_FLAG_NAMED(string, data_model, "data-model", "lp64",
                "Source integer/pointer layout: lp64, llp64, or ilp32.");
IREE_FLAG_LIST_NAMED(string, include_path, "I", "User include directory.");
IREE_FLAG_LIST_NAMED(string, system_include_path, "isystem",
                     "System include directory.");
IREE_FLAG_LIST_NAMED(string, define, "D",
                     "Preprocessor definition NAME[=VALUE].");
IREE_FLAG_LIST(
    string, root,
    "Qualified source function to export; repeat for multiple roots.");
IREE_FLAG(bool, cleanup, true,
          "Canonicalize, eliminate common expressions, and remove dead IR.");
IREE_FLAG_NAMED(bool, approximate_functions, "approximate-functions", false,
                "Permit approximate mathematical function results.");
IREE_FLAG_NAMED(bool, builtin_includes, "builtin-includes", true,
                "Search embedded Loom/HIP source headers when available.");

static iree_status_t loom_cxx_cli_write_block(void* user_data,
                                              iree_const_byte_span_t block) {
  return loom_output_stream_write(
      (loom_output_stream_t*)user_data,
      iree_make_string_view((const char*)block.data, block.data_length));
}

static iree_status_t loom_cxx_cli_write_module(loom_module_t* module,
                                               iree_arena_block_pool_t* pool,
                                               iree_allocator_t allocator) {
  loom_tooling_output_stream_t output = {0};
  IREE_RETURN_IF_ERROR(loom_tooling_output_stream_open(
      iree_make_cstring_view(FLAG_output), allocator, &output));
  iree_status_t status = iree_ok_status();
  if (strcmp(FLAG_to, "text") == 0) {
    status =
        loom_text_print_module(module, &output.stream, LOOM_TEXT_PRINT_DEFAULT);
  } else {
    iree_io_stream_t* stream = NULL;
    status = iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, allocator, &stream);
    if (iree_status_is_ok(status)) {
      loom_bytecode_write_options_t options = {
          .producer = IREE_SV("loom-import-cxx"),
      };
      status = loom_bytecode_write_module(module, stream, &options, pool);
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_vec_stream_enumerate_blocks(
          stream, loom_cxx_cli_write_block, &output.stream);
    }
    iree_io_stream_release(stream);
  }
  return iree_status_join(status, loom_tooling_output_stream_close(&output));
}

// Sets *out_succeeded only after importing and writing the module. Diagnosed
// source rejection returns OK with *out_succeeded=false; infrastructure and
// option failures return a non-OK status for main to report.
static iree_status_t loom_cxx_cli_import(iree_string_view_t filename,
                                         loom_context_t* context,
                                         iree_arena_block_pool_t* pool,
                                         iree_allocator_t allocator,
                                         bool* out_succeeded) {
  *out_succeeded = false;
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
  options.standard = iree_make_cstring_view(FLAG_std);
  options.triple = iree_make_cstring_view(FLAG_triple);
  if (strcmp(FLAG_data_model, "lp64") == 0) {
    options.data_model = LOOM_CXX_DATA_MODEL_LP64;
  } else if (strcmp(FLAG_data_model, "llp64") == 0) {
    options.data_model = LOOM_CXX_DATA_MODEL_LLP64;
  } else if (strcmp(FLAG_data_model, "ilp32") == 0) {
    options.data_model = LOOM_CXX_DATA_MODEL_ILP32;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown source data model '%s'", FLAG_data_model);
  }
  if (strcmp(FLAG_to, "text") != 0 && strcmp(FLAG_to, "bc") != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown output format '%s'", FLAG_to);
  }
  if (FLAG_approximate_functions) {
    options.flags |= LOOM_CXX_IMPORT_FLAG_APPROXIMATE_FUNCTIONS;
  }
  if (!FLAG_builtin_includes) {
    options.flags |= LOOM_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES;
  }
  const iree_flag_string_list_t includes = FLAG_include_path_list();
  options.include_paths = includes.values;
  options.include_path_count = includes.count;
  const iree_flag_string_list_t system_includes =
      FLAG_system_include_path_list();
  options.system_include_paths = system_includes.values;
  options.system_include_path_count = system_includes.count;
  const iree_flag_string_list_t roots = FLAG_root_list();
  options.roots = roots.values;
  options.root_count = roots.count;
  const iree_flag_string_list_t definitions = FLAG_define_list();
  loom_cxx_define_t* defines = NULL;
  if (definitions.count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        allocator, definitions.count * sizeof(*defines), (void**)&defines));
    for (iree_host_size_t i = 0; i < definitions.count; ++i) {
      iree_string_view_t value = definitions.values[i];
      iree_host_size_t equal = iree_string_view_find_char(value, '=', 0);
      defines[i].name = iree_string_view_substr(value, 0, equal);
      defines[i].value =
          equal == IREE_STRING_VIEW_NPOS
              ? IREE_SV("1")
              : iree_string_view_substr(value, equal + 1, IREE_HOST_SIZE_MAX);
    }
  }
  options.defines = defines;
  options.define_count = definitions.count;
  iree_io_file_contents_t* contents = NULL;
  loom_module_t* module = NULL;
  iree_status_t status =
      loom_tooling_read_input_file(filename, allocator, &contents);
  if (iree_status_is_ok(status)) {
    status =
        loom_cxx_import(loom_tooling_file_contents_string_view(contents),
                        filename, context, pool, &options, allocator, &module);
  }
  if (iree_status_is_ok(status) && module && FLAG_cleanup) {
    loom_pass_tool_run_options_t pass_options = {
        .registry = loom_pass_builtin_registry(),
        .block_pool = pool,
    };
    loom_pass_run_result_t result = {0};
    status = loom_pass_tool_run_flat_pipeline(
        module, IREE_SV("canonicalize,cse,dce"), &pass_options, &result);
    if (iree_status_is_ok(status) && result.error_count) {
      status = iree_make_status(IREE_STATUS_INTERNAL, "import cleanup failed");
    }
  }
  if (iree_status_is_ok(status) && module) {
    loom_verify_options_t verification_options = {.sink =
                                                      options.diagnostic_sink};
    loom_verify_result_t result = {0};
    status = loom_verify_module(module, &verification_options, &result);
    if (iree_status_is_ok(status) && result.error_count) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "import cleanup produced invalid IR");
    }
  }
  if (iree_status_is_ok(status) && module) {
    status = loom_cxx_cli_write_module(module, pool, allocator);
  }
  *out_succeeded = iree_status_is_ok(status) && module != NULL;
  loom_module_free(module);
  iree_io_file_contents_free(contents);
  iree_allocator_free(allocator, defines);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-import-cxx",
      "Imports one C/C++ translation unit into editable Loom IR.\n"
      "Usage: loom-import-cxx [options] source.cpp\n"
      "Use --I=path and --D=NAME=VALUE for source configuration.\n"
      "Unspecified kernel dimensions become config.decl symbols.\n");
  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 2) {
    fprintf(stderr, "expected exactly one source translation unit\n");
    return 1;
  }
  iree_allocator_t allocator = iree_allocator_system();
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(64 * 1024, allocator, &pool);
  loom_context_t context;
  loom_context_initialize(allocator, &context);
  iree_status_t status = loom_tooling_context_register_tool_dialects(&context);
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&context);
  }
  bool import_succeeded = false;
  if (iree_status_is_ok(status)) {
    status = loom_cxx_cli_import(iree_make_cstring_view(argv[1]), &context,
                                 &pool, allocator, &import_succeeded);
  }
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&pool);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
  }
  iree_status_free(status);
  return import_succeeded ? 0 : 1;
}
