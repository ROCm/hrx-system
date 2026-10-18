// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/io/file.h"
#include "loomc/import/cxx.h"
#include "loomc/iree.h"
#include "loomc/target/configured_ir.h"

IREE_FLAG(string, output, "-", "Output path, or '-' for stdout.");
IREE_FLAG(string, to, "text", "Output format: text or bc.");
IREE_FLAG(string, std, "c++26", "Source standard: c23, c++14/17/20/23/26.");
IREE_FLAG(string, triple, "",
          "Source ABI triple; does not select an output target.");
IREE_FLAG_NAMED(string, data_model, "data-model", "lp64",
                "Source integer/pointer layout: lp64, llp64, or ilp32.");
IREE_FLAG_NAMED(string, fp_model, "fp-model", "precise",
                "Floating-point transformation policy: precise or fast.");
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
IREE_FLAG_NAMED(bool, builtin_includes, "builtin-includes", true,
                "Search embedded Loom/HIP source headers when available.");

static void loom_cxx_cli_copy_strings(iree_flag_string_list_t list,
                                      loomc_string_view_t* output) {
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    output[i] = loomc_string_view_from_iree(list.values[i]);
  }
}

static iree_status_t loom_cxx_cli_write_module(const loomc_module_t* module,
                                               iree_allocator_t allocator) {
  const loomc_module_serialize_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(options),
      .format = strcmp(FLAG_to, "text") == 0 ? LOOMC_SOURCE_FORMAT_TEXT
                                             : LOOMC_SOURCE_FORMAT_BYTECODE,
  };
  loomc_source_t* output = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_module_serialize_to_source(
          module, &options, loomc_allocator_from_iree(allocator), &output));
  if (iree_status_is_ok(status)) {
    const loomc_byte_span_t contents = loomc_source_contents(output);
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(FLAG_output),
        iree_make_string_view((const char*)contents.data, contents.data_length),
        allocator);
  }
  loomc_source_release(output);
  return status;
}

// Sets *out_succeeded only after importing and writing the module. Diagnosed
// source rejection returns OK with *out_succeeded=false; infrastructure and
// option failures return a non-OK status for main to report.
static iree_status_t loom_cxx_cli_import(iree_string_view_t filename,
                                         loomc_context_t* context,
                                         loomc_workspace_t* workspace,
                                         iree_allocator_t allocator,
                                         bool* out_succeeded) {
  *out_succeeded = false;
  const loomc_allocator_t host_allocator = loomc_allocator_from_iree(allocator);
  loomc_cxx_import_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_CXX_IMPORT_OPTIONS,
      .structure_size = sizeof(options),
      .standard = loomc_make_cstring_view(FLAG_std),
      .triple = loomc_make_cstring_view(FLAG_triple),
  };
  if (strcmp(FLAG_data_model, "lp64") == 0) {
    options.data_model = LOOMC_CXX_DATA_MODEL_LP64;
  } else if (strcmp(FLAG_data_model, "llp64") == 0) {
    options.data_model = LOOMC_CXX_DATA_MODEL_LLP64;
  } else if (strcmp(FLAG_data_model, "ilp32") == 0) {
    options.data_model = LOOMC_CXX_DATA_MODEL_ILP32;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown source data model '%s'", FLAG_data_model);
  }
  if (strcmp(FLAG_fp_model, "precise") == 0) {
    options.fp_model = LOOMC_CXX_FP_MODEL_PRECISE;
  } else if (strcmp(FLAG_fp_model, "fast") == 0) {
    options.fp_model = LOOMC_CXX_FP_MODEL_FAST;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown floating-point model '%s'", FLAG_fp_model);
  }
  if (strcmp(FLAG_to, "text") != 0 && strcmp(FLAG_to, "bc") != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown output format '%s'", FLAG_to);
  }
  if (!FLAG_builtin_includes) {
    options.flags |= LOOMC_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES;
  }
  const iree_flag_string_list_t includes = FLAG_include_path_list();
  const iree_flag_string_list_t system_includes =
      FLAG_system_include_path_list();
  const iree_flag_string_list_t roots = FLAG_root_list();
  const iree_flag_string_list_t definitions = FLAG_define_list();
  const iree_host_size_t string_count =
      includes.count + system_includes.count + roots.count;
  loomc_string_view_t* strings = NULL;
  if (string_count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, string_count, sizeof(*strings), (void**)&strings));
    loom_cxx_cli_copy_strings(includes, strings);
    loom_cxx_cli_copy_strings(system_includes, strings + includes.count);
    loom_cxx_cli_copy_strings(roots,
                              strings + includes.count + system_includes.count);
    options.include_paths = strings;
    options.include_path_count = includes.count;
    options.system_include_paths = strings + includes.count;
    options.system_include_path_count = system_includes.count;
    options.roots = strings + includes.count + system_includes.count;
    options.root_count = roots.count;
  }
  loomc_cxx_define_t* defines = NULL;
  iree_status_t status = iree_ok_status();
  if (definitions.count != 0) {
    status = iree_allocator_malloc_array(allocator, definitions.count,
                                         sizeof(*defines), (void**)&defines);
    for (iree_host_size_t i = 0;
         iree_status_is_ok(status) && i < definitions.count; ++i) {
      const iree_string_view_t value = definitions.values[i];
      const iree_host_size_t equal = iree_string_view_find_char(value, '=', 0);
      defines[i].name =
          loomc_string_view_from_iree(iree_string_view_substr(value, 0, equal));
      defines[i].value =
          equal == IREE_STRING_VIEW_NPOS
              ? loomc_make_cstring_view("1")
              : loomc_string_view_from_iree(iree_string_view_substr(
                    value, equal + 1, IREE_HOST_SIZE_MAX));
    }
  }
  options.defines = defines;
  options.define_count = definitions.count;
  iree_io_file_contents_t* contents = NULL;
  loomc_source_t* source = NULL;
  loomc_module_t* module = NULL;
  loomc_pass_program_t* pass_program = NULL;
  bool succeeded = false;
  if (iree_status_is_ok(status)) {
    status = loom_tooling_read_input_file(filename, allocator, &contents);
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t source_text =
        loom_tooling_file_contents_string_view(contents);
    const loomc_source_options_t source_options = {
        .identifier = loomc_string_view_from_iree(filename),
        .contents = loomc_make_byte_span(source_text.data, source_text.size),
        .storage = LOOMC_SOURCE_STORAGE_BORROWED,
    };
    status = iree_status_from_loomc(
        loomc_source_create(&source_options, host_allocator, &source));
  }
  if (iree_status_is_ok(status)) {
    loomc_result_t* result = NULL;
    status = iree_status_from_loomc(
        loomc_module_import_cxx(context, workspace, source, &options,
                                host_allocator, &module, &result));
    if (iree_status_is_ok(status)) {
      status = loom_tooling_cli_print_loomc_result(stderr, result, &succeeded);
    }
    loomc_result_release(result);
  }
  if (iree_status_is_ok(status) && succeeded && FLAG_cleanup) {
    loomc_result_t* result = NULL;
    status =
        iree_status_from_loomc(loomc_pass_program_create_from_pipeline_text(
            context, loomc_make_cstring_view("combine,cse,dce"), NULL,
            host_allocator, &pass_program, &result));
    if (iree_status_is_ok(status)) {
      status = loom_tooling_cli_print_loomc_result(stderr, result, &succeeded);
    }
    loomc_result_release(result);
  }
  if (iree_status_is_ok(status) && succeeded && FLAG_cleanup) {
    loomc_result_t* result = NULL;
    status = iree_status_from_loomc(loomc_pass_program_run(
        pass_program, workspace, module, NULL, host_allocator, &result));
    if (iree_status_is_ok(status)) {
      status = loom_tooling_cli_print_loomc_result(stderr, result, &succeeded);
    }
    loomc_result_release(result);
  }
  if (iree_status_is_ok(status) && succeeded) {
    loomc_result_t* result = NULL;
    status = iree_status_from_loomc(
        loomc_module_verify(module, NULL, host_allocator, &result));
    if (iree_status_is_ok(status)) {
      status = loom_tooling_cli_print_loomc_result(stderr, result, &succeeded);
    }
    loomc_result_release(result);
  }
  if (iree_status_is_ok(status) && succeeded) {
    status = loom_cxx_cli_write_module(module, allocator);
  }
  *out_succeeded = iree_status_is_ok(status) && succeeded;
  loomc_pass_program_release(pass_program);
  loomc_module_release(module);
  loomc_source_release(source);
  iree_io_file_contents_free(contents);
  iree_allocator_free(allocator, defines);
  iree_allocator_free(allocator, strings);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-import-cxx",
      "Imports one C/C++ translation unit into editable Loom IR.\n"
      "Usage: loom-import-cxx [options] source.cxx\n"
      "Use --I=path and --D=NAME=VALUE for source configuration.\n"
      "Unspecified kernel dimensions become config.decl symbols.\n");
  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 2) {
    fprintf(stderr, "expected exactly one source translation unit\n");
    return 1;
  }
  iree_allocator_t allocator = iree_allocator_system();
  const loomc_allocator_t host_allocator = loomc_allocator_from_iree(allocator);
  loomc_target_environment_t* target_environment = NULL;
  loomc_context_t* context = NULL;
  loomc_workspace_t* workspace = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_target_environment_create_configured_ir(
          host_allocator, &target_environment));
  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = iree_status_from_loomc(
        loomc_context_create(&context_options, host_allocator, &context));
  }
  if (iree_status_is_ok(status)) {
    // 64 KiB blocks avoid per-block mappings on allocators with a 128 KiB
    // mapping threshold during short-lived imports.
    const loomc_workspace_options_t workspace_options = {
        .type = LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS,
        .structure_size = sizeof(workspace_options),
        .block_size = 64 * 1024,
    };
    status = iree_status_from_loomc(
        loomc_workspace_create(&workspace_options, host_allocator, &workspace));
  }
  bool import_succeeded = false;
  if (iree_status_is_ok(status)) {
    status = loom_cxx_cli_import(iree_make_cstring_view(argv[1]), context,
                                 workspace, allocator, &import_succeeded);
  }
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(target_environment);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
  }
  iree_status_free(status);
  return import_succeeded ? 0 : 1;
}
