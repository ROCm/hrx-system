// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify.h"

#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"

//===----------------------------------------------------------------------===//
// Verify execution
//===----------------------------------------------------------------------===//

iree_status_t loom_check_execute_verify(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  // Arena for collector storage. Entries and message copies are released
  // together at the end of this case.
  iree_arena_allocator_t collector_arena;
  iree_arena_initialize(block_pool, &collector_arena);

  loom_check_diagnostic_collector_t collector = {
      .arena = &collector_arena,
      .host_allocator = allocator,
      .result = result,
  };

  // Strip standalone comment lines from input. Comments become blank
  // lines to preserve line count for diagnostic source locations.
  iree_string_builder_t stripped_input;
  iree_string_builder_initialize(allocator, &stripped_input);
  iree_status_t status =
      loom_check_strip_comments(test_case->input, &stripped_input);

  // Parse the stripped input with the collector as the diagnostic sink.
  // Parse errors are emitted as diagnostics (not status failures).
  loom_module_t* module = NULL;
  iree_string_view_t stripped_view = iree_string_builder_view(&stripped_input);
  loom_target_low_descriptor_registry_t low_registry;
  if (iree_status_is_ok(status)) {
    status = loom_check_environment_initialize_low_descriptor_registry(
        environment, &low_registry);
  }
  if (iree_status_is_ok(status)) {
    loom_low_descriptor_text_print_context_initialize(
        &low_registry.registry, &collector.type_print_context);
  }
  if (iree_status_is_ok(status)) {
    loom_text_parse_options_t parse_options = {
        .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                            .user_data = &collector},
        .max_errors = 100,
    };
    loom_low_descriptor_text_asm_environment_storage_t low_asm_storage = {0};
    loom_low_descriptor_text_asm_environment_initialize_with_diagnostics(
        &low_registry.registry, environment->low_asm_diagnostic_provider_list,
        &low_asm_storage, &parse_options.low_asm_environment);
    status = loom_text_parse(stripped_view, filename, context, block_pool,
                             &parse_options, &module);
    collector.module = module;
  }

  // If parsing succeeded (module != NULL), run the verifier to collect
  // additional diagnostics. The source resolver uses the stripped input
  // so verifier diagnostics get the same line numbers as parse diagnostics.
  if (iree_status_is_ok(status) && module) {
    loom_source_entry_t source_entry = {0};
    loom_source_table_resolver_t resolver_data = {0};
    if (iree_status_is_ok(status)) {
      status = loom_check_source_resolver_for_case(
          module, filename, stripped_view, &source_entry, &resolver_data);
    }
    loom_verify_options_t verify_options = {
        .sink = {.fn = loom_check_diagnostic_collector_sink,
                 .user_data = &collector},
        .max_errors = 100,
        .source_resolver = {.fn = loom_source_table_resolve,
                            .user_data = &resolver_data},
    };

    loom_verify_result_t verify_result = {0};
    if (iree_status_is_ok(status)) {
      status = loom_verify_module(module, &verify_options, &verify_result);
    }
    if (iree_status_is_ok(status) && verify_result.error_count == 0) {
      loom_check_diagnostic_emitter_capture_t low_diagnostic_capture = {
          .diagnostic_collector = &collector,
          .module = module,
          .source_resolver = {.fn = loom_source_table_resolve,
                              .user_data = &resolver_data},
          .emitter = LOOM_EMITTER_VERIFIER,
      };
      loom_low_verify_options_t low_verify_options = {
          .descriptor_registry = &low_registry.registry,
          .emitter =
              {
                  .fn = loom_check_diagnostic_emitter_capture_emit,
                  .user_data = &low_diagnostic_capture,
              },
          .provider_list = environment->low_verify_provider_list,
          .max_errors = 100,
      };
      loom_low_verify_result_t low_verify_result = {0};
      loom_low_verify_scratch_t low_verify_scratch =
          loom_low_verify_scratch_for_module(module);
      status = loom_low_verify_module(module, &low_verify_options,
                                      &low_verify_scratch, &low_verify_result);
    }
  }

  // The stripped input builder is no longer needed (diagnostics already
  // collected with arena-copied messages).
  iree_string_builder_deinitialize(&stripped_input);

  if (iree_status_is_ok(status)) {
    status = loom_check_diagnostic_collector_finish(
        &collector, test_case, case_index, report, allocator, result);
  }

  loom_module_free(module);
  iree_arena_deinitialize(&collector_arena);
  return status;
}
