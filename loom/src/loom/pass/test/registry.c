// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/test/registry.h"

#include <stdint.h>

#include "loom/analysis/ownership_lifetime.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/report.h"

static bool loom_test_pass_target_record_satisfies_requirement(
    const loom_pass_environment_capability_t* capability,
    iree_string_view_t requirement) {
  (void)capability;
  return iree_string_view_equal(requirement, IREE_SV("target.record"));
}

const loom_pass_environment_capability_type_t
    loom_test_pass_target_record_capability_type = {
        .name = IREE_SVL("test.target-record"),
        .satisfies_requirement =
            loom_test_pass_target_record_satisfies_requirement,
};

const loom_pass_environment_capability_t
    loom_test_pass_target_record_capability = {
        .type = &loom_test_pass_target_record_capability_type,
};

const loom_pass_environment_capability_type_t
    loom_test_pass_trace_capability_type = {
        .name = IREE_SVL("test.pass-trace"),
};

loom_test_pass_trace_capability_t loom_test_pass_trace_capability_make(
    loom_test_pass_trace_t* trace) {
  return (loom_test_pass_trace_capability_t){
      .base = {.type = &loom_test_pass_trace_capability_type},
      .trace = trace,
  };
}

loom_test_pass_trace_t* loom_test_pass_trace_from_pass(
    const loom_pass_t* pass) {
  if (!pass || !pass->environment) {
    return NULL;
  }
  const loom_test_pass_trace_capability_t* capability =
      (const loom_test_pass_trace_capability_t*)loom_pass_environment_lookup(
          pass->environment, &loom_test_pass_trace_capability_type);
  return capability ? capability->trace : NULL;
}

static iree_string_view_t loom_test_pass_function_name(
    const loom_module_t* module, loom_func_like_t function) {
  loom_symbol_ref_t symbol_ref = loom_func_like_callee(function);
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<none>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<none>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_test_pass_trace_record(
    loom_test_pass_trace_t* trace, iree_string_view_t pass_name,
    iree_string_view_t symbol_name) {
  if (!trace) {
    return iree_ok_status();
  }
  if (trace->event_count >= LOOM_TEST_PASS_TRACE_EVENT_CAPACITY) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "test pass trace capacity exceeded");
  }
  trace->events[trace->event_count++] = (loom_test_pass_trace_event_t){
      .pass_name = pass_name,
      .symbol_name = symbol_name,
  };
  return iree_ok_status();
}

#define LOOM_TEST_INVOCATION_STATISTICS(V, statistics_type) \
  V(statistics_type, invocations, "invocations", "Number of pass invocations.")

LOOM_PASS_STATISTICS_DEFINE(loom_test_invocation_statistics,
                            loom_test_invocation_statistics_t,
                            LOOM_TEST_INVOCATION_STATISTICS)

#define LOOM_TEST_MARK_CHANGED_STATISTICS(V, statistics_type) \
  V(statistics_type, invocations, "invocations",              \
    "Number of pass invocations.")                            \
  V(statistics_type, synthetic_events, "synthetic-events",    \
    "Number of deterministic test events.")

LOOM_PASS_STATISTICS_DEFINE(loom_test_mark_changed_statistics,
                            loom_test_mark_changed_statistics_t,
                            LOOM_TEST_MARK_CHANGED_STATISTICS)

#define LOOM_TEST_RESOURCE_LIFETIME_STATISTICS(V, statistics_type) \
  V(statistics_type, blocks_checked, "blocks-checked",             \
    "Number of blocks checked for resource lifetimes.")            \
  V(statistics_type, ops_checked, "ops-checked",                   \
    "Number of operations visited for resource lifetimes.")        \
  V(statistics_type, effects_checked, "effects-checked",           \
    "Number of ownership effects interpreted.")                    \
  V(statistics_type, releases_inserted, "releases-inserted",       \
    "Number of test.resource.release operations inserted.")        \
  V(statistics_type, edges_split, "edges-split",                   \
    "Number of CFG edges split for resource cleanup.")

LOOM_PASS_STATISTICS_DEFINE(loom_test_resource_lifetime_statistics,
                            loom_test_resource_lifetime_statistics_t,
                            LOOM_TEST_RESOURCE_LIFETIME_STATISTICS)

static iree_status_t loom_test_module_noop_run(loom_pass_t* pass,
                                               loom_module_t* module) {
  (void)module;
  loom_test_pass_trace_t* trace = loom_test_pass_trace_from_pass(pass);
  IREE_RETURN_IF_ERROR(loom_test_pass_trace_record(
      trace, IREE_SV("test.module-noop"), IREE_SV("<module>")));
  if (trace) ++trace->module_noop_invocation_count;
  loom_test_invocation_statistics_t* statistics =
      loom_test_invocation_statistics(pass);
  ++statistics->invocations;
  return iree_ok_status();
}

static iree_status_t loom_test_noop_run(loom_pass_t* pass,
                                        loom_module_t* module,
                                        loom_func_like_t function) {
  loom_test_pass_trace_t* trace = loom_test_pass_trace_from_pass(pass);
  IREE_RETURN_IF_ERROR(loom_test_pass_trace_record(
      trace, IREE_SV("test.noop"),
      loom_test_pass_function_name(module, function)));
  if (trace) ++trace->noop_invocation_count;
  loom_test_invocation_statistics_t* statistics =
      loom_test_invocation_statistics(pass);
  ++statistics->invocations;
  return iree_ok_status();
}

static iree_status_t loom_test_mark_changed_run(loom_pass_t* pass,
                                                loom_module_t* module,
                                                loom_func_like_t function) {
  loom_test_pass_trace_t* trace = loom_test_pass_trace_from_pass(pass);
  IREE_RETURN_IF_ERROR(loom_test_pass_trace_record(
      trace, IREE_SV("test.mark-changed"),
      loom_test_pass_function_name(module, function)));
  if (trace) ++trace->mark_changed_invocation_count;
  loom_pass_mark_changed(pass);
  loom_test_mark_changed_statistics_t* statistics =
      loom_test_mark_changed_statistics(pass);
  ++statistics->invocations;
  ++statistics->synthetic_events;
  const iree_string_view_t function_name =
      loom_test_pass_function_name(module, function);
  const loom_pass_report_detail_field_t fields[] = {
      loom_pass_report_detail_string_field(IREE_SV("event"),
                                           IREE_SV("synthetic-change")),
      loom_pass_report_detail_string_field(IREE_SV("symbol"), function_name),
      loom_pass_report_detail_uint64_field(IREE_SV("synthetic_event_count"),
                                           (uint64_t)1),
  };
  IREE_RETURN_IF_ERROR(loom_pass_report_append_detail(
      pass, IREE_SV("test.mark-changed"), fields, IREE_ARRAYSIZE(fields)));
  return iree_ok_status();
}

static iree_status_t loom_test_options_create(loom_pass_t* pass,
                                              iree_string_view_t options) {
  (void)options;
  loom_test_pass_trace_t* trace = loom_test_pass_trace_from_pass(pass);
  if (trace && pass->decoded_options) {
    ++trace->options_decoded_create_count;
    if (pass->decoded_options->option_count > 0 &&
        pass->decoded_options->options[0].present) {
      trace->decoded_options_count_value =
          pass->decoded_options->options[0].uint32_value;
    }
    if (pass->decoded_options->option_count > 1 &&
        pass->decoded_options->options[1].present) {
      trace->decoded_options_mode_index =
          pass->decoded_options->options[1].enum_value_index;
    }
    if (pass->decoded_options->option_count > 2 &&
        pass->decoded_options->options[2].present) {
      trace->decoded_options_string_value =
          pass->decoded_options->options[2].string_value;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_test_options_run(loom_pass_t* pass,
                                           loom_module_t* module,
                                           loom_func_like_t function) {
  loom_test_pass_trace_t* trace = loom_test_pass_trace_from_pass(pass);
  IREE_RETURN_IF_ERROR(loom_test_pass_trace_record(
      trace, IREE_SV("test.options"),
      loom_test_pass_function_name(module, function)));
  if (trace) ++trace->options_invocation_count;
  loom_test_invocation_statistics_t* statistics =
      loom_test_invocation_statistics(pass);
  ++statistics->invocations;
  return iree_ok_status();
}

static iree_status_t loom_test_required_create(loom_pass_t* pass,
                                               iree_string_view_t options) {
  (void)pass;
  (void)options;
  return iree_ok_status();
}

static iree_status_t loom_test_required_run(loom_pass_t* pass,
                                            loom_module_t* module,
                                            loom_func_like_t function) {
  (void)pass;
  (void)module;
  (void)function;
  return iree_ok_status();
}

static iree_status_t loom_test_requires_target_run(loom_pass_t* pass,
                                                   loom_module_t* module,
                                                   loom_func_like_t function) {
  (void)pass;
  (void)module;
  (void)function;
  return iree_ok_status();
}

static bool loom_test_resource_lifetime_type_matches(loom_type_t type,
                                                     void* user_data) {
  (void)user_data;
  return loom_type_is_pool(type);
}

static iree_status_t loom_test_resource_lifetime_build_release(
    loom_builder_t* builder, loom_value_id_t value_id,
    loom_location_id_t location, void* user_data, loom_op_t** out_op) {
  (void)user_data;
  return loom_test_resource_release_build(builder, value_id, location, out_op);
}

static iree_status_t loom_test_resource_lifetime_run(loom_pass_t* pass,
                                                     loom_module_t* module) {
  const loom_ownership_lifetime_materialization_policy_t policy = {
      .family =
          {
              .name = IREE_SVL("test.resource"),
              .type_matches = loom_test_resource_lifetime_type_matches,
          },
      .build_release = loom_test_resource_lifetime_build_release,
  };
  loom_ownership_lifetime_materialize_options_t options = {
      .arena = pass->arena,
      .emitter = pass->diagnostic_emitter,
      .phase_name = pass->info->name,
      .policies = &policy,
      .policy_count = 1,
  };
  loom_ownership_lifetime_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_ownership_lifetime_materialize_module(module, &options, &result));
  loom_test_resource_lifetime_statistics_t* statistics =
      loom_test_resource_lifetime_statistics(pass);
  statistics->blocks_checked += (int64_t)result.blocks_checked;
  statistics->ops_checked += (int64_t)result.ops_checked;
  statistics->effects_checked += (int64_t)result.effects_checked;
  statistics->releases_inserted += (int64_t)result.releases_inserted;
  statistics->edges_split += (int64_t)result.edges_split;
  if (result.releases_inserted != 0 || result.edges_split != 0) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}

static iree_status_t loom_test_fail_run(loom_pass_t* pass,
                                        loom_module_t* module) {
  (void)module;
  loom_test_pass_trace_t* trace = loom_test_pass_trace_from_pass(pass);
  IREE_RETURN_IF_ERROR(loom_test_pass_trace_record(trace, IREE_SV("test.fail"),
                                                   IREE_SV("<module>")));
  if (trace) ++trace->fail_invocation_count;
  loom_test_invocation_statistics_t* statistics =
      loom_test_invocation_statistics(pass);
  ++statistics->invocations;
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "intentional test pass failure");
}

static const loom_pass_option_def_t kTestOptionsOptionDefs[] = {
    {
        .name = IREE_SVL("count"),
        .description = IREE_SVL("Synthetic bounded integer option."),
    },
    {
        .name = IREE_SVL("mode"),
        .description = IREE_SVL("Synthetic enum option."),
    },
    {
        .name = IREE_SVL("string"),
        .description = IREE_SVL("Synthetic string option."),
    },
};

static const loom_pass_option_enum_value_t kTestModeValues[] = {
    {.value = IREE_SVL("alpha")},
    {.value = IREE_SVL("beta")},
};

static const loom_pass_option_schema_t kTestOptionsSchema[] = {
    {
        .name = IREE_SVL("count"),
        .kind = LOOM_PASS_OPTION_SCHEMA_UINT32,
        .minimum_uint32 = 1,
        .maximum_uint32 = 8,
    },
    {
        .name = IREE_SVL("mode"),
        .kind = LOOM_PASS_OPTION_SCHEMA_ENUM,
        .enum_values = kTestModeValues,
        .enum_value_count = IREE_ARRAYSIZE(kTestModeValues),
    },
    {
        .name = IREE_SVL("string"),
        .kind = LOOM_PASS_OPTION_SCHEMA_STRING,
    },
};

static const loom_pass_option_def_t kTestRequiredOptionDefs[] = {
    {
        .name = IREE_SVL("required"),
        .description = IREE_SVL("Synthetic required option."),
    },
};

static const loom_pass_option_schema_t kTestRequiredSchema[] = {
    {
        .name = IREE_SVL("required"),
        .kind = LOOM_PASS_OPTION_SCHEMA_STRING,
        .flags = LOOM_PASS_OPTION_SCHEMA_REQUIRED,
    },
};

static const loom_pass_requirement_def_t kTestRequiresTargetRequirements[] = {
    {
        .capability_type = &loom_test_pass_target_record_capability_type,
        .key = IREE_SVL("target.record"),
        .description = IREE_SVL("Synthetic target record availability."),
    },
};

static const loom_pass_info_t kTestFailPassInfo = {
    .name = IREE_SVL("test.fail"),
    .description = IREE_SVL("Synthetic module pass that always fails."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_test_invocation_statistics_layout,
};

static const loom_pass_info_t* loom_test_fail_pass_info(void) {
  return &kTestFailPassInfo;
}

static const loom_pass_info_t kTestMarkChangedPassInfo = {
    .name = IREE_SVL("test.mark-changed"),
    .description = IREE_SVL("Synthetic function pass that reports a change."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_test_mark_changed_statistics_layout,
};

static const loom_pass_info_t* loom_test_mark_changed_pass_info(void) {
  return &kTestMarkChangedPassInfo;
}

static const loom_pass_info_t kTestModuleNoopPassInfo = {
    .name = IREE_SVL("test.module-noop"),
    .description = IREE_SVL("Synthetic module no-op pass."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_test_invocation_statistics_layout,
};

static const loom_pass_info_t* loom_test_module_noop_pass_info(void) {
  return &kTestModuleNoopPassInfo;
}

static const loom_pass_info_t kTestNoopPassInfo = {
    .name = IREE_SVL("test.noop"),
    .description = IREE_SVL("Synthetic function no-op pass."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_test_invocation_statistics_layout,
};

static const loom_pass_info_t* loom_test_noop_pass_info(void) {
  return &kTestNoopPassInfo;
}

static const loom_pass_info_t kTestOptionsPassInfo = {
    .name = IREE_SVL("test.options"),
    .description = IREE_SVL("Synthetic function pass with typed options."),
    .kind = LOOM_PASS_FUNCTION,
    .option_defs = kTestOptionsOptionDefs,
    .option_count = IREE_ARRAYSIZE(kTestOptionsOptionDefs),
    .statistic_layout = &loom_test_invocation_statistics_layout,
};

static const loom_pass_info_t* loom_test_options_pass_info(void) {
  return &kTestOptionsPassInfo;
}

static const loom_pass_info_t kTestRequiredPassInfo = {
    .name = IREE_SVL("test.required"),
    .description = IREE_SVL("Synthetic function pass with a required option."),
    .kind = LOOM_PASS_FUNCTION,
    .option_defs = kTestRequiredOptionDefs,
    .option_count = IREE_ARRAYSIZE(kTestRequiredOptionDefs),
};

static const loom_pass_info_t* loom_test_required_pass_info(void) {
  return &kTestRequiredPassInfo;
}

static const loom_pass_info_t kTestResourceLifetimePassInfo = {
    .name = IREE_SVL("test.resource-lifetime"),
    .description =
        IREE_SVL("Materialize explicit test resource lifetime operations."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_test_resource_lifetime_statistics_layout,
};

static const loom_pass_info_t* loom_test_resource_lifetime_pass_info(void) {
  return &kTestResourceLifetimePassInfo;
}

static const loom_pass_info_t kTestRequiresTargetPassInfo = {
    .name = IREE_SVL("test.requires-target"),
    .description = IREE_SVL("Synthetic function pass with a requirement."),
    .kind = LOOM_PASS_FUNCTION,
};

static const loom_pass_info_t* loom_test_requires_target_pass_info(void) {
  return &kTestRequiresTargetPassInfo;
}

static const loom_pass_info_t kTestUnavailablePassInfo = {
    .name = IREE_SVL("test.unavailable"),
    .description = IREE_SVL("Synthetic unavailable function pass."),
    .kind = LOOM_PASS_FUNCTION,
};

static const loom_pass_info_t* loom_test_unavailable_pass_info(void) {
  return &kTestUnavailablePassInfo;
}

static const loom_pass_descriptor_t kTestPassDescriptors[] = {
    {
        .key = IREE_SVL("test.fail"),
        .info = loom_test_fail_pass_info,
        .module_run = loom_test_fail_run,
    },
    {
        .key = IREE_SVL("test.mark-changed"),
        .info = loom_test_mark_changed_pass_info,
        .function_run = loom_test_mark_changed_run,
    },
    {
        .key = IREE_SVL("test.module-noop"),
        .info = loom_test_module_noop_pass_info,
        .module_run = loom_test_module_noop_run,
    },
    {
        .key = IREE_SVL("test.noop"),
        .info = loom_test_noop_pass_info,
        .function_run = loom_test_noop_run,
    },
    {
        .key = IREE_SVL("test.options"),
        .info = loom_test_options_pass_info,
        .function_run = loom_test_options_run,
        .create = loom_test_options_create,
        .option_schema = kTestOptionsSchema,
        .option_schema_count = IREE_ARRAYSIZE(kTestOptionsSchema),
    },
    {
        .key = IREE_SVL("test.required"),
        .info = loom_test_required_pass_info,
        .function_run = loom_test_required_run,
        .create = loom_test_required_create,
        .option_schema = kTestRequiredSchema,
        .option_schema_count = IREE_ARRAYSIZE(kTestRequiredSchema),
    },
    {
        .key = IREE_SVL("test.requires-target"),
        .info = loom_test_requires_target_pass_info,
        .function_run = loom_test_requires_target_run,
        .requirement_defs = kTestRequiresTargetRequirements,
        .requirement_count = IREE_ARRAYSIZE(kTestRequiresTargetRequirements),
    },
    {
        .key = IREE_SVL("test.resource-lifetime"),
        .info = loom_test_resource_lifetime_pass_info,
        .module_run = loom_test_resource_lifetime_run,
    },
    {
        .key = IREE_SVL("test.unavailable"),
        .info = loom_test_unavailable_pass_info,
        .function_run = loom_test_noop_run,
        .flags = LOOM_PASS_DESCRIPTOR_UNAVAILABLE,
        .unavailable_reason = IREE_SVL("disabled for test"),
    },
};

const loom_pass_registry_t loom_test_pass_registry_storage = {
    .descriptors = kTestPassDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kTestPassDescriptors),
};

const loom_pass_registry_t* loom_test_pass_registry(void) {
  return &loom_test_pass_registry_storage;
}
