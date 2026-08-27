// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/dependency_report.h"

#include "loom/util/json.h"

typedef struct loom_link_dependency_named_flag_t {
  uint32_t flag;
  iree_string_view_t name;
} loom_link_dependency_named_flag_t;

static const loom_link_dependency_named_flag_t
    loom_link_dependency_interface_names[] = {
        {LOOM_SYMBOL_INTERFACE_FUNC_LIKE, IREE_SVL("func_like")},
        {LOOM_SYMBOL_INTERFACE_GLOBAL, IREE_SVL("global")},
        {LOOM_SYMBOL_INTERFACE_EXECUTABLE, IREE_SVL("executable")},
        {LOOM_SYMBOL_INTERFACE_RECORD, IREE_SVL("record")},
        {LOOM_SYMBOL_INTERFACE_TARGET, IREE_SVL("target")},
        {LOOM_SYMBOL_INTERFACE_CONFIG, IREE_SVL("config")},
        {LOOM_SYMBOL_INTERFACE_RODATA, IREE_SVL("rodata")},
        {LOOM_SYMBOL_INTERFACE_KERNEL, IREE_SVL("kernel")},
        {LOOM_SYMBOL_INTERFACE_CALLABLE, IREE_SVL("callable")},
        {LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM, IREE_SVL("command_program")},
        {LOOM_SYMBOL_INTERFACE_TEMPLATE_FAMILY, IREE_SVL("template_family")},
        {LOOM_SYMBOL_INTERFACE_TEMPLATE_PROVIDER,
         IREE_SVL("template_provider")},
        {LOOM_SYMBOL_INTERFACE_KERNEL_ENTRY, IREE_SVL("kernel_entry")},
};

static const loom_link_dependency_named_flag_t
    loom_link_dependency_usage_names[] = {
        {LOOM_LINK_DEPENDENCY_USAGE_FLAG_EXACT, IREE_SVL("exact")},
        {LOOM_LINK_DEPENDENCY_USAGE_FLAG_INTERFACE, IREE_SVL("interface")},
        {LOOM_LINK_DEPENDENCY_USAGE_FLAG_TEMPLATE, IREE_SVL("template")},
};

static iree_string_view_t loom_link_dependency_ownership_name(
    loom_link_dependency_ownership_t ownership) {
  switch (ownership) {
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_LOCAL:
      return IREE_SV("local");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_DIRECT:
      return IREE_SV("direct");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_OPEN:
      return IREE_SV("open");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_MISSING_DIRECT:
      return IREE_SV("missing_direct");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_UNSATISFIED:
      return IREE_SV("unsatisfied");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_INACCESSIBLE:
      return IREE_SV("inaccessible");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_INCOMPATIBLE:
      return IREE_SV("incompatible");
  }
  return IREE_SV("unknown");
}

static iree_string_view_t loom_link_dependency_resolution_name(
    loom_link_dependency_resolution_t resolution) {
  switch (resolution) {
    case LOOM_LINK_DEPENDENCY_RESOLUTION_NOT_APPLICABLE:
      return IREE_SV("not_applicable");
    case LOOM_LINK_DEPENDENCY_RESOLUTION_LOCAL:
      return IREE_SV("local");
    case LOOM_LINK_DEPENDENCY_RESOLUTION_UNIQUE:
      return IREE_SV("unique");
    case LOOM_LINK_DEPENDENCY_RESOLUTION_DEFERRED:
      return IREE_SV("deferred");
    case LOOM_LINK_DEPENDENCY_RESOLUTION_UNRESOLVED:
      return IREE_SV("unresolved");
    case LOOM_LINK_DEPENDENCY_RESOLUTION_AMBIGUOUS:
      return IREE_SV("ambiguous");
    case LOOM_LINK_DEPENDENCY_RESOLUTION_INCOMPATIBLE:
      return IREE_SV("incompatible");
  }
  return IREE_SV("unknown");
}

static iree_string_view_t loom_link_dependency_candidate_origin_name(
    loom_link_dependency_candidate_origin_t origin) {
  switch (origin) {
    case LOOM_LINK_DEPENDENCY_CANDIDATE_INPUT:
      return IREE_SV("input");
    case LOOM_LINK_DEPENDENCY_CANDIDATE_DIRECT_LIBRARY:
      return IREE_SV("direct");
    case LOOM_LINK_DEPENDENCY_CANDIDATE_TRANSITIVE_LIBRARY:
      return IREE_SV("transitive");
  }
  return IREE_SV("unknown");
}

static iree_string_view_t loom_link_dependency_facet_name(
    loom_link_symbol_facet_kind_t facet) {
  switch (facet) {
    case LOOM_LINK_SYMBOL_FACET_DEFINITION:
      return IREE_SV("body");
    case LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT:
      return IREE_SV("kernel_contract");
    case LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION:
      return IREE_SV("kernel_configuration");
    case LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION:
      return IREE_SV("kernel_implementation");
    case LOOM_LINK_SYMBOL_FACET_COMMAND_CONTRACT:
      return IREE_SV("command_contract");
    case LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION:
      return IREE_SV("command_implementation");
    case LOOM_LINK_SYMBOL_FACET_INVALID:
      return IREE_SV("contract");
  }
  return IREE_SV("unknown");
}

static iree_string_view_t loom_link_dependency_mismatch_name(
    loom_link_func_contract_mismatch_kind_t kind) {
  switch (kind) {
    case LOOM_LINK_FUNC_CONTRACT_MISMATCH_NONE:
      return IREE_SV("none");
    case LOOM_LINK_FUNC_CONTRACT_MISMATCH_FIELD:
      return IREE_SV("field");
    case LOOM_LINK_FUNC_CONTRACT_MISMATCH_COUNT:
      return IREE_SV("count");
    case LOOM_LINK_FUNC_CONTRACT_MISMATCH_TYPE:
      return IREE_SV("type");
  }
  return IREE_SV("unknown");
}

static const loom_link_module_index_symbol_t*
loom_link_dependency_requirement_target_symbol(
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_requirement_t* requirement) {
  if (requirement->kind != LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL) {
    return NULL;
  }
  return loom_link_module_index_symbol_at(analysis->index,
                                          requirement->target.symbol_ordinal);
}

static iree_string_view_t loom_link_dependency_requirement_name(
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_requirement_t* requirement) {
  if (requirement->kind == LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL) {
    return loom_link_dependency_requirement_target_symbol(analysis, requirement)
        ->name;
  }
  return loom_link_module_index_template_family_at(
             analysis->index, requirement->target.template_family_ordinal)
      ->name;
}

static iree_status_t loom_link_dependency_write_symbol_name(
    loom_output_stream_t* stream, iree_string_view_t name) {
  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\"@"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&escape_stream, name));
  return loom_output_stream_write_char(stream, '"');
}

static iree_status_t loom_link_dependency_write_symbol_field(
    loom_json_object_writer_t* writer, iree_string_view_t field_name,
    iree_string_view_t symbol_name) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(writer, field_name));
  return loom_link_dependency_write_symbol_name(writer->stream, symbol_name);
}

static iree_status_t loom_link_dependency_write_named_flags(
    loom_output_stream_t* stream, uint32_t flags,
    const loom_link_dependency_named_flag_t* names,
    iree_host_size_t name_count) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < name_count; ++i) {
    if (iree_any_bit_set(flags, names[i].flag)) {
      IREE_RETURN_IF_ERROR(
          loom_json_array_write_string_element(&array, names[i].name));
    }
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_link_dependency_write_interfaces(
    loom_output_stream_t* stream, loom_symbol_interface_flags_t interfaces) {
  return loom_link_dependency_write_named_flags(
      stream, interfaces, loom_link_dependency_interface_names,
      IREE_ARRAYSIZE(loom_link_dependency_interface_names));
}

static iree_status_t loom_link_dependency_write_source(
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_requirement_t* requirement,
    loom_output_stream_t* stream) {
  const loom_link_module_index_symbol_t* source = NULL;
  if (requirement->first_source_symbol_ordinal !=
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    source = loom_link_module_index_symbol_at(
        analysis->index, requirement->first_source_symbol_ordinal);
  }
  const loom_link_module_index_symbol_t* target =
      loom_link_dependency_requirement_target_symbol(analysis, requirement);
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_symbol_provider(analysis->index,
                                             source ? source : target);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("provider"), provider->name));
  if (source != NULL) {
    IREE_RETURN_IF_ERROR(loom_link_dependency_write_symbol_field(
        &object, IREE_SV("symbol"), source->name));
    const loom_link_symbol_facet_kind_t facet =
        loom_link_module_index_symbol_source_root_facet_kind(
            source, requirement->first_source_root_region_index_plus_one);
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("facet"), loom_link_dependency_facet_name(facet)));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("symbol")));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("facet"), IREE_SV("module")));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_link_dependency_write_mismatch(
    const loom_link_func_contract_mismatch_t* mismatch,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      loom_link_dependency_mismatch_name(mismatch->kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("field"), mismatch->field_name));
  if (mismatch->kind == LOOM_LINK_FUNC_CONTRACT_MISMATCH_COUNT) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("required_count"), mismatch->detail.counts.source));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("candidate_count"), mismatch->detail.counts.selected));
  } else if (mismatch->kind == LOOM_LINK_FUNC_CONTRACT_MISMATCH_TYPE) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("element"), mismatch->detail.type_ordinal));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_link_dependency_write_candidates(
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_requirement_t* requirement,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < requirement->candidates.count; ++i) {
    const loom_link_dependency_candidate_t* candidate =
        &analysis->candidates.values[requirement->candidates.first + i];
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(analysis->index,
                                         candidate->symbol_ordinal);
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(analysis->index,
                                           candidate->provider_ordinal);
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("provider"), provider->name));
    IREE_RETURN_IF_ERROR(loom_link_dependency_write_symbol_field(
        &object, IREE_SV("symbol"), symbol->name));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("origin"),
        loom_link_dependency_candidate_origin_name(candidate->origin)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("exported"),
        iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("definition"),
        iree_any_bit_set(symbol->flags,
                         LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("compatible"), candidate->compatible));
    if (loom_link_func_contract_mismatch_present(&candidate->mismatch)) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("mismatch")));
      IREE_RETURN_IF_ERROR(
          loom_link_dependency_write_mismatch(&candidate->mismatch, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_link_dependency_write_requirement(
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_requirement_t* requirement,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("source")));
  IREE_RETURN_IF_ERROR(
      loom_link_dependency_write_source(analysis, requirement, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("target")));
  loom_json_object_writer_t target;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &target));
  IREE_RETURN_IF_ERROR(loom_link_dependency_write_symbol_field(
      &target, IREE_SV("symbol"),
      loom_link_dependency_requirement_name(analysis, requirement)));
  if (requirement->kind == LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&target, IREE_SV("interfaces")));
    IREE_RETURN_IF_ERROR(loom_link_dependency_write_interfaces(
        stream, requirement->target_interfaces));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &target, IREE_SV("exported"), requirement->exported));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&target));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("occurrence_count"), requirement->occurrence_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("dependency"),
      loom_link_dependency_ownership_name(requirement->ownership)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("resolution"),
      loom_link_dependency_resolution_name(requirement->resolution)));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("candidates")));
  IREE_RETURN_IF_ERROR(
      loom_link_dependency_write_candidates(analysis, requirement, stream));
  return loom_json_object_end(&object);
}

static iree_status_t loom_link_dependency_write_requirements(
    const loom_link_dependency_analysis_t* analysis,
    loom_link_dependency_requirement_kind_t kind,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < analysis->requirements.count; ++i) {
    const loom_link_dependency_requirement_t* requirement =
        &analysis->requirements.values[i];
    if (requirement->kind != kind) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    IREE_RETURN_IF_ERROR(
        loom_link_dependency_write_requirement(analysis, requirement, stream));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_link_dependency_write_direct_providers(
    const loom_link_dependency_analysis_t* analysis,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < analysis->direct_providers.count; ++i) {
    const loom_link_dependency_direct_provider_t* direct =
        &analysis->direct_providers.values[i];
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(analysis->index,
                                           direct->provider_ordinal);
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("provider"), provider->name));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("usage")));
    IREE_RETURN_IF_ERROR(loom_link_dependency_write_named_flags(
        stream, direct->usage_flags, loom_link_dependency_usage_names,
        IREE_ARRAYSIZE(loom_link_dependency_usage_names)));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

bool loom_link_dependency_analysis_succeeded(
    const loom_link_dependency_analysis_t* analysis) {
  for (iree_host_size_t i = 0; i < analysis->requirements.count; ++i) {
    if (!loom_link_dependency_requirement_satisfied(
            &analysis->requirements.values[i])) {
      return false;
    }
  }
  return true;
}

iree_string_view_t loom_link_dependency_diagnostic_code(
    const loom_link_dependency_requirement_t* requirement) {
  if (requirement->resolution == LOOM_LINK_DEPENDENCY_RESOLUTION_AMBIGUOUS) {
    return IREE_SV("LINK/DEPENDENCY/AMBIGUOUS");
  }
  switch (requirement->ownership) {
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_MISSING_DIRECT:
      return IREE_SV("LINK/DEPENDENCY/MISSING_DIRECT");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_UNSATISFIED:
      return IREE_SV("LINK/DEPENDENCY/UNSATISFIED");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_INACCESSIBLE:
      return IREE_SV("LINK/DEPENDENCY/INACCESSIBLE");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_INCOMPATIBLE:
      return IREE_SV("LINK/DEPENDENCY/INCOMPATIBLE");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_LOCAL:
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_DIRECT:
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_OPEN:
      return iree_string_view_empty();
  }
  return IREE_SV("LINK/DEPENDENCY/INVALID");
}

static iree_status_t loom_link_dependency_write_diagnostic_subject(
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_requirement_t* requirement,
    loom_output_stream_t* stream) {
  if (requirement->first_source_symbol_ordinal !=
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    const loom_link_module_index_symbol_t* source =
        loom_link_module_index_symbol_at(
            analysis->index, requirement->first_source_symbol_ordinal);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '@'));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, source->name));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "module"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " requires @"));
  return loom_output_stream_write(
      stream, loom_link_dependency_requirement_name(analysis, requirement));
}

typedef enum loom_link_dependency_candidate_filter_e {
  LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_ALL = 0,
  LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_COMPATIBLE_DEFINITIONS = 1,
  LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_COMPATIBLE_TRANSITIVE = 2,
  LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_COMPATIBLE_PRIVATE = 3,
} loom_link_dependency_candidate_filter_t;

static bool loom_link_dependency_candidate_matches_filter(
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_candidate_t* candidate,
    loom_link_dependency_candidate_filter_t filter) {
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(analysis->index,
                                       candidate->symbol_ordinal);
  switch (filter) {
    case LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_ALL:
      return true;
    case LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_COMPATIBLE_DEFINITIONS:
      return candidate->compatible &&
             iree_any_bit_set(symbol->flags,
                              LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION);
    case LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_COMPATIBLE_TRANSITIVE:
      return candidate->compatible &&
             candidate->origin ==
                 LOOM_LINK_DEPENDENCY_CANDIDATE_TRANSITIVE_LIBRARY;
    case LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_COMPATIBLE_PRIVATE:
      return candidate->compatible &&
             symbol->identity == LOOM_LINK_SYMBOL_IDENTITY_PRIVATE;
  }
  return false;
}

static iree_status_t loom_link_dependency_write_candidate_providers(
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_requirement_t* requirement,
    loom_link_dependency_candidate_filter_t filter,
    loom_output_stream_t* stream) {
  iree_host_size_t emitted_count = 0;
  for (iree_host_size_t i = 0; i < requirement->candidates.count; ++i) {
    const loom_link_dependency_candidate_t* candidate =
        &analysis->candidates.values[requirement->candidates.first + i];
    if (!loom_link_dependency_candidate_matches_filter(analysis, candidate,
                                                       filter)) {
      continue;
    }
    if (emitted_count++ != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(analysis->index,
                                           candidate->provider_ordinal);
    IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, provider->name));
  }
  return iree_ok_status();
}

iree_status_t loom_link_dependency_format_diagnostic(
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_requirement_t* requirement,
    iree_string_view_t component_name, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_link_dependency_write_diagnostic_subject(
      analysis, requirement, stream));
  if (requirement->resolution == LOOM_LINK_DEPENDENCY_RESOLUTION_AMBIGUOUS) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream,
        ", but the supplied libraries expose multiple compatible "
        "definitions: "));
    return loom_link_dependency_write_candidate_providers(
        analysis, requirement,
        LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_COMPATIBLE_DEFINITIONS, stream);
  }
  switch (requirement->ownership) {
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_MISSING_DIRECT: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
          stream, ", but no direct dependency"));
      if (!iree_string_view_is_empty(component_name)) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " of "));
        IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, component_name));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
          stream, " exports a compatible contract; available from "));
      return loom_link_dependency_write_candidate_providers(
          analysis, requirement,
          LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_COMPATIBLE_TRANSITIVE, stream);
    }
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_UNSATISFIED:
      return loom_output_stream_write_cstring(
          stream, ", but no supplied library exports a compatible contract");
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_INACCESSIBLE: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
          stream, ", but the compatible definition is private to "));
      return loom_link_dependency_write_candidate_providers(
          analysis, requirement,
          LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_COMPATIBLE_PRIVATE, stream);
    }
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_INCOMPATIBLE: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
          stream, ", but the available same-name contract is incompatible"));
      if (requirement->candidates.count != 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " in "));
        IREE_RETURN_IF_ERROR(loom_link_dependency_write_candidate_providers(
            analysis, requirement, LOOM_LINK_DEPENDENCY_CANDIDATE_FILTER_ALL,
            stream));
      }
      return iree_ok_status();
    }
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_LOCAL:
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_DIRECT:
    case LOOM_LINK_DEPENDENCY_OWNERSHIP_OPEN:
      return loom_output_stream_write_cstring(
          stream, ", and the dependency is satisfied");
  }
  return loom_output_stream_write_cstring(
      stream, ", but dependency ownership is invalid");
}

iree_status_t loom_link_dependency_format_json(
    const loom_link_dependency_analysis_t* analysis,
    iree_string_view_t component_name, loom_output_stream_t* stream) {
  loom_json_object_writer_t root;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &root));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&root, IREE_SV("schema_version"), 1));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("component"), component_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &root, IREE_SV("succeeded"),
      loom_link_dependency_analysis_succeeded(analysis)));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&root, IREE_SV("summary")));
  loom_json_object_writer_t summary;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &summary));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &summary, IREE_SV("exact_occurrences"),
      analysis->exact_occurrence_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &summary, IREE_SV("template_occurrences"),
      analysis->template_demand_occurrence_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &summary, IREE_SV("direct_dependencies"),
      analysis->direct_providers.count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&summary));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("exact_requirements")));
  IREE_RETURN_IF_ERROR(loom_link_dependency_write_requirements(
      analysis, LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("template_demands")));
  IREE_RETURN_IF_ERROR(loom_link_dependency_write_requirements(
      analysis, LOOM_LINK_DEPENDENCY_REQUIREMENT_TEMPLATE_FAMILY, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("direct_dependencies")));
  IREE_RETURN_IF_ERROR(
      loom_link_dependency_write_direct_providers(analysis, stream));
  return loom_json_object_end(&root);
}
