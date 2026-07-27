// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TESTING_DIAGNOSTIC_MATCHERS_H_
#define LOOM_TESTING_DIAGNOSTIC_MATCHERS_H_

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/error/diagnostic.h"
#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"

namespace loom::testing {

// A deep-copied lightweight diagnostic emission payload for exact verifier and
// lowering tests.
struct CapturedDiagnosticEmission {
  // Structured error definition emitted by the producer.
  const loom_error_def_t* error = nullptr;
  // Operation associated with the diagnostic emission.
  const loom_op_t* op = nullptr;
  // String parameters copied in parameter order.
  std::vector<std::string> string_params;
  // String-list parameters copied in parameter order.
  std::vector<std::vector<std::string>> string_list_params;
  // Type parameters copied in parameter order.
  std::vector<loom_type_t> type_params;
  // I64 parameters copied in parameter order.
  std::vector<int64_t> i64_params;
  // U32 parameters copied in parameter order.
  std::vector<uint32_t> u32_params;
  // U64 parameters copied in parameter order.
  std::vector<uint64_t> u64_params;
  // Field references copied for all parameters in full parameter order.
  std::vector<loom_diagnostic_field_ref_t> field_refs;
  // Number of related ops carried by the emission.
  iree_host_size_t related_count = 0;
  // Deep-copied parameters in full parameter order.
  std::vector<loom_diagnostic_param_t> params;
  // Storage backing deep-copied string-list parameters in |params|.
  std::vector<std::vector<iree_string_view_t>> string_list_param_views;
};

struct DiagnosticEmissionCapture {
  // Deep-copied emissions captured in emission order.
  std::vector<CapturedDiagnosticEmission> emissions;

  void Reset() { emissions.clear(); }

  iree_diagnostic_emitter_t emitter() {
    return iree_diagnostic_emitter_t{
        /*.fn=*/CaptureEmission,
        /*.user_data=*/this,
    };
  }

 private:
  static std::string CopyStringView(iree_string_view_t value) {
    if (!value.data) {
      EXPECT_EQ(value.size, 0u);
      return "";
    }
    return std::string(value.data, value.size);
  }

  static iree_status_t CaptureEmission(
      void* user_data, const loom_diagnostic_emission_t* emission) {
    auto* capture = static_cast<DiagnosticEmissionCapture*>(user_data);
    CapturedDiagnosticEmission entry;
    entry.error = emission->error;
    entry.op = emission->op;
    entry.related_count = emission->related_op_count;
    entry.params.reserve(emission->param_count);
    entry.string_params.reserve(emission->param_count);
    entry.string_list_params.reserve(emission->param_count);
    entry.string_list_param_views.reserve(emission->param_count);
    for (iree_host_size_t i = 0; i < emission->param_count; ++i) {
      loom_diagnostic_param_t param = emission->params[i];
      entry.field_refs.push_back(param.field_ref);
      if (param.kind == LOOM_PARAM_STRING) {
        entry.string_params.emplace_back(CopyStringView(param.string));
        const std::string& copy = entry.string_params.back();
        param.string = iree_make_string_view(copy.data(), copy.size());
      } else if (param.kind == LOOM_PARAM_I64) {
        entry.i64_params.push_back(param.i64);
      } else if (param.kind == LOOM_PARAM_TYPE) {
        entry.type_params.push_back(param.type);
      } else if (param.kind == LOOM_PARAM_U32) {
        entry.u32_params.push_back(param.u32);
      } else if (param.kind == LOOM_PARAM_U64) {
        entry.u64_params.push_back(param.u64);
      } else if (param.kind == LOOM_PARAM_STRING_LIST) {
        std::vector<std::string> string_list;
        string_list.reserve(param.string_list.count);
        for (iree_host_size_t j = 0; j < param.string_list.count; ++j) {
          string_list.emplace_back(CopyStringView(param.string_list.values[j]));
        }
        entry.string_list_params.push_back(std::move(string_list));
        const std::vector<std::string>& copy = entry.string_list_params.back();
        std::vector<iree_string_view_t> views;
        views.reserve(copy.size());
        for (const std::string& value : copy) {
          views.push_back(iree_make_string_view(value.data(), value.size()));
        }
        entry.string_list_param_views.push_back(std::move(views));
        const std::vector<iree_string_view_t>& param_views =
            entry.string_list_param_views.back();
        param.string_list = loom_diagnostic_string_list_t{
            /*.values=*/param_views.data(),
            /*.count=*/param_views.size(),
        };
      }
      entry.params.push_back(param);
    }
    capture->emissions.push_back(std::move(entry));
    return iree_ok_status();
  }
};

struct CapturedRelatedLocation {
  std::string label;
  loom_source_range_t source_location = {};
  bool has_source_range = false;
  std::vector<loom_highlight_range_t> highlights;

 private:
  friend struct DiagnosticCapture;

  std::string source_location_filename_storage;
  std::string source_location_source_storage;
};

// A deep-copied diagnostic payload for exact test assertions.
//
// Error def pointers reference stable .rodata definitions, so pointer equality
// is the right test for "did this specific structured error fire?".
struct CapturedDiagnostic {
  const loom_error_def_t* error = nullptr;
  loom_diagnostic_severity_t severity = LOOM_DIAGNOSTIC_ERROR;
  loom_emitter_t emitter = LOOM_EMITTER_COUNT_;

  loom_source_range_t origin = {};
  loom_source_range_t source_location = {};
  uint32_t origin_line = 0;
  uint32_t origin_column = 0;
  uint32_t origin_end_column = 0;
  bool has_source_range = false;
  std::string filename;
  std::string source_text;

  std::vector<loom_highlight_range_t> highlights;
  iree_host_size_t highlight_omitted_count = 0;
  std::vector<CapturedRelatedLocation> related_locations;
  iree_host_size_t related_location_omitted_count = 0;
  std::vector<loom_diagnostic_param_t> params;

 private:
  friend struct DiagnosticCapture;

  std::vector<std::string> string_storage;
  std::string origin_filename_storage;
  std::string origin_source_storage;
  std::string source_location_filename_storage;
  std::string source_location_source_storage;
};

struct DiagnosticCapture {
  std::vector<CapturedDiagnostic> diagnostics;

  void Reset() { diagnostics.clear(); }

  loom_diagnostic_sink_t sink() {
    return loom_diagnostic_sink_t{
        /*.fn=*/CaptureDiagnostic,
        /*.user_data=*/this,
    };
  }

 private:
  static std::string CopyStringView(iree_string_view_t value) {
    if (!value.data) {
      EXPECT_EQ(value.size, 0u);
      return "";
    }
    return std::string(value.data, value.size);
  }

  static void CopySourceRange(loom_source_range_t source_range,
                              std::string* filename_storage,
                              std::string* source_storage,
                              loom_source_range_t* out_source_range) {
    *filename_storage = CopyStringView(source_range.filename);
    *source_storage = CopyStringView(source_range.source);
    *out_source_range = source_range;
    out_source_range->filename = iree_make_string_view(
        filename_storage->data(), filename_storage->size());
    out_source_range->source =
        iree_make_string_view(source_storage->data(), source_storage->size());
  }

  static iree_status_t CaptureDiagnostic(void* user_data,
                                         const loom_diagnostic_t* diagnostic) {
    auto* capture = static_cast<DiagnosticCapture*>(user_data);
    CapturedDiagnostic entry;
    entry.error = diagnostic->error;
    entry.severity = diagnostic->severity;
    entry.emitter = diagnostic->emitter;

    CopySourceRange(diagnostic->origin, &entry.origin_filename_storage,
                    &entry.origin_source_storage, &entry.origin);
    CopySourceRange(
        diagnostic->source_location, &entry.source_location_filename_storage,
        &entry.source_location_source_storage, &entry.source_location);
    entry.origin_line = diagnostic->origin.start_line;
    entry.origin_column = diagnostic->origin.start_column;
    entry.origin_end_column = diagnostic->origin.end_column;
    entry.has_source_range = diagnostic->origin.source.size > 0;
    entry.filename = entry.origin_filename_storage;
    entry.source_text = entry.origin_source_storage;

    entry.highlights.reserve(diagnostic->highlight_count);
    entry.highlight_omitted_count = diagnostic->highlight_omitted_count;
    if (diagnostic->highlight_count > 0) {
      EXPECT_NE(diagnostic->highlights, nullptr);
    }
    if (diagnostic->highlights) {
      for (iree_host_size_t i = 0; i < diagnostic->highlight_count; ++i) {
        entry.highlights.push_back(diagnostic->highlights[i]);
      }
    }

    entry.related_locations.reserve(diagnostic->related_location_count);
    entry.related_location_omitted_count =
        diagnostic->related_location_omitted_count;
    if (diagnostic->related_location_count > 0) {
      EXPECT_NE(diagnostic->related_locations, nullptr);
    }
    if (diagnostic->related_locations) {
      for (iree_host_size_t i = 0; i < diagnostic->related_location_count;
           ++i) {
        const loom_diagnostic_related_location_t* related =
            &diagnostic->related_locations[i];
        CapturedRelatedLocation copied_related;
        copied_related.label = CopyStringView(related->label);
        CopySourceRange(related->source_location,
                        &copied_related.source_location_filename_storage,
                        &copied_related.source_location_source_storage,
                        &copied_related.source_location);
        copied_related.has_source_range =
            copied_related.source_location.source.size > 0;
        copied_related.highlights.reserve(related->highlight_count);
        if (related->highlight_count > 0) {
          EXPECT_NE(related->highlights, nullptr);
        }
        if (related->highlights) {
          for (iree_host_size_t highlight_index = 0;
               highlight_index < related->highlight_count; ++highlight_index) {
            copied_related.highlights.push_back(
                related->highlights[highlight_index]);
          }
        }
        entry.related_locations.push_back(std::move(copied_related));
      }
    }

    entry.params.reserve(diagnostic->param_count);
    entry.string_storage.reserve(diagnostic->param_count);
    for (iree_host_size_t i = 0; i < diagnostic->param_count; ++i) {
      loom_diagnostic_param_t param = diagnostic->params[i];
      if (param.kind == LOOM_PARAM_STRING) {
        entry.string_storage.emplace_back(CopyStringView(param.string));
        const std::string& copy = entry.string_storage.back();
        param.string = iree_make_string_view(copy.data(), copy.size());
      }
      entry.params.push_back(param);
    }

    capture->diagnostics.push_back(std::move(entry));
    return iree_ok_status();
  }
};

inline const CapturedDiagnostic* FindDiagnostic(
    const DiagnosticCapture& capture, const loom_error_def_t* expected_error) {
  for (const auto& diagnostic : capture.diagnostics) {
    if (diagnostic.error == expected_error) {
      return &diagnostic;
    }
  }
  return nullptr;
}

inline void ExpectError(const CapturedDiagnostic& diagnostic,
                        const loom_error_def_t* expected_error,
                        loom_emitter_t expected_emitter) {
  EXPECT_EQ(diagnostic.error, expected_error);
  EXPECT_EQ(diagnostic.severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_EQ(diagnostic.emitter, expected_emitter);
}

inline void ExpectError(const CapturedDiagnostic& diagnostic,
                        const loom_error_def_t* expected_error) {
  ExpectError(diagnostic, expected_error, LOOM_EMITTER_PARSER);
}

inline std::string GetStringParam(const CapturedDiagnostic& diagnostic,
                                  iree_host_size_t param_index) {
  EXPECT_LT(param_index, diagnostic.params.size());
  if (param_index >= diagnostic.params.size()) return "";
  EXPECT_EQ(diagnostic.params[param_index].kind, LOOM_PARAM_STRING);
  return std::string(diagnostic.params[param_index].string.data,
                     diagnostic.params[param_index].string.size);
}

inline void ExpectU32Param(const CapturedDiagnostic& diagnostic,
                           iree_host_size_t param_index, uint32_t expected) {
  ASSERT_LT(param_index, diagnostic.params.size());
  EXPECT_EQ(diagnostic.params[param_index].kind, LOOM_PARAM_U32);
  EXPECT_EQ(diagnostic.params[param_index].u32, expected);
}

inline void ExpectI64Param(const CapturedDiagnostic& diagnostic,
                           iree_host_size_t param_index, int64_t expected) {
  ASSERT_LT(param_index, diagnostic.params.size());
  EXPECT_EQ(diagnostic.params[param_index].kind, LOOM_PARAM_I64);
  EXPECT_EQ(diagnostic.params[param_index].i64, expected);
}

inline void ExpectTypeParam(const CapturedDiagnostic& diagnostic,
                            iree_host_size_t param_index,
                            loom_type_t expected) {
  ASSERT_LT(param_index, diagnostic.params.size());
  EXPECT_EQ(diagnostic.params[param_index].kind, LOOM_PARAM_TYPE);
  EXPECT_TRUE(loom_type_equal(diagnostic.params[param_index].type, expected));
}

inline void ExpectFieldRefParam(const CapturedDiagnostic& diagnostic,
                                iree_host_size_t param_index,
                                loom_diagnostic_field_kind_t expected_kind,
                                uint16_t expected_index,
                                uint16_t expected_occurrence = 0) {
  ASSERT_LT(param_index, diagnostic.params.size());
  EXPECT_EQ(diagnostic.params[param_index].field_ref.kind, expected_kind);
  EXPECT_EQ(diagnostic.params[param_index].field_ref.index, expected_index);
  EXPECT_EQ(diagnostic.params[param_index].field_ref.occurrence,
            expected_occurrence);
}

inline void ExpectNoFieldRefParam(const CapturedDiagnostic& diagnostic,
                                  iree_host_size_t param_index) {
  ASSERT_LT(param_index, diagnostic.params.size());
  EXPECT_EQ(diagnostic.params[param_index].field_ref.kind,
            LOOM_DIAGNOSTIC_FIELD_NONE);
  EXPECT_EQ(diagnostic.params[param_index].field_ref.index, 0u);
  EXPECT_EQ(diagnostic.params[param_index].field_ref.occurrence, 0u);
}

inline void ExpectHighlightFieldRef(const CapturedDiagnostic& diagnostic,
                                    iree_host_size_t highlight_index,
                                    loom_diagnostic_field_kind_t expected_kind,
                                    uint16_t expected_index,
                                    iree_host_size_t expected_param_index,
                                    uint16_t expected_occurrence = 0) {
  ASSERT_LT(highlight_index, diagnostic.highlights.size());
  EXPECT_EQ(diagnostic.highlights[highlight_index].field_ref.kind,
            expected_kind);
  EXPECT_EQ(diagnostic.highlights[highlight_index].field_ref.index,
            expected_index);
  EXPECT_EQ(diagnostic.highlights[highlight_index].field_ref.occurrence,
            expected_occurrence);
  EXPECT_EQ(diagnostic.highlights[highlight_index].param_index,
            expected_param_index);
}

inline void ExpectRelatedHighlightFieldRef(
    const CapturedDiagnostic& diagnostic, iree_host_size_t related_index,
    iree_host_size_t highlight_index,
    loom_diagnostic_field_kind_t expected_kind, uint16_t expected_index,
    uint16_t expected_occurrence = 0) {
  ASSERT_LT(related_index, diagnostic.related_locations.size());
  const CapturedRelatedLocation& related =
      diagnostic.related_locations[related_index];
  ASSERT_LT(highlight_index, related.highlights.size());
  EXPECT_EQ(related.highlights[highlight_index].field_ref.kind, expected_kind);
  EXPECT_EQ(related.highlights[highlight_index].field_ref.index,
            expected_index);
  EXPECT_EQ(related.highlights[highlight_index].field_ref.occurrence,
            expected_occurrence);
}

}  // namespace loom::testing

#endif  // LOOM_TESTING_DIAGNOSTIC_MATCHERS_H_
