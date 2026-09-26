// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "src/diagnostic.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "loom/error/error_defs.h"
#include "test/util.h"

namespace {
using loomc::testing::HandlePtr;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;

ResultPtr Capture(const loom_source_range_t& range,
                  const loomc_source_t* source = nullptr) {
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_result_create(LOOMC_RESULT_STATE_FAILED,
                                      loomc_allocator_system(), &result));
  loom_diagnostic_param_t param = loom_param_string(IREE_SV("x"));
  loom_diagnostic_t diagnostic = {};
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.emitter = LOOM_EMITTER_PARSER;
  diagnostic.params = &param;
  diagnostic.param_count = 1;
  diagnostic.source_location = range;
  LOOMC_EXPECT_OK(
      loomc_result_add_loom_diagnostic(result, source, &diagnostic));
  return ResultPtr(result);
}

SourcePtr CreateSource(const char* identifier, const char* contents) {
  loomc_source_options_t options = {};
  options.identifier = loomc_make_cstring_view(identifier);
  options.contents = loomc_make_byte_span(contents, strlen(contents));
  options.storage = LOOMC_SOURCE_STORAGE_COPY;
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

TEST(DiagnosticTest, OwnsRecordedIdentityWithoutText) {
  std::string filename = "virtual/kernel.cxx";
  loom_source_range_t range = {};
  range.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;
  range.filename = iree_make_string_view(filename.data(), filename.size());
  range.start_line = 7;
  range.start_column = 12;
  range.end_line = 7;
  range.end_column = 25;
  auto result = Capture(range);
  filename.assign("released");
  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  const auto& stored = loomc_result_diagnostic_at(result.get(), 0)->range;
  ASSERT_NE(stored.source, nullptr);
  auto identifier = loomc_source_identifier(stored.source);
  EXPECT_EQ(std::string(identifier.data, identifier.size),
            "virtual/kernel.cxx");
  EXPECT_EQ(stored.start_line, 7u);
  EXPECT_EQ(stored.start_column, 12u);
  EXPECT_EQ(stored.end_line, 7u);
  EXPECT_EQ(stored.end_column, 25u);
  EXPECT_EQ(stored.start, 0u);
  EXPECT_EQ(stored.end, 0u);
  EXPECT_EQ(loomc_source_contents(stored.source).data_length, 0u);
}

TEST(DiagnosticTest, InputIdentityAloneCannotSupplyOriginalText) {
  for (const char* input_name : {"input.loombc", "header.h"}) {
    for (const char* original : {"", "original"}) {
      SCOPED_TRACE(input_name);
      SCOPED_TRACE(original);
      auto source = CreateSource(input_name, "different contents");
      std::string text = original;
      loom_source_range_t range = {};
      range.filename = IREE_SV("header.h");
      range.source = iree_make_string_view(text.data(), text.size());
      range.provenance = text.empty()
                             ? LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE
                             : LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
      range.start_line = range.start_column = range.end_line = 1;
      range.end_column = 9;
      range.end = text.size();
      auto result = Capture(range, source.get());
      source.reset();
      text.assign("released");
      ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
      const auto& stored = loomc_result_diagnostic_at(result.get(), 0)->range;
      ASSERT_NE(stored.source, nullptr);
      auto identifier = loomc_source_identifier(stored.source);
      EXPECT_EQ(std::string(identifier.data, identifier.size), "header.h");
      auto contents = loomc_source_contents(stored.source);
      EXPECT_EQ(contents.data_length, strlen(original));
      if (contents.data_length) {
        EXPECT_EQ(std::string(reinterpret_cast<const char*>(contents.data),
                              contents.data_length),
                  original);
      }
    }
  }
}

TEST(DiagnosticTest, RetainsMatchingSourceOwner) {
  auto source = CreateSource("source.loom", "original");
  auto contents = loomc_source_contents(source.get());
  loom_source_range_t range = {};
  range.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  range.filename = IREE_SV("source.loom");
  range.source = iree_make_string_view(
      reinterpret_cast<const char*>(contents.data), contents.data_length);
  range.start_line = range.start_column = range.end_line = 1;
  range.end_column = 9;
  range.end = contents.data_length;
  auto result = Capture(range, source.get());
  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  const auto& stored = loomc_result_diagnostic_at(result.get(), 0)->range;
  EXPECT_EQ(stored.source, source.get());
  source.reset();
  EXPECT_EQ(loomc_source_contents(stored.source).data, contents.data);
}

TEST(DiagnosticTest, RelatedLocationsOwnLabelsAndShareOnlyMatchingSnapshots) {
  for (bool has_contents : {false, true}) {
    SCOPED_TRACE(has_contents);
    auto input = CreateSource("input.loom", "original");
    const auto contents = loomc_source_contents(input.get());
    std::string filename = "header.h";
    std::string text = "x\ndeclaration";
    std::string label = "defined here";
    loom_source_range_t primary = {};
    primary.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;
    primary.filename = IREE_SV("input.loom");
    primary.start_line = primary.start_column = primary.end_line = 1;
    primary.end_column = 9;
    if (has_contents) {
      primary.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
      primary.source = iree_make_string_view(
          reinterpret_cast<const char*>(contents.data), contents.data_length);
      primary.end = contents.data_length;
    }
    loom_diagnostic_related_location_t related[4] = {};
    related[0].label = IREE_SV("same source");
    related[0].source_location = primary;
    related[1].label = iree_make_string_view(label.data(), label.size());
    auto& header = related[1].source_location;
    header.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;
    header.filename = iree_make_string_view(filename.data(), filename.size());
    header.start_line = header.end_line = 2;
    header.start_column = 3;
    header.end_column = 7;
    if (has_contents) {
      header.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
      header.source = iree_make_string_view(text.data(), text.size());
      header.start = 4;
      header.end = 8;
    }
    related[2] = related[1];
    related[2].label = IREE_SV("same snapshot");
    related[3] = related[1];
    related[3].label = IREE_SV("another snapshot");
    related[3].source_location.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
    related[3].source_location.source = IREE_SV("x\ndifferent declaration");
    related[3].source_location.start = 4;
    related[3].source_location.end = 8;

    loom_diagnostic_param_t param = loom_param_string(IREE_SV("x"));
    loom_diagnostic_t diagnostic = {};
    diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
    diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
    diagnostic.emitter = LOOM_EMITTER_PARSER;
    diagnostic.params = &param;
    diagnostic.param_count = 1;
    diagnostic.source_location = primary;
    diagnostic.related_locations = related;
    diagnostic.related_location_count = IREE_ARRAYSIZE(related);
    diagnostic.related_location_omitted_count = 3;
    loomc_result_t* raw_result = nullptr;
    LOOMC_ASSERT_OK(loomc_result_create(LOOMC_RESULT_STATE_FAILED,
                                        loomc_allocator_system(), &raw_result));
    ResultPtr result(raw_result);
    LOOMC_ASSERT_OK(loomc_result_add_loom_diagnostic(result.get(), input.get(),
                                                     &diagnostic));
    input.reset();
    filename.assign("released");
    text.assign("released");
    label.assign("released");
    for (auto& note : related) {
      note = {};
    }

    const auto* stored = loomc_result_diagnostic_at(result.get(), 0);
    ASSERT_EQ(stored->related_location_count, 4u);
    EXPECT_EQ(stored->related_location_omitted_count, 3u);
    const auto* notes = stored->related_locations;
    EXPECT_EQ(notes[0].range.source, stored->range.source);
    EXPECT_EQ(notes[1].range.source, notes[2].range.source);
    EXPECT_NE(notes[1].range.source, notes[3].range.source);
    EXPECT_EQ(std::string(notes[1].label.data, notes[1].label.size),
              "defined here");
    auto identifier = loomc_source_identifier(notes[1].range.source);
    EXPECT_EQ(std::string(identifier.data, identifier.size), "header.h");
    EXPECT_EQ(notes[1].range.start_line, 2u);
    EXPECT_EQ(notes[1].range.start_column, 3u);
    EXPECT_EQ(notes[1].range.end_line, 2u);
    EXPECT_EQ(notes[1].range.end_column, 7u);
    auto snapshot = loomc_source_contents(notes[1].range.source);
    if (has_contents) {
      EXPECT_EQ(std::string(reinterpret_cast<const char*>(snapshot.data),
                            snapshot.data_length),
                "x\ndeclaration");
      EXPECT_EQ(notes[1].range.start, 4u);
      EXPECT_EQ(notes[1].range.end, 8u);
    } else {
      EXPECT_EQ(snapshot.data_length, 0u);
      EXPECT_EQ(notes[1].range.start, 0u);
      EXPECT_EQ(notes[1].range.end, 0u);
    }
    auto other = loomc_source_contents(notes[3].range.source);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(other.data),
                          other.data_length),
              "x\ndifferent declaration");
  }
}

struct DiagnosticAllocator {
  // Allocation ordinal that fails once, or the maximum to allow all requests.
  size_t failure_at = LOOMC_HOST_SIZE_MAX;
  // Number of allocation requests since the start of capture.
  size_t allocation_count = 0;
  // Live allocations that must be released on both success and failure.
  size_t live_count = 0;

  static loomc_status_t Control(void* user_data,
                                loomc_allocator_command_t command,
                                const void* params, void** pointer) {
    auto& self = *static_cast<DiagnosticAllocator*>(user_data);
    const bool was_live = *pointer != nullptr;
    if (command != LOOMC_ALLOCATOR_COMMAND_FREE &&
        self.allocation_count++ == self.failure_at) {
      return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                               "injected diagnostic allocation failure");
    }
    auto system = loomc_allocator_system();
    auto status = system.ctl(system.self, command, params, pointer);
    if (loomc_status_is_ok(status)) {
      if (command == LOOMC_ALLOCATOR_COMMAND_FREE) {
        self.live_count -= was_live;
      } else if (!was_live && *pointer) {
        ++self.live_count;
      }
    }
    return status;
  }
};

TEST(DiagnosticTest,
     FailedRelatedCaptureReleasesSourcesAndLeavesResultReusable) {
  loom_diagnostic_param_t param = loom_param_string(IREE_SV("x"));
  loom_diagnostic_t diagnostic = {};
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.emitter = LOOM_EMITTER_PARSER;
  diagnostic.params = &param;
  diagnostic.param_count = 1;
  diagnostic.source_location.filename = IREE_SV("caller.cxx");
  diagnostic.source_location.source = IREE_SV("caller");
  diagnostic.source_location.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  diagnostic.source_location.start_line = diagnostic.source_location.end_line =
      1;
  diagnostic.source_location.start_column = 1;
  diagnostic.source_location.end_column = 7;
  diagnostic.source_location.end = 6;
  loom_diagnostic_related_location_t related[2] = {};
  related[0].label = IREE_SV("defined here");
  related[0].source_location.filename = IREE_SV("callee.h");
  related[0].source_location.source = IREE_SV("declaration");
  related[0].source_location.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  related[0].source_location.start_line = related[0].source_location.end_line =
      1;
  related[0].source_location.start_column = 1;
  related[0].source_location.end_column = 12;
  related[0].source_location.end = 11;
  related[1] = related[0];
  related[1].label = IREE_SV("same source retained again");
  diagnostic.related_locations = related;
  diagnostic.related_location_count = IREE_ARRAYSIZE(related);

  DiagnosticAllocator baseline;
  loomc_result_t* baseline_result = nullptr;
  LOOMC_ASSERT_OK(loomc_result_create(
      LOOMC_RESULT_STATE_FAILED,
      loomc_allocator_t{&baseline, DiagnosticAllocator::Control},
      &baseline_result));
  ResultPtr baseline_owner(baseline_result);
  baseline.allocation_count = 0;
  LOOMC_ASSERT_OK(
      loomc_result_add_loom_diagnostic(baseline_result, nullptr, &diagnostic));
  const size_t allocation_count = baseline.allocation_count;
  baseline_owner.reset();
  EXPECT_EQ(baseline.live_count, 0u);
  for (size_t i = 0; i < allocation_count; ++i) {
    SCOPED_TRACE(i);
    DiagnosticAllocator failing;
    loomc_result_t* result = nullptr;
    auto allocator = loomc_allocator_t{&failing, DiagnosticAllocator::Control};
    LOOMC_ASSERT_OK(
        loomc_result_create(LOOMC_RESULT_STATE_FAILED, allocator, &result));
    ResultPtr owner(result);
    failing.allocation_count = 0;
    failing.failure_at = i;
    LOOMC_EXPECT_STATUS_IS(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        loomc_result_add_loom_diagnostic(result, nullptr, &diagnostic));
    EXPECT_EQ(loomc_result_diagnostic_count(result), 0u);
    failing.failure_at = LOOMC_HOST_SIZE_MAX;
    LOOMC_ASSERT_OK(
        loomc_result_add_loom_diagnostic(result, nullptr, &diagnostic));
    ASSERT_EQ(loomc_result_diagnostic_count(result), 1u);
    EXPECT_EQ(loomc_result_diagnostic_at(result, 0)->related_location_count,
              2u);
    owner.reset();
    EXPECT_EQ(failing.live_count, 0u);
  }
}

}  // namespace
