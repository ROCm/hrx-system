// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>

#include "iree/base/internal/arena.h"
#include "loom/format/text/printer.h"
#include "loom/import/cxx/import.h"
#include "loom/import/cxx/testing/source_generator.h"
#include "loom/ops/op_registry.h"

namespace {

constexpr size_t kImportOptionByteCount = 2;
constexpr size_t kMaximumInputByteCount =
    loom::cxx_import::testing::kMaximumMutationByteCount +
    kImportOptionByteCount;

constexpr std::array<loom_cxx_data_model_t, 3> kDataModels = {
    LOOM_CXX_DATA_MODEL_LP64,
    LOOM_CXX_DATA_MODEL_LLP64,
    LOOM_CXX_DATA_MODEL_ILP32,
};

[[noreturn]] void Fail(const std::string& source, const char* message) {
  std::fprintf(stderr, "%s\n--- generated source ---\n%s\n", message,
               source.c_str());
  std::abort();
}

void Check(const std::string& source, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  std::fprintf(stderr, "--- generated source ---\n%s\n", source.c_str());
  iree_status_abort(status);
}

class ImportFuzzState {
 public:
  ImportFuzzState() {
    loom_context_initialize(iree_allocator_system(), &context_);
    Check({}, loom_op_registry_register_all_dialects(&context_));
    Check({}, loom_context_finalize(&context_));

    const std::string representative_source =
        loom::cxx_import::testing::GenerateSource({});
    for (size_t i = 0; i < kDataModels.size(); ++i) {
      loom_cxx_import_options_t options;
      loom_cxx_import_options_initialize(&options);
      options.data_model = kDataModels[i];
      Check(representative_source,
            loom_cxx_preamble_create(
                iree_make_string_view(representative_source.data(),
                                      representative_source.size()),
                IREE_SV("/fuzz/generated.cxx"), &options,
                iree_allocator_system(), &preambles_[i]));
      if (!preambles_[i]) {
        Fail(representative_source, "generated source preamble was rejected");
      }
    }
  }

  ~ImportFuzzState() {
    for (loom_cxx_preamble_t* preamble : preambles_) {
      loom_cxx_preamble_release(preamble);
    }
    loom_context_deinitialize(&context_);
  }

  std::string ImportAndPrint(const std::string& source, size_t data_model_index,
                             loom_cxx_fp_model_t fp_model) {
    iree_arena_block_pool_t block_pool;
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &block_pool);
    loom_cxx_import_options_t options;
    loom_cxx_import_options_initialize(&options);
    options.data_model = kDataModels[data_model_index];
    options.fp_model = fp_model;
    options.preamble = preambles_[data_model_index];
    loom_module_t* module = nullptr;
    Check(source, loom_cxx_import(
                      iree_make_string_view(source.data(), source.size()),
                      IREE_SV("/fuzz/generated.cxx"), &context_, &block_pool,
                      &options, iree_allocator_system(), &module));
    if (!module) {
      iree_arena_block_pool_deinitialize(&block_pool);
      Fail(source, "generated source was rejected");
    }

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    Check(source, loom_text_print_module_to_builder(module, &builder,
                                                    LOOM_TEXT_PRINT_DEFAULT));
    std::string text(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    loom_module_free(module);
    iree_arena_block_pool_deinitialize(&block_pool);
    return text;
  }

 private:
  // Finalized shared dialect registry used by every generated module.
  loom_context_t context_ = {};
  // Immutable semantic prefixes for each supported source data model.
  std::array<loom_cxx_preamble_t*, kDataModels.size()> preambles_ = {};
};

ImportFuzzState& GetImportFuzzState() {
  static ImportFuzzState state;
  return state;
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  (void)argc;
  (void)argv;
  // Keep one-time registry and preamble construction outside the per-input
  // coverage window. Otherwise libFuzzer attributes setup counters to whichever
  // corpus file happens to run first.
  (void)GetImportFuzzState();
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > kMaximumInputByteCount) {
    return 0;
  }
  ImportFuzzState& state = GetImportFuzzState();
  const size_t data_model_index = size ? data[0] % kDataModels.size() : 0;
  const loom_cxx_fp_model_t fp_model = size > 1 && (data[1] & 1)
                                           ? LOOM_CXX_FP_MODEL_FAST
                                           : LOOM_CXX_FP_MODEL_PRECISE;
  const std::span<const uint8_t> input =
      size ? std::span<const uint8_t>(data, size) : std::span<const uint8_t>();
  const size_t source_offset = std::min(size, kImportOptionByteCount);
  const std::string source =
      loom::cxx_import::testing::GenerateSource(input.subspan(source_offset));
  if (source.size() >
      loom::cxx_import::testing::kMaximumGeneratedSourceByteCount) {
    Fail(source, "generated source exceeded its structural bound");
  }
  const std::string first =
      state.ImportAndPrint(source, data_model_index, fp_model);
  const std::string second =
      state.ImportAndPrint(source, data_model_index, fp_model);
  if (first != second) {
    Fail(source, "independent imports did not print identically");
  }
  return 0;
}
