// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/preamble.h"

#include <cxx/token.h>

#include <atomic>
#include <future>
#include <latch>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/import/cxx/source/frontend.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

TEST(PreambleTest, RetainedConsumersOutliveThePublishingReference) {
  static constexpr std::string_view kSource =
      "#include \"library.h\"\n"
      "constexpr auto consumer_file = __FILE__;\n"
      "static_assert(__LDBL_MANT_DIG__ == 53);\n"
      "static_assert(__LDBL_MAX_EXP__ == 1024);\n";
  struct ProviderState {
    // Header bytes available only while the publisher constructs the preamble.
    std::string header = "constexpr int header_value = 42;\n";
    // True after the publisher releases its catalog reference.
    std::atomic<bool> closed = false;
    // Total source-provider calls across producer and consumers.
    std::atomic<unsigned> call_count = 0;
  } provider_state;

  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.source_provider = {
      [](void* user_data, iree_string_view_t path, bool* out_found,
         iree_string_view_t* out_source) {
        auto& state = *static_cast<ProviderState*>(user_data);
        state.call_count.fetch_add(1, std::memory_order_relaxed);
        if (state.closed.load(std::memory_order_acquire)) {
          return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "retired catalog was queried");
        }
        *out_found = iree_string_view_equal(path, IREE_SV("/app/library.h"));
        *out_source =
            *out_found ? view(state.header) : iree_string_view_empty();
        return iree_ok_status();
      },
      &provider_state};

  loom_cxx_preamble_t* raw_preamble = nullptr;
  IREE_ASSERT_OK(loom_cxx_preamble_create(
      view(kSource), IREE_SV("/app/producer.cpp"), &options,
      iree_allocator_system(), &raw_preamble));
  std::unique_ptr<loom_cxx_preamble_t, decltype(&loom_cxx_preamble_release)>
      publisher(raw_preamble, loom_cxx_preamble_release);
  ASSERT_NE(publisher, nullptr);

  constexpr unsigned kConsumerCount = 8;
  std::latch consumers_ready(kConsumerCount);
  std::latch publisher_released(1);
  std::vector<std::future<bool>> consumers;
  consumers.reserve(kConsumerCount);
  for (unsigned i = 0; i < kConsumerCount; ++i) {
    loom_cxx_preamble_retain(raw_preamble);
    auto consumer_options = options;
    consumer_options.preamble = raw_preamble;
    consumers.push_back(std::async(std::launch::async, [raw_preamble,
                                                        consumer_options,
                                                        &consumers_ready,
                                                        &publisher_released,
                                                        i] {
      std::unique_ptr<loom_cxx_preamble_t, decltype(&loom_cxx_preamble_release)>
          retained(raw_preamble, loom_cxx_preamble_release);
      consumers_ready.count_down();
      publisher_released.wait();
      const auto filename = "/app/consumer_" + std::to_string(i) + ".cpp";
      Source source(view(kSource), view(filename), consumer_options);
      const auto expected_file = '"' + filename + '"';
      for (const auto& token : source.unit().tokens()) {
        if (token.kind() == cxx::TokenKind::T_STRING_LITERAL &&
            token.spell() == expected_file) {
          return source.unit().ast() != nullptr;
        }
      }
      return false;
    }));
  }

  consumers_ready.wait();
  provider_state.closed.store(true, std::memory_order_release);
  publisher.reset();
  publisher_released.count_down();
  for (auto& consumer : consumers) {
    EXPECT_TRUE(consumer.get());
  }
  EXPECT_EQ(provider_state.call_count.load(std::memory_order_relaxed), 1u);
}

}  // namespace
}  // namespace loom::cxx_import
