// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_MODELS_FIXTURES_TEXT_ORACLE_H_
#define EXPERIMENTAL_LOOM_SERVE_MODELS_FIXTURES_TEXT_ORACLE_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace loom_serve::testing::text {

// Independent dense arithmetic, with no knowledge of device packets, pooled
// addresses, slot ownership or scheduling chunks. EOS is a consumed delimiter
// in replayed history but an unconsumed prediction at generation completion.
class Oracle {
 public:
  explicit Oracle(uint32_t bias = 2) : bias_(bias) {}

  void Consume(int32_t token) {
    hidden_ = (5 * hidden_ + token + bias_) % 97;
    history_ = (7 * history_ + hidden_ + position_) % 101;
    sum_ += history_;
    ++position_;
  }

  void Consume(const std::string& text) {
    for (char character : text) {
      Consume(Id(character));
    }
  }

  int32_t Predict() const {
    return position_ % 11 == 0 ? 3 : (hidden_ + sum_) % 3;
  }

  static int32_t Id(char character) {
    return character == 'a'   ? 0
           : character == 'b' ? 1
           : character == 'x' ? 2
                              : 3;
  }

  static std::string Complete(const std::string& prompt, size_t limit,
                              uint32_t bias = 2) {
    Oracle oracle(bias);
    oracle.Consume(prompt);
    std::string response;
    for (size_t i = 0; i < limit; ++i) {
      const int32_t token = oracle.Predict();
      if (token == 3) {
        break;
      }
      response.push_back("abx"[token]);
      oracle.Consume(token);
    }
    return response;
  }

 private:
  // Actual immutable checkpoint value.
  const uint32_t bias_;
  // Recurrent value, independent of row identity.
  uint32_t hidden_ = 0;
  // Last cached history value.
  uint32_t history_ = 0;
  // Sum of all cached history values used by selection.
  uint32_t sum_ = 0;
  // Number of consumed inputs, excluding a pending prediction.
  uint32_t position_ = 0;
};

}  // namespace loom_serve::testing::text

#endif  // EXPERIMENTAL_LOOM_SERVE_MODELS_FIXTURES_TEXT_ORACLE_H_
