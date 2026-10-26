// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_MODEL_WAIT_H_
#define LOOM_SERVE_TESTING_MODEL_WAIT_H_

#include <string>

#include "iree/testing/gtest.h"
#include "loom/serve/testing/http_harness.h"

namespace loom_serve::testing {

// Waits for a named model to expose |marker| in its serialized status. If the
// marker is not present, a terminal registration failure makes the requested
// state unreachable and is reported immediately. |query| returns an empty
// string while the model is not visible.
template <typename Query>
std::string WaitModelState(HttpHarness& http, const std::string& name,
                           const std::string& marker, Query&& query) {
  if (marker.empty()) {
    [&] { FAIL() << "Model-state waits require a nonempty marker"; }();
    return {};
  }
  while (!http.failed() && !::testing::Test::HasFatalFailure()) {
    const std::string state = query();
    if (state.find(marker) != std::string::npos) {
      return state;
    }
    if (state.find(R"("registration":"failed")") != std::string::npos) {
      [&] {
        FAIL() << "Model " << name << " failed before reaching " << marker
               << ": " << state;
      }();
      return {};
    }
  }
  return {};
}

// Waits until a named model is absent from serialized status. Failed and
// retiring registrations remain valid observations until asynchronous
// ownership retirement releases their name.
template <typename Query>
void WaitModelAbsent(HttpHarness& http, Query&& query) {
  while (!http.failed() && !::testing::Test::HasFatalFailure()) {
    if (query().empty()) {
      return;
    }
  }
}

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_TESTING_MODEL_WAIT_H_
