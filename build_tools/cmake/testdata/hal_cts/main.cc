// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "iree/hal/cts/util/registry.h"

static unsigned registrations = 0;

void iree::hal::cts::CtsRegistry::RegisterExecutableTarget(
    const char* backend, ExecutableTarget target) {
  using namespace std::literals;
  if (target.name != "target"sv || target.family != "family"sv ||
      target.target_key != "family:target"sv) {
    std::fprintf(stderr, "Incorrect target: %s %s %s\n", target.name,
                 target.family, target.target_key);
    std::exit(EXIT_FAILURE);
  }
  const auto data = target.data_fn("kernel.bin");
  if (!data.data ||
      std::string_view(reinterpret_cast<const char*>(data.data),
                       data.data_length) != "kernel bytes" ||
      target.data_fn("missing.bin").data) {
    std::fprintf(stderr, "Incorrect embedded data: %zu bytes [%.*s]\n",
                 data.data_length, static_cast<int>(data.data_length),
                 data.data ? reinterpret_cast<const char*>(data.data) : "");
    std::exit(EXIT_FAILURE);
  }
  if (backend == "primary"sv) {
    registrations |= 1u;
  } else if (backend == "alias"sv) {
    registrations |= 2u;
  } else {
    std::fprintf(stderr, "Incorrect backend: %s\n", backend);
    std::exit(EXIT_FAILURE);
  }
}

int main() { return registrations == 3u ? EXIT_SUCCESS : EXIT_FAILURE; }
