// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

#include "iree/hal/cts/util/registry.h"

static unsigned registrations = 0;

void iree::hal::cts::CtsRegistry::RegisterExecutableTarget(
    const char* backend, ExecutableTarget target) {
  using namespace std::literals;
  unsigned target_index;
  if (target.target_key == "first-exact"sv && target.name == "first_exact"sv) {
    target_index = 0;
  } else if (target.target_key == "first-generic"sv &&
             target.name == "first_generic"sv) {
    target_index = 1;
  } else if (target.target_key == "second"sv && target.name == "second"sv) {
    target_index = 2;
  } else {
    std::fprintf(stderr, "Unexpected target: %s\n", target.target_key);
    std::exit(EXIT_FAILURE);
  }
  unsigned backend_index;
  if (backend == "primary"sv) {
    backend_index = 0;
  } else if (backend == "alias"sv) {
    backend_index = 1;
  } else if (backend == "secondary"sv) {
    backend_index = 2;
  } else {
    std::exit(EXIT_FAILURE);
  }
  if (target.family != "amdgpu"sv || target.data_fn("missing.bin").data) {
    std::exit(EXIT_FAILURE);
  }
  for (const char* program : {"one.kernel", "two"}) {
    const auto data = target.data_fn(std::string(program) + ".bin");
    if (backend_index == 2 && program == "two"sv) {
      if (data.data) {
        std::exit(EXIT_FAILURE);
      }
      continue;
    }
    if (!data.data) {
      std::exit(EXIT_FAILURE);
    }
    const std::string_view text(reinterpret_cast<const char*>(data.data),
                                data.data_length);
    if (text.find(std::string(program) + " content") == text.npos ||
        text.find(std::string("--target=amdgpu:") + target.target_key) ==
            text.npos) {
      std::fprintf(stderr, "Incorrect program: %s/%s\n", backend, program);
      std::exit(EXIT_FAILURE);
    }
  }
  const unsigned bit = 1u << (target_index * 3 + backend_index);
  if (registrations & bit) {
    std::exit(EXIT_FAILURE);
  }
  registrations |= bit;
}

int main(int argc, char** argv) {
  const unsigned target_count = argc == 2 ? std::atoi(argv[1]) : 0;
  return registrations == (1u << (3 * target_count)) - 1u ? EXIT_SUCCESS
                                                          : EXIT_FAILURE;
}
