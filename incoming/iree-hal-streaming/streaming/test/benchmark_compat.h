#ifndef IREE_STREAMING_TEST_BENCHMARK_COMPAT_H_
#define IREE_STREAMING_TEST_BENCHMARK_COMPAT_H_

#include <string>

#include "iree/testing/benchmark.h"
#include "benchmark/benchmark.h"

// Compatibility macro for IREE_BENCHMARK_REGISTER_ARGS which was removed.
// This directly uses the Google Benchmark C++ API to register benchmarks
// with arguments.
#define IREE_BENCHMARK_REGISTER_ARGS(name, arg) \
  static ::benchmark::Benchmark* IREE_BENCHMARK_IMPL_NAME_(name) \
      IREE_ATTRIBUTE_UNUSED = \
          ::benchmark::RegisterBenchmark(#name, \
              [](::benchmark::State& state) { \
                iree_benchmark_state_t iree_state; \
                iree_state.impl = &state; \
                iree_state.host_allocator = iree_allocator_system(); \
                iree_status_t status = name(nullptr, &iree_state); \
                if (!iree_status_is_ok(status)) { \
                  iree_allocator_t alloc = iree_allocator_system(); \
                  char* buf = NULL; \
                  iree_host_size_t len = 0; \
                  if (iree_status_to_string(status, &alloc, &buf, &len)) { \
                    state.SkipWithError(std::string(buf, len)); \
                    iree_allocator_free(alloc, buf); \
                  } else { \
                    state.SkipWithError("unknown error"); \
                  } \
                  iree_status_ignore(status); \
                } \
              })->Arg(arg)
#endif // IREE_STREAMING_TEST_BENCHMARK_COMPAT_H_
