// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS Registry for link-time HAL test composition.
//
// This registry enables HAL test suites to be compiled once and linked against
// multiple backends, eliminating per-backend instantiation boilerplate.
//
// Architecture:
//   1. Test suites register themselves at static init time via
//      CTS_REGISTER_TEST_SUITE().
//   2. Backends register themselves at static init time via
//      CtsRegistry::RegisterBackend().
//   3. main() calls CtsRegistry::InstantiateAll() before RUN_ALL_TESTS().
//   4. InstantiateAll() creates gtest parameterized test instances for each
//      suite × backend pair that passes tag filtering.
//
// Tag filtering:
//   - required_tags: Backend must have ALL of these tags.
//   - excluded_tags: Backend must have NONE of these tags.
//
// Example test suite registration:
//   CTS_REGISTER_TEST_SUITE(AllocatorTest);  // All backends, once each.
//   CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(CopyBufferTest);  //
//   Direct+indirect. CTS_REGISTER_EXECUTABLE_TEST_SUITE(ExecutableTest);  //
//   Per (device, format).
//   CTS_REGISTER_EXECUTABLE_COMMAND_BUFFER_TEST_SUITE(DispatchTest);  // All
//   axes.
//
// Example backend registration:
//   CtsRegistry::RegisterBackend({
//       "local_task",
//       {.name = "local_task", .factory = CreateLocalTaskDevice},
//       {"events", "indirect"},
//       {{"native", "embedded-elf", GetNativeExecutableData}},
//   });

#ifndef IREE_HAL_CTS_UTIL_REGISTRY_H_
#define IREE_HAL_CTS_UTIL_REGISTRY_H_

#include <functional>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"

namespace iree::hal::cts {

//===----------------------------------------------------------------------===//
// Backend factory
//===----------------------------------------------------------------------===//

// Factory function that creates a HAL driver and device pair.
//
// The shared CTS fixture owns |create_params| storage and keeps all borrowed
// resources in it alive for the lifetime of the returned device. Factories must
// pass it through to driver device creation unchanged unless the backend has a
// documented driver-specific extension chain to add.
//
// Returns:
//   iree_ok_status(): Success. Both out params populated with retained refs.
//   IREE_STATUS_UNAVAILABLE: Backend not present on this system. Tests skip.
//   Any other error: Creation failed. Tests fail.
//
// The caller takes ownership of both returned objects and must release them.
using DeviceFactory = std::function<iree_status_t(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device)>;

// Predicate reporting whether a backend configuration is compatible with the
// host build before attempting device creation.
//
// Returning true does not imply hardware support or driver availability. It
// only means the host process configuration does not rule the backend out
// before the production device creation path gets to query hardware.
using BackendHostCompatibilityFn = std::function<bool(std::string* out_reason)>;

// Function called during CTS process teardown before cached backend resources
// are released.
using CleanupFn = std::function<void()>;

// Function that returns pre-compiled executable data for a given file name.
// Used by dispatch tests to load backend-specific device code.
// Returns an empty span if the file is not found.
using ExecutableDataFn =
    iree_const_byte_span_t (*)(iree_string_view_t file_name);

// Selects the device target compatible with one linked executable artifact.
//
// The target family owns compatibility between |artifact_target_key| and the
// immutable targets advertised by |device_spec|. No-match and ambiguous
// outcomes are returned in |out_result| without creating a status.
using ExecutableTargetSelectorFn = iree_status_t (*)(
    const iree_hal_device_spec_t* device_spec,
    iree_string_view_t artifact_target_family,
    iree_string_view_t artifact_target_key,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    iree_hal_executable_target_selection_result_t* out_result);

// Selects a device target by exact family-owned identity.
inline iree_status_t SelectExactExecutableTarget(
    const iree_hal_device_spec_t* device_spec,
    iree_string_view_t artifact_target_family,
    iree_string_view_t artifact_target_key,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    iree_hal_executable_target_selection_result_t* out_result) {
  iree_hal_executable_target_selection_t selection = {};
  selection.family = artifact_target_family;
  selection.target_key = artifact_target_key;
  selection.physical_device_affinity = physical_device_affinity;
  *out_result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Executable target
//===----------------------------------------------------------------------===//

// A pre-compiled executable target available for a backend.
// Backends declare the targets they support; executable test suites expand
// parameterizations across all available targets. This keeps non-executable
// tests (allocator, semaphore, etc.) running once per device while executable
// tests run once per (device, target) pair.
struct ExecutableTarget {
  // Suffix appended to the backend name, such as "gfx942" or "spirv".
  std::string name;
  // HAL target family, such as "amdgpu" or "spirv".
  const char* family;
  // Canonical family-owned target key.
  const char* target_key;
  // Looks up linked executable data by CTS fixture name.
  ExecutableDataFn data_fn;
};

//===----------------------------------------------------------------------===//
// Recording mode
//===----------------------------------------------------------------------===//

// Command buffer recording mode for parameterized command buffer tests.
// CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE creates instantiations for both modes.
enum class RecordingMode {
  kDirect,    // Inline buffer references, binding_capacity = 0.
  kIndirect,  // Binding table slots, binding_capacity > 0.
};

//===----------------------------------------------------------------------===//
// Test exclusions and expected failures
//===----------------------------------------------------------------------===//

// Permanent categorical exclusion. The backend fundamentally cannot support
// this test (e.g., no file I/O, no indirect command buffers). These entries
// should never be removed — they document inherent backend limitations.
//
// Pattern is matched against "TestClass.TestMethod" using
// iree_string_view_match_pattern() glob syntax (* and ?).
struct TestUnsupported {
  std::string pattern;  // Glob: "TestClass.Method" or "TestClass.*"
  std::string reason;   // Why this will never work.
};

// Temporary expected failure. The test should eventually pass but currently
// doesn't due to incomplete implementation. These shrink over time as the
// backend matures.
//
// Pattern is matched against "TestClass.TestMethod" using
// iree_string_view_match_pattern() glob syntax (* and ?).
//
// In verify mode (IREE_CTS_VERIFY_XFAILS=1), xfail tests run instead of
// being skipped, and unexpected passes (XPASS) are flagged as test failures
// so stale entries can be detected and removed.
struct TestExpectedFailure {
  std::string pattern;  // Glob: "TestClass.Method" or "TestClass.*"
  std::string reason;   // What's broken and what the fix requires.
};

//===----------------------------------------------------------------------===//
// Backend info (test parameterization type)
//===----------------------------------------------------------------------===//

// Identifies a HAL backend for test parameterization.
// Tests receive this via GetParam() and use it to create devices and load
// executables.
struct BackendInfo {
  // Human-readable backend name used as the gtest parameter suffix.
  std::string name;
  // Creates driver + device.
  DeviceFactory factory;
  // HAL target family for the linked executable data.
  const char* executable_target_family = nullptr;
  // Canonical family-owned target key for the linked executable data.
  const char* executable_target_key = nullptr;
  // Compiled executable lookup.
  ExecutableDataFn executable_data = nullptr;
  RecordingMode recording_mode = RecordingMode::kDirect;
  std::vector<TestUnsupported> unsupported_tests;
  std::vector<TestExpectedFailure> expected_failures;
  // Stable key for caches that own driver/device resources.
  //
  // Executable tests use |name| to distinguish one executable target from
  // another while still running on the same physical backend. Device caches
  // should use this key so target parameterization does not multiply expensive
  // device state.
  std::string device_cache_key;
  // Optional pre-device-creation host compatibility query.
  BackendHostCompatibilityFn host_compatibility_fn;
  // Family-owned artifact compatibility selector.
  ExecutableTargetSelectorFn executable_target_selector =
      SelectExactExecutableTarget;
};

// Returns the stable cache key for driver/device resources owned by |info|.
inline const std::string& GetBackendDeviceCacheKey(const BackendInfo& info) {
  return info.device_cache_key.empty() ? info.name : info.device_cache_key;
}

// Returns true if |info| is compatible with this host build/configuration.
inline bool IsBackendHostCompatible(const BackendInfo& info,
                                    std::string* out_reason) {
  if (!info.host_compatibility_fn) return true;
  return info.host_compatibility_fn(out_reason);
}

// Returns human-readable test suffix from BackendInfo.
// Used as the generator for INSTANTIATE_TEST_SUITE_P.
struct BackendName {
  std::string operator()(
      const ::testing::TestParamInfo<BackendInfo>& info) const {
    return info.param.name;
  }
};

// Printer for BackendInfo in gtest assertions and failure messages.
// Without this, gtest prints the raw bytes of the struct.
inline void PrintTo(const BackendInfo& info, std::ostream* os) {
  *os << info.name;
}

//===----------------------------------------------------------------------===//
// Backend configuration with tags
//===----------------------------------------------------------------------===//

// Identifies a backend configuration for test instantiation.
// Extends BackendInfo with tags for filtering which test suites apply.
struct BackendConfig {
  const char* name;               // "local_task", etc.
  BackendInfo info;               // Factory + capabilities.
  std::vector<std::string> tags;  // {"events", "indirect", ...}
  std::vector<ExecutableTarget> executable_targets;  // Available targets.
};

//===----------------------------------------------------------------------===//
// Test suite registration
//===----------------------------------------------------------------------===//

// Function called for each matching backend to accumulate backends for a suite.
using TestSuiteAccumulator = std::function<void(const BackendConfig& config)>;

// Function called once after all backends are accumulated to create gtest
// instances.
using TestSuiteFinalizer = std::function<void(const char* file, int line)>;

// Metadata about a registered test suite.
struct TestSuiteInfo {
  const char* name;
  TestSuiteAccumulator accumulator;
  TestSuiteFinalizer finalizer;
  const char* file;
  int line;

  // Required tags: suite only instantiated for backends with ALL of these.
  std::vector<std::string> required_tags;

  // Excluded tags: suite skipped for backends with ANY of these.
  std::vector<std::string> excluded_tags;
};

//===----------------------------------------------------------------------===//
// CTS Registry
//===----------------------------------------------------------------------===//

class CtsRegistry {
 public:
  //===--------------------------------------------------------------------===//
  // Registration (called at static init time)
  //===--------------------------------------------------------------------===//

  // Register a test suite. Called by CTS_REGISTER_TEST_SUITE macro.
  static void RegisterSuite(TestSuiteInfo info);

  // Register a backend configuration. Called by backend factory files.
  static void RegisterBackend(BackendConfig config);

  // Register a process-level cleanup hook. Called by helpers that own cached
  // resources outside of the common backend cache.
  static void RegisterCleanup(CleanupFn cleanup);

  // Registers an executable target for an already-registered (or
  // not-yet-registered) backend. Targets are stored in a pending list and
  // merged into their backends at InstantiateAll() time, so static init
  // ordering between RegisterBackend() and RegisterExecutableTarget() does
  // not matter.
  static void RegisterExecutableTarget(const char* backend_name,
                                       ExecutableTarget target);

  // Returns executable targets currently registered for |backend_name|.
  //
  // This is intentionally independent from test-suite instantiation so tools
  // and benchmarks can load linked CTS-style executable data without creating
  // gtest parameterizations or requiring a BackendConfig registration.
  static std::vector<ExecutableTarget> ListExecutableTargets(
      const char* backend_name);

  //===--------------------------------------------------------------------===//
  // Instantiation (called from main, before RUN_ALL_TESTS)
  //===--------------------------------------------------------------------===//

  // Instantiate all suites × all backends, respecting tag filters.
  static void InstantiateAll();

  // Instantiate specific suite for specific backend (for debugging).
  static void Instantiate(const char* suite_name, const char* backend_name);

  // Runs registered cleanup hooks exactly once.
  static void RunCleanups();

  //===--------------------------------------------------------------------===//
  // Introspection (for tooling, test listing)
  //===--------------------------------------------------------------------===//

  static std::vector<std::string> ListSuites();
  static std::vector<std::string> ListBackends();

 private:
  // Check if backend has all required tags and none of the excluded tags.
  static bool TagsMatch(const BackendConfig& backend,
                        const std::vector<std::string>& required,
                        const std::vector<std::string>& excluded);
};

//===----------------------------------------------------------------------===//
// Test Registration Helpers
//===----------------------------------------------------------------------===//

namespace internal {

// Name generator function for test instantiation.
inline std::string GetBackendName(
    const ::testing::TestParamInfo<BackendInfo>& info) {
  return BackendName()(info);
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// Test Registration Macros
//===----------------------------------------------------------------------===//

// Basic registration - suite runs against all backends.
#define CTS_REGISTER_TEST_SUITE(TestClass) \
  CTS_REGISTER_TEST_SUITE_WITH_TAGS(TestClass, {}, {})

// Registration with tag requirements.
// required_tags: backend must have ALL of these tags.
// excluded_tags: backend must have NONE of these tags.
//
// Each test suite has a static vector of BackendInfos that accumulates
// matching backends during InstantiateAll(). After all backends are added,
// the suite is instantiated once with ValuesIn() covering all backends.
#define CTS_REGISTER_TEST_SUITE_WITH_TAGS(TestClass, required_tags,           \
                                          excluded_tags)                      \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestClass);                   \
  namespace {                                                                 \
  struct TestClass##_Backends {                                               \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  }                                                                           \
  static bool TestClass##_registered_ =                                       \
      (::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass,                                                       \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              TestClass##_Backends::Get().push_back(cfg.info);                \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_Backends::Get().empty()) return;                \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS", &TestClass##_Backends::Generator,                \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__, __LINE__, required_tags, excluded_tags}),               \
       true)

// Dual-mode registration for command buffer tests.
//
// Creates two gtest instantiation sets from a single test class:
//   - CTS/TestClass.* parameterized with direct-mode BackendInfos
//   - CTS_Indirect/TestClass.* parameterized with indirect-mode BackendInfos
//     (only for backends with the "indirect" tag)
//
// Tests call recording_mode() to get the current mode and create command
// buffers accordingly (direct uses inline buffer references, indirect uses
// binding table slots).
#define CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(TestClass)                     \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestClass);                   \
  namespace {                                                                 \
  struct TestClass##_DirectBackends {                                         \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  struct TestClass##_IndirectBackends {                                       \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  }                                                                           \
  static bool TestClass##_registered_ =                                       \
      (::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass,                                                       \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              auto info = cfg.info;                                           \
              info.recording_mode = ::iree::hal::cts::RecordingMode::kDirect; \
              TestClass##_DirectBackends::Get().push_back(std::move(info));   \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_DirectBackends::Get().empty()) return;          \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS", &TestClass##_DirectBackends::Generator,          \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__,                                                         \
            __LINE__,                                                         \
            {},                                                               \
            {}}),                                                             \
       ::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass "_Indirect",                                           \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              auto info = cfg.info;                                           \
              info.recording_mode =                                           \
                  ::iree::hal::cts::RecordingMode::kIndirect;                 \
              TestClass##_IndirectBackends::Get().push_back(std::move(info)); \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_IndirectBackends::Get().empty()) return;        \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS_Indirect",                                         \
                      &TestClass##_IndirectBackends::Generator,               \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__,                                                         \
            __LINE__,                                                         \
            {"indirect"},                                                     \
            {}}),                                                             \
       true)

// Executable test registration - suite runs once per (device, target) pair.
//
// Backends with no executable_targets are silently skipped (the accumulator
// pushes nothing, and GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST handles
// the empty suite). The "executables" tag is unnecessary — target availability
// is structural, not string-based.
//
// For each matching backend, the accumulator iterates executable_targets and
// pushes one BackendInfo per target with name = "backend_target" and the
// target identity and executable data populated from ExecutableTarget.
#define CTS_REGISTER_EXECUTABLE_TEST_SUITE(TestClass)                         \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestClass);                   \
  namespace {                                                                 \
  struct TestClass##_Backends {                                               \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  }                                                                           \
  static bool TestClass##_registered_ =                                       \
      (::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass,                                                       \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              for (const auto& target : cfg.executable_targets) {             \
                auto info = cfg.info;                                         \
                info.device_cache_key = cfg.name;                             \
                info.name = std::string(cfg.name) + "_" + target.name;        \
                info.executable_target_family = target.family;                \
                info.executable_target_key = target.target_key;               \
                info.executable_data = target.data_fn;                        \
                TestClass##_Backends::Get().push_back(std::move(info));       \
              }                                                               \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_Backends::Get().empty()) return;                \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS", &TestClass##_Backends::Generator,                \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__,                                                         \
            __LINE__,                                                         \
            {},                                                               \
            {}}),                                                             \
       true)

// Executable command buffer test registration - suite runs once per
// (device, target, recording_mode) triple.
//
// Combines executable target expansion with direct/indirect recording mode
// expansion. Creates two gtest instantiation sets:
//   - CTS/TestClass.* with direct-mode BackendInfos for each target
//   - CTS_Indirect/TestClass.* with indirect-mode BackendInfos for each target
//     (only for backends with the "indirect" tag)
#define CTS_REGISTER_EXECUTABLE_COMMAND_BUFFER_TEST_SUITE(TestClass)          \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestClass);                   \
  namespace {                                                                 \
  struct TestClass##_DirectBackends {                                         \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  struct TestClass##_IndirectBackends {                                       \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  }                                                                           \
  static bool TestClass##_registered_ =                                       \
      (::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass,                                                       \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              for (const auto& target : cfg.executable_targets) {             \
                auto info = cfg.info;                                         \
                info.device_cache_key = cfg.name;                             \
                info.name = std::string(cfg.name) + "_" + target.name;        \
                info.executable_target_family = target.family;                \
                info.executable_target_key = target.target_key;               \
                info.executable_data = target.data_fn;                        \
                info.recording_mode =                                         \
                    ::iree::hal::cts::RecordingMode::kDirect;                 \
                TestClass##_DirectBackends::Get().push_back(std::move(info)); \
              }                                                               \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_DirectBackends::Get().empty()) return;          \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS", &TestClass##_DirectBackends::Generator,          \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__,                                                         \
            __LINE__,                                                         \
            {},                                                               \
            {}}),                                                             \
       ::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass "_Indirect",                                           \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              for (const auto& target : cfg.executable_targets) {             \
                auto info = cfg.info;                                         \
                info.device_cache_key = cfg.name;                             \
                info.name = std::string(cfg.name) + "_" + target.name;        \
                info.executable_target_family = target.family;                \
                info.executable_target_key = target.target_key;               \
                info.executable_data = target.data_fn;                        \
                info.recording_mode =                                         \
                    ::iree::hal::cts::RecordingMode::kIndirect;               \
                TestClass##_IndirectBackends::Get().push_back(                \
                    std::move(info));                                         \
              }                                                               \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_IndirectBackends::Get().empty()) return;        \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS_Indirect",                                         \
                      &TestClass##_IndirectBackends::Generator,               \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__,                                                         \
            __LINE__,                                                         \
            {"indirect"},                                                     \
            {}}),                                                             \
       true)

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_UTIL_REGISTRY_H_
