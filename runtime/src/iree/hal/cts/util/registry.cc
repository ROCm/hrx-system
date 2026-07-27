// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/registry.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace iree::hal::cts {
namespace {

// Global storage for registered suites and backends.
// Protected by a mutex for thread-safe static initialization.
// Executable target registered before its backend (or separately from it).
struct PendingTarget {
  // Backend name the executable target should attach to.
  std::string backend_name;
  // Executable target waiting for backend registration.
  ExecutableTarget target;
};

struct RegistryData {
  // Protects all mutable registry state.
  std::mutex mutex;
  // True after test suites have been instantiated.
  bool instantiated = false;
  // True after process-level cleanup hooks have run.
  bool cleanups_run = false;
  // Registered CTS test suites.
  std::vector<TestSuiteInfo> test_suites;
  // Registered CTS backend configurations.
  std::vector<BackendConfig> backends;
  // Executable targets waiting for matching backend registration.
  std::vector<PendingTarget> pending_targets;
  // Process-level cleanup hooks registered by CTS helper libraries.
  std::vector<CleanupFn> cleanups;
};

RegistryData& GetRegistryData() {
  static RegistryData data;
  return data;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void CtsRegistry::RegisterSuite(TestSuiteInfo info) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);
  data.test_suites.push_back(std::move(info));
}

void CtsRegistry::RegisterBackend(BackendConfig config) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);
  data.backends.push_back(std::move(config));
}

void CtsRegistry::RegisterCleanup(CleanupFn cleanup) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);
  data.cleanups.push_back(std::move(cleanup));
}

void CtsRegistry::RegisterExecutableTarget(const char* backend_name,
                                           ExecutableTarget target) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);
  data.pending_targets.push_back(
      {std::string(backend_name), std::move(target)});
}

std::vector<ExecutableTarget> CtsRegistry::ListExecutableTargets(
    const char* backend_name) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  std::vector<ExecutableTarget> targets;
  for (const auto& pending : data.pending_targets) {
    if (pending.backend_name == backend_name) {
      targets.push_back(pending.target);
    }
  }
  for (const auto& backend : data.backends) {
    if (std::string(backend.name) == backend_name) {
      targets.insert(targets.end(), backend.executable_targets.begin(),
                     backend.executable_targets.end());
    }
  }
  return targets;
}

//===----------------------------------------------------------------------===//
// Instantiation
//===----------------------------------------------------------------------===//

bool CtsRegistry::TagsMatch(const BackendConfig& backend,
                            const std::vector<std::string>& required,
                            const std::vector<std::string>& excluded) {
  // Check required tags: backend must have ALL of them.
  for (const auto& tag : required) {
    if (std::find(backend.tags.begin(), backend.tags.end(), tag) ==
        backend.tags.end()) {
      return false;  // Missing required tag.
    }
  }

  // Check excluded tags: backend must have NONE of them.
  for (const auto& tag : excluded) {
    if (std::find(backend.tags.begin(), backend.tags.end(), tag) !=
        backend.tags.end()) {
      return false;  // Has excluded tag.
    }
  }

  return true;
}

// Merges pending executable targets into their matching backends.
// Must be called with data.mutex held.
static void MergePendingTargets(RegistryData& data) {
  for (auto& pending : data.pending_targets) {
    bool found = false;
    for (auto& backend : data.backends) {
      if (pending.backend_name == backend.name) {
        backend.executable_targets.push_back(std::move(pending.target));
        found = true;
        break;
      }
    }
    if (!found) {
      std::cerr << "CtsRegistry: executable target for '"
                << pending.backend_name
                << "' has no matching backend registration.\n";
      std::abort();
    }
  }
  data.pending_targets.clear();
}

void CtsRegistry::InstantiateAll() {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  if (data.instantiated) {
    std::cerr << "CtsRegistry::InstantiateAll called twice.\n";
    std::abort();
  }
  data.instantiated = true;

  // Merge pending executable targets into their backends. This handles the
  // case where RegisterExecutableTarget() was called before or after the
  // corresponding RegisterBackend() — static init ordering is unspecified.
  MergePendingTargets(data);

  if (data.backends.empty() || data.test_suites.empty()) {
    return;
  }

  // Accumulate matching backends for each test suite, then finalize.
  for (const auto& suite : data.test_suites) {
    for (const auto& backend : data.backends) {
      if (TagsMatch(backend, suite.required_tags, suite.excluded_tags)) {
        suite.accumulator(backend);
      }
    }
    // Finalize: create gtest instances for all accumulated backends.
    suite.finalizer(suite.file, suite.line);
  }
}

void CtsRegistry::Instantiate(const char* suite_name,
                              const char* backend_name) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  // Merge pending executable targets (same as InstantiateAll) so that targets
  // registered separately from backends are available.
  MergePendingTargets(data);

  // Find the specified suite.
  const TestSuiteInfo* suite = nullptr;
  for (const auto& s : data.test_suites) {
    if (std::string(s.name) == suite_name) {
      suite = &s;
      break;
    }
  }
  if (!suite) {
    std::cerr << "CtsRegistry::Instantiate: Suite '" << suite_name
              << "' not found.\n";
    std::abort();
  }

  // Find the specified backend.
  const BackendConfig* backend = nullptr;
  for (const auto& b : data.backends) {
    if (std::string(b.name) == backend_name) {
      backend = &b;
      break;
    }
  }
  if (!backend) {
    std::cerr << "CtsRegistry::Instantiate: Backend '" << backend_name
              << "' not found.\n";
    std::abort();
  }

  // Check tag compatibility.
  if (!TagsMatch(*backend, suite->required_tags, suite->excluded_tags)) {
    std::cerr << "CtsRegistry::Instantiate: Tags don't match for suite '"
              << suite_name << "' with backend '" << backend_name << "'.\n";
    std::abort();
  }

  suite->accumulator(*backend);
  suite->finalizer(suite->file, suite->line);
}

void CtsRegistry::RunCleanups() {
  std::vector<CleanupFn> cleanups;
  {
    auto& data = GetRegistryData();
    std::lock_guard<std::mutex> lock(data.mutex);
    if (data.cleanups_run) return;
    data.cleanups_run = true;
    cleanups = std::move(data.cleanups);
    data.cleanups.clear();
  }
  for (auto it = cleanups.rbegin(); it != cleanups.rend(); ++it) {
    (*it)();
  }
}

//===----------------------------------------------------------------------===//
// Introspection
//===----------------------------------------------------------------------===//

std::vector<std::string> CtsRegistry::ListSuites() {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  std::vector<std::string> names;
  names.reserve(data.test_suites.size());
  for (const auto& suite : data.test_suites) {
    names.push_back(suite.name);
  }
  return names;
}

std::vector<std::string> CtsRegistry::ListBackends() {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  std::vector<std::string> names;
  names.reserve(data.backends.size());
  for (const auto& backend : data.backends) {
    names.push_back(backend.name);
  }
  return names;
}

}  // namespace iree::hal::cts
