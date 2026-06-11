// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "clang-tidy/ClangTidyModule.h"
#include "iree/LifecycleChecks.h"
#include "iree/RefCountChecks.h"
#include "iree/SmokeCheck.h"
#include "iree/StatusChecks.h"
#include "iree/StyleChecks.h"
#include "iree/TraceChecks.h"

namespace clang::tidy::iree {
namespace {

class IreeTidyModule final : public ClangTidyModule {
 public:
  void addCheckFactories(ClangTidyCheckFactories& CheckFactories) override {
    CheckFactories.registerCheck<SmokeCheck>("iree-smoke");
    CheckFactories.registerCheck<BorrowedStatusParameterCheck>(
        "iree-status-borrowed-parameter");
    CheckFactories.registerCheck<DiscardedStatusCheck>("iree-status-discarded");
    CheckFactories.registerCheck<StatusLifetimeCheck>("iree-status-lifetime");
    CheckFactories.registerCheck<StatusTransferOrderCheck>(
        "iree-status-transfer-order");
    CheckFactories.registerCheck<DirectGotoCheck>("iree-direct-goto");
    CheckFactories.registerCheck<GuardedReleaseCheck>("iree-guarded-release");
    CheckFactories.registerCheck<LifecycleNamingCheck>("iree-lifecycle-naming");
    CheckFactories.registerCheck<RefCountLifecycleCheck>(
        "iree-refcount-lifecycle");
    CheckFactories.registerCheck<TestStatusMacroCheck>(
        "iree-test-status-macro-scope");
    CheckFactories.registerCheck<TestStatusPredicateCheck>(
        "iree-test-status-predicate");
    CheckFactories.registerCheck<TraceZoneCheck>("iree-trace-zone-balance");
  }
};

}  // namespace
}  // namespace clang::tidy::iree

static clang::tidy::ClangTidyModuleRegistry::Add<
    clang::tidy::iree::IreeTidyModule>
    X("iree-module", "Adds IREE-specific checks.");

volatile int IreeClangTidyModuleAnchorSource = 0;
