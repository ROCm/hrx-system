// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

extern "C" int fixture_registration;

namespace {
struct Registration {
  Registration() { fixture_registration = 42; }
};
Registration registration;
}  // namespace
