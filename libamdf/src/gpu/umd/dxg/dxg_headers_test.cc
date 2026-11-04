// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <type_traits>

#include "gtest/gtest.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"

namespace {

TEST(ThunkProxyHeadersTest, DeclaresPublishedPayloadApi) {
  using MakePowerPayload = thunk_proxy::PrivData (*)(bool);
  using MakeContextPayload = thunk_proxy::PrivData (*)(bool);
  using MakeAllocationPayload = thunk_proxy::AllocPrivData (*)(int);

  static_assert(std::is_same_v<decltype(&thunk_proxy::MakePowerOptPrivData),
                               MakePowerPayload>);
  static_assert(std::is_same_v<decltype(&thunk_proxy::MakeContextPrivData),
                               MakeContextPayload>);
  static_assert(std::is_same_v<decltype(&thunk_proxy::MakeAllocPrivData),
                               MakeAllocationPayload>);
  SUCCEED();
}

}  // namespace
