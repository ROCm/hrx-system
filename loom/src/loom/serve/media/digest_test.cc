// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/digest.h"

#include <string>

#include "iree/testing/gtest.h"

namespace {

std::string Hex(const loom_serve_media_digest_t& digest) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string value(sizeof(digest.bytes) * 2, '0');
  for (iree_host_size_t i = 0; i < sizeof(digest.bytes); ++i) {
    value[i * 2] = kDigits[digest.bytes[i] >> 4];
    value[i * 2 + 1] = kDigits[digest.bytes[i] & 0x0f];
  }
  return value;
}

loom_serve_media_digest_t Digest(iree_string_view_t value) {
  loom_serve_media_digest_t digest;
  loom_serve_media_digest_compute(
      iree_make_const_byte_span(value.data, value.size), &digest);
  return digest;
}

TEST(DigestTest, StandardVectors) {
  EXPECT_EQ(Hex(Digest(IREE_SV(""))),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(Hex(Digest(IREE_SV("abc"))),
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(Hex(Digest(IREE_SV("abcdbcdecdefdefgefghfghighijhijk"
                               "ijkljklmklmnlmnomnopnopq"))),
            "248d6a61d20638b8e5c026930c3e6039"
            "a33ce45964ff2167f6ecedd419db06c1");
}

TEST(DigestTest, CompleteIdentityComparison) {
  const auto first = Digest(IREE_SV("first"));
  auto second = first;
  EXPECT_TRUE(loom_serve_media_digest_equal(&first, &second));
  second.bytes[31] ^= 1;
  EXPECT_FALSE(loom_serve_media_digest_equal(&first, &second));
}

}  // namespace
