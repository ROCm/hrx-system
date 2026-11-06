// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/runtime/testing/stalled_stderr.h"
#include "loom/serve/service/finite/embedding/source_service_test_fixture.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/package.h"

namespace loom_serve::testing::embedding {

using loom_runtime::testing::StalledStderr;

TEST_F(SourceEmbeddingServiceTest, ReconfiguresDirectDispatchAndRollsBack) {
  ASSERT_NO_FATAL_FAILURE(UseDirectPackage());
  StartRouter();
  const auto inventory =
      Read(Send({}, "/v1/models/variants", "GET", "X-Loom-Model: fixture\r\n"));
  ASSERT_EQ(inventory.find("HTTP/1.1 200"), 0u) << inventory;
  const uint64_t original = Number(inventory, "selected");
  const std::filesystem::path path = SourceServiceDirectStageSource()[0]
                                         ? SourceServiceDirectStageSource()
                                         : directory_ + "/variants/direct.loom";
  const auto staged =
      UploadVariant(original, path.parent_path().string(),
                    path.filename().c_str(), "scalar.constant 2 : i32");
  ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
  const uint64_t candidate = Number(staged, "id", "variant");
  ASSERT_EQ(SelectVariant(candidate, original).find("HTTP/1.1 200"), 0u);
  const std::string request = R"({"model":"fixture","input":"a"})";
  const auto changed = Read(Send(request));
  EXPECT_NE(changed.find(R"("embedding":[2,1,2,1])"), std::string::npos)
      << changed;
  ASSERT_EQ(SelectVariant(original, candidate).find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(Send(request)).find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
}

TEST_F(SourceEmbeddingServiceTest, DirectDispatchReturnsCompactRaggedVectors) {
  ASSERT_NO_FATAL_FAILURE(UseDirectPackage());
  const int a = Send(R"({"model":"fixture","input":["a","bbb"]})");
  ASSERT_NO_FATAL_FAILURE(Accept(a));
  const int b = Send(R"({"model":"fixture","input":[[2,0]],"dimensions":2})");
  ASSERT_NO_FATAL_FAILURE(Accept(b));
  Drain();
  const std::string first = Read(a);
  SCOPED_TRACE(first);
  EXPECT_EQ(first.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(first.find(R"("index":0,"embedding":[1,1,1,3])"),
            std::string::npos);
  EXPECT_NE(first.find(R"("index":1,"embedding":[6,3,2,3])"),
            std::string::npos);
  const std::string second = Read(b);
  EXPECT_NE(second.find(R"("embedding":[4,2])"), std::string::npos) << second;
  const int next = Send(R"({"model":"fixture","input":[1]})");
  ASSERT_NO_FATAL_FAILURE(Accept(next));
  Drain();
  const std::string last = Read(next);
  EXPECT_NE(last.find(R"("embedding":[2,1,2,1])"), std::string::npos) << last;
}

TEST_F(SourceEmbeddingServiceTest, DirectDispatchRetiresCancelledCohort) {
  ASSERT_NO_FATAL_FAILURE(UseDirectPackage());
  CheckAbandonedCohort(0);
}

}  // namespace loom_serve::testing::embedding
