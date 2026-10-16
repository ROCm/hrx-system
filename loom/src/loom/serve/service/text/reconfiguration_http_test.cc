// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/json.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/testing/package.h"
#include "loomc/context.h"
#include "loomc/iree.h"
#include "loomc/module.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class ReconfigurationHttpTextTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_service_options_t ServiceOptions() const override {
    auto options = SourceTextServiceTest::ServiceOptions();
    options.variant_capacity = 4;
    return options;
  }

  std::string Inventory(const std::string& headers = "") {
    return Read(Send("", "X-Loom-Model: fixture-text\r\n" + headers,
                     "/v1/models/variants", "GET"));
  }

  static uint64_t Number(const std::string& response, const char* field,
                         const char* object = nullptr) {
    uint64_t result = 0;
    [&] {
      const auto start = response.find("\r\n\r\n");
      ASSERT_NE(start, std::string::npos) << response;
      auto value = iree_make_string_view(response.data() + start + 4,
                                         response.size() - start - 4);
      if (object) {
        IREE_ASSERT_OK(iree_json_lookup_object_value(
            value, iree_make_cstring_view(object), &value));
      }
      IREE_ASSERT_OK(iree_json_lookup_object_value(
          value, iree_make_cstring_view(field), &value));
      IREE_ASSERT_OK(iree_json_parse_uint64(value, &result));
    }();
    return result;
  }

  void Provider(const std::string& replacement, std::string* out_path) {
    std::ifstream input(directory_ + "/advance.loom");
    ASSERT_TRUE(input.good());
    std::string source((std::istreambuf_iterator<char>(input)), {});
    const std::string original = "scalar.remui %total, %c3_i32";
    const size_t position = source.find(original);
    ASSERT_NE(position, std::string::npos);
    source.replace(position, original.size(), replacement);
    *out_path = iree::testing::MakeTempFilePath("http_variant", ".loom");
    std::ofstream output(*out_path);
    output << source;
    output.close();
    ASSERT_TRUE(output.good());
  }

  std::string Stage(uint64_t base, const std::string& path, bool wait = true) {
    return Read(
        Send("{\"model\":\"fixture-text\",\"base\":" + std::to_string(base) +
                 ",\"sources\":[\"" + std::filesystem::absolute(path).string() +
                 "\"],\"wait\":" + (wait ? "true}" : "false}"),
             "", "/v1/models/variants"));
  }

  static iree_status_t Bytecode(const std::string& path,
                                std::string* out_bytes) {
    const auto allocator = loomc_allocator_system();
    loomc_source_t* source = nullptr;
    auto status = iree_status_from_loomc(loomc_source_create_from_path(
        loomc_make_cstring_view(path.c_str()), nullptr, allocator, &source));
    loomc_context_t* context = nullptr;
    loomc_workspace_t* workspace = nullptr;
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    loomc_source_t* bytecode = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(
          loomc_context_create(nullptr, allocator, &context));
    }
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(
          loomc_workspace_create(nullptr, allocator, &workspace));
    }
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(loomc_module_deserialize_from_source(
          context, workspace, source, nullptr, allocator, &module, &result));
    }
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(loomc_module_serialize_bytecode_to_source(
          module, nullptr, allocator, &bytecode));
    }
    if (iree_status_is_ok(status)) {
      const auto bytes = loomc_source_contents(bytecode);
      out_bytes->assign(reinterpret_cast<const char*>(bytes.data),
                        bytes.data_length);
    }
    loomc_source_release(bytecode);
    loomc_result_release(result);
    loomc_module_release(module);
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_source_release(source);
    return status;
  }

  std::string Upload(uint64_t base, const std::string& path,
                     bool binary = false, bool wait = true) {
    std::ifstream input(path, std::ios::binary);
    EXPECT_TRUE(input.good());
    std::string source((std::istreambuf_iterator<char>(input)), {});
    if (binary) {
      [&] { IREE_ASSERT_OK(Bytecode(path, &source)); }();
      if (HasFatalFailure()) {
        return {};
      }
    }
    const std::string name = binary ? "native.loombc" : "native.loom";
    const std::string json =
        "{\"model\":\"fixture-text\",\"base\":" + std::to_string(base) +
        ",\"sources\":[\"" + name +
        "\"],\"wait\":" + (wait ? "true" : "false") + "}";
    loom_serve::testing::SourcePackage package;
    package.Add("model/" + name, source);
    const auto request = package.Request("/v1/models/variants", json);
    int client = -1;
    [&] { IREE_ASSERT_OK(http_.Send(request, &client)); }();
    return Read(client);
  }

  std::string Select(uint64_t id, uint64_t expected) {
    return Read(
        Send("{\"model\":\"fixture-text\",\"variant\":" + std::to_string(id) +
                 ",\"expected\":" + std::to_string(expected) + "}",
             "", "/v1/models/variants/select"));
  }

  std::string Remove(uint64_t id) {
    return Read(Send(
        "{\"model\":\"fixture-text\",\"variant\":" + std::to_string(id) + "}",
        "", "/v1/models/variants/remove"));
  }
};

class RegisteredReconfigurationHttpTextTest
    : public ReconfigurationHttpTextTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }
};

TEST_F(RegisteredReconfigurationHttpTextTest,
       UploadedRegistrationOwnsDeferredVmAndMetadata) {
  StartManaged();
  {
    loom_serve::testing::SourcePackage package;
    for (const auto* file :
         {"prepare.loom", "advance.loom", "weights.loom", "tokenizer.json"}) {
      IREE_ASSERT_OK(package.AddFile(std::string("model/") + file,
                                     directory_ + "/" + file));
    }
    std::string bytecode;
    IREE_ASSERT_OK(Bytecode(directory_ + "/control.loom", &bytecode));
    package.Add("model/control.loombc", bytecode);
    const std::string declaration =
        R"({"name":"fixture-text","admission":"drained","definition":{)"
        R"("kind":"text","prepare":"prepare.loom","control":"control.loombc",)"
        R"("sources":["advance.loom"],"tokenizer":"tokenizer.json","weights":")" +
        std::filesystem::absolute(weights_).string() +
        R"(","rows":4,"context_capacity":96,"pool_capacity":256,"prefill_capacity":32}})";
    int client = -1;
    IREE_ASSERT_OK(http_.Send(
        package.Request("/v1/models/register", declaration), &client));
    const auto registered = Read(client);
    ASSERT_EQ(registered.find("HTTP/1.1 202"), 0u) << registered;
    EXPECT_NE(registered.find(R"("registration":"registered")"),
              std::string::npos);
  }
  // Both the encoder and the upload socket are gone before VM preparation.
  ASSERT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "abxa"), 2)))),
            Oracle::Complete("abxa", 2));
  EXPECT_NE(
      Read(Send(R"({"model":"fixture-text","content":"abx"})", "", "/tokenize"))
          .find(R"({"tokens":[0,1,2]})"),
      std::string::npos);
  ASSERT_EQ(
      Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/unregister"))
          .find("HTTP/1.1 202"),
      0u);
  WaitModel("fixture-text", {});
}

TEST_F(ReconfigurationHttpTextTest,
       StageSelectRollbackPreserveWeightsAndConversation) {
  IREE_ASSERT_OK(loom_serve_text_model_activate(model_));
  const auto weights = loom_serve_text_model_weight_activity(model_);
  Start();
  const auto listing = Inventory();
  ASSERT_EQ(listing.find("HTTP/1.1 200"), 0u) << listing;
  const uint64_t original = Number(listing, "selected");
  ASSERT_NE(original, 0u);
  EXPECT_EQ(Number(listing, "count"), 1u);
  const std::string prompt = "abxa";
  const auto first = Content(Read(Send(Request(Message("user", prompt), 2),
                                       "X-Loom-Session: retained\r\n")));
  EXPECT_EQ(first, Oracle::Complete(prompt, 2));

  std::string path;
  ASSERT_NO_FATAL_FAILURE(Provider("scalar.remui %total, %c1_i32", &path));
  const auto accepted = Upload(original, path, true, false);
  ASSERT_EQ(accepted.find("HTTP/1.1 202"), 0u) << accepted;
  const uint64_t candidate = Number(accepted, "id", "variant");
  std::string staged;
  while (!http_.failed() && !HasFatalFailure()) {
    staged = Inventory("X-Loom-Variant: " + std::to_string(candidate) + "\r\n");
    ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
    if (staged.find("\"phase\":\"compiling\"") == std::string::npos) {
      break;
    }
  }
  ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
  EXPECT_NE(staged.find("\"phase\":\"ready\""), std::string::npos);
  EXPECT_NE(staged.find("\"compilation\":"), std::string::npos);
  EXPECT_EQ(Number(staged, "id", "variant"), candidate);
  EXPECT_NE(candidate, original);
  EXPECT_EQ(Number(staged, "selected"), original);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", prompt), 2)))), first);

  const auto selected = Select(candidate, original);
  ASSERT_EQ(selected.find("HTTP/1.1 200"), 0u) << selected;
  EXPECT_EQ(Number(selected, "selected"), candidate);
  EXPECT_EQ(Remove(candidate).find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(Select(original, original).find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(Number(Inventory(), "selected"), candidate);

  const auto messages = Message("user", prompt) + "," +
                        Message("assistant", first) + "," +
                        Message("user", "b");
  const auto changed =
      Read(Send(Request(messages, 2), "X-Loom-Session: retained\r\n"));
  EXPECT_EQ(Content(changed), "aa");
  EXPECT_NE(changed.find("\"cached_tokens\":5"), std::string::npos) << changed;
  const auto restored = Select(original, candidate);
  ASSERT_EQ(restored.find("HTTP/1.1 200"), 0u) << restored;
  EXPECT_EQ(Number(restored, "selected"), original);
  const auto continued =
      Read(Send(Request(messages + "," + Message("assistant", "aa") + "," +
                            Message("user", "b"),
                        2),
                "X-Loom-Session: retained\r\n"));
  EXPECT_EQ(Content(continued), Oracle::Complete(prompt + first + "#baa#b", 2));
  EXPECT_NE(continued.find("\"cached_tokens\":9"), std::string::npos)
      << continued;
  EXPECT_EQ(Remove(candidate).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Number(Inventory(), "count"), 1u);
  EXPECT_EQ(Inventory("X-Loom-Variant: " + std::to_string(candidate) + "\r\n")
                .find("HTTP/1.1 404"),
            0u);

  // Join the real owner before inspecting its counters; HTTP never borrows
  // mutable model state into this test thread.
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  const auto retained = loom_serve_text_model_weight_activity(model_);
  EXPECT_EQ(retained.read_bytes, weights.read_bytes);
  EXPECT_EQ(retained.prepared_bytes, weights.prepared_bytes);
  EXPECT_EQ(retained.activations, weights.activations);
}

TEST_F(ReconfigurationHttpTextTest, CompilationFailuresRemainInspectable) {
  Start();
  const uint64_t original = Number(Inventory(), "selected");
  std::string path;
  ASSERT_NO_FATAL_FAILURE(Provider("scalar.not_an_operation %total", &path));
  const auto failed = Upload(original, path);
  ASSERT_EQ(failed.find("HTTP/1.1 422"), 0u) << failed;
  EXPECT_EQ(Number(failed, "selected"), original);
  const uint64_t rejected = Number(failed, "id", "variant");
  EXPECT_NE(failed.find("\"phase\":\"failed\""), std::string::npos);
  EXPECT_NE(failed.find("not_an_operation"), std::string::npos) << failed;
  const auto retained =
      Inventory("X-Loom-Variant: " + std::to_string(rejected) + "\r\n");
  EXPECT_EQ(retained.find("HTTP/1.1 200"), 0u) << retained;
  EXPECT_NE(retained.find("not_an_operation"), std::string::npos);
  EXPECT_EQ(Select(rejected, original).find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "abxa"), 3)))),
            Oracle::Complete("abxa", 3));
  EXPECT_EQ(Remove(rejected).find("HTTP/1.1 200"), 0u);

  // A detached request has the same persistent observation contract.
  const auto accepted = Stage(original, directory_ + "/advance.loom", false);
  ASSERT_EQ(accepted.find("HTTP/1.1 202"), 0u) << accepted;
  const uint64_t candidate = Number(accepted, "id", "variant");
  std::string terminal;
  while (!http_.failed() && !HasFatalFailure()) {
    terminal =
        Inventory("X-Loom-Variant: " + std::to_string(candidate) + "\r\n");
    ASSERT_EQ(terminal.find("HTTP/1.1 200"), 0u) << terminal;
    if (terminal.find("\"phase\":\"compiling\"") == std::string::npos) {
      break;
    }
  }
  EXPECT_NE(terminal.find("\"phase\":\"ready\""), std::string::npos);
  EXPECT_EQ(Number(terminal, "selected"), original);
  EXPECT_EQ(Remove(candidate).find("HTTP/1.1 200"), 0u);
}

}  // namespace
