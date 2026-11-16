// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/image/service_test_fixture.h"

namespace loom_serve::testing::image {

TEST_P(SourceImageServiceTest,
       UploadedExecutionChangesPartsAndPixelsWithoutRelinking) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  IREE_ASSERT_OK(loom_serve_image_model_activate(model_));
  const auto weights = loom_serve_image_model_parameter_activity(model_);
  Start();
  const auto inventory = Read(Send("GET", "/v1/models/variants", {},
                                   "X-Loom-Model: fixture-image\r\n"));
  ASSERT_EQ(inventory.find("HTTP/1.1 200"), 0u) << inventory;
  const uint64_t original = Number(inventory, "selected");
  const std::string request =
      R"({"model":"fixture-image","prompt":"aba","seed":1})";
  auto generate = [&](uint64_t seed) {
    const auto response = Read(Send("POST", "/v1/images/generations", request));
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
    EXPECT_NE(response.find("\"b64_json\":\"" + EncodedPixels(3, seed) + "\""),
              std::string::npos)
        << response;
  };
  generate(1);
  std::ifstream input(directory_ + "/generate.loom");
  ASSERT_TRUE(input.good());
  std::string source((std::istreambuf_iterator<char>(input)), {});
  const std::string red = "scf.select %odd_seed, %high, %low";
  const auto red_position = source.find(red);
  ASSERT_NE(red_position, std::string::npos);
  source.replace(red_position, red.size(), "scf.select %odd_seed, %low, %high");
  const std::string launch = "  kernel.launch @pixels(%colors, %rgba)";
  const auto launch_position = source.find(launch);
  ASSERT_NE(launch_position, std::string::npos);
  // A second producer is a third barrier wave under command_quantum=1. The
  // replacement table must grow, while the retained source VM sees new counts.
  source.insert(
      launch_position,
      "  kernel.launch @colors(%request, %colors) : (buffer, buffer)\n");
  const std::string allocation = "index.constant 8 : offset";
  const auto allocation_position = source.find(allocation);
  ASSERT_NE(allocation_position, std::string::npos);
  source.replace(allocation_position, allocation.size(),
                 "index.constant 4160 : offset");
  loom_serve::testing::SourcePackage package;
  package.Add("model/generate.loom", source);
  IREE_ASSERT_OK(
      package.AddFile("model/config.loom", directory_ + "/config.loom"));
  int client = -1;
  IREE_ASSERT_OK(http_.Send(
      package.Request(
          "/v1/models/variants",
          "{\"model\":\"fixture-image\",\"base\":" + std::to_string(original) +
              ",\"sources\":[\"config.loom\",\"generate.loom\"],\"wait\":"
              "true}"),
      &client));
  const auto staged = Read(client);
  ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
  const uint64_t candidate = Number(staged, "id", "variant");
  EXPECT_EQ(Number(staged, "selected"), original);
  generate(1);
  const auto selected = SelectVariant(candidate, original);
  ASSERT_EQ(selected.find("HTTP/1.1 200"), 0u) << selected;
  generate(0);
  EXPECT_EQ(SelectVariant(original, original).find("HTTP/1.1 409"), 0u);
  ASSERT_EQ(SelectVariant(original, candidate).find("HTTP/1.1 200"), 0u);
  generate(1);
  const auto metrics = Read(Send("GET", "/metrics"));
  const std::string batches = "\nloom_serve_server_batches_total ";
  const auto position = metrics.find(batches);
  ASSERT_NE(position, std::string::npos) << metrics;
  EXPECT_EQ(std::stoull(metrics.substr(position + batches.size())),
            GetParam() ? 9u : 4u);
  loom_serve::testing::SourcePackage broken;
  broken.Add("model/generate.loom", "not a Loom program");
  IREE_ASSERT_OK(http_.Send(
      broken.Request(
          "/v1/models/variants",
          "{\"model\":\"fixture-image\",\"base\":" + std::to_string(original) +
              ",\"sources\":[\"generate.loom\"],\"wait\":true}"),
      &client));
  const auto failed = Read(client);
  EXPECT_EQ(failed.find("HTTP/1.1 422"), 0u) << failed;
  EXPECT_EQ(Number(failed, "selected"), original);
  generate(1);
  // The original preparation owns the budget even when a candidate's command
  // program grows. Reject before selection without disturbing the live model.
  const std::string grown_allocation = "index.constant 4160 : offset";
  source.replace(source.find(grown_allocation), grown_allocation.size(),
                 "index.constant 16384 : offset");
  loom_serve::testing::SourcePackage over_budget;
  over_budget.Add("model/generate.loom", source);
  IREE_ASSERT_OK(
      over_budget.AddFile("model/config.loom", directory_ + "/config.loom"));
  IREE_ASSERT_OK(http_.Send(
      over_budget.Request(
          "/v1/models/variants",
          "{\"model\":\"fixture-image\",\"base\":" + std::to_string(original) +
              ",\"sources\":[\"config.loom\",\"generate.loom\"]}"),
      &client));
  const auto rejected = Read(client);
  EXPECT_EQ(rejected.find("HTTP/1.1 422"), 0u) << rejected;
  EXPECT_NE(rejected.find("workspace"), std::string::npos) << rejected;
  EXPECT_EQ(Number(rejected, "selected"), original);
  generate(1);
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  const auto retained = loom_serve_image_model_parameter_activity(model_);
  EXPECT_EQ(retained.read_bytes, weights.read_bytes);
  EXPECT_EQ(retained.prepared_bytes, weights.prepared_bytes);
  EXPECT_EQ(retained.activations, weights.activations);
}

TEST_P(SourceImageServiceTest, PreparedShapePinsCodeAcrossSelectionAndRemoval) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  const auto generator = loom_serve_image_model_generator(model_);
  const auto update = generator.execution_update;
  const std::string paths[] = {directory_ + "/config.loom",
                               directory_ + "/generate.loom"};
  std::string contents[2];
  for (size_t i = 0; i < 2; ++i) {
    std::ifstream input(paths[i]);
    ASSERT_TRUE(input.good());
    contents[i].assign(std::istreambuf_iterator<char>(input), {});
  }
  const std::string red = "scf.select %odd_seed, %high, %low";
  const auto red_position = contents[1].find(red);
  ASSERT_NE(red_position, std::string::npos);
  contents[1].replace(red_position, red.size(),
                      "scf.select %odd_seed, %low, %high");
  const std::string launch = "  kernel.launch @pixels(%colors, %rgba)";
  const auto launch_position = contents[1].find(launch);
  ASSERT_NE(launch_position, std::string::npos);
  contents[1].insert(
      launch_position,
      "  kernel.launch @colors(%request, %colors) : (buffer, buffer)\n");
  const loom_runtime_package_file_t files[] = {
      {IREE_SV("model/config.loom"),
       iree_make_const_byte_span(contents[0].data(), contents[0].size())},
      {IREE_SV("model/generate.loom"),
       iree_make_const_byte_span(contents[1].data(), contents[1].size())}};
  loom_runtime_package_t* package = nullptr;
  IREE_ASSERT_OK(loom_runtime_package_create(IREE_SV("model/model.json"),
                                             IREE_ARRAYSIZE(files), files,
                                             host_allocator_, &package));
  std::unique_ptr<loom_runtime_package_t,
                  decltype(&loom_runtime_package_release)>
      retained_package(package, loom_runtime_package_release);
  const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                          IREE_SV("generate.loom")};
  const loom_runtime_sources_t sources = {
      .count = IREE_ARRAYSIZE(providers),
      .values = providers,
      .package = package,
  };
  struct Compilation {
    // Joins the accepted compiler's publication with this outer test thread.
    std::mutex mutex;
    // Notifies completion without blocking a compiler or serving worker.
    std::condition_variable condition;
    // Compiler result, transferred after the callback's final access.
    void* candidate = nullptr;
    // Owned compilation outcome, consumed after publication.
    iree_status_t status = iree_ok_status();
    // Terminal callback has transferred ownership.
    bool complete = false;
    // Request compilation publications observed by the serialized test owner.
    uint64_t wakes = 0;
  } compilation;
  IREE_ASSERT_OK(update.compile(
      update.self, &sources,
      [](void* self, iree_status_t status, void* value,
         const loom_runtime_compilation_report_t*) {
        auto* compilation = static_cast<Compilation*>(self);
        std::lock_guard<std::mutex> lock(compilation->mutex);
        compilation->candidate = value;
        compilation->status = status;
        compilation->complete = true;
        compilation->condition.notify_one();
      },
      &compilation));
  {
    std::unique_lock<std::mutex> lock(compilation.mutex);
    compilation.condition.wait(lock, [&] { return compilation.complete; });
  }
  std::unique_ptr<void, loom_runtime_variant_dispose_fn_t> candidate(
      compilation.candidate, update.dispose);
  IREE_ASSERT_OK(std::exchange(compilation.status, iree_ok_status()));
  retained_package.reset();
  generator.set_observer(
      generator.self, {[](void* self) {
                         auto* publication = static_cast<Compilation*>(self);
                         std::lock_guard<std::mutex> lock(publication->mutex);
                         ++publication->wakes;
                         publication->condition.notify_all();
                       },
                       &compilation});

  auto complete = [&](uint64_t expected_seed, uint32_t expected_parts) {
    bool resolved = false;
    while (!resolved) {
      uint64_t before;
      {
        std::lock_guard<std::mutex> lock(compilation.mutex);
        before = compilation.wakes;
      }
      IREE_ASSERT_OK(generator.advance_batch(generator.self, false, &resolved));
      if (!resolved) {
        std::unique_lock<std::mutex> lock(compilation.mutex);
        compilation.condition.wait(lock,
                                   [&] { return compilation.wakes != before; });
      }
    }
    IREE_ASSERT_OK(loom_runtime_residency_acquire_wait(
        loom_serve_image_model_residency(model_)));
    bool finished = false;
    uint32_t parts = 0;
    while (!finished) {
      loom_runtime_execution_frontier_t frontier;
      IREE_ASSERT_OK(generator.submit(generator.self, 0, &frontier, &finished));
      ASSERT_NO_FATAL_FAILURE(Wait(frontier));
      ++parts;
    }
    EXPECT_EQ(parts, expected_parts);
    const auto rgba = generator.complete(generator.self);
    const auto expected = Pixels(3, expected_seed, 15 * 9);
    ASSERT_EQ(rgba.data_length, expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(rgba.data[i], expected[i]);
    }
  };
  loom_service_image_prepared_t prepared;
  IREE_ASSERT_OK(
      PrepareSingle(IREE_SV(R"({"prompt":"aba","seed":1})"), 15, 9, &prepared));
  IREE_ASSERT_OK(update.select(update.self, candidate.get()));
  ASSERT_NO_FATAL_FAILURE(complete(1, GetParam() ? 2 : 1));

  IREE_ASSERT_OK(
      PrepareSingle(IREE_SV(R"({"prompt":"aba","seed":1})"), 15, 9, &prepared));
  IREE_ASSERT_OK(update.select(update.self, nullptr));
  candidate.reset();
  ASSERT_NO_FATAL_FAILURE(complete(0, GetParam() ? 3 : 1));

  IREE_ASSERT_OK(
      PrepareSingle(IREE_SV(R"({"prompt":"aba","seed":1})"), 15, 9, &prepared));
  ASSERT_NO_FATAL_FAILURE(complete(1, GetParam() ? 2 : 1));
  generator.set_observer(generator.self, {});
}

TEST_P(SourceImageServiceTest, SelectionWaitsForWholeImageAndEncoderRelease) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  const loom_serve_image_service_options_t options = {
      .name = loom_serve_image_model_name(model_),
      .width = kWidth,
      .height = kHeight,
      .pending_capacity = 8,
      .variant_capacity = 4,
  };
  loom_serve_image_service_t* value = nullptr;
  IREE_ASSERT_OK(loom_serve_image_service_allocate(
      host_, loom_serve_image_model_generator(model_), &options, nullptr,
      scheduling_.binding(), host_allocator_, &value));
  std::unique_ptr<loom_serve_image_service_t,
                  decltype(&loom_serve_image_service_free)>
      service(value, loom_serve_image_service_free);
  const auto interface = loom_serve_image_service_interface(value);
  auto* notification = loom_serve_http_server_notification(server_);
  interface.runtime.vtable->set_observer(
      interface.runtime.self, {[](void* self) {
                                 iree_notification_post(
                                     static_cast<iree_notification_t*>(self),
                                     IREE_ALL_WAITERS);
                               },
                               notification});
  auto advance_until = [&](auto ready) {
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      IREE_ASSERT_OK(interface.runtime.vtable->advance(interface.runtime.self,
                                                       0, &progress));
      if (ready() || progress) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  };
  loom_serve::testing::ModelDocument document;
  IREE_ASSERT_OK(document.Load(directory_ + "/model.json"));
  auto sources = document.value().sources;
  sources.prepare = {};
  sources.control = {};
  const auto& configuration = interface.reconfiguration;
  const auto original = configuration.selected(configuration.self);
  loom_runtime_variant_id_t candidate = 0;
  IREE_ASSERT_OK(configuration.stage_native(configuration.self, original,
                                            &sources, &candidate));
  loom_runtime_variant_snapshot_t snapshot;
  ASSERT_NO_FATAL_FAILURE(advance_until([&] {
    return loom_runtime_variant_pool_snapshot(configuration.pool, candidate,
                                              &snapshot) &&
           snapshot.phase != LOOM_RUNTIME_VARIANT_COMPILING;
  }));
  ASSERT_EQ(snapshot.phase, LOOM_RUNTIME_VARIANT_READY)
      << std::string(snapshot.message.data ? snapshot.message.data : "",
                     snapshot.message.size);
  const int client =
      Send("POST", "/v1/images/generations",
           R"({"model":"fixture-image","prompt":"aba","seed":1})");
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(http_.TakeRequest(client, &connection, &request));
  IREE_ASSERT_OK(
      interface.accept(interface.runtime.self, connection, request, 0));
  ASSERT_NO_FATAL_FAILURE(advance_until([&] {
    return std::string(interface.snapshot(interface.runtime.self).phase) ==
           "generating";
  }));
  struct loom_serve_test_image_selection_t {
    // Borrowed live service, accessed only on this serialized owner.
    const loom_serve_service_t* interface;
    // Terminal selection publication after the entire invocation retires.
    bool complete;
  } selection{&interface, false};
  IREE_ASSERT_OK(configuration.select(
      configuration.self, candidate, original,
      [](void* user_data, iree_status_t status,
         loom_runtime_variant_id_t selected) {
        auto* selection =
            static_cast<loom_serve_test_image_selection_t*>(user_data);
        IREE_EXPECT_OK(status);
        const auto* interface = selection->interface;
        EXPECT_STREQ(interface->snapshot(interface->runtime.self).phase,
                     "idle");
        EXPECT_EQ(interface->telemetry.counters->completed, 1u);
        selection->complete = true;
      },
      &selection));
  EXPECT_FALSE(selection.complete);
  ASSERT_NO_FATAL_FAILURE(advance_until([&] { return selection.complete; }));
  EXPECT_EQ(configuration.selected(configuration.self), candidate);
  EXPECT_NE(Read(client).find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos);
  EXPECT_EQ(interface.telemetry.counters->batches, GetParam() ? 2u : 1u);
  interface.runtime.vtable->set_observer(interface.runtime.self, {});
}

TEST_P(SourceImageServiceTest, SourcePixelsAndHttpPngPreserveRequestIdentity) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  // A parameter-free model needs no device allocation until an invocation is
  // admitted. Compilation and tokenizer preparation leave its timeline idle.
  const auto loaded =
      loom_runtime_residency_frontier(loom_serve_image_model_residency(model_));
  EXPECT_EQ(loaded.values[0], 0u);
  EXPECT_EQ(loaded.values[1], 0u);
  loom_service_image_prepared_t prepared;
  IREE_ASSERT_OK(
      PrepareSingle(IREE_SV(R"({"prompt":"x"})"), kWidth, kHeight, &prepared));
  EXPECT_EQ(prepared.stage, 0u);
  EXPECT_EQ(prepared.tag, 32);
  EXPECT_EQ(prepared.input_bytes, 16u + 32u);
  loom_serve_image_model_discard_batch(model_);
  // Verify source arithmetic independently of the image encoder.
  for (uint64_t seed : {0u, 1u}) {
    const std::string body =
        R"({"prompt":"aba","seed":)" + std::to_string(seed) + "}";
    ImageRequest input;
    IREE_ASSERT_OK(input.Initialize(
        iree_make_string_view(body.data(), body.size()), kWidth, kHeight));
    iree_const_byte_span_t rgba;
    IREE_ASSERT_OK(loom_serve_image_model_generate(model_, &input.value,
                                                   &prepared, &rgba));
    EXPECT_EQ(prepared.stage, 0u);
    EXPECT_EQ(prepared.tag, 32);
    EXPECT_EQ(prepared.input_bytes, 16u + 32u);
    const auto expected = Pixels(3, seed);
    ASSERT_EQ(rgba.data_length, expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(rgba.data[i], expected[i]);
    }
  }
  Start();
  EXPECT_NE(
      Read(Send("GET", "/v1/models"))
          .find(
              R"("input_modalities":["text"],"output_modalities":["image"],"loom":{"kind":"image","endpoint":"/v1/images/generations","tokenize":true,"variants":)"),
      std::string::npos);
  EXPECT_NE(
      Read(Send("GET", "/v1/status"))
          .find(
              R"("status":{"phase":"idle","queued_requests":0,"active_requests":0,"oldest_queued_age_seconds":0.000000000,"oldest_active_age_seconds":0.000000000,"admission":"open","held_requests":0,"tokenizer_requests":0})"),
      std::string::npos);
  EXPECT_NE(Read(Send("GET", "/readyz")).find(R"({"status":"ready"})"),
            std::string::npos);
  EXPECT_NE(Read(Send("POST", "/tokenize",
                      R"({"model":"fixture-image","content":"aba"})"))
                .find(R"({"tokens":[0,1,0]})"),
            std::string::npos);
  EXPECT_NE(Read(Send("POST", "/detokenize",
                      R"({"model":"fixture-image","tokens":[1,0,1]})"))
                .find(R"({"content":"bab"})"),
            std::string::npos);
  // A rejected request must leave the same listener/model usable.
  EXPECT_EQ(Read(Send("POST", "/v1/images/generations",
                      R"({"model":"fixture-image","prompt":"ab","n":2})"))
                .find("HTTP/1.1 400"),
            0u);
  // This rejection comes from the source VM after ordinary HTTP parsing.
  const std::string too_long =
      Read(Send("POST", "/v1/images/generations",
                "{\"model\":\"fixture-image\",\"prompt\":\"" +
                    std::string(33, 'a') + "\"}"));
  ASSERT_EQ(too_long.find("HTTP/1.1 400"), 0u);
  EXPECT_NE(too_long.find("prompt exceeds fixture text capacity"),
            std::string::npos);
  const auto unknown_lora = Read(Send(
      "POST", "/v1/images/generations",
      R"({"model":"fixture-image","prompt":"ab","loras":[{"id":"missing"}]})"));
  EXPECT_EQ(unknown_lora.find("HTTP/1.1 400"), 0u) << unknown_lora;
  EXPECT_NE(unknown_lora.find("unknown or duplicate fixture LoRA ID"),
            std::string::npos);
  const char* prompts[] = {"ab", "aba", "b", "abba"};
  std::vector<int> clients;
  for (size_t i = 0; i < IREE_ARRAYSIZE(prompts); ++i) {
    clients.push_back(
        Send("POST", "/v1/images/generations",
             std::string("{\"model\":\"fixture-image\",\"prompt\":\"") +
                 prompts[i] + "\",\"seed\":" + std::to_string(i) + "}"));
  }
  for (size_t i = 0; i < clients.size(); ++i) {
    SCOPED_TRACE(i);
    const std::string encoded =
        EncodedPixels(std::string(prompts[i]).size(), i);
    const std::string response = Read(clients[i]);
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
    EXPECT_NE(response.find("\"b64_json\":\"" + encoded + "\""),
              std::string::npos);
  }
  EXPECT_NE(Read(Send("GET", "/healthz")).find("\"status\":\"ready\""),
            std::string::npos);
}

TEST_P(SourceImageServiceTest, ImpossibleWorkingSetLeavesModelUsable) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  Start();
  const auto large = Read(Send(
      "POST", "/v1/images/generations",
      R"({"model":"fixture-image","prompt":"aba","seed":1,"size":"64x64"})"));
  if (loom_runtime_pool_memory(storage_owner_)) {
    ASSERT_EQ(large.find("HTTP/1.1 503"), 0u) << large;
    EXPECT_NE(large.find("invocation working set"), std::string::npos);
  } else {
    ASSERT_EQ(large.find("HTTP/1.1 200"), 0u) << large;
    EXPECT_NE(large.find(EncodedPixels(3, 1, 64, 64)), std::string::npos);
  }
  const auto small =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"fixture-image","prompt":"aba","seed":1})"));
  ASSERT_EQ(small.find("HTTP/1.1 200"), 0u) << small;
  EXPECT_NE(small.find(EncodedPixels(3, 1)), std::string::npos);
}

TEST_P(SourceImageServiceTest, RequestGeometryOwnsPixelsAndPreservesDefault) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  Start();
  const std::array<uint32_t, 2> dimensions[] = {
      {15, 9}, {9, 15}, {16, 11}, {1, 1}, {15, 9}, {kWidth, kHeight}};
  std::vector<int> clients;
  for (size_t i = 0; i < IREE_ARRAYSIZE(dimensions); ++i) {
    const auto [width, height] = dimensions[i];
    const std::string size = i + 1 == IREE_ARRAYSIZE(dimensions)
                                 ? ""
                                 : ",\"size\":\"" + std::to_string(width) +
                                       "x" + std::to_string(height) + "\"";
    clients.push_back(
        Send("POST", "/v1/images/generations",
             "{\"model\":\"fixture-image\",\"prompt\":\"aba\",\"seed\":" +
                 std::to_string(i) + size + "}"));
  }
  uint64_t generated_pixels = 0;
  for (size_t i = 0; i < clients.size(); ++i) {
    const auto [width, height] = dimensions[i];
    const auto response = Read(clients[i]);
    ASSERT_EQ(response.find("HTTP/1.1 200 OK\r\n"), 0u) << response;
    EXPECT_NE(response.find("\"b64_json\":\"" +
                            EncodedPixels(3, i, width, height) + "\""),
              std::string::npos);
    generated_pixels += (uint64_t)width * height;
  }
  const auto rejected =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"fixture-image","prompt":"aba","size":"65x9"})"));
  EXPECT_EQ(rejected.find("HTTP/1.1 400 Bad Request\r\n"), 0u) << rejected;
  EXPECT_NE(rejected.find("1-64 pixel extents"), std::string::npos);
  const auto restored =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"fixture-image","prompt":"aba","seed":1})"));
  EXPECT_EQ(restored.find("HTTP/1.1 200 OK\r\n"), 0u) << restored;
  EXPECT_NE(restored.find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos);
  generated_pixels += kPixels;
  // The model log has a bounded page size. Follow its cursor to the final
  // cumulative snapshot instead of treating the first page as the whole log.
  std::string cursor;
  std::string events;
  std::string latest;
  for (;;) {
    latest =
        Read(Send("GET", "/logz", {},
                  "X-Loom-Model: fixture-image\r\n" +
                      (cursor.empty() ? std::string()
                                      : "X-Loom-Cursor: " + cursor + "\r\n")));
    ASSERT_EQ(latest.find("HTTP/1.1 200 OK\r\n"), 0u) << latest;
    ASSERT_EQ(latest.find(R"("event":"gap")"), std::string::npos) << latest;
    events += latest;
    const std::string marker = R"({"event":"cursor","value":")";
    const size_t position = latest.find(marker);
    ASSERT_NE(position, std::string::npos) << latest;
    const size_t first = position + marker.size();
    const std::string next =
        latest.substr(first, latest.find('"', first) - first);
    if (cursor == next) {
      break;
    }
    cursor = next;
  }
  EXPECT_NE(latest.find(R"("requests":8,"completed":7,"rejected":1)"),
            std::string::npos)
      << latest;
  EXPECT_NE(events.find("\"partial\":false"), std::string::npos) << events;
  const std::string pixel_count = "\"generated_pixels\":";
  const size_t position = events.rfind(pixel_count);
  ASSERT_NE(position, std::string::npos) << events;
  const uint64_t recorded_pixels =
      std::stoull(events.substr(position + pixel_count.size()));
  EXPECT_EQ(recorded_pixels, generated_pixels);
}

TEST_P(SourceImageServiceTest, ClosedTransportTerminatesModelPolling) {
  StartManaged();
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  ::testing::TestPartResultArray failures;
  {
    ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::
            INTERCEPT_ONLY_CURRENT_THREAD,
        &failures);
    EXPECT_TRUE(WaitModelState("absent", "ready").empty());
  }
  ASSERT_EQ(failures.size(), 1);
  EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
  EXPECT_TRUE(http_.failed());
}

TEST_P(SourceImageServiceTest, ShapeCompilationFailureLeavesModelReusable) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  Start();
  const auto inventory = Read(Send("GET", "/v1/models/variants", {},
                                   "X-Loom-Model: fixture-image\r\n"));
  const uint64_t original = Number(inventory, "selected");
  std::ifstream input(directory_ + "/generate.loom");
  ASSERT_TRUE(input.good());
  std::string source((std::istreambuf_iterator<char>(input)), {});
  const std::string range = "range(%value, 1, 4096)";
  const auto offset = source.find(range);
  ASSERT_NE(offset, std::string::npos);
  source.replace(offset, range.size(), "range(%value, 1, 100)");
  loom_serve::testing::SourcePackage package;
  package.Add("model/generate.loom", source);
  IREE_ASSERT_OK(
      package.AddFile("model/config.loom", directory_ + "/config.loom"));
  int client = -1;
  IREE_ASSERT_OK(http_.Send(
      package.Request(
          "/v1/models/variants",
          "{\"model\":\"fixture-image\",\"base\":" + std::to_string(original) +
              ",\"sources\":[\"config.loom\",\"generate.loom\"],\"wait\":"
              "true}"),
      &client));
  const auto staged = Read(client);
  ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
  ASSERT_EQ(SelectVariant(Number(staged, "id", "variant"), original)
                .find("HTTP/1.1 200"),
            0u);
  const auto rejected =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"fixture-image","prompt":"aba","size":"15x9"})"));
  EXPECT_EQ(rejected.find("HTTP/1.1 422"), 0u) << rejected;
  const auto valid = Read(Send(
      "POST", "/v1/images/generations",
      R"({"model":"fixture-image","prompt":"aba","seed":1,"size":"9x9"})"));
  EXPECT_EQ(valid.find("HTTP/1.1 200"), 0u) << valid;
  EXPECT_NE(valid.find("\"b64_json\":\"" + EncodedPixels(3, 1, 9, 9) + "\""),
            std::string::npos);
}

TEST_P(SourceImageServiceTest, IndependentLogReadersCannotBlockColdImageWork) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  loom_runtime::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  Start();
  auto poll = [&](const std::string& cursor) {
    return Read(
        Send("GET", "/logz", "",
             "X-Loom-Model: fixture-image\r\n" +
                 (cursor.empty() ? std::string()
                                 : "X-Loom-Cursor: " + cursor + "\r\n")));
  };
  auto cursor_of = [](const std::string& response) {
    const std::string marker = R"({"event":"cursor","value":")";
    const size_t start = response.find(marker);
    EXPECT_NE(start, std::string::npos) << response;
    if (start == std::string::npos) {
      return std::string();
    }
    const size_t first = start + marker.size();
    return response.substr(first, response.find('"', first) - first);
  };
  const std::string initial = poll("");
  ASSERT_NE(initial.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  const std::string slow = cursor_of(initial);
  std::string fast = slow;
  std::string latest;
  std::string events;
  auto drain = [&] {
    for (;;) {
      latest = poll(fast);
      ASSERT_FALSE(HasFatalFailure());
      EXPECT_EQ(latest.find(R"("event":"gap")"), std::string::npos);
      events += latest;
      const std::string next = cursor_of(latest);
      if (next == fast) {
        break;
      }
      fast = next;
    }
  };
  for (size_t wave = 0; wave < 10; ++wave) {
    std::vector<int> clients;
    for (size_t i = 0; i < 4; ++i) {
      const std::string prompt = i & 1 ? "aba" : "ab";
      clients.push_back(Send("POST", "/v1/images/generations",
                             "{\"model\":\"fixture-image\",\"prompt\":\"" +
                                 prompt + "\",\"seed\":" + std::to_string(i) +
                                 "}"));
    }
    for (size_t i = 0; i < clients.size(); ++i) {
      const std::string response = Read(clients[i]);
      EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
      EXPECT_NE(response.find("\"b64_json\":\"" +
                              EncodedPixels(i & 1 ? 3 : 2, i) + "\""),
                std::string::npos);
    }
    drain();
  }
  EXPECT_EQ(Read(Send("POST", "/v1/images/generations",
                      R"({"model":"fixture-image","prompt":"aba","n":2})"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/v1/images/generations",
                      "{\"model\":\"fixture-image\",\"prompt\":\"" +
                          std::string(33, 'a') + "\"}"))
                .find("HTTP/1.1 400"),
            0u);
  drain();
  // Arrival timing controls cohort size; image and request totals do not depend
  // on it. Deterministic owner-driven tests check exact command-part counts.
  const std::string totals =
      R"("requests":42,"completed":40,"rejected":2,"cancelled":0,"failed":0,"input_tokens":0,"output_tokens":0)";
  EXPECT_NE(latest.find(totals), std::string::npos);
  EXPECT_NE(events.find(R"("generated_images":40,"generated_pixels":1400)"),
            std::string::npos);
  EXPECT_NE(events.find(R"("event":"image_request_stats")"), std::string::npos);
  EXPECT_NE(events.find(R"("event":"image_prepared")"), std::string::npos);
  EXPECT_NE(events.find(R"("tag":32,"input_bytes":)"), std::string::npos);
  EXPECT_NE(events.find(R"("event":"image_generated")"), std::string::npos);
  EXPECT_NE(events.find(R"("event":"parameter_activity")"), std::string::npos);
  EXPECT_NE(
      events.find(
          R"("activations":0,"deactivations":0,"read_bytes":0,"prepared_bytes":0,"groups":0)"),
      std::string::npos);
  for (const std::string event : {"image_prepared", "image_generated"}) {
    const std::string key = "\"event\":\"" + event + "\"";
    size_t count = 0;
    for (size_t offset = events.find(key); offset != std::string::npos;
         offset = events.find(key, offset + key.size())) {
      ++count;
    }
    EXPECT_EQ(count, 40u) << event;
  }
  EXPECT_EQ(events.find("aba"), std::string::npos);
  const std::string resumed = poll(slow);
  EXPECT_NE(resumed.find(R"({"event":"gap","from":1,)"), std::string::npos);
  EXPECT_NE(resumed.find(totals), std::string::npos);
  EXPECT_NE(poll("").find(R"("partial":true)"), std::string::npos);
  EXPECT_EQ(cursor_of(poll(fast)), fast);
  loom_serve_http_server_request_stop(server_);
  IREE_EXPECT_OK(http_.Join());
}

TEST_P(SourceImageServiceTest, DynamicImageAdmissionRetirementAndReuse) {
  StartManaged();
  EXPECT_NE(Read(Send("GET", "/v1/models")).find(R"("data":[])"),
            std::string::npos);
  loom_runtime::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(Read(Send("POST", "/v1/models/register",
                      Declaration("pixels", directory_, "drained")))
                .find("HTTP/1.1 202"),
            0u);
  const auto initial = ModelStatus("pixels");
  EXPECT_NE(initial.find(R"("registration":"registered")"), std::string::npos);
  EXPECT_NE(initial.find(R"("admission":"drained")"), std::string::npos);
  const auto initial_log =
      Read(Send("GET", "/logz", "", "X-Loom-Model: pixels\r\n"));
  EXPECT_NE(initial_log.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  const int held = Send("POST", "/v1/images/generations",
                        R"({"model":"pixels","prompt":"aba","seed":1})");
  const int tokens =
      Send("POST", "/tokenize", R"({"model":"pixels","content":"aba"})");
  WaitModelState("pixels", R"("held_requests":2)");
  EXPECT_EQ(Read(Send("POST", "/v1/models/undrain", R"({"model":"pixels"})"))
                .find("HTTP/1.1 200"),
            0u);
  const auto response = Read(held);
  const auto tokenizer_response = Read(tokens);
  ASSERT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
  EXPECT_NE(response.find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos);
  EXPECT_NE(tokenizer_response.find(R"({"tokens":[0,1,0]})"),
            std::string::npos);
  const auto ready = ModelStatus("pixels");
  EXPECT_NE(ready.find(R"("registration":"ready")"), std::string::npos);
  // The process-owned device cache may already contain this exact source
  // generation. Both native entries must be accounted as one complete cold
  // compilation or one complete-stage reuse, never a partial mixture.
  const bool compiled =
      ready.find(
          R"("commands":1,"kernels":2,"reused_kernels":0,"entries":2)") !=
      std::string::npos;
  const bool reused =
      ready.find(
          R"("commands":1,"kernels":0,"reused_kernels":2,"entries":2)") !=
      std::string::npos;
  EXPECT_NE(compiled, reused) << ready;
  const auto log = Read(Send("GET", "/logz", "", "X-Loom-Model: pixels\r\n"));
  EXPECT_NE(log.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  EXPECT_NE(log.find(R"("event":"image_generated")"), std::string::npos);

  // The accepted registration belongs to the registry, not its admin peer.
  const int detached =
      Send("POST", "/v1/models/register", Declaration("peer", directory_));
  iree_status_t detached_status = http_.AwaitResponse(detached);
  EXPECT_EQ(close(detached), 0);
  IREE_ASSERT_OK(detached_status);
  const int peer = Send("POST", "/v1/images/generations",
                        R"({"model":"peer","prompt":"ab","seed":0})");
  EXPECT_EQ(Read(Send("POST", "/v1/models/drain", R"({"model":"pixels"})"))
                .find("HTTP/1.1 200"),
            0u);
  const int removed = Send("POST", "/v1/images/generations",
                           R"({"model":"pixels","prompt":"a"})");
  WaitModelState("pixels", R"("held_requests":1)");
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"pixels","mode":"drain"})"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Read(removed).find("HTTP/1.1 410"), 0u);
  WaitModelAbsent("pixels");
  EXPECT_NE(Read(peer).find("\"b64_json\":\"" + EncodedPixels(2, 0) + "\""),
            std::string::npos);

  // Reusing the route neither reuses its counters nor invalidates its peer.
  EXPECT_EQ(Read(Send("POST", "/v1/models/register",
                      Declaration("pixels", directory_)))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_NE(Read(Send("POST", "/v1/images/generations",
                      R"({"model":"pixels","prompt":"ab","seed":1})"))
                .find("\"b64_json\":\"" + EncodedPixels(2, 1) + "\""),
            std::string::npos);
  const auto metrics = Read(Send("GET", "/metrics"));
  EXPECT_NE(metrics.find("loom_serve_server_completed_total 3\n"),
            std::string::npos);
  EXPECT_NE(metrics.find("loom_serve_model_completed_total{model=\"pixels\","
                         "kind=\"image\",incarnation=\"3\"} 1\n"),
            std::string::npos);
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"pixels","mode":"abort"})"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModelAbsent("pixels");
}

TEST_P(SourceImageServiceTest, UploadedImageRegistrationProducesExactPixels) {
  StartManaged();
  {
    loom_serve::testing::SourcePackage package;
    for (const auto* file : {"prepare.loom", "control.loom", "config.loom",
                             "generate.loom", "tokenizer.json"}) {
      IREE_ASSERT_OK(package.AddFile(std::string("model/") + file,
                                     directory_ + "/" + file));
    }
    const std::string declaration =
        R"({"name":"pixels","admission":"drained","definition":{)"
        R"("kind":"image","prepare":"prepare.loom","control":"control.loom",)"
        R"("sources":["config.loom","generate.loom"],"checkpoint":".",)"
        R"("width":7,"height":5,"text_tokens":32,"command_quantum":)" +
        std::to_string(GetParam()) + "}}";
    int client = -1;
    IREE_ASSERT_OK(http_.Send(
        package.Request("/v1/models/register", declaration), &client));
    const auto registered = Read(client);
    ASSERT_EQ(registered.find("HTTP/1.1 202"), 0u) << registered;
    EXPECT_NE(registered.find(R"("registration":"registered")"),
              std::string::npos);
  }
  ASSERT_EQ(Read(Send("POST", "/v1/models/undrain", R"({"model":"pixels"})"))
                .find("HTTP/1.1 200"),
            0u);
  const auto response =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"pixels","prompt":"aba","seed":1})"));
  EXPECT_NE(response.find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos)
      << response;
  // Registration retirement also joins accepted, detached compilation, not
  // just the last image's device timeline. Uploaded providers outlive this
  // request and are released before the model's parameter roots.
  const uint64_t base = Number(
      Read(Send("GET", "/v1/models/variants", {}, "X-Loom-Model: pixels\r\n")),
      "selected");
  {
    loom_serve::testing::SourcePackage package;
    for (const char* file : {"config.loom", "generate.loom"}) {
      IREE_ASSERT_OK(package.AddFile(std::string("model/") + file,
                                     directory_ + "/" + file));
    }
    int client = -1;
    IREE_ASSERT_OK(http_.Send(
        package.Request(
            "/v1/models/variants",
            "{\"model\":\"pixels\",\"base\":" + std::to_string(base) +
                ",\"sources\":[\"config.loom\",\"generate.loom\"],\"wait\":"
                "false}"),
        &client));
    const auto accepted = Read(client);
    ASSERT_EQ(accepted.find("HTTP/1.1 202"), 0u) << accepted;
  }
  ASSERT_EQ(Read(Send("POST", "/v1/models/unregister", R"({"model":"pixels"})"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModelAbsent("pixels");
}

TEST_P(SourceImageServiceTest, TokenizerFailureRetiresCompiledModel) {
  const auto assets =
      iree::testing::MakeTempFilePath("missing_image_tokenizer");
  ASSERT_TRUE(std::filesystem::create_directory(assets));
  loom_serve::testing::ModelDocument document;
  IREE_ASSERT_OK(document.Load(directory_ + "/model.json"));
  const loom_serve_image_model_options_t options = {
      .sources = document.value().sources,
      .checkpoint_directory = iree_make_cstring_view(assets.c_str()),
      .height = kHeight,
      .width = kWidth,
      .text_tokens = 32,
      .command_quantum = GetParam(),
  };
  const auto before = loom_runtime_pool_statistics(storage_owner_);
  loom_serve_image_model_t* failed = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_image_model_allocate(device_, storage_owner_, &options,
                                      host_allocator_, &failed));
  EXPECT_EQ(failed, nullptr);
  const auto after = loom_runtime_pool_statistics(storage_owner_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);

  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  Start();
  const auto response =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"fixture-image","prompt":"aba","seed":1})"));
  EXPECT_NE(response.find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos)
      << response;
}

TEST_P(SourceImageServiceTest, FailedImageSourceLeavesPeerUsable) {
  const auto package = iree::testing::MakeTempFilePath("failed_image_source");
  ASSERT_TRUE(std::filesystem::create_directory(package));
  for (const char* file :
       {"config.loom", "control.loom", "prepare.loom", "model.json"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           package + "/" + file));
  }
  {
    std::ofstream source(package + "/generate.loom");
    source << "this is not a source program\n";
    ASSERT_TRUE(source.good());
  }
  StartManaged();
  loom_runtime::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(
      Read(Send("POST", "/v1/models/register", Declaration("peer", directory_)))
          .find("HTTP/1.1 202"),
      0u);
  EXPECT_EQ(Read(Send("POST", "/v1/models/register",
                      Declaration("broken", package, "drained")))
                .find("HTTP/1.1 202"),
            0u);
  const int held = Send("POST", "/v1/images/generations",
                        R"({"model":"broken","prompt":"a"})");
  WaitModelState("broken", R"("held_requests":1)");
  EXPECT_EQ(Read(Send("POST", "/v1/models/undrain", R"({"model":"broken"})"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(held).find("HTTP/1.1 503"), 0u);
  const auto failed = WaitModelState("broken", R"("registration":"failed")");
  EXPECT_NE(failed.find("generate.loom"), std::string::npos);
  EXPECT_NE(failed.find(R"("diagnostics":[{)"), std::string::npos);
  EXPECT_NE(Read(Send("POST", "/v1/images/generations",
                      R"({"model":"peer","prompt":"aba","seed":1})"))
                .find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos);
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister", R"({"model":"broken"})"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModelAbsent("broken");
}

TEST_P(SourceImageServiceTest, CancelledDeviceImageSkipsEncoding) {
  CheckAbandonedImage(0);
}

TEST_P(SourceImageServiceTest, CompatibleCohortPreservesHttpOutputs) {
  CheckHttpCohort(nullptr);
}

TEST_P(SourceImageServiceTest, CohortPeerResetPreservesSurvivingHttpOutput) {
  CheckHttpCohort("generating");
}

TEST_P(SourceImageServiceTest,
       CohortPeerResetDuringEncodingPreservesNextOutput) {
  CheckHttpCohort("encoding");
}

TEST_P(SourceImageServiceTest, StoppingDeviceImageSkipsEncoding) {
  CheckAbandonedImage(LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
}

TEST_P(SourceImageServiceTest, FailedPartReadinessPreservesActualRetirement) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  loom_service_image_prepared_t prepared;
  IREE_ASSERT_OK(PrepareSingle(IREE_SV(R"({"prompt":"aba","seed":1})"), kWidth,
                               kHeight, &prepared));
  const auto generator = loom_serve_image_model_generator(model_);
  bool resolved = false;
  IREE_ASSERT_OK(generator.advance_batch(generator.self, false, &resolved));
  ASSERT_TRUE(resolved);
  loom_runtime_residency_admission_t admission;
  IREE_ASSERT_OK(generator.try_admit(generator.self, &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
  loom_runtime_execution_frontier_t frontier;
  bool complete = false;
  IREE_ASSERT_OK(generator.submit(generator.self, 0, &frontier, &complete));
  EXPECT_EQ(complete, GetParam() == 0);
  // Failure readiness does not authorize releasing the workspace or upload
  // payload while the native queue still owns its accepted first part.
  iree_hal_semaphore_fail(
      frontier.semaphores[0],
      iree_make_status(IREE_STATUS_ABORTED, "failed image part"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_serve_image_model_free(model_));
  model_ = nullptr;
}

INSTANTIATE_TEST_SUITE_P(CommandQuanta, SourceImageServiceTest,
                         ::testing::Values(0u, 1u));

}  // namespace loom_serve::testing::image
