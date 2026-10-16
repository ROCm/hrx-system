// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_EMBEDDING_SOURCE_SERVICE_TEST_FIXTURE_H_
#define LOOM_SERVE_SERVICE_FINITE_EMBEDDING_SOURCE_SERVICE_TEST_FIXTURE_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>

#include "iree/base/threading/notification.h"
#include "iree/task/executor.h"
#include "iree/task/process.h"
#include "iree/testing/gtest.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/service/finite/model.h"
#include "loom/serve/service/finite/service.h"
#include "loom/serve/telemetry/record.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/transport/http/listener.h"

namespace loom_serve::testing::embedding {

// Returns the optional resident stage source configured for this test process.
const char* SourceServiceDirectStageSource();

// Drives one asynchronous model load or retirement on the serving host.
struct loom_serve_test_embedding_model_lifecycle_t {
  static void Wake(void* user_data);

  static iree_status_t Drain(iree_task_process_t* process,
                             const iree_task_worker_context_t* context,
                             iree_task_process_drain_result_t* result);

  static void Release(iree_task_process_t* process);

  iree_status_t Run(loom_serve_device_t* device,
                    const loom_serve_finite_model_options_t& options,
                    iree_allocator_t host_allocator = iree_allocator_system());

  iree_status_t Retire(loom_serve_device_t* device);

  iree_status_t RunOwner(loom_serve_device_t* device);

  // Borrowed shared executor for both owner turns and private preparation.
  iree_task_executor_t* executor = nullptr;
  // Borrowed capacity domain whose wakes keep this owner runnable under
  // pressure.
  loom_serve_memory_pool_t* pool = nullptr;
  // Optional live service advanced by this same serialized pool owner.
  loom_serve_service_t* service = nullptr;
  // Owned in-progress load, consumed at its terminal owner turn.
  loom_serve_finite_load_t* load = nullptr;
  // Prepared model returned by loading, or owned model consumed by retirement.
  loom_serve_finite_model_t* model = nullptr;
  // Copied terminal evidence, surviving failed or cancelled preparation.
  loom_serve_compilation_report_t report = {};
  // Exclusive single-drainer process; asynchronous callbacks only wake it.
  iree_task_process_t process = {};
  // Publishes final release to the outer test, independent of owner turns.
  std::mutex mutex;
  // Outer-only notification of actual process retirement.
  std::condition_variable condition;
  // Exercises cancellation after actual task acceptance, not a fake callback.
  bool cancel_after_start = false;
  // This owner removes a model instead of preparing one.
  bool retiring = false;
  // Cached consumers were released once and ownership observation is armed.
  bool retirement_started = false;
  // Terminal preparation or retirement outcome has been consumed.
  bool complete = false;
  // Executor has made its final access to this process.
  bool released = false;
  // Terminal outcome, owned by the outer caller after Run.
  iree_status_t status = iree_ok_status();
};

// Per-case HTTP/model state borrowing the process-owned compiler, task runtime,
// and physical device group.
class SourceEmbeddingServiceTest : public ::testing::Test {
 protected:
  virtual iree_host_size_t DeviceCount() const;

  void SetUp() override;

  static iree_status_t ListenerAllocatorControl(
      void* self, iree_allocator_command_t command, const void* parameters,
      void** inout_pointer);

  void CreateModel(const std::string& directory);

  void CreateService(iree_duration_t heartbeat_interval);

  void TearDown() override;

  int Send(const std::string& body, const char* target = "/v1/embeddings",
           const char* method = "POST", const std::string& headers = {});

  std::string Read(int client, bool allow_empty = false);

  void Accept(int client,
              loom_serve_http_connection_t** out_connection = nullptr);

  bool Step(loom_serve_service_flags_t flags = 0);

  void Drain();

  void CheckAbandonedCohort(loom_serve_service_flags_t flags,
                            const char* expected = R"("embedding":[1,1,1,1])");

  void StartRouter();

  void StartManagedRouter(iree_host_size_t capacity = 2);

  std::string Declaration(const std::string& name, const std::string& source,
                          const char* admission = "open",
                          const std::string& base = "");

  std::string ModelStatus(const std::string& name);

  std::string WaitModel(const std::string& name, const std::string& marker);

  void CheckInvalidSubmissionRetirement(const std::string& package);

  uint64_t CompletedRequests();

  static uint64_t Number(const std::string& response, const char* field,
                         const char* object = nullptr);

  std::string UploadVariant(uint64_t base, const std::string& directory,
                            const char* provider, const char* replacement,
                            bool include_parameter_source = false);

  std::string SelectVariant(uint64_t variant, uint64_t expected);

  // Writes a package with a scalar checkpoint initialized to 1.0. Its device
  // preparation doubles the factor in place before serving scales the output.
  void WriteCheckpointPackage(const std::string& target);

  void UseDirectPackage();

  // Host allocator used by all production owners.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // Rejects listener receive allocation after startup without changing
  // teardown.
  std::atomic<bool> fail_listener_allocations_{false};
  // Shared CPU/JIT/task runtime.
  loom_serve_host_t* host_ = nullptr;
  // Application group enclosing every device and model resource.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed placement of the source-defined embedding model.
  loom_serve_device_t* device_ = nullptr;
  // Shared source-JIT finite model.
  loom_serve_finite_model_t* model_ = nullptr;
  // A second actual model retaining parameter capacity during the pressure
  // test.
  loom_serve_finite_model_t* retained_ = nullptr;
  // Source package directory, independent of temporary replacements.
  std::string directory_;
  // Production batched protocol service.
  loom_serve_finite_service_t* service_ = nullptr;
  // Serialized callbacks used by both direct turns and the shared router.
  loom_serve_service_t interface_ = {};
  // Production loopback carrier, independent of scheduler turns.
  loom_serve_http_server_t* server_ = nullptr;
  // Independent binding borrowing the HTTP owner and shared host pool.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Outer driver notification for actual task/queue readiness.
  iree_notification_t notification_;
  // Published callback generations.
  std::atomic<uint64_t> wakes_{0};
  // HTTP readiness and optional outer owner, joined before model destruction.
  loom_serve::testing::HttpHarness http_;
  // Copied after the router joins, including counters from removed
  // incarnations.
  loom_serve_telemetry_counters_t retired_counters_ = {};
};

}  // namespace loom_serve::testing::embedding

#endif  // LOOM_SERVE_SERVICE_FINITE_EMBEDDING_SOURCE_SERVICE_TEST_FIXTURE_H_
