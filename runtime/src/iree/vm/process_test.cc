// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/process.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "iree/vm/test_allocator.h"

namespace {

using iree::vm::testing::CountingAllocator;

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

enum class EventKind : uint8_t {
  kAttach = 1,
  kSeal = 2,
  kDetach = 3,
  kStart = 4,
  kResume = 5,
  kCleanup = 6,
};

uint32_t MakeEvent(uint8_t module_id, EventKind kind) {
  return (uint32_t{module_id} << 8) | static_cast<uint8_t>(kind);
}

struct EventLog {
  std::array<uint32_t, 32> events = {};
  iree_host_size_t count = 0;

  void Record(uint8_t module_id, EventKind kind) {
    events[count++] = MakeEvent(module_id, kind);
  }
};

void ExpectEvents(const EventLog& log,
                  std::initializer_list<uint32_t> expected_events) {
  ASSERT_EQ(log.count, expected_events.size());
  iree_host_size_t i = 0;
  for (uint32_t expected_event : expected_events) {
    EXPECT_EQ(log.events[i++], expected_event);
  }
}

enum class FailurePhase : uint8_t {
  kNone = 0,
  kAttach = 1,
  kSeal = 2,
};

enum class FunctionBehavior : uint8_t {
  kNone = 0,
  kYieldingInitializer = 1,
};

struct TestModule {
  // Generic module base published at offset zero.
  iree_vm_module_t base = {};
  // Immutable descriptor published through |base|.
  iree_vm_module_descriptor_t descriptor = {};
  // Shared caller-owned lifecycle event storage.
  EventLog* log = nullptr;
  // Small identity encoded into lifecycle events and attached bytes.
  uint8_t module_id = 0;
  // Injected lifecycle failure point.
  FailurePhase failure_phase = FailurePhase::kNone;
  // Optional execution behavior.
  FunctionBehavior function_behavior = FunctionBehavior::kNone;
  // Whether attach observed an entirely zeroed state slice.
  bool saw_zeroed_storage = false;
  // Whether a zero-state callback received canonical empty storage.
  bool saw_canonical_empty_storage = false;
  // Initializer value observed by the seal callback.
  uint32_t sealed_value = 0;
  // Number of final module-owner releases.
  int destroy_count = 0;
};

TestModule* CastTestModule(iree_vm_module_t* base_module) {
  return iree_containerof(base_module, TestModule, base);
}

const TestModule* CastTestModule(const iree_vm_module_t* base_module) {
  return iree_containerof(base_module, TestModule, base);
}

void DestroyTestModule(iree_vm_module_t* base_module) {
  ++CastTestModule(base_module)->destroy_count;
}

iree_status_t AttachTestState(iree_vm_module_t* base_module,
                              iree_byte_span_t storage,
                              iree_allocator_t host_allocator) {
  (void)host_allocator;
  TestModule* module = CastTestModule(base_module);
  module->log->Record(module->module_id, EventKind::kAttach);
  module->saw_canonical_empty_storage = iree_byte_span_is_empty(storage);
  module->saw_zeroed_storage = true;
  for (iree_host_size_t i = 0; i < storage.data_length; ++i) {
    module->saw_zeroed_storage &= storage.data[i] == 0;
  }
  if (storage.data_length) {
    std::memset(storage.data, module->module_id, storage.data_length);
  }
  if (module->failure_phase == FailurePhase::kAttach) {
    if (storage.data_length) std::memset(storage.data, 0, storage.data_length);
    return iree_make_status(IREE_STATUS_ABORTED, "injected attach failure");
  }
  return iree_ok_status();
}

iree_status_t SealTestState(iree_vm_module_t* base_module,
                            iree_byte_span_t storage) {
  TestModule* module = CastTestModule(base_module);
  module->log->Record(module->module_id, EventKind::kSeal);
  if (module->function_behavior == FunctionBehavior::kYieldingInitializer) {
    std::memcpy(&module->sealed_value, storage.data,
                sizeof(module->sealed_value));
  }
  if (module->failure_phase == FailurePhase::kSeal) {
    return iree_make_status(IREE_STATUS_ABORTED, "injected seal failure");
  }
  return iree_ok_status();
}

void DetachTestState(iree_vm_module_t* base_module, iree_byte_span_t storage) {
  TestModule* module = CastTestModule(base_module);
  module->log->Record(module->module_id, EventKind::kDetach);
  if (storage.data_length) std::memset(storage.data, 0, storage.data_length);
}

struct InitializerFrame {
  // Module receiving the eventual frame cleanup callback.
  TestModule* module;
};

void CleanupInitializerFrame(iree_vm_frame_t* frame) {
  auto* payload = static_cast<InitializerFrame*>(iree_vm_frame_storage(frame));
  payload->module->log->Record(payload->module->module_id, EventKind::kCleanup);
}

iree_status_t StartTestFunction(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  TestModule* module = CastTestModule(base_module);
  module->log->Record(module->module_id, EventKind::kStart);
  const uint32_t value =
      static_cast<uint32_t>(iree_vm_call_value_argument_load(&params->call, 0));
  std::memcpy(params->execution.process_storage, &value, sizeof(value));
  const iree_vm_frame_layout_t layout = {sizeof(InitializerFrame),
                                         alignof(InitializerFrame)};
  iree_vm_frame_t* frame = nullptr;
  IREE_RETURN_IF_ERROR(iree_vm_invocation_push_frame(
      params, layout, CleanupInitializerFrame, &frame));
  auto* payload = static_cast<InitializerFrame*>(iree_vm_frame_storage(frame));
  payload->module = module;
  const iree_vm_invocation_wake_callback_t wake_callback =
      iree_vm_invocation_wake_callback(params->execution.invocation);
  if (wake_callback.fn) wake_callback.fn(wake_callback.user_data);
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  return iree_ok_status();
}

iree_status_t ResumeTestFunction(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  TestModule* module = CastTestModule(base_module);
  module->log->Record(module->module_id, EventKind::kResume);
  iree_vm_invocation_pop_frame(params->execution.invocation, params->frame);
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

void QueryTestImportGroup(const iree_vm_module_t* module,
                          iree_host_size_t ordinal,
                          iree_vm_module_import_group_t* out_group) {
  (void)module;
  (void)ordinal;
  (void)out_group;
}

void QueryTestImport(const iree_vm_module_t* module, iree_host_size_t ordinal,
                     iree_vm_module_import_declaration_t* out_import) {
  (void)module;
  (void)ordinal;
  (void)out_import;
}

static const iree_vm_module_signature_type_t kInitializerArguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
};
static const iree_vm_module_callable_type_declaration_t kInitializerType = {
    {{kInitializerArguments, IREE_ARRAYSIZE(kInitializerArguments)},
     {nullptr, 0}},
    IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
    0,
    0,
};
static const iree_vm_module_export_declaration_t kInitializerExport = {
    IREE_SVL("initialize"), 0, 0, 0};

void QueryTestExport(const iree_vm_module_t* module, iree_host_size_t ordinal,
                     iree_vm_module_export_declaration_t* out_export) {
  (void)module;
  (void)ordinal;
  *out_export = kInitializerExport;
}

void QueryTestCallableType(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type) {
  (void)module;
  (void)ordinal;
  *out_callable_type = kInitializerType;
}

const iree_vm_module_vtable_t kTestModuleVtable = {
    sizeof(kTestModuleVtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    DestroyTestModule,
    StartTestFunction,
    ResumeTestFunction,
    AttachTestState,
    SealTestState,
    DetachTestState,
    QueryTestImportGroup,
    QueryTestImport,
    QueryTestExport,
    QueryTestCallableType,
    iree_vm_module_query_presentation_none,
    iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t InitializeTestModule(iree_string_view_t name, uint8_t module_id,
                                   iree_host_size_t process_storage_size,
                                   FailurePhase failure_phase,
                                   FunctionBehavior function_behavior,
                                   EventLog* log, TestModule* out_module) {
  *out_module = TestModule{};
  out_module->log = log;
  out_module->module_id = module_id;
  out_module->failure_phase = failure_phase;
  out_module->function_behavior = function_behavior;
  const bool has_initializer =
      function_behavior == FunctionBehavior::kYieldingInitializer;
  out_module->descriptor = iree_vm_module_descriptor_t{
      name,
      IREE_VM_MODULE_FLAG_LINKABLE,
      {nullptr, 0},
      {has_initializer ? 1u : 0u, has_initializer ? 1u : 0u, 0, 0,
       has_initializer ? 1u : 0u, 0},
      process_storage_size,
  };
  return iree_vm_module_initialize(&kTestModuleVtable, &out_module->descriptor,
                                   &out_module->base);
}

void ReleaseTestModules(std::initializer_list<TestModule*> modules) {
  for (TestModule* module : modules) iree_vm_module_release(&module->base);
}

void CountWake(void* user_data) { ++*static_cast<int*>(user_data); }

TEST(VMProcessTest, NullLifetimeOperationsAreNoOps) {
  iree_vm_process_retain(nullptr);
  iree_vm_process_release(nullptr);
}

TEST(VMProcessTest, AttachesAndSealsSortedModulesInOneSlab) {
  EventLog log;
  TestModule alpha;
  TestModule middle;
  TestModule zeta;
  IREE_ASSERT_OK(InitializeTestModule(IREE_SV("alpha"), 1, 4,
                                      FailurePhase::kNone,
                                      FunctionBehavior::kNone, &log, &alpha));
  IREE_ASSERT_OK(InitializeTestModule(IREE_SV("middle"), 2, 0,
                                      FailurePhase::kNone,
                                      FunctionBehavior::kNone, &log, &middle));
  IREE_ASSERT_OK(InitializeTestModule(IREE_SV("zeta"), 3, 8,
                                      FailurePhase::kNone,
                                      FunctionBehavior::kNone, &log, &zeta));
  iree_vm_module_t* libraries[] = {&zeta.base, &alpha.base};
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create(
      {&middle.base, iree_vm_module_span_from_array(libraries)},
      iree_allocator_system(), &program));

  alignas(iree_max_align_t) std::array<uint8_t, kInvocationStorageSize>
      storage = {};
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(storage.data(), storage.size()), &invocation));
  CountingAllocator allocator;
  iree_vm_process_create_outcome_t outcome = {};
  IREE_ASSERT_OK(iree_vm_process_create_start(program, invocation,
                                              iree_vm_variant_span_empty(), {},
                                              allocator.allocator(), &outcome));
  ASSERT_EQ(outcome.execution_outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  ASSERT_NE(outcome.process, nullptr);
  EXPECT_EQ(allocator.allocation_count(), 1u);
  EXPECT_EQ(allocator.free_count(), 0u);
  EXPECT_TRUE(alpha.saw_zeroed_storage);
  EXPECT_TRUE(middle.saw_zeroed_storage);
  EXPECT_TRUE(middle.saw_canonical_empty_storage);
  EXPECT_TRUE(zeta.saw_zeroed_storage);
  ExpectEvents(
      log, {MakeEvent(1, EventKind::kAttach), MakeEvent(2, EventKind::kAttach),
            MakeEvent(3, EventKind::kAttach), MakeEvent(1, EventKind::kSeal),
            MakeEvent(2, EventKind::kSeal), MakeEvent(3, EventKind::kSeal)});

  iree_vm_process_release(outcome.process);
  EXPECT_EQ(allocator.free_count(), 1u);
  ExpectEvents(
      log, {MakeEvent(1, EventKind::kAttach), MakeEvent(2, EventKind::kAttach),
            MakeEvent(3, EventKind::kAttach), MakeEvent(1, EventKind::kSeal),
            MakeEvent(2, EventKind::kSeal), MakeEvent(3, EventKind::kSeal),
            MakeEvent(3, EventKind::kDetach), MakeEvent(2, EventKind::kDetach),
            MakeEvent(1, EventKind::kDetach)});
  iree_vm_invocation_deinitialize(invocation);
  iree_vm_program_release(program);
  ReleaseTestModules({&alpha, &middle, &zeta});
  EXPECT_EQ(alpha.destroy_count, 1);
  EXPECT_EQ(middle.destroy_count, 1);
  EXPECT_EQ(zeta.destroy_count, 1);
}

TEST(VMProcessTest, DetachesOnlyCompletedAttachesAfterAttachFailure) {
  EventLog log;
  TestModule alpha;
  TestModule middle;
  TestModule zeta;
  IREE_ASSERT_OK(InitializeTestModule(IREE_SV("alpha"), 1, 4,
                                      FailurePhase::kNone,
                                      FunctionBehavior::kNone, &log, &alpha));
  IREE_ASSERT_OK(InitializeTestModule(IREE_SV("middle"), 2, 4,
                                      FailurePhase::kAttach,
                                      FunctionBehavior::kNone, &log, &middle));
  IREE_ASSERT_OK(InitializeTestModule(IREE_SV("zeta"), 3, 4,
                                      FailurePhase::kNone,
                                      FunctionBehavior::kNone, &log, &zeta));
  iree_vm_module_t* libraries[] = {&middle.base, &alpha.base};
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create(
      {&zeta.base, iree_vm_module_span_from_array(libraries)},
      iree_allocator_system(), &program));
  alignas(iree_max_align_t) std::array<uint8_t, kInvocationStorageSize>
      storage = {};
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(storage.data(), storage.size()), &invocation));
  CountingAllocator allocator;
  iree_vm_process_create_outcome_t outcome = {
      IREE_VM_EXECUTION_OUTCOME_SUSPENDED,
      reinterpret_cast<iree_vm_process_t*>(uintptr_t{1}),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_vm_process_create_start(
                            program, invocation, iree_vm_variant_span_empty(),
                            {}, allocator.allocator(), &outcome));
  EXPECT_EQ(outcome.execution_outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(outcome.process,
            reinterpret_cast<iree_vm_process_t*>(uintptr_t{1}));
  EXPECT_EQ(allocator.allocation_count(), 1u);
  EXPECT_EQ(allocator.free_count(), 1u);
  ExpectEvents(
      log, {MakeEvent(1, EventKind::kAttach), MakeEvent(2, EventKind::kAttach),
            MakeEvent(1, EventKind::kDetach)});

  iree_vm_invocation_deinitialize(invocation);
  iree_vm_program_release(program);
  ReleaseTestModules({&alpha, &middle, &zeta});
}

TEST(VMProcessTest, ReverseDetachesAllModulesAfterSealFailure) {
  EventLog log;
  TestModule alpha;
  TestModule middle;
  TestModule zeta;
  IREE_ASSERT_OK(InitializeTestModule(IREE_SV("alpha"), 1, 4,
                                      FailurePhase::kNone,
                                      FunctionBehavior::kNone, &log, &alpha));
  IREE_ASSERT_OK(InitializeTestModule(IREE_SV("middle"), 2, 4,
                                      FailurePhase::kSeal,
                                      FunctionBehavior::kNone, &log, &middle));
  IREE_ASSERT_OK(InitializeTestModule(IREE_SV("zeta"), 3, 4,
                                      FailurePhase::kNone,
                                      FunctionBehavior::kNone, &log, &zeta));
  iree_vm_module_t* libraries[] = {&middle.base, &alpha.base};
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create(
      {&zeta.base, iree_vm_module_span_from_array(libraries)},
      iree_allocator_system(), &program));
  alignas(iree_max_align_t) std::array<uint8_t, kInvocationStorageSize>
      storage = {};
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(storage.data(), storage.size()), &invocation));
  CountingAllocator allocator;
  iree_vm_process_create_outcome_t outcome = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_vm_process_create_start(
                            program, invocation, iree_vm_variant_span_empty(),
                            {}, allocator.allocator(), &outcome));
  EXPECT_EQ(allocator.allocation_count(), 1u);
  EXPECT_EQ(allocator.free_count(), 1u);
  ExpectEvents(
      log,
      {MakeEvent(1, EventKind::kAttach), MakeEvent(2, EventKind::kAttach),
       MakeEvent(3, EventKind::kAttach), MakeEvent(1, EventKind::kSeal),
       MakeEvent(2, EventKind::kSeal), MakeEvent(3, EventKind::kDetach),
       MakeEvent(2, EventKind::kDetach), MakeEvent(1, EventKind::kDetach)});

  iree_vm_invocation_deinitialize(invocation);
  iree_vm_program_release(program);
  ReleaseTestModules({&alpha, &middle, &zeta});
}

TEST(VMProcessTest, PublishesOnlyAfterYieldingInitializerCompletes) {
  EventLog log;
  TestModule module;
  IREE_ASSERT_OK(InitializeTestModule(
      IREE_SV("initializer.test"), 1, sizeof(uint32_t), FailurePhase::kNone,
      FunctionBehavior::kYieldingInitializer, &log, &module));
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(
      iree_vm_program_create({&module.base, iree_vm_module_span_empty()},
                             iree_allocator_system(), &program));
  alignas(iree_max_align_t) std::array<uint8_t, kInvocationStorageSize>
      storage = {};
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(storage.data(), storage.size()), &invocation));
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_i32(42),
  };
  int wake_count = 0;
  CountingAllocator allocator;
  iree_vm_process_create_outcome_t outcome = {};
  IREE_ASSERT_OK(iree_vm_process_create_start(
      program, invocation,
      iree_vm_variant_span_from_ptr(arguments.data(), arguments.size()),
      {CountWake, &wake_count}, allocator.allocator(), &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_EQ(outcome.execution_outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(outcome.process, nullptr);
  EXPECT_EQ(wake_count, 1);
  EXPECT_EQ(module.sealed_value, 0u);
  ExpectEvents(
      log, {MakeEvent(1, EventKind::kAttach), MakeEvent(1, EventKind::kStart)});

  IREE_ASSERT_OK(iree_vm_process_create_resume(invocation, &outcome));
  ASSERT_EQ(outcome.execution_outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  ASSERT_NE(outcome.process, nullptr);
  EXPECT_EQ(module.sealed_value, 42u);
  ExpectEvents(
      log, {MakeEvent(1, EventKind::kAttach), MakeEvent(1, EventKind::kStart),
            MakeEvent(1, EventKind::kResume), MakeEvent(1, EventKind::kCleanup),
            MakeEvent(1, EventKind::kSeal)});

  iree_vm_process_release(outcome.process);
  EXPECT_EQ(allocator.allocation_count(), 1u);
  EXPECT_EQ(allocator.free_count(), 1u);
  iree_vm_invocation_deinitialize(invocation);
  iree_vm_program_release(program);
  ReleaseTestModules({&module});
}

TEST(VMProcessTest, SynchronousCreatePreservesAnEarlyInitializerWake) {
  EventLog log;
  TestModule module;
  IREE_ASSERT_OK(InitializeTestModule(
      IREE_SV("initializer.test"), 1, sizeof(uint32_t), FailurePhase::kNone,
      FunctionBehavior::kYieldingInitializer, &log, &module));
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(
      iree_vm_program_create({&module.base, iree_vm_module_span_empty()},
                             iree_allocator_system(), &program));
  alignas(iree_max_align_t) std::array<uint8_t, kInvocationStorageSize>
      storage = {};
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(storage.data(), storage.size()), &invocation));
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_i32(42),
  };
  CountingAllocator allocator;
  iree_vm_process_t* process = nullptr;
  IREE_ASSERT_OK(iree_vm_process_create(
      program, invocation,
      iree_vm_variant_span_from_ptr(arguments.data(), arguments.size()),
      allocator.allocator(), &process));

  ASSERT_NE(process, nullptr);
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_EQ(module.sealed_value, 42u);
  EXPECT_EQ(allocator.allocation_count(), 1u);
  EXPECT_EQ(allocator.free_count(), 0u);
  ExpectEvents(
      log, {MakeEvent(1, EventKind::kAttach), MakeEvent(1, EventKind::kStart),
            MakeEvent(1, EventKind::kResume), MakeEvent(1, EventKind::kCleanup),
            MakeEvent(1, EventKind::kSeal)});

  iree_vm_process_release(process);
  EXPECT_EQ(allocator.free_count(), 1u);
  iree_vm_invocation_deinitialize(invocation);
  iree_vm_program_release(program);
  ReleaseTestModules({&module});
}

TEST(VMProcessTest, CancellationDiscardsUnpublishedInitializedProcess) {
  EventLog log;
  TestModule module;
  IREE_ASSERT_OK(InitializeTestModule(
      IREE_SV("initializer.test"), 1, sizeof(uint32_t), FailurePhase::kNone,
      FunctionBehavior::kYieldingInitializer, &log, &module));
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(
      iree_vm_program_create({&module.base, iree_vm_module_span_empty()},
                             iree_allocator_system(), &program));
  alignas(iree_max_align_t) std::array<uint8_t, kInvocationStorageSize>
      storage = {};
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(storage.data(), storage.size()), &invocation));
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_i32(42),
  };
  int wake_count = 0;
  CountingAllocator allocator;
  iree_vm_process_create_outcome_t outcome = {};
  IREE_ASSERT_OK(iree_vm_process_create_start(
      program, invocation,
      iree_vm_variant_span_from_ptr(arguments.data(), arguments.size()),
      {CountWake, &wake_count}, allocator.allocator(), &outcome));
  ASSERT_EQ(outcome.execution_outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  ASSERT_TRUE(iree_vm_invocation_request_cancel(
      invocation, IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED));
  EXPECT_EQ(wake_count, 2);

  outcome.process = reinterpret_cast<iree_vm_process_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_vm_process_create_resume(invocation, &outcome));
  EXPECT_EQ(outcome.execution_outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(outcome.process,
            reinterpret_cast<iree_vm_process_t*>(uintptr_t{1}));
  EXPECT_EQ(allocator.allocation_count(), 1u);
  EXPECT_EQ(allocator.free_count(), 1u);
  ExpectEvents(
      log, {MakeEvent(1, EventKind::kAttach), MakeEvent(1, EventKind::kStart),
            MakeEvent(1, EventKind::kResume), MakeEvent(1, EventKind::kCleanup),
            MakeEvent(1, EventKind::kSeal), MakeEvent(1, EventKind::kDetach)});

  iree_vm_invocation_deinitialize(invocation);
  iree_vm_program_release(program);
  ReleaseTestModules({&module});
}

}  // namespace
