// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/trace.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

struct CountingAllocatorState {
  // Allocator receiving all commands after lifetime observation.
  iree_allocator_t delegate = iree_allocator_system();
  // Module allocation whose release is observed.
  void* watched_allocation = nullptr;
  // True after the watched allocation is released.
  bool watched_allocation_freed = false;
};

static iree_status_t CountingAllocatorCtl(void* self,
                                          iree_allocator_command_t command,
                                          const void* params,
                                          void** inout_ptr) {
  auto* state = static_cast<CountingAllocatorState*>(self);
  void* old_ptr = *inout_ptr;
  iree_status_t status =
      state->delegate.ctl(state->delegate.self, command, params, inout_ptr);
  if (iree_status_is_ok(status) && command == IREE_ALLOCATOR_COMMAND_FREE &&
      old_ptr == state->watched_allocation) {
    state->watched_allocation_freed = true;
  }
  return status;
}

static iree_allocator_t CountingAllocator(CountingAllocatorState* state) {
  return (iree_allocator_t){
      /*.self=*/state,
      /*.ctl=*/CountingAllocatorCtl,
  };
}

static iree_status_t BuildCounterModule(loom_context_t* context,
                                        iree_arena_block_pool_t* block_pool,
                                        int64_t value,
                                        iree_allocator_t allocator,
                                        loom_module_t** out_module) {
  *out_module = nullptr;
  loom_module_t* module = nullptr;
  iree_status_t status =
      loom_module_allocate(context, IREE_SV("trace_snapshot"), block_pool,
                           nullptr, allocator, &module);
  if (iree_status_is_ok(status)) {
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_op_t* counter_op = nullptr;
    status = loom_test_counter_build(&builder, value,
                                     loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                     LOOM_LOCATION_UNKNOWN, &counter_op);
  }
  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else if (module != nullptr) {
    loom_module_free(module);
  }
  return status;
}

struct ProjectionState {
  // Context used to build the projected module.
  loom_context_t* context = nullptr;
  // Block pool used to build the projected module.
  iree_arena_block_pool_t* block_pool = nullptr;
  // Allocator used to make projection lifetime observable.
  CountingAllocatorState* allocator_state = nullptr;
  // Counter value distinguishing projected IR from source IR.
  int64_t counter_value = 0;
  // Number of times the projection callback ran.
  int invocation_count = 0;
};

static iree_status_t ProjectCounterModule(
    void* user_data, const loom_module_t* source_module,
    loom_module_t** out_projected_module) {
  (void)source_module;
  auto* state = static_cast<ProjectionState*>(user_data);
  ++state->invocation_count;
  IREE_RETURN_IF_ERROR(BuildCounterModule(
      state->context, state->block_pool, state->counter_value,
      CountingAllocator(state->allocator_state), out_projected_module));
  state->allocator_state->watched_allocation = *out_projected_module;
  return iree_ok_status();
}

static iree_status_t ProjectCounterModuleThenFail(
    void* user_data, const loom_module_t* source_module,
    loom_module_t** out_projected_module) {
  IREE_RETURN_IF_ERROR(
      ProjectCounterModule(user_data, source_module, out_projected_module));
  return iree_make_status(IREE_STATUS_ABORTED, "projection failed");
}

static iree_status_t DeclineProjection(void* user_data,
                                       const loom_module_t* source_module,
                                       loom_module_t** out_projected_module) {
  (void)source_module;
  auto* invocation_count = static_cast<int*>(user_data);
  ++*invocation_count;
  *out_projected_module = nullptr;
  return iree_ok_status();
}

struct ArtifactSinkState {
  // Stream returned for the per-event IR artifact.
  loom_output_stream_t stream = {};
  // Number of artifact open calls.
  int open_count = 0;
  // Number of artifact close calls.
  int close_count = 0;
};

static iree_status_t FailArtifactWrite(void* user_data,
                                       iree_string_view_t text) {
  (void)user_data;
  (void)text;
  return iree_make_status(IREE_STATUS_DATA_LOSS, "artifact write failed");
}

static iree_status_t OpenArtifact(void* user_data,
                                  const loom_pass_trace_event_t* event,
                                  iree_host_size_t event_ordinal,
                                  loom_pass_trace_artifact_t* out_artifact) {
  (void)event;
  (void)event_ordinal;
  auto* state = static_cast<ArtifactSinkState*>(user_data);
  ++state->open_count;
  *out_artifact = (loom_pass_trace_artifact_t){
      /*.stream=*/&state->stream,
      /*.path=*/IREE_SV("ir/000000.loom"),
  };
  return iree_ok_status();
}

static iree_status_t CloseArtifact(void* user_data,
                                   loom_pass_trace_artifact_t* artifact) {
  (void)artifact;
  auto* state = static_cast<ArtifactSinkState*>(user_data);
  ++state->close_count;
  return iree_ok_status();
}

class TraceOutput {
 public:
  TraceOutput() {
    iree_string_builder_initialize(iree_allocator_system(), &builder_);
    loom_output_stream_for_builder(&builder_, &stream_);
    loom_pass_trace_options_initialize(&options);
    options.stream = &stream_;
  }

  ~TraceOutput() { iree_string_builder_deinitialize(&builder_); }

  void BindProjector(iree_status_t (*project)(void*, const loom_module_t*,
                                              loom_module_t**),
                     void* user_data) {
    loom_pass_trace_initialize(&options, &trace);
    loom_pass_trace_bind_snapshot_projector(
        &trace, (loom_pass_trace_snapshot_projector_t){
                    /*.project=*/project,
                    /*.user_data=*/user_data,
                });
  }

  std::string text() const {
    iree_string_view_t view = iree_string_builder_view(&builder_);
    return std::string(view.data, view.size);
  }

  bool empty() const {
    return iree_string_view_is_empty(iree_string_builder_view(&builder_));
  }

  // Mutable formatting and filtering options initialized by the constructor.
  loom_pass_trace_options_t options = {};
  // Trace state initialized when |BindProjector| is called.
  loom_pass_trace_t trace = {};

 private:
  // Builder retaining the formatted trace for assertions.
  iree_string_builder_t builder_ = {};
  // Output stream appending to |builder_|.
  loom_output_stream_t stream_ = {};
};

class PassTraceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr BuildSourceModule(int64_t counter_value) {
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(BuildCounterModule(&context_, &block_pool_, counter_value,
                                      iree_allocator_system(), &module));
    return ModulePtr(module);
  }

  ProjectionState MakeProjectionState(CountingAllocatorState* allocator_state,
                                      int64_t counter_value = 2) {
    return (ProjectionState){
        /*.context=*/&context_,
        /*.block_pool=*/&block_pool_,
        /*.allocator_state=*/allocator_state,
        /*.counter_value=*/counter_value,
    };
  }

  static loom_pass_trace_event_t MakeAfterEvent(const loom_module_t* module) {
    static const loom_pass_program_instruction_t instruction = {
        /*.kind=*/LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE,
        /*.anchor_kind=*/LOOM_PASS_MODULE,
    };
    return (loom_pass_trace_event_t){
        /*.module=*/module,
        /*.instruction=*/&instruction,
        /*.instruction_index=*/0,
        /*.invocation_ordinal=*/0,
        /*.pipeline_symbol=*/IREE_SV("@pipeline"),
        /*.symbol_name=*/IREE_SV("<none>"),
        /*.anchor_kind=*/LOOM_PASS_MODULE,
        /*.point=*/LOOM_PASS_TRACE_POINT_AFTER,
        /*.changed=*/false,
        /*.status_code=*/IREE_STATUS_OK,
        /*.error_count=*/0,
        /*.warning_count=*/0,
        /*.remark_count=*/0,
    };
  }

  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
};

TEST(PassTraceOptionsTest, PrefersTargetLowAssembly) {
  loom_pass_trace_options_t options = {};
  loom_pass_trace_options_initialize(&options);
  EXPECT_TRUE(iree_any_bit_set(options.print_options.flags,
                               LOOM_TEXT_PRINT_PREFER_LOW_ASM));
}

TEST_F(PassTraceTest, ProjectsMatchedSnapshotAndReleasesIt) {
  ModulePtr source_module = BuildSourceModule(1);
  ASSERT_NE(source_module, nullptr);

  CountingAllocatorState allocator_state;
  ProjectionState projection_state = MakeProjectionState(&allocator_state);
  TraceOutput output;
  output.options.dump_after_all = true;
  output.BindProjector(ProjectCounterModule, &projection_state);

  loom_pass_trace_event_t event = MakeAfterEvent(source_module.get());
  IREE_ASSERT_OK(loom_pass_trace_emit(&output.trace, &event));

  EXPECT_EQ(projection_state.invocation_count, 1);
  EXPECT_TRUE(allocator_state.watched_allocation_freed);
  EXPECT_EQ(output.trace.next_event_ordinal, 1u);
  std::string output_string = output.text();
  EXPECT_NE(output_string.find("test.counter 2"), std::string::npos);
  EXPECT_EQ(output_string.find("test.counter 1"), std::string::npos);
}

TEST_F(PassTraceTest, SkipsProjectionForUnmatchedEvent) {
  ModulePtr source_module = BuildSourceModule(1);
  ASSERT_NE(source_module, nullptr);

  CountingAllocatorState allocator_state;
  ProjectionState projection_state = MakeProjectionState(&allocator_state);
  TraceOutput output;
  output.options.dump_before_all = true;
  output.BindProjector(ProjectCounterModule, &projection_state);

  loom_pass_trace_event_t event = MakeAfterEvent(source_module.get());
  IREE_ASSERT_OK(loom_pass_trace_emit(&output.trace, &event));

  EXPECT_EQ(projection_state.invocation_count, 0);
  EXPECT_FALSE(allocator_state.watched_allocation_freed);
  EXPECT_EQ(output.trace.next_event_ordinal, 0u);
  EXPECT_TRUE(output.empty());
}

TEST_F(PassTraceTest, RejectsInvalidConfigurationBeforeProjection) {
  ModulePtr source_module = BuildSourceModule(1);
  ASSERT_NE(source_module, nullptr);

  int invocation_count = 0;
  ArtifactSinkState artifact_state;
  TraceOutput output;
  output.options.artifact_sink = {
      /*.open=*/OpenArtifact,
      /*.close=*/nullptr,
      /*.user_data=*/&artifact_state,
  };
  output.options.dump_after_all = true;
  output.BindProjector(DeclineProjection, &invocation_count);

  loom_pass_trace_event_t event = MakeAfterEvent(source_module.get());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_pass_trace_emit(&output.trace, &event));

  EXPECT_EQ(invocation_count, 0);
  EXPECT_EQ(artifact_state.open_count, 0);
  EXPECT_EQ(output.trace.next_event_ordinal, 0u);
  EXPECT_TRUE(output.empty());
}

TEST_F(PassTraceTest, ProjectorMayDeclineWithoutCloning) {
  ModulePtr source_module = BuildSourceModule(1);
  ASSERT_NE(source_module, nullptr);

  int invocation_count = 0;
  TraceOutput output;
  output.options.dump_after_all = true;
  output.BindProjector(DeclineProjection, &invocation_count);

  loom_pass_trace_event_t event = MakeAfterEvent(source_module.get());
  IREE_ASSERT_OK(loom_pass_trace_emit(&output.trace, &event));

  EXPECT_EQ(invocation_count, 1);
  std::string output_string = output.text();
  EXPECT_NE(output_string.find("test.counter 1"), std::string::npos);
}

TEST_F(PassTraceTest, ProjectionFailurePublishesNothingAndReleasesModule) {
  ModulePtr source_module = BuildSourceModule(1);
  ASSERT_NE(source_module, nullptr);

  CountingAllocatorState allocator_state;
  ProjectionState projection_state = MakeProjectionState(&allocator_state);
  TraceOutput output;
  output.options.dump_after_all = true;
  output.BindProjector(ProjectCounterModuleThenFail, &projection_state);

  loom_pass_trace_event_t event = MakeAfterEvent(source_module.get());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_pass_trace_emit(&output.trace, &event));

  EXPECT_EQ(projection_state.invocation_count, 1);
  EXPECT_TRUE(allocator_state.watched_allocation_freed);
  EXPECT_EQ(output.trace.next_event_ordinal, 0u);
  EXPECT_TRUE(output.empty());
}

TEST_F(PassTraceTest, ClosesArtifactAndReleasesProjectionAfterWriteFailure) {
  ModulePtr source_module = BuildSourceModule(1);
  ASSERT_NE(source_module, nullptr);

  CountingAllocatorState allocator_state;
  ProjectionState projection_state = MakeProjectionState(&allocator_state);
  ArtifactSinkState artifact_state = {
      /*.stream=*/{
          /*.write=*/FailArtifactWrite,
          /*.user_data=*/nullptr,
      },
  };
  TraceOutput output;
  output.options.artifact_sink = {
      /*.open=*/OpenArtifact,
      /*.close=*/CloseArtifact,
      /*.user_data=*/&artifact_state,
  };
  output.options.format = LOOM_PASS_TRACE_FORMAT_JSONL;
  output.options.dump_after_all = true;
  output.BindProjector(ProjectCounterModule, &projection_state);

  loom_pass_trace_event_t event = MakeAfterEvent(source_module.get());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        loom_pass_trace_emit(&output.trace, &event));

  EXPECT_EQ(artifact_state.open_count, 1);
  EXPECT_EQ(artifact_state.close_count, 1);
  EXPECT_TRUE(allocator_state.watched_allocation_freed);
  EXPECT_TRUE(output.empty());
}

}  // namespace
}  // namespace loom
