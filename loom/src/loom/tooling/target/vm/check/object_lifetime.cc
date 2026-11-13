// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/vm/check/object_lifetime.h"

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "iree/vm/bytecode/module.h"
#include "iree/vm/module_test_util.h"
#include "iree/vm/sync.h"
#include "loom/tools/loom-check/compile.h"
#include "loomc/iree.h"

namespace {

struct Event {
  // Native operation or lifecycle boundary, with static storage duration.
  const char* name;
  // Object identity, or live-object count for a lifecycle boundary.
  int32_t value;
};

struct Trace {
  // Ordered native observations, independent of compiler allocation choices.
  std::vector<Event> events;
  // Objects created but not yet destroyed by their actual final-release
  // callback.
  int32_t live_count = 0;
};

struct Object {
  // Intrusive VM prefix required by the native object contract.
  iree_vm_ref_object_t ref_object;
  // Observer that outlives the invocation, process, and provider.
  Trace* trace;
  // Payload read through the object, exposing stale access to ASAN.
  int32_t identity;
};

void DestroyObject(void* pointer) {
  auto* object = static_cast<Object*>(pointer);
  object->trace->events.push_back({"destroy", object->identity});
  --object->trace->live_count;
  delete object;
}

extern const iree_vm_ref_type_table_t kObjectTypes;
const iree_vm_ref_type_descriptor_t kObjectType = {DestroyObject, &kObjectTypes,
                                                   IREE_SV("object")};
const iree_vm_ref_type_t kRefTypes[] = {&kObjectType};
const iree_vm_ref_type_table_t kObjectTypes = {
    sizeof(kObjectTypes),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SV("lifetime"),
    {kRefTypes, IREE_ARRAYSIZE(kRefTypes)}};

using NativeCall = iree_vm_module_function_start_params_t;

iree_status_t ObserveObject(Trace* trace, const char* event,
                            iree_vm_ref_t value) {
  void* pointer = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_vm_ptr_from_ref_borrowed(value, &kObjectType, &pointer));
  auto* object = static_cast<Object*>(pointer);
  trace->events.push_back({event, object ? object->identity : -1});
  return iree_ok_status();
}

iree_status_t Consume(Trace* trace, const NativeCall* params) {
  iree_vm_ref_t owner = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_move(&params->call, 0, &owner);
  iree_status_t status = ObserveObject(trace, "consume", owner);
  iree_vm_ref_reset(&owner);
  return status;
}

iree_status_t ConsumeRead(Trace* trace, const NativeCall* params) {
  IREE_RETURN_IF_ERROR(Consume(trace, params));
  iree_vm_ref_t pointer = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(&params->call, 1, &pointer);
  return ObserveObject(trace, "read", pointer);
}

iree_vm_ref_t CreateObject(Trace* trace, int32_t identity) {
  auto* object = new Object;
  iree_vm_ref_object_initialize(&object->ref_object);
  object->trace = trace;
  object->identity = identity;
  ++trace->live_count;
  trace->events.push_back({"create", object->identity});
  void* pointer = object;
  return iree_vm_ref_from_ptr_move(&pointer, &kObjectType);
}

iree_status_t Create(Trace* trace, const NativeCall* params) {
  iree_vm_ref_t owner = CreateObject(
      trace,
      static_cast<int32_t>(iree_vm_call_value_argument_load(&params->call, 0)));
  iree_vm_ref_t borrow =
      iree_vm_ref_from_ptr_borrowed(owner.object, &kObjectType);
  // Publish the borrowed result while the native owner is still live. The
  // actual call boundary acquires its guard; the fixture keeps no extra owner.
  iree_vm_call_ref_result_store_move(&params->call, 1, &borrow);
  iree_vm_call_ref_result_store_move(&params->call, 0, &owner);
  return iree_ok_status();
}

iree_status_t CreateFail(Trace* trace, const NativeCall* params) {
  iree_vm_ref_t owner = CreateObject(
      trace,
      static_cast<int32_t>(iree_vm_call_value_argument_load(&params->call, 0)));
  iree_vm_call_ref_result_store_move(&params->call, 0, &owner);
  trace->events.push_back({"fail", trace->live_count});
  // The second result is intentionally unpublished when the provider fails.
  return iree_make_status(IREE_STATUS_ABORTED, "partial native result failure");
}

iree_status_t CreateNull(Trace*, const NativeCall* params) {
  iree_vm_ref_t value = iree_vm_ref_null();
  iree_vm_call_ref_result_store_move(&params->call, 0, &value);
  iree_vm_call_ref_result_store_move(&params->call, 1, &value);
  return iree_ok_status();
}

iree_status_t Fail(Trace* trace, const NativeCall*) {
  trace->events.push_back({"fail", trace->live_count});
  return iree_make_status(IREE_STATUS_ABORTED, "native lifetime test failure");
}

iree_status_t Observe(Trace* trace, const NativeCall*) {
  trace->events.push_back({"live", trace->live_count});
  return iree_ok_status();
}

iree_status_t RecordEvent(Trace* trace, const NativeCall* params) {
  trace->events.push_back(
      {"event", static_cast<int32_t>(
                    iree_vm_call_value_argument_load(&params->call, 0))});
  return iree_ok_status();
}

iree_status_t Read(Trace* trace, const NativeCall* params) {
  iree_vm_ref_t borrow = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(&params->call, 0, &borrow);
  return ObserveObject(trace, "read", borrow);
}

iree_status_t ReadPair(Trace* trace, const NativeCall* params) {
  iree_vm_ref_t left = iree_vm_ref_null();
  iree_vm_ref_t right = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(&params->call, 0, &left);
  iree_vm_call_ref_argument_load_borrow(&params->call, 1, &right);
  IREE_RETURN_IF_ERROR(ObserveObject(trace, "read", left));
  return ObserveObject(trace, "read", right);
}

iree_status_t Count(Trace* trace, const NativeCall* params) {
  iree_vm_ref_t borrow = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(&params->call, 0, &borrow);
  void* pointer = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_vm_ptr_from_ref_borrowed(borrow, &kObjectType, &pointer));
  auto* object = static_cast<Object*>(pointer);
  trace->events.push_back(
      {"references",
       object ? iree_atomic_ref_count_load(&object->ref_object.ref_count) : 0});
  return iree_ok_status();
}

const iree_vm_module_signature_type_t kOwnedFields[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
const iree_vm_module_signature_type_t kBorrowedFields[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 0}};
const iree_vm_module_signature_type_t kPairFields[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 0}};
const iree_vm_module_signature_type_t kFactoryFields[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 0}};
const iree_vm_module_signature_type_t kIdentityFields[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}};
const iree_vm_module_callable_type_declaration_t kCallableTypes[] = {
    {{{}, {}}, 0, 0, 0},
    {{{}, {kFactoryFields, 2, 0, 2, 0}}, 0, 0, 0},
    {{{kIdentityFields, 1, 1, 0, 0}, {}}, 0, 0, 0},
    {{{kIdentityFields, 1, 1, 0, 0}, {kFactoryFields, 2, 0, 2, 0}}, 0, 0, 0},
    {{{kOwnedFields, 1, 0, 1, 0}, {}}, 0, 0, 0},
    {{{kBorrowedFields, 1, 0, 1, 0}, {}}, 0, 0, 0},
    {{{kFactoryFields, 2, 0, 2, 0}, {}}, 0, 0, 0},
    {{{kPairFields, 2, 0, 2, 0}, {}}, 0, 0, 0}};
const iree_vm_module_export_declaration_t kExports[] = {
    {IREE_SV("consume"), 4, 0, 0},     {IREE_SV("consume_read"), 6, 9, 0},
    {IREE_SV("count"), 5, 8, 0},       {IREE_SV("create"), 3, 1, 0},
    {IREE_SV("create_fail"), 3, 2, 0}, {IREE_SV("create_null"), 1, 3, 0},
    {IREE_SV("event"), 2, 10, 0},      {IREE_SV("fail"), 0, 4, 0},
    {IREE_SV("observe"), 0, 5, 0},     {IREE_SV("read"), 5, 6, 0},
    {IREE_SV("read_pair"), 7, 7, 0}};
using NativeFunction = iree_status_t (*)(Trace*, const NativeCall*);
constexpr NativeFunction kFunctions[] = {
    Consume, Create,   CreateFail, CreateNull,  Fail,       Observe,
    Read,    ReadPair, Count,      ConsumeRead, RecordEvent};

struct ObjectModule {
  // TableModule and its contained VM module both occupy offset zero.
  iree::vm::testing::TableModule table;
  // Observer borrowed for the complete provider lifetime.
  Trace* trace;

  static iree_status_t Start(iree_vm_module_t* module, const NativeCall* params,
                             iree_vm_execution_outcome_t* outcome) {
    auto* self = reinterpret_cast<ObjectModule*>(module);
    IREE_RETURN_IF_ERROR(
        kFunctions[params->function_ordinal](self->trace, params));
    *outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
    return iree_ok_status();
  }
};
static_assert(std::is_standard_layout_v<ObjectModule>);
static_assert(offsetof(ObjectModule, table) == 0);

// Owns runtime handles across partial setup, terminal failure, and normal exit.
struct Execution {
  // Construction registry borrowing the static native type table.
  iree_vm_environment_t* environment = nullptr;
  // Bytecode module borrowing the compiler artifact until execution completes.
  iree_vm_module_t* module = nullptr;
  // Linked executable and native module ownership.
  iree_vm_program_t* program = nullptr;
  // Process state, released before the invocation storage is retired.
  iree_vm_process_t* process = nullptr;
  // Initialized invocation, or null when initialization did not succeed.
  iree_vm_invocation_t* invocation = nullptr;
  // Host-provided storage for real VM frames, including overflow packets.
  alignas(iree_max_align_t) std::array<uint8_t, 64 * 1024> storage = {};

  ~Execution() {
    iree_vm_process_release(process);
    if (invocation) {
      iree_vm_invocation_deinitialize(invocation);
    }
    iree_vm_program_release(program);
    iree_vm_module_release(module);
    iree_vm_environment_free(environment);
  }

  iree_status_t Initialize(iree_const_byte_span_t contents,
                           iree_vm_module_t* provider,
                           iree_allocator_t allocator) {
    IREE_RETURN_IF_ERROR(iree_vm_environment_allocate(allocator, &environment));
    IREE_RETURN_IF_ERROR(iree_vm_environment_register_ref_type_table(
        environment, &kObjectTypes));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
        environment, IREE_SV("client"), {contents, iree_allocator_null()},
        allocator, &module));
    iree_vm_module_t* libraries[] = {provider};
    IREE_RETURN_IF_ERROR(iree_vm_program_create(
        {module, iree_vm_module_span_from_array(libraries)}, allocator,
        &program));
    IREE_RETURN_IF_ERROR(iree_vm_invocation_initialize(
        iree_make_byte_span(storage.data(), storage.size()), &invocation));
    return iree_vm_process_create(
        program, invocation, iree_vm_variant_span_empty(), allocator, &process);
  }
};

iree_status_t Execute(iree_const_byte_span_t contents, bool borrowed_argument,
                      iree_allocator_t allocator,
                      iree_string_builder_t* output) {
  Trace trace;
  {
    ObjectModule provider;
    provider.trace = &trace;
    provider.table.vtable().function_start = ObjectModule::Start;
    IREE_RETURN_IF_ERROR(provider.table.Publish(
        iree::vm::testing::TableModuleDefinition(IREE_SV("lifetime"),
                                                 IREE_ARRAYSIZE(kFunctions))
            .WithRefTypes(kRefTypes)
            .WithCallableTypes(kCallableTypes)
            .WithExports(kExports)));
    Execution execution;
    IREE_RETURN_IF_ERROR(
        execution.Initialize(contents, provider.table.module(), allocator));
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
        execution.process, IREE_SV("client"), IREE_SV("run"), &function));
    // The host owner covers the complete synchronous invocation, as it would
    // when passing a native frame or tokenizer to compiled methods.
    iree_vm_ref_t owner =
        borrowed_argument ? CreateObject(&trace, 1) : iree_vm_ref_null();
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_ref_borrowed(owner)};
    iree_status_t status = iree_vm_invoke(
        execution.invocation, function,
        borrowed_argument ? iree_vm_variant_span_from_array(arguments)
                          : iree_vm_variant_span_empty(),
        iree_vm_variant_span_empty());
    trace.events.push_back(
        {iree_status_code_string(iree_status_code(status)), trace.live_count});
    iree_status_ignore(status);
    iree_vm_ref_reset(&owner);
  }
  trace.events.push_back({"retired", trace.live_count});
  iree_status_t status = iree_ok_status();
  for (size_t i = 0; i < trace.events.size() && iree_status_is_ok(status);
       ++i) {
    const Event& event = trace.events[i];
    status = iree_string_builder_append_format(output, "%s %d\n", event.name,
                                               event.value);
  }
  return status;
}

bool Match(const loom_check_emit_provider_t*, iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV("vm-object-lifetime"));
}

iree_status_t Emit(const loom_check_emit_provider_t*,
                   const loom_check_emit_provider_request_t* request) {
  const iree_string_view_t target_options =
      iree_string_view_trim(request->target_options);
  const bool borrowed_argument =
      iree_string_view_equal(target_options, IREE_SV("borrowed-argument"));
  if (!iree_string_view_is_empty(target_options) && !borrowed_argument) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "vm-object-lifetime runs @run with optional borrowed-argument");
  }
  loomc_target_profile_t* target_profile = nullptr;
  IREE_RETURN_IF_ERROR(loom_check_compile_session_select_target_profile(
      request->environment->compile_session, IREE_SV("vm:core"),
      &target_profile));
  const loom_check_compile_artifact_options_t options = {
      .artifact_format = IREE_SV("vm"),
      .root = IREE_SV("run"),
      .target_profile = target_profile,
      .lower_source_to_low = true,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
  };
  loomc_result_t* result = nullptr;
  iree_status_t status =
      loom_check_compile_artifact(request, &options, &result);
  loomc_target_profile_release(target_profile);
  const loomc_artifact_t* artifact = nullptr;
  if (iree_status_is_ok(status) && result) {
    status = loom_check_compile_select_artifact(
        result, LOOMC_ARTIFACT_KIND_EXECUTABLE, options.artifact_format,
        &artifact);
  }
  loomc_source_t* source = nullptr;
  if (iree_status_is_ok(status) && artifact) {
    status = iree_status_from_loomc(loomc_artifact_create_source(
        artifact, LOOMC_SOURCE_FORMAT_UNKNOWN,
        loomc_allocator_from_iree(request->host_allocator), &source));
  }
  loomc_result_release(result);
  if (iree_status_is_ok(status) && source) {
    const loomc_byte_span_t contents = loomc_source_contents(source);
    status =
        Execute(iree_make_const_byte_span(contents.data, contents.data_length),
                borrowed_argument, request->host_allocator,
                &request->result->actual_output);
  }
  loomc_source_release(source);
  return status;
}

iree_status_t AppendNames(const loom_check_emit_provider_t*,
                          iree_string_builder_t* output) {
  return iree_string_builder_append_cstring(output, "vm-object-lifetime");
}

const loom_check_emit_provider_t kEmitProvider = {
    .name = IREE_SVL("vm-object-lifetime"),
    .match = Match,
    .execute = Emit,
    .append_names = AppendNames,
};
const loom_check_emit_provider_t* const kEmitProviders[] = {&kEmitProvider};

}  // namespace

const loom_check_provider_t loom_vm_object_lifetime_check_provider = {
    .name = IREE_SVL("vm-object-lifetime"),
    .emit_providers = kEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kEmitProviders),
};
