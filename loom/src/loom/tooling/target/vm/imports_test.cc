// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/location.h"
#include "loom/ops/func/location_capture.h"
#include "loom/ops/func/location_capture_test_data.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/tooling/input/input.h"
#include "loom/tooling/target/vm/imports_bytecode.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"
#include "loomc/target/vm.h"

namespace {

constexpr iree_vm_module_signature_type_t kStep[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}};
constexpr iree_vm_module_signature_type_t kFail[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
constexpr iree_vm_module_signature_type_t kMixed[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
constexpr auto kWide = [] {
  std::array<iree_vm_module_signature_type_t, 34> types = {};
  for (int i = 0; i < 17; ++i) {
    types[i] = {IREE_VM_SCALAR_TYPE_I64, 0};
    types[17 + i] = {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0};
  }
  return types;
}();

const iree_vm_module_callable_type_declaration_t kCallables[] = {
    {{{kStep, 1, 1, 0, 0}, {kStep, 1, 1, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
    {{{kFail, 2, 0, 2, 0}, {nullptr, 0, 0, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
    {{{kMixed, 3, 1, 2, 0}, {kMixed, 3, 1, 2, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
    {{{kWide.data(), 34, 17, 17, 0}, {kWide.data(), 34, 17, 17, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
};
const iree_vm_module_export_declaration_t kExports[] = {
    {IREE_SVL("fail"), 1, 0, 0},
    {IREE_SVL("mixed"), 2, 1, 0},
    {IREE_SVL("step"), 0, 2, 0},
    {IREE_SVL("wide"), 3, 3, 0},
};

iree_status_t Start(iree_vm_module_t*,
                    const iree_vm_module_function_start_params_t* params,
                    iree_vm_execution_outcome_t* outcome) {
  if (params->function_ordinal == 0) {
    return iree_make_status(IREE_STATUS_ABORTED, "native failure");
  }
  const uint16_t value_count = params->function_ordinal == 3 ? 17 : 1;
  const uint16_t ref_count = params->function_ordinal == 3   ? 17
                             : params->function_ordinal == 1 ? 2
                                                             : 0;
  // Result banks are disjoint from argument banks, including overflow slots.
  for (uint16_t i = 0; i < value_count; ++i) {
    iree_vm_call_value_result_store(
        &params->call, i,
        iree_vm_call_value_argument_load(&params->call, i) + i + 1);
  }
  for (uint16_t i = 0; i < ref_count; ++i) {
    iree_vm_ref_t ref = iree_vm_ref_null();
    iree_vm_call_ref_argument_load_move(&params->call, i, &ref);
    iree_vm_call_ref_result_store_move(&params->call, i, &ref);
  }
  *outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

void QueryExport(const iree_vm_module_t*, iree_host_size_t ordinal,
                 iree_vm_module_export_declaration_t* value) {
  *value = kExports[ordinal];
}
void QueryCallable(const iree_vm_module_t*, iree_host_size_t ordinal,
                   iree_vm_module_callable_type_declaration_t* value) {
  *value = kCallables[ordinal];
}
void QueryImportGroup(const iree_vm_module_t*, iree_host_size_t,
                      iree_vm_module_import_group_t*) {
  IREE_CHECK_UNREACHABLE("native module has no imports");
}
void QueryImport(const iree_vm_module_t*, iree_host_size_t,
                 iree_vm_module_import_declaration_t*) {
  IREE_CHECK_UNREACHABLE("native module has no imports");
}
void DestroyModule(iree_vm_module_t*) {}
const iree_vm_module_vtable_t kVtable = {
    sizeof(kVtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    DestroyModule,
    Start,
    iree_vm_module_function_resume_unreachable,
    nullptr,
    nullptr,
    nullptr,
    QueryImportGroup,
    QueryImport,
    QueryExport,
    QueryCallable,
    iree_vm_module_query_presentation_none,
    iree_vm_module_metadata_by_ordinal_none};

void CountRelease(void* user_data, iree_byte_span_t) {
  ++*static_cast<int*>(user_data);
}

class VMImportsTest : public ::testing::Test {
 protected:
  virtual void CreateImage(std::vector<uint8_t>* out_image) {
    const auto* source = loom_vm_imports_bytecode_create();
    out_image->assign(source[0].data, source[0].data + source[0].size);
  }

  void SetUp() override {
    std::vector<uint8_t> contents;
    CreateImage(&contents);
    ASSERT_FALSE(contents.empty());
    iree_vm_environment_t* environment = nullptr;
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types_));
    uint8_t* image = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         contents.size(),
                                         reinterpret_cast<void**>(&image)));
    std::memcpy(image, contents.data(), contents.size());
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment, IREE_SV("compiled"),
        {iree_make_const_byte_span(image, contents.size()),
         iree_allocator_system()},
        iree_allocator_system(), &bytecode_));
    iree_vm_environment_free(environment);

    descriptor_ = {IREE_SVL("native"),
                   IREE_VM_MODULE_FLAG_LINKABLE,
                   {&types_.buffer, 1},
                   {4, 4, 0, 0, 4, 0, {38, 40, 0}},
                   0};
    IREE_ASSERT_OK(iree_vm_module_initialize(&kVtable, &descriptor_, &native_));
    iree_vm_module_t* libraries[] = {&native_};
    IREE_ASSERT_OK(iree_vm_program_create(
        {bytecode_, iree_vm_module_span_from_array(libraries)},
        iree_allocator_system(), &program_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(storage_.data(), storage_.size()), &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_,
                                          iree_vm_variant_span_empty(),
                                          iree_allocator_system(), &process_));
  }

  void TearDown() override {
    iree_vm_process_release(process_);
    iree_vm_invocation_deinitialize(invocation_);
    iree_vm_program_release(program_);
    iree_vm_module_release(bytecode_);
    iree_vm_module_release(&native_);
  }

  iree_status_t Invoke(iree_string_view_t name,
                       iree_vm_variant_span_t arguments,
                       iree_vm_variant_span_t results) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
        process_, IREE_SV("compiled"), name, &function));
    return iree_vm_invoke(invocation_, function, arguments, results);
  }

  iree_vm_variant_t WrapBuffer(int* release_count) {
    iree_vm_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span(bytes_.data(), bytes_.size()),
        {CountRelease, release_count}, iree_allocator_system(), &buffer));
    iree_vm_buffer_t* alias = nullptr;
    IREE_CHECK_OK(iree_vm_buffer_subspan(buffer, 2, 4,
                                         IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                         iree_allocator_system(), &alias));
    iree_vm_buffer_release(buffer);
    return iree_vm_buffer_variant_from_ptr_move(&types_, &alias);
  }

  void ExpectAliases(iree_vm_variant_t left, iree_vm_variant_t right) {
    void* lhs = nullptr;
    void* rhs = nullptr;
    IREE_ASSERT_OK(
        iree_vm_ptr_from_variant_borrowed(left, types_.buffer, &lhs));
    IREE_ASSERT_OK(
        iree_vm_ptr_from_variant_borrowed(right, types_.buffer, &rhs));
    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(iree_vm_buffer_length(static_cast<iree_vm_buffer_t*>(lhs)), 4u);
  }

  // Resolved core reference types shared by both modules.
  iree_vm_ref_types_t types_ = {};
  // Native provider storage borrowed throughout the fixture.
  iree_vm_module_t native_ = {};
  // Immutable native provider description.
  iree_vm_module_descriptor_t descriptor_ = {};
  // Bytecode compiled from the authored fixture by loom-compile.
  iree_vm_module_t* bytecode_ = nullptr;
  // Linked native and compiled modules.
  iree_vm_program_t* program_ = nullptr;
  // Independent execution state for the linked program.
  iree_vm_process_t* process_ = nullptr;
  // Host-owned invocation storage.
  alignas(iree_max_align_t) std::array<uint8_t, 16384> storage_ = {};
  // Reusable invocation borrowing storage_.
  iree_vm_invocation_t* invocation_ = nullptr;
  // Host buffer backing kept alive until every returned alias is released.
  std::array<uint8_t, 8> bytes_ = {};
};

TEST_F(VMImportsTest, NativeCallsExecuteInsideRuntimeLoop) {
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(4)};
  iree_vm_variant_t result = {};
  IREE_ASSERT_OK(Invoke(IREE_SV("loop"),
                        iree_vm_variant_span_from_array(arguments),
                        {&result, 1}));
  int32_t value = 0;
  IREE_ASSERT_OK(iree_vm_i32_from_variant(result, &value));
  EXPECT_EQ(value, 10);
}

TEST_F(VMImportsTest, MixedCallPreservesLiveValuesAndAliasedReferences) {
  int release_count = 0;
  iree_vm_variant_t arguments[] = {WrapBuffer(&release_count),
                                   iree_vm_variant_from_i64(42)};
  iree_vm_variant_t results[3] = {};
  IREE_ASSERT_OK(Invoke(IREE_SV("aliases"),
                        iree_vm_variant_span_from_array(arguments),
                        iree_vm_variant_span_from_array(results)));
  int64_t value = 0;
  IREE_ASSERT_OK(iree_vm_i64_from_variant(results[1], &value));
  EXPECT_EQ(value, 85);
  ExpectAliases(results[0], results[2]);
  EXPECT_EQ(release_count, 0);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  EXPECT_EQ(release_count, 1);
}

TEST_F(VMImportsTest, OverflowArgumentsAndResultsUseBothBanks) {
  int release_count = 0;
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i64(42),
                                   WrapBuffer(&release_count)};
  iree_vm_variant_t results[4] = {};
  IREE_ASSERT_OK(Invoke(IREE_SV("overflow"),
                        iree_vm_variant_span_from_array(arguments),
                        iree_vm_variant_span_from_array(results)));
  int64_t first = 0;
  int64_t last = 0;
  IREE_ASSERT_OK(iree_vm_i64_from_variant(results[0], &first));
  IREE_ASSERT_OK(iree_vm_i64_from_variant(results[1], &last));
  EXPECT_EQ(first, 43);
  EXPECT_EQ(last, 101);
  ExpectAliases(results[2], results[3]);
  EXPECT_EQ(release_count, 0);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  EXPECT_EQ(release_count, 1);
}

TEST_F(VMImportsTest, NativeFailureUnwindsAliasedBufferArguments) {
  int release_count = 0;
  iree_vm_variant_t argument = WrapBuffer(&release_count);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      Invoke(IREE_SV("failure"), {&argument, 1}, iree_vm_variant_span_empty()));
  EXPECT_EQ(release_count, 1);
}

TEST_F(VMImportsTest, CapturedLocationOutlivesCompiledProgram) {
  iree_vm_variant_t result = {};
  IREE_ASSERT_OK(
      Invoke(IREE_SV("location"), iree_vm_variant_span_empty(), {&result, 1}));
  iree_vm_process_release(std::exchange(process_, nullptr));
  iree_vm_program_release(std::exchange(program_, nullptr));
  iree_vm_module_release(std::exchange(bytecode_, nullptr));

  void* pointer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_ptr_from_variant_borrowed(result, types_.buffer, &pointer));
  auto* buffer = static_cast<iree_vm_buffer_t*>(pointer);
  iree_const_byte_span_t bytes;
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      buffer, 0, iree_vm_buffer_length(buffer), &bytes));
  loom_location_value_t location;
  IREE_ASSERT_OK(loom_location_value_parse(bytes, &location));
  ASSERT_EQ(location.node_count, 2u);
  EXPECT_EQ(loom_location_value_kind(location, 1), LOOM_LOCATION_VALUE_TAGGED);
  EXPECT_EQ(loom_location_value_flags(location, 1),
            LOOM_LOCATION_VALUE_FLAG_SYNTHETIC);
  const auto tag = loom_location_value_tagged(location, 1);
  EXPECT_EQ(tag.tag, 2u);
  EXPECT_EQ(tag.child, 0u);
  ASSERT_EQ(tag.data.data_length, 2u);
  EXPECT_EQ(tag.data.data[1], 2u);
  const auto file = loom_location_value_file(location, tag.child);
  EXPECT_TRUE(iree_string_view_equal(file.source, IREE_SV("helper.cc")));
  EXPECT_EQ(file.range.start_line, 70000u);
  EXPECT_EQ(file.range.end_column, 15u);
  EXPECT_TRUE(file.has_text);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(file.text.data),
                        file.text.data_length),
            "  check(value);\n");
  ASSERT_EQ(file.field_count, 1u);
  const auto field = loom_location_value_field(location, tag.child, 0);
  EXPECT_EQ(field.kind, 0u);
  EXPECT_EQ(field.index, 1u);
  EXPECT_EQ(field.range.start_column, 9u);
  iree_vm_variant_span_reset({&result, 1});
}

// Exercises the compiler-client lifecycle: capture while admitted sources are
// alive, freeze in bytecode without debug locations, link, compile, and only
// then hand independently owned executable bytes to the native-call fixture.
class VMSourceCaptureTest : public VMImportsTest {
 protected:
  void CreateImage(std::vector<uint8_t>* out_image) override {
    iree_arena_block_pool_t pool;
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(), &pool);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&pool, &arena);
    loom_context_t context;
    loom_context_initialize(iree_allocator_system(), &context);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context));
    IREE_ASSERT_OK(loom_context_finalize(&context));
    const auto* data = loom_location_capture_test_data_create();
    loom_input_request_t request = {
        .source = iree_make_string_view(
            reinterpret_cast<const char*>(data[0].data), data[0].size),
        .path = IREE_SV("admitted.loom"),
    };
    loom_input_module_t input;
    IREE_ASSERT_OK(loom_input_module_load(&loom_input_text_provider, &request,
                                          &context, &pool,
                                          iree_allocator_system(), &input));
    ASSERT_NE(input.module, nullptr);
    auto find = [](loom_module_t* module, iree_string_view_t name) {
      auto symbol = loom_module_find_symbol(
          module, loom_module_lookup_string(module, name));
      return module->symbols.entries[symbol].defining_op;
    };
    loom_parameterized_attr_array_t nodes;
    IREE_ASSERT_OK(loom_func_location_capture(
        input.module, find(input.module, IREE_SV("original"))->location,
        loom_input_module_source_resolver(&input), &arena, &nodes));
    auto function = loom_func_like_cast(
        input.module, find(input.module, IREE_SV("captured")));
    auto* capture =
        loom_region_entry_block(loom_func_like_body(function))->first_op;
    ASSERT_TRUE(loom_func_location_isa(capture));
    IREE_ASSERT_OK(loom_func_location_set_nodes(
        input.module, capture,
        loom_attr_parameterized_array(nodes.values, nodes.count)));

    iree_io_stream_t* stream = nullptr;
    IREE_ASSERT_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_READABLE |
            IREE_IO_STREAM_MODE_SEEKABLE,
        4096, iree_allocator_system(), &stream));
    loom_bytecode_write_options_t write_options = {
        .location_mode = LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS,
    };
    IREE_ASSERT_OK(loom_bytecode_write_module(input.module, stream,
                                              &write_options, &pool));
    std::vector<uint8_t> serialized(iree_io_stream_length(stream));
    IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_ASSERT_OK(iree_io_stream_read(stream, serialized.size(),
                                       serialized.data(), nullptr));
    iree_io_stream_release(stream);
    loom_input_module_deinitialize(&input);
    iree_arena_reset(&arena);

    const loomc_allocator_t compiler_allocator = loomc_allocator_system();
    loomc_target_environment_t* target_environment = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_environment_create_vm(
        compiler_allocator, &target_environment)));
    loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = nullptr,
        .target_environment = target_environment,
    };
    loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    loomc_context_t* compiler_context = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_context_create(
        &context_options, compiler_allocator, &compiler_context)));
    loomc_workspace_t* workspace = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_workspace_create(
        /*options=*/nullptr, compiler_allocator, &workspace)));
    loomc_compiler_t* compiler = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_compiler_create(
        compiler_context, /*options=*/nullptr, compiler_allocator, &compiler)));
    loomc_target_profile_t* profile = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_profile_select(
        target_environment, loomc_make_cstring_view("vm:core"),
        compiler_allocator, &profile)));

    const loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .next = nullptr,
        .format = LOOMC_SOURCE_FORMAT_BYTECODE,
        .identifier = loomc_make_cstring_view("stripped.loombc"),
        .contents = loomc_make_byte_span(serialized.data(), serialized.size()),
        .storage = LOOMC_SOURCE_STORAGE_BORROWED,
    };
    loomc_source_t* compiler_source = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_source_create(
        &source_options, compiler_allocator, &compiler_source)));
    loomc_module_t* module = nullptr;
    loomc_result_t* deserialize_result = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_module_deserialize_from_source(
        compiler_context, workspace, compiler_source, /*options=*/nullptr,
        compiler_allocator, &module, &deserialize_result)));
    ASSERT_NE(deserialize_result, nullptr);
    ASSERT_TRUE(loomc_result_succeeded(deserialize_result));
    loomc_result_release(deserialize_result);

    const loomc_string_view_t root = loomc_make_cstring_view("captured");
    const loomc_emit_options_t emit_options = {
        .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        .structure_size = sizeof(emit_options),
        .next = nullptr,
        .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM),
    };
    const loomc_compile_artifact_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
        .structure_size = sizeof(compile_options),
        .next = nullptr,
        .roots = &root,
        .root_count = 1,
        .excluded_roots = nullptr,
        .excluded_root_count = 0,
        .target_profile = profile,
        .config = nullptr,
        .emit_options = &emit_options,
        .artifact_flags = 0,
    };
    loomc_result_t* compile_result = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_compile_artifact(
        compiler, workspace, /*pass_program=*/nullptr, module, &compile_options,
        compiler_allocator, &compile_result)));
    ASSERT_NE(compile_result, nullptr);
    ASSERT_TRUE(loomc_result_succeeded(compile_result));
    ASSERT_EQ(loomc_result_artifact_count(compile_result), 1u);
    const loomc_artifact_t* artifact =
        loomc_result_artifact_at(compile_result, 0);
    ASSERT_NE(artifact, nullptr);
    loomc_byte_span_t image = loomc_byte_span_empty();
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_byte_sequence_clone(
        artifact->contents, compiler_allocator, &image)));
    out_image->assign(image.data, image.data + image.data_length);
    loomc_allocator_free(compiler_allocator, const_cast<uint8_t*>(image.data));

    loomc_result_release(compile_result);
    loomc_module_release(module);
    loomc_source_release(compiler_source);
    loomc_target_profile_release(profile);
    loomc_compiler_release(compiler);
    loomc_workspace_release(workspace);
    loomc_context_release(compiler_context);
    loomc_target_environment_release(target_environment);
    loom_context_deinitialize(&context);
    iree_arena_deinitialize(&arena);
    iree_arena_block_pool_deinitialize(&pool);
  }
};

TEST_F(VMSourceCaptureTest, OriginalSourceOutlivesCompilerAndProgram) {
  iree_vm_variant_t result = {};
  IREE_ASSERT_OK(
      Invoke(IREE_SV("captured"), iree_vm_variant_span_empty(), {&result, 1}));
  iree_vm_process_release(std::exchange(process_, nullptr));
  iree_vm_program_release(std::exchange(program_, nullptr));
  iree_vm_module_release(std::exchange(bytecode_, nullptr));
  void* pointer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_ptr_from_variant_borrowed(result, types_.buffer, &pointer));
  auto* buffer = static_cast<iree_vm_buffer_t*>(pointer);
  iree_const_byte_span_t bytes;
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      buffer, 0, iree_vm_buffer_length(buffer), &bytes));
  loom_location_value_t value;
  IREE_ASSERT_OK(loom_location_value_parse(bytes, &value));
  ASSERT_EQ(value.node_count, 1u);
  const auto file = loom_location_value_file(value, 0);
  EXPECT_TRUE(iree_string_view_equal(file.source, IREE_SV("admitted.loom")));
  EXPECT_TRUE(file.has_text);
  EXPECT_EQ(file.range.start_line, 4u);
  EXPECT_GT(file.field_count, 0u);
  const std::string text(reinterpret_cast<const char*>(file.text.data),
                         file.text.data_length);
  EXPECT_EQ(text,
            "func.def export(\"résumé\") @original(%left: i32, %right: i32) -> "
            "(i32) {\n"
            "  %sum = scalar.addi %left, %right : i32\n"
            "  func.return %sum : i32\n}\n");
  iree_vm_variant_span_reset({&result, 1});
}

}  // namespace
