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
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/location.h"
#include "loom/link/linker.h"
#include "loom/ops/func/location_capture.h"
#include "loom/ops/func/location_capture_test_data.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/vm/module.h"
#include "loom/target/arch/vm/provider.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/input/input.h"
#include "loom/tooling/target/vm/imports_bytecode.h"
#include "loom/transforms/cleanup/configured.h"

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
    const loom_target_provider_t* providers[] = {&loom_vm_target_provider};
    const auto provider_set = loom_target_provider_set_make(providers, 1);
    loom_target_environment_t environment;
    IREE_ASSERT_OK(
        loom_target_environment_initialize(&provider_set, &environment));
    loom_context_t context;
    loom_context_initialize(iree_allocator_system(), &context);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context));
    IREE_ASSERT_OK(
        loom_target_environment_register_context(&environment, &context));
    IREE_ASSERT_OK(loom_context_finalize(&context));
    const auto* data = loom_location_capture_test_data_create();
    loom_input_request_t request = {};
    request.source = iree_make_string_view(
        reinterpret_cast<const char*>(data[0].data), data[0].size);
    request.path = IREE_SV("admitted.loom");
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
    loom_bytecode_write_options_t write_options = {};
    write_options.location_mode = LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS;
    IREE_ASSERT_OK(loom_bytecode_write_module(input.module, stream,
                                              &write_options, &pool));
    std::vector<uint8_t> serialized(iree_io_stream_length(stream));
    IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_ASSERT_OK(iree_io_stream_read(stream, serialized.size(),
                                       serialized.data(), nullptr));
    iree_io_stream_release(stream);
    loom_input_module_deinitialize(&input);
    iree_arena_reset(&arena);

    loom_bytecode_read_result_t read_result;
    loom_module_t* decoded = nullptr;
    IREE_ASSERT_OK(loom_bytecode_read_module(
        {serialized.data(), serialized.size()}, IREE_SV("stripped.loombc"),
        &context, &pool, nullptr, &read_result, &decoded,
        iree_allocator_system()));
    ASSERT_EQ(read_result.error_count, 0u);
    ASSERT_NE(decoded, nullptr);
    EXPECT_LE(decoded->locations.count, 1u);
    const loom_module_t* sources[] = {decoded};
    const iree_string_view_t root = IREE_SV("captured");
    loom_link_options_t link_options = {};
    link_options.module_name = IREE_SV("captured");
    link_options.root_symbols = {1, &root};
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(loom_link_materialized_modules(
        sources, 1, &link_options, &pool, iree_allocator_system(), &module));
    loom_module_free(decoded);
    serialized.clear();

    const loom_target_profile_t* profile = nullptr;
    IREE_ASSERT_OK(
        loom_vm_target_provider.select_profile(IREE_SV("core"), &profile));
    loom_target_low_descriptor_registry_t registry;
    IREE_ASSERT_OK(loom_target_environment_initialize_low_descriptor_registry(
        &environment, &registry));
    loom_target_specialization_request_t specialization = {};
    specialization.function_name = root;
    specialization.target_profile = profile;
    loom_compile_pipeline_options_t options;
    loom_compile_pipeline_options_initialize(&options);
    options.target_environment = &environment;
    options.low_descriptor_registry = &registry;
    options.cleanup_pattern_provider_set =
        loom_cleanup_configured_pattern_provider_set();
    options.target_specializations = {&specialization, 1};
    loom_compile_pipeline_result_t pipeline;
    IREE_ASSERT_OK(
        loom_compile_run_pipeline(module, &options, &pool, &pipeline));
    ASSERT_EQ(pipeline.pass.error_count, 0u);
    loom_target_emit_request_t emission = {};
    emission.target_environment = &environment;
    emission.low_descriptor_registry = &registry.registry;
    emission.module = module;
    emission.function_versions = &pipeline.function_versions.list;
    emission.scratch_arena = &arena;
    emission.allocator = iree_allocator_system();
    loom_target_emit_artifact_t artifact;
    bool artifact_emitted = false;
    IREE_ASSERT_OK(
        loom_vm_module_emit(&emission, &artifact_emitted, &artifact));
    ASSERT_TRUE(artifact_emitted);
    iree_byte_span_t image;
    IREE_ASSERT_OK(iree_byte_sequence_clone(artifact.contents,
                                            iree_allocator_system(), &image));
    out_image->assign(image.data, image.data + image.data_length);
    iree_allocator_free(iree_allocator_system(), image.data);
    loom_target_emit_artifact_release(&artifact);
    loom_compile_pipeline_result_deinitialize(&pipeline);
    loom_module_free(module);
    loom_context_deinitialize(&context);
    loom_target_environment_deinitialize(&environment);
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
