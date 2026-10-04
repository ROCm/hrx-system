// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/program.h"

#include <cstring>
#include <string>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"

IREE_FLAG(string, policy_source, "", "Production model weight policy source.");

namespace {

class ProgramTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    iree_vm_buffer_release(root);
    loom_serve_program_destroy(program);
    iree_vm_environment_free(environment);
  }

  // Allocation policy shared by the VM and borrowed key wrapper.
  const iree_allocator_t allocator = iree_allocator_system();
  // Owned VM environment whose provider scopes outlive all results.
  iree_vm_environment_t* environment = nullptr;
  // Owned source-JIT host program.
  loom_serve_program_t* program = nullptr;
  // Invocation arguments owning the temporary key wrapper.
  iree_vm_variant_t arguments[1] = {};
  // Invocation results with ordinary VM reference ownership.
  iree_vm_variant_t results[2] = {};
  // Owned returned root, independent of argument storage.
  iree_vm_buffer_t* root = nullptr;
};

TEST_F(ProgramTest, SourcePolicyQueriesBorrowedKeysAndOwnsResults) {
  IREE_ASSERT_OK(iree_vm_environment_allocate(allocator, &environment));
  iree_vm_ref_types_t types = {};
  IREE_ASSERT_OK(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types));
  const iree_string_view_t roots[] = {IREE_SVL("prepare_weight")};
  IREE_ASSERT_OK(loom_serve_program_create(
      environment, iree_make_cstring_view(FLAG_policy_source),
      IREE_ARRAYSIZE(roots), roots, iree_vm_module_span_empty(), allocator,
      &program));
  iree_vm_function_t function = {};
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      loom_serve_program_process(program), IREE_SV("model"),
      IREE_SV("prepare_weight"), &function));
  struct Query {
    // Actual reflected tensor key or nonmatching boundary case.
    const char* key;
    // Whether source selects the FFN preparation command.
    bool prepared;
  };
  const Query queries[] = {
      {"", false},
      {"b", false},
      {"blk", false},
      {"blk.", false},
      {".ffn_gate.weight", false},
      {"other.0.ffn_gate.weight", false},
      {"blk.0.ffn_gate.weight", true},
      {"blk.64.ffn_up.weight", true},
      {"blk.0.ffn_down.weight", false},
      {"token_embd.weight", false},
      {"blk.0.ffn_gate.weight.extra", false},
      {"blk.0.ffn_up.weigh", false},
  };
  for (const auto& query : queries) {
    SCOPED_TRACE(query.key);
    // Padding is not part of the VM key; this is not a C string ABI.
    std::string storage = std::string(query.key) + "not part of the key";
    iree_vm_buffer_t* key = nullptr;
    IREE_ASSERT_OK(iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span(storage.data(), strlen(query.key)),
        iree_vm_buffer_release_callback_null(), allocator, &key));
    arguments[0] = iree_vm_buffer_variant_from_ptr_move(&types, &key);
    IREE_ASSERT_OK(iree_vm_invoke(loom_serve_program_invocation(program),
                                  function,
                                  iree_vm_variant_span_from_array(arguments),
                                  iree_vm_variant_span_from_array(results)));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    storage.clear();
    IREE_ASSERT_OK(
        iree_vm_buffer_ptr_from_variant_move(&types, &results[0], &root));
    int64_t byte_length = -1;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(results[1], &byte_length));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    iree_const_byte_span_t bytes;
    IREE_ASSERT_OK(
        iree_vm_buffer_map_read(root, 0, iree_vm_buffer_length(root), &bytes));
    const std::string name(
        bytes.data_length ? reinterpret_cast<const char*>(bytes.data) : "",
        bytes.data_length);
    EXPECT_EQ(name, query.prepared ? "qwen38_prepare_ffn" : "");
    EXPECT_EQ(byte_length, query.prepared ? 61276160 : 0);
    iree_vm_buffer_release(root);
    root = nullptr;
  }
}

}  // namespace
