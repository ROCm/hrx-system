// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_compiler_cxx.h"
#include "pyre_runtime_cxx.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

extern pyre_device_t g_test_device;

namespace {

void requireOk(pyre_status_t status) {
  if (pyre_status_is_ok(status)) {
    return;
  }

  char* message = nullptr;
  size_t length = 0;
  pyre_status_t format_status =
      pyre_status_to_string(status, &message, &length);
  INFO("pyre error: " << (message ? message : "?"));
  pyre_status_free_message(message);
  pyre_status_ignore(format_status);
  pyre_status_ignore(status);
  REQUIRE(false);
}

#define REQUIRE_PYRE_OK(expr) requireOk((expr))

std::string loadMlir(const char* relative_path) {
  std::string path = std::string(PYRE_CTS_SOURCE_DIR) + "/" + relative_path;
  std::ifstream file(path);
  REQUIRE(file.is_open());
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

bool cliCompilerConfigured() {
  const char* cli_path = std::getenv("PYRE_IREE_COMPILER_CLI");
  if (!cli_path || !*cli_path) {
    return false;
  }
  std::ifstream file(cli_path);
  return file.good();
}

std::unique_ptr<pyre::compiler::PyreGraphCompiler> createCompilerIfAvailable(
    pyre_compiler_backend_t backend) {
  try {
    return std::make_unique<pyre::compiler::PyreGraphCompiler>(backend);
  } catch (const std::runtime_error& e) {
    INFO("compiler backend unavailable: " << e.what());
    return nullptr;
  }
}

void verifyAddI64(pyre_device_t device,
                  const pyre::compiler::compiler_output_ptr& output) {
  REQUIRE(output);
  REQUIRE(pyre_compiler_output_data(output.get()) != nullptr);
  REQUIRE(pyre_compiler_output_size(output.get()) > 0);

  pyre::runtime::module_ptr module;
  REQUIRE_PYRE_OK(pyre_module_load_compiler_output(
      device, output.get(), module.for_output()));

  pyre::runtime::function_ptr function;
  REQUIRE_PYRE_OK(pyre_module_lookup_function(
      module.get(), "module.add_i64", function.for_output()));

  pyre::runtime::value_list_ptr args;
  pyre::runtime::value_list_ptr rets;
  REQUIRE_PYRE_OK(pyre_value_list_create(2, args.for_output()));
  REQUIRE_PYRE_OK(pyre_value_list_create(1, rets.for_output()));
  REQUIRE_PYRE_OK(pyre_value_list_push_i64(args.get(), 40));
  REQUIRE_PYRE_OK(pyre_value_list_push_i64(args.get(), 2));
  REQUIRE_PYRE_OK(pyre_function_invoke(
      module.get(), function.get(), args.get(), rets.get()));

  int64_t result = 0;
  REQUIRE_PYRE_OK(pyre_value_list_get_i64(rets.get(), 0, &result));
  REQUIRE(result == 42);
}

void verifyBadMlirDiagnostics(pyre::compiler::PyreGraphCompiler& compiler) {
  auto session = compiler.createSession();
  session.setFlags({
      "--iree-hal-target-backends=llvm-cpu",
  });

  try {
    (void)session.compileMlir("module { util.func public @broken(");
    FAIL("expected malformed MLIR to fail");
  } catch (const std::runtime_error& e) {
    std::string message = e.what();
    INFO("diagnostics: " << message);
    REQUIRE(message.find("failed to parse MLIR") != std::string::npos);
  }
}

}  // namespace

TEST_CASE("Compile MLIR with default dylib backend", "[compiler][dylib]") {
  auto compiler =
      createCompilerIfAvailable(PYRE_COMPILER_BACKEND_DEFAULT);
  if (!compiler) {
    SKIP("IREE compiler dylib unavailable");
  }

  pyre::compiler::compiler_output_ptr output = compiler->compileMlir(
      loadMlir("testdata/add_i64.mlir"),
      {"--iree-hal-target-backends=llvm-cpu"});
  verifyAddI64(g_test_device, output);
}

TEST_CASE("Compile MLIR with explicit CLI backend", "[compiler][cli]") {
  if (!cliCompilerConfigured()) {
    SKIP("PYRE_IREE_COMPILER_CLI is not configured");
  }

  auto compiler = createCompilerIfAvailable(PYRE_COMPILER_BACKEND_CLI);
  REQUIRE(compiler != nullptr);

  pyre::compiler::compiler_output_ptr output = compiler->compileMlir(
      loadMlir("testdata/add_i64.mlir"),
      {"--iree-hal-target-backends=llvm-cpu"});
  verifyAddI64(g_test_device, output);
}

TEST_CASE("Compiler reports diagnostics for bad MLIR",
          "[compiler][diagnostics]") {
  auto compiler =
      createCompilerIfAvailable(PYRE_COMPILER_BACKEND_DEFAULT);
  if (!compiler) {
    if (!cliCompilerConfigured()) {
      SKIP("no compiler backend available");
    }
    compiler = createCompilerIfAvailable(PYRE_COMPILER_BACKEND_CLI);
    REQUIRE(compiler != nullptr);
  }

  verifyBadMlirDiagnostics(*compiler);
}
