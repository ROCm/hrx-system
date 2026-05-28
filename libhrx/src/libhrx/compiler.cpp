// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_internal.h"

#include "iree/compiler/embedding_api.h"
#include "iree/compiler/loader.h"

#include <errno.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

extern char **environ;

namespace {

#if defined(_WIN32)
constexpr bool kIsWindows = true;
#else
constexpr bool kIsWindows = false;
#endif
constexpr bool kIsPosix = !kIsWindows;

class CompilerDiscovery {
public:
  CompilerDiscovery() {
    if (const char *path = std::getenv("HRX_IREE_COMPILE")) {
      dylib_path_ = path;
      has_dylib_override_ = true;
    }
    if (const char *path = std::getenv("HRX_IREE_COMPILER_CLI")) {
      cli_path_ = path;
    }
  }

  const std::string &dylibPath() const { return dylib_path_; }
  bool hasDylibOverride() const { return has_dylib_override_; }
  const std::string &cliPath() const { return cli_path_; }
  bool hasCliOverride() const { return !cli_path_.empty(); }

  hrx_status_t requireCliPath() {
    if (cli_path_.empty()) {
      return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                             "HRX_IREE_COMPILER_CLI is unset");
    }
    std::optional<std::filesystem::path> executable_path =
        resolveExecutablePath(cli_path_);
    if (!executable_path) {
      return hrx_make_status(
          HRX_STATUS_UNAVAILABLE,
          "HRX_IREE_COMPILER_CLI does not point to an executable file");
    }
    cli_path_ = executable_path->string();
    return hrx_ok_status();
  }

private:
  static bool isExecutableFile(const std::filesystem::file_status &status) {
    if (!std::filesystem::is_regular_file(status)) {
      return false;
    }
    if (kIsWindows) {
      return true;
    }
    if (!kIsPosix) {
      return false;
    }
    std::filesystem::perms perms = status.permissions();
    return (perms & (std::filesystem::perms::owner_exec |
                     std::filesystem::perms::group_exec |
                     std::filesystem::perms::others_exec)) !=
           std::filesystem::perms::none;
  }

  static std::optional<std::filesystem::path>
  resolveExecutablePath(const std::filesystem::path &path) {
    std::error_code ec;
    std::filesystem::file_status status = std::filesystem::status(path, ec);
    if (!ec && isExecutableFile(status)) {
      return path;
    }
    if (kIsPosix || path.extension() == ".exe") {
      return std::nullopt;
    }
    std::filesystem::path exe_path = path;
    exe_path += ".exe";
    ec.clear();
    status = std::filesystem::status(exe_path, ec);
    if (!ec && isExecutableFile(status)) {
      return exe_path;
    }
    return std::nullopt;
  }

  std::string dylib_path_ = "libIREECompiler.so";
  bool has_dylib_override_ = false;
  std::string cli_path_;
};

struct CompilerDylibState {
  std::once_flag init_once;
  bool available = false;
  std::string error_message;
};

CompilerDylibState &dylibState() {
  static CompilerDylibState state;
  return state;
}

void initializeCompilerDylib() {
  static constexpr const char *kDefaultLibs[] = {
      "libIREECompiler.so",
      "libIREECompiler.so.0",
  };

  CompilerDiscovery discovery;
  // TODO(hrx): on Windows, probe the directory containing libhrx and look
  // for an adjacent libIREECompiler DLL before falling back to the default
  // loader search path.
  if (discovery.hasDylibOverride()) {
    if (ireeCompilerLoadLibrary(discovery.dylibPath().c_str())) {
      ireeCompilerGlobalInitialize();
      dylibState().available = true;
      return;
    }
    dylibState().error_message =
        "failed to load HRX_IREE_COMPILE=" + discovery.dylibPath();
    return;
  }

  for (const char *lib_path : kDefaultLibs) {
    if (ireeCompilerLoadLibrary(lib_path)) {
      ireeCompilerGlobalInitialize();
      dylibState().available = true;
      return;
    }
  }

  dylibState().error_message =
      "failed to load libIREECompiler.so from the default search path";
}

hrx_status_t ensureCompilerDylibLoaded() {
  CompilerDylibState &state = dylibState();
  std::call_once(state.init_once, initializeCompilerDylib);
  if (state.available) {
    return hrx_ok_status();
  }
  return hrx_make_status(HRX_STATUS_UNAVAILABLE, state.error_message.c_str());
}

hrx_status_t
statusFromCompilerError(const char *context, iree_compiler_error_t *error,
                        const std::string &diagnostics = std::string()) {
  std::string message(context);
  if (error) {
    message.append(": ");
    message.append(ireeCompilerErrorGetMessage(error));
    ireeCompilerErrorDestroy(error);
  }
  if (!diagnostics.empty()) {
    message.append("\n");
    message.append(diagnostics);
  }
  return hrx_make_status(HRX_STATUS_INTERNAL, message.c_str());
}

void destroyDylibOutput(hrx_compiler_output_t output) {
  ireeCompilerOutputDestroy(
      static_cast<iree_compiler_output_t *>(output->impl));
  free(output);
}

void destroyCliOutput(hrx_compiler_output_t output) {
  hrx_host_allocator_free_aligned(output->host_allocator,
                                  const_cast<uint8_t *>(output->data));
  free(output);
}

void freeFlags(char **flags, size_t flag_count) {
  for (size_t i = 0; i < flag_count; ++i) {
    free(flags[i]);
  }
  free(flags);
}

hrx_status_t cloneFlags(const char *const *flags, size_t flag_count,
                        char ***out_flags) {
  *out_flags = nullptr;
  if (flag_count == 0) {
    return hrx_ok_status();
  }

  char **cloned = static_cast<char **>(calloc(flag_count, sizeof(char *)));
  if (!cloned) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate compiler flags");
  }

  for (size_t i = 0; i < flag_count; ++i) {
    if (!flags || !flags[i]) {
      freeFlags(cloned, flag_count);
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "compiler flag is NULL");
    }
    cloned[i] = strdup(flags[i]);
    if (!cloned[i]) {
      freeFlags(cloned, flag_count);
      return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                             "failed to copy compiler flag");
    }
  }

  *out_flags = cloned;
  return hrx_ok_status();
}

hrx_status_t compileWithDylib(hrx_compiler_session_t session,
                              std::string_view mlir,
                              hrx_compiler_output_t *output) {
  struct InvocationState {
    iree_compiler_invocation_t *invocation = nullptr;
    iree_compiler_source_t *source = nullptr;
    iree_compiler_output_t *iree_output = nullptr;
    std::string diagnostics;
    ~InvocationState() {
      if (source) {
        ireeCompilerSourceDestroy(source);
      }
      if (invocation) {
        ireeCompilerInvocationDestroy(invocation);
      }
    }
  } state;

  state.invocation = ireeCompilerInvocationCreate(session->iree_session);
  ireeCompilerInvocationEnableCallbackDiagnostics(
      state.invocation, 0,
      [](iree_compiler_diagnostic_severity_t severity, const char *message,
         size_t message_size, void *user_data) {
        if (severity < IREE_COMPILER_DIAGNOSTIC_SEVERITY_ERROR) {
          return;
        }
        auto *diagnostics = static_cast<std::string *>(user_data);
        if (!diagnostics->empty()) {
          diagnostics->push_back('\n');
        }
        diagnostics->append(message, message_size);
      },
      &state.diagnostics);

  std::string mlir_copy(mlir.data(), mlir.size());
  if (iree_compiler_error_t *error = ireeCompilerSourceWrapBuffer(
          session->iree_session, "hrx_graph.mlir", mlir_copy.c_str(),
          mlir_copy.size() + 1, /*isNullTerminated=*/true, &state.source)) {
    return statusFromCompilerError("failed to create compiler source", error);
  }

  if (!ireeCompilerInvocationParseSource(state.invocation, state.source)) {
    return statusFromCompilerError("failed to parse MLIR", nullptr,
                                   state.diagnostics);
  }

  if (!ireeCompilerInvocationPipeline(state.invocation,
                                      IREE_COMPILER_PIPELINE_STD)) {
    return statusFromCompilerError("IREE compilation failed", nullptr,
                                   state.diagnostics);
  }

  if (iree_compiler_error_t *error =
          ireeCompilerOutputOpenMembuffer(&state.iree_output)) {
    return statusFromCompilerError("failed to create compiler output", error);
  }

  if (iree_compiler_error_t *error = ireeCompilerInvocationOutputVMBytecode(
          state.invocation, state.iree_output)) {
    ireeCompilerOutputDestroy(state.iree_output);
    return statusFromCompilerError("failed to emit VMFB", error,
                                   state.diagnostics);
  }

  void *contents = nullptr;
  uint64_t size = 0;
  if (iree_compiler_error_t *error =
          ireeCompilerOutputMapMemory(state.iree_output, &contents, &size)) {
    ireeCompilerOutputDestroy(state.iree_output);
    return statusFromCompilerError("failed to map compiler output", error);
  }

  hrx_compiler_output_t compiled =
      static_cast<hrx_compiler_output_t>(calloc(1, sizeof(*compiled)));
  if (!compiled) {
    ireeCompilerOutputDestroy(state.iree_output);
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate compiler output");
  }

  iree_atomic_ref_count_init(&compiled->ref_count);
  compiled->data = static_cast<const uint8_t *>(contents);
  compiled->size = static_cast<size_t>(size);
  compiled->impl = state.iree_output;
  compiled->destroy = destroyDylibOutput;
  compiled->host_allocator = hrx_host_allocator_system();
  *output = compiled;
  return hrx_ok_status();
}

hrx_status_t readCliVmfb(const std::filesystem::path &path,
                         hrx_compiler_output_t *output) {
  FILE *file = fopen(path.c_str(), "rb");
  if (!file) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "failed to open iree-compile output");
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "failed to seek iree-compile output");
  }
  long size = ftell(file);
  if (size < 0) {
    fclose(file);
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "failed to stat iree-compile output");
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "failed to rewind iree-compile output");
  }

  hrx_host_allocator_t host_allocator = hrx_host_allocator_system();
  uint8_t *data = nullptr;
  hrx_status_t status = hrx_host_allocator_malloc_aligned(
      host_allocator, static_cast<size_t>(size) + 1, 4096, 0,
      reinterpret_cast<void **>(&data));
  if (!hrx_status_is_ok(status)) {
    fclose(file);
    return status;
  }

  size_t bytes_read = 0;
  if (size > 0) {
    bytes_read = fread(data, 1, static_cast<size_t>(size), file);
  }
  fclose(file);
  if (bytes_read != static_cast<size_t>(size)) {
    hrx_host_allocator_free_aligned(host_allocator, data);
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "failed to read iree-compile output");
  }
  data[size] = 0;

  hrx_compiler_output_t compiled =
      static_cast<hrx_compiler_output_t>(calloc(1, sizeof(*compiled)));
  if (!compiled) {
    hrx_host_allocator_free_aligned(host_allocator, data);
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate compiler output");
  }

  iree_atomic_ref_count_init(&compiled->ref_count);
  compiled->data = data;
  compiled->size = static_cast<size_t>(size);
  compiled->impl = nullptr;
  compiled->destroy = destroyCliOutput;
  compiled->host_allocator = host_allocator;
  *output = compiled;
  return hrx_ok_status();
}

hrx_status_t compileWithCli(hrx_compiler_session_t session,
                            std::string_view mlir,
                            hrx_compiler_output_t *output) {
  std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
  std::string mlir_template = (temp_dir / "hrx_graph_XXXXXX.mlir").string();
  std::string vmfb_template = (temp_dir / "hrx_graph_XXXXXX.vmfb").string();

  int mlir_fd = mkstemps(mlir_template.data(), 5);
  if (mlir_fd < 0) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "failed to create temporary MLIR input");
  }
  int vmfb_fd = mkstemps(vmfb_template.data(), 5);
  if (vmfb_fd < 0) {
    close(mlir_fd);
    std::filesystem::remove(mlir_template);
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "failed to create temporary VMFB output");
  }

  size_t total_written = 0;
  while (total_written < mlir.size()) {
    ssize_t wrote = write(mlir_fd, mlir.data() + total_written,
                          mlir.size() - total_written);
    if (wrote < 0 && errno == EINTR) {
      continue;
    }
    if (wrote <= 0) {
      close(mlir_fd);
      close(vmfb_fd);
      std::filesystem::remove(mlir_template);
      std::filesystem::remove(vmfb_template);
      return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                             "failed to write temporary MLIR input");
    }
    total_written += static_cast<size_t>(wrote);
  }
  close(mlir_fd);
  close(vmfb_fd);

  std::vector<std::string> argv_storage;
  argv_storage.emplace_back(session->compiler->cli_path);
  for (size_t i = 0; i < session->flag_count; ++i) {
    argv_storage.emplace_back(session->flags[i]);
  }
  argv_storage.emplace_back("-o");
  argv_storage.emplace_back(vmfb_template);
  argv_storage.emplace_back(mlir_template);

  std::vector<char *> argv;
  argv.reserve(argv_storage.size() + 1);
  for (std::string &arg : argv_storage) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  int output_pipe[2];
  if (pipe(output_pipe) != 0) {
    std::filesystem::remove(mlir_template);
    std::filesystem::remove(vmfb_template);
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "failed to create compiler diagnostics pipe");
  }

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDERR_FILENO);
  posix_spawn_file_actions_addclose(&actions, output_pipe[0]);

  pid_t pid = -1;
  int spawn_rc = posix_spawn(&pid, session->compiler->cli_path, &actions,
                             nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(output_pipe[1]);

  std::string diagnostics;
  char buffer[4096];
  for (;;) {
    ssize_t n = read(output_pipe[0], buffer, sizeof(buffer));
    if (n > 0) {
      diagnostics.append(buffer, static_cast<size_t>(n));
      continue;
    }
    if (n < 0 && errno == EINTR) {
      continue;
    }
    break;
  }
  close(output_pipe[0]);

  std::filesystem::remove(mlir_template);

  if (spawn_rc != 0) {
    std::filesystem::remove(vmfb_template);
    std::string message =
        "failed to launch iree-compile: " + std::to_string(spawn_rc);
    return hrx_make_status(HRX_STATUS_UNAVAILABLE, message.c_str());
  }

  int child_status = 0;
  while (waitpid(pid, &child_status, 0) < 0) {
    if (errno == EINTR) {
      continue;
    }
    std::filesystem::remove(vmfb_template);
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "failed to wait for iree-compile");
  }

  if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    std::filesystem::remove(vmfb_template);
    std::string message = "iree-compile failed";
    if (WIFEXITED(child_status)) {
      message.append(" (exit ");
      message.append(std::to_string(WEXITSTATUS(child_status)));
      message.push_back(')');
    }
    if (!diagnostics.empty()) {
      message.append(":\n");
      message.append(diagnostics);
    }
    return hrx_make_status(HRX_STATUS_INTERNAL, message.c_str());
  }

  hrx_status_t status = readCliVmfb(vmfb_template, output);
  std::filesystem::remove(vmfb_template);
  return status;
}

} // namespace

extern "C" {

hrx_status_t hrx_compiler_create(hrx_compiler_backend_t backend,
                                 hrx_compiler_t *compiler) {
  if (!compiler) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "compiler is NULL");
  }
  *compiler = nullptr;

  CompilerDiscovery discovery;
  hrx_status_t status = hrx_ok_status();
  if (backend == HRX_COMPILER_BACKEND_AUTO ||
      backend == HRX_COMPILER_BACKEND_DYLIB) {
    status = ensureCompilerDylibLoaded();
    if (!hrx_status_is_ok(status)) {
      if (backend == HRX_COMPILER_BACKEND_AUTO && discovery.hasCliOverride()) {
        hrx_status_ignore(status);
        status = discovery.requireCliPath();
        if (!hrx_status_is_ok(status)) {
          return status;
        }
        backend = HRX_COMPILER_BACKEND_CLI;
      } else {
        return status;
      }
    } else {
      backend = HRX_COMPILER_BACKEND_DYLIB;
    }
  } else if (backend == HRX_COMPILER_BACKEND_CLI) {
    status = discovery.requireCliPath();
    if (!hrx_status_is_ok(status)) {
      return status;
    }
  } else if (backend != HRX_COMPILER_BACKEND_CLI) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "unknown compiler backend");
  }

  hrx_compiler_t created =
      static_cast<hrx_compiler_t>(calloc(1, sizeof(*created)));
  if (!created) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate compiler");
  }

  iree_atomic_ref_count_init(&created->ref_count);
  created->backend = backend;
  if (backend == HRX_COMPILER_BACKEND_CLI) {
    created->cli_path = strdup(discovery.cliPath().c_str());
    if (!created->cli_path) {
      free(created);
      return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                             "failed to copy compiler CLI path");
    }
  }

  *compiler = created;
  return hrx_ok_status();
}

void hrx_compiler_retain(hrx_compiler_t compiler) {
  iree_atomic_ref_count_inc(&compiler->ref_count);
}

void hrx_compiler_release(hrx_compiler_t compiler) {
  if (iree_atomic_ref_count_dec(&compiler->ref_count) == 1) {
    free(compiler->cli_path);
    free(compiler);
  }
}

hrx_compiler_backend_t hrx_compiler_backend(hrx_compiler_t compiler) {
  return compiler->backend;
}

hrx_status_t hrx_compiler_session_create(hrx_compiler_t compiler,
                                         hrx_compiler_session_t *session) {
  if (!compiler || !session) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "compiler or session is NULL");
  }
  *session = nullptr;

  hrx_compiler_session_t created =
      static_cast<hrx_compiler_session_t>(calloc(1, sizeof(*created)));
  if (!created) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate compiler session");
  }

  iree_atomic_ref_count_init(&created->ref_count);
  created->compiler = compiler;
  hrx_compiler_retain(compiler);
  if (compiler->backend == HRX_COMPILER_BACKEND_DYLIB) {
    created->iree_session = ireeCompilerSessionCreate();
  }

  *session = created;
  return hrx_ok_status();
}

void hrx_compiler_session_retain(hrx_compiler_session_t session) {
  hrx_compiler_retain(session->compiler);
  iree_atomic_ref_count_inc(&session->ref_count);
}

void hrx_compiler_session_release(hrx_compiler_session_t session) {
  if (iree_atomic_ref_count_dec(&session->ref_count) == 1) {
    if (session->iree_session) {
      ireeCompilerSessionDestroy(session->iree_session);
    }
    freeFlags(session->flags, session->flag_count);
    hrx_compiler_release(session->compiler);
    free(session);
  } else {
    hrx_compiler_release(session->compiler);
  }
}

hrx_status_t hrx_compiler_session_set_flags(hrx_compiler_session_t session,
                                            const char *const *flags,
                                            size_t flag_count) {
  if (!session || (flag_count > 0 && !flags)) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "session or flags is NULL");
  }

  char **cloned_flags = nullptr;
  hrx_status_t status = cloneFlags(flags, flag_count, &cloned_flags);
  if (!hrx_status_is_ok(status)) {
    return status;
  }

  if (session->compiler->backend == HRX_COMPILER_BACKEND_DYLIB) {
    if (iree_compiler_error_t *error = ireeCompilerSessionSetFlags(
            session->iree_session, static_cast<int>(flag_count), flags)) {
      freeFlags(cloned_flags, flag_count);
      return statusFromCompilerError("failed to set compiler flags", error);
    }
  }

  freeFlags(session->flags, session->flag_count);
  session->flags = cloned_flags;
  session->flag_count = flag_count;
  return hrx_ok_status();
}

hrx_status_t hrx_compiler_session_compile_mlir(hrx_compiler_session_t session,
                                               const char *mlir_data,
                                               size_t mlir_size,
                                               hrx_compiler_output_t *output) {
  if (!session || !mlir_data || !output) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "session, mlir_data, or output is NULL");
  }
  *output = nullptr;
  if (mlir_size == 0) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "mlir_size must be > 0");
  }

  std::string_view mlir(mlir_data, mlir_size);
  if (session->compiler->backend == HRX_COMPILER_BACKEND_CLI) {
    return compileWithCli(session, mlir, output);
  }
  return compileWithDylib(session, mlir, output);
}

void hrx_compiler_output_retain(hrx_compiler_output_t output) {
  iree_atomic_ref_count_inc(&output->ref_count);
}

void hrx_compiler_output_release(hrx_compiler_output_t output) {
  if (iree_atomic_ref_count_dec(&output->ref_count) == 1) {
    output->destroy(output);
  }
}

const uint8_t *hrx_compiler_output_data(hrx_compiler_output_t output) {
  return output ? output->data : nullptr;
}

size_t hrx_compiler_output_size(hrx_compiler_output_t output) {
  return output ? output->size : 0;
}

} // extern "C"
