// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/process.h"

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>
#include <string_view>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_PLATFORM_WINDOWS)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_MACOS) || \
    defined(IREE_PLATFORM_ANDROID)
#include <errno.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_WINDOWS

namespace {

static constexpr char kArgvProbeArgument[] = "--loom-process-argv-probe";
static constexpr char kArgvProbeArguments[][32] = {
    "",           "space value",        "tab\tvalue",     "quote\"value",
    "trailing\\", R"(before\\\"quote)", "space \xCF\x80",
};

TEST(ToolOutputTest, NormalizesCrLfNewlines) {
  char text[] = "first\r\nsecond\r\r\nthird\nfourth\rfifth\r\rtext";
  loom_tool_output_t output = {text, sizeof(text) - 1};

  loom_tool_output_normalize_newlines(&output);

  EXPECT_EQ(std::string_view(output.data, output.length),
            "first\nsecond\nthird\nfourth\rfifth\r\rtext");
  EXPECT_EQ(output.data[output.length], '\0');
}

TEST(ToolOutputTest, AcceptsEmptyOutput) {
  loom_tool_output_t output = {0};

  loom_tool_output_normalize_newlines(&output);

  EXPECT_EQ(output.data, nullptr);
  EXPECT_EQ(output.length, 0u);
}

#if defined(IREE_PLATFORM_WINDOWS)

static constexpr char kUnicodeProbeArgumentUtf8[] =
    "--loom-process-unicode-probe=\xCF\x80";
static constexpr wchar_t kUnicodeProbeArgumentWide[] =
    L"--loom-process-unicode-probe=\x03C0";
static constexpr char kUnicodeProbeOutput[] = "unicode-probe:\xCF\x80";
static constexpr char kHandleProbePrefixUtf8[] = "--loom-process-handle-probe=";
static constexpr wchar_t kHandleProbePrefixWide[] =
    L"--loom-process-handle-probe=";

static std::string Win32WideToUtf8(const std::wstring& value) {
  int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.c_str(),
                                   -1, NULL, 0, NULL, NULL);
  if (length == 0) {
    return std::string();
  }
  std::string result((size_t)length, '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.c_str(), -1,
                          result.data(), length, NULL, NULL) == 0) {
    return std::string();
  }
  result.resize((size_t)length - 1);
  return result;
}

static std::string Win32ExecutablePathUtf8() {
  std::vector<wchar_t> module_path(32768);
  DWORD module_path_length =
      GetModuleFileNameW(NULL, module_path.data(), (DWORD)module_path.size());
  if (module_path_length == 0 ||
      module_path_length >= (DWORD)module_path.size()) {
    return std::string();
  }
  return Win32WideToUtf8(
      std::wstring(module_path.data(), (size_t)module_path_length));
}

class ScopedWin32Handle {
 public:
  explicit ScopedWin32Handle(HANDLE handle) : handle_(handle) {}
  ~ScopedWin32Handle() {
    if (handle_ != NULL && handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
    }
  }

  HANDLE get() const { return handle_; }

 private:
  HANDLE handle_;
};

class ScopedUnicodeProcessProbe {
 public:
  ~ScopedUnicodeProcessProbe() {
    if (tmp_changed_) {
      SetEnvironmentVariableW(L"TMP",
                              had_previous_tmp_ ? previous_tmp_.c_str() : NULL);
    }
    if (!executable_path_.empty()) {
      DeleteFileW(executable_path_.c_str());
    }
    if (!directory_path_.empty()) {
      RemoveDirectoryW(directory_path_.c_str());
    }
  }

  DWORD Initialize() {
    SetLastError(ERROR_SUCCESS);
    DWORD previous_tmp_length = GetEnvironmentVariableW(L"TMP", NULL, 0);
    if (previous_tmp_length != 0) {
      std::vector<wchar_t> previous_tmp(previous_tmp_length);
      if (GetEnvironmentVariableW(L"TMP", previous_tmp.data(),
                                  previous_tmp_length) == 0) {
        return GetLastError();
      }
      previous_tmp_ = previous_tmp.data();
      had_previous_tmp_ = true;
    } else {
      DWORD error = GetLastError();
      if (error == ERROR_SUCCESS) {
        had_previous_tmp_ = true;
      } else if (error != ERROR_ENVVAR_NOT_FOUND) {
        return error;
      }
    }

    wchar_t temp_directory[4096] = {0};
    DWORD temp_directory_length =
        GetTempPathW(IREE_ARRAYSIZE(temp_directory), temp_directory);
    if (temp_directory_length == 0 ||
        temp_directory_length >= IREE_ARRAYSIZE(temp_directory)) {
      return GetLastError();
    }
    directory_path_ = temp_directory;
    directory_path_.append(L"loom_process_\x03C0_");
    directory_path_.append(std::to_wstring(GetCurrentProcessId()));
    directory_path_.push_back(L'_');
    directory_path_.append(std::to_wstring(GetTickCount64()));
    if (!CreateDirectoryW(directory_path_.c_str(), NULL)) {
      return GetLastError();
    }

    std::vector<wchar_t> module_path(32768);
    DWORD module_path_length =
        GetModuleFileNameW(NULL, module_path.data(), (DWORD)module_path.size());
    if (module_path_length == 0 ||
        module_path_length >= (DWORD)module_path.size()) {
      return GetLastError();
    }
    // Keep the probe beside the original image so that any adjacent runtime
    // dependencies remain discoverable while exercising a Unicode image name.
    executable_path_.assign(module_path.data(), (size_t)module_path_length);
    size_t directory_separator = executable_path_.find_last_of(L"\\/");
    if (directory_separator == std::wstring::npos) {
      return ERROR_BAD_PATHNAME;
    }
    executable_path_.resize(directory_separator + 1);
    executable_path_.append(L"process_\x03C0_");
    executable_path_.append(std::to_wstring(GetCurrentProcessId()));
    executable_path_.push_back(L'_');
    executable_path_.append(std::to_wstring(GetTickCount64()));
    executable_path_.append(L".exe");
    if (!CopyFileW(module_path.data(), executable_path_.c_str(), TRUE)) {
      return GetLastError();
    }
    if (!SetEnvironmentVariableW(L"TMP", directory_path_.c_str())) {
      return GetLastError();
    }
    tmp_changed_ = true;
    return ERROR_SUCCESS;
  }

  const std::wstring& executable_path() const { return executable_path_; }

 private:
  bool had_previous_tmp_ = false;
  bool tmp_changed_ = false;
  std::wstring previous_tmp_;
  std::wstring directory_path_;
  std::wstring executable_path_;
};

TEST(ToolProcessTest, PreservesUtf8AtWin32Boundaries) {
  ScopedUnicodeProcessProbe probe;
  ASSERT_EQ(probe.Initialize(), ERROR_SUCCESS);

  std::string executable_path = Win32WideToUtf8(probe.executable_path());
  ASSERT_FALSE(executable_path.empty());
  iree_string_view_t arguments[] = {
      iree_make_cstring_view(kUnicodeProbeArgumentUtf8),
  };
  loom_tool_process_result_t result = {0};
  IREE_ASSERT_OK(loom_tool_process_run(
      iree_make_cstring_view(executable_path.c_str()), /*search_path=*/false,
      arguments, IREE_ARRAYSIZE(arguments), iree_allocator_system(), &result));
  EXPECT_TRUE(loom_tool_process_result_succeeded(&result));
  EXPECT_EQ(
      std::string_view(result.stdout_bytes.data, result.stdout_bytes.length),
      kUnicodeProbeOutput);
  EXPECT_EQ(result.stderr_bytes.length, 0u);
  loom_tool_process_result_deinitialize(&result, iree_allocator_system());

  loom_tool_temp_file_t temp_file;
  IREE_ASSERT_OK(
      loom_tool_temp_file_initialize(IREE_SV("unicode"), &temp_file));
  EXPECT_NE(std::string_view(temp_file.path).find("\xCF\x80"),
            std::string_view::npos);
  IREE_ASSERT_OK(loom_tool_temp_file_deinitialize(&temp_file));
}

TEST(ToolProcessTest, DoesNotInheritUnrelatedWin32Handles) {
  SECURITY_ATTRIBUTES security_attributes = {0};
  security_attributes.nLength = sizeof(security_attributes);
  security_attributes.bInheritHandle = TRUE;
  ScopedWin32Handle event(
      CreateEventW(&security_attributes, TRUE, FALSE, NULL));
  ASSERT_NE(event.get(), nullptr);

  std::string executable_path = Win32ExecutablePathUtf8();
  ASSERT_FALSE(executable_path.empty());
  std::string probe_argument =
      std::string(kHandleProbePrefixUtf8) +
      std::to_string(reinterpret_cast<std::uintptr_t>(event.get()));
  iree_string_view_t arguments[] = {
      iree_make_string_view(probe_argument.data(), probe_argument.size()),
  };
  loom_tool_process_result_t result = {0};
  IREE_ASSERT_OK(loom_tool_process_run(
      iree_make_string_view(executable_path.data(), executable_path.size()),
      /*search_path=*/false, arguments, IREE_ARRAYSIZE(arguments),
      iree_allocator_system(), &result));
  EXPECT_TRUE(loom_tool_process_result_succeeded(&result));
  EXPECT_EQ(WaitForSingleObject(event.get(), 0), WAIT_TIMEOUT);
  loom_tool_process_result_deinitialize(&result, iree_allocator_system());
}

TEST(ToolProcessTest, MapsMissingPathExecutableToNotFound) {
  loom_tool_process_result_t result = {0};
  iree_status_t status = loom_tool_process_run(
      IREE_SV("loom_process_missing_executable_7d56240f.exe"),
      /*search_path=*/true, /*arguments=*/NULL, /*argument_count=*/0,
      iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
}

#endif  // IREE_PLATFORM_WINDOWS

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_MACOS) || \
    defined(IREE_PLATFORM_ANDROID)

static constexpr char kDescriptorProbePrefix[] =
    "--loom-process-descriptor-probe=";
static std::string g_executable_path;

class ScopedPosixDescriptor {
 public:
  explicit ScopedPosixDescriptor(int fd) : fd_(fd) {}
  ~ScopedPosixDescriptor() { reset(); }

  int get() const { return fd_; }

  void reset() {
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }

 private:
  int fd_;
};

TEST(ToolProcessTest, DoesNotInheritUnrelatedPosixDescriptors) {
  int pipe_descriptors[2] = {-1, -1};
  ASSERT_EQ(pipe(pipe_descriptors), 0);
  ScopedPosixDescriptor read_descriptor(pipe_descriptors[0]);
  ScopedPosixDescriptor write_descriptor(pipe_descriptors[1]);

  std::string probe_argument = std::string(kDescriptorProbePrefix) +
                               std::to_string(write_descriptor.get());
  iree_string_view_t arguments[] = {
      iree_make_string_view(probe_argument.data(), probe_argument.size()),
  };
  loom_tool_process_result_t result = {0};
  iree_status_t status = loom_tool_process_run(
      iree_make_string_view(g_executable_path.data(), g_executable_path.size()),
      /*search_path=*/false, arguments, IREE_ARRAYSIZE(arguments),
      iree_allocator_system(), &result);
  write_descriptor.reset();
  IREE_ASSERT_OK(status);
  EXPECT_TRUE(loom_tool_process_result_succeeded(&result));

  char marker = 0;
  ssize_t read_result = -1;
  do {
    read_result = read(read_descriptor.get(), &marker, sizeof(marker));
  } while (read_result < 0 && errno == EINTR);
  EXPECT_EQ(read_result, 0);
  loom_tool_process_result_deinitialize(&result, iree_allocator_system());
}

#endif  // POSIX platforms

#if defined(IREE_PLATFORM_WINDOWS) || defined(IREE_PLATFORM_LINUX) || \
    defined(IREE_PLATFORM_MACOS) || defined(IREE_PLATFORM_ANDROID)

TEST(ToolTempFileTest, ReportsDeleteFailures) {
  loom_tool_temp_file_t temp_file = {};
  IREE_ASSERT_OK(
      loom_tool_temp_file_initialize(IREE_SV("cleanup"), &temp_file));
  IREE_ASSERT_OK(loom_tool_temp_file_deinitialize(&temp_file));

  std::strcpy(temp_file.path, ".");
  iree_status_t status = loom_tool_temp_file_deinitialize(&temp_file);
  IREE_EXPECT_NOT_OK(status);
  EXPECT_EQ(temp_file.path[0], '\0');
}

TEST(ToolProcessTest, PreservesCompleteArgumentVector) {
#if defined(IREE_PLATFORM_WINDOWS)
  std::string executable_path = Win32ExecutablePathUtf8();
#else
  const std::string& executable_path = g_executable_path;
#endif  // IREE_PLATFORM_WINDOWS
  ASSERT_FALSE(executable_path.empty());

  iree_string_view_t arguments[IREE_ARRAYSIZE(kArgvProbeArguments) + 1];
  arguments[0] = iree_make_cstring_view(kArgvProbeArgument);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kArgvProbeArguments); ++i) {
    arguments[i + 1] = iree_make_cstring_view(kArgvProbeArguments[i]);
  }

  loom_tool_process_result_t result = {0};
  IREE_ASSERT_OK(loom_tool_process_run(
      iree_make_string_view(executable_path.data(), executable_path.size()),
      /*search_path=*/false, arguments, IREE_ARRAYSIZE(arguments),
      iree_allocator_system(), &result));
  EXPECT_TRUE(loom_tool_process_result_succeeded(&result));
  EXPECT_EQ(result.stdout_bytes.length, 0u);
  EXPECT_EQ(result.stderr_bytes.length, 0u);
  loom_tool_process_result_deinitialize(&result, iree_allocator_system());
}

#endif  // supported process platforms

}  // namespace

#if defined(IREE_PLATFORM_WINDOWS)

int wmain(int argc, wchar_t** argv) {
  static constexpr wchar_t kArgvProbeArgumentWide[] =
      L"--loom-process-argv-probe";
  static constexpr wchar_t kArgvProbeArgumentsWide[][32] = {
      L"",           L"space value",        L"tab\tvalue",   L"quote\"value",
      L"trailing\\", LR"(before\\\"quote)", L"space \x03C0",
  };
  if (argc == (int)IREE_ARRAYSIZE(kArgvProbeArgumentsWide) + 2 &&
      std::wcscmp(argv[1], kArgvProbeArgumentWide) == 0) {
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kArgvProbeArgumentsWide);
         ++i) {
      if (std::wcscmp(argv[i + 2], kArgvProbeArgumentsWide[i]) != 0) {
        return 1;
      }
    }
    return 0;
  }
  if (argc == 2 && std::wcscmp(argv[1], kUnicodeProbeArgumentWide) == 0) {
    DWORD bytes_written = 0;
    BOOL write_result =
        WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), kUnicodeProbeOutput,
                  (DWORD)sizeof(kUnicodeProbeOutput) - 1, &bytes_written, NULL);
    if (!write_result || bytes_written != sizeof(kUnicodeProbeOutput) - 1) {
      return 1;
    }
    return 0;
  }
  if (argc == 2 && std::wcsncmp(argv[1], kHandleProbePrefixWide,
                                std::wcslen(kHandleProbePrefixWide)) == 0) {
    wchar_t* end = NULL;
    unsigned long long value =
        std::wcstoull(argv[1] + std::wcslen(kHandleProbePrefixWide), &end, 10);
    if (end == NULL || *end != L'\0') {
      return 1;
    }
    HANDLE handle =
        reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(value));
    (void)SetEvent(handle);
    return 0;
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#else

int main(int argc, char** argv) {
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_MACOS) || \
    defined(IREE_PLATFORM_ANDROID)
  if (argc == (int)IREE_ARRAYSIZE(kArgvProbeArguments) + 2 &&
      std::strcmp(argv[1], kArgvProbeArgument) == 0) {
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kArgvProbeArguments); ++i) {
      if (std::strcmp(argv[i + 2], kArgvProbeArguments[i]) != 0) {
        return 1;
      }
    }
    return 0;
  }
  if (argc == 2 && strncmp(argv[1], kDescriptorProbePrefix,
                           strlen(kDescriptorProbePrefix)) == 0) {
    char* end = NULL;
    long fd = strtol(argv[1] + strlen(kDescriptorProbePrefix), &end, 10);
    if (end == NULL || *end != '\0' || fd < 0 || fd > INT_MAX) {
      return 1;
    }
    char marker = 'x';
    ssize_t write_result = write(static_cast<int>(fd), &marker, sizeof(marker));
    (void)write_result;
    return 0;
  }
  g_executable_path = argv[0];
#endif  // POSIX platforms
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif  // IREE_PLATFORM_WINDOWS
