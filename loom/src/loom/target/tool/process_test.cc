// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/process.h"

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
#endif  // IREE_PLATFORM_WINDOWS

namespace {

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
    executable_path_ = directory_path_;
    executable_path_.append(L"\\process_\x03C0.exe");
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
      std::string_view(result.stdout_text.data, result.stdout_text.length),
      kUnicodeProbeOutput);
  EXPECT_EQ(result.stderr_text.length, 0u);
  loom_tool_process_result_deinitialize(&result, iree_allocator_system());

  loom_tool_temp_file_t temp_file;
  IREE_ASSERT_OK(
      loom_tool_temp_file_initialize(IREE_SV("unicode"), &temp_file));
  EXPECT_NE(std::string_view(temp_file.path).find("\xCF\x80"),
            std::string_view::npos);
  loom_tool_temp_file_deinitialize(&temp_file);
}

#endif  // IREE_PLATFORM_WINDOWS

}  // namespace

#if defined(IREE_PLATFORM_WINDOWS)

int wmain(int argc, wchar_t** argv) {
  if (argc == 2 && std::wcscmp(argv[1], kUnicodeProbeArgumentWide) == 0) {
    DWORD bytes_written = 0;
    BOOL write_result =
        WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), kUnicodeProbeOutput,
                  (DWORD)sizeof(kUnicodeProbeOutput) - 1, &bytes_written, NULL);
    return write_result && bytes_written == sizeof(kUnicodeProbeOutput) - 1 ? 0
                                                                            : 1;
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#else

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif  // IREE_PLATFORM_WINDOWS
