// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The compiler dependency records its arguments and inputs. This fixture tests
// build graph ownership and freshness, without duplicating compiler semantics.
#include <stdio.h>
#include <string.h>

static const char* Value(const char* argument, const char* prefix) {
  size_t length = strlen(prefix);
  return strncmp(argument, prefix, length) == 0 ? argument + length : NULL;
}

static int CopyInput(FILE* output, const char* path) {
  FILE* input = fopen(path, "rb");
  if (!input) {
    return 1;
  }
  char bytes[4096];
  size_t count;
  while ((count = fread(bytes, 1, sizeof(bytes), input)) != 0) {
    if (fwrite(bytes, 1, count, output) != count) {
      fclose(input);
      return 1;
    }
  }
  int failed = ferror(input);
  return fclose(input) != 0 || failed;
}

int main(int argc, char** argv) {
  const char* path = NULL;
  for (int i = 1; i < argc; ++i) {
    const char* value = Value(argv[i], "--output=");
    if (value) {
      path = value;
    }
  }
  if (!path) {
    return 1;
  }
  FILE* output = fopen(path, "wb");
  if (!output) {
    return 1;
  }
  int failed = 0;
  for (int i = 1; i < argc && !failed; ++i) {
    failed = fprintf(output, "%s\n", argv[i]) < 0;
    const char* input = argv[i][0] != '-' ? argv[i] : NULL;
    const char* prefixes[] = {"--library=", "--root-library=",
                              "--test-library=", "--transitive-library="};
    for (size_t j = 0; j < sizeof(prefixes) / sizeof(prefixes[0]); ++j) {
      const char* value = Value(argv[i], prefixes[j]);
      if (value) {
        input = value;
      }
    }
    if (input && !failed) {
      failed = CopyInput(output, input);
    }
  }
  failed = fclose(output) != 0 || failed;
  for (int i = 1; i < argc && !failed; ++i) {
    const char* report = Value(argv[i], "--dependency-report=");
    if (!report) {
      report = Value(argv[i], "--compile-report-output=");
    }
    if (report) {
      FILE* file = fopen(report, "wb");
      if (!file) {
        return 1;
      }
      failed = fputs("{}\n", file) < 0;
      failed = fclose(file) != 0 || failed;
    }
  }
  return failed;
}
