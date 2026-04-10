// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <hrx_runtime.h>

int main(void) {
  int major = -1;
  int minor = -1;
  int patch = -1;
  hrx_runtime_version(&major, &minor, &patch);
  return major == HRX_VERSION_MAJOR && minor == HRX_VERSION_MINOR &&
          patch == HRX_VERSION_PATCH
      ? 0
      : 1;
}
