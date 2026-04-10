// Auto-generated passthrough stubs for all HIP symbols
// Generated from /opt/rocm/lib/libamdhip64.so
//
// Each function dynamically forwards to the real HIP library.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static void *g_real_lib = NULL;

__attribute__((constructor)) static void stubs_init(void) {
  const char *lib_path = getenv("HIP_PASSTHROUGH_BACKEND_LIB");
  if (!lib_path)
    lib_path = "/opt/rocm/lib/libamdhip64.so.bak";
  g_real_lib = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
  if (!g_real_lib) {
    fprintf(stderr, "passthrough: failed to load backend: %s\n", dlerror());
  }
}

static void *get_real_sym(const char *name) {
  if (!g_real_lib)
    return NULL;
  return dlsym(g_real_lib, name);
}

// Generic variadic forwarder - use for functions we don't know the signature of
#define DEFINE_STUB(name)                                                      \
  __attribute__((visibility("default"))) void *name(void) {                    \
    static void *(*real_fn)(void) = NULL;                                      \
    if (!real_fn)                                                              \
      real_fn = get_real_sym(#name);                                           \
    if (!real_fn) {                                                            \
      fprintf(stderr, "passthrough: missing: " #name "\n");                    \
      return NULL;                                                             \
    }                                                                          \
    return real_fn();                                                          \
  }

// __hipGetPCH@hip_4.2
__attribute__((visibility("default"))) int __hipGetPCH() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("__hipGetPCH");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: __hipGetPCH\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver __hipGetPCH, __hipGetPCH@hip_4.2");

// __hipPopCallConfiguration@hip_4.2
__attribute__((visibility("default"))) int __hipPopCallConfiguration() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("__hipPopCallConfiguration");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: __hipPopCallConfiguration\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver __hipPopCallConfiguration, __hipPopCallConfiguration@hip_4.2");

// __hipPushCallConfiguration@hip_4.2
__attribute__((visibility("default"))) int __hipPushCallConfiguration() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("__hipPushCallConfiguration");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: __hipPushCallConfiguration\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver __hipPushCallConfiguration, __hipPushCallConfiguration@hip_4.2");

// __hipRegisterFatBinary@hip_4.2
__attribute__((visibility("default"))) int __hipRegisterFatBinary() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("__hipRegisterFatBinary");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: __hipRegisterFatBinary\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver __hipRegisterFatBinary, __hipRegisterFatBinary@hip_4.2");

// __hipRegisterFunction@hip_4.2
__attribute__((visibility("default"))) int __hipRegisterFunction() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("__hipRegisterFunction");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: __hipRegisterFunction\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver __hipRegisterFunction, __hipRegisterFunction@hip_4.2");

// __hipRegisterManagedVar@hip_4.2
__attribute__((visibility("default"))) int __hipRegisterManagedVar() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("__hipRegisterManagedVar");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: __hipRegisterManagedVar\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver __hipRegisterManagedVar, __hipRegisterManagedVar@hip_4.2");

// __hipRegisterSurface@hip_4.2
__attribute__((visibility("default"))) int __hipRegisterSurface() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("__hipRegisterSurface");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: __hipRegisterSurface\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver __hipRegisterSurface, __hipRegisterSurface@hip_4.2");

// __hipRegisterTexture@hip_4.2
__attribute__((visibility("default"))) int __hipRegisterTexture() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("__hipRegisterTexture");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: __hipRegisterTexture\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver __hipRegisterTexture, __hipRegisterTexture@hip_4.2");

// __hipRegisterVar@hip_4.2
__attribute__((visibility("default"))) int __hipRegisterVar() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("__hipRegisterVar");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: __hipRegisterVar\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver __hipRegisterVar, __hipRegisterVar@hip_4.2");

// __hipUnregisterFatBinary@hip_4.2
__attribute__((visibility("default"))) int __hipUnregisterFatBinary() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("__hipUnregisterFatBinary");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: __hipUnregisterFatBinary\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver __hipUnregisterFatBinary, __hipUnregisterFatBinary@hip_4.2");

// amd_dbgapi_get_build_id@hip_4.5
__attribute__((visibility("default"))) int amd_dbgapi_get_build_id() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("amd_dbgapi_get_build_id");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: amd_dbgapi_get_build_id\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver amd_dbgapi_get_build_id, amd_dbgapi_get_build_id@hip_4.5");

// amd_dbgapi_get_build_name@hip_4.5
__attribute__((visibility("default"))) int amd_dbgapi_get_build_name() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("amd_dbgapi_get_build_name");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: amd_dbgapi_get_build_name\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver amd_dbgapi_get_build_name, amd_dbgapi_get_build_name@hip_4.5");

// amd_dbgapi_get_git_hash@hip_4.5
__attribute__((visibility("default"))) int amd_dbgapi_get_git_hash() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("amd_dbgapi_get_git_hash");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: amd_dbgapi_get_git_hash\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver amd_dbgapi_get_git_hash, amd_dbgapi_get_git_hash@hip_4.5");

// hipApiName@hip_4.2
__attribute__((visibility("default"))) int hipApiName() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipApiName");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipApiName\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipApiName, hipApiName@hip_4.2");

// hipArray3DCreate@hip_4.2
__attribute__((visibility("default"))) int hipArray3DCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipArray3DCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipArray3DCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipArray3DCreate, hipArray3DCreate@hip_4.2");

// hipArray3DGetDescriptor@hip_5.6
__attribute__((visibility("default"))) int hipArray3DGetDescriptor() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipArray3DGetDescriptor");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipArray3DGetDescriptor\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipArray3DGetDescriptor, hipArray3DGetDescriptor@hip_5.6");

// hipArrayCreate@hip_4.2
__attribute__((visibility("default"))) int hipArrayCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipArrayCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipArrayCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipArrayCreate, hipArrayCreate@hip_4.2");

// hipArrayDestroy@hip_4.3
__attribute__((visibility("default"))) int hipArrayDestroy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipArrayDestroy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipArrayDestroy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipArrayDestroy, hipArrayDestroy@hip_4.3");

// hipArrayGetDescriptor@hip_5.6
__attribute__((visibility("default"))) int hipArrayGetDescriptor() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipArrayGetDescriptor");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipArrayGetDescriptor\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipArrayGetDescriptor, hipArrayGetDescriptor@hip_5.6");

// hipArrayGetInfo@hip_5.6
__attribute__((visibility("default"))) int hipArrayGetInfo() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipArrayGetInfo");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipArrayGetInfo\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipArrayGetInfo, hipArrayGetInfo@hip_5.6");

// hipBindTexture@hip_4.2
__attribute__((visibility("default"))) int hipBindTexture() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipBindTexture");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipBindTexture\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipBindTexture, hipBindTexture@hip_4.2");

// hipBindTexture2D@hip_4.2
__attribute__((visibility("default"))) int hipBindTexture2D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipBindTexture2D");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipBindTexture2D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipBindTexture2D, hipBindTexture2D@hip_4.2");

// hipBindTextureToArray@hip_4.2
__attribute__((visibility("default"))) int hipBindTextureToArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipBindTextureToArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipBindTextureToArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipBindTextureToArray, hipBindTextureToArray@hip_4.2");

// hipBindTextureToMipmappedArray@hip_4.2
__attribute__((visibility("default"))) int hipBindTextureToMipmappedArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipBindTextureToMipmappedArray");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipBindTextureToMipmappedArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipBindTextureToMipmappedArray, "
        "hipBindTextureToMipmappedArray@hip_4.2");

// hipChooseDevice@hip_4.2
__attribute__((visibility("default"))) int hipChooseDevice() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipChooseDevice");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipChooseDevice\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipChooseDevice, hipChooseDevice@hip_4.2");

// hipChooseDeviceR0000@hip_4.2
__attribute__((visibility("default"))) int hipChooseDeviceR0000() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipChooseDeviceR0000");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipChooseDeviceR0000\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipChooseDeviceR0000, hipChooseDeviceR0000@hip_4.2");

// hipChooseDeviceR0600@hip_6.0
__attribute__((visibility("default"))) int hipChooseDeviceR0600() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipChooseDeviceR0600");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipChooseDeviceR0600\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipChooseDeviceR0600, hipChooseDeviceR0600@hip_6.0");

// hipConfigureCall@hip_4.2
__attribute__((visibility("default"))) int hipConfigureCall() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipConfigureCall");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipConfigureCall\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipConfigureCall, hipConfigureCall@hip_4.2");

// hipCreateChannelDesc@hip_4.2
__attribute__((visibility("default"))) int hipCreateChannelDesc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCreateChannelDesc");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCreateChannelDesc\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCreateChannelDesc, hipCreateChannelDesc@hip_4.2");

// hipCreateSurfaceObject@hip_4.2
__attribute__((visibility("default"))) int hipCreateSurfaceObject() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCreateSurfaceObject");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCreateSurfaceObject\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCreateSurfaceObject, hipCreateSurfaceObject@hip_4.2");

// hipCreateTextureObject@hip_4.2
__attribute__((visibility("default"))) int hipCreateTextureObject() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCreateTextureObject");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCreateTextureObject\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCreateTextureObject, hipCreateTextureObject@hip_4.2");

// hipCtxCreate@hip_4.2
__attribute__((visibility("default"))) int hipCtxCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxCreate, hipCtxCreate@hip_4.2");

// hipCtxDestroy@hip_4.2
__attribute__((visibility("default"))) int hipCtxDestroy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxDestroy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxDestroy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxDestroy, hipCtxDestroy@hip_4.2");

// hipCtxDisablePeerAccess@hip_4.2
__attribute__((visibility("default"))) int hipCtxDisablePeerAccess() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxDisablePeerAccess");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxDisablePeerAccess\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxDisablePeerAccess, hipCtxDisablePeerAccess@hip_4.2");

// hipCtxEnablePeerAccess@hip_4.2
__attribute__((visibility("default"))) int hipCtxEnablePeerAccess() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxEnablePeerAccess");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxEnablePeerAccess\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxEnablePeerAccess, hipCtxEnablePeerAccess@hip_4.2");

// hipCtxGetApiVersion@hip_4.2
__attribute__((visibility("default"))) int hipCtxGetApiVersion() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxGetApiVersion");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxGetApiVersion\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxGetApiVersion, hipCtxGetApiVersion@hip_4.2");

// hipCtxGetCacheConfig@hip_4.2
__attribute__((visibility("default"))) int hipCtxGetCacheConfig() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxGetCacheConfig");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxGetCacheConfig\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxGetCacheConfig, hipCtxGetCacheConfig@hip_4.2");

// hipCtxGetCurrent@hip_4.2
__attribute__((visibility("default"))) int hipCtxGetCurrent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxGetCurrent");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxGetCurrent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxGetCurrent, hipCtxGetCurrent@hip_4.2");

// hipCtxGetDevice@hip_4.2
__attribute__((visibility("default"))) int hipCtxGetDevice() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxGetDevice");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxGetDevice\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxGetDevice, hipCtxGetDevice@hip_4.2");

// hipCtxGetFlags@hip_4.2
__attribute__((visibility("default"))) int hipCtxGetFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxGetFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxGetFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxGetFlags, hipCtxGetFlags@hip_4.2");

// hipCtxGetSharedMemConfig@hip_4.2
__attribute__((visibility("default"))) int hipCtxGetSharedMemConfig() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxGetSharedMemConfig");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxGetSharedMemConfig\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxGetSharedMemConfig, hipCtxGetSharedMemConfig@hip_4.2");

// hipCtxPopCurrent@hip_4.2
__attribute__((visibility("default"))) int hipCtxPopCurrent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxPopCurrent");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxPopCurrent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxPopCurrent, hipCtxPopCurrent@hip_4.2");

// hipCtxPushCurrent@hip_4.2
__attribute__((visibility("default"))) int hipCtxPushCurrent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxPushCurrent");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxPushCurrent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxPushCurrent, hipCtxPushCurrent@hip_4.2");

// hipCtxSetCacheConfig@hip_4.2
__attribute__((visibility("default"))) int hipCtxSetCacheConfig() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxSetCacheConfig");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxSetCacheConfig\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxSetCacheConfig, hipCtxSetCacheConfig@hip_4.2");

// hipCtxSetCurrent@hip_4.2
__attribute__((visibility("default"))) int hipCtxSetCurrent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxSetCurrent");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxSetCurrent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxSetCurrent, hipCtxSetCurrent@hip_4.2");

// hipCtxSetSharedMemConfig@hip_4.2
__attribute__((visibility("default"))) int hipCtxSetSharedMemConfig() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxSetSharedMemConfig");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxSetSharedMemConfig\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxSetSharedMemConfig, hipCtxSetSharedMemConfig@hip_4.2");

// hipCtxSynchronize@hip_4.2
__attribute__((visibility("default"))) int hipCtxSynchronize() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipCtxSynchronize");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipCtxSynchronize\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipCtxSynchronize, hipCtxSynchronize@hip_4.2");

// hipDestroyExternalMemory@hip_4.3
__attribute__((visibility("default"))) int hipDestroyExternalMemory() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDestroyExternalMemory");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDestroyExternalMemory\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDestroyExternalMemory, hipDestroyExternalMemory@hip_4.3");

// hipDestroyExternalSemaphore@hip_4.3
__attribute__((visibility("default"))) int hipDestroyExternalSemaphore() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDestroyExternalSemaphore");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDestroyExternalSemaphore\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipDestroyExternalSemaphore, hipDestroyExternalSemaphore@hip_4.3");

// hipDestroySurfaceObject@hip_4.2
__attribute__((visibility("default"))) int hipDestroySurfaceObject() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDestroySurfaceObject");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDestroySurfaceObject\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDestroySurfaceObject, hipDestroySurfaceObject@hip_4.2");

// hipDestroyTextureObject@hip_4.2
__attribute__((visibility("default"))) int hipDestroyTextureObject() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDestroyTextureObject");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDestroyTextureObject\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDestroyTextureObject, hipDestroyTextureObject@hip_4.2");

// hipDeviceCanAccessPeer@hip_4.2
__attribute__((visibility("default"))) int hipDeviceCanAccessPeer() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceCanAccessPeer");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceCanAccessPeer\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceCanAccessPeer, hipDeviceCanAccessPeer@hip_4.2");

// hipDeviceComputeCapability@hip_4.2
__attribute__((visibility("default"))) int hipDeviceComputeCapability() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceComputeCapability");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDeviceComputeCapability\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipDeviceComputeCapability, hipDeviceComputeCapability@hip_4.2");

// hipDeviceDisablePeerAccess@hip_4.2
__attribute__((visibility("default"))) int hipDeviceDisablePeerAccess() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceDisablePeerAccess");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDeviceDisablePeerAccess\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipDeviceDisablePeerAccess, hipDeviceDisablePeerAccess@hip_4.2");

// hipDeviceEnablePeerAccess@hip_4.2
__attribute__((visibility("default"))) int hipDeviceEnablePeerAccess() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceEnablePeerAccess");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceEnablePeerAccess\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceEnablePeerAccess, hipDeviceEnablePeerAccess@hip_4.2");

// hipDeviceGet@hip_4.2
__attribute__((visibility("default"))) int hipDeviceGet() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGet");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGet\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGet, hipDeviceGet@hip_4.2");

// hipDeviceGetAttribute@hip_4.2
__attribute__((visibility("default"))) int hipDeviceGetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetAttribute, hipDeviceGetAttribute@hip_4.2");

// hipDeviceGetByPCIBusId@hip_4.2
__attribute__((visibility("default"))) int hipDeviceGetByPCIBusId() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetByPCIBusId");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGetByPCIBusId\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetByPCIBusId, hipDeviceGetByPCIBusId@hip_4.2");

// hipDeviceGetCacheConfig@hip_4.2
__attribute__((visibility("default"))) int hipDeviceGetCacheConfig() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetCacheConfig");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGetCacheConfig\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetCacheConfig, hipDeviceGetCacheConfig@hip_4.2");

// hipDeviceGetDefaultMemPool@hip_5.1
__attribute__((visibility("default"))) int hipDeviceGetDefaultMemPool() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetDefaultMemPool");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDeviceGetDefaultMemPool\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipDeviceGetDefaultMemPool, hipDeviceGetDefaultMemPool@hip_5.1");

// hipDeviceGetGraphMemAttribute@hip_4.5
__attribute__((visibility("default"))) int hipDeviceGetGraphMemAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetGraphMemAttribute");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDeviceGetGraphMemAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetGraphMemAttribute, "
        "hipDeviceGetGraphMemAttribute@hip_4.5");

// hipDeviceGetLimit@hip_4.2
__attribute__((visibility("default"))) int hipDeviceGetLimit() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetLimit");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGetLimit\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetLimit, hipDeviceGetLimit@hip_4.2");

// hipDeviceGetMemPool@hip_5.1
__attribute__((visibility("default"))) int hipDeviceGetMemPool() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetMemPool");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGetMemPool\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetMemPool, hipDeviceGetMemPool@hip_5.1");

// hipDeviceGetName@hip_4.2
__attribute__((visibility("default"))) int hipDeviceGetName() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetName");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGetName\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetName, hipDeviceGetName@hip_4.2");

// hipDeviceGetP2PAttribute@hip_4.2
__attribute__((visibility("default"))) int hipDeviceGetP2PAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetP2PAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGetP2PAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetP2PAttribute, hipDeviceGetP2PAttribute@hip_4.2");

// hipDeviceGetPCIBusId@hip_4.2
__attribute__((visibility("default"))) int hipDeviceGetPCIBusId() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetPCIBusId");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGetPCIBusId\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetPCIBusId, hipDeviceGetPCIBusId@hip_4.2");

// hipDeviceGetSharedMemConfig@hip_4.2
__attribute__((visibility("default"))) int hipDeviceGetSharedMemConfig() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetSharedMemConfig");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDeviceGetSharedMemConfig\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipDeviceGetSharedMemConfig, hipDeviceGetSharedMemConfig@hip_4.2");

// hipDeviceGetStreamPriorityRange@hip_4.2
__attribute__((visibility("default"))) int hipDeviceGetStreamPriorityRange() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetStreamPriorityRange");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDeviceGetStreamPriorityRange\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetStreamPriorityRange, "
        "hipDeviceGetStreamPriorityRange@hip_4.2");

// hipDeviceGetTexture1DLinearMaxWidth@hip_4.2
__attribute__((visibility("default"))) int
hipDeviceGetTexture1DLinearMaxWidth() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetTexture1DLinearMaxWidth");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipDeviceGetTexture1DLinearMaxWidth\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetTexture1DLinearMaxWidth, "
        "hipDeviceGetTexture1DLinearMaxWidth@hip_4.2");

// hipDeviceGetUuid@hip_5.1
__attribute__((visibility("default"))) int hipDeviceGetUuid() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGetUuid");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGetUuid\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGetUuid, hipDeviceGetUuid@hip_5.1");

// hipDeviceGraphMemTrim@hip_4.5
__attribute__((visibility("default"))) int hipDeviceGraphMemTrim() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceGraphMemTrim");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceGraphMemTrim\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceGraphMemTrim, hipDeviceGraphMemTrim@hip_4.5");

// hipDevicePrimaryCtxGetState@hip_4.2
__attribute__((visibility("default"))) int hipDevicePrimaryCtxGetState() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDevicePrimaryCtxGetState");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDevicePrimaryCtxGetState\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipDevicePrimaryCtxGetState, hipDevicePrimaryCtxGetState@hip_4.2");

// hipDevicePrimaryCtxRelease@hip_4.2
__attribute__((visibility("default"))) int hipDevicePrimaryCtxRelease() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDevicePrimaryCtxRelease");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDevicePrimaryCtxRelease\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipDevicePrimaryCtxRelease, hipDevicePrimaryCtxRelease@hip_4.2");

// hipDevicePrimaryCtxReset@hip_4.2
__attribute__((visibility("default"))) int hipDevicePrimaryCtxReset() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDevicePrimaryCtxReset");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDevicePrimaryCtxReset\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDevicePrimaryCtxReset, hipDevicePrimaryCtxReset@hip_4.2");

// hipDevicePrimaryCtxRetain@hip_4.2
__attribute__((visibility("default"))) int hipDevicePrimaryCtxRetain() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDevicePrimaryCtxRetain");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDevicePrimaryCtxRetain\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDevicePrimaryCtxRetain, hipDevicePrimaryCtxRetain@hip_4.2");

// hipDevicePrimaryCtxSetFlags@hip_4.2
__attribute__((visibility("default"))) int hipDevicePrimaryCtxSetFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDevicePrimaryCtxSetFlags");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDevicePrimaryCtxSetFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipDevicePrimaryCtxSetFlags, hipDevicePrimaryCtxSetFlags@hip_4.2");

// hipDeviceReset@hip_4.2
__attribute__((visibility("default"))) int hipDeviceReset() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceReset");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceReset\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceReset, hipDeviceReset@hip_4.2");

// hipDeviceSetCacheConfig@hip_4.2
__attribute__((visibility("default"))) int hipDeviceSetCacheConfig() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceSetCacheConfig");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceSetCacheConfig\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceSetCacheConfig, hipDeviceSetCacheConfig@hip_4.2");

// hipDeviceSetGraphMemAttribute@hip_4.5
__attribute__((visibility("default"))) int hipDeviceSetGraphMemAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceSetGraphMemAttribute");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDeviceSetGraphMemAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceSetGraphMemAttribute, "
        "hipDeviceSetGraphMemAttribute@hip_4.5");

// hipDeviceSetLimit@hip_5.3
__attribute__((visibility("default"))) int hipDeviceSetLimit() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceSetLimit");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceSetLimit\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceSetLimit, hipDeviceSetLimit@hip_5.3");

// hipDeviceSetMemPool@hip_5.1
__attribute__((visibility("default"))) int hipDeviceSetMemPool() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceSetMemPool");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceSetMemPool\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceSetMemPool, hipDeviceSetMemPool@hip_5.1");

// hipDeviceSetSharedMemConfig@hip_4.2
__attribute__((visibility("default"))) int hipDeviceSetSharedMemConfig() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceSetSharedMemConfig");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDeviceSetSharedMemConfig\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipDeviceSetSharedMemConfig, hipDeviceSetSharedMemConfig@hip_4.2");

// hipDeviceSynchronize@hip_4.2
__attribute__((visibility("default"))) int hipDeviceSynchronize() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceSynchronize");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceSynchronize\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceSynchronize, hipDeviceSynchronize@hip_4.2");

// hipDeviceTotalMem@hip_4.2
__attribute__((visibility("default"))) int hipDeviceTotalMem() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDeviceTotalMem");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDeviceTotalMem\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDeviceTotalMem, hipDeviceTotalMem@hip_4.2");

// hipDriverGetVersion@hip_4.2
__attribute__((visibility("default"))) int hipDriverGetVersion() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDriverGetVersion");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDriverGetVersion\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDriverGetVersion, hipDriverGetVersion@hip_4.2");

// hipDrvGetErrorName@hip_5.3
__attribute__((visibility("default"))) int hipDrvGetErrorName() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvGetErrorName");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDrvGetErrorName\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvGetErrorName, hipDrvGetErrorName@hip_5.3");

// hipDrvGetErrorString@hip_5.3
__attribute__((visibility("default"))) int hipDrvGetErrorString() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvGetErrorString");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDrvGetErrorString\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvGetErrorString, hipDrvGetErrorString@hip_5.3");

// hipDrvGraphAddMemFreeNode@hip_6.2
__attribute__((visibility("default"))) int hipDrvGraphAddMemFreeNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvGraphAddMemFreeNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDrvGraphAddMemFreeNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvGraphAddMemFreeNode, hipDrvGraphAddMemFreeNode@hip_6.2");

// hipDrvGraphAddMemcpyNode@hip_5.6
__attribute__((visibility("default"))) int hipDrvGraphAddMemcpyNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvGraphAddMemcpyNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDrvGraphAddMemcpyNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvGraphAddMemcpyNode, hipDrvGraphAddMemcpyNode@hip_5.6");

// hipDrvGraphAddMemsetNode@hip_5.6
__attribute__((visibility("default"))) int hipDrvGraphAddMemsetNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvGraphAddMemsetNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDrvGraphAddMemsetNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvGraphAddMemsetNode, hipDrvGraphAddMemsetNode@hip_5.6");

// hipDrvGraphExecMemcpyNodeSetParams@hip_6.2
__attribute__((visibility("default"))) int
hipDrvGraphExecMemcpyNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvGraphExecMemcpyNodeSetParams");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipDrvGraphExecMemcpyNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvGraphExecMemcpyNodeSetParams, "
        "hipDrvGraphExecMemcpyNodeSetParams@hip_6.2");

// hipDrvGraphExecMemsetNodeSetParams@hip_6.2
__attribute__((visibility("default"))) int
hipDrvGraphExecMemsetNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvGraphExecMemsetNodeSetParams");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipDrvGraphExecMemsetNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvGraphExecMemsetNodeSetParams, "
        "hipDrvGraphExecMemsetNodeSetParams@hip_6.2");

// hipDrvGraphMemcpyNodeGetParams@hip_6.0
__attribute__((visibility("default"))) int hipDrvGraphMemcpyNodeGetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvGraphMemcpyNodeGetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDrvGraphMemcpyNodeGetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvGraphMemcpyNodeGetParams, "
        "hipDrvGraphMemcpyNodeGetParams@hip_6.0");

// hipDrvGraphMemcpyNodeSetParams@hip_6.0
__attribute__((visibility("default"))) int hipDrvGraphMemcpyNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvGraphMemcpyNodeSetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDrvGraphMemcpyNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvGraphMemcpyNodeSetParams, "
        "hipDrvGraphMemcpyNodeSetParams@hip_6.0");

// hipDrvLaunchKernelEx@hip_6.5
__attribute__((visibility("default"))) int hipDrvLaunchKernelEx() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvLaunchKernelEx");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDrvLaunchKernelEx\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvLaunchKernelEx, hipDrvLaunchKernelEx@hip_6.5");

// hipDrvMemcpy2DUnaligned@hip_4.3
__attribute__((visibility("default"))) int hipDrvMemcpy2DUnaligned() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvMemcpy2DUnaligned");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDrvMemcpy2DUnaligned\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvMemcpy2DUnaligned, hipDrvMemcpy2DUnaligned@hip_4.3");

// hipDrvMemcpy3D@hip_4.2
__attribute__((visibility("default"))) int hipDrvMemcpy3D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvMemcpy3D");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDrvMemcpy3D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvMemcpy3D, hipDrvMemcpy3D@hip_4.2");

// hipDrvMemcpy3DAsync@hip_4.2
__attribute__((visibility("default"))) int hipDrvMemcpy3DAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvMemcpy3DAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipDrvMemcpy3DAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipDrvMemcpy3DAsync, hipDrvMemcpy3DAsync@hip_4.2");

// hipDrvPointerGetAttributes@hip_5.0
__attribute__((visibility("default"))) int hipDrvPointerGetAttributes() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipDrvPointerGetAttributes");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipDrvPointerGetAttributes\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipDrvPointerGetAttributes, hipDrvPointerGetAttributes@hip_5.0");

// hipEventCreate@hip_4.2
__attribute__((visibility("default"))) int hipEventCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipEventCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipEventCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipEventCreate, hipEventCreate@hip_4.2");

// hipEventCreateWithFlags@hip_4.2
__attribute__((visibility("default"))) int hipEventCreateWithFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipEventCreateWithFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipEventCreateWithFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipEventCreateWithFlags, hipEventCreateWithFlags@hip_4.2");

// hipEventDestroy@hip_4.2
__attribute__((visibility("default"))) int hipEventDestroy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipEventDestroy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipEventDestroy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipEventDestroy, hipEventDestroy@hip_4.2");

// hipEventElapsedTime@hip_4.2
__attribute__((visibility("default"))) int hipEventElapsedTime() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipEventElapsedTime");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipEventElapsedTime\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipEventElapsedTime, hipEventElapsedTime@hip_4.2");

// hipEventQuery@hip_4.2
__attribute__((visibility("default"))) int hipEventQuery() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipEventQuery");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipEventQuery\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipEventQuery, hipEventQuery@hip_4.2");

// hipEventRecord@hip_4.2
__attribute__((visibility("default"))) int hipEventRecord() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipEventRecord");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipEventRecord\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipEventRecord, hipEventRecord@hip_4.2");

// hipEventRecordWithFlags@hip_6.4
__attribute__((visibility("default"))) int hipEventRecordWithFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipEventRecordWithFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipEventRecordWithFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipEventRecordWithFlags, hipEventRecordWithFlags@hip_6.4");

// hipEventRecord_spt@hip_5.2
__attribute__((visibility("default"))) int hipEventRecord_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipEventRecord_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipEventRecord_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipEventRecord_spt, hipEventRecord_spt@hip_5.2");

// hipEventSynchronize@hip_4.2
__attribute__((visibility("default"))) int hipEventSynchronize() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipEventSynchronize");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipEventSynchronize\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipEventSynchronize, hipEventSynchronize@hip_4.2");

// hipExtGetLastError@hip_6.0
__attribute__((visibility("default"))) int hipExtGetLastError() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipExtGetLastError");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipExtGetLastError\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipExtGetLastError, hipExtGetLastError@hip_6.0");

// hipExtGetLinkTypeAndHopCount@hip_4.2
__attribute__((visibility("default"))) int hipExtGetLinkTypeAndHopCount() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipExtGetLinkTypeAndHopCount");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipExtGetLinkTypeAndHopCount\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipExtGetLinkTypeAndHopCount, "
        "hipExtGetLinkTypeAndHopCount@hip_4.2");

// hipExtLaunchKernel@hip_4.2
__attribute__((visibility("default"))) int hipExtLaunchKernel() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipExtLaunchKernel");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipExtLaunchKernel\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipExtLaunchKernel, hipExtLaunchKernel@hip_4.2");

// hipExtLaunchMultiKernelMultiDevice@hip_4.2
__attribute__((visibility("default"))) int
hipExtLaunchMultiKernelMultiDevice() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipExtLaunchMultiKernelMultiDevice");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipExtLaunchMultiKernelMultiDevice\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipExtLaunchMultiKernelMultiDevice, "
        "hipExtLaunchMultiKernelMultiDevice@hip_4.2");

// hipExtMallocWithFlags@hip_4.2
__attribute__((visibility("default"))) int hipExtMallocWithFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipExtMallocWithFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipExtMallocWithFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipExtMallocWithFlags, hipExtMallocWithFlags@hip_4.2");

// hipExtModuleLaunchKernel@hip_4.2
__attribute__((visibility("default"))) int hipExtModuleLaunchKernel() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipExtModuleLaunchKernel");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipExtModuleLaunchKernel\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipExtModuleLaunchKernel, hipExtModuleLaunchKernel@hip_4.2");

// hipExtStreamCreateWithCUMask@hip_4.2
__attribute__((visibility("default"))) int hipExtStreamCreateWithCUMask() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipExtStreamCreateWithCUMask");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipExtStreamCreateWithCUMask\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipExtStreamCreateWithCUMask, "
        "hipExtStreamCreateWithCUMask@hip_4.2");

// hipExtStreamGetCUMask@hip_4.2
__attribute__((visibility("default"))) int hipExtStreamGetCUMask() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipExtStreamGetCUMask");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipExtStreamGetCUMask\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipExtStreamGetCUMask, hipExtStreamGetCUMask@hip_4.2");

// hipExternalMemoryGetMappedBuffer@hip_4.3
__attribute__((visibility("default"))) int hipExternalMemoryGetMappedBuffer() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipExternalMemoryGetMappedBuffer");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipExternalMemoryGetMappedBuffer\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipExternalMemoryGetMappedBuffer, "
        "hipExternalMemoryGetMappedBuffer@hip_4.3");

// hipFree@hip_4.2
__attribute__((visibility("default"))) int hipFree() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipFree");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipFree\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipFree, hipFree@hip_4.2");

// hipFreeArray@hip_4.2
__attribute__((visibility("default"))) int hipFreeArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipFreeArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipFreeArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipFreeArray, hipFreeArray@hip_4.2");

// hipFreeAsync@hip_5.1
__attribute__((visibility("default"))) int hipFreeAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipFreeAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipFreeAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipFreeAsync, hipFreeAsync@hip_5.1");

// hipFreeHost@hip_4.2
__attribute__((visibility("default"))) int hipFreeHost() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipFreeHost");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipFreeHost\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipFreeHost, hipFreeHost@hip_4.2");

// hipFreeMipmappedArray@hip_4.2
__attribute__((visibility("default"))) int hipFreeMipmappedArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipFreeMipmappedArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipFreeMipmappedArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipFreeMipmappedArray, hipFreeMipmappedArray@hip_4.2");

// hipFuncGetAttribute@hip_4.2
__attribute__((visibility("default"))) int hipFuncGetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipFuncGetAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipFuncGetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipFuncGetAttribute, hipFuncGetAttribute@hip_4.2");

// hipFuncGetAttributes@hip_4.2
__attribute__((visibility("default"))) int hipFuncGetAttributes() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipFuncGetAttributes");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipFuncGetAttributes\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipFuncGetAttributes, hipFuncGetAttributes@hip_4.2");

// hipFuncSetAttribute@hip_4.2
__attribute__((visibility("default"))) int hipFuncSetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipFuncSetAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipFuncSetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipFuncSetAttribute, hipFuncSetAttribute@hip_4.2");

// hipFuncSetCacheConfig@hip_4.2
__attribute__((visibility("default"))) int hipFuncSetCacheConfig() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipFuncSetCacheConfig");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipFuncSetCacheConfig\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipFuncSetCacheConfig, hipFuncSetCacheConfig@hip_4.2");

// hipFuncSetSharedMemConfig@hip_4.2
__attribute__((visibility("default"))) int hipFuncSetSharedMemConfig() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipFuncSetSharedMemConfig");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipFuncSetSharedMemConfig\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipFuncSetSharedMemConfig, hipFuncSetSharedMemConfig@hip_4.2");

// hipGLGetDevices@hip_4.3
__attribute__((visibility("default"))) int hipGLGetDevices() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGLGetDevices");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGLGetDevices\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGLGetDevices, hipGLGetDevices@hip_4.3");

// hipGetChannelDesc@hip_4.2
__attribute__((visibility("default"))) int hipGetChannelDesc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetChannelDesc");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetChannelDesc\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetChannelDesc, hipGetChannelDesc@hip_4.2");

// hipGetCmdName@hip_4.2
__attribute__((visibility("default"))) int hipGetCmdName() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetCmdName");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetCmdName\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetCmdName, hipGetCmdName@hip_4.2");

// hipGetDevice@hip_4.2
__attribute__((visibility("default"))) int hipGetDevice() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetDevice");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetDevice\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetDevice, hipGetDevice@hip_4.2");

// hipGetDeviceCount@hip_4.2
__attribute__((visibility("default"))) int hipGetDeviceCount() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetDeviceCount");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetDeviceCount\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetDeviceCount, hipGetDeviceCount@hip_4.2");

// hipGetDeviceFlags@hip_4.2
__attribute__((visibility("default"))) int hipGetDeviceFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetDeviceFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetDeviceFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetDeviceFlags, hipGetDeviceFlags@hip_4.2");

// hipGetDeviceProperties@hip_4.2
__attribute__((visibility("default"))) int hipGetDeviceProperties() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetDeviceProperties");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetDeviceProperties\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetDeviceProperties, hipGetDeviceProperties@hip_4.2");

// hipGetDevicePropertiesR0000@hip_4.2
__attribute__((visibility("default"))) int hipGetDevicePropertiesR0000() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetDevicePropertiesR0000");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGetDevicePropertiesR0000\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGetDevicePropertiesR0000, hipGetDevicePropertiesR0000@hip_4.2");

// hipGetDevicePropertiesR0600@hip_6.0
__attribute__((visibility("default"))) int hipGetDevicePropertiesR0600() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetDevicePropertiesR0600");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGetDevicePropertiesR0600\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGetDevicePropertiesR0600, hipGetDevicePropertiesR0600@hip_6.0");

// hipGetDriverEntryPoint@hip_7.1
__attribute__((visibility("default"))) int hipGetDriverEntryPoint() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetDriverEntryPoint");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetDriverEntryPoint\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetDriverEntryPoint, hipGetDriverEntryPoint@hip_7.1");

// hipGetDriverEntryPoint_spt@hip_7.1
__attribute__((visibility("default"))) int hipGetDriverEntryPoint_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetDriverEntryPoint_spt");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGetDriverEntryPoint_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGetDriverEntryPoint_spt, hipGetDriverEntryPoint_spt@hip_7.1");

// hipGetErrorName@hip_4.2
__attribute__((visibility("default"))) int hipGetErrorName() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetErrorName");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetErrorName\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetErrorName, hipGetErrorName@hip_4.2");

// hipGetErrorString@hip_4.2
__attribute__((visibility("default"))) int hipGetErrorString() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetErrorString");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetErrorString\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetErrorString, hipGetErrorString@hip_4.2");

// hipGetFuncBySymbol@hip_6.2
__attribute__((visibility("default"))) int hipGetFuncBySymbol() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetFuncBySymbol");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetFuncBySymbol\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetFuncBySymbol, hipGetFuncBySymbol@hip_6.2");

// hipGetLastError@hip_4.2
__attribute__((visibility("default"))) int hipGetLastError() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetLastError");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetLastError\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetLastError, hipGetLastError@hip_4.2");

// hipGetMipmappedArrayLevel@hip_4.2
__attribute__((visibility("default"))) int hipGetMipmappedArrayLevel() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetMipmappedArrayLevel");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetMipmappedArrayLevel\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetMipmappedArrayLevel, hipGetMipmappedArrayLevel@hip_4.2");

// hipGetProcAddress@hip_6.1
__attribute__((visibility("default"))) int hipGetProcAddress() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetProcAddress");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetProcAddress\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetProcAddress, hipGetProcAddress@hip_6.1");

// hipGetStreamDeviceId@hip_4.2
__attribute__((visibility("default"))) int hipGetStreamDeviceId() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetStreamDeviceId");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetStreamDeviceId\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetStreamDeviceId, hipGetStreamDeviceId@hip_4.2");

// hipGetSymbolAddress@hip_4.2
__attribute__((visibility("default"))) int hipGetSymbolAddress() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetSymbolAddress");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetSymbolAddress\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetSymbolAddress, hipGetSymbolAddress@hip_4.2");

// hipGetSymbolSize@hip_4.2
__attribute__((visibility("default"))) int hipGetSymbolSize() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetSymbolSize");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetSymbolSize\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetSymbolSize, hipGetSymbolSize@hip_4.2");

// hipGetTextureAlignmentOffset@hip_4.2
__attribute__((visibility("default"))) int hipGetTextureAlignmentOffset() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetTextureAlignmentOffset");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGetTextureAlignmentOffset\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetTextureAlignmentOffset, "
        "hipGetTextureAlignmentOffset@hip_4.2");

// hipGetTextureObjectResourceDesc@hip_4.2
__attribute__((visibility("default"))) int hipGetTextureObjectResourceDesc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetTextureObjectResourceDesc");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGetTextureObjectResourceDesc\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetTextureObjectResourceDesc, "
        "hipGetTextureObjectResourceDesc@hip_4.2");

// hipGetTextureObjectResourceViewDesc@hip_4.2
__attribute__((visibility("default"))) int
hipGetTextureObjectResourceViewDesc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetTextureObjectResourceViewDesc");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipGetTextureObjectResourceViewDesc\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetTextureObjectResourceViewDesc, "
        "hipGetTextureObjectResourceViewDesc@hip_4.2");

// hipGetTextureObjectTextureDesc@hip_4.2
__attribute__((visibility("default"))) int hipGetTextureObjectTextureDesc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetTextureObjectTextureDesc");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGetTextureObjectTextureDesc\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetTextureObjectTextureDesc, "
        "hipGetTextureObjectTextureDesc@hip_4.2");

// hipGetTextureReference@hip_4.2
__attribute__((visibility("default"))) int hipGetTextureReference() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGetTextureReference");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGetTextureReference\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGetTextureReference, hipGetTextureReference@hip_4.2");

// hipGraphAddBatchMemOpNode@hip_6.4
__attribute__((visibility("default"))) int hipGraphAddBatchMemOpNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddBatchMemOpNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddBatchMemOpNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddBatchMemOpNode, hipGraphAddBatchMemOpNode@hip_6.4");

// hipGraphAddChildGraphNode@hip_4.5
__attribute__((visibility("default"))) int hipGraphAddChildGraphNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddChildGraphNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddChildGraphNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddChildGraphNode, hipGraphAddChildGraphNode@hip_4.5");

// hipGraphAddDependencies@hip_4.4
__attribute__((visibility("default"))) int hipGraphAddDependencies() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddDependencies");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddDependencies\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddDependencies, hipGraphAddDependencies@hip_4.4");

// hipGraphAddEmptyNode@hip_4.4
__attribute__((visibility("default"))) int hipGraphAddEmptyNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddEmptyNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddEmptyNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddEmptyNode, hipGraphAddEmptyNode@hip_4.4");

// hipGraphAddEventRecordNode@hip_4.5
__attribute__((visibility("default"))) int hipGraphAddEventRecordNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddEventRecordNode");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphAddEventRecordNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphAddEventRecordNode, hipGraphAddEventRecordNode@hip_4.5");

// hipGraphAddEventWaitNode@hip_4.5
__attribute__((visibility("default"))) int hipGraphAddEventWaitNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddEventWaitNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddEventWaitNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddEventWaitNode, hipGraphAddEventWaitNode@hip_4.5");

// hipGraphAddExternalSemaphoresSignalNode@hip_5.3
__attribute__((visibility("default"))) int
hipGraphAddExternalSemaphoresSignalNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn =
        (int (*)())get_real_sym("hipGraphAddExternalSemaphoresSignalNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipGraphAddExternalSemaphoresSignalNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddExternalSemaphoresSignalNode, "
        "hipGraphAddExternalSemaphoresSignalNode@hip_5.3");

// hipGraphAddExternalSemaphoresWaitNode@hip_5.3
__attribute__((visibility("default"))) int
hipGraphAddExternalSemaphoresWaitNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddExternalSemaphoresWaitNode");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipGraphAddExternalSemaphoresWaitNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddExternalSemaphoresWaitNode, "
        "hipGraphAddExternalSemaphoresWaitNode@hip_5.3");

// hipGraphAddHostNode@hip_4.5
__attribute__((visibility("default"))) int hipGraphAddHostNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddHostNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddHostNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddHostNode, hipGraphAddHostNode@hip_4.5");

// hipGraphAddKernelNode@hip_4.3
__attribute__((visibility("default"))) int hipGraphAddKernelNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddKernelNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddKernelNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddKernelNode, hipGraphAddKernelNode@hip_4.3");

// hipGraphAddMemAllocNode@hip_5.5
__attribute__((visibility("default"))) int hipGraphAddMemAllocNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddMemAllocNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddMemAllocNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddMemAllocNode, hipGraphAddMemAllocNode@hip_5.5");

// hipGraphAddMemFreeNode@hip_5.5
__attribute__((visibility("default"))) int hipGraphAddMemFreeNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddMemFreeNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddMemFreeNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddMemFreeNode, hipGraphAddMemFreeNode@hip_5.5");

// hipGraphAddMemcpyNode@hip_4.3
__attribute__((visibility("default"))) int hipGraphAddMemcpyNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddMemcpyNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddMemcpyNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddMemcpyNode, hipGraphAddMemcpyNode@hip_4.3");

// hipGraphAddMemcpyNode1D@hip_4.3
__attribute__((visibility("default"))) int hipGraphAddMemcpyNode1D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddMemcpyNode1D");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddMemcpyNode1D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddMemcpyNode1D, hipGraphAddMemcpyNode1D@hip_4.3");

// hipGraphAddMemcpyNodeFromSymbol@hip_4.5
__attribute__((visibility("default"))) int hipGraphAddMemcpyNodeFromSymbol() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddMemcpyNodeFromSymbol");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphAddMemcpyNodeFromSymbol\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddMemcpyNodeFromSymbol, "
        "hipGraphAddMemcpyNodeFromSymbol@hip_4.5");

// hipGraphAddMemcpyNodeToSymbol@hip_4.5
__attribute__((visibility("default"))) int hipGraphAddMemcpyNodeToSymbol() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddMemcpyNodeToSymbol");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphAddMemcpyNodeToSymbol\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddMemcpyNodeToSymbol, "
        "hipGraphAddMemcpyNodeToSymbol@hip_4.5");

// hipGraphAddMemsetNode@hip_4.3
__attribute__((visibility("default"))) int hipGraphAddMemsetNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddMemsetNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddMemsetNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddMemsetNode, hipGraphAddMemsetNode@hip_4.3");

// hipGraphAddNode@hip_5.5
__attribute__((visibility("default"))) int hipGraphAddNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphAddNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphAddNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphAddNode, hipGraphAddNode@hip_5.5");

// hipGraphBatchMemOpNodeGetParams@hip_6.4
__attribute__((visibility("default"))) int hipGraphBatchMemOpNodeGetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphBatchMemOpNodeGetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphBatchMemOpNodeGetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphBatchMemOpNodeGetParams, "
        "hipGraphBatchMemOpNodeGetParams@hip_6.4");

// hipGraphBatchMemOpNodeSetParams@hip_6.4
__attribute__((visibility("default"))) int hipGraphBatchMemOpNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphBatchMemOpNodeSetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphBatchMemOpNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphBatchMemOpNodeSetParams, "
        "hipGraphBatchMemOpNodeSetParams@hip_6.4");

// hipGraphChildGraphNodeGetGraph@hip_4.5
__attribute__((visibility("default"))) int hipGraphChildGraphNodeGetGraph() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphChildGraphNodeGetGraph");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphChildGraphNodeGetGraph\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphChildGraphNodeGetGraph, "
        "hipGraphChildGraphNodeGetGraph@hip_4.5");

// hipGraphClone@hip_4.5
__attribute__((visibility("default"))) int hipGraphClone() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphClone");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphClone\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphClone, hipGraphClone@hip_4.5");

// hipGraphCreate@hip_4.3
__attribute__((visibility("default"))) int hipGraphCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphCreate, hipGraphCreate@hip_4.3");

// hipGraphDebugDotPrint@hip_5.3
__attribute__((visibility("default"))) int hipGraphDebugDotPrint() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphDebugDotPrint");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphDebugDotPrint\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphDebugDotPrint, hipGraphDebugDotPrint@hip_5.3");

// hipGraphDestroy@hip_4.3
__attribute__((visibility("default"))) int hipGraphDestroy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphDestroy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphDestroy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphDestroy, hipGraphDestroy@hip_4.3");

// hipGraphDestroyNode@hip_4.5
__attribute__((visibility("default"))) int hipGraphDestroyNode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphDestroyNode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphDestroyNode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphDestroyNode, hipGraphDestroyNode@hip_4.5");

// hipGraphEventRecordNodeGetEvent@hip_4.5
__attribute__((visibility("default"))) int hipGraphEventRecordNodeGetEvent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphEventRecordNodeGetEvent");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphEventRecordNodeGetEvent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphEventRecordNodeGetEvent, "
        "hipGraphEventRecordNodeGetEvent@hip_4.5");

// hipGraphEventRecordNodeSetEvent@hip_4.5
__attribute__((visibility("default"))) int hipGraphEventRecordNodeSetEvent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphEventRecordNodeSetEvent");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphEventRecordNodeSetEvent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphEventRecordNodeSetEvent, "
        "hipGraphEventRecordNodeSetEvent@hip_4.5");

// hipGraphEventWaitNodeGetEvent@hip_4.5
__attribute__((visibility("default"))) int hipGraphEventWaitNodeGetEvent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphEventWaitNodeGetEvent");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphEventWaitNodeGetEvent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphEventWaitNodeGetEvent, "
        "hipGraphEventWaitNodeGetEvent@hip_4.5");

// hipGraphEventWaitNodeSetEvent@hip_4.5
__attribute__((visibility("default"))) int hipGraphEventWaitNodeSetEvent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphEventWaitNodeSetEvent");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphEventWaitNodeSetEvent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphEventWaitNodeSetEvent, "
        "hipGraphEventWaitNodeSetEvent@hip_4.5");

// hipGraphExecBatchMemOpNodeSetParams@hip_6.4
__attribute__((visibility("default"))) int
hipGraphExecBatchMemOpNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecBatchMemOpNodeSetParams");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipGraphExecBatchMemOpNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecBatchMemOpNodeSetParams, "
        "hipGraphExecBatchMemOpNodeSetParams@hip_6.4");

// hipGraphExecChildGraphNodeSetParams@hip_4.5
__attribute__((visibility("default"))) int
hipGraphExecChildGraphNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecChildGraphNodeSetParams");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipGraphExecChildGraphNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecChildGraphNodeSetParams, "
        "hipGraphExecChildGraphNodeSetParams@hip_4.5");

// hipGraphExecDestroy@hip_4.3
__attribute__((visibility("default"))) int hipGraphExecDestroy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecDestroy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphExecDestroy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecDestroy, hipGraphExecDestroy@hip_4.3");

// hipGraphExecEventRecordNodeSetEvent@hip_4.5
__attribute__((visibility("default"))) int
hipGraphExecEventRecordNodeSetEvent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecEventRecordNodeSetEvent");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipGraphExecEventRecordNodeSetEvent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecEventRecordNodeSetEvent, "
        "hipGraphExecEventRecordNodeSetEvent@hip_4.5");

// hipGraphExecEventWaitNodeSetEvent@hip_4.5
__attribute__((visibility("default"))) int hipGraphExecEventWaitNodeSetEvent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecEventWaitNodeSetEvent");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphExecEventWaitNodeSetEvent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecEventWaitNodeSetEvent, "
        "hipGraphExecEventWaitNodeSetEvent@hip_4.5");

// hipGraphExecExternalSemaphoresSignalNodeSetParams@hip_5.3
__attribute__((visibility("default"))) int
hipGraphExecExternalSemaphoresSignalNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym(
        "hipGraphExecExternalSemaphoresSignalNodeSetParams");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipGraphExecExternalSemaphoresSignalNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecExternalSemaphoresSignalNodeSetParams, "
        "hipGraphExecExternalSemaphoresSignalNodeSetParams@hip_5.3");

// hipGraphExecExternalSemaphoresWaitNodeSetParams@hip_5.3
__attribute__((visibility("default"))) int
hipGraphExecExternalSemaphoresWaitNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym(
        "hipGraphExecExternalSemaphoresWaitNodeSetParams");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipGraphExecExternalSemaphoresWaitNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecExternalSemaphoresWaitNodeSetParams, "
        "hipGraphExecExternalSemaphoresWaitNodeSetParams@hip_5.3");

// hipGraphExecGetFlags@hip_6.2
__attribute__((visibility("default"))) int hipGraphExecGetFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecGetFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphExecGetFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecGetFlags, hipGraphExecGetFlags@hip_6.2");

// hipGraphExecHostNodeSetParams@hip_4.5
__attribute__((visibility("default"))) int hipGraphExecHostNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecHostNodeSetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphExecHostNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecHostNodeSetParams, "
        "hipGraphExecHostNodeSetParams@hip_4.5");

// hipGraphExecKernelNodeSetParams@hip_4.4
__attribute__((visibility("default"))) int hipGraphExecKernelNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecKernelNodeSetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphExecKernelNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecKernelNodeSetParams, "
        "hipGraphExecKernelNodeSetParams@hip_4.4");

// hipGraphExecMemcpyNodeSetParams@hip_4.5
__attribute__((visibility("default"))) int hipGraphExecMemcpyNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecMemcpyNodeSetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphExecMemcpyNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecMemcpyNodeSetParams, "
        "hipGraphExecMemcpyNodeSetParams@hip_4.5");

// hipGraphExecMemcpyNodeSetParams1D@hip_4.5
__attribute__((visibility("default"))) int hipGraphExecMemcpyNodeSetParams1D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecMemcpyNodeSetParams1D");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphExecMemcpyNodeSetParams1D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecMemcpyNodeSetParams1D, "
        "hipGraphExecMemcpyNodeSetParams1D@hip_4.5");

// hipGraphExecMemcpyNodeSetParamsFromSymbol@hip_4.5
__attribute__((visibility("default"))) int
hipGraphExecMemcpyNodeSetParamsFromSymbol() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn =
        (int (*)())get_real_sym("hipGraphExecMemcpyNodeSetParamsFromSymbol");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipGraphExecMemcpyNodeSetParamsFromSymbol\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecMemcpyNodeSetParamsFromSymbol, "
        "hipGraphExecMemcpyNodeSetParamsFromSymbol@hip_4.5");

// hipGraphExecMemcpyNodeSetParamsToSymbol@hip_4.5
__attribute__((visibility("default"))) int
hipGraphExecMemcpyNodeSetParamsToSymbol() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn =
        (int (*)())get_real_sym("hipGraphExecMemcpyNodeSetParamsToSymbol");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipGraphExecMemcpyNodeSetParamsToSymbol\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecMemcpyNodeSetParamsToSymbol, "
        "hipGraphExecMemcpyNodeSetParamsToSymbol@hip_4.5");

// hipGraphExecMemsetNodeSetParams@hip_4.5
__attribute__((visibility("default"))) int hipGraphExecMemsetNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecMemsetNodeSetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphExecMemsetNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecMemsetNodeSetParams, "
        "hipGraphExecMemsetNodeSetParams@hip_4.5");

// hipGraphExecNodeSetParams@hip_6.2
__attribute__((visibility("default"))) int hipGraphExecNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecNodeSetParams");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphExecNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecNodeSetParams, hipGraphExecNodeSetParams@hip_6.2");

// hipGraphExecUpdate@hip_4.5
__attribute__((visibility("default"))) int hipGraphExecUpdate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphExecUpdate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphExecUpdate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExecUpdate, hipGraphExecUpdate@hip_4.5");

// hipGraphExternalSemaphoresSignalNodeGetParams@hip_5.3
__attribute__((visibility("default"))) int
hipGraphExternalSemaphoresSignalNodeGetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym(
        "hipGraphExternalSemaphoresSignalNodeGetParams");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipGraphExternalSemaphoresSignalNodeGetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExternalSemaphoresSignalNodeGetParams, "
        "hipGraphExternalSemaphoresSignalNodeGetParams@hip_5.3");

// hipGraphExternalSemaphoresSignalNodeSetParams@hip_5.3
__attribute__((visibility("default"))) int
hipGraphExternalSemaphoresSignalNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym(
        "hipGraphExternalSemaphoresSignalNodeSetParams");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipGraphExternalSemaphoresSignalNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExternalSemaphoresSignalNodeSetParams, "
        "hipGraphExternalSemaphoresSignalNodeSetParams@hip_5.3");

// hipGraphExternalSemaphoresWaitNodeGetParams@hip_5.3
__attribute__((visibility("default"))) int
hipGraphExternalSemaphoresWaitNodeGetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn =
        (int (*)())get_real_sym("hipGraphExternalSemaphoresWaitNodeGetParams");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipGraphExternalSemaphoresWaitNodeGetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExternalSemaphoresWaitNodeGetParams, "
        "hipGraphExternalSemaphoresWaitNodeGetParams@hip_5.3");

// hipGraphExternalSemaphoresWaitNodeSetParams@hip_5.3
__attribute__((visibility("default"))) int
hipGraphExternalSemaphoresWaitNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn =
        (int (*)())get_real_sym("hipGraphExternalSemaphoresWaitNodeSetParams");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipGraphExternalSemaphoresWaitNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphExternalSemaphoresWaitNodeSetParams, "
        "hipGraphExternalSemaphoresWaitNodeSetParams@hip_5.3");

// hipGraphGetEdges@hip_4.5
__attribute__((visibility("default"))) int hipGraphGetEdges() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphGetEdges");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphGetEdges\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphGetEdges, hipGraphGetEdges@hip_4.5");

// hipGraphGetNodes@hip_4.4
__attribute__((visibility("default"))) int hipGraphGetNodes() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphGetNodes");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphGetNodes\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphGetNodes, hipGraphGetNodes@hip_4.4");

// hipGraphGetRootNodes@hip_4.4
__attribute__((visibility("default"))) int hipGraphGetRootNodes() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphGetRootNodes");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphGetRootNodes\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphGetRootNodes, hipGraphGetRootNodes@hip_4.4");

// hipGraphHostNodeGetParams@hip_4.5
__attribute__((visibility("default"))) int hipGraphHostNodeGetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphHostNodeGetParams");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphHostNodeGetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphHostNodeGetParams, hipGraphHostNodeGetParams@hip_4.5");

// hipGraphHostNodeSetParams@hip_4.5
__attribute__((visibility("default"))) int hipGraphHostNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphHostNodeSetParams");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphHostNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphHostNodeSetParams, hipGraphHostNodeSetParams@hip_4.5");

// hipGraphInstantiate@hip_4.3
__attribute__((visibility("default"))) int hipGraphInstantiate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphInstantiate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphInstantiate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphInstantiate, hipGraphInstantiate@hip_4.3");

// hipGraphInstantiateWithFlags@hip_4.5
__attribute__((visibility("default"))) int hipGraphInstantiateWithFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphInstantiateWithFlags");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphInstantiateWithFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphInstantiateWithFlags, "
        "hipGraphInstantiateWithFlags@hip_4.5");

// hipGraphInstantiateWithParams@hip_6.1
__attribute__((visibility("default"))) int hipGraphInstantiateWithParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphInstantiateWithParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphInstantiateWithParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphInstantiateWithParams, "
        "hipGraphInstantiateWithParams@hip_6.1");

// hipGraphKernelNodeCopyAttributes@hip_5.3
__attribute__((visibility("default"))) int hipGraphKernelNodeCopyAttributes() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphKernelNodeCopyAttributes");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphKernelNodeCopyAttributes\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphKernelNodeCopyAttributes, "
        "hipGraphKernelNodeCopyAttributes@hip_5.3");

// hipGraphKernelNodeGetAttribute@hip_5.0
__attribute__((visibility("default"))) int hipGraphKernelNodeGetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphKernelNodeGetAttribute");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphKernelNodeGetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphKernelNodeGetAttribute, "
        "hipGraphKernelNodeGetAttribute@hip_5.0");

// hipGraphKernelNodeGetParams@hip_4.4
__attribute__((visibility("default"))) int hipGraphKernelNodeGetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphKernelNodeGetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphKernelNodeGetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphKernelNodeGetParams, hipGraphKernelNodeGetParams@hip_4.4");

// hipGraphKernelNodeSetAttribute@hip_5.0
__attribute__((visibility("default"))) int hipGraphKernelNodeSetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphKernelNodeSetAttribute");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphKernelNodeSetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphKernelNodeSetAttribute, "
        "hipGraphKernelNodeSetAttribute@hip_5.0");

// hipGraphKernelNodeSetParams@hip_4.4
__attribute__((visibility("default"))) int hipGraphKernelNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphKernelNodeSetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphKernelNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphKernelNodeSetParams, hipGraphKernelNodeSetParams@hip_4.4");

// hipGraphLaunch@hip_4.3
__attribute__((visibility("default"))) int hipGraphLaunch() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphLaunch");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphLaunch\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphLaunch, hipGraphLaunch@hip_4.3");

// hipGraphLaunch_spt@hip_5.3
__attribute__((visibility("default"))) int hipGraphLaunch_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphLaunch_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphLaunch_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphLaunch_spt, hipGraphLaunch_spt@hip_5.3");

// hipGraphMemAllocNodeGetParams@hip_5.5
__attribute__((visibility("default"))) int hipGraphMemAllocNodeGetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphMemAllocNodeGetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphMemAllocNodeGetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphMemAllocNodeGetParams, "
        "hipGraphMemAllocNodeGetParams@hip_5.5");

// hipGraphMemFreeNodeGetParams@hip_5.5
__attribute__((visibility("default"))) int hipGraphMemFreeNodeGetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphMemFreeNodeGetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphMemFreeNodeGetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphMemFreeNodeGetParams, "
        "hipGraphMemFreeNodeGetParams@hip_5.5");

// hipGraphMemcpyNodeGetParams@hip_4.4
__attribute__((visibility("default"))) int hipGraphMemcpyNodeGetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphMemcpyNodeGetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphMemcpyNodeGetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphMemcpyNodeGetParams, hipGraphMemcpyNodeGetParams@hip_4.4");

// hipGraphMemcpyNodeSetParams@hip_4.4
__attribute__((visibility("default"))) int hipGraphMemcpyNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphMemcpyNodeSetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphMemcpyNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphMemcpyNodeSetParams, hipGraphMemcpyNodeSetParams@hip_4.4");

// hipGraphMemcpyNodeSetParams1D@hip_4.5
__attribute__((visibility("default"))) int hipGraphMemcpyNodeSetParams1D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphMemcpyNodeSetParams1D");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphMemcpyNodeSetParams1D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphMemcpyNodeSetParams1D, "
        "hipGraphMemcpyNodeSetParams1D@hip_4.5");

// hipGraphMemcpyNodeSetParamsFromSymbol@hip_4.5
__attribute__((visibility("default"))) int
hipGraphMemcpyNodeSetParamsFromSymbol() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphMemcpyNodeSetParamsFromSymbol");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipGraphMemcpyNodeSetParamsFromSymbol\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphMemcpyNodeSetParamsFromSymbol, "
        "hipGraphMemcpyNodeSetParamsFromSymbol@hip_4.5");

// hipGraphMemcpyNodeSetParamsToSymbol@hip_4.5
__attribute__((visibility("default"))) int
hipGraphMemcpyNodeSetParamsToSymbol() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphMemcpyNodeSetParamsToSymbol");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipGraphMemcpyNodeSetParamsToSymbol\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphMemcpyNodeSetParamsToSymbol, "
        "hipGraphMemcpyNodeSetParamsToSymbol@hip_4.5");

// hipGraphMemsetNodeGetParams@hip_4.4
__attribute__((visibility("default"))) int hipGraphMemsetNodeGetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphMemsetNodeGetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphMemsetNodeGetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphMemsetNodeGetParams, hipGraphMemsetNodeGetParams@hip_4.4");

// hipGraphMemsetNodeSetParams@hip_4.4
__attribute__((visibility("default"))) int hipGraphMemsetNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphMemsetNodeSetParams");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphMemsetNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphMemsetNodeSetParams, hipGraphMemsetNodeSetParams@hip_4.4");

// hipGraphNodeFindInClone@hip_4.5
__attribute__((visibility("default"))) int hipGraphNodeFindInClone() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphNodeFindInClone");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphNodeFindInClone\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphNodeFindInClone, hipGraphNodeFindInClone@hip_4.5");

// hipGraphNodeGetDependencies@hip_4.5
__attribute__((visibility("default"))) int hipGraphNodeGetDependencies() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphNodeGetDependencies");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphNodeGetDependencies\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphNodeGetDependencies, hipGraphNodeGetDependencies@hip_4.5");

// hipGraphNodeGetDependentNodes@hip_4.5
__attribute__((visibility("default"))) int hipGraphNodeGetDependentNodes() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphNodeGetDependentNodes");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphNodeGetDependentNodes\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphNodeGetDependentNodes, "
        "hipGraphNodeGetDependentNodes@hip_4.5");

// hipGraphNodeGetEnabled@hip_5.3
__attribute__((visibility("default"))) int hipGraphNodeGetEnabled() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphNodeGetEnabled");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphNodeGetEnabled\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphNodeGetEnabled, hipGraphNodeGetEnabled@hip_5.3");

// hipGraphNodeGetType@hip_4.5
__attribute__((visibility("default"))) int hipGraphNodeGetType() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphNodeGetType");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphNodeGetType\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphNodeGetType, hipGraphNodeGetType@hip_4.5");

// hipGraphNodeSetEnabled@hip_5.3
__attribute__((visibility("default"))) int hipGraphNodeSetEnabled() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphNodeSetEnabled");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphNodeSetEnabled\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphNodeSetEnabled, hipGraphNodeSetEnabled@hip_5.3");

// hipGraphNodeSetParams@hip_6.2
__attribute__((visibility("default"))) int hipGraphNodeSetParams() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphNodeSetParams");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphNodeSetParams\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphNodeSetParams, hipGraphNodeSetParams@hip_6.2");

// hipGraphReleaseUserObject@hip_5.3
__attribute__((visibility("default"))) int hipGraphReleaseUserObject() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphReleaseUserObject");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphReleaseUserObject\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphReleaseUserObject, hipGraphReleaseUserObject@hip_5.3");

// hipGraphRemoveDependencies@hip_4.5
__attribute__((visibility("default"))) int hipGraphRemoveDependencies() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphRemoveDependencies");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphRemoveDependencies\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphRemoveDependencies, hipGraphRemoveDependencies@hip_4.5");

// hipGraphRetainUserObject@hip_5.3
__attribute__((visibility("default"))) int hipGraphRetainUserObject() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphRetainUserObject");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphRetainUserObject\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphRetainUserObject, hipGraphRetainUserObject@hip_5.3");

// hipGraphUpload@hip_5.3
__attribute__((visibility("default"))) int hipGraphUpload() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphUpload");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphUpload\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphUpload, hipGraphUpload@hip_5.3");

// hipGraphicsGLRegisterBuffer@hip_4.3
__attribute__((visibility("default"))) int hipGraphicsGLRegisterBuffer() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphicsGLRegisterBuffer");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphicsGLRegisterBuffer\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphicsGLRegisterBuffer, hipGraphicsGLRegisterBuffer@hip_4.3");

// hipGraphicsGLRegisterImage@hip_4.5
__attribute__((visibility("default"))) int hipGraphicsGLRegisterImage() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphicsGLRegisterImage");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphicsGLRegisterImage\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipGraphicsGLRegisterImage, hipGraphicsGLRegisterImage@hip_4.5");

// hipGraphicsMapResources@hip_4.3
__attribute__((visibility("default"))) int hipGraphicsMapResources() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphicsMapResources");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphicsMapResources\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphicsMapResources, hipGraphicsMapResources@hip_4.3");

// hipGraphicsResourceGetMappedPointer@hip_4.3
__attribute__((visibility("default"))) int
hipGraphicsResourceGetMappedPointer() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphicsResourceGetMappedPointer");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipGraphicsResourceGetMappedPointer\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphicsResourceGetMappedPointer, "
        "hipGraphicsResourceGetMappedPointer@hip_4.3");

// hipGraphicsSubResourceGetMappedArray@hip_4.5
__attribute__((visibility("default"))) int
hipGraphicsSubResourceGetMappedArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphicsSubResourceGetMappedArray");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipGraphicsSubResourceGetMappedArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphicsSubResourceGetMappedArray, "
        "hipGraphicsSubResourceGetMappedArray@hip_4.5");

// hipGraphicsUnmapResources@hip_4.3
__attribute__((visibility("default"))) int hipGraphicsUnmapResources() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphicsUnmapResources");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipGraphicsUnmapResources\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphicsUnmapResources, hipGraphicsUnmapResources@hip_4.3");

// hipGraphicsUnregisterResource@hip_4.3
__attribute__((visibility("default"))) int hipGraphicsUnregisterResource() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipGraphicsUnregisterResource");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipGraphicsUnregisterResource\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipGraphicsUnregisterResource, "
        "hipGraphicsUnregisterResource@hip_4.3");

// hipHccModuleLaunchKernel@hip_4.2
__attribute__((visibility("default"))) int hipHccModuleLaunchKernel() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipHccModuleLaunchKernel");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipHccModuleLaunchKernel\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipHccModuleLaunchKernel, hipHccModuleLaunchKernel@hip_4.2");

// hipHostAlloc@hip_4.2
__attribute__((visibility("default"))) int hipHostAlloc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipHostAlloc");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipHostAlloc\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipHostAlloc, hipHostAlloc@hip_4.2");

// hipHostFree@hip_4.2
__attribute__((visibility("default"))) int hipHostFree() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipHostFree");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipHostFree\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipHostFree, hipHostFree@hip_4.2");

// hipHostGetDevicePointer@hip_4.2
__attribute__((visibility("default"))) int hipHostGetDevicePointer() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipHostGetDevicePointer");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipHostGetDevicePointer\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipHostGetDevicePointer, hipHostGetDevicePointer@hip_4.2");

// hipHostGetFlags@hip_4.2
__attribute__((visibility("default"))) int hipHostGetFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipHostGetFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipHostGetFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipHostGetFlags, hipHostGetFlags@hip_4.2");

// hipHostMalloc@hip_4.2
__attribute__((visibility("default"))) int hipHostMalloc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipHostMalloc");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipHostMalloc\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipHostMalloc, hipHostMalloc@hip_4.2");

// hipHostRegister@hip_4.2
__attribute__((visibility("default"))) int hipHostRegister() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipHostRegister");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipHostRegister\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipHostRegister, hipHostRegister@hip_4.2");

// hipHostUnregister@hip_4.2
__attribute__((visibility("default"))) int hipHostUnregister() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipHostUnregister");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipHostUnregister\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipHostUnregister, hipHostUnregister@hip_4.2");

// hipImportExternalMemory@hip_4.3
__attribute__((visibility("default"))) int hipImportExternalMemory() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipImportExternalMemory");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipImportExternalMemory\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipImportExternalMemory, hipImportExternalMemory@hip_4.3");

// hipImportExternalSemaphore@hip_4.3
__attribute__((visibility("default"))) int hipImportExternalSemaphore() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipImportExternalSemaphore");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipImportExternalSemaphore\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipImportExternalSemaphore, hipImportExternalSemaphore@hip_4.3");

// hipInit@hip_4.2
__attribute__((visibility("default"))) int hipInit() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipInit");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipInit\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipInit, hipInit@hip_4.2");

// hipIpcCloseMemHandle@hip_4.2
__attribute__((visibility("default"))) int hipIpcCloseMemHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipIpcCloseMemHandle");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipIpcCloseMemHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipIpcCloseMemHandle, hipIpcCloseMemHandle@hip_4.2");

// hipIpcGetEventHandle@hip_4.2
__attribute__((visibility("default"))) int hipIpcGetEventHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipIpcGetEventHandle");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipIpcGetEventHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipIpcGetEventHandle, hipIpcGetEventHandle@hip_4.2");

// hipIpcGetMemHandle@hip_4.2
__attribute__((visibility("default"))) int hipIpcGetMemHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipIpcGetMemHandle");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipIpcGetMemHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipIpcGetMemHandle, hipIpcGetMemHandle@hip_4.2");

// hipIpcOpenEventHandle@hip_4.2
__attribute__((visibility("default"))) int hipIpcOpenEventHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipIpcOpenEventHandle");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipIpcOpenEventHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipIpcOpenEventHandle, hipIpcOpenEventHandle@hip_4.2");

// hipIpcOpenMemHandle@hip_4.2
__attribute__((visibility("default"))) int hipIpcOpenMemHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipIpcOpenMemHandle");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipIpcOpenMemHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipIpcOpenMemHandle, hipIpcOpenMemHandle@hip_4.2");

// hipKernelNameRef@hip_4.2
__attribute__((visibility("default"))) int hipKernelNameRef() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipKernelNameRef");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipKernelNameRef\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipKernelNameRef, hipKernelNameRef@hip_4.2");

// hipKernelNameRefByPtr@hip_4.2
__attribute__((visibility("default"))) int hipKernelNameRefByPtr() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipKernelNameRefByPtr");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipKernelNameRefByPtr\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipKernelNameRefByPtr, hipKernelNameRefByPtr@hip_4.2");

// hipLaunchByPtr@hip_4.2
__attribute__((visibility("default"))) int hipLaunchByPtr() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLaunchByPtr");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLaunchByPtr\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLaunchByPtr, hipLaunchByPtr@hip_4.2");

// hipLaunchCooperativeKernel@hip_4.2
__attribute__((visibility("default"))) int hipLaunchCooperativeKernel() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLaunchCooperativeKernel");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipLaunchCooperativeKernel\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipLaunchCooperativeKernel, hipLaunchCooperativeKernel@hip_4.2");

// hipLaunchCooperativeKernelMultiDevice@hip_4.2
__attribute__((visibility("default"))) int
hipLaunchCooperativeKernelMultiDevice() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLaunchCooperativeKernelMultiDevice");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipLaunchCooperativeKernelMultiDevice\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLaunchCooperativeKernelMultiDevice, "
        "hipLaunchCooperativeKernelMultiDevice@hip_4.2");

// hipLaunchCooperativeKernel_spt@hip_5.2
__attribute__((visibility("default"))) int hipLaunchCooperativeKernel_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLaunchCooperativeKernel_spt");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipLaunchCooperativeKernel_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLaunchCooperativeKernel_spt, "
        "hipLaunchCooperativeKernel_spt@hip_5.2");

// hipLaunchHostFunc@hip_5.3
__attribute__((visibility("default"))) int hipLaunchHostFunc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLaunchHostFunc");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLaunchHostFunc\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLaunchHostFunc, hipLaunchHostFunc@hip_5.3");

// hipLaunchHostFunc_spt@hip_5.3
__attribute__((visibility("default"))) int hipLaunchHostFunc_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLaunchHostFunc_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLaunchHostFunc_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLaunchHostFunc_spt, hipLaunchHostFunc_spt@hip_5.3");

// hipLaunchKernel@hip_4.2
__attribute__((visibility("default"))) int hipLaunchKernel() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLaunchKernel");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLaunchKernel\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLaunchKernel, hipLaunchKernel@hip_4.2");

// hipLaunchKernelExC@hip_6.5
__attribute__((visibility("default"))) int hipLaunchKernelExC() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLaunchKernelExC");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLaunchKernelExC\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLaunchKernelExC, hipLaunchKernelExC@hip_6.5");

// hipLaunchKernel_spt@hip_5.2
__attribute__((visibility("default"))) int hipLaunchKernel_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLaunchKernel_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLaunchKernel_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLaunchKernel_spt, hipLaunchKernel_spt@hip_5.2");

// hipLibraryGetKernel@hip_7.2
__attribute__((visibility("default"))) int hipLibraryGetKernel() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLibraryGetKernel");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLibraryGetKernel\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLibraryGetKernel, hipLibraryGetKernel@hip_7.2");

// hipLibraryGetKernelCount@hip_7.2
__attribute__((visibility("default"))) int hipLibraryGetKernelCount() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLibraryGetKernelCount");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLibraryGetKernelCount\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLibraryGetKernelCount, hipLibraryGetKernelCount@hip_7.2");

// hipLibraryLoadData@hip_7.2
__attribute__((visibility("default"))) int hipLibraryLoadData() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLibraryLoadData");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLibraryLoadData\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLibraryLoadData, hipLibraryLoadData@hip_7.2");

// hipLibraryLoadFromFile@hip_7.2
__attribute__((visibility("default"))) int hipLibraryLoadFromFile() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLibraryLoadFromFile");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLibraryLoadFromFile\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLibraryLoadFromFile, hipLibraryLoadFromFile@hip_7.2");

// hipLibraryUnload@hip_7.2
__attribute__((visibility("default"))) int hipLibraryUnload() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLibraryUnload");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLibraryUnload\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLibraryUnload, hipLibraryUnload@hip_7.2");

// hipLinkAddData@hip_6.4
__attribute__((visibility("default"))) int hipLinkAddData() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLinkAddData");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLinkAddData\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLinkAddData, hipLinkAddData@hip_6.4");

// hipLinkAddFile@hip_6.4
__attribute__((visibility("default"))) int hipLinkAddFile() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLinkAddFile");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLinkAddFile\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLinkAddFile, hipLinkAddFile@hip_6.4");

// hipLinkComplete@hip_6.4
__attribute__((visibility("default"))) int hipLinkComplete() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLinkComplete");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLinkComplete\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLinkComplete, hipLinkComplete@hip_6.4");

// hipLinkCreate@hip_6.4
__attribute__((visibility("default"))) int hipLinkCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLinkCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLinkCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLinkCreate, hipLinkCreate@hip_6.4");

// hipLinkDestroy@hip_6.4
__attribute__((visibility("default"))) int hipLinkDestroy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipLinkDestroy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipLinkDestroy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipLinkDestroy, hipLinkDestroy@hip_6.4");

// hipMalloc@hip_4.2
__attribute__((visibility("default"))) int hipMalloc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMalloc");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMalloc\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMalloc, hipMalloc@hip_4.2");

// hipMalloc3D@hip_4.2
__attribute__((visibility("default"))) int hipMalloc3D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMalloc3D");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMalloc3D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMalloc3D, hipMalloc3D@hip_4.2");

// hipMalloc3DArray@hip_4.2
__attribute__((visibility("default"))) int hipMalloc3DArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMalloc3DArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMalloc3DArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMalloc3DArray, hipMalloc3DArray@hip_4.2");

// hipMallocArray@hip_4.2
__attribute__((visibility("default"))) int hipMallocArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMallocArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMallocArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMallocArray, hipMallocArray@hip_4.2");

// hipMallocAsync@hip_5.1
__attribute__((visibility("default"))) int hipMallocAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMallocAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMallocAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMallocAsync, hipMallocAsync@hip_5.1");

// hipMallocFromPoolAsync@hip_5.1
__attribute__((visibility("default"))) int hipMallocFromPoolAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMallocFromPoolAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMallocFromPoolAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMallocFromPoolAsync, hipMallocFromPoolAsync@hip_5.1");

// hipMallocHost@hip_4.2
__attribute__((visibility("default"))) int hipMallocHost() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMallocHost");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMallocHost\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMallocHost, hipMallocHost@hip_4.2");

// hipMallocManaged@hip_4.2
__attribute__((visibility("default"))) int hipMallocManaged() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMallocManaged");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMallocManaged\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMallocManaged, hipMallocManaged@hip_4.2");

// hipMallocMipmappedArray@hip_4.2
__attribute__((visibility("default"))) int hipMallocMipmappedArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMallocMipmappedArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMallocMipmappedArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMallocMipmappedArray, hipMallocMipmappedArray@hip_4.2");

// hipMallocPitch@hip_4.2
__attribute__((visibility("default"))) int hipMallocPitch() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMallocPitch");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMallocPitch\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMallocPitch, hipMallocPitch@hip_4.2");

// hipMemAddressFree@hip_5.1
__attribute__((visibility("default"))) int hipMemAddressFree() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemAddressFree");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemAddressFree\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemAddressFree, hipMemAddressFree@hip_5.1");

// hipMemAddressReserve@hip_5.1
__attribute__((visibility("default"))) int hipMemAddressReserve() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemAddressReserve");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemAddressReserve\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemAddressReserve, hipMemAddressReserve@hip_5.1");

// hipMemAdvise@hip_4.2
__attribute__((visibility("default"))) int hipMemAdvise() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemAdvise");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemAdvise\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemAdvise, hipMemAdvise@hip_4.2");

// hipMemAdvise_v2@hip_7.1
__attribute__((visibility("default"))) int hipMemAdvise_v2() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemAdvise_v2");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemAdvise_v2\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemAdvise_v2, hipMemAdvise_v2@hip_7.1");

// hipMemAllocHost@hip_4.2
__attribute__((visibility("default"))) int hipMemAllocHost() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemAllocHost");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemAllocHost\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemAllocHost, hipMemAllocHost@hip_4.2");

// hipMemAllocPitch@hip_4.2
__attribute__((visibility("default"))) int hipMemAllocPitch() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemAllocPitch");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemAllocPitch\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemAllocPitch, hipMemAllocPitch@hip_4.2");

// hipMemCreate@hip_5.1
__attribute__((visibility("default"))) int hipMemCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemCreate, hipMemCreate@hip_5.1");

// hipMemExportToShareableHandle@hip_5.1
__attribute__((visibility("default"))) int hipMemExportToShareableHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemExportToShareableHandle");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipMemExportToShareableHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemExportToShareableHandle, "
        "hipMemExportToShareableHandle@hip_5.1");

// hipMemGetAccess@hip_5.1
__attribute__((visibility("default"))) int hipMemGetAccess() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemGetAccess");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemGetAccess\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemGetAccess, hipMemGetAccess@hip_5.1");

// hipMemGetAddressRange@hip_4.2
__attribute__((visibility("default"))) int hipMemGetAddressRange() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemGetAddressRange");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemGetAddressRange\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemGetAddressRange, hipMemGetAddressRange@hip_4.2");

// hipMemGetAllocationGranularity@hip_5.1
__attribute__((visibility("default"))) int hipMemGetAllocationGranularity() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemGetAllocationGranularity");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipMemGetAllocationGranularity\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemGetAllocationGranularity, "
        "hipMemGetAllocationGranularity@hip_5.1");

// hipMemGetAllocationPropertiesFromHandle@hip_5.1
__attribute__((visibility("default"))) int
hipMemGetAllocationPropertiesFromHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn =
        (int (*)())get_real_sym("hipMemGetAllocationPropertiesFromHandle");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipMemGetAllocationPropertiesFromHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemGetAllocationPropertiesFromHandle, "
        "hipMemGetAllocationPropertiesFromHandle@hip_5.1");

// hipMemGetHandleForAddressRange@hip_6.5
__attribute__((visibility("default"))) int hipMemGetHandleForAddressRange() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemGetHandleForAddressRange");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipMemGetHandleForAddressRange\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemGetHandleForAddressRange, "
        "hipMemGetHandleForAddressRange@hip_6.5");

// hipMemGetInfo@hip_4.2
__attribute__((visibility("default"))) int hipMemGetInfo() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemGetInfo");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemGetInfo\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemGetInfo, hipMemGetInfo@hip_4.2");

// hipMemImportFromShareableHandle@hip_5.1
__attribute__((visibility("default"))) int hipMemImportFromShareableHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemImportFromShareableHandle");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipMemImportFromShareableHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemImportFromShareableHandle, "
        "hipMemImportFromShareableHandle@hip_5.1");

// hipMemMap@hip_5.1
__attribute__((visibility("default"))) int hipMemMap() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemMap");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemMap\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemMap, hipMemMap@hip_5.1");

// hipMemMapArrayAsync@hip_5.1
__attribute__((visibility("default"))) int hipMemMapArrayAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemMapArrayAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemMapArrayAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemMapArrayAsync, hipMemMapArrayAsync@hip_5.1");

// hipMemPoolCreate@hip_5.1
__attribute__((visibility("default"))) int hipMemPoolCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPoolCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolCreate, hipMemPoolCreate@hip_5.1");

// hipMemPoolDestroy@hip_5.1
__attribute__((visibility("default"))) int hipMemPoolDestroy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolDestroy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPoolDestroy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolDestroy, hipMemPoolDestroy@hip_5.1");

// hipMemPoolExportPointer@hip_5.1
__attribute__((visibility("default"))) int hipMemPoolExportPointer() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolExportPointer");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPoolExportPointer\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolExportPointer, hipMemPoolExportPointer@hip_5.1");

// hipMemPoolExportToShareableHandle@hip_5.1
__attribute__((visibility("default"))) int hipMemPoolExportToShareableHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolExportToShareableHandle");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipMemPoolExportToShareableHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolExportToShareableHandle, "
        "hipMemPoolExportToShareableHandle@hip_5.1");

// hipMemPoolGetAccess@hip_5.1
__attribute__((visibility("default"))) int hipMemPoolGetAccess() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolGetAccess");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPoolGetAccess\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolGetAccess, hipMemPoolGetAccess@hip_5.1");

// hipMemPoolGetAttribute@hip_5.1
__attribute__((visibility("default"))) int hipMemPoolGetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolGetAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPoolGetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolGetAttribute, hipMemPoolGetAttribute@hip_5.1");

// hipMemPoolImportFromShareableHandle@hip_5.1
__attribute__((visibility("default"))) int
hipMemPoolImportFromShareableHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolImportFromShareableHandle");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipMemPoolImportFromShareableHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolImportFromShareableHandle, "
        "hipMemPoolImportFromShareableHandle@hip_5.1");

// hipMemPoolImportPointer@hip_5.1
__attribute__((visibility("default"))) int hipMemPoolImportPointer() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolImportPointer");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPoolImportPointer\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolImportPointer, hipMemPoolImportPointer@hip_5.1");

// hipMemPoolSetAccess@hip_5.1
__attribute__((visibility("default"))) int hipMemPoolSetAccess() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolSetAccess");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPoolSetAccess\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolSetAccess, hipMemPoolSetAccess@hip_5.1");

// hipMemPoolSetAttribute@hip_5.1
__attribute__((visibility("default"))) int hipMemPoolSetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolSetAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPoolSetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolSetAttribute, hipMemPoolSetAttribute@hip_5.1");

// hipMemPoolTrimTo@hip_5.1
__attribute__((visibility("default"))) int hipMemPoolTrimTo() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPoolTrimTo");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPoolTrimTo\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPoolTrimTo, hipMemPoolTrimTo@hip_5.1");

// hipMemPrefetchAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemPrefetchAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPrefetchAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPrefetchAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPrefetchAsync, hipMemPrefetchAsync@hip_4.2");

// hipMemPrefetchAsync_v2@hip_7.1
__attribute__((visibility("default"))) int hipMemPrefetchAsync_v2() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPrefetchAsync_v2");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPrefetchAsync_v2\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPrefetchAsync_v2, hipMemPrefetchAsync_v2@hip_7.1");

// hipMemPtrGetInfo@hip_4.2
__attribute__((visibility("default"))) int hipMemPtrGetInfo() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemPtrGetInfo");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemPtrGetInfo\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemPtrGetInfo, hipMemPtrGetInfo@hip_4.2");

// hipMemRangeGetAttribute@hip_4.2
__attribute__((visibility("default"))) int hipMemRangeGetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemRangeGetAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemRangeGetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemRangeGetAttribute, hipMemRangeGetAttribute@hip_4.2");

// hipMemRangeGetAttributes@hip_4.2
__attribute__((visibility("default"))) int hipMemRangeGetAttributes() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemRangeGetAttributes");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemRangeGetAttributes\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemRangeGetAttributes, hipMemRangeGetAttributes@hip_4.2");

// hipMemRelease@hip_5.1
__attribute__((visibility("default"))) int hipMemRelease() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemRelease");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemRelease\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemRelease, hipMemRelease@hip_5.1");

// hipMemRetainAllocationHandle@hip_5.1
__attribute__((visibility("default"))) int hipMemRetainAllocationHandle() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemRetainAllocationHandle");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipMemRetainAllocationHandle\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemRetainAllocationHandle, "
        "hipMemRetainAllocationHandle@hip_5.1");

// hipMemSetAccess@hip_5.1
__attribute__((visibility("default"))) int hipMemSetAccess() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemSetAccess");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemSetAccess\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemSetAccess, hipMemSetAccess@hip_5.1");

// hipMemUnmap@hip_5.1
__attribute__((visibility("default"))) int hipMemUnmap() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemUnmap");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemUnmap\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemUnmap, hipMemUnmap@hip_5.1");

// hipMemcpy@hip_4.2
__attribute__((visibility("default"))) int hipMemcpy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy, hipMemcpy@hip_4.2");

// hipMemcpy2D@hip_4.2
__attribute__((visibility("default"))) int hipMemcpy2D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2D");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2D, hipMemcpy2D@hip_4.2");

// hipMemcpy2DArrayToArray@hip_6.2
__attribute__((visibility("default"))) int hipMemcpy2DArrayToArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DArrayToArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2DArrayToArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2DArrayToArray, hipMemcpy2DArrayToArray@hip_6.2");

// hipMemcpy2DAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpy2DAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2DAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2DAsync, hipMemcpy2DAsync@hip_4.2");

// hipMemcpy2DAsync_spt@hip_5.3
__attribute__((visibility("default"))) int hipMemcpy2DAsync_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DAsync_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2DAsync_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2DAsync_spt, hipMemcpy2DAsync_spt@hip_5.3");

// hipMemcpy2DFromArray@hip_4.2
__attribute__((visibility("default"))) int hipMemcpy2DFromArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DFromArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2DFromArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2DFromArray, hipMemcpy2DFromArray@hip_4.2");

// hipMemcpy2DFromArrayAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpy2DFromArrayAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DFromArrayAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2DFromArrayAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2DFromArrayAsync, hipMemcpy2DFromArrayAsync@hip_4.2");

// hipMemcpy2DFromArrayAsync_spt@hip_5.3
__attribute__((visibility("default"))) int hipMemcpy2DFromArrayAsync_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DFromArrayAsync_spt");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipMemcpy2DFromArrayAsync_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2DFromArrayAsync_spt, "
        "hipMemcpy2DFromArrayAsync_spt@hip_5.3");

// hipMemcpy2DFromArray_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemcpy2DFromArray_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DFromArray_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2DFromArray_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2DFromArray_spt, hipMemcpy2DFromArray_spt@hip_5.2");

// hipMemcpy2DToArray@hip_4.2
__attribute__((visibility("default"))) int hipMemcpy2DToArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DToArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2DToArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2DToArray, hipMemcpy2DToArray@hip_4.2");

// hipMemcpy2DToArrayAsync@hip_4.3
__attribute__((visibility("default"))) int hipMemcpy2DToArrayAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DToArrayAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2DToArrayAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2DToArrayAsync, hipMemcpy2DToArrayAsync@hip_4.3");

// hipMemcpy2DToArrayAsync_spt@hip_5.3
__attribute__((visibility("default"))) int hipMemcpy2DToArrayAsync_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DToArrayAsync_spt");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipMemcpy2DToArrayAsync_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipMemcpy2DToArrayAsync_spt, hipMemcpy2DToArrayAsync_spt@hip_5.3");

// hipMemcpy2DToArray_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemcpy2DToArray_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2DToArray_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2DToArray_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2DToArray_spt, hipMemcpy2DToArray_spt@hip_5.2");

// hipMemcpy2D_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemcpy2D_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy2D_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy2D_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy2D_spt, hipMemcpy2D_spt@hip_5.2");

// hipMemcpy3D@hip_4.2
__attribute__((visibility("default"))) int hipMemcpy3D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy3D");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy3D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy3D, hipMemcpy3D@hip_4.2");

// hipMemcpy3DAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpy3DAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy3DAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy3DAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy3DAsync, hipMemcpy3DAsync@hip_4.2");

// hipMemcpy3DAsync_spt@hip_5.3
__attribute__((visibility("default"))) int hipMemcpy3DAsync_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy3DAsync_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy3DAsync_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy3DAsync_spt, hipMemcpy3DAsync_spt@hip_5.3");

// hipMemcpy3DBatchAsync@hip_7.1
__attribute__((visibility("default"))) int hipMemcpy3DBatchAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy3DBatchAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy3DBatchAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy3DBatchAsync, hipMemcpy3DBatchAsync@hip_7.1");

// hipMemcpy3DPeer@hip_7.1
__attribute__((visibility("default"))) int hipMemcpy3DPeer() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy3DPeer");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy3DPeer\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy3DPeer, hipMemcpy3DPeer@hip_7.1");

// hipMemcpy3DPeerAsync@hip_7.1
__attribute__((visibility("default"))) int hipMemcpy3DPeerAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy3DPeerAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy3DPeerAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy3DPeerAsync, hipMemcpy3DPeerAsync@hip_7.1");

// hipMemcpy3D_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemcpy3D_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy3D_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy3D_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy3D_spt, hipMemcpy3D_spt@hip_5.2");

// hipMemcpyAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyAsync, hipMemcpyAsync@hip_4.2");

// hipMemcpyAsync_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemcpyAsync_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyAsync_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyAsync_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyAsync_spt, hipMemcpyAsync_spt@hip_5.2");

// hipMemcpyAtoA@hip_6.2
__attribute__((visibility("default"))) int hipMemcpyAtoA() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyAtoA");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyAtoA\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyAtoA, hipMemcpyAtoA@hip_6.2");

// hipMemcpyAtoD@hip_6.2
__attribute__((visibility("default"))) int hipMemcpyAtoD() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyAtoD");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyAtoD\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyAtoD, hipMemcpyAtoD@hip_6.2");

// hipMemcpyAtoH@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyAtoH() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyAtoH");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyAtoH\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyAtoH, hipMemcpyAtoH@hip_4.2");

// hipMemcpyAtoHAsync@hip_6.2
__attribute__((visibility("default"))) int hipMemcpyAtoHAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyAtoHAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyAtoHAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyAtoHAsync, hipMemcpyAtoHAsync@hip_6.2");

// hipMemcpyBatchAsync@hip_7.1
__attribute__((visibility("default"))) int hipMemcpyBatchAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyBatchAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyBatchAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyBatchAsync, hipMemcpyBatchAsync@hip_7.1");

// hipMemcpyDtoA@hip_6.2
__attribute__((visibility("default"))) int hipMemcpyDtoA() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyDtoA");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyDtoA\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyDtoA, hipMemcpyDtoA@hip_6.2");

// hipMemcpyDtoD@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyDtoD() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyDtoD");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyDtoD\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyDtoD, hipMemcpyDtoD@hip_4.2");

// hipMemcpyDtoDAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyDtoDAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyDtoDAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyDtoDAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyDtoDAsync, hipMemcpyDtoDAsync@hip_4.2");

// hipMemcpyDtoH@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyDtoH() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyDtoH");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyDtoH\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyDtoH, hipMemcpyDtoH@hip_4.2");

// hipMemcpyDtoHAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyDtoHAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyDtoHAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyDtoHAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyDtoHAsync, hipMemcpyDtoHAsync@hip_4.2");

// hipMemcpyFromArray@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyFromArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyFromArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyFromArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyFromArray, hipMemcpyFromArray@hip_4.2");

// hipMemcpyFromArray_spt@hip_5.3
__attribute__((visibility("default"))) int hipMemcpyFromArray_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyFromArray_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyFromArray_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyFromArray_spt, hipMemcpyFromArray_spt@hip_5.3");

// hipMemcpyFromSymbol@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyFromSymbol() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyFromSymbol");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyFromSymbol\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyFromSymbol, hipMemcpyFromSymbol@hip_4.2");

// hipMemcpyFromSymbolAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyFromSymbolAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyFromSymbolAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyFromSymbolAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyFromSymbolAsync, hipMemcpyFromSymbolAsync@hip_4.2");

// hipMemcpyFromSymbolAsync_spt@hip_5.3
__attribute__((visibility("default"))) int hipMemcpyFromSymbolAsync_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyFromSymbolAsync_spt");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipMemcpyFromSymbolAsync_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyFromSymbolAsync_spt, "
        "hipMemcpyFromSymbolAsync_spt@hip_5.3");

// hipMemcpyFromSymbol_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemcpyFromSymbol_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyFromSymbol_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyFromSymbol_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyFromSymbol_spt, hipMemcpyFromSymbol_spt@hip_5.2");

// hipMemcpyHtoA@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyHtoA() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyHtoA");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyHtoA\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyHtoA, hipMemcpyHtoA@hip_4.2");

// hipMemcpyHtoAAsync@hip_6.2
__attribute__((visibility("default"))) int hipMemcpyHtoAAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyHtoAAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyHtoAAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyHtoAAsync, hipMemcpyHtoAAsync@hip_6.2");

// hipMemcpyHtoD@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyHtoD() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyHtoD");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyHtoD\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyHtoD, hipMemcpyHtoD@hip_4.2");

// hipMemcpyHtoDAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyHtoDAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyHtoDAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyHtoDAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyHtoDAsync, hipMemcpyHtoDAsync@hip_4.2");

// hipMemcpyParam2D@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyParam2D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyParam2D");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyParam2D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyParam2D, hipMemcpyParam2D@hip_4.2");

// hipMemcpyParam2DAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyParam2DAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyParam2DAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyParam2DAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyParam2DAsync, hipMemcpyParam2DAsync@hip_4.2");

// hipMemcpyPeer@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyPeer() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyPeer");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyPeer\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyPeer, hipMemcpyPeer@hip_4.2");

// hipMemcpyPeerAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyPeerAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyPeerAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyPeerAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyPeerAsync, hipMemcpyPeerAsync@hip_4.2");

// hipMemcpyToArray@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyToArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyToArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyToArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyToArray, hipMemcpyToArray@hip_4.2");

// hipMemcpyToSymbol@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyToSymbol() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyToSymbol");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyToSymbol\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyToSymbol, hipMemcpyToSymbol@hip_4.2");

// hipMemcpyToSymbolAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyToSymbolAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyToSymbolAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyToSymbolAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyToSymbolAsync, hipMemcpyToSymbolAsync@hip_4.2");

// hipMemcpyToSymbolAsync_spt@hip_5.3
__attribute__((visibility("default"))) int hipMemcpyToSymbolAsync_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyToSymbolAsync_spt");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipMemcpyToSymbolAsync_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipMemcpyToSymbolAsync_spt, hipMemcpyToSymbolAsync_spt@hip_5.3");

// hipMemcpyToSymbol_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemcpyToSymbol_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyToSymbol_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyToSymbol_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyToSymbol_spt, hipMemcpyToSymbol_spt@hip_5.2");

// hipMemcpyWithStream@hip_4.2
__attribute__((visibility("default"))) int hipMemcpyWithStream() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpyWithStream");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpyWithStream\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpyWithStream, hipMemcpyWithStream@hip_4.2");

// hipMemcpy_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemcpy_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemcpy_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemcpy_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemcpy_spt, hipMemcpy_spt@hip_5.2");

// hipMemset@hip_4.2
__attribute__((visibility("default"))) int hipMemset() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemset");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemset\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemset, hipMemset@hip_4.2");

// hipMemset2D@hip_4.2
__attribute__((visibility("default"))) int hipMemset2D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemset2D");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemset2D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemset2D, hipMemset2D@hip_4.2");

// hipMemset2DAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemset2DAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemset2DAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemset2DAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemset2DAsync, hipMemset2DAsync@hip_4.2");

// hipMemset2DAsync_spt@hip_5.3
__attribute__((visibility("default"))) int hipMemset2DAsync_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemset2DAsync_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemset2DAsync_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemset2DAsync_spt, hipMemset2DAsync_spt@hip_5.3");

// hipMemset2D_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemset2D_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemset2D_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemset2D_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemset2D_spt, hipMemset2D_spt@hip_5.2");

// hipMemset3D@hip_4.2
__attribute__((visibility("default"))) int hipMemset3D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemset3D");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemset3D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemset3D, hipMemset3D@hip_4.2");

// hipMemset3DAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemset3DAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemset3DAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemset3DAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemset3DAsync, hipMemset3DAsync@hip_4.2");

// hipMemset3DAsync_spt@hip_5.3
__attribute__((visibility("default"))) int hipMemset3DAsync_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemset3DAsync_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemset3DAsync_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemset3DAsync_spt, hipMemset3DAsync_spt@hip_5.3");

// hipMemset3D_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemset3D_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemset3D_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemset3D_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemset3D_spt, hipMemset3D_spt@hip_5.2");

// hipMemsetAsync@hip_4.2
__attribute__((visibility("default"))) int hipMemsetAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetAsync, hipMemsetAsync@hip_4.2");

// hipMemsetAsync_spt@hip_5.3
__attribute__((visibility("default"))) int hipMemsetAsync_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetAsync_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetAsync_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetAsync_spt, hipMemsetAsync_spt@hip_5.3");

// hipMemsetD16@hip_4.2
__attribute__((visibility("default"))) int hipMemsetD16() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD16");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD16\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD16, hipMemsetD16@hip_4.2");

// hipMemsetD16Async@hip_4.2
__attribute__((visibility("default"))) int hipMemsetD16Async() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD16Async");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD16Async\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD16Async, hipMemsetD16Async@hip_4.2");

// hipMemsetD2D16@hip_7.1
__attribute__((visibility("default"))) int hipMemsetD2D16() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD2D16");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD2D16\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD2D16, hipMemsetD2D16@hip_7.1");

// hipMemsetD2D16Async@hip_7.1
__attribute__((visibility("default"))) int hipMemsetD2D16Async() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD2D16Async");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD2D16Async\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD2D16Async, hipMemsetD2D16Async@hip_7.1");

// hipMemsetD2D32@hip_7.1
__attribute__((visibility("default"))) int hipMemsetD2D32() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD2D32");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD2D32\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD2D32, hipMemsetD2D32@hip_7.1");

// hipMemsetD2D32Async@hip_7.1
__attribute__((visibility("default"))) int hipMemsetD2D32Async() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD2D32Async");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD2D32Async\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD2D32Async, hipMemsetD2D32Async@hip_7.1");

// hipMemsetD2D8@hip_7.1
__attribute__((visibility("default"))) int hipMemsetD2D8() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD2D8");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD2D8\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD2D8, hipMemsetD2D8@hip_7.1");

// hipMemsetD2D8Async@hip_7.1
__attribute__((visibility("default"))) int hipMemsetD2D8Async() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD2D8Async");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD2D8Async\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD2D8Async, hipMemsetD2D8Async@hip_7.1");

// hipMemsetD32@hip_4.2
__attribute__((visibility("default"))) int hipMemsetD32() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD32");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD32\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD32, hipMemsetD32@hip_4.2");

// hipMemsetD32Async@hip_4.2
__attribute__((visibility("default"))) int hipMemsetD32Async() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD32Async");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD32Async\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD32Async, hipMemsetD32Async@hip_4.2");

// hipMemsetD8@hip_4.2
__attribute__((visibility("default"))) int hipMemsetD8() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD8");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD8\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD8, hipMemsetD8@hip_4.2");

// hipMemsetD8Async@hip_4.2
__attribute__((visibility("default"))) int hipMemsetD8Async() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemsetD8Async");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemsetD8Async\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemsetD8Async, hipMemsetD8Async@hip_4.2");

// hipMemset_spt@hip_5.2
__attribute__((visibility("default"))) int hipMemset_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMemset_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMemset_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMemset_spt, hipMemset_spt@hip_5.2");

// hipMipmappedArrayCreate@hip_4.2
__attribute__((visibility("default"))) int hipMipmappedArrayCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMipmappedArrayCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMipmappedArrayCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMipmappedArrayCreate, hipMipmappedArrayCreate@hip_4.2");

// hipMipmappedArrayDestroy@hip_4.2
__attribute__((visibility("default"))) int hipMipmappedArrayDestroy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMipmappedArrayDestroy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMipmappedArrayDestroy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMipmappedArrayDestroy, hipMipmappedArrayDestroy@hip_4.2");

// hipMipmappedArrayGetLevel@hip_4.2
__attribute__((visibility("default"))) int hipMipmappedArrayGetLevel() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipMipmappedArrayGetLevel");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipMipmappedArrayGetLevel\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipMipmappedArrayGetLevel, hipMipmappedArrayGetLevel@hip_4.2");

// hipModuleGetFunction@hip_4.2
__attribute__((visibility("default"))) int hipModuleGetFunction() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleGetFunction");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipModuleGetFunction\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleGetFunction, hipModuleGetFunction@hip_4.2");

// hipModuleGetFunctionCount@hip_7.1
__attribute__((visibility("default"))) int hipModuleGetFunctionCount() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleGetFunctionCount");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipModuleGetFunctionCount\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleGetFunctionCount, hipModuleGetFunctionCount@hip_7.1");

// hipModuleGetGlobal@hip_4.2
__attribute__((visibility("default"))) int hipModuleGetGlobal() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleGetGlobal");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipModuleGetGlobal\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleGetGlobal, hipModuleGetGlobal@hip_4.2");

// hipModuleGetTexRef@hip_4.2
__attribute__((visibility("default"))) int hipModuleGetTexRef() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleGetTexRef");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipModuleGetTexRef\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleGetTexRef, hipModuleGetTexRef@hip_4.2");

// hipModuleLaunchCooperativeKernel@hip_5.5
__attribute__((visibility("default"))) int hipModuleLaunchCooperativeKernel() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleLaunchCooperativeKernel");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipModuleLaunchCooperativeKernel\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleLaunchCooperativeKernel, "
        "hipModuleLaunchCooperativeKernel@hip_5.5");

// hipModuleLaunchCooperativeKernelMultiDevice@hip_5.5
__attribute__((visibility("default"))) int
hipModuleLaunchCooperativeKernelMultiDevice() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn =
        (int (*)())get_real_sym("hipModuleLaunchCooperativeKernelMultiDevice");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipModuleLaunchCooperativeKernelMultiDevice\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleLaunchCooperativeKernelMultiDevice, "
        "hipModuleLaunchCooperativeKernelMultiDevice@hip_5.5");

// hipModuleLaunchKernel@hip_4.2
__attribute__((visibility("default"))) int hipModuleLaunchKernel() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleLaunchKernel");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipModuleLaunchKernel\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleLaunchKernel, hipModuleLaunchKernel@hip_4.2");

// hipModuleLoad@hip_4.2
__attribute__((visibility("default"))) int hipModuleLoad() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleLoad");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipModuleLoad\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleLoad, hipModuleLoad@hip_4.2");

// hipModuleLoadData@hip_4.2
__attribute__((visibility("default"))) int hipModuleLoadData() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleLoadData");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipModuleLoadData\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleLoadData, hipModuleLoadData@hip_4.2");

// hipModuleLoadDataEx@hip_4.2
__attribute__((visibility("default"))) int hipModuleLoadDataEx() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleLoadDataEx");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipModuleLoadDataEx\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleLoadDataEx, hipModuleLoadDataEx@hip_4.2");

// hipModuleLoadFatBinary@hip_7.1
__attribute__((visibility("default"))) int hipModuleLoadFatBinary() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleLoadFatBinary");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipModuleLoadFatBinary\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleLoadFatBinary, hipModuleLoadFatBinary@hip_7.1");

// hipModuleOccupancyMaxActiveBlocksPerMultiprocessor@hip_4.2
__attribute__((visibility("default"))) int
hipModuleOccupancyMaxActiveBlocksPerMultiprocessor() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym(
        "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, "
        "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor@hip_4.2");

// hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags@hip_4.2
__attribute__((visibility("default"))) int
hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym(
        "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: "
            "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, "
        "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags@hip_4.2");

// hipModuleOccupancyMaxPotentialBlockSize@hip_4.2
__attribute__((visibility("default"))) int
hipModuleOccupancyMaxPotentialBlockSize() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn =
        (int (*)())get_real_sym("hipModuleOccupancyMaxPotentialBlockSize");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipModuleOccupancyMaxPotentialBlockSize\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleOccupancyMaxPotentialBlockSize, "
        "hipModuleOccupancyMaxPotentialBlockSize@hip_4.2");

// hipModuleOccupancyMaxPotentialBlockSizeWithFlags@hip_4.2
__attribute__((visibility("default"))) int
hipModuleOccupancyMaxPotentialBlockSizeWithFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym(
        "hipModuleOccupancyMaxPotentialBlockSizeWithFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipModuleOccupancyMaxPotentialBlockSizeWithFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleOccupancyMaxPotentialBlockSizeWithFlags, "
        "hipModuleOccupancyMaxPotentialBlockSizeWithFlags@hip_4.2");

// hipModuleUnload@hip_4.2
__attribute__((visibility("default"))) int hipModuleUnload() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipModuleUnload");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipModuleUnload\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipModuleUnload, hipModuleUnload@hip_4.2");

// hipOccupancyMaxActiveBlocksPerMultiprocessor@hip_4.2
__attribute__((visibility("default"))) int
hipOccupancyMaxActiveBlocksPerMultiprocessor() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn =
        (int (*)())get_real_sym("hipOccupancyMaxActiveBlocksPerMultiprocessor");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipOccupancyMaxActiveBlocksPerMultiprocessor\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipOccupancyMaxActiveBlocksPerMultiprocessor, "
        "hipOccupancyMaxActiveBlocksPerMultiprocessor@hip_4.2");

// hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags@hip_4.2
__attribute__((visibility("default"))) int
hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym(
        "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: "
                    "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, "
        "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags@hip_4.2");

// hipOccupancyMaxPotentialBlockSize@hip_4.2
__attribute__((visibility("default"))) int hipOccupancyMaxPotentialBlockSize() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipOccupancyMaxPotentialBlockSize");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipOccupancyMaxPotentialBlockSize\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipOccupancyMaxPotentialBlockSize, "
        "hipOccupancyMaxPotentialBlockSize@hip_4.2");

// hipPeekAtLastError@hip_4.2
__attribute__((visibility("default"))) int hipPeekAtLastError() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipPeekAtLastError");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipPeekAtLastError\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipPeekAtLastError, hipPeekAtLastError@hip_4.2");

// hipPointerGetAttribute@hip_5.0
__attribute__((visibility("default"))) int hipPointerGetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipPointerGetAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipPointerGetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipPointerGetAttribute, hipPointerGetAttribute@hip_5.0");

// hipPointerGetAttributes@hip_4.2
__attribute__((visibility("default"))) int hipPointerGetAttributes() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipPointerGetAttributes");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipPointerGetAttributes\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipPointerGetAttributes, hipPointerGetAttributes@hip_4.2");

// hipPointerSetAttribute@hip_4.2
__attribute__((visibility("default"))) int hipPointerSetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipPointerSetAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipPointerSetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipPointerSetAttribute, hipPointerSetAttribute@hip_4.2");

// hipProfilerStart@hip_4.2
__attribute__((visibility("default"))) int hipProfilerStart() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipProfilerStart");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipProfilerStart\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipProfilerStart, hipProfilerStart@hip_4.2");

// hipProfilerStop@hip_4.2
__attribute__((visibility("default"))) int hipProfilerStop() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipProfilerStop");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipProfilerStop\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipProfilerStop, hipProfilerStop@hip_4.2");

// hipRegisterTracerCallback@hip_5.3
__attribute__((visibility("default"))) int hipRegisterTracerCallback() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipRegisterTracerCallback");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipRegisterTracerCallback\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipRegisterTracerCallback, hipRegisterTracerCallback@hip_5.3");

// hipRuntimeGetVersion@hip_4.2
__attribute__((visibility("default"))) int hipRuntimeGetVersion() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipRuntimeGetVersion");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipRuntimeGetVersion\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipRuntimeGetVersion, hipRuntimeGetVersion@hip_4.2");

// hipSetDevice@hip_4.2
__attribute__((visibility("default"))) int hipSetDevice() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipSetDevice");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipSetDevice\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipSetDevice, hipSetDevice@hip_4.2");

// hipSetDeviceFlags@hip_4.2
__attribute__((visibility("default"))) int hipSetDeviceFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipSetDeviceFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipSetDeviceFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipSetDeviceFlags, hipSetDeviceFlags@hip_4.2");

// hipSetValidDevices@hip_6.2
__attribute__((visibility("default"))) int hipSetValidDevices() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipSetValidDevices");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipSetValidDevices\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipSetValidDevices, hipSetValidDevices@hip_6.2");

// hipSetupArgument@hip_4.2
__attribute__((visibility("default"))) int hipSetupArgument() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipSetupArgument");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipSetupArgument\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipSetupArgument, hipSetupArgument@hip_4.2");

// hipSignalExternalSemaphoresAsync@hip_4.3
__attribute__((visibility("default"))) int hipSignalExternalSemaphoresAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipSignalExternalSemaphoresAsync");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipSignalExternalSemaphoresAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipSignalExternalSemaphoresAsync, "
        "hipSignalExternalSemaphoresAsync@hip_4.3");

// hipStreamAddCallback@hip_4.2
__attribute__((visibility("default"))) int hipStreamAddCallback() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamAddCallback");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamAddCallback\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamAddCallback, hipStreamAddCallback@hip_4.2");

// hipStreamAddCallback_spt@hip_5.3
__attribute__((visibility("default"))) int hipStreamAddCallback_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamAddCallback_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamAddCallback_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamAddCallback_spt, hipStreamAddCallback_spt@hip_5.3");

// hipStreamAttachMemAsync@hip_4.2
__attribute__((visibility("default"))) int hipStreamAttachMemAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamAttachMemAsync");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamAttachMemAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamAttachMemAsync, hipStreamAttachMemAsync@hip_4.2");

// hipStreamBatchMemOp@hip_6.4
__attribute__((visibility("default"))) int hipStreamBatchMemOp() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamBatchMemOp");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamBatchMemOp\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamBatchMemOp, hipStreamBatchMemOp@hip_6.4");

// hipStreamBeginCapture@hip_4.3
__attribute__((visibility("default"))) int hipStreamBeginCapture() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamBeginCapture");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamBeginCapture\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamBeginCapture, hipStreamBeginCapture@hip_4.3");

// hipStreamBeginCaptureToGraph@hip_6.1
__attribute__((visibility("default"))) int hipStreamBeginCaptureToGraph() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamBeginCaptureToGraph");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipStreamBeginCaptureToGraph\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamBeginCaptureToGraph, "
        "hipStreamBeginCaptureToGraph@hip_6.1");

// hipStreamBeginCapture_spt@hip_5.3
__attribute__((visibility("default"))) int hipStreamBeginCapture_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamBeginCapture_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamBeginCapture_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamBeginCapture_spt, hipStreamBeginCapture_spt@hip_5.3");

// hipStreamCreate@hip_4.2
__attribute__((visibility("default"))) int hipStreamCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamCreate, hipStreamCreate@hip_4.2");

// hipStreamCreateWithFlags@hip_4.2
__attribute__((visibility("default"))) int hipStreamCreateWithFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamCreateWithFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamCreateWithFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamCreateWithFlags, hipStreamCreateWithFlags@hip_4.2");

// hipStreamCreateWithPriority@hip_4.2
__attribute__((visibility("default"))) int hipStreamCreateWithPriority() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamCreateWithPriority");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipStreamCreateWithPriority\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipStreamCreateWithPriority, hipStreamCreateWithPriority@hip_4.2");

// hipStreamDestroy@hip_4.2
__attribute__((visibility("default"))) int hipStreamDestroy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamDestroy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamDestroy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamDestroy, hipStreamDestroy@hip_4.2");

// hipStreamEndCapture@hip_4.3
__attribute__((visibility("default"))) int hipStreamEndCapture() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamEndCapture");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamEndCapture\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamEndCapture, hipStreamEndCapture@hip_4.3");

// hipStreamEndCapture_spt@hip_5.3
__attribute__((visibility("default"))) int hipStreamEndCapture_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamEndCapture_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamEndCapture_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamEndCapture_spt, hipStreamEndCapture_spt@hip_5.3");

// hipStreamGetAttribute@hip_7.1
__attribute__((visibility("default"))) int hipStreamGetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamGetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamGetAttribute, hipStreamGetAttribute@hip_7.1");

// hipStreamGetCaptureInfo@hip_4.5
__attribute__((visibility("default"))) int hipStreamGetCaptureInfo() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetCaptureInfo");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamGetCaptureInfo\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamGetCaptureInfo, hipStreamGetCaptureInfo@hip_4.5");

// hipStreamGetCaptureInfo_spt@hip_5.3
__attribute__((visibility("default"))) int hipStreamGetCaptureInfo_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetCaptureInfo_spt");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipStreamGetCaptureInfo_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipStreamGetCaptureInfo_spt, hipStreamGetCaptureInfo_spt@hip_5.3");

// hipStreamGetCaptureInfo_v2@hip_4.5
__attribute__((visibility("default"))) int hipStreamGetCaptureInfo_v2() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetCaptureInfo_v2");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipStreamGetCaptureInfo_v2\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipStreamGetCaptureInfo_v2, hipStreamGetCaptureInfo_v2@hip_4.5");

// hipStreamGetCaptureInfo_v2_spt@hip_5.3
__attribute__((visibility("default"))) int hipStreamGetCaptureInfo_v2_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetCaptureInfo_v2_spt");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipStreamGetCaptureInfo_v2_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamGetCaptureInfo_v2_spt, "
        "hipStreamGetCaptureInfo_v2_spt@hip_5.3");

// hipStreamGetDevice@hip_4.2
__attribute__((visibility("default"))) int hipStreamGetDevice() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetDevice");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamGetDevice\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamGetDevice, hipStreamGetDevice@hip_4.2");

// hipStreamGetFlags@hip_4.2
__attribute__((visibility("default"))) int hipStreamGetFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamGetFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamGetFlags, hipStreamGetFlags@hip_4.2");

// hipStreamGetFlags_spt@hip_5.2
__attribute__((visibility("default"))) int hipStreamGetFlags_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetFlags_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamGetFlags_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamGetFlags_spt, hipStreamGetFlags_spt@hip_5.2");

// hipStreamGetId@hip_7.1
__attribute__((visibility("default"))) int hipStreamGetId() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetId");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamGetId\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamGetId, hipStreamGetId@hip_7.1");

// hipStreamGetPriority@hip_4.2
__attribute__((visibility("default"))) int hipStreamGetPriority() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetPriority");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamGetPriority\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamGetPriority, hipStreamGetPriority@hip_4.2");

// hipStreamGetPriority_spt@hip_5.2
__attribute__((visibility("default"))) int hipStreamGetPriority_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamGetPriority_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamGetPriority_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamGetPriority_spt, hipStreamGetPriority_spt@hip_5.2");

// hipStreamIsCapturing@hip_4.3
__attribute__((visibility("default"))) int hipStreamIsCapturing() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamIsCapturing");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamIsCapturing\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamIsCapturing, hipStreamIsCapturing@hip_4.3");

// hipStreamIsCapturing_spt@hip_5.3
__attribute__((visibility("default"))) int hipStreamIsCapturing_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamIsCapturing_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamIsCapturing_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamIsCapturing_spt, hipStreamIsCapturing_spt@hip_5.3");

// hipStreamQuery@hip_4.2
__attribute__((visibility("default"))) int hipStreamQuery() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamQuery");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamQuery\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamQuery, hipStreamQuery@hip_4.2");

// hipStreamQuery_spt@hip_5.2
__attribute__((visibility("default"))) int hipStreamQuery_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamQuery_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamQuery_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamQuery_spt, hipStreamQuery_spt@hip_5.2");

// hipStreamSetAttribute@hip_7.1
__attribute__((visibility("default"))) int hipStreamSetAttribute() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamSetAttribute");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamSetAttribute\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamSetAttribute, hipStreamSetAttribute@hip_7.1");

// hipStreamSynchronize@hip_4.2
__attribute__((visibility("default"))) int hipStreamSynchronize() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamSynchronize");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamSynchronize\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamSynchronize, hipStreamSynchronize@hip_4.2");

// hipStreamSynchronize_spt@hip_5.2
__attribute__((visibility("default"))) int hipStreamSynchronize_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamSynchronize_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamSynchronize_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamSynchronize_spt, hipStreamSynchronize_spt@hip_5.2");

// hipStreamUpdateCaptureDependencies@hip_4.5
__attribute__((visibility("default"))) int
hipStreamUpdateCaptureDependencies() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamUpdateCaptureDependencies");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipStreamUpdateCaptureDependencies\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamUpdateCaptureDependencies, "
        "hipStreamUpdateCaptureDependencies@hip_4.5");

// hipStreamWaitEvent@hip_4.2
__attribute__((visibility("default"))) int hipStreamWaitEvent() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamWaitEvent");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamWaitEvent\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamWaitEvent, hipStreamWaitEvent@hip_4.2");

// hipStreamWaitEvent_spt@hip_5.2
__attribute__((visibility("default"))) int hipStreamWaitEvent_spt() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamWaitEvent_spt");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamWaitEvent_spt\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamWaitEvent_spt, hipStreamWaitEvent_spt@hip_5.2");

// hipStreamWaitValue32@hip_4.4
__attribute__((visibility("default"))) int hipStreamWaitValue32() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamWaitValue32");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamWaitValue32\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamWaitValue32, hipStreamWaitValue32@hip_4.4");

// hipStreamWaitValue64@hip_4.4
__attribute__((visibility("default"))) int hipStreamWaitValue64() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamWaitValue64");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamWaitValue64\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamWaitValue64, hipStreamWaitValue64@hip_4.4");

// hipStreamWriteValue32@hip_4.4
__attribute__((visibility("default"))) int hipStreamWriteValue32() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamWriteValue32");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamWriteValue32\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamWriteValue32, hipStreamWriteValue32@hip_4.4");

// hipStreamWriteValue64@hip_4.4
__attribute__((visibility("default"))) int hipStreamWriteValue64() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipStreamWriteValue64");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipStreamWriteValue64\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipStreamWriteValue64, hipStreamWriteValue64@hip_4.4");

// hipTexObjectCreate@hip_4.2
__attribute__((visibility("default"))) int hipTexObjectCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexObjectCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexObjectCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexObjectCreate, hipTexObjectCreate@hip_4.2");

// hipTexObjectDestroy@hip_4.2
__attribute__((visibility("default"))) int hipTexObjectDestroy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexObjectDestroy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexObjectDestroy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexObjectDestroy, hipTexObjectDestroy@hip_4.2");

// hipTexObjectGetResourceDesc@hip_4.2
__attribute__((visibility("default"))) int hipTexObjectGetResourceDesc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexObjectGetResourceDesc");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexObjectGetResourceDesc\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipTexObjectGetResourceDesc, hipTexObjectGetResourceDesc@hip_4.2");

// hipTexObjectGetResourceViewDesc@hip_4.2
__attribute__((visibility("default"))) int hipTexObjectGetResourceViewDesc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexObjectGetResourceViewDesc");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexObjectGetResourceViewDesc\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexObjectGetResourceViewDesc, "
        "hipTexObjectGetResourceViewDesc@hip_4.2");

// hipTexObjectGetTextureDesc@hip_4.2
__attribute__((visibility("default"))) int hipTexObjectGetTextureDesc() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexObjectGetTextureDesc");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexObjectGetTextureDesc\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipTexObjectGetTextureDesc, hipTexObjectGetTextureDesc@hip_4.2");

// hipTexRefGetAddress@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetAddress() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetAddress");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefGetAddress\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefGetAddress, hipTexRefGetAddress@hip_4.2");

// hipTexRefGetAddressMode@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetAddressMode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetAddressMode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefGetAddressMode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefGetAddressMode, hipTexRefGetAddressMode@hip_4.2");

// hipTexRefGetArray@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefGetArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefGetArray, hipTexRefGetArray@hip_4.2");

// hipTexRefGetBorderColor@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetBorderColor() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetBorderColor");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefGetBorderColor\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefGetBorderColor, hipTexRefGetBorderColor@hip_4.2");

// hipTexRefGetFilterMode@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetFilterMode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetFilterMode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefGetFilterMode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefGetFilterMode, hipTexRefGetFilterMode@hip_4.2");

// hipTexRefGetFlags@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefGetFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefGetFlags, hipTexRefGetFlags@hip_4.2");

// hipTexRefGetFormat@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetFormat() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetFormat");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefGetFormat\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefGetFormat, hipTexRefGetFormat@hip_4.2");

// hipTexRefGetMaxAnisotropy@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetMaxAnisotropy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetMaxAnisotropy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefGetMaxAnisotropy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefGetMaxAnisotropy, hipTexRefGetMaxAnisotropy@hip_4.2");

// hipTexRefGetMipMappedArray@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetMipMappedArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetMipMappedArray");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexRefGetMipMappedArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipTexRefGetMipMappedArray, hipTexRefGetMipMappedArray@hip_4.2");

// hipTexRefGetMipmapFilterMode@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetMipmapFilterMode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetMipmapFilterMode");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexRefGetMipmapFilterMode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefGetMipmapFilterMode, "
        "hipTexRefGetMipmapFilterMode@hip_4.2");

// hipTexRefGetMipmapLevelBias@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetMipmapLevelBias() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetMipmapLevelBias");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexRefGetMipmapLevelBias\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipTexRefGetMipmapLevelBias, hipTexRefGetMipmapLevelBias@hip_4.2");

// hipTexRefGetMipmapLevelClamp@hip_4.2
__attribute__((visibility("default"))) int hipTexRefGetMipmapLevelClamp() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefGetMipmapLevelClamp");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexRefGetMipmapLevelClamp\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefGetMipmapLevelClamp, "
        "hipTexRefGetMipmapLevelClamp@hip_4.2");

// hipTexRefSetAddress@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetAddress() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetAddress");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefSetAddress\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetAddress, hipTexRefSetAddress@hip_4.2");

// hipTexRefSetAddress2D@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetAddress2D() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetAddress2D");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefSetAddress2D\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetAddress2D, hipTexRefSetAddress2D@hip_4.2");

// hipTexRefSetAddressMode@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetAddressMode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetAddressMode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefSetAddressMode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetAddressMode, hipTexRefSetAddressMode@hip_4.2");

// hipTexRefSetArray@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetArray");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefSetArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetArray, hipTexRefSetArray@hip_4.2");

// hipTexRefSetBorderColor@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetBorderColor() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetBorderColor");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefSetBorderColor\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetBorderColor, hipTexRefSetBorderColor@hip_4.2");

// hipTexRefSetFilterMode@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetFilterMode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetFilterMode");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefSetFilterMode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetFilterMode, hipTexRefSetFilterMode@hip_4.2");

// hipTexRefSetFlags@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetFlags() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetFlags");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefSetFlags\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetFlags, hipTexRefSetFlags@hip_4.2");

// hipTexRefSetFormat@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetFormat() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetFormat");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefSetFormat\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetFormat, hipTexRefSetFormat@hip_4.2");

// hipTexRefSetMaxAnisotropy@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetMaxAnisotropy() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetMaxAnisotropy");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipTexRefSetMaxAnisotropy\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetMaxAnisotropy, hipTexRefSetMaxAnisotropy@hip_4.2");

// hipTexRefSetMipmapFilterMode@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetMipmapFilterMode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetMipmapFilterMode");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexRefSetMipmapFilterMode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetMipmapFilterMode, "
        "hipTexRefSetMipmapFilterMode@hip_4.2");

// hipTexRefSetMipmapLevelBias@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetMipmapLevelBias() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetMipmapLevelBias");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexRefSetMipmapLevelBias\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipTexRefSetMipmapLevelBias, hipTexRefSetMipmapLevelBias@hip_4.2");

// hipTexRefSetMipmapLevelClamp@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetMipmapLevelClamp() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetMipmapLevelClamp");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexRefSetMipmapLevelClamp\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipTexRefSetMipmapLevelClamp, "
        "hipTexRefSetMipmapLevelClamp@hip_4.2");

// hipTexRefSetMipmappedArray@hip_4.2
__attribute__((visibility("default"))) int hipTexRefSetMipmappedArray() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipTexRefSetMipmappedArray");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipTexRefSetMipmappedArray\n");
    return 1;
  }
  return real_fn();
}
__asm__(
    ".symver hipTexRefSetMipmappedArray, hipTexRefSetMipmappedArray@hip_4.2");

// hipThreadExchangeStreamCaptureMode@hip_5.0
__attribute__((visibility("default"))) int
hipThreadExchangeStreamCaptureMode() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipThreadExchangeStreamCaptureMode");
  if (!real_fn) {
    fprintf(
        stderr,
        "passthrough: missing symbol: hipThreadExchangeStreamCaptureMode\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipThreadExchangeStreamCaptureMode, "
        "hipThreadExchangeStreamCaptureMode@hip_5.0");

// hipUnbindTexture@hip_4.2
__attribute__((visibility("default"))) int hipUnbindTexture() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipUnbindTexture");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipUnbindTexture\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipUnbindTexture, hipUnbindTexture@hip_4.2");

// hipUserObjectCreate@hip_5.3
__attribute__((visibility("default"))) int hipUserObjectCreate() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipUserObjectCreate");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipUserObjectCreate\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipUserObjectCreate, hipUserObjectCreate@hip_5.3");

// hipUserObjectRelease@hip_5.3
__attribute__((visibility("default"))) int hipUserObjectRelease() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipUserObjectRelease");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipUserObjectRelease\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipUserObjectRelease, hipUserObjectRelease@hip_5.3");

// hipUserObjectRetain@hip_5.3
__attribute__((visibility("default"))) int hipUserObjectRetain() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipUserObjectRetain");
  if (!real_fn) {
    fprintf(stderr, "passthrough: missing symbol: hipUserObjectRetain\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipUserObjectRetain, hipUserObjectRetain@hip_5.3");

// hipWaitExternalSemaphoresAsync@hip_4.3
__attribute__((visibility("default"))) int hipWaitExternalSemaphoresAsync() {
  static int (*real_fn)() = NULL;
  if (!real_fn)
    real_fn = (int (*)())get_real_sym("hipWaitExternalSemaphoresAsync");
  if (!real_fn) {
    fprintf(stderr,
            "passthrough: missing symbol: hipWaitExternalSemaphoresAsync\n");
    return 1;
  }
  return real_fn();
}
__asm__(".symver hipWaitExternalSemaphoresAsync, "
        "hipWaitExternalSemaphoresAsync@hip_4.3");
