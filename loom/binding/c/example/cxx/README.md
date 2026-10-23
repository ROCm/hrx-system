# C++ source integration

## Native C++ ABI interop

`native_abi_library.cxx` compiles into a normal x86-64 ELF relocatable object
and links directly with `native_abi_main.cc`, which is compiled by the host C++
toolchain. Both translation units include `native_abi.h`. They exchange the
same records and pointers in both directions without a generated wrapper, a
Loom runtime, or a custom loader:

```cpp
extern "C" Pair loom_pair(Pair value) {
  return {value.second + 7u, value.first * 3u};
}

extern "C" Pair native_pair(Pair value);

extern "C" Pair loom_call_native_pair(Pair value) {
  return native_pair(value);
}
```

The example also crosses a mixed `{float, unsigned}` record, a native 128-bit
SIMD value, a 24-byte record that uses the platform's hidden-result convention,
and a pointer with nonzero byte displacement. Each definition transforms its
input so an identity-shaped ABI mistake cannot pass accidentally.

```sh
iree-bazel-run --config=loom-importer-cxx \
  --//loom/config/target:enable=x86 \
  //loom/binding/c/example/cxx:native_abi
```

The build target performs the same three ordinary steps available to another
build system: import C++ into Loom bytecode, compile that module into an ELF
object, and pass the object to the system C++ linker. The equivalent command
line is:

```sh
iree-bazel-run --config=loom-importer-cxx \
  //loom/src/loom/tools/loom-import-cxx -- \
  loom/binding/c/example/cxx/native_abi_library.cxx \
  --to=bc --output=/tmp/native_abi.loombc
iree-bazel-run --//loom/config/target:enable=x86 \
  //loom/src/loom/tools/loom-compile -- \
  /tmp/native_abi.loombc --target=x86:avx2 --output=/tmp/native_abi.o
c++ -std=c++20 -mavx2 -I. \
  loom/binding/c/example/cxx/native_abi_main.cc \
  /tmp/native_abi.o -o /tmp/native_abi
/tmp/native_abi
```

The source contract retains complete C object layout while Loom High IR keeps
record fields as named values. Target lowering performs the SysV register,
stack, and hidden-result projection only at the native boundary. Internal Loom
optimization therefore sees ordinary structured values instead of prematurely
packed machine words.

## C++ source to GPU execution

`jit_amdgpu.c` is a C embedding that compiles the standalone HIP-style C++
translation unit in `kernels.cxx`, emits one HSACO, loads it through IREE HAL
and executes both kernels. Each kernel checks 128 values and surrounding
guards. The shared template helper calls the generated scalar FMA binding
through the ordinary HIP header facade.

```sh
iree-bazel-run --config=loom-importer-cxx \
  --//runtime/config/hal:drivers=amdgpu,task \
  --//loom/config/emit:enable=amdgpu \
  //loom/binding/c/example/cxx:jit_amdgpu
```

The first optional argument is a HAL device URI, defaulting to `legacy-amdgpu`.
The second supplies an external facade include root, disabling embedded
headers for the import:

```sh
iree-bazel-run --config=loom-importer-cxx \
  --//runtime/config/hal:drivers=amdgpu,task \
  --//loom/config/emit:enable=amdgpu \
  --//loom/config/import/cxx:embed_includes=false \
  //loom/binding/c/example/cxx:jit_amdgpu -- \
  legacy-amdgpu loom/src/loom/import/cxx/include
```

The driver prepares the context, compiler and target pipeline before accepting
source. Those objects can be reused across JIT submissions. Import creates a
normal module and destroys the C++ frontend before compilation; source buffers
can be released immediately. The API performs no cleanup pipeline during
import. The target pipeline owns subsequent optimization and lowering.

The kernels carry explicit workgroup dimensions. `affine` fixes both size and
count; `residual` bounds its count and gets exact values from a normal Loom
config module. Config validation rejects choices outside the source contract.
The compiler emits launch configuration bytecode, and the host evaluates that
artifact before each dispatch. Upload, dispatch and download are ordered by
semaphore dependencies, and every successful submission completes before its
buffers are released.

Only the example depends on a GPU runtime. The importer extension itself has
no target or runtime dependency and also accepts functions-only units for the
normal Loom compilation and linking APIs.
