# C API reference

`loomc` is the stable C ABI for embedding Loom in native drivers, JITs,
autotuners, package builders, language bindings, and caller-owned artifact
caches. Its reference is generated directly from the installed public headers;
ownership, lifetime, threading, and failure contracts stay next to the
declarations they govern.

[Open the generated `loomc` reference](generated/index.html){ .md-button .md-button--primary }

The core API has no GPU runtime dependency. Target packages add the profile and
emission surfaces needed for AMDGPU, SPIR-V, and other output families, while
optional adapters bridge runtimes such as Vulkan and IREE HAL.
