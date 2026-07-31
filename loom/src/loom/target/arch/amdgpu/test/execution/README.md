# AMDGPU Backend Execution Tests

This package owns backend qualification that must compile and execute through
the AMDGPU HAL. These are target tests written in Loom, not authoring examples:
cases may enumerate rounding boundaries, exceptional payloads, matrix
instruction families, allocation regressions, and architecture-dependent
native versus portable routes.

Source-to-low tests beside this directory pin target selection and emitted low
IR without requiring hardware. The execution cases here prove the resulting
artifacts against exact numeric expectations on physical devices or a qualified
simulator. Host-only compile and planning coverage checks the same sources
without requiring a local AMDGPU device.
