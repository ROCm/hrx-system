# amdf-info

`amdf-info` reports the GPUs, NPUs, system-memory pools, and native capabilities visible through libamdf on Linux and Windows. The default output is a compact machine inventory; detailed text and JSON expose the same public API records. It statically links libamdf and needs neither ROCr nor XRT.

```sh
amdf-info
amdf-info > machine.txt
amdf-info --details
amdf-info --format=json > machine.json
```

System RAM appears once under **Memory pools**, followed by one section per accelerator. Each endpoint reports its compiler target, native PCI attachment when available, and queue formats. `--details` expands exact identities, compute geometry, queue capabilities, atomic operations, and ring limits. It does not activate a device. Native names are escaped so control characters cannot inject terminal commands or extra report lines. Color is automatic for supported terminals, disabled for redirected output, and controlled by `--color=auto|always|never`; automatic color respects `NO_COLOR` and `TERM=dumb`.

## Selecting a device

Device selectors follow the native AMD HAL vocabulary used by Loom tools:

```sh
amdf-info --device=rdna://0 --details
amdf-info --device=xdna --details
amdf-info --device=amdgpu://GPU-4939e1d93d24ff77
amdf-info --device=xdna://id/0000010500000000aeed6694ed14d173
amdf-info --device=/dev/dri/renderD128
```

The schemes are `rdna`, `cdna`, `amdgpu`, and `xdna`. A bare scheme selects ordinal zero; `rdna:0` and `rdna://0` are equivalent. `amdgpu` covers all GPU architectures. `GPU-` selects a nonzero, 16-digit hexadecimal GPU identity; `id/` selects the complete 32-digit endpoint ID. Linux also accepts a character-device pathname, following symlinks and matching its device numbers without opening an execution file. A native path is not a file descriptor, and the namespace is unavailable on Windows.

`--engine=gpu|xdna` filters execution domains. `--target=NAME` filters the exact compiler target identifier reported for the endpoint, such as a GPU's `gfx` name or an NPU's `amd.xdna` target. Filters intersect; they do not renumber a device selector's family view.

```sh
amdf-info --engine=gpu --target=gfx1151 --format=json
amdf-info select --engine=gpu --target=gfx1151 --format=json
```

Inventory may contain several matching endpoints. `select` and an explicit `--device` require exactly one match: duplicate targets or GPU identities fail instead of selecting the first device. An identity or target query failure cannot turn a possible match into a definite non-match. A selected device remains the same endpoint through activation; removal produces an error rather than retargeting the request.

The diagnostic inventory does not apply HAL, HIP, or ROCr visibility masks. Its short ordinals describe its own unmasked family view and are retained separately from the report's identity-sorted display order. When an earlier endpoint could not be classified, affected later family ordinals are null and text uses exact selectors. Copy the exact `uri` from JSON or detailed text to carry a selection into another tool. Endpoint IDs and Windows LUIDs identify the current native snapshot, not a persistent identity across reboot or removal. HAL execution parameters such as queue counts and column placement are not diagnostic URI options.

## Checking activation

```sh
amdf-info --device=rdna --activate
amdf-info --device=xdna --activate --details --format=json
```

`--activate` creates each selected device, queries its achieved properties and queue placement contracts, and checks destruction of the device, endpoint, and instance. It creates no queue, workload context, or allocation and submits no commands. GPU activation exercises the native execution connection and address-space setup; XDNA activation validates the native device interface. This checks whether the provider can acquire and release the device, not whether a particular program executes correctly.

Live GPU records expose native memory/event features and reset epochs. Live XDNA records expose array geometry, context admission, placement modes, and instruction limits. Queue placement records distinguish placement support and host-publication requirements from the passive queue formats. Physical-local storage and cross-device memory relationships are separate live memory queries, not inferred from PCI attachment or system RAM capacity.

The final activation result includes cleanup failures. A failed native destruction retains the failed owner and its borrowed parents until process exit; the tool does not retry cleanup, close a live child's parent, or claim a successful smoke. An outer process supervisor owns hang detection when a native call stops making progress. The tool adds no driver timeouts, background worker, or retry loop.

## JSON representation

`--format=json` writes one UTF-8 document to stdout, without terminal emphasis. `--details` only affects text; JSON always includes the acquired semantic fields. The root contains `schema_version`, `platform`, `library`, `memory_pools`, `endpoints`, `cleanup_errors`, and `result` when those stages are reached. An earlier failed operation supplies its attributed error instead of inventing later records. The schema version changes for incompatible representation changes; readers can ignore additional fields.

| Representation | Meaning |
| --- | --- |
| JSON number | Unsigned fields no wider than 32 bits, including ordinals and enum values. |
| Decimal string | 64-bit quantities such as byte lengths, capacities, and reset epochs, regardless of magnitude. |
| Fixed hexadecimal string | 64-bit masks and native identities use `0x` followed by 16 digits. Endpoint and device IDs concatenate two 16-digit words without a prefix. |
| `{ "value": 1, "name": "gpu" }` | An enum preserves its numeric value; an unknown value has a null name. |
| `{ "value": "0x0000000000000001", "names": ["user"] }` | A bitset retains all bits and lists recognized symbolic names. |
| Null PCI address or NUMA node | The successful query did not establish that optional attachment. It is not node zero or a failed query. |
| `{ "error": { ... } }` | An operation failed; no failed output bytes are serialized as capability data. |

Full-width values never pass through floating-point JSON numbers. Units remain in field names or the public record's named quantity; human-scaled values such as GiB exist only in text. Serialization excludes C ABI headers, pointers, reserved fields, and padding.

Each endpoint retains `id`, `uri`, `family`, `enumeration_ordinal`, `family_ordinal`, and a typed `native_identity`. Linux identity contains character-device `major` and `minor`. Windows identity contains all LUID bits and the `physical_adapter_index`; the logical adapter alone does not identify a linked physical node. Queue arrays retain their native family ordinals. The tool streams individual public records and retains only the small endpoint identity index needed for selection and deterministic display.

Errors preserve `operation`, status `domain` (numeric value and symbolic name), native `code`, and a readable `message`. Endpoint errors stay inside the endpoint; cleanup errors identify the owner when available. An absent family extension describes the loaded library's composition and does not remove visible hardware from inventory. Optional access denial or unsupported metadata remains visible while independent queries continue. Unexpected native errors and required selection or activation failures fail the invocation.

`result.complete` indicates whether all requested queries completed. A partial inventory can have `exit_code: 0` when only optional access or service availability prevented completion. Automation requiring complete metadata checks both fields. An empty machine is a valid inventory; an empty required selection or activation is not.

| Exit code | Meaning |
| --- | --- |
| 0 | Requested operation succeeded; inspect `result.complete` for optional metadata gaps. |
| 1 | Required operation, unexpected native query, cleanup, or output write failed. |
| 2 | Invalid command-line arguments; discovery was not started. |
| 3 | No endpoint matches a required selection or activation. |
| 4 | A scalar selection matches multiple endpoints. |

A failed output write can truncate JSON and makes the process fail even if an earlier buffered result said success. Process exit status remains authoritative. Native enumeration growth fails the bounded snapshot instead of silently changing ordinals through rediscovery.

## Building and installation

```sh
iree-bazel-run //libamdf/tool/amdf-info -- --help
iree-bazel-run //libamdf/tool/amdf-info -- --format=json
```

The CMake `AMDF` install component places `amdf-info` in `bin/` alongside the library distribution. The tool embeds the static libamdf implementation and does not require a companion libamdf shared library. `find_package(amdf CONFIG REQUIRED)` exposes `amdf_TOOL_DIR` for locating the installed executable. Family-selective library builds retain the same tool; extension availability reports the composition actually linked. The `hrx-loom` wheel also installs the tool into its Python environment's command directory.

Loom SDK releases also provide a standalone `amdf-info-VERSION-PLATFORM.zip` containing the same executable as the Release wheel. Extract it and run it directly; no Python environment, SDK installation, ROCr, or XRT is required. The Linux build uses the release's glibc baseline, and Windows requires the Microsoft C runtime. `SHA256SUMS` identifies the exact archive bytes for pinned CI bootstrap installations.

The executable is a public libamdf client. Argument parsing and selection, query traversal and lifetime ownership, typed record renderers, and stream encoding have separate implementation boundaries. OS leaves supply terminal detection and pathname identity only; driver discovery remains in libamdf. Hardware smoke targets declare GPU or XDNA resources independently. CPU-only tests exercise the real CLI against a test-only public API dependency, including partial reports, failed acquisition, checked teardown, selection ambiguity, and output errors.
