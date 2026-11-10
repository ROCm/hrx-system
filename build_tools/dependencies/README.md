# Dependency Inventory

These tools derive dependency identity from the declarations that already own
it. They do not maintain another version manifest or update dependencies.

`dependency_inventory` combines two inputs:

- Bazel's resolved direct-module graph, which is authoritative for BCR module
  selection and overrides;
- the source dependency declarations parsed by
  `build_tools/bazel_to_cmake/deps.py`, which own custom archives, files,
  digests, and patches.

The output is ephemeral `hrx.dependency.inventory.v0` JSON. Source declarations
that resolve to the same upstream package and revision, such as the Linux UAPI
header set, share one package identity while retaining every declaration and
digest. A source host or archive shape that cannot be identified fails instead
of silently disappearing from the inventory.

`github_snapshot_json` is a pure projection of that inventory into GitHub's
dependency-submission schema. It does not contact GitHub. The caller owns Bazel
execution, repository credentials, and submission. Registry modules and custom
sources remain separate manifests because GitHub interprets package identity in
the context of each manifest ecosystem:

```bash
dependency_tmp="$(mktemp -d)"
bazelisk mod deps '<root>' --depth=1 --output=json \
  > "${dependency_tmp}/direct-deps.json"
iree-bazel-run //build_tools/dependencies:dependency_inventory -- \
  --bazel-deps="${dependency_tmp}/direct-deps.json" \
  > "${dependency_tmp}/inventory.json"
iree-bazel-run //build_tools/dependencies:github_snapshot_json -- \
  --inventory="${dependency_tmp}/inventory.json" \
  --sha="$(git rev-parse HEAD)" \
  > "${dependency_tmp}/snapshot.json"
gh api --method POST \
  "repos/ROCm/hrx-system/dependency-graph/snapshots" \
  --input "${dependency_tmp}/snapshot.json"
```

GitHub currently rejects the registered `pkg:bazel` package type in dependency
submissions. The projection uses `pkg:generic/bazel/...` until that API accepts
the registered identity. GitHub is therefore a visibility projection, not the
source of update policy or proof that an unreported dependency is safe.

`github_alerts_jsonl` turns a saved Dependabot API response into stable,
action-oriented JSONL without contacting GitHub. Multiple alert instances for
one package and GHSA become one event carrying all manifests, requirements,
links, vulnerable ranges, and fixed versions. Its fingerprint changes when
that evidence changes, allowing an append-only local consumer to suppress
unchanged observations:

```bash
dependency_tmp="$(mktemp -d)"
gh api --paginate --slurp \
  "repos/ROCm/hrx-system/dependabot/alerts?state=open&per_page=100" \
  > "${dependency_tmp}/dependabot-alerts.json"
iree-bazel-run //build_tools/dependencies:github_alerts_jsonl -- \
  --alerts="${dependency_tmp}/dependabot-alerts.json" \
  > "${dependency_tmp}/advisories.jsonl"
```

Fetching alerts and retaining or acting on events remain caller-owned steps.
The formatter has no repository credentials and does not open issues, branches,
or pull requests.
