# Serve source-defined models

`loom-serve` runs text, image, embedding and decision models in one HTTP server.
Models are runtime inputs: a JSON deployment document, Loom programs and assets.
Registering a compatible package does not require rebuilding the executable.
The process can start empty and keep serving while models are added or removed.

This walkthrough produces an independently checkable embedding response on an
x86 CPU. The example performs arithmetic, not learned inference; it exercises
the real JIT, device execution, batching and HTTP path without downloading weights.
It requires an installed `loom-serve` with the CPU target and task HAL enabled,
plus `curl` and `jq`. [Acquiring Loom](../getting-started/acquiring-loom.md)
describes obtaining the tools.

## Start with no models

```sh
loom-serve --device=task --pool_backing=fixed --listen=127.0.0.1:8080
```

Leave this process running and use another terminal for the requests below.
The JSON `ready` record on stderr reports the bound address. It means the HTTP
server is accepting connections, not that any model has finished preparation.

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/readyz
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
```

The first response is `{"status":"ready"}`; the second has an empty `data`
array. Device selection is explicit. `task` executes native x86 kernels using
the shared CPU workers; it does not emulate a GPU. Fixed backing works without
virtual-memory support. The [command reference](../reference/loom-serve.md)
describes other device and memory choices.

## Obtain the example package

The maintained [embedding example](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/embedding)
contains six files. Download them into a new directory on the server machine:

```sh
mkdir vectors
source_url=https://raw.githubusercontent.com/ROCm/hrx-system/main/loom/src/loom/serve/fixture/embedding
for name in model.json prepare.loom control.loom config.loom embed.loom tokenizer.json; do
  curl --fail --silent --show-error --location "$source_url/$name" --output "vectors/$name"
done
```

`model.json` gives the deployment the public name `vectors`. It names the cold
preparation program, warm control program, native source providers and asset
directory. The tokenizer recognizes `a`, `b` and `x`; the example's output is
deliberately simple enough to check without another inference framework.

## Register and execute

Registration takes the same [model document](../reference/serving-models.md)
accepted by `--model`. An HTTP document needs an absolute `base_directory` for
relative file references. That directory is on the server, not the client.

```sh
jq --arg directory "$(pwd)/vectors" '. + {base_directory: $directory}' vectors/model.json |
  curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/register \
    -H 'Content-Type: application/json' --data-binary @-

curl --fail --silent --show-error http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"vectors","input":["a","bbb"],"dimensions":2}' |
  jq '{data: [.data[] | {index, embedding}], usage}'
```

Registration returns HTTP 202 for accepted intent. Preparation and JIT run
asynchronously; the inference request can wait for readiness within the
server's bounded admission capacity. The response contains these vectors, in
input order, and accounts for four prompt tokens:

```json
{
  "data": [
    {"index": 0, "embedding": [1, 1]},
    {"index": 1, "embedding": [6, 3]}
  ],
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

The two components are the sum of `token_id + 1` and the token count.
`a` has ID 0 and `b` has ID 1. The package executes two native kernels and
restores the request's row ordering after producing rows in reverse order.

## Inspect and remove

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
curl --fail --silent --show-error http://127.0.0.1:8080/metrics
curl --fail --silent --show-error http://127.0.0.1:8080/logz \
  -H 'X-Loom-Model: vectors'

curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/drain \
  -H 'Content-Type: application/json' -d '{"model":"vectors"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/unregister \
  -H 'Content-Type: application/json' -d '{"model":"vectors","mode":"drain"}'
```

Drain closes admission while accepted requests finish. New inference waits
within the holding budget until undrain, or fails if that budget is full.
Unregister is different: it removes the registration after accepted work and
resource borrowers retire. Its HTTP 202 acknowledges the removal request;
`/v1/status` shows progress until the name disappears. Other models and the
server remain alive. Ctrl-C in the server terminal stops admission and joins
accepted work before process teardown.

## Use your own package

An existing package can be registered at startup instead:

```sh
loom-serve --device=task --pool_backing=fixed \
  --model=/srv/models/vectors/model.json \
  --model=/srv/models/scout/model.json
```

Each package must provide programs for the chosen device and the source
interface of the installed server. A checkpoint alone is not a model package.
Text and portable bytecode (`.loom` and `.loombc`) can be mixed; both still JIT
for the selected live device. Provider dependencies are resolved by Loom's
symbol linker, not by a native list of model layers.

The [model-document reference](../reference/serving-models.md) explains path
binding, source roles and modality capacities. The
[command reference](../reference/loom-serve.md) covers device groups, multiple
TCP/Unix listeners and process limits. [Operation](operate-model-server.md) and
[observation](observe-model-server.md) cover model swapping, pressure, counters
and log streams. HTTP administration and model loading
assume trusted clients and programs; deployment-owned isolation or tunnels
supply any access-control boundary.

[HTTP profiles](../reference/serving-http.md) enumerate the supported request
fields, raw tokenizer utilities and retained-text checkpoint controls.
