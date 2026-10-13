# Server composition

This package composes transport, device resources and source-defined model
services. It owns deployment configuration and model lifetime; numerical
programs remain runtime inputs. Process flags, signal policy and the application
entry point belong to the executable, not these libraries.

## Deployment and lifetime

[Configuration](configuration.h) parses one complete JSON model document for
both startup files and HTTP registration. The definition contains program and
asset references, modality capacities and source-owned options. The deployment
adds its name, admission state and device-group index. Parsing copies the
document's strings and preserves authored references separately from their
local directory base; it does not load numerical programs or weights.

The [registry](registry.h) owns stable deployment slots. Each slot retains its
configuration, cumulative counters, diagnostic history and registration
identity until every accepted borrower retires. Closing admission is reversible
and independent of residency. Unregistering is irreversible: graceful removal
finishes admitted work, while abort requests cancellation; both wait for actual
resource retirement before freeing the name. A control connection acknowledges
intent without owning the operation's lifetime.

The [registration adapter](model_registration.h) is the concrete composition
point for text, image and finite-vector services. It translates deployment
options into their native constructors, then exposes preparation, residency and
retirement callbacks to the registry. The registry consumes that callback
contract without depending on concrete model implementations. Source loading
and compilation use the application's shared task pool. Metadata-only drained
registrations perform neither source I/O nor model allocation.

Each deployment selects a member of one application-owned device group. The
group and its host outlive every registration. Weight pinning, activation and
deactivation operate through the registration's parameter owner without
discarding mutable sessions. A failed platform release retains the owning
resources and shared parents; it never authorizes callback-thread destruction
of a queue or pool.

## Routing and progress

The [router](router.h) is a cooperative task process. Bounded serialized turns
claim requests, release held requests, advance model services and format
observation batches. Service priority rotates; CPU preparation, I/O and device
work progress independently and wake the owner when necessary. The outer
application caller joins final process release. Task workers do not wait for
model completion.

Requests held behind closed admission retain transport claims rather than
execution rows. Holding and live observation share a bound below connection
capacity so they cannot occupy every control connection. Each service applies
its own admission and output-credit policy after routing.

Raw tokenization runs as independent finite CPU tasks. A tokenizer borrower
keeps its deployment alive through task release, including after disconnect.
Log readers retain copied identity and cursor state, not model resources or
producer ring slots. A slow reader can miss overwritten records; model removal
does not wait for it.

## Observation and failures

[Resources](resources.h) publishes the application's listeners, immutable
device identities and serving-managed memory accounting. Status and metrics
copy owner state without polling hardware or waiting for device completion.
`/healthz` and `/readyz` report server availability, not readiness of every
registered model; `/v1/status` carries each model's preparation and failure
state. `/v1/models` lists prepared deployments available for inference.

Invalid packages remain inspectable while other models continue serving.
Failure affecting shared backing instead stops admission and drains accepted
work. The registry's callback contract distinguishes these cases using the
resource owner's evidence, not by guessing from an error code.

Adjacent tests exercise document parsing and the real router, carriers and
task system with controlled model dependencies. Source-defined catalog tests
exercise concrete registration adapters, device execution and final HTTP output.
