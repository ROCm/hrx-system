# Authoring Loom SDK release notes

Loom uses two editorial surfaces during a release cycle because release
candidates and the final stable release have different review and GitHub
cross-reference costs.

- An unlisted Gist carries the evolving full release notes across the candidate
  cycle. It contains migrations, examples, measured results, notable pull
  requests, and the complete comparison link. The same Gist is updated in place
  for `rc1`, `rc2`, and later candidates.
- Each candidate identity pull request carries a short marked summary and a
  link to that Gist. Its public prerelease copies the summary, followed by the
  generated installation and qualification appendix.
- The final stable identity pull request carries the reviewed full notes copied
  from the Gist. This is the one intentional point at which notable pull-request
  links create repository cross-references.

GitHub creates persistent timeline references when one pull request description
links another. Repeating the full notable-change index in every candidate would
spray the same release bookkeeping across the entire development range. A Gist
keeps the working editorial public to testers without creating those repository
backreferences.

The identity pull request remains the publication boundary. The release tag
points at the exact commit GitHub records as that pull request's merge result,
and the publisher extracts one version-marked section from its description. A
missing, duplicate, mismatched, empty, or placeholder-bearing section stops
publication before any release mutation.

## Evidence boundary

Every authoring pass fixes these inputs before drafting:

- the previous public release tag;
- the exact candidate commit and proposed tag;
- the release channel;
- the qualification record for the candidate artifacts.

The evidence set is the exact first-parent merge range, changed public docs and
examples, release qualification, and the implementation and review record for
candidate notable changes. Agent prose, scratch summaries, labels, and pull
request titles can locate that evidence but cannot establish a claim.

The range is refreshed after the release line is frozen. Claims that no longer
describe that exact comparison are revised or removed. Performance results keep
their workload, scale, hardware, and measurement boundary; a microbenchmark is
never promoted into an end-to-end product claim.

## Candidate editorial

[CANDIDATE_TEMPLATE.md](CANDIDATE_TEMPLATE.md) is the identity pull-request
template for `aN`, `bN`, and `rcN` versions. Its marked section contains:

- the candidate theme and compatibility headline;
- the intended evaluators and actionable feedback fields;
- one link to the release line's unlisted full-notes Gist.

The candidate body omits the notable pull-request index and complete comparison
link. Those remain in the Gist. A later candidate updates the same Gist and uses
another short marked summary for its own exact identity and feedback delta.

The Gist is mutable working editorial rather than release provenance. Candidate
publication snapshots the short marked summary and records the source pull
request, merge commit, version, editorial digest, artifacts, and qualification.
The generated release appendix supplies artifact names, installation commands,
checksums, platform inventory, and profile coverage.

## Full stable editorial

[TEMPLATE.md](TEMPLATE.md) is the final stable identity pull-request template.
Before that pull request opens, the Gist is refreshed against the exact stable
range and qualification record, then its full prose is copied between the
stable pull request's marked boundaries.

The full notes lead with `Behavioral changes and migrations`. Each row names the
affected user surface, previous contract, current contract, and required action.
`None` is valid only after the public-surface audit.

The remaining evidence is clustered by user outcome rather than repository
ownership, subsystem prefix, author, or merge order. Three to six highlights
link no more than six major areas. Each area carries a real command, API or IR
fragment, report excerpt, or bounded measured result. The final section selects
only the pull requests needed to navigate headline capabilities, migrations,
measured product improvements, and critical correctness boundaries, followed by
one complete comparison link.

User outcomes lead; implementation details appear when they explain a
constraint, migration, cost, or meaningful measured result. Supported
platforms, qualified hardware, and compiler representation support remain
distinct claims. Costs sit beside benefits when users experience them. Exact
model or product names appear only when reproduction or interpretation requires
them.

## Review and publication

The marked version always matches the exact `package-version` carried by the
merge commit. Pull-request-specific guidance stays outside the marked section,
and placeholder syntax inside it is rejected.

Editorial and technical review finish before merge. Any description edit after
merge receives the same review before the tag is created. Once a publication
draft exists, changing the marked prose is a conflict rather than an implicit
release-note update.

The immutable GitHub release body is authoritative after publication. A
candidate release contains its short snapshot and Gist link; a stable release
contains the complete reviewed notes. Retries validate the recorded source and
editorial digest rather than treating the mutable Gist or pull request as the
published record.

The previous release is a style reference, not evidence for the next release.
Each release is rederived from its own exact range.
