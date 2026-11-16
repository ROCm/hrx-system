# Authoring Loom release notes

Loom stable release notes are a reviewed product document carried by the SDK
version-bump pull request. They explain the user-visible outcomes of a release,
show how those outcomes are exercised, and make required migrations
conspicuous. GitHub's generated notes and pull request titles are discovery
inputs, not publishable release notes.

During a release-candidate cycle, that document evolves in one unlisted Gist
shared by the entire release line. Candidate pull requests carry only a concise
summary of their release focus and delta, required migrations or test focus, a
link to the Gist, and the feedback channel. An unlisted Gist is accessible to
anyone with its URL; it is not private and contains no confidential material.
It is mutable working editorial rather than a publication source or durable
release record.

The release tag points at the exact commit GitHub records as that pull request's
merge result. The publisher resolves the pull request from the commit, extracts
one version-marked notes section from its description, and places the prose
before the generated installation, artifact, checksum, and qualification
appendix. A missing, duplicate, mismatched, empty, or placeholder-bearing
section stops publication before any release mutation.

The stable pull request is the final editorial review surface; individual
release documents do not accumulate in the source tree. Pull request links in a
Gist do not create GitHub timeline cross-references. Copying the completed
editorial into the stable version-bump pull request creates those references
once, from the durable source selected by the publisher.

Publication records the pull request, merge commit, version, and editorial
digest in the release body. A partial draft must continue to match the pull
request description. Once public, the immutable release body is authoritative
and validates without consulting the mutable pull request or Gist.

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

## Authoring sequence

1. Verify the previous tag and candidate commit, then enumerate the exact
   first-parent merge range.
2. For the first candidate, create one unlisted Gist for the `X.Y.Z` release
   line and seed it with the contents of the stable template's marked section.
   Reuse that Gist for every candidate.
3. Audit changes to public CLI, API, IR, configuration, serialized formats,
   target or driver defaults, packaging, and supported platforms.
4. Draft `Behavioral changes and migrations` before the feature narrative so
   required user action cannot be hidden by highlights.
5. Cluster the remaining evidence by user outcome. Repository ownership,
   subsystem prefixes, authors, and merge order are not release-note sections.
6. Select no more than six major areas. Each area carries one real command,
   API or IR fragment, report excerpt, or bounded measured result.
7. Select only the pull requests needed to navigate headline capabilities,
   migrations, measured product improvements, and critical correctness
   boundaries. The complete range is represented by one comparison link.
8. Update the Gist as candidate evidence, feedback repairs, and migrations
   change. Each candidate pull request summarizes only its own delta and links
   the same Gist.
9. Recheck every command, number, link, supported-platform statement, and
   migration against the frozen stable candidate and its qualification record.
10. Copy the completed Gist into the stable version-bump pull request's marked
    section, then review the complete composed GitHub body as the user will read
    it.

## Candidate page structure

Candidate notes state the intended evaluators and release focus, summarize the
material changes since the previous candidate or selected main revision, make
required migrations and test focus explicit, and link the shared Gist and
feedback channel. Potential security issues follow the repository's private
reporting policy.

The release-wide feature narrative and notable pull request list remain in the
Gist until stable promotion. This keeps every candidate attributable to a small
immutable summary while the longer editorial can change without duplicating its
pull request references across the RC series.

## Stable page structure

The opening paragraph states the release theme and intended audience. Three to
six highlights link the major user outcomes to their detailed sections. Each
major section explains what is newly possible or materially different, gives a
representative example, and links the smallest useful evidence set.

`Behavioral changes and migrations` is mandatory. Each row identifies the
affected user surface, previous contract, current contract, and required action.
`None` is valid only after the public-surface audit in the authoring sequence.

Patch releases use the same stable contract with a shorter feature narrative
and a complete user-visible fix list.

The final section contains notable pull requests and one complete comparison
link. Artifact filenames, install commands, checksums, platform inventory, and
qualification profiles remain in the generated appendix so editorial prose
cannot drift from the bytes being published.

## Editorial contract

User outcomes lead; implementation details appear when they explain a
constraint, migration, cost, or meaningful measured result. Examples come from
an executable or qualified production path rather than decorative pseudocode.

Supported platforms, qualified hardware, and compiler representation support
are distinct claims. Costs sit beside benefits when users experience them.
Uncertainty remains visible: a bounded production witness stays bounded, and an
unmeasured internal improvement does not become a product performance claim.

Exact model or product names appear only when reproduction or interpretation
requires them. Representative workload descriptions otherwise keep the release
story useful beyond one integration witness.

## Starting a release pull request

For a candidate, copy
[CANDIDATE_TEMPLATE.md](CANDIDATE_TEMPLATE.md) into the version-bump pull request
description and link the release line's existing Gist. For the stable release,
copy [TEMPLATE.md](TEMPLATE.md) and replace its marked-section placeholders with
the completed Gist content. In both forms, set the marked version to the exact
`package-version` that the merge commit will carry and replace every remaining
`{{PLACEHOLDER}}`.

Only the marked section is copied into the public release; pull-request-specific
review guidance remains outside it. The publisher rejects placeholder syntax
inside the marked section. Candidate descriptions omit the release-wide notable
pull request list. The stable description contains the exact pull request links
from the completed Gist so GitHub creates the release's durable cross-references.

Finish editorial and technical review before merging. The merge result is the
release revision, and the release tag must point directly at it. Any description
edit after merge must receive the same review before the tag is created. Once a
publication draft has been created, changing the marked prose is a conflict
rather than an implicit release-note update.

After stable publication, replace the Gist draft with a short link to the
immutable GitHub release. Existing candidate descriptions continue to resolve,
and the mutable Gist cannot drift into a competing release record.

The previous release is a style reference, not evidence for the next release.
Each release is rederived from its own exact range.
