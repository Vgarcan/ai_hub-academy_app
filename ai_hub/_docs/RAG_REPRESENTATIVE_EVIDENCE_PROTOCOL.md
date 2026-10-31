# Representative Retrieval Evidence Protocol

This protocol prepares an independently adjudicated retrieval corpus. It does
not create a corpus, mark evidence representative, set a no-answer threshold,
or enable any runtime path. It is the required handoff between a data owner and
a future operator-facing evaluation adapter.

`s29-policy-candidate-v1` remains controlled-candidate evidence. It must not be
relabelled by copying its shape or its results into this protocol.

## Scope and boundary

The protocol evaluates retrieval only: whether an authorized result is judged
relevant to a query. It does not evaluate the final Agent answer, and it does
not let a model judge its own output.

The actual corpus, query text, documents, source identifiers and reviewer
identities may be sensitive. They stay in the data owner's approved private
environment. Versioned repository artifacts contain only the protocol and
content-free aggregate results.

## Required private evidence package

Before running an evaluation, the data owner prepares one immutable package:

1. A corpus snapshot manifest bound to one `ApplicationScope`, one set of
   authorized collections, one embedding contract (`e1`) and one retrieval
   configuration.
2. A stable opaque case identifier for every query.
3. A stable opaque target key for every judged chunk, plus the chunk's `k1` at
   the time of adjudication. Database primary keys alone are not a snapshot
   identity.
4. The query text and relevance judgments, held only in the approved private
   workspace.
5. An immutable package digest and a redacted manifest recording the digest,
   schema version, case counts and evidence class. The redacted manifest must
   not contain query text, document text, vectors, provider bodies, credentials
   or reviewer names.

If a chunk changes, its `k1` changes and the package no longer describes the
corpus being evaluated. Re-adjudicate or make a new package; never silently
reuse judgments against modified content.

## Sampling plan

Pre-register a sampling plan before reviewers see rankings. It must cover the
actual application's intended use, including each applicable category:

- exact identifiers and codes;
- natural paraphrases and synonyms;
- noisy or incomplete user phrasing;
- ambiguous and multi-evidence questions;
- known no-answer and plausible-near-miss questions;
- restricted or cross-scope attempts, evaluated only through an authorized
  security test procedure.

The plan records the source population, inclusion/exclusion rules, known gaps
and the intended meaning of “representative”. A convenient sample is not made
representative by its size, and this repository provides no preset quota or
quality threshold.

## Independent adjudication

Each answerable case is reviewed blindly by two distinct qualified reviewers.
They assess the query against the frozen candidate targets, not a retrieval
ranking supplied by the system under evaluation. Their private records include
the package digest, reviewer role, timestamp and their grade for each target.

Use the existing four-grade relevance scale:

| Grade | Meaning | Counts as relevant |
| --- | --- | --- |
| 0 | Irrelevant | No |
| 1 | Marginal or contextual only | No |
| 2 | Relevant | Yes |
| 3 | Directly answers the case | Yes |

For a no-answer case, the reconciled judgment set must contain no grade 2 or 3
target. It tests abstention evidence; it is not a request to select the least
bad result.

A third person, distinct from both initial reviewers, reconciles disagreements.
The reconciler records the resolution and reason in the private package. An
unresolved conflict is a blocked case, not an implicit grade and not evidence
for promotion.

## Evaluation handoff

A future representative-evaluation adapter must fail closed unless all of the
following are true before dispatching a query:

1. The Agent's resolved authorization scope contains the package collection.
2. The live corpus target keys and `k1` values exactly match the frozen package.
3. Embedding coverage is complete and current under the recorded `e1`.
4. The package records two independent initial reviews and a reconciliation for
   every included case.
5. The adapter can emit only content-free result IDs, counts, metrics and
   bounded failure codes.

It must not seed or alter Knowledge, embed missing content, persist query text,
write rankings, or change retrieval routing. Any mismatch invalidates the run;
it must not be averaged, retried until it passes, or downgraded to a warning.

## Interpreting results

Run lexical, exact semantic, hybrid and any ANN candidate over the same frozen
package. Preserve the system's current authorization boundaries and compare the
same query set and cutoffs. Report at least relevance metrics, no-answer empty
rate, degradation/failure rate and authorization failures separately.

Only an explicit reviewed policy may interpret representative evidence as an
activation or backend-promotion decision. In particular, the observed S-29
no-answer failures mean that no confidence or abstention threshold exists yet.

## Current blocker and next implementation slice

There is no authorized private representative package in this repository, and
no current command accepts one. The next implementation slice begins only when
the data owner supplies a package satisfying this protocol. It should add a
separate, report-first adapter rather than widening
`knowledge_retrieval_evaluate`, whose contract remains intentionally limited to
controlled synthetic corpora.
