# Procedural memory: retain the instrument, not just the story

2026-11-01 · INSIGHT-CODE-WORLD-MODELS-2026-11-01  
Dennis Palatov's 'code is your magic spell' framing; NOVA / ChatGPT synthesis. Research/design note, not a cartridge-format or server change.

The [SAGE insight, PR #621](https://github.com/dp-web4/SAGE/pull/621), treats code as executable world model, experimental instrument and prime computational effector. For Membot, the implication is that useful memory may include a **runnable procedure plus the evidence and conditions that make it useful**, rather than only a summary that the agent once solved a problem.

## What a useful remembered procedure contains

A retrieval record can point to source or a content-addressed artifact and carry its purpose, input/output contract, assumptions, dependencies, applicable environment/version, observed successes, counterexamples, revisions and provenance. These are candidate application metadata, not newly required fields in the existing cartridge schema.

Preserve the difference between source code, a description of source code, and a reference whose bytes are no longer available. A short semantic snippet may locate the artifact without being sufficient to execute or review it. Keep full code in a durable appropriate store and preserve a verifiable pointer when embedding only the description.

## Retrieval is neither applicability nor execution

The [README](../README.md) describes search and recall over swappable cartridges. Search similarity says why an item was retrieved; it does not establish that a procedure is valid for the present state. A high-ranked result may rely on old API semantics, a different grid representation or authority the recipient lacks.

Membot should remain a memory surface, not silently become an interpreter for retrieved content. A retrieved program is untrusted data until the agent checks applicability and uses an authorized execution path. Source text and comments can contain instructions; their presence in a trusted cartridge must not turn them into higher-priority instructions.

The model should be able to revise a remembered procedure instead of being forced to select an old one unchanged. Retain the original, revision lineage and counterexample; do not erase the failure that explains the change.

## Memory that can compound

A successful function can save repeated generation, but board-specific variables may become stale. Separate reusable procedure from episodic state. Record whether a later encounter merely retrieved the item, actually invoked it, adapted it correctly, or failed because its assumptions did not hold.

An experiment can compare narrative-only recall with source-plus-contract-and-evidence recall on held-out task variations, keeping model, task information and time budget fixed. Measure task success, tokens/time, applicability errors, unsafe execution attempts and successful adaptation. A retrieval hit alone is not the outcome.

## Boundaries

This note adds no execution tool, schema migration, embedding change or auto-run behavior. It makes a design target explicit: **memory should preserve the means of knowing and doing, alongside the account of what happened.** The agent remains responsible for grounding and governed use; the memory server need not perform a language-model inference to store that relationship.
