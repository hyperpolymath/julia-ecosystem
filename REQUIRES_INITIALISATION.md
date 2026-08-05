<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# REQUIRES INITIALISATION

**This repository is not finished being set up.** 32 substitution token(s) across 82 file(s) still have no value.

## Why this is not already done

This repo was created from `hyperpolymath/rsr-template-repo`. The mint
(`just repo-init`) fills every token that has a single mechanical answer —
owner, repo, author, dates, licence, branch — and it has done so here.

The tokens below are the ones it *deliberately cannot* answer. They need a
decision or a fact that exists only in your head: what this project is for,
what command builds it, which port the service listens on, whether a PGP key
is held at all. The template's own token vocabulary says as much — you cannot
sensibly answer "required invariants" in a thirty-second bootstrap.

They were left **visibly unfilled on purpose**. The alternatives were both
worse: inventing plausible values would put confident falsehoods into a
security policy and an architecture document, and silently deleting the
sections would hide the fact that a decision is owed. A visible gap is
honest; a fabricated answer is not.

## Do not delete this file until every item below is resolved

This file is the only marker that the work is outstanding. Deleting it early
does not finish the setup, it just conceals it — and the next person or agent
to arrive will reasonably assume the repo is complete.

- **If you are a person:** delete this file yourself once the last item is done.
- **If you are an agent:** resolve what you legitimately can, leave the rest,
  and delete this file only when no token below remains anywhere in the tree.
  Do not delete it to make a gate go green.

Re-running the estate top-up tool will remove this file automatically once
nothing is outstanding, so the safest way to finish is to fix the tokens and
let the check confirm it.

## What is needed, and where it goes

### `{{AUTHOR_EMAIL_ALT}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.mailmap`
- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/.mailmap`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/.mailmap`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/PRComms.jl/.mailmap`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PostDisciplinary.jl/.mailmap`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/TradeUnionist.jl/.mailmap`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`

### `{{AUTHOR_ORG}}`

Author's organisation. NOTE: no filled instance of this exists anywhere in the estate — consider deleting the field instead.

Appears in:

- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`

### `{{BUILD_CMD}}`

The exact command that builds this project.

Appears in:

- `QUICKSTART-DEV.adoc`

### `{{BUILD_OUTPUT_PATH}}`

Where the build artefact lands.

Appears in:

- `QUICKSTART-MAINTAINER.adoc`

### `{{CONDUCT_TEAM}}`

Name of the conduct body. If there is no committee, rewrite the sentence rather than substituting a plural noun into 'a {{CONDUCT_TEAM}} member'.

Appears in:

- `packages/Axiology.jl/CODE_OF_CONDUCT.md`
- `packages/Causals.jl/CODE_OF_CONDUCT.md`
- `packages/Cladistics.jl/CODE_OF_CONDUCT.md`
- `packages/Cliodynamics.jl/CODE_OF_CONDUCT.md`
- `packages/Cliometrics.jl/CODE_OF_CONDUCT.md`
- `packages/Exnovation.jl/SONNET-TASKS.md`
- `packages/HackenbushGames.jl/SONNET-TASKS.md`
- `packages/InvestigativeJournalist.jl/CODE_OF_CONDUCT.md`
- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/CODE_OF_CONDUCT.md`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/CODE_OF_CONDUCT.md`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/KnotTheory.jl/CODE_OF_CONDUCT.md`
- `packages/KnotTheory.jl/SONNET-TASKS.md`
- `packages/PRComms.jl/CODE_OF_CONDUCT.md`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PolyglotFormalisms.jl/CODE_OF_CONDUCT.md`
- `packages/PostDisciplinary.jl/CODE_OF_CONDUCT.md`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/ProvenCrypto.jl/CODE_OF_CONDUCT.md`
- `packages/TradeUnionist.jl/CODE_OF_CONDUCT.md`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`
- `packages/ZeroProb.jl/CODE_OF_CONDUCT.md`

### `{{DEPS}}`

Prose summary of runtime/build dependencies.

Appears in:

- `QUICKSTART-MAINTAINER.adoc`

### `{{DILITHIUM5_PUBLIC_KEY}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{DOMAIN}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{ED448_PUBLIC_KEY}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{EXPIRES_AT}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{FALLBACK_SIGNATURE}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{GENERATED_AT}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{LANG_STACK}}`

The language stack, in prose.

Appears in:

- `QUICKSTART-DEV.adoc`

### `{{LICENSE}}`

SPDX identifier for this repo's licence.

Appears in:

- `packages/Axiology.jl/ABI-FFI-README.md`
- `packages/Causals.jl/ABI-FFI-README.md`
- `packages/Cladistics.jl/ABI-FFI-README.md`
- `packages/Cliometrics.jl/ABI-FFI-README.md`
- `packages/InvestigativeJournalist.jl/ABI-FFI-README.md`
- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/ABI-FFI-README.md`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/ABI-FFI-README.md`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/KnotTheory.jl/ABI-FFI-README.md`
- `packages/KnotTheory.jl/SONNET-TASKS.md`
- `packages/PRComms.jl/ABI-FFI-README.md`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PolyglotFormalisms.jl/ABI-FFI-README.md`
- `packages/PostDisciplinary.jl/ABI-FFI-README.md`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/ProvenCrypto.jl/ABI-FFI-README.md`
- `packages/SMTLib.jl/SONNET-TASKS.md`
- `packages/TradeUnionist.jl/ABI-FFI-README.md`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`
- `packages/ZeroProb.jl/ABI-FFI-README.md`

### `{{MUST_INVARIANTS}}`

The invariants this project guarantees. Not answerable in a bootstrap; it is the point of the repo.

Appears in:

- `QUICKSTART-DEV.adoc`

### `{{PGP_FINGERPRINT}}`

Full fingerprint of the security-contact PGP key. NOTE: no key is published anywhere in this estate — if none is held, delete the PGP block rather than inventing one.

Appears in:

- `packages/Axiology.jl/SECURITY.md`
- `packages/Causals.jl/SECURITY.md`
- `packages/Cladistics.jl/SECURITY.md`
- `packages/Cliodynamics.jl/SECURITY.md`
- `packages/Cliometrics.jl/SECURITY.md`
- `packages/Exnovation.jl/SONNET-TASKS.md`
- `packages/HackenbushGames.jl/SONNET-TASKS.md`
- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/InvestigativeJournalist.jl/SECURITY.md`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/SECURITY.md`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/SECURITY.md`
- `packages/KnotTheory.jl/SECURITY.md`
- `packages/KnotTheory.jl/SONNET-TASKS.md`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PRComms.jl/SECURITY.md`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/PostDisciplinary.jl/SECURITY.md`
- `packages/ProvenCrypto.jl/SECURITY.md`
- `packages/SMTLib.jl/SONNET-TASKS.md`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`
- `packages/TradeUnionist.jl/SECURITY.md`
- `packages/ZeroProb.jl/SECURITY.md`

### `{{PGP_KEY_URL}}`

Public URL the PGP key can be fetched from. Same caveat as PGP_FINGERPRINT.

Appears in:

- `packages/Axiology.jl/SECURITY.md`
- `packages/Causals.jl/SECURITY.md`
- `packages/Cladistics.jl/SECURITY.md`
- `packages/Cliodynamics.jl/SECURITY.md`
- `packages/Cliometrics.jl/SECURITY.md`
- `packages/Exnovation.jl/SONNET-TASKS.md`
- `packages/HackenbushGames.jl/SONNET-TASKS.md`
- `packages/InvestigativeJournalist.jl/.well-known/security.txt`
- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/InvestigativeJournalist.jl/SECURITY.md`
- `packages/JuliaKids.jl/.well-known/security.txt`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/SECURITY.md`
- `packages/JuliaPackage-Reuse-Audit.jl/.well-known/security.txt`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/SECURITY.md`
- `packages/KnotTheory.jl/SECURITY.md`
- `packages/KnotTheory.jl/SONNET-TASKS.md`
- `packages/PRComms.jl/.well-known/security.txt`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PRComms.jl/SECURITY.md`
- `packages/PostDisciplinary.jl/.well-known/security.txt`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/PostDisciplinary.jl/SECURITY.md`
- `packages/ProvenCrypto.jl/SECURITY.md`
- `packages/TradeUnionist.jl/.well-known/security.txt`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`
- `packages/TradeUnionist.jl/SECURITY.md`
- `packages/ZeroProb.jl/SECURITY.md`

### `{{PLACEHOLDERS}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{PRIMARY_SIGNATURE}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{PROJECT_DESCRIPTION}}`

One-line description, matching the forge description.

Appears in:

- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`

### `{{PROJECT_PURPOSE}}`

One line: what this exists to do.

Appears in:

- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/InvestigativeJournalist.jl/guix.scm`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/guix.scm`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/guix.scm`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PRComms.jl/guix.scm`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/PostDisciplinary.jl/guix.scm`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`
- `packages/TradeUnionist.jl/guix.scm`

### `{{PROJECT_UNIQUE_STRENGTH}}`

What this does that its alternatives do not.

Appears in:

- `.machine_readable/bot_directives/methodology.a2ml`

### `{{RESPONSE_TIME}}`

Initial-response SLA for a security or conduct report. Promise only what a solo maintainer can actually meet.

Appears in:

- `packages/Axiology.jl/CODE_OF_CONDUCT.md`
- `packages/Causals.jl/CODE_OF_CONDUCT.md`
- `packages/Cladistics.jl/CODE_OF_CONDUCT.md`
- `packages/Cliodynamics.jl/CODE_OF_CONDUCT.md`
- `packages/Cliometrics.jl/CODE_OF_CONDUCT.md`
- `packages/Exnovation.jl/SONNET-TASKS.md`
- `packages/HackenbushGames.jl/SONNET-TASKS.md`
- `packages/InvestigativeJournalist.jl/CODE_OF_CONDUCT.md`
- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/CODE_OF_CONDUCT.md`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/CODE_OF_CONDUCT.md`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/KnotTheory.jl/CODE_OF_CONDUCT.md`
- `packages/KnotTheory.jl/SONNET-TASKS.md`
- `packages/PRComms.jl/CODE_OF_CONDUCT.md`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PolyglotFormalisms.jl/CODE_OF_CONDUCT.md`
- `packages/PostDisciplinary.jl/CODE_OF_CONDUCT.md`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/ProvenCrypto.jl/CODE_OF_CONDUCT.md`
- `packages/TradeUnionist.jl/CODE_OF_CONDUCT.md`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`
- `packages/ZeroProb.jl/CODE_OF_CONDUCT.md`

### `{{SECURITY_EMAIL}}`

Address for private vulnerability reports. Two competing values exist in the estate (`6759885+hyperpolymath@users.noreply.github.com` and `security@hyperpolymath.org`) — pick one deliberately.

Appears in:

- `packages/Axiology.jl/SECURITY.md`
- `packages/BowtieRisk.jl/SONNET-TASKS.md`
- `packages/Causals.jl/SECURITY.md`
- `packages/Cladistics.jl/SECURITY.md`
- `packages/Cliodynamics.jl/SECURITY.md`
- `packages/Cliometrics.jl/SONNET-TASKS.md`
- `packages/Exnovation.jl/SONNET-TASKS.md`
- `packages/HackenbushGames.jl/SONNET-TASKS.md`
- `packages/InvestigativeJournalist.jl/.well-known/security.txt`
- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/InvestigativeJournalist.jl/SECURITY.md`
- `packages/JuliaKids.jl/.well-known/security.txt`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/SECURITY.md`
- `packages/JuliaPackage-Reuse-Audit.jl/.well-known/security.txt`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/SECURITY.md`
- `packages/KnotTheory.jl/SECURITY.md`
- `packages/KnotTheory.jl/SONNET-TASKS.md`
- `packages/PRComms.jl/.well-known/security.txt`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PRComms.jl/SECURITY.md`
- `packages/PostDisciplinary.jl/.well-known/security.txt`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/PostDisciplinary.jl/SECURITY.md`
- `packages/ProvenCrypto.jl/SECURITY.md`
- `packages/SMTLib.jl/SONNET-TASKS.md`
- `packages/TradeUnionist.jl/.well-known/security.txt`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`
- `packages/TradeUnionist.jl/SECURITY.md`
- `packages/ZeroProb.jl/SECURITY.md`

### `{{SHA3_512}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{SHAKE256}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{SPHINCS_PLUS_PUBLIC_KEY}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{TEST_CMD}}`

The exact command that runs its tests.

Appears in:

- `QUICKSTART-DEV.adoc`

### `{{TRUSTFILE_PATH}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{TRUSTFILE_VERSION}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

### `{{WEBSITE}}`

Project homepage URL, or delete the field if there is none.

Appears in:

- `packages/Axiology.jl/SECURITY.md`
- `packages/Causals.jl/SECURITY.md`
- `packages/Cladistics.jl/SECURITY.md`
- `packages/Cliodynamics.jl/SECURITY.md`
- `packages/Cliometrics.jl/SECURITY.md`
- `packages/Exnovation.jl/SONNET-TASKS.md`
- `packages/HackenbushGames.jl/SONNET-TASKS.md`
- `packages/InvestigativeJournalist.jl/.well-known/security.txt`
- `packages/InvestigativeJournalist.jl/PLACEHOLDERS.md`
- `packages/InvestigativeJournalist.jl/SECURITY.md`
- `packages/JuliaKids.jl/.well-known/security.txt`
- `packages/JuliaKids.jl/PLACEHOLDERS.md`
- `packages/JuliaKids.jl/SECURITY.md`
- `packages/JuliaPackage-Reuse-Audit.jl/.well-known/security.txt`
- `packages/JuliaPackage-Reuse-Audit.jl/PLACEHOLDERS.md`
- `packages/JuliaPackage-Reuse-Audit.jl/SECURITY.md`
- `packages/KnotTheory.jl/SECURITY.md`
- `packages/KnotTheory.jl/SONNET-TASKS.md`
- `packages/PRComms.jl/.well-known/security.txt`
- `packages/PRComms.jl/PLACEHOLDERS.md`
- `packages/PRComms.jl/SECURITY.md`
- `packages/PostDisciplinary.jl/.well-known/security.txt`
- `packages/PostDisciplinary.jl/PLACEHOLDERS.md`
- `packages/PostDisciplinary.jl/SECURITY.md`
- `packages/ProvenCrypto.jl/SECURITY.md`
- `packages/TradeUnionist.jl/.well-known/security.txt`
- `packages/TradeUnionist.jl/PLACEHOLDERS.md`
- `packages/TradeUnionist.jl/SECURITY.md`
- `packages/ZeroProb.jl/SECURITY.md`

### `{{ZONEMD}}`

Appears in:

- `packages/InvestigativeJournalist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaKids.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/JuliaPackage-Reuse-Audit.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PRComms.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/PostDisciplinary.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`
- `packages/TradeUnionist.jl/.machine_readable/contractiles/trust/Trustfile.a2ml`

---

Generated by the estate top-up pass. Rationale and the governing rulings are
in `hyperpolymath/standards`; the token vocabulary is
`.machine_readable/ai/PLACEHOLDERS.adoc` in `rsr-template-repo`.
