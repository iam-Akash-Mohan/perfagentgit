# Java standards for generated patches

Loaded into the patch prompt at runtime. Edit this file to change what the
agent is told — no Python change required.

Four of these rules are **enforced**: a lint pass reads the diff and fails the
phase if it finds one. The rest are prompt guidance, because they need taste
rather than a regex. `docs/patch-standards.md` says which is which and why.

## Required

- Constructor injection only. No field `@Autowired`.                [ENFORCED]
- `@Transactional(readOnly = true)` on read-only service methods.
- New repository query methods carry a Javadoc line stating what they return.
- Projections are interfaces or records, not entity subclasses.
- New query methods are named for what they return, not how they work.
- A JPQL constructor expression (`SELECT NEW com.ex.dto.Dto(...)`) needs no
  `@SqlResultSetMapping` — that annotation is for native queries only.
  Prefer the JPQL form for a DTO projection: it stays inside the repository
  interface and needs no entity annotation or second file.
- Never widen a repository method's parameter from `LocalDate` to
  `LocalDateTime` to compare it against a `DATETIME` column. A bound
  `LocalDateTime` parameter is converted using the JVM's timezone and the
  datasource's `serverTimezone`; when those differ, the comparison silently
  shifts by the offset and the query returns the wrong rows with no error.
  Keep the method's parameters as `LocalDate` and widen the comparison on
  the parameter side instead — wrap the parameter in a function that
  produces a timestamp, not the indexed column, so the index is still used
  and the caller's argument types never change.

## Forbidden

- No new static mutable state.
- No `new` on a Spring-managed type.                                [ENFORCED]
- No raw JDBC outside the repository layer.
- No changes to public method signatures on controllers.            [ENFORCED]

## Style

- Prefer `Optional<T>` over null returns.
- Streams where they read more clearly than an index loop; not otherwise.
- Keep imports explicit; no wildcard imports.                       [ENFORCED]
