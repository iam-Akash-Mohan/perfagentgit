# Java standards: what is enforced, what is advice

`standards/java.md` is loaded into the patch prompt verbatim at runtime. Edit
that file to change what the agent is told; no Python change is needed.

Four of its rules are also **enforced**: `senapathy/patch/standards.py` lints
the diff when the agent calls `emit_patch`, and a violation is a rejection the
agent must fix before the patch can be accepted. The rest are prompt guidance.

## Enforced — the lint fails the phase

| Rule | How it is checked |
|---|---|
| Constructor injection only. No field `@Autowired`. | An added `@Autowired` whose next added line is a field declaration rather than a constructor or setter. |
| No wildcard imports. | An added `import x.y.*;`. |
| No `new` on a Spring-managed type. | The stereotype annotations (`@Service`, `@Component`, `@Repository`, `@Controller`, `@RestController`, `@Configuration`) are read out of the target repo's sources, and any added `new <ThatType>(` is a violation. Read rather than guessed from the name: a `@Component` called `Clock` is container-managed and a plain `OrderCalculator` is not. |
| No changes to public method signatures on controllers. | A removed public method in a file carrying a mapping annotation, or one kept by name with different parameters. |

Each violation names the rule it broke. "Fix the lint" is not actionable;
"constructor injection only" is.

## Advisory — prompt only

`@Transactional(readOnly = true)` on read-only service methods · Javadoc on new
repository query methods · projections as interfaces or records · query methods
named for what they return · no new static mutable state · no raw JDBC outside
the repository layer · `Optional<T>` over null · streams where they read better.

These need taste rather than a regex. A lint that rejects correct code is a
lint someone switches off, and then the four that matter stop running too.

## Why the lint runs at `emit_patch` and not at `apply_patch`

A single edit is not a diff. "No field `@Autowired`" can only be judged against
the finished change — an agent may add the annotation in one call and convert
it to constructor injection in the next, and refusing the first edit would stop
work that was about to be correct. The lint therefore runs at the last possible
moment, which is also the first moment the whole diff exists.
