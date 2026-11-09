from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

import httpx

from ..contracts import RequestCorpus, RequestSpec, SqlDigestStat
from ..errors import ContractViolation, ToolError
from ..measure.baseline import is_session_control
from ..measure.bench import latency_stats
from ..measure.sql import MySQLTarget, collect_digests, reset_digests

# Hard caps. The agent picks the numbers, so the numbers have to be bounded:
# page_size=100000 against a 50k-row table with a per-row query is a denial of
# service against the thing we are trying to measure.
MAX_ITERATIONS = 20
DEFAULT_MAX_PAGE_SIZE = 100
PROBE_WARMUP = 1
PROBE_TIMEOUT_S = 180.0

_SIZE_QUERY_RE = re.compile(r"([?&]size=)(\d+)")


class ProbeLog:
    """Record of probes that were ACTUALLY executed.

    The agent also reports the probes it ran, but a self-reported probe list is
    a claim, not evidence. Validation checks the agent's story against this.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def record(self, args: dict[str, Any], summary: str, *,
              method: str | None = None, url: str | None = None) -> None:
        entry: dict[str, Any] = {"tool": "run_probe", "args": args, "result": summary}
        if method is not None and url is not None:
            entry["endpoint_called"] = f"{method} {url}"
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)


def _app_sql(digests: list[SqlDigestStat]) -> tuple[int, float]:
    """(statement count, total ms) for application SQL only."""
    app = [d for d in digests if not is_session_control(d)]
    return sum(d.calls for d in app), sum(d.total_ms for d in app)


def _unbounded_collection_name(endpoint_model: dict[str, Any]) -> str | None:
    shape = endpoint_model.get("response_shape") or {}
    fields = shape.get("unbounded_collections") or []
    for entry in fields:
        name = str(entry).split("(")[0].strip().strip("`'\" ")
        if name:
            return name
    return None


def _count_rows(body: bytes, endpoint_model: dict[str, Any]) -> int | None:
    """Result rows in a probe response, by shape — not by this one
    endpoint's convention. Tried in order:

    1. the whole body is a JSON array
    2. a Spring `Page<>` response: a dict with a `content` list (the
       convention this app's paginated GET endpoints already use)
    3. the endpoint's own declared unbounded response field (from
       response_shape.unbounded_collections), if the body has it as a list

    None — not 0 — when nothing matches, so a probe against a response shape
    this cannot read is reported honestly as unparsed rather than silently
    scored as zero rows.
    """
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, dict):
        content = parsed.get("content")
        if isinstance(content, list):
            return len(content)
        field = _unbounded_collection_name(endpoint_model)
        if field and isinstance(parsed.get(field), list):
            return len(parsed[field])
    return None


class ProbeCatalog:
    """What `run_probe` is allowed to fire, derived from THIS run's own
    endpoint model and corpus — never a fixed route.

    Built once, at diagnose setup, so a run against an endpoint this cannot
    build a probe for fails LOUDLY and immediately (`ContractViolation`),
    rather than silently measuring a different endpoint instead. That
    silent fallback is exactly what made every diagnosis in Scenarios 7 and
    8 describe `POST /api/customers/search` — a real, correctly-measured
    N+1 on an endpoint nobody asked about — while reporting it as if it
    described the endpoint actually under test.
    """

    def __init__(self, base_url: str, endpoint_model: dict[str, Any],
                corpus: RequestCorpus) -> None:
        requests = list(corpus.benchmark)
        if not requests:
            raise ContractViolation(
                "the corpus has no benchmark requests to probe — diagnosis "
                "cannot build an experiment for an endpoint with nothing to "
                "replay."
            )
        non_get = [r for r in requests if r.method.upper() != "GET"]
        if non_get:
            raise ContractViolation(
                f"{len(non_get)} corpus request(s) are not GET "
                f"(e.g. {non_get[0].method} {non_get[0].path}) — run_probe "
                f"only knows how to replay a GET request with no body, "
                f"the only shape this project's corpus generator produces. "
                f"A POST/PUT/DELETE endpoint has no probe support yet."
            )

        self.base_url = base_url.rstrip("/")
        self.endpoint_model = endpoint_model
        self.by_id: dict[str, RequestSpec] = {r.id: r for r in requests}

        pagination = endpoint_model.get("pagination") or {}
        mechanism = pagination.get("mechanism")
        # Sanity check the derived axis against reality: pagination exists
        # in the model AND at least one request actually carries a size=
        # parameter (the one convention this project's own corpus generator
        # uses for every paginating endpoint — see workload.py's
        # _PAGINATION_PARAM_NAMES). Either alone, without the other, means
        # the two disagree and the size axis is not trustworthy.
        has_size_param = any(_SIZE_QUERY_RE.search(r.path) for r in requests)
        self.paginates = (mechanism in ("offset", "keyset") and has_size_param)
        self.max_page_size = pagination.get("max_page_size") or DEFAULT_MAX_PAGE_SIZE

    def describe_requests(self) -> str:
        lines = []
        for request_id, spec in self.by_id.items():
            note = f" — {spec.note}" if spec.note else ""
            lines.append(f"  {request_id}: {spec.method} {spec.path}{note}")
        return "\n".join(lines)

    def build(self, request_id: str, page_size: int | None) -> tuple[str, str]:
        """(method, url) for one probe call, or raises ToolError naming why."""
        spec = self.by_id.get(request_id)
        if spec is None:
            raise ToolError(
                f"{request_id!r} is not one of this run's own corpus "
                f"requests. Available ids:\n{self.describe_requests()}",
                suggestion="Pick a request_id from the list above.",
            )
        if page_size is not None and not self.paginates:
            raise ToolError(
                f"this endpoint has no size parameter to vary "
                f"(pagination mechanism: "
                f"{(self.endpoint_model.get('pagination') or {}).get('mechanism', 'none')}) "
                f"— page_size must be omitted.",
                suggestion="Omit page_size. Vary request_id instead to probe "
                           "a different magnitude (see the ids and their "
                           "notes you were given).",
            )
        path = spec.path
        if page_size is not None:
            bounded = max(1, min(int(page_size), self.max_page_size))
            path, count = _SIZE_QUERY_RE.subn(
                lambda m: f"{m.group(1)}{bounded}", path)
            if count == 0:
                raise ToolError(
                    f"request {request_id!r} has no size= parameter to "
                    f"override even though this endpoint paginates — its "
                    f"own path is {spec.path!r}.",
                )
        return spec.method, self.base_url + path


def make_run_probe(
    base_url: str,
    target: MySQLTarget,
    *,
    endpoint_model: dict[str, Any],
    corpus: RequestCorpus,
    log: ProbeLog | None = None,
    catalog: ProbeCatalog | None = None,
) -> Callable[..., str]:
    """Build the `run_probe` handler for THIS run's own endpoint.

    The route, method and request shape come from `endpoint_model`/`corpus` —
    the same comprehension model and generated corpus every other phase
    already uses — never a fixed path. The only thing the agent can still
    choose is WHICH of this endpoint's own corpus requests to fire
    (`request_id`, varying magnitude the same way the benchmark/held-back
    split already does) and, when this endpoint paginates, a `page_size`
    override (clamped, same discipline as before).

    Raises at construction (not at the first tool call) when no probe can
    be built for this endpoint at all — see `ProbeCatalog`. Pass an
    already-built `catalog` (the caller needs one anyway, to build
    `run_probe_tool`'s schema, before the handler) rather than validating
    the same corpus twice.
    """
    catalog = catalog or ProbeCatalog(base_url, endpoint_model, corpus)
    probe_log = log if log is not None else ProbeLog()

    def run_probe(request_id: str, iterations: int = 10,
                 page_size: int | None = None) -> str:
        try:
            iterations = int(iterations)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"iterations must be an integer: {exc}") from exc
        if page_size is not None:
            try:
                page_size = int(page_size)
            except (TypeError, ValueError) as exc:
                raise ToolError(f"page_size must be an integer: {exc}") from exc

        notes: list[str] = []
        if iterations > MAX_ITERATIONS:
            notes.append(f"iterations clamped from {iterations} to {MAX_ITERATIONS}")
            iterations = MAX_ITERATIONS
        if iterations < 1:
            notes.append(f"iterations raised from {iterations} to 1")
            iterations = 1

        method, url = catalog.build(request_id, page_size)

        samples_ms: list[float] = []
        body_sizes: list[int] = []
        statuses: list[int] = []
        last_body: bytes = b""

        try:
            with httpx.Client(timeout=httpx.Timeout(PROBE_TIMEOUT_S)) as client:
                for _ in range(PROBE_WARMUP):
                    client.request(method, url).read()

                # Reset AFTER warmup, immediately before the measured requests,
                # so the digests describe exactly these `iterations` calls.
                reset_digests(target)

                for _ in range(iterations):
                    started = time.perf_counter()
                    response = client.request(method, url)
                    content = response.content
                    samples_ms.append((time.perf_counter() - started) * 1000.0)
                    body_sizes.append(len(content))
                    statuses.append(response.status_code)
                    last_body = content

                digests = collect_digests(target)
        except httpx.HTTPError as exc:
            raise ToolError(
                f"probe could not reach {url}: {type(exc).__name__}: {exc}",
                suggestion="The app may have stopped. Do not retry more than once.",
            ) from exc

        statements, sql_ms = _app_sql(digests)
        stats = latency_stats(samples_ms)
        median_body = sorted(body_sizes)[len(body_sizes) // 2]
        rows = _count_rows(last_body, catalog.endpoint_model)

        status_counts: dict[str, int] = {}
        for code in statuses:
            status_counts[str(code)] = status_counts.get(str(code), 0) + 1

        sql_ms_per_request = sql_ms / iterations
        sql_share = (100.0 * sql_ms_per_request / stats.p50_ms) if stats.p50_ms else 0.0

        out: list[str] = []
        out.append(
            f"probe: {method} {url}  (request_id={request_id!r}"
            + (f", page_size={page_size}" if page_size is not None else "")
            + f") iterations={iterations} ({PROBE_WARMUP} warmup discarded)"
        )
        for note in notes:
            out.append(f"  NOTE: {note}")
        out.append(f"  statuses                     : {status_counts}")
        out.append(f"  p50 latency                  : {stats.p50_ms:.1f} ms")
        out.append(f"  p95 latency                  : {stats.p95_ms:.1f} ms")
        out.append(
            f"  result rows in response      : {rows if rows is not None else 'unparsed'}"
        )
        out.append(f"  response body size (median)  : {median_body} bytes")
        out.append(f"  application SQL per request  : {statements / iterations:.1f} statements")
        out.append(
            f"  server-side SQL per request  : {sql_ms_per_request:.1f} ms "
            f"({sql_share:.1f} % of p50)"
        )
        if rows:
            out.append(f"  latency per result row       : {stats.p50_ms / rows:.1f} ms")
            out.append(f"  body bytes per result row    : {median_body // rows}")
        summary = "\n".join(out)

        args: dict[str, Any] = {"request_id": request_id, "iterations": iterations}
        if page_size is not None:
            args["page_size"] = page_size
        probe_log.record(args, summary, method=method, url=url)
        return summary

    return run_probe


def run_probe_tool(catalog_description: str, *, paginates: bool,
                   max_page_size: int) -> dict[str, Any]:
    """RUN_PROBE_TOOL's schema, parameterised by what THIS endpoint actually
    supports — an endpoint with no size parameter is not offered one."""
    properties: dict[str, Any] = {
        "request_id": {
            "type": "string",
            "description": (
                "Which of this endpoint's own corpus requests to fire, "
                "repeatedly. Each id varies this endpoint's own real "
                "magnitude (a denser or sparser path parameter, a deeper "
                "page) — the same requests the benchmark phase itself "
                "uses, not an invented one. Available ids and what they "
                f"are:\n{catalog_description}"
            ),
        },
        "iterations": {
            "type": "integer",
            "description": f"Measured requests, default 10, maximum {MAX_ITERATIONS}.",
        },
    }
    required = ["request_id"]
    if paginates:
        properties["page_size"] = {
            "type": "integer",
            "description": (
                f"Override this request's own page size, 1 to "
                f"{max_page_size}. Sweep it to find out whether cost "
                f"scales with the number of result rows. Omit to use the "
                f"request's own recorded size unchanged."
            ),
        }
    return {
        "type": "function",
        "function": {
            "name": "run_probe",
            "description": (
                "Run a live experiment against THIS RUN'S OWN ENDPOINT and "
                "measure what happens. Fires one of this endpoint's own "
                "corpus requests `iterations` times, resets and re-reads "
                "performance_schema around the measured calls, and reports "
                "p50 latency, application SQL statements per request, "
                "server-side SQL ms per request, result row count and "
                "response body size.\n\n"
                "This is READ-ONLY and cannot call any other endpoint, "
                "verb or body shape — only the requests this run's own "
                "corpus already generated for the endpoint under test.\n\n"
                f"Caps: iterations at most {MAX_ITERATIONS} (clamped, not "
                "rejected)."
                + (f" page_size at most {max_page_size} (clamped)."
                   if paginates else
                   " This endpoint does not paginate — vary request_id "
                   "instead to probe a different magnitude.")
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
