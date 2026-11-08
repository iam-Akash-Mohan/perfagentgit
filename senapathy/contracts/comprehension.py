"""The comprehension artifact: what one endpoint does, as data.

Split out of `contracts.py` when it passed 400 lines. Every name here is
re-exported from `senapathy.contracts`, so no import path changed.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator

from .primitives import Citation


class CallHop(BaseModel):
    """One method on the execution path, with what it actually does.

    Flat rather than wrapping a Citation because this is the shape the
    diagnosis phase already reads off disk, and because `does` is not a
    property of the pointer — it is the finding.
    """
    method: str                     # e.g. CustomerService.search
    file: str
    line: int
    does: str

    def citation(self) -> "Citation":
        return Citation(file=self.file, line=self.line, symbol=self.method)


DataAccessKind = Literal[
    "jpql_query", "native_query", "derived_query",
    "lazy_collection", "lazy_association", "criteria", "jdbc_template", "other",
]

# How a data access maps onto the older SqlSource vocabulary. Derived in code
# so the model is never asked to fill in two overlapping fields.
_SQL_SOURCE_KIND: dict[str, str] = {
    "jpql_query": "@Query JPQL",
    "native_query": "@Query native",
    "derived_query": "derived query",
    "lazy_collection": "lazy collection",
    "lazy_association": "lazy collection",
    "criteria": "criteria",
    "jdbc_template": "jdbc template",
    "other": "other",
}


class DataAccess(BaseModel):
    """One way this endpoint can reach the database.

    `executes_per_row` is the field the whole phase exists for: a query that
    runs once per request and the identical query run once per result row are
    the same SQL and a completely different bottleneck.
    """
    trigger: str                    # "repo.findByFilter", "lazy load of Customer.invoices"
    kind: DataAccessKind
    file: str
    line: int
    sql: str | None = None
    tables: list[str] = []
    executes_per_row: bool
    note: str | None = None

    def citation(self) -> "Citation":
        return Citation(file=self.file, line=self.line, symbol=None)


class PerRowWork(BaseModel):
    """Work that executes once per RESULT ROW and is not a database access.

    This is the field the target endpoint needed and did not have. A loop that
    builds one string per invoice does not appear in `data_access`, does not
    appear in `sql_sources`, and was 93% of the measured latency. Every phase
    downstream was reasoning about queries because queries were the only thing
    the map could describe.
    """
    description: str        # "appends one 'overdue' string per overdue invoice"
    file: str
    line: int
    kind: Literal["string_building", "collection_processing", "object_creation",
                  "computation", "serialization", "external_call", "other"]
    scales_with: str        # what drives the count: "invoices per customer"
    bounded: bool           # is there ANY limit on how much work this does?
    bound: str | None = None    # if bounded, what bounds it: "page size, max 100"

    def citation(self) -> "Citation":
        return Citation(file=self.file, line=self.line, symbol=None)


class ResponseShape(BaseModel):
    """What the endpoint returns, and which parts of it are unbounded.

    `unbounded_collections` exists so that a later phase learns BEFORE writing
    a patch that the response size is part of the API contract. The first patch
    attempt spent four build iterations discovering that by hand.
    """
    dto: str | None = None
    fields_returned: list[str] = []
    computed_fields: list[str] = []       # getters that do real work, not accessors
    unbounded_collections: list[str] = []  # fields whose size scales with data
    citation: Citation | None = None


RequestParamType = Literal["string", "integer", "number", "boolean", "array"]


class RequestParameter(BaseModel):
    """One value the endpoint reads OFF the request — a path segment or a
    query string key. The symmetric counterpart of a field in ResponseShape.

    `allowed_values` is populated only when the CODE constrains it: an enum
    type, a `@Pattern`, an explicit check that rejects anything else. A
    required parameter with no default and no allowed_values is still a
    valid thing to report — it means the corpus generator cannot invent a
    value for it either, and must refuse by name rather than omit it.
    """
    name: str
    location: Literal["path", "query"]
    type: RequestParamType
    required: bool
    default: Any = None
    allowed_values: list[Any] = []
    citation: Citation


class RequestShape(BaseModel):
    """What the endpoint's REQUEST looks like, as data.

    Added after a generated corpus silently omitted a required query
    parameter (`status`) that nothing had recorded existed, and all 100
    baseline requests measured a 400 instead of the endpoint. `ResponseShape`
    had a home for every returned field; nothing symmetric existed for what
    the request itself requires.
    """
    parameters: list[RequestParameter] = []


class PaginationInfo(BaseModel):
    """How the endpoint pages, and what that costs.

    The measured baseline showed `SELECT COUNT(id) FROM customer` at 1,640 ms
    with no_index_used — a real finding with nowhere to be recorded.
    """
    mechanism: Literal["offset", "keyset", "none", "unbounded"]
    fires_count_query: bool = False
    max_page_size: int | None = None      # None means uncapped
    citation: Citation | None = None


class CacheInfo(BaseModel):
    kind: Literal["spring_cacheable", "hibernate_second_level",
                  "query_cache", "manual", "other"]
    on: str                               # method or entity
    cache_name: str | None = None
    citation: Citation


class HiddenQueryTrigger(BaseModel):
    """SQL that fires from somewhere other than the call site.

    @Formula, @PostLoad, entity listeners, Hibernate filters. Invisible when
    reading the service layer, and a classic source of "where did that query
    come from".
    """
    kind: Literal["formula", "post_load", "pre_persist", "entity_listener",
                  "hibernate_filter", "audit", "other"]
    on: str                               # entity or field
    detail: str
    citation: Citation


class IndexInfo(BaseModel):
    name: str
    columns: list[str]          # in index order — order matters for composites
    unique: bool
    cardinality: int | None = None


class TableInfo(BaseModel):
    table: str
    entity: str | None = None
    columns_read: list[str] = []      # columns this endpoint actually reads
    all_columns: list[str] = []
    existing_indexes: list[IndexInfo] = []
    approx_rows: int | None = None
    foreign_keys: list[dict[str, str]] = []


class SqlSource(BaseModel):
    method: str
    kind: Literal["@Query JPQL", "@Query native", "derived query",
                  "lazy collection", "criteria", "jdbc template", "other"]
    tables: list[str] = []
    citation: Citation


class EntityRelation(BaseModel):
    """Facts about a mapping, and nothing else.

    `batch_size` and `fetch_mode` are recorded because Hibernate's behaviour
    is unreadable without them. Whether an already-batched relation still
    needs work is diagnosis's call, not comprehension's — so there is
    deliberately no `already_optimized` flag here.
    """
    field: str
    type: Literal["OneToMany", "ManyToOne", "OneToOne", "ManyToMany"]
    fetch: Literal["LAZY", "EAGER"]
    target: str
    batch_size: int | None = None                                      # @BatchSize
    fetch_mode: Literal["SELECT", "JOIN", "SUBSELECT"] | None = None   # @Fetch


class EntityInfo(BaseModel):
    name: str
    table: str
    citation: Citation
    relations: list[EntityRelation] = []


class TransactionInfo(BaseModel):
    method: str
    propagation: str
    read_only: bool = False
    citation: Citation


class ExternalCall(BaseModel):
    method: str
    target: str
    kind: Literal["http", "grpc", "queue", "other"]
    citation: Citation


class EndpointModel(BaseModel):
    """The entire output of the comprehension phase.

    `call_chain` / `data_access` are what an agent submits. `entry`, `path`
    and `sql_sources` are DERIVED from them in `_derive` below rather than
    asked for separately: two fields describing the same fact are two fields
    that can disagree, and the model does not get to decide which is right.
    """
    route: str
    entry: Citation
    path: list[Citation]
    call_chain: list[CallHop] = []
    data_access: list[DataAccess] = []
    sql_sources: list[SqlSource] = []
    entities: list[EntityInfo] = []
    transactions: list[TransactionInfo] = []
    external_calls: list[ExternalCall] = []
    tables: list[TableInfo] = []
    per_row_work: list[PerRowWork] = []
    response_shape: ResponseShape | None = None
    request_shape: RequestShape | None = None
    pagination: PaginationInfo | None = None
    caching: list[CacheInfo] = []
    hidden_query_triggers: list[HiddenQueryTrigger] = []
    tests_covering: list[str] = []
    uncertainties: list[str] = Field(
        default_factory=list,
        description="What the agent was NOT sure about. An agent forced to "
                    "declare uncertainty is more useful than a confident one.")
    prompt_hash: str | None = None      # which prompt version produced this
    base_commit: str | None = Field(
        default=None,
        description="The git sha of the target repo at analysis time, filled "
                    "by deterministic code and never by the model. Without it "
                    "there is no way to tell a stale artifact from a current "
                    "one, and resume logic is guessing.")

    @model_validator(mode="before")
    @classmethod
    def _derive(cls, data: Any) -> Any:
        """Fill entry / path / sql_sources from call_chain / data_access.

        Only fills what is absent, so hand-built models that supply the older
        fields directly keep working.
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)

        chain = out.get("call_chain") or []
        hops = [h if isinstance(h, CallHop) else CallHop.model_validate(h)
                for h in chain]
        if not out.get("entry") and hops:
            out["entry"] = hops[0].citation()
        if not out.get("path") and hops:
            out["path"] = [h.citation() for h in hops]

        access = out.get("data_access") or []
        if not out.get("sql_sources") and access:
            reads = [a if isinstance(a, DataAccess) else DataAccess.model_validate(a)
                     for a in access]
            out["sql_sources"] = [
                SqlSource(method=a.trigger,
                          kind=cast(Any, _SQL_SOURCE_KIND.get(a.kind, "other")),
                          tables=list(a.tables),
                          citation=a.citation())
                for a in reads
            ]
        return out

    def per_row_accesses(self) -> list[DataAccess]:
        """Computed here, never counted by a model (invariant 5)."""
        return [a for a in self.data_access if a.executes_per_row]

    def unbounded_per_row_work(self) -> list[PerRowWork]:
        """The findings that usually matter most. Computed, never counted."""
        return [w for w in self.per_row_work if not w.bounded]

    def all_citations(self) -> list[Citation]:
        out: list[Citation] = [self.entry, *self.path]
        out.extend(s.citation for s in self.sql_sources)
        out.extend(e.citation for e in self.entities)
        out.extend(t.citation for t in self.transactions)
        out.extend(c.citation for c in self.external_calls)
        out.extend(w.citation() for w in self.per_row_work)
        out.extend(c.citation for c in self.caching)
        out.extend(h.citation for h in self.hidden_query_triggers)
        if self.response_shape is not None and self.response_shape.citation:
            out.append(self.response_shape.citation)
        if self.request_shape is not None:
            out.extend(p.citation for p in self.request_shape.parameters)
        if self.pagination is not None and self.pagination.citation:
            out.append(self.pagination.citation)
        return out
