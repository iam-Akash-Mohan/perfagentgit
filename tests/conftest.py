"""Shared fixtures. Nothing here touches the network or a database."""

from __future__ import annotations

from pathlib import Path

import pytest

# A miniature Spring MVC repository. It exists so that the tool layer, the
# route parser and the citation validator can be exercised against real files
# with known line numbers instead of against one developer's clone.
FIXTURE_FILES: dict[str, str] = {
    "src/main/java/com/ex/web/OrderController.java": """package com.ex.web;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/orders")
public class OrderController {

    private final OrderService service;

    @PostMapping("/search")
    public Page<OrderDto> search(@RequestBody SearchRequest req) {
        return service.search(req);
    }

    @GetMapping("/{id}")
    public OrderDto findOne(@PathVariable Long id) {
        return service.findOne(id);
    }

    @RequestMapping(value = {"/legacy", "/old"}, method = RequestMethod.GET)
    public String legacy() {
        return "ok";
    }
}
""",
    "src/main/java/com/ex/web/HealthController.java": """package com.ex.web;

import org.springframework.web.bind.annotation.*;

@RestController
public class HealthController {

    @GetMapping(path = "/health")
    public String health() {
        return "UP";
    }

    @GetMapping("/api/orders/{id}")
    public String shadow(@PathVariable Long id) {
        return "shadow";
    }
}
""",
    "src/main/java/com/ex/service/OrderService.java": """package com.ex.service;

public class OrderService {

    public Page<OrderDto> search(SearchRequest req) {
        return repo.findByFilter(req.getName(), req.toPageable())
                   .map(this::enrich);
    }

    private OrderDto enrich(Order o) {
        return new OrderDto(o.getId(), o.getLines().size());
    }
}
""",
    "src/main/java/com/ex/repo/OrderRepository.java": """package com.ex.repo;

public interface OrderRepository {

    Page<Order> findByFilter(String name, Pageable pageable);
}
""",
    "src/main/resources/application.yml": "server:\n  port: 8080\n",
    "README.md": "fixture repo\n",
}


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A throwaway Spring-shaped repository on disk."""
    root = tmp_path / "fixture-spring"
    for relative, body in FIXTURE_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _clear_route_cache():
    """The route table caches per repo path; tmp_path reuse would poison it."""
    from senapathy.tools.routes import clear_cache

    clear_cache()
    yield
    clear_cache()
