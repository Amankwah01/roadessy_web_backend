"""The road GeoJSON endpoints must not be able to take the API down with them.

`GET /api/analysis/roads` used to hold the event loop for the length of a
country-sized query. It is an `async def` that called `db.execute(...).fetchall()`
directly, so psycopg2 blocked the loop rather than yielding to it, and while it
ran nothing else on the server could be scheduled: `/health` stopped answering,
every other route timed out, and one unauthenticated GET was enough to do it.

Two things caused that and both are pinned here, because either one alone is
enough to bring it back:

* the handler being a coroutine, so its blocking query ran on the event loop;
* the work being unbounded and unauthenticated, so there was nothing to stop a
  caller asking for all 370,374 roads (280 MB of GeoJSON) in the first place.

Nothing here touches the road table. These are the properties that made the
outage possible, and they are all decidable without a database.
"""
import inspect

import pytest
from fastapi import HTTPException

from app.api.routes import analysis
from app.api.routes.analysis import (
    DEFAULT_ROAD_FEATURE_LIMIT,
    MAX_ROAD_FEATURE_LIMIT,
    list_gh_roads_with_survey_data,
    list_roads_with_survey_data,
)


class TestTheLoopStaysFree:
    def test_neither_road_handler_is_a_coroutine_function(self):
        """A blocking query inside `async def` runs on the event loop.

        FastAPI runs a plain `def` handler in the threadpool, so a slow query
        costs one worker instead of the whole server. This is the whole fix for
        the outage, and it is one word wide, so it is worth a test.
        """
        for handler in (list_roads_with_survey_data, list_gh_roads_with_survey_data):
            assert not inspect.iscoroutinefunction(handler), (
                f"{handler.__name__} blocks the event loop while it queries"
            )


class TestTheWorkIsBounded:
    def test_the_default_page_is_far_below_the_road_count(self):
        """The table holds hundreds of thousands of roads.

        Unbounded, this answered with all of them as GeoJSON. The default has to
        be a page, and the cap has to be low enough that a page is a page a
        browser can hold.
        """
        assert DEFAULT_ROAD_FEATURE_LIMIT <= 20_000
        assert MAX_ROAD_FEATURE_LIMIT <= 50_000
        assert DEFAULT_ROAD_FEATURE_LIMIT < MAX_ROAD_FEATURE_LIMIT

    def test_a_caller_cannot_ask_for_more_than_the_cap(self):
        assert MAX_ROAD_FEATURE_LIMIT < 1_000_000


class TestBboxParsing:
    def test_no_bbox_means_no_filter(self):
        assert analysis._parse_bbox(None) is None
        assert analysis._parse_bbox("") is None

    def test_a_well_formed_bbox_is_read_in_order(self):
        assert analysis._parse_bbox("-0.30,5.50,0.10,6.10") == (-0.30, 5.50, 0.10, 6.10)

    def test_surrounding_whitespace_is_tolerated(self):
        assert analysis._parse_bbox(" -0.30 , 5.50 , 0.10 , 6.10 ") == (
            -0.30,
            5.50,
            0.10,
            6.10,
        )

    @pytest.mark.parametrize(
        "value",
        [
            "1,2,3",  # too few
            "1,2,3,4,5",  # too many
            "a,b,c,d",  # not numbers
            "1,2,x,4",  # one bad value among good ones
        ],
    )
    def test_a_malformed_bbox_is_refused_rather_than_repaired(self, value):
        """Silently dropping a bad corner returns the wrong half of the country.

        That reads as an empty map rather than as a bad request, so it is
        refused instead.
        """
        with pytest.raises(HTTPException) as caught:
            analysis._parse_bbox(value)
        assert caught.value.status_code == 422

    def test_swapped_corners_are_refused(self):
        """`min,max` reversed means the other side of the country, not an error."""
        with pytest.raises(HTTPException) as caught:
            analysis._parse_bbox("0.10,6.10,-0.30,5.50")
        assert caught.value.status_code == 422

    def test_an_absent_bbox_leaves_the_predicate_open(self):
        assert analysis._road_bbox_predicate(None) == "TRUE"

    def test_a_bbox_predicate_is_indexed_rather_than_a_python_filter(self):
        """Filtering after the fetch is what made this query country-sized.

        The index-friendly operator and `ST_Intersects` are both required:
        without them the database reads every road and discards most of them.
        """
        predicate = analysis._road_bbox_predicate("-0.30,5.50,0.10,6.10")
        assert "&&" in predicate
        assert "ST_Intersects" in predicate


class TestAuthenticationIsRequired:
    @pytest.mark.parametrize("path", ["/api/analysis/roads", "/api/analysis/roads/gh"])
    def test_the_route_declares_the_authenticated_user_dependency(self, path):
        """The outage was reachable without a token.

        The refusal has to be declared on the route rather than assumed of it:
        `get_principal` answers an anonymous caller happily outside production,
        so a route that depends on it alone is open in development, which is
        exactly where this was found. `require_authenticated_user` is the
        dependency that refuses an anonymous principal in either environment.
        """
        route = next(r for r in analysis.router.routes if r.path == path)
        declared = {getattr(d.call, "__name__", None) for d in route.dependant.dependencies}

        assert "require_authenticated_user" in declared, (
            f"{path} accepts anonymous callers: {sorted(n for n in declared if n)}"
        )

    def test_an_anonymous_principal_is_refused_by_that_dependency(self):
        """Why the dependency matters: it 401s outside production too."""
        from app.core.security import require_authenticated_user

        with pytest.raises(HTTPException) as caught:
            require_authenticated_user({"user_id": None, "org_id": None, "kind": "anonymous"})

        assert caught.value.status_code == 401