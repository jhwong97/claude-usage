"""Tests for dashboard.py - API endpoint and data retrieval."""

import json
import os
import sqlite3
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from scanner import get_db, init_db, upsert_sessions, insert_turns, insert_prompts
from dashboard import (
    get_dashboard_data, get_session_detail, DashboardHandler, HTML_TEMPLATE,
    _group_prompts, _score_session,
)

try:
    from http.server import HTTPServer
except ImportError:
    HTTPServer = None


class TestGetDashboardData(unittest.TestCase):
    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmpfile.close()
        self.db_path = Path(self.tmpfile.name)
        conn = get_db(self.db_path)
        init_db(conn)
        # Insert sample data
        sessions = [{
            "session_id": "sess-abc123", "project_name": "user/myproject",
            "first_timestamp": "2026-04-08T09:00:00Z",
            "last_timestamp": "2026-04-08T10:00:00Z",
            "git_branch": "main", "model": "claude-sonnet-4-6",
            "total_input_tokens": 5000, "total_output_tokens": 2000,
            "total_cache_read": 500, "total_cache_creation": 200,
            "turn_count": 10,
        }]
        upsert_sessions(conn, sessions)
        turns = [
            {
                "session_id": "sess-abc123", "timestamp": "2026-04-08T09:30:00Z",
                "model": "claude-sonnet-4-6", "input_tokens": 500,
                "output_tokens": 200, "cache_read_tokens": 50,
                "cache_creation_tokens": 20, "tool_name": None, "cwd": "/tmp",
            },
            {
                "session_id": "sess-abc123", "timestamp": "2026-04-08T14:15:00Z",
                "model": "claude-sonnet-4-6", "input_tokens": 300,
                "output_tokens": 150, "cache_read_tokens": 0,
                "cache_creation_tokens": 0, "tool_name": None, "cwd": "/tmp",
            },
        ]
        insert_turns(conn, turns)
        conn.commit()
        conn.close()

    def tearDown(self):
        os.unlink(self.db_path)

    def test_returns_valid_structure(self):
        data = get_dashboard_data(db_path=self.db_path)
        self.assertIn("all_models", data)
        self.assertIn("daily_by_model", data)
        self.assertIn("sessions_all", data)
        self.assertIn("generated_at", data)

    def test_models_populated(self):
        data = get_dashboard_data(db_path=self.db_path)
        self.assertIn("claude-sonnet-4-6", data["all_models"])

    def test_sessions_populated(self):
        data = get_dashboard_data(db_path=self.db_path)
        self.assertEqual(len(data["sessions_all"]), 1)
        session = data["sessions_all"][0]
        self.assertEqual(session["project"], "user/myproject")
        self.assertEqual(session["model"], "claude-sonnet-4-6")
        self.assertEqual(session["input"], 5000)

    def test_session_by_model_buckets_partition_turn_tokens(self):
        data = get_dashboard_data(db_path=self.db_path)
        session = data["sessions_all"][0]
        self.assertIn("by_model", session)
        buckets = session["by_model"]
        # Both turns are sonnet → one bucket summing the turn tokens.
        self.assertEqual(len(buckets), 1)
        b = buckets[0]
        self.assertEqual(b["model"], "claude-sonnet-4-6")
        self.assertEqual(b["input"], 800)   # 500 + 300
        self.assertEqual(b["output"], 350)  # 200 + 150
        self.assertEqual(b["cache_read"], 50)
        self.assertEqual(b["cache_creation"], 20)

    def test_daily_by_model_populated(self):
        data = get_dashboard_data(db_path=self.db_path)
        self.assertGreater(len(data["daily_by_model"]), 0)
        day = data["daily_by_model"][0]
        self.assertIn("day", day)
        self.assertIn("model", day)
        self.assertIn("input", day)

    def test_missing_db_returns_error(self):
        data = get_dashboard_data(db_path=Path("/nonexistent/path/usage.db"))
        self.assertIn("error", data)

    def test_session_id_truncated(self):
        data = get_dashboard_data(db_path=self.db_path)
        session = data["sessions_all"][0]
        self.assertEqual(len(session["session_id"]), 8)

    def test_session_has_efficiency_and_full_id(self):
        data = get_dashboard_data(db_path=self.db_path)
        session = data["sessions_all"][0]
        self.assertIn(session["efficiency"], ("green", "yellow", "red"))
        self.assertEqual(session["session_id_full"], "sess-abc123")

    def test_session_duration_calculated(self):
        data = get_dashboard_data(db_path=self.db_path)
        session = data["sessions_all"][0]
        # 1 hour = 60 minutes
        self.assertEqual(session["duration_min"], 60.0)

    def test_hourly_by_model_present(self):
        data = get_dashboard_data(db_path=self.db_path)
        self.assertIn("hourly_by_model", data)
        self.assertIsInstance(data["hourly_by_model"], list)

    def test_hourly_by_model_buckets_by_utc_hour(self):
        data = get_dashboard_data(db_path=self.db_path)
        rows = data["hourly_by_model"]
        # Two turns at UTC 09:30 and 14:15 → two hour buckets
        by_hour = {r["hour"]: r for r in rows}
        self.assertIn(9, by_hour)
        self.assertIn(14, by_hour)
        self.assertEqual(by_hour[9]["turns"], 1)
        self.assertEqual(by_hour[9]["output"], 200)
        self.assertEqual(by_hour[14]["turns"], 1)
        self.assertEqual(by_hour[14]["output"], 150)

    def test_hourly_by_model_carries_day_and_model(self):
        data = get_dashboard_data(db_path=self.db_path)
        rows = data["hourly_by_model"]
        self.assertTrue(all("day" in r and "model" in r for r in rows))
        self.assertTrue(all(r["model"] == "claude-sonnet-4-6" for r in rows))
        self.assertTrue(all(r["day"] == "2026-04-08" for r in rows))


class TestEmptyStringModelNormalization(unittest.TestCase):
    """Regression: turns with model='' (empty string) must group as 'unknown'.
    COALESCE(model, 'unknown') alone returns '' because empty string isn't NULL;
    NULLIF(model, '') is needed first."""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmpfile.close()
        self.db_path = Path(self.tmpfile.name)
        conn = get_db(self.db_path)
        init_db(conn)
        upsert_sessions(conn, [{
            "session_id": "sess-empty", "project_name": "u/p",
            "first_timestamp": "2026-04-08T09:00:00Z",
            "last_timestamp": "2026-04-08T09:05:00Z",
            "git_branch": "", "model": "",
            "total_input_tokens": 100, "total_output_tokens": 50,
            "total_cache_read": 0, "total_cache_creation": 0,
            "turn_count": 1,
        }])
        insert_turns(conn, [{
            "session_id": "sess-empty", "timestamp": "2026-04-08T09:05:00Z",
            "model": "", "input_tokens": 100, "output_tokens": 50,
            "cache_read_tokens": 0, "cache_creation_tokens": 0,
            "tool_name": None, "cwd": "/tmp",
        }])
        conn.commit()
        conn.close()

    def tearDown(self):
        os.unlink(self.db_path)

    def test_all_models_contains_unknown_not_empty(self):
        data = get_dashboard_data(db_path=self.db_path)
        self.assertIn("unknown", data["all_models"])
        self.assertNotIn("", data["all_models"])

    def test_daily_by_model_contains_unknown_not_empty(self):
        data = get_dashboard_data(db_path=self.db_path)
        models = {r["model"] for r in data["daily_by_model"]}
        self.assertIn("unknown", models)
        self.assertNotIn("", models)

    def test_hourly_by_model_contains_unknown_not_empty(self):
        data = get_dashboard_data(db_path=self.db_path)
        models = {r["model"] for r in data["hourly_by_model"]}
        self.assertIn("unknown", models)
        self.assertNotIn("", models)


class TestMixedNullAndEmptyModel(unittest.TestCase):
    """Regression: a mix of model=NULL and model='' rows must collapse into a
    SINGLE 'unknown' group across all aggregations. Without `GROUP BY
    COALESCE(NULLIF(model, ''), 'unknown')` (matching the SELECT expression),
    SQLite groups by raw value and emits two distinct 'unknown' rows."""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmpfile.close()
        self.db_path = Path(self.tmpfile.name)
        conn = get_db(self.db_path)
        init_db(conn)
        upsert_sessions(conn, [{
            "session_id": "sess-mix", "project_name": "u/p",
            "first_timestamp": "2026-04-08T09:00:00Z",
            "last_timestamp": "2026-04-08T10:00:00Z",
            "git_branch": "", "model": "",
            "total_input_tokens": 200, "total_output_tokens": 100,
            "total_cache_read": 0, "total_cache_creation": 0,
            "turn_count": 2,
        }])
        # Insert one turn with model='' and one with model=NULL on the same day.
        # Use raw INSERT for the NULL row because insert_turns() requires the
        # model key to exist (would error on missing key, not on None).
        insert_turns(conn, [{
            "session_id": "sess-mix", "timestamp": "2026-04-08T09:00:00Z",
            "model": "", "input_tokens": 100, "output_tokens": 50,
            "cache_read_tokens": 0, "cache_creation_tokens": 0,
            "tool_name": None, "cwd": "/tmp",
        }])
        conn.execute("""
            INSERT INTO turns (session_id, timestamp, model, input_tokens,
                output_tokens, cache_read_tokens, cache_creation_tokens,
                tool_name, cwd)
            VALUES ('sess-mix', '2026-04-08T09:30:00Z', NULL, 100, 50, 0, 0, NULL, '/tmp')
        """)
        conn.commit()
        conn.close()

    def tearDown(self):
        os.unlink(self.db_path)

    def test_all_models_collapses_to_single_unknown(self):
        data = get_dashboard_data(db_path=self.db_path)
        unknowns = [m for m in data["all_models"] if m == "unknown"]
        self.assertEqual(len(unknowns), 1, f"got duplicate 'unknown' rows: {data['all_models']}")

    def test_daily_collapses_to_single_unknown(self):
        data = get_dashboard_data(db_path=self.db_path)
        unknown_rows = [r for r in data["daily_by_model"] if r["model"] == "unknown"]
        # One day, one model bucket
        self.assertEqual(len(unknown_rows), 1, f"got {unknown_rows}")
        self.assertEqual(unknown_rows[0]["turns"], 2)
        self.assertEqual(unknown_rows[0]["input"], 200)

    def test_hourly_collapses_to_single_unknown(self):
        data = get_dashboard_data(db_path=self.db_path)
        # Both turns are in UTC hour 9 — must be one row, not two
        hour9 = [r for r in data["hourly_by_model"]
                 if r["hour"] == 9 and r["model"] == "unknown"]
        self.assertEqual(len(hour9), 1, f"got {hour9}")
        self.assertEqual(hour9[0]["turns"], 2)


class TestNonBillableModelFallback(unittest.TestCase):
    """Regression: when the user has only non-billable models (e.g. gemma, glm,
    local LLMs) — or all turns lack a model field — the default model selection
    must fall back to ALL models so the dashboard isn't blank."""

    def test_readurlmodels_fallback_in_html_template(self):
        # The fallback logic is JS; we assert the source contains the guard so
        # a future refactor doesn't silently remove it.
        self.assertIn("billable.length ? billable : allModels", HTML_TEMPLATE)


class TestGroupPrompts(unittest.TestCase):
    """Read-time attribution of turns to the prompt that triggered them."""

    def _turn(self, ts, inp=100, out=50, tool=None, model="claude-sonnet-4-6"):
        return {
            "timestamp": ts, "model": model, "input_tokens": inp,
            "output_tokens": out, "cache_read_tokens": 0,
            "cache_creation_tokens": 0, "tool_name": tool,
        }

    def test_turns_attributed_to_preceding_prompt(self):
        prompts = [
            {"timestamp": "2026-04-08T09:00:00Z", "text": "First"},
            {"timestamp": "2026-04-08T10:00:00Z", "text": "Second"},
        ]
        turns = [
            self._turn("2026-04-08T09:01:00Z", inp=100),
            self._turn("2026-04-08T09:02:00Z", inp=200),
            self._turn("2026-04-08T10:05:00Z", inp=300),
        ]
        groups = _group_prompts(prompts, turns)
        by_text = {g["text"]: g for g in groups}
        self.assertEqual(by_text["First"]["turn_count"], 2)
        self.assertEqual(by_text["First"]["input"], 300)
        self.assertEqual(by_text["Second"]["turn_count"], 1)
        self.assertEqual(by_text["Second"]["input"], 300)

    def test_turns_before_first_prompt_go_to_initial_bucket(self):
        prompts = [{"timestamp": "2026-04-08T10:00:00Z", "text": "Later"}]
        turns = [
            self._turn("2026-04-08T09:00:00Z", inp=500),  # before any prompt
            self._turn("2026-04-08T10:01:00Z", inp=100),
        ]
        groups = _group_prompts(prompts, turns)
        initial = [g for g in groups if g["is_initial"]]
        self.assertEqual(len(initial), 1)
        self.assertEqual(initial[0]["input"], 500)

    def test_tools_and_model_aggregated(self):
        prompts = [{"timestamp": "2026-04-08T09:00:00Z", "text": "Do"}]
        turns = [
            self._turn("2026-04-08T09:01:00Z", tool="Read"),
            self._turn("2026-04-08T09:02:00Z", tool="Read"),
            self._turn("2026-04-08T09:03:00Z", tool="Edit"),
        ]
        g = _group_prompts(prompts, turns)[0]
        tool_map = {t["name"]: t["count"] for t in g["tools"]}
        self.assertEqual(tool_map, {"Read": 2, "Edit": 1})
        self.assertEqual(g["model"], "claude-sonnet-4-6")

    def test_zero_turn_prompts_dropped(self):
        prompts = [{"timestamp": "2026-04-08T09:00:00Z", "text": "No reply yet"}]
        groups = _group_prompts(prompts, [])
        self.assertEqual(groups, [])

    def test_by_model_buckets_partition_group_tokens(self):
        prompts = [{"timestamp": "2026-04-08T09:00:00Z", "text": "Do"}]
        turns = [
            self._turn("2026-04-08T09:01:00Z", inp=100, out=10, model="claude-opus-4-7"),
            self._turn("2026-04-08T09:02:00Z", inp=200, out=20, model="claude-opus-4-7"),
            self._turn("2026-04-08T09:03:00Z", inp=300, out=30, model="claude-haiku-4-5"),
        ]
        g = _group_prompts(prompts, turns)[0]
        # Dominant model is still the single "model" tag (opus, 2 of 3 turns).
        self.assertEqual(g["model"], "claude-opus-4-7")
        buckets = {b["model"]: b for b in g["by_model"]}
        self.assertEqual(set(buckets), {"claude-opus-4-7", "claude-haiku-4-5"})
        self.assertEqual(buckets["claude-opus-4-7"]["input"], 300)   # 100 + 200
        self.assertEqual(buckets["claude-opus-4-7"]["output"], 30)   # 10 + 20
        self.assertEqual(buckets["claude-haiku-4-5"]["input"], 300)
        self.assertEqual(buckets["claude-haiku-4-5"]["output"], 30)
        # Buckets must partition the group's totals exactly.
        self.assertEqual(sum(b["input"] for b in g["by_model"]), g["input"])
        self.assertEqual(sum(b["output"] for b in g["by_model"]), g["output"])

    def test_null_model_turns_bucket_as_unknown(self):
        prompts = [{"timestamp": "2026-04-08T09:00:00Z", "text": "Do"}]
        turns = [self._turn("2026-04-08T09:01:00Z", inp=100, model=None)]
        g = _group_prompts(prompts, turns)[0]
        self.assertEqual([b["model"] for b in g["by_model"]], ["unknown"])
        self.assertEqual(g["by_model"][0]["input"], 100)


class TestScoreSession(unittest.TestCase):
    """Efficiency scoring heuristics."""

    def _turns(self, n, tool=False, model="claude-sonnet-4-6", inp=100):
        return [{
            "input_tokens": inp, "output_tokens": 50, "cache_read_tokens": 0,
            "cache_creation_tokens": 0, "tool_name": "Read" if tool else None,
            "model": model,
        } for _ in range(n)]

    def test_small_session_is_green(self):
        self.assertEqual(_score_session(self._turns(3), 1), "green")

    def test_marathon_is_red(self):
        self.assertEqual(_score_session(self._turns(250), 20), "red")

    def test_opus_on_tiny_session_is_yellow(self):
        turns = self._turns(4, model="claude-opus-4-8")
        self.assertEqual(_score_session(turns, 2), "yellow")

    def test_runaway_growth_is_red(self):
        turns = self._turns(3, inp=100) + self._turns(5, inp=100_000)
        self.assertEqual(_score_session(turns, 2), "red")


class TestSessionDetail(unittest.TestCase):
    """End-to-end get_session_detail against a seeded DB."""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmpfile.close()
        self.db_path = Path(self.tmpfile.name)
        conn = get_db(self.db_path)
        init_db(conn)
        upsert_sessions(conn, [{
            "session_id": "sess-detail", "project_name": "u/p",
            "first_timestamp": "2026-04-08T09:00:00Z",
            "last_timestamp": "2026-04-08T09:10:00Z",
            "git_branch": "main", "model": "claude-sonnet-4-6",
            "total_input_tokens": 300, "total_output_tokens": 150,
            "total_cache_read": 0, "total_cache_creation": 0,
            "turn_count": 3,
        }])
        insert_turns(conn, [
            {"session_id": "sess-detail", "timestamp": "2026-04-08T09:01:00Z",
             "model": "claude-sonnet-4-6", "input_tokens": 100, "output_tokens": 50,
             "cache_read_tokens": 0, "cache_creation_tokens": 0, "tool_name": "Read",
             "cwd": "/tmp", "message_id": "m1"},
            {"session_id": "sess-detail", "timestamp": "2026-04-08T09:06:00Z",
             "model": "claude-sonnet-4-6", "input_tokens": 200, "output_tokens": 100,
             "cache_read_tokens": 0, "cache_creation_tokens": 0, "tool_name": None,
             "cwd": "/tmp", "message_id": "m2"},
        ])
        insert_prompts(conn, [
            {"uuid": "p1", "session_id": "sess-detail",
             "timestamp": "2026-04-08T09:00:30Z", "text": "First task"},
            {"uuid": "p2", "session_id": "sess-detail",
             "timestamp": "2026-04-08T09:05:30Z", "text": "Second task"},
        ])
        conn.commit()
        conn.close()

    def tearDown(self):
        os.unlink(self.db_path)

    def test_returns_prompt_groups(self):
        d = get_session_detail("sess-detail", db_path=self.db_path)
        self.assertNotIn("error", d)
        texts = [g["text"] for g in d["prompts"]]
        self.assertEqual(texts, ["First task", "Second task"])
        self.assertEqual(d["prompts"][0]["input"], 100)
        self.assertEqual(d["prompts"][1]["input"], 200)

    def test_insights_is_a_list(self):
        d = get_session_detail("sess-detail", db_path=self.db_path)
        self.assertIsInstance(d["insights"], list)

    def test_unknown_session_returns_error(self):
        d = get_session_detail("does-not-exist", db_path=self.db_path)
        self.assertIn("error", d)

    def test_command_wrapper_prompts_filtered_at_read(self):
        conn = get_db(self.db_path)
        insert_prompts(conn, [{
            "uuid": "p3", "session_id": "sess-detail",
            "timestamp": "2026-04-08T09:00:40Z",
            "text": "<command-name>/model</command-name>",
        }])
        conn.commit()
        conn.close()
        d = get_session_detail("sess-detail", db_path=self.db_path)
        for g in d["prompts"]:
            self.assertNotIn("command-name", (g["text"] or ""))


class TestSessionEndpointHTTP(unittest.TestCase):
    """/api/session route behavior on the running server."""

    @classmethod
    def setUpClass(cls):
        import dashboard as _d
        cls._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls._tmpfile.close()
        cls._db_path = Path(cls._tmpfile.name)
        conn = get_db(cls._db_path)
        init_db(conn)
        upsert_sessions(conn, [{
            "session_id": "http-sess", "project_name": "u/p",
            "first_timestamp": "2026-04-08T09:00:00Z",
            "last_timestamp": "2026-04-08T09:05:00Z",
            "git_branch": "main", "model": "claude-sonnet-4-6",
            "total_input_tokens": 100, "total_output_tokens": 50,
            "total_cache_read": 0, "total_cache_creation": 0, "turn_count": 1,
        }])
        insert_turns(conn, [{
            "session_id": "http-sess", "timestamp": "2026-04-08T09:01:00Z",
            "model": "claude-sonnet-4-6", "input_tokens": 100, "output_tokens": 50,
            "cache_read_tokens": 0, "cache_creation_tokens": 0, "tool_name": None,
            "cwd": "/tmp", "message_id": "hm1"},
        ])
        insert_prompts(conn, [{
            "uuid": "hp1", "session_id": "http-sess",
            "timestamp": "2026-04-08T09:00:30Z", "text": "HTTP prompt"}])
        conn.commit()
        conn.close()

        cls._orig_db = _d.DB_PATH
        _d.DB_PATH = cls._db_path
        cls.server = HTTPServer(("127.0.0.1", 0), DashboardHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        import dashboard as _d
        cls.server.shutdown()
        _d.DB_PATH = cls._orig_db
        os.unlink(cls._db_path)

    def test_session_endpoint_returns_detail(self):
        url = f"http://127.0.0.1:{self.port}/api/session?id=http-sess"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read())
            self.assertEqual(data["session_id"], "http-sess")
            self.assertEqual(data["prompts"][0]["text"], "HTTP prompt")

    def test_session_endpoint_missing_id(self):
        url = f"http://127.0.0.1:{self.port}/api/session"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read())
            self.assertIn("error", data)


class TestDashboardHTTP(unittest.TestCase):
    """Integration test: start server and make HTTP requests."""

    @classmethod
    def setUpClass(cls):
        # Redirect DB_PATH + projects dirs to a tempdir so /api/rescan
        # doesn't unlink the user's real ~/.claude/usage.db or scan their
        # real transcript directory during tests.
        import dashboard as _d
        import scanner as _s
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmpdir.name)
        tmp_projects = tmp / "projects"
        tmp_projects.mkdir()
        cls._patches = {
            (_d, "DB_PATH"):                (_d.DB_PATH,                tmp / "usage.db"),
            (_s, "DB_PATH"):                (_s.DB_PATH,                tmp / "usage.db"),
            (_s, "PROJECTS_DIR"):           (_s.PROJECTS_DIR,           tmp_projects),
            (_s, "DEFAULT_PROJECTS_DIRS"):  (_s.DEFAULT_PROJECTS_DIRS,  [tmp_projects]),
        }
        for (mod, name), (_orig, new) in cls._patches.items():
            setattr(mod, name, new)

        cls.server = HTTPServer(("127.0.0.1", 0), DashboardHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        for (mod, name), (orig, _new) in cls._patches.items():
            setattr(mod, name, orig)
        cls._tmpdir.cleanup()

    def test_index_returns_html(self):
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers["Content-Type"])

    def test_index_with_query_string_returns_html(self):
        # Regression: ?range=... and ?models=... must not 404. The dashboard
        # itself rewrites the URL with these params via history.replaceState,
        # so anything that reloads or bookmarks the page hits this path.
        for qs in ("?range=all", "?range=30d&models=claude-opus-4-7"):
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/{qs}") as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn(b"Claude Code Usage", resp.read())

    def test_api_data_with_query_string(self):
        # /api/data is fetched without query parameters today, but the route
        # should be tolerant if any are tacked on (e.g. cache-busting).
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/api/data?_=cachebust"
        ) as resp:
            self.assertEqual(resp.status, 200)

    def test_api_data_returns_json(self):
        url = f"http://127.0.0.1:{self.port}/api/data"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/json", resp.headers["Content-Type"])
            data = json.loads(resp.read())
            # Should have expected keys (or error if no DB)
            self.assertTrue("all_models" in data or "error" in data)

    def test_api_rescan_returns_json(self):
        url = f"http://127.0.0.1:{self.port}/api/rescan"
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/json", resp.headers["Content-Type"])
            data = json.loads(resp.read())
            self.assertIn("new", data)
            self.assertIn("updated", data)
            self.assertIn("skipped", data)

    def test_404_for_unknown_path(self):
        url = f"http://127.0.0.1:{self.port}/nonexistent"
        try:
            urllib.request.urlopen(url)
            self.fail("Expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


class TestHTMLTemplate(unittest.TestCase):
    def test_template_is_valid_html(self):
        self.assertIn("<!DOCTYPE html>", HTML_TEMPLATE)
        self.assertIn("</html>", HTML_TEMPLATE)

    def test_template_has_esc_function(self):
        """Verify XSS protection is present (PR #10)."""
        self.assertIn("function esc(", HTML_TEMPLATE)

    def test_template_has_chart_js(self):
        self.assertIn("chart.js", HTML_TEMPLATE.lower())

    def test_template_has_substring_matching(self):
        """Verify getPricing falls back to substring match for unknown models."""
        self.assertIn("m.includes('opus')", HTML_TEMPLATE)
        self.assertIn("m.includes('sonnet')", HTML_TEMPLATE)
        self.assertIn("m.includes('haiku')", HTML_TEMPLATE)

    def test_unknown_models_return_null(self):
        """Verify getPricing returns null for non-Anthropic models."""
        self.assertIn("return null;", HTML_TEMPLATE)

    def test_hourly_chart_canvas_present(self):
        """Hourly distribution chart has a canvas + TZ toggle."""
        self.assertIn('id="chart-hourly"', HTML_TEMPLATE)
        self.assertIn('data-tz="local"', HTML_TEMPLATE)
        self.assertIn('data-tz="utc"', HTML_TEMPLATE)

    def test_hourly_peak_hour_constants(self):
        """Peak-hour set covers UTC 12–17 (Mon–Fri 05:00–11:00 PT)."""
        self.assertIn('PEAK_HOURS_UTC', HTML_TEMPLATE)
        self.assertIn('[12, 13, 14, 15, 16, 17]', HTML_TEMPLATE)

    def test_efficiency_dot_and_expand_present(self):
        """Per-session efficiency dot + expandable prompt breakdown wiring."""
        self.assertIn(".eff-dot", HTML_TEMPLATE)
        self.assertIn("function toggleSession", HTML_TEMPLATE)
        self.assertIn("function renderSessionDetail", HTML_TEMPLATE)
        self.assertIn("/api/session?id=", HTML_TEMPLATE)

    def test_today_range_button_present(self):
        """The 'Today' range button is wired into RANGE_LABELS, RANGE_TICKS,
        getRangeBounds, and the filter-bar HTML."""
        self.assertIn("data-range=\"today\"", HTML_TEMPLATE)
        self.assertIn("'today': 'Today'", HTML_TEMPLATE)
        self.assertIn("'today': 1", HTML_TEMPLATE)
        # Bounds case: today returns start === end === today's ISO date
        self.assertIn("range === 'today'", HTML_TEMPLATE)


class TestPricingParity(unittest.TestCase):
    """Verify CLI and dashboard pricing tables stay in sync."""

    def _extract_js_pricing(self):
        """Extract pricing values from the dashboard JS PRICING object."""
        import re
        prices = {}
        for match in re.finditer(
            r"'(claude-[^']+)':\s*\{\s*input:\s*([\d.]+),\s*output:\s*([\d.]+)",
            HTML_TEMPLATE
        ):
            model, inp, out = match.group(1), float(match.group(2)), float(match.group(3))
            prices[model] = {"input": inp, "output": out}
        return prices

    def test_all_cli_models_in_dashboard(self):
        from cli import PRICING as CLI_PRICING
        js_prices = self._extract_js_pricing()
        for model in CLI_PRICING:
            self.assertIn(model, js_prices, f"{model} missing from dashboard JS")

    def test_prices_match(self):
        from cli import PRICING as CLI_PRICING
        js_prices = self._extract_js_pricing()
        for model in CLI_PRICING:
            self.assertAlmostEqual(
                CLI_PRICING[model]["input"], js_prices[model]["input"],
                msg=f"{model} input price mismatch"
            )
            self.assertAlmostEqual(
                CLI_PRICING[model]["output"], js_prices[model]["output"],
                msg=f"{model} output price mismatch"
            )


if __name__ == "__main__":
    unittest.main()
