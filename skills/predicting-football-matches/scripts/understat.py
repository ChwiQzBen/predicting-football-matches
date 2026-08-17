#!/usr/bin/env python3

"""
Understat data collector using AJAX API.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://understat.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://understat.com/",
}


class UnderstatClient:
    def __init__(
        self,
        cache_dir: str | Path = "data/raw/understat",
        request_delay: float = 1.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay

        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _request_ajax(self, endpoint: str) -> Any:
        url = f"{BASE_URL}/{endpoint}"

        response = self.session.get(
            url,
            timeout=30,
        )

        response.raise_for_status()

        time.sleep(self.request_delay)

        return response.json()

    def league_matches(
        self,
        league: str,
        season: int,
    ) -> list[dict[str, Any]]:
        cache_file = (
            self.cache_dir
            / f"{league}_{season}_matches.json"
        )

        if cache_file.exists():
            return json.loads(
                cache_file.read_text(encoding="utf-8")
            )

        data = self._request_ajax(
            f"getLeagueData/{league}/{season}"
        )

        matches = data.get("dates", [])

        if not isinstance(matches, list):
            raise ValueError(
                "Unexpected Understat league response."
            )

        cache_file.write_text(
            json.dumps(
                matches,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return matches

    def team_matches(
        self,
        team: str,
        season: int,
    ) -> list[dict[str, Any]]:
        cache_file = (
            self.cache_dir
            / f"team_{team}_{season}.json"
        )

        if cache_file.exists():
            return json.loads(
                cache_file.read_text(encoding="utf-8")
            )

        data = self._request_ajax(
            f"getTeamData/{team}/{season}"
        )

        matches = data.get("dates", [])

        if not isinstance(matches, list):
            raise ValueError(
                "Unexpected Understat team response."
            )

        cache_file.write_text(
            json.dumps(
                matches,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return matches