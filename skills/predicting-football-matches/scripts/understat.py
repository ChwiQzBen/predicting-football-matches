#!/usr/bin/env python3

"""
Understat data collector.

Collects historical match-level xG data and stores it locally.

No bookmaker odds are used anywhere in this module.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://understat.com"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/140.0 Safari/537.36"
    )
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

    def _get(self, url: str) -> str:
        response = requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=30,
        )
        response.raise_for_status()

        time.sleep(self.request_delay)

        return response.text

    @staticmethod
    def _extract_json(html: str, variable: str) -> Any:
        """
        Extract Understat's JSON blob from a script variable.
        """

        pattern = rf"datesData\s*=\s*JSON\.parse\('(.*?)'\)"

        if variable != "datesData":
            pattern = (
                rf"{re.escape(variable)}\s*=\s*JSON\.parse\('(.*?)'\)"
            )

        match = re.search(pattern, html)

        if not match:
            raise ValueError(
                f"Could not find Understat variable: {variable}"
            )

        encoded = match.group(1)

        decoded = bytes(encoded, "utf-8").decode("unicode_escape")

        return json.loads(decoded)

    def league_matches(
        self,
        league: str,
        season: int,
    ) -> list[dict[str, Any]]:
        """
        Retrieve match-level data for an Understat league season.

        Examples:
            EPL
            La_liga
            Bundesliga
            Serie_A
            Ligue_1
        """

        cache_file = (
            self.cache_dir
            / f"{league}_{season}_matches.json"
        )

        if cache_file.exists():
            return json.loads(
                cache_file.read_text(encoding="utf-8")
            )

        url = (
            f"{BASE_URL}/league/{league}/{season}"
        )

        html = self._get(url)

        matches = self._extract_json(
            html,
            "datesData",
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
        """
        Retrieve a team's historical matches.
        """

        cache_file = (
            self.cache_dir
            / f"team_{team}_{season}.json"
        )

        if cache_file.exists():
            return json.loads(
                cache_file.read_text(encoding="utf-8")
            )

        url = (
            f"{BASE_URL}/team/{team}/{season}"
        )

        html = self._get(url)

        matches = self._extract_json(
            html,
            "datesData",
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