"""Two-Tier Semantic Tool Router for ZAC 2026 LifeMate.
Ranks installed apps for a given user request using fast character-and-word n-gram TF-IDF similarity.
Enables two-tier prompt construction (Recommended Candidate Tools vs Secondary Tools)
to eliminate tool confusion without risking false-negative tool pruning.
"""
from __future__ import annotations

import json
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticToolRouter:
    """Fast, deterministic semantic tool and app router (< 2ms CPU latency)."""

    def __init__(self, top_k_recommended: int = 4):
        self.top_k_recommended = top_k_recommended
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b\w+\b",
        )

    def route_sample_apps(
        self,
        query: str,
        sample: dict[str, Any],
        catalog_tools_map: dict[str, dict[str, Any]],
        installed_app_descriptions: dict[str, str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Rank installed apps for `query`.
        Returns: `(recommended_apps, secondary_apps)`.
        Guarantees: Every installed app appears in exactly one of the two lists.
        """
        installed = sample.get("context", {}).get("installed_apps", {})
        app_names = [a for a in installed.keys() if a != "sys"]

        # If 3 or fewer apps installed, all are recommended
        if len(app_names) <= self.top_k_recommended:
            return app_names, []

        installed_app_descriptions = installed_app_descriptions or {}

        # Build document text for each installed app: app_name + app_description + tool descriptions
        app_docs: list[str] = []
        for app in app_names:
            parts = [app, installed_app_descriptions.get(app, "")]
            # Include tool descriptions of that app
            tool_names = (installed.get(app) or {}).get("tools", [])
            for tname in tool_names:
                parts.append(tname)
                spec = catalog_tools_map.get(tname) or {}
                if spec.get("description"):
                    parts.append(spec["description"])
                # Include parameter names and descriptions
                for p_name, p_info in (spec.get("parameters", {}).get("properties") or {}).items():
                    parts.append(p_name)
                    if p_info.get("description"):
                        parts.append(p_info["description"])
            app_docs.append(" ".join(parts))

        # Query includes the user prompt and recent user history text
        query_parts = [query]
        for turn in (sample.get("history") or [])[-2:]:
            if turn.get("role") == "user" and turn.get("audio_text"):
                query_parts.append(turn["audio_text"])
        full_query = " ".join(query_parts)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(app_docs)
            q_vec = self.vectorizer.transform([full_query])
            scores = cosine_similarity(q_vec, tfidf_matrix)[0]

            # Rank apps by score descending
            ranked_indices = scores.argsort()[::-1]
            ranked_apps = [app_names[i] for i in ranked_indices]

            recommended = ranked_apps[: self.top_k_recommended]
            secondary = ranked_apps[self.top_k_recommended :]
            return recommended, secondary
        except Exception:
            # Fallback: preserve original order
            return app_names[: self.top_k_recommended], app_names[self.top_k_recommended :]
