"""Deterministic Entity Normalizer & Database-Aware Fuzzy Fallback for ZAC 2026 LifeMate.
Solution 2: Bridges phonetic/colloquial speech to concrete database IDs.
Handles empty read recovery, relationship aliases, and query broadening.
"""
from __future__ import annotations

import copy
import re
from typing import Any

# Vietnamese kinship / nickname synonyms
RELATIONSHIP_ALIASES: dict[str, list[str]] = {
    "bố": ["bố", "ba", "cha", "bố yêu", "ba yêu", "bố đt", "ba đt"],
    "ba": ["ba", "bố", "cha", "ba yêu", "bố yêu"],
    "mẹ": ["mẹ", "má", "u", "mẹ yêu", "má yêu", "mẹ đt"],
    "má": ["má", "mẹ", "má yêu", "mẹ yêu"],
    "vợ": ["vợ", "bà xã", "vợ yêu", "bx", "nhà"],
    "chồng": ["chồng", "ông xã", "chồng yêu", "ox"],
    "anh": ["anh", "anh yêu", "anh trai"],
    "chị": ["chị", "chị gái", "chị yêu"],
    "em": ["em", "em gái", "em trai", "em yêu"],
    "con": ["con", "con gái", "con trai"],
    "sếp": ["sếp", "thủ trưởng", "lead"],
}


class EntityNormalizer:
    """Deterministic normalizer and recovery engine for entities in LifeMate."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def expand_query_variants(self, query: str) -> list[str]:
        """Generate search query variants for colloquial terms or compound names."""
        if not self.enabled or not query:
            return [query] if query else []

        q_clean = query.strip()
        variants = [q_clean]
        q_lower = q_clean.lower()

        # Check kinship dictionary
        for canonical, aliases in RELATIONSHIP_ALIASES.items():
            if q_lower == canonical or any(a == q_lower for a in aliases):
                for a in aliases:
                    if a not in variants:
                        variants.append(a)
                break

        # Check multi-word Vietnamese names: e.g. "Lê Minh" -> also try "Minh"
        parts = q_clean.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            if last_name not in variants and len(last_name) >= 2:
                variants.append(last_name)

        return variants

    def resolve_fuzzy_arguments(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        observed_entities: dict[str, Any],
        sample: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve colloquial or fuzzy entity references (e.g. contact names, relationship aliases,
        or device names) to concrete database IDs observed in the session.
        """
        if not self.enabled or not isinstance(tool_args, dict) or not observed_entities:
            return tool_args

        resolved = copy.deepcopy(tool_args)

        # 1. Resolve contact_id if passed as a relationship alias or name instead of ID
        if "contact_id" in resolved and isinstance(resolved["contact_id"], str):
            cid = resolved["contact_id"].strip()
            if cid and cid not in observed_entities:
                cid_lower = cid.lower()
                cid_variants = self.expand_query_variants(cid)
                for ent_id, ent in observed_entities.items():
                    if isinstance(ent, dict):
                        e_name = str(ent.get("name", "")).strip().lower()
                        e_rel = str(ent.get("relation", "")).strip().lower()
                        if (
                            e_name == cid_lower
                            or e_rel == cid_lower
                            or any(v.lower() == e_name or v.lower() == e_rel for v in cid_variants)
                            or (e_name and e_name in cid_lower)
                            or (cid_lower and cid_lower in e_name)
                        ):
                            resolved["contact_id"] = str(ent.get("id") or ent_id)
                            break

        # 2. Resolve device_id if passed as a device name/room instead of ID
        if "device_id" in resolved and isinstance(resolved["device_id"], str):
            did = resolved["device_id"].strip()
            if did and did not in observed_entities:
                did_lower = did.lower()
                for ent_id, ent in observed_entities.items():
                    if isinstance(ent, dict) and "type" in ent:
                        e_name = str(ent.get("name", "")).strip().lower()
                        if (
                            e_name == did_lower
                            or (e_name and e_name in did_lower)
                            or (did_lower and did_lower in e_name)
                        ):
                            resolved["device_id"] = str(ent.get("id") or ent_id)
                            break

        return resolved

    def recover_empty_read(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: dict[str, Any],
        sample: dict[str, Any],
        observed_entities: dict[str, Any],
    ) -> dict[str, Any]:
        """If a search/read tool returned an empty collection, attach grounded recovery hints."""
        if not self.enabled or not isinstance(tool_result, dict):
            return tool_result

        # Check if any list in result is empty
        is_empty = False
        for k, v in tool_result.items():
            if isinstance(v, list) and len(v) == 0:
                is_empty = True
                break

        if not is_empty:
            return tool_result

        result_copy = copy.deepcopy(tool_result)

        # 1. contacts.search returned []
        if tool_name == "contacts.search":
            orig_q = str(tool_args.get("query") or "")
            q_variants = self.expand_query_variants(orig_q)
            found_candidate = None
            for ent_id, ent in observed_entities.items():
                if isinstance(ent, dict) and "name" in ent:
                    c_name = str(ent.get("name", "")).lower()
                    if any(v.lower() in c_name for v in q_variants):
                        found_candidate = ent
                        break

            if found_candidate and isinstance(result_copy.get("contacts"), list):
                # Return synthetic match compliant with real contacts schema
                result_copy["contacts"] = [found_candidate]
            return result_copy

        return tool_result
