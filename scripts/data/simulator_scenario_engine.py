"""Scenario-driven, simulator-grounded SFT trajectory generator for ZAC 2026 LifeMate.

Separates three core responsibilities:
  A. Scenario specification (ScenarioSpec):
     - Scenario ID, template family, split assignment ('train' or 'val')
     - Target apps and distractor apps
     - Initial synthetic application state conforming to real zaic_tools app schemas
     - Realistic Vietnamese user request, current_time, city, and user profile
     - Expected high-level objective & post-condition verification function
     - Dynamic execution plan function (plan_fn)
  B. Simulator execution (execute_scenario):
     - Runs on a fresh official `zaic_tools.executor.Session`
     - Validates every tool call via `session.validate()` before execution
     - Executes via `session.call()` and captures the actual returned tool result
     - Extracts entity IDs and quantitative fields dynamically from live tool results
     - Never fabricates response dicts or final states
  C. Trajectory serialization (serialize_trajectory):
     - Reuses `build_initial_messages()` with `SemanticToolRouter` and `catalog_apps_map`
     - Formats concise, grounded `<think>` + `<tool_call>` turns and `<tool_response>` observations
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ScenarioSpec:
    scenario_id: str
    template_id: str
    split: str  # "train" or "val"
    scenario_type: str  # e.g. "read_before_write", "multi_result_selection", "multi_step_chain", "arithmetic_condition", "negative_condition", "missing_info", "cancellation"
    primary_apps: list[str]
    installed_apps: list[str]
    user_request: str
    current_time: str
    city: str
    user_profile: dict[str, Any]
    location: dict[str, float]
    initial_app_states: dict[str, dict[str, Any]]
    objective: str
    expect_mutation: bool
    plan_fn: Callable[[Any], None]
    verify_fn: Callable[..., tuple[bool, str]] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class SimulatorRunner:
    """Wraps an official `zaic_tools.executor.Session` to enforce validate-before-call
    and record exact step-by-step tool calls and actual simulator results.
    """

    def __init__(self, session: Any):
        self.session = session
        self.executed_calls: list[dict[str, Any]] = []
        self.executed_results: list[dict[str, Any]] = []

    def step(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = copy.deepcopy(arguments or {})
        val = self.session.validate(tool_name, args)
        if not val.get("ok"):
            raise ValueError(f"Schema validation failed for {tool_name}({args}): {val.get('error')}")
        res = self.session.call(tool_name, args)
        self.executed_calls.append({"name": tool_name, "arguments": copy.deepcopy(args)})
        self.executed_results.append(copy.deepcopy(res))
        return res


def build_session_entry_and_sample(
    spec: ScenarioSpec,
    catalog_tools_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct the official `Session` entry dict and the agent `sample` dict for a ScenarioSpec."""
    app_tools: dict[str, list[str]] = {}
    for t_name in catalog_tools_map:
        app_id = t_name.split(".", 1)[0]
        if app_id != "sys":
            app_tools.setdefault(app_id, []).append(t_name)

    tools_list: list[str] = []
    installed_apps_dict: dict[str, dict[str, Any]] = {}

    for app_id in spec.installed_apps:
        t_list = app_tools.get(app_id, [])
        tools_list.extend(t_list)
        app_state = copy.deepcopy(spec.initial_app_states.get(app_id, {}))
        installed_apps_dict[app_id] = {
            "tools": list(t_list),
            "state": app_state,
        }

    entry: dict[str, Any] = {
        "_meta": {
            "tools": tools_list,
            "current_time": spec.current_time,
            "city": spec.city,
        }
    }
    for app_id in spec.installed_apps:
        entry[app_id] = copy.deepcopy(spec.initial_app_states.get(app_id, {}))

    sample: dict[str, Any] = {
        "id": spec.scenario_id,
        "audio_text": spec.user_request,
        "context": {
            "current_time": spec.current_time,
            "timezone": "Asia/Ho_Chi_Minh",
            "location": copy.deepcopy(spec.location),
            "user_profile": copy.deepcopy(spec.user_profile),
            "installed_apps": installed_apps_dict,
        },
        "history": copy.deepcopy(spec.history),
    }
    return entry, sample


def execute_scenario_on_simulator(
    spec: ScenarioSpec,
    catalog: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    session_cls: Any,
) -> dict[str, Any]:
    """Execute a ScenarioSpec on a fresh official `Session` and capture the full trace."""
    entry, sample = build_session_entry_and_sample(spec, catalog_tools_map)
    session = session_cls(sample_id=spec.scenario_id, entry=copy.deepcopy(entry), catalog=catalog)
    initial_state = session.get_state()

    runner = SimulatorRunner(session)
    execution_exception: str | None = None
    try:
        spec.plan_fn(runner)
    except Exception as exc:
        execution_exception = f"{type(exc).__name__}: {exc}"

    final_state = session.get_state()
    final_hash = session.get_state_hash()

    sample["label"] = {
        "tool_calls": copy.deepcopy(runner.executed_calls),
        "tool_results": copy.deepcopy(runner.executed_results),
        "final_state": {
            app: st for app, st in final_state.items() if st != initial_state.get(app)
        },
    }

    return {
        "scenario_id": spec.scenario_id,
        "template_id": spec.template_id,
        "split": spec.split,
        "scenario_type": spec.scenario_type,
        "primary_apps": spec.primary_apps,
        "installed_apps": spec.installed_apps,
        "objective": spec.objective,
        "expect_mutation": spec.expect_mutation,
        "entry": entry,
        "sample": sample,
        "initial_state": initial_state,
        "final_state": final_state,
        "changed_state": sample["label"]["final_state"],
        "state_hash": final_hash,
        "tool_calls": runner.executed_calls,
        "tool_results": runner.executed_results,
        "execution_exception": execution_exception,
    }
