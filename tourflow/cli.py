"""业务使用路径的命令行入口。

    python3 -m tourflow.cli validate
    python3 -m tourflow.cli analyze
    python3 -m tourflow.cli forecast
    python3 -m tourflow.cli plan [--commit] [--conservatism 0.7]
    python3 -m tourflow.cli accept --group G5
    python3 -m tourflow.cli fulfill
    python3 -m tourflow.cli audit
    python3 -m tourflow.cli demo        # 依次跑通整条路径
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import __version__
from .audit import RunRegistry, commitment_lineage
from .capacity import build_ledger
from .commit import CommitEngine, replay_commitments, seed_obligations
from .demand import Forecast, backtest, fit_demand, predict
from .did import DidConfig
from .fulfill import apply_events, apply_feedback_closures
from .incrementality import analyze
from .materials import load
from .model import build_structure, parse_date, parse_moment
from .policy import Policy
from .report import (
    render_audit,
    render_forecast,
    render_fulfill,
    render_incrementality,
    render_plan,
)
from .store import StateStore

FORECAST_CACHE = "forecast_cache.json"


def _now(args) -> dt.datetime:
    if getattr(args, "now", None):
        return parse_moment(args.now)
    return dt.datetime.now(dt.timezone.utc)


def _policy(args) -> Policy:
    return Policy(
        conservatism=args.conservatism,
        pilot_fraction=args.pilot_fraction,
        protected_util_cap=args.protected_util_cap,
    )


def _write_outputs(out_dir: Path, name: str, payload: dict, markdown: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    (out_dir / f"{name}.md").write_text(markdown, encoding="utf-8")


def _requests(materials) -> dict[str, dict]:
    return {r["id"]: r for r in materials.table("group_requests")}


def _latest_run(runs: dict, kind: str) -> dict | None:
    for run in reversed(runs["runs"]):
        if run["kind"] == kind:
            return run
    return None


def cmd_validate(args) -> int:
    materials = load(args.root)
    print(json.dumps(materials.counts(), ensure_ascii=False, sort_keys=True))
    return 0


def _run_analyze(args, materials, structure, store) -> tuple[dict, str]:
    config = DidConfig(
        min_controls=args.min_controls,
        bootstrap=args.bootstrap,
        seed=args.seed,
        control_scope=args.control_scope,
    )
    at = _now(args).isoformat()
    report = analyze(materials, structure, config, created_at=at)
    payload = report.to_dict()
    runs = store.load_runs()
    registry = RunRegistry(runs)
    run_id = registry.record(
        "analyze", report.config, {"verdicts": {k: v.verdict for k, v in report.routes.items()}},
        materials.fingerprint, at=at,
    )
    store.save_runs(runs)
    payload["run_id"] = run_id
    _write_outputs(Path(args.output_dir), "incrementality", payload, render_incrementality(report))
    return payload, run_id


def cmd_analyze(args) -> int:
    materials = load(args.root)
    structure = build_structure(materials)
    store = StateStore(args.state_dir)
    payload, run_id = _run_analyze(args, materials, structure, store)
    verdicts = {k: v["verdict"] for k, v in payload["routes"].items()}
    print(json.dumps({"run_id": run_id, "verdicts": verdicts}, ensure_ascii=False))
    return 0


def _run_forecast(args, materials, structure, store) -> tuple[dict, str]:
    at = _now(args).isoformat()
    runs = store.load_runs()
    analyze_run = _latest_run(runs, "analyze")
    if analyze_run is None:
        _, analyze_run_id = _run_analyze(args, materials, structure, store)
        runs = store.load_runs()
        analyze_run = _latest_run(runs, "analyze")
    evidence = analyze_run["summary"]["verdicts"]
    models = fit_demand(materials, structure)
    supply_dates = sorted({parse_date(r["on"]) for r in materials.table("supply")})
    holiday_dates = {
        dt.date.fromisoformat(r["on"]) for r in materials.table("daily_panel") if r["holiday"]
    }
    forecasts = []
    for (origin, route), model in sorted(models.items()):
        forecasts.extend(
            predict(model, supply_dates, holiday_dates=holiday_dates, seed=args.seed)
        )
    bt = backtest(materials, structure, cutoff=args.backtest_cutoff, seed=args.seed)
    registry = RunRegistry(runs)
    summary = {
        "n_models": len(models),
        "backtest": bt,
        "evidence": evidence,
    }
    run_id = registry.record(
        "forecast",
        {"seed": args.seed, "backtest_cutoff": args.backtest_cutoff.isoformat()},
        {"n_models": len(models), "pooled": bt.get("pooled")},
        materials.fingerprint,
        at=at,
    )
    cache = {
        "run_id": run_id,
        "incrementality_run": analyze_run["id"],
        "created_at": at,
        "evidence": evidence,
        "forecasts": [
            {
                "origin": f.origin,
                "route": f.route,
                "on": f.on.isoformat(),
                "lam": f.lam,
                "quantiles": f.quantiles,
                "natural_lam": f.natural_lam,
                "uplift_lam": f.uplift_lam,
                "extrapolated": f.extrapolated,
                "notes": f.notes,
            }
            for f in forecasts
        ],
    }
    (Path(args.state_dir)).mkdir(parents=True, exist_ok=True)
    (Path(args.state_dir) / FORECAST_CACHE).write_text(
        json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # 归档每次预测，供审计比较两轮可承诺量
    (Path(args.state_dir) / f"forecast_cache_{run_id}.json").write_text(
        json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    store.save_runs(runs)
    md_payload = {
        "created_at": at,
        "n_models": len(models),
        "backtest": bt,
        "forecasts": [
            {
                "origin": f.origin,
                "route": f.route,
                "on": f.on.isoformat(),
                "q05": f.quantiles["q05"],
                "q50": f.quantiles["q50"],
                "q95": f.quantiles["q95"],
                "natural": f.natural_lam,
                "uplift": f.uplift_lam,
                "notes": f.notes,
            }
            for f in forecasts
        ],
    }
    _write_outputs(Path(args.output_dir), "forecast", {**md_payload, "run_id": run_id},
                   render_forecast(md_payload))
    return cache, run_id


def cmd_forecast(args) -> int:
    materials = load(args.root)
    structure = build_structure(materials)
    store = StateStore(args.state_dir)
    cache, run_id = _run_forecast(args, materials, structure, store)
    print(json.dumps({"run_id": run_id, "n_forecasts": len(cache["forecasts"])},
                     ensure_ascii=False))
    return 0


def _load_forecast_cache(state_dir: str) -> dict:
    path = Path(state_dir) / FORECAST_CACHE
    if not path.exists():
        raise SystemExit("缺少需求预测，请先运行 forecast")
    return json.loads(path.read_text(encoding="utf-8"))


def _forecast_map(cache: dict) -> dict[tuple[str, str, dt.date], Forecast]:
    out = {}
    for row in cache["forecasts"]:
        out[(row["origin"], row["route"], parse_date(row["on"]))] = Forecast(
            origin=row["origin"],
            route=row["route"],
            on=parse_date(row["on"]),
            lam=row["lam"],
            quantiles=row["quantiles"],
            natural_lam=row["natural_lam"],
            uplift_lam=row["uplift_lam"],
            extrapolated=row["extrapolated"],
            notes=row["notes"],
        )
    return out


def _build_engine(args, materials, structure, store):
    ledger = store.load_ledger()
    capacity = build_ledger(materials, structure)
    requests = _requests(materials)
    seed_obligations(structure, ledger, materials.table("obligations"), requests)
    replay_commitments(structure, ledger, capacity)
    apply_feedback_closures(capacity, ledger)
    cache = _load_forecast_cache(args.state_dir)
    policy = _policy(args)
    engine = CommitEngine(
        structure,
        ledger,
        capacity,
        _forecast_map(cache),
        cache["evidence"],
        policy,
        basis={
            "forecast_run": cache["run_id"],
            "incrementality_run": cache["incrementality_run"],
        },
        show_up=ledger["feedback"].get("show_up", {}),
    )
    return engine, requests, cache


def cmd_plan(args) -> int:
    materials = load(args.root)
    structure = build_structure(materials)
    store = StateStore(args.state_dir)
    engine, requests, cache = _build_engine(args, materials, structure, store)
    now = _now(args)
    result = engine.plan(requests, now, commit=args.commit, at=now.isoformat())
    payload = {
        "created_at": now.isoformat(),
        "commit": args.commit,
        "policy": {
            "conservatism": engine.policy.conservatism,
            "pilot_fraction": engine.policy.pilot_fraction,
            "protected_util_cap": engine.policy.protected_util_cap,
        },
        "decisions": [
            {
                "group": d.group,
                "status": d.status,
                "route": d.route,
                "people": d.people,
                "committable": d.committable,
                "reasons": d.reasons,
            }
            for d in result.decisions
        ],
        "accepted_people": result.accepted_people,
        "rejected_people": result.rejected_people,
        "idle_risk_people": result.idle_risk_people,
        "protected_pressure": result.protected_pressure,
        "notes": result.notes,
    }
    if args.commit:
        runs = store.load_runs()
        registry = RunRegistry(runs)
        registry.record(
            "plan",
            payload["policy"],
            {
                "accepted_people": result.accepted_people,
                "rejected_people": result.rejected_people,
                "n_commitments": len(engine.ledger["commitments"]),
            },
            materials.fingerprint,
            at=now.isoformat(),
        )
        store.save_runs(runs)
        store.save_ledger(engine.ledger)
    _write_outputs(Path(args.output_dir), "plan", payload, render_plan(payload))
    print(json.dumps({"accepted_people": result.accepted_people,
                      "rejected_people": result.rejected_people,
                      "commit": args.commit}, ensure_ascii=False))
    return 0


def cmd_accept(args) -> int:
    materials = load(args.root)
    structure = build_structure(materials)
    store = StateStore(args.state_dir)
    engine, requests, cache = _build_engine(args, materials, structure, store)
    now = _now(args)
    try:
        commitment = engine.accept(requests, args.group, now, at=now.isoformat())
    except KeyError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2
    store.save_ledger(engine.ledger)
    print(json.dumps({"id": commitment["id"], "state": commitment["state"],
                      "route": commitment["route"]}, ensure_ascii=False))
    return 0


def cmd_fulfill(args) -> int:
    materials = load(args.root)
    structure = build_structure(materials)
    store = StateStore(args.state_dir)
    ledger = store.load_ledger()
    capacity = build_ledger(materials, structure)
    requests = _requests(materials)
    # 恢复台账上既有承诺占用的容量与已知关闭，再应用后续事实
    seed_obligations(structure, ledger, materials.table("obligations"), requests)
    replay_commitments(structure, ledger, capacity)
    apply_feedback_closures(capacity, ledger)
    now = _now(args)
    result = apply_events(
        structure, ledger, capacity, requests, materials.table("later_events"),
        at=now.isoformat(),
    )
    store.save_ledger(ledger)
    payload = {
        "created_at": now.isoformat(),
        "substitutions": result.substitutions,
        "refunds": result.refunds,
        "manual_reviews": result.manual_reviews,
        "outcomes": result.outcomes,
        "feedback": result.feedback,
    }
    _write_outputs(Path(args.output_dir), "fulfill", payload, render_fulfill(payload))
    print(json.dumps({"substitutions": len(result.substitutions),
                      "refunds": len(result.refunds),
                      "manual_reviews": len(result.manual_reviews)}, ensure_ascii=False))
    return 0


def _capacity_map(structure, capacity, requests) -> dict[tuple, int]:
    """每个请求键（客源地, 线路, 日期）上，线路全部资源剩余的最小值。"""
    out = {}
    for request in requests.values():
        day = parse_date(request["on"])
        for route in {request["route"], *request["allowed_substitutes"]}:
            sites = structure.routes[route].sites
            if any((day, site) in capacity.closed for site in sites):
                out[(request["origin"], route, request["on"])] = 0
                continue
            remain = []
            for site in sites:
                remain.append(capacity.remaining_total.get((day, site), 0))
                remain.append(capacity.remaining_protected.get((day, site), 0))
            bus = structure.routes[route].bus
            remain.append(capacity.remaining_bus.get((day, bus), 0))
            out[(request["origin"], route, request["on"])] = min(remain) if remain else 0
    return out


def _demand_map(cache: dict, requests, policy: Policy) -> dict[tuple, int]:
    from .policy import committable as policy_committable

    forecasts = _forecast_map(cache)
    out = {}
    for request in requests.values():
        day = parse_date(request["on"])
        for route in {request["route"], *request["allowed_substitutes"]}:
            forecast = forecasts.get((request["origin"], route, day))
            result = policy_committable(forecast, cache["evidence"].get(route, "无法识别"), policy)
            out[(request["origin"], route, request["on"])] = result.quantity
    return out


def cmd_audit(args) -> int:
    materials = load(args.root)
    structure = build_structure(materials)
    store = StateStore(args.state_dir)
    ledger = store.load_ledger()
    runs = store.load_runs()
    registry = RunRegistry(runs)
    items = [
        commitment_lineage(c, registry, ledger["outcomes"])
        for c in sorted(ledger["commitments"].values(), key=lambda c: c["id"])
    ]
    # 可承诺量变化：比较最近两轮预测，容量侧分解承诺占用与供给变化
    committable_change = []
    caches = sorted(Path(args.state_dir).glob("forecast_cache_*.json"))
    if len(caches) >= 2:
        from .audit import explain_committable_change
        from .capacity import build_ledger as build_capacity

        requests = _requests(materials)
        policy = _policy(args)
        before_cache = json.loads(caches[-2].read_text(encoding="utf-8"))
        after_cache = json.loads(caches[-1].read_text(encoding="utf-8"))
        demand_before = _demand_map(before_cache, requests, policy)
        demand_after = _demand_map(after_cache, requests, policy)
        cap_free = build_capacity(materials, structure)
        capacity_before = _capacity_map(structure, cap_free, requests)
        cap_used = build_capacity(materials, structure)
        seed_obligations(structure, ledger, materials.table("obligations"), requests)
        replay_commitments(structure, ledger, cap_used)
        apply_feedback_closures(cap_used, ledger)
        capacity_after = _capacity_map(structure, cap_used, requests)
        keys = set(demand_before) | set(demand_after)
        before = {k: min(capacity_before.get(k, 0), demand_before.get(k, 0)) for k in keys}
        after = {k: min(capacity_after.get(k, 0), demand_after.get(k, 0)) for k in keys}
        committable_change = explain_committable_change(
            before, after,
            capacity_before=capacity_before, capacity_after=capacity_after,
            demand_before=demand_before, demand_after=demand_after,
        )
    payload = {
        "items": items,
        "runs": runs["runs"],
        "feedback": ledger["feedback"],
        "committable_change": committable_change,
    }
    _write_outputs(Path(args.output_dir), "audit", payload,
                   render_audit(items, committable_change))
    print(json.dumps({"commitments": len(items), "runs": len(runs["runs"])},
                     ensure_ascii=False))
    return 0


def cmd_demo(args) -> int:
    """用固定时点依次跑通整条业务路径，产出全部报告。

    履约后再跑一轮预测，让审计能展示新到访资料如何改变下一轮可承诺量。
    """
    args.now = args.now or "2026-09-18T09:00:00Z"
    steps = (
        cmd_validate, cmd_analyze, cmd_forecast, cmd_plan,
        cmd_fulfill, cmd_forecast, cmd_audit,
    )
    for step in steps:
        rc = step(args)
        if rc != 0:
            return rc
    print("demo 完成，报告见 " + args.output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tourflow", description=__doc__)
    parser.add_argument("--root", default=".", help="资料根目录（含 data/）")
    parser.add_argument("--state-dir", default="state", help="本地状态目录")
    parser.add_argument("--output-dir", default="outputs", help="报告输出目录")
    parser.add_argument("--now", default=None, help="业务时点（UTC，ISO 格式）")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--min-controls", type=int, default=2)
    parser.add_argument("--control-scope", default="strict",
                        choices=["strict", "include_spillover"])
    parser.add_argument("--conservatism", type=float, default=0.5)
    parser.add_argument("--pilot-fraction", type=float, default=0.25)
    parser.add_argument("--protected-util-cap", type=float, default=0.95)
    parser.add_argument("--backtest-cutoff", type=parse_date, default=dt.date(2026, 7, 30))
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="校验合成资料")
    sub.add_parser("analyze", help="增量测算")
    sub.add_parser("forecast", help="需求预测与回测")
    plan = sub.add_parser("plan", help="承诺试算")
    plan.add_argument("--commit", action="store_true", help="把可行承诺落账")
    accept = sub.add_parser("accept", help="接受单个团体（幂等）")
    accept.add_argument("--group", required=True)
    sub.add_parser("fulfill", help="应用后续事实（关闭/核销/接驳）")
    sub.add_parser("audit", help="承诺溯源")
    sub.add_parser("demo", help="依次跑通整条业务路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "validate": cmd_validate,
        "analyze": cmd_analyze,
        "forecast": cmd_forecast,
        "plan": cmd_plan,
        "accept": cmd_accept,
        "fulfill": cmd_fulfill,
        "audit": cmd_audit,
        "demo": cmd_demo,
    }
    handler = handlers[args.command]
    if args.command == "demo":
        args.commit = True  # demo 中 plan 步骤直接落账
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
