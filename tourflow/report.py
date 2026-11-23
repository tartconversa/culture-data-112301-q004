"""把分析结果渲染为 Markdown / JSON 报告，供运营阅读与归档。"""
from __future__ import annotations

from .incrementality import IncrementalityReport


def render_incrementality(report: IncrementalityReport) -> str:
    lines = [
        "# 增量测算报告",
        "",
        f"- 生成时间：{report.created_at}",
        f"- 资料指纹：{report.data_fingerprint[:16]}",
        f"- 对照范围：{report.config['control_scope']}（min_controls={report.config['min_controls']}）",
        "",
        "| 线路 | 归因判定 | 到访次数增量 ATT | 95% 区间 | 唯一游客增量区间 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for route_id, r in sorted(report.routes.items()):
        att = f"{r.att_visits:.2f}" if r.att_visits is not None else "—"
        ci = f"[{r.att_visits_ci[0]:.2f}, {r.att_visits_ci[1]:.2f}]" if r.att_visits_ci else "—"
        uniq = (
            f"[{r.att_unique[0]:.2f}, {r.att_unique[1]:.2f}]" if r.att_unique else "—"
        )
        lines.append(f"| {route_id} | {r.verdict} | {att} | {ci} | {uniq} |")
    lines.append("")
    for route_id, r in sorted(report.routes.items()):
        lines.append(f"## {route_id}")
        lines.append("")
        lines.append(f"- 判定：{r.verdict}")
        lines.append(f"- 唯一游客换算：{r.att_unique_note}")
        pre = r.did.pretrend
        weekly = pre.get("weekly_leads", [])
        if weekly:
            bound = max(abs(w["estimate"]) for w in weekly)
            wald_p = pre.get("wald_p")
            wald_text = "<0.001" if isinstance(wald_p, float) and wald_p < 0.001 else (
                f"{wald_p:.3f}" if isinstance(wald_p, float) else "—"
            )
            lines.append(
                f"- 实施前趋势：前置偏差量级 {bound:.2f}，"
                f"Wald p={wald_text}，均值 {pre.get('mean_weekly_lead'):.2f}"
            )
        bal = r.did.balance

        def _fmt(value):
            return f"{value:.2f}" if isinstance(value, float) else "—"

        lines.append(
            f"- 实施前均衡：降雨 处理 {_fmt(bal.get('treated_rain_mean'))} / "
            f"对照 {_fmt(bal.get('control_rain_mean'))}；"
            f"假日占比 处理 {_fmt(bal.get('treated_holiday_share'))} / "
            f"对照 {_fmt(bal.get('control_holiday_share'))}"
        )
        lines.append(
            f"- 单元：使用 {r.did.cells_used}，无法识别 {len(r.did.cells_unidentified)}，"
            f"剔除售罄截尾 {r.did.censored_cells_dropped}，剔除关闭 {r.did.closed_cells_dropped}"
        )
        if r.did.spillover["items"]:
            lines.append("- 外溢描述（不作归因）：")
            for item in r.did.spillover["items"]:
                lines.append(
                    f"  - {item['origin']} 自 {item['exposed_from']} 被波及，"
                    f"相对干净对照均值差 {item['mean_diff']:.2f}（{item['n_periods']} 期）"
                )
        if r.gaps:
            lines.append("- 识别缺口：")
            for gap in r.gaps:
                lines.append(f"  - {gap}")
        if r.did.notes:
            lines.append("- 备注：")
            for note in r.did.notes:
                lines.append(f"  - {note}")
        lines.append("")
    if report.sensitivity:
        lines.append("## 敏感性口径（保留售罄截尾单元）")
        lines.append("")
        for route_id, s in sorted(report.sensitivity.items()):
            att = f"{s['att']:.2f}" if s["att"] is not None else "—"
            lines.append(f"- {route_id}: ATT {att}（{s['note']}）")
        lines.append("")
    if report.gaps:
        lines.append("## 总体识别缺口")
        lines.append("")
        for gap in report.gaps:
            lines.append(f"- {gap}")
        lines.append("")
    return "\n".join(lines)


def render_forecast(summary: dict) -> str:
    lines = [
        "# 需求预测报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 模型数：{summary['n_models']}（客源地×线路）",
        "",
        "## 回测表现",
        "",
    ]
    pooled = summary["backtest"].get("pooled")
    if pooled:
        lines.append(
            f"- 汇总：n={pooled['n']}，MAE={pooled['mae']:.2f}，"
            f"偏差={pooled['bias']:.2f}，80% 区间覆盖率={pooled['coverage_80']:.0%}"
        )
        if pooled["coverage_80"] > 0.9:
            lines.append("- 覆盖率高于名义水平：区间偏保守，承诺侧偏安全")
        elif pooled["coverage_80"] < 0.7:
            lines.append("- 覆盖率低于名义水平：区间偏窄，承诺需额外留缓冲")
    else:
        lines.append("- 回测窗口内无可评估单元")
    lines.append("")
    lines.append("## 承诺用预测（按客源地×线路×日，节选）")
    lines.append("")
    lines.append("| 客源地 | 线路 | 日期 | 中位 | q05 | q95 | 自然需求 | 推广变化 | 备注 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary["forecasts"][:80]:
        notes = "；".join(row["notes"])
        lines.append(
            f"| {row['origin']} | {row['route']} | {row['on']} "
            f"| {row['q50']:.0f} | {row['q05']:.0f} | {row['q95']:.0f} "
            f"| {row['natural']:.1f} | {row['uplift']:.1f} | {notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_plan(plan: dict) -> str:
    lines = [
        "# 承诺试算报告",
        "",
        f"- 策略：conservatism={plan['policy']['conservatism']}，"
        f"pilot_fraction={plan['policy']['pilot_fraction']}，"
        f"protected_util_cap={plan['policy']['protected_util_cap']}",
        f"- 接受人数：{plan['accepted_people']}；拒绝人数：{plan['rejected_people']}；"
        f"预计闲置（不到访）：{plan['idle_risk_people']:.1f}",
        "",
        "| 团体 | 决定 | 线路 | 人数 | 可承诺量 | 理由 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for d in plan["decisions"]:
        reasons = "；".join(d["reasons"]) if d["reasons"] else ""
        lines.append(
            f"| {d['group']} | {d['status']} | {d['route'] or '—'} "
            f"| {d['people']} | {d['committable']} | {reasons} |"
        )
    lines.append("")
    over = [p for p in plan["protected_pressure"] if p["over_cap"]]
    lines.append("## 保护区压力")
    lines.append("")
    if over:
        for p in over:
            lines.append(
                f"- {p['on']} {p['site']}：利用率 {p['protected_utilization']:.0%}（超上限）"
            )
    else:
        lines.append("- 全部场馆×日均在保护区上限内")
    lines.append("")
    for note in plan.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines)


def render_fulfill(result: dict) -> str:
    lines = [
        "# 履约处理报告",
        "",
        f"- 替代：{len(result['substitutions'])}；退款：{len(result['refunds'])}；"
        f"人工协商：{len(result['manual_reviews'])}",
        "",
    ]
    for s in result["substitutions"]:
        lines.append(f"- 替代：{s['group']} {s['from']} → {s['to']}（{s['people']} 人）")
    for r in result["refunds"]:
        lines.append(f"- 退款：{r['group']}（{r['people']} 人）")
    for m in result["manual_reviews"]:
        lines.append(f"- 人工协商：{m['group']}（{m['people']} 人，{m['reason']}）")
    if result.get("outcomes"):
        lines.append("")
        lines.append("## 核销分类")
        lines.append("")
        for group, outcome in sorted(result["outcomes"].items()):
            lines.append(
                f"- {group}：{outcome.get('classification')}，"
                f"承诺 {outcome.get('committed_people')}，实际 {outcome.get('actual_people')}"
            )
    lines.append("")
    return "\n".join(lines)


def render_audit(items: list[dict], committable_change: list[dict] | None = None) -> str:
    lines = [
        "# 承诺溯源报告",
        "",
    ]
    for item in items:
        lines.append(f"## {item['commitment']}（{item['group']}）")
        lines.append("")
        lines.append(
            f"- 状态：{item['state']}；线路 {item['route']}；{item['on']}；{item['people']} 人"
        )
        basis = item["basis"]
        if item["source"] == "seed":
            lines.append(f"- 依据：{basis.get('note', '历史承诺')}")
        else:
            policy = basis.get("policy", {})
            lines.append(
                f"- 依据：增量运行 {basis.get('incrementality_run', '—')}；"
                f"预测运行 {basis.get('forecast_run', '—')}；"
                f"策略 conservatism={policy.get('conservatism')}"
            )
        outcome = item["outcome"]
        if outcome.get("classification"):
            lines.append(
                f"- 履约：{outcome['classification']}，承诺 {outcome.get('committed_people')}，"
                f"实际 {outcome.get('actual_people')}"
            )
        for h in item["history"]:
            lines.append(f"  - {h.get('ts')}: {h.get('action')}")
        lines.append("")
    if committable_change is not None:
        lines.append("## 两轮可承诺量变化")
        lines.append("")
        if not committable_change:
            lines.append("- 两轮之间可承诺量无变化")
        else:
            lines.append("| 键（客源地\|线路\|日期） | 上轮 | 本轮 | 容量效应 | 需求效应 |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in committable_change:
                key = row["key"].replace("|", " \\| ")
                lines.append(
                    f"| {key} | {row['before']} | {row['after']} "
                    f"| {row['capacity_effect']:+d} | {row['demand_effect']:+d} |"
                )
            lines.append("")
            lines.append("- 分解顺序：先替换容量约束（承诺占用与关闭），再替换需求上限")
        lines.append("")
    return "\n".join(lines)
