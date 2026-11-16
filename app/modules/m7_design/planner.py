"""M7 多实验对照编排：把整组实验的对照分配当成一个联合优化问题。

三个刻意的选择：
1. 目标函数是「整组实验的前置趋势偏差之和」，不是每个实验各自挑最优——
   各自挑会把同一个对照商品在重叠日期重复占用；
2. 已开跑（locked）实验的对照钉死不动，只作为约束进入求解，
   新实验到来只能重排草案（queued/confirmed）；
3. 同商品同渠道只在「窗口重叠」时互斥——错开日期的两个实验可以复用同一个对照，
   占用表按日落库，互斥由 (sku, channel, date) 主键兜底。

求解用连通分量内的精确枚举：分量 = 同渠道且实验窗两两可能相碰的一组实验，
分量之间互不影响，可以各自独立求最优。分量内先最大化可分配实验数，
再最小化总趋势差，最后按字典序取确定解（同一状态重算结果一定相同）。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from . import control, power
from .service import clean_trend_window

# 分量规模上限：超过就退回贪心并标注，避免枚举爆炸（实际分量只有个位数）
MAX_ENUMERATE_NODES = 10
MAX_ENUMERATE_COMBOS = 200_000


@dataclass
class Node:
    """一个待分配实验：候选域 + 是否锁定。"""
    experiment_id: str
    sku: str
    channel: str
    start: dt.date
    end: dt.date
    status: str
    locked: bool
    domain: list[dict] = field(default_factory=list)   # 可用候选，按 bias 升序
    pinned: dict | None = None                          # 锁定时的占用（唯一选择）
    best_bias: float | None = None                      # 各自不受约束时的最佳候选
    best_sku: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.sku, self.channel)


def overlaps(a: Node, b: Node) -> bool:
    """窗口重叠才互斥：左闭右开，相接不算重叠。"""
    return a.channel == b.channel and a.start < b.end and b.start < a.end


def _candidate_rows(store, batch_id: str, sku: str, channel: str,
                    start: dt.date, end: dt.date, cache: dict) -> list:
    """候选对照：与单实验设计同一套打分（control.build），只留「可用」进域。"""
    if batch_id not in cache:
        cache[batch_id] = (store.sales(batch_id), store.promo_windows(batch_id))
    rows, windows = cache[batch_id]
    series = power.build_series(rows, sku, channel)
    if series is None:
        return []
    trend_dates, _note = clean_trend_window(series, windows, sku, channel)
    return control.build(rows, windows, series, (sku, channel), start, end,
                         trend_dates)


def build_node(store, experiment: dict, cache: dict, conn=None) -> Node:
    start = dt.date.fromisoformat(experiment["start_date"])
    end = dt.date.fromisoformat(experiment["end_date"])
    locked = experiment["status"] == "running"
    candidates = _candidate_rows(store, experiment["batch_id"], experiment["sku"],
                                 experiment["channel"], start, end, cache)
    node = Node(experiment_id=experiment["experiment_id"],
                sku=experiment["sku"], channel=experiment["channel"],
                start=start, end=end, status=experiment["status"], locked=locked)
    usable = [item for item in candidates if item.status == "可用"]
    node.domain = [{"sku": item.sku, "channel": item.channel, "bias": item.bias}
                   for item in sorted(usable, key=lambda c: (c.bias, c.sku))]
    if node.domain:
        node.best_bias = node.domain[0]["bias"]
        node.best_sku = node.domain[0]["sku"]
    if locked:
        pinned_sku = experiment["control_sku"]
        hit = next((item for item in candidates if item.sku == pinned_sku), None)
        if hit is not None:
            bias = hit.trend_gap
        elif experiment.get("plan_bias"):
            bias = float(experiment["plan_bias"])
        else:
            bias = 0.0
        node.pinned = {"sku": pinned_sku,
                       "channel": experiment["control_channel"] or experiment["channel"],
                       "bias": bias}
    return node


def components(nodes: list[Node]) -> list[list[Node]]:
    """冲突图的连通分量：同渠道且窗口重叠就连一条边。"""
    parent = {node.experiment_id: node.experiment_id for node in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if overlaps(a, b):
                parent[find(a.experiment_id)] = find(b.experiment_id)
    groups: dict[str, list[Node]] = {}
    for node in nodes:
        groups.setdefault(find(node.experiment_id), []).append(node)
    return [sorted(group, key=lambda n: n.experiment_id) for group in groups.values()]


def _conflicts(pick: dict, node: Node, placed: list[tuple[dict, Node]]) -> bool:
    for other_pick, other in placed:
        if pick["sku"] == other_pick["sku"] and overlaps(node, other):
            return True
    return False


def _solve_exact(free: list[Node], placed: list[tuple[dict, Node]]) -> dict:
    """分支限界枚举：先比可分配数，再比总趋势差，最后比字典序保证确定。"""
    best = {"count": -1, "bias": float("inf"), "assign": None, "stamp": None}

    def stamp_of(assign: dict) -> tuple:
        return tuple(sorted((eid, pick["sku"]) for eid, pick in assign.items()))

    def dfs(index: int, assign: dict, used: list[tuple[dict, Node]], bias: float):
        remaining = len(free) - index
        if len(assign) + remaining < best["count"]:
            return
        if len(assign) + remaining == best["count"] and bias >= best["bias"]:
            return
        if index == len(free):
            stamp = stamp_of(assign)
            count = len(assign)
            if (count > best["count"] or
                    (count == best["count"] and bias < best["bias"] - 1e-12) or
                    (count == best["count"] and abs(bias - best["bias"]) <= 1e-12
                     and (best["stamp"] is None or stamp < best["stamp"]))):
                best.update(count=count, bias=bias, assign=dict(assign), stamp=stamp)
            return
        node = free[index]
        for pick in node.domain:
            if not _conflicts(pick, node, used):
                assign[node.experiment_id] = pick
                used.append((pick, node))
                dfs(index + 1, assign, used, bias + pick["bias"])
                used.pop()
                del assign[node.experiment_id]
        # 不分配这个实验也是一种选择（让位给分量里其它实验）
        dfs(index + 1, assign, used, bias)

    dfs(0, {}, list(placed), 0.0)
    return best


def _solve_greedy(free: list[Node], placed: list[tuple[dict, Node]]) -> dict:
    """分量过大时的退路：按候选数升序逐个挑不冲突的最优候选。"""
    used = list(placed)
    assign: dict[str, dict] = {}
    bias = 0.0
    for node in free:
        for pick in node.domain:
            if not _conflicts(pick, node, used):
                assign[node.experiment_id] = pick
                used.append((pick, node))
                bias += pick["bias"]
                break
    return {"count": len(assign), "bias": bias, "assign": assign, "stamp": None}


def solve(group: list[Node]) -> tuple[dict[str, dict], bool]:
    """求一个分量的最优分配；返回 {experiment_id: 占用} 与是否精确解。"""
    placed: list[tuple[dict, Node]] = []
    free: list[Node] = []
    for node in group:
        if node.locked and node.pinned is not None:
            placed.append((node.pinned, node))
        elif not node.locked:
            free.append(node)
    free.sort(key=lambda n: (len(n.domain), n.experiment_id))
    combos = 1
    for node in free:
        combos *= max(1, len(node.domain) + 1)
    if len(free) > MAX_ENUMERATE_NODES or combos > MAX_ENUMERATE_COMBOS:
        return _solve_greedy(free, placed)["assign"], False
    return _solve_exact(free, placed)["assign"], True


def _busy_detail(node: Node, group: list[Node],
                 assignments: dict[str, dict]) -> str:
    """无解原因：逐个候选说明被谁占住（锁定优先，其次同组已分配）。"""
    by_id = {item.experiment_id: item for item in group}
    blockers: list[str] = []
    for pick in node.domain:
        holders: list[str] = []
        for other in group:
            if other.experiment_id == node.experiment_id:
                continue
            if not overlaps(node, other):
                continue
            if other.locked and other.pinned and other.pinned["sku"] == pick["sku"]:
                holders.append(f"{other.experiment_id}(已开跑)")
                continue
            taken = assignments.get(other.experiment_id)
            if taken and taken["sku"] == pick["sku"]:
                holders.append(other.experiment_id)
        if holders:
            blockers.append(f"{pick['sku']} 被 {'、'.join(sorted(set(holders)))} 占用")
        else:
            blockers.append(f"{pick['sku']} 与整组分配冲突")
    return "；".join(blockers[:4])


def plan(store, conn=None) -> dict:
    """整组实验的联合分配方案：每个实验一行，含相对各自最佳候选的损失与无解原因。"""
    experiments = store.experiments(("queued", "confirmed", "running"), conn)
    revision = store.experiment_revision(conn)
    cache: dict = {}
    nodes = [build_node(store, exp, cache, conn) for exp in experiments]
    by_id = {exp["experiment_id"]: exp for exp in experiments}
    groups = components(nodes)
    group_of = {node.experiment_id: group for group in groups for node in group}
    assignments: dict[str, dict] = {}
    exact = True
    for group in groups:
        solved, is_exact = solve(group)
        exact = exact and is_exact
        assignments.update(solved)
        for node in group:
            if node.locked and node.pinned is not None:
                assignments[node.experiment_id] = node.pinned

    rows: list[dict] = []
    total_bias = 0.0
    total_loss = 0.0
    assigned_count = 0
    for node in nodes:
        exp = by_id[node.experiment_id]
        assigned = assignments.get(node.experiment_id)
        row = {
            "experiment_id": node.experiment_id,
            "sku": node.sku,
            "channel": node.channel,
            "start_date": exp["start_date"],
            "end_date": exp["end_date"],
            "status": node.status,
            "locked": 1 if node.locked else 0,
            "best_sku": node.best_sku,
            "best_bias": (power.dec(node.best_bias) if node.best_bias is not None
                          else ""),
        }
        if assigned is not None:
            bias = assigned["bias"]
            loss = max(0.0, bias - node.best_bias) if node.best_bias is not None else 0.0
            total_bias += bias
            total_loss += loss
            assigned_count += 1
            row.update({
                "planned_sku": assigned["sku"],
                "planned_channel": assigned["channel"],
                "planned_bias": power.dec(bias),
                "loss": power.dec(loss),
                "feasible": 1,
                "code": "",
                "detail": "",
            })
        else:
            if not node.domain:
                code, detail = "D_NO_CONTROL", "同渠道没有可用对照候选"
            else:
                code = "D_CONTROL_BUSY"
                detail = _busy_detail(node, group_of[node.experiment_id],
                                      assignments)
            row.update({
                "planned_sku": "",
                "planned_channel": "",
                "planned_bias": "",
                "loss": "",
                "feasible": 0,
                "code": code,
                "detail": detail,
            })
        rows.append(row)

    return {
        "revision": revision,
        "exact": 1 if exact else 0,
        "assigned": assigned_count,
        "infeasible": len(rows) - assigned_count,
        "total_bias": power.dec(total_bias),
        "total_loss": power.dec(total_loss),
        "rows": rows,
    }
