"""M7 多实验对照编排：联合分配、锁定、占用互斥、并发、恢复与幂等。

合成批次直接写库（不读时钟），期望值由构造数据决定；求解器用纯 Node 测。
"""
from __future__ import annotations

import datetime as dt
import json
import threading

import pytest

from app import config
from app.errors import Conflict
from app.main import create_app
from app.modules.m7_design import experiments, planner
from app.store import Store

DAY = dt.timedelta(days=1)
HIST_START = dt.date(2026, 9, 1)


def _node(eid, channel, start, end, domain, locked=False, pinned=None):
    return planner.Node(
        experiment_id=eid, sku=f"T-{eid}", channel=channel,
        start=dt.date.fromisoformat(start), end=dt.date.fromisoformat(end),
        status="running" if locked else "queued", locked=locked,
        domain=[{"sku": sku, "channel": channel, "bias": bias}
                for sku, bias in domain],
        pinned=pinned,
        best_bias=domain[0][1] if domain else None,
        best_sku=domain[0][0] if domain else "")


# ---------- 求解器（纯函数） ----------

def test_solver_prefers_global_optimum_over_per_experiment_greedy():
    # 各自挑：E1 拿 A(0.01)，E2 无解；联合最优：E1→B、E2→A，整组 0.07 都可分配
    e1 = _node("E1", "电商", "2026-11-01", "2026-11-15",
               [("A", 0.01), ("B", 0.05)])
    e2 = _node("E2", "电商", "2026-11-05", "2026-11-20", [("A", 0.02)])
    group = planner.components([e1, e2])
    assert len(group) == 1
    assign, exact = planner.solve(group[0])
    assert exact is True
    assert assign["E1"]["sku"] == "B"
    assert assign["E2"]["sku"] == "A"
    assert abs(assign["E1"]["bias"] + assign["E2"]["bias"] - 0.07) < 1e-9


def test_solver_minimizes_total_bias_not_individual():
    # E1→A、E2→B 合计 0.10；联合最优 E1→B、E2→A 合计 0.035
    e1 = _node("E1", "电商", "2026-11-01", "2026-11-15",
               [("A", 0.01), ("B", 0.02)])
    e2 = _node("E2", "电商", "2026-11-01", "2026-11-15",
               [("A", 0.015), ("B", 0.09)])
    assign, _ = planner.solve(planner.components([e1, e2])[0])
    assert (assign["E1"]["sku"], assign["E2"]["sku"]) == ("B", "A")


def test_solver_keeps_locked_control_and_works_around_it():
    locked = _node("E0", "电商", "2026-11-01", "2026-11-30", [],
                   locked=True, pinned={"sku": "A", "channel": "电商", "bias": 0.02})
    free = _node("E1", "电商", "2026-11-10", "2026-11-20",
                 [("A", 0.01), ("B", 0.03)])
    assign, _ = planner.solve(planner.components([locked, free])[0])
    assert assign["E1"]["sku"] == "B"  # A 被锁定实验占住，草案只能选 B


def test_solver_reuses_control_when_windows_do_not_overlap():
    # 左闭右开：E1 结束日 = E2 开始日，不算重叠，同一对照可复用
    e1 = _node("E1", "电商", "2026-11-01", "2026-11-15", [("A", 0.01)])
    e2 = _node("E2", "电商", "2026-11-15", "2026-11-30", [("A", 0.01)])
    groups = planner.components([e1, e2])
    assert len(groups) == 2  # 不重叠就不在一个分量里
    assign, _ = planner.solve(planner.components([e1, e2])[0])
    assert assign["E1"]["sku"] == "A"


def test_solver_is_deterministic():
    def run():
        e1 = _node("E1", "电商", "2026-11-01", "2026-11-15",
                   [("A", 0.02), ("B", 0.02)])
        e2 = _node("E2", "电商", "2026-11-01", "2026-11-15",
                   [("A", 0.02), ("B", 0.02)])
        assign, _ = planner.solve(planner.components([e1, e2])[0])
        return {k: v["sku"] for k, v in assign.items()}

    assert run() == run() == {"E1": "A", "E2": "B"}


# ---------- 合成批次 ----------

def _sales_rows(skus_by_channel: dict, values_by_sku: dict, days: int = 20):
    rows = []
    line = 0
    for channel, skus in skus_by_channel.items():
        for sku in skus:
            series = values_by_sku.get(sku, [1000] * days)
            for index in range(days):
                line += 1
                day = (HIST_START + DAY * index).isoformat()
                gmv = series[index] if index < len(series) else series[-1]
                rows.append({"line_no": line, "sale_date": day, "channel": channel,
                             "sku": sku, "sku_name": sku, "price_incl": "10.00",
                             "price_unit": "元/件", "tax_rate": "0.13",
                             "price_excl": "8.8496", "qty": 100,
                             "promo_id": "", "amount_excl": str(gmv)})
    return rows


def make_batch(store: Store, batch_id: str, skus_by_channel: dict,
               values_by_sku: dict | None = None, promo_windows=()) -> None:
    store.upsert_batch({
        "batch_id": batch_id, "batch_key": f"key-{batch_id}", "run_id": "r1",
        "dataset": "synthetic", "status": "done",
        "created_at": "2026-09-21T00:00:00+00:00",
        "finished_at": "2026-09-21T00:00:01+00:00", "input_digest": "d",
        "params": {}, "note": ""})
    store.write_batch(batch_id, {
        "sales": _sales_rows(skus_by_channel, values_by_sku or {}),
        "promo_windows": [
            {"promo_id": w.get("promo_id", "PX"), "sku": w["sku"],
             "channel": w["channel"], "start_date": w["start_date"],
             "end_date": w["end_date"], "version": 1, "price_incl": "9.00",
             "price_excl": "7.9646", "days": 5, "base_price_excl": "8.8496"}
            for w in promo_windows
        ]})


def _enqueue(store, batch_id, sku, channel, start, end):
    return experiments.enqueue(store, {
        "batch_id": batch_id, "sku": sku, "channel": channel,
        "start_date": start, "end_date": end})["experiment"]["experiment_id"]


# 趋势构造：前半 1000、后半抬升，趋势差由抬升幅度决定
VALUES = {
    "T": [1000] * 10 + [1080] * 10,   # 目标趋势 ≈ 0.0770
    "B": [1000] * 10 + [1100] * 10,   # 差距 ≈ 0.0183（各自最佳）
    "C": [1000] * 10 + [1050] * 10,   # 差距 ≈ 0.0282（次佳）
    "D": [1000] * 20,                 # 差距 ≈ 0.0770
}
GAP_B = abs(__import__("math").log(1.10) - __import__("math").log(1.08))
GAP_C = abs(__import__("math").log(1.05) - __import__("math").log(1.08))


@pytest.fixture()
def synth(store):
    make_batch(store, "TB1", {"电商": ["T", "B", "C", "D"], "孤渠道": ["L"]},
               VALUES)
    return store


def test_plan_assigns_jointly_and_reports_loss_vs_best(synth):
    # 两个实验同一个目标商品：B 只能给一家，另一家拿 C，损失 = 次佳 − 最佳
    e1 = _enqueue(synth, "TB1", "T", "电商", "2026-11-01", "2026-11-21")
    e2 = _enqueue(synth, "TB1", "T", "电商", "2026-11-10", "2026-11-30")
    proposal = planner.plan(synth)
    rows = {row["experiment_id"]: row for row in proposal["rows"]}
    picks = {rows[e1]["planned_sku"], rows[e2]["planned_sku"]}
    assert picks == {"B", "C"}  # 不重复占用
    loser = e1 if rows[e1]["planned_sku"] == "C" else e2
    winner = e2 if loser == e1 else e1
    assert rows[winner]["planned_sku"] == "B"
    assert float(rows[winner]["loss"]) == pytest.approx(0.0, abs=1e-4)
    assert float(rows[loser]["loss"]) == pytest.approx(GAP_C - GAP_B, abs=1e-4)
    assert rows[loser]["best_sku"] == "B"  # 各自最佳仍是 B，只是没抢到
    assert float(proposal["total_loss"]) == pytest.approx(GAP_C - GAP_B, abs=1e-4)


def test_plan_reports_no_control_and_busy_reasons(synth):
    # 孤渠道只有一个商品：没有候选 → D_NO_CONTROL
    alone = _enqueue(synth, "TB1", "L", "孤渠道", "2026-11-01", "2026-11-21")
    # 电商渠道再加两个同目标实验抢 B、C、D：候选够分，不出现 BUSY
    proposal = planner.plan(synth)
    row = next(r for r in proposal["rows"] if r["experiment_id"] == alone)
    assert row["feasible"] == 0
    assert row["code"] == "D_NO_CONTROL"

    # 同目标两个实验 + 只有一个可用对照（B、C、D 被促销排除）→ D_CONTROL_BUSY
    make_batch(synth, "TB2", {"电商": ["T", "B"]}, VALUES,
               promo_windows=[{"sku": "B", "channel": "电商",
                               "start_date": "2026-11-01",
                               "end_date": "2026-11-30"}])
    # B 在实验窗内自己有促销被排除后，两个实验都没有候选 → 都是 D_NO_CONTROL
    e1 = _enqueue(synth, "TB2", "T", "电商", "2026-11-01", "2026-11-21")
    e2 = _enqueue(synth, "TB2", "T", "电商", "2026-11-05", "2026-11-25")
    proposal = planner.plan(synth)
    rows = {row["experiment_id"]: row for row in proposal["rows"]
            if row["experiment_id"] in (e1, e2)}
    assert len(rows) == 2
    assert {row["code"] for row in rows.values()} == {"D_NO_CONTROL"}

    # 三个同目标实验抢 B、C 两个可用对照：总有一个无解 → D_CONTROL_BUSY
    make_batch(synth, "TB3", {"电商": ["T", "B", "C"]}, VALUES)
    e1 = _enqueue(synth, "TB3", "T", "电商", "2026-11-01", "2026-11-21")
    e2 = _enqueue(synth, "TB3", "T", "电商", "2026-11-05", "2026-11-25")
    e3 = _enqueue(synth, "TB3", "T", "电商", "2026-11-10", "2026-11-30")
    proposal = planner.plan(synth)
    rows = {row["experiment_id"]: row for row in proposal["rows"]
            if row["experiment_id"] in (e1, e2, e3)}
    assert len(rows) == 3
    busy = [row for row in rows.values() if row["code"] == "D_CONTROL_BUSY"]
    assert len(busy) == 1
    assert "占用" in busy[0]["detail"]
    assert proposal["assigned"] >= 2


def test_locked_experiment_keeps_control_when_new_experiment_arrives(synth):
    # E0 开跑后锁定对照 B；新实验唯一候选也是 B，只能无解，不能调走 E0 的 B
    make_batch(synth, "TB4", {"电商": ["T", "B", "C"]}, VALUES,
               promo_windows=[{"sku": "C", "channel": "电商",
                               "start_date": "2026-10-15",
                               "end_date": "2026-12-31"}])
    e0 = _enqueue(synth, "TB4", "T", "电商", "2026-11-01", "2026-11-21")
    revision = synth.experiment_revision()
    experiments.confirm(synth, {"request_id": "c0", "base_revision": revision})
    experiments.launch(synth, e0)
    held_before = synth.occupancy_of(e0)
    assert {row["sku"] for row in held_before} == {"B"}

    e1 = _enqueue(synth, "TB4", "T", "电商", "2026-11-05", "2026-11-25")
    proposal = planner.plan(synth)
    rows = {row["experiment_id"]: row for row in proposal["rows"]}
    assert rows[e0]["locked"] == 1
    assert rows[e0]["planned_sku"] == "B"  # 已开跑的不被调走
    assert rows[e1]["feasible"] == 0
    assert rows[e1]["code"] == "D_CONTROL_BUSY"
    assert "已开跑" in rows[e1]["detail"]
    # 占用记录一行没动
    assert synth.occupancy_of(e0) == held_before


def test_confirm_falls_back_infeasible_experiment_to_queued(synth):
    e1 = _enqueue(synth, "TB1", "L", "孤渠道", "2026-11-01", "2026-11-21")
    revision = synth.experiment_revision()
    result = experiments.confirm(synth, {"request_id": "c1",
                                         "base_revision": revision})
    assert result["result"]["applied"] == []
    experiment = synth.experiment_get(e1)
    assert experiment["status"] == "queued"  # 确认不上就回退为排队
    assert experiment["infeasible_code"] == "D_NO_CONTROL"
    assert synth.occupancy_of(e1) == []


# ---------- 真实样例上的生命周期 ----------

def test_enqueue_is_idempotent_on_natural_key(client_run):
    payload = {"sku": "MY-0012", "channel": "电商",
               "start_date": "2026-11-01", "end_date": "2026-12-13"}
    first = client_run.post("/api/experiments", json=payload).json()
    second = client_run.post("/api/experiments", json=payload).json()
    assert first["created"] is True
    assert second["created"] is False
    assert first["experiment"]["experiment_id"] == second["experiment"]["experiment_id"]
    board = client_run.get("/api/experiments").json()
    assert len(board["experiments"]) == 1  # 重试不产生重复排队


def test_confirm_then_retry_does_not_double_occupy(client_run):
    client_run.post("/api/experiments", json={
        "sku": "MY-0012", "channel": "电商",
        "start_date": "2026-11-01", "end_date": "2026-12-13"})
    board = client_run.get("/api/experiments").json()
    revision = board["revision"]
    first = client_run.post("/api/experiments/confirm", json={
        "request_id": "confirm-1", "base_revision": revision})
    assert first.status_code == 200
    days_after_first = client_run.get("/api/experiments").json()[
        "experiments"][0]["occupancy_days"]
    assert days_after_first == 42
    # 同一请求键重试：返回已存结果，占用不翻倍
    retry = client_run.post("/api/experiments/confirm", json={
        "request_id": "confirm-1", "base_revision": revision})
    assert retry.status_code == 200
    assert retry.json()["idempotent"] is True
    board = client_run.get("/api/experiments").json()
    assert board["experiments"][0]["occupancy_days"] == 42
    # 换一个请求键但修订号已旧：409，不抢走已有占用
    stale = client_run.post("/api/experiments/confirm", json={
        "request_id": "confirm-2", "base_revision": revision})
    assert stale.status_code == 409
    assert stale.json()["error"] == "D_PLAN_STALE"


def test_launch_cancel_finish_state_machine(client_run):
    exp = client_run.post("/api/experiments", json={
        "sku": "MY-0012", "channel": "电商",
        "start_date": "2026-11-01", "end_date": "2026-12-13"}).json()["experiment"]
    exp_id = exp["experiment_id"]
    # 排队中不能开跑
    early = client_run.post(f"/api/experiments/{exp_id}/launch")
    assert early.status_code == 409
    assert early.json()["error"] == "D_EXP_STATE"
    board = client_run.get("/api/experiments").json()
    client_run.post("/api/experiments/confirm",
                    json={"base_revision": board["revision"]})
    launched = client_run.post(f"/api/experiments/{exp_id}/launch").json()
    assert launched["experiment"]["status"] == "running"
    assert launched["experiment"]["locked"] == 1
    again = client_run.post(f"/api/experiments/{exp_id}/launch").json()
    assert again["changed"] is False  # 幂等
    # 开跑后不能取消失败状态；结束释放占用
    finished = client_run.post(f"/api/experiments/{exp_id}/finish").json()
    assert finished["experiment"]["status"] == "finished"
    board = client_run.get("/api/experiments").json()
    assert board["experiments"][0]["occupancy_days"] == 0
    late = client_run.post(f"/api/experiments/{exp_id}/cancel")
    assert late.status_code == 409  # 已结束不能再取消


def test_cancel_releases_occupancy_for_others(client_run):
    first = client_run.post("/api/experiments", json={
        "sku": "MY-0012", "channel": "电商",
        "start_date": "2026-11-01", "end_date": "2026-12-13"}).json()["experiment"]
    board = client_run.get("/api/experiments").json()
    client_run.post("/api/experiments/confirm",
                    json={"base_revision": board["revision"]})
    held = client_run.get("/api/experiments").json()["experiments"][0]
    assert held["occupancy_days"] == 42
    client_run.post(f"/api/experiments/{first['experiment_id']}/cancel")
    board = client_run.get("/api/experiments").json()
    assert board["experiments"][0]["status"] == "cancelled"
    assert board["experiments"][0]["occupancy_days"] == 0
    # 取消后同一商品同一窗口可以再来一次
    second = client_run.post("/api/experiments", json={
        "sku": "MY-0012", "channel": "电商",
        "start_date": "2026-11-01", "end_date": "2026-12-13"}).json()
    assert second["created"] is True
    assert second["experiment"]["experiment_id"] != first["experiment_id"]


def test_concurrent_confirm_exactly_one_wins(settings, store, batch_id):
    for sku in ("MY-0012", "MY-0021"):
        experiments.enqueue(store, {"batch_id": batch_id, "sku": sku,
                                    "channel": "电商", "start_date": "2026-11-01",
                                    "end_date": "2026-12-13"})
    revision = store.experiment_revision()
    outcomes = []

    def worker(name):
        try:
            experiments.confirm(store, {"request_id": f"race-{name}",
                                        "base_revision": revision})
            outcomes.append("ok")
        except Conflict:
            outcomes.append("conflict")

    threads = [threading.Thread(target=worker, args=(index,))
               for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("ok") == 1  # 只有一家确认成功，其余 409
    assert outcomes.count("conflict") == 3
    # 占用与分配一致：每个实验 42 天，无重复占用
    occupancy = store.occupancy_rows()
    assert len(occupancy) == 2 * 42
    keys = {(row["sku"], row["channel"], row["date"]) for row in occupancy}
    assert len(keys) == len(occupancy)


def test_concurrent_confirm_same_key_never_clobbers_journal(settings, store,
                                                            batch_id):
    experiments.enqueue(store, {"batch_id": batch_id, "sku": "MY-0012",
                                "channel": "电商", "start_date": "2026-11-01",
                                "end_date": "2026-12-13"})
    revision = store.experiment_revision()
    outcomes = []

    def worker():
        try:
            result = experiments.confirm(store, {"request_id": "same-key",
                                                 "base_revision": revision})
            outcomes.append(("ok", result["idempotent"]))
        except Conflict:
            outcomes.append(("conflict", None))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    oks = [flag for outcome, flag in outcomes if outcome == "ok"]
    assert len(oks) >= 1
    assert oks.count(False) <= 1  # 至多一个真实执行，其余都是幂等命中
    # 关键不变量：成功确认后的日志不会被并发误标 aborted
    assert store.journal_get("same-key")["status"] == "done"
    occupancy = store.occupancy_rows()
    assert len(occupancy) == 42  # 同键并发只占一份


# ---------- 中断恢复 ----------

def test_recover_replays_interrupted_confirm(settings, store, batch_id):
    exp_id = experiments.enqueue(store, {
        "batch_id": batch_id, "sku": "MY-0012", "channel": "电商",
        "start_date": "2026-11-01", "end_date": "2026-12-13"})["experiment"]["experiment_id"]
    revision = store.experiment_revision()
    # 模拟崩溃：日志已写（pending），应用那一步没走到
    store.journal_insert({"request_id": "crash-1", "kind": "confirm",
                          "payload": json.dumps({"base_revision": revision}),
                          "status": "pending",
                          "created_at": "2026-11-01T00:00:00+00:00"})
    report = experiments.recover(store)
    assert report["replayed"] == ["crash-1"]
    assert store.journal_get("crash-1")["status"] == "done"
    experiment = store.experiment_get(exp_id)
    assert experiment["status"] == "confirmed"
    assert len(store.occupancy_of(exp_id)) == 42


def test_recover_aborts_stale_pending_journal(settings, store, batch_id):
    experiments.enqueue(store, {"batch_id": batch_id, "sku": "MY-0012",
                                "channel": "电商", "start_date": "2026-11-01",
                                "end_date": "2026-12-13"})
    store.journal_insert({"request_id": "crash-2", "kind": "confirm",
                          "payload": json.dumps({"base_revision": 999}),
                          "status": "pending",
                          "created_at": "2026-11-01T00:00:00+00:00"})
    report = experiments.recover(store)
    assert report["aborted"] == ["crash-2"]
    assert store.journal_get("crash-2")["status"] == "aborted"
    assert store.occupancy_rows() == []  # 没应用就不留占用


def test_recover_releases_orphans_and_rebuilds_missing(settings, store, batch_id):
    exp_id = experiments.enqueue(store, {
        "batch_id": batch_id, "sku": "MY-0012", "channel": "电商",
        "start_date": "2026-11-01", "end_date": "2026-12-13"})["experiment"]["experiment_id"]
    revision = store.experiment_revision()
    experiments.confirm(store, {"request_id": "c1", "base_revision": revision})
    assert len(store.occupancy_of(exp_id)) == 42
    # 模拟撕裂写：确认实验的占用行丢了；取消实验的占用行还在
    ghost_id = experiments.enqueue(store, {
        "batch_id": batch_id, "sku": "MY-0021", "channel": "电商",
        "start_date": "2026-11-01", "end_date": "2026-11-15"})["experiment"]["experiment_id"]
    experiments.cancel(store, ghost_id)
    with store.write() as conn:
        conn.execute("DELETE FROM control_occupancy WHERE experiment_id=?", (exp_id,))
        conn.execute("INSERT INTO control_occupancy (sku, channel, date, "
                     "experiment_id) VALUES ('MY-0034', '电商', '2026-11-01', ?)",
                     (ghost_id,))
    report = experiments.recover(store)
    assert report["rebuilt"] == [exp_id]
    assert report["released"] == [ghost_id]
    assert len(store.occupancy_of(exp_id)) == 42
    assert store.occupancy_of(ghost_id) == []
    # 恢复是幂等的：再跑一次没有可修的
    second = experiments.recover(store)
    assert second["replayed"] == second["released"] == second["rebuilt"] == []


def test_startup_recovers_before_serving(settings, store, batch_id):
    exp_id = experiments.enqueue(store, {
        "batch_id": batch_id, "sku": "MY-0012", "channel": "电商",
        "start_date": "2026-11-01", "end_date": "2026-12-13"})["experiment"]["experiment_id"]
    revision = store.experiment_revision()
    store.journal_insert({"request_id": "boot-1", "kind": "confirm",
                          "payload": json.dumps({"base_revision": revision}),
                          "status": "pending",
                          "created_at": "2026-11-01T00:00:00+00:00"})
    create_app(settings)  # 工厂里先跑恢复再服务
    assert store.journal_get("boot-1")["status"] == "done"
    assert store.experiment_get(exp_id)["status"] == "confirmed"


# ---------- 页面 ----------

def test_design_page_shows_queue_board_and_enqueue_form(client_run):
    client_run.post("/api/experiments", json={
        "sku": "MY-0012", "channel": "电商",
        "start_date": "2026-11-01", "end_date": "2026-12-13"})
    text = client_run.get("/design").text
    for title in ("实验队列", "对照编排", "新建实验"):
        assert title in text, title
    assert "排队中 · 可重排" in text
    assert "确认占用" in text
    assert "恢复占用记录" in text
    # 走完 排队 → 确认 → 开跑，页面上能看到锁定标记
    board = client_run.get("/api/experiments").json()
    client_run.post("/api/experiments/confirm",
                    json={"base_revision": board["revision"]})
    exp_id = board["experiments"][0]["experiment_id"]
    client_run.post(f"/api/experiments/{exp_id}/launch")
    text = client_run.get("/design").text
    assert "已开跑 · 已锁定" in text
    assert "可分配" in text
