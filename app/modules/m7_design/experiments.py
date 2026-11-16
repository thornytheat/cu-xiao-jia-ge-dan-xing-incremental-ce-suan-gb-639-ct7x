"""M7 实验队列生命周期：排队 → 确认占用 → 开跑 → 结束/取消。

四条约定的落点：
1. 幂等——入队按「请求键或自然键（批次+商品+渠道+窗口）」去重，确认按
   分配日志的请求键去重；重试返回已有结果，不重复占用；
2. 并发——确认走 BEGIN IMMEDIATE 写事务，先校修订号再落占用，
   后到者 409，不抢走已有占用；占用互斥由 (sku, channel, date) 主键兜底；
3. 中断恢复——确认分两段：先写分配日志（pending），再单事务应用并置 done；
   崩溃后 recover 把 pending 日志重放（修订号已变则置 aborted），
   并修复孤儿占用与缺失占用；
4. 状态机——queued/confirmed 是可重排的草案，running 锁定，finished/cancelled
   释放占用；非法迁移一律 409，不悄悄改状态。
"""
from __future__ import annotations

import datetime as dt
import json
import uuid

from ... import config, reasons
from ...errors import Conflict, NotFound, Unprocessable
from ..m2_price import coerce
from . import planner, power
from .service import series_options

ACTIVE_STATUSES = ("queued", "confirmed", "running")
STATUS_LABELS = {
    "queued": "排队中",
    "confirmed": "已确认",
    "running": "已开跑",
    "finished": "已结束",
    "cancelled": "已取消",
}
# 可重排的草案：queued/confirmed；已锁定：running
REARRANGEABLE = ("queued", "confirmed")
MAX_WINDOW_DAYS = power.MAX_EXPERIMENT_DAYS


def now_iso() -> str:
    return dt.datetime.now(config.UTC).isoformat(timespec="seconds")


def _bad(code: str, message: str, detail: dict | None = None):
    return Unprocessable(message, code=code, detail=detail or {})


def _conflict(code: str, message: str, detail: dict | None = None):
    return Conflict(message, code=code, detail=detail or {})


def _dates_of(experiment: dict) -> list[str]:
    start = dt.date.fromisoformat(experiment["start_date"])
    end = dt.date.fromisoformat(experiment["end_date"])
    return [day.isoformat() for day in power.dates_between(start, end)]


def _public(row: dict, occupancy_days: int | None = None) -> dict:
    out = dict(row)
    out["status_label"] = STATUS_LABELS.get(row["status"], row["status"])
    out["locked"] = 1 if row["status"] == "running" else 0
    out["rearrangeable"] = 1 if row["status"] in REARRANGEABLE else 0
    if occupancy_days is not None:
        out["occupancy_days"] = occupancy_days
    return out


# ---------- 排队 ----------

def enqueue(store, params: dict) -> dict:
    """新建实验并入队；同一请求键或同一自然键重试直接返回已有实验。"""
    batch_id = (params.get("batch_id") or "").strip()
    if not batch_id or store.get_batch(batch_id) is None:
        raise _bad("D_NO_BATCH", "入队前先有批次", {"batch_id": batch_id})
    sku = (params.get("sku") or "").strip()
    channel = (params.get("channel") or "").strip()
    options = {(item["sku"], item["channel"])
               for item in series_options(store, batch_id)}
    if (sku, channel) not in options:
        raise _bad("D_UNKNOWN_SERIES", f"本批次没有 {sku}/{channel} 的日销记录",
                   {"sku": sku, "channel": channel})
    start = coerce.to_date((params.get("start_date") or "").strip())
    end = coerce.to_date((params.get("end_date") or "").strip())
    if start is None or end is None or end <= start:
        raise _bad("D_EXP_WINDOW",
                   f"实验窗不合法：{params.get('start_date')} → "
                   f"{params.get('end_date')}",
                   {"accepts": "yyyy-mm-dd，结束日晚于开始日"})
    if (end - start).days > MAX_WINDOW_DAYS:
        raise _bad("D_EXP_WINDOW", f"实验窗超过 {MAX_WINDOW_DAYS} 天",
                   {"days": (end - start).days})

    request_id = (params.get("request_id") or "").strip()
    natural = f"nat:{batch_id}:{sku}:{channel}:{start.isoformat()}:{end.isoformat()}"
    if request_id:
        # 显式请求键：严格幂等，同键返回同一个实验（无论状态）
        existing = store.experiment_by_request(request_id)
        if existing is not None:
            return {"experiment": _public(existing), "created": False,
                    "plan": planner.plan(store)}
        final_key = request_id
    else:
        # 自然键：只对活跃实验幂等；已取消/已结束的同窗口实验不挡新建，
        # 用 #N 后缀派生新键，重试仍能命中同一个活跃实验
        final_key = natural
        suffix = 1
        while True:
            existing = store.experiment_by_request(final_key)
            if existing is None:
                break
            if existing["status"] in ACTIVE_STATUSES:
                return {"experiment": _public(existing), "created": False,
                        "plan": planner.plan(store)}
            suffix += 1
            final_key = f"{natural}#{suffix}"

    experiment_id = f"X{dt.datetime.now(config.UTC).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    stamp = now_iso()
    row = {
        "experiment_id": experiment_id,
        "request_id": final_key,
        "batch_id": batch_id,
        "sku": sku,
        "channel": channel,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "target_lift": (params.get("target_lift") or "").strip(),
        "status": "queued",
        "created_at": stamp,
        "updated_at": stamp,
        "note": (params.get("note") or "").strip(),
    }
    created = True
    with store.write() as conn:
        # 并发入队：唯一索引兜底，撞键说明对方先落库，直接复用对方那行
        clash = conn.execute("SELECT * FROM experiments WHERE request_id=?",
                             (row["request_id"],)).fetchone()
        if clash is not None:
            row = dict(clash)
            experiment_id = row["experiment_id"]
            created = False
        else:
            store.experiment_insert(row, conn)
            store.event_add(experiment_id, "enqueue",
                            f"{sku}/{channel} {row['start_date']}→{row['end_date']}",
                            stamp, conn)
    return {"experiment": _public(store.experiment_get(experiment_id)),
            "created": created, "plan": planner.plan(store)}


# ---------- 确认占用 ----------

def _apply_plan(store, base_revision: int) -> dict:
    """单事务应用当前编排：校修订号 → 重算 → 重排占用 → 记事件。

    全程持有写锁，修订号与占用快照一致；任何一步冲突都整体回滚。
    占用重排先整体释放草案的旧占用再逐实验插入——新方案可能把 A 的
    对照换给 B，边删边插会撞上还没释放的旧行。
    """
    with store.write() as conn:
        revision = store.experiment_revision(conn)
        if revision != base_revision:
            raise _conflict("D_PLAN_STALE",
                            f"编排已变化（基于 {base_revision}，当前 {revision}），"
                            f"重新规划后再确认",
                            {"base_revision": base_revision, "current": revision})
        proposal = planner.plan(store, conn)
        stamp = now_iso()
        movable = [row for row in proposal["rows"] if not row["locked"]]
        for row in movable:
            conn.execute("DELETE FROM control_occupancy WHERE experiment_id=?",
                         (row["experiment_id"],))
        applied: list[dict] = []
        for row in movable:
            experiment = store.experiment_get(row["experiment_id"], conn)
            if experiment is None:
                continue
            if not row["feasible"]:
                # 确认失败：回退为排队，清空持有与计划
                store.experiment_update(experiment["experiment_id"], {
                    "status": "queued",
                    "control_sku": "",
                    "control_channel": "",
                    "planned_sku": "",
                    "planned_channel": "",
                    "infeasible_code": row["code"],
                    "infeasible_detail": row["detail"],
                    "updated_at": stamp,
                }, conn)
                store.event_add(experiment["experiment_id"], "infeasible",
                                f"{row['code']} {row['detail']}", stamp, conn)
                continue
            dates = [day.isoformat() for day in power.dates_between(
                dt.date.fromisoformat(experiment["start_date"]),
                dt.date.fromisoformat(experiment["end_date"]))]
            for day in dates:
                try:
                    conn.execute(
                        "INSERT INTO control_occupancy (sku, channel, date, "
                        "experiment_id) VALUES (?,?,?,?)",
                        (row["planned_sku"], row["planned_channel"], day,
                         experiment["experiment_id"]))
                except Exception as exc:  # 占用主键冲突：不抢占，整体回滚
                    if "UNIQUE" in str(exc).upper():
                        raise _conflict(
                            "D_OCCUPANCY_HELD",
                            f"{row['planned_sku']} 在实验窗内已被其它实验占用，未抢占",
                            {"experiment_id": experiment["experiment_id"],
                             "control": row["planned_sku"]}) from exc
                    raise
            store.experiment_update(experiment["experiment_id"], {
                "status": "confirmed",
                "control_sku": row["planned_sku"],
                "control_channel": row["planned_channel"],
                "planned_sku": row["planned_sku"],
                "planned_channel": row["planned_channel"],
                "plan_bias": row["planned_bias"],
                "best_bias": row["best_bias"],
                "infeasible_code": "",
                "infeasible_detail": "",
                "updated_at": stamp,
            }, conn)
            store.event_add(experiment["experiment_id"], "confirm",
                            f"占用 {row['planned_sku']} × {len(dates)} 天", stamp,
                            conn)
            applied.append({"experiment_id": experiment["experiment_id"],
                            "control_sku": row["planned_sku"], "days": len(dates)})
        return {"revision": store.experiment_revision(conn),
                "applied": applied, "plan": proposal}


def _journal_done(store, request_id: str, base_revision: int, result: dict) -> None:
    payload = {"base_revision": base_revision, "result": result}
    with store.write() as conn:
        conn.execute(
            "UPDATE allocation_journal SET payload=?, status='done', applied_at=? "
            "WHERE request_id=?",
            (json.dumps(payload, ensure_ascii=False), now_iso(), request_id))


def _journal_abort(store, request_id: str) -> None:
    """标中止只认 pending：同键的并发确认可能已置 done，不能回覆盖。"""
    with store.write() as conn:
        conn.execute(
            "UPDATE allocation_journal SET status='aborted', applied_at=? "
            "WHERE request_id=? AND status='pending'", (now_iso(), request_id))


def confirm(store, params: dict) -> dict:
    """确认当前编排：幂等（请求键）+ 两段写（日志 pending → 应用 done）。"""
    request_id = (params.get("request_id") or "").strip()
    try:
        base_revision = int(params.get("base_revision"))
    except (TypeError, ValueError):
        raise _bad("D_BAD_PARAM", "base_revision 必须是整数",
                   {"field": "base_revision"})
    if not request_id:
        request_id = f"confirm-r{base_revision}"

    journal = store.journal_get(request_id)
    if journal is not None and journal["status"] == "done":
        return {"confirmed": True, "idempotent": True,
                "result": json.loads(journal["payload"])["result"]}
    if journal is not None and journal["status"] == "pending":
        # 上一次确认写到一半：就地重放，等价于一次恢复
        stored_revision = json.loads(journal["payload"])["base_revision"]
        try:
            result = _apply_plan(store, stored_revision)
        except Conflict:
            _journal_abort(store, request_id)
            raise
        _journal_done(store, request_id, stored_revision, result)
        return {"confirmed": True, "idempotent": True, "result": result}

    result_holder: dict = {}
    try:
        with store.write() as conn:
            if store.journal_get(request_id, conn) is None:
                store.journal_insert({
                    "request_id": request_id, "kind": "confirm",
                    "payload": json.dumps({"base_revision": base_revision},
                                          ensure_ascii=False),
                    "status": "pending", "created_at": now_iso()}, conn)
        result = _apply_plan(store, base_revision)
        result_holder["result"] = result
    except Conflict:
        _journal_abort(store, request_id)
        # 同键的并发确认可能已先一步完成：那次成功就是本次的幂等结果
        current = store.journal_get(request_id)
        if current is not None and current["status"] == "done":
            return {"confirmed": True, "idempotent": True,
                    "result": json.loads(current["payload"])["result"]}
        raise
    _journal_done(store, request_id, base_revision, result_holder["result"])
    return {"confirmed": True, "idempotent": False,
            "result": result_holder["result"]}


# ---------- 状态迁移 ----------

def _transition(store, experiment_id: str, expect: tuple[str, ...],
                target: str, action: str) -> dict:
    with store.write() as conn:
        experiment = store.experiment_get(experiment_id, conn)
        if experiment is None:
            raise NotFound(f"实验 {experiment_id} 不存在", code="D_EXP_UNKNOWN")
        if experiment["status"] == target:
            return {"experiment": _public(experiment), "changed": False}
        if experiment["status"] not in expect:
            raise _conflict(
                "D_EXP_STATE",
                f"实验当前是「{STATUS_LABELS.get(experiment['status'])}」，"
                f"不能{action}",
                {"experiment_id": experiment_id, "status": experiment["status"]})
        stamp = now_iso()
        fields = {"status": target, "updated_at": stamp}
        detail = ""
        if target == "running":
            fields["launched_at"] = stamp
            fields["control_sku"] = experiment["planned_sku"] or experiment["control_sku"]
            fields["control_channel"] = (experiment["planned_channel"] or
                                         experiment["control_channel"])
            detail = f"锁定对照 {fields['control_sku']}"
            # 开跑前自愈：占用行缺失、与窗口不符或对照不一致，都按锁定对照重建
            held = store.occupancy_of(experiment_id, conn)
            want = set(_dates_of({**experiment, **fields}))
            have = {row["date"] for row in held}
            mismatched = any(
                row["sku"] != fields["control_sku"] or
                row["channel"] != fields["control_channel"] for row in held)
            if have != want or mismatched:
                store.occupancy_replace(experiment_id, fields["control_sku"],
                                        fields["control_channel"], sorted(want), conn)
                detail += f"（占用自愈 {len(want)} 天）"
        elif target in ("finished", "cancelled"):
            released = store.occupancy_delete(experiment_id, conn)
            fields["finished_at" if target == "finished" else "cancelled_at"] = stamp
            detail = f"释放占用 {released} 天"
        store.experiment_update(experiment_id, fields, conn)
        store.event_add(experiment_id, target, detail or action, stamp, conn)
    return {"experiment": _public(store.experiment_get(experiment_id)),
            "changed": True}


def launch(store, experiment_id: str) -> dict:
    """开跑：只有已确认（占用在手）的实验能开跑；开跑即锁定，不再参与重排。"""
    return _transition(store, experiment_id, ("confirmed",), "running", "开跑")


def cancel(store, experiment_id: str) -> dict:
    """取消：任何在途状态都可取消，占用立即释放；已结束的不能再取消。"""
    return _transition(store, experiment_id, ACTIVE_STATUSES, "cancelled", "取消")


def finish(store, experiment_id: str) -> dict:
    """结束：只有开跑中的实验能结束；占用释放，对照回到可用池。"""
    return _transition(store, experiment_id, ("running",), "finished", "结束")


# ---------- 恢复 ----------

def recover(store) -> dict:
    """中断恢复：重放未完成的分配日志，清理孤儿占用，补齐缺失占用。

    返回修复清单；没有可修的就是全 0。启动时与手动触发共用这一套。
    """
    report = {"replayed": [], "aborted": [], "released": [], "rebuilt": []}

    for journal in store.journal_pending():
        request_id = journal["request_id"]
        try:
            base_revision = json.loads(journal["payload"])["base_revision"]
            _apply_plan(store, base_revision)
        except Conflict:
            _journal_abort(store, request_id)
            report["aborted"].append(request_id)
        else:
            store.journal_mark(request_id, "done", now_iso())
            report["replayed"].append(request_id)

    experiments = {row["experiment_id"]: row
                   for row in store.experiments(None)}
    held_by: dict[str, list[dict]] = {}
    for row in store.occupancy_rows():
        held_by.setdefault(row["experiment_id"], []).append(row)

    with store.write() as conn:
        stamp = now_iso()
        for experiment_id, rows in held_by.items():
            experiment = experiments.get(experiment_id)
            if experiment is None or experiment["status"] in ("finished", "cancelled"):
                conn.execute("DELETE FROM control_occupancy WHERE experiment_id=?",
                             (experiment_id,))
                store.event_add(experiment_id, "recover_release",
                                f"释放孤儿占用 {len(rows)} 天", stamp, conn)
                report["released"].append(experiment_id)
        for experiment in experiments.values():
            if experiment["status"] not in ("confirmed", "running"):
                continue
            if not experiment["control_sku"]:
                continue
            want = set(_dates_of(experiment))
            held = held_by.get(experiment["experiment_id"], [])
            have = {row["date"] for row in held}
            mismatched = any(
                row["sku"] != experiment["control_sku"] or
                row["channel"] != (experiment["control_channel"] or
                                   experiment["channel"]) for row in held)
            if have != want or mismatched:
                store.occupancy_replace(experiment["experiment_id"],
                                        experiment["control_sku"],
                                        experiment["control_channel"] or
                                        experiment["channel"],
                                        sorted(want), conn)
                store.event_add(experiment["experiment_id"], "recover_rebuild",
                                f"按锁定对照重建占用 {len(want)} 天", stamp, conn)
                report["rebuilt"].append(experiment["experiment_id"])
    report["ok"] = 1
    return report


# ---------- 看板 ----------

def board(store) -> dict:
    """队列 + 当前编排 + 修订号：页面与 API 共用。"""
    experiments = store.experiments(None)
    occupancy = store.occupancy_rows()
    days_of: dict[str, int] = {}
    for row in occupancy:
        days_of[row["experiment_id"]] = days_of.get(row["experiment_id"], 0) + 1
    rows = [_public(exp, days_of.get(exp["experiment_id"], 0))
            for exp in experiments]
    proposal = planner.plan(store)
    return {
        "revision": proposal["revision"],
        "experiments": rows,
        "plan": proposal,
        "status_labels": STATUS_LABELS,
        "events": store.experiment_events(20),
    }
