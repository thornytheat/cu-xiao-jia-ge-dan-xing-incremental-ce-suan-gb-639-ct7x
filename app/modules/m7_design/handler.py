"""M7 入口：实验设计 API 与设计页。

三个入口共用一套计算：`POST /api/design`（给脚本用）、`GET /api/design`（给只读看板用）、
`GET /design`（页面）。页面参数不合法时把原因就地渲染成一行提示，不返回 5xx。

多实验编排走 `/api/experiments*`：排队 → 确认占用 → 开跑 → 结束/取消，
确认按修订号做乐观并发控制，崩溃恢复用 `POST /api/experiments/recover`。
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ... import web
from . import control, experiments, impute, service

router = APIRouter()

FIELDS = ("sku", "channel", "target_lift", "alpha", "power", "budget_days",
          "start_date", "impute")


class DesignIn(BaseModel):
    sku: str | None = None
    channel: str | None = None
    target_lift: str | float | int | None = None
    alpha: str | float | int | None = None
    power: str | float | int | None = None
    budget_days: str | float | int | None = None
    start_date: str | None = None
    impute: str | None = None

    def as_params(self) -> dict:
        return {name: getattr(self, name) for name in FIELDS}


def _batch_id(request: Request) -> str | None:
    ctx = web.get_ctx(request)
    batch = ctx.current_batch(request.query_params.get("batch_id"))
    return batch["batch_id"] if batch else None


@router.post("/api/design")
def api_design(body: DesignIn, request: Request) -> dict:
    ctx = web.get_ctx(request)
    batch_id = _batch_id(request)
    if batch_id is None:
        from ...errors import Unprocessable
        raise Unprocessable("还没有批次，先导入再设计", code="D_NO_BATCH")
    return service.design(ctx.store, batch_id, body.as_params())


@router.get("/api/design")
def api_design_state(request: Request) -> dict:
    ctx = web.get_ctx(request)
    batch_id = _batch_id(request)
    if batch_id is None:
        return {"batch_id": None, "plan": None, "options": [],
                "defaults": service.default_request(ctx.store, None)}
    params = {name: request.query_params.get(name) for name in FIELDS}
    if not any(value for value in params.values()):
        params = service.default_request(ctx.store, batch_id)
    return {
        "batch_id": batch_id,
        "plan": service.design(ctx.store, batch_id, params),
        "options": service.series_options(ctx.store, batch_id),
        "defaults": params,
    }


@router.get("/design")
def page_design(request: Request):
    ctx = web.get_ctx(request)
    batch_id = _batch_id(request)
    options = service.series_options(ctx.store, batch_id) if batch_id else []
    defaults = service.default_request(ctx.store, batch_id)
    params = {name: request.query_params.get(name) for name in FIELDS}
    submitted = any(value for value in params.values())
    if not submitted:
        params = defaults
    plan = None
    error = None
    from ...errors import PriceliftError
    if batch_id is None:
        error = {"code": "D_NO_BATCH", "message": "还没有批次，先导入再设计"}
    else:
        try:
            plan = service.design(ctx.store, batch_id, params)
        except PriceliftError as exc:
            error = {"code": exc.code, "message": exc.message}
            params = defaults if not submitted else params
    return web.render(
        request, "design.html",
        plan=plan,
        error=error,
        options=options,
        params=params,
        impute_modes=impute.MODES,
        trend_slack=f"{control.TREND_SLACK:.2f}",
        submitted=submitted,
        board=experiments.board(ctx.store),
    )


# ---------- 多实验对照编排 ----------

class ExperimentIn(BaseModel):
    request_id: str | None = None
    batch_id: str | None = None
    sku: str | None = None
    channel: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    target_lift: str | None = None
    note: str | None = None


class ConfirmIn(BaseModel):
    request_id: str | None = None
    base_revision: int | str | None = None


@router.get("/api/experiments")
def api_experiments(request: Request) -> dict:
    return experiments.board(web.get_ctx(request).store)


@router.post("/api/experiments")
def api_experiment_enqueue(body: ExperimentIn, request: Request) -> dict:
    ctx = web.get_ctx(request)
    params = body.model_dump()
    if not params.get("batch_id"):
        params["batch_id"] = _batch_id(request)
    return experiments.enqueue(ctx.store, params)


@router.post("/api/experiments/confirm")
def api_experiments_confirm(body: ConfirmIn, request: Request) -> dict:
    return experiments.confirm(web.get_ctx(request).store, body.model_dump())


@router.post("/api/experiments/recover")
def api_experiments_recover(request: Request) -> dict:
    return experiments.recover(web.get_ctx(request).store)


@router.post("/api/experiments/{experiment_id}/launch")
def api_experiment_launch(experiment_id: str, request: Request) -> dict:
    return experiments.launch(web.get_ctx(request).store, experiment_id)


@router.post("/api/experiments/{experiment_id}/cancel")
def api_experiment_cancel(experiment_id: str, request: Request) -> dict:
    return experiments.cancel(web.get_ctx(request).store, experiment_id)


@router.post("/api/experiments/{experiment_id}/finish")
def api_experiment_finish(experiment_id: str, request: Request) -> dict:
    return experiments.finish(web.get_ctx(request).store, experiment_id)
