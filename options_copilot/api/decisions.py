"""The only state-changing routes: approve, reject, defer. Every one is a deliberate human act.

* Approve calls `briefs.approve` and nothing else. The request must echo the brief ID and ticker
  the owner confirmed on screen; a mismatch changes nothing. There is no bulk or automatic path,
  and no other route can approve a brief or write a ticket (tests/test_api.py proves both).
* Reject needs a code from the playbook's `guardrails.reject_codes` and a reason. The reason
  becomes a Reflexion lesson: written by the local LLM when it is free, otherwise by the template.
* Defer keeps the brief pending and moves it to the end of the queue.

The CSRF and Origin checks run in security.LocalOnlyMiddleware before any of these handlers.
Decisions only ever touch the main database.
"""
from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import briefs


class ApproveBody(BaseModel):
    brief_id: str = Field(min_length=1, max_length=64)   # echoed from the confirm step
    ticker: str = Field(min_length=1, max_length=12)
    note: str = Field(default="", max_length=500)


class RejectBody(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=1000)


class DeferBody(BaseModel):
    brief_id: str = Field(min_length=1, max_length=64)


def register(app: FastAPI, st: Any, out: Callable[[Any], Any]) -> None:

    def pending(db, brief_id: str) -> dict:
        b = db.brief(brief_id)
        if b is None:
            raise HTTPException(404, f"no brief {brief_id}")
        if b["status"] != "pending":
            raise HTTPException(409, f"brief {brief_id} is {b['status']}, not pending")
        return b

    @app.post("/api/briefs/{brief_id}/approve")
    def approve(brief_id: str, body: ApproveBody) -> dict:
        with st.db() as db:
            b = pending(db, brief_id)
            if body.brief_id != brief_id or body.ticker.upper() != b["ticker"].upper():
                raise HTTPException(400, "the confirmation does not match this brief; nothing was approved")
            try:
                ticket_id = briefs.approve(db, brief_id, body.note.strip())
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
        return out({"brief_id": brief_id, "status": "approved", "ticket_id": ticket_id, "paper_only": True,
                    "message": "Paper ticket written. Nothing was sent to any broker."})

    @app.post("/api/briefs/{brief_id}/reject")
    def reject(brief_id: str, body: RejectBody) -> dict:
        codes = tuple(st.pb["guardrails.reject_codes"])
        if body.code not in codes:
            raise HTTPException(422, f"unknown reject code {body.code!r}; choose one of: {', '.join(codes)}")
        if not body.reason.strip():
            raise HTTPException(422, "a rejection needs a reason (it becomes a lesson)")
        got_llm = st.llm_lock.acquire(blocking=False)   # never wait behind a long run
        try:
            llm = st.llm_factory() if got_llm and st.llm_factory else None
            with st.db() as db:
                pending(db, brief_id)
                try:
                    lesson = briefs.reject(db, brief_id, body.reason.strip(), llm=llm, store=st.store(),
                                           masker=st.masker, code=body.code, codes=codes)
                except ValueError as exc:
                    raise HTTPException(409, str(exc)) from exc
        finally:
            if got_llm:
                st.llm_lock.release()
        return out({"brief_id": brief_id, "status": "rejected", "code": body.code, "lesson": lesson.lesson,
                    "lesson_by": "local LLM" if llm is not None else "template (the LLM was busy)"})

    @app.post("/api/briefs/{brief_id}/defer")
    def defer(brief_id: str, body: DeferBody) -> dict:
        if body.brief_id != brief_id:
            raise HTTPException(400, "the confirmation does not match this brief; nothing was deferred")
        with st.db() as db:
            pending(db, brief_id)
            db.defer(brief_id)
        return out({"brief_id": brief_id, "status": "pending", "deferred": True,
                    "message": "Deferred: still pending, moved to the end of the queue. Nothing was approved."})
