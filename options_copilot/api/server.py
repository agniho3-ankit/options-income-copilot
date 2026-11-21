"""Start the local UI: build the app from settings and serve it on 127.0.0.1 only."""
from __future__ import annotations

from ..config import Settings
from ..db import Database
from ..eval_harness import SNAPSHOT_ROOT
from ..playbook import load_playbook
from .app import create_app
from .security import check_bind


def build(settings: Settings, port: int):
    eval_path = settings.repo_root / SNAPSHOT_ROOT / "eval.db"
    dist = settings.repo_root / "ui" / "dist"
    return create_app(settings, load_playbook(settings.playbook_path), main_db=Database(settings.db_path),
                      eval_db=Database(eval_path) if eval_path.is_file() else None,
                      dist_dir=dist if dist.is_dir() else None, port=port)


def serve(settings: Settings, host: str, port: int) -> None:
    check_bind(host)   # refuse before anything else starts
    import uvicorn
    app = build(settings, port)
    print(f"Options Income Copilot UI on http://{host}:{port}  (paper only; Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning", server_header=False, proxy_headers=False,
                access_log=False)
