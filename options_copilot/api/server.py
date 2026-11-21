"""Start the local UI: build the app from settings and serve it on 127.0.0.1 only."""
from __future__ import annotations

from ..config import Settings
from ..db import Database
from ..eval_harness import SNAPSHOT_ROOT
from ..playbook import load_playbook
from .app import create_app
from .security import check_bind


DEMO_DB = "ui_demo.db"
DEMO_CHROMA = "ui_demo_chroma"


def build(settings: Settings, port: int, db: str = "main"):
    """`db="demo"` serves the synthetic data/ui_demo.db (built if missing) instead of data/copilot.db."""
    eval_path = settings.repo_root / SNAPSHOT_ROOT / "eval.db"
    dist = settings.repo_root / "ui" / "dist"
    main_path = settings.db_path
    extra = {}
    if db == "demo":
        from ..ui_demo import build_demo_db
        main_path = settings.data_dir / DEMO_DB
        if not main_path.is_file():
            build_demo_db(main_path)
        # demo decisions must never touch real data: reject lessons go to a separate Chroma store
        # (never data/chroma, which real runs read), and background runs are switched off
        demo_store = settings.data_dir / DEMO_CHROMA

        def store_factory():
            from ..store import Store
            return Store(demo_store)
        extra = {"store_factory": store_factory, "demo": True}
    return create_app(settings, load_playbook(settings.playbook_path), main_db=Database(main_path),
                      eval_db=Database(eval_path) if eval_path.is_file() else None,
                      dist_dir=dist if dist.is_dir() else None, port=port,
                      db_factory=lambda: Database(main_path), **extra)


def serve(settings: Settings, host: str, port: int, db: str = "main") -> None:
    check_bind(host)   # refuse before anything else starts
    from ..multi_agent import privacy
    privacy.configure()   # OpenTelemetry SDK disabled, no exporters, framework telemetry off
    import uvicorn
    app = build(settings, port, db)
    label = " (SYNTHETIC DEMO DATA)" if db == "demo" else ""
    print(f"Options Income Copilot UI on http://{host}:{port}{label}  (paper only; Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning", server_header=False, proxy_headers=False,
                access_log=False)
