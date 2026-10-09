"""HTTP API package.

**Deliberately empty of imports.** ``app.py`` runs
``create_app()`` at module scope so uvicorn can find it; if this
``__init__`` re-exported ``app``, then any import of
``recsys.api.<anything>`` (a schema, a router, a middleware module)
would trigger the app factory at import time and pull the whole
serving-path dependency tree with it. A test that only needs
``recsys.api.schemas.events`` should not need a database pool.

Import from the module you need directly:

    from recsys.api.app import create_app
    from recsys.api.schemas.events import EventEnvelope
"""
