"""
Session lifecycle manager.

Each session gets its own Chroma in-memory collection, identified by a UUID.
- persist_directory is NEVER set (CLAUDE.md hard rule).
- Sessions auto-destroy after SESSION_TIMEOUT_MINUTES of inactivity.
- All create/destroy events are logged.
- A session from one UUID never touches another session's collection.
"""
import logging
import os
import threading
import time

import chromadb

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = int(os.getenv("SESSION_TIMEOUT_MINUTES", "120")) * 60


class SessionManager:
    """
    Thread-safe manager of per-session Chroma in-memory collections.

    Usage:
        mgr = SessionManager()
        mgr.create_session("uuid-abc")
        col = mgr.get_collection("uuid-abc")   # chromadb Collection
        mgr.destroy_session("uuid-abc")
    """

    def __init__(self) -> None:
        # Single in-memory Chroma client — no persist_directory.
        self._client = chromadb.Client()
        self._collections: dict[str, chromadb.Collection] = {}
        self._last_active: dict[str, float] = {}
        self._lock = threading.Lock()
        self._start_reaper()

    # ── Public API ─────────────────────────────────────────────────────────────

    def create_session(self, session_id: str) -> None:
        """Create a new in-memory Chroma collection for this session."""
        with self._lock:
            if session_id in self._collections:
                logger.warning("Session already exists: %s — reusing", session_id)
                return
            collection = self._client.create_collection(
                name=f"session_{session_id}",
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[session_id] = collection
            self._last_active[session_id] = time.monotonic()
            logger.info("Session created: %s", session_id)

    def get_collection(self, session_id: str) -> chromadb.Collection:
        """
        Return the Chroma collection for this session.
        Updates the last-active timestamp. Raises KeyError for unknown sessions.
        """
        with self._lock:
            if session_id not in self._collections:
                raise KeyError(f"No active session: {session_id}")
            self._last_active[session_id] = time.monotonic()
            return self._collections[session_id]

    def destroy_session(self, session_id: str) -> None:
        """Destroy a session's collection and remove it from the registry."""
        with self._lock:
            self._destroy_locked(session_id)

    def active_sessions(self) -> list[str]:
        """Return list of currently active session IDs."""
        with self._lock:
            return list(self._collections.keys())

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _destroy_locked(self, session_id: str) -> None:
        """Must be called with self._lock held."""
        if session_id not in self._collections:
            return
        try:
            self._client.delete_collection(f"session_{session_id}")
        except Exception:
            logger.exception("Error deleting Chroma collection for session %s", session_id)
        del self._collections[session_id]
        del self._last_active[session_id]
        logger.info("Session destroyed: %s", session_id)

    def _reap_expired(self) -> None:
        """Destroy all sessions that have been inactive for longer than the timeout."""
        now = time.monotonic()
        with self._lock:
            expired = [
                sid for sid, last in self._last_active.items()
                if now - last >= _TIMEOUT_SECONDS
            ]
            for sid in expired:
                logger.info(
                    "Session %s expired after %.0f minutes of inactivity — destroying",
                    sid, _TIMEOUT_SECONDS / 60,
                )
                self._destroy_locked(sid)

    def _start_reaper(self) -> None:
        """Start a background daemon thread that checks for expired sessions every minute."""
        def loop():
            while True:
                time.sleep(60)
                try:
                    self._reap_expired()
                except Exception:
                    logger.exception("Reaper error")

        t = threading.Thread(target=loop, daemon=True, name="session-reaper")
        t.start()
