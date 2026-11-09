"""
Incremental Indexer & Manifest Orchestrator for Unified RAG
- Manages SHA-256 content hashing to index only added, updated, or modified files
- Persists state in data/manifests/index_manifest.json with atomic writes for crash resilience
- Synchronizes Document RAG and Codebase RAG into isolated ChromaDB collections
- Implements versioned generation-based chunk replacement:
  * Generates unique generation identifiers per indexing operation
  * Versions chunk IDs to avoid overwriting during staging
  * Verifies storage before switching manifest
  * Preserves previous valid generation until commit succeeds
  * Deletes old generation chunks only post-commit
  * Rolls back newly inserted chunks on any failure prior to commit
"""

from datetime import datetime
import hashlib
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from rag import logger
from rag.config import (
    CODEBASE_COLLECTION_NAME,
    DOCUMENT_COLLECTION_NAME,
    DOCUMENTS_DIR,
    INDEX_MANIFEST_PATH,
    PROJECT_ROOT,
    is_path_ignored,
    is_safe_path,
)
from rag.document_pipeline import DocumentPipeline
from rag.codebase_pipeline import CodebasePipeline
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStoreManager


class IncrementalIndexer:
    """
    Orchestrates differential indexing across documentation and codebase.
    Employs versioned generation replacement to ensure zero data loss on failure,
    crash-safe manifest updates, and clean deletion of obsolete chunks.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        manifest_path: str = INDEX_MANIFEST_PATH,
        base_dir: str = PROJECT_ROOT,
    ):
        self.base_dir = os.path.abspath(base_dir)
        self.manifest_path = os.path.abspath(manifest_path)

        self.vector_store = vector_store or VectorStoreManager()
        self.embedding_service = embedding_service or EmbeddingService()

        self.doc_pipeline = DocumentPipeline(base_dir=self.base_dir)
        self.code_pipeline = CodebasePipeline(base_dir=self.base_dir)

        self.manifest: Dict[str, Any] = self.load_manifest()

    # ── Manifest Management ──────────────────────────────────────────────────

    def load_manifest(self) -> Dict[str, Any]:
        """Load manifest from disk or return clean initial structure."""
        default_manifest: Dict[str, Any] = {
            "version": "1.0",
            "last_updated": None,
            "documents": {},
            "codebase": {},
        }

        if not os.path.exists(self.manifest_path):
            return default_manifest

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("version", "1.0")
                    data.setdefault("last_updated", None)
                    data.setdefault("documents", {})
                    data.setdefault("codebase", {})
                    return data
        except Exception as e:
            logger.warning(
                f"Manifest at '{self.manifest_path}' was unreadable or corrupted: {e}. Starting fresh."
            )
            # Create backup of corrupted manifest
            try:
                backup_path = f"{self.manifest_path}.bak"
                shutil.copy2(self.manifest_path, backup_path)
            except Exception:
                pass

        return default_manifest

    def save_manifest(self) -> None:
        """Atomically persist manifest to disk to prevent corruption on interruption."""
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        self.manifest["last_updated"] = datetime.now().isoformat()

        tmp_path = f"{self.manifest_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.manifest_path)
        except Exception as e:
            logger.error(f"Failed to persist manifest to '{self.manifest_path}': {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    # ── Content Hashing & Inspection ─────────────────────────────────────────

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute SHA-256 checksum of a file in 64KB blocks."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except OSError as e:
            logger.warning(f"Failed to compute hash for '{file_path}': {e}")
            return ""

    def _get_relative_path(self, file_path: str) -> str:
        """Compute relative path with forward slashes."""
        try:
            rel = os.path.relpath(file_path, self.base_dir)
            return rel.replace("\\", "/")
        except ValueError:
            return os.path.basename(file_path)

    # ── Versioned Generation-Based Chunk Replacement ─────────────────────────

    def _safe_replace_file_entry(
        self,
        collection_name: str,
        manifest_category: str,
        rel_path: str,
        chunks: List[Dict[str, Any]],
        cur_hash: str,
        mtime: float,
        size_bytes: int,
        old_ids: Optional[List[str]] = None,
        old_rel_to_prune: Optional[str] = None,
    ) -> List[str]:
        """
        Executes versioned generation-based replacement for a file entry:
        1. Assigns a unique generation ID to this indexing operation.
        2. Versions chunk IDs so new chunks never collide with or overwrite old chunks during staging.
        3. Embeds and stores new chunks in ChromaDB.
        4. Verifies storage in ChromaDB.
        5. Commits manifest to disk with the new generation.
        6. Only after successful commit, deletes old-generation chunks.
        7. If any step fails before commit, rolls back newly inserted chunks leaving old chunks intact.
        """
        old_ids = old_ids or []
        cat_manifest = self.manifest.setdefault(manifest_category, {})
        prev_entry = cat_manifest.get(rel_path)

        # Case: file now produces no chunks (e.g. emptied or comments only)
        if not chunks:
            cat_manifest[rel_path] = {
                "sha256": cur_hash,
                "mtime": mtime,
                "size_bytes": size_bytes,
                "generation_id": None,
                "chunk_ids": [],
                "chunk_count": 0,
                "indexed_at": datetime.now().isoformat(),
            }
            if old_rel_to_prune:
                cat_manifest.pop(old_rel_to_prune, None)

            self.save_manifest()
            if old_ids:
                try:
                    self.vector_store.delete_by_ids(collection_name, old_ids)
                except Exception as e:
                    logger.warning(f"Failed to delete old chunks after emptying file '{rel_path}': {e}")
            return []

        # Step 1 & 2: Assign unique generation ID and version all chunk IDs
        gen_id = uuid.uuid4().hex[:8]
        new_ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for c in chunks:
            raw_id = c["id"]
            # Strip previous generation tag if re-processing (e.g. :ga1b2c3d4 at end of id)
            clean_base = re.sub(r":g[0-9a-fA-F]{8}$", "", raw_id)
            v_id = f"{clean_base}:g{gen_id}"
            c["id"] = v_id
            c["metadata"]["generation_id"] = gen_id

            new_ids.append(v_id)
            texts.append(c["text"])
            metadatas.append(c["metadata"])

        # Step 3: Generate embeddings and insert into vector store
        try:
            if hasattr(self.embedding_service, "embed_batch"):
                embeddings = self.embedding_service.embed_batch(texts)
            else:
                embeddings = self.embedding_service.embed_documents(texts)
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
                )

            inserted_count = self.vector_store.add_documents(
                collection_name=collection_name,
                ids=new_ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            if inserted_count != len(new_ids):
                raise RuntimeError(
                    f"Vector store upsert count mismatch: expected {len(new_ids)}, inserted {inserted_count}"
                )

            # Step 4: Storage verification
            col = self.vector_store.get_collection(collection_name)
            stored = col.get(ids=new_ids)
            stored_ids = stored.get("ids", []) if stored else []
            if len(stored_ids) != len(new_ids):
                raise RuntimeError(
                    f"Storage verification failure: expected {len(new_ids)} chunks, found {len(stored_ids)}"
                )

        except Exception as staging_exc:
            # Staging failed: rollback any newly inserted chunks, preserving old generation intact
            try:
                self.vector_store.delete_by_ids(collection_name, new_ids)
            except Exception:
                pass
            raise staging_exc

        # Step 5: Commit manifest to the new generation
        cat_manifest[rel_path] = {
            "sha256": cur_hash,
            "mtime": mtime,
            "size_bytes": size_bytes,
            "generation_id": gen_id,
            "chunk_ids": new_ids,
            "chunk_count": len(new_ids),
            "indexed_at": datetime.now().isoformat(),
        }
        if old_rel_to_prune:
            cat_manifest.pop(old_rel_to_prune, None)

        try:
            self.save_manifest()
        except Exception as commit_exc:
            # Manifest commit failed: rollback in-memory manifest and purge newly added chunks
            if prev_entry is not None:
                cat_manifest[rel_path] = prev_entry
            else:
                cat_manifest.pop(rel_path, None)

            try:
                self.vector_store.delete_by_ids(collection_name, new_ids)
            except Exception:
                pass
            raise commit_exc

        # Step 6: Post-commit cleanup of previous generation chunks
        if old_ids:
            try:
                self.vector_store.delete_by_ids(collection_name, old_ids)
            except Exception as cleanup_err:
                logger.warning(
                    f"Post-commit cleanup of old generation chunks failed for '{rel_path}': {cleanup_err}"
                )

        return new_ids

    # ── Indexing Workflows ───────────────────────────────────────────────────

    def index_documents(self, target_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Scan and incrementally index unstructured documents into DOCUMENT_COLLECTION_NAME.
        Tracks new, modified, unchanged, renamed, and deleted files with versioned replacement.
        """
        if not self.vector_store.is_initialized():
            self.vector_store.initialize()

        scan_dir = target_dir or DOCUMENTS_DIR
        scan_dir = os.path.abspath(scan_dir)

        logger.info(f"Starting Document RAG indexing in '{scan_dir}'...")
        summary = {
            "indexed": 0,
            "updated": 0,
            "skipped": 0,
            "deleted": 0,
            "errors": 0,
            "total_chunks": 0,
        }

        # 1. Discover all candidate files
        current_files: Dict[str, str] = {}
        if os.path.isdir(scan_dir):
            for root, dirs, files in os.walk(scan_dir):
                dirs[:] = [
                    d for d in dirs
                    if not is_path_ignored(os.path.join(root, d), self.base_dir)
                ]
                for file_name in files:
                    abs_path = os.path.join(root, file_name)
                    if self.doc_pipeline.can_process(abs_path):
                        rel_path = self._get_relative_path(abs_path)
                        current_files[rel_path] = abs_path

        # 2. Reconcile with manifest
        doc_manifest = self.manifest.setdefault("documents", {})

        manifest_rel_paths = set(doc_manifest.keys())
        current_rel_paths = set(current_files.keys())
        missing_rel_paths = manifest_rel_paths - current_rel_paths

        missing_by_hash: Dict[str, str] = {}
        for m_path in missing_rel_paths:
            h = doc_manifest[m_path].get("sha256")
            if h:
                missing_by_hash[h] = m_path

        # Process each current file
        for rel_path, abs_path in current_files.items():
            try:
                cur_hash = self.compute_file_hash(abs_path)
                if not cur_hash:
                    summary["errors"] += 1
                    continue

                mtime = os.path.getmtime(abs_path)
                size_bytes = os.path.getsize(abs_path)

                # Case A: Existing file in manifest
                if rel_path in doc_manifest:
                    entry = doc_manifest[rel_path]
                    if entry.get("sha256") == cur_hash:
                        # Unchanged - skip completely
                        summary["skipped"] += 1
                        summary["total_chunks"] += len(entry.get("chunk_ids", []))
                        continue

                    # File modified: versioned replacement
                    old_ids = entry.get("chunk_ids", [])
                    chunks = self.doc_pipeline.process_file(abs_path)

                    new_chunk_ids = self._safe_replace_file_entry(
                        collection_name=DOCUMENT_COLLECTION_NAME,
                        manifest_category="documents",
                        rel_path=rel_path,
                        chunks=chunks,
                        cur_hash=cur_hash,
                        mtime=mtime,
                        size_bytes=size_bytes,
                        old_ids=old_ids,
                    )

                    summary["updated"] += 1
                    summary["total_chunks"] += len(new_chunk_ids)

                else:
                    # Check for rename: same hash was recorded under a missing path
                    old_ids = []
                    old_rel_to_prune = None
                    if cur_hash in missing_by_hash:
                        old_rel = missing_by_hash.pop(cur_hash)
                        missing_rel_paths.discard(old_rel)
                        old_entry = doc_manifest.get(old_rel, {})
                        old_ids = old_entry.get("chunk_ids", [])
                        old_rel_to_prune = old_rel
                        logger.info(f"Detected document rename: '{old_rel}' -> '{rel_path}'")

                    # Process file
                    chunks = self.doc_pipeline.process_file(abs_path)
                    new_chunk_ids = self._safe_replace_file_entry(
                        collection_name=DOCUMENT_COLLECTION_NAME,
                        manifest_category="documents",
                        rel_path=rel_path,
                        chunks=chunks,
                        cur_hash=cur_hash,
                        mtime=mtime,
                        size_bytes=size_bytes,
                        old_ids=old_ids,
                        old_rel_to_prune=old_rel_to_prune,
                    )

                    summary["indexed"] += 1
                    summary["total_chunks"] += len(new_chunk_ids)

            except Exception as e:
                logger.error(f"Error indexing document '{abs_path}': {e}")
                summary["errors"] += 1

        # Purge remaining missing files (deleted)
        for dead_rel in missing_rel_paths:
            try:
                dead_entry = doc_manifest.pop(dead_rel, {})
                dead_ids = dead_entry.get("chunk_ids", [])
                if dead_ids:
                    self.vector_store.delete_by_ids(DOCUMENT_COLLECTION_NAME, dead_ids)
                summary["deleted"] += 1
                logger.info(f"Pruned deleted document from index: '{dead_rel}'")
            except Exception as e:
                logger.error(f"Error purging deleted document '{dead_rel}': {e}")
                summary["errors"] += 1

        if missing_rel_paths:
            self.save_manifest()

        logger.info(f"Document RAG indexing finished: {summary}")
        return summary

    def index_codebase(self, root_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Scan and incrementally index codebase and config files into CODEBASE_COLLECTION_NAME.
        Uses Python AST parsing for .py and safe line chunking for configs with versioned replacement.
        """
        if not self.vector_store.is_initialized():
            self.vector_store.initialize()

        scan_root = root_dir or self.base_dir
        scan_root = os.path.abspath(scan_root)

        logger.info(f"Starting Codebase RAG indexing in '{scan_root}'...")
        summary = {
            "indexed": 0,
            "updated": 0,
            "skipped": 0,
            "deleted": 0,
            "errors": 0,
            "total_chunks": 0,
        }

        # 1. Discover all candidate files
        current_files: Dict[str, str] = {}
        if os.path.isdir(scan_root):
            for root, dirs, files in os.walk(scan_root):
                dirs[:] = [
                    d for d in dirs
                    if not is_path_ignored(os.path.join(root, d), self.base_dir)
                ]
                for file_name in files:
                    abs_path = os.path.join(root, file_name)
                    if self.code_pipeline.can_process(abs_path):
                        rel_path = self._get_relative_path(abs_path)
                        current_files[rel_path] = abs_path

        # 2. Reconcile with codebase manifest
        code_manifest = self.manifest.setdefault("codebase", {})

        manifest_rel_paths = set(code_manifest.keys())
        current_rel_paths = set(current_files.keys())
        missing_rel_paths = manifest_rel_paths - current_rel_paths

        missing_by_hash: Dict[str, str] = {}
        for m_path in missing_rel_paths:
            h = code_manifest[m_path].get("sha256")
            if h:
                missing_by_hash[h] = m_path

        # Process each current code file
        for rel_path, abs_path in current_files.items():
            try:
                cur_hash = self.compute_file_hash(abs_path)
                if not cur_hash:
                    summary["errors"] += 1
                    continue

                mtime = os.path.getmtime(abs_path)
                size_bytes = os.path.getsize(abs_path)

                # Case A: Existing file
                if rel_path in code_manifest:
                    entry = code_manifest[rel_path]
                    if entry.get("sha256") == cur_hash:
                        # Unchanged - skip completely
                        summary["skipped"] += 1
                        summary["total_chunks"] += len(entry.get("chunk_ids", []))
                        continue

                    # File modified: versioned replacement
                    old_ids = entry.get("chunk_ids", [])
                    chunks = self.code_pipeline.process_file(abs_path)

                    new_chunk_ids = self._safe_replace_file_entry(
                        collection_name=CODEBASE_COLLECTION_NAME,
                        manifest_category="codebase",
                        rel_path=rel_path,
                        chunks=chunks,
                        cur_hash=cur_hash,
                        mtime=mtime,
                        size_bytes=size_bytes,
                        old_ids=old_ids,
                    )

                    summary["updated"] += 1
                    summary["total_chunks"] += len(new_chunk_ids)

                else:
                    # Check for rename
                    old_ids = []
                    old_rel_to_prune = None
                    if cur_hash in missing_by_hash:
                        old_rel = missing_by_hash.pop(cur_hash)
                        missing_rel_paths.discard(old_rel)
                        old_entry = code_manifest.get(old_rel, {})
                        old_ids = old_entry.get("chunk_ids", [])
                        old_rel_to_prune = old_rel
                        logger.info(f"Detected codebase rename: '{old_rel}' -> '{rel_path}'")

                    # New file
                    chunks = self.code_pipeline.process_file(abs_path)
                    new_chunk_ids = self._safe_replace_file_entry(
                        collection_name=CODEBASE_COLLECTION_NAME,
                        manifest_category="codebase",
                        rel_path=rel_path,
                        chunks=chunks,
                        cur_hash=cur_hash,
                        mtime=mtime,
                        size_bytes=size_bytes,
                        old_ids=old_ids,
                        old_rel_to_prune=old_rel_to_prune,
                    )

                    summary["indexed"] += 1
                    summary["total_chunks"] += len(new_chunk_ids)

            except Exception as e:
                logger.error(f"Error indexing code file '{abs_path}': {e}")
                summary["errors"] += 1

        # Purge remaining missing files
        for dead_rel in missing_rel_paths:
            try:
                dead_entry = code_manifest.pop(dead_rel, {})
                dead_ids = dead_entry.get("chunk_ids", [])
                if dead_ids:
                    self.vector_store.delete_by_ids(CODEBASE_COLLECTION_NAME, dead_ids)
                summary["deleted"] += 1
                logger.info(f"Pruned deleted code file from index: '{dead_rel}'")
            except Exception as e:
                logger.error(f"Error purging deleted code file '{dead_rel}': {e}")
                summary["errors"] += 1

        if missing_rel_paths:
            self.save_manifest()

        logger.info(f"Codebase RAG indexing finished: {summary}")
        return summary

    def clean_orphaned_generations(
        self, collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Safely identifies and removes orphaned, stale, or uncommitted chunk generations from ChromaDB.
        - An active chunk is defined as any chunk ID currently registered in the committed manifest.
        - Chunks belonging to the current active generation of any file are NEVER deleted.
        - Orphaned chunks (from failed staging, previous generations post-crash, or deleted files)
          are purged in batches.
        - Cleanup failure logs a warning and does not raise or invalidate successful indexing.
        """
        if not self.vector_store.is_initialized():
            self.vector_store.initialize()

        # Reload manifest to ensure clean, committed state
        self.manifest = self.load_manifest()

        summary: Dict[str, Any] = {
            "scanned_collections": [],
            "total_chunks_scanned": 0,
            "active_chunks_retained": 0,
            "orphaned_chunks_deleted": 0,
            "errors": 0,
            "details": {},
        }

        if collection_name:
            target_collections = [collection_name]
        else:
            target_collections = [DOCUMENT_COLLECTION_NAME, CODEBASE_COLLECTION_NAME]

        for col_name in target_collections:
            cat_key = "documents" if col_name == DOCUMENT_COLLECTION_NAME else "codebase"
            cat_manifest = self.manifest.get(cat_key, {})

            # Gather all active chunk IDs registered in the manifest
            active_chunk_ids: Set[str] = set()
            if isinstance(cat_manifest, dict):
                for file_entry in cat_manifest.values():
                    if isinstance(file_entry, dict):
                        active_chunk_ids.update(file_entry.get("chunk_ids", []))

            col_detail = {
                "scanned": 0,
                "active": 0,
                "deleted": 0,
                "errors": 0,
            }
            summary["scanned_collections"].append(col_name)

            try:
                col = self.vector_store.get_collection(col_name)
                total_in_col = col.count()
                if total_in_col == 0:
                    summary["details"][col_name] = col_detail
                    continue

                # Fetch all existing chunk IDs from collection
                stored = col.get(include=["metadatas"])
                stored_ids = stored.get("ids", []) if stored else []
                col_detail["scanned"] = len(stored_ids)
                summary["total_chunks_scanned"] += len(stored_ids)

                # Identify orphaned IDs (any ID not registered in active manifest)
                orphaned_ids: List[str] = [
                    cid for cid in stored_ids if cid not in active_chunk_ids
                ]
                active_count = len(stored_ids) - len(orphaned_ids)
                col_detail["active"] = active_count
                summary["active_chunks_retained"] += active_count

                if orphaned_ids:
                    logger.info(
                        f"Found {len(orphaned_ids)} orphaned chunk(s) in collection '{col_name}'. Purging..."
                    )
                    batch_size = 500
                    for i in range(0, len(orphaned_ids), batch_size):
                        batch = orphaned_ids[i : i + batch_size]
                        try:
                            self.vector_store.delete_by_ids(col_name, batch)
                            col_detail["deleted"] += len(batch)
                            summary["orphaned_chunks_deleted"] += len(batch)
                        except Exception as del_err:
                            logger.warning(
                                f"Failed to delete batch of {len(batch)} orphaned chunks from '{col_name}': {del_err}"
                            )
                            col_detail["errors"] += 1
                            summary["errors"] += 1

            except Exception as col_err:
                logger.warning(f"Error scanning/cleaning collection '{col_name}': {col_err}")
                col_detail["errors"] += 1
                summary["errors"] += 1

            summary["details"][col_name] = col_detail

        logger.info(
            f"Orphan cleanup completed: {summary['orphaned_chunks_deleted']} deleted, "
            f"{summary['active_chunks_retained']} active retained across {summary['scanned_collections']}."
        )
        return summary

    def sync(self) -> Dict[str, Any]:
        """Perform comprehensive differential sync across documents and codebase."""
        doc_summary = self.index_documents()
        code_summary = self.index_codebase()

        # Safely execute orphan cleanup without invalidating successful indexing
        try:
            orphan_summary = self.clean_orphaned_generations()
        except Exception as e:
            logger.warning(f"Orphan cleanup during sync encountered an error: {e}")
            orphan_summary = {"error": str(e), "orphaned_chunks_deleted": 0}

        return {
            "documents": doc_summary,
            "codebase": code_summary,
            "orphan_cleanup": orphan_summary,
            "status": "completed",
            "last_updated": self.manifest.get("last_updated"),
        }

    def get_status(self) -> Dict[str, Any]:
        """Return real-time indexing status and collection statistics."""
        if not self.vector_store.is_initialized():
            self.vector_store.initialize()

        doc_stats = self.vector_store.get_collection_stats(DOCUMENT_COLLECTION_NAME)
        code_stats = self.vector_store.get_collection_stats(CODEBASE_COLLECTION_NAME)

        doc_files = self.manifest.get("documents", {})
        code_files = self.manifest.get("codebase", {})

        return {
            "documents": {
                "files_count": len(doc_files),
                "chunks_count": sum(len(e.get("chunk_ids", [])) for e in doc_files.values()),
                "collection_count": doc_stats.get("count", 0),
            },
            "codebase": {
                "files_count": len(code_files),
                "chunks_count": sum(len(e.get("chunk_ids", [])) for e in code_files.values()),
                "collection_count": code_stats.get("count", 0),
            },
            "manifest_path": self.manifest_path,
            "last_updated": self.manifest.get("last_updated"),
        }
