"""Ingestion pipeline: published docs -> citable, chunked JSONL.

Stages:
    1. manifest — parse llms.txt -> in-scope corpus URL list
    2. fetch    — download each page as .md into a local cache
    3. build    — clean MDX -> chunk by heading -> metadata -> chunks.jsonl
"""

__all__ = ["scope", "manifest", "fetch", "clean", "chunk"]
