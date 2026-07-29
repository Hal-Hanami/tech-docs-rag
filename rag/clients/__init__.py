"""Adapters: the only modules in this package that open a network connection.

Everything here implements a Protocol from `rag.ports`. Nothing here contains
retrieval, prompting, or scoring policy — if a decision about *what the system
should do* ends up in this package, it is in the wrong place.
"""
