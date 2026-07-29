"""Retrieval, grounded generation, and evaluation over a chunked docs corpus.

Layout, outermost to innermost:

    cli / commands      flags in, wiring out — the only place that knows both
                        which client implements what, and which policy wants it
    clients/            the only modules that open a network connection
    ports               the four Protocols those clients implement
    search, generate,   pure policy: retrieval strategy, prompt assembly,
    eval, report,       scoring, rendering, tracing — no I/O, no keys
    observe
    store, index        the SQLite index and how it gets built

Dependencies point inward: policy declares the interfaces it needs (`ports`),
and the clients conform. That is what lets the whole retrieval -> generate ->
judge -> aggregate path run in tests with no API key and no network.
"""
