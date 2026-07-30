# Working in this repository

Read [`docs/DESIGN.md`](docs/DESIGN.md) before changing behaviour. It states what
the system must do, in numbered sections that code and tests cite. If a change
makes one of those statements false, the statement is what to fix first.

Then read [CONTRIBUTING.md](CONTRIBUTING.md). Its **"Five rules the build
enforces"** section is the part that is easy to violate by accident: every design
section pinned by a test, no module at 0% coverage, every published number tied to
something that produces it, `tests/test_public_api.py` updated in the same commit
as any change to what other repositories import, and a restructuring verified with
an AST comparison rather than asserted.

**One concern per file.** A module's docstring carries the reason that module
exists; a promise spanning modules goes in `docs/DESIGN.md`; a measured number goes
in `docs/EVALUATION.md` with its date and reproduction command; the README quotes
those rather than repeating them.
