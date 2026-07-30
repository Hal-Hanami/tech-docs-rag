# Working in this repository

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing anything. Its **"Four rules
the build enforces"** section is the part that is easy to violate by accident:
no module at 0% coverage, every published number tied to something that produces it,
`tests/test_public_api.py` updated in the same commit as any change to what other
repositories import, and a restructuring verified with an AST comparison rather
than asserted.

Commit and comment conventions are in the same file.
