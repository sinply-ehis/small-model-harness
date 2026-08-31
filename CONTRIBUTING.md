# Contributing to small-model-harness

Thanks for your interest in contributing! This guide covers how to get started.

## Development setup

```bash
git clone https://github.com/sinply-ehis/small-model-harness.git
cd small-model-harness
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                  # Run all tests
pytest -v               # Verbose
pytest tests/test_harness.py  # Core only
```

All 142 tests must pass before submitting a PR.

## Code style

- Python 3.10+ only (use `X | Y` unions, not `Optional[X]`)
- Pydantic v2 for all models (`BaseModel`, `Field`, `model_dump`)
- No comments unless the logic is genuinely non-obvious
- Max 500 lines per file, 100 lines per function
- Type hints on all public functions

## Adding a new feature

1. Add tests first — cover the happy path and edge cases
2. Implement the feature
3. Run `pytest` — all tests must pass
4. Update the README if it's a public API change
5. Submit a PR with a clear description

## Adding a new tool repair strategy

The `tool_repair.py` module handles deterministic output repair. To add a new strategy:

1. Add a function like `repair_your_thing(args, schema) -> (repaired_args, fixes)`
2. Wire it into `repair_tool_call()` in the pipeline
3. Add tests in `tests/test_tool_repair.py`

## Reporting issues

Open a GitHub issue with:
- What you expected
- What actually happened
- Steps to reproduce
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under MIT.
