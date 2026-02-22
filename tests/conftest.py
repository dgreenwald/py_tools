"""Shared pytest configuration for py_tools tests.

This file intentionally stays minimal for now and can be extended with
project-wide fixtures as test coverage expands in Phase 5.2.

Bayesian tests are currently serial-only by default: they exercise
`parallel=False` paths and avoid launching MPI workers. Parallel MPI behavior
should be covered by separate integration tests run under an MPI launcher.
"""
