# Video Generation Blueprint

Use [README.md](README.md) for setup and [operator/Cargo.toml](operator/Cargo.toml) for supported dependencies.
Keep shared payment validation, billing, health, and metrics in the `tangle-inference-core` dependency.

## Verification

For API changes, extend [e2e.rs](operator/tests/e2e.rs): run the actual server and verify submission, polling, invalid input, and missing jobs.
Replace only the external generation backend when hardware is unavailable.
[lifecycle.rs](operator/tests/lifecycle.rs) invokes the real handler with a backend substitute; it does not prove on-chain submission or settlement.
Chain changes need submission, operator processing, and recorded results through the production runner.
Use the SDK revision selected by the manifests and lockfile for test APIs and fixtures.
Exercise contract changes with actual deployments under `contracts/`.
Choose tests for the changed behavior and report substitutes or skipped prerequisites with the result.
