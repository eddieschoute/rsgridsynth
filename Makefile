.PHONY: help build build-cli fmt fmt-check clippy test ci clean

CARGO_TERM_COLOR ?= always
export CARGO_TERM_COLOR

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Build the library (no CLI, default feature set)
	cargo build

build-cli: ## Build the release CLI binary (requires the `cli` feature)
	cargo build --bin rsgridsynth -F cli --release

fmt: ## Reformat all code in place
	cargo fmt --all

fmt-check: ## Check formatting without modifying files (CI gate)
	cargo fmt --all -- --check

clippy: ## Lint with all features and all targets, warnings are errors
	cargo clippy --all-features --all-targets -- -D warnings

test: ## Run all tests (lib, bins, integration tests, examples) plus doctests
	cargo test --all-features --all-targets
	cargo test --all-features --doc

ci: fmt-check clippy test ## Full CI gate: fmt-check, clippy, test

clean: ## Remove build artifacts
	cargo clean
