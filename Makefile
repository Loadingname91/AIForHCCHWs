# CS6961 · convenience targets. Everything here also works as a plain command.
ENV_NAME ?= cs6961-agents

.PHONY: help setup ollama models doctor test test-live examples clean

help:
	@echo "make setup      create the conda env and install everything"
	@echo "make ollama     install Ollama and start the server"
	@echo "make models     pull a Qwen3 model sized for this machine"
	@echo "make doctor     check the environment end to end"
	@echo "make test       run the offline tests"
	@echo "make test-live  also test against the running LLM server"
	@echo "make examples   run every example in order"
	@echo ""
	@echo "Remember: conda activate $(ENV_NAME)"

setup:
	bash scripts/setup_env.sh $(ENV_NAME)

ollama:
	bash scripts/install_ollama.sh

models:
	bash scripts/pull_models.sh

doctor:
	python scripts/doctor.py

test:
	pytest -q

test-live:
	pytest -q -m live

examples:
	@for f in examples/0*.py; do \
		case "$$f" in *06_*) continue;; esac; \
		printf "\n=== %s ===\n" "$$f"; python "$$f" || exit 1; \
	done

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache storage
