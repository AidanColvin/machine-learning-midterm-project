.PHONY: run preprocessing

PYTHON := python3
SRC    := src

run:
	@$(MAKE) --no-print-directory $(filter-out $@,$(MAKECMDGOALS))

%:
	@:

preprocessing:
	@$(PYTHON) $(SRC)/load-raw-training-data.py
	@$(PYTHON) $(SRC)/preprocessing.py