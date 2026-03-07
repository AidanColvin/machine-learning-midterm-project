# -----------------------------------
# MACHINE LEARNING PIPELINE COMMANDS
# -----------------------------------

PYTHON = python3
SRC = src

# Default target
all: preprocess

# ----------------------
# DATA PREPROCESSING
# ----------------------
preprocess:
	$(PYTHON) $(SRC)/preprocessing.py

# ----------------------
# LOAD RAW DATA
# ----------------------
load:
	$(PYTHON) $(SRC)/load-raw-training-data.py

# ----------------------
# CLEAN TEMP FILES
# ----------------------
clean:
	rm -rf __pycache__
	rm -rf $(SRC)/__pycache__
	rm -rf data/processed

# ----------------------
# HELP
# ----------------------
help:
	@echo ""
	@echo "Available commands:"
	@echo " make preprocess   -> run preprocessing pipeline"
	@echo " make load         -> load raw training data"
	@echo " make clean        -> remove temp files"
	@echo ""