# Wake Chess Engine Makefile
# Automates building, testing, and deployment of the UCI chess engine

# Variables
PYTHON := python3
ENGINE_NAME := wake_chess_engine
ENTRY_POINT := wake_engine.py
DIST_DIR := dist
BUILD_DIR := build
SPEC_FILE := $(ENGINE_NAME).spec

# Default target
.PHONY: all
all: build

# Build the standalone executable
.PHONY: build
build: clean-build
	@echo "Building Wake Chess Engine executable..."
	$(PYTHON) -m PyInstaller --onefile --console --name $(ENGINE_NAME) $(ENTRY_POINT)
	@echo "Build complete! Executable at: $(DIST_DIR)/$(ENGINE_NAME)"

# Quick rebuild (without cleaning)
.PHONY: rebuild
rebuild:
	@echo "Rebuilding Wake Chess Engine executable..."
	$(PYTHON) -m PyInstaller --onefile --console --name $(ENGINE_NAME) $(ENTRY_POINT)
	@echo "Rebuild complete! Executable at: $(DIST_DIR)/$(ENGINE_NAME)"

# Test the Python script directly
.PHONY: test-python
test-python:
	@echo "Testing Python UCI script..."
	@echo -e "uci\nisready\nposition startpos\ngo depth 2\nquit" | $(PYTHON) $(ENTRY_POINT)

# Test the standalone executable
.PHONY: test-exe
test-exe: build
	@echo "Testing standalone executable..."
	@echo -e "uci\nisready\nposition startpos\ngo depth 2\nquit" | ./$(DIST_DIR)/$(ENGINE_NAME)

# Test both versions
.PHONY: test
test: test-python test-exe
	@echo "All tests completed successfully!"

# Clean build artifacts
.PHONY: clean-build
clean-build:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(DIST_DIR)
	@rm -f $(SPEC_FILE)

# Clean Python cache files
.PHONY: clean-cache
clean-cache:
	@echo "Cleaning Python cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Full clean
.PHONY: clean
clean: clean-build clean-cache
	@echo "Full cleanup completed!"

# Install/check dependencies
.PHONY: deps
deps:
	@echo "Installing/checking dependencies..."
	$(PYTHON) -m pip install --user pyinstaller
	$(PYTHON) -m pip install --user -r requirements.txt

# Show executable info
.PHONY: info
info:
	@if [ -f "$(DIST_DIR)/$(ENGINE_NAME)" ]; then \
		echo "Wake Chess Engine Executable Info:"; \
		echo "  Path: $(DIST_DIR)/$(ENGINE_NAME)"; \
		echo "  Size: $$(du -h $(DIST_DIR)/$(ENGINE_NAME) | cut -f1)"; \
		echo "  Modified: $$(stat -f '%Sm' $(DIST_DIR)/$(ENGINE_NAME))"; \
		echo "  Permissions: $$(ls -l $(DIST_DIR)/$(ENGINE_NAME) | cut -d' ' -f1)"; \
	else \
		echo "Executable not found. Run 'make build' first."; \
	fi

# Install to /usr/local/bin for system-wide access
.PHONY: install
install: build
	@echo "Installing Wake Chess Engine to /usr/local/bin..."
	@sudo cp $(DIST_DIR)/$(ENGINE_NAME) /usr/local/bin/
	@sudo chmod +x /usr/local/bin/$(ENGINE_NAME)
	@echo "Installed! You can now run 'wake_chess_engine' from anywhere."

# Uninstall from system
.PHONY: uninstall
uninstall:
	@echo "Removing Wake Chess Engine from /usr/local/bin..."
	@sudo rm -f /usr/local/bin/$(ENGINE_NAME)
	@echo "Uninstalled."

# Development mode - watch for changes and rebuild
.PHONY: dev
dev:
	@echo "Development mode: watching for changes..."
	@echo "Run 'make rebuild' after making changes, or use a file watcher."
	@echo "Example: 'fswatch -o wake/ $(ENTRY_POINT) | xargs -n1 make rebuild'"

# Package for distribution
.PHONY: package
package: build
	@echo "Creating distribution package..."
	@mkdir -p release
	@cp $(DIST_DIR)/$(ENGINE_NAME) release/
	@cp README.md release/ 2>/dev/null || echo "# Wake Chess Engine\n\nUCI-compliant chess engine executable.\n\nUsage: ./wake_chess_engine" > release/README.md
	@echo "Distribution package created in 'release/' directory"

# Quick UCI test with custom position
.PHONY: test-position
test-position: build
	@echo "Testing with custom position (Scholar's mate setup)..."
	@echo -e "uci\nisready\nposition startpos moves e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7\ngo depth 3\nquit" | ./$(DIST_DIR)/$(ENGINE_NAME)

# Benchmark test
.PHONY: benchmark
benchmark: build
	@echo "Running benchmark (depth 4 from starting position)..."
	@time echo -e "uci\nisready\nposition startpos\ngo depth 4\nquit" | ./$(DIST_DIR)/$(ENGINE_NAME)

# Help
.PHONY: help
help:
	@echo "Wake Chess Engine Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build         - Build the standalone executable"
	@echo "  rebuild       - Quick rebuild without cleaning"
	@echo "  test          - Test both Python script and executable"
	@echo "  test-python   - Test the Python UCI script"
	@echo "  test-exe      - Test the standalone executable"
	@echo "  clean         - Clean all build artifacts and cache"
	@echo "  deps          - Install/check dependencies"
	@echo "  info          - Show executable information"
	@echo "  install       - Install to /usr/local/bin (requires sudo)"
	@echo "  uninstall     - Remove from /usr/local/bin (requires sudo)"
	@echo "  package       - Create distribution package"
	@echo "  test-position - Test with custom chess position"
	@echo "  benchmark     - Run performance benchmark"
	@echo "  dev           - Show development mode info"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Example workflow:"
	@echo "  make deps     # Install dependencies"
	@echo "  make build    # Build executable"
	@echo "  make test     # Test everything"
	@echo "  make install  # Install system-wide (optional)" 