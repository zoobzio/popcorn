# Popcorn - CUDA Elementwise Kernels for Tendo
# Builds a static library (libpopcorn.a) for linking via cgo

# CUDA paths - adjust if your installation differs
CUDA_PATH ?= /opt/cuda
CUDA_INCLUDE = $(CUDA_PATH)/targets/x86_64-linux/include
CUDA_LIB = $(CUDA_PATH)/targets/x86_64-linux/lib

# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 --compiler-options -fPIC
NVCC_FLAGS += -I$(CUDA_INCLUDE) -Iinclude -Isrc

# Target architecture - adjust for your GPU
# Common options: sm_75 (Turing), sm_80 (Ampere), sm_86 (GA102), sm_89 (Ada)
CUDA_ARCH ?= sm_80

# Fat binary architectures (all common GPUs)
FAT_ARCHS = -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89

# Single-arch flags (default)
NVCC_FLAGS += -arch=$(CUDA_ARCH)

# Directories
SRC_DIR = src
BUILD_DIR = build
LIB_DIR = lib
TEST_DIR = test

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Test files
TEST_SOURCES = $(wildcard $(TEST_DIR)/test_*.cu)
TEST_BINARIES = $(TEST_SOURCES:$(TEST_DIR)/%.cu=$(BUILD_DIR)/%)

# Output
TARGET = $(LIB_DIR)/libpopcorn.a

.PHONY: all clean test tests fat info

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(LIB_DIR)
	ar rcs $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

# -----------------------------------------------------------------------------
# Fat Binary (multi-architecture)
# -----------------------------------------------------------------------------

FAT_BUILD_DIR = build/fat
FAT_OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(FAT_BUILD_DIR)/%.o)

$(FAT_BUILD_DIR):
	mkdir -p $@

$(FAT_BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(FAT_BUILD_DIR)
	$(NVCC) -std=c++17 -O3 --compiler-options -fPIC -I$(CUDA_INCLUDE) -Iinclude -Isrc $(FAT_ARCHS) -c $< -o $@

# Build fat binary (all architectures)
fat: $(FAT_OBJECTS) | $(LIB_DIR)
	ar rcs $(TARGET) $^
	@echo ""
	@echo "Built fat binary with architectures: sm_75, sm_80, sm_86, sm_89"
	@ls -lh $(TARGET)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

# Build test binaries
$(BUILD_DIR)/test_%: $(TEST_DIR)/test_%.cu $(TARGET) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(TEST_DIR) $< -L$(LIB_DIR) -lpopcorn -o $@

# Build all tests
tests: $(TEST_BINARIES)

# Run all tests
test: $(TEST_BINARIES)
	@echo ""
	@echo "Running tests..."
	@echo "================"
	@failed=0; \
	for t in $(TEST_BINARIES); do \
		echo ""; \
		$$t || failed=1; \
	done; \
	echo ""; \
	if [ $$failed -eq 0 ]; then \
		echo "\033[0;32mAll test suites passed\033[0m"; \
	else \
		echo "\033[0;31mSome tests failed\033[0m"; \
		exit 1; \
	fi

# -----------------------------------------------------------------------------
# Info
# -----------------------------------------------------------------------------

info:
	@echo "CUDA_PATH:     $(CUDA_PATH)"
	@echo "CUDA_ARCH:     $(CUDA_ARCH)"
	@echo "NVCC_FLAGS:    $(NVCC_FLAGS)"
	@echo "SOURCES:       $(SOURCES)"
	@echo "OBJECTS:       $(OBJECTS)"
	@echo "TEST_SOURCES:  $(TEST_SOURCES)"
	@echo "TEST_BINARIES: $(TEST_BINARIES)"
