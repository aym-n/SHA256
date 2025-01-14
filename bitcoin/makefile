
SRC_DIR = ./src
BUILD_DIR = ./build
OBJECTS = $(BUILD_DIR)/gpu_miner

all: $(OBJECTS)

clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/gpu_miner: $(SRC_DIR)/main.cu $(BUILD_DIR)/utils.o $(BUILD_DIR)/sha256.o | $(BUILD_DIR)
	nvcc -ccbin clang++ -O1 -v -lrt -lm -arch=sm_75 -o $@ $^

$(BUILD_DIR)/verify_gpu: $(SRC_DIR)/main.cu $(BUILD_DIR)/utils.o $(BUILD_DIR)/sha256.o | $(BUILD_DIR)
	nvcc -ccbin clang++ -O1 -v -lrt -lm -D VERIFY_HASH -arch=sm_75 -o $@ $^

$(BUILD_DIR)/sha256.o: $(SRC_DIR)/sha256.c | $(BUILD_DIR)
	gcc -O1 -v -c -o $@ $^

$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.c | $(BUILD_DIR)
	gcc -O1 -v -c -o $@ $^ -lrt
