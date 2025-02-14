# Compiler and flags
CC = gcc
CFLAGS = -Wall -g -I./include  # Added -g for debugging and include directory
LDFLAGS = -lm

# Source and object files
SRC_DIR = src
SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(SRC:.c=.o)

# Output executable
TARGET = bin/main.exe

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJ)
	@mkdir -p bin  # Ensure the bin directory exists
	$(CC) -o $@ $^ $(LDFLAGS)

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean