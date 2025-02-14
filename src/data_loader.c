#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_loader.h"

// Helper function to count lines in a file
static int count_lines(FILE *file) {
    int lines = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), file)) {
        lines++;
    }
    rewind(file); // Reset file pointer to the beginning
    return lines;
}

// Load CSV data into the Dataset struct
void load_csv(const char *filename, Dataset *data) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Count total lines (excluding header)
    int total_lines = count_lines(file);
    data->n_samples = total_lines - 1; // Skip header
    data->n_features = 30; // Fixed for this dataset

    // Allocate memory for features and labels
    data->features = malloc(data->n_samples * sizeof(double *));
    data->labels = malloc(data->n_samples * sizeof(int));

    // Read and skip the header line
    char header[1024];
    fgets(header, sizeof(header), file);

    // Parse each line
    char line[1024];
    int sample_idx = 0;
    while (fgets(line, sizeof(line), file)) {
        // Allocate memory for features of this sample
        data->features[sample_idx] = malloc(data->n_features * sizeof(double));

        // Split line into tokens
        char *token = strtok(line, ",");

        // Skip the first column (ID)
        token = strtok(NULL, ",");

        // Second column: diagnosis (M=1, B=0)
        if (strcmp(token, "M") == 0) {
            data->labels[sample_idx] = 1;
        } else {
            data->labels[sample_idx] = 0;
        }

        // Read 30 feature columns
        for (int i = 0; i < data->n_features; i++) {
            token = strtok(NULL, ",");
            data->features[sample_idx][i] = atof(token);
        }

        sample_idx++;
    }

    fclose(file);
}

// Free dynamically allocated memory
void free_dataset(Dataset *data) {
    for (int i = 0; i < data->n_samples; i++) {
        free(data->features[i]);
    }
    free(data->features);
    free(data->labels);
    data->n_samples = 0;
    data->n_features = 0;
}