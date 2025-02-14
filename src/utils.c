#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

// Gaussian PDF with numerical stability (prevents division by zero)
double gaussian_pdf(double x, double mean, double variance) {
    const double epsilon = 1e-9;
    variance += epsilon; // Avoid zero variance
    double exponent = exp(-pow(x - mean, 2) / (2 * variance));
    return (1.0 / sqrt(2 * M_PI * variance)) * exponent;
}

// Create a deep copy of an array
double *copy_array(const double *src, int n) {
    double *dest = safe_malloc(n * sizeof(double));
    memcpy(dest, src, n * sizeof(double));
    return dest;
}

// Fisher-Yates shuffle
void shuffle(int *array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Safe memory allocation
void *safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Free 2D array of doubles
void free_2d_array(double **array, int n_rows) {
    for (int i = 0; i < n_rows; i++) {
        free(array[i]); // Free each row
    }
    free(array); // Free the array of pointers
}