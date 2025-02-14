#ifndef UTILS_H
#define UTILS_H

#include <math.h>

// Define M_PI if not already provided by math.h
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Gaussian probability density function (PDF)
double gaussian_pdf(double x, double mean, double variance);

// Create a deep copy of an array
double *copy_array(const double *src, int n);

// Shuffle an array of integers (Fisher-Yates algorithm)
void shuffle(int *array, int n);

// Safe memory allocation with error checking
void *safe_malloc(size_t size);

// Free 2D array of doubles
void free_2d_array(double **array, int n_rows);

#endif // UTILS_H