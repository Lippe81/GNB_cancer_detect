#ifndef DATA_LOADER_H
#define DATA_LOADER_H

// Structure to hold the dataset
typedef struct {
    double **features;  // 2D array [n_samples][n_features]
    int *labels;        // Array [n_samples] (0 for benign, 1 for malignant)
    int n_samples;      // Total number of samples
    int n_features;     // Total features per sample (30)
} Dataset;

// Function to load CSV data into a Dataset struct
void load_csv(const char *filename, Dataset *data);

// Function to free memory allocated for the Dataset
void free_dataset(Dataset *data);

#endif // DATA_LOADER_H