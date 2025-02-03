#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "data_loader.h" // For Dataset struct

// Normalize features to [0, 1] using min-max scaling (prevents data leakage)
void normalize(Dataset *train, Dataset *test);

// Split dataset into training and testing sets (stratified split)
void train_test_split(const Dataset *data, Dataset *train, Dataset *test, float test_size);

#endif // PREPROCESSING_H