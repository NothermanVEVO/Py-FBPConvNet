import os
import phantoms.Dataset as Dataset

QUANT_OF_TRAIN_IMGS = 2000
QUANT_OF_TEST_IMGS = 500

TRAIN_PATH = "Dataset/GroundTruth/Train"
TEST_PATH = "Dataset/GroundTruth/Test"

# print("Gerando dataset de ground truth...")
# Dataset.generate_ground_truth_dataset(
#     QUANT_OF_TRAIN_IMGS, TRAIN_PATH)

# print("Gerando dataset de ground truth...")
# Dataset.generate_ground_truth_dataset(
#     QUANT_OF_TEST_IMGS, TEST_PATH)

projection = 15

print("gerando TRAIN dataset de low: ", projection, " projections")
Dataset.generate_low_projections_dataset(
    TRAIN_PATH,
    os.path.join("Dataset", str(projection), "Train"),
    QUANT_OF_TRAIN_IMGS,
    projection
)

print("gerando TEST dataset de low: ", projection, " projections")
Dataset.generate_low_projections_dataset(
    TEST_PATH,
    os.path.join("Dataset", str(projection), "Test"),
    QUANT_OF_TEST_IMGS,
    projection
)

projection = 30

print("gerando TRAIN dataset de low: ", projection, " projections")
Dataset.generate_low_projections_dataset(
    TRAIN_PATH,
    os.path.join("Dataset", str(projection), "Train"),
    QUANT_OF_TRAIN_IMGS,
    projection
)

print("gerando TEST dataset de low: ", projection, " projections")
Dataset.generate_low_projections_dataset(
    TEST_PATH,
    os.path.join("Dataset", str(projection), "Test"),
    QUANT_OF_TEST_IMGS,
    projection
)

projection = 60

print("gerando TEST dataset de low: ", projection, " projections")
Dataset.generate_low_projections_dataset(
    TRAIN_PATH,
    os.path.join("Dataset", str(projection), "Train"),
    QUANT_OF_TRAIN_IMGS,
    projection
)

print("gerando TEST dataset de low: ", projection, " projections")
Dataset.generate_low_projections_dataset(
    TEST_PATH,
    os.path.join("Dataset", str(projection), "Test"),
    QUANT_OF_TEST_IMGS,
    projection
)