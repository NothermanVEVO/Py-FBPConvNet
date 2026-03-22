import phantoms.Dataset as Dataset
import os

QUANT_OF_TRAIN_IMGS = 12500
X_TRAIN_PATH = "dataset/x_train"
Y_TRAIN_PATH = "dataset/y_train"

QUANT_OF_TEST_IMGS = 50
X_TEST_PATH = "dataset/x_test"
Y_TEST_PATH = "dataset/y_test"

PROJECTION = 15

def _generate_datasets() -> None:
    print("Generating dataset of \"", PROJECTION, "\" projections")
    print("Generating TRAIN dataset...", QUANT_OF_TRAIN_IMGS)

    os.makedirs(X_TRAIN_PATH, exist_ok=True)
    os.makedirs(Y_TRAIN_PATH, exist_ok=True)

    Dataset.generate_custom_data_set(
        QUANT_OF_TRAIN_IMGS,
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        [PROJECTION]
    )

    print("Generating TEST dataset...", QUANT_OF_TEST_IMGS)

    os.makedirs(X_TEST_PATH, exist_ok=True)
    os.makedirs(Y_TEST_PATH, exist_ok=True)

    Dataset.generate_custom_data_set(
        QUANT_OF_TEST_IMGS,
        X_TEST_PATH,
        Y_TEST_PATH,
        [PROJECTION]
    )

if __name__ == "__main__":
    _generate_datasets()