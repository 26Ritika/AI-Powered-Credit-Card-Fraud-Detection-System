from src.preprocessing import prepare_data

data = prepare_data()

print("\nPreprocessing completed!")

print("Training samples:", len(data["X_train"]))
print("Validation samples:", len(data["X_val"]))
print("Testing samples:", len(data["X_test"]))
print("Number of features:", len(data["feature_names"]))