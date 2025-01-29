def print_label_mapping(class_mapping, model_name):
    """
    Prints the mapping of original labels to encoded labels for a given model.

    Parameters:
        class_mapping (dict): The dictionary mapping labels to encoded values.
        model_name (str): The name of the model (e.g., "SVM", "PLS_DA", "XGBoost").
    """
    # Determine the maximum label length for formatting
    max_label_length = max(len(label) for label in class_mapping)
    
    print(f"\nModel: {model_name}")
    print("Structure".ljust(max_label_length + 4) + " | Encoded label")
    print("=" * (max_label_length + 20))

    # Print each label and its encoded value with proper spacing
    for label, encoded in class_mapping.items():
        print(f"{label.ljust(max_label_length + 5)}| {encoded}")