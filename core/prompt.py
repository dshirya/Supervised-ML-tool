def print_label_mapping(encoder):
    # Get the mapping of original labels to encoded labels
    label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    # Determine the maximum label length for formatting
    max_label_length = max(len(label) for label in label_mapping)
    print("\nStructure".ljust(max_label_length + 4) + " | Encoded label")
    print("=" * (max_label_length + 20))

    # Print each label and its encoded value with proper spacing
    for label, encoded in label_mapping.items():
        print(f"{label.ljust(max_label_length + 5)}| {encoded}")
