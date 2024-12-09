def reverse_min_max_normalization(normalized_value, min_value=0, max_value=4.5):
    # Reverse the min-max scaling
    return (normalized_value * (max_value - min_value)) + min_value


print(reverse_min_max_normalization(0.5855299830436707))