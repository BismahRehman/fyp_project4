import pickle

# Load the encoder object
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Get all encoded feature names
if hasattr(encoder, 'get_feature_names_out'):
    feature_names = encoder.get_feature_names_out()
else:
    feature_names = encoder.get_feature_names()

# Print the feature names
print("Total Encoded Columns:", len(feature_names))
print("Encoded Columns:")
for col in feature_names:
    print(col)
