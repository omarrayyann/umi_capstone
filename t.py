import pickle

def view_pkl_file(file_path):
    """Load and print the content of a .pkl file."""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("Contents of the .pkl file:")
            print(data)
    except Exception as e:
        print(f"Error reading the .pkl file: {e}")

# Example usage
file_path = 'nn.pkl'  # Replace with your .pkl file path
view_pkl_file(file_path)
