import pickle

def save_object(file_path, obj):
    """Saves a Python object as a pickle file"""
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)
