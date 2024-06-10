import os

def create_and_check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def join_file_path(path, filename):
    return os.path.join(path, filename)

def check_file_exists(filepath):
    return os.path.exists(filepath)
