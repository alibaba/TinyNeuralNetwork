import subprocess
import sys

import glob

# List of test files to run
test_files = glob.glob('tests/*_test.py', recursive=True)

print("Test files:")
for file in test_files:
    print(f" - {file}")

# Function to run a test file
def run_test(file):
    try:
        subprocess.run(["pytest", "-v", file], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


# Run all test files
def main():
    error_flag = True
    for test_file in test_files:
        error = run_test(test_file)
        error_flag = error_flag and error

    return error_flag


if __name__ == "__main__":
    error_flag = True
    for test_file in test_files:
        error = run_test(test_file)
        error_flag = error_flag and error

    if error_flag:
        sys.exit(1)
