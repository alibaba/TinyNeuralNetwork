import subprocess
import sys

import glob

# List of test files to run
test_files = glob.glob('**/*_test.py', recursive=True)

# Function to run a test file
def run_test(file):
    try:
        result = subprocess.run(["pytest", "-v", file], check=True, capture_output=True, text=True)
        print(result.stdout)
        return None  # No error
    except subprocess.CalledProcessError as e:
        error_message = f"Tests failed in {file}\n{e.stdout}\n{e.stderr}"
        return error_message  # Return the error message


# Run all test files
def main():
    errors = []
    for test_file in test_files:
        error = run_test(test_file)
        if error:
            errors.append(error)

    if errors:
        raise Exception("\n".join(errors))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
