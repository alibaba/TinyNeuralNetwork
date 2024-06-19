import subprocess
import sys

# Function to run a test file
def run_test(file):
    try:
        subprocess.run(["pytest", "-v", file], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


if __name__ == "__main__":
    # List of test files to run
    test_files = ["tests/converter_op_test.py"]

    print("Test files:")
    for file in test_files:
        print(f" - {file}")

    pass_flag = True
    for test_file in test_files:
        passed = run_test(test_file)
        pass_flag = pass_flag and passed

    if not pass_flag:
        sys.exit(1)
