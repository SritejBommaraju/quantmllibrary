import subprocess
import sys

def run_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_activations.py", "--no-cov", "-v"],
        capture_output=True,
        text=True
    )
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

if __name__ == "__main__":
    run_tests()
