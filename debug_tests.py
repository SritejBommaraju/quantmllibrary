import subprocess
import sys

def run_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--no-cov", "-v"],
        capture_output=True,
        text=True
    )
    
    # Print everything if there are failures
    if result.returncode != 0:
        print("TEST FAILURES DETECTED")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
    else:
        print("ALL TESTS PASSED")

if __name__ == "__main__":
    run_tests()
