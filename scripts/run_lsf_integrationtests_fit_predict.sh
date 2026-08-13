#!/bin/bash
# run_lsf_integrationtests_fit_predict.sh
# Run each test (either file or individual test function) in separate LSF jobs with coverage + logs.

# Environment variables:
#   TEST_DIR    - Base directory for tests (default: current directory or first argument)
#   TEST_FILE   - Test file to run (default: integrationtests/test_base_set.py)
#   VENV_PATH   - Path to Python virtual environment (default: .venv)
#   TYPE        - Type of tests to run: "fit" or "predict" (default: "fit")

set -euo pipefail


TEST_DIR="/dccstor/terratorch/users/wanjiru/terratorch"
LOG_DIR="$TEST_DIR/logs"
COV_DIR="$TEST_DIR/.coverage_jobs"
VENV_PATH="/dccstor/terratorch/users/wanjiru/.wanjiru_tests/bin/activate"
TYPE="fit" # Or predict
mkdir -p "$LOG_DIR" "$COV_DIR"

source $VENV_PATH

echo "Grabbing tests" 
all_tests_str=$(cd "$TEST_DIR" && \
  pytest --collect-only -q integrationtests/test_base_set.py 2>/dev/null | \
  grep -E '^integrationtests/test_base_set\.py::' || true)

mapfile -t all_tests <<< "$all_tests_str"

echo "Found tests in array : ${#all_tests[@]}"



# Removing all prior tests artifacts to makes sure the new tests 
# run cleanly with correct terratorch version
shopt -s extglob
rm -rf /dccstor/terratorch/tmp/!("Python"*)

predict_tests=()
fit_tests=()
unknown_tests=()


# Run fit separately from current_predict to ensure the checkpoints exist
# grab from the file_names automatically
for test in "${all_tests[@]}"; do
    if [[ "$test" == *::test_*fit* ]]; then
        fit_tests+=("$test")
    elif [[ "$test" == *::test_*predict* ]]; then
        predict_tests+=("$test")
    else
        unknown_tests+=("$test")
    fi
done


submit_test_job() {
  local test_name="$1"

  test_name=$(echo "$test" | tr '/:' '_')
  out="$LOG_DIR/${test_name}.out"
  err="$LOG_DIR/${test_name}.err"

  echo "Submitting job for $test"
  hash=$(echo -n "$test" | sha1sum | cut -c1-10)
  job_name="tt_${hash}"


  bsub -gpu num=1 -R "rusage[ngpus=1,cpu=4,mem=32GB]" \
        -J "terratorch_${job_name}" \
        -oo "$out" -eo "$err" \
        "cd $TEST_DIR && \
        source /dccstor/terratorch/users/wanjiru/.wanjiru_tests/bin/activate && \
        pytest -s -v $test"


  # add other repeated commands here
}

run_tests_for_type() {
    local test_list=("$@")  # all arguments passed as an array
    for test in "${test_list[@]}"; do
        submit_test_job "$test"
        echo "Submiited job for $test"
    done
}

if [[ "$TYPE" == "fit" ]]; then
  # run fit tests
  echo "Running Fit tests..."
  run_tests_for_type "${fit_tests[@]}"  
  echo "All jobs submitted. Monitor with: bjobs -u \$USER"

elif [[ "$TYPE" == "predict" ]]; then
  # run predict tests
  echo "Running Predict tests..."
  run_tests_for_type "${predict_tests[@]}"
  echo "All jobs submitted. Monitor with: bjobs -u \$USER"

else
  echo "Running unknown tests"
  run_tests_for_type "${unknown_tests[@]}"
  echo "All jobs submitted. Monitor with: bjobs -u \$USER"
  fi
