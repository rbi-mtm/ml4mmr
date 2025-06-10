# Activate conda environment to ensure rlaunch command is available
source /opt/conda/bin/activate base && conda activate T3

# Ensure the environment variables pointing to the configuration files are correctly set
export ATOMATE2_CONFIG_FILE="$PWD/atomate2/config/atomate2.yaml"
export JOBFLOW_CONFIG_FILE="$PWD/atomate2/config/jobflow.yaml"

# Launch the jobs
rlaunch -w my_fworker.yaml -c . rapidfire
