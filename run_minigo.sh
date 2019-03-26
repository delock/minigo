#gsutil -m cp gs://tensor-go-ml-perf/models/9x9/target.pb ml_perf/
#md5sum ml_perf/target.pb

BASE_DIR=$1
FLAG_DIR=$2

# Run training loop
BOARD_SIZE=9  python3  ml_perf/reference_implementation.py \
  --base_dir=$BASE_DIR \
  --flagfile=$FLAG_DIR/rl_loop.flags

# Once the training loop has finished, run model evaluation to find the
# first trained model that's better than the target
#BOARD_SIZE=9  python3  ml_perf/eval_models.py \
#  --base_dir=$BASE_DIR \
#  --flags_dir=$FLAG_DIR
