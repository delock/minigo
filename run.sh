export KMP_HW_SUBSET=28c,2T
export KMP_BLOCKTIME=1
export KMP_AFFINITY=compact,granularity=fine
export OMP_NUM_THREADS=56
ulimit -u 760000
LOG_FILE=train_log/origin/result-`hostname`-`date +%m-%d-%H%M`.txt
LOG_CURRENT_FILE=train_log/result_current.txt
./run_minigo.sh $(pwd)/results/$(hostname) ml_perf/flags/9.lite 2>&1 |tee $LOG_FILE |tee $LOG_CURRENT_FILE
