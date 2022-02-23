#!/usr/bin/bash

#pause with showing image and no timer
#./cifar10_train_v1_2020.py > cpu.log &
#./cifar10_gpu_train_v1_2020.py > gpu.log &

#no pause and timer
#./cifar10_train_timer_no_pause_v2_2020.py > cpu.log &
#./cifar10_gpu_train_timer_no_pause_v2_2020.py > gpu.log &

#larger 60 or 600
./cifar10_cpu_larger_NN_v3_2020.py > cpu.log &
./cifar10_gpu_larger_NN_v3_2020.py > gpu.log &

#even larger 6000 and batch 40 instead of 4
#./cifar10_cpu_even_larger_NN_larger_batch_v4_2020.py > cpu.log &
#./cifar10_gpu_even_larger_NN_larger_batch_v4_2020.py > gpu.log &

#6000 and 40 and proper reporting
#./cifar10_cpu_report_w_larger_batch_v5_2020.py > cpu.log &
#./cifar10_gpu_report_w_larger_batch_v5_2020.py > gpu.log &

#6000 and 4 (original) batch
#./cifar10_cpu_6k_same_4origbatch_v5b_2020.py > cpu.log &
#./cifar10_gpu_6k_same_4origbatch_v5b_2020.py > gpu.log &

