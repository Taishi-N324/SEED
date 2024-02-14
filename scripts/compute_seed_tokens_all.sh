#!/bin/bash

start=0
end=20

for file_index in {0..20}
do
    padded_file_index=$(printf "%05d" $file_index)
    padded_start=$(printf "%05d" $start)
    padded_end=$(printf "%05d" $end)
	echo $padded_start $padded_end grit-train-$padded_file_index-of-00021
    sbatch scripts/compute_seed_tokens_grit_1gpu.sh $padded_start $padded_end grit-train-$padded_file_index-of-00021
done

for i in {1..2}
do
    start=$((i * 20 + 1))
    end=$((start + 19))

    for file_index in {0..20}
    do
        padded_file_index=$(printf "%05d" $file_index)
        padded_start=$(printf "%05d" $start)
        padded_end=$(printf "%05d" $end)
		echo $padded_start $padded_end grit-train-$padded_file_index-of-00021
        sbatch scripts/compute_seed_tokens_grit_1gpu.sh $padded_start $padded_end grit-train-$padded_file_index-of-00021
    done
done
