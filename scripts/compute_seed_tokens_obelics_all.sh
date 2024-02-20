#!/bin/bash

# 開始と終了のシャードを指定
START=0
END=1335
INCREMENT=40

# STARTからENDまでINCREMENTごとにループ
for ((i=START; i<=END; i+=INCREMENT)); do
  # 終了シャードを計算。ただし、ループの最終イテレーションではENDを使用
  if ((i + INCREMENT > END)); then
    END_SHARD=$END
  else
    END_SHARD=$((i+INCREMENT-1))
  fi
  echo $i $END_SHARD
  # sbatchコマンドでスクリプトをサブミット
  sbatch scripts/compute_seed_tokens_obelics.sh $i $END_SHARD
done
