#!/bin/bash

src="ImageNet"
target="${1^^}"
arch="$2"
shift 2

mkdir -p logs
mkdir -p runs

dataset="Digits"
rootdir="data/$dataset"

params="--epochs 50 --lr-steps 10 20 30 40"

for mode in single # dual triple
do
	for scorer in entropy consistency
	do
		for sampler in random toprank uniform
		do
			for ratio in 0.01 0.1
			do
				for pretrained in 1 # 0 1
				do
					for useall in 0 # 1
					do
						run="python -u digits.py -r $rootdir"
						logfile="${dataset,,}"_"${src,,}"_"${target,,}"_"${arch,,}"_"${mode,,}"_"${sampler,,}"_"$ratio"_"${scorer,,}"
						if [ "$pretrained" -eq 0 ]
						then
							logfile="$logfile"_"scratch"
						else
							run="$run --pretrained"
							logfile="$logfile"_"finetune"
						fi
						if [ "$useall" -eq 0 ]
						then
							logfile="$logfile"_"budget"
						else
							run="$run --useall"
							logfile="$logfile"_"useall"
						fi
						if [ $scorer == "consistency" ]
						then
							for count in {0..9}
							do
								if [ ! -e runs/"$logfile"_"$count" ]
								then
									echo "$run --source=$src --target=$target --arch=$arch --mode=$mode --sampler=$sampler --sample-ratio=$ratio --scorer=$scorer $params $@" |& tee -a logs/$logfile.log
									eval  $run --source=$src --target=$target --arch=$arch --mode=$mode --sampler=$sampler --sample-ratio=$ratio --scorer=$scorer $params $@  |& tee -a logs/$logfile.log
									mv runs/"$logfile" runs/"$logfile"_"$count"
									mv logs/"$logfile".log logs/"$logfile"_"$count".log
								fi
							done
						else
							if [ ! -e runs/"$logfile" ]
							then
								echo "$run --source=$src --target=$target --arch=$arch --mode=$mode --sampler=$sampler --sample-ratio=$ratio --scorer=$scorer $params $@" |& tee -a logs/$logfile.log
								eval  $run --source=$src --target=$target --arch=$arch --mode=$mode --sampler=$sampler --sample-ratio=$ratio --scorer=$scorer $params $@  |& tee -a logs/$logfile.log
							fi
						fi
					done
				done
			done
		done
	done
done

