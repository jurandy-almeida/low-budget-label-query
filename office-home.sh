#!/bin/bash

src="${1^}"
target="${2^}"
arch="$3"
shift 3

mkdir -p logs
mkdir -p runs
mkdir -p clusters

dataset="OfficeHomeDataset_10072016"
rootdir="data/$dataset"
if [ ! -d "$rootdir" ]
then
	echo "Please, download the Office-Home dataset from"
	echo "https://drive.google.com/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg&export=download"
	echo "and unzip the downloaded file in the 'data' folder"
	exit
fi

params="--workers 2"

for mode in single dual triple
do
	run="python -u office-home.py -r $rootdir"
	logfile="${dataset,,}"_"${src,,}"_"${target,,}"_"${arch,,}"_"${mode,,}"
	if [ ! -e clusters/"$logfile" ]
	then
		echo "$run --clustering --source=$src --target=$target --arch=$arch --mode=$mode $params $@" |& tee -a clusters/$logfile.log
		eval  $run --clustering --source=$src --target=$target --arch=$arch --mode=$mode $params $@  |& tee -a clusters/$logfile.log
	fi
done

params="--epochs 25 --lr-steps 10 20"

for mode in single dual triple
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
						run="python -u office-home.py -r $rootdir"
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

