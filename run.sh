#!/bin/bash

case $1 in

	41)
		echo "shared memory"
		cp ece408_src/new-forward_4_sharedMemory.cuh ece408_src/new-forward.cuh
		cd ..
		rai -p UIUC_2_1_ECE408_FinalProject --queue rai_amd64_ece408
		cd UIUC_2_1_ECE408_FinalProject
		;;

	42)
		echo "shared and constant memory"
		cp ece408_src/new-forward_4_shared_constMem.cuh ece408_src/new-forward.cuh
		cd ..
		rai -p UIUC_2_1_ECE408_FinalProject --queue rai_amd64_ece408
		cd UIUC_2_1_ECE408_FinalProject
		;;

	43)
		echo "unrolling"
		cp ece408_src/new-forward_4_unroll.cuh ece408_src/new-forward.cuh
		cd ..
		rai -p UIUC_2_1_ECE408_FinalProject --queue rai_amd64_ece408
		cd UIUC_2_1_ECE408_FinalProject
		;;

	51)
		echo "tiled matrix multiplication"
		cp ece408_src/new-forward_5_tiledMatrixMultiplication.cuh ece408_src/new-forward.cuh
		cd ..
		rai -p UIUC_2_1_ECE408_FinalProject --queue rai_amd64_ece408
		cd UIUC_2_1_ECE408_FinalProject
		;;

	52)
		echo "two layers tiled matrix multiplication"
		cp ece408_src/new-forward_5_twoLayers.cuh ece408_src/new-forward.cuh
		cd ..
		rai -p UIUC_2_1_ECE408_FinalProject --queue rai_amd64_ece408
		cd UIUC_2_1_ECE408_FinalProject
		;;

	52)
		echo "tiled matrix multiplication with tuned parameters"
		cp ece408_src/new-forward_5_parameterTuning.cuh ece408_src/new-forward.cuh
		cd ..
		rai -p UIUC_2_1_ECE408_FinalProject --queue rai_amd64_ece408
		cd UIUC_2_1_ECE408_FinalProject
		;;

	*)
		echo "Illegal argument!"
		;;

esac