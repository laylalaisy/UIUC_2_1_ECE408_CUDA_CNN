#!/bin/bash

case $1 in

	41)
		echo -n "shared memory"
		cp ece408_src/new-forward_4_sharedMemory.cuh ece408_src/new-forward.cuh
		cd ..
		rai -p UIUC_2_1_ECE408_FinalProject --queue rai_amd64_ece408
		cd UIUC_2_1_ECE408_FinalProject
		;;

	42)
		echo -n "shared and constant memory"
		cp ece408_src/new-forward_4_shared_constMem.cuh ece408_src/new-forward.cuh
		cd ..
		rai -p UIUC_2_1_ECE408_FinalProject --queue rai_amd64_ece408
		cd UIUC_2_1_ECE408_FinalProject
		;;

	43)
		echo -n "unrolling"
		cp ece408_src/new-forward_4_unroll.cuh ece408_src/new-forward.cuh
		cd ..
		rai -p UIUC_2_1_ECE408_FinalProject --queue rai_amd64_ece408
		cd UIUC_2_1_ECE408_FinalProject
		;;

	51)
		echo -n "shared and constant memory"
		cp ece408_src/new-forward_5_sharedMemory.cuh ece408_src/new-forward.cuh
		cd ..
		rai -p UIUC_2_1_ECE408_FinalProject --queue rai_amd64_ece408
		cd UIUC_2_1_ECE408_FinalProject
		;;

	*)
		echo -n "Illegal argument!"
		;;

esac