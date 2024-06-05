#!/usr/bin/python
# Checks if the current ML can find all the elves that has been foundd so far

import os

batch_size=8

#for i in (2,3,4,5,6,7,8,9,10,11,12,13,14,15)+tuple(range(20,110,10)):
for i in (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)+tuple(range(20,110,10)):
	# Check if the result exists
	if not os.path.exists(f"/home/lewhoo/workspace/minieuso_elves_cnn/pytorch/res_batchsize{batch_size}/curve_snapshot_epoch{i}.model"):
		continue

	# Run the search in the directory containing all the found elves files
	os.chdir("/home/lewhoo/data/Mini-EUSO/found_elves_files")
	#"""
	if os.path.exists("tle_occnn_results.txt"):
		os.remove("tle_occnn_results.txt")
	os.system(f"/home/lewhoo/workspace/minieuso_elves_cnn/pytorch/find_elves_occnn_wholedir.py /home/lewhoo/workspace/minieuso_elves_cnn/pytorch/res_batchsize{batch_size}/curve_snapshot_epoch{i}.model")
	#"""

	# Open the file with detected elves
	fdet = open("/home/lewhoo/data/Mini-EUSO/found_elves_files/tle_occnn_results.txt", "r")
	ldet = [(l.split()[0], int(l.split()[1])) for l in fdet]

	# Open the file with all elves
	fall = open("/home/lewhoo/workspace/minieuso_elves_cnn/pytorch/all_elves.txt", "r")
	lall = [(l.split()[0], int(l.split()[1])) for l in fall]

	#print(len(lall), len(set(lall) & set(ldet)))

	print(i, f"Found {len(set(lall) & set(ldet))} of {len(lall)} elves, which gives efficiency of {len(set(lall) & set(ldet))/len(lall)}, bg {len(set(ldet)-set(lall))}")
	print("Not found elves:", set(lall)-set(ldet))
	
	with open(f"/home/lewhoo/workspace/minieuso_elves_cnn/pytorch/res_batchsize{batch_size}/res_b{batch_size}", "a") as f:
		print(i, len(set(lall) & set(ldet))/len(lall), len(set(ldet)-set(lall)), file=f)
	#exit()


