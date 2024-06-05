#!/usr/bin/python

import glob
import subprocess

flist = glob.glob("23????/tle_occnn*")
flist.extend(glob.glob("24????/tle_occnn*"))
flist = sorted(flist)
#print(flist)
#exit()

#flist = glob.glob("*/tle_results")

for i,fl in enumerate(flist):
	with open(fl, "r") as f:
		lines = f.readlines()
		print(fl, len(lines))
		for j,l in enumerate(lines):
			if float(l.split()[-2])>20: # and float(l.split()[-2])<15: 
				print(l)
				parts = l.split()
				filename, frame_num = parts[0], parts[1]
				print(f"******* OPENING {filename} on frame {frame_num}, event {i},{j}")
				file_path = fl.split("/")[0]+"/"+filename
				subprocess.check_call([f"etos.py -sf {frame_num} {file_path}"], shell=True)
