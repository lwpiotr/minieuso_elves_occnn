# minieuso_elves_occnn

1. To look for elves, inside a directory with Mini-EUSO root files call

find_elves_bigsample

it will produce tle_occnn_results_bigsample0.txt file with filename, frame and distance to the ELVESs hyperspace centre. The smaller the distance (usually <0.1), the bigger the chance it is an ELVES. But sometimes ELVESs have also pretty high distances...

2. Another model to look for ELVESs:

find_elves_smallsample

this model in principle gave worse result than the one above, but it was also tested to generalise quite well, while the one above wasn't.

3. Sun filtering of the found ELVES

A lot of found events are not ELVESs but sun influenced data. If you are operating on S1 data, one can create filtered lists of ELVESs with 

filter_sun_tles_occnn.py

4. Browing through the elves that were found with etos: modify the file below to suit your directories:

browse_tles_tmp.py

4. Training is far more complicated

a. You need to generate samples from Mini-EUSO data and augument them like in generate_samples.py
b. You need to run training (after modyfying absolute paths and the initial model) with train_elves_occnn.py
