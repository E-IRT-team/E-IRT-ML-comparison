This is GitHub folder for reproducing the results from:
"Comparing the Prediction Performance of Item Response Theory and Machine Learning Methods on Item Responses for Educational Assessments"

preprequisites:

Python >= 3.7.7 and standard libraries such as os, numpy, pandas, sklearn.
The subprocess library is needed to run the EIRT.py script

R >= 4.0.2 and the lme4 library

For reproducing results from the ML methods, simply run the corresponding Python scripts
For reproducing the EIRM results, the EIRT.py script will run a wrapper around the EIRT.R script. Running time might be significantly higher.

The output will be stored int eh /results folder both on a .txt and .npy format. To facilitate results collection, 
a print_results.py is provided. The user can decide which methods to comapre with a Friedman_Nemenyi test.

Finally, the test can be run through the Friedman_Nemenyi_test.R script, the significance plots will also be ouputted


For questions about the paper and th EIRT methodology, you can contact the corresponding author
JinHo Kim at jinhokim@uso.ac.kr

For questions about the python code, you can contact the 2nd name author, Klest Dedja, at
klest.dedja@gmail.com

