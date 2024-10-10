Sampath Rapuri srapuri1
William Li wli128

JHU Computer Integrated Surgery
Fall 2024
Programming Assignment 1

DataIO.py: Handles reading and writing of calibration data files, including parsing various input formats.

Debug.py: Contains utility functions for debugging and generating synthetic test data for validation.

LinAlg_test.py: Unit testing module for of all linear algebra and transformation operations used in this assignment

LinAlg.py: Provides linear algebra functions and classes for vector and frame operations, transformations, and registrations.

PA1_Final.py: Main executable script that solves the problems outlined in Assignment 1, including running the full calibration pipeline.

PivotCalibration.py: Implements methods for performing pivot calibration to compute the probe tip and dimple positions based on given tracker data.


To run the code: 
conda env create -f environment.yaml
conda activate pa1_env

Then to run the PA1_Final.py run  the following command:

python PA1_Final.py

This will generate all outputs from pa1-debug-a-output1.txt to pa1-unknown-k-output1.txt


To run unit tests contained in LinAlg_test.py:

python LinAlg_test.py