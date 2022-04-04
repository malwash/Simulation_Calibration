[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/malwash/Simulation_Calibration.git/HEAD)

# Simulation_Calibration
Codebase for the SimCalibration project, which seeks to employ experimental simulation to test, evaluate and calibrate environments/ML pipelines towards Data-generating processes (DGPs)

The primary executable python file runs the meta-simulation workflows or different parameterised ML estimators and structural learners across Linear, Non-linear, Sparse and Dimensional settings (in the form of pipelines) and reports the results and recommendations

1. logistic_regression_pipeline.py
The main python script which first runs the benchmarking of simulation workflows, and second outputs these measures to plots, csv and print stream

2. Simulation_Calibration.ipynb
The python workbook which allows for an interactive full-scale, low sample form of the SimCal benchmarking with graphics, tunable parameters and transparent presentation of code

3. simulation_dagsim.py
Python file pertaining to the generation of simulated data for the real-world using DagSim (https://github.com/uio-bmi/dagsim), this is organised using a pipeline_type parameter passed in from the main python application (logistic_regression_pipeline.py) that organises which data to generate for the current workflow

4. simulation_models.py
Python file which organises the pre-configuring of Machine Learning algorithms (estimators), such as Random Forest, Decision Tree in the main application file. The parameters of each ML method is set at a high-level (i.e., penalty, weight, criterion, solver, kernel) and utilised across each pipeline and regarded as a single workflow

5. simulation_bnlearn.py
Python file pertaining to the generation of simulated data for the learned world using BNLEARN (https://cran.r-project.org/web/packages/bnlearn/index.html) given a pipeline_type. This library provides a number of different algorithms for structural learning, due to different behaviour between python and R implementations, the current version emulates the R functionality using rpy2 as to avoid unconnected edges in a DAG being discarded in the case with the python implementation

5. simulation_notears.py
Python file pertaining to the generation of simulated data for the learned world using NOTEARS (https://github.com/xunzheng/notears)

6. simulation_pgmpy.py
Python file pertaining to the generation of simulated data for the learned world using PGMPY (https://pgmpy.org/)

7. simulation_pomegranate.py
Python file pertaining to the generation of simulated data for the learned world using Pomegranate (https://pomegranate.readthedocs.io/en/latest/)

