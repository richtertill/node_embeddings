# README

## What is the project about?
This project compares different node embeddings with one shared evaluation protocol.
More information can be found on the [wiki] (https://wiki.tum.de/display/mllab/Project+5%3A+Node+embedding+survey).

## Architecture of the project

/embedding
- contains the implementation of 3 embedding methods
- contains the implementation of several embedding methods

/evaluation
- contains the implementation of 3 evaluation tasks

/experiment
- contains 2 jupyter notebooks two run and visualize experiments

/gust
- contains a helper library developed by the chair to load and preprocess data

/utils
- contains helper code

## How to run the code?
1. Install all requirements  
`pip install -r requirements.txt`

2. Define and run an experiment  
Open the jupyter notebook `Experiment_pipeline.ipynb` in the folder "experiments" and follow its instructions.

3. Visualize the results  
Open the jupyter notebook `Visualize_Results.ipynb` in the folder "experiments" and follow its instructions.