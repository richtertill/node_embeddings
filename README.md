# README

## Authors
This is a survey on Node Embeddings from [Christos Ziakas](https://www.linkedin.com/in/christos-ziakas-5783079a/), [Jan Rüttinger](https://github.com/JanRuettinger), and [Till Richter](https://www.linkedin.com/in/till-richter-659334157/). It was conducted in the [Machine Learning Lab](https://www.in.tum.de/en/daml/teaching/machine-learning-lab/) of the Data Analytics and Machine Learning Group from TUM.
We thank [Oleksandr Shchur](https://www.in.tum.de/en/daml/team/oleksandr-shchur/) for supervising our project and [Prof. Dr. Günnemann](https://www.in.tum.de/en/daml/team/damlguennemann/) for the possibility to conduct research at his group.

## What is the project about?
This project compares different node embeddings with one shared evaluation protocol.

## Architecture of the project

/embedding
- contains the implementation of 3 embedding methods
- contains the implementation of several embedding methods

/evaluation
- contains the implementation of 3 evaluation tasks

/experiment
- contains 3 jupyter notebooks two run and visualize experiments

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
