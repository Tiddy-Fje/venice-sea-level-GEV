# Sea level extremes in Venice

## Project overview  

The city of Venice and its lagoonal ecosystem are vulnerable to changes in sea levels. The occurrences of high tides (acqua alta) has an increasing trend, usually attributed to global warming. 

In this report, we analyze data from sea levels in Venice with the goal of assessing the significance of severity increase over recent years. We use a Generalized Extreme Value (GEV) model to fit the data and compute its associated return levels. We then compare these to the extreme events that occurred between 2013 and 2023. 

See report.pdf for a detailed analysis and discussion of the project. It includes a quick exploration of the dataset, model selection and validation, as well as a risk analysis.

## Running the code 

To manage the `python` packages needed to run the files `conda` was used. The `requirements.yml` file can be used to create the associated environment easily as `conda create --n <env-name> --file <relative-path-to-this-file>` (or using similar non-`conda` commands).

`exploration.py` is used to explore the data using the `pyextremes` package. `implementation.py` implements the functions needed to fit GEVs with time varying parameters (not supported in `pyextremes`). The results were validated on the constant model, for which there is support in the `pyextremes` package. 

Finally, `results.py` generates the figures and p-values associated with the analysis. It can be run (analogously to `exploration.py`) as `python <filename>.py` in the command line. 

## Generating the report PDF
To generate a pdf of the report, standard `LaTeX` command line prompts of your local machine apply. 

## Acknowledgements
This project was developed as part of the Applied Statistics course at EPFL. I thank Dr. Linda Mhalla for providing the project statement and initial guidance. I also thank Laurent Brugnard and Rayan Harfouche for valuable discussions. 

## Note about Git history
This project was initially submitted to a Github classroom repo. This version is a copy of that submission that is posted on my personal Github account. I could unfortunately not recover the original Git history.  