### Code anda data for: Anticipated versus Actual Effects of Platform Design Change: A Case Study of Twitter's Character Limit

### Repository structure

------------

    ├── README.md          <- The top-level README
    ├── constants.py       <- Project constants: data paths, data sources
    ├── data               
    │   ├── measurements   <- Intermediate .csv data files with cramming and runover measurements created with scripts in `src`
    │   └── batches        <- Raw data used to create .csvs in measurements, check out instructions below on how to download the files
    │
    ├── notebooks          <- Jupyter notebooks with code to plot figures from the paper
    │   ├── Fig1: Diagram.ipynb
    │   ├── Fig2: Cramming.ipynb
    │   ├── Fig3: POS_analysis.ipynb
    │   ├── Fig4: POS_analysis.ipynb
    │   ├── Fig5: Topics LIWC v2.ipynb
    │   ├── Fig6: Diagram.ipynb
    │   ├── Fig7: Diagram.ipynb
    │   ├── Fig8: Cramming.ipynb
    │   ├── Fig9: Cramming.ipynb
    │   ├── FigS1:  FigS1a.ipynb and FigS1b.ipynb
    │   ├── FigS2:  FigS2.ipynb
    │   ├── FigS3:  FigS3.ipynb
    │   ├── FigS4:  Cramming.ipynb
    │   └── FigS5:  Cramming.ipynb
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code used to create .csv data files in data/measurements.
    └──
------------

###  System requirements and installation guide

We recommend a local installation of new Python virtual environment. The code was tested on Ubuntu 18.04.
Please use the packages versions provided in requirements.txt

1. To obtain the raw data, go to our [Zenodo repository](https://zenodo.org/record/7009935), download 40 batch files and place them under `data/batches`.

2. Open the terminal. To avoid any incompatibility issue,
 please create a new virtual environment. This project was created using [virtualenvwrapper](]https://virtualenvwrapper.readthedocs.io/en/latest/)

`pip install virtualenvwrapper` <br>
`mkvirtualenv tweets -r requirements.txt -p python3.7` <br>

The environment should be activated automatically, if not use: <br>
`workon tweets`

To deactivate the environment simply use: <br>
`deactivate`

3. Start a Jupyter notebook server if you want to reproduce the plots. <br>
`cd notebooks` <br>
`jupyter notebook`

4. To regenerate the intermediate .csv files you can do it using scripts from `src`.


### Cite us

