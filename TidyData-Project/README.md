# TidyData-Project

## Project Overview

The goal of this project is to use tidy data principles to clean a dataset and easily perform analysis and visualizations on it. Tidy data follows these principles:

1. Each variable forms a column  
2. Each observation forms a row  
3. Each type of observational unit forms a table  

## Instructions

### 1. Open the notebook
Open the `TidyProject.ipynb` file in Jupyter Notebook or Google Colab by using this [link](https://raw.githubusercontent.com/ldeend/TidyData-Project/refs/heads/main/TidyProject.ipynb). No local file path setup is needed for the `mutant_moneyball.csv` dataset since the it is loaded from GitHub

### 2. Install dependencies
Install the required packages pandas (for data analysis) and matplotlib (for visualizations).

```bash
pip install pandas matplotlib
```

### 3. Run notebook

Finally, run all cells in the notebook. The notebook will load the dataset, clean it to match tidy data principles, and utilize that cleaned dataset to make pivot tables and visualizations.



## Dataset Description

The dataset `mutant_moneyball.csv` contains valuation data for comics featuring different X-Men members across the late 1900s and various sources. The original dataset and more information about it can be found [here from EliCash82](https://github.com/EliCash82/mutantmoneyball/tree/main). It is called `MutantMoneyballOpenData.csv`. 

To get from this dataset to `mutant_moneyball.csv`, the dataset was reduced for this project to only include the columns that follow this format: `TotalValue[Decade]_[Source]`. For example, `TotalValue60s_ebay`, which means the total value of each X-Men team member's total number of issues as reflected by ebay's highest sale of comics released in the 1960s.

## References 

### Tiny Data Principles
For background knowledge and motivation of this project, I read [Tidy Data Principles](https://vita.had.co.nz/papers/tidy-data.pdf) by Hadley Wickham, which covered what tidy data is, why having tidy data is important, and how to make your dataset tidy. 

### Pandas Cheat Sheet
To help in my data cleaning and analysis, I referred to a [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf).

