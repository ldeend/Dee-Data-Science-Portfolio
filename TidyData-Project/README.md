# TidyData-Project

## Project Overview

The goal of this project is to use tidy data principles to clean a dataset and easily perform analysis and visualizations on it. Tidy data follows these principles:

1. Each variable forms a column  
2. Each observation forms a row  
3. Each type of observational unit forms a table  

## Instructions

### 1. Open the notebook
Open the `TidyProject.ipynb` file in Jupyter Notebook or Google Colab from [here](https://github.com/ldeend/Dee-Data-Science-Portfolio/blob/main/TidyData-Project/TidyProject.ipynb). No local file path setup is needed for the `mutant_moneyball.csv` dataset since the it is loaded from GitHub

### 2. Install dependencies
Install the required packages pandas (for data analysis) and matplotlib (for visualizations).

```bash
pip install pandas matplotlib
```

### 3. Run notebook

Finally, run all cells in the notebook. The notebook will load the dataset, clean it to match tidy data principles, and utilize that cleaned dataset to make pivot tables and visualizations.



## Dataset Description

### Dataset Background
The dataset `mutant_moneyball.csv` contains valuation data for comics featuring different X-Men members across the late 1900s and various sources. The original dataset and more information about it can be found [here from EliCash82](https://github.com/EliCash82/mutantmoneyball/tree/main). It is called `MutantMoneyballOpenData.csv`. 

To get from this dataset to `mutant_moneyball.csv`, the dataset was reduced for this project to only include the columns that follow this format: `TotalValue[Decade]_[Source]`. For example, `TotalValue60s_ebay`.

### Variable names explained

Members is straightforward and contains either the superhero name of the X-Man, such as `longshot`, or the civilian name of the X-Man, such as `warrenWorthington`. No X-Men are included more than once. Source idicates where the valuation came from, whether it is ebay, oStreet, Wiz, or Heritage. Similarily, Decade indicates which timeframe the comic came from: 60s (1963-1969), 70s (1970-1979), 80s (1980-1989), 90s (1990-1992). 

The term value means something slighlty different for each source, so for more detail again go to [the original repo](https://github.com/EliCash82/mutantmoneyball/tree/main). In general, the term value means total sales/highest sale. An example column `TotalValue60s_ebay` means the total value of each X-Men team member's total number of issues released between 1963 and 1969 as reflected by ebay sales in 2022 in which sellers tagged the issue as VG (Very Good) Condition.


## References 

### Tiny Data Principles
For background knowledge and motivation of this project, I read [Tidy Data Principles](https://vita.had.co.nz/papers/tidy-data.pdf) by Hadley Wickham, which covered what tidy data is, why having tidy data is important, and how to make your dataset tidy. 

### Pandas Cheat Sheet
To help in my data cleaning and analysis, I referred to a [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf).

