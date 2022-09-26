# Airbnb-Host-Score-Prediction
 Brainstation Data Science Capstone Project

To start, for this project all environments needed will be in the environments folder. 

Each notebook that was used in the final iteration of the project will have a number indicating it's place in the workflow, for example: 02-Exploratory Data Analysis is the second step in the workflow, after 01-NYC Data cleaner.

The purpose of each notebook is as follows:

01-NYC Data Cleaner
	This notebook is meant to be plug and play, meaning that you specify a file at the beginning, specify a filename to be outputted, and run the entire notebook at once. This notebook will only work with .csv files from the data source
	
	
02-Exploratory Data Analysis
	This notebook includes all of the plots and visuals used in the EDA process. Most of the plots are done with plotly, their html files can be found in the plotly_charts folder. This notebook also contains some minor data cleaning as well.
	
	
03-Encoding and Preprocessing
	This notebook takes in the file output by the previous notebook and performs a series of encoding operations to prep the data for the machine learning models. There are some charts here, but they are minimal
	
04-Modeling
	This notebook contains all of the models that were run for the binary classification problem. There are other modeling notebooks. Those can be found in Supplementary-materials/Failed-Modeling-Notebooks. These notebooks do not have as much markdown and code comments as the main ones. I just didn't have the time to format them correctly.
	
clean_data folder:
	This folder contains the output files of the 01-NYC Data Cleaner notebook
	
environments folder:
	Contains .yml files that need to be installed to run all notebooks but the 03-Encoding and 	Preprocessing notebook
	
Modeling_Datasets folder:
	This folder contains the output of the 03-Encoding and Preprocessing notebook. Note the different filenames for different modeling applications.
	
plotly_charts folder:
	Folder containing all plotly charts from the project saved as .html files per project instructions
	
	
raw_data folder:
	This folder contains the files that were fed into the 01-NYC Data Cleaner notebook. They can also be directly downloaded from the InsideAirbnb website
	
Supplementary-Materials folder:
	This folder contains a bunch of items that I have started, but didn't have time to complete within the time constraints of this project. That means a lot of uncommented code, I apologize. Within this folder are the following:
	
- Failed-Modeling-Notebooks: previous modeling attempts that just didn't work out. They are not as pretty as the notebooks in the main folder

- Python Scripts: in here is two scripts, datacleaner.py is an automated form of the datacleaning notebook, and it's cleaned files can be found in the other test_script_data subfolder within this folder. The preprocessor.py file is an attempt to automate the third notebook in the main file, but I did not have time to finish it.

- streamlit: app.py is a file that is the start of deploying this project/model to a streamlit app

- test_script_data: The output of datacleaner.py mentioned above. Files are for multiple cities around the US that may be used in future iterations of the project
	
	
	
	
	
	