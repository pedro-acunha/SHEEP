<image src="SHEEP.png" width="400" align="left"/> 

# SHEEP: an ML pipeline for astronomy classification
Photometric redshift-aided classification pipeline using ensemble learning to classify astronomical sources into galaxies, quasars and stars.
<br>
<br>
<br>
<br>
It was built to deal with tabular data, while allowing the use of sparse data.
Our approach uses SDSS and WISE photometry, however others types of data (if tabular) are compatible with our architecture, e.g. radio fluxes/magnitudes.
You can read the paper here: https://arxiv.org/pdf/2204.02080.pdf .

## How to use
This work was made using Python and Jupyter Notebook. For better visualisation and interactivity, use Jupyter Notebook. 
The code is ready to be used and reproduced the results in the paper. If you use it with a different dataset, please remember that the models are optimised for our problem. Do not forget to adapt them to your personal needs.

### Data

To create the data you can use the SQL queries provided in the file "sql_query_casjobs". The data can be retrieved from https://skyserver.sdss.org/CasJobs/ .

All the feature engineering process can be seen in the file "create_data.py". Make the necessary changes, creating and eliminating features, as needed for your project.

### Photo-z predictions

The chain-regressor approach used is described in the file "photo_z.py". 

### Galaxy, QSO and star classification

The two methodologies are described in separate files:
<ul>
  <li> Multi-class: see file "clf_multi.py". 
    <li> One vs all: see file "clf_1vsall.py".
</ul>

The hyper-parameters obtained using FLAML are described. It is recommended to re-run FLAML everytime changes are made in the input data.

## Cite us
Thank you for your interest in the SHEEP pipeline.
If this work was helpful, please do not forget to cite us in your publications.

https://doi.org/10.1051/0004-6361/202243135
