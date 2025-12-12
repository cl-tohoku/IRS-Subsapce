# IRS-Subsapce
Codes and datasets for our paper **"Indexed Relational Schema: How Do LLMs Encode Discourse?"**

### Setup
Install dependencies (creating a virtual environment called "finetuning") :
~~~~
conda env create -f environment.yml
conda activate finetuning
~~~~


### Activation Extraction
Run the following script for extracting attribute activations for IRS subspace analysis.
~~~
python script/activation_extraction.py
~~~

### Sampling IRS Subspace
Run the following script for sampling IRS subspace.
~~~
python script/activation_sampling.py
~~~

### Perturbing IRS Subspace
Run the following script for perturbing IRS subspace.
~~~
python script/activation_perturbing.py
~~~

### Steering IRS Subspace
~~~
python script/activation_steering.py
~~~
