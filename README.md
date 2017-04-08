<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Bayesian Model Evaluation and Comparison
## SPS Modeling of Compact Dwarf Galaxies

### Michael Hoffman, Charlie Bonfield, Patrick O'Brien

### ASTR 703: Galastrostats

## Introduction

Stellar population synthesis models are valuable tools for determining how
stellar evolution leads to the galaxy characteristics we observe in nature.  In
light of the recent gravitational wave discovery by LIGO, it is important to
properly model the frequency and behavior of binary stars.  By determining the
best places to look for massive binary progenitors (and thus potential LIGO
event binary black holes), scientists can better predict the frequency of future
gravitational wave detections.  The goal of this project is to compare stellar
population synthesis models with and without binary star evolution and
investigate how these compare with observed data for dwarf galaxies in the
RESOLVE survey.  We will determine the likelihood of models from BPASS by
comparing broadband magnitudes to those from RESOLVE using a Bayesian approach,
resulting in estimates of various parameters (age, metallicity, binary fraction, mass fraction ) after marginalization. We will also provide galaxy mass estimates
using a method mirroring the one used in the RESOLVE survey. Through careful
analysis of our results, we hope to determine how significant a role binary
stars play in the evolution of dwarf galaxies, thereby telling us where to point
(or not to point) our gravitational wave detectors in the future!

### Code Acknowledgements
Conversion from SPS model fluxes to broadband magnitudes heavily based on IDL
code provided by S. Kannappan.

The project members would also like to acknowledge those who created/maintain
numpy, pandas, and any other python packages used in this code.

### Databases
1. RESOLVE: Eckert et al., *Astrophysical Journal*, 810, 166 (2015).
2. BPASSv2: Stanway et al., *Monthly Notices of the Royal Astronomical Society*,
456, 485 (2015).
   BPASSv1: Eldridge & Stanway, *Monthly Notices of the Royal Astronomical
Society*, 400, 1019 (2009).  
   *The project members would also like to acknowledge NeSI Pan Cluster, where
the code for BPASS was run.*


### Table of Contents
1. Loading the Data  
   (1.1) RESOLVE Data  
   (1.2) BPASS (Binary Population and Spectral Synthesis Code) Data
       (1.2a) Model Parameters
2. Model Calculations  
   (2.1) Fraction of Massive Binary Black Holes  
   (2.2) Normalization and Chi-Squared  
   (2.3) Galaxy-Specific Calculations
       (2.3a) Computing the Model-Weighted Parameters for Each Galaxy
       (2.3b) Mass Comparison with RESOLVE
       (2.3c) Cross-Validation and the Stellar Mass Function
       (2.3d) Galaxy Parameters as a Function of Binary Fraction
   (2.4) Bayesian Analysis
       (2.4a) Priors
       (2.4b) Likelihood Function
       (2.4c) Marginalization
3. Method Comparison  
   (3.1) Simple Random Sampling  
   (3.2) Metropolis-Hastings Algorithm  
   (3.3) Slice Sampling and NUTS (Failures)  
4. Fraction of Massive Binary Black Holes  
5. Discussion of Results  
6. References

# 1. Loading the Data

## (1.1) RESOLVE Data

To compare BPASS models with real data, we will be using broadband magnitude
measurements for dwarf galaxies in the RESOLVE survey. (The unit conversion and
data analysis will be outlined in a later section.) First, we download a csv
containing all of the RESOLVE data, opting to have all the data at our disposal
initially before selecting only the columns necessary for our calculations. To
accomplish this task, we run the following SQL query on the private RESOLVE data
site:

```select where name != "bunnyrabbits"```

We use pandas to convert this to a single, centralized DataFrame.  With pandas,
we are able to easily select out the columns for `name`, `grpcz` (for
calculating distance), `logmstar`, and eight magnitudes and their uncertainties.
The bands used in our analysis are those which are included in both the BPASS
models and RESOLVE (J, H, K, u, g, r, i, z).

```python
# Import statements 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astroML.plotting import hist
from astroML.plotting import scatter_contour
import seaborn as sns
import sampyl as smp
from sklearn.neighbors import KernelDensity
import astropy.stats
import scipy.stats as ss
import corner
import random
import time
from IPython.display import Image

# set font rendering to LaTeX 
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=12, usetex=True)

%matplotlib inline

pd.set_option('display.max_columns', None)

# Read in the full RESOLVE dataset. 
resolve_full = pd.read_csv('resolve_all.csv')

# Select only parameters of interest.
resolve = resolve_full[['name','cz','grpcz','2jmag','2hmag','2kmag','umag',
                        'gmag','rmag','imag','zmag','e_2jmag','e_2hmag','e_2kmag',
                        'e_umag','e_gmag','e_rmag','e_imag','e_zmag','logmstar']]
resolve.to_pickle('resolve.pkl')

# Pickle DataFrame for quick storage/loading.
resolve_data = pd.read_pickle('resolve.pkl')

# Display RESOLVE data. 
resolve_data.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>cz</th>
      <th>grpcz</th>
      <th>2jmag</th>
      <th>2hmag</th>
      <th>2kmag</th>
      <th>umag</th>
      <th>gmag</th>
      <th>rmag</th>
      <th>imag</th>
      <th>zmag</th>
      <th>e_2jmag</th>
      <th>e_2hmag</th>
      <th>e_2kmag</th>
      <th>e_umag</th>
      <th>e_gmag</th>
      <th>e_rmag</th>
      <th>e_imag</th>
      <th>e_zmag</th>
      <th>logmstar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>rf0001</td>
      <td>6683.8</td>
      <td>6683.8</td>
      <td>15.84</td>
      <td>-99.00</td>
      <td>-99.00</td>
      <td>18.64</td>
      <td>17.69</td>
      <td>17.32</td>
      <td>17.15</td>
      <td>17.05</td>
      <td>0.24</td>
      <td>-99.00</td>
      <td>-99.00</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.12</td>
      <td>8.51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rf0002</td>
      <td>6583.8</td>
      <td>6583.8</td>
      <td>14.53</td>
      <td>14.27</td>
      <td>12.72</td>
      <td>18.62</td>
      <td>17.08</td>
      <td>16.33</td>
      <td>15.94</td>
      <td>15.67</td>
      <td>0.09</td>
      <td>0.29</td>
      <td>0.44</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>9.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rf0003</td>
      <td>5962.9</td>
      <td>5962.9</td>
      <td>15.33</td>
      <td>14.72</td>
      <td>14.35</td>
      <td>18.89</td>
      <td>17.72</td>
      <td>17.07</td>
      <td>16.78</td>
      <td>16.51</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>0.09</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>8.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rf0004</td>
      <td>6365.9</td>
      <td>6365.9</td>
      <td>13.24</td>
      <td>13.09</td>
      <td>14.11</td>
      <td>16.20</td>
      <td>15.20</td>
      <td>14.61</td>
      <td>13.80</td>
      <td>14.27</td>
      <td>0.51</td>
      <td>0.15</td>
      <td>0.98</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.09</td>
      <td>9.67</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rf0005</td>
      <td>5565.4</td>
      <td>5565.4</td>
      <td>11.48</td>
      <td>10.75</td>
      <td>10.54</td>
      <td>15.90</td>
      <td>14.21</td>
      <td>13.40</td>
      <td>13.00</td>
      <td>12.73</td>
      <td>0.05</td>
      <td>0.13</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>10.67</td>
    </tr>
  </tbody>
</table>
</div>

To select dwarf galaxies from the RESOLVE dataset, we use the following criteria:

* Galaxy stellar mass less than \\(10^{9.5} M_{\odot}\\) (`logmstar` < 9.5)

(This is the threshold scale and was recommended by Dr. Kannappan in a private correspondence.)

Now, our DataFrame is reduced to only the columns we need: eight magnitudes and their uncertainties, as well as `name`, `logmstar`, and `grpcz`.

