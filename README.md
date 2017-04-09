<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

[back to main page](https://michael-hoffman.github.io)
# Bayesian Model Evaluation and Comparison
## SPS Modeling of Compact Dwarf Galaxies
### Michael Hoffman, Charlie Bonfield, Patrick O'Brien


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

```python
# Select the dwarf galaxies.
resolve = resolve_data[(resolve_data['logmstar'] < 9.5)]

print 'Number of galaxies in RESOLVE matching our dwarf galaxy criterion: ', resolve.shape[0]
```

    Number of galaxies in RESOLVE matching our dwarf galaxy criterion:  1657

Missing photometric data in RESOLVE is indicated by `-99`. Since we do not wish to include these values in our subsequent data analysis, we treat them by ignoring all instances of `-99`. We also exclude data from these bands when comparing to our BPASS model fluxes.

*We perform this process externally, but the relevant code has been extracted from our pipeline and is shown below.*  

```python
errs =[]
    for eband in emag_list:
        if (gal[eband] != -99.0):
            errs.append(gal[eband])
    num_good_values = len(errs)

    # add 0.1 mag to all errors Kannappan (2007)
    errs = np.array(np.sqrt(np.array(errs)**2 + 0.1**2))

    gmags = []
    for gband in rmag_list:
        if (gal[gband] != -99.0):
            gmags.append(gal[gband])
gmags = np.array(gmags)
```
## (1.2) BPASS (Binary Population and Spectral Synthesis Code) Data  

BPASS is a stellar population synthesis (SPS) modeling code that models stellar populations that include binary stars.   This is a unique endeavor in that most SPS codes do not include binary star evolution, and it makes BPASS well-suited for our hunt for massive binaries in RESOLVE dwarf galaxies.

**Advantages:  **
1. Includes data for separate SPS models for populations of single and binary star populations.  
2. Data files are freely available and well-organized.  
3. Both spectral energy distributions (SEDs) and broadband magnitudes are available. 

**Disadvantages:  **
1. Cannot provide any input to the code - you are stuck with the provided IMF slopes, ages, and metallicities.  
    * In the context of our project, this means that our model grid has already been defined.  
2. Binary star populations are not merged with single star populations - you have to mix the two on your own.  
    * We will be mixing binary and single star populations in varying proportions.

We read in all of the data from the BPASS models and store it as a pandas DataFrame. <br/>

### (1.2a) Model Parameters

We have three parameters from BPASS that characterize each model:

- Age: 1 Myr (\\(10^6\\) years) to 10 Gyr (\\(10^{10}\\) years) on a logarithmic scale
    - Stored as `log(Age)`, ranging from 6.0 to 10.0.
- Metallicity: \\(Z = 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.014, 0.020, 0.030, 0.040\\)
    - Stored as: `Metallicity`
- IMF Slopes (for ranges of \\(M_{*}\\) in units of \\(M_{\odot}\\)):  
    - \\(0.1 < M_{\textrm{galaxy}} < 0.5\\): -1.30, -2.35
        - Stored as: `IMF (0.1-0.5)`
    - \\(0.5 < M_{\textrm{galaxy}} < 100\\): -2.0, -2.35, -2.70
        - Stored as: `IMF (0.5-100)`
    - \\(100 < M_{\textrm{galaxy}} < 300\\): 0.0, -2.0, -2.35, -2.70 
        - Stored as: `IMF (100-300)`
                
We will also be calculating a fourth parameter, \\(f_{MBBH}\\), to go along with each model, which we will discuss in further detail below. 

![IMF](DG_SPSv4_files/DG_SPSv4_10_0.png)



The image above shows a number of popular initial mass functions. The BPASS initial mass functions most closely resemble `Kroupa01`, in that we have three linear functions that are defined over different mass ranges and stitched together at the boundaries of each interval. We believe this to be a practical choice of IMF, and it is simple to integrate later on in the code to determine $f_{MBBH}$.  

**Note: The BPASS models that we use contain a single simple stellar population** (delta-function burst in star formation at time zero). We predict that this will limit the degree to which the models match the observed data, and we believe incorporating models with continuous star formation (which are available from BPASS) represents an area of further study.  

*Due to the nature of the data (there are many BPASS data files placed in a set of seven folders), we perform this process externally and pickle the resulting DataFrame. For the interested members of the audience, the data files may be found under "BROAD-BAND COLOURS - UBVRIJHKugriz" at http://bpass.auckland.ac.nz/2.html. Locally, we have stored these files in our `DataRepo` folder.* 

```python
# ind: number of IMF slope combinations (number of blue links on BPASS website)
for i in range(len(ind)):
        
    os.chdir('/afs/cas.unc.edu/classes/fall2016/astr_703_001/bonfield/group_project/
              BPASS_mags/BPASSv2_'+ind[i]+'/OUTPUT_POP/'+key+'/')
    # print os.getcwd()     # check that we are moving through directories
        
    dat_files = glob.glob('*.dat')
    mod1,mod2,mod3,mod4,mod5,mod6,mod7 = [],[],[],[],[],[],[]
    df_sub = [mod1, mod2, mod3, mod4, mod5, mod6, mod7]
    df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11 = [],[],[],[],[],[],[],[],[],[],[]
    df_arr = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11]
        
    df_sub[i] = pd.DataFrame()
    
    for j in range(len(dat_files)):
            
        bpass = np.genfromtxt(dat_files[j],delimiter='',dtype=float)
        # pull out metallicity from data file name (start/stop are index positions)
        if key == 'binary':
            start = 13
            stop = 16
        elif key == 'single':
            start = 9
            stop = 12
        metallicity = float(dat_files[j][start:stop])/1000.
            
        age = bpass[:,0]
        metallicity = metallicity*np.ones(len(age))
        imf_1 = imf1[i]*np.ones(len(age))         
        imf_2 = imf2[i]*np.ones(len(age))         
        imf_3 = imf3[i]*np.ones(len(age))
            
        df_arr[j] = pd.DataFrame()            
            
        df_arr[j]['IMF (0.1-0.5)'] = pd.Series(imf_1)
        df_arr[j]['IMF (0.5-100)'] = pd.Series(imf_2)
        df_arr[j]['IMF (100-300)'] = pd.Series(imf_3)
        df_arr[j]['Metallicity'] = pd.Series(metallicity) 
        df_arr[j]['log(Age)'] = pd.Series(age)
        df_arr[j]['u'] = pd.Series(bpass[:,10])
        df_arr[j]['g'] = pd.Series(bpass[:,11])    
        df_arr[j]['r'] = pd.Series(bpass[:,12])    
        df_arr[j]['i'] = pd.Series(bpass[:,13])
        df_arr[j]['z'] = pd.Series(bpass[:,14])
        df_arr[j]['J'] = pd.Series(bpass[:,7])
        df_arr[j]['H'] = pd.Series(bpass[:,8])
        df_arr[j]['K'] = pd.Series(bpass[:,9])
```


```python
# Read in the BPASS model data 
bpass_single = pd.read_pickle('bpass_sin_mags.pkl')
bpass_binary = pd.read_pickle('bpass_bin_mags.pkl')

# Display data frame for binary models as example 
bpass_binary.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IMF (0.1-0.5)</th>
      <th>IMF (0.5-100)</th>
      <th>IMF (100-300)</th>
      <th>Metallicity</th>
      <th>log(Age)</th>
      <th>u</th>
      <th>g</th>
      <th>r</th>
      <th>i</th>
      <th>z</th>
      <th>J</th>
      <th>H</th>
      <th>K</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.3</td>
      <td>-2.0</td>
      <td>-2.0</td>
      <td>0.001</td>
      <td>6.0</td>
      <td>-16.72410</td>
      <td>-15.25659</td>
      <td>-14.97677</td>
      <td>-14.81755</td>
      <td>-14.64745</td>
      <td>-14.35990</td>
      <td>-14.23688</td>
      <td>-14.12323</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.3</td>
      <td>-2.0</td>
      <td>-2.0</td>
      <td>0.001</td>
      <td>6.1</td>
      <td>-16.89995</td>
      <td>-15.43458</td>
      <td>-15.15283</td>
      <td>-14.99264</td>
      <td>-14.82397</td>
      <td>-14.54499</td>
      <td>-14.42667</td>
      <td>-14.31568</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.3</td>
      <td>-2.0</td>
      <td>-2.0</td>
      <td>0.001</td>
      <td>6.2</td>
      <td>-17.12357</td>
      <td>-15.66747</td>
      <td>-15.39094</td>
      <td>-15.23271</td>
      <td>-15.06762</td>
      <td>-14.79439</td>
      <td>-14.68084</td>
      <td>-14.57534</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.3</td>
      <td>-2.0</td>
      <td>-2.0</td>
      <td>0.001</td>
      <td>6.3</td>
      <td>-17.60489</td>
      <td>-16.40239</td>
      <td>-16.45614</td>
      <td>-16.59758</td>
      <td>-16.69958</td>
      <td>-16.89289</td>
      <td>-17.24655</td>
      <td>-17.36608</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.3</td>
      <td>-2.0</td>
      <td>-2.0</td>
      <td>0.001</td>
      <td>6.4</td>
      <td>-17.63741</td>
      <td>-16.54304</td>
      <td>-16.77804</td>
      <td>-17.05414</td>
      <td>-17.24676</td>
      <td>-17.56112</td>
      <td>-18.00511</td>
      <td>-18.15740</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Model Calculations

For each model under consideration, we have *four* parameters that provide us with our model grid for BPASS:   

- Age
- Metallicity
- IMF Slopes (proxy for \\(f_{MBBH}\\))
- \\(\alpha\\) (binary fraction) : 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    - We mix the BPASS models in a manner such that \\(\alpha\\) of our population is binary stars, while \\((1-\alpha)\\) 
      is single stars. 

```python
# combine magnitudes based on alpha (binary fraction)
def combine_mags(single,binary,alpha):
    
    single = np.array(single)
    binary = np.array(binary)
    return -2.5 * np.log10((alpha * 10**(-0.4*binary))+(1. - alpha) * (10**(-0.4*single))) 
```

Each combination of these parameters defines a unique model with a set of model magnitudes. We must first scale these values to put ourselves on the same footing as our RESOLVE galaxies, with the normalization providing an estimate of the galaxy mass. Before we do, however, we will discuss how we compute \\(f_{MMBH}\\). 

## (2.1) Fraction of Massive Binary Black Holes

The fundamental science question that we address in this project pertains to the fraction of massive binary black holes in dwarf galaxies (\\(f_{MBBH}\\)) that would be massive enough to emit gravitational waves that could be detected by Earth-based gravitational wave detectors. For each model, we can calculate this parameter using the BPASS initial mass functions and alpha parameter.

We define the following piecewise function for the initial mass functions from BPASS:

$$
IMF(M_*) = 
   \begin{cases} 
      aM_* + C_1, & 0.1M_{\odot} \lt M_* \lt 0.5M_{\odot} \\
      bM_* + C_2, & 0.5M_{\odot} \lt M_* \lt 100M_{\odot} \\
      cM_* + C_3, & 100M_{\odot} \lt M_* \lt 300M_{\odot} 
   \end{cases},
$$

where \\((a,b,c)\\) are the IMF slopes in the corresponding intervals (provided by BPASS) and \\((C_1,C_2,C_3)\\) are unknown constants.

Making use of the mass normalization condition (all IMFs are normalized to \\(10^6 M_{\odot}\\)) and boundary conditions, we are able to solve for the unknown constants. Once the system is solved, we can integrate the first moment of the IMF from \\(90M_{\odot}\\) (progenitor cutoff is from Belczynski et al.) to \\(300M_{\odot}\\) in order to find the fraction of (potential) massive binary black holes in our binary star population. To find \\(f_{MBBH}\\) (which is the fraction of massive binary black holes in the *total* population), we merely multiply the result from evaluating the integral under the curve by the binary fraction, \\(\alpha\\).

In summary, we walk away with \\(f_{MBBH}\\) for each unique combination of IMF slopes and alpha (70).

```python
"""
Function to calculate fraction of massive progenitors in binaries. 
Uses IMF slopes and alpha (binary fraction). The numerator is the integral 
of the binary IMF integrated from 90 M_sun (massive progenitor cutoff from 
Belczynski et. al. 2016) through 300 M_sun. The denominator is the 
normalized mass of the model (10^6 M_sun). 
"""
def fbin_calc(alpha, a, b, c):
    import scipy
    
    def intercept2(a, b, c):
        return ((1./(0.5*(300.**2-0.1**2)))*(10**6 - (1./3.)*a*(0.5**3 - 0.1**3) - (1./3.)*b*(100**3 - 0.5**3) 
               - (1./3.)*c*(300**3 - 100**3) - (b-a)*0.5*(0.5*(0.5**2-0.1**2)) - (b-c)*(100.0)*(0.5*(300**2-100**2))))
    
    def imf2(x):
        return (b*x + yint2)*alpha
        
    def imf3(x):
        return (c*x + yint3)*alpha

    yint2 = intercept2(a, b, c)
    yint1 = (b-a)*(0.5) + yint2
    yint3 = (b-c)*(100.) + yint2
    
    a1 = float(scipy.integrate.quad(imf2, 90, 100)[0])
    a2 = float(scipy.integrate.quad(imf3, 100, 300)[0])
    
    f_bin = (a1+a2)/(10.**6)
    
    return f_bin
```

## (2.2) Normalization and Chi-Squared
- BPASS models are based on instantaneous formation of stars with a total mass of \\(10^6 M_{\odot}\\).
- Since galaxies in RESOLVE contain far more mass than \\(10^6 M_{\odot}\\), we must normalize. To do so, we assume that the uncertainties on the model magnitudes are Poissonian when converted to fluxes, then perform the normalization in flux units (this is the natural linear scale for light output).

```python
# convert RESOLVE absolute mags to fluxes
gfluxs = 10**(-0.4*gmags)

# upper - lower bound on flux error
hfluxs = 10**(-0.4*(gmags+errs))
lfluxs = 10**(-0.4*(gmags-errs))
errs = 0.5*(hfluxs - lfluxs)
```

We define the normalization to be the scale factor that minimizes \\({\chi}^2\\), where \\({x_i}\\) are the model fluxes, \\({d_i}\\) are the observed (RESOLVE) fluxes, \\({\epsilon_i}\\) are the observed (RESOLVE) uncertainties, and \\(i\\) is the band index. We display the result below:  

$$c = \frac{\sum{\frac{x_i d_i}{\epsilon_i^2}}}{\sum{\frac{x_i^2}{\epsilon_i^2}}} $$  

