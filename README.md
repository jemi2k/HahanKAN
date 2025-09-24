# HaKAN
**Time Series Forecasting with Hahn Kolmogorov-Arnold Networks**

This repository contains the code for our project on long-term time series forecasting.

### Requirements
<pre><code>pip install numpy==1.24.3 matplotlib==3.7.2 pandas==2.0.3 scikit-learn==1.3.0 torch==2.4.1+cu121 </code></pre>

### Datasets
------
The datasets are hosted on Google Drive by Autoformer. Please download them and place them in the `./datasets/` directory before running the experiments.

👉 [Access the Datasets on Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)

### Experiments
If you want to run an experiment, just run the following script and edit it as you need. This script is for the look-back window is 96.
<pre><code>sh ./scripts/SHORT/etth1.sh</code></pre>
If you want to run an experiment for the look-back window 336, you should run the following script:
<pre><code>sh ./scripts/LONG/etth1.sh</code></pre>


