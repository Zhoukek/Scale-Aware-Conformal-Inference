# Conformal Optimistic Prediction

This repository contains the data and code required to reproduce the results in paper **"Scale-Aware Conformal Inference: Adapting to Score Function Dynamics in Time Series Forecasting"**.


## Environment
<p>
Please clone this repo and run following command locally for install the environment:
<pre>
conda create --name SACI python=3.10 -y
conda activate SACI
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
</pre>
</p>



## Demo

Please run following command locally for local test:
<pre>
cd tests
python base_test.py configs/AMZN_test.yaml
python base_plots.py results/AMZN_test.pkl
</pre>

The plot results will be saved in <code>test/plots</code> folder. More commands can be seen in <code>tests/expbook.ipynb</code>. Users can modify the YAML files in the <code>configs</code> folder to configure the experimental settings.


## Implemented Methods

In addition to our proposed method, this repository provides implementations of several state-of-the-art online conformal prediction algorithms for benchmarking and reproducibility. The detailed information can be seen in our paper.

- ACI (Adaptive Conformal Inference)

- OGD (Online Gradient Descent)

- SF-OGD (Scale-Free OGD)

- Decay-OGD (OGD with decaying step sizes)

- Conformal PID (proportional-integral-derivative)

- ECI (Error-quantified Conformal Inference)

- LQT (Linear Quantile Tracking)

- COF（Conformal Optimistic Prediction）

## Acknowledgements

This project is built upon the excellent framework
[ECI](https://github.com/creator-xi/Error-quantified-Conformal-Inference) and [COP](https://github.com/creator-xi/Conformal-Optimistic-Prediction).
We thank the authors for making their work publicly available.



