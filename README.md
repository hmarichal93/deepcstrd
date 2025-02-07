# DeepCS-TRD
DeepCS-TRD, a Deep Learning-based Cross-Section Tree Ring Detector in Macro images of Pinus taeda and Gleditsia triacanthos.

***
<img src="assets/deepCS-TRD_pinus2.png" alt="Example input image and detected tree rings"/>

<img src="assets/deepCS-TRD_pinus.png" alt="Example input image and detected tree rings"/>

<img src="assets/deepCS-TRD_gleditsia.png" alt="Example input image and detected tree rings"/>

***

## Setup:
### Set conda environment 
```bash
conda env create -f environment.yml
conda activate deep_cstrd
pip install -r requirements.txt
```

### Install dependencies
1) CS-TRD
```bash 
git clone https://github.com/hmarichal93/cstrd_ipol.git
cd cstrd_ipol/
pip install .
cd .. 
```
2) UruDendro
```bash
git clone https://github.com/hmarichal93/uruDendro.git
cd uruDendro/
pip install .
```

### Test
Results should appear in the output/F02c folder
```bash
python main.py inference
```

Or use GitHub Codespaces: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=dev&repo=574937325&machine=standardLinux32gb&location=WestEurope)
