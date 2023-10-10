# DeepLigType: Predicting Ligand Types of Protein-Ligand Binding Sites Using a Deep Learning Model

The paper "DeepLigType: Predicting Ligand Types of Protein-Ligand Binding Sites Using a Deep Learning Model" is currently under review. More details will be provided shortly. Stay tuned!


[Link - Download Files](https://drive.google.com/drive/folders/1WVIpv6CbHBnRcE_0qR8E9AEwu7-CGgCZ?usp=sharing) 

**_Requirements_**

To run the program, you'll need an operating system with a Linux kernel. Please note that certain libraries, such as libmolgrid, are not compatible with Windows. It should be executed on a GPU-equipped machine.

The below libraries and their versions ensure the program functions without issues

!pip3 install torch torchvision torchaudio<br/>
!pip install molgrid<br/>
!pip install biopython<br/>
!mamba install fpocket<br/>
!pip install scikit-learn<br/>

``` pip install -r requirements.txt```

As a second option, Google Colab is a good choice if you want to use our program without needing a specific operating system or GPU on your local machine. If you use Google Colab, you can open '_Run_Colab.ipynb_' from your Google Drive and follow the instructions within.


**Prediction**

predict.py

pdb dosyasi projenin icinde olmali

**Evaluation**

evaluation.py

**Train**

train.py