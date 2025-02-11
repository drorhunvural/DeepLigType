<div align="center" style="text-align: center;">
  <h1>DeepLigType: Predicting Ligand Types of Protein-Ligand Binding Sites Using a Deep Learning Model</h1>
</div>

The paper "DeepLigType: Predicting Ligand Types of Protein-Ligand Binding Sites Using a Deep Learning Model" 

**_Citation_**

https://ieeexplore.ieee.org/abstract/document/10747127

https://pubmed.ncbi.nlm.nih.gov/39509302/
```
@ARTICLE{10747127,
  author={Vural, Orhun and Jololian, Leon and Pan, Lurong},
  journal={IEEE Transactions on Computational Biology and Bioinformatics}, 
  title={DeepLigType: Predicting Ligand Types of Protein-Ligand Binding Sites Using a Deep Learning Model}, 
  year={2025},
  volume={22},
  number={1},
  pages={116-123},
  keywords={Proteins;Deep learning;Computational modeling;Predictive models;Computer architecture;Inhibitors;Drugs;Convolutional neural networks;Accuracy;Training;Predict ligand type;Drug discovery;deep learning},
  doi={10.1109/TCBB.2024.3493820}}
```
**_Files_**

[Link - Download Files](https://drive.google.com/drive/folders/1WVIpv6CbHBnRcE_0qR8E9AEwu7-CGgCZ?usp=sharing) => The Molcache file and trained models (.pth) can be found via the link.

**_Requirements & Installation_**

Please note that certain libraries, such as libmolgrid, are not compatible with Windows. To run the program, you'll need an operating system with a Linux kernel. Additionally, It should be executed on a GPU-equipped machine.

The below libraries and their versions ensure the program functions without issues. The latest library versions will also work without any problems. If you face any issues, report them by opening an issue.

[Python](https://github.com/python) 3.10.12<br/>
[torch](https://github.com/pytorch/pytorch)==2.1.0<br/>
[torchvision](https://github.com/pytorch/pytorch)==0.16.0<br/>
[torchaudio](https://github.com/pytorch/pytorch)==2.1.0<br/>
[biopython](https://github.com/biopython/biopython)==1.81<br/>
[molgrid](https://github.com/gnina/libmolgrid)==0.5.3<br/>
[scikit-learn](https://github.com/scikit-learn/scikit-learn)==1.3.1<br/>
[fpocket](https://github.com/Discngine/fpocket)==4.1<br/>
[tqdm](https://github.com/tqdm/tqdm)==4.66.1<br/>


As a second option, Google Colab is a good choice if you want to use our program without needing a specific operating system or GPU on your local machine. If you use Google Colab, you can open '_Run_Colab.ipynb_' from your Google Drive and follow the instructions within.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drorhunvural/DeepLigType/blob/main/Run_Colab.ipynb)

**_Prediction_**

```python '.../DeepLigType/predict.py' -p '.../DeepLigType/xxxx.pdb' -t 'trainedmodel'```

Example: <pre><code> python '.../DeepLigType/predict.py' -p '.../DeepLigType/5xpp.pdb' -t "CNN_2023-08-30_acc_0.717969_59.26.pth" </code></pre>


**_Evaluation_**

```!python '.../DeepLigType/evaluation.py' -t ".../trained_model_pth"```

Example: <pre><code> python '.../DeepLigType/evaluation.py' -t "CBAM_2023-08-29_acc_0.915781_74.16.pth" </code></pre>

**_Training_**

```python '.../DeepLigType/train.py' -m "deeplearning_model_name" ```

Example: <pre><code> python '.../DeepLigType/train.py' -m "cbam" </code></pre>

**_Visualization_**

Visualization is coming! Stay tuned!

**Running DeepLigType: Video**<br/>


[![YouTube Video](https://img.shields.io/badge/Watch%20on%20YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/SobCab1fZeA?si=u-soXHHxLBHp8lUL)
<br/>
[![YouTube Video](https://img.youtube.com/vi/SobCab1fZeA/0.jpg)](https://youtu.be/SobCab1fZeA?si=u-soXHHxLBHp8lUL)


