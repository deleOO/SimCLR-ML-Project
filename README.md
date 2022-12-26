# SimCLR-ML-Project
Final project exam of Introduction to Machine Learning, University of Trento, 2021-22.

The project is based on the following assignments:

- Implement SimCLR (https://arxiv.org/abs/2002.05709) and pretrain any CNN on CIFAR-100. 
- After that, evaluate the top-1 accuracy of a linear classifier (CIFAR-100 that you will train after freezing the learned CNN) and the top-1 accuracy of a K-NN classifier. 
- Plot a T-SNE visualisation of the features learned by your model: you should extract the representations of all images, reduce their dimensionality to 2 (with T-SNE) and show each point as a unique colour (related to its class).

## General info
This project is based on the paper [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) by Chen et al. (2020). 

<p align="center">
  <img src="https://camo.githubusercontent.com/d92c0e914af70fe618cf3ea555e2da1737d84bc4/68747470733a2f2f312e62702e626c6f6773706f742e636f6d2f2d2d764834504b704539596f2f586f3461324259657276492f414141414141414146704d2f766146447750584f79416f6b4143385868383532447a4f67457332324e68625877434c63424741735948512f73313630302f696d616765342e676966" alt="alt text" width="300"/>
  <br>
  <m>Fig.1 - SimCLR Illustration <a href="https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html"> [1] </a> </m>
</p>

The goal is to train a neural network to learn visual representations of images, using a contrastive learning approach. The network is trained on the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, and the learned representations are used to train a linear classifier and a KNN classifier. 

Our implementation did not reach the results reported in the paper due to the lack of time and resources. Indeed, we were constrained by the computational resources available on our personal computers, and we were not able to train the network for a sufficient number of epochs. For this reason, we used google colab that has a GPU available for free, but the GPU is disconnected after 3 hours of activity.

There are two ways to run the code: locally and on Google Colab. We recommend to use Google Colab, because it is easier to setup and it is faster to run the code. Please, follow carefully setup instructions. 

## Setup
Tested on Windows 10 Home 20H2 and Google Colab.

### Local setup

1. Clone the repository
```
git clone https://github.com/deleOO/SimCLR-ML-Project.git
```

2. Create a virtual environment
```
python -m venv .venv
.venv\Scripts\activate
```

3. Install the requirements
```
pip install -r requirements.txt
```

4. Run download_cifar100.py to download the CIFAR-100 dataset
```
python download_cifar100.py
```

5. Use the jupyter notebooks in the repository to play with the code 
- `Cifar100_SimCLR_Implementation.ipynb`
- `Cifar100_Downstream_task_LINEAR.ipynb`
- `Cifar100_Downstream_task_KNN.ipynb`

### Google Colab setup
You can run the code on Google Colab. To do so, you have just to follow the instructions in the notebook that we provide.

1. SimCLR Implementation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1nOwzxxv0_cNwMoAVQDiE3ghHZ6ePQ_a9/view?usp=share_link)

2. Linear Classifier: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1LxiKM2wsFLxTCfVSrn9lld7V2FgNFHth/view?usp=share_link)

3. KNN Classifier: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1mivEERWBYb1GoLY8OnioNxj-PgoCJ83-/view?usp=share_link)




