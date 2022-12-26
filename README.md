# SimCLR-ML-Project
Final project exam of Introduction to Machine Learning, University of Trento, 2021-22.

The project is based on the following assignments:

- Implement SimCLR (https://arxiv.org/abs/2002.05709) and pretrain any CNN on CIFAR-100. 
- After that, evaluate the top-1 accuracy of a linear classifier (CIFAR-100 that you will train after freezing the learned CNN) and the top-1 accuracy of a K-NN classifier. 
- Plot a T-SNE visualisation of the features learned by your model: you should extract the representations of all images, reduce their dimensionality to 2 (with T-SNE) and show each point as a unique colour (related to its class).

## General info
This project is based on the paper [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) by Chen et al. (2020). The goal is to train a neural network to learn visual representations of images, using a contrastive learning approach. The network is trained on the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, and the learned representations are used to train a linear classifier and a KNN classifier. 

Our implementation did not reach the results reported in the paper due to the lack of time and resources. Indeed, we were constrained by the computational resources available on our personal computers, and we were not able to train the network for a sufficient number of epochs. For this reason, we used google colab that has a GPU available for free, but the GPU is disconnected after 3 hours of activity.

Please, follow carefully setup instructions. 

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

5. Use the jupyter notebook in the repository to play with the code for SimCLR implementation `Cifar100_SimCLR_Implementation.ipynb` and for the downstream tasks (Linear Classifier `Cifar100_Downstream_task_LINEAR.ipynb`and KNN classifier `Cifar100_Downstream_task_KNN.ipynb`)

### Google Colab setup
You can run the code on Google Colab. To do so, you have just to follow the instructions in the notebook that we provide.

1. SimCLR Implementation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1nOwzxxv0_cNwMoAVQDiE3ghHZ6ePQ_a9/view?usp=share_link)

2. Linear Classifier: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1LxiKM2wsFLxTCfVSrn9lld7V2FgNFHth/view?usp=share_link)

3. KNN Classifier: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1mivEERWBYb1GoLY8OnioNxj-PgoCJ83-/view?usp=share_link)
```




