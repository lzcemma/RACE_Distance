## One-Pass Distribution Sketch for Measuring Data Heterogeneity in Federated Learning

Federated learning (FL) is a machine learning paradigm where multiple client devices train models collaboratively without data exchange. Data heterogeneity problem is naturally inherited in FL since data in different clients follow diverse distributions. To mitigate the negative influence of data heterogeneity, we need to start by measuring it across clients. However, the efficient measurement between distributions is a challenging problem, especially in high dimensionality. In this paper, we propose a one-pass distribution sketch to represent the client data distribution. Our sketching algorithm only requires a single pass of the client data, which is efficient in terms of time and memory. Moreover, we show in both theory and practice that the distance between two distribution sketches represents the divergence between their corresponding distributions. Furthermore, we demonstrate with extensive experiments that our distribution sketch improves the client selection in the FL training. We also showcase that our distribution sketch is an efficient solution to the cold start problem in FL for new clients with unlabeled data.

Paper Link:  [https://openreview.net/pdf?id=KMxRQO7P98](https://openreview.net/pdf?id=KMxRQO7P98)

This repo is consisting of three parts: (1)  Sketch-based client sampling strategy for federated optimization (2) Cold start personalized FL with sketch

## Generate Data
We provide `generate_data.py` to generate client data with different distribution. 
Example usage to replicate Figure 2(b) : `python generate_data.py --partition dirichlet_uniform`

## Sketch-based client sampling strategy for federated optimization**
Main algorithm is contained in `src/algo_sketch_sample.py`. One need to specify the generated data path in SPLITPATH_TRAIN
Example usage: `python main.py --dataset mnist_dirichlet_uniform --algo sample`

## Cold start personalized FL with sketch
Main algorithm is contained in `src/algo_personalization.py.`One need to specify the generated data path in SPLITPATH_TRAIN
Example usage: `python main.py --dataset mnist_dirichlet_uniform --train_local --train_global`