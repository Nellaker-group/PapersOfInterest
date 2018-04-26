# Papers Of Interest

## Adverserials
[Learning to Pivot with Adversarial Networks](https://arxiv.org/abs/1611.01046)
```
Several techniques for domain adaptation have been proposed to account for differences in the distribution of the data 
used for training and testing. The majority of this work focuses on a binary domain label. Similar problems occur in a 
scientific context where there may be a continuous family of plausible data generation processes associated to the presence
of systematic uncertainties. Robust inference is possible if it is based on a pivot -- a quantity whose distribution does not 
depend on the unknown values of the nuisance parameters that parametrize this family of data generation processes. In this 
work, we introduce and derive theoretical results for a training procedure based on adversarial networks for enforcing the 
pivotal property (or, equivalently, fairness with respect to continuous attributes) on a predictive model. The method includes 
a hyperparameter to control the trade-off between accuracy and robustness. We demonstrate the effectiveness of this approach 
with a toy example and examples from particle physics.
```

## GANs

[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

```
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an 
input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will
not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the
absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is 
indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we 
couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to push F(G(X))≈X (and vice versa). Qualitative
results are presented on several tasks where paired training data does not exist, including collection style transfer, object
transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate 
the superiority of our approach.
```

## Applied
[Predicting multicellular function through multi-layer tissue networks](https://arxiv.org/abs/1707.04638)

```
Motivation: Understanding functions of proteins in specific human tissues is essential for insights into disease diagnostics
and therapeutics, yet prediction of tissue-specific cellular function remains a critical challenge for biomedicine. 
Results: Here we present OhmNet, a hierarchy-aware unsupervised node feature learning approach for multi-layer networks. We 
build a multi-layer network, where each layer represents molecular interactions in a different human tissue. OhmNet then 
automatically learns a mapping of proteins, represented as nodes, to a neural embedding based low-dimensional space of 
features. OhmNet encourages sharing of similar features among proteins with similar network neighborhoods and among proteins
activated in similar tissues. The algorithm generalizes prior work, which generally ignores relationships between tissues, 
by modeling tissue organization with a rich multiscale tissue hierarchy. We use OhmNet to study multicellular function in a 
multi-layer protein interaction network of 107 human tissues. In 48 tissues with known tissue-specific cellular functions, 
OhmNet provides more accurate predictions of cellular function than alternative approaches, and also generates more accurate
hypotheses about tissue-specific protein actions. We show that taking into account the tissue hierarchy leads to improved 
predictive power. Remarkably, we also demonstrate that it is possible to leverage the tissue hierarchy in order to 
effectively transfer cellular functions to a functionally uncharacterized tissue. Overall, OhmNet moves from flat networks 
to multiscale models able to predict a range of phenotypes spanning cellular subsystems
```

## Unsupervised / AE
[$\beta$-VAE](https://openreview.net/references/pdf?id=Sy2fzU9gl)
```

## Advice
[A DISCIPLINED APPROACH TO NEURAL NETWORK HYPER-PARAMETERS: PART 1 – LEARNING RATE, BATCH SIZE, MOMENTUM, AND WEIGHT DECAY](https://arxiv.org/abs/1803.09820)

```
Learning an interpretable factorised representation of the independent data generative factors of the world without 
supervision is an important precursor for the development of artificial intelligence that is able to learn and reason in the 
same way that humans do. We introduce beta-VAE, a new state-of-the-art framework for automated discovery of interpretable 
factorised latent representations from raw image data in a completely unsupervised manner. Our approach is a modification of 
the variational autoencoder (VAE) framework. We introduce an adjustable hyperparameter beta that balances latent channel 
capacity and independence constraints with reconstruction accuracy. We demonstrate that beta-VAE with appropriately tuned
beta > 1 qualitatively outperforms VAE (beta = 1), as well as state of the art unsupervised (InfoGAN) and semi-supervised (DC-
IGN) approaches to disentangled factor learning on a variety of datasets (celebA, faces and chairs). Furthermore, we devise a
protocol to quantitatively compare the degree of disentanglement learnt by different models, and show that our approach also 
significantly outperforms all baselines quantitatively. Unlike InfoGAN, beta-VAE is stable to train, makes few assumptions 
about the data and relies on tuning a single hyperparameter, which can be directly optimised through a hyper parameter search 
using weakly labelled data or through heuristic visual inspection for purely unsupervised data.
```

## Genetics / Genomics

[A Likelihood-Free Inference Framework for Population Genetic Data using Exchangeable Neural Networks](https://arxiv.org/abs/1802.06153)

```
Inference for population genetics models is hindered by computationally intractable likelihoods. While this issue is tackled
by likelihood-free methods, these approaches typically rely on hand-crafted summary statistics of the data. In complex 
settings, designing and selecting suitable summary statistics is problematic and results are very sensitive to such choices. 
In this paper, we learn the first exchangeable feature representation for population genetic data to work directly with 
genotype data. This is achieved by means of a novel Bayesian likelihood-free inference framework, where a permutation-
invariant convolutional neural network learns the inverse functional relationship from the data to the posterior. We 
leverage access to scientific simulators to learn such likelihood-free function mappings, and establish a general framework 
for inference in a variety of simulation-based tasks. We demonstrate the power of our method on the recombination hotspot 
testing problem, outperforming the state-of-the-art.
```

## Misc


[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)
```
In the past few years, the field of computer vision has gone through a revolution fueled mainly by the advent
of large datasets and the adoption of deep convolutional neural networks for end-to-end learning. 
The person re-identification subfield is no exception to this. Unfortunately, a prevailing belief in the community
seems to be that the triplet loss is inferior to using surrogate losses (classification, verification) 
followed by a separate metric learning step. We show that, for models trained from scratch as well as pretrained ones, 
using a variant of the triplet loss to perform end-to-end deep metric learning outperforms most other published 
methods by a large margin.
```

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

```
The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, 
where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically 
increasing model capacity without a proportional increase in computation. In practice, however, there are significant 
algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of 
conditional computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational
efficiency on modern GPU clusters. We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to 
thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use 
for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is 
critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in 
which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language 
modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at 
lower computational cost.
```

[Pixel Recursive Super Resolution](https://arxiv.org/abs/1702.00783)
```
We present a pixel recursive super resolution model that synthesizes realistic details into images while enhancing their 
resolution. A low resolution image may correspond to multiple plausible high resolution images, thus modeling the super 
resolution process with a pixel independent conditional model often results in averaging different details--hence blurry 
edges. By contrast, our model is able to represent a multimodal conditional distribution by properly modeling the 
statistical dependencies among the high resolution image pixels, conditioned on a low resolution input. We employ a PixelCNN 
architecture to define a strong prior over natural images and jointly optimize this prior with a deep conditioning 
convolutional network. Human evaluations indicate that samples from our proposed model look more photo realistic than a 
strong L2 regression baseline.
```
