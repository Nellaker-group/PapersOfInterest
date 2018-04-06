# Papers Of Interest

 blahdy blah
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
[VAE with a VampPrior](https://arxiv.org/abs/1705.07120)
```
Many different methods to train deep generative models have been introduced in the past. In this paper, we propose to extend
the variational auto-encoder (VAE) framework with a new type of prior which we call "Variational Mixture of Posteriors" 
prior, or VampPrior for short. The VampPrior consists of a mixture distribution (e.g., a mixture of Gaussians) with 
components given by variational posteriors conditioned on learnable pseudo-inputs. We further extend this prior to a two 
layer hierarchical model and show that this architecture with a coupled prior and posterior, learns significantly better 
models. The model also avoids the usual local optima issues related to useless latent dimensions that plague VAEs. We 
provide empirical studies on six datasets, namely, static and binary MNIST, OMNIGLOT, Caltech 101 Silhouettes, Frey Faces 
and Histopathology patches, and show that applying the hierarchical VampPrior delivers state-of-the-art results on all 
datasets in the unsupervised permutation invariant setting and the best results or comparable to SOTA methods for the 
approach with convolutional networks.
```

## Advice
[A DISCIPLINED APPROACH TO NEURAL NETWORK HYPER-PARAMETERS: PART 1 – LEARNING RATE, BATCH SIZE, MOMENTUM, AND WEIGHT DECAY](https://arxiv.org/abs/1803.09820)

```
Although deep learning has produced dazzling successes for applications of image, speech, and video processing in the past 
few years, most trainings are with suboptimal hyper-parameters, requiring unnecessarily long training times. Setting the 
hyper-parameters remains a black art that requires years of experience to acquire. This report proposes several efficient 
ways to set the hyper-parameters that significantly reduce training time and improves performance. Specifically, this report 
shows how to examine the training validation/test loss function for subtle clues of underfitting and overfitting and 
suggests guidelines for moving toward the optimal balance point. Then it discusses how to increase/decrease the learning 
rate/momentum to speed up training. Our experiments show that it is crucial to balance every manner of regularization for 
each dataset and architecture. Weight decay is used as a sample regularizer to show how its optimal value is tightly coupled
with the learning rates and momentums.
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
