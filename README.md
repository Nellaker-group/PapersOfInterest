# Papers Of Interest

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
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
```

```Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to push F(G(X))≈X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.
