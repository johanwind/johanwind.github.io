---
title: "Start here: Why I care about implicit biases"
layout: post
description: I explain what I mean by "implicit biases" in deep learning and my motivations for researching them.
keywords: introduction, neural networks
---

My research, and this blog, are centered around implicit biases in deep learning. But what even are "implicit biases", and why do I care about them? In this post, I try to explain my motivations, why I think understanding implicit biases is the key to unlocking the potential of deep learning.

# Why I care about deep learning
It works *really* well for some problems. Just look at the following image generated using deep learning (by [Midjourney](https://www.midjourney.com/showcase)):

<img src="https://www.creativeshrimp.com/wp-content/uploads/2022/06/midjourney_aiart_gleb_alexandrov_06-1170x731.jpg" style="width: 100%; display: block; margin: 0 auto;"/>

Making a program to automaticaly draw a beautiful image from a text prompt would be practically impossible, before deep learning came along and did it. Other problems where deep learning is miles ahead of the competition include [image classification](https://paperswithcode.com/sota/image-classification-on-imagenet), [playing games from pixels](https://www.deepmind.com/publications/playing-atari-with-deep-reinforcement-learning) and [lossless text compression](http://www.mattmahoney.net/dc/text.html). I believe deep learning will continue to deliver breakthroughs in other areas, and I'm excited to see what those are.

# The problem with deep learning
Ok, so deep learning has amazing potential, what are the problems we need to overcome to acheive that potential? In my opinion, the main problem with modern deep learning is the huge amount of engineering effort it currently requires.

To get the best practical performance from deep learning, you need to add a bunch of small (but hugely important) tricks. These tricks take time to learn, and take time to apply and adjust to new problems. An example of such a trick is using *data augmentation* when training an image classifier. Concretely, a simple data augmentation would be adding horizontally flipped images to your dataset, effectively doubling the size of your dataset.

Another problem is that modern deep learning requires enormous computations. More compute gives better results, so naturally you put in as much compute as you can afford. In practice, this means waiting hours or days for a model to train. This drastically increases the time required to test new code, and in general slows down development.

Tuning the *hyperparameters* is also a time consuming part of modern deep learning. The performance of a deep learning model critically depends on a large number of tuning parameters, which need to be carefully chosen when applying the model to a new problem. Here is a list of some common hyperparameters:

| Hyperparameter | Typical value | Affects model expressivity |
| --- | ---: | ---: | ---: |
| Learning rate | 3e-4 | No |
| Momentum | 0.9 | No |
| Learning rate schedule | Cosine | No |
| Optimizer | Adam | No |
| Batch size | 32 | No |
| Number of training epochs | 300 | No | 
| Weight decay | 0 | No |
| Activation function | ReLU | Yes |
| Weight initialization | Xavier | No |
| Feature dimension | 512 | Yes |
| Label smoothing | 0.1 | No |
| Dropout probability | 0.2 | No |

In practice, these hyperparameters are chosen using a combination of the engineer's experience, and repeatedly testing the model with different hyperparameter configurations. Note that changing one parameter might change the effects of other parameters, making it exponentially harder. Recall that testing the model is computationally expensive and slow. The combination makes for a painful engineering experience. The "correct" solution is to run an automated search through many hyperparameter combinations, and pick the best. However, that is computationally expensive. So you would generally rather train a more computationally expensive model giving better results.

I believe the applicability of deep learning is severely limited by the huge amount of engineering effort it requires. So what is the solution?

# The need for theory
Let's compare with classical machine learning algorithms, things like linear regression (on possibly nonlinear, hand-engineered features). Applying those algorithms can often be reduced to minimizing some convex loss function $$L$$ plus some convex regularizer $$R$$ times some scalar weight $$\lambda$$,

$$\min_\theta L(\theta)+\lambda R(\theta).$$

We can then use some numerical optimziation algorithm to find the unique minimum at parameter configuration $$\theta^*$$, and use those parameters to produce predictions. There are several hyperparameters in the optimization algorithm, but those only affect the time to convergence, so they can be left to reasonable default values. Mathematicians proved correctness of the optimzation algorithms, so the right $$\theta^*$$ is found every time. This mature theory means practitioners can focus most of their effort on making good models, since then applying the models is relatively straightforward.

But modern deep learning is also optimizing a loss function? We have optimizers which can be guaranteed to find (local) minima. What's the problem? The problems start when we realize that modern deep learning methods can have way more parameters than the number of data points they are trying to fit. Even for small datasets like CIFAR with 50 000 training images, deep learning models use millions of parameters. The models are *overparameterized*. [Deep learning models can fit random training labels](https://arxiv.org/abs/1611.03530).

Because of overparameterization, there are *many* different parameter configurations giving 0 training loss (or arbitrarily small loss in the case of cross entropy classification loss). To make matters worse, deep learning models don't seem to require explicit regularizers (specifically, weight decay is optional). As a result, in deep learning, our training algorithm and hyperparameters *do* affect what model we end up with. I call it the *implicit bias* which determines the final model, among the many models minimizing the loss function. The long list of hyperparameters in the table above can (and empirically does!) affect the implicit bias and performance of the final model. 

Classically, the main important aspect of a model is what kind of functions it can express, its *expressivity*. However, in deep learning, most of the important choices don't even affect the expressivity, they only affect the implicit bias (see the table above). Sadly, we have no better description of this implicit bias than "run exactly this algorithm with these hyperparameters, that should give you the implicit bias baked into your final model".

As a concrete example of why this is a problem, say you made a fantastic new second order optimization algorithm, it optimizes the loss function 10x faster than standard first order methods! The current deep learning models were developed and compared under the implicit bias give by current optimizers. Chances are that your second order optimizer significantly changes that implicit bias, giving worse performance, since the model was not built for the new implicit bias. The algorithm couldn't be used.

Let's say we found a way to better characerize the implicit bias of modern deep learning models. Maybe we could improve it. Maybe we could train models faster, without losing out on performance. Maybe we could get rid of annoying hyperparameters. I want to find out.
