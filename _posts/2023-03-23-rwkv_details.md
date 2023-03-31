---
title: "How the RWKV language model works"
layout: post
description: I go through and explain a minimal implementation of RWKV in detail.
keywords: code, neural networks
---

In this post, I will explain the details of how RWKV generates text. For a high level overview of what RWKV is and what is so special about it, check out [the other post about RWKV](/2023/03/23/rwkv_overview.html).

To explain exactly how RWKV works, I think it is easiest to look at a simple implementation of it. The following ~100 line code (based on [RWKV in 150 lines](https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py)) is a minimal implementation of a relatively small (430m parameter) RWKV model which generates text.

<p><details> <summary>Minimal RWKV code</summary> {% highlight python %}
import numpy as np
from torch import load as torch_load  # Only for loading the model weights
from tokenizers import Tokenizer

layer_norm = lambda x, w, b : (x - np.mean(x)) / np.std(x) * w + b
exp = np.exp
sigmoid = lambda x : 1/(1 + exp(-x))

def time_mixing(x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout):
    k = Wk @ ( x * mix_k + last_x * (1 - mix_k) )
    v = Wv @ ( x * mix_v + last_x * (1 - mix_v) )
    r = Wr @ ( x * mix_r + last_x * (1 - mix_r) )

    wkv = (last_num + exp(bonus + k) * v) /      \
          (last_den + exp(bonus + k))
    rwkv = sigmoid(r) * wkv

    num = exp(-exp(decay)) * last_num + exp(k) * v
    den = exp(-exp(decay)) * last_den + exp(k)

    return Wout @ rwkv, (x,num,den)


def channel_mixing(x, last_x, mix_k, mix_r, Wk, Wr, Wv):
    k = Wk @ ( x * mix_k + last_x * (1 - mix_k) )
    r = Wr @ ( x * mix_r + last_x * (1 - mix_r) )
    vk = Wv @ np.maximum(k, 0)**2
    return sigmoid(r) * vk, x


def RWKV(model, token, state):
    params = lambda prefix : [model[key] for key in model.keys() if key.startswith(prefix)]

    x = params('emb')[0][token]
    x = layer_norm(x, *params('blocks.0.ln0'))

    for i in range(N_LAYER):
        x_ = layer_norm(x, *params(f'blocks.{i}.ln1'))
        dx, state[i][:3] = time_mixing(x_, *state[i][:3], *params(f'blocks.{i}.att'))
        x = x + dx

        x_ = layer_norm(x, *params(f'blocks.{i}.ln2'))
        dx, state[i][3] = channel_mixing(x_, state[i][3], *params(f'blocks.{i}.ffn'))
        x = x + dx

    x = layer_norm(x, *params('ln_out'))
    x = params('head')[0] @ x

    e_x = exp(x-np.max(x))
    probs = e_x / e_x.sum() # Softmax of x

    return probs, state

##########################################################################################################

def sample_probs(probs, temperature=1.0, top_p=0.85):
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = sorted_probs[np.argmax(cumulative_probs > top_p)]
    probs[probs < cutoff] = 0
    probs = probs**(1/temperature)
    return np.random.choice(a=len(probs), p=probs/np.sum(probs))


# Available at https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth
MODEL_FILE = '/data/rwkv/RWKV-4-Pile-430M-20220808-8066.pth'
N_LAYER = 24
N_EMBD = 1024

print(f'\nLoading {MODEL_FILE}')
weights = torch_load(MODEL_FILE, map_location='cpu')
for k in weights.keys():
    if '.time_' in k: weights[k] = weights[k].squeeze()
    weights[k] = weights[k].float().numpy() # convert to f32 type


# Available at https://github.com/BlinkDL/ChatRWKV/blob/main/20B_tokenizer.json
tokenizer = Tokenizer.from_file("/data/rwkv/20B_tokenizer.json")

print(f'\nPreprocessing context')

context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

state = np.zeros((N_LAYER, 4, N_EMBD), dtype=np.float32)
for token in tokenizer.encode(context).ids:
    probs, state = RWKV(weights, token, state)

print(context, end="")
for i in range(100):
    token = sample_probs(probs)
    print(tokenizer.decode([token]), end="", flush=True)
    probs, state = RWKV(weights, token, state)
{% endhighlight %} </details></p>

To avoid hiding complexity, the model computation itself is written entirely in python, with numpy for matrix / vector operations. However, I needed to use `torch.load` to load the model weights from a file, and `tokenizers.Tokenizer` to make the text into tokens the model can work with.



## Text generation with RWKV
The code uses RWKV to continue the following text:

"In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

We first need to convert this text into a series of tokens (numbers from 0 to 50276 representing words/symbols/tokens in our vocabulary). That is not the focus of this blog post, so I just do it with an external library `tokenizer.encode(context).ids`.

Next, we need to process this sequence of tokens into an RWKV state. Essentially, RWKV represents a function which takes a token and a state, and outputs a probability distribution over the next token, and a new state. Of course, the function also depends on the RWKV model parameters, but since we use a trained model (downloaded from [here](https://huggingface.co/BlinkDL/rwkv-4-pile-430m)), we view those parameters as fixed. To convert the text to a state, we just initialize the state to all zeros, and feed the tokens through the RWKV function one by one.

```
state = np.zeros((N_LAYER, 4, N_EMBD), dtype=np.float32)
for token in tokenizer.encode(context).ids:
    probs, state = RWKV(weights, token, state)
```

Now the variable `state` contains a state representation of our input text, and the variable "probs" contain the probability distribution the model predicts for the next token.

We can now simply sample the probability distribution (in practice, we avoid low probability tokens in `sample_probs()`) and add another token to the text. Then we feed the new token into RWKV and repeat.

```
for i in range(100):
    token = sample_probs(probs)
    print(tokenizer.decode([token]), end="", flush=True)
    probs, state = RWKV(weights, token, state)       
```

A typical, generated continuation is:

"They’re just like us. They use Tibetan for communication, and for a different reason – they use a language that they’re afraid to use. To protect their secret, they prefer to speak a different language to the local public."

Of course, larger models will perform better than this relatively small 430m RWKV.

## What goes on inside RWKV()

The first thing RWKV does is look up the embedding vector of the input token. I.e `x = params('emb')[0][token]`. Here `params('emb')[0]` is simply a $$50277 \times 1024$$ matrix, and we extract a row.

The next line `x = layer_norm(x, *params('blocks.0.ln0'))` requires me to explain what a Layer Normalization is. The easiest way is to just show the definition:

`layer_norm = lambda x, w, b : (x - np.mean(x)) / np.std(x) * w + b`.

The intuition is that it normalizes a vector x to zero mean and unit variance, and then scales and offsets that. Note that the scale `w` and offset `b` are 1024-dimensional vectors, which are learned model parameters.

Now we get to the main part of the model. Which is split into 24 layers, applied sequentially.

```python
for i in range(N_LAYER):
    x_ = layer_norm(x, *params(f'blocks.{i}.ln1'))
    dx, state[i][:3] = time_mixing(x_, *state[i][:3], *params(f'blocks.{i}.att'))
    x = x + dx

    x_ = layer_norm(x, *params(f'blocks.{i}.ln2'))
    dx, state[i][3] = channel_mixing(x_, state[i][3], *params(f'blocks.{i}.ffn'))
    x = x + dx
```

Note that we are only adding updates to `x` like `x = x + dx`, this is called using "residual connections". Each time we make a copy of `x`, we feed it through a layer normalization before mixing it.  Each layer has two mixing functions: a "time mixing" part and a "channel mixing" part. In a typical transformer, the "time mixing" would be done by multi head attention, and the "channel mixing" would be done by a simple feed forward network. RWKV does something a bit different, which we'll explain in the next sections.

### Channel mixing
I'll start with channel mixing, since it's the simpler one of the two mixing functions.
```python
def channel_mixing(x, last_x, mix_k, mix_r, Wk, Wr, Wv):
    k = Wk @ ( x * mix_k + last_x * (1 - mix_k) )
    r = Wr @ ( x * mix_r + last_x * (1 - mix_r) )
    vk = Wv @ np.maximum(k, 0)**2
    return sigmoid(r) * vk, x
```
The channel mixing layer takes an input `x` corresponding to this token, and the `x` corresponding to the previous token, which we call `last_x`. `last_x` was stored in this RWKV layer's `state`. The rest of the inputs are learned RWKV parameters.

First, we linearly interpolate `x` and `last_x`, using learned weights. We run this interpolated `x` as input to a 2 layer feed forward network with squared relu activation, and finally multiply with the sigmoid activations of another feed forward network (in classical RNN terms, this would be called gating).

Note that in terms of memory usage, the matrices `Wk,Wr,Wv` hold almost all the parameters (they are $$1024\times 1024$$ matrices, while the other variables are just 1024-dimensional vectors). And the matrix multiplications (`@` in python) contribute the vast majority of required computations.

### Time mixing
```python
def time_mixing(x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout):
    k = Wk @ ( x * mix_k + last_x * (1 - mix_k) )
    v = Wv @ ( x * mix_v + last_x * (1 - mix_v) )
    r = Wr @ ( x * mix_r + last_x * (1 - mix_r) )

    wkv = (last_num + exp(bonus + k) * v) /      \
          (last_den + exp(bonus + k))
    rwkv = sigmoid(r) * wkv

    num = exp(-exp(decay)) * last_num + exp(k) * v
    den = exp(-exp(decay)) * last_den + exp(k)

    return Wout @ rwkv, (x,num,den)
```
The time mixing starts similarly to the channel mixing, by interpolating this token's `x` with the last token's `x`. We then apply learned $$1024\times 1024$$ matrices to get "key", "value" and "receptance" vectors.

The next part is where the magic happens.

#### The "RWKV attention"
Before getting to the core of the mechanism, we will make the observation that while the variables going into the attention mechanism are all 1024-dimensional (we say they have 1024 channels), all channels are computed independently of each other. We will therefore just look at what happens to a single channel, treating the variables as scalars.

Now, let us look at the variable `num`. To make math notations cleaner, let's rename `num` and `den` to $$\alpha$$ and $$\beta$$. Both $$\alpha$$ and $$\beta$$ are stored in the RWKV state. For each new token, $$\alpha$$ is calculated as $$\alpha_i = e^{-w} \alpha_{i-1} +e^{k_i} v_i$$, where $$i$$ is the index of the current token. We defined `w = exp(decay)`, note that `w` is always positive.

By induction, we have $$\alpha_i = \sum_{j=1}^i e^{-(i-j)w+k_j} v_j$$. Similarly, $$\beta_i = \sum_{j=1}^i e^{-(i-j)w+k_j}$$. Note that $$\alpha_i$$ looks like a weighted sum of the $$v_j$$, while $$\beta_i$$ is just the sum of weights. So $$\frac{\alpha_i}{\beta_i}$$ becomes a weighted average of $$v_j$$.

Plugging in the formulas for $$\alpha_{i-1}$$ and $$\beta_{i-1}$$ into the definition of `wkv`, and denoting `bonus` by $$u$$, we get

$$\text{wkv}_i = \frac{ \sum_{j=1}^{i-1} e^{-(i-1-j)w+k_j} v_j + e^{u+k_i} v_i }{\sum_{j=1}^{i-1} e^{-(i-1-j)w+k_j} + e^{u+k_i}}.$$

So $$\text{wkv}$$ is a weighted average of $$v$$ with weights according to $$k$$, but also the current $$v_i$$ is given a `bonus` ($$u$$) additional weight, and previous $$v_j$$ are given geometrically smaller weights the further away they are.

For reference, standard transformer attention takes "query", "key" and "value" vectors $$q,k,v$$ and outputs

$$\frac{\sum_{j=1}^i e^{q_i^\top k_j} v_j}{\sum_{j=1}^i e^{q_i^\top k_j}}.$$

After calculating `wkv`, the time mixing multiplies by the "receptance" `sigmoid(r)`. It does a final linear transformation before returning the result.

### Converting to output probabilities
After going through the 24 layers of time mixing and channel mixing, we need to convert the final output to predicted probabilities for the next token.
```python
x = layer_norm(x, *params('ln_out'))
x = params('head')[0] @ x

e_x = exp(x-np.max(x))
probs = e_x / e_x.sum() # Softmax of x
```
First, we do a layer normalization. Then, we multiply by a $$50277 \times 1024$$ matrix `params('head')[0]` given by the RWKV parameters, giving us a 50277-dimensional vector. To get a probability distribution over tokens (i.e. a 50277-dimensional, non-negative vector which sums to 1), we run our `x` through a "softmax" function. The softmax of `x` is just `exp(x)/sum(exp(x))`. However, calculating `exp(x)` can cause numerical overflows, so we calculate the equivalent function `exp(x-max(x))/sum(exp(x-max(x)))`.

That's it! Now you know exactly how RWKV works for generating text.

## Practical considerations
In practice, there are some issues which I ignored in my simplified code. Most importantly, in practice, we care a lot about the performance / run-time of the code. This leads us to run RWKV in parallel on GPUs, use specialized GPU code written in CUDA, use 16-bit floating point numbers, and more.

### Numerical issues
The largest number a 16-bit floating point number (float16) can represent is 65 504, anything above that overflows, which is bad. Most of the code has no problems with this, partially because the Layer Normalizations keep values in a reasonable range. However, the RWKV attention contains exponentially large numbers (`exp(bonus + k)`). In practice, the RWKV attention is implemented in a way where we factor out an exponential factor from `num` and `den` to keep everything within float16 range. See for example the time_mixing function in [RWKV in 150 lines](https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py).

### Training
We simply loaded a pretrained model in our example. To train the model, one would calculate the cross entropy loss of the predicted probabilities on a long text (our example model was trained on [the pile](https://pile.eleuther.ai/)). Next, calculate the gradient of that loss with respect to all the RWKV parameters. That gradient is used to improve the parameters using a variant of Gradient Descent called Adam. Repeat for a long time, and you get a trained RWKV model.

### GPT-mode
My simplified code processes the tokens one by one, which is much slower than processing them in parallel, especially when running on GPUs. For inference, there is no way around this, as we need to sample a token before we can use it to calculate the next one. However, for training, all the text is already available. This lets us parallelize across tokens. Most of the code is fairly straightforward to parallelize like this, as there is little dependence through time. For example, all the expensive matrix multiplications work on each token independently, leading to good performance.

However, the RWKV attention is inherently sequential. Fortunately, it has very little computation (on the order of 1024 times less than the matrix multiplications), so it should be fast. Sadly, pytorch does not have a good way of handling this sequential task, so the attention part becomes slow (even compared to the matrix multiplications). Therefore, I wrote optimized CUDA kernels for computing the RWKV attention, which has been my main contribution to the RWKV project.

JAX has jax.lax.scan and jax.lax.associative_scan, which allows a pure JAX implementation to perform better than pure pytorch. However, I still estimate that JAX would lead to [about 40% slower training compared to CUDA](https://discord.com/channels/992359628979568762/992363236370436136/1000838180674736158) (that estimate may be outdated, as it was made for training a relatively small 1.5B model).

# Contribute
RWKV is an open source community project. Join the [Discord](https://discord.gg/neCJHNcDCZ) and contribute! Or just ask questions or lurk.
