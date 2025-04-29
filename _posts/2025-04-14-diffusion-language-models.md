---
layout: post
title: What are Diffusion Language Models?
date: 2025-04-14 11:59:00
description: A gentle, in-depth introduction of existing diffusion language models.
giscus_comments: true
related_posts: false
---

#### **Preface** 

[Dear Reader](https://www.youtube.com/watch?v=X0Jti9F-oQA), I feel like writing this blog for a long time. Diffusion model for language generation is an exciting and emerging field that receives increasing attention. However, up until the moment I am writing this blog, there isn't a comprehensive guide/intro covering such topics for the members of NLP/ML community who wish to conduct research in this area, prospectively. In this blog, we will walk through the history of diffusion language model, different paradigms in implementation, possible future research directions and applications (*also a few of my personal opinion, which might be biased, in italics*). The blog will be updated constantly, and I will probably turn this into a survey paper when the time is right.

This blog targets audience who already has sufficient knowledge in Diffusion Models, and traditional autoregressive LLMs. If you don't, no worries, here are resources I found extremely helpful.
* **For Diffusion Models**: 
    - [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) by Lilian Weng *Strongly recommended*
    - [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) by Calvin Luo

* **For Autoregressive LLMs**:
    - [Stanford CS224N](https://www.youtube.com/watch?v=LWMzyfvuehA&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=8) *GOAT course*

#### **What's diffusion language model (DLM)?**

A brief recap on all the trending autoregressive language model (AR-LM) nowadays, from GPT-2, Llama, to Gemini, ChatGPT, and Claude, these models are using transformer as the common backbone for **auoregressive** decoding (AR), that is predicting token by token in an left-to-right fashion. By contrast, diffusion language model (DLM) is iteratively refines and predicts the entire sequence from a sampled noisy input as **non-autoregressive** decoding (NAR). A straightforward (*simplified*) demonstration of the difference between two paradigms can be shown in the pipeline figure below.

<div class="row mt-2">
    <div class="col-sm-10 col-md-8 mt-2 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/diffusion_vs_ar.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. AR-LM predicted the sequence token-by-token. Output tokens are used as input for the next token-prediction in left-to-right manner (top). DLM iteratively refines the entire output sequence from a noisy sample (bottom).
</div>

Putting it into mathematical term is, given a sequence we want to predict, $$\mathbf{x} = \{x_1, x_2, \ldots x_N\}$$ , AR-LM with parameter $$\theta$$ models the following distribution.
$$
\begin{equation}
\label{eq:AR-LM}
P(\mathbf{x}; \theta) = \prod_{t=1}^{N} P(x_n | \mathbf{x}_{<n}; \theta)
\end{equation}
$$

Where DLM take a holistic view of the entire sequence, it models the following distribution, where $$t$$ is the timestep in **reverse diffusion process** (*we will get to the details very soon*). A larger $$t$$ means noiser sequence, closer to sampled from a standard gaussian noise. Consider we are having a very messy paragraph in the beginning and iteratively refines into the passage we want.

$$
\begin{equation}
\label{eq:DLM}
\mathbf{x}_{t-1} \sim p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t, t)
\end{equation}
$$

#### **Why we need DLM (or NAR)? What's wrong with AR?**

Autoregressive model is extremely successful nowadays, so why do we still need another paradigm? What makes DLM a field worth looking into? Well here are some arguments for you to consider.

* **Inherent Disadvantage of AR-LM**
    - **Error Propagation** For autoregressive models, if you made mistake in predicting the current token, there is no chance for going back for revision. Future predictions are always based on this flawed prediction, propagating and accumulating such errors. We call such struggle as [error propagation](https://aclanthology.org/D18-1396/).
    - **Indirect Generation Control** Current strategies for AR controlled generation depends on extensive training or hacks in decoding strategy. Such techniques are often indirect and inconvienient for users to control the generation. For example, if you want to generate a passage with certain length, you need to train a length predictor or use heuristics like [top-k sampling](https://arxiv.org/abs/1904.09751) to control the generation. And what's more, there is no guarantee 😥.
    - **Computational Constraints** Sequential token-by-token generation incurs high computational costs, and the left-to-right modeling limits effectiveness in reversal reasoning tasks, the "[Reversal Curse](https://arxiv.org/abs/2309.12288)".

* **(Potential) Advantage of DLM**
    - **NAR** As sequence are generated holistically, you can fix and correct previous mistakes and refine the sequence as a whole.
    - **Controllability** Diffusion models are known to provide simple, training efficient style controlls using either classifier free guidance or classifier based guidance. Such controllability can be easily extended to DLM, where we can control the style of generation using the same techniques ([Prafulla et al., 2021](https://arxiv.org/abs/2105.05233), [Radford et al., 2021](https://arxiv.org/abs/2103.00020)). We can also take this a step further to even more fine-grained controles in lengths, specific text editing, infillings because the model works on the whole sequence representation iteratively ([Li et al, 2022](https://arxiv.org/abs/2205.14217), [Nie et al., 2025](https://arxiv.org/abs/2502.09992),). This may also extends to tasks with stronger structural constraints (e.g., code, table).
    - **Diversity** Can produce more diverse outputs compared to beam search in AR. *You just need to sample different initial noise.*
    - **Speed Up** Poential for faster generation as stops *could* be paralleilized, as tokens are generated all-together, instead of waiting for previous outputs iteratively.

#### **Diffusion Model Recap**

Diffusion models are very successful and widely adopted in computer vision tasks, such as image generation, super-resolution, and inpainting. The core idea of diffusion models is to learn a generative model by reversing a diffusion process that gradually adds noise to the data. Using the famous [DDPM](https://arxiv.org/abs/2006.11239) as an example, given a data sample from a real data distribution $$\mathbf{x}_0 \sim \mathcal{D}(x)$$, we use a **forward process** to gradually perturb the data with small amounts of Gaussian noise over $$T$$ steps:

$$
\begin{equation}
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}) 
\end{equation}
$$
$$
\begin{equation}
q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t | \mathbf{x}_{t-1})
\end{equation}
$$

where $$\beta_t \in (0, 1)$$ is a variance schedule that controls the amount of noise added at each step. As $$T \to \infty$$, $$\mathbf{x}_T$$ approaches a sample from a standard Gaussian distribution:

$$
\begin{equation}
\lim_{T \to \infty} \mathbf{x}_T \approx \mathcal{N}(0, \mathbf{I})
\end{equation}
$$

The **reverse process** then learns to gradually denoise the data, starting from pure noise $$\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$$ and working backwards, where $$\mu_\theta$$ and $$\Sigma_\theta$$ are learned by a fancy neural network model. Again, if you are not comfortable with these concepts, please refer to [Lilian's amazing blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

$$
\begin{equation}
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
\end{equation}
$$


$$
\begin{equation}
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))
\end{equation}
$$

<div class="row mt-2">
    <div class="col-sm-8 col-md-6 mt-2 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/DDPM.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 2. The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise. (Image source: <a href="https://arxiv.org/abs/2006.11239">Ho et al. 2020</a> with a few additional annotations)
</div>


#### **What's the fundamental challenge?**

Well, if the Diffusion Model is well-established and it has all these exciting perks, why this is not as trending as in the field of computer vision? Good spot, now it comes to the fundamental challenge, the discrepancy between traditional **continuous** diffusion models, which achieved great success in image generation (such as with Denoising Diffusion Probabilistic Models - [Ho et al., 2020](https://arxiv.org/abs/2006.11239)), and the domain of **discrete text**.

Think about an image. It's made of pixels, and each pixel has colour values (like RGB) that are essentially numbers on a continuous scale. Adding "noise" is intuitive: you can slightly perturb these numerical values, typically by adding small random numbers drawn from a Gaussian (bell curve) distribution. Gradually adding more noise smoothly transitions the image to random static. The reverse process involves training a model to predict and subtract that small amount of noise at each step, gradually recovering the image.

Now, consider text. Language is composed of discrete units – words or tokens selected from a finite vocabulary (e.g., "cat", "dog", "runs"). You can't simply add "0.1 Gaussian noise" to the word "cat" and get something meaningful that's slightly different but still related in a smooth, continuous way. Applying the original continuous noise formulation directly just doesn't work.

This **discrete nature** of text is the core hurdle that requires specific adaptations. How do you define a "forward process" that gradually corrupts text into noise, and critically, a "reverse process" that a model can learn to invert step-by-step?

Researchers have developed clever workarounds to bridge this gap:

1.  **Operating on Continuous Variables:** One approach is to work not with the tokens themselves, but with their continuous variables. Traditional langauge models gives well-constructed word embedding and hidden outputs representations. We could leverage these representations to define a continuous forward process, where the model learns to predict the noise added to these continuous representations at each step. This is similar to how diffusion models operate in the image domain, where the forward process is defined in the latent space of a VAE or similar architecture. 
    * **Word Embedding (Token Level)** Noise *can* be added to these word embedding vectors, a technique used in models like [Diffusion-LM](https://arxiv.org/abs/2205.14217), and the pre-trained DLM [GENIE](https://arxiv.org/abs/2212.11685). However, mapping potentially noisy embeddings back to specific discrete tokens at each step introduces its own complexities.
    * **Higher Level Latent Representations** Works like [PLANNER](https://arxiv.org/abs/2306.02531) and [LD4LG](https://arxiv.org/pdf/2212.09462) are operating on a higher level latent representations of paragraphs of texts. However, the latent representations are fragile when perturbed by noise, resulting in abrupt semantic changes in reverse diffusion process. **My own paper** [SLD](https://arxiv.org/abs/2412.11333) attempts to mitigate this issue *cleverly* by text-segementation and improved representation learning techniques. Meta's [Large Concept Model](https://arxiv.org/pdf/2412.08821) is probably the *ultimate form* of pre-trained DLM following this path.
2.  **Discrete Diffusion over Tokens:** Now a few *bold geniuses* are thinking, "Hey, if tokens are discrete, why not make the diffusion process discrete as well?". So we have discrete diffusion over categorical supports. [Eminel et al, 2021](https://arxiv.org/pdf/2102.05379) introduces extensions of diffusion for categorical data such as language. We model each token as a probability mass vector distributing over $$p \in \mathbb{R}^{V}$$, where $$V$$ is the size of the vocabulary and use transition matrices $$\mathbf{Q}$$ to model transitions between denoising timestops, for example $$q(\mathbf{x}_t |\mathbf{x}_{t-1}) = Categorical(\mathbf{x}_t; \mathbf{p}=\mathbf{x}_{t-1}\mathbf{Q}_t)$$.
Models like [D3PM](https://arxiv.org/abs/2107.03006), and [SEDD](https://arxiv.org/pdf/2310.16834) (*ICML 2024 Best Paper*) follows this path. More commonly, diffusion models for text define analogous discrete "noising" or "corruption" processes, as explored in papers like . Instead of adding mathematical noise, the forward process might involve:
    * **Masking:** Randomly replacing tokens with a special `[MASK]` token, with the number of masked tokens increasing over diffusion steps. The recent trending work of [LLaDA](https://arxiv.org/pdf/2502.09992), is the pre-trained DLM (8B parameters, largest of all kinds) following this path.
    * **Token Substitution:** Randomly replacing tokens with other tokens from the vocabulary.
    * **Hybrids:** Combining these or other discrete corruption methods.

3. 🔥🖼️👊  **The Maverick--Text in Image:** *"Discrete text? What text? It's an image!"* I personally wish you to checkout this [GlyphDiffusion](https://arxiv.org/pdf/2304.12519). Instead of dealing with the discrepancy, they bypassed by rendering the target text as a glyph image containing visual language content 🤣.

The reverse process then becomes about learning to **undo** this specific type of discrete corruption. For instance, the model learns to predict the original tokens at the `[MASK]` positions or identify and correct a randomly sampled variable into meaningful tokens , iteratively refining the sequence from a highly corrupted state back to coherent text. So, while the core *idea* of diffusion (iterative refinement from noise) remains, the *mechanisms* for the forward (corruption) and reverse (denoising) processes have to be specifically adapted for the discrete world of language. 

Now I will select the representative work of each of these methods to further explain the concept of DLM. <span style="color:blue"> For each of the paradigm, I will introduce papers explaining the mechanism and direct you to an off-the-shelf pre-trained model for you. </span>

#### **Embedding-Level diffusion -- where it begins**

##### **1.Token-level Embeddings**
*As far as I know*, [Diffusion-LM](https://arxiv.org/abs/2205.14217) is probably the first, influencial work that starts the era of DLM. Now suppose we have a sequence of words $$\mathbf{w} = \{w_1, w_2, \ldots, w_n\}$$, an embedding fucntion is defined to map each word to a word vector $$Emb(w_i) \in \mathbb{R}^{d}$$. So the entire sequence weill be encoded into $$\mathbf{x}_0 = Emb(\mathbf{w}) \in \mathbb{R}^{d}$$. So yeah! There we go, we have a **continuous** space where we can run the conventional diffusion models. We use the typical simplified KL-divergence term in evidence lower bound (which I will not reiterate here) to derive a loss function.

$$
\begin{equation}
\mathcal{L}_{simple}(\mathbf{x}_0)  = \Sigma_{t=1}^T \underset{q(\mathbf{x}_t | \mathbf{x}_0)}{\mathbb{E}} ||\mu(\mathbf{x}_t, t) - \hat{\mu}(\mathbf{x}_t, \mathbf{x}_0)||^2
\end{equation}
$$

But don't forget, we need to convert the embedding back to discrete tokens, you will say this is easy, let's just have another function transform them back to tokens. Indeed, that's how it's done. In Li's implementation, they model theses steps into the diffusion process as an extra timestep. As shown in the figure below, the forward process consists of an additional Markov transition to obtain the embeddings parametrized by 
$$q_{\phi}(\mathbf{x}_0 | \mathbf{w}) = \mathcal{N}(Emb(\mathbf{w}); \sigma_{0} I)$$
. Then in the reverse process you will have an additional trainble **rounding** step, parameterized by 
$$p_{\theta}(\mathbf{w} | \mathbf{x}_0) = \prod_{i=1}^n p_{\theta}(w_i | x_i)$$
, where $$p_{\theta}(w_i | x_i)$$ is a simple softmax distribution.

<div class="row mt-2">
    <div class="col-sm-10 col-md-8 mt-4 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/diffusion-lm.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 3. A graphical model representing the forward and reverse diffusion processes for Diffusion-LM. (Image source: <a href="https://arxiv.org/abs/2205.14217">Li et al. 2022</a>)
</div>

Then we can adjust the loss function accordingly. For end-to-end training, we will have the final loss function as below. During the reverse process, you run the inferece by sampling a random embedding sequence containing $$n$$ token embeddings (fixed, the same as in the training process) and gradually remove the noise. You are always predicting $$n$$ embeddings together, as the diffusion model requires a fixed shape I/O (A bit wasteful when the sequence is underfull right? We will talk about it later.). 

$$
\begin{equation}
\mathcal{L}_{simple}^{e2e}(\mathbf{w}) = \underset{q_{\phi}(\mathbf{x}_{0:T} | \mathbf{w})}{\mathbb{E}} \left[\underbrace{\mathcal{L}_{simple}(\mathbf{x}_0)}_{\text{diffusion Loss}} + ||Emb(\mathbf{w}) - \overbrace{\mu_{\theta}(\mathbf{x}_1, 1)}^{\text{predicted }\mathbf{x}_0}||^2  - \underbrace{\log p_{\theta}(\mathbf{w} | \mathbf{x}_0)}_{\text{rounding}} \right]
\end{equation}
$$

So, everything seems extremely straightforward right? Or does it? Unfortunately, no 😅. The conversion between continous embedding space and discrete tokens is non-trivial, harder than you think. <span style="color:red">This rounding is a key challenge in token embedding-level diffusion models. The discretization step can lead to errors that accumulate across the diffusion process, as the embedding space is not uniformly populated with valid tokens.</span> *Well isn't this the notorious data sparsity revisited.*

In the paper, there is a major chapter talking about the techniques of how they managed to reduce the rounding error to obtain admissible outputs. For example using reparameterisation trick to make sure every term in the loss models $$\mathbf{x}_0$$ explicitly. They also introduce a **clamping trick** that is maping the predicted vector $$\mathbf{x}_t$$ to its nearest word embedding sequence in every reverse diffusion sampling step. Still, a lot of work needs to be done in the future for the sake of generation quality.

Back to the model, with the diffusion pipeline, you can do the fancy conditioning and controlled generation during your inference now. For example, you could have a separate neural network classifier and a class condition $$\mathbf{c}$$. During the backward process, you obtain the $$\mathbf{x}_{t-1}$$ with respect to the posterior probability using the gradient update below.

$$
\begin{equation}
\nabla_{\mathbf{x}_{t-1}} \log p(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{c}) = \nabla_{\mathbf{x}_{t-1}} \log p(\mathbf{x}_{t-1} | \mathbf{x}_{t}) + \underbrace{\nabla_{\mathbf{x}_{t-1}} \log p(\mathbf{c} | \mathbf{x}_{t-1})}_{\text{classifier guidance}}
\end{equation}
$$

<div class="row mt-2">
    <div class="col-sm-10 col-md-8 mt-4 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/diffusion-lm-classifier.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 4. For controllable generation, we iteratively perform gradient updates on these continuous latents to optimize for fluency (parametrized by Diffusion-LM) and satisfy control requirements (parametrized by a classifier). (Image source: <a href="https://arxiv.org/abs/2205.14217">Li et al. 2022</a>)
</div>

The paper provides many experiments for controlled generation (e.g., semantics, length, POS and etc.). I want to mention about **infilling** specifically. That is during the inference process, instead of denoising all the embeddings, some context embeddings are given and fixed (e.g., $$\mathbf{x}_t = $$`[w_1] [noise] [w_2]`), the diffusion model will mask the gradient of other tokens, only generated the noised token in the middle. However, the other embeddings are used together as conditions naturally during the reverse sampling, as a **classifier-free guidance**. And you, my clever reader, immediately understands how traditional sequence-to-sequence task can be modelled by giving the input as left contexts only.

[GENIE](https://arxiv.org/pdf/2212.11685) is the first pre-trained DLM (*again, as I know*) following this token embedding-level diffusion. The model uses **continuous paragraph denoise** objective for pre-training. The object first applied a hybrid noising techniques to the original text, including token masking (e.g., `[w_1] [MASK] [w_2]`) and forward diffusion process sampling, then ask the model to recover the clean original text.

<div class="row mt-2">
    <div class="col-sm-10 col-md-8 mt-4 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/GENIE.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 5. The framework of GENIE. Source sequence is encoded as the condition of the transformer DLM through cross attention. DLM restores the randomly initial Gaussian noise to the output text through the iterative denoising and grounding process (Image source: <a href="https://arxiv.org/pdf/2212.11685">Lin et al., 2022</a>)
</div>

Notably, GENIE doesn't use infilling as the default sequence-to-sequence generation method. Instead, it is using a more Encoder-Decoder approach (yes, like BART or T5), which is another analogues of classifier-free guidance. Similar rounding techniques is applied, they are using an effective KNN algorithm to retreive closets word embedding of each token during reverse sampling and apply rounding to the closest learnt word embedding.

##### **2. Higher-level Embeddings**

To tackle this **rounding error** of translating predicted word-embeddings to discrete tokens, attempts have been made to bring diffusion to high-level semantic latent space (e.g., sentences, paragraphs). The diffusion will predict some latent representation of a piece of text, and a separate autoregressive decoder is used for decoding the representation back to text. We use the work of Lovelace et al., Latent Diffusion For Language Generation ([LD4LG](https://arxiv.org/abs/2212.09462)) as an example to explain the concept.

<div class="row mt-2">
    <div class="col-sm-10 col-md-8 mt-4 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/LD4LG.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 6. Overview of Latent Diffusion For Language Generation (Image source: <a href="https://arxiv.org/abs/2212.09462">Lovelace et al., 2023</a>)
</div>

Firstly, an encoder model (e.g. BART or T5 Encoder) $$Enc()$$ to convert the piece of text $$\mathbf{w} = {w_1, \ldots, w_n}$$ into hidden states, $$Enc(\mathbf{w}) \in \mathbb{R}^{n \times h_{lm}}$$, where $$h_{lm}$$ is the hidden state dimension of your encoder (usually 768 or 1024...). Notice that, $$n$$ here represents the variable token length of the text, and the diffusion requires a fixed shape representation. In the token-level embedding diffusion, this is a constraint for the context window. For higher-level embeddings, an additional pooling/compressing unit $$f()$$ is used to project the hidden states to a fixed-length latent representation $$\mathbf{z} = f(Enc(\mathbf{w})), \mathbf{z} \in \mathbb{R}^{k \times h_{rep}}$$, where $$k$$ and $$h_{rep}$$ are the dimension of the latent representation (hyperparameters). Typically you want $$k<n$$ and $$ h_{rep} << h_{lm}$$, as you don't want a very sparse latent representation, a more compact latent space helps diffusion model to learn the distribution more easily.

Then the diffusion model 
$$R(\cdot |\theta)$$
, which is usually a diffusion transformer ([DiT](https://arxiv.org/abs/2212.09748)) deployed to learn how to recover $$\mathbf{z}_0$$ from its noised version over the forward diffusion process as usual.

$$
\begin{equation}
    \mathcal{L}(\theta_{R}) = \sum_{t=1}^T \underset{q(\mathbf{z}_{t} | \mathbf{z}_0)} {\mathbb{E}} \left\| \hat{\mathbf{z}}_{t} - \mathbf{z}_t \right\|^2_2
\end{equation}
$$

For inference, you sample a latent representation from gaussian noise, and do the reverse process as usual $$\hat{\mathbf{z}}_{t-1} =  R(\hat{\mathbf{z}}_t, t); \theta_{R}$$. Conditioning can be added to forward and backward process similarly as well. Then we employ a reconstruction unit $$g()$$, projecting the latent state back to the dimension of the backbone language model hidden state, using the decoder to generate text $$Dec(g(\mathbf{z}_0))$$. In this case, as we are always doing diffusion over the continuous latent space of high level semantics, we bypass the phase of rounding.

**BUT**, again, is it that simple? Actually, no ☹️. <span style="color:red"> It is non-trivial to find a good latent representation for text, not mentioning generate one from noisy examples. </span> For example, if I perturb a latent representation of sentence `It is sunny today` a bit, in this case, what are we doing here? Just as it's name that high-level semantic diffusion, it should preturb it's semantics, not surface forms (tokens) right? We want to make sure I add a small noise to the sentence, it should give me some thing similar in meaning like `Today is sunny` or `Today has sun` some thing like that, definitely not `It isn't sunny today` or `It is savvy tody` which is similar by appearance but not in meaning. But this is extremely hard to do, as a the meaning of a paragraph of text is extremely rich and diverse, how are we suppose to regularize that? This pose a great challenge for high-level semantic diffusion doing long-form generation.

So back to square one, what's the definition of a good latent representation for high-level semantics diffusion? In [Zhang et al., 2024](https://arxiv.org/pdf/2306.02531) gives a formal definition of this, describing the desirata in three aspects.

- **Low conversion error** Give you a piece of text $$\mathbf{w}$$, we can transform it into latent representations and convert it back, $$\tilde{\mathbf{w}} = Dec(g(f(Enc(\mathbf{w}))))$$. The difference between $$\mathbf{w}$$ and $$\tilde{\mathbf{w}}$$ should be minimal, or none. However this is not very achievable when $$\mathbf{w}$$ is long. The hyperparameter $$k, h_{rep}$$ are fixed, when we are doing projections, we are losing information. The longer the sequence, the more information is lost, right?

- **Local Smoothness** Consider if a person stutters or drop a few minor words while speaking, you can still recover what the person is trying to convey. That said, given a piece of text $$\mathbf{w}$$, and its slightly variant version $$\mathbf{w'}$$, their encoded latent representation should not differ to much, $$\mathbf{z}_{w} \approx \mathbf{z}_{w'}$$.

- **Distributional Smoothness** In the high level latent space, we want the meanings of paragraphs are distributed smoothly. That is consider you have a piece of text $$\mathbf{w}$$ and it's latent representation $$\mathbf{z}_{w}$$, in the latent space it should be closer to similar meanings, like paraphrases of $$\mathbf{w}$$. When we are preturbing the text with a small amount of text, the decoded representation shouldn't differ to much by its meaning, as mentioned above. Vice versa, text with different meanings should be far away from each other in the latent space. However, when the sequence gets longer it's hard to control this as it contains too much complex concepts and meanings. If we increase the size of the latent space, the diffusion model may face difficulty in learning a distribution, $$p(\mathbf{z})$$ that is highly multimodal, or has density that are associated with a large Lipchitz constant (i.e., has abrupt changes).

So the fact is without proper regularization, the learned distribution may be susceptible to abrupt semantic changes due to small perturbations, increasing the difficulty of the task for the diffusion model and catastrophically corrupts the quality of decoded texts. So how do we fix this?

[Zhu et al., 2024](https://arxiv.org/pdf/2412.11333) (*yeah that's me 🤪*) provides Segment-Level Diffusion. Extending the concept **patching** from image generation, we patch texts into segments (e.g. sentences, dialogue utterances). This way we have effectively control the length of text and the meanings in the segment. We further regulate the latent representations by doing additional training for representation learning, using contrastive learning, and adversial noise preturbation, ensuring local and distributional smoothness. The diffusion model will now predict multiple latent representations with one-to-one correspondence to each segments. These representations will be independently decoded in parallel.

<div class="row mt-2">
    <div class="col-sm-12 col-md-10 mt-6 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/SLD.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 7. Overview of the training pipeline of SLD. In the first stage, gold output is divided into segments. In the second stage, we use contrastive and adversarial learning to ensure latent representations are robust to drastic semantic changes. Finally, we train a diffusion model as an inherent semantic planner conditioned on given inputs. (Image source: <a href="https://arxiv.org/pdf/2412.11333">Zhu et al., 2024</a>)
</div>

A contemporary work from Meta, [Large Concept Model](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/) uses a similar paradigm. They are doing diffusion over concepts, which is the segment in my case. Do check their work out! They provide the model pre-trained on much more data than I have! *I was submitting mine for 15th of Dec 2024 ARR, but they released their paper on the 12th. Well I'd be lying if I say I am not a bit frustrated. But hey! This also proves my idea, our idea, is very promising! Hope this is helpful to you!*

<div class="row mt-2">
    <div class="col-sm-12 col-md-10 mt-6 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/LCM.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 8. Left: visualization of reasoning in an embedding space of concepts (task of summarization). Right: fundamental architecture of an Large Concept Model (LCM). Note that the LCM is a diffusion model! (Image source: <a href="https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/"> LCM Team 2024</a>)
</div>

#### **Discrete Diffusion over Tokens**

In this section, we describe how diffusion over tokens is done using Masked Diffusion Model (MDM) by using examples of Sahoo et al., 's [Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524v2) as a concrete sample. Please also check out [D3PM](https://arxiv.org/abs/2107.03006), and [SEDD](https://arxiv.org/pdf/2310.16834) for more details, but they are too mathematically dense and terse for this introduction blog.

As mentioned before, [D3PM](https://arxiv.org/abs/2107.03006) introduces the Markov forward process as categorical distributions using multiplications of matrices $$\mathbf{Q}_{t}$$ over $$T$$ discrete timesteps. We have a series of multiplication $$\mathbf{Q}_{T-1}\cdot\mathbf{Q}_{T-1} \ldots \mathbf{Q}_{1} \mathbf{x}$$ that converges to a stationary distribution.

<div class="row mt-2">
    <div class="col-sm-12 col-md-10 mt-6 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/MDLM.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    (Left) Masked diffusion language model (MDLM) is trained using a weighted average of masked cross entropy losses. (Top Right) In comparison to masked language models (MLM), MDLM's objective correspond to a principled variational lower bound, and supports generation via ancestral sampling.(Bottom Right) Perplexity (PPL) on One Billion Words benchmark. (Image source: <a href="https://arxiv.org/abs/2406.07524v2"> Sahoo et al., 2024</a>)
</div>

In their work, they have tokens represented as 
$$\mathbf{x} \in \mathcal{V}$$, where $$\mathcal{V}, |\mathcal{V}| = K$$
is a set of all the one-hot vectors of the vocabulary. They define the $$Cat(\cdot;\mathbf{\pi})$$ as the categorical distribution over K token classes with probabilities given by $$\mathbf{\pi} \in \Delta ^ K$$, where $$\Delta ^ k$$ denotes the $$K$$-simplex. They also define a special `[MASK]` token $$\mathbf{m} \in \mathcal{V}$$.

During the forward process, they interpolate the discrete diffusion by converting $$\mathbf{x}$$ into increasingly noisy variables $$\mathbf{z}_t$$. The marginal of $$\mathbf{z}_t$$ conditioned on $$\mathbf{x}$$ is given below, $$\alpha$$ is still the term derived from noise schedules in standard diffusion process.

$$
\begin{equation}
q(\mathbf{z}_t | \mathbf{x}) = Cat(\mathbf{z}_t; \alpha_t \mathbf{x} + (1-\alpha_t)\mathbf{\pi})
\end{equation}
$$

During masked diffusion, $$\pi = \mathbf{m}$$, which means at each noising step, t, the input x transitions to a "masked" state $$\mathbf{m}$$ with some probability. If an input transitions to $$\mathbf{m}$$ at any time $$t$$, it will remain as the masked token afterwards.

$$
\begin{equation}
q(\mathbf{z}_s \mid \mathbf{z}_t, \mathbf{x}) =
\begin{cases}
\text{Cat}(\mathbf{z}_s; \mathbf{z}_t), & \mathbf{z}_t \neq \mathbf{m},\\
\text{Cat}\left( \mathbf{z}_s; \frac{(1 - \alpha_s)\mathbf{m} + (\alpha_s - \alpha_t)\mathbf{x}}{1 - \alpha_t} \right), & \mathbf{z}_t = \mathbf{m}. 
\end{cases}
\end{equation}
$$

So, consequently, for the reverse diffusion process, we train a network to do $$p_{\theta} (\mathbf{z}_s | \mathbf{z}_t)$$ to convert masked tokens back to concrete tokens.

$$
\begin{equation}
p_\theta(\mathbf{z}_s \mid \mathbf{z}_t) = q(\mathbf{z}_s \mid \mathbf{z}_t, \mathbf{x} = \mathbf{x}_\theta(\mathbf{z}_t, t)) =
\begin{cases}
\text{Cat}(\mathbf{z}_s; \mathbf{z}_t), & \mathbf{z}_t \neq \mathbf{m}, \\
\text{Cat}\left(\mathbf{z}_s; \frac{(1 - \alpha_s)\mathbf{m} + (\alpha_s - \alpha_t)\mathbf{x}_\theta(\mathbf{z}_t, t)}{1 - \alpha_t} \right), & \mathbf{z}_t = \mathbf{m}.
\end{cases}
\end{equation}
$$

Great, right? In this case they have successfully extended the diffusion from continous into discrete domain. I've omitted a few details which you can check out in detail in their [blog](https://s-sahoo.com/mdlm/), or their paper. However, in their project, note that their design is not flexible enough. *While the prototype naturally has some limitations typical of early-stage systems, it marks a significant advancement in the field.* For example, during decoding, the token stays the same after it is unmasked. This is not ideal, as in early stage of the reverse diffusion, the entire paragraph is noisy, and the few tokens decoded will are probably not optimal. But since they are fixed, your later decoding process is conditioned on these tokens causing **error-propagation** again. [Lin et al.,](https://arxiv.org/abs/2302.05737) from [Lingpeng Kong's Group](https://ikekonglp.github.io/) (*I like their work a lot*) therefore uses a **routing** technique to mitigate this issue. Basically they model that a decoded token can be remasked into `[MASK]` or noised token during the decoding stage if model's confidence on that token is low.

<div class="row mt-2">
    <div class="col-sm-12 col-md-10 mt-6 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/LLaDA.svg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    (a) Pre-training. LLaDA is trained on text with random masks applied independently to all tokens at the same ratio t ∼ U[0, 1]. (b) SFT. Only response tokens are possibly masked. (c) Sampling. LLaDA simulates a diffusion process from t = 1 (fully masked) to t = 0 (unmasked), predicting all masks simultaneously at each step with flexible remask strategies. (Image source: <a href="https://arxiv.org/abs/2502.09992"> Nie et al., 2025</a>)
</div>

Finally, [LLaDA 8B](https://arxiv.org/abs/2502.09992) combines both techniques above, MDLM and routing, demonstrating the potential of MDM as a new paradigm of pre-trained language models with on-par and even superior generation quality than AR LLMs in the same size. Amazing isn't it? Try [it](https://huggingface.co/spaces/multimodalart/LLaDA) out yourself!


#### **Text-in-Image Diffusion? (Brain-teaser)**

Think CV! Nowadays, we are borrowing concepts from each other for a long time. Like [VAR](https://arxiv.org/abs/2404.02905) styled auto-regressive image generation. So maybe we can go a step further since we are already using diffusion, the most trending paradigm in CV. But I will skip great CV works like [DeepFloyd IF](https://github.com/deep-floyd/IF) for conciseness. As a brain-teaser, I will just introduce this [GlyphDiffusion](https://arxiv.org/pdf/2304.12519v2). Their key idea is to render the target text as a glyph image containing visual language content. The conditional text generation can be cast as a glyph image generation task, and it is then natural to apply continuous diffusion models to discrete texts. *Let's stop worrying whether this is scalable or not, at least it's fun.*

<div class="row mt-2">
    <div class="col-sm-10 col-md-8 mt-4 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/GlyphDiffusion.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    GlyphDiffusion generate patches of image containing visual language contents. (Image source: <a href="https://arxiv.org/abs/2304.12519v2"> Li et al., 2023</a>)
</div>

#### **Challenges and Oppurtunities**

You might wonder, "Hey after reading through this blog, seems like the concept of DLM is well-established as we have already have all the pre-trained models already. What stops us from using it?" Well, the major challenge is **speed**, the efficiency and computational cost during inference sampling. I know that we claimed NAR methods could be potentially faster than traditional AR methods, but we haven't achieve that so far. AR methods is benefiting from acceleration techniques like KV caching, which is not applicable for NAR generation. Now naive DLM implementations is about 50 times slower than AR methods. **But!** Don't get frustrated, we are gradually adapting the techniques of acceleration to this field, (e.g., [Consistency Models](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/), [Adaptive Decay Sampling](https://arxiv.org/abs/2305.04465), etc.). Yeah, another work ([Ye et al., 2024 Diffusion of Thoughts](https://arxiv.org/abs/2402.07754)), Diffu from Kong's group has succesfully used ODE solver to speed up the decoding process. I believe we will catch up soon. And I think some institute has already done it, check out Inception Labs' [Mercury Coder](https://chat.inceptionlabs.ai/), the first, fast, commercial DLM!

There are lots of potential of DLMs yet to be explored. For example, since we are generating everything all together, how does this changes searching/reasoning? Will this give us better CoT results, avoiding bad trajectories if the model can look ahead? Checkout [Diffusion-of-Thought](https://arxiv.org/abs/2402.07754) and [Dream 7B Diffusion Reasoning Model](https://hkunlp.github.io/blog/2025/dream/)! Since we can use diffusion to do generation with fine-grained controls such as infilling easily, we can generate better data with complex grammar/format constraints (e.g., tables, code) with guarantees. Check out [TabDiff](https://arxiv.org/pdf/2410.20626) and [Mario et al., 2024](https://arxiv.org/abs/2407.02549)'s work. The random sampling of diffusion can also help us augment data easily, which might be useful for expensive/rare data, such as low-resource langauge. [Chen et al, 2024](https://aclanthology.org/2024.emnlp-main.109.pdf) used this for low-resource sentiment classification. We can even finally start unifying all the modalities, generating image and tokens all at once, like Meta's [Transfusion](https://ai.meta.com/research/publications/transfusion-predict-the-next-token-and-diffuse-images-with-one-multi-modal-model/)! *But I think we have to call it FusionTrans, as they are using AR for image, we are using Diffusion for text.*

Yeah, these people are continuously updating the list of existing [DLM papers](https://github.com/bansky-cl/diffusion-nlp-paper-arxiv).


##### **Epilogue**
I will end this introductory blog here without a conclusion, as the Era of DLM has just begun 🚀 ! I hope this blog helps you in understanding DLMs, and make you better prepared or even sparkled some ideas if you want to do any DLM related research. *Oh Gosh, I hope I didn't write too much dad jokes.* If you guys like it, I will probably write more blogs to describe some aspects in depth in the future.
If you want to do me a favour and find this blog helpful, you can cite my [Segment-level Diffusion](https://arxiv.org/abs/2412.11333) paper in your paper, as I've included most of the blogs content into my related work section. Leave a comment if you have any suggestions for the blog. 

```bibtex
@misc{zhu2024segmentleveldiffusionframeworkcontrollable,
  title={Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models}, 
  author={Xiaochen Zhu and Georgi Karadzhov and Chenxi Whitehouse and Andreas Vlachos},
  year={2024},
  eprint={2412.11333},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2412.11333},
}
```

Until next time...