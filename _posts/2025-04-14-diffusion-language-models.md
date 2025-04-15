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
    Figure 2. AR-LM predicted the sequence token-by-token. Output tokens are used as input for the next token-prediction in left-to-right manner (top). DLM iteratively refines the entire output sequence from a noisy sample (bottom).
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

* **(Potential) Advantage of DLM**
    - **NAR** As sequence are generated holistically, you can fix and correct previous mistakes and refine the sequence as a whole.
    - **Controllability** Diffusion models are known to provide simple, training efficient style controlls using either classifier free guidance or classifier based guidance. Such controllability can be easily extended to DLM, where we can control the style of generation using the same techniques ([Prafulla et al., 2021](https://arxiv.org/abs/2105.05233), [Radford et al., 2021](https://arxiv.org/abs/2103.00020)). We can also take this a step further to even more fine-grained controles in lengths, specific text editing, infillings because the model works on the whole sequence representation iteratively ([Li et al, 2023](https://arxiv.org/abs/2205.14217), [Nie et al., 2025](https://arxiv.org/abs/2502.09992),). This may also extends to tasks with stronger structural constraints (eg., code, table).
    - **Diversity** Can produce more diverse outputs compared to beam search in AR. *You just need to sample different initial noise.*
    - **Speed Up** Poential for faster generation as stops *could* be paralleilized, as tokens are generated all-together, instead of waiting for previous outputs iteratively.

#### **Diffusion Model Recap**

Diffusion models are very successful and widely adopted in computer vision tasks, such as image generation, super-resolution, and inpainting. The core idea of diffusion models is to learn a generative model by reversing a diffusion process that gradually adds noise to the data. Using the famous [DDPM](https://arxiv.org/abs/2006.11239) as an example, given a data sample from a real data distribution $$\mathbf{x}_0 \sim \mathcal{D}(x)$$, we use a **forward process** to gradually perturb the data with small amounts of Gaussian noise over $$T$$ steps:

$$
\begin{align}
q(\mathbf{x}_t | \mathbf{x}_{t-1}) &= \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}) \\
q(\mathbf{x}_{1:T} | \mathbf{x}_0) &= \prod_{t=1}^T q(\mathbf{x}_t | \mathbf{x}_{t-1})
\end{align}
$$

where $$\beta_t \in (0, 1)$$ is a variance schedule that controls the amount of noise added at each step. As $$T \to \infty$$, $$\mathbf{x}_T$$ approaches a sample from a standard Gaussian distribution:

\begin{align}
\lim_{T \to \infty} \mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})
\end{align}

The **reverse process** then learns to gradually denoise the data, starting from pure noise $$\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$$ and working backwards, where $$\mu_\theta$$ and $$\Sigma_\theta$$ are learned by a fancy neural network model. Again, if you are not comfortable with these concepts, please refer to [Lilian's amazing blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

$$
\begin{align}
p_\theta(\mathbf{x}_{0:T}) &= p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) \\
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) &= \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))
\end{align}
$$

<div class="row mt-2">
    <div class="col-sm-8 col-md-6 mt-2 mt-md-0 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/DDPM.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise. (Image source: <a href="https://arxiv.org/abs/2006.11239">Ho et al. 2020</a> with a few additional annotations)
</div>


#### **What's the fundamental challenge?**

Well, if the Diffusion Model is well-established and it has all these exciting perks, why this is not as trending as in the field of computer vision? Good spot, now it comes to the fundamental challenge, the discrepancy between traditional **continuous** diffusion models, which achieved great success in image generation (such as with Denoising Diffusion Probabilistic Models - [Ho et al., 2020](https://arxiv.org/abs/2006.11239)), and the domain of **discrete text**.

Think about an image. It's made of pixels, and each pixel has colour values (like RGB) that are essentially numbers on a continuous scale. Adding "noise" is intuitive: you can slightly perturb these numerical values, typically by adding small random numbers drawn from a Gaussian (bell curve) distribution. Gradually adding more noise smoothly transitions the image to random static. The reverse process involves training a model to predict and subtract that small amount of noise at each step, gradually recovering the image.

Now, consider text. Language is composed of discrete units – words or tokens selected from a finite vocabulary (e.g., "cat", "dog", "runs"). You can't simply add "0.1 Gaussian noise" to the word "cat" and get something meaningful that's slightly different but still related in a smooth, continuous way. Applying the original continuous noise formulation directly just doesn't work.

This **discrete nature** of text is the core hurdle that requires specific adaptations, a challenge explored in models for discrete state-spaces like [Structured Denoising Diffusion Models (D3PM)](https://arxiv.org/abs/2107.03006). How do you define a "forward process" that gradually corrupts text into noise, and critically, a "reverse process" that a model can learn to invert step-by-step?

Researchers have developed clever workarounds to bridge this gap:

1.  **Operating on Continuous Variables:** One approach is to work not with the tokens themselves, but with their continuous variables. Traditional langauge models gives well-constructed word embedding and hidden outputs representations. We could leverage these representations to define a continuous forward process, where the model learns to predict the noise added to these continuous representations at each step. This is similar to how diffusion models operate in the image domain, where the forward process is defined in the latent space of a VAE or similar architecture. 
    * **Word Embedding (Token Level)** Noise *can* be added to these word embedding vectors, a technique used in models like [Diffusion-LM](https://arxiv.org/abs/2205.14217), and the pre-trained DLM [GENIE](https://arxiv.org/abs/2212.11685). However, mapping potentially noisy embeddings back to specific discrete tokens at each step introduces its own complexities.
    * **Higher Level Latent Representations** Works like [PLANNER](https://arxiv.org/abs/2306.02531) and [LD4LG](https://arxiv.org/pdf/2212.09462) are operating on a higher level latent representations of paragraphs of texts. However, the latent representations are fragile when perturbed by noise, resulting in abrupt semantic changes in reverse diffusion process. **My own paper** [SLD](https://arxiv.org/abs/2412.11333) attempts to mitigate this issue *cleverly* by text-segementation and improved representation learning techniques. Meta's [Large Concept Model](https://arxiv.org/pdf/2412.08821) is probably the *ultimate form* of pre-trained DLM following this path.
2.  **Discrete Corruption Surrogate:** More commonly, diffusion models for text define analogous discrete "noising" or "corruption" processes, as explored in papers like . Instead of adding mathematical noise, the forward process might involve:
    * **Masking:** Randomly replacing tokens with a special `[MASK]` token, with the number of masked tokens increasing over diffusion steps. The recent trending work of [LLaDA](https://arxiv.org/pdf/2502.09992), is the pre-trained DLM (8B parameters, largest of all kinds) following this path.
    * **Token Substitution:** Randomly replacing tokens with other tokens from the vocabulary.
    * **Hybrids:** Combining these or other discrete corruption methods.
3. **Discrete Diffusion:** Now a few *bold geniuses* are thinking, "Hey, if tokens are discrete, why not make the diffusion process discrete as well?". So we have discrete diffusion over categorical supports. [Eminel et al, 2021](https://arxiv.org/pdf/2102.05379) introduces extensions of diffusion for categorical data such as language. We model each token as a probability mass vector distributing over $$p \in \mathbb{R}^{V}$$, where $$V$$ is the size of the vocabulary and use transition matrices $$\mathbf{Q}$$ to model transitions between denoising timestops, for example $$q(\mathbf{x}_t |\mathbf{x}_{t-1}) = Categorical(\mathbf{x}_t; \mathbf{p}=\mathbf{x}_{t-1}\mathbf{Q}_t)$$.
Models like [D3PM](https://arxiv.org/abs/2107.03006), and [SEDD](https://arxiv.org/pdf/2310.16834) (*ICML 2024 Best Paper*) follows this path.


The reverse process then becomes about learning to **undo** this specific type of discrete corruption. For instance, the model learns to predict the original tokens at the `[MASK]` positions or identify and correct the randomly substituted tokens (as explored in the [Diffusion-LM paper](https://arxiv.org/abs/2205.14217)), iteratively refining the sequence from a highly corrupted state back to coherent text. So, while the core *idea* of diffusion (iterative refinement from noise) remains, the *mechanisms* for the forward (corruption) and reverse (denoising) processes have to be specifically adapted for the discrete world of language. 

Now I will select the representative work of each of these methods to further explain the concept of DLM.