---
layout: post
title: What are Diffusion Language Models?
date: 2025-04-14 11:59:00
description: A gentle, in-depth introduction of existing diffusion language models.
giscus_comments: true
related_posts: false
---

### **Preface** 

[Dear Reader](https://www.youtube.com/watch?v=X0Jti9F-oQA), I feel like writing this blog for a long time. Diffusion model for language generation is an exciting and emerging field that receives increasing attention. However, up until the moment I am writing this blog, there isn't a comprehensive guide/intro covering such topics for the members of NLP/ML community who wish to conduct research in this area, prospectively. In this blog, we will walk through the history of diffusion language model, different paradigms in implementation, possible future research directions and applications (*also a few of my personal opinion, which might be biased, in italics*). The blog will be updated constantly, and I will probably turn this into a survey paper when the time is right.

This blog targets audience who already has sufficient knowledge in Diffusion Models, and traditional autoregressive LLMs. If you don't, no worries, here are resources I found extremely helpful.
* **For Diffusion Models**: 
    - [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) by Lilian Weng *Strongly recommended*
    - [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) by Calvin Luo

* **For Autoregressive LLMs**:
    - [Stanford CS224N](https://www.youtube.com/watch?v=LWMzyfvuehA&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=8) *GOAT course*

#### What's diffusion language model (DLM)?

A brief recap on all the trending autoregressive language model (AR-LM) nowadays, from GPT-2, Llama, to Gemini, ChatGPT, and Claude, these models are using transformer as the common backbone for **auoregressive decoding**, that is predicting token by token in an left-to-right fashion. By contrast, diffusion language model (DLM) is iteratively refines and predicts the entire sequence from a sampled noisy input as **non-autoregressive decoding**.

<div class="row mt-2">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/diffusionlm_blog/diffusion_vs_ar.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. AR-LM predicted the sequence token-by-token. Output tokens are used as input for the next token-prediction in left-to-right manner (top). DLM iteratively refines the entire output sequence from a noisy sample (bottom).
</div>


\begin{equation}
\label{eq:cauchy-schwarz}
P(\mathbf{x}; \theta) = \prod_{t=1}^{T} P(x_t | \mathbf{x}_{<t}; \theta)
\end{equation}
