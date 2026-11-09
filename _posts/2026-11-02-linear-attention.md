---
layout: post
title: "Linear Attention"
date: 2026-11-10 00:30:00 +0800
description: The mathematics and code implementation of linear attention.
tags: machine-learning
categories: notes
---

> Recently, all eight Harry Potter movies were added to Amazon Prime, which I happen to subscribe to, just in time for a long weekend in Singapore. Unsurprisingly, this made for a wonderfully unproductive weekend.

This post serves as a note or reference for my own personal use, but I think it's an interesting topic to write about. I also thought it'd be fun to write a post with a lot of LaTEX math.

Attention is a mechanism that allows information to be passed on from one token to another. It is a crucial part of the transformer model which enables modern day LLMs.

In the attention mechanism, we have three tensors obtained by projecting token representations with learned weights:

* Query $\mathbf{Q}$
* Key $\mathbf{K}$
* Value $\mathbf{V}$

In batched multi-head attention, $$\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{B \times T \times H \times D_k}$$ where $$B$$ is the batch size, $$T$$ is the sequence length, $$H$$ is the number of heads, and $$D_k$$ is the dimension of the key vector, while $$\mathbf{O}, \mathbf{V} \in \mathbb{R}^{B \times T \times H \times D_v}$$ where $$D_v$$ is the dimension of the value vector (and the rest are the same). For the following complexity notations, we let head dimension $d$ be equal to $$D_k = D_v$$.

Matrix equations apply independently to each batch example and head, with tokens stored as rows. The softmax formulation here omits the usual scaling factor $1/\sqrt{D_k}$ and describes an unscaled causal attention. Here, $\mathbf{O}$ denotes the per-head output before concatenation and output projection.

The output $\mathbf{O}$ of $\text{attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ is given by:

$$
\mathbf{O} = \text{softmax}(\mathbf{Q}\mathbf{K}^{\mathsf{T}} + \mathbf{M})\mathbf{V}
$$

$\mathbf{M}$ is the causal mask:

$$
\mathbf{M}_{ij} = \begin{cases}
0, & j \le i, \\
-\infty, & j > i.
\end{cases}
$$

For some step $t$, the output of softmax attention is:

$$
\begin{aligned}
\mathbf{o}_t^{\mathsf{T}} &= \text{softmax}(\mathbf{q_t}^{\mathsf{T}}\mathbf{K}_{1:t}^{\mathsf{T}} )\mathbf{V}_{1:t} \\
&= \sum_{i=1}^{t}
\frac{
\exp\!\left(\mathbf{q}_t^{\mathsf T}\mathbf{k}_i\right)
}{
\sum_{j=1}^{t}
\exp\!\left(\mathbf{q}_t^{\mathsf T}\mathbf{k}_j\right)
}
\mathbf{v}_i^{\mathsf T}
\end{aligned}
$$

Here's an implementation of softmax attention using PyTorch:

```py
def softmax_attn(
    q: Float[Tensor, "B T H Dk"],
    k: Float[Tensor, "B T H Dk"],
    v: Float[Tensor, "B T H Dv"],
) -> Float[Tensor, "B T H Dv"]:
    T = q.shape[1]
    
    scores = einops.einsum(q, k, "B i H Dk, B j H Dk -> B H i j")
    mask = t.tril(t.ones([T, T], dtype=t.bool, device=q.device))
    scores = scores.masked_fill(~mask, -float("inf"))
    
    o = einops.einsum(t.softmax(scores, dim=-1), v, "B H i j, B j H Dv -> B i H Dv")
    return o
```

Softmax attention computes all query-key similarities. For sequence length $T$ and head dimension $d$, this requires $\mathcal{O}(T^2d)$ work per head and batch example. The explicit score matrix uses $\mathcal{O}(T^2)$ memory, although implementations such as FlashAttention avoid materializing the full matrix.

## Linear Attention

Linear attention replaces softmax attention with a kernel-based formulation that enables linear-time computation, as described by [Katharopoulos et al. (2020)](https://proceedings.mlr.press/v119/katharopoulos20a.html).

Unnormalized dot-product attention replaces the softmax weights with raw dot products:

$$
\begin{aligned}
\mathbf{o}^{\mathsf{T}}_t &= \sum^{t}_{i=1}{\left(\mathbf{q}_t^{\mathsf{T}}\mathbf{k}_i\right)\mathbf{v}_i^{\mathsf{T}}}
\end{aligned}
$$

Removing softmax allows us to reassociate the products and factor the query outside the sum, using associativity and distributivity. With keys and values fixed, the resulting output is linear in the query:

$$
\begin{aligned}
\mathbf{o}^{\mathsf{T}}_t &= \sum^{t}_{i=1}{\mathbf{q}_t^{\mathsf{T}}\left(\mathbf{k}_i\mathbf{v}_i^{\mathsf{T}}\right)} \\
&= \mathbf{q}_t^{\mathsf{T}}\sum^{t}_{i=1}{\mathbf{k}_i\mathbf{v}_i^{\mathsf{T}}}
\end{aligned}
$$


Define the state matrix $\mathbf{S}_t\in\mathbb{R}^{D_k\times D_v}$:

$$
\begin{aligned}
\mathbf{S}_t &= \sum_{i=1}^{t}\mathbf{k}_i\mathbf{v}_i^{\mathsf T},\\
\mathbf{S}_t &= \mathbf{S}_{t-1}+\mathbf{k}_t\mathbf{v}_t^{\mathsf T},
\qquad \mathbf{S}_0=\mathbf{0},\\
\mathbf{o}_t &= \mathbf{S}_t^{\mathsf T}\mathbf{q}_t.
\end{aligned}
$$

Applying a feature map to queries and keys gives an unnormalized recurrent implementation. An optional initial state summarizes earlier tokens using the same feature map. Inputs have a nonempty sequence and matching devices and dtypes.

```py
def feature_map(x: Tensor) -> Tensor:
    return t.nn.functional.elu(x) + 1

def linear_attn_recurrent(
    q: Float[Tensor, "B T H Dk"],
    k: Float[Tensor, "B T H Dk"],
    v: Float[Tensor, "B T H Dv"],
    initial_state: Float[Tensor, "B H Dk Dv"] | None = None,
) -> tuple[Float[Tensor, "B T H Dv"], Float[Tensor, "B H Dk Dv"]]:
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]

    phi_q = feature_map(q)
    phi_k = feature_map(k)

    S = (
        initial_state
        if initial_state is not None
        else t.zeros([B, H, Dk, Dv], dtype=q.dtype, device=q.device)
    )

    outs = []
    for tok in range(T):
        S = S + einops.einsum(phi_k[:, tok], v[:, tok], "B H Dk, B H Dv -> B H Dk Dv")
        o_t = einops.einsum(phi_q[:, tok], S, "B H Dk, B H Dk Dv -> B H Dv")
        outs.append(o_t)

    o = t.stack(outs, dim=1)

    return o, S
```

The original formulation in the paper uses a feature map $$\phi(x) = \text{ELU}(x) + 1$$ on $\mathbf{q}$ and $\mathbf{k}$ and a normalizer state $\mathbf{z}_t$. The linear attention equation in the paper is given by: 

$$
\begin{aligned}
\mathbf{S}_t &= \sum^t_{i=1}{\phi(\mathbf{k}_i)\mathbf{v}_i^{\mathsf T}}, \\
\mathbf{z}_t &= \sum^t_{i=1}{\phi(\mathbf{k}_i)}, \\
\mathbf{o}_t &= \frac{\mathbf{S}_t^{\mathsf T}\phi(\mathbf{q}_t)}{\phi(\mathbf{q}_t)^\mathsf{T}\mathbf{z}_t} \\
\end{aligned}
$$

The positive feature map gives positive similarity scores, and the denominator makes attention weights sum to one. A normalized streaming implementation must carry both $\mathbf{S}_t$ and $\mathbf{z}_t$.

This formulation allows an algorithm with a linear time complexity of $$\mathcal{O}(Td^2)$$.

Linear attention is generally wonderful. Aside from having to perform fewer operations (given sufficient sequence length), it also does not require a growing [key-value (KV) cache](https://huggingface.co/blog/not-lain/kv-caching), instead summarizes past keys and values in a fixed-size state $$\mathbf{S}_t$$. However, naturally, there are downsides to this. With softmax attention, each previous key and value remains available in the KV cache. A new query scores every past key individually and combines the corresponding values. With linear attention, the keys and values are merged into the state matrix. New queries can only read that condensed summary. 

In addition to that, a recurrent implementation of linear attention processes tokens sequentially. This is generally not great if you spent a lot on a GPU (you might want to exploit more parallelism to make full use of them). Therefore, efforts have been made to make it more GPU friendly.

### Parallel Implementation

Thus, we require a parallel implementation of linear attention.

```py
def linear_attn_parallel(
    q: Float[Tensor, "B T H Dk"],
    k: Float[Tensor, "B T H Dk"],
    v: Float[Tensor, "B T H Dv"],
    initial_state: Float[Tensor, "B H Dk Dv"] | None = None,
) -> tuple[Float[Tensor, "B T H Dv"], Float[Tensor, "B H Dk Dv"]]:
    T = q.shape[1]

    phi_q = feature_map(q)
    phi_k = feature_map(k)

    scores = einops.einsum(
        phi_q, phi_k, "B i H Dk, B j H Dk -> B H i j"
    )
    mask = t.ones((T, T), dtype=t.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, 0)

    o = einops.einsum(
        scores, v, "B H i j, B j H Dv -> B i H Dv"
    )
    S = einops.einsum(
        phi_k, v, "B T H Dk, B T H Dv -> B H Dk Dv"
    )

    if initial_state is not None:
        o = o + einops.einsum(
            phi_q, initial_state,
            "B T H Dk, B H Dk Dv -> B T H Dv",
        )
        S = S + initial_state

    return o, S
```

The implementation above seems awfully similar to a traditional softmax attention. It computes all query-key scores at once, applies the causal mask, and uses the scores to combine the values. This is easy to parallelize, but we are back to constructing a $$T \times T$$ score matrix. It takes $$\mathcal{O}(T^2d + Td^2)$$ work and $$\mathcal{O}(T^2 + d^2)$$ intermediate memory per batch example and head. For $$T \ge d$$, this simplifies to $$\mathcal{O}(T^2d)$$ work and $$\mathcal{O}(T^2)$$ intermediate memory.

### Chunkwise Linear Attention

We now have a sequential linear algorithm and a parallel quadratic algorithm. A fair compromise is to perform some parts in parallel and some parts sequentially. Chunkwise linear attention is a mechanism that divides the sequence into smaller chunks, computes attention within each chunk in parallel, and passes a state between chunks. The basic implementation handles chunks sequentially; more advanced implementations can also parallelize the state propagation.

Formally, let chunk $c$ contain query, key, and value matrices denoted as $$\mathbf{Q}_c$$, $$\mathbf{K}_c$$, and $$\mathbf{V}_c$$ respectively. Here $\mathbf{S}_{c-1}$ is the state after the previous chunk. Within each chunk, a binary causal mask excludes future positions:

$$
A_{ij}=\begin{cases}1,&j\le i,\\0,&j>i.\end{cases}
$$

The unnormalized output and state update are:

$$
\begin{aligned}
\mathbf{O}_c
&=\underbrace{\phi({\mathbf{Q}}_c)\mathbf{S}_{c-1}}_{\text{inter-chunk}}
+\underbrace{\left((\phi({\mathbf{Q}}_c)\phi({\mathbf{K}}_c)^{\mathsf T})\odot\mathbf{A}\right)\mathbf{V}_c}_{\text{intra-chunk}},\\
\mathbf{S}_c&=\mathbf{S}_{c-1}+\phi({\mathbf{K}}_c)^{\mathsf T}\mathbf{V}_c.
\end{aligned}
$$

The chunk output combines the incoming state with causal attention within the chunk:

```py
def linear_attn_chunkwise(
    q: Float[Tensor, "B T H Dk"],
    k: Float[Tensor, "B T H Dk"],
    v: Float[Tensor, "B T H Dv"],
    chunk_size: int = 64,
    initial_state: Float[Tensor, "B H Dk Dv"] | None = None,
) -> tuple[Float[Tensor, "B T H Dv"], Float[Tensor, "B H Dk Dv"]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]

    S = (
        initial_state
        if initial_state is not None
        else t.zeros([B, H, Dk, Dv], dtype=q.dtype, device=q.device)
    )

    outs = []
    for start in range(0, T, chunk_size):
        end = min(T, start + chunk_size)
        
        phi_q = feature_map(q[:, start:end])
        phi_k = feature_map(k[:, start:end])
        
        vc = v[:, start:end]
        inter = einops.einsum(
            phi_q, S, "B i H Dk, B H Dk Dv -> B i H Dv"
        )
        scores = einops.einsum(
            phi_q, phi_k, "B i H Dk, B j H Dk -> B H i j"
        )
        mask = t.ones(
            (end - start, end - start), dtype=t.bool, device=q.device
        ).tril()
        scores = scores.masked_fill(~mask, 0)
        intra = einops.einsum(
            scores, vc, "B H i j, B j H Dv -> B i H Dv"
        )
        outs.append(inter + intra)
        S = S + einops.einsum(
            phi_k, vc, "B j H Dk, B j H Dv -> B H Dk Dv"
        )

    o = t.cat(outs, dim=1)

    return o, S
```

For sequence length $T$, chunk size $C$, and head dimension $d$, the matrix-based chunkwise formulation takes $$\mathcal{O}(TCd+Td^2)$$.

Intra-chunk attention costs $\mathcal{O}(TCd)$ across all chunks, while reading and updating the state costs $\mathcal{O}(Td^2)$. These work bounds omit the multiplicative batch and head factors $BH$. For fixed $C$ and $d$, the work is linear in $T$.

## Summary
* Linear attention is a variation of vanilla attention without softmax.
