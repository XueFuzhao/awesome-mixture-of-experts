# awesome-mixture-of-experts [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

A collection of AWESOME things about mixture-of-experts

This repo is a collection of AWESOME things about mixture-of-experts, including papers, code, etc. Feel free to star and fork.

# Contents
- [awesome-mixture-of-experts](#awesome-mixture-of-experts)
- [Contents](#contents)
- [Papers](#papers)
  - [MoE Model](#moe-model)
  - [MoE System](#moe-system)
- [Library](#library)

# Papers
## MoE Model
**Publication**
- Taming Sparsely Activated Transformer with Stochastic Experts [[ICLR 2022]](https://arxiv.org/abs/2110.04260)
- Go Wider Instead of Deeper [[AAAI2022]](https://arxiv.org/abs/2107.11817)
- Hash layers for large sparse models [[NeurIPS2021]](https://arxiv.org/abs/2106.04426)
- DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning [[NeurIPS2021]](https://arxiv.org/abs/2106.03760)
- Scaling Vision with Sparse Mixture of Experts [[NeurIPS2021]](https://proceedings.neurips.cc/paper/2021/hash/48237d9f2dea8c74c2a72126cf63d933-Abstract.html)
- BASE Layers: Simplifying Training of Large, Sparse Models [[ICML2021]](https://arxiv.org/abs/2103.16716)
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer [[ICLR2017]](https://openreview.net/forum?id=B1ckMDqlg)
- CPM-2: Large-scale cost-effective pre-trained language models [[AI Open]](https://www.sciencedirect.com/science/article/pii/S2666651021000310)
- Mixture of experts: a literature survey [[Artificial Intelligence Review]](https://link.springer.com/article/10.1007/s10462-012-9338-y)


**Arxiv**
- Mixture-of-experts VAEs can disregard variation in surjective multimodal data [[11 Apr 2022]](https://arxiv.org/abs/2204.05229)
- Efficient Language Modeling with Sparse all-MLP [[14 Mar 2022]](https://arxiv.org/abs/2203.06850)
- Parameter-Efficient Mixture-of-Experts Architecture for Pre-trained Language Models [[2 Mar 2022]](https://arxiv.org/abs/2203.01104)
- Mixture-of-Experts with Expert Choice Routing [[18 Feb 2022]](https://arxiv.org/abs/2101.03961)
- Designing Effective Sparse Expert Models [[17 Feb 2022]](https://arxiv.org/abs/2202.08906)
- Unified Scaling Laws for Routed Language Models [[2 Feb 2022]](https://arxiv.org/abs/2202.01169)
- Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model [[28 Jan 2022]](https://arxiv.org/abs/2201.11990)
- One Student Knows All Experts Know: From Sparse to Dense [[26 Jan 2022]](https://arxiv.org/abs/2201.10890)
- Dense-to-Sparse Gate for Mixture-of-Experts [[29 Dec 2021]](https://arxiv.org/abs/2112.14397)
- Efficient Large Scale Language Modeling with Mixtures of Experts [[20 Dec 2021]](https://arxiv.org/abs/2112.10684)
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts [[13 Dec 2021]](https://arxiv.org/abs/2112.06905)
- Building a great multi-lingual teacher with sparsely-gated mixture of experts for speech recognition [[10 Dec 2021]](https://arxiv.org/abs/2112.05820)
- SpeechMoE2: Mixture-of-Experts Model with Improved Routing [[23 Nov 2021]](https://arxiv.org/abs/2111.11831)
- VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts [[23 Nov 2021]](https://arxiv.org/abs/2111.02358)
- Towards More Effective and Economic Sparsely-Activated Model [[14 Oct 2021]](https://arxiv.org/abs/2110.07431)
- M6-10T: A Sharing-Delinking Paradigm for Efficient Multi-Trillion Parameter Pretraining [[8 Oct 2021]](https://arxiv.org/abs/2110.03888)
- Sparse MoEs meet Efficient Ensembles [[7 Oct 2021]](https://arxiv.org/abs/2110.03360)
- MoEfication: Conditional Computation of Transformer Models for Efficient Inference [[5 Oct 2021]](https://arxiv.org/abs/2110.01786)
- Cross-token Modeling with Conditional Computation [[5 Sep 2021]](https://arxiv.org/abs/2109.02008)
- M6-T: Exploring Sparse Expert Models and Beyond [[31 May 2021]](https://arxiv.org/abs/2105.15082)
- SpeechMoE: Scaling to Large Acoustic Models with Dynamic Routing Mixture of Experts [[7 May 2021]](https://arxiv.org/abs/2105.03036)
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity [[11 Jan 2021]](https://arxiv.org/abs/2101.03961)
- Exploring Routing Strategies for Multilingual Mixture-of-Experts Models [[28 Sept 2020]](https://openreview.net/forum?id=ey1XXNzcIZS)


## MoE System

**Publication**
- Pathways: Asynchronous Distributed Dataflow for ML [[MLSys2022]](https://arxiv.org/abs/2203.12533)
- Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning [[OSDI2022]](https://arxiv.org/abs/2201.12023)
- FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models[[PPoPP2022]](https://dl.acm.org/doi/abs/10.1145/3503221.3508418)
- BaGuaLu: Targeting Brain Scale Pretrained Models with over 37 Million Cores [[PPoPP2022]](http://keg.cs.tsinghua.edu.cn/jietang/publications/PPOPP22-Ma%20et%20al.-BaGuaLu%20Targeting%20Brain%20Scale%20Pretrained%20Models%20w.pdf)
- GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding [[ICLR2021]](https://openreview.net/forum?id=qrwe7XHTmYb)


**Arxiv**
- HetuMoE: An Efficient Trillion-scale Mixture-of-Expert Distributed Training System [[28 Mar 2022]](https://arxiv.org/abs/2203.14685)
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale [[14 Jan 2022]](https://arxiv.org/abs/2201.05596)
- SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient [[29 Sep 2021]](https://openreview.net/forum?id=U1edbV4kNu_)
- FastMoE: A Fast Mixture-of-Expert Training System [[24 Mar 2021]](https://arxiv.org/abs/2103.13262)


# Library
- [Tutel: An efficient mixture-of-experts implementation for large DNN model training](https://github.com/microsoft/tutel)
- [Mesh-TensorFlow: Deep Learning for Supercomputers](https://github.com/tensorflow/mesh)
- [FastMoE: A Fast Mixture-of-Expert Training System](https://github.com/laekov/fastmoe)
- [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://github.com/microsoft/DeepSpeed)
