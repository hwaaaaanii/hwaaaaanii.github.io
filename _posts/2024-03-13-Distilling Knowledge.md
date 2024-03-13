---
layout : post
title: Paper Review - "Distilling Knowledge From Reader To Retriever For Question Answering", Gautier Izacard et al., 4 Aug 2022
date : 2024-03-13 14:41:00 +09:00
categories : [NLP(Natural Language Process), Data-Centric]
tags : [논문 리뷰, Paper Review, 연구, Research, NLP, Natural Language Process, AI, Artificial Intelligence, Data Science, Data-centric, RAG, ICLR, Retrival Augmented Generation, Gautier Izacard, FiD, Fusion-in-Decoder model, cross-attention, self-attention, KL divergence, SpaCy, max-margin loss]
lastmod : 2024-03-13 14:41:00 +09:00
sitemap :
  changefreq : daily
  priority : 1.0
---

<!-- MathJax Script for this post only -->
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ['\\(','\\)'] ],
      displayMath: [ ['$$','$$'], ['\\[','\\]'] ],
      processEscapes: true
    }
  });
</script>


---

Conference: ICLR (International Conference on Learning Representations), 2021

[논문 Link](https://arxiv.org/abs/2012.04584)

---

## **1. Contributions**

RAG System(Retrieval-Augmented Generation)은 external data base에서 문서 검색을 통해 얻은 정보를 기반으로 answer generation을 할 수 있게 해주는 system입니다. RAG System을 학습시키기 위해서는 query와 document의 쌍에 대한 data가 필요합니다. 그러나 Data를 얻는 것은 challenge가 되는 부분입니다. 또한, 어떤 방법론에서는 정답을 포함한 document는 positive하도록 분류하여 학습하는 방식을 택합니다. 이러한 방식은 “Where was Ada Lovelace born?”이라는 질문의 정답이 “London”일 때, “Ada Lovelace died in 1852 in Londen”을 포함하는 document도 positive로 분류될 수 있다는 문제점이 있습니다. 이 외에도 supervised learning은 여러 한계들이 있습니다.

본 논문에서는 unsupervised learning을 통해서 query, document pair가 없이도 RAG System을 학습할 수 있는 방법론을 제안했습니다. 본 논문의 핵심 기여는 다음과 같습니다.

1-1. Knowledge distillation에서 영감을 받아 reader(teacher model)와 retriever(student model)을 통해 unsupervised learning을 했습니다.

1-2. Reader model(Seq2Seq model)이 retriever를 지도하는 데에 있어서, 적절한 document를 찾아내게 하기 위해서 attention score를 활용했습니다.

1-3. 본 논문에서의 방법론을 통해서 SOTA를 달성했습니다.

본 논문이 Attention Score를 어떻게 활용했는지, Knowledge distillation에서의 방법론을 어떻게 활용했는지 그리고 이러한 방법론이 왜 중요한지에 대해서 다뤄보도록 하겠습니다.

---

## **2. Backgrounds**

### **2-1. FiD(Fusion-in-Decoder model)**

[Reference : “Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering”, Gautier Izacard et al., 2020](https://arxiv.org/abs/2007.01282)

FiD 모델은 Transformer의 attention mechanism을 decoder에서 활용하여, 여러 document로부터 정보를 통합하고 처리합니다. 이 과정에서 주요한 두 가지 attention 작업이 수행됩니다.

**2-1-1. Cross-attention**

FiD의 decoder는 검색된 여러 document의 정보를 종합하기 위해 cross-attention 메커니즘을 사용합니다. 여기서 각 document는 Value와 Key로 변환되며, 질문은 Query로 변환됩니다. decoder는 질문 query와 각 document key 사이의 관계를 분석하여 문서의 value에 가중치를 부여합니다. 이를 통해 model은 질문에 **가장 관련성이 높은 정보를 식별**하고, 이 정보를 기반으로 답변을 생성합니다.

**2-1-2. Self-attention** 

Decoder 내부에서도 self-attention mechanism이 작동하여, 생성된 답변의 각 단어(또는 token) 사이의 관계를 modeling합니다. 이 과정은 문맥적으로 **일관되고 자연스러운 답변**을 생성하는 데 기여합니다.

<div style="text-align:center;">
  <img src="/assets/img/8/1.png" alt="Image 1" style="width:100%; margin:auto; display:block;" />
</div>


위의 attention mechanism을 활용하면서, 기존의 encoder를 통과한 document를 decoder에 하나씩 개별적으로 넣어주는 방식을 하나의 sequence로 concat하는 방식으로 바꾸었습니다. Concat을 하면서 document들 끼리는 분류해주기 위해서 ‘question:’, ‘title:’, ‘context:’와 같은 special tokens를 concatenated sequence에 넣어주었습니다. 이러한 방식으로 FiD model은 당시에 SOTA를 달성한 model이라고 합니다.

### **2-2. KL(Kullback-Leibler) Divergence**

KL Divergence는 $P(x)$와 $Q(x)$분포가 있을 때, 두 분포가 얼마나 차이가 나는지를 측정해주는 지표라고 생각할 수 있습니다. $P(x)$와 $Q(x)$가 discrete probability distribution이라고 가정할 때, 수식은 다음과 같습니다.

$$
D_{\text{KL}}(P||Q) = \sum_{x \in X} P(x) \log(\frac{P(x)}{Q(x)})
$$

수식을 볼 때, $P(x) \approx Q(x)$일 때, $\frac{P(x)}{Q(x)} \approx 1$이 되어, $D_{\text{KL}}(P\|\|Q) \approx 0$이 되는 것을 알 수 있습니다. 이는 두 distribution이 유사할 때, $D_{\text{KL}}(P\|\|Q)$이 작다는 것을 의미합니다.

KL Divergence는 entropy와 cross entropy의 차이로도 표현할 수 있습니다.

$$
\sum_{x \in X} P(x) \log P(x) - \sum_{x \in X} P(x) \log Q(x) = H(P) - H(P,Q)
$$

또한, ELBO(Evidence Lower BOund)를 계산할 때에도 쓰이기도 하고 Loss function에 포함시켜주면 두 분포를 유사하게끔 학습시킬 수 있는 만큼, KL Divergence는 Machine Learning에서 중요하다고 볼 수 있습니다.

### **2-3. SpaCy**

SpaCy는 NLP를 위한 Open source library로, Python으로 구현되어 있습니다. 다양한 NLP를 위한 강력하고 효율적인 기능을 제공하며, 특히 대규모 정보 추출 작업에 유용합니다. SpaCy는 tokenizing, 품사 tagging, syntactic analysis, NER(Named entity recognition), dependency parsing 등 다양한 NLP 작업을 지원합니다. 특히, SpaCy는 학습된 모델을 사용하여 빠르게 결과를 얻을 수 있으며, 사용자가 자신의 data set으로 모델을 추가로 학습시키거나 fine-tuning하는 것도 가능합니다. 이를 통해 더 나은 성능과 정확도를 얻을 수 있습니다. SpaCy는 특히 document classification, sementic analysis, named entity recognition 등의 작업에 자주 사용됩니다.

---

## **3. Methodology**

본 논문에서 소개한 전반적인 방법론은 attention score가 usefulness of passage의 좋은 척도라고 가정하며, 다음 두 개의 module을 활용하는 것입니다. 

- Reader(teacher model) - 앞서 소개한 FiD model을 활용하여 Attention score를 계산하고, 이를 토대로 Retriever(student model)을 학습합니다.
- Retriever(student model) - Reader로부터 제공된 attention score를 따라할 수 있도록 학습하면서, attention score를 토대 가장 연관성이 높은 passage를 선별하는 것을 목표로 합니다.

두 module이 구체적으로 어떻게 작동하는지 살펴보겠습니다. 

### **3-1. Cross-Attention Mechanism**

FiD에서 decoder부분의 cross attention에 대해서 수식으로 살펴보겠습니다. 

$$
Q = W_Q H ,\quad K = W_KX, \quad V = W_V X
$$

$H \in R^d$ denotes the output from the previous self-attention.

$Q$ : Query, $K$ : Key, $V$ : Value

$W_Q, W_K, W_V$ are the weight matrices of queries, keys, and values, respectively.

위의 수식을 통해서 $Q, K, V$를 먼저 선형 변환 해준 다음에 아래의 수식으로 similarity를 계산해줍니다. 

$$
\alpha_{i,j} = Q_i^TK_j, \quad \tilde{\alpha}_{i,j} = \frac{\exp(\alpha_{i,j})}{\sum_m \exp(\alpha_{i,m})}
$$

마지막으로 Value에 앞서 구한 weight들로 가중합을 해준 다음에 다시 한 번 선형변환을 거치게 됩니다.

$$
O_i = W_O \sum_{j} \tilde{\alpha}_{i,j}V_{i,j}
$$

위와 같이 선형변환을 해주는 이유는 같은 공간에 mapping을 해줌으로 차원을 맞출 수 있게 되며, learnable parameter를 제공해주는 등의 이유가 있습니다. 

### **3-2. Dense Bi-Encoder for Passage Retriever**

본 논문에서는 Cross-attention score를 통해서 문서를 ranking하고자 합니다. 그러나, 모든 document와 query를 동시에 처리하는 것은 비효율적이라는 문제가 있습니다.. 논문에서는 이러한 문제를 해결하기 위해서, d차원 vector로 embedding하는 함수 $E$를 retriever와 함께 사용했습니다. $E$는 BERT가 사용했으며, DPR(Dense Passage Retrieval)과는 달리 Query인 $q$와 passgae인 $p$에 대해서 같은 $E$를 적용하여, $E(q)$와 $E(p)$를 얻을 수 있게 됩니다. 결과적으로 다음과 같이 similarity를 계산할 수 있게 됩니다. 

$$
S_{\theta}(q,p) = E(q)^TE(p)/\sqrt{d}
$$

### **3-3 Distilling the Cross-Attention Score to a Bi-Encoder**

이제 어떠한 방식으로 retriever model과 reader model을 학습했는지 그리고 Loss에 따라서 어떤 차이들이 있는지 보겠습니다. 

- Option 1 (Empirical result에서 확인해보면 가장 잘 작동한 Loss function입니다)
    
    첫 번째로 논문에서는 normalized output인 $S_{\theta}(q,p)$와 normalized attention score인 $G_{q,p}$의 KL-divergence를 최소화하는 것을 제안했습니다. 
    

$$
L_{\text{KL}}(\theta, Q) = \sum_{q \in Q,\text{ } p \in D_{q}} \tilde{G}_{q,p} (\log{\tilde{G}}_{q,p} - \log \tilde{S}_\theta(q,p)) 
$$

$$
where, \quad \tilde{G}_{q,p} = \frac{\exp(G_{q,p})}{\sum_{p' \in D_q}\exp(G_{q, p'})}, \quad \tilde{S}_{\theta}(q,p) = \frac{\exp(S_{\theta}(q,p))}{\sum_{p' \in D_q}\exp(S_{\theta}(q,p'))}
$$

- Option 2
    
    Option 2에서는 MSE를 최소화하는 Loss를 적용했습니다.
    

$$
L_{\text{MSE}}(\theta, Q) = \sum_{q \in Q, \text{ } p \in D_q} (S_{\theta}(q,p) - G_{q,p})^2
$$

- Option 3
    
    Option 3에서는 max-margin loss를 이용해 model이 순위를 잘못 예측했을 때, penalty를 주는 방식을 적용했습니다.
    
    $$
    L_{\text{ranking}}(\theta, Q) =\sum_{q \in Q, \text{ } p_q, p_2 \in D_q} \max(0, \gamma - \text{sign}(G_{q,p_1} - G_{q,p_2})(S_{\theta}(q,p_1) - S_{\theta}(q,p_2)))
    $$
    
    위의 수식을 직관적으로 접근해보자면, $p_1$이 $p_2$보다 클 때 ( $i.e. \quad G_{q,p_1}>G_{q,p2}$),  $\max(0, \gamma - (S_{\theta}(q,p_1) - S_{\theta}(q,p_2)))$ 가 되어, $S_{\theta}(q,p_1) - S_{\theta}(q,p_2)$이 적어도 $\gamma$보다 크게 만들어줍니다.
    
    $$
    \text{consider} \quad \max(S_{\theta}(q,p_1) - S_{\theta}(q,p_2), \gamma)
    $$
    
    즉, 두 score 사이에 margin을 주는 방식으로 score가 충분히 벌어지도록 하여 순위 예측을 용이하게 해준다고 볼 수 있습니다.  
    

위의 방법론들을 통해 teacher model과 student model을 반복적으로 학습했습니다. Iterative learning에서 특히 중요한 점은 current retriever가 negative sampling도 한다는 것입니다.[Reference : “Relevance-guided Supervision for OpenQA with ColBERT”, Omar Khattab et al., 2021](https://arxiv.org/abs/2007.00814) 
negative sample을 next step에서 retriever를 학습하는 데에 쓰는 discriminative training을 함으로써 model의 능력을 높이고자 했습니다. 순서는 다음과 같습니다. 

**Iterative Training**

1. Train the reader $R$ using the set of support documents for each question $D_{q}^0$
2. Compute aggregated attention scores  $$ (G_{q,p})_{q \in Q, \text{ } p \in D_{q}^0}$$ with the reader $$R$$.
3. Train the retriever $E$ using the scores  $(G_{q,p})_{q \in Q, \text{ } p \in D{q}^0}$ 
4. Retrieve top-passages with the new trained retriever $E$

---

## **4.Empirical Results**

논문에서 제시한 방법론을 통해서 SOTA를 달성했다고 합니다. Table들을 통해서 확인해보겠습니다. 

<div style="text-align:center;">
  <img src="/assets/img/8/2.png" alt="Image 2" style="width:60%; margin:auto; display:block;" />
</div>


논문에서 주장한 대로 성능이 잘 나온 것을 확인할 수 있습니다. 특히 제가 위 테이블에서 눈이 가는 부분은 더 많은 document를 top-k로 선정했을 때, 성능이 높아진다는 점입니다(R@k means extracted top-k documents). ROUGE나 BLEU score도 잘 나온 것을 확인할 수 있습니다. 한 가지 의문인 것은 왜 Iter를 0과 1만 넣어뒀는지가 의문입니다.

앞서서 제시한 Loss function들 중에서 무엇이 가장 잘 작동했는지는 아래 table에 있습니다.

<div style="text-align:center;">
  <img src="/assets/img/8/3.png" alt="Image 3" style="width:60%; margin:auto; display:block;" />
</div>


MSE는 성능이 안 좋았고, Max-margin loss에 있어서는 $\gamma$값이 유의미한 차이를 주지 못했다는 점, 그리고 KL-divergence가 가장 성능이 좋았다는 점을 확인할 수 있습니다. 또한 R@k에서 k가 높을 수록 성능이 더 높아지는 부분도 눈에 들어옵니다. k가 높을 수록 computation cost가 높아지는 문제가 있긴 하겠지만, 성능이 좋아진다는 점은 기억에 남겨두고 싶은 부분입니다.

---

이상으로 포스팅을 마치겠습니다.  

감사합니다.