---
layout : post
title: Paper Review - "Adaptive-RAG ; Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity", Soyeong Jeong et al., Mar 2024
date : 2024-03-28 14:41:00 +09:00
categories : [NLP(Natural Language Process), Data-Centric]
tags : [논문 리뷰, Paper Review, 연구, Research, NLP, Natural Language Process, AI, Artificial Intelligence, Data Science, RAG, Noise, Document, IR, LLM, Retrieval-Augmented Generation, Information Retrieval, MDQA, Multi document QA, NQ dataset, PDFImage-Searcher,  Markdown Formatter(MF), Hierarchical Contextual Augmentor(HCA),  Multi-Route Retriever(MRR), Data augmentation, soft partitioning]
lastmod : 2024-03-28 14:41:00 +09:00
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

Conference : NAACL, Mar 2024

[논문 Link](https://arxiv.org/abs/2403.14403)

---

# **1. Contributions**

RAG System은 Question-Answering(QA) task를 비롯한 다양한 task들에서 LLM의 성능을 높이는 데에 기여를 해왔습니다. 그러나, 사용자들이 실제로 쓰는 query들의 complexity는 다양합니다. 때로는, 간단한 query에 대해서 Multi-Step Approach를 하면서 불필요한 계산을 할 때도 있고, 복잡한 query에 대해서는 Single-Step Approach를 하며 적절하게 다루지 못할 때도 있습니다. 이러한 문제를 위해서 본 논문에서는 query의 complexity를 LLM을 활용해 미리 분류하고 활용하는 방안으로 Adaptive-RAG를 제안했습니다. 논문의 핵심 기여는 다음과 같습니다.

1-1.  실제 사용자들이 쓰는 현실적인 query들은 complexity가 매우 다양하다는 점을 지적했습니다. 

1-2.  LLM을 활용한 Classifier를 통해 query의 complexity를 평가하고, 각각의 complexity에 대해서 더 적절한 접근 방법들을 제안했습니다.

1-3. 특히, Adaptive-RAG를 통해서 complexity와 simplicity 사이의 균형을 맞추어 다양한 query들에 대해서 효율적으로 RAG가 작동할 수 있도록 했습니다. 

‘When did the people who captured Malakoff come to the region where Philipsburg is located?’와 같은 query는 ‘The region where Philipsburg is located’, ‘The people who captured Malakoff’, ‘When did the people went the region’ 과 같은 여러 reasoning을 거치며 답변을 생성할 수 있어야합니다. 이러한 경우에는 Document에 여러 차례 접근하여 정보들을 연결하는 Multi-hop QA 방식으로 answer를 생성하는 것이 적절합니다. 논문에서는 query의 complexity에 따라서 1) document에 접근하지 않고  답변을 생성 2) Single-hop QA 3) Multi-hop QA 을 수행할 수 있도록 했습니다.

---

# **2. Backgrounds**

### **2-1. Open-domain QA(ODQA)**

Open-domain Question Answering (ODQA)는 주제나 도메인에 제한 없이 다양한 질문에 답변할 수 있는 system입니다. 이러한 system은 방대한 정보 source에서 정확한 답변을 찾아내야 하기 때문에, NLP, IR, ML 같은 여러 기술을 종합적으로 활용합니다. ODQA의 주요 도전 과제 중 하나는 정확도와 효율성을 균형 있게 유지하면서 다양한 형태의 질문에 대해 정확한 정보를 신속하게 제공하는 것입니다.

### **2-2. Single-hop QA vs Multi-hop QA**

- **2-2-1. Single-hop QA**

Single-hop QA는 질문에 답하기 위해 단일 정보 source나 문서에서 직접적으로 답을 찾을 수 있는 Question-Answering system을 말합니다. 이러한 system은 주로 간단하거나 구체적인 정보를 요구하는 질문에 효과적입니다. 예를 들어, "한국의 수도는 어디인가요?" 같은 질문은 직접적으로 답변할 수 있으며, 복잡한 추론이나 여러 정보 source의 결합이 필요하지 않습니다. 

- **2-2-2. Multi-hop QA**

Multi-hop QA는 여러 단계의 추론을 거쳐야 하거나 다수의 정보 소스를 종합해야만 답변할 수 있는 복잡한 질문에 초점을 맞춘 system입니다. 이러한 시스템은 하나의 문서나 정보 소스만으로는 충분한 정보를 얻을 수 없는 경우 사용됩니다. 예를 들어, "바이든이 졸업한 대학의 설립자는 누구인가요?"와 같은 질문에 답하기 위해서는 먼저 "바이든이 졸업한 대학"을 찾고, 그 다음에 해당 대학의 "설립자"에 대한 정보를 찾는 여러 단계를 거쳐야 합니다. 이 과정에서 여러 문서를 통합하고, 정보 간의 연결을 추론하는 과정이 필요합니다.

---

# **3. Methodology**

<div style="text-align:center;">
  <img src="/assets/img/10/1.png" alt="Image 1" style="width:80%; margin:auto; display:block;" />
</div>