# VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos

Bao Tran Gia <sup>1,2</sup>, Khiem Le <sup>1,2</sup>, Tien Do <sup>1,2</sup>, Tien-Dung Mai <sup>1,2</sup>,  
Thanh Duc Ngo <sup>1,2</sup>, Duy-Dinh Le <sup>1,2</sup>, Shin'ichi Satoh <sup>3</sup>

<sup>1</sup> University of Information Technology, VNU-HCM, Vietnam

<sup>2</sup> Vietnam National University, Ho Chi Minh City, Vietnam

<sup>3</sup> National Institute of Informatics, Japan

22520121@gm.uit.edu.vn

{khiemltt, tiendv, dungmt, thanhnd, duyld}@uit.edu.vn

satoh@nii.ac.jp

## Abstract

*The rapid expansion of video data across various domains has heightened the demand for efficient retrieval and question-answering systems, particularly for long-form videos. Existing Video Question Answering (VQA) approaches struggle with processing extended video sequences due to high computational costs, loss of contextual coherence, and challenges in retrieving relevant information. To tackle these limitations, we introduce VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos, a novel framework that brings a retrieval-augmented generation (RAG) architecture to the video domain. VRAG first retrieves the most relevant video segments and then applies chunking and refinement to identify key sub-segments, enabling precise and focused answer generation. This approach maximizes the effectiveness of the Multimodal Large Language Model (MLLM) by ensuring only the most relevant content is processed. Our experimental evaluation on a benchmark demonstrates significant improvements in retrieval precision and answer quality. These results highlight the effectiveness of retrieval-augmented reasoning for scalable and accurate VQA in long-form video datasets.*

## 1. Introduction

The rapid growth of video data across digital platforms and surveillance systems has created an urgent need for accurate video retrieval from large datasets using natural language queries. As video archives expand, retrieving precise segments relevant to a given query becomes increasingly challenging, especially in long-form content with complex narratives. For example, in surveillance footage, quickly locating a specific event, such as an individual entering a

restricted area, is crucial for security response.

In computer vision and natural language processing, the retrieval of detailed information from videos is formalized as the VQA task on large-scale datasets [27]. However, this task presents several significant challenges. Unlike image-based VQA, video VQA requires understanding complex spatiotemporal dynamics, tracking objects and actions over time, and integrating multi-modal information, including visual, audio, and textual cues. The inherent variability in question types, ranging from factual queries to reasoning-based inference, further complicates the problem. Additionally, the vast length and redundancy in video data demand efficient attention mechanisms to identify and focus on the most relevant segments.

Current VQA approaches primarily rely on deep learning-based methods, including transformer architectures [19], multimodal fusion techniques [18], and memory-augmented networks [21]. Transformer models like VidL and MERLOT [32] leverage self-attention for joint video-text reasoning, while multimodal fusion integrates video, audio, and text through cross-modal attention. Memory-augmented methods enhance retrieval by storing relevant video segments for reasoning. However, these methods struggle with long-form videos due to challenges in maintaining contextual coherence over extended sequences. Transformer models face scalability issues as video length increases [32], multimodal fusion suffers from information bottlenecks, and memory-augmented networks often fail to select relevant frames efficiently. These limitations highlight the need for more scalable and context-aware approaches that can efficiently process long videos while preserving spatio-temporal relationships.

To address these challenges, recent research has focused on two primary tasks: Known-Item Search (KIS) [29] and Video Question Answering (VQA) [12]. KIS aims to en-

hance retrieval efficiency by developing scalable indexing and search models, while VQA seeks to improve answer accuracy through advanced multimodal representations and temporal reasoning. Advancements in these tasks contribute to the broader goal of improving video-based information retrieval, enabling more accurate and efficient content understanding.

In this paper, we introduce VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos, a solution that integrates retrieval and reasoning to enhance both accuracy and efficiency in video-based question answering. Inspired by retrieval-augmented generation (RAG) architectures used in chatbots, VRAG first retrieves the most relevant video segments, then applies a chunking-based refinement step to focus on key sub-segments before generating answers. This structured retrieval process ensures that only the most relevant content is processed, maximizing the effectiveness of large language models (LLMs) and vision-language models (VLMs). Unlike existing approaches that process entire videos end-to-end, VRAG first retrieves the most relevant segments, reducing computational overhead while preserving critical spatiotemporal information for reasoning. VRAG consists of three key modules: (1) Multi-modal Search, which leverages diverse retrieval techniques to locate relevant video segments; (2) Re-ranking, which refines retrieval results by prioritizing keyframe-based representations while ensuring temporal coherence; and (3) VQA, which includes a Filtering Module to pre-select relevant segments and an Answer Module to aggregate information and generate precise, context-aware responses. By combining these components, VRAG improves retrieval precision, enhances reasoning over sequential video content, and optimizes computational efficiency, making it a robust solution for VQA in large-scale, long-form video datasets.

Our experimental evaluation assesses both the retrieval and VQA components of VRAG on a large-scale dataset collected from the Video Browser Showdown (VBS) 2019-2025, providing a comprehensive analysis of its effectiveness. Retrieval performance is evaluated based on the system’s ability to accurately locate relevant video segments, while the VQA module is assessed using metrics such as answer accuracy, contextual relevance, and computational efficiency. Experimental results indicate that VRAG achieves a retrieval score of 40.5 out of 45 for the KIS task and a VQA performance of 4 out of 5 on queries derived from VBS, demonstrating significant improvements in retrieval precision and answer quality. These improvements are attributed to the effective integration of multi-modal search, re-ranking, and retrieval-augmented reasoning. Overall, the findings highlight the robustness of VRAG in handling long-form video content, positioning it as a promising solution for large-scale video understanding and question an-

sowering.

## 2. Related Work

### 2.1. Video Retrieval

Video retrieval has gained significant attention in recent years due to the rapid growth of video content and the increasing demand for efficient retrieval systems

Traditional video retrieval systems predominantly relied on content-based video retrieval (CBVR) methods, which utilized handcrafted features and keyframe extraction. Videos were segmented into shots, with keyframes indexed using global descriptors such as color histograms and texture patterns or local descriptors. The Bag-of-Visual-Words (BoVW) [11] model, clustered local descriptors into a codebook to facilitate efficient retrieval. While effective, these approaches were limited by their reliance on manually designed features, which constrained their generalization across diverse video datasets.

The advent of deep learning transformed video retrieval by replacing handcrafted features with learned representations. Convolutional Neural Networks (CNNs) [23], pre-trained on large-scale datasets, improved feature extraction, while Long Short-Term Memory (LSTM) [14] networks enabled modeling of spatio-temporal relationships. Hybrid architectures such as the Long-Term Recurrent Convolutional Network (LRCN) [9] combined CNN-based spatial feature extraction with LSTM-based sequence modeling, improving retrieval accuracy but struggling with long video sequences due to computational constraints.

Transformer-based models further advanced video retrieval by employing self-attention mechanisms to capture long-range dependencies. Vision Transformers (ViTs) [10] inspired video-based adaptations such as TimeSformer [4] and ViViT [1], which process video frames as token sequences, achieving superior spatio-temporal modeling compared to CNNs and RNNs. However, these models require extensive training data and substantial computational resources to learn meaningful representations.

Contemporary video retrieval systems increasingly leverage multimodal approaches, incorporating visual, textual, and audio data for enhanced retrieval accuracy. Inspired by CLIP [24], video-language models utilize dual encoders to establish a joint embedding space, facilitating zero-shot retrieval and improved performance on standard benchmarks. The integration of audio features, including speech transcripts and environmental sounds, provides complementary cues that refine retrieval outcomes. Multimodal transformers enhance cross-modal alignment, enabling more robust query matching. As research progresses, the fusion of vision, language, and audio continues to advance video retrieval, making systems more adaptive to complex and diverse queries.

![Figure 1: Overview of the VRAG framework for retrieval-augmented video question answering. The diagram shows a flow from user queries (KIS or VQA) to a Retrieval Query and a Question. The Retrieval Query is processed by a Multimodal Retrieval System (containing Semantic-based Search, On-screen Text Search, Audio-based Search, Object Filtering, and Temporal Search) to produce Top-N Results. These results are then processed by a Re-ranking Module (containing a Multimodal LLM, Score Assessment, and Re-ranking) to produce Top-K Results. The Top-K Results are then processed by a VQA Module (containing Chunking, a Multimodal LLM, Relevant chunks, Concatenated Segment, and Answer) to produce the final Answer. The VQA Module also receives the Question and the Video as input.](b230b8f21d8e82d55c0d311c8c32ef73_img.jpg)

Figure 1: Overview of the VRAG framework for retrieval-augmented video question answering. The diagram shows a flow from user queries (KIS or VQA) to a Retrieval Query and a Question. The Retrieval Query is processed by a Multimodal Retrieval System (containing Semantic-based Search, On-screen Text Search, Audio-based Search, Object Filtering, and Temporal Search) to produce Top-N Results. These results are then processed by a Re-ranking Module (containing a Multimodal LLM, Score Assessment, and Re-ranking) to produce Top-K Results. The Top-K Results are then processed by a VQA Module (containing Chunking, a Multimodal LLM, Relevant chunks, Concatenated Segment, and Answer) to produce the final Answer. The VQA Module also receives the Question and the Video as input.

Figure 1. Overview of the VRAG framework for retrieval-augmented video question answering. The system processes user queries as either KIS or VQA. For VQA queries, GPT-4o generates both a retrieval query and a corresponding question. The retrieval module identifies the top-N candidate segments, which are then refined through a re-ranking mechanism to obtain the top-K most relevant segments. The selected segments are either used for KIS tasks or processed by the VQA module, where they undergo chunking, filtering, and reasoning within the Answering Module to generate an accurate, context-aware response.

### 2.2. Long-form Video Understanding

Long-form video understanding is a critical task in multi-media computing that involves analyzing and interpreting extended video sequences to extract meaningful semantic information. Unlike short-form video analysis, long-form video understanding requires capturing complex temporal dependencies, handling varying scene transitions, and integrating multimodal information over extended durations.

Early approaches to video understanding relied on hand-crafted feature representations such as Histograms of Oriented Gradients (HOG) and Scale-Invariant Feature Transform (SIFT) to describe individual frames. While these methods were effective for basic recognition tasks, they were limited in their ability to model long-range temporal dependencies and contextual relationships across video sequences.

The advent of deep learning has significantly advanced video analysis, particularly with the widespread adoption of CNNs for spatial feature extraction. To capture temporal dependencies, RNNs, including Long LSTM networks and Gated Recurrent Units (GRUs), have been extensively utilized. More recently, transformer-based architectures have further improved the modeling of long-range dependencies in video sequences, demonstrating superior performance in capturing global context and complex temporal interactions.

Self-supervised learning has emerged as a promising paradigm to reduce reliance on large-scale labeled datasets. Techniques such as contrastive learning, masked frame prediction, and temporal consistency learning enable models to learn meaningful video representations from unlabeled

data. Approaches such as MoCo [13], SimCLR [6], and VideoBERT [28] leverage self-supervised objectives to pre-train video models, improving generalization across diverse video analysis tasks.

Recent advancements in large language models (LLMs) have further transformed long-form video understanding by introducing in-context learning capabilities. Pretrained on extensive multimodal datasets, LLMs can generalize across multiple tasks using natural language prompts, reducing the need for task-specific fine-tuning. Ongoing research explores the integration of LLMs with vision models, such as InternVL2.5 [7], Qwen2.5-VL [2], and VideoLLaMA3 [33], to enhance multimodal video analysis, facilitating more robust and flexible long-form video interpretation.

### 2.3. Retrieval-Augmented Generation (RAG)

#### 2.3.1. RAG in Natural Language Processing

RAG has emerged as a pivotal framework in natural language processing (NLP), enhancing LLMs by integrating non-parametric memory retrieval to supplement parametric knowledge. Dense retrieval techniques, such as Dense Passage Retrieval (DPR) [15] and ColBERT [17], leverage neural embeddings to retrieve semantically relevant documents, demonstrating superior performance over traditional sparse retrieval methods like BM25. Hybrid retrieval approaches, which combine sparse lexical matching with dense retrieval, further improve recall and precision in knowledge-intensive tasks. Additionally, memory-based retrieval methods, including KNN-LM [16] and RETRO [5], employ external knowledge sources to enhance contextual understanding

and reduce reliance on static parametric memory. These retrieval-augmented paradigms have significantly advanced NLP applications, including open-domain question answering, abstractive summarization, and multimodal learning, by providing models with dynamically retrieved, contextually relevant knowledge.

#### 2.3.2. Multimodal RAG

Beyond text-only applications, researchers have extended the RAG paradigm to vision and multimodal tasks, creating models that retrieve and incorporate external knowledge for image understanding. Multimodal RAG systems integrate image understanding, retrieval, and generation components to enhance vision-language tasks. Given an image and an optional query, the system extracts visual features or text representations (e.g., object tags, captions) to retrieve relevant information from external knowledge bases like Wikipedia. Early approaches relied on unimodal text retrievers (e.g., DPR) using detected object names for retrieval, overlooking richer visual context. Recent methods, such as [26], introduce joint multimodal retrievers that encode both image and text, leveraging techniques like inverse cloze task pre-training to improve retrieval alignment. These models significantly enhance recall, with one study reporting a 28% improvement in OK-VQA [22] compared to using a text-only retriever. This allows vision models to access external knowledge for identifying and describing unfamiliar objects, improving reasoning and generation capabilities.

## 3. Methodologies

The proposed VRAG framework, illustrated in Figure 2, integrates video retrieval and reasoning to enhance VQA performance. Unlike conventional end-to-end video processing methods, the VRAG framework first retrieves the most relevant segments, optimizing both accuracy and efficiency. For VQA queries, GPT-4o generates a retrieval query along with the corresponding question. The retrieval module, powered by a multimodal search system, identifies the top-N candidate segments based on diverse modalities, including semantic similarity, on-screen text, audio features, and object-based filtering. These results are then refined by a re-ranking module, which prioritizes the most relevant segments, selecting the top-K results. While KIS queries directly utilize these top-K results, VQA queries undergo additional processing through the VRAG-VQA module, where retrieved segments are chunked, filtered, and analyzed to generate accurate, context-aware responses.

### 3.1. Multi-modal Retrieval System

**Semantic-based Retrieval:** Our multimodal retrieval system employs a late-fusion approach at the shot level, integrating results from InternVL-G [7], BLIP-2 [20], BEiT-

3 [30], and CLIP [8]. This strategy leverages the complementary semantic representations of vision-language models, enhancing retrieval accuracy through improved cross-modal alignment.

**On-Screen Text Retrieval:** To enhance text-based video retrieval, we employ optical character recognition (OCR) to extract textual content from video frames. Specifically, DeepSolo [31] is utilized for text detection, while PARSeq [3] is employed for text recognition. This approach enables efficient retrieval based on on-screen text, facilitating precise content searches within video data.

**Audio-based Retrieval:** Our system enables video retrieval based on spoken content by utilizing Whisper [25] for automatic speech transcription. This allows for precise retrieval of video segments containing specific spoken words.

**Object Filtering:** As a post-processing step, this module refines retrieval results by filtering out shots that do not satisfy predefined object conditions. Object extraction is performed using Co-DETR [34] to ensure the retrieved segments align with the query constraints.

**Temporal Search:** This module enhances retrieval by incorporating temporal constraints, allowing users to locate relevant video segments based on temporal relationships between events.

### 3.2. Re-ranking Module

To improve the quality of retrieval results, we integrate a re-ranking module that refines the order of retrieved video segments. This module aims to enhance ranking precision by incorporating additional factors beyond the initial retrieval stage. While keyframe-based retrieval is efficient in identifying relevant segments, it tends to perform poorly on queries that require an understanding of temporal information. This limitation arises because keyframe-based methods primarily focus on visual similarity at discrete moments, thus discarding the temporal relationships between frames.

The re-ranking process begins with the retrieval of relevant shots from a multi-modal retrieval system. To incorporate temporal continuity, the re-ranking module expands each potentially relevant shot  $X$  by including three preceding and three succeeding shots, merging into a potentially meaningful short video segment. These expanded segments are then processed through a video understanding MLLM model, which evaluates their alignment with the query and assigns a relevance score on a scale from 0 to 1. Once the relevance scores are computed, the video segments are

![Figure 2: Overview of the Re-ranking Module in the VRAG framework. The diagram shows a flow from a 'Retrieval Query' box to a 'Multimodal Retrieval System' box, which then points to a 'Shot T' box. A dashed arrow from 'Shot T' points to a vertical stack of six boxes labeled 'Shot T-3', 'Shot T-2', 'Shot T-1', 'Shot T', 'Shot T+1', and 'Shot T+3'. A dashed arrow from the 'Shot T' box in the stack points to a 'Merged Shot' box. A dashed arrow from the 'Merged Shot' box points to a 'Multimodal LLM' box (represented by a brain icon). A dashed arrow from the 'Multimodal LLM' box points to a 'Relevance score' box.](f9a14fbfecbd7d059226cc93677d721b_img.jpg)

Figure 2: Overview of the Re-ranking Module in the VRAG framework. The diagram shows a flow from a 'Retrieval Query' box to a 'Multimodal Retrieval System' box, which then points to a 'Shot T' box. A dashed arrow from 'Shot T' points to a vertical stack of six boxes labeled 'Shot T-3', 'Shot T-2', 'Shot T-1', 'Shot T', 'Shot T+1', and 'Shot T+3'. A dashed arrow from the 'Shot T' box in the stack points to a 'Merged Shot' box. A dashed arrow from the 'Merged Shot' box points to a 'Multimodal LLM' box (represented by a brain icon). A dashed arrow from the 'Multimodal LLM' box points to a 'Relevance score' box.

Figure 2. Overview of the Re-ranking Module in the VRAG framework. This module enhances retrieval precision by incorporating temporally adjacent shots to construct a short video segment. A multimodal video understanding model then processes the merged segment and assigns a relevance score, facilitating more accurate and context-aware ranking in retrieval-based video analysis.

ranked by their corresponding scores in descending order with the top-ranked ones selected as the final results. This approach ensures that the retrieval process considers both semantic and temporal coherence, thereby enhancing the accuracy and relevance of the results. The corresponding process is illustrated in Figure 2.

### 3.3. VQA Module

VQA on long videos poses significant challenges due to computational constraints and limitations in context retention. The substantial memory and processing power required for analyzing long videos makes direct processing impractical. Consequently, essential details necessary for generating accurate responses may be overlooked, reducing the system’s reliability. Moreover, the increased temporal complexity complicates the identification of relevant information, hindering the model’s ability to retrieve precise answers.

To overcome these challenges, a retrieval-augmented generation approach is adopted, integrating a chunking-based strategy with selective retrieval mechanisms to enhance efficiency and accuracy, as illustrated in Figure 3.

#### 3.3.1. Filtering Module

Long videos are segmented into smaller and manageable chunks, each undergoing a filtering process to assess their relevance to a given query. This process, guided by MLLM, involves making binary decisions that determine whether to retain relevant segments or discard non-relevant ones.

To preserve contextual continuity, the video is divided into fixed-length overlapping segments before evaluation. The Filtering Module determines whether each segment contains answer-relevant information, ensuring that only the most informative portions are selected for further analysis.

By selectively focusing on relevant segments, this approach optimizes computational efficiency and ensures that the generated answers are grounded in pertinent content. The retrieval process plays a crucial role in mitigating memory and performance limitations, while the subsequent answer generation phase enhances response accuracy through multi-modal reasoning.

#### 3.3.2. Answering Module

The segments identified as relevant are then aggregated to construct a coherent and contextually enriched input for answer generation. This process is orchestrated by the MLLM, which synthesizes information across multiple retrieved segments rather than treating them as independent units. By capturing contextual dependencies and integrating temporal cues, the model can generate a comprehensive and well-informed response. This approach significantly enhances the model’s ability to infer relationships between different events, objects, and actions, thereby improving the reliability and comprehensiveness of the generated response. By narrowing its focus to only the most relevant content, the answering module optimizes computational efficiency while preserving high fidelity in its predictions.

## 4. Experiments

### 4.1. Datasets

#### 4.1.1. Introduction

The dataset used for our experiments is the first shard of the Vimeo Creative Commons Collection (V3C), consisting of 7,475 Creative Commons-licensed videos obtained from Vimeo, with a combined runtime of approximately 1,000 hours. This dataset has been extensively used in prior benchmark challenges, such as TRECVID and VBS, making it a suitable choice for evaluation. The dataset contains a diverse range of content types, providing a robust foundation for evaluating video search and exploration methodologies.

#### 4.1.2. Video Pre-processing

For video pre-processing, we leveraged shot boundaries obtained from the master shot boundary detection. Within each shot, keyframe selection was performed based on semantic feature analysis. Specifically, we extracted semantic features using BEiT-3 and applied a threshold-based approach to determine the most representative keyframes. This process ensured that keyframes effectively captured the essential content of each shot while reducing redundancy, thereby facilitating efficient video retrieval and analysis. As a result, a total of 2,143,361 keyframes were selected for further processing.

#### 4.1.3. Queries

The challenge tasks encompassed KIS and VQA. For the experiments, queries were sourced from the VBS competitions held between 2019 and 2025. Only queries with established ground truth in the V3C1 dataset were considered, resulting in a selection of 45 KIS queries and 5 VQA queries. KIS queries consisted of textual descriptions specifying a particular video segment for retrieval, while VQA queries required identifying a relevant video and answering a content-based question.

Table 1. Example queries from KIS and VQA tasks derived from VBS 2019-2025.

| Type | ID | Video ID | Query / Question                                                                                                                                                                |
|------|----|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| KIS  | 1  | 4316     | A young man sitting outdoors and eating. He wears a dark T-shirt and silver headphones. It is evening, trees are visible behind him (out of focus).                             |
|      | 2  | 3589     | Close-up of hands playing the piano, then of hands using a tablet. The tablet shows musical score sheets. 'videoblocks' is superimposed over the shot with the tablet.          |
|      | 3  | 2637     | View out of a high building down to the street, snow is falling, views of other buildings. The camera zooms slowly down to the street. The views of other buildings get blurry. |
| VQA  | 1  | 2799     | In which US state is the paddlers' car registered?                                                                                                                              |
|      | 2  | 4318     | What color is the shirt of the man taking a photo?                                                                                                                              |
|      | 3  | 7220     | What is being carried on the bike of the man passing the newspaper stand?                                                                                                       |

### 4.2. Evaluation

Evaluation was conducted using task-specific scoring metrics. In the fully automated track, scores were computed as:

$$S = 1 - \frac{(r - 1)}{n}$$

where  $r$  is the rank of the correct answer and  $n$  is the number of possible answers per task, fixed at  $n = 10$ . Tasks without a correct submission were assigned a score of 0, and the total score was the sum over all tasks. A KIS result was considered correct if the retrieved video segment matched the ground truth video and the identified time interval was entirely contained within the reference interval. Partial overlaps were deemed incorrect. A VQA result was correct if the submitted textual response matched the expected answer in content, as determined by human assessors. The interactive track incorporated response time, and computing scores as:

$$S = \left(1 - \frac{t}{2d}\right) - \frac{(r - 1)}{n}$$

where  $t$  is the time until the first correct submission and  $d$  is the total task duration (5 minutes). Scores were displayed as scaled values for readability.

### 4.3. Experimental Results

This section report the effectiveness of our multi-modal retrieval system across the retrieval, re-ranking, and visual question answering modules.

#### 4.3.1. Multi-modal Retrieval System

Our experiments evaluated the performance of our multi-modal retrieval system on queries derived from VBS 2019-2025. While leveraging multiple modalities can enhance retrieval accuracy, we observed that incorporating too many modalities simultaneously may lead to suboptimal performance due to increased noise, redundancy, and inconsistencies in modality-specific predictions.

Table 2. Performance comparison of MLLMs in the re-ranking module, showing their impact on ranking scores compared to the baseline without re-ranking. The maximum achievable score is 45.0.

| Method         | Model           | Score       |
|----------------|-----------------|-------------|
| w/o re-ranking | -               | 33.1        |
| w/ re-ranking  | Qwen2.5-VL-7B   | 26.9        |
|                | VideoLLaMA3-7B  | 34.5        |
|                | InternVL2.5-8B  | 35.1        |
|                | InternVL2.5-78B | <b>40.5</b> |

In the interactive track, users could selectively refine retrieval results by interacting with different modalities,

![Figure 3: An illustration of the VQA module within the VRAG framework. The diagram shows a flow from 'Long Video' (via 'Chunking') and 'Question' to a 'Filtering Module' (two 'Multimodal LLM' blocks). The first LLM uses the video segments and question to select 'Segment 1' and 'Segment 3'. The second LLM uses these segments and a 'Short Segment' (selected from the first LLM's output) to generate an 'Answer'.](fc46871d72c65d3381d9201646d23439_img.jpg)

```

graph TD
    LV[Long Video] -- Chunking --> S1[Segment 1]
    LV -- Chunking --> S2[Segment 2]
    LV -- Chunking --> S3[Segment 3]
    LV -- Chunking --> S4[Segment 4]
    LV -- Chunking --> S5[Segment 5]
    Q[Question] --> F1[Multimodal LLM]
    F1 --> S1
    F1 --> S3
    S1 --> F2[Multimodal LLM]
    S3 --> F2
    F2 --> SS[Short Segment]
    SS --> A[Answer]
    Q --> A
    
```

Figure 3: An illustration of the VQA module within the VRAG framework. The diagram shows a flow from 'Long Video' (via 'Chunking') and 'Question' to a 'Filtering Module' (two 'Multimodal LLM' blocks). The first LLM uses the video segments and question to select 'Segment 1' and 'Segment 3'. The second LLM uses these segments and a 'Short Segment' (selected from the first LLM's output) to generate an 'Answer'.

Figure 3. An illustration of the VQA module within the VRAG framework. The system processes long-form videos by segmenting them and employing a **Filtering Module** to extract the most relevant content. The **Answering Module** integrates the selected segments to generate accurate and contextually informed responses, optimizing computational efficiency and reasoning over sequential video content.

which improved retrieval precision. In contrast, the fully-automated track required a streamlined approach to ensure robustness and efficiency without human intervention. Our findings indicate that state-of-the-art vision-language models, such as InternVL2, BLIP-2, and others, demonstrate strong cross-modal understanding capabilities, often mitigating the need for specialized approaches for individual modalities in automated retrieval scenarios.

We adopted a straightforward strategy for the fully-automated track where the query was directly processed using the semantic-based retrieval module for the KIS task.

Table 3. Comparison of the performance of different VQA approaches, including Naive and VRAG methods, across various models. The maximum achievable score is 5.

| Model          | Naive | VRAG     |
|----------------|-------|----------|
| VideoLLaMA3-7B | 2     | <b>4</b> |
| Qwen2.5-VL-7B  | 2     | 2        |
| InternVL2.5-8B | 2     | 2        |

#### 4.3.2. Re-ranking Module

To assess the effectiveness of the re-ranking module, we conducted a comparative analysis of retrieval performance before and after re-ranking. Context expansion was applied by incorporating neighboring shots for each of the top 100 retrieved results, followed by the use of MLLM to evaluate semantic relevance and refine the ranking. Several MLLMs were tested, including VideoLLaMA3, InternVL2.5, and Qwen2.5-VL, to examine their impact on retrieval performance. The results, presented in Table 2, illustrate the ex-

tent to which re-ranking enhances retrieval effectiveness.

The initial experiments with MLLMs in the parameter range of 7B–8B revealed that InternVL2.5-8B demonstrated superior performance compared to other models in effectively understanding and assigning relevance scores. Based on these findings, we scaled up to InternVL2.5-78B to further enhance performance.

Subsequent experimental results indicate that re-ranking significantly improves retrieval accuracy, with InternVL2.5 achieving the highest performance among the tested models. The baseline approach, which did not employ re-ranking, yielded a score of 33.1, whereas the best-performing model attained a score of 40.5 out of a possible 45.0. These results highlight the effectiveness of integrating MLLMs for ranking refinement.

#### 4.3.3. VQA Module

To evaluate the effectiveness of our VQA module, we conducted experiments comparing a naive approach – where the entire video is directly fed into an MLLM for answer generation – with our retrieval-based approach, which selectively extracts the most relevant video segments before processing the VQA query. We employed GPT-4o to decompose the VQA query into structured components. A retrieval query was used to identify the relevant video segment to determine the video ID and question used to generate an answer.

Following the retrieval and re-ranking steps, the video corresponding to each ranked result was used for VQA. Experiments were conducted with multiple vision-language models, including VideoLLaMA3, InternVL2.5, and Qwen2.5-VL, to evaluate their impact on answer generation accuracy.

Table 4. Details of the runs conducted for the fully automated track, including the approach used for each task (KIS and VQA), the specific VQA settings, and the corresponding experimental scores. The maximum experimental score for KIS is 45, while the maximum score for VQA is 5. MRS stands for Multimodal Retrieval System, and CS represents the chunk size (in seconds).

| Task | # Run | Approach                             | VQA Settings |                | Experimental Score |
|------|-------|--------------------------------------|--------------|----------------|--------------------|
|      |       |                                      | CS           | MLLM           |                    |
| KIS  | 1     | MRS + Re-ranking Module              | -            | -              | 40.5               |
|      | 2     | MRS                                  | -            | -              | 33.1               |
| VQA  | 1     | MRS + Re-ranking Module + VQA Module | 15           | VideoLLaMA3-7B | 4                  |
|      | 2     | MRS + Re-ranking Module + VQA Module | 30           | VideoLLaMA3-7B | 2                  |
|      | 3     | MRS + Re-ranking Module + Naive VQA  | -            | VideoLLaMA3-7B | 2                  |
|      | 4     | MRS + Re-ranking Module + VQA Module | 15           | InternVL2.5-8B | 2                  |

The results, summarized in Table 3, indicate that the retrieval-based approach significantly outperforms the naive baseline. Specifically, while the naive approach, which processes entire videos without segment selection, achieved an accuracy score of 2, the retrieval-based strategy consistently improved VQA performance. Among the evaluated models, VideoLLaMA3-7B achieved the highest accuracy score of 4, closely approaching the theoretical maximum of 5. These findings underscore the effectiveness of retrieval-based video selection in enhancing VQA accuracy.

## 5. Submissions for IViSE

To demonstrate the effectiveness of our proposed framework, we will participate in the 1st International Workshop on Interactive Video Search and Exploration (IViSE). This workshop addresses key challenges in video retrieval and understanding by integrating both automatic and interactive methodologies. It focuses on two primary tasks: Known-Item Search (KIS), which improves retrieval efficiency through scalable indexing and search models, and Visual Question Answering (VQA), which enhances answer accuracy using advanced multi-modal representations and temporal reasoning. Through IViSE, we aim to validate VRAG’s retrieval-augmented reasoning, demonstrating its efficiency in locating relevant video segments and generating precise, context-aware answers, while showcasing the system’s scalability and robustness in real-world video search and VQA.

As part of IViSE competition, we participated in the fully automated track and conducted runs to evaluate our system on 10 KIS queries and 10 VQA queries. All results were generated in a fully automated manner without human intervention in the querying process. The details of each run are presented in Table 4.

For the KIS task, the query was processed through a multimodal search system to retrieve the top 100 candidates, which were refined by a re-ranking module to select the top 10 for submission.

For the VQA task, the query was first processed using GPT-4o, which decomposed it into a retrieval query and a corresponding question. The retrieval query was handled similarly to the KIS task, where the top 10 results were obtained after re-ranking. Subsequently, for each video ID among the top 10 results, the corresponding video was processed using the VQA module to generate the final answer.

## 6. Conclusion

In this work, we introduce VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos, a framework designed to improve efficiency and accuracy in video retrieval and question answering. Inspired by retrieval-augmented generation (RAG) architectures used in chatbot systems, VRAG follows a structured retrieval process, first retrieving relevant video segments and then applying chunking-based refinement to focus on key sub-segments before reasoning over the content. This approach ensures that only the most relevant information is processed, maximizing the effectiveness of LLMs and VLMs. By leveraging retrieval-augmented reasoning, VRAG reduces computational overhead while maintaining crucial spatiotemporal context. Experimental evaluations on VBS 2019-2025 demonstrate significant improvements in retrieval precision and answer quality, validating the effectiveness of our approach in handling long-form video content. These findings highlight the importance of integrating retrieval-driven methods with structured reasoning to enhance scalability and contextual coherence in video understanding. Future research should focus on further optimizing retrieval mechanisms, refining ranking strategies, and extending the framework to accommodate more diverse and complex video datasets.

## Acknowledgment

This research is funded by University of Information Technology-Vietnam National University HoChiMinh City under grant number D4-2025-02

## References

- [1] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, and Cordelia Schmid. Vivit: A video vision transformer. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 6836–6846, 2021. [2](#)
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. [3](#)
- [3] Darwin Bautista and Rowel Atienza. Scene text recognition with permuted autoregressive sequence models. In *European conference on computer vision*, pages 178–196. Springer, 2022. [4](#)
- [4] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In *Proceedings of the International Conference on Machine Learning (ICML)*, 2021. [2](#)
- [5] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by retrieving from trillions of tokens. In *International conference on machine learning*, pages 2206–2240. PMLR, 2022. [3](#)
- [6] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In *International conference on machine learning*, pages 1597–1607. Pmlr, 2020. [3](#)
- [7] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhui Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025. [3](#), [4](#)
- [8] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, page 2818–2829. IEEE, 2023. [4](#)
- [9] Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell. Long-term recurrent convolutional networks for visual recognition and description. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 2625–2634, 2015. [2](#)
- [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. [2](#)
- [11] Spyros Gidaris, Andrei Bursuc, Nikos Komodakis, Patrick Pérez, and Matthieu Cord. Learning representations by predicting bags of visual words. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6928–6938, 2020. [2](#)
- [12] Cathal Gurrin, Liting Zhou, Graham Healy, Werner Bailer, Duc-Tien Dang Nguyen, Steve Hodges, Björn Þór Jónsson, Jakub Lokoč, Luca Rossetto, Minh-Triet Tran, and Klaus Schöffmann. Introduction to the seventh annual lifelog search challenge, lsc’24. In *Proceedings of the 2024 International Conference on Multimedia Retrieval*, page 1334–1335, New York, NY, USA, 2024. Association for Computing Machinery. [1](#)
- [13] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 9729–9738, 2020. [3](#)
- [14] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural Comput.*, 9(8):1735–1780, 1997. [2](#)
- [15] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6769–6781, Online, 2020. Association for Computational Linguistics. [3](#)
- [16] Urva Shi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. *arXiv preprint arXiv:1911.00172*, 2019. [3](#)
- [17] Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In *Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval*, pages 39–48, 2020. [3](#)
- [18] Kyung-Min Kim, Seong-Ho Choi, Jin-Hwa Kim, and Byoung-Tak Zhang. Multimodal dual attention memory for video story question answering. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 673–688, 2018. [1](#)
- [19] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 7331–7341, 2021. [1](#)
- [20] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *International conference on machine learning*, pages 19730–19742. PMLR, 2023. [4](#)
- [21] Chao Ma, Chunhua Shen, Anthony Dick, Qi Wu, Peng Wang, Anton Van den Hengel, and Ian Reid. Visual question answering with memory-augmented networks. In *Proceed-*

- ings of the IEEE conference on computer vision and pattern recognition, pages 6975–6984, 2018. 1
- [22] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In *Proceedings of the IEEE/cvf conference on computer vision and pattern recognition*, pages 3195–3204, 2019. 4
- [23] Keiron O’Shea and Ryan Nash. An introduction to convolutional neural networks, 2015. 2
- [24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. Pmlr, 2021. 2
- [25] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In *International conference on machine learning*, pages 28492–28518. PMLR, 2023. 4
- [26] Benjamin Reichman and Larry Heck. Cross-modal dense passage retrieval for outside knowledge visual question answering. In *2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)*, pages 2829–2834, 2023. 4
- [27] Loris Sauter, Ralph Gasser, Heiko Schuld, Abraham Bernstein, and Luca Rossetto. Performance evaluation in multimedia retrieval. *ACM Trans. Multimedia Comput. Commun. Appl.*, 21(1), 2024. 1
- [28] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. Videobert: A joint model for video and language representation learning. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 7464–7473, 2019. 3
- [29] Lucia Vadicamo, Rahel Arnold, Werner Bailer, Fabio Carrara, Cathal Gurrin, Nico Hezel, Xinghan Li, Jakub Lokoc, Sebastian Lubos, Zhixin Ma, Nicola Messina, Thao-Nhu Nguyen, Ladislav Peska, Luca Rossetto, Loris Sauter, Klaus Schöffmann, Florian Spiess, Minh-Triet Tran, and Stefanos Vrochidis. Evaluating performance and trends in interactive video retrieval: Insights from the 12th vbs competition. *IEEE Access*, 12:79342–79366, 2024. 1
- [30] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for vision and vision-language tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 19175–19186, 2023. 4
- [31] Maoyuan Ye, Jing Zhang, Shanshan Zhao, Juhua Liu, Tongliang Liu, Bo Du, and Dacheng Tao. Deepsolo: Let transformer decoder with explicit points solo for text spotting. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 19348–19357, 2023. 4
- [32] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. *Advances in neural information processing systems*, 34:23634–23651, 2021. 1
- [33] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multi-modal foundation models for image and video understanding. *arXiv preprint arXiv:2501.13106*, 2025. 3
- [34] Zhuofan Zong, Guanglu Song, and Yu Liu. Dets with collaborative hybrid assignments training. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 6748–6758, 2023. 4