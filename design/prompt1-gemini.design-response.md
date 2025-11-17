

# **The Neural RAG Database (NRDB): An Expert Architectural Blueprint for AI-Powered Knowledge Retrieval**

The design and implementation of the Neural RAG Database (NRDB) marks a significant advancement in knowledge retrieval systems, transitioning from simple vector similarity search to structured, neurally augmented reasoning. This document outlines the comprehensive architecture required for a high-performance, low-latency solution optimized for Retrieval-Augmented Generation (RAG) tasks involving complex, multi-hop queries.

### **I. Strategic Overview and The Neural RAG Database (NRDB) Architecture**

The technical mandate for the NRDB is to address the inherent limitations of conventional vector-only RAG systems. While dense vectors effectively capture semantic *similarity* between documents and queries, they fail to explicitly represent the structural *relationships* between entities, which is critical for complex, multi-hop reasoning tasks.1 A system purely reliant on vector similarity will struggle to answer structured queries that require synthesizing facts across multiple context fragments.

The NRDB resolves this by establishing a **hybrid architecture** that uses specialized neural networks not just for embedding, but for the fundamental storage and querying mechanism itself. The core value proposition is the explicit creation and querying of a structured Knowledge Graph (KG) derived from raw text, enabling the Large Language Model (LLM) to receive verified, structured reasoning paths alongside semantic context snippets.3

The successful deployment of this complex hybrid architecture is entirely dependent on rigorously optimizing all neural components for low latency. By mandating extreme optimization techniques such as 8-bit weight quantization and deployment via the ONNX Runtime 4, the NRDB creates the necessary computational capacity to execute the computationally intensive KG processing and Graph Neural Network (GNN) re-ranking steps without introducing unacceptable retrieval latency.

#### **I.A. Defining the Core Neural Components and Performance Objectives**

The NRDB architecture is compartmentalized into two functional categories of optimized neural networks:

1. **Storage Neural Networks (Extraction Pipeline):** Responsible for converting unstructured text into structured graph data. This includes high-efficiency embedding models used for initial chunking and vector index creation. Candidates like all-MiniLM-L6-v2 7 or EmbeddingGemma 9 are prioritized for their compact size and speed. Crucially, a specialized Relation Extraction (RE) Model converts raw text into verifiable graph triples.10  
2. **Query Neural Networks (Retrieval and Ranking):** Dedicated to optimizing the search experience. This layer features the Graph Neural Network (GNN) Reasoner, which identifies and scores logical paths across the KG.1 The final output context is processed by an efficient LLM Generator, such as a quantized version of Mistral 7B or Gemma 3.4B, selected specifically for low VRAM RAG inference.6

**Performance Targets:** The target retrieval latency for the NRDB must be aggressive, aiming for p95 latency below 50ms, accommodating the multi-stage nature of the hybrid retrieval process. Achieving this requires mandatory implementation of hardware acceleration via frameworks like ONNX Runtime, batch processing maximization, and in-memory caching.4

#### **I.B. Key Architectural Decision Matrix: Embedding Model Selection**

The selection of the embedding model has system-wide implications, affecting memory footprint, storage size, and ANN search latency. The architecture prioritizes speed and low dimension to maximize computational efficiency, accepting that the GNN will compensate for any marginal loss in semantic depth.

| Model Candidate | Parameter Count (Approx.) | Embedding Dimension | Quantization Recommendation | Primary RAG Benefit |
| :---- | :---- | :---- | :---- | :---- |
| EmbeddingGemma | 308M | 512 | 8-bit (under 200MB RAM) | State-of-the-art multilingual performance under 500M parameters, ideal for efficient retrieval.9 |
| all-MiniLM-L6-v2 | 22.6M | 384 | 8-bit/Q8 (FastEmbed default) | Extremely fast inference; low dimension (384) reduces storage overhead and latency in ANN searches.4 |
| LiquidAI LFM2 | 1.2B | Varies | Q8 GGUF/Quantized ONNX | Demonstrated high RAG quality, specifically designed for retrieval tasks comparable to larger models.12 |

---

### **II. Component 1: Neural Storage and Knowledge Graph Construction**

#### **II.A. The Data Transformation Pipeline: Structured Knowledge Generation**

Data ingestion begins with initial document processing, which involves semantic chunking and embedding generation. The system leverages highly efficient libraries such as FastEmbed to utilize pre-quantized, small models like all-MiniLM-L6-v2.5

A prerequisite for the Relation Extraction (RE) process is accurate and fast dependency parsing. The system must utilize robust linguistic analysis, such as that provided by optimized implementations of dependency parsers (e.g., Stanford CoreNLP or modern neural parsers).13 This syntactic information, in the form of a dependency tree, is essential, as the subsequent RE model relies on it to construct the structured knowledge. If the parsing step is slow or introduces noise, the resulting KG will be structurally flawed, undermining the entire NRDB design principle.

#### **II.B. Architecture of the Relation Extraction (RE) Model**

The NRDB uses a specialized RE model to transform raw text into KG triples, focusing on achieving high accuracy with minimal computational overhead.

**The Entity-Centric Paradigm:** The model is based on the novel concept of the **entity-centric dependency tree**.10 In this method, instead of relying on the sentence's grammatical root, the model reconstructs the tree by designating the entity (subject or object) as the root node. This structural transformation ensures that the system can more easily capture lexical information strongly related to the entities via an attention mechanism. The syntactic distance between the root entity and other words is encoded as explicit edge labels, which are utilized to learn the precise syntactic relationships between the entities.10

**Model Specification:** The recommended architecture is a small, specialized model—such as a Bi-LSTM/Transformer encoder fused with syntactic features.15 This model is deliberately kept lean, demonstrating that such specialized, smaller approaches can achieve comparable or superior results to generalized, larger LMs for focused tasks like relation extraction.10 The model should be trained on high-quality, supervised relation annotation datasets, including **TACRED, Re-TACRED, and SemEval 2010 Task 8**, where this approach has yielded state-of-the-art F1 scores (e.g., 74.9% on TACRED).10

During the feature engineering phase, the system should incorporate established information retrieval principles. Specifically, the observation that the norm of word vectors can serve as a reliable proxy for the importance of words 17 should be leveraged. By integrating this norm weighting, the RE process can implicitly down-weight frequent, less discriminatory tokens, aligning the KG construction with the Luhn paradigm which posits that mid-rank terms are the most significant. This results in a KG that is denser and more semantically relevant, optimizing the efficiency of subsequent graph traversals.

#### **II.C. KG Modeling and Serialization for Disk Residency**

For the NRDB to scale effectively, its Knowledge Graph must be stored persistently on disk in a format optimized for graph traversal I/O, particularly when the graph size significantly exceeds available memory (e.g., 10x or 20x beyond RAM).18

**Data Layout Selection:** While Compressed Sparse Row (CSR) is memory efficient, its linear complexity for updates makes it impractical for a dynamic RAG environment requiring frequent insertions and deletions.19 The NRDB adopts **Packed Compressed Sparse Row (PCSR)**.19 PCSR retains the space efficiency of CSR but strategically incorporates spaces between elements, allowing for insertions and deletions to be performed orders of magnitude faster. This trade-off—a constant-factor slowdown in traversal for a massive improvement in update speed—is necessary for a continuously ingested RAG database.19

**I/O-Efficient Indexing:** Minimizing disk I/O cost during traversal is paramount. The system must implement a physical data layout that optimizes locality, moving beyond the NP-hard problem of generic global graph partitioning.20 The architecture specifies a **Breadth-First Search (BFS) inspired layout** that clusters highly connected nodes and their neighbors onto the same physical disk block or page.21 This strategy, exemplified by frameworks like Starling, which use an in-memory navigation graph alongside a reordered disk-based structure 20, ensures that multi-hop retrievals minimize random disk seeks by maximizing the relevant data retrieved per I/O block.18 For handling adjacency lists that exceed memory limits, external memory vector systems (like STXXL vectors) must be utilized to ensure efficient data management.22

**Property Storage:** Entity and relation properties, including the dense, 384- or 512-dimensional embedding vectors, must be stored in a columnar format optimized for block-oriented reading, such as **Apache Parquet or ORC**.23

##### **Table II: Comparison of Storage Formats for Disk-Resident Knowledge Graphs**

| Storage Format | Structure Type | I/O Efficiency for Traversal | Update Complexity | Suitability for Dynamic NRDB |
| :---- | :---- | :---- | :---- | :---- |
| Compressed Sparse Row (CSR) | Static Sparse Adjacency | High (Sequential Read) | High (Linear time for updates, expensive) | Suitable only for static graphs or archival subsets.24 |
| **Packed Compressed Sparse Row (PCSR)** | Dynamic Sparse Adjacency | High (Slight slowdown vs. CSR) | **Low (Asymptotically faster inserts/deletions)** | **Recommended:** Optimized for the continuous, dynamic updates required in a modern RAG system.19 |
| Apache Parquet/ORC | Columnar Storage (Properties) | High (Block-oriented read) | Low/Moderate | **Recommended:** Ideal for storing large, dense feature vectors (embeddings) and entity metadata efficiently.23 |

---

### **III. Component 2: Dual Indexing and Large-Scale Retrieval Mechanisms**

#### **III.A. Dense Vector Indexing for Semantic Search**

The vector index serves as the initial semantic filter. The **Hierarchical Navigable Small World (HNSW)** algorithm is the favored Approximate Nearest Neighbor (ANN) index due to its superior high-recall/low-latency profile.4 Alternatively, Inverted File Indexing (IVF) offers speed gains by limiting searches to relevant clustered subsets of embeddings.4

**Vector Optimization:** To maximize retrieval speed and minimize memory pressure, three technical mandates are enforced:

1. **Normalization:** All embeddings must be pre-normalized to unit vectors. This allows for the use of the computationally cheaper dot-product operation to calculate cosine similarity, which is the standard metric for RAG.4  
2. **Quantization:** Embeddings must be quantized from 32-bit floats down to 8-bit integers. This directly reduces storage size, memory usage, and computational overhead during distance calculation within the ANN index.4  
3. **Dimensionality:** While high dimensions offer marginal fidelity gains, the optimal dimension for overall low-latency system performance is 384 (e.g., all-MiniLM-L6-v2).8 This lower dimension drastically reduces the memory usage and the computation required for distance metric calculation during the ANN search, ensuring efficient vector I/O.4 The GNN re-ranking step is strategically designed to compensate for any accuracy loss from this dimension reduction.

#### **III.B. Sparse/Structural Indexing for KG Traversal**

The structural index utilizes the disk-resident PCSR layout.19 To ensure optimal I/O for external memory operations, the system must prioritize search locality. Strategies mirroring the Starling framework—using an in-memory graph segment for navigation coupled with a reordered disk layout 20—are employed to minimize the number of disk accesses required to traverse multiple hops, critical for complex KGQA problems. This includes leveraging external memory techniques (e.g., STXXL) when managing adjacency lists that exceed system memory.18

---

### **IV. Component 3: The Neural Query Engine and Ranking Layer**

This component executes the structural reasoning required for advanced RAG, shifting the heavy lifting of multi-hop inference from the LLM to a dedicated Graph Neural Network.

#### **IV.A. Query Parsing and Candidate Subgraph Retrieval**

The process begins by converting the query into a structured form, specifically a **Question Graph** ($G\_q$), using a dependency parser.1 A simultaneous **Hybrid Initial Retrieval** is executed: vector search on the HNSW index and structural search (entity matching and limited traversal) on the PCSR index.

The combined results form an initial **Knowledge Subgraph (KSG)**. Since this KSG can be large, containing potentially thousands of nodes 1, it must be partitioned into smaller, manageable sub-KSGs using an efficient partition algorithm.1 This sub-KSG approach is essential for feeding high-quality, focused context to the GNN reasoner. The GNN is explicitly designated as the "dense subgraph reasoner," tasked with finding and scoring logical paths (e.g., shortest paths from question entities to potential answers).3 This approach ensures the LLM receives structured reasoning context, leading to more verifiable and accurate complex query answers.

#### **IV.B. Graph Neural Network (GNN) Re-ranking Architecture**

A high-performance GNN must be utilized to score the relevance of these candidate sub-KSGs.

**GNN Architecture Selection:** The recommended architecture is the **Bi-directional Gated Graph Sequence Neural Network (GGNN)**.1 This network type is proven in Knowledge Graph Question Answering (KGQA) tasks for its ability to capture global contextual and structural information. The GNN iteratively updates node representations ($h^{(l)}\_v$) by aggregating messages from neighbors, distinguishing between incoming ($\\triangleleft$) and outgoing ($\\triangleright$) edges.1 The state update is managed by a **Gated Recurrent Unit (GRU)**, ensuring efficient integration of aggregated information.1 Max-pooling is typically used on the final node representations to generate a fixed-size vector representing the entire sub-KSG for ranking purposes.1

#### **IV.C. Training the Neural Ranker: Listwise Loss Optimization**

The effectiveness of the GNN ranker lies in its ability to order the retrieved sub-KSGs optimally. The architecture mandates the use of a **Listwise Ranking Loss** function.

**Ranking Strategy:** The Listwise approach 25 is superior to Pointwise or Pairwise methods because it optimizes the relevance of the entire retrieved list of context simultaneously.27 In a complex RAG system, the synergistic relationship between retrieved KG facts is crucial; optimizing the global permutation of context maximizes the coherence and quality of the final input to the LLM. Research confirms that listwise approaches perform better in information retrieval ranking tasks than pairwise methods.26

**Recommended Loss:** The GNN ranking model should be trained using a listwise function such as **ListNet**, which minimizes the cross-entropy between the model's predicted ranking distribution and the ground truth ranking.27 This training goal ensures the GNN produces a relevance score ($y$) that results in the most effective ordering of structured context for the subsequent LLM generation.

---

### **V. Model Training, Optimization, and Production Deployment**

#### **V.A. Model Selection and Optimization Strategy**

To meet the high-throughput, low-latency goals, all neural components must undergo a transformation and optimization process focused on minimizing resource consumption.

**RAG LLM Specification:** For the final generation step, lightweight LLMs like the quantized **Gemma 3.4B (Q4\_K\_M)** are recommended, requiring only 4GB of VRAM and demonstrating high performance fidelity (e.g., achieving 59.05 on the OpenLLM benchmark, 99.7% of the unquantized score).6 The use of such highly optimized, low-VRAM models is essential as it significantly reduces the Total Cost of Ownership (TCO) by enabling high-density deployment on commodity GPU hardware.

**The ONNX/Quantization Mandate:** Every neural network—embeddings, RE model, GNN ranker, and LLM—must be converted to the **ONNX (Open Neural Network Exchange) format**.29 This facilitates platform-agnostic optimization. Crucially, all models must employ 8-bit integer quantization, which can reduce disk size and GPU memory requirements by approximately 50% with minimal impact on accuracy.4 The use of tools like the Hugging Face optimum.exporters.onnx package ensures a reliable conversion workflow from PyTorch or Transformer models.29

#### **V.B. Infrastructure and Hardware Acceleration using ONNX Runtime**

Production inference must be managed by the **ONNX Runtime (ORT)**, the dedicated execution engine for optimized ONNX models.5

**CUDA Compatibility Management:** The stability of high-speed deployment relies entirely on ensuring perfect compatibility between the chosen ORT version, the NVIDIA CUDA Toolkit, and the cuDNN library. Version conflicts (e.g., between ORT and specific CUDA 12.x/cuDNN 9.x pairings) are a frequent cause of deployment failure.31 Therefore, the NRDB must standardize its entire stack on a rigidly verified and stable configuration (e.g., ORT 1.20.x, CUDA 12.x, and cuDNN 9.x).33 This mitigates the significant operational risk associated with version mismatch in accelerated machine learning production environments.

**Serving Optimization:** The NRDB deployment must incorporate two essential serving optimizations: **batch processing** (grouping requests for simultaneous accelerator utilization) and **in-memory caching** (caching frequently requested embeddings or GNN outputs, likely via Redis, to minimize disk I/O and recomputation).4

### **VI. Conclusion and Roadmap for Implementation**

The Neural RAG Database provides a robust, expert-level solution for modern RAG requirements by systematically integrating structured reasoning into the retrieval loop. Its core competitive advantages lie in its use of a specialized, neurally derived Knowledge Graph and its aggressive pursuit of low-latency performance through optimization mandates (ONNX/Quantization) and listwise neural ranking.

#### **Phased Implementation Roadmap**

To successfully deploy the NRDB, a phased approach is recommended:

1. **Phase I: Foundation and Extraction:** Implement the document processing pipeline, including the optimized dependency parser and the specialized Entity-Centric Relation Extraction model, trained on TACRED and similar datasets.10 Establish the dual indexing system (HNSW/IVF for vectors and initial CSR for structure). Formalize the ONNX export and ORT deployment pipeline, ensuring strict adherence to the stable CUDA compatibility matrix.33  
2. **Phase II: Neural Query and Ranking:** Develop and train the GNN Reasoner (GGNN architecture).1 Crucially, implement the Listwise Ranking Loss (e.g., ListNet) to optimize the selection and ordering of sub-KSGs.27 Integrate the hybrid search methodology combining vector and structural retrieval.  
3. **Phase III: Scale and Production:** Migrate the static graph structure (CSR) to the dynamic, I/O-efficient **Packed Compressed Sparse Row (PCSR)** format.19 Implement the disk block reordering strategy based on search locality to minimize I/O cost.20 Finalize serving infrastructure with caching and optimized batch processing, deploying the selected quantized RAG LLM (e.g., Gemma 3.4B Q4\_K\_M) for maximized efficiency.6

**Future Development:** Once the core NRDB system is validated, future work should investigate leveraging dedicated Graph Foundation Models (FMs), such as GFM-RAG 34, to further enhance the system's ability to generalize KG retrieval across diverse, automatically extracted graph datasets.

#### **עבודות שצוטטו**

1. Graph-augmented Learning to Rank for Querying ... \- ACL Anthology, נרשמה גישה בתאריך נובמבר 14, 2025, [https://aclanthology.org/2022.aacl-main.7.pdf](https://aclanthology.org/2022.aacl-main.7.pdf)  
2. QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering \- CS Stanford, נרשמה גישה בתאריך נובמבר 14, 2025, [https://cs.stanford.edu/people/jure/pubs/qagnn-naacl21.pdf](https://cs.stanford.edu/people/jure/pubs/qagnn-naacl21.pdf)  
3. Gnn-Rag: Graph Neural Retrieval for Large Language Model Reasoning \- arXiv, נרשמה גישה בתאריך נובמבר 14, 2025, [https://arxiv.org/html/2405.20139v1](https://arxiv.org/html/2405.20139v1)  
4. How do you optimize embeddings for low-latency retrieval? \- Milvus, נרשמה גישה בתאריך נובמבר 14, 2025, [https://milvus.io/ai-quick-reference/how-do-you-optimize-embeddings-for-lowlatency-retrieval](https://milvus.io/ai-quick-reference/how-do-you-optimize-embeddings-for-lowlatency-retrieval)  
5. FastEmbed: Qdrant's Efficient Python Library for Embedding Generation, נרשמה גישה בתאריך נובמבר 14, 2025, [https://qdrant.tech/articles/fastembed/](https://qdrant.tech/articles/fastembed/)  
6. Gemma 2 2b It Quantized.w8a16 · Models \- Dataloop, נרשמה גישה בתאריך נובמבר 14, 2025, [https://dataloop.ai/library/model/neuralmagic\_gemma-2-2b-it-quantizedw8a16/](https://dataloop.ai/library/model/neuralmagic_gemma-2-2b-it-quantizedw8a16/)  
7. sentence-transformers/all-MiniLM-L6-v2 \- Hugging Face, נרשמה גישה בתאריך נובמבר 14, 2025, [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
8. Supported Models \- FastEmbed, נרשמה גישה בתאריך נובמבר 14, 2025, [https://qdrant.github.io/fastembed/examples/Supported\_Models/](https://qdrant.github.io/fastembed/examples/Supported_Models/)  
9. Welcome EmbeddingGemma, Google's new efficient embedding model \- Hugging Face, נרשמה גישה בתאריך נובמבר 14, 2025, [https://huggingface.co/blog/embeddinggemma](https://huggingface.co/blog/embeddinggemma)  
10. Effective sentence-level relation extraction model using entity-centric ..., נרשמה גישה בתאריך נובמבר 14, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11419662/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11419662/)  
11. 10 Best Small Local LLMs to Try Out (\< 8GB) \- Apidog, נרשמה גישה בתאריך נובמבר 14, 2025, [https://apidog.com/blog/small-local-llm/](https://apidog.com/blog/small-local-llm/)  
12. This tiny LLM dominates RAG and is SUPER FAST \- YouTube, נרשמה גישה בתאריך נובמבר 14, 2025, [https://www.youtube.com/watch?v=5LTDuOg9DVo](https://www.youtube.com/watch?v=5LTDuOg9DVo)  
13. Software \> Stanford Parser, נרשמה גישה בתאריך נובמבר 14, 2025, [https://nlp.stanford.edu/software/lex-parser.shtml](https://nlp.stanford.edu/software/lex-parser.shtml)  
14. A Fast and Accurate Dependency Parser using Neural Networks, נרשמה גישה בתאריך נובמבר 14, 2025, [https://courses.grainger.illinois.edu/cs546/sp2020/Slides/Lecture17.pdf](https://courses.grainger.illinois.edu/cs546/sp2020/Slides/Lecture17.pdf)  
15. Pretrained Knowledge Base Embeddings for improved Sentential Relation Extraction \- ACL Anthology, נרשמה גישה בתאריך נובמבר 14, 2025, [https://aclanthology.org/2022.acl-srw.29.pdf](https://aclanthology.org/2022.acl-srw.29.pdf)  
16. Simple Relation Extraction with a Bi-LSTM Model — Part 1 | by Marion Valette | southpigalle, נרשמה גישה בתאריך נובמבר 14, 2025, [https://medium.com/southpigalle/simple-relation-extraction-with-a-bi-lstm-model-part-1-682b670d5e11](https://medium.com/southpigalle/simple-relation-extraction-with-a-bi-lstm-model-part-1-682b670d5e11)  
17. Extracting Sentence Embeddings from Pretrained Transformer Models \- arXiv, נרשמה גישה בתאריך נובמבר 14, 2025, [https://arxiv.org/html/2408.08073v2](https://arxiv.org/html/2408.08073v2)  
18. Load the Edges You Need: A Generic I/O Optimization for Disk-based Graph Processing \- USENIX, נרשמה גישה בתאריך נובמבר 14, 2025, [https://www.usenix.org/system/files/conference/atc16/atc16\_paper-vora.pdf](https://www.usenix.org/system/files/conference/atc16/atc16_paper-vora.pdf)  
19. Packed Compressed Sparse Row: A Dynamic Graph Representation \- Helen Xu, נרשמה גישה בתאריך נובמבר 14, 2025, [https://itshelenxu.github.io/files/papers/pcsr.pdf](https://itshelenxu.github.io/files/papers/pcsr.pdf)  
20. Starling: An I/O-Efficient Disk-Resident Graph Index Framework for High-Dimensional Vector Similarity Search on Data Segment \- arXiv, נרשמה גישה בתאריך נובמבר 14, 2025, [https://arxiv.org/html/2401.02116v3](https://arxiv.org/html/2401.02116v3)  
21. Storing very large graphs on disk/streaming graph partitioning algorithms? \- Stack Overflow, נרשמה גישה בתאריך נובמבר 14, 2025, [https://stackoverflow.com/questions/2153963/storing-very-large-graphs-on-disk-streaming-graph-partitioning-algorithms](https://stackoverflow.com/questions/2153963/storing-very-large-graphs-on-disk-streaming-graph-partitioning-algorithms)  
22. I/O-Efficient Multi-Criteria Shortest Paths Query Processing on Large Graphs \- IEEE Xplore, נרשמה גישה בתאריך נובמבר 14, 2025, [https://ieeexplore.ieee.org/iel7/69/10709365/10496202.pdf](https://ieeexplore.ieee.org/iel7/69/10709365/10496202.pdf)  
23. Format Specification \- Apache GraphAr, נרשמה גישה בתאריך נובמבר 14, 2025, [https://graphar.apache.org/docs/specification/format/](https://graphar.apache.org/docs/specification/format/)  
24. LSMGraph: A High-Performance Dynamic Graph Storage System with Multi-Level CSR, נרשמה גישה בתאריך נובמבר 14, 2025, [https://arxiv.org/html/2411.06392v1](https://arxiv.org/html/2411.06392v1)  
25. Graph-Based Re-ranking: Emerging Techniques, Limitations, and Opportunities \- arXiv, נרשמה גישה בתאריך נובמבר 14, 2025, [https://arxiv.org/html/2503.14802v1](https://arxiv.org/html/2503.14802v1)  
26. Learning to Rank for Information Retrieval Contents, נרשמה גישה בתאריך נובמבר 14, 2025, [https://dmice.ohsu.edu/bedricks/courses/cs635\_spring\_2017/pdf/liu\_2009.pdf](https://dmice.ohsu.edu/bedricks/courses/cs635_spring_2017/pdf/liu_2009.pdf)  
27. Learning to Rank: From Pairwise Approach to Listwise Approach \- Microsoft, נרשמה גישה בתאריך נובמבר 14, 2025, [https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)  
28. Ranking Structured Objects with Graph Neural Networks \- Open Access LMU, נרשמה גישה בתאריך נובמבר 14, 2025, [https://epub.ub.uni-muenchen.de/91630/1/2104.08869v2.pdf](https://epub.ub.uni-muenchen.de/91630/1/2104.08869v2.pdf)  
29. Export a model to ONNX with optimum.exporters.onnx \- Hugging Face, נרשמה גישה בתאריך נובמבר 14, 2025, [https://huggingface.co/docs/optimum-onnx/onnx/usage\_guides/export\_a\_model](https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model)  
30. Export to ONNX \- Hugging Face, נרשמה גישה בתאריך נובמבר 14, 2025, [https://huggingface.co/docs/transformers/v4.29.1/serialization](https://huggingface.co/docs/transformers/v4.29.1/serialization)  
31. Fail to Run OnnxRuntime Session in C\# with CUDA Device \- Stack Overflow, נרשמה גישה בתאריך נובמבר 14, 2025, [https://stackoverflow.com/questions/79819810/fail-to-run-onnxruntime-session-in-c-sharp-with-cuda-device](https://stackoverflow.com/questions/79819810/fail-to-run-onnxruntime-session-in-c-sharp-with-cuda-device)  
32. Keep falling into CPU Path · Issue \#26232 · microsoft/onnxruntime \- GitHub, נרשמה גישה בתאריך נובמבר 14, 2025, [https://github.com/microsoft/onnxruntime/issues/26232](https://github.com/microsoft/onnxruntime/issues/26232)  
33. NVIDIA \- CUDA | onnxruntime, נרשמה גישה בתאריך נובמבר 14, 2025, [https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)  
34. KGGen: Extracting Knowledge Graphs from Plain Text with Language Models \- arXiv, נרשמה גישה בתאריך נובמבר 14, 2025, [https://arxiv.org/html/2502.09956v1](https://arxiv.org/html/2502.09956v1)