I want to create a new AI-powered database with a new way of storing and querying data using neural networks.
This new database will be used primarily as a RAG for advanced and modern language models.

The “index” method should be able to:
1. Get full raw documents in Markdown format in the request
2. Convert the raw documents into embedding vectors.
3. Break the embedding vectors into small pieces of data with relations (might be more than one) between them, by understanding the structure of the sentences, instead of using big language models.
The relations can be for example:
3.1. Operations
3.2. Sequence
3.3. Serial number
3.4. Timestamp
4. Save all the small pieces of data in a big data-graph file.

The “query” method should be able to:
1. Get a short simple question on the indexed data, in a free-text of up to 256 characters.
2. Find only relevant pieces of data and relations in the big data-graph file.
3. Pass the simple question (from #1) + the found pieces of data and relations (from #2) into a small language model, to build from it a coherent and simple answer.

I assume we will need for that mission 3 new neural network models:
1. Index-Model - that will be able to get an embedding vector represents a single document, and break it down into small pieces of data and relations.
2. Search-Model - that will be able to get a simple question of up to 256 characters, and to find the relevant pieces of data and relations within the data-graph file.
3. Answer-Model - that will be able to get the simple question + the found pieces of data and to build from it a coherent and simple answer.

Notes:
- The database can work for now in English only.
- The database should be able to get documents in free-text and queries that are actually simple free-text questions.
- The data-graph file might be built from massive abount of big documents, and the new DB must be able to find correlations between the data.
- Besides simple questions about the data, the database should be able to answer about questions related to questions about questions about correlated data by using reasoning.
  For example, even if there is no single document that show all types of neural networks that are considered as DNN, but there are some documents about different neural networks and within them it mentioned that they are types of DNN, the database should correlate the different relevant pieces of data together.
- I don't want to use vector DB or existing graph database behind the scenes.
- The new neural network models must be small ones, so they will run fast.
- The “index” method is allowed to take relatively long time, but the “query” must be efficient and fast.
- The "cost" (in resources) of the "query" is really important, and therefore it should use as minimal RAM/VRAM as possible.

Examples for queries:
If the database contains documents about neural network, I should be able to ask questions like the following, even if the relevant information originated from big number of different documents:
- What is it RNN? 
- Which types of DNN exists? 
- What alternatives there are for CNN? 
- Describe to me the Backpropagation algoritem and it's phases?
- How to train language model?
- Which DNN type can be the best to do X?

Technologies to be used:
- Python as a programing language foromodels creationd+tion + Pyas the Tramework to createch as the f and to export them as ONNX
- .NET Core as a programing language to run the models in production + to control the index/query via gRPC APIs 
- CUDA 12.8.0ramework to create the models and to expiort them as ONNX
- .NET Core as a programing language to run the models in production + to control the index/query via gRPC APIs 
- CUDA 12.8.0

Your mission:
Write for me a very detailed design document for a full working solution, based on my ideas above + strategies and libraries to be used for the training of the models + algoritems to be used for index and query the data.
Explain very detailed the way that things should work and connected, how the data will be break.
Note that if there are already recommended models that can be reused and will be a **perfect** fit to the project (e.g. for embedding), use them instead of creating new custom models into small pieces, how to search should work.
You can change the way things are working, if you have better ideas, but the overall solution (AI-driven database) should remain the same.
Note that if there are already recommended models that can be reused and will be a **perfect** fit to the project (e.g. for embedding), use them instead of creating new custom models.