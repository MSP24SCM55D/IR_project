## Information-Retrieval-Project-A20549555

## MAITHREYEE DOMA

## ABSTRACT

This project showcases an Information Retrieval System that comprises a web crawler built utilizing Scrapy, an indexer fueled by Scikit-Learn, and an inquiry processor created with Flask. The crawler recovers web substance, the indexer makes an altered list utilizing TF-IDF representation, and the processor oversees free-text questions utilizing cosine similitude scoring. Using WordNet for inquiry development progresses the quality of the look comes about. The innovation exhibits capable recovery of relevant records by using client questions. Future enhancements will center on upgrading execution for gigantic datasets and upgrading mistakes dealing with capabilities. In outline, the consideration sets up a strong premise for building flexible data recovery frameworks that have the potential to be connected in different fields.
# OVERVIEW

The Information Recovery Framework described here consists of three main components: a web crawler, an indexer, and an inquiry processor. The goal of this system is to streamline the process of finding useful information from web documents when users explore specific topics.

The Scrapy-based web crawler is responsible for navigating online sites, extracting content, and populating the document collection. The indexer utilizes Scikit-Learn to create a customized index using TF-IDF representation and enables cosine similarity scoring for query processing.

The inquiry processor, built using Jar, receives free-text queries, conducts query expansion using WordNet, and ranks relevant documents according to cosine similarity. The integration of these components forms a comprehensive Information Retrieval Framework that can efficiently process user queries and retrieve valuable information from a wide variety of web publications.
# DESIGN

The design of the project involves three main components: the web crawler, the document indexer, and the query processor. These components work together seamlessly to enable efficient content retrieval, indexing, and retrieval.

## Web Crawler:

The web crawler, implemented using Scrapy, serves as the backbone of the system. It is responsible for fetching web pages, extracting relevant content, and following links to discover new pages. The crawler begins its journey from seed URLs provided by the user and recursively explores the web graph until reaching the specified maximum number of pages or depth. Concurrent crawling and distributed crawling capabilities can be added to enhance performance and scalability.

## Document Indexer:

The document indexer, powered by Scikit-Learn, constructs an inverted index based on the crawled web documents. It leverages the TF-IDF (Term Frequency-Inverse Document Frequency) representation to assign weights to terms in each document and calculate their importance in the corpus. This inverted index enables efficient searching and retrieval of relevant documents based on user queries. Additionally, advanced techniques such as vector embedding representation (e.g., word2vec) and neural or semantic search (e.g., FAISS) can be incorporated for more sophisticated indexing and retrieval capabilities.

## Query Processor:

The query processor, built using Flask, provides an interface for users to submit free-text queries and retrieve relevant documents. Upon receiving a query, the processor expands it using techniques like query expansion (e.g., WordNet) to enhance search accuracy. The expanded query is then vectorized using the same TF-IDF representation used in indexing. Cosine similarity is employed to compare the query vector with document vectors stored in the inverted index, yielding a ranked list of relevant documents. The processor returns the top-K ranked results to the user, enabling efficient information retrieval.
# ARCHITECTURE

The architecture of the system follows a modular and scalable design, allowing for easy integration of new functionalities and enhancements. Each component operates independently, communicating through well-defined interfaces.

## Web Crawler:

The web crawler component consists of spiders responsible for fetching and parsing web pages. It utilizes Scrapy's built-in features for asynchronous processing and request handling. Concurrent crawling can be achieved using Scrapy's AutoThrottle middleware, while distributed crawling can be implemented using tools like Scrapyd.

## Document Indexer:

The document indexer component is initialized with a corpus of web documents obtained from the crawler. It utilizes the TfidfVectorizer from Scikit-Learn to transform the text data into TF-IDF matrices. These matrices are then stored as an inverted index, allowing for efficient cosine similarity calculations during query processing.

## Query Processor:

The query processor component exposes RESTful endpoints using Flask, enabling users to submit queries via HTTP requests. It utilizes NLTK for query expansion and Scikit-Learn for vectorization and similarity calculations. The processor can handle multiple queries concurrently and return results in JSON format for easy consumption by client applications.
# OPERATION

The operation of the code involves three core processes: crawling, indexing, and querying. Users initiate the process by providing seed URLs and setting parameters for crawling, such as the maximum number of pages and depth. The web crawler then traverses the web, fetching pages, extracting content, and following links to discover new pages.

During the crawling process, the crawler systematically explores the web graph, starting from the seed URLs provided by the user. It retrieves the HTML content of each page and parses it to extract relevant information, such as text content and links to other pages. The crawler follows these links to explore additional pages, recursively traversing the web to gather a diverse set of documents.

To prevent infinite loops and excessive resource consumption, users can specify parameters like the maximum number of pages to crawl and the maximum depth to explore. These parameters help control the scope of the crawling process and ensure efficient resource utilization. Additionally, the crawler keeps track of visited pages to avoid revisiting them, further optimizing the crawling process.

Once the crawling process is complete, the next step is indexing. The indexed documents are processed to create an inverted index, a data structure that maps terms to the documents in which they appear, along with information about their frequency and importance.

The indexer employs techniques like TF-IDF (Term Frequency-Inverse Document Frequency) to assign weights to terms based on their frequency in individual documents and their significance in the entire corpus. This weighted representation enables efficient matching of queries with relevant documents during retrieval.

The indexed documents are stored in a suitable data structure, such as a matrix or dictionary, to facilitate fast lookup and retrieval. Additionally, the index may be saved to disk in a serialized format, such as a pickle file, for persistence and future use.

Once the documents are indexed, users can submit free-text queries to search for relevant information. The query processor receives the query input and preprocesses it to enhance search accuracy. Techniques like query expansion may be employed to broaden the scope of the query and improve recall.

The preprocessed query is then vectorized using the same TF-IDF representation used in indexing. This vectorization transforms the query into a numerical vector that captures its semantic meaning and importance. The cosine similarity between the query vector and document vectors stored in the inverted index is calculated to determine the relevance of each document to the query.

The query processor returns the top-K ranked documents based on their similarity scores, providing users with a ranked list of relevant results. The results are typically presented in a user-friendly format, such as a web page or JSON response, allowing users to easily navigate and access the retrieved documents.

In summary, the operation of the code involves a seamless process of crawling, indexing, and querying web documents. Users can initiate the process by providing seed URLs and parameters for crawling, after which the crawler fetches pages and extracts content. The indexed documents are then used to construct an inverted index, enabling efficient search and retrieval. Users can submit queries to search for relevant documents, which are processed and ranked based on their similarity to the query. Overall, the operation of the code facilitates efficient information retrieval from web documents, empowering users to access relevant content with ease.
# CONCLUSION

The provided Data Retrieval System demonstrates a comprehensive approach to crawling, indexing, and processing queries of web documents. The integration of Scrapy for web crawling, Scikit-Learn for indexing, and Flask for query processing provides a solid foundation for information retrieval tasks. The system efficiently retrieves relevant documents by utilizing TF-IDF representation and cosine similarity scoring. Moreover, the incorporation of query expansion using WordNet enhances the search experience by considering synonyms and related terms.

While the system shows promising possibilities, some areas require additional development and exploration. Implementing optimization techniques such as caching and parallel processing can enhance the performance of query processing, particularly when dealing with large document collections. Additionally, ongoing error handling and input validation would strengthen the system's robustness and improve user experience.

In summary, this framework offers a solid framework for building advanced information retrieval systems and serves as a foundation for future research and advancement in this field.
# DATA SOURCES

The primary data source for the provided codebase is the collection of web documents obtained through the crawling process. Users input seed URLs as starting points for the web crawler, which systematically explores the web, fetching pages, extracting content, and following links to discover new pages. These crawled documents represent a diverse set of web pages spanning various topics and domains. They serve as the foundation for building the search index, enabling users to search for relevant information efficiently.

Additionally, the code relies on external libraries and packages for various functionalities. The Scrapy library facilitates web crawling by providing tools for fetching pages, parsing HTML content, and following links. It allows users to specify parameters such as maximum pages and depth to control the crawling process. Scikit-Learn is utilized for document indexing and similarity calculations, offering algorithms like TF-IDF for constructing the inverted index and cosine similarity for ranking search results. Flask is employed for building the query processor, enabling users to submit queries and retrieve search results through a web interface. Lastly, NLTK may be used for text preprocessing and query expansion, enhancing search accuracy and relevance.

In summary, the primary data source for the project is the collection of web documents obtained through crawling. External libraries and packages such as Scrapy, Scikit-Learn, Flask, and NLTK contribute to various functionalities such as web crawling, indexing, query processing, and text analysis. These data sources collectively enable the development of a robust and effective information retrieval system capable of efficiently searching and retrieving relevant content from web documents.
# TEST CASES

The test cases section outlines the methodologies employed to validate the functionality and reliability of the system. It discusses the selection of test cases, the criteria for evaluating success, and the framework used for testing. The section underscores the importance of rigorous testing in ensuring the quality and robustness of the system.

<img width="468" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/5437eda2-f635-4f91-ab33-b57b766fe47b">

The above screenshot shows the terminal while executing.
# SOURCE CODE

## Crawler

The Crawler module of our Data Recovery Framework is a fundamental tool for gathering web documents in HTML format. Utilizing Scrapy, this tool provides essential capabilities for accurate data acquisition.

<img width="472" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/6ef9dccc-d3e9-4997-b154-338d058c8bff">

#Figure 1: Defining Myspider (Crawler)

During the initialization handle, the crawler permits clients to input certain settings, just like the seed URL, the maximum number of pages to crawl, and the most extreme depth for navigating links. This enables the execution of customized crawling methodologies that are well-suited to meet one-of-a-kind data recovery necessities (Hossain et al., 2024). In expansion, it gives bolster for concurrent crawling, progressing effectiveness through AutoThrottle strategies and empowering disseminated slithering with scraped, permitting for versatility in large-scale information collecting exercises.

<img width="472" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/4425d9da-d7f5-446d-9474-484be359a5dd">

#Figure 2: Defining start request

During the process of crawling, the crawler methodically navigates over web pages, starting from the seed URL. The framework recovers pertinent data from each page, encompassing titles and textual content within paragraphs, to ensure a thorough representation of the required subject area. In this way, the information is organized and saved for subsequent analysis.

<img width="472" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/e1a9abe9-180a-4be0-81a7-06229c716a45">

#Figure 3: Defining the condition

In order to avoid perpetual crawling and guarantee optimal execution, the crawler uses strategies to confine the number of pages crept and the degree of traversal. This ensures that the crawler concentrates on related substances within a feasible run, hence avoiding unnecessary utilization of resources.

The crawler part plays a vital part in getting data for the Data Retrieval Framework (Monterrubio et al., 2021). The software's capable features, such as the ability to customize crawling parameters, perform multiple tasks simultaneously, and lay down certain boundaries on the relevance of web document recovery, enable users to effectively gather web pages for future querying and analysis operations.

## Indexer

The Indexer part of the Data Retrieval Framework plays a crucial role in enabling powerful and accurate search capabilities. The Indexer is a software tool that utilizes the Scikit-Learn library to form a modified index. This index is constructed using the Term Frequency-Inverse Document Frequency (TF-IDF) scheme, which allows efficient document indexing and retrieval.

<img width="401" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/36a45e05-a7be-4195-bce1-6c3521c1c6fd">

#Figure 4: Defining Indexer class

When the Indexer is created, it sets up a TF-IDF vectorizer, which is a tool that transforms text documents into numerical feature vectors by considering the frequency of terms within the corpus of documents (Costola et al., 2023). The process of vectorization enables the assessment of the importance of each term in a document relative to the entire collection of documents, making it easier to compute similarity scores between queries and documents.

<img width="431" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/acf08d6f-a37a-45cf-8ac3-34c153cfe8e5">

#Figure 5: Defining cosine_sim

The primary role of the Indexer is to create and manipulate the modified index. The Indexer uses the TF-IDF vectorizer to create a sparse matrix that represents the document collection. Each row in the matrix represents a document, and each column represents a unique term within the corpus. This matrix serves as the basis for calculating cosine similarity scores between queries and documents, facilitating the effective retrieval of relevant materials based on query terms.

<img width="413" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/e2c99b38-6d43-4542-9db0-746fa329b63b">

#Figure 6: Loading the index with an example query

The Indexer offers strategies for protecting and recovering file information, empowering long-term conservation and recovery of the record. This includes ensuring that the ordering handle can be executed once and after that used for numerous look questions, thus making strides in proficiency and limiting computational burden.

<img width="472" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/a34436bc-ba13-4196-a7e0-b366732c9798">

#Figure 7: Output of the Indexer query

The Indexer component is pivotal within the Information Recovery Framework because it offers a solid and compelling strategy for ordering reports and empowering exact look capabilities by calculating TF-IDF-based similarity scores (Rahman et al., 2023). The integration of this framework with the bigger system permits for easy recovery of germane records in reaction to client requests, thus progressing the in general client involvement and value of the framework.
## Processor
The Processor component of our Data Retrieval System capacities as the interface for overseeing free-text inquiries and recovering related records from the recorded collection. Created on Flask, it gives a coordinated HTTP-based interface for questioning and taking care of client inquiries.

<img width="454" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/3caaed66-31ac-407c-9224-7d6c413c7e03">

#Figure 8: Defining flask

When the Processor gets an inquiry in JSON arrange, it confirms the input and extricates the inquiry string alongside any discretionary parameters, such as the required number of comes about (top-K). Next, the framework uses a TF-IDF vectorizer to change the inquiry into a numerical organization that's congruous with the recorded records (Urchs et al., 2020). In expansion, the Processor utilizes cosine similitude to calculate the significance scores between the inquiry vector and the TF-IDF framework of the report collection.

<img width="344" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/8bd64008-93b7-462d-8c35-2bccfecd5f84">

#Figure 9: Defining TF-IDF vectorizer

In arrange to make strides in the exactness of looks, the Processor joins an inquiry extension component that utilizes WordNet, an English lexical database. This strategy improves the introductory inquiry by counting equivalent words obtained from WordNet, expanding the look run, and expanding the likelihood of obtaining relevant reports.


<img width="435" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/49232408-7abd-4df6-b2de-f315baa3efd4">

#Figure 10: Calculating cosine similarity

The Processor delivers the top-K-ranked results to the client, enclosed in a JSON reaction (Costola et al., 2023). Each result contains the substance of the archive and its related score of similitude, which makes it less demanding and faster for the user to assess and show look comes about.


<img width="402" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/83de3bb3-ace9-478b-a742-1f404ba00a71">

#Figure 11: Defining expanding query
Although the existing arrangement is successful for inquiry handling, there are openings for extra upgrades (Urchs et al., 2020). Future modifications ought to organize advancing speed to effortlessly handle bigger record collections, create advanced positioning algorithms, and update the client involvement with highlights such as spell correction and query recommendations.

<img width="446" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/52f73853-2574-4c85-87f4-c623790cf580">

#Figure 13: Running flask on http://127.0.0.1:5000

The Processor component is significant in connecting client questions with the fundamental archive corpus, giving an adaptable and user-friendly strategy for recovering data inside the system.
Code:
#crawler.py

<img width="462" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/56b4af13-90c8-4a0e-ad94-a77b82943400">

#Indexer.py

<img width="420" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/612fcc2b-539d-4c93-a7e0-298a65df0152">

#Processor.py

<img width="422" alt="image" src="https://github.com/MSP24SCM55D/Information-Retrieval-Project-A20549555/assets/164919823/b4adb945-ebb6-43a0-af11-06438f1af683">

# BIBILIOGRAPHY

1. Costola, M., Hinz, O., Nofer, M. and Pelizzon, L., 2023. Machine learning sentiment analysis, COVID-19 news and stock market reactions. Research in international business and finance, 64, p.101881.
2. Hossain, M.R., Hoque, M.M., Siddique, N. and Dewan, M.A.A., 2024. AraCovTexFinder: Leveraging the transformer-based language model for Arabic COVID-19 text identification. Engineering Applications of Artificial Intelligence, 133, p.107987.
3. Monterrubio, S.M.M., Naranjo, J.E.A., López, L.I.B. and Caraguay, Á.L.V., 2021, March. Black Widow Crawler for TOR network to search for criminal patterns. In 2021 Second International Conference on Information Systems and Software Technologies (ICI2ST) (pp. 108-113). IEEE.
4. Rahman, M., 2023. Query Search VS ChatAI:: The nature of users’ discourse of two search paradigms.
5. Urchs, S. and Kosch, M.G.P.D.H., 2020. Extracting definition and subsumption from german law (Doctoral dissertation, Master’s thesis).



