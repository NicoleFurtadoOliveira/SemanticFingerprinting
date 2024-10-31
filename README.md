## Semantic Fingerprinting

This app is an example of natural language processing (NLP). It uses TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to perform text analysis and comparison using the 20 Newsgroups dataset, providing the similarity between two selected documents based on their content. It can be interacted with through a Streamlit UI and ran containerized using Docker.


## No docker setup by creating a virtual environment
python3 -m venv SemanticFingerprinting
source SemanticFingerprinting/bin/activate
cd SemanticFingerprinting
pip install scikit-learn   
pip install streamlit
...
deactivate  # When done


## No docker run
streamlit run similarity_checker.py


## Docker build and run
docker build -t streamlit-similarity-app .
docker run -p 8501:8501 streamlit-similarity-app


## Example interpretation

As an example I compared two documents. Document 1 (index 19) belonging to the rec.motorcycles category and Document 2 (index 78) belonging to the soc.religion.christian:

Document 1 - rec.motorcycles (index 19) 

"ed 1. All of us that argue about gyroscopes, etc., throughly understand ed the technique of countersteering. me Including all the ones who think that they countersteer all the way me through a corner?? ed Well... all the way through a decreasing radius corner, anyway... Maybe they are riding around an ever decreasing circle of lies which eventually leads to the truth.... me The official line here though I do have my doubts about it is that the me front brake is applied first, followed by the rear brake, the idea being me that you avoid locking up the rear after weight transfer takes place. Me too, though unfortunately the Official Line is the one that you have to adhere to if you want to get a full licence. The examiner s guidelines are laid down by the government, and the basic rider education courses have no choice but to follow them. It surprises me that none of the rider groups here, either MAG or the BMF make much noise about the fact that the riding test requires you to ride three feet from the kerb all the time in order to pass, that the front brake must be applied before the rear, that you have to keep looking over your shoulder all the time instead of just when it is justified there s probably a few more too, which I can t think of for the moment. If the riding test could be rejigged a bit to include more of the real world survival skills and less of the woefully simplistic crap that it contains now, then the accident figures would imho reduce still further. Don t think we should include countersteering knowledge in our test though..."

Document 2 - soc.religion.christian (index 78)

"Subject pretty much says it all I m looking for Johnny Hart s creator of the B.C. comic stip mailing address. For those of you who haven t seen them, take a look at his strips for Good Friday and Easter Sunday. Remarkable witness! If anyone can help me get in touch with him, I d really appreciate it! I ve contacted the paper that carries his strip and they ll get back to me with it! Thanks for your help, Dave Arndt St. Peter s Evangelical Lutheran Church St. Peter, MN 56082"

The obtained similarity score was 0.0050 between these two documents which is extremely low, indicating that the content of these documents is almost entirely dissimilar. 

Cosine similarity measures the angle between two vectors. A score close to 1 means documents are highly similar, while a score near 0 (like 0.0050) suggests almost no similarity.

Document 1 discusses topics related to motorcycles, riding techniques, and regulations, whereas Document 2 is a message about contacting a comic strip artist associated with Christian themes. Since the subject matter and vocabulary differ significantly, it’s not surprising that the similarity score is low.

This low score suggests that these documents don’t share common themes, topics, or language patterns, which aligns with the distinct subjects in each document.


## Code explanation

This code uses the 20 Newsgroups dataset to compare documents and give users the similarity between two selected documents based on their content.

First, the code loads the documents and then applies some cleaning steps to make the text ready for analysis. The cleaning steps involve stripping away extra spaces, newlines, and unnecessary special characters. This is essential because cleaner text means more accurate results when we analyze the content. The cleaning is done in parallel, using all the available CPUs, and also caching is used to avoid re-loading and re-processing the data every time, which improves efficiency, speed and user experience.

To compare documents, we need to turn them into a format that a computer can easily understand. That’s where TF-IDF (Term Frequency-Inverse Document Frequency) comes in. TF-IDF is a method that converts each document into a numerical vector, emphasizing words that are unique to each document while downplaying words that are too common / stopwords (like "the" or "and"). This is good practice for document comparison because it gives more weight to the words that define a document's unique content.

Once we have each document as a TF-IDF vector, we can calculate their similarity. Cosine similarity measures the angle between two vectors—essentially telling us if they’re pointing in the same direction (indicating similarity). A score close to 1 means the documents are quite similar, while scores near 0 indicate they’re very different. This code focuses on absolute cosine similarity, meaning all scores will be between 0 and 1, which makes it easier for users to interpret.

Streamlit was used to create a simple, interactive interface where users can select two documents and see how similar they are. After choosing a category and document within it, they click a button to see the similarity score and read an excerpt from each document, giving context to the score. This is a great practice in user experience design because it allows users to easily interact with the model and understand the results without diving into the code.

The integer values in labels are indexes into the categories list. For example, if labels[i] is 0, then the i-th document belongs to the category categories[0], which is alt.atheism. labels[1] is 3, so the second document belongs to categories[3], which is comp.os.ms-windows.misc. This is how we know which documents belong to which category of these:
categories = [
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
    'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
    'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 
    'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 
    'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    

## Theory explanation 

TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity are two key techniques used in text analysis, particularly in information retrieval and natural language processing (NLP). 

In the semantic fingerprinting context, TF-IDF creates a vector for each document that emphasizes unique words within the context of the corpus and Cosine Similarity then compares these vectors, allowing to measure the semantic similarity between documents.


1. TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a statistical measure that evaluates how relevant a word is to a document within a larger collection (or corpus). This metric helps identify the most important words in a document by assigning weights based on two factors:

Term Frequency (TF): This measures how frequently a term appears in a document. If a word appears frequently in a document, it is likely important to that document. TF is typically calculated as:

TF = Number of times term t appears in a document / Total number of terms in the document

Inverse Document Frequency (IDF): This measures how common or rare a term is across all documents in the corpus. Words that appear in many documents, like common stop words ("the," "is," etc.), receive a lower weight, as they are less unique or meaningful. IDF is typically calculated as:

IDF = log⁡(Total number of documents / Number of documents containing term t)

The TF-IDF score is then calculated as the product of TF and IDF, assigning higher values to words that are more frequent in a single document but less common across the corpus.

TF-IDF is particularly useful for feature extraction in machine learning, as it captures the relevance of terms while filtering out common, less meaningful words.


2. Cosine Similarity

Cosine similarity is a measure that calculates the similarity between two non-zero vectors. In text analysis, these vectors often represent TF-IDF values for terms within documents. Cosine similarity evaluates the cosine of the angle between two vectors; the closer the angle is to zero, the more similar the vectors (and hence the documents).

Mathematically, cosine similarity is defined as:

Cosine Similarity = cos⁡(θ) = vA⋅vB / ( ∣∣vA∣∣ × ∣∣vB∣∣ )

​where:

vA⋅vB is the dot product of vectors A and B and ∣∣vA∣∣ and ∣∣vB∣∣ are the magnitudes (or norms) of A and B.

Cosine similarity ranges from -1 (completely opposite) to 1 (identical), with 0 indicating no correlation. In text analysis, cosine similarity is commonly used to compare document vectors (often derived using TF-IDF) to assess how similar the content is. This is useful in applications like document clustering, plagiarism detection, and recommendation systems. In most practical applications for document similarity, the cosine similarity is constrained between 0 and 1 by focusing on the absolute angle between vectors, meaning values are usually non-negative.