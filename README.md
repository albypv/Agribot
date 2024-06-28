# Agribot

**Problem Statement**:
	Farmers often lack timely, reliable information on best agricultural practices, pest control, weather forecasts, and market prices, leading to suboptimal yields and financial instability. Researchers and policymakers need centralized, up-to-date data to support their work. Additionally, consumers seek transparency about food origins and sustainable farming practices. This agricultural bot aims to address these needs by providing accurate, real-time information to farmers, curated research for academics, policy insights for organizations, and educational content for consumers, ultimately fostering informed decisions and sustainable agricultural practices.
	
**Solution**:
	The proposed agricultural bot will leverage artificial intelligence and machine learning to deliver personalized, real-time information and recommendations to users. For farmers, the bot will provide timely updates on weather conditions, pest control methods, and market prices, enhancing decision-making and productivity. Researchers and academics will benefit from curated research articles, data analysis, and trend reports. Policymakers will receive data-driven insights and policy recommendations to inform their strategies. Additionally, the bot will educate consumers on sustainable farming practices and food origins, promoting transparency and informed choices. This comprehensive approach will support the agricultural community, driving sustainable practices and improved outcomes.
	
**Limitations**:
	The agricultural bot, while proficient in delivering domain-specific information, exhibits several limitations:
		• Lack of Conversational Flexibility: The bot struggles with handling general greetings and basic conversational queries due to its training focus being predominantly on agricultural content.
		• Limited Training Data: The bot's inability to respond effectively to untrained data highlights a gap in its training set, which has not encompassed sufficient examples of everyday language and general inquiries.
		• Scalability Issues: The bot's current framework may not easily scale to accommodate a wider range of topics beyond agriculture without significant retraining and data augmentation efforts.
		• Contextual Understanding: The bot often fails to maintain context in multi-turn conversations, especially when switching between agricultural and non-agricultural topics, leading to disjointed and less coherent interactions.
		
**Conclusion**:
  The development of the agricultural bot marks a significant step towards providing targeted, real-time information and support to farmers, researchers, policymakers, and consumers. While the bot excels in delivering domain-specific knowledge and recommendations, its limitations in handling general conversational queries highlight areas for further improvement. Enhancing the bot's conversational flexibility and contextual understanding will be crucial in creating a more engaging and user-friendly experience. Future iterations should focus on broadening the training data to include everyday language and refining natural language processing capabilities. With these advancements, the bot can become a comprehensive tool, driving informed decision-making and sustainable practices in the agricultural sector.

**____________________________________________________________________________________________________________________________________________________**

**Link: LLM Model**
https://colab.research.google.com/drive/1QZpAyBN-4Vd2zyxaYIYegzLGY9Vyjdre?usp=sharing

**Link: RAG**
https://colab.research.google.com/drive/1K1Y3nslteoB8IY5hy3ir9I567GQvSEe7?usp=sharing

**Limitations [RAG Model]**
	The agricultural bot, developed using a large language model (LLM), effectively provides domain-specific information but struggles with handling general conversational queries, such as greetings. To address this, we explored the integration of Retrieval-Augmented Generation (RAG) without significant improvement.

 **Conclusion [RAG Model]**
 	This model enhances response accuracy by retrieving relevant information from a large knowledge base or document repository, ensuring up-to-date and precise answers. It excels in handling rare or specific queries that may not be well-covered by traditional training data alone, thereby expanding the bot's capability to address diverse user inquiries. Moreover, RAG's scalability allows for seamless integration of extensive external data sources, supporting broader and more comprehensive responses. Ultimately, RAG enables the generation of contextually rich answers, improving the overall user experience in interactive applications.

**____________________________________________________________________________________________________________________________________________________**

**NOTES**

6 layers in gpt

Self attention helps to find the important words in a sentence

input/prompt → model which is trained on a large dataset →
output/response

Self attention parameters : QKV —\> Query Key Value

Self attention = Softmax(QK<sup>T</sup> / sqrt(d<sub>k</sub>)) V

V: In- depth features

K: Response

Q: Question

d<sub>k</sub> : Dimensionality of the matrix

Softmax : we get the probability distribution

*<u>Architecture of Transformer</u>*

All the components are separate blocks which work together connecting
one output to the other input to finally give the output of the
transformer.

Input embedding -—-\>Positional Embedding —--\> Self Attention -—-\>Feed
forward —\> Normalization —-\>Output

*<u>Input Embedding</u>*

→Splits the individual words into different components, ie
*tokenization*.

*<u>Position Embedding</u>*

→Word positions are encoded in the sentence

*<u>Self Attention</u>*

→Used to find the relationship between different elements of the input
sequence and to find in depth features of the input sequence

*<u>Feed Forward</u>*

→Provides in depth feature extraction

*<u>Normalization</u>*

→Prove solutions for errors and other issues that might occur with the
model

Eg: Gradient explosion, Errors etc

—---------------------------------------------------------------------------------------------------------

*<u>Training</u>*

→Means you are inputting your data to a pre existing model and then make
the model do well on your input data

*<u>Fine Tuning</u>*

Eg: The quick brown fox jumps over the lazy dog.

*Layer 1* (Input Encoding) : It tokenizes the sentence and places it
as tokens.

Converts each word into a 4d matrix

*Layer 2* (Positional Encoding) :

PE<sub>(pos,2i)</sub>= sin(pos/10,000 <sup>2i/dmodel</sup>)

PE<sub>(pos,2i+1)</sub> = cos(pos/10,000 <sup>2i/dmodel</sup>)

*Layer 3* (Self Attention) : Comparing itself(Each word) with other
words to get more features.

Self attention = Softmax(QK<sup>T</sup> / sqrt(d<sub>k</sub>)) V

Q=XW<sub>Q</sub>

K=XW<sub>K</sub>

V=XW<sub>V</sub>

For example, we are querying on “quick”, K compares the relevance of the
words with each other and how much importance can be given to the query.
In this particular example, it shows how much the input is relevant to
the key.

V is the value vector which gives the aggregation score, i.e. it is an
aggregator vector which is highly dependent on Q. We obtain it during
training and it represents the in-depth and detailed features of the
data.

*Layer 4* (Feed forward) : Max(0,x) i.e. RELU function

FF<sub>N</sub>= max (0,xW<sub>1</sub>+ b<sub>1</sub>) W<sub>2</sub> +
b<sub>2</sub>

It has MLP layers, i.e. multilayer perceptron layers.

W : weight matrices

b : biases

FFN gives the high dimensional representation which is very hard to
visualize. This layer is responsible for artificial generative AI, AI
with higher intelligence than humans etc etc

*Layer 5* (Normalization) : mapping the outputs to sensible values in
the output domain.

Layer normalization(x) = ($\frac{x\\ - \\\mu}{\sigma}$) γ + β

X: input(h)

μ: mean of input

σ: standard deviation

Gamma beta are learned parameters

*Layer 6* (Multihead attention) : Part of the decoder architecture.
After this , we perform FF<sub>N</sub>, normalization etc

Multihead(Q,K,V) = concat(head<sub>1</sub> , head<sub>2</sub> …..
head<sub>i</sub> )

head<sub>i</sub> = attention(Q W<sub>i</sub><sup>Q</sup> , K
W<sub>i</sub><sup>K</sup> , V W<sub>i</sub><sup>V</sup> )

These heads are used to perform more feature extraction

W<sub>i</sub> are the weight matrices, multiplied with the query , key
and value matrices to perform many attention steps.

*Layer 7* (Feedforward layer) :

Gate —\> Up and down

Gate → High dimensional projection

Up → Deeper high dimensional projection

Down → Back to embedded projection

*Layer 8* (Normalization)

*Layer 9* (Output)

<u>\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\</u>

*<u>Summary</u>*

- Input is tokenized and split into a matrix with all the different
  > words in the query.

- Then a token ID is assigned to each of the tokens.

- This token ID is mapped to embedding vectors wherein each token is
  > converted into matrices of the specified dimensions.

- We get output logits(numbers) which are the raw unnormalized
  > predictions which are generated by a model before applying any
  > activation function.

- Then the softmax of logits give the probabilities which are then used
  > to generate the solution based off of token id’s. This is called
  > token selection wherein each token is monitored and the token with
  > the highest probability is selected and put into the output.

KV cache is a method to reduce the computational cost by storing and
retrieving previously computed data thus enabling the model to not
recompute previously processed data .
