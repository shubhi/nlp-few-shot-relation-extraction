# Few-Shot Relation Extraction

## Introduction
Humans are capable of recognizing new object classes from just a few examples, unlike traditional machine learning techniques which typically require large datasets to achieve comparable results. Recent efforts in computer vision to address few-shot learning tasks aim to overcome the challenges of data scarcity without sacrificing model performance. However, the application of Few-Shot Learning (FSL) in NLP is still largely uncharted, with only a few datasets available.

## FewRel Dataset
The Few-Shot Relation Classification Dataset (FewRel) consists of 70,000 sentences on 100 relations derived from Wikipedia, where each relation has approximately 700 sentences. Relation extraction in NLP involves identifying semantic relationships between entities within a text.

### Example
- **Sentence:** "London is the capital of the U.K."
- **Relation:** `capital_of`

## Dataset Features
The dataset is stored as a JSON and includes features such as tokens, head entities, tail entities, and their positions. These features are crucial for masking and positional encodings in the model.

### Pre-processing Steps
1. Convert tokens to lowercase to reduce vocabulary set.
2. Create word vectors using pre-trained embeddings like BERT or GloVe.
3. Map each word to an ID and handle unknown words with a special "UNK" vector.
4. Map each instance to its relation ID, defining the start and end indices in the processed data arrays.
5. Create masks using the positional encodings of the head and tail entities.

## Models and Evaluation
We will test the models in various few-shot settings, such as 5-way 1-shot and 10-way 5-shot. The evaluation metric will be accuracy.

### Models
- **k-NNs:** Used as a baseline with pre-trained word embeddings.
- **Prototypical Networks:** Classify new instances based on their distance to prototype representations of each class.
- **MAML:** Adapt quickly to new tasks with minimal training data by optimizing initial model parameters for fast learning.

## References
- Achiam, J., et al. (2023). GPT-4 technical report.
- Han, X., et al. (2018). FewRel: A large-scale supervised few-shot relation classification dataset.
- Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning.
- Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks.

## Collaborators
- Jacob
- Shubhi
- Pavan
- Dev
