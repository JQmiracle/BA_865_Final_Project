# BA_865_Final_Project

Predicting Supreme Court Decisions


## Introduction
* **Case background**: The Supreme court is the highest tribunal for all cases and interpretation of the Constitution or the laws in the United States. Supreme court decisions impact parties in each case, stakeholders, government and society. Supreme court decisions regulate individuals' life, rights and obligations. Therefore, predicting supreme court decision is critical that it helps stakeholder decision making.

* **Problem statement**: This project aims to predict whether petitioner will win or respondent will win in each case using multiple neural network models. This is a binary classification problem given winner index, party names, and case facts. Winner index indicates whether petitioner won or respondent won.

* **Dataset**: The source of this data is the Oyez project. Oyez project is a free law project from Cornell’s Legal Information Institute, Justia, and Chicago-Kent College of Law to archive Supreme Court data. We used dataset(task1_data.pkl) gathered by Mohammed Alsayed et all, in https://github.com/smitp415/CSCI_544_Final_Project.git

## Data Preprocessing
Address the imbalanced dataset by upsampling minor class (winner index=1) using Sklearn resample functions. Since only partial case facts include party names, we decided to merge 'facts', 'first_party', and 'second_party' to preserve party information.
  - 13.05% of facts don't contain the first party name
  - 17.18% of facts don't contain  the second party name
  - 1.93% of facts don't contain both first party the second party names


## Modeling
Applied advanced NLP Algorithm (GloVe, Word2Vec, spaCy) to analyze supreme court cases, constructed deep neural
networks using 1D CNN, LSTM and Textual Embeddings to predict a court's judgment given the case's facts, increased the model accuracy from 0.66 to 0.92


<img width="833" alt="Screen Shot 2022-06-14 at 2 47 05 PM" src="https://user-images.githubusercontent.com/87022634/173666024-81f27594-4b3d-4543-81ec-7c514d22a3ac.png">


## Model Selection & Interpretation

* **Best model**: Dense layer with text-vectorizatoin(bigram, TD-IDF) performed best(AUC) among our models
  * **Sigmoid/Binary-crossentrophy**: Since our prediction problem was binary classification, we used sigmoid output activation function that it returns values between 0 and 1, which can be treated as probabilities of a data point belonging to binary class. Likewise, we used binary-crossentrophy as loss function.
  * **Test accuracy/AUC**: We measured test accuracy for each model. To choose best model, we generated AUC.

* **LIME**: We used LIME to explain our model and to see what words in text contributed to the prediction


<img width="853" alt="Screen Shot 2022-06-14 at 2 48 33 PM" src="https://user-images.githubusercontent.com/87022634/173666255-66a8a3be-4bd0-4250-bbea-4dc0a864a3ec.png">



## Limitation and Future Steps

* **Cross-validation with upsampled data**: For better measurement, we could have done upsampling manually in each cross validation folds. However, since our goal was exploring multiple NN models, upsampling in each folds hurted runtime efficiency and code-reuse. We decided to upsample train set first. As we kept test set aside, we obtained a valid measure of model performance on test set.

* **Domain specific pretrained model**: We could further work using domain specific pretrained model. We found https://github.com/ashkonf/LeGloVe, which is python implementation of GloVe word vectors for legal domain-specific corpuses.

* **Gather more features**: In Oyez database, we could find more information such as advocate, location, lower court and date. Gathering this information as new features might be able to improve our model performance.
