# Machine Learning Projects


## NHL Chatbot

This is an implementation of the notebook in NHL Draft Predictions. It takes the models trained in the notebook and creates a retrieval-based chatbot that can make draft pick predictions and retrieve NHL player stats. To understand user intent, I built an algorithm that converts user input to Word2Vec embeddings and compares the similarity to a predefined set of responses. This allows the chatbot to move down either branch of the decision logic to make draft pick predictions or retrieve stats. 

This project was presented at the CUCAI 2023 design showcase to AI industry leaders. Check out my LinkedIn for a demo of the full-stack development my team and I built to implement this code


https://user-images.githubusercontent.com/118325725/230187248-d8dce823-fafd-4532-9974-e955e3744bf2.mp4


Academic Paper: [Predicting the NHL Draft - CUCAI Paper.pdf](https://github.com/eobrien2002/ML/files/11162484/Predicting.the.NHL.Draft.-.CUCAI.Paper.pdf)



## DataQuests 2023

This project was for the 2023 DataQuest Hackathon as part of the Western AI Club. The notebook walks through exploratory data analysis, feature engineering, feature reduction with PCA, and various ML models. Logistic Regression, K-Neighest Neighbours, Support Vector Machines, Decision Trees, Random Forest, XG Boost, MLP, CNN, and RNN models were tested. These base models were improved with ensemble methods such as bagging and AdaBoost and tuned with a Gird Search algorithm. The Neural Networks were tuned using a Hyperband algorithm.


## NHL Draft predictions

This notebook aims to predict the draft round of an amateur player based on amateur statistics. I built the dataset using the NHL API to get past draft picks and then scraped their amateur statistics from the website Elite Hockey Prospects. Throughout the notebook, I clean the data, perform exploratory data analysis, feature engineering, and feature reduction with PCA. I then used Sequential Forward Selection, sequential backward selection, and Recursive Feature selection algorithms to determine the features that improved model performance.

I tested base models including Logistic Regression, K-Neighest Neighbours, Support Vector Machines, Decision Trees, Random Forest, XG Boost, MLP, CNN, and RNN models, and improved them with ensemble methods such as bagging and AdaBoost and tuned with a Gird Search algorithm. The Neural Networks were tuned using a Hyperband algorithm.

