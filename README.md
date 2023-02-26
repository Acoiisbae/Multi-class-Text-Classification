# Multi-class Text Classification
 
 To categorize unseen  articles into 5 categories namely Sport, Tech, Business, Entertainment and  Politics.
 
# Project Description
Text documents are essential as they are one of the richest sources of data for businesses. Text documents often contain crucial information which might shape the market trends or influence the investment flows. Therefore, companies often hire analysts to monitor the trend via articles posted online, tweets on social media platforms such as Twitter or articles from newspaper. However, some companies may wish to only focus on articles related to technologies and politics. Thus, filtering of the articles into different categories is required.

Often the categorization of the articles is conduced manually and retrospectively; thus, causing the waste of time and resources due to this arduous task. Hence, the purpose of this project is to categorize unseen articles into 5 categories namely Sport, Tech, Business, Entertainment and Politics by using LSTM model and get the best accuracy.

Project Details
* Data Loading

Load the dataset into this project using pandas by simply passing the URL dataset into pd.read_csv.
* Data Inspection

Check the data type of all columns
Check if the dataset has duplicates data - found 99 duplicates
Check if the dataset has any missing values - 0 missing values found

* Data Cleaning

Define the features x and target y
Features - text
Target - category
Remove unimportant character in the text column
Remove all duplicates data

* Features Selection
Select column 'text' and 'category' as features and target

* Data Pre-Processing

Change the text into numbers using Tokenization for Feature x(separate piece of texts into smaller units called tokens)
Padding and Truncating
Splitting the train and test data - X_train, X_test, y_train, y_test
Pre-processing the Target y using One-Hot Encoder

* Model Development

Create Sequential Model
Add Embedding layer
Add Bidirectional LSTM layer
Add LSTM layer
Add Dropout layers
Add Dense layers
Model Summary
Model Architecture
My model architecture used in this project are as follows where I used Bidirectional, LSTM and Dropout as my hidden layers and Dense as my output layer with 64 nodes for each layer.

![architecture model](https://user-images.githubusercontent.com/125865422/220282347-c80d2b04-839b-4759-a1cb-e23ed66b8bf2.png)

* Model Compilation

Compile the model
Optimizer - adam
Loss - 'categorical_crossentropy'
Metrics - 'acc'

* Callbacks - Early Stopping and TensorBoard

Tensorboard logs after every batch of training to monitor metrics
Save model to disk

* Model Training

Train the model for 10 epochs and get the model accuracy

* Model Evaluation

Get the model prediction
Evaluate the model confusion matrix
Evaluate the model classification report
Evaluate the model accuracy score
Model Performance
My model performance that I get from this project are as follows where I used Confusion Matrix, Classification Report, and Accuracy Score to evaluate the performance. This value can still be improved by adjusting the hidden layers in Model Development.

