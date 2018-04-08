# CKME136-Final-Code
This repository include Final code of my project. There are two files in this repository, one is for text categorization and the other is sentiment analysis. 
title: "Project Text Categorization"
output:
  html_document: default
  word_document: default
---

This Text categorization method uses 9 different analytical techniques and they are SVM(Support Vector Machines), GLMNET(Generalized Linear Model), MAXENT(Maximum Entropy), SLDA, Bagging, Boosting, RF(Random Forest), NNET(Neural Networks), TREE(Classification Tree).   

Load Required Libraries
```{r}
library(stringi)
library(qdapDictionaries)
library(caTools)
library(SparseM)
library(RTextTools)
```

Load Data from csv file
```{r}
My_data<-read.csv(file="C:/Users/Jayan/Documents/Ryerson - Big Data, Analytics, Predictive Analytics/CKME 136 - Capstone Project/Programming Part/My Data.csv", header=T, sep=",", na.strings=c("","NA"))
```

The Dataset is cleaned by first removing empty records then removing records which are non-english and gibberish
```{r}
My_data<-My_data[complete.cases(My_data),]
is.word  <- function(x) x %in% GradyAugmented    
My_data$Response<-tolower(My_data$Response)
split_word<-stri_extract_all_words(My_data$Response, simplify=TRUE)  
split_word[split_word==""]<-NA    
nonengrownums<-which(apply(split_word, 1, function(x) sum(is.word(x)/sum(!is.na(x))))<0.75)   
My_data<-My_data[-nonengrownums,]  
```

Randomly shuffle the dataset and making the Action class in the dataset a numeric type
```{r}
My_data<-My_data[sample(nrow(My_data)),]
My_data$Action<-as.numeric(as.factor(My_data$Action))
```

Create the document term matrix
```{r}
doc_matrix <- create_matrix(My_data$Response, language="english", removeNumbers=TRUE,stemWords=TRUE, removePunctuation=TRUE, toLower=TRUE, removeSparseTerms=.998)
```

Creating the container with 80% Train data and 20% Test data
```{r}
train_size<-round(0.8*(nrow(My_data)), digits=0)
container <- create_container(doc_matrix, My_data$Action, trainSize=1:train_size, testSize=(train_size+1):(nrow(My_data)),virgin=FALSE)
```

Train and classify the model using the allocated train and test sets
```{r}
SVM <- train_model(container,"SVM")
GLMNET <- train_model(container,"GLMNET")
MAXENT <- train_model(container,"MAXENT")
SLDA <- train_model(container,"SLDA")
BOOSTING <- train_model(container,"BOOSTING")
BAGGING <- train_model(container,"BAGGING")
RF <- train_model(container,"RF")
NNET <- train_model(container,"NNET")
TREE <- train_model(container,"TREE")

SVM_CLASSIFY <- classify_model(container, SVM)
GLMNET_CLASSIFY <- classify_model(container, GLMNET)
MAXENT_CLASSIFY <- classify_model(container, MAXENT)
SLDA_CLASSIFY <- classify_model(container, SLDA)
BOOSTING_CLASSIFY <- classify_model(container, BOOSTING)
BAGGING_CLASSIFY <- classify_model(container, BAGGING)
RF_CLASSIFY <- classify_model(container, RF)
NNET_CLASSIFY <- classify_model(container, NNET)
TREE_CLASSIFY <- classify_model(container, TREE)
```

Create the analytics summary data
```{r}
analytics <- create_analytics(container,
                              cbind(SVM_CLASSIFY, SLDA_CLASSIFY,
                                    BOOSTING_CLASSIFY, BAGGING_CLASSIFY,
                                    RF_CLASSIFY, GLMNET_CLASSIFY,
                                    NNET_CLASSIFY, TREE_CLASSIFY,
                                    MAXENT_CLASSIFY))
summary(analytics)
```
