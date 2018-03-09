install.packages("dplyr")
install.packages("data.table")
install.packages("dtplyr")
install.packages("topicmodels")
install.packages("tidytext")
install.packages("ggplot2")
install.packages("randomForest")
install.packages("tm")

# LIST OF PACKAGES TO BE USED
library(dplyr)
library(data.table)
library(dtplyr)
library(topicmodels)
library(tidytext)
library(ggplot2)
library(randomForest)
library(tm)
library(ROCR)
library(rattle)

############################################################################################################################
#Reading the data

##Train data
df_train <- fread("train.csv",nrows = 17591)
str(df_train)
df_train$id<-as.numeric(df_train$id)
df_train$qid1<-as.numeric(df_train$qid1)
df_train$qid2<-as.numeric(df_train$qid2)
df_train$is_duplicate<-as.factor(df_train$is_duplicate)

head(df_train, 3)

##Test data
df_test <- fread("test.csv",nrows = 29473)
colnames(df_test)[1] <- c("id")
df_test$id<-as.numeric(df$id)

head(df_test, 3)
df_test_1 <- df_test
############################################################################################################################
#Cleaning the data

# lowercase conversion, removing html/http/image links, and then the STOPWORDS 
cleanup <- function(x){
  x <- tolower(x)
  x <- gsub("<img src.*?>", "", x)
  x <- gsub("[^a-zA-Z0-9 ]", "", x)
  x <- gsub("http\\S+", "", x)
  x <- gsub("\\[math\\]", "", x)    # text between [] refers to tags e.g. [math]
  x <- gsub("<.*?>", "", x)
  x <- gsub("\n", " ", x)                 # replace newline with a space
  x <- gsub("\\s+", " ", x)                # multiple spaces into one
  # using tm_map to remove stopwords
  docs <- Corpus(VectorSource(x))
  docs <- tm_map(docs, removeWords, stopwords('en'))
  docs <- tm_map(docs, removePunctuation)    # dont remove punct so early in the analysis
  docs <- tm_map(docs, stripWhitespace)
  xxx <- sapply(docs, function(i) i)
  data_content <- data.frame(text = xxx, stringsAsFactors = FALSE)
  return(data_content$text)
}


#Train Data
df_train$question1 <- cleanup(df_train$question1)
head(df_train,3)

df_train$question2 <- cleanup(df_train$question2)
head(df_train,3)

#Test data
df_test$question1 <- cleanup(df_test$question1)
head(df_test,3)

df_test$question2 <- cleanup(df_test$question2)
head(df_test,3)
################################################################################################################################
set_data <- function(df,is_train)
{
	### below code shall create tokens(words) per question - 

	### baseline model  ####
	### Cosine similarity
	### common words
	### diff between nchar of q2, q1

	## for question1
	tokens_q1 <- df %>%
				unnest_tokens(word, question1, drop = FALSE, token = "regex", pattern = " ") %>%
				count(id, word) %>%
				ungroup()
	colnames(tokens_q1)[1:3] <- c("id1", "word1", "n1")

	## for question2
	tokens_q2 <- df %>%
				unnest_tokens(word, question2, drop = FALSE, token = "regex", pattern = " ") %>%
				count(id, word) %>%
				ungroup()
	colnames(tokens_q2)[1:3] <- c("id2", "word2", "n2")
		
	#Cosine Similarity

	#create data frame to find the cosine similarity
	words_q1_q2<-merge(tokens_q1,tokens_q2,by.x=c("id1","word1"),by.y=c("id2","word2"),all=TRUE)
	words_q1_q2[is.na(words_q1_q2)]<-0
	colnames(words_q1_q2)[1:2] <- c("id", "word")

	list_total<-words_q1_q2%>%
				group_by(id)%>%
				summarise(total=n())

	list_com<-words_q1_q2%>%
				group_by(id)%>%
				filter(n1!=0 & n2!=0)%>%
				summarise(common=n())

	list_uncommon<-words_q1_q2%>%
				group_by(id)%>%
				filter((n1==0 & n2!=0)|(n1!=0 & n2==0)|(n1==0 & n2==0))%>%
				summarise(uncommon=n())

	list_n2_n1<-words_q1_q2%>%
				group_by(id)%>%
				summarise(diff_words=abs(sum(n1)-sum(n2)))

	list_word_1<-merge(list_total,list_com,by="id",all=TRUE)
	list_word_count<-merge(list_word_1,list_n2_n1,by="id",all=TRUE)

	list_word_count[is.na(list_word_count)]<-0

	list_word_count<-list_word_count%>%
				mutate(common_ratio=(common/total))

	temp1<- aggregate(n1~id,data=words_q1_q2,c)
	temp2<- aggregate(n2~id,data=words_q1_q2,c)

	for(i in 1:nrow(temp1)) {
	  a = unlist(temp1$n1[i])
	  b = unlist(temp2$n2[i])
	  temp1$cosine_sim[i] = crossprod(a, b)/sqrt(crossprod(a) * crossprod(b))
	}
	temp1<-temp1[,c(1,3)]
	
	if(is_train==1)
	{
		target_variable<-df[,c(1,6)]
	}
	else
	{
		target_variable<-df[,c(1)]
	}
	model_input_temp<-merge(list_word_count,temp1,by="id")
	model_input<-merge(model_input_temp,target_variable,by="id")

	model_input[is.na(model_input)]<-0
	
	return(model_input)
	
}
model_train_input<-set_data(df_train,1)
model_test<-set_data(df_test,0)

set.seed(432)
s<-sample(1:nrow(model_train_input),0.7*nrow(model_train_input),replace = FALSE)
model_train<- model_train_input[s,]
model_valid<- model_train_input[-s,]


################################################################################################################################
##Random Forest
RF <- randomForest(is_duplicate ~ diff_words+common_ratio+cosine_sim, data = model_train, ntree = 1000, mtry = 3)

predictTrainRF <- predict(RF,type="prob")[,2]
ROCRpredTrainRF = prediction(predictTrainRF, model_train$is_duplicate)
perf = performance(ROCRpredTrainRF, "tpr", "fpr")
plot(perf)
as.numeric(performance(ROCRpredTrainRF, "auc")@y.values)

predictValidRF <- predict(RF,newdata = model_valid,type="prob")[,2]
ROCRpredValidRF = prediction(predictValidRF, model_valid$is_duplicate)
perf = performance(ROCRpredValidRF, "tpr", "fpr")
plot(perf)
as.numeric(performance(ROCRpredValidRF, "auc")@y.values)
predictValidRF <- ifelse(predictValidRF>0.5,1,0)
model_valid <- cbind(model_valid, predictValidRF)
model_valid$predictValidRF <- as.factor(model_valid$predictValidRF)

write.csv(model_valid,file = "final_validation.csv")

predictTestRF <- predict(RF,newdata = model_test,type="prob")[,2]
predictTestRF <- ifelse(predictTestRF>0.5,1,0)
model_test <- cbind(model_test,predictTestRF)
model_test_1 <- model_test[,c(1,7)]
final_test <- merge(df_test_1,model_test_1,by="id")

write.csv(final_test,file = "final_test.csv",row.names = FALSE)

