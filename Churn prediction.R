### Churn Prediction ###
churn_train = read.csv("C:/Users/KARISHMA/OneDrive/Desktop/DS projects/Vidhya hackton projects/Churn Prediction/train_PDjVQMB.csv", stringsAsFactors = FALSE)
churn_test = read.csv("C:/Users/KARISHMA/OneDrive/Desktop/DS projects/Vidhya hackton projects/Churn Prediction/test_lTY72QC.csv", stringsAsFactors = FALSE)

library(dplyr)
glimpse(churn_train)

head(churn_train)
dim(churn_train)
dim(churn_test)
length(unique(churn_train$Income))
length(unique(churn_train$Balance))

## Diffrence of data ##
setdiff(names(churn_train), names(churn_test))
churn_test$Is_Churn=NA
churn_train$data='train'
churn_test$data='test'

### Binding a data ##

churn = rbind(churn_train, churn_test)
dim(churn)

glimpse(churn)

## NA covert to 0 ###
churn$Product_Holdings[is.na(churn$Product_Holdings)]= 1

## char convert to int 
churn$Product_Holdings=as.numeric(churn$Product_Holdings)

table(churn$Product_Holdings)
glimpse(churn)


## now create a dummies of income and caredit_category
table(churn$Credit_Category)

##install.packages("fastDummies", "recipes")
library(fastDummies)

churn= dummy_cols(churn, select_columns = 'Credit_Category') 
head(churn)

churn= dummy_cols(churn, select_columns = 'Income')
churn= dummy_cols(churn, select_columns = 'Gender')

#churn$Credit_Category_Poor[is.na(churn$Credit_Category_Poor)] <- mean(churn$Credit_Category_Poor, na.rm=TRUE)
#churn$`Income_More than 15L`[is.na(churn$`Income_More than 15L`)] <- mean(churn$`Income_More than 15L`, na.rm=TRUE)
#churn$Gender_Male[is.na(churn$Gender_Male)] <- mean(churn$Gender_Male, na.rm=TRUE)


dim(churn)

## drop some col 
churn$Income= NULL
churn$Credit_Category= NULL
churn$Gender= NULL
churn$ID = NULL
churn$Credit_Category_Poor= NULL
churn$`Income_More than 15L` = NULL
churn$Gender_Male = NULL

glimpse(churn)

sum(is.na(churn$Age))
sum(is.na(churn$Balance))
sum(is.na(churn$Vintage))
sum(is.na(churn$Transaction_Status))
sum(is.na(churn$Product_Holdings))
sum(is.na(churn$Credit_Card))
sum(is.na(churn_test$Is_Churn))

    
#### no. of Na values------------------------------------------
sum(apply(churn, 2, is.na))

### speration of train and test data------------------------------
churn_train = churn %>%filter(data=='train')%>%select(-data )
churn_test = churn %>% filter(data=='test')%>%select(-data,-Is_Churn )

head(churn_train)
head(churn_test)

##----------------------------------------------------------------------

set.seed(2)
s=sample(1:nrow(churn_train),0.8*nrow(churn_train))
churn_train1=churn_train[s,]
churn_train2=churn_train[-s,]

dim(churn_train2)

library(car)

for_vif=lm(Is_Churn ~ Age + Balance + Vintage + Transaction_Status + Product_Holdings + 
             Credit_Card + Credit_Category_Average + Credit_Category_Good + 
             `Income_5L - 10L` + `Income_10L - 15L` + `Income_Less than 5L` + 
             Gender_Female,data=churn_train1)

sort(vif(for_vif),decreasing = T)[1:9]
formula(for_vif)


log_fit <- glm(Is_Churn ~ Age + Balance + Vintage + Transaction_Status + Product_Holdings + 
                 Credit_Card + Credit_Category_Average + Credit_Category_Good + 
                 `Income_5L - 10L` + `Income_10L - 15L` + `Income_Less than 5L` + 
                 Gender_Female,data=churn_train1, family = "binomial")

log_fit=step(log_fit)

head(churn_train1)
formula(log_fit)


summary(log_fit)

library(pROC)
train1.score = predict(log_fit, newdata = churn_train1, type= 'response')
train2.score = predict(log_fit, newdata = churn_train2, type= 'response')


auc(roc(churn_train1$Is_Churn, train1.score))
auc(roc(churn_train2$Is_Churn, train2.score))

#caTools::colAUC(train1.score, churn_train1$Is_Churn, plotROC = TRUE)
#caTools::colAUC(train2.score, churn_train2$Is_Churn, plotROC = TRUE)

train1.score = predict(log_fit,newdata = churn_train,type='response')
train1.score

real=churn_train$Is_Churn ## Positives
cutoffs=seq(0.001,0.999,0.001)
length(cutoffs)
cutoffs[1]

cutoff_data=data.frame(cutoff=99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)




for(cutoff in cutoffs){
  print(paste0('cutoff is: ' , cutoff))
  predicted=as.numeric(train1.score>cutoff) ## call it 1
  
  ## for every cut off there is 1 / 0 generated for each row
  
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  
  P=TP+FN
  N=TN+FP
  
  Sn=TP/P ## TP/p
  Sp=TN/N ## TN/N
  precision=TP/(TP+FP)
  recall=Sn
  ## Ks = tp/p - fp/n
  KS=(TP/P)-(FP/N)
  F5=(26*precision*recall)/((25*precision)+recall)
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  M=(4*FP+FN)/(5*(P+N))
  
  cutoff_data=rbind(cutoff_data,c(cutoff,Sn,Sp,KS,F5,F.1,M))
}

cutoff_data=cutoff_data[-1,]
View(cutoff_data)


#### visualise how these measures move across cutoffs
library(ggplot2)
ggplot(cutoff_data,aes(x=cutoff,y=Sp))+geom_line()


library(tidyr)

head(cutoff_long)

cutoff_long[cutoff_long$cutoff==.001,]

cutoff_long=cutoff_data %>% 
  gather(Measure,Value,Sn:M)

cutoff_long_small <- cutoff_long[cutoff_long$Measure %in% c('Sn', 'Sp'),]

ggplot(cutoff_long_small,aes(x=cutoff,y=Value,color=Measure))+geom_line()

my_cutoff=cutoff_data$cutoff[which.max(cutoff_data$KS)]

my_cutoff ## or .127 first 4 decimal places are same
# now that we have our cutoff we can convert score to hard classes
#

my_cutoff <- .08
train.predicted <- as.numeric(train1.score > my_cutoff)
test.predicted <- as.numeric(predict(log_fit, churn_test) > my_cutoff)

churn_train$train_predicted <- train.predicted
churn_train$score <- train.score
churn_test$test_predicted <- test.predicted

write.csv(test.predicted,"churn_prediction.csv",row.names = F)
head(churn_test)








