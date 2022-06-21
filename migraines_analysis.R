if (!require("dplyr")) install.packages("dplyr")
if (!require("caret")) install.packages("caret", dependencies = c("Depends", "Suggests"))
if (!require("gam")) install.packages("gam")
if (!require("randomForest")) install.packages("randomForest")
if (!require("mda")) install.packages("mda")
if (!require("klaR")) install.packages("klaR")
if (!require("party")) install.packages("party")
if (!require("corrplot")) install.packages("corrplot")
if (!require("readr")) install.packages("readr")

library(dplyr)
library(caret)
library(corrplot)
library (readr)

#Download and Load the dataset
urlfile="https://raw.githubusercontent.com/calresearch/dsedx/main/data.csv"
migraines<-read_csv(url(urlfile))

############# Exploratory Data Analysis #############
#Number of training examples
nrow(migraines)
ncol(migraines)
colnames(migraines)
head(migraines)
#Categories
migraines <- migraines %>% group_by(Type) 
summarise(migraines)
#verify that all labels are correct
unique(migraines$Type)
head(migraines)
summary(migraines)

cor_var = colnames(migraines) != "Type" & colnames(migraines) != "Ataxia"
res <- cor(migraines[cor_var])

corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

default_margins <- par("mar")
new_margins <- c(12, 2.5, 2.5, 2.5)
par(mar = new_margins)
barplot(table(migraines$Type), main="Label Distri. of Migraines Dataset", ylab="Frequency", las=3)


set.seed(1, sample.kind="Rounding")
#Partition the dataset
test_index <- createDataPartition(y = migraines$Type, times = 1, p = 0.2, list = FALSE)
test_set <- migraines[test_index,]
train_set <- migraines[-test_index,]
summary(test_set$Type)
barplot(table(test_set$Type), main="Label Dist. of Migraines Test Dataset", ylab="Frequency", las=3)
summary(train_set$Type)
barplot(table(train_set$Type), main="Label Dist. of Migraines Train Dataset", ylab="Frequency", las=3)

test_set <- test_set %>% ungroup()
train_set <- train_set %>% ungroup()

trellis.par.set(caretTheme())
control <- trainControl(method="cv", number=10, p=0.9)
indep_var = colnames(migraines) != "Type"

#Need to balance the classes for the training dataset using resampling
resample <- function(sample_size){
  Type_1 <- train_set %>% filter(Type=="Basilar-type aura") %>% sample_n(sample_size, replace=TRUE)
  Type_2 <- train_set %>% filter(Type=="Familial hemiplegic migraine") %>% sample_n(sample_size, replace=TRUE)
  Type_3 <- train_set %>% filter(Type=="Migraine without aura") %>% sample_n(sample_size, replace=TRUE)
  Type_4 <- train_set %>% filter(Type=="Other") %>% sample_n(sample_size, replace=TRUE)
  Type_5 <- train_set %>% filter(Type=="Sporadic hemiplegic migraine") %>% sample_n(sample_size, replace=TRUE)
  Type_6 <- train_set %>% filter(Type=="Typical aura with migraine") %>% sample_n(sample_size, replace=FALSE)
  Type_7 <- train_set %>% filter(Type=="Typical aura without migraine") %>% sample_n(sample_size, replace=TRUE)
  bind_rows(Type_1, Type_2, Type_3, Type_4, Type_5, Type_6, Type_7)
}

#Check optimal sample size
n <- seq(27,197,10)
size_acc_check <- function(n){
  bal_train <- resample(n)
  grid <- data.frame(mtry=10)
  model_bal_rf = train(x = bal_train[indep_var], y = bal_train$Type, method = 'rf', trControl=control, tuneGrid = grid)
  c(n, model_bal_rf$results$Accuracy)
}
samp_acc <- sapply(n,size_acc_check)
plot(samp_acc[1,], samp_acc[2,], xlab="Sample Size", ylab="Accurary", main="Sample Size vs Accurary (RF) on Training Set")
sample_size <- samp_acc[1,which.max(samp_acc[2,])]
sample_size

bal_train <- resample(sample_size)
barplot(table(bal_train$Type), main="Label Distri. of Balanced Training Set", ylab="Frequency", las=3)


############# Random Forest ############# 
grid <- data.frame(mtry=seq(2,24,2))
model_bal_rf = train(x = bal_train[indep_var], y = bal_train$Type, method = 'rf', trControl=control, tuneGrid = grid)
varImp(model_bal_rf)
ggplot(model_bal_rf) + labs(title="RF Model Tuning")

y_hat_bal <- predict(model_bal_rf, test_set[indep_var])
y_hat_bal
as_tibble()
#Overall Acuracy
acc_rf <- confusionMatrix(y_hat_bal, as.factor(test_set$Type))$overall[["Accuracy"]]
acc_rf
#Mean F1 score all classes
F1_rf <- mean(as_tibble(confusionMatrix(y_hat_bal, as.factor(test_set$Type))$byClass)$F1)
F1_rf
#Detailed results by class
confusionMatrix(y_hat_bal, as.factor(test_set$Type), mode="everything")$byClass


############# knn ############# 
grid <- data.frame(k=seq(3,13,2))
model_bal_knn = train(x = bal_train[indep_var], y = bal_train$Type, method = 'knn', trControl=control, tuneGrid = grid)
ggplot(model_bal_knn) + labs(title="KNN Model Tuning")

y_hat_bal <- predict(model_bal_knn, test_set[indep_var])
#Overall Acuracy
acc_knn <- confusionMatrix(y_hat_bal, as.factor(test_set$Type))$overall[["Accuracy"]]
acc_knn
#Mean F1 score all classes
F1_knn <- mean(as_tibble(confusionMatrix(y_hat_bal, as.factor(test_set$Type))$byClass)$F1)
F1_knn
#Detailed results by class
confusionMatrix(y_hat_bal, as.factor(test_set$Type), mode="everything")$byClass


############# PDA2 ############# 
grid <- data.frame(df=seq(1,21,2))
model_bal_pda = train(x = bal_train[indep_var], y = bal_train$Type, method = 'pda2', trControl=control, tuneGrid = grid)
ggplot(model_bal_pda) + labs(title="PDA Model Tuning")

y_hat_bal <- predict(model_bal_pda, test_set[indep_var])
#Overall Acuracy
acc_pda <- confusionMatrix(y_hat_bal, as.factor(test_set$Type))$overall[["Accuracy"]]
acc_pda
#Mean F1 score all classes
F1_pda <- mean(as_tibble(confusionMatrix(y_hat_bal, as.factor(test_set$Type))$byClass)$F1)
F1_pda
#Detailed results by class
confusionMatrix(y_hat_bal, as.factor(test_set$Type), mode="everything")$byClass


############# RDA ############# 
grid <- expand.grid(gamma=c(0.01, 0.001, 0.0001, 0.00001), lambda=seq(0.01, 0.1, 0.01))
model_bal_rda = train(x = bal_train[indep_var], y = bal_train$Type, method = 'rda', trControl=control, tuneGrid = grid)
ggplot(model_bal_rda) + labs(title="RDA Model Tuning")

y_hat_bal <- predict(model_bal_rda, test_set[indep_var])
#Overall Acuracy
acc_rda <- confusionMatrix(y_hat_bal, as.factor(test_set$Type))$overall[["Accuracy"]]
#Mean F1 score all classes
F1_rda <- mean(as_tibble(confusionMatrix(y_hat_bal, as.factor(test_set$Type))$byClass)$F1)
F1_rda
#Detailed results by class
confusionMatrix(y_hat_bal, as.factor(test_set$Type), mode="everything")$byClass


############# CIRF ############# 
grid <- data.frame(mtry=seq(4,14,2))
model_bal_cirf = train(x = bal_train[indep_var], y = bal_train$Type, method = 'cforest', trControl=control,tuneGrid = grid)
varImp(model_bal_cirf)
ggplot(model_bal_cirf) + labs(title="CIRF Model Tuning")

y_hat_bal <- predict(model_bal_cirf, test_set[indep_var])
#Overall Acuracy
acc_cirf <- confusionMatrix(y_hat_bal, as.factor(test_set$Type))$overall[["Accuracy"]]
acc_cirf
#Mean F1 score all classes
F1_cirf <- mean(as_tibble(confusionMatrix(y_hat_bal, as.factor(test_set$Type))$byClass)$F1)
F1_cirf 
#Detailed results by class
confusionMatrix(y_hat_bal, as.factor(test_set$Type), mode="everything")$byClass


#Comparison of final models
resamps <- resamples(list(RF = model_bal_rf,
                          KNN = model_bal_knn,
                          PDA = model_bal_pda,
                          RDA = model_bal_rda,
                          CIRF = model_bal_cirf))

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps, layout = c(2, 1), main="Test Set Model Performance")

acc_results <- data.frame(Accuracy=c(acc_rf,acc_knn,acc_pda,acc_rda,acc_cirf), Macro_F1 = c(F1_rf,F1_knn,F1_pda,F1_rda,F1_cirf))
rownames(acc_results) <- c("RF", "KNN", "PDA", "RDA", "CIRF")
acc_results
