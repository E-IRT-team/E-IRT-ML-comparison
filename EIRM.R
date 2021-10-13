
#setwd("C:/Users/u0134289/Box Sync/ML_EIRT/DemoML_EIRT")
#setwd("C:/Users/u0135479/Box Sync/ML_EIRT/DemoML_EIRT")
setwd("C:/Users/u0135479/Documents/GitHub/E-IRT-ML-comparison")
getwd()
rm(list=ls())

#mydata <- read.csv("trainset.csv", header=T,sep=",",na.strings = "-1")
mydata <- read.csv("C:/Users/u0135479/Documents/GitHub/E-IRT-ML-comparison/temp/trainset.csv",header=T,sep=",",na.strings = "999")

#mydatatest <- read.csv("testset.csv", header=T,sep=",",na.strings = "-1")
mydatatest <- read.csv("C:/Users/u0135479/Documents/GitHub/E-IRT-ML-comparison/temp/testset.csv",header=T,sep=",",na.strings = "999")

str(mydata)
names(mydata)[names(mydata)=="X..student_id"] <- "student_id"
names(mydatatest)[names(mydatatest)=="X..student_id"] <- "student_id"


(vn <- names(mydata))
vn <- vn[!(vn %in% c("y", "student_id", "item_id"))]
eq.vn <- paste(vn, sep = " ", collapse = "+")
eq <- as.formula(paste0("y ~ 1+", eq.vn, "+(1|student_id)+(1|item_id)"))

# eq1 <- as.formula(paste0("y ~ 1+", eq.vn))
# summary(glm(eq1, data=mydata, family=binomial(link="logit")))


library(lme4)

IRTfit0 <- glmer(eq, data=mydata, family=binomial(link="logit"))

newpred0 <- predict(IRTfit0,newdata=mydatatest,type='response',allow.new.levels = TRUE)
cat(newpred0)


#mydata <- read.csv("C:/Users/u0106589/Box Sync/workplace/IMEC/LearningEnv/DemoML_EIRT/trainset.csv",
#                   header=T,sep=",",na.strings = "999")
#mydata <- read.csv("C:/Users/u0135479/Documents/Jupiter_Notebooks/DemoML_EIRT/Other_datasets/trainset.csv",
#                   header=T,sep=",",na.strings = "-1")
# mydatatest <- read.csv("C:/Users/u0106589/Box Sync/workplace/IMEC/LearningEnv/DemoML_EIRT/testset.csv",
#                       header=T,sep=",",na.strings = "999")
#mydatatest <- read.csv("C:/Users/u0135479/Documents/Jupiter_Notebooks/DemoML_EIRT/Other_datasets/testset.csv",
#                       header=T,sep=",",na.strings = "-1")
# 
# library(lme4)
# 
# IRTfit <- glmer(y~ 1+ study+gender+school_progress+language_home+number_books_at_home+SES+Dyslexia+
#                   dyscalculia+ADHD+ASS+other_learning_problem+School_type+school_type2+rural_urban+
#                   concentration+Province+language_friends+hours_math_per_week+academic_self_concept_math+
#                   academic_self_concept+attest_2nd_grade+math_ambitions+Attitude_math_parents+question_type+
#                   Tabel+Graph+attainment_target+level_of_processing+standard_deviation + (1|student_id) + (1|item_id), data=mydata, family=binomial(link="logit"))
# 
# 
# newpred <- predict(IRTfit,newdata=mydatatest,type='response',allow.new.levels = TRUE)
# cat(newpred)
#write.table(newpred, "C:/Users/u0135479/Documents/Jupiter_Notebooks/DemoML_EIRT/Other_datasets/new_predict.csv", 
#            row.names = FALSE,col.names = FALSE)

#newpred <- read.csv("C:/Users/u0135479/Documents/Jupiter_Notebooks/DemoML_EIRT/Other_datasets/new_predict.csv",
#                                   header=T,sep=",",na.strings = "-1")
# 
# L2.diff <- (newpred-newpred0)^2
# cat(L2.diff)

