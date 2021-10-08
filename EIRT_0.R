

#mypath <- "C:/Users/u0135479/Documents/GitHub/EML/ML_EIRT/data/"  # Klest's GitHub path
#mypath <- "C:/Users/u0135479/Documents/GitHub/EML/ML_EIRT/temp/"   # Klest's GitHub path for RW1 data
#mypath <- "C:/Users/u0135479/Documents/GitHub/EML/ML_EIRT/data/100-10/"  # Klest's GitHub path for simul data
  
#mydata <- read.csv(file = paste0(mypath,"trainset.csv"), header=T, sep=",", na.strings = "999")
#mydatatest <-  read.csv(file = paste0(mypath,"testset.csv"), header=T, sep=",", na.strings = "999")

#mydata <- read.csv("C:/Users/konst/Box/ML_EIRT/Experiments/temp/trainset.csv",header=T,sep=",",na.strings = "999")
mydata <- read.csv("C:/Users/u0135479/Documents/GitHub/E-IRT-ML-comparison/temp/trainset.csv",header=T,sep=",",na.strings = "999")
#mydatatest <- read.csv("C:/Users/konst/Box/ML_EIRT/Experiments/temp/testset.csv",header=T,sep=",",na.strings = "999")
mydatatest <- read.csv("C:/Users/u0135479/Documents/GitHub/E-IRT-ML-comparison/temp/testset.csv",header=T,sep=",",na.strings = "999")

vn<-names(mydata)
vn2<-vn[!(vn %in% c("y","X..student_id","item_id"))]
n<-paste (vn2, sep = " ", collapse = "+")
eq <- as.formula(paste0("y ~ 1+",n,"+ (1|X..student_id) + (1|item_id)"))


#### here runs the main part of the code #####

### WARNING: the randomly packed variables (1|student_id) and (1|item_id) do not have enough factors in RW1

library(lme4)

#IRTfit <- glmer(y ~ 1 +. + (1|X..student_id) + (1|item_id), data=mydata, family=binomial(link="logit")) # for RW1 dataset
#IRTfit <- glmer(y ~ 1 +. + (1|student_id) - student_id - item_id + (1|item_id), data=mydata, family=binomial(link="logit")) # for RW1 dataset
IRTfit <- glmer(eq, data=mydata, family=binomial(link="logit"))


newpred <- predict(IRTfit,newdata=mydatatest,type='response',allow.new.levels = TRUE)
cat(newpred)
#write.table(newpred, "C:/Users/u0106589/Box Sync/workplace/IMEC/LearningEnv/DemoML_EIRT//RES.txt",row.names = FALSE,col.names = FALSE)