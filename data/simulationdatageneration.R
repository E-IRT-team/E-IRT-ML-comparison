#######################################################
# 100 persons' responses to 10 items 
# 9 latent features and 10% noise  
######################################################## 
library('TAM')
setwd("C:/")



# Set number of persons (NPers) & number of items (NItems)
set.seed(1234) 
NPers <- 100 
NItems <- 10  


# Item vs Latent features Q-matrix  
q1 <- rbinom(NItems,1,0.5)  
q2 <- rbinom(NItems,1,0.5)
q3 <- rbinom(NItems,1,0.5)
q4 <- rbinom(NItems,1,0.5)
q5 <- rbinom(NItems,1,0.5)
q6 <- rbinom(NItems,1,0.5)
q7 <- rbinom(NItems,1,0.5)
q8 <- rbinom(NItems,1,0.5)
q9 <- rbinom(NItems,1,0.5)
qmatrix<-cbind(q1,q2,q3,q4,q5,q6,q7,q8,q9)
NDims <- ncol(qmatrix)


# Create 15 person background variables   
Qmatrix.Per <- matrix(rbinom(n=NPers*15, size=1, prob=0.20), nrow=NPers, ncol=15)  
Xmat <- as.matrix(data.frame(incpt=1, Qmatrix.Per))
a0 <- 1.2
a <- rnorm(15,0,1)
Avector <- c(a0, a) # 16 true reg coefs (including intercept)


# Vary true coefficients by latent features 
adjusted <- runif(NDims,0.2,1)  
Avector1 <- Avector*adjusted[1]
Avector2 <- Avector*adjusted[2]
Avector3 <- Avector*adjusted[3]
Avector4 <- Avector*adjusted[4]
Avector5 <- Avector*adjusted[5]
Avector6 <- Avector*adjusted[6]
Avector7 <- Avector*adjusted[7]
Avector8 <- Avector*adjusted[8]
Avector9 <- Avector*adjusted[9]


# Generate Person Parameters (Linear Predictor for Person Side)
XA1 <- Xmat %*% (Avector1) 
XA2 <- Xmat %*% (Avector2) 
XA3 <- Xmat %*% (Avector3) 
XA4 <- Xmat %*% (Avector4) 
XA5 <- Xmat %*% (Avector5) 
XA6 <- Xmat %*% (Avector6) 
XA7 <- Xmat %*% (Avector7) 
XA8 <- Xmat %*% (Avector8) 
XA9 <- Xmat %*% (Avector9) 
XF <- cbind(XA1, XA2, XA3, XA4, XA5, XA6, XA7, XA8, XA9)


# Set R squared to get SD of Person Errors (random noise)
Rsquared <- 0.90 # noise level=0.10
sd.R2.Avector1 <- sqrt(sum(sapply(Avector1[-1], function(x) x^2))) * sqrt((1 - Rsquared)/Rsquared)
sd.R2.Avector2 <- sqrt(sum(sapply(Avector2[-1], function(x) x^2))) * sqrt((1 - Rsquared)/Rsquared)
sd.R2.Avector3 <- sqrt(sum(sapply(Avector3[-1], function(x) x^2))) * sqrt((1 - Rsquared)/Rsquared)
sd.R2.Avector4 <- sqrt(sum(sapply(Avector4[-1], function(x) x^2))) * sqrt((1 - Rsquared)/Rsquared)
sd.R2.Avector5 <- sqrt(sum(sapply(Avector5[-1], function(x) x^2))) * sqrt((1 - Rsquared)/Rsquared)
sd.R2.Avector6 <- sqrt(sum(sapply(Avector6[-1], function(x) x^2))) * sqrt((1 - Rsquared)/Rsquared)
sd.R2.Avector7 <- sqrt(sum(sapply(Avector7[-1], function(x) x^2))) * sqrt((1 - Rsquared)/Rsquared)
sd.R2.Avector8 <- sqrt(sum(sapply(Avector8[-1], function(x) x^2))) * sqrt((1 - Rsquared)/Rsquared)
sd.R2.Avector9 <- sqrt(sum(sapply(Avector9[-1], function(x) x^2))) * sqrt((1 - Rsquared)/Rsquared)


# Generate Person Errors randomly drawn from the normal dist
sigmad <- c(sd.R2.Avector1,sd.R2.Avector2,sd.R2.Avector3,sd.R2.Avector4,sd.R2.Avector5,sd.R2.Avector6,sd.R2.Avector7,sd.R2.Avector8,sd.R2.Avector9)
Cord <- matrix(0.3, NDims, NDims)
diag(Cord) <- sd.R2 
Vard <- diag(sigmad) %*% Cord %*% diag(sigmad)
pers.error.R2 <- rmvnorm(NPers, mean=rep(0, NDims), sigma=Vard)   
pers.deltas <- XF + pers.error.R2


# Same Steps for Item Component 
Qmatrix.Item <- matrix(rbinom(n=NItems*10, size=1, prob=0.20), nrow=NItems, ncol=10)  
b0 = 0
b = rnorm(10,0.5,1)
Bvector <- c(b0, b) 


# Generate Item Parameters 
Zmat <- as.matrix(data.frame(incpt=1, Qmatrix.Item))
sd.R2 <- sqrt(sum(sapply(Bvector[-1], function(x) x^2))) * sqrt((1 - Rsquared)/Rsquared)
error.R2 <- rnorm(NItems, mean = 0, sd = sd.R2)
deltas <- Zmat %*% Bvector + error.R2


# Calculate Linear Predictor for IRT model 
qmatrix<-as.matrix(qmatrix)
qmatrix1<-matrix(runif(NItems*ncol(qmatrix),0,1),nrow=NItems)*qmatrix 
true.ability<-linpred<-matrix(NA, NPers, NItems)
for(j in 1:NItems)
{
  for(i in 1:NPers)
  {
    true.ability[i,j] <- t(qmatrix[j,]) %*% (pers.deltas[i,])
    linpred[i,j] <- true.ability[i,j] - deltas[j]
  }
} 


# Calculate item response probs, binary responses, and save the data 
prob <- exp(linpred) / (1 + exp(linpred)) 
sim.resp <- matrix(sapply(prob, function(x) rbinom(n=1, size=1, prob=x)), ncol = NItems)
colnames(sim.resp) <- factor(paste0("I", sprintf(paste('%0', max(nchar(1:NItems)), 'i', sep=''), c(1:NItems))))
table(apply(sim.resp,1,sum))

