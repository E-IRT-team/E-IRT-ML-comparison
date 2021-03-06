
library(scmamp)
#library(PMCMRplus)

setwd('C:/Users/u0135479/Documents/GitHub/E-IRT-comparison')

filename <- "results_auroc"

csvname = paste(filename, ".csv", sep = "")
pdfname = paste(filename, "_res.pdf", sep = "")

# simul data read
data <- read.csv(csvname, header = TRUE)

cols.totest <- c("MLP",	"KNN","DT",	"RF",	"GB",	"QDA", "EIRM")

data.totest <- data[,cols.totest]
#clean.data <-clean.data[,sel.cols]
friedmanTest(data.totest)
nemenyiTest(data.totest)

pdf(pdfname, width = 16, height = 6.8)
#set decreasing = False for MSE loss
plotCD(data.totest, cex = 2.25, decreasing = T)
dev.off()