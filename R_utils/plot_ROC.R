plot_ROC <- function(testfile,predictfile,bin=0.05,posneg=0.5,lty=1,lwd=2,...){
	 test <- read.csv(testfile,header=T,row.names=1)
	 predict <- read.csv(predictfile,header=T,row.names=1)

	 type <- colnames(test)
	 tpr <- matrix(0,1/bin,length(type))
	 colnames(tpr) <- type
	 rownames(tpr) <- 1-((1:(1/bin))-1)*bin
	 fpr <- tpr
	 auc <- numeric(length(type))
	 names(auc) <- type

	 for(i in 1:length(type)){
	       pos <- which(test[,i] >= posneg)
	       neg <- which(test[,i] <  posneg)
	       for(j in 1:(1/bin)){
	       cutoff <- 1-(j-1)*bin
	       tp <- length(which(predict[pos,i] >= cutoff))
	       tn <- length(which(predict[neg,i] <  cutoff))
	       fn <- length(which(predict[pos,i] <  cutoff))
	       fp <- length(which(predict[neg,i] >= cutoff))
	       
	       tpr[j,i] <- tp / (tp + fn)
	       fpr[j,i] <- fp / (fp + tn)
	       }
	       if(i != 1){
	       par(new=T)
	       plot(c(0,fpr[,i],1),c(0,tpr[,i],1),type="l",col=i,
	       ann=F,axes=F,lty=lty,lwd=lwd,...)
	       }else{
	       plot(c(0,fpr[,i],1),c(0,tpr[,i],1),type="l",col=i,
	       xlab="FPR",ylab="TPR",lty=lty,lwd=lwd,...)
	       }
	       auc[i] <- sum((c(0,tpr[,i])) * (c(fpr[,i],1) - c(0,fpr[,i])))
	 }
	 legend(0.6,0.4,names(auc),col=1:length(type),lty=lty,lwd=lwd,box.lwd=0)
	 return(auc)
}