args<-commandArgs(TRUE)

id=args[1]


# Create Line Chart
# resultPath=paste("./result/",id, sep="")
resultPath=args[2]
resultPath1=args[3]
jpeg(paste(resultPath,".jpeg",sep=""), width=10, height=4, units="in", res=800)


# mydata <- read.table(paste(resultPath,".txt",sep=""),colClasses=c("integer", "double", "character"))
mydata <- read.table(paste(resultPath1,".txt",sep=""),colClasses=c("integer", "double", "character"),sep = ',')
len=length(mydata$V2)
max_y <- max(mydata$V2)
min_y <- min(mydata$V2)
# print(mydata$V2)
mydata$V2<-mydata$V2
plot(mydata$V2, type="l", xlab="Residue Position", ylab="Prediction Score", ylim=c(min_y,max_y+max_y*0.1), xlim=c(0,len),cex.lab=1.2,cex.axis=1,cex=2,col="black")
exist_p=''
for (i in 1:len) {
  if(mydata$V2[i] >= max_y/2.0*1.8) {
    if(exist_p==''){
      posSite=paste(i,mydata$V3[i],sep="")
      # posSite=i
      text(i-1,mydata$V2[i] + 0.1*mydata$V2[i],posSite,cex=0.4, col="red")
      exist_p=i
    } else {
      if(abs(exist_p-i)>=10) {
      posSite=paste(i,mydata$V3[i],sep="")
      # posSite=i
      text(i-1,mydata$V2[i] + 0.1*mydata$V2[i],posSite,cex=0.4, col="red")
      exist_p=i
        } 
      }
  } 
}
abline(h=(max_y+max_y*0.1)/2.0,col=3,lty =2)
grid() 
title(paste("ProsperousPlus Prediction for",id))
dev.off()
