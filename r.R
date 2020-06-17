require(dplyr)
require('plyr')


data <- read.csv('./iris_data.txt', header=F)

# first suprise right here, indexing
head(data,5)

colnames(data) <- c('sepal_length','sepal_width','petal_length','petal_width','class')
colnames(data)

unique(data$class)
uniques <- unique(data$class)

#dplyr
data$class <- mapvalues(data$class, from=uniques, to=c(0,1,2))

tail(data,5)


knn_like <- function(new,df,k){
  dist_df = data.frame()

  for (i in 1:nrow(df)){
    distance = 0
    # calculate distance d
    for (j in 1:length(new)){
      distance = distance + (new[j] - df[i,j])^2
    }
    # final value per row
    distance <- distance^.5

    temp_df <- data.frame(
      'index' <- i,
      'dist'  <- distance
    )
    dist_df = rbind(dist_df,temp_df)
  }

  key_table = head(dist_df[order(dist_df[,2]),],k)
  print(key_table)
  keys = as.integer(rownames(key_table))
  #keys
  #attributes(df)
  #dplyr
  results = slice(df,keys)
  return(mean(as.integer(results$class)))
}

new <- c(7,3,5,1)
predicted_val <- knn_like(new,data,5)
predicted_val


data[7,]
