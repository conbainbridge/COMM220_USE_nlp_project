########################################################
########### Sentiment analysis of USE essays ###########
################## Connie Bainbridge ###################
########################################################

# USE dataset available at the follow link (link on main site is either broken or otherwise unavailable at least when accessed from the US): https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457

###########################
### Required R packages ###
###########################

install.packages("tidyverse")
install.packages("tidytext")
install.packages("quanteda")
install.packages("quanteda.textmodels")
install.packages("rvest")
install.packages("ggplot2")

library(tidyverse)
library(tidytext)
library(quanteda) # Info on using quanteda: https://quanteda.io/articles/pkgdown/examples/lsa.html
library("quanteda.textmodels")
library(rvest) # For wrangling the .txt files and the html within them
library(ggplot2)



######################
### Setup datasets ###
######################

### Dataset 1 - Literature prompt
setwd("~/Documents/academia/UCLA/Year 1/COMM 220/final_project/COMM220_USE_nlp_project/USE_data/USEcorpus/")
# Set up for loop to pipe each file into a single dataset.
fileList <- list.files(path = "a4", pattern = "txt")
setwd("~/Documents/academia/UCLA/Year 1/COMM 220/final_project/COMM220_USE_nlp_project/USE_data/USEcorpus/a4")

iterations = 185 # Sample size
USE_dataMerge = NULL

# Populate data for USE dataset1 with cleaned up texts
for (i in 1:iterations) {
  # Get file from list, parse out text from HTML, remove weird markers
  file = file(fileList[i])
  targetFile = read_html(file)
  clean.targetFile <- targetFile %>% html_nodes("body") %>% html_text()
  clean.targetFile = gsub("\r","",clean.targetFile)
  clean.targetFile = gsub("\n"," ",clean.targetFile)
  clean.targetFile = gsub("\t","",clean.targetFile)
  clean.targetFile = gsub("'","",clean.targetFile)
  clean.targetFile = gsub(",","",clean.targetFile)
  clean.targetFile = gsub('\"', "",clean.targetFile)
  USE_dataMerge[i] <- clean.targetFile
}


### Dataset 2 - My English prompt
setwd("~/Documents/academia/UCLA/Year 1/COMM 220/final_project/COMM220_USE_nlp_project/USE_data/USEcorpus/")
fileList2 <- list.files(path = "a1", pattern = "txt")
setwd("~/Documents/academia/UCLA/Year 1/COMM 220/final_project/COMM220_USE_nlp_project/USE_data/USEcorpus/a1")

iterations2 = 303 # Sample size
USE2_dataMerge = NULL

# Populate data for with cleaned up texts
for (i in 1:iterations2) {
  # Get file from list, parse out text from HTML, remove weird markers
  file = file(fileList2[i])
  targetFile = read_html(file)
  clean.targetFile <- targetFile %>% html_nodes("body") %>% html_text()
  clean.targetFile = gsub("\r","",clean.targetFile)
  clean.targetFile = gsub("\n"," ",clean.targetFile)
  clean.targetFile = gsub("\t","",clean.targetFile)
  clean.targetFile = gsub("'","",clean.targetFile)
  clean.targetFile = gsub(",","",clean.targetFile)
  clean.targetFile = gsub('\"', "",clean.targetFile)
  USE2_dataMerge[i] <- clean.targetFile
}



##########################
### Sentiment analysis ###
##########################

sents = get_sentiments("nrc")
positive = sents[which(sents[,2]=='positive'),1]$word
negative = sents[which(sents[,2]=='negative'),1]$word

USE1_positive = NULL
USE1_negative = NULL
USE2_positive = NULL
USE2_negative = NULL

USE_nostop = NULL
USE2_nostop = NULL

USE1_length <- c()
USE2_length <- c()

# USE a4 - Literature prompt
for (i in 1:iterations) {
  USE_nostop <- gsub(",","", toString(USE_dataMerge[i]))
  USE_nostop = tolower(unlist(strsplit(USE_nostop," ")))
  USE1_length <- c(USE1_length, length(USE_nostop))
  
  m_positive = mean(USE_nostop %in% positive)
  USE1_positive = rbind(m_positive, USE1_positive)
  USE1_positive
  
  m_negative = mean(USE_nostop %in% negative)
  USE1_negative = rbind(m_negative, USE1_negative)
  USE1_negative
}

# USE a1 - My English prompt
for (i in 1:iterations2) {
  USE2_nostop <- gsub(",","", toString(USE2_dataMerge[i]))
  USE2_nostop = tolower(unlist(strsplit(USE2_nostop," ")))
  USE2_length <- c(USE2_length, length(USE2_nostop))
  
  m_positive = mean(USE2_nostop %in% positive)
  USE2_positive = rbind(m_positive, USE2_positive)
  USE2_positive
  
  m_negative = mean(USE2_nostop %in% negative)
  USE2_negative = rbind(m_negative, USE2_negative)
  USE2_negative
}


# sentsAFINN = get_sentiments("afinn") # Saving for future work - plenty to dig into here already!



##################
### Some stats ###
##################

# Positive / negative sentiment analyses
hist(USE1_positive) # Eyeballing for mostly normal distribution
hist(USE2_positive) # Eyeballing for mostly normal distribution
hist(USE1_negative) # Eyeballing for mostly normal distribution
hist(USE2_negative) # Eyeballing for mostly normal distribution

ttest_positive <- t.test(USE1_positive, USE2_positive, alternative = c("two.sided"))
ttest_positive

ttest_negative <- t.test(USE1_negative, USE2_negative, alternative = c("two.sided"))
ttest_negative


# Compare word counts
hist(USE1_length) # Eyeballing for mostly normal distribution
hist(USE2_length) # Eyeballing for mostly normal distribution

ttest_count <- t.test(USE1_length, USE2_length)
ttest_count


#################
### LSA setup ###
#################

# Literature prompt
USE_lsa <- tokens(USE_dataMerge, remove_numbers = TRUE,  remove_punct = TRUE)
USE_lsa <- tokens_select(USE_lsa, pattern = stopwords("en"), selection = "remove")
mydfm1 <- dfm(USE_lsa)
mylsa1 <- textmodel_lsa(mydfm1)

# My English prompt
USE2_lsa <- tokens(USE2_dataMerge, remove_numbers = TRUE,  remove_punct = TRUE)
USE2_lsa <- tokens_select(USE2_lsa, pattern = stopwords("en"), selection = "remove")
mydfm2 <- dfm(USE2_lsa)
mylsa2 <- textmodel_lsa(mydfm2)


# Looking at Ngrams
ngrams_USE1 <- tokens_ngrams(USE_lsa)
ngram_dfm1 <- dfm(ngrams_USE1)
tstat_freq1 <- textstat_frequency(ngram_dfm1, n = 20)
head(tstat_freq1, 20)

ngrams_USE2 <- tokens_ngrams(USE2_lsa)
ngram_dfm2 <- dfm(ngrams_USE2)
tstat_freq2 <- textstat_frequency(ngram_dfm2, n = 20)
head(tstat_freq2, 20)

# debugger <- dfm(USE2_dataMerge, select = "*�*")  # Tracking down mystery ? symbol to see what the cause is



#######################
### Frequency plots ###
#######################
# Note: Frequencies are sans stop words.

# Literature prompt
frequency_plot1 <- textstat_frequency(mydfm1, n= 1000) %>%
  ggplot(aes(x= rank, y = frequency)) + geom_point() + labs(x= "Frequency Rank", y = "Frequency Term") + ggtitle("Word frequency for Literature prompt")
frequency_plot1

# My English prompt
frequency_plot2 <- textstat_frequency(mydfm2, n= 1000) %>%
  ggplot(aes(x= rank, y = frequency)) + geom_point() + labs(x= "Frequency Rank", y = "Frequency Term") + ggtitle("Word frequency for \"My English\" prompt")
frequency_plot2


# Setting up similarity data for possible future analyses
tstat_similarity <- textstat_dist(
  x = mydfm1,
  y = mydfm2,
  selection = NULL,
  margin = c("documents"),
  p = 2
)
tstat_similarity



################
### Features ###
################

# Get top features
topfeatures(mydfm1, 20) # Lit prompt
topfeatures(mydfm2, 20) # My English prompt

featureV <- c("like") # Tried individual terms for the sake of running t-test -> the terms I looked at were words in common across each essay group's 20 top features - like, can, time, and also. Like was the only one that was significantly different and as such is kept here.

# Literature prompt
querydfm <- dfm_match(dfm(mydfm1), featureV)
querydfm

# My English prompt
querydfm2 <- dfm_match(dfm(mydfm2), featureV)
querydfm2

# Compare words that appear in top features for both prompts - like, can, time, also
t.test(querydfm, querydfm2) # us is significant



# Needs debugging - setup for future investigations of prediction analyses
newq <- predict(mylsa1, newdata = querydfm)
newq$docs_newspace[, 1:2]
##### Resources for debugging this above part?
# https://quanteda.io/articles/pkgdown/examples/lsa.html
# https://quanteda.io/reference/tokens.html
# https://www.rdocumentation.org/packages/quanteda/versions/2.1.2/topics/dfm_match
# https://quanteda.io/articles/pkgdown/examples/lsa.html
# unlist function might be helpful? Make list into vector
# Might consider chi-square to look at proportion of negative/positive words - create table of both essay things



###################
### Text clouds ###
###################

# set.seed(825)
library(RColorBrewer)

# Literature prompt
textplot_wordcloud(mydfm1, max_words = 500, colors = RColorBrewer::brewer.pal(8,  "Dark2"))

# My English prompt
textplot_wordcloud(mydfm2, max_words = 500, colors = RColorBrewer::brewer.pal(8,  "Dark2"))




