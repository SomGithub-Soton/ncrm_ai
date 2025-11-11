#=============================================================================
#=============================================================================

# List of packages
packages <- c("car", "caret", "cluster", "clValid", "corrplot", "cowplot", 
              "DataExplorer", "dbscan", "dplyr", "e1071", "factoextra", 
              "fclust", "GGally", "ggplot2", "gridExtra", "kableExtra", 
              "knitr", "nnet", "patchwork", "pdp", "pROC", "randomForest", 
              "ranger", "reshape2", "rpart", "rpart.plot", "tidyr", "tidyverse", "vip")

# Install packages that are not installed
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages, dependencies = TRUE)

# Load all packages
lapply(packages, library, character.only = TRUE)

install.packages("h2o")
library(h2o)
#=============================================================================
#=============================================================================
