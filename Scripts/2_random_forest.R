
library(tidyverse)
library(GGally)        
library(caret)         
library(randomForest)  
library(ranger)        
library(vip)           
library(cowplot)       

set.seed(42) 

# -----------------------
# 1. Load data & quick peek
# -----------------------
data(iris)
df <- iris
glimpse(df)
summary(df)

# -----------------------
# 2. Exploratory Data Analysis (visualizations)
# -----------------------

# 2.1 Pairwise scatterplot matrix (colored by Species)
p_pairs <- GGally::ggpairs(
  df, 
  columns = 1:4, 
  mapping = aes(color = Species, alpha = 0.6),
  upper = list(continuous = wrap("cor", size = 3))
) + theme_minimal()
# Print or save:
print(p_pairs)

# 2.2 Boxplots (feature distributions by Species)
p_box <- df %>%
  pivot_longer(cols = 1:4, names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = Species, y = value, fill = Species)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~feature, scales = "free_y") +
  labs(title = "Feature distributions by Species") +
  theme_minimal() +
  theme(legend.position = "none")
print(p_box)

# 2.3 Correlation heatmap for numeric features
num_df <- df %>% select(-Species)
cor_mat <- cor(num_df)
cor_df <- as.data.frame(as.table(cor_mat))
p_cor <- ggplot(cor_df, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = round(Freq, 2)), size = 3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  labs(title = "Correlation matrix (numeric features)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p_cor)

# -----------------------
# 3. Preprocessing & Train/Test Split
# -----------------------
# For iris, minimal preprocessing necessary. We'll create train/test split.
set.seed(123)
train_idx <- createDataPartition(df$Species, p = 0.80, list = FALSE)
train <- df[train_idx, ]
test  <- df[-train_idx, ]

# Check class balance
table(train$Species)
table(test$Species)

# -----------------------
# 4. Baseline: simple decision boundary visualization (2 features)
# -----------------------
# Visualize decision boundaries using two features (Sepal.Length, Sepal.Width)
# Fit a simple RF on only two features for boundary plotting
rf_2feat <- randomForest(Species ~ Sepal.Length + Sepal.Width, data = train, ntree = 200)

# Create grid
x_range <- seq(min(df$Sepal.Length) - 0.5, max(df$Sepal.Length) + 0.5, length = 200)
y_range <- seq(min(df$Sepal.Width) - 0.5,  max(df$Sepal.Width) + 0.5,  length = 200)
grid <- expand.grid(Sepal.Length = x_range, Sepal.Width = y_range)

# Predict grid
grid$pred <- predict(rf_2feat, newdata = grid)

p_boundary <- ggplot() +
  geom_tile(data = grid, aes(x = Sepal.Length, y = Sepal.Width, fill = pred), alpha = 0.6) +
  geom_point(data = train, aes(x = Sepal.Length, y = Sepal.Width, color = Species), size = 1.6) +
  labs(title = "RF Decision Boundary (Sepal.Length vs Sepal.Width)",
       subtitle = "Trained on train split (2-feature RF)") +
  theme_minimal()
print(p_boundary)

# -----------------------
# 5. Model tuning and training with caret (randomForest)
# -----------------------
# We'll tune 'mtry' using caret with repeated CV
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = "final"
)

# Candidate mtry values (for 4 predictors)
mtry_grid <- expand.grid(mtry = 1:4)

set.seed(111)
rf_fit <- train(
  Species ~ .,
  data = train,
  method = "rf",
  metric = "Accuracy",
  tuneGrid = mtry_grid,
  trControl = ctrl,
  ntree = 500,
  importance = TRUE
)

# Print model results & best tuning parameter
print(rf_fit)
plot(rf_fit) + ggtitle("Tuning results (mtry)")

# -----------------------
# 6. Training using ranger (fast) with best mtry (optional)
# -----------------------
best_mtry <- rf_fit$bestTune$mtry
set.seed(222)
ranger_fit <- ranger(
  formula = Species ~ .,
  data = train,
  num.trees = 500,
  mtry = best_mtry,
  importance = "permutation",
  probability = FALSE
)
ranger_fit

# -----------------------
# 7. Evaluate on Test Set
# -----------------------
# Predictions
pred_rf <- predict(rf_fit, newdata = test)
pred_prob <- predict(rf_fit, newdata = test, type = "prob")

# Confusion matrix / accuracy
cm <- confusionMatrix(pred_rf, test$Species)
print(cm)

# Per-class metrics (caret confusionMatrix includes by-class stats)
# Show accuracy and kappa explicitly
cat(sprintf("Test Accuracy: %.3f\n", cm$overall["Accuracy"]))
cat(sprintf("Kappa: %.3f\n", cm$overall["Kappa"]))

# -----------------------
# 8. Visualizing predictions vs true
# -----------------------
p_conf <- as.data.frame(cm$table) %>%
  ggplot(aes(Prediction, Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  labs(title = "Confusion Matrix (Test Set)") +
  theme_minimal()
print(p_conf)

# 8.1 Plot class probabilities for test set (example for first 20 obs)
prob_df <- test %>%
  mutate(Predicted = pred_rf) %>%
  bind_cols(as_tibble(pred_prob)) %>%
  slice(1:20) %>%
  pivot_longer(cols = starts_with("setosa") | starts_with("versicolor") | starts_with("virginica"),
               names_to = "class", values_to = "prob")

# Build a probability bar plot for first 20 observations
# (we rely on the probability columns being named exactly like the species)
prob_cols <- colnames(pred_prob)
prob_df <- test %>%
  mutate(Predicted = pred_rf) %>%
  bind_cols(as_tibble(pred_prob)) %>%
  mutate(id = row_number()) %>%
  slice(1:20) %>%
  pivot_longer(cols = all_of(prob_cols), names_to = "class", values_to = "prob")

p_probs <- ggplot(prob_df, aes(x = factor(id), y = prob, fill = class)) +
  geom_col(position = "stack") +
  facet_wrap(~ Predicted, scales = "free_x") +
  labs(title = "Predicted class probabilities (first 20 test obs)", x = "Observation (subset)") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p_probs)

# -----------------------
# 9. Variable importance & interpretation
# -----------------------
# 9.1 caret variable importance
vip_caret <- varImp(rf_fit, scale = TRUE)
print(vip_caret)
p_vip1 <- ggplot(vip_caret, top = 4) + ggtitle("Variable Importance (caret::varImp)")

# 9.2 ranger permutation importance (if ranger_fit used)
vi_ranger <- importance(ranger_fit)
vi_df <- data.frame(Feature = names(vi_ranger), Importance = vi_ranger) %>%
  arrange(desc(Importance))

p_vip2 <- ggplot(vi_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() +
  coord_flip() +
  labs(title = "Permutation Variable Importance (ranger)", x = "", y = "Importance") +
  theme_minimal()
print(p_vip2)


# -----------------------
# 10. Error vs number of trees (if randomForest object available)
# -----------------------
# If rf_fit$finalModel is randomForest object, plot OOB error
rf_obj <- rf_fit$finalModel
if(inherits(rf_obj, "randomForest")) {
  err_df <- data.frame(
    trees = 1:length(rf_obj$err.rate[,1]),
    OOB = rf_obj$err.rate[, "OOB"],
    setosa = rf_obj$err.rate[, "setosa"],
    versicolor = rf_obj$err.rate[, "versicolor"],
    virginica = rf_obj$err.rate[, "virginica"]
  )
  err_df_long <- err_df %>% pivot_longer(-trees, names_to = "series", values_to = "error")
  p_err <- ggplot(err_df_long, aes(x = trees, y = error, color = series)) +
    geom_line() +
    labs(title = "Random Forest Error vs Trees (OOB & class-wise)", y = "Error") +
    theme_minimal()
  print(p_err)
}

