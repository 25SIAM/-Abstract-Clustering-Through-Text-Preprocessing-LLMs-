install.packages(c("tidyverse", "tidytext", "tm", "SnowballC", 
                   "textstem", "cluster", "factoextra", 
                   "dbscan", "irlba", "ggplot2"))
library(tidyverse)
library(tidytext)
library(tm)
library(SnowballC)
library(textstem)
library(cluster)
library(factoextra)
library(dbscan)
library(irlba)
library(ggplot2)

# 1. Load dataset
csv_path <- "C:/UNIVERCIITY/9th Semester/INTRODUCTION TO DATA SCIENCE [E]/FINAL/Final_Project/ids_final_dataset_sample_group_06.csv"
df <- read.csv(csv_path, stringsAsFactors = FALSE)

# Output first 6 rows
cat("\n--- Dataset Preview ---\n")
print(head(df))

# 2. Text preprocessing
text_col <- "Abstract"

text_data <- df %>%
  select(doc_id = 1, !!sym(text_col)) %>%
  rename(text = !!sym(text_col)) %>%
  mutate(
    text = iconv(text, from = "", to = "UTF-8", sub = " "),
    text = tolower(text),
    text = gsub("[^a-z\\s]", " ", text)
  )

cat("\n--- Preprocessed Text Preview ---\n")
print(head(text_data))

# Tokenization
tokens <- text_data %>% unnest_tokens(word, text)

# Remove stopwords and short words
data("stop_words")
tokens <- tokens %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2)

cat("\n--- Tokenized Words Preview ---\n")
print(head(tokens, 30))

# Stemming & Lemmatization
tokens <- tokens %>%
  mutate(stem = wordStem(word),
         lemma = lemmatize_words(word))

cat("\n--- Stem & Lemma Preview ---\n")
print(head(tokens, 30))

# 3. Compute TF-IDF & DTM
# -------------------------------
tokens_for_tfidf <- tokens %>% select(doc_id, word = lemma)
word_counts <- tokens_for_tfidf %>% count(doc_id, word, sort = TRUE)

cat("\n--- Word Counts Preview ---\n")
print(head(word_counts, 30))

tfidf <- word_counts %>% bind_tf_idf(word, doc_id, n)

cat("\n--- TF-IDF Preview ---\n")
print(head(tfidf, 30))

dtm <- tfidf %>% cast_dtm(document = doc_id, term = word, value = tf_idf)
dtm_matrix <- as.matrix(dtm)

cat("\n--- Document-Term Matrix Dimensions ---\n")
print(dim(dtm_matrix))

# 4. PCA for dimensionality reduction 
# -------------------------------
# Scale features before PCA
dtm_scaled <- scale(dtm_matrix)

# Reduce dimensionality (keep first 10 PCs for clustering)
pca <- prcomp(dtm_scaled, center = TRUE, scale. = TRUE)
summary(pca)

# Keep first 10 PCs for clustering
pca_scores <- as.data.frame(pca$x[, 1:10])

# For visualization: only 2D
pca_2d <- as.data.frame(pca$x[, 1:2])
colnames(pca_2d) <- c("PC1", "PC2")

cat("\n--- PCA 2D Preview ---\n")
print(head(pca_2d))

# -------------------------------
# 5. K-Means clustering (on reduced PCA features)
# -------------------------------
set.seed(123)
k <- 5
kmeans_res <- kmeans(pca_scores, centers = k, nstart = 25)
pca_2d$kmeans_cluster <- factor(kmeans_res$cluster)

cat("\n--- K-Means Cluster Counts ---\n")
print(table(pca_2d$kmeans_cluster))

ggplot(pca_2d, aes(PC1, PC2, color = kmeans_cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  ggtitle("K-Means Clustering (on PCA Reduced Features)")

# -------------------------------
# 6. Hierarchical Clustering
# -------------------------------
hclust_res <- hclust(dist(pca_scores), method = "ward.D2")
hc_clusters <- cutree(hclust_res, k = k)
pca_2d$hc_cluster <- factor(hc_clusters)

cat("\n--- Hierarchical Cluster Counts ---\n")
print(table(pca_2d$hc_cluster))

ggplot(pca_2d, aes(PC1, PC2, color = hc_cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  scale_color_manual(values = rainbow(k)) +
  ggtitle("Hierarchical Clustering (on PCA Reduced Features)")

# -------------------------------
# 7. DBSCAN clustering
# -------------------------------
# Scale before DBSCAN (important!)
pca_for_dbscan <- scale(pca_scores[, 1:5])  # try with first 5 PCs

dbscan_res <- dbscan(pca_for_dbscan, eps = 1, minPts = 5)
pca_2d$dbscan_cluster <- factor(dbscan_res$cluster)

cat("\n--- DBSCAN Cluster Counts ---\n")
print(table(pca_2d$dbscan_cluster))

ggplot(pca_2d, aes(PC1, PC2, color = dbscan_cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  scale_color_manual(values = c("white", rainbow(length(unique(dbscan_res$cluster))))) +
  ggtitle("DBSCAN Clustering (on PCA Reduced Features)")

# -------------------------------
# 8. Compare all clustering results
# -------------------------------
pca_2d_long <- pca_2d %>%
  pivot_longer(cols = c(kmeans_cluster, hc_cluster, dbscan_cluster),
               names_to = "algorithm", values_to = "cluster")

ggplot(pca_2d_long, aes(PC1, PC2, color = cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  facet_wrap(~algorithm) +
  ggtitle("Comparison of Clustering Algorithms (PCA Reduced Features)")

