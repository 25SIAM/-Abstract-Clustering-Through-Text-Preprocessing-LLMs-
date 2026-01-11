install.packages(c("tidyverse", "tidytext", "tm", "SnowballC", 
                   "textstem", "cluster", "factoextra", 
                   "dbscan", "irlba", "ggplot2", 
                   "wordcloud", "RColorBrewer"))
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
library(wordcloud)
library(RColorBrewer)

csv_path <- "C:/UNIVERCIITY/9th Semester/INTRODUCTION TO DATA SCIENCE [E]/FINAL/Final_Project/ids_final_dataset_sample_group_06.csv"
df <- read.csv(csv_path, stringsAsFactors = FALSE)

cat("\n--- Dataset Preview ---\n")
print(head(df))

text_col <- "Abstract"
text_data <- df %>%
  select(doc_id = 1, !!sym(text_col)) %>%
  rename(text = !!sym(text_col)) %>%
  mutate(
    text = iconv(text, from = "", to = "UTF-8", sub = " "),
    text = tolower(text),
    text = gsub("[^a-z\\s]", " ", text)
  )

tokens <- text_data %>% unnest_tokens(word, text)
data("stop_words")
tokens <- tokens %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2) %>%
  mutate(stem = wordStem(word),
         lemma = lemmatize_words(word))

tokens_for_tfidf <- tokens %>% select(doc_id, word = lemma)
word_counts <- tokens_for_tfidf %>% count(doc_id, word, sort = TRUE)

tfidf <- word_counts %>% bind_tf_idf(word, doc_id, n)
dtm <- tfidf %>% cast_dtm(document = doc_id, term = word, value = tf_idf)
dtm_matrix <- as.matrix(dtm)

cat("\n--- Document-Term Matrix Dimensions ---\n")
print(dim(dtm_matrix))

set.seed(123)
wss <- vector()
max_k <- 10  
for (i in 1:max_k) {
  kmeans_model <- kmeans(dtm_matrix, centers = i, nstart = 10)
  wss[i] <- kmeans_model$tot.withinss
}
plot(1:max_k, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters (k)",
     ylab = "Total within-cluster sum of squares",
     main = "Elbow Method for Optimal k")

sil_scores <- numeric(max_k)
for (i in 2:max_k) {
  km_res <- kmeans(dtm_matrix, centers = i, nstart = 10)
  ss <- silhouette(km_res$cluster, dist(dtm_matrix))
  sil_scores[i] <- mean(ss[, 3])
}
plot(2:max_k, sil_scores[2:max_k], type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters (k)",
     ylab = "Average Silhouette Width",
     main = "Silhouette Method for Optimal k")

best_k <- which.max(sil_scores)
cat("\n--- Best k based on Silhouette ---\n")
print(best_k)
k <- best_k

kmeans_res <- kmeans(dtm_matrix, centers = k, nstart = 25)
hclust_res <- hclust(dist(dtm_matrix), method = "ward.D2")
k_hc <- k
hc_clusters <- cutree(hclust_res, k = k_hc)

pca <- prcomp_irlba(dtm_matrix, n = 50)
pca_scores <- as.data.frame(pca$x)

kNNdistplot(pca_scores[, 1:20], k = 15)   
abline(h = 1, col = "red", lty = 2)       

dbscan_res <- dbscan(pca_scores[, 1:20], eps = 1, minPts = 15)

cat("\n--- K-Means Cluster Counts ---\n")
print(table(kmeans_res$cluster))
cat("\n--- Hierarchical Cluster Counts ---\n")
print(table(hc_clusters))
cat("\n--- DBSCAN Cluster Counts ---\n")
print(table(dbscan_res$cluster))


  pca_2d <- pca_scores[, 1:2]
colnames(pca_2d) <- c("PC1", "PC2")
pca_2d$kmeans_cluster <- factor(kmeans_res$cluster)
pca_2d$hc_cluster <- factor(hc_clusters)
pca_2d$dbscan_cluster <- factor(dbscan_res$cluster)

ggplot(pca_2d, aes(PC1, PC2, color = kmeans_cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  ggtitle("K-Means Clustering (PCA)")
ggplot(pca_2d, aes(PC1, PC2, color = hc_cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  scale_color_manual(values = rainbow(k_hc)) +
  ggtitle("Hierarchical Clustering (PCA)")
ggplot(pca_2d, aes(PC1, PC2, color = dbscan_cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  scale_color_manual(values = c("grey", rainbow(length(unique(dbscan_res$cluster))-1))) +
  ggtitle("DBSCAN Clustering (PCA)")
pca_2d_long <- pca_2d %>%
  pivot_longer(cols = c(kmeans_cluster, hc_cluster, dbscan_cluster),
               names_to = "algorithm", values_to = "cluster")
ggplot(pca_2d_long, aes(PC1, PC2, color = cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  facet_wrap(~algorithm) +
  ggtitle("Comparison of Clustering Algorithms")


make_wordclouds <- function(cluster_assignments, method_name) {
  tfidf_fixed <- tfidf %>%
    mutate(doc_id = as.character(doc_id))
  
  if (is.null(names(cluster_assignments))) {
    names(cluster_assignments) <- as.character(unique(tfidf_fixed$doc_id))
  }
  
  cluster_words <- tfidf_fixed %>%
    inner_join(
      tibble(doc_id = as.character(names(cluster_assignments)),
             cluster = cluster_assignments),
      by = "doc_id"
    )
  
  top_cluster_terms <- cluster_words %>%
    group_by(cluster, word) %>%
    summarise(mean_tfidf = mean(tf_idf), .groups = "drop")
  
  unique_clusters <- sort(unique(top_cluster_terms$cluster))
  n_clusters <- length(unique_clusters)
  
  par(mfrow = c(ceiling(n_clusters/2), 2))
  for (cl in unique_clusters) {
    terms <- top_cluster_terms %>%
      filter(cluster == cl) %>%
      arrange(desc(mean_tfidf)) %>%
      head(100)
    
    wordcloud(words = terms$word,
              freq = terms$mean_tfidf,
              min.freq = 1,
              max.words = 100,
              random.order = FALSE,
              rot.per = 0.35,
              colors = brewer.pal(8, "Dark2"))
    title(paste(method_name, "- Cluster", cl))
  }
}

make_wordclouds(kmeans_res$cluster, "K-Means")
make_wordclouds(hc_clusters, "Hierarchical")
make_wordclouds(dbscan_res$cluster, "DBSCAN")



plot_top_terms <- function(cluster_assignments, method_name, top_n = 10) {
  tfidf_fixed <- tfidf %>%
    mutate(doc_id = as.character(doc_id))
  
  if (is.null(names(cluster_assignments))) {
    names(cluster_assignments) <- as.character(unique(tfidf_fixed$doc_id))
  }
  
  cluster_words <- tfidf_fixed %>%
    inner_join(
      tibble(doc_id = as.character(names(cluster_assignments)),
             cluster = cluster_assignments),
      by = "doc_id"
    )
  
  top_cluster_terms <- cluster_words %>%
    group_by(cluster, word) %>%
    summarise(mean_tfidf = mean(tf_idf), .groups = "drop") %>%
    group_by(cluster) %>%
    slice_max(mean_tfidf, n = top_n)
  
  ggplot(top_cluster_terms, aes(x = reorder_within(word, mean_tfidf, cluster),
                                y = mean_tfidf, fill = factor(cluster))) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~cluster, scales = "free") +
    coord_flip() +
    scale_x_reordered() +
    labs(title = paste("Top", top_n, "terms per cluster -", method_name),
         x = "Terms", y = "Average TF-IDF")
}
plot_top_terms(kmeans_res$cluster, "K-Means")
plot_top_terms(hc_clusters, "Hierarchical")
plot_top_terms(dbscan_res$cluster, "DBSCAN")

