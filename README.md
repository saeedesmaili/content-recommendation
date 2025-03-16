# Personal Content Recommendation System

## Project Overview

A personalized content recommendation system that analyzes previously liked content (from sources like Pocket) to predict whether new content would be of interest. The system uses natural language processing and embedding similarity to make these predictions.

## Problem Statement

When saving content from across the web, it's difficult to determine which new articles or posts would align with personal interests. This project aims to:

1. Analyze a corpus of previously liked content
2. Build a model of personal interests based on content embeddings
3. Compare new content against this model to predict likelihood of interest
4. Account for diverse interest categories (e.g., photography, programming)

## Proposed Approach

### Data Processing

1. ✅ Get the list of the saved articles from Pocket
2. ✅ Extract the text content from saved articles using r.jina.ai or readibility
3. ✅ Summarize the text content into a few paragraphs using an LLM (gemini flash or a local LLM)
   - Review the text and summaries, clean up the data
4. Generate embeddings for each content summary using an embeddings API (gemini or openai) or sentence-transformers
5. Store these embeddings for comparison with new content
   - [ ] Where to store the embeddings?
6. Use Github actions to check for the new content from desired sources
   - Hackernews new items
   - r/LocalLLaMA and other subreddits
7. Extract the text, summarize, and generate embeddings for the new content
8. Find the ones that are close to our interests and add them into an RSS feed

### Recommendation Methods

#### 1. Nearest Neighbors Approach

- Calculate similarity (cosine similarity) between a new item and each liked item
- Identify top-k most similar items
- Recommend if similarity exceeds a defined threshold

#### 2. Category Clustering

- Apply clustering algorithms (e.g., k-means) to automatically discover interest clusters
- For new content, identify the closest cluster and measure proximity
- Handles diverse interests without manual categorization

#### 3. Weighted Similarity

- Apply weights based on recency or engagement level with liked content
- Recent interests may be more relevant than historical ones

#### 4. Hybrid Approach (Future Enhancement)

- Combine content-based methods with collaborative filtering if data becomes available

## Implementation Plan

1. Extract and preprocess text from liked content sources
2. Generate embeddings using appropriate NLP models
3. Apply clustering to identify interest groups
4. Develop similarity calculation functions for new content
5. Create a recommendation threshold based on testing
6. Build a simple API or UI for evaluating new content

## Technical Considerations

- Choice of embedding model impacts recommendation quality
- Clustering parameters need tuning for optimal interest group detection
- Periodic retraining may be necessary as interests evolve
- Consider dimensionality reduction techniques for large collections

## Next Steps

- Select embedding model and clustering algorithm
- Implement proof of concept with existing saved content
- Develop evaluation metrics to measure recommendation quality
- Create simple interface for testing recommendations

## Current Limitations

Without examples of uninteresting content, our system might:

- Recommend content that's similar to our liked items but still not interesting to me
- Struggle with ambiguous cases that fall between interest clusters
- Have difficulty establishing clear boundaries for what we don't like

### Solutions to Consider

1. **Collect negative examples**: If possible, gathering examples of content we explicitly don't like would be extremely valuable. Even a small set of "not interested" examples gives our model boundaries to learn from.

2. **Implement a threshold approach**: Without negative examples, we can set stricter similarity thresholds, but I'll need to tune these carefully through testing.

3. **Anomaly detection**: Treat our interests as the "normal" pattern, and anything significantly different as potentially uninteresting.

4. **Implicit negative feedback**: Track content I've seen but chosen not to save/like as implicit negative examples.

5. **Active learning**: Implement a feedback mechanism where we explicitly mark false positives (recommended but not interesting) to improve the system over time.

### Updated Implementation Approach

1. Start with our liked content approach
2. Add a mechanism to collect and incorporate negative examples over time
3. Implement an active learning component where we can provide feedback
4. Consider using a binary classifier (interested/not interested) once we have enough negative examples
