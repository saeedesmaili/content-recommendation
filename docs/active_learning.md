## Active learning

Active learning is particularly valuable here because it helps our model improve through targeted feedback.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Union
import pickle
import os

class ContentRecommender:
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        n_clusters: int = 5,
        similarity_threshold: float = 0.5,
        feedback_weight: float = 0.8
    ):
        """
        Initialize the content recommender with active learning capabilities

        Args:
            embedding_model: The sentence transformer model to use for embeddings
            n_clusters: Number of interest clusters to identify
            similarity_threshold: Minimum similarity score to recommend content
            feedback_weight: Weight for user feedback (higher values prioritize explicit feedback)
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.n_clusters = n_clusters
        self.similarity_threshold = similarity_threshold
        self.feedback_weight = feedback_weight

        # Storage for content and feedback
        self.liked_content = []
        self.disliked_content = []
        self.uncertain_content = []  # Content for active learning queries

        # Model components
        self.liked_embeddings = None
        self.disliked_embeddings = None
        self.cluster_model = None
        self.clusters = None
        self.cluster_centers = None

    def add_content(self, content_text: str, content_metadata: Dict, liked: bool = True) -> None:
        """
        Add content with user preference information

        Args:
            content_text: The text of the content to add
            content_metadata: Additional information about the content (title, url, etc.)
            liked: Whether the user liked this content
        """
        content_item = {
            'text': content_text,
            'metadata': content_metadata,
            'embedding': self.embedding_model.encode(content_text)
        }

        if liked:
            self.liked_content.append(content_item)
        else:
            self.disliked_content.append(content_item)

        # Retrain the model if we have enough data
        if len(self.liked_content) >= self.n_clusters:
            self._train_model()

    def _train_model(self) -> None:
        """Train the clustering and similarity models based on current content"""
        # Extract embeddings from liked content
        self.liked_embeddings = np.array([item['embedding'] for item in self.liked_content])

        # Extract embeddings from disliked content if available
        if self.disliked_content:
            self.disliked_embeddings = np.array([item['embedding'] for item in self.disliked_content])

        # Train clustering model on liked content
        if len(self.liked_content) >= self.n_clusters:
            self.cluster_model = KMeans(n_clusters=min(self.n_clusters, len(self.liked_content)))
            self.clusters = self.cluster_model.fit_predict(self.liked_embeddings)
            self.cluster_centers = self.cluster_model.cluster_centers_

    def predict_interest(self, content_text: str) -> Dict:
        """
        Predict whether the user would be interested in the given content

        Args:
            content_text: The text of the content to evaluate

        Returns:
            Dict containing prediction results and confidence
        """
        if not self.liked_embeddings is not None:
            return {'interested': False, 'confidence': 0, 'needs_feedback': True}

        # Get embedding for new content
        content_embedding = self.embedding_model.encode(content_text)

        # Calculate similarity to liked content
        liked_similarities = cosine_similarity([content_embedding], self.liked_embeddings)[0]
        max_liked_similarity = np.max(liked_similarities)
        avg_liked_similarity = np.mean(liked_similarities)

        # Calculate similarity to disliked content if available
        disliked_similarity = 0
        if hasattr(self, 'disliked_embeddings') and self.disliked_embeddings is not None:
            disliked_similarities = cosine_similarity([content_embedding], self.disliked_embeddings)[0]
            disliked_similarity = np.max(disliked_similarities)

        # Find closest cluster
        if self.cluster_centers is not None:
            cluster_similarities = cosine_similarity([content_embedding], self.cluster_centers)[0]
            closest_cluster = np.argmax(cluster_similarities)
            cluster_similarity = cluster_similarities[closest_cluster]
        else:
            cluster_similarity = 0
            closest_cluster = None

        # Calculate composite score
        # Higher when similar to liked, lower when similar to disliked
        if disliked_similarity > 0:
            score = avg_liked_similarity - (self.feedback_weight * disliked_similarity)
        else:
            score = avg_liked_similarity

        # Determine if this is a good candidate for active learning
        uncertainty = 1.0 - abs(score - 0.5) * 2  # Higher when score is close to 0.5
        needs_feedback = uncertainty > 0.6

        # Store content that needs feedback
        if needs_feedback:
            self.uncertain_content.append({
                'text': content_text,
                'embedding': content_embedding,
                'predicted_score': score
            })

            # Limit the uncertain content queue to a reasonable size
            if len(self.uncertain_content) > 20:
                self.uncertain_content = self.uncertain_content[-20:]

        return {
            'interested': score > self.similarity_threshold,
            'confidence': 1.0 - uncertainty,
            'score': score,
            'max_similarity': max_liked_similarity,
            'avg_similarity': avg_liked_similarity,
            'closest_cluster': closest_cluster,
            'cluster_similarity': cluster_similarity if closest_cluster is not None else None,
            'needs_feedback': needs_feedback
        }

    def get_feedback_requests(self, limit: int = 5) -> List[Dict]:
        """
        Get content items that would benefit most from user feedback

        Args:
            limit: Maximum number of items to return

        Returns:
            List of content items with highest uncertainty
        """
        # Sort uncertain content by uncertainty (those closest to 0.5 prediction)
        sorted_uncertain = sorted(
            self.uncertain_content,
            key=lambda x: abs(x['predicted_score'] - 0.5)
        )

        return [{
            'text': item['text'],
            'predicted_score': item['predicted_score']
        } for item in sorted_uncertain[:limit]]

    def provide_feedback(self, content_text: str, liked: bool) -> None:
        """
        Provide explicit feedback on content to improve the model

        Args:
            content_text: The text that was presented to the user
            liked: Whether the user liked the content
        """
        # Find matching content in uncertain items
        matching_items = [item for item in self.uncertain_content
                         if item['text'] == content_text]

        if matching_items:
            # Remove from uncertain content
            self.uncertain_content = [item for item in self.uncertain_content
                                    if item['text'] != content_text]

            # Add to appropriate category with metadata
            content_item = {
                'text': content_text,
                'metadata': {'source': 'feedback'},
                'embedding': matching_items[0]['embedding']
            }

            if liked:
                self.liked_content.append(content_item)
            else:
                self.disliked_content.append(content_item)

            # Retrain model with new feedback
            self._train_model()

    def save_model(self, filepath: str) -> None:
        """Save the recommender to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath: str) -> 'ContentRecommender':
        """Load a recommender from a file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Example usage
def process_content_batch(recommender, new_content_list):
    """Process a batch of new content and identify items for feedback"""
    results = []

    for content in new_content_list:
        prediction = recommender.predict_interest(content['text'])
        results.append({
            'content': content,
            'prediction': prediction
        })

    # Items recommended with high confidence
    confident_recommendations = [
        item for item in results
        if item['prediction']['interested'] and item['prediction']['confidence'] > 0.8
    ]

    # Items that need feedback (uncertain predictions)
    needs_feedback = [
        item for item in results
        if item['prediction']['needs_feedback']
    ]

    return {
        'recommendations': confident_recommendations,
        'feedback_requests': needs_feedback
    }


# Example of how to use the active learning feedback loop
def active_learning_demo():
    # Initialize recommender
    recommender = ContentRecommender(n_clusters=3)

    # Add initial training data
    initial_liked = [
        {"text": "Python's new async features make concurrent programming much easier",
         "metadata": {"category": "programming"}},
        {"text": "How to optimize our photography workflow with Lightroom presets",
         "metadata": {"category": "photography"}},
        {"text": "Building REST APIs with FastAPI and Python",
         "metadata": {"category": "programming"}}
    ]

    initial_disliked = [
        {"text": "Latest political developments in Congress",
         "metadata": {"category": "politics"}},
        {"text": "Celebrity gossip: Who's dating who in Hollywood",
         "metadata": {"category": "entertainment"}}
    ]

    # Train with initial data
    for item in initial_liked:
        recommender.add_content(item["text"], item["metadata"], liked=True)

    for item in initial_disliked:
        recommender.add_content(item["text"], item["metadata"], liked=False)

    # New content to evaluate
    new_content = [
        {"text": "Advanced techniques for landscape photography in low light",
         "metadata": {"category": "photography"}},
        {"text": "Senate debates new tax legislation",
         "metadata": {"category": "politics"}},
        {"text": "Introduction to GraphQL for API development",
         "metadata": {"category": "programming"}},
        {"text": "Using machine learning for image classification",
         "metadata": {"category": "programming/ML"}}
    ]

    # Process the new content
    results = process_content_batch(recommender, new_content)

    print("Recommended content:")
    for item in results['recommendations']:
        print(f"- {item['content']['text']} (Score: {item['prediction']['score']:.2f})")

    print("\nContent needing feedback:")
    for item in results['feedback_requests']:
        print(f"- {item['content']['text']} (Uncertain score: {item['prediction']['score']:.2f})")

        # Simulate user feedback (in a real app, this would come from user interaction)
        is_programming = "programming" in item['content']['metadata'].get('category', '').lower()
        is_photography = "photography" in item['content']['metadata'].get('category', '').lower()
        liked = is_programming or is_photography

        # Provide the feedback to the recommender
        recommender.provide_feedback(item['content']['text'], liked)
        print(f"  → User {'liked' if liked else 'disliked'} this content")

    # Save the trained model
    recommender.save_model("content_recommender.pkl")

    print("\nRecommender model saved.")


if __name__ == "__main__":
    active_learning_demo()

```

### Active Learning Core Concepts

1. **Uncertainty Sampling**

   - The system identifies content where it's most uncertain about our interest level
   - Content with prediction scores near 0.5 (the decision boundary) are flagged for feedback
   - This approach targets feedback where it will be most informative for the model

2. **Balanced Representation**

   - The system maintains separate collections for liked and disliked content
   - This creates a more balanced dataset for learning our preferences

3. **Feedback Integration**
   - When we provide feedback, the content moves from "uncertain" to either liked or disliked
   - The model is retrained with this new information
   - Over time, this refines the decision boundary between interesting and uninteresting content

### Key Implementation Features

1. **Content Representation**

   - Uses SentenceTransformer for high-quality embeddings
   - Maintains embeddings for efficient similarity calculations

2. **Interest Modeling**

   - Combines clustering (for interest categories) with similarity scoring
   - Weighs similarity to liked content against similarity to disliked content

3. **Prediction with Confidence**

   - Returns both a prediction and a confidence score
   - Low confidence triggers feedback requests

4. **Batch Processing**
   - Efficiently processes multiple new content items
   - Separates high-confidence recommendations from uncertain items

### Using the Implementation

The code provides a complete workflow:

1. Initialize the recommender
2. Add initial training data (liked and disliked content)
3. Process new content in batches
4. Request feedback on uncertain predictions
5. Update the model with new feedback
6. Save the trained model for future use

We can integrate this into our project as a module, adding a simple feedback interface to capture our preferences over time. The system will gradually learn the boundaries between content we like and dislike.
