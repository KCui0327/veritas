import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
def find_duplicates(embeddings, sentences, similarity_threshold=0.8, batch_size=1000):
    n_samples = len(embeddings)
    duplicates = set()
    
    print(f"Processing {n_samples} samples in batches of {batch_size}")
    
    # Process in batches to avoid memory issues
    for i in range(0, n_samples, batch_size):
        if i % (batch_size * 10) == 0:  # Print progress every 10 batches
            print(f"Processing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}")
        end_idx = min(i + batch_size, n_samples)
        batch_embeddings = embeddings[i:end_idx]
        
        # Calculate similarity between current batch and all samples
        similarities = cosine_similarity(batch_embeddings, embeddings)
        
        # Find duplicates within this batch
        for j in range(len(batch_embeddings)):
            global_idx = i + j
            
            # Skip if already marked as duplicate
            if global_idx in duplicates:
                continue
                
            # Find similar items (excluding self)
            similar_indices = np.where(similarities[j] > similarity_threshold)[0]
            similar_indices = similar_indices[similar_indices != global_idx]
            
            for similar_idx in similar_indices:
                if similar_idx not in duplicates:
                    print(f"Similar pair found:")
                    print(f"  Index {global_idx}: {sentences.iloc[global_idx][:100]}...")
                    print(f"  Index {similar_idx}: {sentences.iloc[similar_idx][:100]}...")
                    print(f"  Similarity: {similarities[j][similar_idx]:.4f}")
                    print("-" * 50)
                    
                    # Keep the one with lower index, mark the other as duplicate
                    if similar_idx > global_idx:
                        duplicates.add(similar_idx)
                    else:
                        duplicates.add(global_idx)
                        break  # This item is now a duplicate, skip further comparisons
    
    return duplicates

# Load the model
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2') 

# Load data
print("Loading dataset...")
df = pd.read_csv("data/veritas_dataset.csv")
sentences = df['statement']
print(f"Loaded {len(sentences)} sentences")

# Generate embeddings
print("Generating embeddings...")
start_time = time.time()
embeddings = model.encode(sentences.tolist(), show_progress_bar=True, batch_size=32)
end_time = time.time()
print(f"Time for embeddings: {end_time-start_time:.2f} seconds")

# Save embeddings
print("Saving embeddings...")
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.to_csv("src/dataset/processor/embeddings.csv", index=False)
print("Embeddings saved successfully!")

# Find duplicates
print("Finding duplicates...")
start_time = time.time()
duplicates = find_duplicates(embeddings, sentences, similarity_threshold=0.8, batch_size=500)
end_time = time.time()
print(f"Time for deduplication: {end_time-start_time:.2f} seconds")

print(f"Found {len(duplicates)} duplicate entries")

# Create deduplicated dataset
print("Creating deduplicated dataset...")
deduplicated_indices = [i for i in range(len(df)) if i not in duplicates]
deduplicated_df = df.iloc[deduplicated_indices].reset_index(drop=True)

# Save deduplicated dataset
output_path = "data/veritas_dataset_deduplicated.csv"
deduplicated_df.to_csv(output_path, index=False)
print(f"Deduplicated dataset saved to {output_path}")
print(f"Original dataset: {len(df)} entries")
print(f"Deduplicated dataset: {len(deduplicated_df)} entries")
print(f"Removed {len(df) - len(deduplicated_df)} duplicate entries")




