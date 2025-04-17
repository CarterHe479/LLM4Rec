from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import json

# load data
df = pd.read_csv("mini_youtube8m.csv")
df["full_text"] = df["title"] + ". " + df["description"] + ". " + df["tags"]

# load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# encode text
video_embeddings = model.encode(df["full_text"].tolist(), show_progress_bar=True)

# save embeddings
video_id_map = df["video_id"].tolist()
np.save("video_embeddings.npy", video_embeddings)

with open("video_id_map.txt", "w") as f:
    for video_id in video_id_map:
        f.write(video_id + "\n")

# build FAISS index
index = faiss.IndexFlatL2(video_embeddings.shape[1])
index.add(video_embeddings)

# save FAISS index
faiss.write_index(index, "video_embeddings.faiss")

# define user query retrieval
with open("user_behavior.json", "r") as f:
    user_data = json.load(f)

user = user_data["user_1"] # example user
user_query = " ".join(user["preferred_tags"])
query_vec = model.encode([user_query])

# load index and search
index = faiss.read_index("video_embeddings.faiss")
D, I = index.search(query_vec, 5)

# display results
print("User Query:", user_query)
print("Top 5 recommended videos:")
for i in I[0]:
    video_id = video_id_map[i]
    row = df[df["video_id"] == video_id].iloc[0]
    print(f" -{video_id}: {row["title"]} ({row["tags"]})")
