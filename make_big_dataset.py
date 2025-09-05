# make_big_dataset.py
import pandas as pd

# Load the two datasets
print("Loading the real and fake news files...")
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

# Add a 'label' column: 0 for REAL, 1 for FAKE
true_news['label'] = 0
fake_news['label'] = 1

# We only need the text and the label. Let's use the 'title' column as our text.
# You could also use the 'text' column for more detail, but titles are good for a prototype.
combined_data = pd.concat([
    true_news[['title', 'label']].rename(columns={'title': 'text'}),
    fake_news[['title', 'label']].rename(columns={'title': 'text'})
])

# Shuffle the data so real and fake news are mixed up
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Let's not use all 40,000+ examples for our prototype. Let's take a smaller sample.
# We'll take 1000 real and 1000 fake news titles to keep it fast.
sample_size = 1000
true_sample = combined_data[combined_data['label'] == 0].head(sample_size)
fake_sample = combined_data[combined_data['label'] == 1].head(sample_size)

final_dataset = pd.concat([true_sample, fake_sample])
final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle again

# Save this new, better dataset
final_dataset.to_csv('better_news_data.csv', index=False)
print(f"New dataset created with {len(final_dataset)} examples!")
print(f"Number of REAL news: {len(final_dataset[final_dataset['label'] == 0])}")
print(f"Number of FAKE news: {len(final_dataset[final_dataset['label'] == 1])}")
print("Saved as 'better_news_data.csv'")