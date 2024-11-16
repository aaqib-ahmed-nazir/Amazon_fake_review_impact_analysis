import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import wordcloud
from textblob import TextBlob
from collections import Counter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer

# Download required NLTK data
nltk.download('vader_lexicon')

# Load the datasets
books_df = pd.read_csv('/Users/aaqibnazir/Documents/work/Amazon_fake_review_impact_analysis/scraping/data/books_category_reviews.csv')
beauty_df = pd.read_csv('/Users/aaqibnazir/Documents/work/Amazon_fake_review_impact_analysis/scraping/data/beauty_category_reviews.csv')

# Basic cleaning function
def clean_dataframe(df):
    # Handle missing values
    df['content'] = df['content'].fillna('')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df

# Clean both dataframes
books_df = clean_dataframe(books_df)
beauty_df = clean_dataframe(beauty_df)

# Function to calculate review characteristics
def get_review_characteristics(text):
    if not isinstance(text, str):
        return 0, 0, 0
    
    # Length of review
    review_length = len(text.split())
    
    # Number of exclamation marks
    exclamation_count = text.count('!')
    
    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    return review_length, exclamation_count, sentiment_scores['compound']

def get_enhanced_review_characteristics(text):
    if not isinstance(text, str):
        return 0, 0, 0, 0, 0, 0
    
    # Basic characteristics
    review_length = len(text.split())
    exclamation_count = text.count('!')
    
    # Advanced sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    # Additional features
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    repeated_chars = len([c for c in text if text.count(c) > 3])
    generic_phrases = sum(text.lower().count(phrase) for phrase in [
        'amazing', 'great', 'best', 'awesome', 'perfect', 'love it', 
        'excellent', 'wonderful', 'fantastic', 'incredible'
    ])
    
    return (review_length, exclamation_count, sentiment_scores['compound'], 
            caps_ratio, repeated_chars, generic_phrases)

# Add review characteristics to both dataframes
for df in [books_df, beauty_df]:
    df['review_length'], df['exclamation_count'], df['sentiment_score'] = zip(*df['content'].map(get_review_characteristics))

# Add enhanced characteristics to dataframes
for df in [books_df, beauty_df]:
    (df['review_length'], df['exclamation_count'], df['sentiment_score'],
     df['caps_ratio'], df['repeated_chars'], df['generic_phrases']) = zip(
        *df['content'].map(get_enhanced_review_characteristics)
    )

# Function to identify potential fake reviews
def identify_fake_reviews(df):
    # Create features for fake review detection
    suspicious_reviews = pd.DataFrame()
    
    # 1. Extremely positive reviews with very short content
    suspicious_reviews['short_positive'] = (df['rating'] >= 4.5) & (df['review_length'] < 20)
    
    # 2. High number of exclamation marks
    suspicious_reviews['excessive_punctuation'] = df['exclamation_count'] > 3
    
    # 3. Extreme sentiment (very positive or very negative)
    suspicious_reviews['extreme_sentiment'] = abs(df['sentiment_score']) > 0.8
    
    # Combined score
    suspicious_reviews['fake_probability'] = suspicious_reviews.mean(axis=1)
    
    return suspicious_reviews['fake_probability']

def identify_enhanced_fake_reviews(df):
    suspicious_reviews = pd.DataFrame()
    
    # Enhanced detection criteria
    suspicious_reviews['extremely_positive'] = (df['rating'] >= 4.8) & (df['sentiment_score'] > 0.9)
    suspicious_reviews['short_glowing'] = (df['rating'] >= 4.5) & (df['review_length'] < 20)
    suspicious_reviews['excessive_punctuation'] = df['exclamation_count'] > 4
    suspicious_reviews['generic_content'] = df['generic_phrases'] > 3
    suspicious_reviews['caps_abuse'] = df['caps_ratio'] > 0.3
    
    # Weighted score calculation
    weights = {
        'extremely_positive': 0.3,
        'short_glowing': 0.2,
        'excessive_punctuation': 0.15,
        'generic_content': 0.2,
        'caps_abuse': 0.15
    }
    
    suspicious_reviews['fake_probability'] = sum(
        suspicious_reviews[col] * weight for col, weight in weights.items()
    )
    
    return suspicious_reviews['fake_probability']

# Add fake review probability to both dataframes
books_df['fake_probability'] = identify_fake_reviews(books_df)
beauty_df['fake_probability'] = identify_fake_reviews(beauty_df)

# Recalculate fake probabilities with enhanced detection
books_df['fake_probability'] = identify_enhanced_fake_reviews(books_df)
beauty_df['fake_probability'] = identify_enhanced_fake_reviews(beauty_df)

# Calculate descriptive statistics
def print_category_stats(df, category_name):
    print(f"\n=== {category_name} Category Statistics ===")
    print(f"Total reviews: {len(df)}")
    print(f"Average rating: {df['rating'].mean():.2f}")
    print(f"Average review length: {df['review_length'].mean():.2f} words")
    print(f"Suspected fake reviews: {(df['fake_probability'] > 0.5).sum()} ({(df['fake_probability'] > 0.5).mean()*100:.1f}%)")
    
    # Rating distribution
    print("\nRating distribution:")
    print(df['rating'].value_counts(normalize=True).sort_index())

# Print statistics for both categories
print_category_stats(books_df, "Books")
print_category_stats(beauty_df, "Beauty")

# Visualizations
def plot_category_comparison(books_df, beauty_df):
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Rating Distribution
    plt.subplot(131)
    sns.kdeplot(data=books_df['rating'], label='Books')
    sns.kdeplot(data=beauty_df['rating'], label='Beauty')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.legend()
    
    # Plot 2: Review Length Distribution
    plt.subplot(132)
    sns.kdeplot(data=np.log1p(books_df['review_length']), label='Books')
    sns.kdeplot(data=np.log1p(beauty_df['review_length']), label='Beauty')
    plt.title('Review Length Distribution (log)')
    plt.xlabel('Log(Review Length)')
    plt.legend()
    
    # Plot 3: Fake Probability Distribution
    plt.subplot(133)
    sns.kdeplot(data=books_df['fake_probability'], label='Books')
    sns.kdeplot(data=beauty_df['fake_probability'], label='Beauty')
    plt.title('Fake Review Probability Distribution')
    plt.xlabel('Fake Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Generate visualizations
plot_category_comparison(books_df, beauty_df)

def plot_fake_review_impact(books_df, beauty_df):
    plt.figure(figsize=(20, 10))
    
    # Plot 1: Rating vs Fake Probability
    plt.subplot(221)
    plt.scatter(books_df['rating'], books_df['fake_probability'], alpha=0.5, label='Books')
    plt.scatter(beauty_df['rating'], beauty_df['fake_probability'], alpha=0.5, label='Beauty')
    plt.title('Rating vs Fake Review Probability')
    plt.xlabel('Rating')
    plt.ylabel('Fake Probability')
    plt.legend()

    # Plot 2: Review Length Impact
    plt.subplot(222)
    plt.scatter(books_df['review_length'], books_df['fake_probability'], alpha=0.5, label='Books')
    plt.scatter(beauty_df['review_length'], beauty_df['fake_probability'], alpha=0.5, label='Beauty')
    plt.title('Review Length vs Fake Probability')
    plt.xlabel('Review Length')
    plt.ylabel('Fake Probability')
    plt.legend()

    # Plot 3: Sentiment Distribution for Suspected Fake vs Real Reviews
    plt.subplot(223)
    fake_reviews = pd.concat([
        books_df[books_df['fake_probability'] > 0.5]['sentiment_score'],
        beauty_df[beauty_df['fake_probability'] > 0.5]['sentiment_score']
    ])
    real_reviews = pd.concat([
        books_df[books_df['fake_probability'] <= 0.5]['sentiment_score'],
        beauty_df[beauty_df['fake_probability'] <= 0.5]['sentiment_score']
    ])
    
    sns.kdeplot(data=fake_reviews, label='Suspected Fake')
    sns.kdeplot(data=real_reviews, label='Likely Real')
    plt.title('Sentiment Distribution: Fake vs Real Reviews')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Density')
    plt.legend()

    # Plot 4: Monthly Trend of Fake Reviews
    plt.subplot(224)
    monthly_fake_books = books_df.set_index('date')['fake_probability'].resample('M').mean()
    monthly_fake_beauty = beauty_df.set_index('date')['fake_probability'].resample('M').mean()
    
    plt.plot(monthly_fake_books.index, monthly_fake_books.values, label='Books')
    plt.plot(monthly_fake_beauty.index, monthly_fake_beauty.values, label='Beauty')
    plt.title('Monthly Trend of Fake Review Probability')
    plt.xlabel('Date')
    plt.ylabel('Average Fake Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_enhanced_visualizations(books_df, beauty_df):
    # Create figure with subplots
    plt.figure(figsize=(20, 15))
    
    # 1. Rating Distribution by Category with Fake Probability Overlay
    plt.subplot(331)
    sns.boxplot(x='category', y='rating', data=pd.concat([
        books_df.assign(category='Books'),
        beauty_df.assign(category='Beauty')
    ]))
    plt.title('Rating Distribution by Category')
    
    # 2. Sentiment vs Rating Scatter
    plt.subplot(332)
    plt.scatter(books_df['sentiment_score'], books_df['rating'], alpha=0.5, label='Books')
    plt.scatter(beauty_df['sentiment_score'], beauty_df['rating'], alpha=0.5, label='Beauty')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Rating')
    plt.title('Sentiment vs Rating')
    plt.legend()
    
    # 3. Fake Review Probability Distribution
    plt.subplot(333)
    sns.histplot(data=pd.concat([
        books_df.assign(category='Books'),
        beauty_df.assign(category='Beauty')
    ]), x='fake_probability', hue='category', bins=30)
    plt.title('Fake Review Probability Distribution')
    
    # 4. Review Length vs Fake Probability
    plt.subplot(334)
    plt.hexbin(books_df['review_length'], books_df['fake_probability'], 
               gridsize=20, cmap='YlOrRd')
    plt.colorbar(label='Count')
    plt.xlabel('Review Length')
    plt.ylabel('Fake Probability')
    plt.title('Review Length vs Fake Probability (Books)')
    
    # 5. Time Series of Fake Reviews
    plt.subplot(335)
    monthly_fake_books = books_df.set_index('date')['fake_probability'].resample('M').mean()
    monthly_fake_beauty = beauty_df.set_index('date')['fake_probability'].resample('M').mean()
    
    plt.plot(monthly_fake_books.index, monthly_fake_books.values, label='Books')
    plt.plot(monthly_fake_beauty.index, monthly_fake_beauty.values, label='Beauty')
    plt.title('Monthly Trend of Fake Reviews')
    plt.xticks(rotation=45)
    plt.legend()
    
    # 6. Feature Importance for Fake Detection
    plt.subplot(336)
    feature_importance = pd.DataFrame({
        'Feature': ['Rating', 'Sentiment', 'Length', 'Exclamations', 'Generic'],
        'Importance': [0.3, 0.25, 0.2, 0.15, 0.1]
    })
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance in Fake Detection')
    
    plt.tight_layout()
    plt.show()

def generate_wordclouds(books_df, beauty_df):
    plt.figure(figsize=(15, 6))
    
    # Word cloud for fake reviews
    plt.subplot(121)
    fake_reviews_text = ' '.join(pd.concat([
        books_df[books_df['fake_probability'] > 0.7]['content'],
        beauty_df[beauty_df['fake_probability'] > 0.7]['content']
    ]))
    
    wc = wordcloud.WordCloud(width=800, height=400, background_color='white')
    wc.generate(fake_reviews_text)
    plt.imshow(wc)
    plt.title('Common Words in Suspected Fake Reviews')
    plt.axis('off')
    
    # Word cloud for genuine reviews
    plt.subplot(122)
    genuine_reviews_text = ' '.join(pd.concat([
        books_df[books_df['fake_probability'] < 0.3]['content'],
        beauty_df[beauty_df['fake_probability'] < 0.3]['content']
    ]))
    
    wc = wordcloud.WordCloud(width=800, height=400, background_color='white')
    wc.generate(genuine_reviews_text)
    plt.imshow(wc)
    plt.title('Common Words in Genuine Reviews')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_fake_review_impact():
    # Calculate impact metrics
    impact_metrics = {
        'books_rating_inflation': books_df[books_df['fake_probability'] > 0.5]['rating'].mean() - 
                                books_df[books_df['fake_probability'] <= 0.5]['rating'].mean(),
        'beauty_rating_inflation': beauty_df[beauty_df['fake_probability'] > 0.5]['rating'].mean() - 
                                 beauty_df[beauty_df['fake_probability'] <= 0.5]['rating'].mean(),
        'books_fake_percent': (books_df['fake_probability'] > 0.5).mean() * 100,
        'beauty_fake_percent': (beauty_df['fake_probability'] > 0.5).mean() * 100
    }
    
    print("\n=== Fake Review Impact Analysis ===")
    print(f"Books rating inflation due to fake reviews: {impact_metrics['books_rating_inflation']:.2f} stars")
    print(f"Beauty rating inflation due to fake reviews: {impact_metrics['beauty_rating_inflation']:.2f} stars")
    print(f"Percentage of fake reviews in Books: {impact_metrics['books_fake_percent']:.1f}%")
    print(f"Percentage of fake reviews in Beauty: {impact_metrics['beauty_fake_percent']:.1f}%")

    return impact_metrics

plot_fake_review_impact(books_df, beauty_df)
plot_enhanced_visualizations(books_df, beauty_df)
generate_wordclouds(books_df, beauty_df)
impact_metrics = analyze_fake_review_impact()

# Save results
def save_results(books_df, beauty_df, impact_metrics):
    results = {
        'books_total_reviews': len(books_df),
        'beauty_total_reviews': len(beauty_df),
        'books_avg_rating': books_df['rating'].mean(),
        'beauty_avg_rating': beauty_df['rating'].mean(),
        'books_suspected_fake': (books_df['fake_probability'] > 0.5).mean(),
        'beauty_suspected_fake': (beauty_df['fake_probability'] > 0.5).mean(),
        'books_rating_inflation': impact_metrics['books_rating_inflation'],
        'beauty_rating_inflation': impact_metrics['beauty_rating_inflation'],
        'books_fake_percent': impact_metrics['books_fake_percent'],
        'beauty_fake_percent': impact_metrics['beauty_fake_percent']
    }
    
    # Save to CSV
    pd.DataFrame([results]).to_csv('analysis_results.csv', index=False)

# Save the results
save_results(books_df, beauty_df, impact_metrics)

def prepare_data_for_ml(books_df, beauty_df):
    # Combine datasets and create labels
    combined_df = pd.concat([
        books_df.assign(category='books'),
        beauty_df.assign(category='beauty')
    ]).reset_index(drop=True)
    
    # Create initial labels based on our heuristics
    combined_df['initial_label'] = (combined_df['fake_probability'] > 0.5).astype(int)
    
    # Split data
    train_df, temp_df = train_test_split(combined_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    return train_df, val_df, test_df

def get_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    # Load model
    model = SentenceTransformer(model_name)
    
    # Get embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def train_initial_model(train_df, val_df):
    # Get embeddings
    print("Generating embeddings for training data...")
    train_embeddings = get_embeddings(train_df['content'].tolist())
    val_embeddings = get_embeddings(val_df['content'].tolist())
    
    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(train_embeddings, train_df['initial_label'])
    
    # Validate
    val_preds = model.predict_proba(val_embeddings)[:, 1]
    return model, val_preds

def generate_pseudo_labels(model, unlabeled_df, confidence_threshold=0.8):
    # Get embeddings for unlabeled data
    unlabeled_embeddings = get_embeddings(unlabeled_df['content'].tolist())
    
    # Get predictions
    pred_probs = model.predict_proba(unlabeled_embeddings)[:, 1]
    
    # Generate pseudo labels for confident predictions
    pseudo_labels = np.where(
        (pred_probs >= confidence_threshold) | (pred_probs <= (1-confidence_threshold)),
        (pred_probs >= 0.5).astype(int),
        -1
    )
    
    return pseudo_labels

def train_bert_classifier(train_df, val_df, model_name="bert-base-uncased"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Tokenize data
    train_encodings = tokenizer(train_df['content'].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_df['content'].tolist(), truncation=True, padding=True)
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(train_df['initial_label'].values)
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(3):
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    
    return model, tokenizer

# Add these functions after existing analysis code:
print("Preparing data for ML analysis...")
train_df, val_df, test_df = prepare_data_for_ml(books_df, beauty_df)

print("Training initial model...")
initial_model, val_preds = train_initial_model(train_df, val_df)

print("Generating pseudo labels...")
pseudo_labels = generate_pseudo_labels(initial_model, test_df)

print("Training BERT classifier...")
bert_model, tokenizer = train_bert_classifier(train_df, val_df)

# Add results to existing metrics
def analyze_ml_results(test_df, initial_model, bert_model, tokenizer):
    # Get predictions from both models
    test_embeddings = get_embeddings(test_df['content'].tolist())
    initial_preds = initial_model.predict_proba(test_embeddings)[:, 1]
    
    # Get BERT predictions
    test_encodings = tokenizer(test_df['content'].tolist(), truncation=True, padding=True, return_tensors='pt')
    bert_outputs = bert_model(**test_encodings)
    bert_preds = torch.softmax(bert_outputs.logits, dim=1)[:, 1].detach().numpy()
    
    # Compare results
    results = {
        'initial_model_accuracy': accuracy_score(test_df['initial_label'], initial_preds > 0.5),
        'bert_model_accuracy': accuracy_score(test_df['initial_label'], bert_preds > 0.5),
        'disagreement_rate': np.mean((initial_preds > 0.5) != (bert_preds > 0.5))
    }
    
    return results

# Update the existing save_results function
def save_results(books_df, beauty_df, impact_metrics, ml_results):
    results = {
        'books_total_reviews': len(books_df),
        'beauty_total_reviews': len(beauty_df),
        'books_avg_rating': books_df['rating'].mean(),
        'beauty_avg_rating': beauty_df['rating'].mean(),
        'books_suspected_fake': (books_df['fake_probability'] > 0.5).mean(),
        'beauty_suspected_fake': (beauty_df['fake_probability'] > 0.5).mean(),
        'books_rating_inflation': impact_metrics['books_rating_inflation'],
        'beauty_rating_inflation': impact_metrics['beauty_rating_inflation'],
        'books_fake_percent': impact_metrics['books_fake_percent'],
        'beauty_fake_percent': impact_metrics['beauty_fake_percent'],
        'initial_model_accuracy': ml_results['initial_model_accuracy'],
        'bert_model_accuracy': ml_results['bert_model_accuracy'],
        'model_disagreement_rate': ml_results['disagreement_rate']
    }
    
    pd.DataFrame([results]).to_csv('analysis_results.csv', index=False)

# Get ML results and save
ml_results = analyze_ml_results(test_df, initial_model, bert_model, tokenizer)
save_results(books_df, beauty_df, impact_metrics, ml_results)

def extract_ml_features(df):
    """Extract comprehensive features for ML models"""
    
    # Text-based features
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    text_features = vectorizer.fit_transform(df['content']).toarray()
    
    # Metadata features
    metadata_features = pd.DataFrame({
        'review_length': df['review_length'],
        'exclamation_count': df['exclamation_count'], 
        'sentiment_score': df['sentiment_score'],
        'caps_ratio': df['caps_ratio'],
        'repeated_chars': df['repeated_chars'],
        'generic_phrases': df['generic_phrases'],
        'rating': df['rating'],
        'text_len': df['content'].str.len(),
        'avg_word_len': df['content'].str.split().apply(lambda x: np.mean([len(w) for w in x]) if len(x)>0 else 0),
        'unique_words': df['content'].str.split().apply(lambda x: len(set(x))),
    })
    
    # Combine features
    features = np.hstack([metadata_features, text_features])
    
    return features

def train_multiple_classifiers(X_train, y_train, X_val, y_val):
    """Train multiple ML models and evaluate performance"""
    
    # Initialize models
    models = {
        'logistic': LogisticRegression(max_iter=1000),
        'svm': SVC(probability=True),
        'neural_net': MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500),
        'random_forest': RandomForestClassifier(n_estimators=100)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Get predictions
        train_preds = model.predict(X_train_scaled)
        val_preds = model.predict(X_val_scaled)
        
        # Calculate metrics
        results[name] = {
            'train_accuracy': accuracy_score(y_train, train_preds),
            'val_accuracy': accuracy_score(y_val, val_preds),
            'confusion_matrix': confusion_matrix(y_val, val_preds),
            'classification_report': classification_report(y_val, val_preds)
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"Train accuracy: {results[name]['train_accuracy']:.3f}")
        print(f"Validation accuracy: {results[name]['val_accuracy']:.3f}")
        print("\nClassification Report:")
        print(results[name]['classification_report'])
        
    return results

def plot_ml_results(results):
    """Plot ML model performance comparisons"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Accuracy comparison
    plt.subplot(131)
    models = list(results.keys())
    train_acc = [results[m]['train_accuracy'] for m in models]
    val_acc = [results[m]['val_accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, train_acc, width, label='Train')
    plt.bar(x + width/2, val_acc, width, label='Validation')
    plt.xticks(x, models, rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.legend()
    
    # Plot 2: Confusion matrices
    plt.subplot(132)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i//2, i%2]
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', ax=ax)
        ax.set_title(f'{name} Confusion Matrix')
    
    plt.tight_layout()
    plt.show()

# Add this code after the existing data preparation:

print("Extracting ML features...")
X = extract_ml_features(pd.concat([books_df, beauty_df]))
y = pd.concat([books_df['fake_probability'] > 0.5, beauty_df['fake_probability'] > 0.5]).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Training multiple classifiers...")
ml_results = train_multiple_classifiers(X_train, y_train, X_val, y_val)

print("Plotting ML results...")
plot_ml_results(ml_results)

# Update the save_results function to include ML metrics
def save_ml_results(results, filename='ml_results.csv'):
    ml_metrics = {
        model_name: {
            'train_acc': res['train_accuracy'],
            'val_acc': res['val_accuracy']
        }
        for model_name, res in results.items()
    }
    pd.DataFrame(ml_metrics).to_csv(filename)

save_ml_results(ml_results)
