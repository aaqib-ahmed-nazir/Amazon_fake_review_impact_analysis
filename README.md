# Amazon Fake Review Impact Analysis

## Project Structure

```bash
project/
├── scraping/
│   ├── amazon_reviews_scraper.ipynb    
│   └── selectors.yml                   
├── data/
│   ├── books_category_reviews.csv
│   └── beauty_category_reviews.csv
├── analysis/
│   └── analysis.ipynb
└── requirements.txt
```

## Features

- Web scraping of Amazon product reviews
- Text preprocessing and cleaning
- Fake review detection using machine learning
- Comparative analysis between product categories
- Statistical analysis and visualization
- Sentiment analysis
- Pattern detection in review behaviors

## Setup

### Step 1: Clone the repository
    
```bash
git clone https://github.com/yourusername/Amazon_fake_review_impact_analysis.git
cd Amazon_fake_review_impact_analysis
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Data Collection

1. Open `amazon_reviews_scraper.ipynb`
2. Update product URLs in the notebook
3. Run all cells to collect reviews
4. Scraped data will be saved in `data`

### Analysis

1. Open `analysis.ipynb`
2. Run the notebook to:
   - Process the collected reviews
   - Detect potential fake reviews
   - Generate visualizations
   - View comparative statistics

## Key Findings

The analysis provides insights into:

- Fake review patterns across categories
- Impact on product ratings
- Sentiment distribution
- Review length and content patterns
- Temporal patterns in review posting

## Tools & Technologies

- **Python**: 3.9+
- **Jupyter Notebooks**
- **Key Libraries**:
  - `pandas`
  - `scikit-learn`
  - `NLTK`
  - `TextBlob`
  - `Matplotlib/Seaborn`
  - `Requests`
  - `Selectorlib`

## Notes

- Respect Amazon's `robots.txt` and rate limiting
- Results are probabilistic and should be interpreted accordingly
- The model's accuracy depends on training data quality
- Regular updates may be needed as review patterns evolve

## License

This project is licensed under the MIT License - see the [MIT License](https://opensource.org/license/MIT) file for details.

## Disclaimer

This Amazon scraper is intended for educational purposes only. Please ensure that your use of this tool complies with Amazon's terms of service and any applicable laws and regulations.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a pull request
