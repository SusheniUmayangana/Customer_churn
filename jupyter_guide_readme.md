# Jupyter Notebooks for Customer Churn Prediction

This directory contains comprehensive Jupyter notebooks to guide you through building a customer churn prediction model step by step.

## Notebooks Overview

### 1. `customer_churn_model_jupyter_guide.ipynb`

A complete, end-to-end guide covering all aspects of building a customer churn prediction model:

- Data loading and exploration
- Data preprocessing and cleaning
- Feature engineering
- Data visualization
- Model building and training
- Model evaluation
- Feature importance analysis
- Model deployment preparation

### 2. `target_range_churn_model.ipynb`

A focused notebook specifically for building a Random Forest model with target accuracy in the 80-90% range to prevent overfitting:

- Implementation of regularization techniques
- Cross-validation for generalization checking
- Model evaluation within target accuracy range
- Feature importance analysis
- Model saving for deployment

## Requirements

To run these notebooks, you'll need the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Install the requirements with:

```bash
pip install -r jupyter_requirements.txt
```

## How to Use

1. **Data Preparation**:
   Ensure you have the `botswana_bank_customer_churn.csv` dataset in the same directory as the notebooks.

2. **Running the Notebooks**:

   - Start with `customer_churn_model_jupyter_guide.ipynb` for a comprehensive understanding
   - Use `target_range_churn_model.ipynb` for a focused implementation with controlled accuracy

3. **Execution**:
   Run each cell sequentially from top to bottom. All cells are designed to execute in order without errors.

## Key Features

### Comprehensive Guide Notebook:

- Handles missing values and outliers
- Creates meaningful engineered features
- Balances training data to handle class imbalance
- Implements visualization for data understanding
- Provides detailed model evaluation metrics
- Shows how to save and load models for deployment

### Target Range Model Notebook:

- Implements strong regularization to prevent overfitting
- Uses cross-validation to verify generalization
- Ensures accuracy stays within 80-90% target range
- Analyzes feature importance for business insights
- Saves the final model for production use

## Expected Results

### Model Performance:

- Accuracy: 80-90% (target range)
- Good generalization (low difference between CV and test scores)
- Balanced precision and recall for both classes
- ROC AUC > 0.85

### Business Insights:

- Identification of key factors driving customer churn
- Quantification of feature importance for strategic decisions
- Ready-to-use model for predicting customer churn risk

## Files Generated

Running these notebooks will create:

1. `customer_churn_model.pkl` - The trained model from the comprehensive guide
2. `target_range_random_forest_model.pkl` - The target range model
3. `label_encoders.pkl` - Encoders for categorical variables (from comprehensive guide)

## Troubleshooting

### Common Issues:

1. **Missing Data Files**: Ensure all CSV files are in the correct directory
2. **Package Import Errors**: Install all required packages using the requirements file
3. **Memory Issues**: If working with large datasets, consider using data sampling techniques

### Need Help?

If you encounter any issues:

1. Check that all required packages are installed
2. Verify data files are in the correct location
3. Ensure you're running Python 3.7 or higher

## Next Steps

After running these notebooks, you can:

1. Deploy the saved models in production environments
2. Integrate with the Streamlit dashboard (`enhanced_churn_dashboard.py`)
3. Use Power BI dashboards with the prepared data files
4. Extend the models with additional features or algorithms
