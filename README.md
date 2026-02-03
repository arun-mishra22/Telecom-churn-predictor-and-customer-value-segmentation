# Customer Value–Aware Churn Predictor

##  Project Overview

The **Customer Value–Aware Churn Predictor** is an end-to-end machine learning project designed to help telecom companies predict customer churn while also analyzing the business value of each customer.

###  Core Objective

> **Predict churn ➝ Analyze customer value ➝ Segment customers ➝ Enable smarter business decisions**


This project is developed using a real-world telecom dataset containing **7043 customer records and 21 features**, where the target variable is **Churn (Yes/No)**. The system not only predicts whether a customer is likely to leave the service but also provides deeper insights into customer behavior and long-term value.

Unlike traditional churn prediction models, this project integrates predictive analytics with customer segmentation and value analysis. Customers are grouped based on key business factors such as **"tenure , total charges and monthly charges "**, enabling the company to identify high-value, medium-value, low-value, and very low-value customers.

By combining churn prediction with segmentation, the project helps businesses focus retention strategies on the right customers, optimize marketing efforts, and make smarter data-driven decisions. The result is a complete analytics solution rather than just a simple classification model.


## Problem Statement

Customer churn is a major challenge for telecom companies. Acquiring new customers is far more expensive than retaining existing ones. Without proper analytics, businesses struggle to identify:

- Which customers are likely to leave  
- Which customers are most valuable  
- Where to focus retention efforts  

This project aims to solve these problems by predicting churn in advance and identifying customer segments based on business value.

## Dataset Description

- Domain: Telecom Industry  
- Total Records: 7043  
- Total Features: 21  
- Target Variable: Churn (Yes / No)

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand customer behavior and identify key factors influencing churn. The most important insights are summarized below:

### Key Numerical Insights

- **Tenure and TotalCharges are highly correlated (0.83)**, as expected since total charges accumulate over time.  
- No extreme statistical outliers were observed in **Tenure, MonthlyCharges, or TotalCharges**, ensuring stable model training.  
- **Tenure shows a U-shaped distribution**, indicating that both new customers and very long-term customers exhibit different churn behaviors.  
- **MonthlyCharges has a bimodal distribution**, suggesting two major customer groups – low-paying and high-paying customers.  
- **TotalCharges is right-skewed**, which is natural as it represents cumulative spending over time.

### Key Categorical Insights

- The dataset shows class imbalance in certain features:
  - Most customers are **non-senior citizens**  
  - A large majority have **PhoneService enabled**  
  - Most customers are on **month-to-month contracts**, rather than long-term contracts  

### Churn Behavior Patterns

- Customers with **lower tenure are far more likely to churn**, showing an inverse relationship between tenure and churn.  
- Churned customers generally have **higher MonthlyCharges**, indicating price sensitivity.  
- Customers with **lower TotalCharges tend to churn more**, reinforcing that long-term customers are more loyal.

### Service and Contract Insights

- **Senior citizens churn almost twice as much** as non-senior customers.  
- Customers using **Fiber Optic internet show significantly higher churn** compared to DSL users.  
- Customers **without online security or technical support have much higher churn rates**, highlighting the importance of add-on services.  
- **Month-to-month contract customers have the highest churn**, while long-term contracts (1-year and 2-year) strongly reduce churn risk.

These insights played a crucial role in feature selection, model building, and designing effective customer retention strategies.


## Technologies Used

### Programming Language
- Python  

### Libraries and Tools

#### Data Handling & Analysis
- Pandas  
- NumPy  

#### Data Visualization
- Matplotlib  
- Seaborn  

#### Machine Learning & Modeling
- Scikit-learn  
  - Logistic Regression  
  - Random Forest Classifier  
  - Data Preprocessing (StandardScaler, OneHotEncoder)  
  - Pipeline and ColumnTransformer  
  - Train-Test Split  
  - GridSearchCV  
  - Cross Validation  
  - Stratified K-Fold  

- XGBoost  

#### Clustering and Customer Segmentation
- K-Means Clustering  
- Silhouette Score for cluster evaluation  

#### Model Evaluation
- Accuracy Score  
- Precision Score  
- Recall Score  
- F1 Score  
- Confusion Matrix


## Methodology

The project follows a structured and systematic data science workflow to ensure reliable and business-focused results:

1. **Data Understanding** – Analyzed the telecom dataset to understand feature types, distributions, and relationships with the target variable.

2. **Data Cleaning & Preprocessing** – Handled missing values, encoded categorical variables using OneHotEncoder, and scaled numerical features using StandardScaler.

3. **Exploratory Data Analysis (EDA)** – Performed visual and statistical analysis to identify trends, correlations, and patterns influencing customer churn.

4. **Feature Engineering** – Selected and transformed relevant features to improve model performance and predictive power.

5. **Train-Test Split** – Divided the dataset into training and testing sets to evaluate model generalization.

6. **Model Building** – Implemented multiple machine learning models including Logistic Regression, Random Forest, and XGBoost for churn prediction.

7. **Hyperparameter Tuning** – Optimized model performance using GridSearchCV and cross-validation techniques.

8. **Model Evaluation** – Compared models using key metrics such as Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

9. **Threshold Tuning** – Adjusted the decision threshold to increase recall and reduce false negatives, ensuring that more potential churn customers are correctly identified.

10. **Customer Segmentation** – Applied K-Means clustering to segment customers based on business-relevant features such as tenure and total charges.

11. **Cluster Evaluation** – Used Silhouette Score to determine the optimal number of clusters and validate segmentation quality.

12. **Business Insights Generation** – Combined churn predictions with customer segmentation to derive actionable insights for targeted retention strategies.


## Machine Learning Models

The following classification algorithms were implemented and evaluated for churn prediction:

- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

Each model was trained using cross-validation and optimized using GridSearchCV. The models were compared based on performance metrics to select the most effective one.


## Threshold Tuning

After selecting the best performing model, threshold tuning was performed to improve business effectiveness.

Instead of using the default threshold of 0.5, different probability thresholds were tested with the primary goal of:

- Increasing recall  
- Reducing false negatives  
- Identifying more potential churn customers  

The final threshold was chosen to align model performance with real-world business priorities.

## Results

### Best Performing Model
- Model: Logistic Regression  

After comparing multiple models, Logistic Regression delivered the most balanced and reliable performance for churn prediction.

### Performance After Threshold Tuning

To align the model with business priorities, threshold tuning was performed. The final selected threshold was **0.45**, which significantly improved recall.

#### Final Metrics:

- Threshold: **0.45**  
- Recall: **0.8438**  
- Precision: **0.4641**  
- F1 Score: **0.5988**  

The tuned model successfully identifies a large proportion of actual churn customers, making it highly effective for real-world retention strategies.


## Customer Segmentation

Customer segmentation was performed to better understand different types of customers and their overall business value.

K-Means clustering was applied using key business features:

- Tenure  
- Total Charges  
- Monthly Charges  

The optimal number of clusters was determined using the Silhouette Score to ensure meaningful and well-separated customer groups.

Based on clustering results, customers were divided into four value-based segments:

| Segment Type      | Number of Customers |
|------------------|---------------------|
| Low Value        | 2270                |
| High Value       | 1904                |
| Very Low Value   | 1688                |
| Medium Value     | 1159                |

### Business Interpretation

- **High Value Customers:** Loyal, long-term, and high-paying customers who contribute the most revenue  
- **Medium Value Customers:** Customers with moderate engagement and spending  
- **Low Value Customers:** Customers with relatively low tenure or charges  
- **Very Low Value Customers:** Least engaged customers with minimal long-term business impact  

This segmentation, when combined with churn prediction, enables businesses to:

- Prioritize retention strategies for high-value customers  
- Focus marketing efforts on the right customer groups  
- Personalize offers based on customer value  
- Allocate resources more effectively  
- Reduce revenue loss from critical segments  


## Business Impact

This project provides a complete analytics solution that helps telecom companies to:

- Identify potential churn customers in advance  
- Focus retention efforts on high-value customers  
- Design personalized marketing strategies  
- Reduce revenue loss caused by customer churn  
- Allocate resources more efficiently  
- Make smarter, data-driven business decisions  

## Model Deployment

The churn prediction model has been successfully deployed on **Hugging Face Spaces**, making it accessible as an interactive web application.

Users can input customer details and get real-time predictions along with churn probability scores.

This deployment allows non-technical users to easily utilize the model without running any code.


## Live Demo

The deployed application can be accessed here:

 **Hugging Face App Link:** https://huggingface.co/spaces/aruuuuuuuuunnnn/churn_predictor

Anyone can test the model by entering sample customer details and getting instant churn predictions.

## Deployment Tools

- Hugging Face Spaces  
- Python  
- Scikit-learn  
- Gradio 


## How to Use the Application

1. Open the Hugging Face deployment link  
2. Enter customer details in the input fields  
3. Click on **Predict**  
4. View:
   - Churn prediction result  
   - Probability score  
   - Customer value segment  


## Future Improvements

- Build a real-time prediction pipeline  
- Add deep learning-based models  
- Create an interactive analytics dashboard  
- Automate model retraining with new data   



## Author

**Arun Mishra**  
Aspiring Data Scientist  
Machine Learning | Deep Learning | NLP  
