# Telecom Support Ticket Classification using NLP

## Project Overview
This project presents a Natural Language Processing (NLP) solution for the automatic classification of customer support tickets in the telecommunications domain. The goal is to assist customer support operations by automatically routing incoming text-based requests to the appropriate support teams.

The solution is designed in the context of a large telecommunications operator such as **Deutsche Telekom**, where thousands of customer inquiries are received daily through digital channels including web forms, email, mobile applications, and live chat systems.

## Business Context
Modern telecommunications providers face increasing pressure to deliver fast and reliable customer support. Manual routing of support tickets is time-consuming, prone to human error, and difficult to scale as the volume of digital interactions grows.

Misrouted tickets lead to:
- Longer resolution times
- Increased operational costs
- Lower customer satisfaction and Net Promoter Score (NPS)

This project explores an automated, NLP-driven approach to support ticket triage as a practical and scalable solution.

## Project Objectives
The main objectives of this project are:
- To build an NLP-based system that automatically classifies customer support tickets into predefined categories.
- To demonstrate the applicability of classical machine learning techniques in enterprise telecommunications environments.
- To ensure transparency, interpretability, and ease of integration.
- To analyze model performance and extract operational insights.

## Dataset Description
The project uses a **curated dataset of customer support tickets**, reflecting typical inquiry patterns found in telecommunications customer service operations.

Dataset characteristics:
- Total records: 400 customer support tickets
- Languages: German and English
- Text format: short free-text descriptions submitted by customers
- Categories:
  - `billing` – billing and payment-related inquiries
  - `network` – connectivity and performance issues
  - `device` – SIM card and hardware-related problems
  - `contract` – tariff changes, cancellations, and contract questions
  - `other` – general inquiries and portal navigation

All data used in this project is **fully anonymized** and does not contain personally identifiable information (PII).

## Methodology Overview
The project follows a modular and reproducible NLP pipeline:

1. **Data Loading and Validation**
   - Structured dataset loading and integrity checks.

2. **Text Preprocessing**
   - Lowercasing
   - Punctuation removal
   - Stop-word removal for German and English

3. **Feature Extraction**
   - TF-IDF vectorization using unigrams and bigrams.

4. **Model Training**
   - Logistic Regression classifier optimized for short-text classification.

5. **Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix analysis to identify misclassification patterns.

This methodology prioritizes practical deployment considerations over experimental complexity.

## Project Structure
```
telecom-support-nlp/
├── data/
│ └── support_tickets_clean.csv
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── model.py
│ ├── utils.py
│ └── visualization.py
├── reports/
│ └── figures/
│ ├── figure_1_distribution.png
│ ├── figure_2_confusion_matrix.png
│ ├── figure_3_pipeline.png
│ └── figure_5_metrics.png
├── outputs/
│ └── model.joblib
├── main.py
├── demo_simple.py
├── requirements.txt
└── README.md
```


## How to Run the Project

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```
   On Windows:

   ```cmd
   .venv\Scripts\activate
   ```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Generate dataset distribution plot (Figure 1):
```bash
python src/visualization.py
```
4. Run the main pipeline (generates Figures 2, 4 and saves the model):
```bash
python main.py
```
5. Launch the interactive demo:
```bash
python demo_simple.py
```
Enter customer support tickets in English or German
Get real-time category predictions (billing, contract, device, network, other)
Press 'n' to exit
The script trains the model, evaluates performance, and generates visualizations.

## Project Structure

The `demo_simple.py` script provides a lightweight command-line interface for real-time ticket classification:

- Supports both English and German input
- Uses the trained model from `outputs/model.joblib`
- Includes example tickets for quick testing
- Runs in any environment (terminal, PyCharm, Jupyter via `%run`)

This demo illustrates how the model could be integrated into enterprise workflows such as CRM systems or agent dashboards.

## Results Summary

The model demonstrates strong baseline performance suitable for automated ticket triage. 
Most classification errors occur between semantically similar categories, such as `network` and `device`, which reflects realistic overlap in customer issue descriptions. 
The trained model achieves an accuracy of **98%** on the test set (n=100).  

The pipeline generates the following visualizations in `reports/figures/`:

- `figure_1_distribution.png` – Class distribution
- `figure_2_confusion_matrix.png` – Model performance
- `figure_3_pipeline.png` – End-to-End ML Pipeline Architecture
- `figure_4_metrics.png` – Per-class performance metrics (Precision, Recall, F1-score)

> **Note**:  
> - The "Top Keywords per Category" analysis is presented as **Table 1** in the Capstone report (Section 6.2).  
> - The "Example Predictions" analysis is presented as **Table 2** in the Capstone report (Section 6.3).  
> Neither table is saved as a separate image file.

These figures and tables are used in the final Capstone report (Sections 3–6).

## Limitations and Future Work

Current limitations include:
- Limited dataset size
- Absence of continuous feedback loops
- Simplified multilingual preprocessing

Future improvements may include:
- Fine-tuning transformer-based models
- Integration with CRM systems
- Real-time inference pipelines
- Continuous retraining with live data

## Conclusion

This project demonstrates how NLP techniques can be applied to automate customer support ticket classification in telecommunications. The solution improves operational efficiency while maintaining transparency and business relevance, making it suitable as a foundation for enterprise AI systems.

## References

1. Gartner. (2023). AI in Telecommunications: Trends in Customer Experience Automation.  
   https://www.gartner.com/en/articles/ai-in-telecom-trends

2. Scikit-learn Developers. (2024). Text Classification Examples.  
   https://scikit-learn.org/stable/auto_examples/text/index.html

3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.  
   https://nlp.stanford.edu/IR-book/

4. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O’Reilly Media.

5. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?" Explaining the Predictions of Any Classifier. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD).
