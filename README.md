# ML4QS-G28

Welcome to **ML4QS-G28**! This repository is the workspace for Group 28's project in the Machine Learning for the Quantified Self (ML4QS) course.

## Overview

This project explores the application of machine learning techniques to Quantified Self (QS) data. Our goal is to analyze, process, and extract meaningful insights from personal sensor data using machine learning algorithms. For more information, see the report (ML4QS_group28_final.pdf). 

## Features
- Data visualization, cleaning and preprocessing for various sensor sources
- Feature extraction and selection
- Implementation of machine learning models for classification 
- Performance evaluation and visualizations

## Repository Structure

```
.
├── data/               # Raw and processed datasets
├── EDA.py              # Data visualization
├── cleaning.py         # Data cleaning + missing value imputation
├── features.py         # Feature extraction (mean, max, FFT, etc.)
├── train_....py        # Training scripts (DT, KKN, LSTM)
├── results.py          # Run all experiments
├── requirements.txt    # Python dependencies
└── README.md           # Project overview and instructions
```

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/LarsdeWolf/ML4QS-G28.git
   cd ML4QS-G28
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data**
   - Download or collect your own QS data and place it in the `data/` directory.

5. **Run the code**
   ```bash
   python results.py
   ```

## Acknowledgements

- This project is a fork of [mhoogen/ML4QS](https://github.com/mhoogen/ML4QS), which provides the base structure and many utilities used in this project.
- Course: Machine Learning for the Quantified Self

## License

This project is for educational purposes and may not have a formal license.

---

*ML4QS-G28 – Machine Learning for the Quantified Self, Group 28*
