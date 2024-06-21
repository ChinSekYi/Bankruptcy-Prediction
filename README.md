# Predict Company Bankruptcy using Machine Learning


## Project motivation
In today's dynamic business landscape, exemplified by recent events such as the bankruptcy of Silicon Valley Bank, the ability to anticipate and mitigate financial risks is crucial for sustainable growth and stability. This project aims to develop a robust predictive model of company bankruptcy, leveraging advanced machine learning algorithms and financial data analysis techniques. By accurately identifying early warning signs of financial distress, this model will empower stakeholders to make informed decisions and implement timely interventions to prevent bankruptcy. This model will also help investors to mitigate financial risks, minimizing their loss. 

## Description
Our project will explore and compare the usage of different machine learning models to predict company bankruptcy. We will evaluate the different models using various metrics, and improve its accuracy for better model performance. 
We will also use custom functions which will work well on our specific dataset. The dataset is about bankruptcy prediction of Polish companies. The bankrupt companies were analyzed in the period 2000-2012, while the still operating companies were evaluated from 2007 to 2013. The dataset consists of 64 features that pertain to various financial ratios and performance metrics that encompass aspects such as profitability, liquidity, solvency, and operational efficiency of a company.

## Proposed solution
The proposed solution involves employing a diverse array of predictive models, including logistic regression, k nearest neighbors and decision trees, each tailored to identify companies at risk of bankruptcy. To gauge effectiveness, logistic regression will serve as the benchmark classifier, ensuring a thorough evaluation of alternative models. We will also leverage ensemble methods including bagging, boosting and random forests to amplify our model's predictive capabilities, while enriching our understanding of machine learning techniques. Through this multifaceted strategy, our goal is to construct a predictive framework that surpasses traditional limitations, equipping stakeholders with nuanced insights to confidently traverse the unpredictable landscape of financial risk.

## Dataset Description
Source: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

Variable Description:
X1	net profit / total assets
X2	total liabilities / total assets
X3	working capital / total assets
X4	current assets / short-term liabilities
X5	[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365
X6	retained earnings / total assets
X7	EBIT / total assets
X8	book value of equity / total liabilities
X9	sales / total assets
X10	equity / total assets
X11	(gross profit + extraordinary items + financial expenses) / total assets
X12	gross profit / short-term liabilities
X13	(gross profit + depreciation) / sales
X14	(gross profit + interest) / total assets
X15	(total liabilities * 365) / (gross profit + depreciation)
X16	(gross profit + depreciation) / total liabilities
X17	total assets / total liabilities
X18	gross profit / total assets
X19	gross profit / sales
X20	(inventory * 365) / sales
X21	sales (n) / sales (n-1)
X22	profit on operating activities / total assets
X23	net profit / sales
X24	gross profit (in 3 years) / total assets
X25	(equity - share capital) / total assets
X26	(net profit + depreciation) / total liabilities
X27	profit on operating activities / financial expenses
X28	working capital / fixed assets
X29	logarithm of total assets
X30	(total liabilities - cash) / sales
X31	(gross profit + interest) / sales
X32	(current liabilities * 365) / cost of products sold
X33	operating expenses / short-term liabilities
X34	operating expenses / total liabilities
X35	profit on sales / total assets
X36	total sales / total assets
X37	(current assets - inventories) / long-term liabilities
X38	constant capital / total assets
X39	profit on sales / sales
X40	(current assets - inventory - receivables) / short-term liabilities
X41	total liabilities / ((profit on operating activities + depreciation) * (12/365))
X42	profit on operating activities / sales
X43	rotation receivables + inventory turnover in days
X44	(receivables * 365) / sales
X45	net profit / inventory
X46	(current assets - inventory) / short-term liabilities
X47	(inventory * 365) / cost of products sold
X48	EBITDA (profit on operating activities - depreciation) / total assets
X49	EBITDA (profit on operating activities - depreciation) / sales
X50	current assets / total liabilities
X51	short-term liabilities / total assets
X52	(short-term liabilities * 365) / cost of products sold)
X53	equity / fixed assets
X54	constant capital / fixed assets
X55	working capital
X56	(sales - cost of products sold) / sales
X57	(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)
X58	total costs /total sales
X59	long-term liabilities / equity
X60	sales / inventory
X61	sales / receivables
X62	(short-term liabilities *365) / sales
X63	sales / short-term liabilities
X64	sales / fixed assets


### Set up virtual environment 
To ensure collaboraters have the same packages and versions installed. 

Install anaconda: https://docs.anaconda.com/free/miniconda/index.html  
For windows installation guide: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html  

To check if conda is installed
```
cd /your directory/..
conda --version
```

Create conda virtual environment
```
conda create -n dev python=3.11 numpy pandas scikit-learn imbalanced-learn matplotlib seaborn scipy black isort plotly tqdm

conda activate dev
```

Work on individual branch 
```
git checkout -b branchname
```

To merge individual branch to main branch
```
git checkout main
git merge branchname
git commit -m "Merge branchname branch into main"
git push origin main
```

### Run Flask app
```
cd Bankruptcy-Prediction
python3 app.py
```

Go to chrome and enter: 
0.0.0.0:5001 for home page  
0.0.0.0:5001/predictdata for bankruptcy prediction page  
