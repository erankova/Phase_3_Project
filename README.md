# Phase_3_Project

#### David Johnson and Elina Rankova

<div style="width: 100%; text-align: center;">
  <img src="https://media.licdn.com/dms/image/C4D12AQH-Qk_eTZv6iA/article-cover_image-shrink_720_1280/0/1622638706496?e=1715212800&v=beta&t=a3k-gWfqrlzfv7inhVBpUU9xxPjnJ0of4viF4tFu-Oc" width="720" height="450" style="margin: 0 auto;"/>
</div>

<u>image source</u>: <a href="https://www.linkedin.com/pulse/churn-analysis-smriti-saini/">Churn Analysis Article</a>

# Business Problem and Understanding

**Stakeholders:** Director of Member Operations, Member Operations Manager, Member Retention Manager, Member Support Manager

The business problem at hand is to predict customer churn for SyriaTel, a telecommunications company, in order to minimize revenue loss and enhance customer retention efforts. With customer attrition posing a significant challenge to profitability in the telecom industry, SyriaTel seeks to identify patterns and trends within its customer base that indicate potential churn. By leveraging historical data and predictive modeling techniques, the aim is to develop a classifier that can accurately forecast which customers are likely to discontinue their services, enabling SyriaTel to implement targeted retention strategies and ultimately strengthen its competitive position in the market.

**The goal:** Create a model to predict churn in telecom members contacting support. We are aiming to reduce the amount of cases in which members are mistakenly identified as retained (false negative) vs mistakenly identified as churned to ensure we capture all members who may churn (positive).

# Data Understanding and Exploration

For this analysis, the SyriaTel churn data was sourced from <a href = "https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset">Kaggle</a>

The dataset contains data on the customers of a Telecom company. Each row represents a customer and the columns contain customer’s attributes such as minutes, number of calls, and charge for each time of day and international. In addition we also have information about the customer's voicemail and customer call behavior.

### Observations:
- The dataset has no missingness and most columns are numeric
- Of the 3,333 customers in this dataset, 483 terminated their contract with SyriaTel
- Transforming target `churn` as well as `international_plan` and `voice_mail_plan` to binary 0/1 is needed
- `state` appears numeric but is actually categorical and needs to be transformed as well
- `phone_number` can be dropped as there are no duplicate entries

#### Class Imbalance
This is an imbalanced dataset, with 14.5% of customers lost, balancing will be necessary 

![alt text] (Phase_3_Project/Images/Class Imbalance.jpeg)

#### There are several correlations worth noting:
`total_intl_charge`, `total_day_charge`, total_eve_charge`, and `total_night_charge` is perfectly correlated with `total_intl_minutes`, `total_day_minutes`, total_eve_minutes`, and `total_night_minutes` respectively. This makes sense since the company is charging by the minute. 
> If we need to, we can confidently drop the 'charge' column from each category; day, eve, night, and intl. We can keep the 'minutes' category as it is unclear what currency metric 'charge' is referring to.

In addition, there is a near perfect correlation between `number_vmail_messages` and `voice_mail_plan`, this makes sense and these two columns much like 'charge' and 'minutes' are telling us the same thing. If we need to, we can drop `number_vmail_messages`.

Lastly, there are a couple of weak correlations associated with our target `churn` variable; It seems `customer_service_calls`, `international_plan` and `total_day_minutes` have a slight positive correlation with churn. While weak correlations, we would want to consider including these features in our models.

![alt text](Phase_3_Project/Images/Data Exploration Heatmap.jpeg)

# Data Preperation

To prepare the data for modeling, several steps had to be taken as described below.

## Model 1 & 2
Given our selected approach to these `LogisticRegression` models, we had slightly different steps applied depending on the model.

#### Pre-Split

Before splitting our data between train and test, we performed some simple processing:

- Since the column names were formatted with a space between words, we transformed them to include and underscore as per column name standard formatting
- `LabelEncoder` was used to perform transformtions on the following categorical columns:
  - `churn` orriginally in binary True/False format
  - `international_plan` and `voice_mail_plan` in binary Yes/No format
- Dropped `phone_number` column as there were no duplicate entries as mentioned previously

#### Post-Split

After splitting our data into a train and test we had to perform a couple of other transformations depending on the model criteria.

For the first two models we used `OneHotEncoder` to transform the `area_code` and `state` categorical columns to numerical format. This left us with an `X_train` containing 69 features. In addition, we used `SMOTE` to resample our data and handle class imbalance.

**Model 2 with `SelectFromModel`** to aid with important feature selection we called on this meta-transformer to reduce our 

## Model 3

For our 3rd model we took a manual approach and redefined the DataFrame criteria which lead us to having to conduct a fresh train/test split.

#### Pre-Split

We decided to only include highly correlated variables since we had previously stated there were some features which had extremely high correlations with eachother. For this model we were left with only the features seen below in a heatmap no longer demonstrating any co-linearity.

![alt text](Phase_3_Project/Images/Model 3 Heatmap.jpeg)

Since tranforming column names applies generally to the dataframe, we did not have to repeat this as it was already complete as the first pre-processing step.

#### Post-Split

Since we redefined a new `X` and `y` we also applied `SMOTE` to this fresh data set, creating reduced versious of our training data. 
> Sinced we eliminated any categorical columns in need of transformation, the `OneHotEncoder` was not necessary for this model.

   
