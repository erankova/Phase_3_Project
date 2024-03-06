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

![alt text](Phase_3_Project/Images/Data Exploration Heatmap.png)
   
