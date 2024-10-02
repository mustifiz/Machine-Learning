
# Insurance Charges Prediction

This project involves predicting insurance charges using various regression models. The dataset used contains information about individuals' demographics, health, and lifestyle factors.

## Dataset

The dataset includes the following columns:
- **age**: Age of the individual
- **sex**: Gender of the individual
- **bmi**: Body Mass Index
- **children**: Number of children
- **smoker**: Smoking status
- **region**: Residential region
- **charges**: Insurance charges

## Models Used

The following regression models were used to predict insurance charges:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

## Results

The performance of each model was evaluated using Mean Squared Error (MSE), R-squared (R²), and Cross-Validated Root Mean Squared Error (CV RMSE). The results are as follows:

{outputs_dict['9dbaf635']}

## Feature Importance

Feature importance was analyzed using the Random Forest model, which highlighted the most significant factors influencing insurance charges:

{outputs_dict['3e339f3a']}
{outputs_dict['447bc04f']}
{outputs_dict['f62fbafc']}
{outputs_dict['9ea768a3']}
{outputs_dict['c5af0a12']}
{outputs_dict['4e7f6426']}
{outputs_dict['5cc3b670']}
{outputs_dict['2d0e0071']}
{outputs_dict['3e05d370']}

## Visualizations

The R² scores and feature importances are visualized below:

![R2 Scores]({image_urls_dict['6350f08e']})
![Feature Importances]({image_urls_dict['e10ea23a']})

## Conclusion

The analysis demonstrates that the 'smoker' status is the most significant predictor of insurance charges, followed by 'bmi' and 'age'. The Random Forest model provided the best performance among the models tested.

