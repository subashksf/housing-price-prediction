# housing-price-prediction
This repository contains the machine learning model to predict the sale price of a house based on the dataset aquired from [kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

## Dataset
The dataset contains the below attributes:-

+ SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
+ MSSubClass: The building class
+ MSZoning: The general zoning classification
+ LotFrontage: Linear feet of street connected to property
+ LotArea: Lot size in square feet
+ Street: Type of road access
+ Alley: Type of alley access
+ LotShape: General shape of property
+ LandContour: Flatness of the property
+ Utilities: Type of utilities available
+ LotConfig: Lot configuration
+ LandSlope: Slope of property
+ Neighborhood: Physical locations within Ames city limits
+ Condition1: Proximity to main road or railroad
+ Condition2: Proximity to main road or railroad (if a second is present)
+ BldgType: Type of dwelling
+ HouseStyle: Style of dwelling
+ OverallQual: Overall material and finish quality
+ OverallCond: Overall condition rating
+ YearBuilt: Original construction date
+ YearRemodAdd: Remodel date
+ RoofStyle: Type of roof
+ RoofMatl: Roof material
+ Exterior1st: Exterior covering on house
+ Exterior2nd: Exterior covering on house (if more than one material)
+ MasVnrType: Masonry veneer type
+ MasVnrArea: Masonry veneer area in square feet
+ ExterQual: Exterior material quality
+ ExterCond: Present condition of the material on the exterior
+ Foundation: Type of foundation
+ BsmtQual: Height of the basement
+ BsmtCond: General condition of the basement
+ BsmtExposure: Walkout or garden level basement walls
+ BsmtFinType1: Quality of basement finished area
+ BsmtFinSF1: Type 1 finished square feet
+ BsmtFinType2: Quality of second finished area (if present)
+ BsmtFinSF2: Type 2 finished square feet
+ BsmtUnfSF: Unfinished square feet of basement area
+ TotalBsmtSF: Total square feet of basement area
+ Heating: Type of heating
+ HeatingQC: Heating quality and condition
+ CentralAir: Central air conditioning
+ Electrical: Electrical system
+ 1stFlrSF: First Floor square feet
+ 2ndFlrSF: Second floor square feet
+ LowQualFinSF: Low quality finished square feet (all floors)
+ GrLivArea: Above grade (ground) living area square feet
+ BsmtFullBath: Basement full bathrooms
+ BsmtHalfBath: Basement half bathrooms
+ FullBath: Full bathrooms above grade
+ HalfBath: Half baths above grade
+ Bedroom: Number of bedrooms above basement level
+ Kitchen: Number of kitchens
+ KitchenQual: Kitchen quality
+ TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
+ Functional: Home functionality rating
+ Fireplaces: Number of fireplaces
+ FireplaceQu: Fireplace quality
+ GarageType: Garage location
+ GarageYrBlt: Year garage was built
+ GarageFinish: Interior finish of the garage
+ GarageCars: Size of garage in car capacity
+ GarageArea: Size of garage in square feet
+ GarageQual: Garage quality
+ GarageCond: Garage condition
+ PavedDrive: Paved driveway
+ WoodDeckSF: Wood deck area in square feet
+ OpenPorchSF: Open porch area in square feet
+ EnclosedPorch: Enclosed porch area in square feet
+ 3SsnPorch: Three season porch area in square feet
+ ScreenPorch: Screen porch area in square feet
+ PoolArea: Pool area in square feet
+ PoolQC: Pool quality
+ Fence: Fence quality
+ MiscFeature: Miscellaneous feature not covered in other categories
+ MiscVal: $Value of miscellaneous feature
+ MoSold: Month Sold
+ YrSold: Year Sold
+ SaleType: Type of sale
+ SaleCondition: Condition of sale

## Data Preprocessing
I used the below data preprocessing techniques:-
+ dropped the columns that have more than 70% values as null.
+ identified columns that have same value in more than 80% of the records, so that I could delete them, as they do not add any meaningful value to the model.
+ replaced null values in numerical fields with the mean.
+ replaced null values in categorical fields with the mode.
+ deleted the independent variables with high correlation between themselves.
+ lebel encoded categorial values

## Models
I built two models to compare the performance and efficiency - Random Forests and XGBoost, using cross-validation technique for determining the optimal hyperparameters for both of them. The model evaluation metrics can be found in the notebook.

# Initial Feature Engineering and Modeling
The initial feature engineering and modeling processes are documented in the Jupyter notebook house_price_prediction_regression.ipynb. This notebook contains exploratory data analysis (EDA), data preprocessing steps, feature engineering techniques, and initial model building. It serves as a comprehensive guide to understanding the initial stages of the project.

# Finalized Model
The finalized model, along with tuned hyperparameters, is implemented in the Python project. To execute the model, follow these steps:

1. Ensure you have installed all dependencies by running:
pip install -r requirements.txt

2. Run the main Python script main.py:
python main.py

Executing main.py will load the finalized model, apply it to the test data, and produce the desired output.

This modular approach allows for a clear separation between the exploratory and development phases in the Jupyter notebook and the production-ready model in the Python project.
