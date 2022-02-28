EXTERNAL_CENSUS_FEATURES = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']
EXTERNAL_SHOPPERS_FEATURES = ['Revenue']
SIGNIFICANCE_LEVEL = 0.99
SAMPLE_SIZE = 20000
SHOPPERS_DATA_TYPES = {
    'Administrative': int,
    'Administrative_Duration': float,
    'Informational': int,
    'Informational_Duration': float,
    'ProductRelated': int,
    'ProductRelated_Duration': float,
    'BounceRates': float,
    'ExitRates': float,
    'PageValues': float,
    'SpecialDay': float,
    'Month': object,
    'OperatingSystems': object,
    'Browser': object,
    'Region': object,
    'TrafficType': object,
    'VisitorType': object,
    'Weekend': object,
    'Revenue': object
}