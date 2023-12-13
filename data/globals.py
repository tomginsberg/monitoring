# global variables
import numpy as np

DEFAULT_DATA_PATH = '/mnt/nfs/home/dshift_project/geminaid/processed_data'

GENC_ID = 'genc_id'
ADMIT_DATE_TIME = 'admission_date_time'
DISCHARGE_DATE_TIME = 'discharge_date_time'
HOSPITAL_ID = 'hospital_id'
TRIAGE_DATE_TIME = 'triage_date_time'
RESIDENCE_CODE = 'residence_code'
DISCHARGE_DISPOSITION = 'discharge_disposition'
LENGTH_OF_STAY = 'los'
DIAGNOSIS_CODE = 'diagnosis_code'
DIAGNOSIS_TYPE = 'diagnosis_type'
ER_DIAGNOSIS_CODE = 'er_diagnosis_code'
ER_DIAGNOSIS_TYPE = 'er_diagnosis_type'
VALUE = 'result_value'
MEASUREMENT_CODE = 'measurement_mapped_omop'
MEASUREMENT_DATE_TIME = 'measure_date_time'
MEASUREMENT_VALUE = 'measurement_value'
COLLECTION_DATE_TIME = 'collection_date_time'
TEST_CODE = 'test_type_mapped_omop'
CONCEPT_ID = 'concept_id'
UNIT = 'result_unit'
VOCABULARY_ID = 'vocabulary_id'
COMORBIDITY = '1'
UNIT_MAPPED_CODE = 'unit_mapped_code'
AGE = 'age'

# preprocessing parameters
N_BINS = 8
BIN_LENGTH = 6
MIN_YEAR = 2015
MAX_LOS = 100

# files and directories
CIHI_DIAG_GROUPS = 'tables/cihi_diag_groups.csv'
VITALS_META_PATH = 'tables/vital_meta.pkl'
TOP_VITALS_PATH = 'tables/top_vitals.csv'
LABS_META_PATH = 'tables/lab_meta.pkl'
TOP_LABS_PATH = 'tables/top_labs.csv'


# utils
def transform_to_rule(cats):
    keys = list(cats.keys())
    d = {}
    for k in keys:
        for v in cats[k]:
            d[v] = k

    return d


def _transform_to_rule(cats, other=np.nan):
    d = transform_to_rule(cats)

    def _f(x):
        if x in d:
            return d[x]
        else:
            return other

    return _f


def to_sorted_rule(cats: list[str]):
    cats = sorted(cats)
    d = {v: i for i, v in enumerate(cats)}
    return d


def convert_to_float(val):
    try:
        if isinstance(val, str):
            if val[0] == '<':
                return float(val[1:])
            elif val[0] == '>':
                return float(val[1:])
            else:
                return float(val)
        else:
            return float(val)
    except:
        return float('nan')


# rules
AMBULANCE = transform_to_rule({
    1: ['GROUND AMBULANCE', 'G', 'COMBINATION OF AMBULANCES - Includes Air Ambulance', 'A',
        'AIR AMBULANCE (Heli Pad M-Wing)', 'C',
        'COMBINATION OF AMBULANCES - (Use for Air Ambulance arrived at old Helipad Parking Lot 11)'],
    0: ['N', 'No Ambulance']
})

GENDER = transform_to_rule({
    0: ['M', 'MALE'],
    1: ['F', 'FEMALE']
})

READMISSION = {
    '5': 5,
    '9': 6,
    '2': 2,
    '3': 3,
    '4': 4,
    '1': 1,
}

DISCHARGE = transform_to_rule({
    1: {7, 72, 73, 74},  # died
    0: {4, 5, 6, 12},  # home
    2: {1, 10},  # acute care
    3: {20, 30, 40, 90},  # transfer
    4: {61, 62, 65, 67},  # left AMA
    5: {2, 3, 8, 9}  # other
})

HOSP = to_sorted_rule(['SMH', 'SBK', 'THPM', 'MSH', 'THPC', 'UHNTG', 'UHNTW'])

TRIAGE_LEVEL = transform_to_rule({
    1: ['RESUSCITATION', '1', 'L1'],
    2: ['EMERGENCY', '2', 'L2'],
    3: ['URGENT', '3', 'L3'],
    4: ['SEMI-URGENT', '4', 'L4'],
    5: ['NON-URGENT', '5'],
})

COL_TO_RULE = {
    'admit_via_ambulance': AMBULANCE,
    'gender': GENDER,
    'readmission': READMISSION,
    'discharge_disposition': DISCHARGE,
    'hospital_id': HOSP,
    'triage_level': TRIAGE_LEVEL
}

# key-maps
IP_KEYS = [
    GENC_ID, ADMIT_DATE_TIME,
    DISCHARGE_DATE_TIME, 'hospital_id', 'residence_code',
    'discharge_disposition',
    'readmission', 'gender', 'age'
]
ER_KEYS = [GENC_ID, TRIAGE_DATE_TIME, 'triage_level', 'admit_via_ambulance']
DERIVED_KEYS = [GENC_ID, 'mlaps', 'admit_charlson_derived', 'readmission_7d_derived',
                'readmission_30d_derived', 'covid_icd_confirmed_derived']
DATE_TIME_COLS = [TRIAGE_DATE_TIME, ADMIT_DATE_TIME, DISCHARGE_DATE_TIME]
NA_DROP_COLS = DATE_TIME_COLS + ['gender', 'age']

# Statcan features (derived from da16uid code)
DA16UID = 'da16uid'
STATCAN_FEATURES = [
    'atippe',
    'c16_ed_25to64',
    'c16_lab_empl_rate',
    'c16_popdw_popdens_sqkm',
    'ethniccon_q_da16',
    'instability_q_da16']
INCOME, EDUCATION, EMPLOYMENT, POPULATION_DENSITY, ETHNIC_CONCENTRATION, INSTABILITY = STATCAN_FEATURES
STATCAN_HAS_OUTLIER_FEATURES = [INCOME, EDUCATION, EMPLOYMENT, POPULATION_DENSITY]
LOWER_QUANTILE = 0.01
UPPER_QUANTILE = 0.99

# Median length of stay
MEDIAN_LOS = 5.03
