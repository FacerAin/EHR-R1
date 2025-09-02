"""Database schema definitions for EHR-R1 project."""

from typing import Dict, Optional


# MIMIC-IV database schema based on actual database structure
MIMIC_IV_SCHEMA = """CREATE TABLE patients (
    row_id integer, -- internal row identifier
    subject_id integer, -- unique identifier for each patient, example: [10000032, 10000980]
    gender text, -- patient gender (M/F), example: ['F', 'M']
    dob timestamp, -- date of birth
    dod timestamp, -- date of death if applicable, example: ['2180-07-20', NULL]
    PRIMARY KEY (subject_id)
);

CREATE TABLE admissions (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier, example: [10000032, 10000980] 
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    admittime timestamp, -- admission time, example: ['2180-07-19 06:55:00', '2119-06-22 19:28:00']
    dischtime timestamp, -- discharge time, example: ['2180-07-20 17:55:00', '2119-06-26 17:31:00']
    admission_type text, -- type of admission, example: ['URGENT', 'ELECTIVE']
    admission_location text, -- admission location, example: ['TRANSFER FROM HOSP/EXTRAM', 'PHYSICIAN REFERRAL/NORMAL DELI']
    discharge_location text, -- discharge location, example: ['DEAD/EXPIRED', 'HOME HEALTH CARE']
    insurance text, -- insurance type, example: ['Other', 'Medicare']
    language text, -- preferred language, example: ['ENGLISH', 'SPANISH']
    marital_status text, -- marital status, example: ['MARRIED', 'SINGLE']
    age integer, -- patient age at admission, example: [78, 55]
    PRIMARY KEY (hadm_id),
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id)
);

CREATE TABLE icustays (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    stay_id integer, -- ICU stay identifier, example: [39553978, 35841562]
    first_careunit text, -- first care unit, example: ['Coronary Care Unit (CCU)', 'Medical Intensive Care Unit (MICU)']
    last_careunit text, -- last care unit, example: ['Coronary Care Unit (CCU)', 'Medical Intensive Care Unit (MICU)']
    intime timestamp, -- ICU in time, example: ['2180-07-19 10:16:00', '2119-06-22 21:27:00']
    outtime timestamp, -- ICU out time, example: ['2180-07-20 17:28:00', '2119-06-26 15:54:00']
    PRIMARY KEY (stay_id),
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE chartevents (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    stay_id integer, -- ICU stay identifier, example: [39553978, 35841562]
    itemid integer, -- item identifier from d_items, example: [220045, 223762]
    charttime timestamp, -- chart time, example: ['2180-07-19 11:00:00', '2119-06-22 22:00:00']
    valuenum real, -- numeric value, example: [80.0, 36.7]
    valueuom text, -- unit of measurement, example: ['bpm', 'Deg. C']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id),
    FOREIGN KEY (stay_id) REFERENCES icustays (stay_id),
    FOREIGN KEY (itemid) REFERENCES d_items (itemid)
);

CREATE TABLE labevents (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    itemid integer, -- lab item identifier from d_labitems, example: [50868, 50882]
    charttime timestamp, -- chart time, example: ['2180-07-19 09:45:00', '2119-06-23 04:29:00']
    valuenum real, -- numeric lab value, example: [104.0, 7.4]
    valueuom text, -- unit of measurement, example: ['mEq/L', 'units']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id),
    FOREIGN KEY (itemid) REFERENCES d_labitems (itemid)
);

CREATE TABLE prescriptions (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    starttime timestamp, -- prescription start time, example: ['2180-07-19 19:00:00', '2119-06-23 10:00:00']
    stoptime timestamp, -- prescription stop time, example: ['2180-07-20 17:00:00', '2119-06-26 10:00:00']
    drug text, -- drug name, example: ['Heparin', 'Normal Saline']
    dose_val_rx text, -- dose value, example: ['5000', '1000']
    dose_unit_rx text, -- dose unit, example: ['UNIT', 'mL']
    route text, -- administration route, example: ['IV', 'PO']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE diagnoses_icd (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    icd_code text, -- ICD diagnosis code, example: ['A419', 'I5020']
    charttime timestamp, -- diagnosis time
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE procedures_icd (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    icd_code text, -- ICD procedure code, example: ['5A1935Z', '02703DZ']
    charttime timestamp, -- procedure time
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE inputevents (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier
    hadm_id integer, -- hospital admission identifier
    stay_id integer, -- ICU stay identifier
    starttime timestamp, -- input start time
    itemid integer, -- item identifier from d_items
    totalamount real, -- total amount given
    totalamountuom text, -- unit of measurement
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id),
    FOREIGN KEY (stay_id) REFERENCES icustays (stay_id),
    FOREIGN KEY (itemid) REFERENCES d_items (itemid)
);

CREATE TABLE outputevents (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier
    hadm_id integer, -- hospital admission identifier
    stay_id integer, -- ICU stay identifier
    charttime timestamp, -- output time
    itemid integer, -- item identifier from d_items
    value real, -- output value
    valueuom text, -- unit of measurement
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id),
    FOREIGN KEY (stay_id) REFERENCES icustays (stay_id),
    FOREIGN KEY (itemid) REFERENCES d_items (itemid)
);

CREATE TABLE microbiologyevents (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier
    hadm_id integer, -- hospital admission identifier
    charttime timestamp, -- specimen collection time
    spec_type_desc text, -- specimen type description, example: ['BLOOD', 'URINE']
    test_name text, -- test name, example: ['BLOOD CULTURE', 'URINE CULTURE']
    org_name text, -- organism name if detected, example: ['STAPHYLOCOCCUS AUREUS', NULL]
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE transfers (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier
    hadm_id integer, -- hospital admission identifier
    transfer_id integer, -- transfer identifier
    eventtype text, -- event type, example: ['admit', 'transfer', 'discharge']
    careunit text, -- care unit, example: ['Emergency Department', 'Medical ICU']
    intime timestamp, -- transfer in time
    outtime timestamp, -- transfer out time
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE cost (
    row_id integer, -- internal row identifier
    subject_id integer, -- patient identifier
    hadm_id integer, -- hospital admission identifier
    event_type text, -- event type, example: ['diagnoses_icd', 'procedures_icd']
    event_id integer, -- event identifier
    chargetime timestamp, -- charge time
    cost real, -- cost amount
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE d_items (
    row_id integer, -- internal row identifier
    itemid integer, -- item identifier, example: [220045, 223762]
    label text, -- item label, example: ['Heart Rate', 'Temperature Celsius']
    abbreviation text, -- item abbreviation, example: ['HR', 'Temp C']
    linksto text, -- links to table, example: ['chartevents', 'inputevents']
    PRIMARY KEY (itemid)
);

CREATE TABLE d_labitems (
    row_id integer, -- internal row identifier
    itemid integer, -- lab item identifier, example: [50868, 50882]
    label text, -- lab item label, example: ['Anion Gap', 'Bicarbonate']
    PRIMARY KEY (itemid)
);

CREATE TABLE d_icd_diagnoses (
    row_id integer, -- internal row identifier
    icd_code text, -- ICD diagnosis code, example: ['A419', 'I5020']
    long_title text, -- diagnosis description, example: ['Sepsis, unspecified organism', 'Acute systolic heart failure']
    PRIMARY KEY (icd_code)
);

CREATE TABLE d_icd_procedures (
    row_id integer, -- internal row identifier
    icd_code text, -- ICD procedure code, example: ['5A1935Z', '02703DZ']
    long_title text, -- procedure description
    PRIMARY KEY (icd_code)
);"""


def get_schema(db_name: str = "mimic_iv") -> Optional[str]:
    """Get database schema by name.
    
    Args:
        db_name: Database name
        
    Returns:
        Database schema as CREATE TABLE statements or None if not found
    """
    schemas = {
        "mimic_iv": MIMIC_IV_SCHEMA,
    }
    
    return schemas.get(db_name.lower())


def load_schema_from_file(schema_path: str) -> str:
    """Load schema from file.
    
    Args:
        schema_path: Path to schema file
        
    Returns:
        Database schema content
    """
    try:
        with open(schema_path, 'r') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Could not load schema from {schema_path}: {e}")