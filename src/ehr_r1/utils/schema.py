"""Database schema definitions for EHR-R1 project."""

from typing import Dict, Optional


# MIMIC-IV database schema with comments and examples
MIMIC_IV_SCHEMA = """CREATE TABLE patients (
    subject_id integer, -- unique identifier for each patient, example: [10000032, 10000980]
    gender text, -- patient gender (M/F), example: ['F', 'M']
    anchor_age integer, -- patient age at anchor_year, example: [17, 78]
    anchor_year integer, -- shifted year for patient, example: [2180, 2119]
    dod date, -- date of death if applicable, example: ['2180-07-20', NULL]
    PRIMARY KEY (subject_id)
);

CREATE TABLE admissions (
    subject_id integer, -- patient identifier, example: [10000032, 10000980] 
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    admittime timestamp, -- admission time, example: ['2180-07-19 06:55:00', '2119-06-22 19:28:00']
    dischtime timestamp, -- discharge time, example: ['2180-07-20 17:55:00', '2119-06-26 17:31:00']
    deathtime timestamp, -- death time if applicable, example: ['2180-07-20 17:28:00', NULL]
    admission_type text, -- type of admission, example: ['URGENT', 'ELECTIVE']
    admission_location text, -- admission location, example: ['TRANSFER FROM HOSP/EXTRAM', 'PHYSICIAN REFERRAL/NORMAL DELI']
    discharge_location text, -- discharge location, example: ['DEAD/EXPIRED', 'HOME HEALTH CARE']
    insurance text, -- insurance type, example: ['Other', 'Medicare']
    language text, -- preferred language, example: ['ENGLISH', 'SPANISH']
    religion text, -- religion, example: ['CATHOLIC', 'NOT SPECIFIED']
    marital_status text, -- marital status, example: ['MARRIED', 'SINGLE']
    ethnicity text, -- ethnicity, example: ['WHITE', 'BLACK/AFRICAN AMERICAN']
    edregtime timestamp, -- ED registration time, example: ['2180-07-19 06:03:00', '2119-06-22 16:50:00']
    edouttime timestamp, -- ED out time, example: ['2180-07-19 07:15:00', '2119-06-22 19:46:00']
    diagnosis text, -- primary diagnosis, example: ['SEPSIS', 'CORONARY ARTERY DISEASE']
    hospital_expire_flag integer, -- died in hospital (0/1), example: [1, 0]
    has_chartevents_data integer, -- has chart events (0/1), example: [1, 1]
    PRIMARY KEY (hadm_id),
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id)
);

CREATE TABLE icustays (
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    stay_id integer, -- ICU stay identifier, example: [39553978, 35841562]
    first_careunit text, -- first care unit, example: ['Coronary Care Unit (CCU)', 'Medical Intensive Care Unit (MICU)']
    last_careunit text, -- last care unit, example: ['Coronary Care Unit (CCU)', 'Medical Intensive Care Unit (MICU)']
    intime timestamp, -- ICU in time, example: ['2180-07-19 10:16:00', '2119-06-22 21:27:00']
    outtime timestamp, -- ICU out time, example: ['2180-07-20 17:28:00', '2119-06-26 15:54:00']
    los real, -- length of stay in days, example: [1.297, 3.768]
    PRIMARY KEY (stay_id),
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE chartevents (
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    stay_id integer, -- ICU stay identifier, example: [39553978, 35841562]
    caregiver_id integer, -- caregiver identifier, example: [NULL, NULL]
    charttime timestamp, -- chart time, example: ['2180-07-19 11:00:00', '2119-06-22 22:00:00']
    storetime timestamp, -- store time, example: ['2180-07-19 13:27:00', '2119-06-22 22:06:00']
    itemid integer, -- item identifier from d_items, example: [220045, 223762]
    value text, -- recorded value, example: ['80', '36.7']
    valuenum real, -- numeric value, example: [80.0, 36.7]
    valueuom text, -- unit of measurement, example: ['bpm', 'Deg. C']
    warning integer, -- warning flag (0/1), example: [0, 0]
    error integer, -- error flag (0/1), example: [0, 0]
    resultstatus text, -- result status, example: ['Final', NULL]
    stopped text, -- stopped status, example: ['NotStopped', NULL]
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id),
    FOREIGN KEY (stay_id) REFERENCES icustays (stay_id),
    FOREIGN KEY (itemid) REFERENCES d_items (itemid)
);

CREATE TABLE labevents (
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    specimen_id integer, -- specimen identifier, example: [90040549, 90013026]
    itemid integer, -- lab item identifier from d_labitems, example: [50868, 50882]
    charttime timestamp, -- chart time, example: ['2180-07-19 09:45:00', '2119-06-23 04:29:00']
    storetime timestamp, -- store time, example: ['2180-07-19 11:52:00', '2119-06-23 05:55:00']
    value text, -- lab value, example: ['104', '7.4']
    valuenum real, -- numeric lab value, example: [104.0, 7.4]
    valueuom text, -- unit of measurement, example: ['mEq/L', 'units']
    ref_range_lower real, -- reference range lower, example: [98.0, 7.35]
    ref_range_upper real, -- reference range upper, example: [107.0, 7.45]
    flag text, -- abnormal flag, example: ['normal', 'abnormal']
    priority text, -- priority level, example: ['ROUTINE', 'STAT']
    comments text, -- lab comments, example: [NULL, 'hemolyzed specimen']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id),
    FOREIGN KEY (itemid) REFERENCES d_labitems (itemid)
);

CREATE TABLE prescriptions (
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    pharmacy_id integer, -- pharmacy identifier, example: [25742516, 13178454]
    starttime timestamp, -- prescription start time, example: ['2180-07-19 19:00:00', '2119-06-23 10:00:00']
    stoptime timestamp, -- prescription stop time, example: ['2180-07-20 17:00:00', '2119-06-26 10:00:00']
    drug_type text, -- type of drug, example: ['MAIN', 'BASE']
    drug text, -- drug name, example: ['Heparin', 'Normal Saline']
    gsn text, -- generic sequence number, example: ['005508', '048348']
    ndc text, -- national drug code, example: ['63323041201', '409488001']
    prod_strength text, -- product strength, example: ['5000 unit/mL', '0.9% Sodium Chloride']
    form_rx text, -- form of prescription, example: ['VIAL', 'BAG']
    dose_val_rx text, -- dose value, example: ['5000', '1000']
    dose_unit_rx text, -- dose unit, example: ['UNIT', 'mL']
    form_val_disp text, -- dispensed form value, example: ['5000', '1000']
    form_unit_disp text, -- dispensed form unit, example: ['UNIT', 'mL']
    doses_per_24_hrs real, -- doses per 24 hours, example: [24.0, 1.0]
    route text, -- administration route, example: ['IV', 'PO']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE diagnoses_icd (
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    seq_num integer, -- diagnosis sequence number, example: [1, 2]
    icd_code text, -- ICD diagnosis code, example: ['A419', 'I5020']
    icd_version integer, -- ICD version (9/10), example: [10, 10]
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE procedures_icd (
    subject_id integer, -- patient identifier, example: [10000032, 10000980]
    hadm_id integer, -- hospital admission identifier, example: [29079034, 26951159]
    seq_num integer, -- procedure sequence number, example: [1, 2]
    chartdate date, -- procedure date, example: ['2180-07-19', '2119-06-23']
    icd_code text, -- ICD procedure code, example: ['5A1935Z', '02703DZ']
    icd_version integer, -- ICD version (9/10), example: [10, 10]
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE d_items (
    itemid integer, -- item identifier, example: [220045, 223762]
    label text, -- item label, example: ['Heart Rate', 'Temperature Celsius']
    abbreviation text, -- item abbreviation, example: ['HR', 'Temp C']
    dbsource text, -- database source, example: ['metavision', 'carevue']
    linksto text, -- links to table, example: ['chartevents', 'chartevents']
    category text, -- item category, example: ['Routine Vital Signs', 'Routine Vital Signs']
    unitname text, -- unit name, example: ['bpm', 'Deg. C']
    param_type text, -- parameter type, example: ['Numeric', 'Numeric']
    conceptid integer, -- concept identifier, example: [NULL, NULL]
    PRIMARY KEY (itemid)
);

CREATE TABLE d_labitems (
    itemid integer, -- lab item identifier, example: [50868, 50882]
    label text, -- lab item label, example: ['Anion Gap', 'Bicarbonate']  
    fluid text, -- specimen fluid type, example: ['Blood', 'Blood']
    category text, -- lab category, example: ['Chemistry', 'Chemistry']
    loinc_code text, -- LOINC code, example: ['33747-0', '1963-8']
    PRIMARY KEY (itemid)
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