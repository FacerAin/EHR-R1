"""Database schema definitions for EHR-R1 project."""

from typing import Dict, Optional

# MIMIC-IV database schema based on actual database structure
MIMIC_IV_SCHEMA = """CREATE TABLE patients (
    row_id integer,
    subject_id integer, -- example: [10014729]
    gender text, -- example: ['f']
    dob timestamp, -- example: ['2079-07-22 00:00:00']
    dod timestamp, -- example: ['2119-06-22 00:00:00']
    PRIMARY KEY (subject_id)
);

CREATE TABLE admissions (
    row_id integer,
    subject_id integer, -- example: [10004235] 
    hadm_id integer, -- example: [24181354]
    admittime timestamp, -- example: ['2100-03-19 14:38:00']
    dischtime timestamp, -- example: ['2100-03-28 14:02:00']
    admission_type text, -- example: ['urgent']
    admission_location text, -- example: ['transfer from hospital']
    discharge_location text, -- example: ['skilled nursing facility']
    insurance text, -- example: ['medicaid']
    language text, -- example: ['english']
    marital_status text, -- example: ['single']
    age integer, -- example: [47]
    PRIMARY KEY (hadm_id),
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id)
);

CREATE TABLE icustays (
    row_id integer,
    subject_id integer, -- example: [10018328]
    hadm_id integer, -- example: [23786647]
    stay_id integer, -- example: [31269608]
    first_careunit text, -- example: ['neuro stepdown']
    last_careunit text, -- example: ['neuro stepdown']
    intime timestamp, -- example: ['2100-05-07 23:03:44']
    outtime timestamp, -- example: ['2100-05-15 15:55:21']
    PRIMARY KEY (stay_id),
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE chartevents (
    row_id integer,
    subject_id integer, -- example: [10005817]
    hadm_id integer, -- example: [20626031]
    stay_id integer, -- example: [32604416]
    itemid integer, -- example: [220210]
    charttime timestamp, -- example: ['2100-12-24 00:00:00']
    valuenum real, -- example: [19.0]
    valueuom text, -- example: ['insp/min']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id),
    FOREIGN KEY (stay_id) REFERENCES icustays (stay_id),
    FOREIGN KEY (itemid) REFERENCES d_items (itemid)
);

CREATE TABLE labevents (
    row_id integer,
    subject_id integer, -- example: [10031757]
    hadm_id integer, -- example: [28477280]
    itemid integer, -- example: [50970]
    charttime timestamp, -- example: ['2100-10-25 02:00:00']
    valuenum real, -- example: [2.8]
    valueuom text, -- example: ['mg/dl']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id),
    FOREIGN KEY (itemid) REFERENCES d_labitems (itemid)
);

CREATE TABLE prescriptions (
    row_id integer,
    subject_id integer, -- example: [10020740]
    hadm_id integer, -- example: [23831430]
    starttime timestamp, -- example: ['2100-04-19 11:00:00']
    stoptime timestamp, -- example: ['2100-04-20 22:00:00']
    drug text, -- example: ['insulin']
    dose_val_rx text, -- example: ['5000']
    dose_unit_rx text, -- example: ['unit']
    route text, -- example: ['sc']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE diagnoses_icd (
    row_id integer,
    subject_id integer, -- example: [10035185]
    hadm_id integer, -- example: [22580999]
    icd_code text, -- example: ['icd9|4139']
    charttime timestamp, -- example: ['2100-05-17 12:53:00']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE procedures_icd (
    row_id integer,
    subject_id integer, -- example: [10011398]
    hadm_id integer, -- example: [27505812]
    icd_code text, -- example: ['icd9|3961']
    charttime timestamp, -- example: ['2100-12-30 13:37:00']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE inputevents (
    row_id integer,
    subject_id integer,
    hadm_id integer,
    stay_id integer,
    starttime timestamp,
    itemid integer,
    totalamount real,
    totalamountuom text,
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id),
    FOREIGN KEY (stay_id) REFERENCES icustays (stay_id),
    FOREIGN KEY (itemid) REFERENCES d_items (itemid)
);

CREATE TABLE outputevents (
    row_id integer,
    subject_id integer,
    hadm_id integer,
    stay_id integer,
    charttime timestamp,
    itemid integer,
    value real,
    valueuom text,
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id),
    FOREIGN KEY (stay_id) REFERENCES icustays (stay_id),
    FOREIGN KEY (itemid) REFERENCES d_items (itemid)
);

CREATE TABLE microbiologyevents (
    row_id integer,
    subject_id integer, -- example: [10000032]
    hadm_id integer, -- example: [25742920]
    charttime timestamp, -- example: ['2100-08-26 20:35:00']
    spec_type_desc text, -- example: ['swab']
    test_name text, -- example: ['r/o vancomycin resistant enterococcus']
    org_name text, -- example: ['staphylococcus epidermidis']
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE transfers (
    row_id integer,
    subject_id integer,
    hadm_id integer,
    transfer_id integer,
    eventtype text, -- example: ['admit']
    careunit text, -- example: ['Emergency Department']
    intime timestamp,
    outtime timestamp,
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE cost (
    row_id integer,
    subject_id integer,
    hadm_id integer,
    event_type text, -- example: ['diagnoses_icd']
    event_id integer,
    chargetime timestamp,
    cost real,
    FOREIGN KEY (subject_id) REFERENCES patients (subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions (hadm_id)
);

CREATE TABLE d_items (
    row_id integer,
    itemid integer, -- example: [226228]
    label text, -- example: ['gender']
    abbreviation text, -- example: ['gender']
    linksto text, -- example: ['chartevents']
    PRIMARY KEY (itemid)
);

CREATE TABLE d_labitems (
    row_id integer,
    itemid integer, -- example: [50808]
    label text, -- example: ['free calcium']
    PRIMARY KEY (itemid)
);

CREATE TABLE d_icd_diagnoses (
    row_id integer,
    icd_code text, -- example: ['A419']
    long_title text, -- example: ['Sepsis, unspecified organism']
    PRIMARY KEY (icd_code)
);

CREATE TABLE d_icd_procedures (
    row_id integer,
    icd_code text, -- example: ['5A1935Z']
    long_title text,
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
        with open(schema_path, "r") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Could not load schema from {schema_path}: {e}")
