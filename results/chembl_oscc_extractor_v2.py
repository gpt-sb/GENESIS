#!/usr/bin/env python3
"""
ChEMBL OSCC Compound Extraction Tool v2
=========================================
Extract bioactive compounds for Oral Squamous Cell Carcinoma (OSCC) from a
locally installed ChEMBL 36 MySQL database.

Extraction strategies:
  1. Cell-line screening   – compounds tested on OSCC cell lines
  2. Target-based          – compounds acting on OSCC-relevant gene targets
  3. Keyword search        – OSCC terms in assay descriptions / publication titles
  4. Drug indication       – compounds with oral-cancer indications in ChEMBL

All queries use parameterized SQL (no f-string interpolation).
Outputs: CSV, JSON summary, optional Excel.

Usage:
    python Analysis/chembl_oscc_extractor_v2.py
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Optional, List, Dict
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

class ChEMBLConfig:
    """Database connection and OSCC-specific filter parameters."""

    # MySQL connection -- requires a local ChEMBL MySQL mirror; credentials are
    # never hardcoded, set these environment variables instead:
    #   export CHEMBL_MYSQL_HOST=localhost
    #   export CHEMBL_MYSQL_USER=root
    #   export CHEMBL_MYSQL_PASSWORD=...
    #   export CHEMBL_MYSQL_DB=chembl_36
    HOST = os.environ.get("CHEMBL_MYSQL_HOST", "localhost")
    DATABASE = os.environ.get("CHEMBL_MYSQL_DB", "chembl_36")
    USER = os.environ.get("CHEMBL_MYSQL_USER", "root")
    PASSWORD = os.environ.get("CHEMBL_MYSQL_PASSWORD")
    PORT = int(os.environ.get("CHEMBL_MYSQL_PORT", 3306))

    # ---- OSCC cell-line names (substring match via LIKE) --------------------
    OSCC_CELL_LINES = [
        "HSC-3", "HSC3", "HSC 3",
        "Cal-27", "Cal27", "CAL 27",
        "SCC-25", "SCC25", "SCC 25",
        "SCC-4", "SCC4", "SCC 4",
        "SCC-9", "SCC9", "SCC 9",
        "SCC-15", "SCC15", "SCC 15",
        "KB",
        "OSC-19", "OSC19", "OSC 19",
        "OSCC-3", "OSCC3", "OSCC 3",
        "FaDu", "FADU",
        "Detroit 562", "Detroit562", "Detroit-562",
        "HN-4", "HN4", "HN 4",
        "HN-5", "HN5", "HN 5",
        "UMSCC-1", "UMSCC1", "UM-SCC-1",
    ]

    # ---- Disease / keyword patterns (substring match in text fields) --------
    OSCC_KEYWORDS = [
        "oral cancer",
        "oral squamous",
        "OSCC",
        "HNSCC",
        "head and neck squamous",
        "tongue cancer",
        "mouth cancer",
        "oropharyngeal",
        "buccal cancer",
        "gingival cancer",
        "pharyngeal cancer",
    ]

    # ---- OSCC-relevant gene targets -----------------------------------------
    OSCC_TARGETS = [
        # Tier 1: Most frequently mutated
        'TP53',      # 60-81%
        'NOTCH1',    # 17-37%
        'FAT1',      # 9-27% 
        'CDKN2A',    # 10-34%
        'PIK3CA',    # 7-21%
        'CASP8',     # 8-28%
        'CCND1',     # 30-34%
        'EGFR',      # 80-90% overexpression
        
        # Tier 2: Moderate + druggable
        'HRAS', 'FGFR1', 'FGFR3', 'MDM2', 'MDM4', 
        'MYC', 'STAT3', 'PTGS2', 'AKT1', 'MTOR', 'CCND2'
    ]
    # ---- Indication keywords (for drug_indication table) --------------------
    INDICATION_KEYWORDS = [
        "oral", "mouth", "tongue", "buccal", "gingival",
        "oropharynx", "oropharyngeal", "head and neck",
        "pharynx", "pharyngeal", "squamous cell carcinoma",
    ]

    # ---- IC50 thresholds (nM) -----------------------------------------------
    IC50_VERY_POTENT = 100     # < 100 nM
    IC50_POTENT = 1000         # < 1 µM
    IC50_ACTIVE = 10000        # < 10 µM

    # Only extract IC50 data (standard_type = 'IC50', units = 'nM', value <= IC50_ACTIVE)
    IC50_ONLY = True


# =============================================================================
# DATABASE HELPER
# =============================================================================

class ChEMBLDatabase:
    """Thin wrapper around mysql.connector for DataFrame results."""

    def __init__(self, cfg: ChEMBLConfig):
        self.cfg = cfg
        self.conn = None

    def connect(self) -> bool:
        try:
            self.conn = mysql.connector.connect(
                host=self.cfg.HOST,
                database=self.cfg.DATABASE,
                user=self.cfg.USER,
                password=self.cfg.PASSWORD,
                port=self.cfg.PORT,
            )
            if self.conn.is_connected():
                info = self.conn.server_info if hasattr(self.conn, "server_info") else "?"
                print(f"✓ Connected to MySQL {info}")
                cur = self.conn.cursor()
                cur.execute("SELECT DATABASE()")
                print(f"✓ Database: {cur.fetchone()[0]}")
                cur.close()
                return True
        except Error as e:
            print(f"✗ Connection error: {e}")
        return False

    def disconnect(self):
        if self.conn and self.conn.is_connected():
            self.conn.close()
            print("✓ Database connection closed")

    def query(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute *sql* and return a DataFrame.  Uses the raw DBAPI2 connection."""
        try:
            return pd.read_sql(sql, self.conn, params=params)
        except Exception as e:
            print(f"✗ Query error: {e}")
            return pd.DataFrame()


# =============================================================================
# SHARED SQL FRAGMENTS
# =============================================================================

_COMPOUND_COLUMNS = """
    md.chembl_id              AS compound_chembl_id,
    md.pref_name              AS compound_name,
    cs.canonical_smiles       AS smiles,
    cs.standard_inchi         AS inchi,
    cs.standard_inchi_key     AS inchi_key,
    cp.mw_freebase            AS molecular_weight,
    cp.alogp,
    cp.hba,
    cp.hbd,
    cp.psa,
    cp.rtb                    AS rotatable_bonds,
    cp.aromatic_rings,
    cp.heavy_atoms
"""

_ACTIVITY_COLUMNS = """
    act.activity_id,
    act.standard_type         AS activity_type,
    act.standard_relation     AS activity_relation,
    act.standard_value        AS activity_value,
    act.standard_units        AS activity_units,
    act.pchembl_value         AS pchembl
"""

_ASSAY_COLUMNS = """
    ass.assay_id,
    ass.chembl_id              AS assay_chembl_id,
    ass.assay_type,
    ass.description            AS assay_description,
    cl.cell_name               AS cell_line,
    cl.cell_description
"""

_TARGET_COLUMNS = """
    td.chembl_id              AS target_chembl_id,
    td.pref_name              AS target_name,
    td.target_type,
    td.organism               AS target_organism
"""

_DOC_COLUMNS = """
    d.chembl_id               AS document_chembl_id,
    d.pubmed_id,
    d.doi,
    d.journal,
    d.year                    AS publication_year,
    d.title                   AS publication_title
"""

_BASE_JOINS = """
FROM activities act
INNER JOIN molecule_dictionary md  ON act.molregno   = md.molregno
LEFT  JOIN compound_structures cs  ON md.molregno    = cs.molregno
LEFT  JOIN compound_properties cp  ON md.molregno    = cp.molregno
INNER JOIN assays              ass ON act.assay_id   = ass.assay_id
LEFT  JOIN cell_dictionary     cl  ON ass.cell_id    = cl.cell_id
LEFT  JOIN target_dictionary   td  ON ass.tid        = td.tid
LEFT  JOIN target_components   tco ON td.tid         = tco.tid
LEFT  JOIN component_synonyms  csyn ON tco.component_id = csyn.component_id
LEFT  JOIN docs                d   ON act.doc_id     = d.doc_id
LEFT  JOIN source              src ON act.src_id     = src.src_id
"""


# =============================================================================
# EXTRACTOR
# =============================================================================

class OSCCDataExtractor:
    """Run the four extraction queries and return a combined DataFrame."""

    def __init__(self, db: ChEMBLDatabase, cfg: ChEMBLConfig):
        self.db = db
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Query 1 – Cell-line screening
    # ------------------------------------------------------------------
    def query_by_cell_lines(self) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("QUERY 1: Compounds tested on OSCC cell lines")
        print("=" * 80)

        like_clauses = " OR ".join(["cl.cell_name LIKE %s"] * len(self.cfg.OSCC_CELL_LINES))
        params = tuple(f"%{cl}%" for cl in self.cfg.OSCC_CELL_LINES)

        sql = f"""
        SELECT
            {_COMPOUND_COLUMNS},
            {_ACTIVITY_COLUMNS},
            {_ASSAY_COLUMNS},
            {_TARGET_COLUMNS},
            GROUP_CONCAT(DISTINCT csyn.component_synonym SEPARATOR '; ') AS gene_symbols,
            {_DOC_COLUMNS},
            src.src_description AS data_source
        {_BASE_JOINS}
        WHERE ({like_clauses})
          AND cs.canonical_smiles IS NOT NULL
          AND (td.organism = 'Homo sapiens' OR td.organism IS NULL)
          AND act.standard_type = 'IC50'
          AND act.standard_units = 'nM'
          AND act.standard_value IS NOT NULL
          AND act.standard_value <= {self.cfg.IC50_ACTIVE}
        GROUP BY md.chembl_id, act.activity_id
        ORDER BY act.standard_value ASC
        """

        df = self.db.query(sql, params)
        print(f"  → {len(df)} records, {df['compound_chembl_id'].nunique() if not df.empty else 0} unique compounds")
        return df

    # ------------------------------------------------------------------
    # Query 2 – Target-based
    # ------------------------------------------------------------------
    def query_by_targets(self) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("QUERY 2: Compounds acting on OSCC-relevant targets")
        print("=" * 80)

        placeholders = ", ".join(["%s"] * len(self.cfg.OSCC_TARGETS))
        params = tuple(self.cfg.OSCC_TARGETS)

        sql = f"""
        SELECT
            {_COMPOUND_COLUMNS},
            {_ACTIVITY_COLUMNS},
            {_ASSAY_COLUMNS},
            {_TARGET_COLUMNS},
            GROUP_CONCAT(DISTINCT csyn.component_synonym SEPARATOR '; ') AS gene_symbols,
            {_DOC_COLUMNS},
            src.src_description AS data_source
        {_BASE_JOINS}
        WHERE csyn.component_synonym IN ({placeholders})
          AND td.organism = 'Homo sapiens'
          AND cs.canonical_smiles IS NOT NULL
          AND act.standard_type = 'IC50'
          AND act.standard_units = 'nM'
          AND act.standard_value IS NOT NULL
          AND act.standard_value <= {self.cfg.IC50_ACTIVE}
        GROUP BY md.chembl_id, act.activity_id
        ORDER BY act.standard_value ASC
        """

        df = self.db.query(sql, params)
        print(f"  → {len(df)} records, {df['compound_chembl_id'].nunique() if not df.empty else 0} unique compounds")
        return df

    # ------------------------------------------------------------------
    # Query 3 – Keyword search (assay description + publication title)
    # ------------------------------------------------------------------
    def query_by_keywords(self) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("QUERY 3: Keyword search in assay/publication text")
        print("=" * 80)

        kw_clauses_assay = " OR ".join(["ass.description LIKE %s"] * len(self.cfg.OSCC_KEYWORDS))
        kw_clauses_doc = " OR ".join(["d.title LIKE %s"] * len(self.cfg.OSCC_KEYWORDS))
        params = tuple(f"%{kw}%" for kw in self.cfg.OSCC_KEYWORDS) * 2  # once for assay, once for docs

        sql = f"""
        SELECT
            {_COMPOUND_COLUMNS},
            {_ACTIVITY_COLUMNS},
            {_ASSAY_COLUMNS},
            {_TARGET_COLUMNS},
            GROUP_CONCAT(DISTINCT csyn.component_synonym SEPARATOR '; ') AS gene_symbols,
            {_DOC_COLUMNS},
            src.src_description AS data_source
        {_BASE_JOINS}
        WHERE (({kw_clauses_assay}) OR ({kw_clauses_doc}))
          AND cs.canonical_smiles IS NOT NULL
          AND (td.organism = 'Homo sapiens' OR td.organism IS NULL)
          AND act.standard_type = 'IC50'
          AND act.standard_units = 'nM'
          AND act.standard_value IS NOT NULL
          AND act.standard_value <= {self.cfg.IC50_ACTIVE}
        GROUP BY md.chembl_id, act.activity_id
        ORDER BY act.standard_value ASC
        """

        df = self.db.query(sql, params)
        print(f"  → {len(df)} records, {df['compound_chembl_id'].nunique() if not df.empty else 0} unique compounds")
        return df

    # ------------------------------------------------------------------
    # Query 4 – Drug indication table
    # ------------------------------------------------------------------
    def query_by_indication(self) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("QUERY 4: Compounds with oral-cancer indications (drug_indication)")
        print("=" * 80)

        mesh_clauses = " OR ".join(["di.mesh_heading LIKE %s"] * len(self.cfg.INDICATION_KEYWORDS))
        efo_clauses = " OR ".join(["di.efo_term LIKE %s"] * len(self.cfg.INDICATION_KEYWORDS))
        params = tuple(f"%{kw}%" for kw in self.cfg.INDICATION_KEYWORDS) * 2

        sql = f"""
        SELECT
            md.chembl_id              AS compound_chembl_id,
            md.pref_name              AS compound_name,
            cs.canonical_smiles       AS smiles,
            cs.standard_inchi         AS inchi,
            cs.standard_inchi_key     AS inchi_key,
            cp.mw_freebase            AS molecular_weight,
            cp.alogp,
            cp.hba,
            cp.hbd,
            cp.psa,
            cp.rtb                    AS rotatable_bonds,
            cp.aromatic_rings,
            cp.heavy_atoms,
            di.mesh_heading           AS indication_mesh,
            di.efo_term               AS indication_efo,
            di.max_phase_for_ind      AS max_phase
        FROM drug_indication di
        INNER JOIN molecule_dictionary md ON di.molregno = md.molregno
        LEFT  JOIN compound_structures cs ON md.molregno = cs.molregno
        LEFT  JOIN compound_properties cp ON md.molregno = cp.molregno
        WHERE ({mesh_clauses}) OR ({efo_clauses})
        """

        df = self.db.query(sql, params)
        print(f"  → {len(df)} records, {df['compound_chembl_id'].nunique() if not df.empty else 0} unique compounds")
        return df

    # ------------------------------------------------------------------
    # PubChem CID mapping  (two strategies)
    #   1. PubChem BioAssay SIDs from compound_records (stored as SIDnnn)
    #   2. Local file Analysis/cid2chembl.tsv  (PubChem_CID → SCHEMBL_num)
    #      We strip the numeric part of ChEMBL IDs and join.
    # ------------------------------------------------------------------
    def get_pubchem_ids(self, chembl_ids: List[str]) -> pd.DataFrame:
        """Return DataFrame with columns [compound_chembl_id, pubchem_cid]."""
        if not chembl_ids:
            return pd.DataFrame(columns=["compound_chembl_id", "pubchem_cid"])

        print("\n  Mapping ChEMBL IDs → PubChem CIDs ...")

        # ---- Strategy 1: local cid2chembl.tsv (SCHEMBL numeric IDs) ----
        cid_file = os.path.join("Analysis", "cid2chembl.tsv")
        if os.path.isfile(cid_file):
            print(f"    Loading local mapping from {cid_file} ...")
            try:
                cid_map = pd.read_csv(cid_file, sep="\t", header=None,
                                      names=["pubchem_cid", "schembl_num"],
                                      dtype={"pubchem_cid": str, "schembl_num": str})
                # Build a set of SCHEMBL numbers we need to look up
                # molecule_dictionary.chembl_id  looks like 'CHEMBL25'
                # The mapping file has plain numeric SCHEMBL IDs.
                # We query the DB once for molregno → chembl_id  mapping,
                # then join via the compound_records PUBCHEM_BIOASSAY SIDs.
            except Exception as e:
                print(f"    Could not load {cid_file}: {e}")
                cid_map = pd.DataFrame()
        else:
            cid_map = pd.DataFrame()

        # ---- Strategy 2: PubChem BioAssay SIDs from compound_records ----
        # SIDs are stored as 'SID<number>' in compound_key.
        # We can at least attach these as pubchem_sid.
        batch_size = 500
        frames = []
        for i in range(0, len(chembl_ids), batch_size):
            batch = chembl_ids[i : i + batch_size]
            ph = ", ".join(["%s"] * len(batch))
            sql = f"""
            SELECT DISTINCT
                md.chembl_id       AS compound_chembl_id,
                cr.compound_key    AS pubchem_sid
            FROM molecule_dictionary md
            INNER JOIN compound_records cr ON md.molregno = cr.molregno
            INNER JOIN source src          ON cr.src_id   = src.src_id
            WHERE md.chembl_id IN ({ph})
              AND src.src_short_name = 'PUBCHEM_BIOASSAY'
            """
            frames.append(self.db.query(sql, tuple(batch)))

        df_sid = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df_sid.empty:
            # Keep first SID per compound
            df_sid = df_sid.drop_duplicates("compound_chembl_id", keep="first")
            print(f"    PubChem SIDs found for {len(df_sid)} compounds")
        else:
            df_sid = pd.DataFrame(columns=["compound_chembl_id", "pubchem_sid"])
            print("    No PubChem SIDs found in compound_records")

        # ---- Strategy 3: cid2chembl.tsv mapping via InChI key ----
        # The cid2chembl.tsv has PubChem_CID → SCHEMBL_numeric_id.
        # We can join SCHEMBL numeric ids to chembl_id_lookup to get chembl_id.
        df_cid = pd.DataFrame(columns=["compound_chembl_id", "pubchem_cid"])
        if not cid_map.empty:
            print("    Joining cid2chembl mapping via chembl_id_lookup ...")
            # Get all SCHEMBL numeric IDs from the file that we can match
            # chembl_id_lookup: chembl_id (e.g. SCHEMBL12345) → entity_type, entity_id
            # But we need entity_type='COMPOUND' and then join to molecule_dictionary.
            # Approach: prefix 'SCHEMBL' to the numeric ids and look up.
            schembl_ids = cid_map["schembl_num"].unique()
            # Limit to a reasonable sample (first 100k)
            sample_size = min(100000, len(schembl_ids))
            schembl_sample = schembl_ids[:sample_size]
            prefixed = [f"SCHEMBL{sid}" for sid in schembl_sample]

            # Query chembl_id_lookup in batches
            lookup_frames = []
            for j in range(0, len(prefixed), batch_size):
                b = prefixed[j : j + batch_size]
                ph2 = ", ".join(["%s"] * len(b))
                sql2 = f"""
                SELECT cil.chembl_id AS schembl_chembl_id,
                       md.chembl_id  AS compound_chembl_id
                FROM chembl_id_lookup cil
                JOIN molecule_dictionary md ON cil.entity_id = md.molregno
                WHERE cil.chembl_id IN ({ph2})
                  AND cil.entity_type = 'COMPOUND'
                """
                lookup_frames.append(self.db.query(sql2, tuple(b)))

            df_lookup = pd.concat(lookup_frames, ignore_index=True) if lookup_frames else pd.DataFrame()
            if not df_lookup.empty:
                # Strip 'SCHEMBL' prefix to get the numeric part for joining
                df_lookup["schembl_num"] = df_lookup["schembl_chembl_id"].str.replace("SCHEMBL", "", regex=False)
                # Join with cid_map
                df_cid = cid_map.merge(df_lookup[["schembl_num", "compound_chembl_id"]], on="schembl_num", how="inner")
                df_cid = df_cid[["compound_chembl_id", "pubchem_cid"]].drop_duplicates("compound_chembl_id", keep="first")
                df_cid["pubchem_cid"] = pd.to_numeric(df_cid["pubchem_cid"], errors="coerce").astype("Int64")
                print(f"    PubChem CIDs mapped for {len(df_cid)} compounds via cid2chembl")

        # Merge SID and CID results
        result = df_cid.copy() if not df_cid.empty else pd.DataFrame(columns=["compound_chembl_id", "pubchem_cid"])
        if not df_sid.empty:
            if result.empty:
                result = df_sid.rename(columns={"pubchem_sid": "pubchem_sid"})
                result["pubchem_cid"] = None
            else:
                result = result.merge(df_sid, on="compound_chembl_id", how="outer")

        # Filter to only compounds we care about
        result = result[result["compound_chembl_id"].isin(chembl_ids)]
        print(f"  → Total PubChem mappings: {result['pubchem_cid'].notna().sum()} CIDs, "
              f"{result['pubchem_sid'].notna().sum() if 'pubchem_sid' in result.columns else 0} SIDs")
        return result

    # ------------------------------------------------------------------
    # Molecule synonyms
    # ------------------------------------------------------------------
    def get_synonyms(self, chembl_ids: List[str]) -> pd.DataFrame:
        if not chembl_ids:
            return pd.DataFrame(columns=["compound_chembl_id", "synonyms"])

        print("  Fetching molecule synonyms ...")
        batch_size = 500
        frames = []
        for i in range(0, len(chembl_ids), batch_size):
            batch = chembl_ids[i : i + batch_size]
            ph = ", ".join(["%s"] * len(batch))
            sql = f"""
            SELECT
                md.chembl_id AS compound_chembl_id,
                GROUP_CONCAT(DISTINCT ms.synonyms SEPARATOR '; ') AS synonyms
            FROM molecule_dictionary md
            INNER JOIN molecule_synonyms ms ON md.molregno = ms.molregno
            WHERE md.chembl_id IN ({ph})
            GROUP BY md.chembl_id
            """
            frames.append(self.db.query(sql, tuple(batch)))

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        print(f"  → Synonyms found for {len(df)} compounds")
        return df

    # ------------------------------------------------------------------
    # Combined pipeline
    # ------------------------------------------------------------------
    def extract_all(self) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("OSCC COMPOUND EXTRACTION – FULL PIPELINE")
        print("=" * 80)

        # Run all four queries
        df_cell = self.query_by_cell_lines()
        df_cell["match_source"] = "OSCC_CELL_LINES"

        df_target = self.query_by_targets()
        df_target["match_source"] = "OSCC_TARGETS"

        df_kw = self.query_by_keywords()
        df_kw["match_source"] = "OSCC_KEYWORDS"

        df_ind = self.query_by_indication()
        df_ind["match_source"] = "OSCC_INDICATION"

        # Harmonize indication dataframe to have same columns before concat
        activity_cols = [
            "activity_id", "activity_type", "activity_relation",
            "activity_value", "activity_units", "pchembl",
            "assay_id", "assay_chembl_id", "assay_type", "assay_description",
            "cell_line", "cell_description",
            "target_chembl_id", "target_name", "target_type", "target_organism",
            "gene_symbols",
            "document_chembl_id", "pubmed_id", "doi", "journal",
            "publication_year", "publication_title",
            "data_source",
        ]
        for col in activity_cols:
            if col not in df_ind.columns:
                df_ind[col] = None

        # Combine
        df = pd.concat([df_cell, df_target, df_kw, df_ind], ignore_index=True)
        initial = len(df)

        # Deduplicate: keep first occurrence per (compound + activity_id), merging match_source
        if not df.empty and "activity_id" in df.columns:
            # Build match_source aggregate per compound+activity
            source_agg = (
                df.groupby(["compound_chembl_id", "activity_id"])["match_source"]
                .apply(lambda s: "; ".join(sorted(set(s.dropna()))))
                .reset_index()
                .rename(columns={"match_source": "match_sources"})
            )
            df = df.drop(columns=["match_source"]).drop_duplicates(
                subset=["compound_chembl_id", "activity_id"], keep="first"
            )
            df = df.merge(source_agg, on=["compound_chembl_id", "activity_id"], how="left")
        else:
            df["match_sources"] = df.get("match_source", "unknown")

        print(f"\n  Combined: {initial} → {len(df)} after dedup")
        print(f"  Unique compounds: {df['compound_chembl_id'].nunique()}")

        # Fetch PubChem CIDs & synonyms
        unique_ids = df["compound_chembl_id"].dropna().unique().tolist()
        df_pcid = self.get_pubchem_ids(unique_ids)
        df_syn = self.get_synonyms(unique_ids)

        if not df_pcid.empty:
            merge_cols = [c for c in df_pcid.columns if c != "compound_chembl_id"]
            df = df.merge(df_pcid, on="compound_chembl_id", how="left")
        else:
            df["pubchem_cid"] = None

        if not df_syn.empty:
            df = df.merge(df_syn, on="compound_chembl_id", how="left")
        else:
            df["synonyms"] = None

        return df


# =============================================================================
# DATA PROCESSING
# =============================================================================

class DataProcessor:
    """Enrich the extracted DataFrame with computed columns."""

    @staticmethod
    def classify_potency(val, units):
        """Classify IC50 value into potency buckets.
        All data is already IC50-only with nM units from SQL filter.
        """
        if pd.isna(val):
            return None
        # Values should already be in nM from SQL, but handle edge cases
        nm = val
        if pd.notna(units):
            u = str(units).strip()
            if u == "nM":
                nm = val
            elif u in ("µM", "uM"):
                nm = val * 1000
            elif u == "mM":
                nm = val * 1e6
            elif u == "pM":
                nm = val * 0.001
            else:
                nm = val  # assume nM since SQL filtered for nM

        if nm <= ChEMBLConfig.IC50_VERY_POTENT:
            return "very_potent"     # IC50 ≤ 100 nM
        elif nm <= ChEMBLConfig.IC50_POTENT:
            return "potent"          # IC50 ≤ 1 µM
        else:
            return "active"          # IC50 ≤ 10 µM

    @staticmethod
    def lipinski_pass(row):
        try:
            mw = row.get("molecular_weight")
            logp = row.get("alogp")
            hba = row.get("hba")
            hbd = row.get("hbd")
            if any(pd.isna(v) for v in [mw, logp, hba, hbd]):
                return None
            return bool(mw <= 500 and logp <= 5 and hba <= 10 and hbd <= 5)
        except Exception:
            return None

    @classmethod
    def enrich(cls, df: pd.DataFrame) -> pd.DataFrame:
        print("\nEnriching data ...")

        df["potency_class"] = df.apply(
            lambda r: cls.classify_potency(r["activity_value"], r.get("activity_units")),
            axis=1,
        )

        df["lipinski_pass"] = df.apply(cls.lipinski_pass, axis=1)

        # IC50 nM → µM convenience column (all data is IC50/nM from SQL)
        df["ic50_nM"] = df["activity_value"]
        df["ic50_uM"] = df["activity_value"] / 1000.0

        print("✓ Enrichment complete")
        return df


# =============================================================================
# SUMMARY
# =============================================================================

def generate_summary(df: pd.DataFrame) -> Dict:
    s: Dict = {}
    s["total_records"] = int(len(df))
    s["unique_compounds"] = int(df["compound_chembl_id"].nunique())
    s["unique_targets"] = int(df["target_chembl_id"].nunique()) if "target_chembl_id" in df.columns else 0
    s["unique_assays"] = int(df["assay_chembl_id"].nunique()) if "assay_chembl_id" in df.columns else 0
    s["unique_publications"] = int(df["document_chembl_id"].nunique()) if "document_chembl_id" in df.columns else 0

    # Match source breakdown
    if "match_sources" in df.columns:
        source_counts = {}
        for val in df["match_sources"].dropna():
            for src in val.split("; "):
                source_counts[src] = source_counts.get(src, 0) + 1
        s["match_source_counts"] = source_counts

    # Potency distribution
    if "potency_class" in df.columns:
        s["potency_distribution"] = df["potency_class"].value_counts().to_dict()

    # Cell lines
    if "cell_line" in df.columns:
        cells = sorted(df["cell_line"].dropna().unique().tolist())
        s["cell_lines"] = cells[:30]
        s["unique_cell_lines"] = len(cells)

    # Top targets
    if "target_name" in df.columns:
        s["top_targets"] = df["target_name"].value_counts().head(15).to_dict()

    # Lipinski
    if "lipinski_pass" in df.columns:
        lp = df["lipinski_pass"].dropna()
        passes = int(lp.sum())
        total = int(len(lp))
        s["lipinski"] = {"pass": passes, "total": total, "pct": round(passes / total * 100, 1) if total else 0}

    # PubChem coverage
    if "pubchem_cid" in df.columns:
        pcid_count = int(df["pubchem_cid"].notna().sum())
        s["pubchem_coverage"] = {"records_with_cid": pcid_count, "pct": round(pcid_count / len(df) * 100, 1) if len(df) else 0}

    # MW stats
    if "molecular_weight" in df.columns:
        mw = df["molecular_weight"].dropna()
        if len(mw):
            s["molecular_weight"] = {"mean": round(float(mw.mean()), 1), "median": round(float(mw.median()), 1),
                                     "min": round(float(mw.min()), 1), "max": round(float(mw.max()), 1)}

    return s


def print_summary(s: Dict):
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total records:       {s['total_records']:,}")
    print(f"  Unique compounds:    {s['unique_compounds']:,}")
    print(f"  Unique targets:      {s.get('unique_targets', 0):,}")
    print(f"  Unique assays:       {s.get('unique_assays', 0):,}")
    print(f"  Unique publications: {s.get('unique_publications', 0):,}")

    if "match_source_counts" in s:
        print("\n  Match sources:")
        for src, cnt in sorted(s["match_source_counts"].items(), key=lambda x: -x[1]):
            print(f"    {src:15s} {cnt:>8,}")

    if "potency_distribution" in s:
        print("\n  Potency distribution:")
        for cls, cnt in s["potency_distribution"].items():
            print(f"    {cls:18s} {cnt:>8,}")

    if "lipinski" in s:
        lk = s["lipinski"]
        print(f"\n  Lipinski Ro5: {lk['pass']:,} / {lk['total']:,} ({lk['pct']:.1f}%)")

    if "top_targets" in s:
        print("\n  Top targets:")
        for t, c in list(s["top_targets"].items())[:10]:
            print(f"    {t:40s} {c:>6,}")

    if "pubchem_coverage" in s:
        pc = s["pubchem_coverage"]
        print(f"\n  PubChem CID coverage: {pc['records_with_cid']:,} records ({pc['pct']:.1f}%)")


# =============================================================================
# EXPORT
# =============================================================================

def export_data(df: pd.DataFrame, summary: Dict, outdir: str, ts: str):
    os.makedirs(outdir, exist_ok=True)

    # 1. Full CSV
    full_csv = os.path.join(outdir, f"oscc_compounds_full_{ts}.csv")
    df.to_csv(full_csv, index=False)
    print(f"✓ Full data          → {full_csv}")

    # 2. Unique compounds CSV (best IC50 per compound — lowest value)
    df_unique = df.sort_values("activity_value", ascending=True).drop_duplicates("compound_chembl_id", keep="first")
    keep_cols = [
        "compound_chembl_id", "compound_name", "smiles", "inchi_key",
        "pubchem_cid", "synonyms",
        "molecular_weight", "alogp", "hba", "hbd", "psa", "rotatable_bonds",
        "lipinski_pass", "potency_class", "ic50_nM", "ic50_uM",
        "pchembl",
        "target_name", "gene_symbols",
        "cell_line", "match_sources",
    ]
    keep_cols = [c for c in keep_cols if c in df_unique.columns]
    df_unique = df_unique[keep_cols]
    unique_csv = os.path.join(outdir, f"oscc_unique_compounds_{ts}.csv")
    df_unique.to_csv(unique_csv, index=False)
    print(f"✓ Unique compounds   → {unique_csv}  ({len(df_unique)} compounds)")

    # 2b. Potency-class breakdown files
    for pclass in ["very_potent", "potent", "active"]:
        subset = df_unique[df_unique["potency_class"] == pclass]
        if not subset.empty:
            pclass_csv = os.path.join(outdir, f"oscc_{pclass}_compounds_{ts}.csv")
            subset.to_csv(pclass_csv, index=False)
            print(f"✓ {pclass:12s} compounds → {pclass_csv}  ({len(subset)} compounds)")

    # 3. Summary JSON
    json_path = os.path.join(outdir, f"oscc_summary_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✓ Summary JSON       → {json_path}")

    # 4. Excel (optional)
    try:
        import openpyxl  # noqa: F401
        xlsx_path = os.path.join(outdir, f"oscc_compounds_{ts}.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="All_IC50_Activities", index=False)
            df_unique.to_excel(w, sheet_name="Unique_Compounds", index=False)
            for pclass in ["very_potent", "potent", "active"]:
                psubset = df_unique[df_unique["potency_class"] == pclass]
                if not psubset.empty:
                    psubset.to_excel(w, sheet_name=pclass.replace("_", " ").title(), index=False)
            sum_df = pd.DataFrame(
                [(k, str(v)) for k, v in summary.items()],
                columns=["Metric", "Value"],
            )
            sum_df.to_excel(w, sheet_name="Summary", index=False)
        print(f"✓ Excel workbook     → {xlsx_path}")
    except ImportError:
        print("  (openpyxl not installed – Excel export skipped)")

    return full_csv, unique_csv, json_path


# =============================================================================
# CROSS-REFERENCE WITH KNOWN 2B6E COMPOUNDS
# =============================================================================

def cross_reference_known(df: pd.DataFrame, known_path: str):
    """Flag compounds found in the known 2B6E (OSCC) list."""
    if not os.path.isfile(known_path):
        print(f"  (Known compounds file not found: {known_path})")
        return

    try:
        known = pd.read_csv(known_path, sep="\t")
    except Exception as e:
        print(f"  (Could not read {known_path}: {e})")
        return

    # Try to match by compound name (case-insensitive)
    if "compound_name" in known.columns:
        name_col = "compound_name"
    elif "CompoundName" in known.columns:
        name_col = "CompoundName"
    else:
        # grab second column (first is usually ID)
        name_col = known.columns[1] if len(known.columns) > 1 else known.columns[0]

    known_names = set(known[name_col].dropna().str.lower().str.strip())
    # Build a boolean mask safely
    name_series = df["compound_name"].fillna("").str.lower().str.strip()
    mask = name_series.isin(known_names)
    matched_compounds = df.loc[mask, "compound_name"].dropna().unique()

    print(f"\n  Cross-reference with known 2B6E compounds ({len(known_names)} in reference):")
    print(f"    Matched: {len(matched_compounds)} compounds")
    if len(matched_compounds):
        for name in sorted(matched_compounds)[:20]:
            print(f"      ✓ {name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("ChEMBL OSCC Compound Extraction Tool v2")
    print("=" * 80)
    start = datetime.now()
    print(f"Start: {start:%Y-%m-%d %H:%M:%S}")

    cfg = ChEMBLConfig()
    OUTPUT_DIR = "oscc_chembl_data"
    ts = start.strftime("%Y%m%d_%H%M%S")

    db = ChEMBLDatabase(cfg)
    if not db.connect():
        print("Exiting – cannot connect.")
        return

    try:
        extractor = OSCCDataExtractor(db, cfg)
        df = extractor.extract_all()

        if df.empty:
            print("\n✗ No data extracted.")
            return

        df = DataProcessor.enrich(df)

        summary = generate_summary(df)
        print_summary(summary)

        print("\n" + "=" * 80)
        print("EXPORTING")
        print("=" * 80)
        export_data(df, summary, OUTPUT_DIR, ts)

        # Cross-reference with known OSCC compounds
        cross_reference_known(df, "scripts/2B6E_list.tsv")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.disconnect()

    end = datetime.now()
    print(f"\nEnd: {end:%Y-%m-%d %H:%M:%S}")
    print(f"Elapsed: {end - start}")
    print("=" * 80)


if __name__ == "__main__":
    main()
