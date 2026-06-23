import sqlite3
import json
import os
from typing import List, Optional, Dict, Any

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS recipes (
    faiss_index INTEGER PRIMARY KEY,
    recipe_id INTEGER,
    name TEXT,
    author_id INTEGER,
    author_name TEXT,
    cook_time TEXT,
    prep_time TEXT,
    total_time TEXT,
    date_published TEXT,
    description TEXT,
    images TEXT,                       -- JSON list
    recipe_category TEXT,
    keywords TEXT,                     -- JSON list
    recipe_ingredient_quantities TEXT, -- JSON list (original parsed)
    recipe_ingredient_parts TEXT,      -- JSON list (original parsed)
    aggregated_rating REAL,
    review_count INTEGER,
    calories REAL,
    fat_content REAL,
    saturated_fat_content REAL,
    cholesterol_content REAL,
    sodium_content REAL,
    carbohydrate_content REAL,
    fiber_content REAL,
    sugar_content REAL,
    protein_content REAL,
    recipe_servings REAL,
    recipe_yield TEXT,
    recipe_instructions TEXT,          -- JSON list
    ingredients_raw TEXT,              -- JSON list
    ingredients_cleaned TEXT,          -- JSON list
    ingredients_with_quantities TEXT   -- JSON list
);
CREATE INDEX IF NOT EXISTS idx_name ON recipes(name);
"""

# Columns that are stored as JSON lists in SQLite
_LIST_COLUMNS = {
    "images",
    "keywords",
    "recipe_ingredient_quantities",
    "recipe_ingredient_parts",
    "recipe_instructions",
    "ingredients_raw",
    "ingredients_cleaned",
    "ingredients_with_quantities",
}


def _serialize_row(row: Any) -> tuple:
    """Convert a DataFrame row dict into a SQLite-compatible tuple."""
    values = []
    for col in _get_columns():
        val = row.get(col)
        if col in _LIST_COLUMNS:
            values.append(json.dumps(val) if val is not None else "[]")
        elif col == "date_published":
            values.append(str(val) if val is not None else None)
        else:
            values.append(val)
    return tuple(values)


def _deserialize_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a SQLite row into a Python dict, parsing JSON lists."""
    result = {}
    for key in row.keys():
        val = row[key]
        if key in _LIST_COLUMNS and isinstance(val, str):
            try:
                result[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                result[key] = []
        else:
            result[key] = val
    return result


def _get_columns() -> List[str]:
    """Ordered column list matching the schema."""
    return [
        "faiss_index", "recipe_id", "name", "author_id", "author_name",
        "cook_time", "prep_time", "total_time", "date_published",
        "description", "images", "recipe_category", "keywords",
        "recipe_ingredient_quantities", "recipe_ingredient_parts",
        "aggregated_rating", "review_count", "calories", "fat_content",
        "saturated_fat_content", "cholesterol_content", "sodium_content",
        "carbohydrate_content", "fiber_content", "sugar_content",
        "protein_content", "recipe_servings", "recipe_yield",
        "recipe_instructions", "ingredients_raw", "ingredients_cleaned",
        "ingredients_with_quantities",
    ]


class RecipeSQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def insert_recipes(self, df):
        """Bulk insert recipes from a DataFrame."""
        columns = _get_columns()
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO recipes ({', '.join(columns)}) VALUES ({placeholders})"

        records = []
        for _, row in df.iterrows():
            records.append(_serialize_row(row))

        self.conn.executemany(sql, records)
        self.conn.commit()
        print(f"[SQLite] Inserted {len(records)} recipes into {self.db_path}")

    def get_recipe_by_faiss_index(self, idx: int) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM recipes WHERE faiss_index = ?", (idx,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _deserialize_row(row)

    def recipe_exists(self, idx: int) -> bool:
        cursor = self.conn.execute(
            "SELECT 1 FROM recipes WHERE faiss_index = ? LIMIT 1", (idx,)
        )
        return cursor.fetchone() is not None

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM recipes")
        return cursor.fetchone()[0]

    def close(self):
        self.conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
