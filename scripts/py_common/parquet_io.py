from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def save_embeds_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write df to a parquet file with `embedding` stored as raw float32 bytes
    (pa.binary) instead of list<float>.

    Parquet2.jl cannot read the list<float> nested column type (FieldError on
    meta_data). Storing embeddings as binary bytes avoids nested types entirely;
    Julia reconstructs the vector with `reinterpret(Float32, bytes)`.
    """
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    emb_col = tbl.column("embedding")
    emb_bytes = pa.array(
        [np.asarray(v.as_py(), dtype=np.float32).tobytes() for v in emb_col],
        type=pa.binary(),
    )
    tbl = tbl.set_column(tbl.schema.get_field_index("embedding"), "embedding", emb_bytes)
    pq.write_table(tbl, path)
