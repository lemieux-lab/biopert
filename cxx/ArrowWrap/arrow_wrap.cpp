#include <arrow/io/file.h>
#include <arrow/api.h>
#include <arrow/util/compression.h>
#include <parquet/arrow/reader.h>
#include <julia.h>
#include <jlcxx/jlcxx.hpp>
#include <jlcxx/array.hpp>
#include <jlcxx/stl.hpp>
#include <stdexcept>
#include <chrono>


// -----------------------------------------------------------------------------
// Julia CxxWrap interop for jl_sym_t*
// -----------------------------------------------------------------------------
namespace jlcxx {
    template<>
    struct MappingTrait<jl_sym_t*> {
        using type = DirectPtrTrait;
    };

    template<>
    struct julia_type_factory<jl_sym_t*> {
        static jl_datatype_t* julia_type() {
            return (jl_datatype_t*) jl_get_global(jl_base_module, jl_symbol("Symbol"));
        }
    };
}

// -----------------------------------------------------------------------------
// ParquetReaderWrapper: reads Parquet files via Apache Arrow C++
// -----------------------------------------------------------------------------
class ParquetReaderWrapper {
public:
    std::shared_ptr<arrow::io::ReadableFile> infile;
    std::unique_ptr<parquet::arrow::FileReader> reader;
    int32_t nb;  // number of nonzero entries
    int32_t m;   // number of columns (cells)

    // -------------------------------------------------------------------------
    // Constructor: open Parquet file and compute basic sizes
    // -------------------------------------------------------------------------
    ParquetReaderWrapper(const std::string& filename) {
        PARQUET_ASSIGN_OR_THROW(
            infile,
            arrow::io::ReadableFile::Open(filename)
        );

        PARQUET_ASSIGN_OR_THROW(
            reader,
            parquet::arrow::OpenFile(infile, arrow::default_memory_pool())
        );

        auto meta = reader->parquet_reader()->metadata();
        int num_row_groups = reader->num_row_groups();

        nb = 0;
        m = 0;
        for (int rg = 0; rg < num_row_groups; ++rg) {
            auto rg_meta = meta->RowGroup(rg);
            auto col0 = rg_meta->ColumnChunk(0);
            nb += col0->num_values();
            m += rg_meta->num_rows();
        }
    }

    ~ParquetReaderWrapper() = default;

    std::tuple<int32_t, int64_t> get_sizes() {
        return std::make_tuple(m, nb);
    }

    // -------------------------------------------------------------------------
    // load_expr: populate CSC arrays (colptr, rowval, nzval)
    // -------------------------------------------------------------------------
    void load_expr(jlcxx::ArrayRef<int32_t, 1> colptr,
                   jlcxx::ArrayRef<int32_t, 1> rowval,
                   jlcxx::ArrayRef<float,   1> nzval)
    {
        int num_row_groups = reader->num_row_groups();

        int row_idx = 0;  // current column index
        int nb_idx  = 0;  // current nonzero index

        for (int rg = 0; rg < num_row_groups; ++rg) {
            std::shared_ptr<parquet::arrow::RowGroupReader> rowgroup = reader->RowGroup(rg);

            std::shared_ptr<arrow::ChunkedArray> genes_array;
            std::shared_ptr<arrow::ChunkedArray> exprs_array;

            PARQUET_THROW_NOT_OK(rowgroup->Column(0)->Read(&genes_array));  // genes
            PARQUET_THROW_NOT_OK(rowgroup->Column(1)->Read(&exprs_array));  // expressions

            auto genes = std::static_pointer_cast<arrow::ListArray>(genes_array->chunk(0));
            auto exprs = std::static_pointer_cast<arrow::ListArray>(exprs_array->chunk(0));

            auto gene_vals = std::static_pointer_cast<arrow::Int64Array>(genes->values());
            auto expr_vals = std::static_pointer_cast<arrow::FloatArray>(exprs->values());

            const int64_t* gene_data = gene_vals->raw_values();
            const float*   expr_data = expr_vals->raw_values();

            const int32_t* gene_offsets = genes->raw_value_offsets();
            const int32_t* expr_offsets = exprs->raw_value_offsets();

            int64_t num_rows = genes->length();

            for (int64_t i = 0; i < num_rows; ++i) {
                // mark start of this column (Julia uses 1-based indices)
                colptr[row_idx] = nb_idx + 1;

                int32_t g_start = gene_offsets[i];
                int32_t g_end   = gene_offsets[i + 1];
                int32_t x_start = expr_offsets[i];
                int32_t x_end   = expr_offsets[i + 1];

                int32_t len = g_end - g_start;
                for (int32_t j = 0; j < len; ++j) {
                    rowval[nb_idx] = gene_data[g_start + j];
                    nzval[nb_idx]  = expr_data[x_start + j];
                    ++nb_idx;
                }

                ++row_idx;
            }
        }

        // colptr must have length (ncols + 1), and the last entry = nnz + 1
        if (row_idx < static_cast<int>(colptr.size())) {
            colptr[row_idx] = nb_idx + 1;
        }
    }

    // -------------------------------------------------------------------------
    // load_syms: load string column as Julia Symbols
    // -------------------------------------------------------------------------
    void load_syms(jlcxx::ArrayRef<jl_sym_t*> res, int32_t col) {
        int num_row_groups = reader->num_row_groups();

        long int num_rows = -1;
        std::vector<int> v = {col};
        auto st = reader->ScanContents(v, 1, &num_rows);

        int row_idx = 0;

        for (int rg = 0; rg < num_row_groups; ++rg) {
            auto rowgroup = reader->RowGroup(rg);

            std::shared_ptr<arrow::ChunkedArray> drug_array;
            PARQUET_THROW_NOT_OK(rowgroup->Column(col)->Read(&drug_array));

            auto string_array = std::static_pointer_cast<arrow::StringArray>(drug_array->chunk(0));

            for (int64_t i = 0; i < string_array->length(); ++i, ++row_idx) {
                std::string tmp(string_array->GetView(i));
                res[row_idx] = jl_symbol(tmp.c_str());
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Julia module registration
// -----------------------------------------------------------------------------
JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.add_type<ParquetReaderWrapper>("ParquetReaderWrapper")
       .constructor<const std::string&>()
       .method("get_sizes", &ParquetReaderWrapper::get_sizes)
       .method("load_expr", &ParquetReaderWrapper::load_expr)
       .method("load_syms", &ParquetReaderWrapper::load_syms);
}
