module DoseUtils

export build_dose_feats


function remove_dose_unit(dose_list::Vector{Symbol})::Vector{Float32}
    # Dose unit is always uM for treated profiles, for both LINCS and Tahoe,
    # so this should never actually fire.
    dose_strs = string.(dose_list)
    invalid = filter(s -> !endswith(s, " uM"), dose_strs)
    isempty(invalid) || @error "Found dose(s) not in uM: $invalid"

    dose_values = [parse(Float32, strip(s[1:end-3])) for s in dose_strs]
    return dose_values
end


# "gate" uses log1p rather than log10: non-negative by construction (as in CPA), which
# is what lets a multiplicative dose gate attenuate toward zero at zero dose instead of
# flipping sign. See DoserGate in Models.jl. Every other encoding uses plain log10.
function build_dose_log_feats(dose_values::Vector{Float32}, encoding::String)::Matrix{Float32}
    log_dose = encoding == "gate" ? log1p.(dose_values) : log10.(dose_values)
    return reshape(Float32.(log_dose), 1, :)
end


"""
Defined from the full range of doses of A549 (the default reference cell line) on
LINCS (log10 µM from -4.00 to 2.30). Tahoe's doses fall inside this same range, so the
same edges are reused there.

TODO: fix this dose vocabulary from training data only (not shared across train/val/
test), and move dose_feats — like time_feats — out of Obs, computing them on demand
instead of freezing them in at construction time.
"""
const DOSE_LOG10_EDGES = Float32[-3, -2, -1, 0, 1, 2]


function build_dose_one_hot(dose_values::Vector{Float32})::Matrix{Float32}
    log_dose = Float32.(log10.(dose_values))

    # bin i = first edge the value does not exceed; overflow bin = length(edges)+1
    nbins = length(DOSE_LOG10_EDGES) + 1
    D = zeros(Float32, nbins, length(log_dose))
    for (j, v) in enumerate(log_dose)
        i = findfirst(e -> v <= e, DOSE_LOG10_EDGES)
        D[isnothing(i) ? nbins : i, j] = 1f0
    end
    return D
end


function build_dose_feats(dose_list::Vector{Symbol}; encoding::String = "gate")::Matrix{Float32}
    encoding in ("one_hot", "log10", "gate") ||
        error("Unknown dose encoding=$(repr(encoding)). Choose one of: one_hot, log10, gate.")

    dose_values = remove_dose_unit(dose_list)

    if encoding == "one_hot"
        return build_dose_one_hot(dose_values)
    else
        return build_dose_log_feats(dose_values, encoding)
    end
end


end
