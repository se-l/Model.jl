import DataFrames: outerjoin, names

ffill(v) = return v[accumulate(max, [i * !(ismissing(v[i]) | isnan(v[i])) for i in 1:length(v)], init=1)]
isna(v) = return map((i) -> (ismissing(i) | isnan(i)), v)

function bfill(v)
    vr = reverse(v)
    return reverse(vr[accumulate(max, [i * !(ismissing(vr[i]) | isnan(vr[i])) for i in 1:length(vr)], init=1)])
end

function join_trim_fill(dfa, dfb)
    dfj = outerjoin(dfa, dfb, on=:ts)
    start = maximum([dfa[1, "ts"], dfb[1, "ts"]])
    end_ = minimum([dfa[end, "ts"], dfb[end, "ts"]])
    dfj = dfj[(dfj[:, "ts"] .>= start) .& (dfj[:, "ts"] .< end_), :]
    sort!(dfj, ["ts"])
    col_a = names(dfa)[2]
    col_b = names(dfb)[2]
    dfj[:, col_b] .= ffill(dfj[:, col_b])
    filter!(r -> !ismissing(r[col_a]), dfj)
    dfj = dfj[(ismissing.(dfj[:, col_b])|>sum)+1:end, :]
    @assert(ismissing.(dfj[:, col_b]) |> sum == 0)
    @assert(ismissing.(dfj[:, col_a]) |> sum == 0)
    return dfj
end

# function ADF_test(y::AbstractVector)
#     import HypothesisTests: ADFTest, pvalue
#     maxlag = 12*(length(y)/100)^{1/4}
#     for lag in [1]
#         pval = pvalue(ADFTest(y, :constant, lag))
#         println(pval)
#     end
# end