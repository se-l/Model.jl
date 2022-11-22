module BitfinexReader

using PyCall
using Dates
using DataFrames

import ZipFile
import CSV

import ..Exchange, ..Asset, ..bitfinex_tick, ..root, ..path_layers

dir_tick = bitfinex_tick
    
schema_trade = ["ts", "price", "size", "side"]
schema_quote = ["ts", "price", "size", "count"]

function load(start:: Date, stop:: Date, directory:: String, fn_key:: String)::DataFrame
    df_lst = []
    # (root, dirs, filenames) = collect(walkdir(directory))[1]
    for (root, dirs, filenames) in walkdir(directory)
        # fn = filenames[4]
        for fn in filenames
            fn_date = Date(fn[1:8], "yyyymmdd")
            if start <= fn_date && fn_date < (stop + Dates.Day(1)) && occursin(fn_key, fn)
                r = ZipFile.Reader(joinpath(root, fn))
                # f = r.files[1]
                for f in r.files
                    @info("Reading: $(f.name)")
                    df = CSV.File(f, header=false) |> DataFrame
                    if size(df)[1] == 0
                        @info("No Data for $(fn_date)")
                        continue
                    end
                    # bad value
                    # ix_drop = df.index[(~df[0].apply(is_float))]
                    # v_drop = !isa.(df[:, 2], AbstractFloat)
                    # if v_drop |> sum > 0
                    #     @info("Dropping $(len(ix_drop)). Time not float. File capture was interrupted on server likely.")
                    #     df = df[v_drop]
                    # end
                    df[!, 1] = Millisecond.(round.(Int, df[:, 1] .* 1_000)) + DateTime(fn_date)
                    push!(df_lst, df)
                end
                close(r)
            end
        end
    end
    return vcat(df_lst...)
end

# function remove_merged_rows(df::DataFrame)::DataFrame
#     b4 = size(df)[1]
#     df = df[~(df[0].astype(str).apply(len) > 9)]
#     if len(df) < b4
#         print("Removed $(b4 - len(df)) rows")
#     end
#     return df
# end

function load_trades(sym:: Asset.T, start:: Date, stop::Date)::DataFrame
    df = load(start, stop, joinpath(dir_tick, lowercase(string(sym))), "trade")
    if size(df)[1] == 0
        return DataFrame([Vector{T}(undef, 0) for T in [String, String, String, String]], schema_trade, copycols=false)
    end
    rename!(df, schema_trade)

    df[!, :side] = convert.(String, df[:, :side])
    map_direction(s) = Dict([("Sell", -1), ("Buy", 1)])[s]

    df[!, :side] = map_direction.(df[:, :side])
    df[!, :size] = abs.(df[:, :size])
    # df["ts"] = df["ts"].dt.tz_localize("UTC")
    return df
end

function load_quotes(sym::Asset.T, start::Date, stop::Date)::DataFrame
    df = load(start, stop, joinpath(dir_tick, lowercase(string(sym))), "quote")
    if size(df)[1] == 0
        return DataFrame([Vector{T}(undef, 0) for T in [String, String, String, String]], schema_quote, copycols=false)
    end
    rename!(df, schema_quote)
    # amount > 0: Bid < 0 Ask. count 0: deleted
    df = hcat(df, map.((x) -> x > 0 ? 1 : -1,  df[:, :size]))
    rename!(df, :x1 => :side)
    # df["side"].loc[df.index[df["count"] == 0]] = 0
    # df["ts"] = df["ts"].dt.tz_localize("UTC")
    return df
end

function test()
    sym=Asset.ethusd
    start=Date(2022, 2, 8)
    stop=Date(2022, 2, 9)
    directory = joinpath(dir_tick, lowercase(string(sym)))
    fn_key="trade"
    return load_trades(sym, start, stop)
end

function reconcile_py()
    datetime = pyimport("datetime")
    np = pyimport("numpy")

    push!(pyimport("sys")."path", root)
    push!(pyimport("sys")."path", joinpath(path_layers, "raw"))
    BitfinexReaderPy = pyimport("bitfinex_reader").BitfinexReader
    df_py = BitfinexReaderPy.load_quotes("ethusd", datetime.date(2022, 2, 8), datetime.date(2022, 2, 9))
    if df_py === nothing
        return
    end
    return DataFrame(
        ts=Nanosecond.(df_py.get("timestamp").astype(np.int64)) + DateTime(1970),
        price=df_py.get("price").values,
        size=df_py.get("size").values,
        count=df_py.get("count").values,
        side=df_py.get("side").values,
    )
end

end  # module