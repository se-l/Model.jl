module Model

using Revise
import TsDb: Client

root = @__DIR__
path_common = joinpath(root, "common")
path_temp = joinpath(root, "temp")
path_layers = joinpath(root, "layers")
path_connector = joinpath(root, "connector")
path_data = joinpath(root, "..", "data")
path_tsdb = joinpath(path_data, "ts2hdf5")
path_layer_settings = joinpath(root, "layers", "bars", "settings.yaml")

export
    OrderBook,
    ClientTsDb,
    ffill,
    isna,
    bfill,
    # work on estimate.jl
    get_ex, LoadXY, return_attribution_sample_weights, 
    load_inputs, Not,
    optimize, test_optimize


include("common/utils.jl")
include("layers/raw/zip.jl")
include("layers/bars/OrderBook.jl")
include("layers/bars/imbalance_order_book.jl")
include("layers/estimate/estimate.jl")
include("layers/estimate/auto_tune.jl")


end # module Model

# save_order_book_metrics()
# zipem()
