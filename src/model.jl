module Model

using Revise
import YAML: load_file
import TsDb: Client
cfg = load_file(joinpath(@__FILE__, "..", "config.yaml"))

root = @__DIR__
path_common = joinpath(root, "common")
path_temp = joinpath(root, "temp")
path_layers = joinpath(root, "layers")
path_connector = joinpath(root, "connector")
path_layer_settings = joinpath(root, "layers", "bars", "settings.yaml")

const path_data = cfg["path_data"]
path_tsdb = joinpath(path_data, "ts2hdf5")
qc_crypto = joinpath(path_data, "crypto")
qc_bitfinex_crypto = joinpath(qc_crypto, "bitfinex")
bitfinex_tick = joinpath(root, qc_bitfinex_crypto, "tick")

export
    Asset, Exchange,
    BitfinexReader,
    OrderBook,
    ClientTsDb,
    ffill, isna, bfill


include("common/utils.jl")
include("common/enums/exchange.jl")
include("common/enums/asset.jl")
include("layers/raw/zip.jl")
include("layers/raw/bitfinex_reader.jl")
include("layers/bars/OrderBook.jl")
include("layers/bars/imbalance_order_book.jl")
include("layers/estimate/estimate.jl")
include("layers/estimate/auto_tune.jl")


end # module Model

# save_order_book_metrics()
# zipem()
