using PyCall
using Serialization
using Dates
using DataFrames
using DataStructures
using MLJ
using Plots

import .Client
import .path_layers

datetime = pyimport("datetime")
np = pyimport("numpy")
push!(pyimport("sys")."path", joinpath(path_layers, "estimate"))
LoadXY = pyimport("load_xy").LoadXY

get_ex(;sym="") = "ex$(Dates.format(Dates.now(Dates.UTC), dateformat"Y-mm-dd_HHMMSS"))-$(sym)"


function load_inputs(exchange, sym, start, stop, features=missing)
    load_xy = LoadXY(exchange, sym, 
        datetime.datetime.fromisoformat(string(start)),  
        datetime.datetime.fromisoformat(string(stop)), label_ewm_span="64min")
    load_xy.assemble_frames()
    df = load_xy.df
    df = DataFrame(Dict(vcat(
        [(:ts, Nanosecond.(df.index.astype(np.int64).values) + DateTime(1970))],
        [(Symbol(c), df.get(c).values) for c in df.columns]
        )))

    println("Load XY done.")
    # apply sampling and purging separately perhaps
    v_label = load_xy.ps_label.values
    select!(df, :ts, Not(:ts))  # sorting
    return (df, v_label)
end


function split_ho(df, v_label; ho_share=0.3)
    n_ho = round(Int, size(df)[1] * ho_share)
    n_t = size(df)[1] - n_ho
    return  df[1:n_t, :], 
            df[end-n_ho:end, :], 
            v_label[1:n_t],
            v_label[end-n_ho:end]
end

function store_return_attribution(df, sym, ex)
    # need to have ts in names of df. push into upsert function. can use if type is jl DataFrame, otherwise first col
    select!(df, :ts, Not(:ts))
    meta = Dict(
        "measurement_name" => "weights",
        "asset" => sym,
        "information" => "return_attribution_sample_weights",
        "ex" => ex,
    )
    Client.upsert(meta, df)
end

function return_attribution_sample_weights(v_label; return_amplifier=100)
    """
    instance with high absolute return change should get a much higher weight. ignore the noise.
    overflow / underflow: normalize weights to have the smallest weight a weight of 1. Also easier intuitively later when plotting.
    """
    v = abs.(v_label .- 1) .* return_amplifier .+ 1
    return v ./ minimum(v)
end


function train(df, y, sym, ex::String)
    """t, ho  ;   t -> t_cv test_cv"""
    booster = @load LGBMRegressor

    v_ix_t, v_ix_ho = partition(1:size(df)[1], 0.7, shuffle=false)
    df_t, df_ho = df[v_ix_t, :], df[v_ix_ho, :]
    y_t, y_ho = y[v_ix_t], y[v_ix_ho]

    v_weight_t = return_attribution_sample_weights(y_t)
    # geometric_mean()
    # cluster_sample_weight(50).\

    # store_return_attribution(DataFrame(Dict(
    #     :ts => df[v_ix_t, :ts],
    #     :weight => v_weight_t
    # )), sym, ex)

    estimator_params = Dict([
        (:metric, ["l2"]),  # MSE
        # ("objective", "quantile"),
        # ("alpha", quantile),
        (:learning_rate, 0.05),
        # (:early_stopping_round, 1),  # how no effect in IteratedModel
        (:boosting, "gbdt"),
        # (:num_iterations, 1),  # how no effect in IteratedModel. But used in fit!(estimator).
        (:device_type, "gpu"),
    ])

    estimator = booster(;estimator_params...)

    iterated_model = IteratedModel(
        model=estimator,
        # Custom resampling strategies
        # https://alan-turing-institute.github.io/MLJ.jl/stable/evaluating_model_performance/
        # To define a new resampling strategy, make relevant parameters of your strategy the fields of a new type MyResamplingStrategy <: MLJ.ResamplingStrategy, and implement one of the following methods:

        # MLJ.train_test_pairs(my_strategy::MyResamplingStrategy, rows)
        # MLJ.train_test_pairs(my_strategy::MyResamplingStrategy, rows, y)
        # MLJ.train_test_pairs(my_strategy::MyResamplingStrategy, rows, X, y)
        # Each method takes a vector of indices rows and returns a vector [(t1, e1), (t2, e2), ... (tk, ek)] of train/test pairs of row indices selected from rows. Here X, y are the input and target data (ignored in simple strategies, such as Holdout and CV).
        # source code in MLJ was overwritten to enable purging boundaries. Move into separate fork...
        resampling=CV(nfolds=5, nboundary=250),
        iteration_parameter=:(num_iterations),
        measures=l2,
        controls=[
            Step(1),
            Patience(1),
            NumberLimit(100),
            WithLossDo(f=x->@info("WithLossDo: $x"), stop_if_true=false, stop_message=nothing),
        ],
        retrain=true,
        )
    mach = machine(iterated_model, select(df_t, Not(:ts)), y_t, v_weight_t)
    fit!(mach, verbosity=10)
    yhatcv = predict(mach, select(df_t, Not(:ts)))
    l2(yhatcv, y_t) |> mean  # 2.987225673950955e-5  with number limit 5 3.813411156822854e-5 (5)   3.813411152918477e-5 (25)

    learning_rate = range(estimator, :learning_rate, lower=0.01, upper=0.2)
    lambda_l1 = range(estimator, :lambda_l1, lower=0.01, upper=0.2)
    lambda_l2 = range(estimator, :lambda_l1, lower=0.01, upper=0.2)
    num_leaves = range(estimator, :lambda_l1, lower=15, upper=62)
    min_data_in_leaf = range(estimator, :lambda_l1, lower=10, upper=20*2)

    self_tuning_tree = TunedModel(model=estimator,
							  resampling=CV(nfolds=5, nboundary=250),
							  tuning=Grid(resolution=5),
							  range=[learning_rate, lambda_l1, lambda_l2, num_leaves, min_data_in_leaf],
							  measure=rms,
                              n = 20,
                              );
    mach = machine(self_tuning_tree, select(df_t, Not(:ts)), y_t, v_weight_t)
    fit!(mach, verbosity=1)
    # report(mach)
    fitted_params(mach).best_model
    entry = report(mach).best_history_entry
    plot(mach)

    # Train on whole dataset given good num_iterations
    # vs .py matching sum weight and label
    # mach = machine(estimator, select(df_t, Not(:ts)), y_t, v_weight_t)
    # fit!(mach, verbosity=2)
    # yhat = predict(mach, select(df_t, Not(:ts)))
    # l2(yhat, y_t) |> mean  # 2.9779710871222236e-5   vs  2.9872256622699176e-05  in py

    # GC.gc()
end

function main()
    exchange = "bitfinex"
    sym = "ethusd"
    ex = get_ex(sym=sym)
    start = Date(2022, 2, 7)
    stop = Date(2022, 2, 13)
    # end_ = Date(2022, 9, 3)
    println("From $(start) to $(stop)")
    
    features = []
    features = missing
    if false
        df, y = load_inputs(exchange, sym, start, stop, features)
        serialize("temp/df.jls", df)
        serialize("temp/y.jls", y)
    else
        df = deserialize("temp/df.jls")
        y = deserialize("temp/y.jls")
    end
    train(df, y, ex, exchange)
    # save()
    println("Done. Ex: $(ex)")
end