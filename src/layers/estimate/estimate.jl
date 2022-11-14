using PyCall
using Serialization
using Dates
using DataFrames
using DataStructures
using MLJ
using Plots
using LightGBM
import ScikitLearn.CrossValidation: KFold
import .Client
import .path_layers

datetime = pyimport("datetime")
np = pyimport("numpy")
push!(pyimport("sys")."path", root)
push!(pyimport("sys")."path", joinpath(root, "src"))
push!(pyimport("sys")."path", joinpath(path_layers, "estimate"))
LoadXY = pyimport("load_xy").LoadXY
get_ex(;sym="") = "ex$(Dates.format(Dates.now(Dates.UTC), dateformat"Y-mm-dd_HHMMSS"))-$(sym)"


function purge_overlap(ix_tr::Vector{Int}, ix_te::Vector{Int}; n_boundary::Int=250)
    """
    In each iteration remove i periods from around test.
    i should be derived somewhat intelligently ...
    """
    if ix_te[1] == 1  # test on left side
        return ix_tr[n_boundary:end]
    elseif ix_te[1] > ix_tr[end]  # test on right side
        return ix_tr[1:end-n_boundary]
    else  # train surrounded by test
        return ix_tr[n_boundary:end-n_boundary]
    end
end


function load_inputs(exchange, sym, start, stop; features=missing)
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

function return_attribution_sample_weights(v_label; return_amplifier=5000)::Vector{Float64}
    """
    instance with high absolute return change should get a much higher weight. ignore the noise.
    overflow / underflow: normalize weights to have the smallest weight a weight of 1. Also easier intuitively later when plotting.
    """
    v = abs.(v_label .- 1) .* return_amplifier .+ 1
    return v ./ minimum(v)
end

function train_cv(df_t, y_t, v_weight_t, booster, estimator_params, n_cv_folds)
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
        resampling=CV(nfolds=n_cv_folds),
        iteration_parameter=:(num_iterations),
        measures=l2,
        controls=[
            Step(2),
            Patience(5),
            NumberLimit(1000),
            WithLossDo(f=x->@info("WithLossDo: $x"), stop_if_true=false, stop_message=nothing),
        ],
        retrain=false,
        cache=false,
        )
    mach = machine(iterated_model, select(df_t, Not(:ts)), y_t, v_weight_t, cache=false)
    fit!(mach, verbosity=10)
    return mach
end


function train(df, y, sym, ex::String)
    """t, ho  ;   t -> t_cv test_cv"""

    booster = @load LGBMRegressor

    v_ix_t, v_ix_ho = MLJ.partition(1:size(df)[1], 0.3, shuffle=false)
    # reducing df. mem issue
    df = df[v_ix_t, :]
    y = y[v_ix_t]
    # full one
    df_t, df_ho = df[v_ix_t, :], df[v_ix_ho, :]
    y_t, y_ho = y[v_ix_t], y[v_ix_ho]

    v_weight_t = return_attribution_sample_weights(y_t)
    v_weight_ho = return_attribution_sample_weights(y_ho)

    df=nothing
    y=nothing
    GC.gc()
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
        (:learning_rate, 0.001),
        # (:early_stopping_round, 1),  # how no effect in IteratedModel
        (:boosting, "gbdt"),
        (:lambda_l1, 0.95),
        (:lambda_l2, 0.8),
        (:num_leaves, 22),
        (:min_data_in_leaf, 7),
        # (:num_iterations, 1),  # how no effect in IteratedModel. But used in fit!(estimator).
        (:device_type, "gpu"),
    ])
    mach = train_cv(df_t, y_t, v_weight_t, booster, estimator_params, n_cv_folds)
    yhat_t = predict(mach, select(df_t, Not(:ts)))
    yhat_ho = predict(mach, select(df_ho, Not(:ts)))
    println("Train L2: $(l2(yhat_t, y_t, v_weight_t) |> mean)")  # .0005266576406430649
    println("HO L2: $(l2(yhat_ho, y_ho, v_weight_ho) |> mean)")  # 0.00036236425249246775

    

    learning_rate = range(estimator, :learning_rate, lower=0.001, upper=0.02)
    lambda_l1 = range(estimator, :lambda_l1, lower=0.8, upper=1.)
    lambda_l2 = range(estimator, :lambda_l2, lower=0.8, upper=1.)
    num_leaves = range(estimator, :num_leaves, lower=15, upper=41)
    min_data_in_leaf = range(estimator, :min_data_in_leaf, lower=7, upper=42)
    # max_bin = range(estimator, :max_bin, lower=63, upper=510)

    # space = Dict(
    #     :learning_rate => HP.LogUniform(:learning_rate, .01, .2),
    #     :lambda_l1 => HP.LogUniform(:lambda_l1, .5, 1.),
    #     :lambda_l2 => HP.LogUniform(:lambda_l2, .5, 1.),
    #     :num_leaves => HP.Choice(:num_leaves, collect(7:7:70)),
    #     :min_data_in_leaf => HP.Choice(:min_data_in_leaf, collect(10:10:40)),
    #     :max_bin => HP.Choice(:max_bin, collect(63:50:510)),
    #     # :max_depth => HP.QuantUniform(:max_depth, 1., ceil(log2(training_data_per_fold)), 1.0),
    #     # :alpha => HP.LogUniform(:alpha, -5., 2.),
    # )

    self_tuning_tree = TunedModel(
        model=estimator,
        resampling=CV(nfolds=n_cv_folds),
        tuning=Grid(resolution=5),
        range=[learning_rate, lambda_l1, lambda_l2, num_leaves, min_data_in_leaf],
        # tuning=MLJTreeParzenTuning(),
        # range=space,
        measure=rms,
        n = 500,
        train_best = true,
        );
    mach = machine(self_tuning_tree, select(df_t, Not(:ts)), y_t, v_weight_t)
    fit!(mach, verbosity=10)
    if false
        serialize("temp/mach2.jls", mach)
    else
        mach = deserialize("temp/mach2.jls")
    end 

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

    yhat_t = predict(mach, select(df_t, Not(:ts)))
    yhat_ho = predict(mach, select(df_ho, Not(:ts)))
    l2(yhat_t, y_t, v_weight_t) |> mean  # .0005266576406430649
    l2(ones(size(yhat_t)), y_t, v_weight_t) |> mean
    l1(yhat_t, y_t, v_weight_t) |> mean
    l1(ones(size(yhat_t)), y_t, v_weight_t) |> mean

    l2(yhat_ho, y_ho, v_weight_ho) |> mean  # 0.00036236425249246775
    l2(ones(size(y_ho)), y_ho, v_weight_ho) |> mean
    l1(yhat_ho, y_ho, v_weight_ho) |> mean
    l1(ones(size(y_ho)), y_ho, v_weight_ho) |> mean
    serialize("temp/mach2.jls", mach)

    serialize("temp/yhat_ho.jls", rename!(hcat(select(df_ho, :ts), yhat_ho, y_ho, makeunique=true), [:ts, :pred, :label]))
    CSV.write("temp/yhat_ho.csv", rename!(hcat(select(df_ho, :ts), yhat_ho, y_ho, makeunique=true), [:ts, :pred, :label]))
end

function main()
    exchange = "bitfinex"
    sym = "ethusd"
    ex = get_ex(sym=sym)
    start = Date(2022, 2, 7)
    # stop = Date(2022, 2, 13)
    stop = Date(2022, 9, 3)
    println("From $(start) to $(stop)")

    if false
        df, y = load_inputs(exchange, sym, start, stop)
        serialize(joinpath(path_temp, "df.jls"), df)
        serialize(joinpath(path_temp, "y.jls"), y)
    else
        df = deserialize(joinpath(path_temp, "df.jls"))
        y = deserialize(joinpath(path_temp, "y.jls"))
    end
    train(df, y, ex, exchange)
    # save()
    println("Done. Ex: $(ex)")
end