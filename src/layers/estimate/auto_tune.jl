using Serialization
using DataFrames
using DataStructures
using MLJ
using Plots
using LightGBM
using Hyperopt
# using Distributed
using Statistics
import ScikitLearn.CrossValidation: KFold
using PlotlyJS


function optimize(
        df::DataFrame,
        y::Vector{<:Real}; n_trials=3,
        n_processes=0,
        n_cv_folds=4,
        f_partition=0.4  # refactor to max Float64 / mem size
    )::Hyperoptimizer
    # addprocs(n_processes)
    # @everywhere using Model
    # @everywhere include("auto_tune.jl")

    v_ix_t, _ = MLJ.partition(1:size(df)[1], f_partition, shuffle=false)
    df_t = df[v_ix_t, :]
    y_t = y[v_ix_t]
    df = nothing
    y = nothing
    GC.gc()
    feature_names = filter((x)-> x != "ts", names(df_t))

    @info("Splitting df into $(n_cv_folds) datasets")
    vds_tr = []
    vx_te = []
    vy_te = []

    for (ix_tr, ix_te) in KFold(size(df_t)[1], n_folds=n_cv_folds)
        ix_tr = purge_overlap(ix_tr, ix_te)
        # Fit the estimator on the training data and return its scores for the test data.
        ds_tr = LightGBM.LGBM_DatasetCreateFromMat(Matrix(df_t[ix_tr, Not(:ts)]), "")
        LightGBM.LGBM_DatasetSetField(ds_tr, "label", y_t[ix_tr])
        LightGBM.LGBM_DatasetSetField(ds_tr, "weight", return_attribution_sample_weights(y_t[ix_tr]))
        LightGBM.LGBM_DatasetSetFeatureNames(ds_tr, feature_names)
        push!(vds_tr, ds_tr)

        push!(vx_te, Matrix(df_t[ix_te, Not(:ts)]))
        push!(vy_te, y_t[ix_te])
    end

    @info("Optimizing df of shape: $(size(df_t))")
    ho = @hyperopt for i=n_trials,
        sampler = RandomSampler(),
        learning_rate = LinRange(0.001, 0.003, 2),
        num_leaves = round.(Int, LinRange(100, 2000, 10)),
        max_depth = round.(Int, LinRange(3, 12, 10)),
        min_data_in_leaf = round.(Int, LinRange(200, 2000, 10)),
        lambda_l1 = LinRange(0, 100, 10),
        lambda_l2 = LinRange(0, 100, 10),
        min_gain_to_split = LinRange(0, 5, 5),
        bagging_fraction = LinRange(0.4, 0.95, 5),
        bagging_freq = LinRange(1, 1, 1),
        feature_fraction = LinRange(0.1, 1., 10)

        num_leaves::Int, max_depth::Int, min_data_in_leaf

        @show train_cv(vds_tr, vx_te, vy_te, 
            learning_rate, num_leaves, max_depth, min_data_in_leaf, 
            lambda_l1, lambda_l2, min_gain_to_split, bagging_fraction, bagging_freq, feature_fraction)
    end
    return ho
end

function ana()
    histogram(df[:, fs][:, 1])
    histogram(y)
    f = df_imp[:, :feature][1]
    df_f = df[:, f]
    q_low, q_high = quantile(df_f, 0.05), quantile(df_f, 0.95)
    v_ix = findall(x -> x >= q_high || x <= q_low, df_f)
    PlotlyJS.plot(histogram2d(x=y[v_ix], y=df_f[v_ix], title=f))
end

function train_predict(df::DataFrame, y::Vector{Float64}; f_partition=0.7)
    ix_tr, ix_te = MLJ.partition(1:size(df)[1], f_partition, shuffle=false)
    
    estimator = LGBMRegression(
        objective = "regression",
        metric = ["l2"],
        # early_stopping_round = 5,
        num_iterations = 5,
        max_depth=-1,
        learning_rate = 0.003,
        feature_fraction = 0.8,
        bagging_fraction = 0.5375,
        bagging_freq = 1,
        num_leaves = 522,
        min_data_in_leaf = 600,
        lambda_l1=11.11,
        lambda_l2=44.44,
        min_gain_to_split=0,
        device_type = "gpu",
    )
    fs = names(df[:, Not(:ts)])
    w_tr = return_attribution_sample_weights(y[ix_tr])
    y_tr = (y[ix_tr])# .-1).*w_tr .+ 1
    w_te  = return_attribution_sample_weights(y[ix_te])
    y_te = (y[ix_te])# .-1).*w_te .+ 1
    y_trm = mean(y_tr)
    y_tem = mean(y_te)

    LightGBM.fit!(estimator, Matrix(df[ix_tr, fs]), y[ix_tr], weights=w_tr, verbosity=10)
    LightGBM.fit!(estimator, Matrix(df[ix_tr, fs]), y_tr, verbosity=10)
    # eval_metrics(
    yh_tr = LightGBM.predict(estimator, Matrix(df[ix_tr, fs]))[:, 1]
    yh_te = LightGBM.predict(estimator, Matrix(df[ix_te, fs]))[:, 1]
    # )
    
    df_imp = DataFrame(zip(names(df[:, Not(:ts)]), LightGBM.gain_importance(estimator), LightGBM.split_importance(estimator)))
    rename!(df_imp, [Pair(:1, :feature), Pair(:2, :gain),  Pair(:3, :split)])
    sort!(df_imp, :gain, rev=true)
    sort!(df_imp, :split, rev=true)
    sum(df_imp[:, :gain])
    sum(df_imp[1:15, :gain])

    fs = df_imp[1:3, :feature][1:1]

    histogram(w_tr .* sign.(y_tr .- 1) )
    p = histogram(y_tr)
    p = histogram(y_te)
    p = histogram(yh_tr)
    p = histogram(yh_te)
    display(p)

    # 1. New check - are tr and te on same side of the mean. Train Yes, 
        # Test: No - Learnt anything? Learnt nothing!
    # 2. Plot tree splits. Understand feature importance, 
    #   underlying cause => Enhance.
    LightGBM.savemodel(estimator, joinpath(path_temp, "booster.lgb"))

    q_low, q_high = quantile(yh_tr, 0.01), quantile(yh_tr, 0.99)

    vq_high_ix = findall(x -> x >= q_high, yh_tr)
    vqh_high_tr =yh_tr[vq_high_ix]
    vq_high_t =y_tr[vq_high_ix]
    sum((vq_high_t .< y_trm) .==  (vqh_high_tr .< y_trm)) / length(vq_high_ix)
    # 0.998

    vq_low_ix = findall(x->x<=q_low, yh_tr)
    vqh_low_tr =yh_tr[vq_low_ix]
    vq_low_t =y_tr[vq_low_ix]
    sum((vq_low_t .> y_trm) .==  (vqh_low_tr .> y_trm)) / length(vq_low_ix)
    # 0.992

    q_low, q_high = quantile(yh_te, 0.01), quantile(yh_te, 0.99)

    vq_high_ix = findall(x -> x >= q_high, yh_te)
    vqh_high_tr =yh_te[vq_high_ix]
    vq_high_t =y_te[vq_high_ix]
    # histogram(vq_high_t)
    sum((vq_high_t .> y_tem) .& (vqh_high_tr .> y_tem)) / length(vq_high_ix)
    # 0.42

    vq_low_ix = findall(x -> x <= q_low, yh_te)
    vqh_low_tr =yh_te[vq_low_ix]
    vq_low_t =y_te[vq_low_ix]
    histogram(vqh_low_tr)
    histogram(vq_low_t)
    sum((vq_low_t .< y_tem) .&  (vqh_low_tr .< y_tem)) / length(vq_low_ix)
    # 0.31

    sum((yh_tr .> 1) .==  (y_tr .> 1)) / size(y_tr)[1]
    sum((yh_te .> 1) .==  (y_te .> 1)) / size(y_te)[1]
    sum((yh_te .< 0.995) .==  (y_te .> 1)) / size(y_te)[1]
    sum((yh_te .< 0.995) .==  (y_te .> 1)) / size(y_te)[1]
end

function eval_metrics(;yh_tr, y_tr, w_tr, yh_te, y_te, w_te)
    """
    The loss function is not realistic. 
        - Punish opposite side Error heavily
        - Almost no punishment for predicting same side, coz no loss
        - Punish missed out opportunity? -> Encourage taking sides... as long as it's the right one...

    Apparently, frequently opportunity does not lead to high reward, may be on right side though.
    """
    l2(yh_tr, y_tr) |> mean  # .0005266576406430649
    l2(ones(size(yh_tr)), y_tr) |> mean

    l2(yh_tr, y_tr, w_tr) |> mean  # .0005266576406430649
    l2(ones(size(yh_tr)), y_tr, w_tr) |> mean
    l1(yh_tr, y_tr, w_tr) |> mean
    l1(ones(size(yh_tr)), y_tr, w_tr) |> mean

    l2(yh_te, y_te, w_te) |> mean  # 0.00036236425249246775
    l2(ones(size(y_te)), y_te, w_te) |> mean
    l1(yh_te, y_te, w_te) |> mean
    l1(ones(size(y_te)), y_te, w_te) |> mean

    minimum(y_tr)
    maximum(y_tr)

    minimum(yh_tr)
    maximum(yh_tr)

    minimum(yh_te)
    maximum(yh_te)    
end

function train_cv(
        vds_tr::Vector{Any},
        vx_te::Vector{Any},
        vy_te::Vector{Any},
        learning_rate::Real,
        num_leaves::Int,
        max_depth::Int,
        min_data_in_leaf::Int,
        lambda_l1::Real,
        lambda_l2::Real,
        min_gain_to_split::Real,
        bagging_fraction::Real,
        bagging_freq::Real,
        feature_fraction::Real;
    )
    estimator = get_estimator(1000, learning_rate, num_leaves, max_depth, min_data_in_leaf, 
                                lambda_l1, lambda_l2, min_gain_to_split, bagging_fraction, bagging_freq, feature_fraction)
    ds_parameters = LightGBM.stringifyparams(estimator; verbosity=10)
    
    v_err = []
    for i in eachindex(vds_tr)
        # Create Test Dataset
        ds_te = LightGBM.dataset_constructor(vx_te[i], ds_parameters, false, vds_tr[i])
        LightGBM.LGBM_DatasetSetField(ds_te, "label", vy_te[i])
        LightGBM.LGBM_DatasetSetField(ds_te, "weight", return_attribution_sample_weights(vy_te[i]))

        res = LightGBM.fit!(estimator, vds_tr[i], ds_te, verbosity=10)

        if "test_1" in keys(res["metrics"])
            push!(v_err, minimum(res["metrics"]["test_1"]["l2"]))
        else
            push!(v_err, 1)
        end
    end
    # Fully featured auto tune may need this smarter-than-grid params optimization, then optimize features and back to step 1 until min.
    return mean(v_err)  # consider returning whole res and estimator if relevant.
end

# v_err_train = train_early_stopping(estimator, vx_tr[i], vy_tr[i], vw_tr[i], vx_te[i], vy_te[i], vw_te[i])
# function train_early_stopping(
#         estimator::LGBMEstimator,
#         x_tr::LightGBM.Dataset,
#         y_tr::Vector{<:Real},
#         w_tr::Vector{<:Real},
#         x_te::AbstractMatrix{<:Real},
#         y_te::Vector{<:Real},
#         w_te::Vector{<:Real};
#         num_iterations::Int=1000,
#         early_stopping_round::Int=5
#     )::Vector{<:Real}
#     v_err = []
#     for iteration in 1:num_iterations
#         if iteration == 1
#             LightGBM.fit!(estimator, x_tr, y_tr, (x_te[1:1, :], y_te[:1, :]), weights=w_tr, verbosity=10, truncate_booster=false)
#         else
#             LightGBM.train!(estimator, 1, ["test_1"], 10, Dates.now(), truncate_booster=false)
#         end
#         yh_te = LightGBM.predict(estimator, x_te)[:, 1]

#         # replace with passing err function
#         weighted_err = MLJ.l2(yh_te, y_te, w_te) |> mean
#         @info("Weighted Error on Test; CV: $(i); Iteration $(iteration): $(weighted_err)")
#         push!(v_err, weighted_err)
#         min_, ix_min = findmin(v_err)
#         if ix_min + early_stopping_round <= iteration
#             @info("Stopping at iteration $(ix_min). Best iteration: $(ix_min)")
#             break
#         end
#     end
#     return v_err
# end


function get_estimator(num_iterations::Int, learning_rate::Real, num_leaves::Int, max_depth::Int, min_data_in_leaf::Int, 
    lambda_l1::Real, lambda_l2::Real, min_gain_to_split::Real, bagging_fraction::Real, bagging_freq::Real, feature_fraction::Real)::LGBMEstimator
    return LGBMRegression(
        objective = "regression",
        metric = ["l2"],
        early_stopping_round = 5,
        num_iterations = num_iterations,
        max_depth=max_depth,
        learning_rate = learning_rate,
        feature_fraction = feature_fraction,
        bagging_fraction = bagging_fraction,
        bagging_freq = bagging_freq,
        num_leaves = num_leaves,
        min_data_in_leaf = min_data_in_leaf,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        min_gain_to_split=min_gain_to_split,
        device_type = "gpu",
    )
end


function save(ho::Hyperoptimizer)
    serialize(joinpath(path_temp, "hyperopt.jls"), ho)
end

function present(ho::Hyperoptimizer)
    best_params, min_f = ho.minimizer, ho.minimum
    @info("$(best_params)")
    @info("min_f: $(min_f)")
    @info(ho)
    plot(ho)
end


function test_optimize(;from_disk=true)
    exchange = "bitfinex"
    sym = Asset.ethusd
    ex = get_ex(sym)
    start = Date(2022, 2, 7)
    # stop = Date(2022, 2, 13)
    stop = Date(2022, 9, 3)
    @info("From $(start) to $(stop)")
    
    if !from_disk
        df, y = load_inputs(exchange, sym, start, stop)
        serialize(joinpath(path_temp, "df.jls"), df)
        serialize(joinpath(path_temp, "y.jls"), y)
    else
        df = deserialize(joinpath(path_temp, "df.jls"))
        y = deserialize(joinpath(path_temp, "y.jls"))
    end
    ho = optimize(df, y, n_trials=100, n_processes=0, f_partition=0.4)::Hyperoptimizer
    save(ho)
    present(ho)
    @info("Done.")
    return ho
    #     minimum / maximum: (7.697964987407978e-5, 1.0)
    #   minimizer:
    # learning_rate num_leaves max_depth min_data_in_leaf lambda_l1 lambda_l2 min_gain_to_split bagging_fraction bagging_freq feature_fraction
    #     0.003       522        11           600           11.11       44.44         0                0.5375         1       0.8
end