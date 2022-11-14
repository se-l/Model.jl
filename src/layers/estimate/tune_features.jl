using ShapML
using DataFrames
using MLJ  # Machine learning
using Gadfly  # Plotting
using LightGBM

x = df[ix_tr, Not(:ts)]
explain = copy(x[1:30000, :]) # Compute Shapley feature-level predictions for 300 instances.

reference = copy(x)  # An optional reference population to compute the baseline prediction.

sample_size = 60  # Number of Monte Carlo samples.

function predict_function(estimator, df::DataFrame)
    return DataFrame(LightGBM.predict(estimator, Matrix(df)), :auto)
end

data_shap = ShapML.shap(explain = explain,
                        reference = reference,
                        model = estimator,
                        predict_function = predict_function,
                        sample_size = sample_size,
                        seed = 1
                        )

show(data_shap, allcols = true)



data_plot = combine(groupby(data_shap, [:feature_name]), :shap_effect => mean)
rename!(data_plot, :shap_effect_mean => :mean_effect)
data_plot = sort(data_plot, order(:mean_effect, rev = true))

baseline = round(data_shap.intercept[1], digits = 1)

p = Gadfly.plot(data_plot, y = :feature_name, x = :mean_effect, Coord.cartesian(yflip = true),
         Scale.y_discrete, Geom.bar(position = :dodge, orientation = :horizontal),
         Theme(bar_spacing = 1mm),
         Guide.xlabel("|Shapley effect| (baseline = $baseline)"), Guide.ylabel(nothing),
         Guide.title("Feature Importance - Mean Absolute Shapley Value"))
p