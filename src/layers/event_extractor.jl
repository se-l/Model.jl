import Statistics: std

function sample_events(v::Array{Real}; max_events=missing, thresholds=(-3, 3), method="std")::Array{Integer}
    v2 = filter(x -> x != missing, v)
    if method == "std"
        threshold = std(v) * thresholds[end]
        ix = findall(x -> x >= threshold, abs.(v2))
        @info("Sampling $(size(ix)) events")
        return size(ix) > max_events ? [] : ix
    end
end