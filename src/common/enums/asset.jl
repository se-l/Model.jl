module Asset

export T
@enum T usd ethusd solusd xrpusd btcusd adausd

end # module

Base.tryparse(E::Type{<:Enum}, str::String) =
    let insts = instances(E) ,
        p = findfirst(==(Symbol(str)) ∘ Symbol, insts) ;
        p !== nothing ? insts[p] : nothing
    end

# Alternatively https://docs.julialang.org/en/v1/base/base/#Base.@NamedTuple