
function hpd(x::Vector; p = 0.95)
    p1 = round((1-p)/2, digits = 5)
    p2 = round(p + (1-p)/2, digits = 5)

    dens = kde(x)
    Δn = dens.x[2] - dens.x[1]
    inds = (cumsum(dens.density * Δn) .> p1) .& (cumsum(dens.density * Δn) .< p2)

    # lower = dens.x[inds][1]
    # upper = dens.x[inds][end]

    if sum(inds) < 2
        lower = 0
        upper = 0
    else
        lower = dens.x[inds][1]
        upper = dens.x[inds][end]
    end

    return lower, upper
end

# hpd(tmp)
