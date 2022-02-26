
function print_equation(;M, M_prob, par_names, sys_names, cutoff_prob = 0.5, lower = 0.025, upper = 0.975)

    M_est = round.(mean(M, dims = 3)[:,:,1], digits = 3)
    M_prob_est = mean(M_prob, dims = 3)[:,:,1]
    a = size(M_est)[1]
    b = size(M_est)[2]

    lower = round.([quantile(M[i,j,:], lower) for i in 1:a, j in 1:b], digits = 3)
    upper = round.([quantile(M[i,j,:], upper) for i in 1:a, j in 1:b], digits = 3)

    included = M_prob_est .> cutoff_prob
    final_eqs = copy(sys_names)
    final_lower = copy(sys_names)
    final_upper = copy(sys_names)

    n_eqs = size(final_eqs)[1]

    for i in 1:n_eqs

        found = map((x,y) -> string(x, ' ', y), M_est[i,included[i,:]], par_names[included[i,:]])
        no_sign = replace.(found, "-" => "")
        signs = sign.(M_est[i,included[i,:]])
        signs = [(signs[i] == 1 ? "+" : "-") for i in 1:size(signs)[1]]
        if signs[1] == "+"
            replace(signs[1], "+" => "")
        end
        found = join(map((x,y) -> string(x, ' ', y), signs, no_sign), " ")

        final_eqs[i] = string(final_eqs[i], " = ", found)
    end

    for i in 1:n_eqs

        found = map((x,y) -> string(x, ' ', y), lower[i,included[i,:]], par_names[included[i,:]])
        no_sign = replace.(found, "-" => "")
        signs = sign.(lower[i,included[i,:]])
        signs = [(signs[i] == 1 ? "+" : "-") for i in 1:size(signs)[1]]
        if signs[1] == "+"
            replace(signs[1], "+" => "")
        end
        found = join(map((x,y) -> string(x, ' ', y), signs, no_sign), " ")

        final_lower[i] = string(final_lower[i], " = ", found)
    end

    for i in 1:n_eqs

        found = map((x,y) -> string(x, ' ', y), upper[i,included[i,:]], par_names[included[i,:]])
        no_sign = replace.(found, "-" => "")
        signs = sign.(upper[i,included[i,:]])
        signs = [(signs[i] == 1 ? "+" : "-") for i in 1:size(signs)[1]]
        if signs[1] == "+"
            replace(signs[1], "+" => "")
        end
        found = join(map((x,y) -> string(x, ' ', y), signs, no_sign), " ")

        final_upper[i] = string(final_upper[i], " = ", found)
    end

    out = Dict()
    out["mean"] = final_eqs
    out["lower"] = final_lower
    out["upper"] = final_upper

    return out
end
