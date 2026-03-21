# CFR+ for Kuhn Poker - Julia benchmark
# cross-language speed comparison against the Python implementation

const CARDS = ["J", "Q", "K"]
const NUM_ACTIONS = 2  # pass=1, bet=2

# all possible 2-card deals from 3 cards
const ALL_DEALS = [(i, j) for i in 0:2 for j in 0:2 if i != j]

function is_terminal(history::String)
    return history in ("pp", "bp", "bb", "pbp", "pbb")
end

function terminal_payoff(history::String, cards::Tuple{Int,Int})
    # payoff for player 0
    if history == "bp"
        return 1  # p1 folded
    elseif history == "pbp"
        return -1  # p0 folded
    end

    # showdown
    winner_is_p0 = cards[1] > cards[2]
    if history == "pp"
        return winner_is_p0 ? 1 : -1
    else  # "bb" or "pbb"
        return winner_is_p0 ? 2 : -2
    end
end

function info_key(card::Int, history::String)
    name = CARDS[card + 1]
    if isempty(history)
        return name
    end
    return name * " " * history
end

mutable struct InfoNode
    regret_sum::Vector{Float64}
    strategy_sum::Vector{Float64}
    num_actions::Int
end

InfoNode(n::Int) = InfoNode(zeros(n), zeros(n), n)

function current_strategy!(node::InfoNode)
    # regret matching with CFR+ flooring
    strat = max.(node.regret_sum, 0.0)
    total = sum(strat)
    if total > 0
        strat ./= total
    else
        strat .= 1.0 / node.num_actions
    end
    return strat
end

function average_strategy(node::InfoNode)
    total = sum(node.strategy_sum)
    if total > 0
        return node.strategy_sum ./ total
    end
    return fill(1.0 / node.num_actions, node.num_actions)
end

# global node map
const nodes = Dict{String, InfoNode}()

function cfr(cards::Tuple{Int,Int}, history::String, reach0::Float64, reach1::Float64)
    if is_terminal(history)
        return Float64(terminal_payoff(history, cards))
    end

    player = length(history) % 2
    card = player == 0 ? cards[1] : cards[2]
    key = info_key(card, history)

    node = get!(nodes, key) do
        InfoNode(NUM_ACTIONS)
    end

    strat = current_strategy!(node)
    action_values = zeros(NUM_ACTIONS)
    node_value = 0.0

    for a in 1:NUM_ACTIONS
        action_char = a == 1 ? "p" : "b"
        next_hist = history * action_char

        if player == 0
            action_values[a] = cfr(cards, next_hist, reach0 * strat[a], reach1)
        else
            action_values[a] = -cfr(cards, next_hist, reach0, reach1 * strat[a])
        end
        node_value += strat[a] * action_values[a]
    end

    # update regrets and strategy sums
    opp_reach = player == 0 ? reach1 : reach0
    my_reach = player == 0 ? reach0 : reach1

    for a in 1:NUM_ACTIONS
        regret = action_values[a] - node_value
        node.regret_sum[a] += opp_reach * regret
        # CFR+: floor negative regrets
        node.regret_sum[a] = max(node.regret_sum[a], 0.0)
        node.strategy_sum[a] += my_reach * strat[a]
    end

    return node_value
end

function compute_exploitability()
    # rough estimate via best response value difference
    total = 0.0
    for deal in ALL_DEALS
        total += cfr(deal, "", 1.0, 1.0)
    end
    game_val = total / length(ALL_DEALS)
    return abs(game_val - (-1/18))
end

function main()
    num_iters = 50_000
    game_value = 0.0

    println("Training CFR+ on Kuhn Poker for $num_iters iterations...")
    t_start = time()

    for i in 1:num_iters
        for deal in ALL_DEALS
            game_value += cfr(deal, "", 1.0, 1.0)
        end
    end

    elapsed = time() - t_start
    avg_value = game_value / (num_iters * length(ALL_DEALS))
    exploit = compute_exploitability()

    println()
    println("Results:")
    println("  Game value:      $(round(avg_value, digits=6))  (theory: $(round(-1/18, digits=6)))")
    println("  Exploitability:  $(round(exploit, digits=6))")
    println("  Wall time:       $(round(elapsed, digits=3)) s")
    println("  Iterations:      $num_iters")
    println("  Info sets:       $(length(nodes))")
    println()

    # print key strategies
    println("Key strategies:")
    for key in sort(collect(keys(nodes)))
        avg = average_strategy(nodes[key])
        println("  $key: pass=$(round(avg[1], digits=3)) bet=$(round(avg[2], digits=3))")
    end

    println()
    println("Run `python solve.py` and compare wall times")
end

main()
