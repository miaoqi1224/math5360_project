#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct DateTime {
    int year {};
    int month {};
    int day {};
    int hour {};
    int minute {};
};

struct Bar {
    DateTime dt;
    double open {};
    double high {};
    double low {};
    double close {};
    double volume {};
};

struct MarketConfig {
    std::string ticker;
    std::string data_file;
    std::string exchange;
    std::string currency;
    double point_value {};
    double slippage {};
    double tick_size {};
    double tick_value {};
};

struct StrategyParams {
    int channel_length {};
    double stop_pct {};
};

struct BacktestSummary {
    double profit {};
    double worst_drawdown {};
    double pnl_stddev {};
    double trade_count {};
};

struct VarianceRatioResult {
    int horizon_bars {};
    double value {};
    std::string interpretation;
};

struct PushResponseBin {
    double left_edge {};
    double right_edge {};
    double mean_push {};
    double mean_response {};
    int count {};
};

struct PushResponseResult {
    int horizon_bars {};
    double response_beta {};
    double signed_response {};
    std::string interpretation;
    std::vector<PushResponseBin> bins;
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;

    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }

    return fields;
}

DateTime parse_datetime(const std::string& date, const std::string& time) {
    DateTime dt;
    char slash1 = '\0';
    char slash2 = '\0';
    char colon = '\0';

    std::stringstream ds(date);
    ds >> dt.month >> slash1 >> dt.day >> slash2 >> dt.year;
    if (!ds || slash1 != '/' || slash2 != '/') {
        throw std::runtime_error("Failed to parse date: " + date);
    }

    std::stringstream ts(time);
    ts >> dt.hour >> colon >> dt.minute;
    if (!ts || colon != ':') {
        throw std::runtime_error("Failed to parse time: " + time);
    }

    return dt;
}

std::vector<Bar> load_bars_from_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open CSV file: " + path);
    }

    std::vector<Bar> bars;
    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty: " + path);
    }

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() != 7) {
            throw std::runtime_error("Unexpected CSV column count in line: " + line);
        }

        Bar bar;
        bar.dt = parse_datetime(fields[0], fields[1]);
        bar.open = std::stod(fields[2]);
        bar.high = std::stod(fields[3]);
        bar.low = std::stod(fields[4]);
        bar.close = std::stod(fields[5]);
        bar.volume = std::stod(fields[6]);
        bars.push_back(bar);
    }

    return bars;
}

void print_dataset_summary(const std::vector<Bar>& bars) {
    if (bars.empty()) {
        std::cout << "No bars loaded.\n";
        return;
    }

    const Bar& first = bars.front();
    const Bar& last = bars.back();

    std::cout << "Loaded bars: " << bars.size() << "\n";
    std::cout << "First bar: "
              << first.dt.year << "-" << std::setw(2) << std::setfill('0') << first.dt.month
              << "-" << std::setw(2) << first.dt.day
              << " " << std::setw(2) << first.dt.hour
              << ":" << std::setw(2) << first.dt.minute << "\n";
    std::cout << "Last bar:  "
              << last.dt.year << "-" << std::setw(2) << last.dt.month
              << "-" << std::setw(2) << last.dt.day
              << " " << std::setw(2) << last.dt.hour
              << ":" << std::setw(2) << last.dt.minute << "\n";
    std::cout << std::setfill(' ');
}

std::vector<double> compute_log_returns(const std::vector<Bar>& bars) {
    std::vector<double> returns;
    if (bars.size() < 2) {
        return returns;
    }

    returns.reserve(bars.size() - 1);
    for (std::size_t i = 1; i < bars.size(); ++i) {
        returns.push_back(std::log(bars[i].close / bars[i - 1].close));
    }
    return returns;
}

double sample_variance(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }

    double mean = 0.0;
    for (double x : values) {
        mean += x;
    }
    mean /= static_cast<double>(values.size());

    double sum_sq = 0.0;
    for (double x : values) {
        const double diff = x - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(values.size() - 1);
}

double compute_variance_ratio(const std::vector<double>& one_bar_returns, int horizon_bars) {
    if (horizon_bars <= 0) {
        throw std::runtime_error("Variance ratio horizon must be positive.");
    }
    if (one_bar_returns.size() <= static_cast<std::size_t>(horizon_bars)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double one_bar_var = sample_variance(one_bar_returns);
    if (one_bar_var == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<double> aggregated_returns;
    aggregated_returns.reserve(one_bar_returns.size() - horizon_bars + 1);

    double rolling_sum = 0.0;
    for (int i = 0; i < horizon_bars; ++i) {
        rolling_sum += one_bar_returns[static_cast<std::size_t>(i)];
    }
    aggregated_returns.push_back(rolling_sum);

    for (std::size_t i = static_cast<std::size_t>(horizon_bars); i < one_bar_returns.size(); ++i) {
        rolling_sum += one_bar_returns[i];
        rolling_sum -= one_bar_returns[i - static_cast<std::size_t>(horizon_bars)];
        aggregated_returns.push_back(rolling_sum);
    }

    const double aggregated_var = sample_variance(aggregated_returns);
    return aggregated_var / (static_cast<double>(horizon_bars) * one_bar_var);
}

std::string interpret_variance_ratio(double value) {
    if (std::isnan(value)) {
        return "insufficient data";
    }
    if (value > 1.02) {
        return "trend-following";
    }
    if (value < 0.98) {
        return "mean-reverting";
    }
    return "close to random walk";
}

std::vector<VarianceRatioResult> run_variance_ratio_scan(const std::vector<Bar>& bars) {
    const std::vector<int> horizons {1, 3, 6, 12, 24, 48, 96};
    const std::vector<double> one_bar_returns = compute_log_returns(bars);

    std::vector<VarianceRatioResult> results;
    results.reserve(horizons.size());

    for (int horizon : horizons) {
        const double vr = compute_variance_ratio(one_bar_returns, horizon);
        results.push_back({horizon, vr, interpret_variance_ratio(vr)});
    }

    return results;
}

std::vector<double> compute_horizon_price_changes(const std::vector<Bar>& bars, int horizon_bars) {
    std::vector<double> changes;
    if (bars.size() <= static_cast<std::size_t>(horizon_bars)) {
        return changes;
    }

    changes.reserve(bars.size() - horizon_bars);
    for (std::size_t i = static_cast<std::size_t>(horizon_bars); i < bars.size(); ++i) {
        changes.push_back(bars[i].close - bars[i - static_cast<std::size_t>(horizon_bars)].close);
    }
    return changes;
}

double sample_covariance(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        return 0.0;
    }

    double mean_x = 0.0;
    double mean_y = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= static_cast<double>(x.size());
    mean_y /= static_cast<double>(y.size());

    double cov = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        cov += (x[i] - mean_x) * (y[i] - mean_y);
    }
    return cov / static_cast<double>(x.size() - 1);
}

std::string interpret_signed_predictability(double value) {
    if (std::isnan(value)) {
        return "insufficient data";
    }
    if (value > 0.0) {
        return "trend-following";
    }
    if (value < 0.0) {
        return "mean-reverting";
    }
    return "close to random walk";
}

PushResponseResult run_push_response_for_horizon(
    const std::vector<Bar>& bars,
    int horizon_bars,
    int bin_count
) {
    if (horizon_bars <= 0) {
        throw std::runtime_error("Push-response horizon must be positive.");
    }
    if (bars.size() <= static_cast<std::size_t>(2 * horizon_bars + 1)) {
        return {};
    }

    std::vector<double> pushes;
    std::vector<double> responses;

    for (std::size_t start = 0; start + static_cast<std::size_t>(2 * horizon_bars) < bars.size();
         start += static_cast<std::size_t>(horizon_bars)) {
        const std::size_t mid = start + static_cast<std::size_t>(horizon_bars);
        const std::size_t end = mid + static_cast<std::size_t>(horizon_bars);

        const double x = bars[mid].close - bars[start].close;
        const double y = bars[end].close - bars[mid].close;
        pushes.push_back(x);
        responses.push_back(y);
    }

    const double push_var = sample_variance(pushes);
    const double beta = push_var > 0.0
        ? sample_covariance(pushes, responses) / push_var
        : std::numeric_limits<double>::quiet_NaN();

    double signed_response = 0.0;
    for (std::size_t i = 0; i < pushes.size(); ++i) {
        const double sign = pushes[i] > 0.0 ? 1.0 : (pushes[i] < 0.0 ? -1.0 : 0.0);
        signed_response += sign * responses[i];
    }
    signed_response /= static_cast<double>(pushes.size());

    const auto [min_it, max_it] = std::minmax_element(pushes.begin(), pushes.end());
    const double min_push = *min_it;
    const double max_push = *max_it;
    const double width = (max_push - min_push) / static_cast<double>(bin_count);

    std::vector<PushResponseBin> bins;
    bins.reserve(static_cast<std::size_t>(bin_count));
    if (width > 0.0) {
        std::vector<double> push_sum(static_cast<std::size_t>(bin_count), 0.0);
        std::vector<double> response_sum(static_cast<std::size_t>(bin_count), 0.0);
        std::vector<int> counts(static_cast<std::size_t>(bin_count), 0);

        for (std::size_t i = 0; i < pushes.size(); ++i) {
            int idx = static_cast<int>((pushes[i] - min_push) / width);
            if (idx == bin_count) {
                idx = bin_count - 1;
            }
            push_sum[static_cast<std::size_t>(idx)] += pushes[i];
            response_sum[static_cast<std::size_t>(idx)] += responses[i];
            counts[static_cast<std::size_t>(idx)] += 1;
        }

        for (int i = 0; i < bin_count; ++i) {
            PushResponseBin bin;
            bin.left_edge = min_push + width * static_cast<double>(i);
            bin.right_edge = min_push + width * static_cast<double>(i + 1);
            bin.count = counts[static_cast<std::size_t>(i)];
            if (bin.count > 0) {
                bin.mean_push = push_sum[static_cast<std::size_t>(i)] / static_cast<double>(bin.count);
                bin.mean_response = response_sum[static_cast<std::size_t>(i)] / static_cast<double>(bin.count);
            }
            bins.push_back(bin);
        }
    }

    return {
        horizon_bars,
        beta,
        signed_response,
        interpret_signed_predictability(beta),
        bins
    };
}

std::vector<PushResponseResult> run_push_response_scan(const std::vector<Bar>& bars) {
    const std::vector<int> horizons {1, 3, 6, 12, 24, 48, 96};

    std::vector<PushResponseResult> results;
    results.reserve(horizons.size());
    for (int horizon : horizons) {
        results.push_back(run_push_response_for_horizon(bars, horizon, 7));
    }
    return results;
}

void print_market_summary(const MarketConfig& market, const std::vector<Bar>& bars) {
    std::cout << "\nMarket summary:\n";
    std::cout << "Ticker: " << market.ticker << "\n";
    std::cout << "Exchange: " << market.exchange << "\n";
    std::cout << "Currency: " << market.currency << "\n";
    std::cout << "Point value: " << market.point_value << "\n";
    std::cout << "Tick size: " << market.tick_size << "\n";
    std::cout << "Tick value: " << market.tick_value << "\n";
    std::cout << "Suggested round-turn slippage: " << market.slippage << "\n";
    std::cout << "Sample length: " << bars.size() << " five-minute bars\n";
    std::cout << "Interpretation: Brent crude is a liquid global oil benchmark, so it is a natural candidate for serial-dependence tests and trend-following research.\n";
}

void print_variance_ratio_report(const std::vector<VarianceRatioResult>& results) {
    std::cout << "\nVariance Ratio scan on log returns:\n";
    std::cout << "Horizon(bar)\tApprox horizon\tVR\tInterpretation\n";

    for (const auto& result : results) {
        const int minutes = result.horizon_bars * 5;
        std::cout << std::setw(12) << result.horizon_bars
                  << "\t" << std::setw(4) << minutes << " min"
                  << "\t" << std::fixed << std::setprecision(4) << result.value
                  << "\t" << result.interpretation << "\n";
    }

    std::cout << std::setprecision(6);
}

void print_push_response_report(const std::vector<PushResponseResult>& results) {
    std::cout << "\nPush-Response scan using adjacent non-overlapping windows:\n";
    std::cout << "Horizon(bar)\tApprox horizon\tBeta\tSignedResp\tInterpretation\n";

    for (const auto& result : results) {
        const int minutes = result.horizon_bars * 5;
        std::cout << std::setw(12) << result.horizon_bars
                  << "\t" << std::setw(4) << minutes << " min"
                  << "\t" << std::fixed << std::setprecision(4) << result.response_beta
                  << "\t" << std::fixed << std::setprecision(6) << result.signed_response
                  << "\t" << result.interpretation << "\n";
    }

    if (!results.empty()) {
        const PushResponseResult& sample = results[std::min<std::size_t>(2, results.size() - 1)];
        std::cout << "\nSample conditional response bins for horizon "
                  << sample.horizon_bars * 5 << " min:\n";
        std::cout << "MeanPush\tMeanResponse\tCount\n";
        for (const auto& bin : sample.bins) {
            if (bin.count == 0) {
                continue;
            }
            std::cout << std::fixed << std::setprecision(6)
                      << bin.mean_push << "\t" << bin.mean_response
                      << "\t" << bin.count << "\n";
        }
    }

    std::cout << std::setprecision(6);
}

BacktestSummary run_channel_with_dd_control(
    const std::vector<Bar>& bars,
    const MarketConfig& market,
    const StrategyParams& params
) {
    (void)bars;
    (void)market;
    (void)params;

    // TODO:
    // 1. Compute rolling HH/LL over channel_length.
    // 2. Replicate the entry/exit logic from main.m.
    // 3. Subtract slippage on each round turn using market.slippage.
    // 4. Use market.point_value to convert price changes into PnL.
    // 5. Return profit, drawdown, standard deviation, and trade count.
    return {};
}

int main() {
    const MarketConfig co_market {
        .ticker = "CO",
        .data_file = "/Users/regina/Desktop/5360 project/CO-5minHLV.csv",
        .exchange = "ICE",
        .currency = "USD",
        .point_value = 1000.0,
        .slippage = 48.0,
        .tick_size = 0.01,
        .tick_value = 10.0
    };

    const StrategyParams starter_params {
        .channel_length = 500,
        .stop_pct = 0.005
    };

    try {
        const std::vector<Bar> bars = load_bars_from_csv(co_market.data_file);

        std::cout << "Market: " << co_market.ticker << " (Brent Crude)\n";
        std::cout << "Point value: " << co_market.point_value << "\n";
        std::cout << "Slippage: " << co_market.slippage << "\n";
        print_dataset_summary(bars);
        print_market_summary(co_market, bars);

        const std::vector<VarianceRatioResult> vr_results = run_variance_ratio_scan(bars);
        const std::vector<PushResponseResult> pr_results = run_push_response_scan(bars);
        print_variance_ratio_report(vr_results);
        print_push_response_report(pr_results);

        const BacktestSummary summary =
            run_channel_with_dd_control(bars, co_market, starter_params);
        if (summary.trade_count > 0.0 || summary.profit != 0.0 ||
            summary.worst_drawdown != 0.0 || summary.pnl_stddev != 0.0) {
            std::cout << "\nBacktest summary:\n";
            std::cout << "Profit: " << summary.profit << "\n";
            std::cout << "Worst drawdown: " << summary.worst_drawdown << "\n";
            std::cout << "PnL stdev: " << summary.pnl_stddev << "\n";
            std::cout << "Trades: " << summary.trade_count << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
