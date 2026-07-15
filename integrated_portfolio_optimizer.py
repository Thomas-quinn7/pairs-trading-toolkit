import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def get_riskfree_rate():
    r_df = yf.Ticker("^IRX").history(period="5d")
    if r_df.empty:
        raise ValueError(
            "No data returned for ^IRX. Check internet connection or ticker."
        )
    y = r_df["Close"].iloc[-1] / 100
    T = 13 / 52
    r = -np.log(1 - T * y) / T
    return float(r)

def optimize_pairs_portfolio(successful_results, optimization_method='max_sharpe', 
                           risk_free_rate=None, max_weight=0.5, min_weight=0.0):

    print("Optimizing allocation across pairs trading strategies...")
    
    if len(successful_results) < 2:
        print("Need at least 2 successful strategies to optimize portfolio")
        return {'success': False, 'message': 'Insufficient strategies'}

    strategy_returns = pd.DataFrame()
    strategy_info = []

    if risk_free_rate is None:
        risk_free_rate = get_riskfree_rate()
    
    for result in successful_results:
        if result.get('success', False):
            pair_name = f"{result['stock1']}_{result['stock2']}"
            signals = result['signals']

            strategy_returns[pair_name] = signals['net_strategy_returns'].fillna(0)
            strategy_info.append({
                'pair_name': pair_name,
                'stock1': result['stock1'],
                'stock2': result['stock2'],
                'performance': result['performance']
            })
    
    if strategy_returns.empty:
        print("No valid strategy returns found")
        return {'success': False}

    mean_returns = strategy_returns.mean() * 252
    cov_matrix = strategy_returns.cov() * 252
    num_strategies = len(strategy_returns.columns)
    
    def portfolio_performance(weights):
        """Calculate portfolio performance metrics"""
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe(weights):

        return -portfolio_performance(weights)[2]
    
    def portfolio_variance(weights):
        return portfolio_performance(weights)[1] ** 2
    
    def negative_return(weights):
        return -portfolio_performance(weights)[0]

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    bounds = tuple((min_weight, max_weight) for _ in range(num_strategies))
    initial_guess = np.array([1/num_strategies] * num_strategies)

    if optimization_method == 'max_sharpe':
        objective_function = negative_sharpe
    elif optimization_method == 'min_variance':
        objective_function = portfolio_variance
    elif optimization_method == 'max_return':
        objective_function = negative_return
    else:
        raise ValueError("optimization_method must be 'max_sharpe', 'min_variance', or 'max_return'")
    
    result = minimize(objective_function, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_weights = result.x
        opt_return, opt_vol, opt_sharpe = portfolio_performance(optimal_weights)

        results = {
            'optimization_method': optimization_method,
            'weights': dict(zip(strategy_returns.columns, optimal_weights)),
            'expected_return': opt_return,
            'volatility': opt_vol,
            'sharpe_ratio': opt_sharpe,
            'strategy_returns': strategy_returns,
            'strategy_info': strategy_info,
            'individual_performance': mean_returns.to_dict(),
            'correlation_matrix': strategy_returns.corr(),
            'success': True
        }
        
        print(f"\n=== Pairs Portfolio Optimization Results ({optimization_method}) ===")
        print(f"Expected Annual Return: {opt_return:.2%}")
        print(f"Annual Volatility: {opt_vol:.2%}")
        print(f"Sharpe Ratio: {opt_sharpe:.3f}")
        print(f"\nOptimal Strategy Allocation:")
        
        for strategy, weight in results['weights'].items():
            if weight > 0.01: 
                strategy_perf = None
                for info in strategy_info:
                    if info['pair_name'] == strategy:
                        strategy_perf = info['performance']
                        break
                
                if strategy_perf:
                    print(f"  {strategy}: {weight:.1%} "
                          f"(Individual Return: {strategy_perf['net_total_return']:.1f}%, "
                          f"Sharpe: {strategy_perf['net_sharpe_ratio']:.2f})")
        
        return results
    else:
        print("Pairs portfolio optimization failed!")
        return {'success': False, 'message': result.message}


def plot_pairs_portfolio_analysis(optimization_result):

    if not optimization_result.get('success', False):
        print("Cannot plot - optimization failed")
        return
    
    strategy_returns = optimization_result['strategy_returns']
    weights = optimization_result['weights']
    strategy_info = optimization_result['strategy_info']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    correlation_matrix = strategy_returns.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Pairs Strategy Correlation Matrix')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)

    significant_weights = {k: v for k, v in weights.items() if v > 0.01}
    other_weight = sum(v for v in weights.values() if v <= 0.01)
    if other_weight > 0:
        significant_weights['Others'] = other_weight
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(significant_weights)))
    wedges, texts, autotexts = ax2.pie(significant_weights.values(), 
                                      labels=significant_weights.keys(), 
                                      autopct='%1.1f%%', startangle=90, 
                                      colors=colors)
    ax2.set_title('Pairs Portfolio Allocation')

    portfolio_weights = np.array([weights[col] for col in strategy_returns.columns])
    portfolio_returns = (strategy_returns * portfolio_weights).sum(axis=1)
    
    for col in strategy_returns.columns:
        strategy_cumulative = (1 + strategy_returns[col]).cumprod()
        ax3.plot(strategy_cumulative.index, strategy_cumulative.values, 
                alpha=0.6, linewidth=1, label=col)
    
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    ax3.plot(portfolio_cumulative.index, portfolio_cumulative.values, 
             color='red', linewidth=3, label='Optimized Pairs Portfolio')
    ax3.set_title('Portfolio vs Individual Pairs Strategies')
    ax3.set_ylabel('Cumulative Return')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    individual_returns = strategy_returns.mean() * 252
    individual_volatilities = strategy_returns.std() * np.sqrt(252)
    
    colors = [weights[strategy] for strategy in strategy_returns.columns]
    scatter = ax4.scatter(individual_volatilities, individual_returns, 
                         c=colors, cmap='viridis', s=100, alpha=0.7)
    
    for i, strategy in enumerate(strategy_returns.columns):
        ax4.annotate(strategy.replace('_', '/'), 
                    (individual_volatilities.iloc[i], individual_returns.iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    portfolio_return = optimization_result['expected_return']
    portfolio_vol = optimization_result['volatility']
    ax4.scatter(portfolio_vol, portfolio_return, color='red', s=200, marker='*', 
               label='Optimized Portfolio', zorder=5, edgecolors='black')
    
    ax4.set_xlabel('Volatility (Annual)')
    ax4.set_ylabel('Expected Return (Annual)')
    ax4.set_title('Risk-Return Profile of Pairs Strategies')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Portfolio Weight')
    
    plt.tight_layout()
    plt.show()


def pairs_portfolio_backtest(optimization_result, rebalance_frequency='monthly', 
                           transaction_cost=0.001, initial_capital=10000):

    if not optimization_result.get('success', False):
        print("Cannot backtest - optimization failed")
        return
    
    strategy_returns = optimization_result['strategy_returns']
    weights = optimization_result['weights']
    
    portfolio_weights = np.array([weights[col] for col in strategy_returns.columns])
    
    if rebalance_frequency == 'daily':
        portfolio_returns = (strategy_returns * portfolio_weights).sum(axis=1)
        rebalancing_costs = pd.Series(transaction_cost, index=strategy_returns.index) * len(portfolio_weights)
        
    elif rebalance_frequency == 'monthly':
        portfolio_returns = []
        rebalancing_costs = []
        current_weights = portfolio_weights.copy()
        
        for i in range(len(strategy_returns)):
            period_return = (strategy_returns.iloc[i] * current_weights).sum()
            portfolio_returns.append(period_return)
            
            if (i + 1) % 21 == 0 or i == 0:  
                rebal_cost = transaction_cost * np.sum(np.abs(current_weights - portfolio_weights))
                current_weights = portfolio_weights.copy()
            else:
                rebal_cost = 0
                if i > 0:
                    current_weights = current_weights * (1 + strategy_returns.iloc[i])
                    current_weights = current_weights / current_weights.sum()
            
            rebalancing_costs.append(rebal_cost)
        
        portfolio_returns = pd.Series(portfolio_returns, index=strategy_returns.index)
        rebalancing_costs = pd.Series(rebalancing_costs, index=strategy_returns.index)

    net_portfolio_returns = portfolio_returns - rebalancing_costs

    gross_cumulative = (1 + portfolio_returns).cumprod()
    net_cumulative = (1 + net_portfolio_returns).cumprod()
    
    gross_total_return = (gross_cumulative.iloc[-1] - 1) * 100
    net_total_return = (net_cumulative.iloc[-1] - 1) * 100
    
    annual_periods = len(portfolio_returns) / 252
    gross_annual_return = ((gross_cumulative.iloc[-1]) ** (1/annual_periods) - 1) * 100
    net_annual_return = ((net_cumulative.iloc[-1]) ** (1/annual_periods) - 1) * 100
    
    gross_volatility = portfolio_returns.std() * np.sqrt(252) * 100
    net_volatility = net_portfolio_returns.std() * np.sqrt(252) * 100
    
    gross_sharpe = (gross_annual_return - 2) / gross_volatility if gross_volatility > 0 else 0
    net_sharpe = (net_annual_return - 2) / net_volatility if net_volatility > 0 else 0
    
    gross_max_drawdown = ((gross_cumulative / gross_cumulative.cummax()) - 1).min() * 100
    net_max_drawdown = ((net_cumulative / net_cumulative.cummax()) - 1).min() * 100
    
    total_costs = rebalancing_costs.sum() * initial_capital
    cost_impact = gross_total_return - net_total_return
    
    print(f"\n=== Pairs Portfolio Backtest Results ===")
    print(f"Rebalancing: {rebalance_frequency.title()}")
    print(f"Transaction Cost: {transaction_cost*100:.2f}% per rebalance")
    print(f"\nPerformance (Gross → Net):")
    print(f"  Total Return: {gross_total_return:.2f}% → {net_total_return:.2f}%")
    print(f"  Annualized Return: {gross_annual_return:.2f}% → {net_annual_return:.2f}%")
    print(f"  Volatility: {gross_volatility:.2f}% → {net_volatility:.2f}%")
    print(f"  Sharpe Ratio: {gross_sharpe:.3f} → {net_sharpe:.3f}")
    print(f"  Max Drawdown: {gross_max_drawdown:.2f}% → {net_max_drawdown:.2f}%")
    print(f"\nCost Analysis:")
    print(f"  Total Rebalancing Costs: ${total_costs:.2f}")
    print(f"  Cost Impact on Returns: -{cost_impact:.2f}%")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(gross_cumulative.index, gross_cumulative.values, 
             label='Gross Returns', linewidth=2, alpha=0.8)
    plt.plot(net_cumulative.index, net_cumulative.values,
             label='Net Returns (After Costs)', linewidth=2)
    
    equal_weight_returns = strategy_returns.mean(axis=1)
    equal_weight_cumulative = (1 + equal_weight_returns).cumprod()
    plt.plot(equal_weight_cumulative.index, equal_weight_cumulative.values,
             label='Equal Weight Portfolio', linewidth=1, linestyle='--', alpha=0.7)
    
    plt.axhline(1, color='black', linestyle='-', alpha=0.5)
    plt.title('Pairs Portfolio Performance')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    gross_drawdown = (gross_cumulative / gross_cumulative.cummax() - 1) * 100
    net_drawdown = (net_cumulative / net_cumulative.cummax() - 1) * 100
    
    plt.fill_between(gross_drawdown.index, gross_drawdown.values, 0, 
                     alpha=0.3, color='red', label='Gross Drawdown')
    plt.fill_between(net_drawdown.index, net_drawdown.values, 0, 
                     alpha=0.5, color='darkred', label='Net Drawdown')
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'gross_returns': portfolio_returns,
        'net_returns': net_portfolio_returns,
        'gross_total_return': gross_total_return,
        'net_total_return': net_total_return,
        'gross_annual_return': gross_annual_return,
        'net_annual_return': net_annual_return,
        'gross_sharpe': gross_sharpe,
        'net_sharpe': net_sharpe,
        'gross_max_drawdown': gross_max_drawdown,
        'net_max_drawdown': net_max_drawdown,
        'total_costs': total_costs,
        'cost_impact': cost_impact
    }


def pairs_efficient_frontier(successful_results, num_points=20):
    if len(successful_results) < 2:
        print("Need at least 2 strategies for efficient frontier")
        return None
    
    strategy_returns = pd.DataFrame()
    for result in successful_results:
        if result.get('success', False):
            pair_name = f"{result['stock1']}_{result['stock2']}"
            strategy_returns[pair_name] = result['signals']['net_strategy_returns'].fillna(0)
    
    mean_returns = strategy_returns.mean() * 252
    cov_matrix = strategy_returns.cov() * 252
    
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, num_points)
    
    efficient_portfolios = []
    
    for target_ret in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_ret}
        ]
        bounds = tuple((0, 1) for _ in range(len(strategy_returns.columns)))
        
        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
            np.array([1/len(strategy_returns.columns)] * len(strategy_returns.columns)),
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        if result.success:
            weights = result.x
            ret = np.sum(mean_returns * weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = ret / vol if vol > 0 else 0
            
            efficient_portfolios.append({
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'weights': dict(zip(strategy_returns.columns, weights))
            })
    
    if efficient_portfolios:
        returns = [p['return'] for p in efficient_portfolios]
        volatilities = [p['volatility'] for p in efficient_portfolios]
        sharpe_ratios = [p['sharpe_ratio'] for p in efficient_portfolios]
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', s=50)
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        individual_vols = strategy_returns.std() * np.sqrt(252)
        individual_rets = mean_returns
        plt.scatter(individual_vols, individual_rets, 
                   marker='x', s=100, color='red', label='Individual Strategies')
        
        max_sharpe_idx = np.argmax(sharpe_ratios)
        plt.scatter(volatilities[max_sharpe_idx], returns[max_sharpe_idx],
                   marker='*', s=200, color='gold', edgecolor='black',
                   label=f'Max Sharpe (SR: {sharpe_ratios[max_sharpe_idx]:.3f})')
        
        plt.xlabel('Volatility (Annual)')
        plt.ylabel('Expected Return (Annual)')
        plt.title('Efficient Frontier - Pairs Trading Strategies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {
            'efficient_portfolios': efficient_portfolios,
            'max_sharpe_portfolio': efficient_portfolios[max_sharpe_idx],
            'strategy_returns': strategy_returns
        }
    
    return None


if __name__ == "__main__":
    stock_tickers = ['AAPL','GOOG','TSLA','MSFT','NVDA','JPM','AMD','META','AMZN',
                     'BRK-B','PLTR','^SPX','BA','KO','SMCI','RTX','^IXIC','RYA.IR',
                     'A5G.IR','BIRG.IR','KRZ.IR','GL9.IR','AV.L','INTC','PINC','GS','MU']
    
    pairs = coint_tester(stock_tickers)
    print(f"Found {len(pairs)} cointegrated pairs")
    all_results=[]
    if len(pairs) > 0:
        for i in range(len(pairs)):
            pair_info = pairs.iloc[i]
            try:
                signals, performance = moving_average_strategy(
                    pairs, 
                    item=i,                   
                    ma_short=5,                
                    ma_long=15,                
                    z_entry=0.5,              
                    z_exit=0.1,               
                    initial_capital=10000,
                    transaction_cost=0.001    
                )
                if signals is not None and performance is not None:
                    result = {
                        'pair_index': i,
                        'stock1': pair_info['s1'],
                        'stock2': pair_info['s2'],
                        'p_value_spread': pair_info['pvs'],
                        'p_value_ratio': pair_info['pvr'],
                        'signals': signals,
                        'performance': performance,
                        'success': True
                    }
                    all_results.append(result)
            except Exception as e:
                print(f"Error analysising pair {pair_info['s1']}/{pair_info['s2']}:{str(e)}")
                all_results.append({
                    'pair_index': i,
                    'stock1': pair_info['s1'],
                    'stock2': pair_info['s2'],
                    'success': False,
                    'Error': str(e)
                })

    successful_results = [r for r in all_results if r.get('success', False)]
    if len(successful_results)>0:
        summary_data=[]
        for result in successful_results:
            perf = result['performance']
            summary_data.append({
                'Pair': f"{result['stock1']}/{result['stock2']}",
                'Net Return (%)': perf['net_total_return'],
                'Net Annual (%)': perf['net_annual_return'],
                'Net Sharpe': perf['net_sharpe_ratio'],
                'Max Drawdown (%)': perf['net_max_drawdown'],
                'Num Trades': perf['num_trades'],
                'Total Costs ($)': perf['total_transaction_costs'],
                'Volatility (%)': perf['net_volatility']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Net Return (%)', ascending=False)

        print("\nAll successful strategies by Net returns(%) :")
        print(summary_df.to_string(index=False, float_format='%.2f'))

        best_return = summary_df.iloc[0]
        best_sharpe = summary_df.iloc[summary_df['Net Sharpe'].idxmax()]
        best_risk_adj = summary_df.iloc[(summary_df['Net Return (%)'] / summary_df['Max Drawdown (%)'].abs()).idxmax()]
        risk_adj_ratio = best_risk_adj['Net Return (%)'] / abs(best_risk_adj['Max Drawdown (%)'])

        print(f"\nBest net Return {best_return['Pair']}({best_return['Net Return (%)']:.2f}%)")
        print(f"Best Sharpe ratio {best_sharpe['Pair']}({best_sharpe['Net Sharpe']:.2f})")
        print(f"Best Risk-Adjusted: {best_risk_adj['Pair']} (Return/MaxDD: {risk_adj_ratio:.2f})")
    
    optimisation_parms(pairs, 3, optimisation_metric = 'net_total_return')
    
    print("\nPAIRS PORTFOLIO OPTIMIZATION\n")
    
    if len(successful_results) > 1:
        print("\n1. Optimizing allocation across pairs strategies...")
        pairs_portfolio = optimize_pairs_portfolio(
            successful_results,
            optimization_method='max_sharpe',
            max_weight=0.4,  
            min_weight=0.0
        )
        
        if pairs_portfolio.get('success', False):
            print("\n2. Detailed portfolio analysis...")
            plot_pairs_portfolio_analysis(pairs_portfolio)
            
            print("\n3. Backtesting pairs portfolio...")
            backtest_results = pairs_portfolio_backtest(
                pairs_portfolio,
                rebalance_frequency='monthly',
                transaction_cost=0.001
            )
            
            print("\n4. Generating efficient frontier...")
            efficient_frontier = pairs_efficient_frontier(successful_results)
    
    else:
        print("Need at least 2 successful pairs strategies for portfolio optimization")
    
    print("\nPairs trading analysis complete!")