"""
Enhanced Monte Carlo Simulation Engine
Implements production-grade scenario generation with:
- Volatility clustering (GARCH-like)
- Jump diffusion processes
- Mean reversion with regime switching
- Correlated multi-asset paths
- Historical bootstrap resampling
- Advanced risk metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulations"""
    num_paths: int = 100
    steps: int = 50
    confidence_levels: List[float] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.50, 0.68, 0.95]


class EnhancedScenarioGenerator:
    """
    Production-grade Monte Carlo simulation engine for crypto markets
    """
    
    def __init__(self, analyzer=None):
        """
        Initialize the enhanced scenario generator
        
        Args:
            analyzer: TradingAnalyzer instance (optional, for compatibility)
        """
        self.analyzer = analyzer
        
        # GARCH parameters for volatility clustering
        self.garch_omega = 0.000001  # Base volatility
        self.garch_alpha = 0.1       # Recent shock weight
        self.garch_beta = 0.85       # Persistence
        
        # Jump diffusion parameters (calibrated for crypto)
        self.jump_intensity = 0.05   # 5% chance of jump per step
        self.jump_mean = -0.02       # Average jump is -2% (bearish skew)
        self.jump_std = 0.08         # Jumps vary ±8%
        
        # Mean reversion parameters
        self.mean_reversion_kappa = 0.3  # 30% reversion per step
    
    def _clamp_price(self, price: float, initial_price: float, max_multiplier: float = 5.0, min_multiplier: float = 0.2) -> float:
        """
        Clamp price to reasonable bounds to prevent extreme outliers
        
        Args:
            price: Current price
            initial_price: Starting price
            max_multiplier: Maximum allowed price (as multiple of initial)
            min_multiplier: Minimum allowed price (as multiple of initial)
            
        Returns:
            Clamped price within bounds
        """
        max_price = initial_price * max_multiplier
        min_price = initial_price * min_multiplier
        return max(min_price, min(max_price, price))
    
    # ==================== 1. VOLATILITY CLUSTERING ====================
    
    def generate_realistic_paths(
        self, 
        last_price: float, 
        df: pd.DataFrame, 
        num_paths: int = 100, 
        steps: int = 50
    ) -> List[List[Dict]]:
        """
        Generate paths with realistic volatility clustering (GARCH-like behavior)
        
        Args:
            last_price: Current market price
            df: Historical price DataFrame
            num_paths: Number of simulation paths
            steps: Number of time steps per path
            
        Returns:
            List of paths, each containing time-value-volatility dictionaries
        """
        # Calculate historical volatility parameters
        returns = df['Close'].pct_change().dropna()
        
        paths = []
        
        for p in range(num_paths):
            path = [{"time": 0, "value": last_price, "volatility": returns.tail(20).std()}]
            price = last_price
            
            # Initialize volatility with recent realized vol
            current_vol = returns.tail(20).std()
            
            for s in range(1, steps + 1):
                # Update volatility (GARCH process)
                shock = np.random.normal(0, 1)
                
                # New volatility depends on:
                # 1. Base level (omega)
                # 2. Recent shock (alpha * shock²)
                # 3. Previous volatility (beta * current_vol²)
                current_vol = np.sqrt(
                    self.garch_omega + 
                    self.garch_alpha * (shock ** 2) + 
                    self.garch_beta * (current_vol ** 2)
                )
                
                # Generate price change with time-varying volatility
                daily_return = current_vol * shock
                price = price * (1 + daily_return)
                
                # Clamp to reasonable bounds
                price = self._clamp_price(price, last_price)
                
                path.append({
                    "time": s, 
                    "value": price,
                    "volatility": current_vol
                })
            
            paths.append(path)
        
        return paths
    
    # ==================== 2. JUMP DIFFUSION ====================
    
    def generate_jump_diffusion_paths(
        self, 
        last_price: float, 
        atr: float, 
        num_paths: int = 100, 
        steps: int = 50,
        jump_intensity: Optional[float] = None,
        jump_mean: Optional[float] = None,
        jump_std: Optional[float] = None
    ) -> List[List[Dict]]:
        """
        Add discrete jump events to simulate flash crashes and pumps
        
        Args:
            last_price: Current market price
            atr: Average True Range
            num_paths: Number of simulation paths
            steps: Number of time steps
            jump_intensity: Probability of jump per step (default: 0.05)
            jump_mean: Average jump size (default: -0.02)
            jump_std: Jump volatility (default: 0.08)
            
        Returns:
            List of paths with jump events tracked
        """
        vol = atr / last_price if atr > 0 else 0.02
        
        # Use instance defaults if not provided
        jump_intensity = jump_intensity or self.jump_intensity
        jump_mean = jump_mean or self.jump_mean
        jump_std = jump_std or self.jump_std
        
        paths = []
        
        for p in range(num_paths):
            path = [{"time": 0, "value": last_price, "jump": False}]
            price = last_price
            
            for s in range(1, steps + 1):
                # Normal diffusion component
                normal_return = np.random.normal(0, vol)
                
                # Jump component
                if np.random.random() < jump_intensity:
                    jump = np.random.normal(jump_mean, jump_std)
                    total_return = normal_return + jump
                    jump_occurred = True
                else:
                    total_return = normal_return
                    jump_occurred = False
                
                price = price * (1 + total_return)
                
                path.append({
                    "time": s,
                    "value": price,
                    "jump": jump_occurred
                })
            
            paths.append(path)
        
        return paths
    
    # ==================== 3. MEAN REVERSION ====================
    
    def generate_mean_reverting_paths(
        self, 
        last_price: float, 
        support: float, 
        resistance: float, 
        num_paths: int = 100, 
        steps: int = 50,
        kappa: Optional[float] = None
    ) -> List[List[Dict]]:
        """
        Generate paths with Ornstein-Uhlenbeck mean reversion
        
        Args:
            last_price: Current market price
            support: Support level
            resistance: Resistance level
            num_paths: Number of simulation paths
            steps: Number of time steps
            kappa: Mean reversion speed (default: 0.3)
            
        Returns:
            List of mean-reverting paths
        """
        # Calculate equilibrium level
        equilibrium = (support + resistance) / 2
        
        # Mean reversion speed
        kappa = kappa or self.mean_reversion_kappa
        
        # Volatility around equilibrium
        sigma = (resistance - support) / (2 * equilibrium) * 0.5
        
        paths = []
        
        for p in range(num_paths):
            path = [{"time": 0, "value": last_price, "distance_from_equilibrium": abs(last_price - equilibrium) / equilibrium}]
            price = last_price
            
            for s in range(1, steps + 1):
                # Ornstein-Uhlenbeck process
                # dX = kappa * (equilibrium - X) * dt + sigma * dW
                
                drift = kappa * (equilibrium - price)
                diffusion = sigma * price * np.random.normal(0, 1)
                
                price = price + drift + diffusion
                
                # Add boundaries (support/resistance act as walls)
                if price < support * 0.98:
                    price = support * (1 + np.random.uniform(0, 0.02))
                elif price > resistance * 1.02:
                    price = resistance * (1 - np.random.uniform(0, 0.02))
                
                path.append({
                    "time": s,
                    "value": price,
                    "distance_from_equilibrium": abs(price - equilibrium) / equilibrium
                })
            
            paths.append(path)
        
        return paths
    
    # ==================== 4. CORRELATED MULTI-ASSET ====================
    
    def generate_correlated_paths(
        self, 
        assets: Dict[str, Dict], 
        correlation_matrix: np.ndarray,
        num_paths: int = 100, 
        steps: int = 50
    ) -> Dict[str, List[List[Dict]]]:
        """
        Generate correlated paths for multiple assets (e.g., BTC + ETH + TOTAL3)
        
        Args:
            assets: Dictionary of asset configs, e.g.:
                {
                    'BTC': {'price': 45000, 'vol': 0.03},
                    'ETH': {'price': 2800, 'vol': 0.04},
                    'ALTS': {'price': 1.0, 'vol': 0.06}
                }
            correlation_matrix: NxN correlation matrix for N assets
            num_paths: Number of simulation paths
            steps: Number of time steps
            
        Returns:
            Dictionary mapping asset names to their path lists
        """
        n_assets = len(assets)
        asset_names = list(assets.keys())
        
        # Cholesky decomposition for correlated random variables
        L = np.linalg.cholesky(correlation_matrix)
        
        all_paths = {name: [] for name in asset_names}
        
        for p in range(num_paths):
            # Initialize prices
            prices = {name: [assets[name]['price']] for name in asset_names}
            
            for s in range(steps):
                # Generate correlated random shocks
                independent_shocks = np.random.normal(0, 1, n_assets)
                correlated_shocks = L @ independent_shocks
                
                # Update each asset
                for i, name in enumerate(asset_names):
                    vol = assets[name]['vol']
                    current_price = prices[name][-1]
                    
                    # Apply correlated shock
                    new_price = current_price * (1 + vol * correlated_shocks[i])
                    prices[name].append(new_price)
            
            # Store paths
            for name in asset_names:
                all_paths[name].append([
                    {"time": t, "value": prices[name][t]} 
                    for t in range(steps + 1)
                ])
        
        return all_paths
    
    # ==================== 5. BOOTSTRAP RESAMPLING ====================
    
    def generate_bootstrap_paths(
        self, 
        df: pd.DataFrame, 
        last_price: float,
        num_paths: int = 100, 
        steps: int = 50,
        block_size: int = 5
    ) -> List[List[Dict]]:
        """
        Resample historical return blocks to preserve autocorrelation
        
        Args:
            df: Historical price DataFrame
            last_price: Current market price
            num_paths: Number of simulation paths
            steps: Number of time steps
            block_size: Size of return blocks to sample
            
        Returns:
            List of bootstrapped paths
        """
        returns = df['Close'].pct_change().dropna()
        
        paths = []
        
        for p in range(num_paths):
            path = [{"time": 0, "value": last_price}]
            price = last_price
            
            # Randomly sample blocks of returns
            sampled_returns = []
            while len(sampled_returns) < steps:
                # Random starting point
                start_idx = np.random.randint(0, len(returns) - block_size)
                block = returns.iloc[start_idx:start_idx + block_size].values
                sampled_returns.extend(block)
            
            # Apply sampled returns
            for s in range(1, steps + 1):
                ret = sampled_returns[s - 1]
                price = price * (1 + ret)
                
                path.append({"time": s, "value": price})
            
            paths.append(path)
        
        return paths
    
    # ==================== 6. ADVANCED RISK METRICS ====================
    
    def calculate_path_statistics(
        self, 
        paths: List[List[Dict]], 
        initial_price: float
    ) -> Dict:
        """
        Extract actionable risk metrics from Monte Carlo paths
        
        Args:
            paths: List of simulation paths
            initial_price: Starting price
            
        Returns:
            Dictionary of comprehensive risk metrics
        """
        final_prices = [path[-1]['value'] for path in paths]
        returns = [(fp / initial_price - 1) * 100 for fp in final_prices]
        
        # Calculate drawdowns for each path
        max_drawdowns = []
        for path in paths:
            prices = [p['value'] for p in path]
            peak = prices[0]
            max_dd = 0
            
            for price in prices:
                if price > peak:
                    peak = price
                dd = (peak - price) / peak * 100
                max_dd = max(max_dd, dd)
            
            max_drawdowns.append(max_dd)
        
        # Value at Risk (VaR) and Conditional VaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean([r for r in returns if r <= var_95])
        
        # Probability of profit
        prob_profit = sum(1 for r in returns if r > 0) / len(returns) * 100
        
        # Expected profit/loss
        avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        avg_loss = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
        
        # Sharpe-like ratio (assuming risk-free rate = 0)
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Calculate win/loss ratio (avoid infinity)
        if avg_loss != 0:
            win_loss_ratio = float(abs(avg_win / avg_loss))
        else:
            # If no losses, use a large but finite number instead of infinity
            win_loss_ratio = 999.99 if avg_win > 0 else 0.0
        
        return {
            "expected_return": float(np.mean(returns)),
            "volatility": float(np.std(returns)),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95) if not np.isnan(cvar_95) else 0.0,
            "max_drawdown_avg": float(np.mean(max_drawdowns)),
            "max_drawdown_worst": float(np.max(max_drawdowns)),
            "probability_profit": float(prob_profit),
            "average_win": float(avg_win),
            "average_loss": float(avg_loss),
            "win_loss_ratio": win_loss_ratio,
            "sharpe_ratio": float(sharpe) if not np.isnan(sharpe) and not np.isinf(sharpe) else 0.0,
            "price_targets": {
                "p10": float(np.percentile(final_prices, 10)),
                "p25": float(np.percentile(final_prices, 25)),
                "p50": float(np.percentile(final_prices, 50)),
                "p75": float(np.percentile(final_prices, 75)),
                "p90": float(np.percentile(final_prices, 90))
            }
        }
    
    # ==================== 7. CONFIDENCE CONE ====================
    
    def generate_confidence_cone(
        self, 
        paths: List[List[Dict]], 
        confidence_levels: Optional[List[float]] = None
    ) -> List[Dict]:
        """
        Generate data for confidence interval visualization
        
        Args:
            paths: List of simulation paths
            confidence_levels: List of confidence levels (e.g., [0.50, 0.68, 0.95])
            
        Returns:
            List of confidence cone data points
        """
        if confidence_levels is None:
            confidence_levels = [0.50, 0.68, 0.95]
        
        steps = len(paths[0])
        cone_data = []
        
        for step in range(steps):
            prices_at_step = [path[step]['value'] for path in paths]
            
            intervals = {}
            for conf in confidence_levels:
                lower_pct = (1 - conf) / 2 * 100
                upper_pct = (1 - (1 - conf) / 2) * 100
                
                intervals[f'p{int(conf*100)}'] = {
                    'lower': float(np.percentile(prices_at_step, lower_pct)),
                    'upper': float(np.percentile(prices_at_step, upper_pct))
                }
            
            cone_data.append({
                'time': step,
                'median': float(np.median(prices_at_step)),
                **intervals
            })
        
        return cone_data
    
    # ==================== 8. REGIME-AWARE SIMULATIONS ====================
    
    def generate_regime_aware_paths(
        self, 
        df: pd.DataFrame, 
        last_price: float,
        current_regime: str, 
        atr: float,
        support: float,
        resistance: float,
        num_paths: int = 100,
        steps: int = 50
    ) -> List[List[Dict]]:
        """
        Use different simulation models based on detected market regime
        
        Args:
            df: Historical price DataFrame
            last_price: Current market price
            current_regime: Detected market regime
            atr: Average True Range
            support: Support level
            resistance: Resistance level
            num_paths: Number of paths
            steps: Number of time steps
            
        Returns:
            Regime-appropriate simulation paths
        """
        regime_lower = current_regime.lower()
        
        if "trending bull" in regime_lower or "bull" in regime_lower:
            # Geometric Brownian Motion with positive drift
            return self._generate_gbm_paths(
                last_price, 
                drift=0.002,  # 0.2% upward drift per step
                vol=atr/last_price if atr > 0 else 0.02,
                num_paths=num_paths,
                steps=steps
            )
        
        elif "trending bear" in regime_lower or "bear" in regime_lower:
            # GBM with negative drift + jump risk
            return self.generate_jump_diffusion_paths(
                last_price,
                atr,
                num_paths=num_paths,
                steps=steps,
                jump_intensity=0.08  # Higher crash risk
            )
        
        elif "ranging" in regime_lower or "consolidation" in regime_lower:
            # Mean-reverting Ornstein-Uhlenbeck
            return self.generate_mean_reverting_paths(
                last_price, support, resistance, num_paths, steps
            )
        
        elif "volatile" in regime_lower:
            # High vol + volatility clustering
            return self.generate_realistic_paths(
                last_price, df, num_paths, steps
            )
        
        else:
            # Default: bootstrap historical returns
            return self.generate_bootstrap_paths(df, last_price, num_paths, steps)
    
    def _generate_gbm_paths(
        self,
        last_price: float,
        drift: float,
        vol: float,
        num_paths: int,
        steps: int
    ) -> List[List[Dict]]:
        """
        Generate Geometric Brownian Motion paths
        
        Args:
            last_price: Starting price
            drift: Drift parameter
            vol: Volatility parameter
            num_paths: Number of paths
            steps: Number of time steps
            
        Returns:
            List of GBM paths
        """
        paths = []
        
        for p in range(num_paths):
            path = [{"time": 0, "value": last_price}]
            price = last_price
            
            for s in range(1, steps + 1):
                # GBM: dS = μS dt + σS dW
                price = price * (1 + drift + vol * np.random.normal(0, 1))
                price = self._clamp_price(price, last_price)
                path.append({"time": s, "value": price})
            
            paths.append(path)
        
        return paths
    
    # ==================== 9. COMPREHENSIVE SCENARIO GENERATION ====================
    
    def generate_all_scenarios(
        self, 
        df: pd.DataFrame, 
        latest_row: Dict, 
        market_regime: str, 
        num_paths: int = 100,
        steps: int = 50
    ) -> Dict:
        """
        Master function that generates comprehensive scenario analysis
        
        Args:
            df: Historical price DataFrame
            latest_row: Latest market data row
            market_regime: Current market regime
            num_paths: Number of simulation paths
            steps: Number of time steps
            
        Returns:
            Comprehensive scenario analysis dictionary
        """
        last_price = latest_row['Close']
        atr = latest_row.get('ATR', last_price * 0.02)
        support = latest_row.get('S1', last_price * 0.985)
        resistance = latest_row.get('R1', last_price * 1.015)
        
        scenarios = {}
        
        # 1. Regime-appropriate base simulation
        scenarios['base'] = self.generate_regime_aware_paths(
            df, last_price, market_regime, atr, support, resistance, num_paths, steps
        )
        
        # 2. Stress test scenarios
        scenarios['bull_breakout'] = self._generate_breakout_scenario(
            last_price, atr, resistance, num_paths=50, steps=steps
        )
        
        scenarios['bear_breakdown'] = self._generate_breakdown_scenario(
            last_price, atr, support, num_paths=50, steps=steps
        )
        
        scenarios['black_swan'] = self._generate_black_swan_scenario(
            last_price, atr, num_paths=50, steps=steps
        )
        
        # 3. Calculate statistics for each scenario
        stats = {}
        for scenario_name, paths in scenarios.items():
            if isinstance(paths, list) and len(paths) > 0 and isinstance(paths[0], list):
                stats[scenario_name] = self.calculate_path_statistics(
                    paths, last_price
                )
        
        # 4. Generate confidence cones
        cones = {}
        for scenario_name, paths in scenarios.items():
            if isinstance(paths, list) and len(paths) > 0 and isinstance(paths[0], list):
                cones[scenario_name] = self.generate_confidence_cone(paths)
        
        return {
            'scenarios': scenarios,
            'statistics': stats,
            'confidence_cones': cones,
            'current_price': last_price,
            'regime': market_regime,
            'metadata': {
                'num_paths': num_paths,
                'steps': steps,
                'atr': atr,
                'support': support,
                'resistance': resistance
            }
        }
    
    def _generate_breakout_scenario(
        self,
        last_price: float,
        atr: float,
        resistance: float,
        num_paths: int,
        steps: int
    ) -> List[List[Dict]]:
        """
        Generate realistic bullish breakout scenario with multi-phase behavior
        
        Phases:
        1. Consolidation near resistance (compression)
        2. Volume-driven breakout (expansion)
        3. Retest of broken resistance (now support)
        4. Explosive continuation (momentum)
        5. Gradual deceleration (profit-taking)
        """
        vol = atr / last_price if atr > 0 else 0.02
        paths = []
        
        for p in range(num_paths):
            path = [{"time": 0, "value": last_price}]
            price = last_price
            
            # Randomize breakout timing slightly (13-17 steps)
            breakout_step = int(np.random.uniform(13, 17))
            retest_step = breakout_step + int(np.random.uniform(5, 10))
            
            for s in range(1, steps + 1):
                # Phase 1: Consolidation near resistance (volatility compression)
                if s < breakout_step:
                    # Price coils near resistance with decreasing volatility
                    target = resistance * np.random.uniform(0.992, 0.998)
                    compression_factor = 1 - (s / breakout_step) * 0.5  # Vol decreases
                    noise = np.random.normal(0, vol * 0.2 * compression_factor) * price
                    price += (target - price) * 0.15 + noise
                
                # Phase 2: Breakout (volume spike + volatility expansion)
                elif s == breakout_step:
                    # Explosive breakout with high volume
                    breakout_strength = np.random.uniform(1.015, 1.035)  # 1.5-3.5% breakout
                    price = resistance * breakout_strength
                
                # Phase 3: Immediate continuation (FOMO phase)
                elif s < breakout_step + 3:
                    # Strong continuation with expanded volatility
                    momentum = np.random.uniform(0.008, 0.015)  # 0.8-1.5% per step
                    noise = np.random.normal(0, vol * 1.2) * price
                    price *= (1 + momentum + noise / price)
                
                # Phase 4: Retest of breakout level (healthy pullback)
                elif s >= retest_step and s < retest_step + 5:
                    # Price pulls back to test resistance-turned-support
                    retest_target = resistance * np.random.uniform(1.002, 1.008)
                    mean_reversion = (retest_target - price) * 0.25
                    noise = np.random.normal(0, vol * 0.6) * price
                    price += mean_reversion + noise
                
                # Phase 5: Explosive continuation (momentum phase)
                elif s >= retest_step + 5 and s < retest_step + 15:
                    # Strong upward momentum after successful retest
                    momentum = np.random.uniform(0.006, 0.012)
                    noise = np.random.normal(0, vol * 1.0) * price
                    price *= (1 + momentum + noise / price)
                
                # Phase 6: Deceleration (profit-taking)
                else:
                    # Momentum gradually decreases
                    time_since_peak = s - (retest_step + 15)
                    momentum_decay = max(0.002, 0.008 - time_since_peak * 0.0003)
                    noise = np.random.normal(0, vol * 0.8) * price
                    price *= (1 + momentum_decay + noise / price)
                
                # Ensure price doesn't go negative
                price = max(price, last_price * 0.5)
                
                path.append({"time": s, "value": price})
            
            paths.append(path)
        
        return paths
    
    def _generate_breakdown_scenario(
        self,
        last_price: float,
        atr: float,
        support: float,
        num_paths: int,
        steps: int
    ) -> List[List[Dict]]:
        """
        Generate realistic bearish breakdown scenario with multi-phase behavior
        
        Phases:
        1. Weak consolidation near support (fear building)
        2. Panic selling breakdown (capitulation)
        3. Dead cat bounce (failed rally attempt)
        4. Cascading liquidations (acceleration)
        5. Capitulation bottom (max fear)
        6. Stabilization and base formation
        """
        vol = atr / last_price if atr > 0 else 0.02
        paths = []
        
        for p in range(num_paths):
            path = [{"time": 0, "value": last_price}]
            price = last_price
            
            # Randomize breakdown timing (12-16 steps)
            breakdown_step = int(np.random.uniform(12, 16))
            bounce_step = breakdown_step + int(np.random.uniform(3, 7))
            capitulation_step = bounce_step + int(np.random.uniform(5, 10))
            
            for s in range(1, steps + 1):
                # Phase 1: Weak consolidation near support (fear building)
                if s < breakdown_step:
                    # Price weakly holds near support with increasing volatility
                    target = support * np.random.uniform(1.002, 1.008)
                    fear_factor = 1 + (s / breakdown_step) * 0.8  # Vol increases
                    noise = np.random.normal(-0.001, vol * 0.3 * fear_factor) * price  # Slight bearish bias
                    price += (target - price) * 0.12 + noise
                
                # Phase 2: Breakdown (panic selling)
                elif s == breakdown_step:
                    # Sharp breakdown with panic
                    breakdown_severity = np.random.uniform(0.965, 0.985)  # 1.5-3.5% drop
                    price = support * breakdown_severity
                
                # Phase 3: Immediate cascade (stop-loss triggers)
                elif s < breakdown_step + 3:
                    # Accelerating decline as stops get hit
                    panic_momentum = np.random.uniform(-0.012, -0.006)  # -1.2% to -0.6% per step
                    noise = np.random.normal(0, vol * 1.5) * price  # High volatility
                    price *= (1 + panic_momentum + noise / price)
                
                # Phase 4: Dead cat bounce (failed rally)
                elif s >= bounce_step and s < bounce_step + 4:
                    # Weak bounce as shorts take profit
                    bounce_target = support * np.random.uniform(0.992, 1.002)
                    relief_rally = (bounce_target - price) * 0.3
                    noise = np.random.normal(0, vol * 0.8) * price
                    price += relief_rally + noise
                
                # Phase 5: Rejection and continuation (bounce fails)
                elif s >= bounce_step + 4 and s < capitulation_step:
                    # Bounce fails, selling resumes
                    rejection_momentum = np.random.uniform(-0.008, -0.004)
                    noise = np.random.normal(0, vol * 1.2) * price
                    price *= (1 + rejection_momentum + noise / price)
                
                # Phase 6: Capitulation (max fear, highest volume)
                elif s >= capitulation_step and s < capitulation_step + 3:
                    # Final washout with extreme volatility
                    capitulation_drop = np.random.uniform(-0.015, -0.008)
                    noise = np.random.normal(0, vol * 2.0) * price  # Extreme volatility
                    price *= (1 + capitulation_drop + noise / price)
                
                # Phase 7: Stabilization (base formation)
                else:
                    # Price stabilizes at lower level with decreasing volatility
                    time_since_bottom = s - (capitulation_step + 3)
                    stabilization_factor = max(0.3, 1.0 - time_since_bottom * 0.05)
                    
                    # Slight mean reversion to find equilibrium
                    equilibrium = price * 1.01  # Slight upward bias after capitulation
                    mean_reversion = (equilibrium - price) * 0.1
                    noise = np.random.normal(0, vol * 0.5 * stabilization_factor) * price
                    price += mean_reversion + noise
                
                # Ensure price doesn't go negative or too low
                price = max(price, last_price * 0.3)  # Max 70% drawdown
                
                path.append({"time": s, "value": price})
            
            paths.append(path)
        
        return paths
    
    def _generate_black_swan_scenario(
        self,
        last_price: float,
        atr: float,
        num_paths: int,
        steps: int
    ) -> List[List[Dict]]:
        """
        Generate extreme tail-risk black swan event scenario
        
        Phases:
        1. Normal market (calm before storm)
        2. Initial shock (flash crash trigger)
        3. Panic cascade (liquidation spiral)
        4. Volatility explosion (chaos)
        5. Failed recovery attempts (dead cat bounces)
        6. Capitulation (max pain)
        7. Slow stabilization (shell-shocked market)
        
        Characteristics:
        - Multiple crash waves
        - Extreme volatility spikes
        - Liquidity vacuum gaps
        - Failed rallies
        - Long recovery tail
        """
        vol = atr / last_price if atr > 0 else 0.02
        paths = []
        
        for p in range(num_paths):
            path = [{"time": 0, "value": last_price}]
            price = last_price
            
            # Randomize event timing
            shock_step = int(np.random.uniform(8, 15))
            second_wave_step = shock_step + int(np.random.uniform(8, 15))
            capitulation_step = second_wave_step + int(np.random.uniform(5, 10))
            
            for s in range(1, steps + 1):
                # Phase 1: Deceptive calm (normal market)
                if s < shock_step:
                    # Normal price action with slight weakness
                    drift = np.random.uniform(-0.002, 0.001)  # Slight bearish bias
                    noise = np.random.normal(0, vol * 0.6) * price
                    price *= (1 + drift + noise / price)
                
                # Phase 2: Initial shock (flash crash)
                elif s == shock_step:
                    # Sudden catastrophic drop (5-15%)
                    crash_magnitude = np.random.uniform(0.85, 0.95)
                    price *= crash_magnitude
                
                # Phase 3: Immediate panic cascade (3-5 steps)
                elif s < shock_step + 5:
                    # Cascading liquidations with extreme volatility
                    panic_drop = np.random.uniform(-0.08, -0.03)  # -8% to -3% per step
                    
                    # Simulate liquidity gaps (occasional large drops)
                    if np.random.random() < 0.3:  # 30% chance of gap
                        panic_drop *= np.random.uniform(1.5, 2.5)
                    
                    noise = np.random.normal(0, vol * 3.0) * price  # Extreme volatility
                    price *= (1 + panic_drop + noise / price)
                
                # Phase 4: First dead cat bounce (false hope)
                elif s >= shock_step + 5 and s < shock_step + 10:
                    # Weak bounce as some buyers step in
                    bounce_strength = np.random.uniform(0.003, 0.008)
                    noise = np.random.normal(0, vol * 2.0) * price
                    price *= (1 + bounce_strength + noise / price)
                
                # Phase 5: Second wave crash (hope destroyed)
                elif s >= second_wave_step and s < second_wave_step + 4:
                    # Bounce fails, selling resumes with vengeance
                    second_crash = np.random.uniform(-0.06, -0.02)
                    
                    # Occasional capitulation spikes
                    if np.random.random() < 0.25:
                        second_crash *= np.random.uniform(1.3, 2.0)
                    
                    noise = np.random.normal(0, vol * 2.5) * price
                    price *= (1 + second_crash + noise / price)
                
                # Phase 6: Capitulation (absolute bottom)
                elif s >= capitulation_step and s < capitulation_step + 3:
                    # Final washout with maximum fear
                    final_drop = np.random.uniform(-0.04, -0.01)
                    noise = np.random.normal(0, vol * 3.5) * price  # Maximum volatility
                    price *= (1 + final_drop + noise / price)
                
                # Phase 7: Weak stabilization attempts
                elif s >= capitulation_step + 3 and s < capitulation_step + 10:
                    # Price tries to find a bottom
                    time_since_bottom = s - (capitulation_step + 3)
                    
                    # Gradual reduction in volatility
                    vol_decay = max(0.5, 3.0 - time_since_bottom * 0.3)
                    
                    # Slight upward bias as worst is over
                    recovery_drift = np.random.uniform(0.001, 0.005)
                    noise = np.random.normal(0, vol * vol_decay) * price
                    price *= (1 + recovery_drift + noise / price)
                
                # Phase 8: Slow recovery (shell-shocked market)
                else:
                    # Very slow recovery with high volatility
                    time_in_recovery = s - (capitulation_step + 10)
                    
                    # Gradual healing
                    recovery_rate = min(0.008, 0.002 + time_in_recovery * 0.0003)
                    remaining_vol = max(0.8, 2.0 - time_in_recovery * 0.08)
                    
                    noise = np.random.normal(0, vol * remaining_vol) * price
                    price *= (1 + recovery_rate + noise / price)
                
                # Ensure price doesn't go negative (circuit breakers)
                # Max drawdown: 80% (realistic for crypto black swans)
                price = max(price, last_price * 0.20)
                
                path.append({"time": s, "value": price})
            
            paths.append(path)
        
        return paths
    
    # ==================== LEGACY COMPATIBILITY ====================
    
    def generate_monte_carlo_paths(
        self, 
        last_price: float, 
        atr: float, 
        market_regime: str, 
        num_paths: int = 50, 
        steps: int = 30
    ) -> Dict:
        """
        Legacy compatibility method - generates basic Monte Carlo paths
        Maintained for backward compatibility with existing code
        """
        # Handle cases where ATR might be zero or None
        if not atr or atr <= 0:
            atr = last_price * 0.01 
            
        volatility = atr / last_price
        paths = []
        
        # Adjust drift and vol based on regime
        drift = 0
        vol_mult = 1.0
        
        regime_lower = market_regime.lower()
        if "bull" in regime_lower:
            drift = 0.001
            vol_mult = 1.1
        elif "bear" in regime_lower:
            drift = -0.001
            vol_mult = 1.1
        elif "volatile" in regime_lower:
            vol_mult = 1.8
        elif "range" in regime_lower or "consolidation" in regime_lower:
            vol_mult = 0.6
            drift = 0
        
        for p in range(num_paths):
            current_path = [{"time": 0, "value": last_price}]
            price = last_price
            for s in range(1, steps + 1):
                if "range" in regime_lower or "consolidation" in regime_lower:
                    reversion = (last_price - price) * 0.1
                    change_pct = reversion / last_price + np.random.normal(0, volatility * vol_mult)
                else:
                    change_pct = drift + np.random.normal(0, volatility * vol_mult)
                
                price = price * (1 + change_pct)
                current_path.append({"time": s, "value": price})
            paths.append(current_path)
            
        # Calculate percentiles for "Heat Fan"
        steps_data = [[] for _ in range(steps + 1)]
        for path in paths:
            for s, point in enumerate(path):
                steps_data[s].append(point["value"])
        
        fan_data = []
        for s in range(steps + 1):
            prices = steps_data[s]
            fan_data.append({
                "time": s,
                "p25": float(np.percentile(prices, 25)),
                "p50": float(np.percentile(prices, 50)),
                "p75": float(np.percentile(prices, 75)),
                "min": float(np.min(prices)),
                "max": float(np.max(prices))
            })
            
        return {
            "paths": paths[:10],
            "fan": fan_data,
            "meta": {
                "steps": steps,
                "num_paths": num_paths,
                "regime": market_regime,
                "vol_multiplier": vol_mult,
                "volatility_used": volatility
            }
        }
    
    def generate_regime_injection(
        self, 
        injection_type: str, 
        last_price: float, 
        atr: float, 
        key_levels: dict
    ) -> Tuple[List[Dict], Dict]:
        """
        Legacy compatibility method - generates specific curated scenarios
        Maintained for backward compatibility with existing code
        """
        steps = 40
        path = [{"time": 0, "value": last_price}]
        
        support = key_levels.get("support")
        resistance = key_levels.get("resistance")
        
        # Sane Defaults if levels are 0 or missing
        if not support or support <= 0 or support >= last_price:
            support = last_price * 0.985
        if not resistance or resistance <= 0 or resistance <= last_price:
            resistance = last_price * 1.015
            
        vol = max(atr / last_price, 0.01)
        
        current_price = last_price
        
        if injection_type == "bullish_breakout":
            for s in range(1, steps + 1):
                if s < 15:
                    target = resistance * 0.995
                    current_price += (target - current_price) * 0.2 + np.random.normal(0, vol * 0.3) * current_price
                elif s == 15:
                    current_price = resistance * 1.01
                else:
                    current_price *= (1 + 0.005 + np.random.normal(0, vol * 0.8))
                path.append({"time": s, "value": current_price})
                
        elif injection_type == "black_swan":
            for s in range(1, steps + 1):
                if s == 5:
                    current_price *= 0.90 
                elif 5 < s < 10:
                    current_price *= (1 + np.random.normal(-0.01, vol * 2))
                else:
                    current_price *= (1 + 0.002 + np.random.normal(0, vol * 0.5))
                path.append({"time": s, "value": current_price})
        
        elif injection_type == "institutional_flush":
            for s in range(1, steps + 1):
                if s < 10:
                    current_price += (resistance - current_price) * 0.3
                elif s == 10:
                    current_price = resistance * 1.015
                elif 10 < s < 25:
                    current_price += (support - current_price) * 0.2
                else:
                    current_price += (support - current_price) * 0.1 + np.random.normal(0, vol * 0.2) * current_price
                path.append({"time": s, "value": current_price})
        
        else:  # Default random walk
            for s in range(1, steps + 1):
                current_price *= (1 + np.random.normal(0, vol))
                path.append({"time": s, "value": current_price})
                
        meta = {
            "steps": steps,
            "volatility_used": vol,
            "support_used": support,
            "resistance_used": resistance
        }
        
        return path, meta
