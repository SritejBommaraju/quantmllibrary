"""
Futures-specific backtesting engine.

Handles contract rolls, margin requirements, overnight gaps, and session-based trading.
"""

from typing import List, Optional, Dict, Any, Tuple
from quantml.training.backtest import BacktestEngine
from quantml.training.metrics import sharpe_ratio, max_drawdown


class FuturesBacktestEngine(BacktestEngine):
    """
    Backtesting engine for futures contracts.
    
    Extends BacktestEngine with futures-specific features:
    - Contract roll handling
    - Margin requirements
    - Overnight gap simulation
    - Session-based trading (RTH vs ETH)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        margin_requirement: float = 0.05,  # 5% margin
        contract_size: float = 50.0,  # ES contract multiplier
        roll_dates: Optional[List[int]] = None,
        session_type: str = "RTH"  # RTH or ETH
    ):
        """
        Initialize futures backtesting engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade
            slippage: Slippage per trade
            margin_requirement: Margin requirement as fraction (e.g., 0.05 = 5%)
            contract_size: Contract multiplier (50 for ES, 20 for NQ)
            roll_dates: List of indices where contract rolls occur
            session_type: "RTH" (regular trading hours) or "ETH" (extended)
        """
        super().__init__(initial_capital, commission, slippage)
        self.margin_requirement = margin_requirement
        self.contract_size = contract_size
        self.roll_dates = roll_dates or []
        self.session_type = session_type
    
    def _calculate_margin(self, position: float, price: float) -> float:
        """
        Calculate margin requirement for position.
        
        Args:
            position: Position size (number of contracts)
            price: Current price
        
        Returns:
            Required margin
        """
        position_value = abs(position) * price * self.contract_size
        return position_value * self.margin_requirement
    
    def _apply_overnight_gap(
        self,
        position: float,
        prev_close: float,
        current_open: float
    ) -> Tuple[float, float]:
        """
        Apply overnight gap to position.
        
        Args:
            position: Current position
            prev_close: Previous day's close
            current_open: Current day's open
        
        Returns:
            (updated_capital, gap_pnl)
        """
        if position == 0 or prev_close == 0:
            return 0.0, 0.0
        
        gap = (current_open - prev_close) / prev_close
        position_value = abs(position) * prev_close * self.contract_size
        
        if position > 0:  # Long
            gap_pnl = position_value * gap
        else:  # Short
            gap_pnl = -position_value * gap
        
        return gap_pnl, gap_pnl
    
    def run_futures(
        self,
        signals: List[float],
        prices: List[float],
        opens: Optional[List[float]] = None,
        closes: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Run futures backtest with contract rolls and overnight gaps.
        
        Args:
            signals: Trading signals
            prices: Price data (can be close prices)
            opens: Opening prices (for gap calculation)
            closes: Closing prices (for gap calculation)
            volumes: Volume data
        
        Returns:
            Backtest results with futures-specific metrics
        """
        if len(signals) != len(prices):
            raise ValueError("signals and prices must have same length")
        
        # Use closes if provided, otherwise use prices
        if closes is None:
            closes = prices
        if opens is None:
            opens = prices
        
        n = len(signals)
        capital = self.initial_capital
        position = 0.0  # Position in contracts
        equity_curve = [capital]
        trades = []
        returns = []
        overnight_gaps = []
        margin_used = []
        
        for i in range(n):
            price = prices[i]
            signal = signals[i]
            
            # Check for contract roll
            if i in self.roll_dates:
                # Close position before roll
                if position != 0:
                    trade_value = abs(position) * price * self.contract_size
                    commission_cost = trade_value * self.commission
                    capital -= commission_cost
                    
                    trades.append({
                        'index': i,
                        'price': price,
                        'execution_price': price,
                        'size': -position,  # Close position
                        'cost': commission_cost,
                        'type': 'roll'
                    })
                    position = 0.0
            
            # Apply overnight gap (if not first bar)
            if i > 0:
                prev_close = closes[i-1] if i-1 < len(closes) else prices[i-1]
                current_open = opens[i] if i < len(opens) else price
                
                if position != 0:
                    gap_pnl, _ = self._apply_overnight_gap(position, prev_close, current_open)
                    capital += gap_pnl
                    overnight_gaps.append({
                        'index': i,
                        'gap': (current_open - prev_close) / prev_close if prev_close > 0 else 0.0,
                        'pnl': gap_pnl
                    })
            
            # Determine target position
            target_position_value = self.position_sizing(signal, capital, price)
            target_position = target_position_value / (price * self.contract_size) if price > 0 else 0.0
            
            # Round to integer contracts
            target_position = round(target_position)
            
            # Calculate trade
            trade_size = target_position - position
            
            if abs(trade_size) > 0.5:  # Only trade if at least 1 contract
                # Check margin requirement
                new_position_value = abs(target_position) * price * self.contract_size
                required_margin = new_position_value * self.margin_requirement
                
                if required_margin > capital:
                    # Insufficient margin, reduce position
                    max_position = int(capital / (price * self.contract_size * self.margin_requirement))
                    target_position = max_position if target_position > 0 else -max_position
                    trade_size = target_position - position
                
                if abs(trade_size) > 0.5:
                    # Apply slippage
                    execution_price = price * (1 + self.slippage * (1 if trade_size > 0 else -1))
                    
                    # Calculate costs
                    trade_value = abs(trade_size) * execution_price * self.contract_size
                    commission_cost = trade_value * self.commission
                    slippage_cost = abs(trade_size) * price * self.contract_size * self.slippage
                    total_cost = commission_cost + slippage_cost
                    
                    # Update capital
                    capital -= trade_size * execution_price * self.contract_size + total_cost
                    
                    # Update position
                    position = target_position
                    
                    trades.append({
                        'index': i,
                        'price': price,
                        'execution_price': execution_price,
                        'size': trade_size,
                        'cost': total_cost,
                        'type': 'trade'
                    })
            
            # Update equity (mark-to-market)
            current_value = capital + position * price * self.contract_size
            equity_curve.append(current_value)
            
            # Track margin usage
            margin = self._calculate_margin(position, price)
            margin_used.append(margin)
            
            # Calculate return
            if i > 0:
                prev_value = equity_curve[-2]
                ret = (current_value - prev_value) / prev_value if prev_value > 0 else 0.0
                returns.append(ret)
        
        # Calculate metrics
        final_value = equity_curve[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        sharpe = sharpe_ratio(returns) if returns else 0.0
        max_dd = max_drawdown(returns) if returns else 0.0
        
        # Trade statistics
        n_trades = len([t for t in trades if t.get('type') == 'trade'])
        avg_margin_usage = sum(margin_used) / len(margin_used) if margin_used else 0.0
        max_margin_usage = max(margin_used) if margin_used else 0.0
        
        # Overnight gap statistics
        gap_pnls = [g['pnl'] for g in overnight_gaps]
        total_gap_pnl = sum(gap_pnls)
        avg_gap = sum(g['gap'] for g in overnight_gaps) / len(overnight_gaps) if overnight_gaps else 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'equity_curve': equity_curve,
            'returns': returns,
            'trades': trades,
            'n_trades': n_trades,
            'overnight_gaps': overnight_gaps,
            'total_gap_pnl': total_gap_pnl,
            'avg_gap': avg_gap,
            'margin_used': margin_used,
            'avg_margin_usage': avg_margin_usage,
            'max_margin_usage': max_margin_usage,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'contract_size': self.contract_size,
            'margin_requirement': self.margin_requirement
        }

