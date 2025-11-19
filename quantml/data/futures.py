"""
Futures-specific data handling.

Handles contract rolls, holidays, gaps, and session-based data.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


class FuturesDataHandler:
    """
    Handler for futures-specific data operations.
    
    Features:
    - Contract roll detection and handling
    - Holiday and partial session handling
    - Overnight gap detection
    - Session-based filtering (RTH vs ETH)
    """
    
    def __init__(
        self,
        instrument: str,
        roll_method: str = "volume"  # "volume", "open_interest", "date"
    ):
        """
        Initialize futures data handler.
        
        Args:
            instrument: Instrument symbol (ES, MES, NQ, MNQ)
            roll_method: Method for detecting rolls ("volume", "open_interest", "date")
        """
        self.instrument = instrument
        self.roll_method = roll_method
        self.contract_size = self._get_contract_size(instrument)
    
    def _get_contract_size(self, instrument: str) -> float:
        """Get contract multiplier for instrument."""
        sizes = {
            'ES': 50.0,
            'MES': 5.0,
            'NQ': 20.0,
            'MNQ': 2.0,
            'YM': 5.0,
            'MYM': 0.5
        }
        return sizes.get(instrument.upper(), 50.0)
    
    def detect_contract_rolls(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None,
        dates: Optional[List[datetime]] = None
    ) -> List[int]:
        """
        Detect contract roll dates.
        
        Args:
            prices: Price data
            volumes: Volume data (for volume-based detection)
            dates: Date data (for date-based detection)
        
        Returns:
            List of indices where rolls occur
        """
        roll_indices = []
        
        if self.roll_method == "volume":
            if volumes is None:
                raise ValueError("Volume data required for volume-based roll detection")
            
            # Detect roll when volume drops significantly
            for i in range(1, len(volumes)):
                if volumes[i] < volumes[i-1] * 0.3:  # Volume drops by 70%
                    roll_indices.append(i)
        
        elif self.roll_method == "date":
            if dates is None:
                raise ValueError("Date data required for date-based roll detection")
            
            # Typical roll dates (third Friday of month)
            for i, date in enumerate(dates):
                if date.weekday() == 4 and 15 <= date.day <= 21:  # Third Friday
                    roll_indices.append(i)
        
        return roll_indices
    
    def detect_overnight_gaps(
        self,
        closes: List[float],
        opens: List[float],
        dates: Optional[List[datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect overnight gaps.
        
        Args:
            closes: Closing prices
            opens: Opening prices
            dates: Date data (to identify holiday gaps)
        
        Returns:
            List of gap dictionaries
        """
        gaps = []
        
        for i in range(1, len(opens)):
            if closes[i-1] > 0:
                gap = (opens[i] - closes[i-1]) / closes[i-1]
                
                # Check if holiday gap (more than 1 day between dates)
                is_holiday = False
                if dates and i < len(dates) and i-1 < len(dates):
                    days_diff = (dates[i] - dates[i-1]).days
                    is_holiday = days_diff > 1
                
                gaps.append({
                    'index': i,
                    'gap': gap,
                    'gap_size': abs(gap),
                    'is_holiday': is_holiday,
                    'prev_close': closes[i-1],
                    'current_open': opens[i]
                })
        
        return gaps
    
    def filter_session(
        self,
        data: Dict[str, List[Any]],
        session_type: str = "RTH",
        timestamps: Optional[List[datetime]] = None
    ) -> Dict[str, List[Any]]:
        """
        Filter data by trading session.
        
        Args:
            data: Data dictionary
            session_type: "RTH" (9:30-16:00 ET) or "ETH" (all hours)
            timestamps: Timestamp data
        
        Returns:
            Filtered data dictionary
        """
        if session_type == "ETH":
            return data  # No filtering
        
        if timestamps is None:
            # Assume all data is RTH if no timestamps
            return data
        
        # RTH: 9:30 AM - 4:00 PM ET
        filtered_indices = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            minute = ts.minute
            
            # Convert to ET (simplified, assumes UTC-5)
            # In production, use proper timezone handling
            if 9 <= hour < 16 or (hour == 9 and minute >= 30):
                filtered_indices.append(i)
        
        filtered_data = {}
        for key, values in data.items():
            filtered_data[key] = [values[i] for i in filtered_indices]
        
        return filtered_data
    
    def handle_missing_data(
        self,
        data: Dict[str, List[float]],
        method: str = "forward_fill"
    ) -> Dict[str, List[float]]:
        """
        Handle missing data (NaN, None, zeros).
        
        Args:
            data: Data dictionary
            method: "forward_fill", "backward_fill", "interpolate", "drop"
        
        Returns:
            Data with missing values handled
        """
        cleaned_data = {}
        
        for key, values in data.items():
            cleaned = []
            last_valid = None
            
            for val in values:
                if val is None or (isinstance(val, float) and (val != val or val == 0.0)):
                    # Missing or invalid
                    if method == "forward_fill" and last_valid is not None:
                        cleaned.append(last_valid)
                    elif method == "drop":
                        continue  # Skip
                    else:
                        cleaned.append(0.0)  # Default
                else:
                    cleaned.append(val)
                    last_valid = val
            
            if method == "backward_fill":
                # Fill backwards
                for i in range(len(cleaned)-2, -1, -1):
                    if cleaned[i] == 0.0 and cleaned[i+1] != 0.0:
                        cleaned[i] = cleaned[i+1]
            
            cleaned_data[key] = cleaned
        
        return cleaned_data
    
    def align_contract_data(
        self,
        front_month: Dict[str, List[float]],
        back_month: Dict[str, List[float]],
        roll_indices: List[int]
    ) -> Dict[str, List[float]]:
        """
        Align data across contract rolls.
        
        Args:
            front_month: Front month contract data
            back_month: Back month contract data
            roll_indices: Indices where rolls occur
        
        Returns:
            Aligned data dictionary
        """
        aligned = {}
        
        for key in front_month.keys():
            if key not in back_month:
                continue
            
            aligned_values = []
            front_data = front_month[key]
            back_data = back_month[key]
            
            for i in range(len(front_data)):
                if i in roll_indices:
                    # Use back month data after roll
                    if i < len(back_data):
                        aligned_values.append(back_data[i])
                    else:
                        aligned_values.append(front_data[i])
                else:
                    aligned_values.append(front_data[i])
            
            aligned[key] = aligned_values
        
        return aligned

