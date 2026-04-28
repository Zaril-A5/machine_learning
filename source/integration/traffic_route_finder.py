"""
traffic_route_finder.py

Integration module that connects graph, ML model predictions, and routing.
Provides the main interface: get_edge_travel_time(site_A, site_B, timestamp)

Author: Member 1
"""

import json
from typing import Optional, List, Tuple
from datetime import datetime
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_models'))

from data_loader import TrafficLocationLoader
from graph_builder import GraphBuilder


class TrafficRouteFinder:
    """
    Integration layer that combines:
    - Graph with distance information
    - ML model predictions for traffic flow
    - Travel time conversion function
    """

    def __init__(self, config: dict, graph_builder: GraphBuilder):
        """
        Initialize route finder.
        
        Parameters:
        - config (dict): configuration with speed_limit, intersection_delay, etc.
        - graph_builder (GraphBuilder): built graph with nodes and edges
        """
        self.config = config
        self.graph = graph_builder
        self.ml_predictor = None  # Will be loaded when available from Member 2
        self.scaler = None  # Will be loaded from Member 2

    def set_ml_predictor(self, predictor) -> None:
        """
        Set the ML predictor for traffic flow prediction.
        Called after Member 2 provides trained model.
        
        Parameters:
        - predictor: ML predictor with predict_flow(site_id, timestamp) method
        """
        self.ml_predictor = predictor

    def flow_to_travel_time(self, flow_15min: float, distance_m: float) -> float:
        """
        Convert traffic flow to travel time.
        
        Based on "Traffic Flow to Travel Time Conversion v1.0.pdf" from lecturer.
        
        Parameters:
        - flow_15min (float): traffic flow (cars per 15 minutes)
        - distance_m (float): distance in meters
        
        Returns:
        - travel time in seconds
        
        FORMULA (simplified from the PDF):
        Travel time depends on flow and distance:
        - Low flow: faster travel (less congestion)
        - High flow: slower travel (more congestion)
        - Distance also affects time
        
        Using simplified model:
        travel_time = (distance / speed) + congestion_factor
        """
        # Convert distance from meters to kilometers
        distance_km = distance_m / 1000.0
        
        # Base speed from config (km/h)
        speed_limit_kmh = self.config.get('speed_limit_kmh', 60)
        
        # Base travel time without congestion (in seconds)
        base_travel_time = (distance_km / speed_limit_kmh) * 3600  # convert hours to seconds
        
        # Congestion factor based on flow
        # Higher flow = more congestion = slower travel
        # Simplified model: flow affects speed reduction
        
        if flow_15min > 0:
            # Normalize flow to a congestion multiplier
            # Assuming reasonable flow is around 30-50 cars per 15 min
            congestion_factor = 1.0 + (flow_15min / 100.0)  # increases with flow
        else:
            congestion_factor = 1.0
        
        # Apply congestion factor
        congested_travel_time = base_travel_time * congestion_factor
        
        # Add intersection delay
        intersection_delay = self.config.get('intersection_delay_s', 30)
        total_travel_time = congested_travel_time + intersection_delay
        
        return total_travel_time

    def get_edge_travel_time(self, site_a: int, site_b: int, 
                             timestamp: Optional[str] = None) -> float:
        """
        Main function: Get estimated travel time from site A to site B at given time.
        
        This function integrates:
        1. Graph distance (from GraphBuilder)
        2. ML predicted traffic flow (from Member 2's predictor)
        3. Flow-to-time conversion
        
        Parameters:
        - site_a (int): starting SCATS site
        - site_b (int): destination SCATS site
        - timestamp (str): time in format "HH:MM" or None for current time
        
        Returns:
        - estimated travel time in seconds
        
        Raises:
        - ValueError if sites don't exist or no edge between them
        """
        # Default to current time if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M")
        
        # Get distance from graph
        distance_km = self.graph.get_edge_distance(site_a, site_b)
        
        if distance_km is None:
            raise ValueError(f"No direct edge between site {site_a} and site {site_b}")
        
        distance_m = distance_km * 1000  # Convert to meters
        
        # Step 1: Get ML prediction for traffic flow at destination site
        # (flow at destination affects how quickly you can enter that intersection)
        if self.ml_predictor is not None:
            try:
                flow_15min = self.ml_predictor.predict_flow(site_b, timestamp)
            except Exception as e:
                print(f"[WARNING] Could not get ML prediction for site {site_b}: {e}")
                print("[FALLBACK] Using default flow value")
                flow_15min = 40.0  # Default fallback value
        else:
            print(f"[WARNING] ML predictor not available")
            flow_15min = 40.0  # Default fallback
        
        # Step 2: Convert flow to travel time
        travel_time_s = self.flow_to_travel_time(flow_15min, distance_m)
        
        return travel_time_s

    def get_travel_times_for_route(self, route: List[int], 
                                    timestamp: Optional[str] = None) -> Tuple[float, List[float]]:
        """
        Calculate total travel time and segment times for a route.
        
        Parameters:
        - route (List[int]): list of SCATS sites in order
        - timestamp (str): departure time
        
        Returns:
        - (total_time_seconds, segment_times_list)
        """
        if len(route) < 2:
            raise ValueError("Route must have at least 2 sites")
        
        segment_times = []
        total_time = 0
        
        for i in range(len(route) - 1):
            site_a = route[i]
            site_b = route[i + 1]
            
            segment_time = self.get_edge_travel_time(site_a, site_b, timestamp)
            segment_times.append(segment_time)
            total_time += segment_time
        
        return total_time, segment_times

    def format_travel_time(self, seconds: float) -> str:
        """
        Convert seconds to readable format "Xm Ys" or "Xh Ym".
        
        Parameters:
        - seconds (float): time in seconds
        
        Returns:
        - formatted string
        """
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class RouteResult:
    """Represents a found route with its travel time."""
    
    def __init__(self, path: List[int], travel_time: float, route_finder: TrafficRouteFinder):
        """
        Initialize route result.
        
        Parameters:
        - path (List[int]): sequence of SCATS sites
        - travel_time (float): total travel time in seconds
        - route_finder (TrafficRouteFinder): finder that calculated this
        """
        self.path = path
        self.travel_time = travel_time
        self.route_finder = route_finder

    def format_route(self) -> str:
        """Return formatted route string."""
        site_strings = [str(site) for site in self.path]
        path_str = " -> ".join(site_strings)
        time_str = self.route_finder.format_travel_time(self.travel_time)
        return f"Route: {path_str} | Time: {time_str}"

    def __repr__(self) -> str:
        return self.format_route()
