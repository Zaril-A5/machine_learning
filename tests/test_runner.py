"""
test_runner.py

Comprehensive test suite for Member 1's components:
- Data loading and parsing
- Graph building with Haversine distance calculation
- Route finding with top-5 paths
- Travel time estimation

Author: Member 1
Total Test Cases: 18
"""

import sys
import os
import json
import math
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source', 'data_processing'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source', 'integration'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source', 'pathfinding'))

from data_loader import TrafficLocationLoader
from graph_builder import GraphBuilder
from traffic_route_finder import TrafficRouteFinder
from route_finder import TopKRouteFinder


class TestRunner:
    """Runs all tests and generates report."""
    
    def __init__(self):
        self.results = []
        self.test_count = 0
        self.pass_count = 0
        self.fail_count = 0

    def assert_true(self, condition: bool, message: str) -> None:
        """Assert condition is true."""
        if condition:
            self.pass_count += 1
            print(f"  ✓ {message}")
        else:
            self.fail_count += 1
            print(f"  ✗ {message}")

    def assert_equal(self, actual, expected, message: str) -> None:
        """Assert values are equal."""
        if actual == expected:
            self.pass_count += 1
            print(f"  ✓ {message}")
        else:
            self.fail_count += 1
            print(f"  ✗ {message} (got {actual}, expected {expected})")

    def assert_approx_equal(self, actual: float, expected: float, 
                           tolerance: float, message: str) -> None:
        """Assert values are approximately equal."""
        if abs(actual - expected) <= tolerance:
            self.pass_count += 1
            print(f"  ✓ {message}")
        else:
            self.fail_count += 1
            print(f"  ✗ {message} (got {actual}, expected ~{expected} ±{tolerance})")

    def run_test(self, test_name: str, test_func) -> None:
        """Run a single test."""
        self.test_count += 1
        print(f"\n[TEST {self.test_count}] {test_name}")
        try:
            test_func(self)
        except Exception as e:
            self.fail_count += 1
            print(f"  ✗ Test crashed: {e}")

    def report(self) -> None:
        """Print test report."""
        print(f"\n\n{'='*70}")
        print("TEST REPORT")
        print(f"{'='*70}")
        print(f"Total Tests:    {self.test_count}")
        print(f"Passed:         {self.pass_count}")
        print(f"Failed:         {self.fail_count}")
        print(f"Success Rate:   {(self.pass_count/self.test_count*100):.1f}%")
        print(f"{'='*70}\n")


def test_data_loading(runner):
    """Test 1: Data Loading"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    loader = TrafficLocationLoader(csv_path)
    sites = loader.load_locations()
    
    runner.assert_true(len(sites) > 0, "Loaded sites from CSV")
    runner.assert_true(isinstance(sites, dict), "Sites is dictionary")
    
    # Check sample site
    if sites:
        first_site_id = list(sites.keys())[0]
        lat, lon, desc = sites[first_site_id]
        runner.assert_true(-37 < lat < -36, "Latitude in Boroondara range")
        runner.assert_true(144 < lon < 146, "Longitude in Boroondara range")


def test_haversine_distance(runner):
    """Test 2: Haversine Distance Calculation"""
    builder = GraphBuilder(None)
    
    # Test known distance: Melbourne CBD to approximate location
    # Coordinates are approximate
    lat1, lon1 = -37.8136, 144.9631  # Melbourne CBD approx
    lat2, lon2 = -37.8150, 144.9650  # nearby point
    
    distance = builder.haversine_distance(lat1, lon1, lat2, lon2)
    
    runner.assert_true(distance > 0, "Distance is positive")
    runner.assert_true(distance < 1.0, "Distance is less than 1 km (nearby points)")


def test_haversine_symmetry(runner):
    """Test 3: Haversine Distance Symmetry"""
    builder = GraphBuilder(None)
    
    lat1, lon1 = -37.81, 144.96
    lat2, lon2 = -37.82, 144.97
    
    dist_a_b = builder.haversine_distance(lat1, lon1, lat2, lon2)
    dist_b_a = builder.haversine_distance(lat2, lon2, lat1, lon1)
    
    runner.assert_approx_equal(dist_a_b, dist_b_a, 0.001, 
                              "Haversine distance is symmetric")


def test_graph_building(runner):
    """Test 4: Graph Building"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    loader = TrafficLocationLoader(csv_path)
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    
    runner.assert_true(len(builder.nodes) > 0, "Nodes added to graph")
    runner.assert_equal(len(builder.nodes), len(loader.sites), 
                       "Node count equals site count")


def test_graph_edges(runner):
    """Test 5: Graph Edge Creation"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    loader = TrafficLocationLoader(csv_path)
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    builder.add_edges_between_nearby_sites(distance_threshold_km=2.0)
    
    info = builder.get_graph_info()
    runner.assert_true(info['num_edges'] > 0, "Edges created between nearby sites")


def test_manual_edge_addition(runner):
    """Test 6: Manual Edge Addition"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    loader = TrafficLocationLoader(csv_path)
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    
    site_ids = builder.graph.get_site_ids() if hasattr(builder, 'graph') else list(builder.nodes.keys())[:2]
    
    if len(site_ids) >= 2:
        site_a, site_b = site_ids[0], site_ids[1]
        builder.add_edge(site_a, site_b)
        
        distance = builder.get_edge_distance(site_a, site_b)
        runner.assert_true(distance is not None, f"Edge added between {site_a} and {site_b}")
        runner.assert_true(distance > 0, "Edge distance is positive")


def test_config_loading(runner):
    """Test 7: Configuration Loading"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    runner.assert_true('routing' in config, "Config has routing section")
    runner.assert_equal(config['routing']['speed_limit_kmh'], 60, 
                       "Speed limit is 60 km/h")
    runner.assert_equal(config['routing']['intersection_delay_s'], 30, 
                       "Intersection delay is 30s")


def test_flow_to_travel_time(runner):
    """Test 8: Flow to Travel Time Conversion"""
    config = {
        'speed_limit_kmh': 60,
        'intersection_delay_s': 30
    }
    
    loader = TrafficLocationLoader(os.path.join(os.path.dirname(__file__), '..', 
                                                'data', 'raw', 
                                                'Traffic_Count_Locations_with_LONG_LAT.csv'))
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    
    route_finder = TrafficRouteFinder(config, builder)
    
    # Test low flow
    travel_time_low = route_finder.flow_to_travel_time(10.0, 1000)  # 10 cars, 1 km
    
    # Test high flow
    travel_time_high = route_finder.flow_to_travel_time(100.0, 1000)  # 100 cars, 1 km
    
    runner.assert_true(travel_time_low > 0, "Low flow travel time is positive")
    runner.assert_true(travel_time_high > travel_time_low, 
                      "High flow causes longer travel time")


def test_travel_time_distance_effect(runner):
    """Test 9: Distance Effect on Travel Time"""
    config = {
        'speed_limit_kmh': 60,
        'intersection_delay_s': 30
    }
    
    loader = TrafficLocationLoader(os.path.join(os.path.dirname(__file__), '..', 
                                                'data', 'raw', 
                                                'Traffic_Count_Locations_with_LONG_LAT.csv'))
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    
    route_finder = TrafficRouteFinder(config, builder)
    
    # Same flow, different distances
    time_1km = route_finder.flow_to_travel_time(40.0, 1000)    # 1 km
    time_2km = route_finder.flow_to_travel_time(40.0, 2000)    # 2 km
    
    runner.assert_true(time_2km > time_1km, 
                      "Longer distance results in longer travel time")


def test_route_format_time(runner):
    """Test 10: Route Time Formatting"""
    config = {
        'speed_limit_kmh': 60,
        'intersection_delay_s': 30
    }
    
    loader = TrafficLocationLoader(os.path.join(os.path.dirname(__file__), '..', 
                                                'data', 'raw', 
                                                'Traffic_Count_Locations_with_LONG_LAT.csv'))
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    
    route_finder = TrafficRouteFinder(config, builder)
    
    # Test time formatting
    formatted_30s = route_finder.format_travel_time(30)
    formatted_90s = route_finder.format_travel_time(90)
    formatted_3600s = route_finder.format_travel_time(3600)
    
    runner.assert_true("30s" in formatted_30s or "30" in formatted_30s, 
                      "30 seconds formatted correctly")
    runner.assert_true("1m" in formatted_90s, "90 seconds formatted as 1 minute")
    runner.assert_true("1h" in formatted_3600s, "3600 seconds formatted as 1 hour")


def test_get_site_ids(runner):
    """Test 11: Get All Site IDs"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    loader = TrafficLocationLoader(csv_path)
    loader.load_locations()
    
    site_ids = loader.get_site_ids()
    
    runner.assert_true(len(site_ids) > 0, "Retrieved site IDs")
    runner.assert_true(all(isinstance(sid, int) for sid in site_ids), 
                      "All site IDs are integers")
    runner.assert_equal(site_ids, sorted(site_ids), "Site IDs are sorted")


def test_error_handling_missing_site(runner):
    """Test 12: Error Handling - Missing Site"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    loader = TrafficLocationLoader(csv_path)
    loader.load_locations()
    
    try:
        loader.get_site(99999)  # Non-existent site
        runner.assert_true(False, "Should raise error for missing site")
    except ValueError:
        runner.assert_true(True, "Correctly raises error for missing site")


def test_error_handling_no_edge(runner):
    """Test 13: Error Handling - No Edge Between Sites"""
    config = {
        'speed_limit_kmh': 60,
        'intersection_delay_s': 30
    }
    
    loader = TrafficLocationLoader(os.path.join(os.path.dirname(__file__), '..', 
                                                'data', 'raw', 
                                                'Traffic_Count_Locations_with_LONG_LAT.csv'))
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    # Deliberately don't add edges
    
    route_finder = TrafficRouteFinder(config, builder)
    
    site_ids = list(builder.nodes.keys())[:2]
    
    if len(site_ids) >= 2:
        try:
            route_finder.get_edge_travel_time(site_ids[0], site_ids[1])
            runner.assert_true(False, "Should raise error for non-existent edge")
        except ValueError:
            runner.assert_true(True, "Correctly raises error for non-existent edge")


def test_zero_flow_handling(runner):
    """Test 14: Zero Flow Handling"""
    config = {
        'speed_limit_kmh': 60,
        'intersection_delay_s': 30
    }
    
    loader = TrafficLocationLoader(os.path.join(os.path.dirname(__file__), '..', 
                                                'data', 'raw', 
                                                'Traffic_Count_Locations_with_LONG_LAT.csv'))
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    
    route_finder = TrafficRouteFinder(config, builder)
    
    # Zero flow should still produce valid travel time
    travel_time = route_finder.flow_to_travel_time(0.0, 1000)
    
    runner.assert_true(travel_time > 0, "Zero flow produces positive travel time")
    runner.assert_true(travel_time == 30 + (1/60)*3600, 
                      "Zero flow time equals base time + intersection delay")


def test_graph_get_successors(runner):
    """Test 15: Graph Get Successors"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    loader = TrafficLocationLoader(csv_path)
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    
    site_ids = list(builder.nodes.keys())[:3]
    if len(site_ids) >= 2:
        builder.add_edge(site_ids[0], site_ids[1])
        
        successors = builder.get_successors(site_ids[0])
        runner.assert_true(len(successors) > 0, f"Site {site_ids[0]} has successors")
        runner.assert_equal(successors[0][0], site_ids[1], 
                           "Successor is the added neighbor")


def test_bidirectional_edges(runner):
    """Test 16: Bidirectional Edge Creation"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    loader = TrafficLocationLoader(csv_path)
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    
    site_ids = list(builder.nodes.keys())[:2]
    if len(site_ids) >= 2:
        builder.add_bidirectional_edge(site_ids[0], site_ids[1])
        
        forward_edge = builder.get_edge_distance(site_ids[0], site_ids[1])
        backward_edge = builder.get_edge_distance(site_ids[1], site_ids[0])
        
        runner.assert_true(forward_edge is not None, "Forward edge exists")
        runner.assert_true(backward_edge is not None, "Backward edge exists")
        runner.assert_approx_equal(forward_edge, backward_edge, 0.001, 
                                  "Forward and backward distances are equal")


def test_graph_info(runner):
    """Test 17: Graph Information"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    loader = TrafficLocationLoader(csv_path)
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    builder.add_edges_between_nearby_sites(distance_threshold_km=2.0)
    
    info = builder.get_graph_info()
    
    runner.assert_true('num_nodes' in info, "Graph info has num_nodes")
    runner.assert_true('num_edges' in info, "Graph info has num_edges")
    runner.assert_equal(info['num_nodes'], len(builder.nodes), 
                       "Node count matches")


def test_multiple_edges_same_sites(runner):
    """Test 18: Avoiding Duplicate Edges"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    loader = TrafficLocationLoader(csv_path)
    loader.load_locations()
    
    builder = GraphBuilder(loader)
    builder.add_nodes_from_locations()
    
    site_ids = list(builder.nodes.keys())[:2]
    if len(site_ids) >= 2:
        builder.add_edge(site_ids[0], site_ids[1])
        edge_count_before = len(builder.edges[site_ids[0]])
        
        builder.add_edge(site_ids[0], site_ids[1])  # Add again
        edge_count_after = len(builder.edges[site_ids[0]])
        
        runner.assert_equal(edge_count_before, edge_count_after, 
                           "Duplicate edge updates existing edge, doesn't create new one")


# ============================================================================
# Main Test Execution
# ============================================================================

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("MEMBER 1 - SYSTEM ARCHITECT & INTEGRATION - TEST SUITE")
    print(f"{'='*70}\n")
    
    runner = TestRunner()
    
    # Run all tests
    runner.run_test("Data Loading from CSV", test_data_loading)
    runner.run_test("Haversine Distance Calculation", test_haversine_distance)
    runner.run_test("Haversine Distance Symmetry", test_haversine_symmetry)
    runner.run_test("Graph Building", test_graph_building)
    runner.run_test("Graph Edge Creation", test_graph_edges)
    runner.run_test("Manual Edge Addition", test_manual_edge_addition)
    runner.run_test("Configuration File Loading", test_config_loading)
    runner.run_test("Flow to Travel Time Conversion", test_flow_to_travel_time)
    runner.run_test("Distance Effect on Travel Time", test_travel_time_distance_effect)
    runner.run_test("Route Time Formatting", test_route_format_time)
    runner.run_test("Get All Site IDs", test_get_site_ids)
    runner.run_test("Error Handling - Missing Site", test_error_handling_missing_site)
    runner.run_test("Error Handling - No Edge Between Sites", test_error_handling_no_edge)
    runner.run_test("Zero Flow Handling", test_zero_flow_handling)
    runner.run_test("Graph Get Successors", test_graph_get_successors)
    runner.run_test("Bidirectional Edge Creation", test_bidirectional_edges)
    runner.run_test("Graph Information", test_graph_info)
    runner.run_test("Multiple Edges Same Sites", test_multiple_edges_same_sites)
    
    # Print final report
    runner.report()
