"""
travel_time_estimator.py

This module converts predicted traffic flow into estimated travel time.
Used by the routing system to calculate edge costs.

Author: Member 3
"""

def flow_to_travel_time(flow, distance_km, speed_kmh=60, intersection_delay=30):
    """
    Convert traffic flow to travel time.

    Parameters:
    - flow (float): number of cars per 15 minutes
    - distance_km (float): distance between two intersections in km
    - speed_kmh (float): average speed (default = 60 km/h)
    - intersection_delay (int): delay per intersection in seconds (default = 30)

    Returns:
    - travel_time (float): estimated travel time in seconds
    """

    # --- Safety checks ---
    if distance_km < 0:
        raise ValueError("Distance cannot be negative")

    if distance_km == 0:
        return 0

    # Ensure flow is non-negative
    flow = max(flow, 0)

    # --- Step 1: Base travel time (no traffic) ---
    # time = distance / speed → convert hours to seconds
    base_time = (distance_km / speed_kmh) * 3600

    # --- Step 2: Congestion factor ---
    # Slightly more sensitive scaling
    congestion_factor = 1 + (flow / 800)

    # Limit congestion to avoid unrealistic values
    congestion_factor = min(congestion_factor, 5)

    # --- Step 3: Apply congestion ---
    travel_time = base_time * congestion_factor

    # --- Step 4: Add intersection delay ---
    travel_time += intersection_delay

    return travel_time


def estimate_edge_time(flow, distance_km):
    """
    Wrapper function for cleaner integration with other modules.
    """
    return flow_to_travel_time(flow, distance_km)


# ---------------------------
# TESTING
# ---------------------------
if __name__ == "__main__":
    print("Testing travel_time_estimator...\n")

    test_cases = [
        {"flow": 100, "distance": 1.0},
        {"flow": 300, "distance": 2.5},
        {"flow": 800, "distance": 3.0},
        {"flow": 0,   "distance": 1.5},
    ]

    for i, test in enumerate(test_cases):
        flow = test["flow"]
        distance = test["distance"]

        time_sec = flow_to_travel_time(flow, distance)

        print(f"Test {i+1}:")
        print(f"  Flow: {flow} cars")
        print(f"  Distance: {distance} km")
        print(f"  Estimated Time: {time_sec:.2f} seconds\n")