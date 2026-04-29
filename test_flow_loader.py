from source.data_processing.traffic_flow_loader import TrafficFlowLoader

def test_traffic_flow_loader():

    loader = TrafficFlowLoader("data/raw/Scats Data October 2006.xls")
    
    try:
        flow_data = loader.load_flow_data()
        assert len(flow_data) > 0, "No flow data loaded"
        print(f"Successfully loaded {len(flow_data)} traffic flow records")
        
        # Test displaying sample
        print("\n--- Sample Data ---")
        loader.display_sample(2)
        
        # Test getting all sites
        sites = loader.get_all_sites()
        assert len(sites) > 0, "No sites found"
        assert 970 in sites, "Expected site 970 not found"
        print(f"Found {len(sites)} unique sites")
        print(f"First 5 sites: {sites[:5]}")
        
        # Test getting site dates
        dates = loader.get_site_dates(970)
        assert len(dates) > 0, "No dates found for site 970"
        print(f"Site 970 has {len(dates)} dates available")
        
        # Test getting specific flow values
        print("\n--- Testing Flow Values ---")
        
        # Test with specific location
        flow1 = loader.get_flow(970, "2006-10-01 00:15:00", "0:00", "WARRIGAL_RD N of HIGH STREET_RD")
        assert flow1 > 0, "Flow value should be positive"
        print(f"Flow at 0:00 (WARRIGAL_RD N of HIGH STREET_RD): {flow1}")
        
        # Test with different location
        flow2 = loader.get_flow(970, "2006-10-01 00:15:00", "0:00", "HIGH STREET_RD E of WARRIGAL_RD")
        print(f"Flow at 0:00 (HIGH STREET_RD E of WARRIGAL_RD): {flow2}")
        
        # Test different time intervals
        flow3 = loader.get_flow(970, "2006-10-01 00:15:00", "14:30", "WARRIGAL_RD N of HIGH STREET_RD")
        print(f"Flow at 14:30: {flow3}")
        
        flow4 = loader.get_flow(970, "2006-10-01 00:15:00", "23:45", "WARRIGAL_RD N of HIGH STREET_RD")
        print(f"Flow at 23:45: {flow4}")
        
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    test_traffic_flow_loader()
