import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source', 'ml_models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source', 'data_processing'))

from traffic_flow_loader import TrafficFlowLoader
from predictor import TrafficPredictor
import joblib

def test_extract_features():
    """Test _extract_features with specific case"""

    # Load data
    loader = TrafficFlowLoader('data/raw/Scats Data October 2006.xls')
    loader.load_flow_data()
    
    # Create predictor
    predictor = TrafficPredictor(timesteps=12)
    predictor.historical_data = loader.flow_data
    predictor.scaler = joblib.load('data/processed/scaler.pkl')
    
    # Test parameters
    site_id = 970
    date = "2006-10-10"
    time = "14:00"
    location = "WARRIGAL_RD N of HIGH STREET_RD"
    
    print(f"\nTesting:")
    print(f"  Site: {site_id}")
    print(f"  Date: {date}")
    print(f"  Time: {time}")
    print(f"  Location: {location}")
    
    # Get features
    try:
        features = predictor._extract_features(site_id, date, time, location)
        print(f"\nFeatures shape: {features.shape}")
        print(f"Normalized features: {features[0]}")
        
        # Denormalize to see actual flow values
        denormalized = predictor.scaler.inverse_transform(features.reshape(1, -1))
        print(f"\nDenormalized flow values:")
        for i, flow in enumerate(denormalized[0]):
            print(f"  T-{12-i}: {flow:.1f}")
        
        print(f"\nThese are the 12 flow values before {time} on {date}")
        
        # Expected results
        expected_flows = [296, 305, 278, 287, 301, 275, 303, 311, 265, 294, 258, 277]
        
        # Show comparison
        all_match = True
        for i, actual_flow in enumerate(denormalized[0]):
            expected_flow = expected_flows[i] if i < len(expected_flows) else "N/A"
            match = abs(actual_flow - expected_flow) < 1e-6 if expected_flow != "N/A" else True
            print(f"  T-{12-i}: Expected {expected_flow}, Got {actual_flow:.1f}, Match: {match}")
            if not match:
                all_match = False
        
        if all_match:
            print(f"\nAll flow values match expected results")
        else:
            print(f"\nFlow values don't match expected results")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == 'source':
        os.chdir('..')
    
    test_extract_features()