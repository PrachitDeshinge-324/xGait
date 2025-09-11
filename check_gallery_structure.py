import pickle
import os

# Check gallery structure
gallery_path = "visualization_analysis/faiss_gallery.pkl"

if os.path.exists(gallery_path):
    with open(gallery_path, 'rb') as f:
        gallery_data = pickle.load(f)
    
    print(f"Type of gallery_data: {type(gallery_data)}")
    print(f"Length: {len(gallery_data)}")
    
    if isinstance(gallery_data, dict):
        print(f"Dictionary keys: {list(gallery_data.keys())}")
        
        # Show first few items
        for i, (key, value) in enumerate(gallery_data.items()):
            if i < 3:  # Show first 3 items
                print(f"Key: {key}, Value type: {type(value)}")
                if hasattr(value, '__dict__'):
                    print(f"  Attributes: {vars(value)}")
                elif hasattr(value, 'keys'):
                    print(f"  Dict keys: {list(value.keys())}")
                else:
                    print(f"  Value: {value}")
    elif isinstance(gallery_data, list) and len(gallery_data) > 0:
        print(f"First item type: {type(gallery_data[0])}")
        print(f"First item: {gallery_data[0]}")
else:
    print("Gallery file not found")
