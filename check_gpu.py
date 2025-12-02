"""
Quick script to verify GPU support for XGBoost, LightGBM, and CatBoost
"""

import sys

print("=" * 60)
print("GPU Support Check for Boosting Libraries")
print("=" * 60)

# Check XGBoost
print("\n1. XGBoost:")
try:
    import xgboost as xgb

    print(f"   Version: {xgb.__version__}")
    print(f"   Build Info: {xgb.build_info()}")
    # Try to create a GPU booster
    try:
        import numpy as np

        dtrain = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
        params = {"device": "cuda", "tree_method": "gpu_hist"}
        bst = xgb.train(params, dtrain, num_boost_round=1)
        print("   ✅ GPU support: WORKING")
    except Exception as e:
        print(f"   ❌ GPU support: FAILED - {e}")
except ImportError as e:
    print(f"   ❌ Not installed: {e}")

# Check LightGBM
print("\n2. LightGBM:")
try:
    import lightgbm as lgb

    print(f"   Version: {lgb.__version__}")
    # Try to create a GPU dataset
    try:
        import numpy as np

        train_data = lgb.Dataset(np.random.rand(10, 5), label=np.random.rand(10))
        params = {"device": "gpu", "verbose": -1}
        bst = lgb.train(params, train_data, num_boost_round=1)
        print("   ✅ GPU support: WORKING")
    except Exception as e:
        print(f"   ❌ GPU support: FAILED - {e}")
except ImportError as e:
    print(f"   ❌ Not installed: {e}")

# Check CatBoost
print("\n3. CatBoost:")
try:
    import catboost as cb

    print(f"   Version: {cb.__version__}")
    # Try to create a GPU model
    try:
        import numpy as np

        train_data = cb.Pool(np.random.rand(10, 5), label=np.random.rand(10))
        model = cb.CatBoostRegressor(iterations=1, task_type="GPU", verbose=0)
        model.fit(train_data)
        print("   ✅ GPU support: WORKING")
    except Exception as e:
        print(f"   ❌ GPU support: FAILED - {e}")
except ImportError as e:
    print(f"   ❌ Not installed: {e}")

# Check CUDA availability
print("\n4. CUDA Check:")
try:
    import torch

    print(f"   PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("   PyTorch not installed (optional)")

print("\n" + "=" * 60)
print("Check complete!")
print("=" * 60)
