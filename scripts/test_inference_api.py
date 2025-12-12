#!/usr/bin/env python3
"""
Test script for inference API.
Usage: python scripts/test_inference_api.py
"""
import requests
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\nTesting /predict endpoint...")
    payload = {
        "plot_id": 1,
        "latitude": -0.499127,
        "longitude": 37.612253,
        "planting_date": "2021-10-07",
        "season": "Short Rains",
        "altitude": 1252.08,
        "model_name": "ensemble"
    }
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\nTesting /predict/batch endpoint...")
    payload = {
        "plots": [
            {
                "plot_id": 1,
                "latitude": -0.499127,
                "longitude": 37.612253,
                "planting_date": "2021-10-07",
                "season": "Short Rains"
            },
            {
                "plot_id": 2,
                "latitude": -0.501234,
                "longitude": 37.614567,
                "planting_date": "2021-10-10",
                "season": "Short Rains"
            }
        ],
        "model_name": "ensemble"
    }
    response = requests.post(
        f"{API_BASE_URL}/predict/batch",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Total plots: {result['total_plots']}")
        print(f"Successful: {result['successful']}")
        print(f"Failed: {result['failed']}")
        print(f"Sample prediction: {json.dumps(result['predictions'][0] if result['predictions'] else {}, indent=2)}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200


def test_list_models():
    """Test models list endpoint."""
    print("\nTesting /models endpoint...")
    response = requests.get(f"{API_BASE_URL}/models")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200


if __name__ == "__main__":
    print("=" * 60)
    print("Inference API Test Suite")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("Health Check", test_health()))
        results.append(("List Models", test_list_models()))
        results.append(("Single Prediction", test_single_prediction()))
        results.append(("Batch Prediction", test_batch_prediction()))
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API. Is it running?")
        print("   Start with: docker-compose up -d inference-api")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        exit(1)
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)

