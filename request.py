import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={"depreciation": 13460, "mileage": 66000,
                             "age": 1336, "car_model": 311, "brand": 38, "category": 3})

print(r.json())
