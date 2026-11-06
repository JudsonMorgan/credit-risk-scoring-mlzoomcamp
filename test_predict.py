
import requests


url = "http://127.0.0.1:9696/predict"

# Sample customer data
customer = {
    "limit_bal": 20000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 24,

    "pay_0": 2,
    "pay_2": 2,
    "pay_3": -1,
    "pay_4": -1,
    "pay_5": -2,
    "pay_6": -2,

    "bill_amt1": 3913,
    "bill_amt2": 3102,
    "bill_amt3": 689,
    "bill_amt4": 0,
    "bill_amt5": 0,
    "bill_amt6": 0,

    "pay_amt1": 0,
    "pay_amt2": 689,
    "pay_amt3": 0,
    "pay_amt4": 0,
    "pay_amt5": 0,
    "pay_amt6": 0
}


response = requests.post(url, json=customer, timeout=30)

print("HTTP Status:", response.status_code)
print("Prediction:", response.json())
