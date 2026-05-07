import requests
import os
import time
from tqdm import tqdm

HEADERS = {
    "User-Agent": "research-project your_email@example.com"
}

# next 20 big companies
companies = {
    "oracle":"0001341439",
    "ibm":"0000051143",
    "qualcomm":"0000804328",
    "broadcom":"0001730168",
    "netflix":"0001065280",
    "uber":"0001543151",
    "airbnb":"0001559720",
    "paypal":"0001633917",
    "visa":"0001403161",
    "mastercard":"0001141391",

    "home_depot":"0000354950",
    "mcdonalds":"0000063908",
    "starbucks":"0000829224",
    "nike":"0000320187",
    "lowes":"0000060667",

    "lockheed_martin":"0000936468",
    "raytheon":"0000101829",
    "deere":"0000315189",

    "conocophillips":"0001163165",
    "valero":"0001035002"
}

BASE = "https://data.sec.gov/submissions/CIK{}.json"

os.makedirs("sec_10k_more", exist_ok=True)

for name, cik in companies.items():

    print("\nFetching:", name)

    folder = os.path.join("sec_10k_more", name)
    os.makedirs(folder, exist_ok=True)

    try:
        url = BASE.format(cik)
        data = requests.get(url, headers=HEADERS).json()

        forms = data["filings"]["recent"]["form"]
        accessions = data["filings"]["recent"]["accessionNumber"]
        docs = data["filings"]["recent"]["primaryDocument"]

        for form, acc, doc in zip(forms, accessions, docs):

            if form == "10-K":

                acc_clean = acc.replace("-", "")

                filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"

                r = requests.get(filing_url, headers=HEADERS)

                file_path = os.path.join(folder, f"{name}_{acc}.txt")

                with open(file_path, "wb") as f:
                    f.write(r.content)

                print("Saved:", file_path)

                time.sleep(0.7)

    except Exception as e:
        print("Failed:", name, e)