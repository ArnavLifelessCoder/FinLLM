import requests
import os
import time

headers = {
    "User-Agent": "research-project your_email@example.com"
}

companies = {
"apple":"0000320193",
"microsoft":"0000789019",
"amazon":"0001018724",
"nvidia":"0001045810",
"meta":"0001326801",
"alphabet":"0001652044",
"intel":"0000050863",
"amd":"0000002488",
"adobe":"0000796343",
"salesforce":"0001108524",

"jpmorgan":"0000019617",
"goldman_sachs":"0000886982",
"morgan_stanley":"0000895421",
"bank_of_america":"0000070858",
"citigroup":"0000831001",

"walmart":"0000104169",
"costco":"0000909832",
"target":"0000027419",
"coca_cola":"0000021344",
"pepsico":"0000077476",

"tesla":"0001318605",
"ford":"0000037996",
"general_motors":"0001467858",
"boeing":"0000012927",
"caterpillar":"0000018230",

"johnson_johnson":"0000200406",
"pfizer":"0000078003",
"moderna":"0001682852",

"exxonmobil":"0000034088",
"chevron":"0000093410"
}

base = "https://data.sec.gov/submissions/CIK{}.json"

os.makedirs("sec_10k", exist_ok=True)

for company, cik in companies.items():

    print("Fetching:", company)

    url = base.format(cik)
    data = requests.get(url, headers=headers).json()

    forms = data["filings"]["recent"]["form"]
    accessions = data["filings"]["recent"]["accessionNumber"]
    docs = data["filings"]["recent"]["primaryDocument"]

    for form, acc, doc in zip(forms, accessions, docs):

        if form == "10-K":

            acc_clean = acc.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"

            try:
                r = requests.get(filing_url, headers=headers)

                filename = f"sec_10k/{company}_{acc}.txt"

                with open(filename, "wb") as f:
                    f.write(r.content)

                print("Downloaded:", filename)

                time.sleep(0.5)

            except:
                print("Skipped", company, acc)