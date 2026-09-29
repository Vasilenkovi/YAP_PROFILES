import requests
import xml.etree.ElementTree as ET
import os
import json
import pandas as pd
import time

BASE_OAI = "https://dspace.mit.edu/oai/request"
PDF_DIR = "pdf"
PROGRESS_FILE = "progress.json"
CSV_FILE = "processed.csv"
TARGET_COUNT = 1000

os.makedirs(PDF_DIR, exist_ok=True)

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
else:
    progress = {"resumptionToken": None}

if os.path.exists(CSV_FILE):
    df_processed = pd.read_csv(CSV_FILE)
    processed_handles = set(df_processed["handle"])
else:
    df_processed = pd.DataFrame(columns=["handle", "pdf_url", "status"])
    processed_handles = set()

downloaded_count = len(df_processed[df_processed["status"] == "success"])
print("Уже успешно скачано:", downloaded_count)

while downloaded_count < TARGET_COUNT:

    if progress["resumptionToken"]:
        params = {
            "verb": "ListRecords",
            "resumptionToken": progress["resumptionToken"]
        }
    else:
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "oai_dc"
        }

    try:
        r = requests.get(BASE_OAI, params=params, timeout=60)
        root = ET.fromstring(r.text)
    except Exception as e:
        print("Ошибка запроса OAI:", e)
        time.sleep(10)
        continue

    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/"
    }

    records = root.findall(".//oai:record", ns)
    if not records:
        print("Записи закончились.")
        break

    for record in records:
        if downloaded_count >= TARGET_COUNT:
            break

        identifier_elem = record.find(".//oai:identifier", ns)
        if identifier_elem is None:
            continue

        oai_id = identifier_elem.text
        handle = oai_id.split(":")[-1]

        if handle in processed_handles:
            continue

        success = False
        for seq in range(1, 4):
            pdf_url = f"https://dspace.mit.edu/bitstream/handle/{handle}/filename.pdf?sequence={seq}"
            print(f"Обработка {handle}, sequence={seq}")

            try:
                pdf_resp = requests.get(pdf_url, timeout=60)
                
                if pdf_resp.status_code == 429:
                    print("429 Too Many Requests — ждем 5 минут")
                    time.sleep(300)
                    continue
                    
                if pdf_resp.status_code == 404:
                    print(f"404 — PDF sequence {seq} не найден")
                    continue
                    
                if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                    filename = f"{handle.replace('/', '_')}_seq{seq}.pdf"
                    filepath = os.path.join(PDF_DIR, filename)
                    with open(filepath, "wb") as f:
                        f.write(pdf_resp.content)
                    success = True
                    downloaded_count += 1
                    print("Скачано:", downloaded_count)
                    break

                else:
                    print(f"Не удалось скачать PDF, статус: {pdf_resp.status_code}")

            except Exception as e:
                print("Ошибка скачивания:", e)
                continue

        status = "success" if success else "fail"
        df_processed = pd.concat([df_processed, pd.DataFrame([{
            "handle": handle,
            "pdf_url": pdf_url,
            "status": status
        }])], ignore_index=True)
        df_processed.to_csv(CSV_FILE, index=False)
        processed_handles.add(handle)

        time.sleep(0.5)  # polite pause

    token_elem = root.find(".//oai:resumptionToken", ns)
    if token_elem is not None and token_elem.text:
        progress["resumptionToken"] = token_elem.text
    else:
        print("resumptionToken отсутствует — архив закончился.")
        break

    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

    time.sleep(1)

print("Готово. Всего успешно скачано:", downloaded_count)
