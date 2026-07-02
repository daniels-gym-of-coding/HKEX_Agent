import os
import csv
import json
import requests
from datetime import datetime

def get_hkex_filings(stock_id: str, start_date: str, end_date: str) -> list:
    """
    Retrieves the list of filings for a given HKEX stock ID within a date range,
    saves the output to a CSV file (filings/filings.csv), and returns the parsed list.

    Args:
        stock_id: The HKEX stock ID (e.g., "7609").
        start_date: The start date in YYYYMMDD format.
        end_date: The end date in YYYYMMDD format.

    Returns:
        A list of dictionaries, where each dictionary contains:
        - "id": The ID of the filing.
        - "name": The title/name of the filing.
        - "download_link": The absolute URL to download the filing.
        - "filing_date": The filing date formatted as 'YYYY-MM-DD'.

    Raises:
        ValueError: If inputs are invalid, request fails, response format is incorrect,
                    JSON parsing fails, or CSV write fails.
    """
    if not stock_id:
        raise ValueError("stock_id must be a non-empty string or identifier.")
    if not start_date or not end_date:
        raise ValueError("Both start_date and end_date must be provided in YYYYMMDD format.")

    stock_id = str(stock_id).strip()
    start_date = str(start_date).strip()
    end_date = str(end_date).strip()

    url = (
        f"https://www1.hkexnews.hk/search/titleSearchServlet.do?"
        f"sortDir=0&sortByOptions=DateTime&category=0&market=SEHK&lang=en&rowRange=1000"
        f"&stockId={stock_id}&fromDate={start_date}&toDate={end_date}"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to fetch filings for stockId '{stock_id}' due to network/HTTP error: {e}")

    try:
        data = response.json()
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response from HKEX API for stockId '{stock_id}': {e}. Raw response: {response.text[:200]}")

    result_str = data.get("result")
    if result_str is None:
        raise ValueError(f"Missing 'result' key in HKEX API response for stockId '{stock_id}'")

    try:
        filings_list = json.loads(result_str)
    except Exception as e:
        raise ValueError(f"Failed to parse inner 'result' JSON string for stockId '{stock_id}': {e}. Raw 'result': {result_str[:200]}")

    if not isinstance(filings_list, list):
        raise ValueError(f"Expected inner 'result' to be a JSON list, but got {type(filings_list)}")

    parsed_filings = []
    for item in filings_list:
        if not isinstance(item, dict):
            continue

        # Extract filing ID
        filing_id = str(item.get("NEWS_ID", "")).strip()

        # Extract name (TITLE)
        filing_name = str(item.get("TITLE", "")).strip()

        # Construct download link
        file_link = str(item.get("FILE_LINK", "")).strip()
        if file_link:
            if file_link.startswith("http"):
                download_link = file_link
            else:
                if not file_link.startswith("/"):
                    file_link = "/" + file_link
                download_link = f"https://www1.hkexnews.hk{file_link}"
        else:
            download_link = ""

        # Extract/parse filing date (using exact format %d/%m/%Y %H:%M)
        date_time_str = str(item.get("DATE_TIME", "")).strip()
        filing_date = ""
        if date_time_str:
            try:
                dt = datetime.strptime(date_time_str, "%d/%m/%Y %H:%M")
                filing_date = dt.strftime("%Y-%m-%d")
            except Exception:
                filing_date = date_time_str

        parsed_filings.append({
            "id": filing_id,
            "name": filing_name,
            "download_link": download_link,
            "filing_date": filing_date
        })

    # Determine path for filings directory and csv file
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filings_dir = os.path.join(current_dir, "filings")
    os.makedirs(filings_dir, exist_ok=True)
    csv_path = os.path.join(filings_dir, "filings.csv")

    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "download_link", "filing_date"])
            for f_dict in parsed_filings:
                writer.writerow([
                    f_dict["id"],
                    f_dict["name"],
                    f_dict["download_link"],
                    f_dict["filing_date"]
                ])
    except Exception as e:
        raise ValueError(f"Failed to write CSV file at '{csv_path}': {e}")

    return parsed_filings
