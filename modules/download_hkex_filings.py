import os
import csv
import time
import requests

def download_hkex_filings(csv_path: str = "filings/filings.csv") -> None:
    """
    Downloads all filing files listed in the CSV file and saves them
    directly in the filings folder.

    Args:
        csv_path: Path to the filings CSV file.

    Raises:
        ValueError: If the CSV file does not exist, is malformed, or downloading fails.
    """
    # Normalize path to ensure absolute correctness
    if not os.path.isabs(csv_path):
        # Resolve relative to the project root
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(current_dir, csv_path)

    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file not found at '{csv_path}'. Cannot download filings.")

    filings_dir = os.path.dirname(csv_path)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"Reading filings from {csv_path}...")
    filings = []
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("download_link"):
                    filings.append(row)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    total = len(filings)
    print(f"Found {total} filings to download. Starting downloads...")

    for idx, filing in enumerate(filings, 1):
        url = filing["download_link"]
        filing_id = filing.get("id")
        if not filing_id:
            filing_id = f"filing_{idx}"
        
        # Get extension or default to .pdf
        parsed_url = url.split("?")[0]
        ext = os.path.splitext(parsed_url)[1]
        if not ext:
            ext = ".pdf"
            
        # Determine filename (e.g., NEWS_ID.pdf)
        filename = f"{filing_id}{ext}"
        dest_path = os.path.join(filings_dir, filename)

        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"[{idx}/{total}] [BYPASSED] {filename} already exists. Skipping download.")
            continue

        print(f"[{idx}/{total}] Downloading {filename}...")
        
        # Polite download with retries
        success = False
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=30, stream=True)
                response.raise_for_status()
                
                with open(dest_path, "wb") as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            out_file.write(chunk)
                success = True
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed for {filename}: {e}")
                time.sleep(1)

        if not success:
            print(f"  WARNING: Failed to download {filename} after 3 attempts.")
        
        # Polite delay to avoid rate limiting
        time.sleep(0.5)

    print(f"Completed downloading process. Files are stored under {filings_dir}/")
