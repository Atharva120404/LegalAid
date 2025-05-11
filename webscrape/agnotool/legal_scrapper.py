import csv
import time
import re
from playwright.sync_api import sync_playwright

def scrape_ipc_sections_with_playwright():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        )
        
        page = context.new_page()
        
        print("[*] Navigating to devgan.in...")
        try:
            page.goto("https://devgan.in/all_sections_ipc.php", timeout=60000)
            print("[*] Page loaded successfully")
            
            # Wait for the page to stabilize
            page.wait_for_load_state("networkidle", timeout=10000)
            time.sleep(3)  # Additional wait for JS rendering
            
            # Save initial state for debugging
            page.screenshot(path="devgan_loaded.png")
            with open("devgan_source.html", "w", encoding="utf-8") as f:
                f.write(page.content())
            print("[*] Saved initial page state for inspection")
            
            # Try different selectors for the IPC sections
            selectors_to_try = [
                "table a[href*='section_ipc.php']",  # Links in tables
                "a[href*='section_ipc.php']",        # Any link with section_ipc.php
                "#leftpanel a",                      # Links in leftpanel
                ".container a",                      # Links in container class
                "body a"                             # All links as last resort
            ]
            
            links = []
            used_selector = ""
            
            for selector in selectors_to_try:
                print(f"[*] Trying selector: {selector}")
                links = page.query_selector_all(selector)
                if links:
                    used_selector = selector
                    print(f"[+] Found {len(links)} links with selector: {selector}")
                    break
            
            if not links:
                # Direct regex extraction from page content as last resort
                print("[*] No links found with selectors, trying regex pattern matching...")
                content = page.content()
                
                # Pattern to match section_ipc.php links with their text
                pattern = r'<a[^>]*href="([^"]*section_ipc\.php\?[^"]*)"[^>]*>(.*?)</a>'
                matches = re.findall(pattern, content)
                
                data = []
                base_url = "https://devgan.in/"
                
                for href, link_text in matches:
                    # Clean the text from HTML tags
                    clean_text = re.sub(r'<[^>]*>', '', link_text).strip()
                    
                    if ":" in clean_text:
                        section_part, title_part = clean_text.split(":", 1)
                        section_number = section_part.strip().split()[-1]
                        title = title_part.strip()
                    else:
                        # If no colon, assume the entire text is a section reference
                        parts = clean_text.split()
                        if len(parts) >= 2 and parts[0].lower() == "section":
                            section_number = parts[1].rstrip('.')
                            title = " ".join(parts[2:])
                        else:
                            section_number = "Unknown"
                            title = clean_text
                    
                    # Build full URL if relative
                    full_href = href if href.startswith('http') else base_url + href
                    
                    data.append({
                        "Section Number": section_number,
                        "Title": title,
                        "Link": full_href
                    })
                    print(f"[+] Found via regex: Section {section_number}: {title}")
                
                browser.close()
                print("[*] Scraping complete via regex. Total sections:", len(data))
                return data
            
            data = []
            
            # Process the links we found via selectors
            print(f"[*] Processing {len(links)} links found via selector: {used_selector}")
            for i, link in enumerate(links):
                try:
                    text = link.inner_text().strip()
                    href = link.get_attribute("href")
                    
                    if not text or not href:
                        continue
                    
                    if ":" in text:
                        # Format like "Section 123: Title here"
                        section_part, title_part = text.split(":", 1)
                        section_number = section_part.strip().split()[-1]
                        title = title_part.strip()
                    else:
                        # Format might be just "Section 123" or something else
                        parts = text.split()
                        if len(parts) >= 2 and parts[0].lower() == "section":
                            section_number = parts[1].rstrip('.')
                            title = " ".join(parts[2:]) if len(parts) > 2 else f"Section {section_number}"
                        else:
                            section_number = f"Unknown-{i+1}"
                            title = text
                    
                    # Build full URL if relative
                    full_href = href if href.startswith('http') else f"https://devgan.in/{href}"
                    
                    entry = {
                        "Section Number": section_number,
                        "Title": title,
                        "Link": full_href
                    }
                    print(f"[+] Found: Section {section_number}: {title}")
                    data.append(entry)
                    
                except Exception as link_error:
                    print(f"[!] Error processing link {i}: {link_error}")
            
            # If we found very few items, try an alternative approach
            if len(data) < 10:
                print("[*] Found too few items, trying alternative extraction...")
                # Execute JavaScript to get all links on the page
                links_info = page.evaluate("""
                    Array.from(document.querySelectorAll('a'))
                        .filter(a => a.href.includes('section_ipc.php'))
                        .map(a => ({
                            text: a.innerText.trim(),
                            href: a.href
                        }));
                """)
                
                print(f"[*] Found {len(links_info)} links via JavaScript")
                
                for info in links_info:
                    text = info['text']
                    href = info['href']
                    
                    if not text:
                        continue
                    
                    if ":" in text:
                        section_part, title_part = text.split(":", 1)
                        section_number = section_part.strip().split()[-1]
                        title = title_part.strip()
                    else:
                        parts = text.split()
                        if len(parts) >= 2 and parts[0].lower() == "section":
                            section_number = parts[1].rstrip('.')
                            title = " ".join(parts[2:]) if len(parts) > 2 else f"Section {section_number}"
                        else:
                            # Skip items that don't match our expected format
                            continue
                    
                    entry = {
                        "Section Number": section_number,
                        "Title": title,
                        "Link": href
                    }
                    print(f"[+] Found via JS: Section {section_number}: {title}")
                    
                    # Check if this entry is already in data
                    if not any(d['Section Number'] == section_number for d in data):
                        data.append(entry)
            
            browser.close()
            print("[*] Scraping complete. Total sections collected:", len(data))
            return data
            
        except Exception as e:
            print(f"[!] Error during scraping: {e}")
            # Save the screenshot and HTML for debugging
            try:
                page.screenshot(path="error_screenshot.png")
                with open("page_source.html", "w", encoding="utf-8") as f:
                    f.write(page.content())
                print("[*] Saved error screenshot and page source for debugging")
            except:
                print("[!] Failed to save debug files")
            
            browser.close()
            return []

def save_to_csv(data, filename="ipc_sections.csv"):
    if not data:
        print("[!] No data to save")
        return
    
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Section Number", "Title", "Link"])
        writer.writeheader()
        writer.writerows(data)
    print(f"[*] Data saved to {filename}")

if __name__ == "__main__":
    print("[*] Starting IPC sections scraper for devgan.in...")
    sections = scrape_ipc_sections_with_playwright()
    if sections:
        save_to_csv(sections)
    else:
        print("[!] No sections found. Check error_screenshot.png and page_source.html for debugging.")

# import requests
# from bs4 import BeautifulSoup
# import time

# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
#     "Accept-Language": "en-US,en;q=0.9",
# }

# def scrape_indiankanoon():
#     url = "https://devgan.in/all_sections_ipc.php"
    
#     response = requests.get(url, headers=headers)
    
#     if response.status_code != 200:
#         print(f"Failed to fetch page. Status code: {response.status_code}")
#         return
    
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Try to extract categories/law links
#     law_sections = soup.select('ul li a')  # Adjust this selector based on what you observe in page source
    
#     if not law_sections:
#         print("No data found. Cloudflare may have blocked the response.")
#         return

#     print("Laws/Categories Found:")
#     for a in law_sections:
#         href = a.get('href')  # safely returns None if 'href' not present
#         text = a.text.strip()
#         if href:  # Only print if href exists
#             print(f"- {text} => {href}")

# if __name__ == "__main__":
#     scrape_indiankanoon()
