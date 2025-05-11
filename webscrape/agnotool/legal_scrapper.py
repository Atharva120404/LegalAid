import csv
import time
from playwright.sync_api import sync_playwright

def scrape_ipc_sections():
    """Reliably scrape all IPC sections from devgan.in"""
    sections_data = []
    base_url = "https://devgan.in/section_ipc.php?section="

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=1000)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            viewport={"width": 1280, "height": 1024}
        )
        page = context.new_page()

        for section_num in range(1, 512):
            url = f"{base_url}{section_num}"
            print(f"[*] Processing Section {section_num}...")

            try:
                # Load page with long timeout
                page.goto(url, timeout=120000)
                
                # Wait for basic content
                page.wait_for_selector("body", state="attached", timeout=30000)

                # Extract all visible text as fallback
                full_text = page.evaluate('''() => {
                    return document.body.innerText.trim();
                }''')

                # Extract structured data using properly escaped JS
                structured_data = page.evaluate('''() => {
                    const result = {
                        main_text: "",
                        explanations: [],
                        cases: []
                    };

                    // Try to find main provision text
                    const paragraphs = Array.from(document.querySelectorAll('p'));
                    for (const p of paragraphs) {
                        const text = p.innerText.trim();
                        if (text.includes("shall be punished") || 
                            text.includes("Whoever") || 
                            text.includes("imprisonment")) {
                            result.main_text = text;
                            break;
                        }
                    }

                    // Find explanations (long text blocks)
                    const divs = Array.from(document.querySelectorAll('div'));
                    for (const div of divs) {
                        const text = div.innerText.trim();
                        if (text.length > 200 && 
                            !text.includes(" v. ") && 
                            !text.includes(" vs. ")) {
                            result.explanations.push(text);
                        }
                    }

                    // Find case laws
                    const elements = Array.from(document.querySelectorAll('*'));
                    for (const el of elements) {
                        const text = el.innerText.trim();
                        if ((text.includes(" v. ") || 
                             text.includes(" vs. ") || 
                             text.match(/\\d{4} (AIR|SCC|SCR) \\d+/)) &&
                            text.length < 500) {
                            result.cases.push(text);
                        }
                    }

                    return result;
                }''')

                # Clean data
                main_text = structured_data['main_text'] or "Not found"
                explanations = "\n---\n".join(list(set(structured_data['explanations']))) or "None"
                cases = "\n---\n".join(list(set(structured_data['cases']))) or "None"

                sections_data.append({
                    "Section": section_num,
                    "Provision Text": main_text,
                    "Explanations": explanations,
                    "Case Laws": cases,
                    "Full Text": full_text[:10000]  # First 10k chars
                })

                print(f"[✓] Section {section_num}: Found {len(structured_data['cases'])} cases")

            except Exception as e:
                print(f"[!] Failed Section {section_num}: {str(e)}")
                sections_data.append({
                    "Section": section_num,
                    "Provision Text": "SCRAPE FAILED",
                    "Explanations": "SCRAPE FAILED",
                    "Case Laws": "SCRAPE FAILED",
                    "Full Text": "SCRAPE FAILED"
                })
                continue

        browser.close()

    return sections_data

def save_to_csv(data, filename="ipc_sections_final.csv"):
    """Save data to CSV."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"[+] Data saved to {filename}")

if __name__ == "__main__":
    print("[*] Starting IPC scraping...")
    data = scrape_ipc_sections()
    save_to_csv(data)
    print("[+] Scraping complete!")