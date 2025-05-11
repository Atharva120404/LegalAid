import csv
import time
import logging
import re
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Set up logging
logging.basicConfig(
    filename="scraper_v3.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

def clean_text(text):
    """Clean text by removing extra whitespace, HTML, and irrelevant content."""
    if not text or text == "None":
        return "None"
    # Remove excessive newlines, tabs, and spaces
    text = " ".join(text.split())
    # Truncate if too long
    if len(text) > 1000:
        text = text[:997] + "..."
    # Remove JavaScript warnings, navigation, and metadata
    if any(keyword in text.lower() for keyword in [
        "no javascript", "select language", "home", "prev", "next",
        "chapter i", "chapter ii", "ipcchapter", "best viewed",
        "click here to leave a comment"
    ]):
        return "None"
    return text

def scrape_page_content(page, section_num, url, max_retries=3):
    """Attempt to scrape page content with retries and JavaScript handling."""
    for attempt in range(max_retries):
        try:
            page.goto(url, timeout=60000)
            page.wait_for_load_state("domcontentloaded", timeout=30000)
            # Wait for dynamic content
            page.wait_for_selector("div#content", timeout=10000)
            time.sleep(5)  # Extended wait for JavaScript rendering
            # Dismiss any overlays
            page.evaluate('''() => {
                const overlay = document.querySelector(".overlay, .notification");
                if (overlay) overlay.style.display = "none";
            }''')
            return True
        except PlaywrightTimeoutError as e:
            logging.warning(f"Timeout on Section {section_num}, attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt == max_retries - 1:
                logging.error(f"Failed to load Section {section_num} after {max_retries} attempts")
                return False
        except Exception as e:
            logging.error(f"Error loading Section {section_num}, attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt == max_retries - 1:
                return False
    return False

def scrape_ipc_section(start_section=1, end_section=50):
    """Scrape IPC sections with detailed legal content."""
    sections_data = []
    base_url = "https://devgan.in/ipc/section/"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=1000)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            viewport={"width": 1280, "height": 1024}
        )
        page = context.new_page()

        for section_num in range(start_section, end_section + 1):
            url = f"{base_url}{section_num}/"
            print(f"[*] Scraping Section {section_num}...")
            logging.info(f"Scraping Section {section_num}: {url}")

            try:
                # Load the page
                if not scrape_page_content(page, section_num, url):
                    raise Exception("Failed to load page after retries")

                # Debug: Log page structure
                page_structure = page.evaluate('''() => {
                    const content = document.querySelector("div#content");
                    return content ? content.innerHTML.slice(0, 200) : "No content div found";
                }''')
                logging.info(f"Section {section_num}: Page structure: {page_structure}...")

                # Extract Provision Text
                provision_text = page.evaluate('''() => {
                    const selectors = [
                        "div#content div.section-content p",
                        "div#content div.section-text",
                        "div#content p:not(.nav, .footer, .explanation, .case-law)",
                        "div#content div.provision",
                        "div#content div.text-start"
                    ];
                    for (const sel of selectors) {
                        const elements = document.querySelectorAll(sel);
                        for (const el of elements) {
                            const text = el.textContent.trim();
                            if (text.length > 20 && !text.match(/^(Home|Prev|Next|Chapter|Select Language|Click here)/i)) {
                                return text;
                            }
                        }
                    }
                    // Fallback: Capture any paragraph in content
                    const fallback = document.querySelectorAll("div#content p");
                    for (const p of fallback) {
                        const text = p.textContent.trim();
                        if (text.length > 20 && !text.match(/^(Home|Prev|Next|Chapter|Select Language|Click here)/i)) {
                            return text;
                        }
                    }
                    return "No provision text found";
                }''')
                provision_text = clean_text(provision_text)
                if provision_text == "None":
                    provision_text = "SCRAPE FAILED"
                logging.info(f"Section {section_num}: Provision text: {provision_text[:50]}...")

                # Extract Explanations
                explanations = page.evaluate('''() => {
                    const selectors = [
                        "div#content div.explanation",
                        "div#content div.notes",
                        "div#content p[class*='explan']",
                        "div#content p:not(.section-text, .case-law, .nav, .footer)"
                    ];
                    let explanations = [];
                    for (const sel of selectors) {
                        const elements = document.querySelectorAll(sel);
                        for (const el of elements) {
                            const text = el.textContent.trim();
                            if (text.length > 50 &&
                                !text.match(/v\.|vs\.|AIR|SCC|SCR|SC|HC/i) && // Exclude case laws
                                !text.match(/^(Home|Prev|Next|Chapter|Select Language|Click here)/i)) {
                                explanations.push(text);
                            }
                        }
                    }
                    return explanations.join("\\n---\\n") || "None";
                }''')
                explanations = clean_text(explanations)
                logging.info(f"Section {section_num}: Explanations: {explanations[:50]}...")

                # Extract Case Laws
                case_laws = page.evaluate('''() => {
                    const selectors = [
                        "div#content div.case-law",
                        "div#content ul li",
                        "div#content p[class*='case']",
                        "div#content blockquote",
                        "div#content p"
                    ];
                    let cases = [];
                    for (const sel of selectors) {
                        const elements = document.querySelectorAll(sel);
                        for (const el of elements) {
                            const text = el.textContent.trim();
                            if ((text.match(/v\.|vs\.|AIR|SCC|SCR|SC|HC/i) ||
                                 text.match(/\d{4}\s*(AIR|SCC|SCR|SC|HC)/i)) &&
                                text.length > 20 &&
                                !text.match(/^(Home|Prev|Next|Chapter|Select Language|Click here)/i)) {
                                cases.push(text);
                            }
                        }
                    }
                    return cases.join("\\n---\\n") || "None";
                }''')
                case_laws = clean_text(case_laws)
                logging.info(f"Section {section_num}: Case laws: {case_laws[:50]}...")

                # Extract Related Sections
                related_sections = page.evaluate('''() => {
                    const bodyText = document.body.textContent;
                    const sectionRegex = /Section\s+(\d+[A-Z]*(?:,\s*\d+[A-Z]*)*)/gi;
                    const matches = [];
                    let match;
                    while ((match = sectionRegex.exec(bodyText)) !== null) {
                        const sections = match[1].split(",").map(s => s.trim());
                        matches.push(...sections);
                    }
                    return [...new Set(matches)]
                        .filter(s => s.match(/^\d+[A-Z]*$/) && !s.includes("Indian"))
                        .join(", ") || "None";
                }''')
                related_sections = clean_text(related_sections)
                logging.info(f"Section {section_num}: Related sections: {related_sections}")

                # Extract and Scrape Links
                links = page.evaluate('''() => {
                    const anchors = document.querySelectorAll("div#content a");
                    const urls = [];
                    for (const a of anchors) {
                        const href = a.href;
                        const text = a.textContent.trim();
                        if (href.match(/devgan\.in\/.*(case|judgment|order)\/.*/i) &&
                            !text.match(/^(Home|Prev|Next|Select Language|Click here|S\.\s*\d+)/i)) {
                            urls.push({ url: href, text: text });
                        }
                    }
                    return urls;
                }''')
                logging.info(f"Section {section_num}: Found {len(links)} links: {[link['url'] for link in links]}")

                # Scrape linked content (limit to 3 links)
                linked_content = []
                for link in links[:3]:
                    try:
                        print(f"[*] Scraping linked page: {link['url']}...")
                        logging.info(f"Scraping link for Section {section_num}: {link['url']}")
                        if scrape_page_content(page, section_num, link['url']):
                            link_content = page.evaluate('''() => {
                                const selectors = [
                                    "div#content div.case-details",
                                    "div#content div.judgment-text",
                                    "div#content article",
                                    "div#content p:not(.nav, .footer)"
                                ];
                                for (const sel of selectors) {
                                    const elements = document.querySelectorAll(sel);
                                    for (const el of elements) {
                                        const text = el.textContent.trim();
                                        if (text.length > 50 &&
                                            !text.match(/^(Home|Prev|Next|Chapter|Select Language|Click here)/i)) {
                                            return text;
                                        }
                                    }
                                }
                                return "No content found";
                            }''')
                            link_content = clean_text(link_content)
                            linked_content.append(f"Link: {link['text']} ({link['url']})\nContent: {link_content}")
                            logging.info(f"Section {section_num}: Scraped link {link['url']}: {link_content[:50]}...")
                        else:
                            linked_content.append(f"Link: {link['text']} ({link['url']})\nContent: SCRAPE FAILED")
                        # Return to main page
                        scrape_page_content(page, section_num, url)
                    except Exception as link_error:
                        print(f"[!] Failed to scrape link {link['url']}: {str(link_error)}")
                        logging.error(f"Failed link {link['url']}: {str(link_error)}")
                        linked_content.append(f"Link: {link['text']} ({link['url']})\nContent: SCRAPE FAILED")

                linked_content_text = "\n---\n".join(linked_content) if linked_content else "None"
                logging.info(f"Section {section_num}: {len(linked_content)} linked pages scraped")

                # Store data
                section_data = {
                    "Section": section_num,
                    "Provision Text": provision_text,
                    "Explanations": explanations,
                    "Case Laws": case_laws,
                    "Related Sections": related_sections,
                    "Linked Content": linked_content_text
                }
                sections_data.append(section_data)

                # Take screenshot if failed
                if provision_text == "SCRAPE FAILED" or case_laws == "None":
                    page.screenshot(path=f"failed_section_{section_num}.png")
                    logging.info(f"Section {section_num}: Screenshot saved to failed_section_{section_num}.png")

                print(f"[✓] Section {section_num}: {len(case_laws.split('---')) if case_laws != 'None' else 0} cases, {len(linked_content)} linked pages")
                logging.info(f"Completed Section {section_num}")

                # Save periodically
                if section_num % 10 == 0:
                    save_to_csv(sections_data, f"ipc_sections_partial_{section_num}.csv")

            except Exception as e:
                print(f"[!] Failed Section {section_num}: {str(e)}")
                logging.error(f"Failed Section {section_num}: {str(e)}")
                page.screenshot(path=f"failed_section_{section_num}.png")
                logging.info(f"Section {section_num}: Screenshot saved to failed_section_{section_num}.png")
                sections_data.append({
                    "Section": section_num,
                    "Provision Text": "SCRAPE FAILED",
                    "Explanations": "None",
                    "Case Laws": "None",
                    "Related Sections": "None",
                    "Linked Content": "None"
                })

        browser.close()

    return sections_data

def save_to_csv(data, filename="ipc_sections.csv"):
    """Save data to CSV with headers."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Section", "Provision Text", "Explanations", "Case Laws", "Related Sections", "Linked Content"])
        writer.writeheader()
        writer.writerows(data)
    print(f"[+] Data saved to {filename}")
    logging.info(f"Data saved to {filename}")

if __name__ == "__main__":
    print("[*] Launching IPC Section Scraper...")
    logging.info("Starting IPC Section Scraper")
    data = scrape_ipc_section(start_section=1, end_section=10)  # Limited to 10 for testing
    save_to_csv(data)
    print("[+] Done! Check ipc_sections.csv")
    logging.info("Scraper completed")