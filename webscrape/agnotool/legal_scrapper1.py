import csv
import time
import logging
from playwright.sync_api import sync_playwright

# Set up logging
logging.basicConfig(filename="scraper.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def rephrase_and_clean(data):
    """Clean scraped data to ensure readability."""
    cleaned_data = data.copy()

    # Clean Explanations
    if cleaned_data["Explanations"] != "None":
        explanations = cleaned_data["Explanations"]
        explanation_parts = explanations.split("\n")
        cleaned_parts = []
        for part in explanation_parts:
            part = part.strip()
            if len(part) > 50:
                if len(part) > 200:
                    part = part[:197] + "..."
                cleaned_parts.append(part)
        cleaned_data["Explanations"] = "\n---\n".join(cleaned_parts[:3]) if cleaned_parts else "None"

    # Clean Case Laws
    if cleaned_data["Case Laws"] != "None":
        case_laws = cleaned_data["Case Laws"]
        case_law_parts = case_laws.split("\n---\n")
        cleaned_cases = []
        for case in case_law_parts:
            case = case.strip()
            if " v. " in case or " vs. " in case:
                if "(" not in case and ")" not in case:
                    case = f"{case} (Citation not found)"
                cleaned_cases.append(case)
        cleaned_data["Case Laws"] = "\n---\n".join(cleaned_cases[:3]) if cleaned_cases else "None"

    # Clean Linked Content
    if cleaned_data["Linked Content"] != "None":
        linked_content = cleaned_data["Linked Content"]
        linked_parts = linked_content.split("\n---\n")
        cleaned_linked = []
        for part in linked_parts:
            if "Link: " in part and "Content: " in part:
                link_part, content_part = part.split("\nContent: ", 1)
                if len(content_part) > 500:
                    content_part = content_part[:497] + "..."
                cleaned_linked.append(f"{link_part}\nContent: {content_part}")
        cleaned_data["Linked Content"] = "\n---\n".join(cleaned_linked[:3]) if cleaned_linked else "None"

    return cleaned_data

def scrape_deep_ipc_knowledge():
    """Scrape IPC sections with structured legal knowledge (text, cases, explanations, linked content)."""
    sections_data = []
    base_url = "https://devgan.in/section_ipc.php?section="

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=200)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            viewport={"width": 1280, "height": 1024}
        )
        page = context.new_page()

        for section_num in range(1, 11):  # Scrape Sections 1-10
            url = f"{base_url}{section_num}"
            print(f"[*] Scraping Section {section_num}...")
            logging.info(f"Scraping Section {section_num}: {url}")

            try:
                page.goto(url, timeout=60000)
                page.wait_for_load_state("domcontentloaded", timeout=15000)
                time.sleep(1)

                # Debug: Log the page structure
                page_structure = page.evaluate('''() => {
                    const sectxt = document.querySelector(".sectxt");
                    return sectxt ? sectxt.outerHTML : "No .sectxt found";
                }''')
                logging.info(f"Section {section_num}: Page structure: {page_structure[:200]}...")

                # Extract MAIN LEGAL TEXT (Provision Text)
                main_text = "Not found"
                main_text = page.evaluate('''() => {
                    const sectxt = document.querySelector(".sectxt");
                    if (!sectxt) return "No .sectxt found";
                    const paragraphs = sectxt.querySelectorAll("p");
                    if (paragraphs.length > 0) {
                        for (const p of paragraphs) {
                            const text = p.textContent.trim();
                            if (text.length > 20) {  // Ensure it's a substantial paragraph
                                return text;
                            }
                        }
                    }
                    return sectxt.textContent.trim();
                }''')
                if main_text in ["Not found", "No .sectxt found"]:
                    main_text = page.evaluate('''() => {
                        const contentDiv = document.querySelector("div.panel-body, div.well, article, div[class*='content']");
                        return contentDiv ? contentDiv.textContent.trim() : "No content found";
                    }''')
                logging.info(f"Section {section_num}: Main text extracted: {main_text[:50]}...")

                # Extract EXPLANATIONS
                explanations = page.evaluate('''() => {
                    const explanationDivs = document.querySelectorAll(".panel-body, .well, .explanation");
                    let explanations = [];
                    for (const div of explanationDivs) {
                        const text = div.textContent.trim();
                        if (text.length > 50 && !text.includes(" v. ") && !text.includes(" vs. ")) {
                            explanations.push(text);
                        }
                    }
                    return explanations.join("\\n---\\n");
                }''')
                logging.info(f"Section {section_num}: {len(explanations.split('---')) if explanations else 0} explanations found")

                # Extract CASE LAWS
                case_laws = page.evaluate('''() => {
                    const cases = [];
                    const elements = document.querySelectorAll("li, p, blockquote, div");
                    for (const el of elements) {
                        const text = el.textContent.trim();
                        if (text.includes(" v. ") || text.includes(" vs. ") || text.match(/\\d{4} (AIR|SCC|SCR|SC|HC)/i)) {
                            cases.push(text);
                        }
                    }
                    return cases.join("\\n---\\n");
                }''')
                logging.info(f"Section {section_num}: {len(case_laws.split('---')) if case_laws else 0} case laws found")

                # Extract RELATED SECTIONS (Exclude main content)
                related_sections = page.evaluate('''() => {
                    const mainContent = document.querySelector(".sectxt")?.textContent || "";
                    const bodyText = document.body.textContent;
                    const sectionRegex = /Sections? (\\d+[A-Z]*(?:,\\s?\\d+[A-Z]*)*)/gi;
                    const matches = [];
                    let match;
                    while ((match = sectionRegex.exec(bodyText)) !== null) {
                        const sectionMatch = match[0];
                        if (!mainContent.includes(sectionMatch)) {
                            matches.push(sectionMatch);
                        }
                    }
                    return [...new Set(matches)].join(", ");
                }''')
                logging.info(f"Section {section_num}: Related sections: {related_sections}")

                # Extract LINKS (Fixed regex for excluding navigation links)
                links = page.evaluate('''() => {
                    const anchors = document.querySelectorAll("a");
                    let allLinks = Array.from(anchors).map(a => ({ url: a.href, text: a.textContent.trim() }));
                    let urls = [];
                    for (const a of anchors) {
                        const href = a.href;
                        const text = a.textContent.trim().toLowerCase();
                        // Exclude navigation links and irrelevant links (Home, About, Select Language)
                        if (href.includes("devgan.in") &&
                            !href.match(new RegExp("^https://devgan\\.in/(ipc|bns|crpc|nia|hma|ida|iea|cpc|mva)/?$")) &&
                            !href.match(new RegExp("^https://devgan\\.in(/$|\\?.*|about\\.php$)")) &&
                            !text.includes("select language") &&
                            !text.includes("home") &&
                            !text.includes("about")) {
                            urls.push({ url: href, text: a.textContent.trim() });
                        }
                    }
                    console.log("All links on page:", allLinks);
                    return urls;
                }''')
                logging.info(f"Section {section_num}: Found {len(links)} potential links: {[link['url'] for link in links]}")

                # Visit each link (limit to 5)
                linked_content = []
                for link in links[:5]:
                    try:
                        print(f"[*] Scraping linked page: {link['url']}...")
                        logging.info(f"Scraping link for Section {section_num}: {link['url']}")
                        page.goto(link['url'], timeout=60000)
                        page.wait_for_load_state("networkidle", timeout=15000)

                        link_content = page.evaluate('''() => {
                            const contentDiv = document.querySelector(".panel-body, .well, .sectxt, article, .case-details, .judgment-text, [class*='content']");
                            return contentDiv ? contentDiv.textContent.trim() : "No content found";
                        }''')
                        linked_content.append(f"Link: {link['text']} ({link['url']})\nContent: {link_content}")
                        logging.info(f"Section {section_num}: Scraped link {link['url']}: {link_content[:50]}...")

                        # Return to main page
                        page.goto(url, timeout=60000)
                        page.wait_for_load_state("domcontentloaded", timeout=15000)

                    except Exception as link_error:
                        print(f"[!] Failed to scrape link {link['url']}: {str(link_error)}")
                        logging.error(f"Failed link {link['url']}: {str(link_error)}")
                        linked_content.append(f"Link: {link['text']} ({link['url']})\nContent: SCRAPE FAILED")

                linked_content_text = "\n---\n".join(linked_content) if linked_content else "None"
                logging.info(f"Section {section_num}: {len(linked_content)} linked pages scraped")

                # Store data
                section_data = {
                    "Section": section_num,
                    "Provision Text": main_text,
                    "Explanations": explanations if explanations else "None",
                    "Case Laws": case_laws if case_laws else "None",
                    "Related Sections": related_sections if related_sections else "None",
                    "Linked Content": linked_content_text
                }

                # Clean the data
                cleaned_section_data = rephrase_and_clean(section_data)
                sections_data.append(cleaned_section_data)

                print(f"[✓] Section {section_num}: {len(case_laws.split('---')) if case_laws else 0} cases, {len(linked_content)} linked pages")
                logging.info(f"Completed Section {section_num}")

            except Exception as e:
                print(f"[!] Failed Section {section_num}: {str(e)}")
                logging.error(f"Failed Section {section_num}: {str(e)}")
                sections_data.append({
                    "Section": section_num,
                    "Provision Text": "SCRAPE FAILED",
                    "Explanations": "SCRAPE FAILED",
                    "Case Laws": "SCRAPE FAILED",
                    "Related Sections": "SCRAPE FAILED",
                    "Linked Content": "SCRAPE FAILED"
                })
                continue

        browser.close()

    return sections_data

def save_to_csv(data, filename="ipc_deep_knowledge.csv"):
    """Save data to CSV with headers."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"[+] Data saved to {filename}")
    logging.info(f"Data saved to {filename}")

if __name__ == "__main__":
    print("[*] Launching Deep IPC Knowledge Scraper...")
    logging.info("Starting Deep IPC Knowledge Scraper")
    data = scrape_deep_ipc_knowledge()
    save_to_csv(data)
    print("[+] Done! Check ipc_deep_knowledge.csv")
    logging.info("Scraper completed")