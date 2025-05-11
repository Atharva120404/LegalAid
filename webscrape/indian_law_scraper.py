"""
Indian Constitutional Law Scraper

This script scrapes Indian constitutional laws and related legal documents from authoritative websites
and stores them in structured formats (CSV and JSON).

Supported websites:
- Indian Kanoon (indiankanoon.org)
- India Code (indiacode.nic.in)
- Legislative Department (legislative.gov.in)

Features:
- Scrapes full text of laws, amendments, and related metadata
- Respects website rate limits with delays
- Error handling and logging
- Stores data in CSV and JSON formats
"""

import requests
from bs4 import BeautifulSoup
import csv
import json
import os
import time
import logging
import random
import re
from urllib.parse import urljoin
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("indian_law_scraper")

# Create output directories if they don't exist
os.makedirs('data/csv', exist_ok=True)
os.makedirs('data/json', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

# Headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Rate limiting functions
def random_delay(min_seconds=1, max_seconds=3):
    """Add a random delay between requests to avoid overloading the server"""
    delay = random.uniform(min_seconds, max_seconds)
    logger.info(f"Waiting for {delay:.2f} seconds before next request")
    time.sleep(delay)

def make_request(url, max_retries=3):
    """Make a request with retries and error handling"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt+1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt + random.uniform(0, 1)
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to retrieve {url} after {max_retries} attempts")
                return None

# Helper functions
def clean_text(text):
    """Clean up text by removing extra whitespace and normalizing newlines"""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Clean up newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def sanitize_filename(name):
    """Convert a string to a valid filename"""
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')

# Indian Kanoon scraper
class IndianKanoonScraper:
    BASE_URL = "https://indiankanoon.org"
    
    def __init__(self):
        self.laws = []

    def scrape_constitution(self):
        """Scrape the Indian Constitution"""
        logger.info("Scraping Indian Constitution from Indian Kanoon")
        
        # First, get the Constitution landing page
        url = f"{self.BASE_URL}/browse/constitution"
        response = make_request(url)
        if not response:
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find links to constitution parts
        parts_links = []
        for link in soup.find_all('a', href=True):
            if '/doc/constitution' in link['href']:
                parts_links.append(link['href'])
        
        # Process each part
        for part_link in parts_links:
            random_delay()
            self._process_constitution_part(part_link)
    
    def _process_constitution_part(self, part_link):
        """Process a specific part of the constitution"""
        full_url = urljoin(self.BASE_URL, part_link)
        logger.info(f"Processing constitution part: {full_url}")
        
        response = make_request(full_url)
        if not response:
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title_elem = soup.find('div', class_='doc_title')
        title = title_elem.text.strip() if title_elem else "Unknown Part"
        
        # Extract content
        content_elem = soup.find('div', class_='doc_content')
        content = content_elem.text.strip() if content_elem else ""
        
        # Extract articles and their content
        articles = []
        article_elems = soup.find_all('div', class_='article')
        for article_elem in article_elems:
            article_num = article_elem.find('div', class_='article_number')
            article_title = article_elem.find('div', class_='article_title')
            article_content = article_elem.find('div', class_='article_content')
            
            article = {
                'number': article_num.text.strip() if article_num else "",
                'title': article_title.text.strip() if article_title else "",
                'content': clean_text(article_content.text) if article_content else ""
            }
            articles.append(article)
        
        law = {
            'source': 'Indian Kanoon',
            'type': 'Constitution',
            'title': title,
            'url': full_url,
            'content': clean_text(content),
            'articles': articles
        }
        
        self.laws.append(law)
        logger.info(f"Added constitution part: {title}")
    
    def scrape_bare_acts(self, limit=10):
        """Scrape bare acts (limits the number to avoid overloading)"""
        logger.info(f"Scraping up to {limit} bare acts from Indian Kanoon")
        
        url = f"{self.BASE_URL}/browse/bareacts"
        response = make_request(url)
        if not response:
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find links to bare acts
        act_links = []
        for link in soup.find_all('a', href=True):
            if '/doc/' in link['href'] and 'bareact' in link['href']:
                act_links.append((link.text.strip(), link['href']))
        
        # Process a limited number of acts
        for i, (act_name, act_link) in enumerate(act_links[:limit]):
            if i > 0:  # Skip delay for the first request
                random_delay(2, 5)  # Longer delay for bare acts
            self._process_bare_act(act_name, act_link)
    
    def _process_bare_act(self, act_name, act_link):
        """Process a specific bare act"""
        full_url = urljoin(self.BASE_URL, act_link)
        logger.info(f"Processing bare act: {act_name} at {full_url}")
        
        response = make_request(full_url)
        if not response:
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract content
        content_elem = soup.find('div', class_='doc_content')
        content = content_elem.text.strip() if content_elem else ""
        
        # Extract sections
        sections = []
        section_elems = soup.find_all('div', class_='section')
        for section_elem in section_elems:
            section_num = section_elem.find('div', class_='section_number')
            section_title = section_elem.find('div', class_='section_title')
            section_content = section_elem.find('div', class_='section_content')
            
            section = {
                'number': section_num.text.strip() if section_num else "",
                'title': section_title.text.strip() if section_title else "",
                'content': clean_text(section_content.text) if section_content else ""
            }
            sections.append(section)
        
        law = {
            'source': 'Indian Kanoon',
            'type': 'Bare Act',
            'title': act_name,
            'url': full_url,
            'content': clean_text(content),
            'sections': sections
        }
        
        self.laws.append(law)
        logger.info(f"Added bare act: {act_name}")
    
    def save_data(self):
        """Save the scraped data to files"""
        if not self.laws:
            logger.warning("No laws to save from Indian Kanoon")
            return
        
        # Save as JSON
        with open('data/json/indian_kanoon_laws.json', 'w', encoding='utf-8') as f:
            json.dump(self.laws, f, ensure_ascii=False, indent=2)
        
        # Save main CSV
        with open('data/csv/indian_kanoon_laws.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'type', 'title', 'url', 'content'])
            
            for law in self.laws:
                writer.writerow([
                    law['source'],
                    law['type'],
                    law['title'], 
                    law['url'],
                    law['content']
                ])
        
        # Save articles/sections as separate CSVs
        articles_data = []
        sections_data = []
        
        for law in self.laws:
            if 'articles' in law:
                for article in law['articles']:
                    articles_data.append({
                        'law_title': law['title'],
                        'article_number': article['number'],
                        'article_title': article['title'],
                        'article_content': article['content']
                    })
            
            if 'sections' in law:
                for section in law['sections']:
                    sections_data.append({
                        'law_title': law['title'],
                        'section_number': section['number'],
                        'section_title': section['title'],
                        'section_content': section['content']
                    })
        
        if articles_data:
            pd.DataFrame(articles_data).to_csv('data/csv/indian_kanoon_articles.csv', index=False)
        
        if sections_data:
            pd.DataFrame(sections_data).to_csv('data/csv/indian_kanoon_sections.csv', index=False)
        
        logger.info(f"Saved {len(self.laws)} laws from Indian Kanoon")


# India Code Scraper
class IndiaCodeScraper:
    BASE_URL = "https://www.indiacode.nic.in"
    
    def __init__(self):
        self.laws = []
    
    def scrape_acts(self, limit=10):
        """Scrape central acts from India Code website"""
        logger.info(f"Scraping up to {limit} central acts from India Code")
        
        # The URL for central acts
        url = f"{self.BASE_URL}/coiweb/welcome.html"
        response = make_request(url)
        if not response:
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find links to constitution parts or acts
        act_links = []
        for link in soup.find_all('a', href=True):
            # This is a simplified version - the actual site structure might be different
            if 'act' in link['href'].lower() or 'constitution' in link['href'].lower():
                act_links.append((link.text.strip(), link['href']))
        
        # Process a limited number of acts
        for i, (act_name, act_link) in enumerate(act_links[:limit]):
            if i > 0:
                random_delay(2, 5)
            self._process_act(act_name, act_link)
    
    def _process_act(self, act_name, act_link):
        """Process a specific act"""
        full_url = urljoin(self.BASE_URL, act_link)
        logger.info(f"Processing act: {act_name} at {full_url}")
        
        response = make_request(full_url)
        if not response:
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # The exact structure will depend on the website, this is a simplified approach
        content = ""
        main_content = soup.find('div', id='content') or soup.find('div', class_='content')
        if main_content:
            content = main_content.text.strip()
        
        # Extract sections if available
        sections = []
        section_elems = soup.find_all('div', class_='section') or soup.find_all('p', class_='section')
        for section_elem in section_elems:
            section_text = section_elem.text.strip()
            section_match = re.search(r'Section\s+(\d+)[.\s:]+(.+)', section_text, re.IGNORECASE)
            
            if section_match:
                section = {
                    'number': section_match.group(1),
                    'title': '',  # India Code might not have separate titles
                    'content': clean_text(section_match.group(2))
                }
                sections.append(section)
        
        law = {
            'source': 'India Code',
            'type': 'Act' if 'constitution' not in act_name.lower() else 'Constitution',
            'title': act_name,
            'url': full_url,
            'content': clean_text(content),
            'sections': sections
        }
        
        self.laws.append(law)
        logger.info(f"Added act: {act_name}")
    
    def scrape_constitution(self):
        """Specifically scrape the constitution from India Code"""
        url = f"{self.BASE_URL}/coiweb/welcome.html"
        logger.info(f"Scraping Constitution from India Code at {url}")
        
        # Implementation similar to scrape_acts but focused on constitutional parts
        # For now, we'll use the general method as the site structure needs to be explored
        response = make_request(url)
        if not response:
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find links specifically to constitution parts
        const_links = []
        for link in soup.find_all('a', href=True):
            if 'constitution' in link['href'].lower():
                const_links.append((link.text.strip(), link['href']))
        
        for i, (part_name, part_link) in enumerate(const_links):
            if i > 0:
                random_delay()
            self._process_act(part_name, part_link)
    
    def save_data(self):
        """Save the scraped data to files"""
        if not self.laws:
            logger.warning("No laws to save from India Code")
            return
        
        # Save as JSON
        with open('data/json/india_code_laws.json', 'w', encoding='utf-8') as f:
            json.dump(self.laws, f, ensure_ascii=False, indent=2)
        
        # Save main CSV
        with open('data/csv/india_code_laws.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'type', 'title', 'url', 'content'])
            
            for law in self.laws:
                writer.writerow([
                    law['source'],
                    law['type'],
                    law['title'], 
                    law['url'],
                    law['content']
                ])
        
        # Save sections as separate CSV
        sections_data = []
        
        for law in self.laws:
            if 'sections' in law:
                for section in law['sections']:
                    sections_data.append({
                        'law_title': law['title'],
                        'section_number': section['number'],
                        'section_title': section['title'],
                        'section_content': section['content']
                    })
        
        if sections_data:
            pd.DataFrame(sections_data).to_csv('data/csv/india_code_sections.csv', index=False)
        
        logger.info(f"Saved {len(self.laws)} laws from India Code")


# Legislative Department Scraper
class LegislativeDeptScraper:
    BASE_URL = "https://legislative.gov.in"
    
    def __init__(self):
        self.laws = []
    
    def scrape_constitution(self):
        """Scrape the constitution from Legislative Department"""
        logger.info("Scraping Constitution from Legislative Department")
        
        url = f"{self.BASE_URL}/sites/default/files/coi-4March2016.pdf"
        # Note: PDF handling requires additional libraries like PyPDF2 or pdfplumber
        logger.info(f"Constitution PDF available at {url} - needs PDF processing")
        
        # For demonstration, we'll add a placeholder entry
        law = {
            'source': 'Legislative Department',
            'type': 'Constitution',
            'title': 'Constitution of India',
            'url': url,
            'content': "PDF content needs to be extracted separately",
            'requires_pdf_processing': True
        }
        
        self.laws.append(law)
    
    def scrape_acts(self, limit=5):
        """Scrape acts from Legislative Department"""
        logger.info(f"Scraping up to {limit} acts from Legislative Department")
        
        url = f"{self.BASE_URL}/constitution-of-india"
        response = make_request(url)
        if not response:
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find links to acts
        act_links = []
        for link in soup.find_all('a', href=True):
            if '.pdf' in link['href'] and ('act' in link['href'].lower() or 'law' in link['href'].lower()):
                act_links.append((link.text.strip() or "Unnamed Act", link['href']))
        
        # Process a limited number of acts
        for i, (act_name, act_link) in enumerate(act_links[:limit]):
            if i > 0:
                random_delay()
            
            full_url = urljoin(self.BASE_URL, act_link)
            
            law = {
                'source': 'Legislative Department',
                'type': 'Act',
                'title': act_name,
                'url': full_url,
                'content': "PDF content needs to be extracted separately",
                'requires_pdf_processing': True
            }
            
            self.laws.append(law)
            logger.info(f"Added act reference (PDF): {act_name}")
    
    def save_data(self):
        """Save the scraped data to files"""
        if not self.laws:
            logger.warning("No laws to save from Legislative Department")
            return
        
        # Save as JSON
        with open('data/json/legislative_dept_laws.json', 'w', encoding='utf-8') as f:
            json.dump(self.laws, f, ensure_ascii=False, indent=2)
        
        # Save main CSV
        with open('data/csv/legislative_dept_laws.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'type', 'title', 'url', 'requires_pdf_processing'])
            
            for law in self.laws:
                writer.writerow([
                    law['source'],
                    law['type'],
                    law['title'], 
                    law['url'],
                    law['requires_pdf_processing']
                ])
        
        logger.info(f"Saved {len(self.laws)} law references from Legislative Department")


class PaidSiteHandler:
    """Handler for paid sites - provides instructions instead of scraping"""
    
    def __init__(self):
        pass
    
    def generate_instructions(self):
        """Generate instructions for accessing paid sites"""
        logger.info("Generating instructions for paid sites")
        
        instructions = {
            'SCC Online': {
                'url': 'https://www.scconline.com',
                'access_method': 'Institutional login or subscription required',
                'api_availability': 'May offer API access for institutional subscribers',
                'alternative': 'Contact their support for bulk data access options'
            },
            'Manupatra': {
                'url': 'https://www.manupatrafast.com',
                'access_method': 'Paid subscription required',
                'api_availability': 'May have API for enterprise clients',
                'alternative': 'Consider reaching out for research partnerships'
            },
            'AIR Online': {
                'url': 'https://www.aironline.in',
                'access_method': 'Subscription-based access',
                'api_availability': 'Limited information available',
                'alternative': 'Check if they offer academic or bulk access programs'
            }
        }
        
        # Save instructions as JSON
        with open('data/json/paid_sites_instructions.json', 'w', encoding='utf-8') as f:
            json.dump(instructions, f, ensure_ascii=False, indent=2)
        
        # Create markdown file with instructions
        with open('data/paid_sites_instructions.md', 'w', encoding='utf-8') as f:
            f.write("# Accessing Indian Legal Data from Paid Sites\n\n")
            f.write("This document provides guidance for accessing legal data from subscription-based services.\n\n")
            
            for site, info in instructions.items():
                f.write(f"## {site}\n\n")
                f.write(f"- **URL**: {info['url']}\n")
                f.write(f"- **Access Method**: {info['access_method']}\n")
                f.write(f"- **API Availability**: {info['api_availability']}\n")
                f.write(f"- **Alternative Approach**: {info['alternative']}\n\n")
            
            f.write("\n## General Recommendations\n\n")
            f.write("1. **Institutional Access**: If you're affiliated with a university or research institution, check if they have subscriptions.\n")
            f.write("2. **Trial Access**: Many services offer free trials that could be useful for limited data collection.\n")
            f.write("3. **Research Collaboration**: Consider reaching out for academic or research partnerships.\n")
            f.write("4. **Terms of Service**: Always review the terms of service before scraping or bulk downloading.\n")
        
        logger.info("Generated instructions for paid sites")


def generate_summary_file():
    """Generate a summary of all scraped data"""
    logger.info("Generating summary report")
    
    # Count how many files we have
    csv_files = [f for f in os.listdir('data/csv') if f.endswith('.csv')]
    json_files = [f for f in os.listdir('data/json') if f.endswith('.json')]
    
    # Create summary
    with open('data/scraping_summary.md', 'w', encoding='utf-8') as f:
        f.write("# Indian Legal Data Scraping Summary\n\n")
        f.write(f"Data collection completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Data Sources\n\n")
        f.write("- Indian Kanoon (free, comprehensive)\n")
        f.write("- India Code (official government repository)\n")
        f.write("- Legislative Department (official government source)\n")
        f.write("- Information on paid sources (SCC Online, Manupatra, AIR Online)\n\n")
        
        f.write("## Files Generated\n\n")
        f.write(f"- {len(csv_files)} CSV files\n")
        f.write(f"- {len(json_files)} JSON files\n\n")
        
        f.write("## Data Organization\n\n")
        f.write("- `data/csv/`: Contains all CSV files with structured data\n")
        f.write("- `data/json/`: Contains all JSON files with complete data\n")
        f.write("- `data/raw/`: Contains any raw downloaded content\n")
        f.write("- `scraper.log`: Contains detailed logging information\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review the data for quality and coverage\n")
        f.write("2. Process PDF files that were identified but not parsed\n")
        f.write("3. Consider accessing paid sources if more comprehensive data is needed\n")
    
    logger.info("Summary report generated")


def main():
    """Main function to run the scrapers"""
    logger.info("Starting Indian legal data scraping")
    
    try:
        # Scrape from Indian Kanoon
        ik_scraper = IndianKanoonScraper()
        ik_scraper.scrape_constitution()
        ik_scraper.scrape_bare_acts(limit=5)  # Limiting to 5 for demonstration
        ik_scraper.save_data()
        
        # Scrape from India Code
        # ic_scraper = IndiaCodeScraper()
        # ic_scraper.scrape_constitution()
        # ic_scraper.scrape_acts(limit=5)  # Limiting to 5 for demonstration
        # ic_scraper.save_data()
        
        # # Scrape from Legislative Department
        # ld_scraper = LegislativeDeptScraper()
        # ld_scraper.scrape_constitution()
        # ld_scraper.scrape_acts(limit=3)  # Limiting to 3 for demonstration
        # ld_scraper.save_data()
        
        # Handle paid sites
        paid_handler = PaidSiteHandler()
        paid_handler.generate_instructions()
        
        # Generate summary
        generate_summary_file()
        
        logger.info("Scraping completed successfully")
        print("\nScraping completed successfully. Data is stored in the 'data' directory.")
        print("Check 'scraping_summary.md' for an overview of the collected data.")
        
    except Exception as e:
        logger.error(f"An error occurred during scraping: {str(e)}", exc_info=True)
        print(f"\nAn error occurred: {str(e)}")
        print("Check the log file for details.")


if __name__ == "__main__":
    main()