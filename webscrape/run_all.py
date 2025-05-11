"""
Indian Constitutional Law Scraper - Main Runner

This script runs all components of the Indian legal data scraping system:
1. Scrapes freely available sources (Indian Kanoon, India Code, Legislative Department)
2. Processes PDFs found during scraping
3. Generates instructions for accessing paid sites
4. Creates a comprehensive summary report

Usage:
    python run_all.py [--limit N] [--skip-pdfs]

Options:
    --limit N      Limit the number of documents scraped per source (default: 10)
    --skip-pdfs    Skip PDF processing (useful if PDF libraries aren't installed)
"""

import os
import sys
import time
import logging
import argparse

# Import scraper modules
try:
    from indian_law_scraper import (
        IndianKanoonScraper, 
        IndiaCodeScraper, 
        LegislativeDeptScraper, 
        PaidSiteHandler,
        generate_summary_file
    )
except ImportError:
    print("Error: Cannot import from indian_law_scraper.py")
    print("Make sure it's in the same directory as this script.")
    sys.exit(1)

# Try to import PDF processor
try:
    from pdf_processing_extension import PDFProcessor
    PDF_PROCESSOR_AVAILABLE = True
except ImportError:
    PDF_PROCESSOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_all.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_all")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run the Indian legal data scraping system')
    parser.add_argument('--limit', type=int, default=10, 
                        help='Limit the number of documents scraped per source')
    parser.add_argument('--skip-pdfs', action='store_true',
                        help='Skip PDF processing')
    return parser.parse_args()

def run_all(args):
    """Run all components of the system"""
    # Record start time
    start_time = time.time()
    
    # Create output directories if they don't exist
    os.makedirs('data/csv', exist_ok=True)
    os.makedirs('data/json', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    logger.info(f"Starting full scraping run with document limit: {args.limit}")
    
    try:
        # Step 1: Scrape from Indian Kanoon
        logger.info("Step 1: Scraping from Indian Kanoon")
        ik_scraper = IndianKanoonScraper()
        ik_scraper.scrape_constitution()
        ik_scraper.scrape_bare_acts(limit=args.limit)
        ik_scraper.save_data()
        
        # Step 2: Scrape from India Code
        logger.info("Step 2: Scraping from India Code")
        ic_scraper = IndiaCodeScraper()
        ic_scraper.scrape_constitution()
        ic_scraper.scrape_acts(limit=args.limit)
        ic_scraper.save_data()
        
        # Step 3: Scrape from Legislative Department
        logger.info("Step 3: Scraping from Legislative Department")
        ld_scraper = LegislativeDeptScraper()
        ld_scraper.scrape_constitution()
        ld_scraper.scrape_acts(limit=args.limit)
        ld_scraper.save_data()
        
        # Step 4: Handle paid sites
        logger.info("Step 4: Generating instructions for paid sites")
        paid_handler = PaidSiteHandler()
        paid_handler.generate_instructions()
        
        # Step 5: Process PDFs if available and not skipped
        if not args.skip_pdfs:
            if PDF_PROCESSOR_AVAILABLE:
                logger.info("Step 5: Processing PDF files")
                pdf_processor = PDFProcessor()
                pdf_processor.process_all_pdfs()
            else:
                logger.warning("PDF processing module not available. Skipping PDF processing.")
                print("\nWarning: PDF processing module not available.")
                print("Install PDF libraries with: pip install PyPDF2 or pip install pdfplumber")
        else:
            logger.info("Step 5: PDF processing skipped as requested")
        
        # Step 6: Generate summary report
        logger.info("Step 6: Generating summary report")
        generate_summary_file()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"All tasks completed successfully in {execution_time:.2f} seconds")
        
        print(f"\nScraping completed successfully in {execution_time:.2f} seconds.")
        print("Data is stored in the 'data' directory.")
        print("Check 'data/scraping_summary.md' for an overview of the collected data.")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        print(f"\nAn error occurred: {str(e)}")
        print("Check the log files for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(run_all(args))