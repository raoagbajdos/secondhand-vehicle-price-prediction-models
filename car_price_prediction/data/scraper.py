"""
Web scraping functionality for car data collection.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..config import settings
from ..utils import get_logger

logger = get_logger(__name__)


class BaseScraper(ABC):
    """Base class for car data scrapers."""
    
    def __init__(self, brand: str, delay: float = None):
        self.brand = brand.lower()
        self.delay = delay or settings.scraping_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    @abstractmethod
    def get_car_listings(self, max_pages: int = None) -> List[Dict]:
        """Scrape car listings for the brand."""
        pass
    
    @abstractmethod
    def parse_car_details(self, listing_url: str) -> Dict:
        """Parse detailed information from a car listing."""
        pass
    
    def save_data(self, data: List[Dict], filename: Optional[str] = None) -> Path:
        """Save scraped data to CSV file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.brand}_scraped_data_{timestamp}.csv"
        
        output_path = settings.raw_data_dir / filename
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(data)} records to {output_path}")
        return output_path


class AutotraderScraper(BaseScraper):
    """Scraper for Autotrader UK."""
    
    BASE_URL = "https://www.autotrader.co.uk"
    
    def get_car_listings(self, max_pages: int = None) -> List[Dict]:
        """Scrape car listings from Autotrader."""
        max_pages = max_pages or settings.max_pages_per_brand
        cars = []
        
        search_url = f"{self.BASE_URL}/car-search?make={self.brand.upper()}"
        
        for page in range(1, max_pages + 1):
            try:
                logger.info(f"Scraping {self.brand} page {page}")
                
                response = self.session.get(
                    f"{search_url}&page={page}",
                    timeout=settings.request_timeout
                )
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                listings = soup.find_all('article', class_='product-card')
                
                if not listings:
                    logger.info(f"No more listings found on page {page}")
                    break
                
                for listing in listings:
                    car_data = self._parse_listing_card(listing)
                    if car_data:
                        cars.append(car_data)
                
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                continue
        
        return cars
    
    def _parse_listing_card(self, listing) -> Optional[Dict]:
        """Parse a car listing card."""
        try:
            # Extract basic information
            title_elem = listing.find('h3', class_='product-card-details__title')
            price_elem = listing.find('div', class_='product-card-pricing__price')
            mileage_elem = listing.find('li', string=lambda x: x and 'miles' in x.lower())
            year_elem = listing.find('li', string=lambda x: x and x.isdigit() and len(x) == 4)
            
            if not title_elem or not price_elem:
                return None
            
            title = title_elem.get_text(strip=True)
            price_text = price_elem.get_text(strip=True)
            
            # Clean price
            price = self._clean_price(price_text)
            mileage = self._clean_mileage(mileage_elem.get_text(strip=True) if mileage_elem else '')
            year = int(year_elem.get_text(strip=True)) if year_elem else None
            
            return {
                'brand': self.brand,
                'title': title,
                'price': price,
                'mileage': mileage,
                'year': year,
                'source': 'autotrader',
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error parsing listing: {e}")
            return None
    
    def parse_car_details(self, listing_url: str) -> Dict:
        """Parse detailed car information."""
        # Implementation for detailed scraping
        # This would extract more detailed specs from individual listing pages
        pass
    
    def _clean_price(self, price_text: str) -> Optional[float]:
        """Clean and convert price text to float."""
        try:
            # Remove currency symbols and commas
            price_clean = ''.join(c for c in price_text if c.isdigit())
            return float(price_clean) if price_clean else None
        except (ValueError, TypeError):
            return None
    
    def _clean_mileage(self, mileage_text: str) -> Optional[int]:
        """Clean and convert mileage text to int."""
        try:
            mileage_clean = ''.join(c for c in mileage_text if c.isdigit())
            return int(mileage_clean) if mileage_clean else None
        except (ValueError, TypeError):
            return None


class CarDataScraper:
    """Main scraper orchestrator."""
    
    SCRAPERS = {
        'autotrader': AutotraderScraper,
        # Add more scrapers here
    }
    
    def __init__(self, sources: List[str] = None):
        self.sources = sources or ['autotrader']
    
    def scrape_brand(self, brand: str, max_pages: int = None) -> List[Dict]:
        """Scrape data for a specific brand from all sources."""
        all_data = []
        
        for source in self.sources:
            if source in self.SCRAPERS:
                scraper_class = self.SCRAPERS[source]
                scraper = scraper_class(brand)
                
                try:
                    data = scraper.get_car_listings(max_pages)
                    all_data.extend(data)
                    logger.info(f"Scraped {len(data)} records from {source} for {brand}")
                except Exception as e:
                    logger.error(f"Error scraping {brand} from {source}: {e}")
        
        return all_data
    
    def scrape_all_brands(self, brands: List[str] = None, max_pages: int = None) -> Dict[str, List[Dict]]:
        """Scrape data for all specified brands."""
        brands = brands or settings.supported_brands
        results = {}
        
        for brand in brands:
            logger.info(f"Starting scraping for {brand}")
            brand_data = self.scrape_brand(brand, max_pages)
            results[brand] = brand_data
            
            # Save brand data
            if brand_data:
                scraper = AutotraderScraper(brand)  # Use default scraper for saving
                scraper.save_data(brand_data)
        
        return results


def scrape_brand_data(
    brands: Union[str, List[str]], 
    sources: List[str] = None,
    max_pages: int = None
) -> Dict[str, List[Dict]]:
    """
    Convenience function to scrape data for specified brands.
    
    Args:
        brands: Brand name or list of brand names
        sources: List of sources to scrape from
        max_pages: Maximum pages to scrape per brand
        
    Returns:
        Dictionary mapping brand names to scraped data
    """
    if isinstance(brands, str):
        brands = [brands]
    
    scraper = CarDataScraper(sources)
    return scraper.scrape_all_brands(brands, max_pages)


def main():
    """CLI entry point for scraping."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape car data")
    parser.add_argument(
        "--brands", 
        type=str, 
        default=",".join(settings.supported_brands),
        help="Comma-separated list of brands to scrape"
    )
    parser.add_argument(
        "--max-pages", 
        type=int, 
        default=settings.max_pages_per_brand,
        help="Maximum pages to scrape per brand"
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="autotrader",
        help="Comma-separated list of sources"
    )
    
    args = parser.parse_args()
    
    brands = [b.strip() for b in args.brands.split(",")]
    sources = [s.strip() for s in args.sources.split(",")]
    
    scraper = CarDataScraper(sources)
    results = scraper.scrape_all_brands(brands, args.max_pages)
    
    total_records = sum(len(data) for data in results.values())
    logger.info(f"Scraping completed. Total records: {total_records}")


if __name__ == "__main__":
    main()
