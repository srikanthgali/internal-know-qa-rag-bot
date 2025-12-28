"""
Document Ingestion Script for RAG Chatbot

This script downloads documents from a specified website to build the knowledge base.
It supports downloading PDF, DOCX, TXT, MD and other document formats.

Usage:
    python scripts/ingest_documents.py --url https://handbook.gitlab.com/handbook/engineering --output data/raw --max-pages 50
"""

import os
import sys
import argparse
import requests
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import List, Set, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentDownloader:
    """
    Downloads documents from websites for RAG knowledge base.

    Supports multiple document formats and handles duplicate detection.
    """

    # Supported document extensions
    DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".md"}

    def __init__(
        self,
        base_url: str,
        output_dir: str = "data/raw",
        max_depth: int = 2,
        max_pages: int = 100,
        allowed_paths: Optional[List[str]] = None,
        rate_limit: float = 1.0,
    ):
        """
        Initialize document downloader.

        Args:
            base_url (str): Base URL to start crawling
            output_dir (str): Directory to save downloaded documents
            max_depth (int): Maximum depth for crawling links
            max_pages (int): Maximum number of pages to crawl
            allowed_paths (List[str]): List of URL path patterns to include
            rate_limit (float): Delay between requests in seconds
        """
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.allowed_paths = allowed_paths or []
        self.rate_limit = rate_limit
        self.visited_urls: Set[str] = set()
        self.downloaded_files: List[str] = []
        self.pages_crawled = 0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir.absolute()}")

    def is_allowed_url(self, url: str) -> bool:
        """
        Check if URL matches allowed path patterns.

        Args:
            url (str): URL to check

        Returns:
            bool: True if URL is allowed
        """
        if not self.allowed_paths:
            return True

        parsed = urlparse(url)
        return any(path in parsed.path for path in self.allowed_paths)

    def is_valid_document_url(self, url: str) -> bool:
        """
        Check if URL points to a downloadable document.

        Args:
            url (str): URL to check

        Returns:
            bool: True if URL is a document
        """
        parsed = urlparse(url)
        path = parsed.path.lower()
        return any(path.endswith(ext) for ext in self.DOCUMENT_EXTENSIONS)

    def save_html_as_markdown(self, url: str, html_content: str) -> bool:
        """
        Save HTML page content as markdown/text for RAG processing.

        Args:
            url (str): Source URL
            html_content (str): HTML content

        Returns:
            bool: True if save successful
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get text content
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            # Generate filename from URL
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]
            filename = "_".join(path_parts[-2:]) if len(path_parts) >= 2 else "page"
            filename = f"{filename}_{url_hash}.txt"

            file_path = self.output_dir / filename

            # Skip if already exists
            if file_path.exists():
                logger.debug(f"File already exists: {filename}")
                return True

            # Save content with metadata
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Source: {url}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(text)

            self.downloaded_files.append(str(file_path))
            logger.info(f"✓ Saved: {filename}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to save {url}: {e}")
            return False

    def download_file(self, url: str, filename: str = None) -> bool:
        """
        Download a file from URL.

        Args:
            url (str): URL of the file to download
            filename (str): Optional custom filename

        Returns:
            bool: True if download successful
        """
        try:
            # Generate filename if not provided
            if not filename:
                filename = os.path.basename(urlparse(url).path)
                if not filename:
                    filename = f"document_{len(self.downloaded_files)}.pdf"

            file_path = self.output_dir / filename

            # Skip if already exists
            if file_path.exists():
                logger.info(f"File already exists: {filename}")
                return True

            # Download with progress bar
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(file_path, "wb") as f:
                if total_size:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True, desc=filename
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            self.downloaded_files.append(str(file_path))
            logger.info(f"✓ Downloaded: {filename}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to download {url}: {e}")
            return False

    def extract_links(self, url: str, html_content: str) -> List[str]:
        """
        Extract all links from HTML content.

        Args:
            url (str): Current page URL
            html_content (str): HTML content to parse

        Returns:
            List[str]: List of extracted URLs
        """
        soup = BeautifulSoup(html_content, "html.parser")
        links = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(url, href)

            # Remove fragments
            full_url = full_url.split("#")[0]

            # Only process URLs from same domain
            if urlparse(full_url).netloc == urlparse(self.base_url).netloc:
                # Check if URL matches allowed paths
                if self.is_allowed_url(full_url):
                    links.append(full_url)

        return list(set(links))  # Remove duplicates

    def crawl_and_download(self, url: str, depth: int = 0):
        """
        Recursively crawl website and download documents.

        Args:
            url (str): URL to crawl
            depth (int): Current crawl depth
        """
        # Check limits
        if depth > self.max_depth:
            logger.debug(f"Max depth reached for {url}")
            return

        if self.pages_crawled >= self.max_pages:
            logger.warning(f"Max pages limit ({self.max_pages}) reached")
            return

        # Skip if already visited
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)
        self.pages_crawled += 1
        logger.info(
            f"Crawling [{depth}] ({self.pages_crawled}/{self.max_pages}): {url}"
        )

        try:
            # Rate limiting
            time.sleep(self.rate_limit)

            # If URL is a document, download it
            if self.is_valid_document_url(url):
                self.download_file(url)
                return

            # Otherwise, fetch and parse HTML
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Save HTML content as text
            self.save_html_as_markdown(url, response.text)

            # Extract and process links
            links = self.extract_links(url, response.text)

            for link in links:
                if self.pages_crawled >= self.max_pages:
                    break

                if self.is_valid_document_url(link):
                    self.download_file(link)
                else:
                    self.crawl_and_download(link, depth + 1)

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")

    def download_from_url_list(self, urls: List[str]):
        """
        Download documents from a list of URLs.

        Args:
            urls (List[str]): List of document URLs
        """
        logger.info(f"Downloading {len(urls)} documents...")

        for url in urls:
            if self.pages_crawled >= self.max_pages:
                logger.warning(f"Max pages limit ({self.max_pages}) reached")
                break

            self.download_file(url)
            self.pages_crawled += 1
            time.sleep(self.rate_limit)

    def generate_report(self):
        """Generate download report."""
        print("\n" + "=" * 70)
        print("📊 DOWNLOAD REPORT")
        print("=" * 70)
        print(f"Base URL: {self.base_url}")
        print(f"Output Directory: {self.output_dir.absolute()}")
        print(f"Pages Crawled: {self.pages_crawled}")
        print(f"Pages Visited: {len(self.visited_urls)}")
        print(f"Documents Downloaded: {len(self.downloaded_files)}")
        if self.allowed_paths:
            print(f"Allowed Paths: {', '.join(self.allowed_paths)}")
        print("\n📄 Downloaded Files:")
        for i, file_path in enumerate(self.downloaded_files[:20], 1):
            print(f"   {i}. {Path(file_path).name}")
        if len(self.downloaded_files) > 20:
            print(f"   ... and {len(self.downloaded_files) - 20} more files")
        print("=" * 70 + "\n")


def main():
    """Main function for document ingestion."""
    parser = argparse.ArgumentParser(
        description="Download documents for RAG knowledge base",
        epilog="""
Examples:
  # Crawl GitLab Engineering section (limited)
  python scripts/ingest_documents.py --url https://handbook.gitlab.com/handbook/engineering --max-pages 50

  # Crawl multiple sections
  python scripts/ingest_documents.py --url https://handbook.gitlab.com/handbook --paths engineering product --max-pages 100

  # Download from URL list
  python scripts/ingest_documents.py --urls-file urls.txt --max-pages 50
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url", type=str, help="Website URL to crawl and download documents from"
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        help="Text file containing list of document URLs (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for downloaded documents (default: data/raw)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum crawl depth (default: 2, recommended: 2-3)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum number of pages to crawl (default: 100, recommended: 50-200)",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Allowed URL path patterns to include (e.g., engineering product)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.url and not args.urls_file:
        parser.error("Either --url or --urls-file must be provided")

    # Initialize downloader
    downloader = DocumentDownloader(
        base_url=args.url or "",
        output_dir=args.output,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        allowed_paths=args.paths,
        rate_limit=args.rate_limit,
    )

    try:
        if args.urls_file:
            # Download from URL list
            with open(args.urls_file, "r") as f:
                urls = [line.strip() for line in f if line.strip()]
            downloader.download_from_url_list(urls)
        else:
            # Crawl and download
            downloader.crawl_and_download(args.url)

        # Generate report
        downloader.generate_report()

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
        downloader.generate_report()
    except Exception as e:
        logger.error(f"Error during document ingestion: {e}")
        raise


if __name__ == "__main__":
    main()
