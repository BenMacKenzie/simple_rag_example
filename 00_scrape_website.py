# Databricks notebook source
pip install pyyaml

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import yaml



with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

config = config['data']
tables_config = config['tables_config']

output_dir = tables_config['source_path']

base_url = "https://kumo.ai"



# Keep track of visited URLs to avoid duplicates
visited_urls = set()

def save_html(url, html):
    # Generate a valid filename from the URL path
    parsed_url = urlparse(url)
    filename = parsed_url.path.strip("/").replace("/", "_") + ".html"
    filepath = os.path.join(output_dir, filename)
    
    # Save the HTML content to a file
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(html)
    print(f"Saved: {filepath}")

def scrape(url):
    # Avoid revisiting the same URL
    if url in visited_urls:
        return
    visited_urls.add(url)
    
    try:
        # Fetch the page content
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the HTML content
        save_html(url, response.text)
        
        # Parse HTML and find new links
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href")
            if href:
                full_url = urljoin(base_url, href)
                
                # Check if the URL belongs to the same domain and has not been visited
                if full_url.startswith(base_url) and full_url not in visited_urls:
                    scrape(full_url)
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")

# Start the scraping process
scrape(base_url)

