"""
Academic search tools for CoScientist.

This module contains tools for searching academic sources like arXiv and Google Scholar.
"""

import os
import re
import json
import logging
import requests
import time
from typing import Dict, List, Any, Optional, Union
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

class AcademicSearchTool:
    """
    Tool for searching academic sources.
    """
    
    def __init__(self, max_results: int = 5):
        """
        Initialize the academic search tool.
        
        Args:
            max_results (int, optional): Maximum number of results to return. Defaults to 5.
        """
        self.max_results = max_results
    
    def search_arxiv(self, query: str, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers related to the query.
        
        Args:
            query (str): Search query
            categories (List[str], optional): List of arXiv categories. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: List of papers found
        """
        try:
            # Construct the arXiv API query
            base_url = "http://export.arxiv.org/api/query?"
            
            # Clean and format the query
            query = query.replace(' ', '+')
            
            # Add categories if provided
            cat_str = ""
            if categories:
                cat_str = "cat:" + "+OR+cat:".join(categories)
                
            # Full search query
            if categories:
                search_query = f"search_query=all:{query}+AND+({cat_str})"
            else:
                search_query = f"search_query=all:{query}"
            
            # Add max results and sort by relevance
            params = f"{search_query}&start=0&max_results={self.max_results}&sortBy=relevance"
            
            # Send request to arXiv API
            response = requests.get(base_url + params)
            response.raise_for_status()
            
            # Parse XML response
            root = ElementTree.fromstring(response.content)
            
            # Extract namespace
            namespace = {'atom': 'http://www.w3.org/2005/Atom',
                         'arxiv': 'http://arxiv.org/schemas/atom'}
            
            # Get entries
            entries = root.findall('.//atom:entry', namespace)
            
            results = []
            for entry in entries:
                # Extract paper details
                title = entry.find('./atom:title', namespace).text.strip()
                summary = entry.find('./atom:summary', namespace).text.strip()
                published = entry.find('./atom:published', namespace).text
                
                # Get authors
                author_elements = entry.findall('./atom:author/atom:name', namespace)
                authors = [author.text for author in author_elements]
                
                # Get link to paper
                links = entry.findall('./atom:link', namespace)
                paper_url = ""
                for link in links:
                    if link.get('rel') == 'alternate':
                        paper_url = link.get('href')
                        break
                
                # Get categories
                category_elements = entry.findall('./arxiv:category', namespace)
                categories = [cat.get('term') for cat in category_elements]
                
                # Get DOI if available
                doi = ""
                for link in links:
                    if link.get('title') == 'doi':
                        doi = link.get('href')
                        break
                
                # Format summary for better readability
                summary = re.sub(r'\s+', ' ', summary)
                
                # Add to results
                paper = {
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "published": published,
                    "url": paper_url,
                    "categories": categories,
                    "doi": doi,
                    "source": "arXiv"
                }
                
                results.append(paper)
            
            logger.info(f"Found {len(results)} papers on arXiv for query: {query}")
            return results
        
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def search_semantic_scholar(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for papers related to the query.
        
        Args:
            query (str): Search query
            limit (int, optional): Maximum number of results. Defaults to self.max_results.
            
        Returns:
            List[Dict[str, Any]]: List of papers found
        """
        try:
            # Use the Semantic Scholar API
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            
            # Set parameters
            if limit is None:
                limit = self.max_results
                
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,authors,abstract,url,venue,year,citationCount,influentialCitationCount,referenceCount"
            }
            
            # Send request
            headers = {"Accept": "application/json"}
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            results = []
            
            for paper in data.get("data", []):
                # Extract author names
                authors = [author.get("name", "") for author in paper.get("authors", [])]
                
                # Add to results
                result = {
                    "title": paper.get("title", ""),
                    "authors": authors,
                    "summary": paper.get("abstract", ""),
                    "published": paper.get("year", ""),
                    "url": paper.get("url", ""),
                    "venue": paper.get("venue", ""),
                    "citation_count": paper.get("citationCount", 0),
                    "influential_citation_count": paper.get("influentialCitationCount", 0),
                    "reference_count": paper.get("referenceCount", 0),
                    "source": "Semantic Scholar"
                }
                
                results.append(result)
            
            logger.info(f"Found {len(results)} papers on Semantic Scholar for query: {query}")
            return results
        
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    def search_all_sources(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search all academic sources.
        
        Args:
            query (str): Search query
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Results from all sources
        """
        results = {}
        
        # Search arXiv
        arxiv_results = self.search_arxiv(query)
        results["arxiv"] = arxiv_results
        
        # Sleep briefly to avoid overwhelming APIs
        time.sleep(1)
        
        # Search Semantic Scholar
        semantic_scholar_results = self.search_semantic_scholar(query)
        results["semantic_scholar"] = semantic_scholar_results
        
        return results
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract keywords from text for better search queries.
        
        Args:
            text (str): Text to extract keywords from
            max_keywords (int, optional): Maximum number of keywords. Defaults to 5.
            
        Returns:
            List[str]: Extracted keywords
        """
        # Simple keyword extraction based on term frequency
        # In a real implementation, you could use more sophisticated NLP techniques
        
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Remove common stopwords (a simple list for demonstration)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 
                    'by', 'about', 'of', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
                    'should', 'can', 'could', 'may', 'might', 'that', 'this', 'these', 'those'}
        
        words = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Count word frequency
        word_counts = {}
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        keywords = [word for word, count in sorted_words[:max_keywords]]
        
        return keywords
    
    def format_references(self, papers: List[Dict[str, Any]]) -> str:
        """
        Format a list of papers as a bibliography.
        
        Args:
            papers (List[Dict[str, Any]]): List of papers
            
        Returns:
            str: Formatted bibliography
        """
        if not papers:
            return "No references found."
        
        bibliography = []
        
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Untitled")
            authors = paper.get("authors", [])
            year = paper.get("published", "").split('-')[0] if paper.get("published") else ""
            
            if not year and "year" in paper:
                year = str(paper["year"])
                
            url = paper.get("url", "")
            
            # Format authors
            if len(authors) > 5:
                authors_str = ", ".join(authors[:5]) + ", et al."
            else:
                authors_str = ", ".join(authors)
            
            # Create citation
            if authors and year:
                citation = f"[{i}] {authors_str} ({year}). {title}. "
            elif authors:
                citation = f"[{i}] {authors_str}. {title}. "
            else:
                citation = f"[{i}] {title}. "
            
            if url:
                citation += f"URL: {url}"
            
            bibliography.append(citation)
        
        return "\n\n".join(bibliography)

def search_academic_sources(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search academic sources for a query.
    
    Args:
        query (str): Search query
        max_results (int, optional): Maximum number of results. Defaults to 5.
        
    Returns:
        Dict[str, Any]: Search results
    """
    search_tool = AcademicSearchTool(max_results=max_results)
    
    # Extract keywords if the query is long
    if len(query.split()) > 7:
        keywords = search_tool.extract_keywords(query)
        if keywords:
            # Use the top keywords for a more focused search
            keyword_query = " ".join(keywords)
            logger.info(f"Using keywords for search: {keyword_query}")
            query = keyword_query
    
    # Search all sources
    results = search_tool.search_all_sources(query)
    
    # Combine and format results
    all_papers = []
    all_papers.extend(results.get("arxiv", []))
    all_papers.extend(results.get("semantic_scholar", []))
    
    # Deduplicate by title (simple approach)
    seen_titles = set()
    unique_papers = []
    
    for paper in all_papers:
        title = paper.get("title", "").lower()
        if title not in seen_titles:
            seen_titles.add(title)
            unique_papers.append(paper)
    
    # Sort by recency (if available)
    sorted_papers = sorted(
        unique_papers, 
        key=lambda x: x.get("published", "0000") if x.get("published") else "0000", 
        reverse=True
    )
    
    # Format as bibliography
    bibliography = search_tool.format_references(sorted_papers[:max_results])
    
    return {
        "papers": sorted_papers[:max_results],
        "bibliography": bibliography,
        "query": query
    } 