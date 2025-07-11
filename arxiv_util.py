import arxiv
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)

def get_arxiv_results(query: str, max_results: int):
    # Validate and sanitize inputs
    if not isinstance(query, str) or not query.strip():
        logging.error("Invalid query provided.")
        return []
    if not isinstance(max_results, int) or max_results <= 0:
        logging.error("Invalid max_results provided.")
        return []
    
    # Sanitize the query (optional, based on requirements)
    query = query.strip()
    
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query, 
            max_results=max_results, 
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        results = client.results(search)
        return list(results)
    except Exception as e:
        logging.error(f"Error fetching results: {e}")
        return []

def get_arxiv_message(result):
    try:
        # Safely handle summary, authors, and other attributes
        summary = getattr(result, 'summary', 'No summary available').replace('\n', ' ')
        authors = ', '.join([author.name for author in getattr(result, 'authors', [])]) 
        title = getattr(result, 'title', 'No title available')
        entry_id = getattr(result, 'entry_id', 'No URL available')
        upvote = getattr(result, 'upvotes', 'None')
        
        message = (
            f"**Upvote:** {upvote}\n"
            f"**Source:** arXiv\n"
            f"**Title:** {title}\n"
            f"**Authors:** {authors}\n"
            f"**Summary:** {summary}\n"
            f"**URL:** {entry_id}"
        )
        return message
    except Exception as e:
        logging.error(f"Error creating message: {e}")
        return "Unable to retrieve the message for this result."

# Example usage:
if __name__ == "__main__":
    query = "machine learning"
    max_results = 5
    results = get_arxiv_results(query, max_results)
    for result in results:
        message = get_arxiv_message(result)
        print(message)

