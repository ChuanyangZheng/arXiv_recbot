# refer to https://github.com/AK391/dailypapersHN/blob/main/app.py
# another refer: https://github.com/elsatch/daily_hf_papers_abstracts/tree/main
import requests
from datetime import datetime, timezone
from easydict import EasyDict
import logging
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.ERROR)

API_URL = "https://huggingface.co/api/daily_papers"

class PaperManager:
    def __init__(self, papers_per_page=30):
        self.papers_per_page = papers_per_page
        self.current_page = 1
        self.papers = []
        self.total_pages = 1
        self.sort_method = "hot"  # Default sort method
        self.raw_papers = []  # To store fetched data

        print('loadding embeeding model: sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # 推荐小巧快的模型
        )

    def calculate_score(self, paper):
        """
        Calculate the score of a paper based on upvotes and age.
        This mimics the "hotness" algorithm used by platforms like Hacker News.
        """
        upvotes = paper.get('paper', {}).get('upvotes', 0)
        published_at_str = paper.get('publishedAt', datetime.now(timezone.utc).isoformat())
        try:
            published_time = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
        except ValueError:
            # If parsing fails, use current time to minimize the impact on sorting
            published_time = datetime.now(timezone.utc)
        
        time_diff = datetime.now(timezone.utc) - published_time
        time_diff_hours = time_diff.total_seconds() / 3600  # Convert time difference to hours

        # Avoid division by zero and apply the hotness formula
        score = upvotes / ((time_diff_hours + 2) ** 1.5)
        return score

    def calculate_rising_score(self, paper):
        """
        Calculate the rising score of a paper.
        This emphasizes recent upvotes and the rate of upvote accumulation.
        """
        upvotes = paper.get('paper', {}).get('upvotes', 0)
        published_at_str = paper.get('publishedAt', datetime.now(timezone.utc).isoformat())
        try:
            published_time = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
        except ValueError:
            published_time = datetime.now(timezone.utc)

        time_diff = datetime.now(timezone.utc) - published_time
        time_diff_hours = time_diff.total_seconds() / 3600  # Convert time difference to hours

        # Rising score favors papers that are gaining upvotes quickly
        # Adjusted to have a linear decay over time
        score = upvotes / (time_diff_hours + 1)
        return score

    def fetch_papers(self):
        try:
            response = requests.get(f"{API_URL}?limit=100")
            response.raise_for_status()
            data = response.json()

            if not data:
                print("No data received from API.")
                return False

            # Debug: Print keys of the first paper
            print("Keys in the first paper:", data[0].keys())

            self.papers = data  # Store raw data
            self.select_papers()
            self.sort_papers()
            self.total_pages = max((len(self.papers) + self.papers_per_page - 1) // self.papers_per_page, 1)
            self.current_page = 1
            return True
        except requests.RequestException as e:
            print(f"Error fetching papers: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def select_papers(self):
        paper_list = []
        for result in self.papers:
            result = EasyDict(result)
            summary = getattr(result, 'summary', 'No summary available').replace('\n', ' ')
            authors = ', '.join([author.name for author in getattr(result.paper, 'authors', [])]) 
            title = getattr(result, 'title', 'No title available')
            final_paper_text = f"**title**: {title}\n, **summary**: {summary}\n, **authors**: {authors}\n"
            paper_list.append(final_paper_text)
        metadatas = [{"index": i} for i in range(len(paper_list))]
        assert len(paper_list) == len(self.papers)
        vectorstore = FAISS.from_texts(paper_list, self.embeddings, metadatas=metadatas)
        docs_and_scores = vectorstore.similarity_search_with_score(self.query, k=30)
        select_paper_list = []
        for doc, score in docs_and_scores:
            idx = doc.metadata.get("index")
            select_paper_list.append(self.papers[idx])
        self.papers = select_paper_list

    def sort_papers(self):
        if self.sort_method == "hot":
            self.papers = sorted(
                self.papers,
                key=lambda x: self.calculate_score(x),
                reverse=True
            )
        elif self.sort_method == "new":
            self.papers = sorted(
                self.papers,
                # key=lambda x: x.get('publishedAt', ''),
                key=lambda x: x['paper'].get('submittedOnDailyAt', ''),
                reverse=True
            )
        elif self.sort_method == "rising":
            self.papers = sorted(
                self.papers,
                key=lambda x: self.calculate_rising_score(x),
                reverse=True
            )
        else:
            self.papers = sorted(
                self.papers,
                key=lambda x: self.calculate_score(x),
                reverse=True
            )

    def set_sort_method(self, method):
        if method not in ["hot", "new", "rising"]:
            method = "hot"
        print(f"Setting sort method to: {method}")
        self.sort_method = method
        self.sort_papers()
        self.current_page = 1
        return True  # Assume success

    def format_paper(self, paper, rank):
        title = paper.get('title', 'No title')
        paper_id = paper.get('paper', {}).get('id', '')
        url = f"https://huggingface.co/papers/{paper_id}"
        authors = ', '.join([author.get('name', '') for author in paper.get('paper', {}).get('authors', [])]) or 'Unknown'
        upvotes = paper.get('paper', {}).get('upvotes', 0)
        comments = paper.get('numComments', 0)
        published_time_str = paper.get('publishedAt', datetime.now(timezone.utc).isoformat())
        try:
            published_time = datetime.fromisoformat(published_time_str.replace('Z', '+00:00'))
        except ValueError:
            published_time = datetime.now(timezone.utc)
        time_diff = datetime.now(timezone.utc) - published_time
        time_ago_days = time_diff.days
        time_ago = f"{time_ago_days} days ago" if time_ago_days > 0 else "today"

        return f"""
        <tr class="athing">
            <td align="right" valign="top" class="title"><span class="rank">{rank}.</span></td>
            <td valign="top" class="title">
                <a href="{url}" class="storylink" target="_blank">{title}</a>
            </td>
        </tr>
        <tr>
            <td colspan="2" class="subtext">
                <span class="score">{upvotes} points</span> Authors: {authors} {time_ago} | {comments} comments
            </td>
        </tr>
        <tr class="spacer"><td colspan="2"></td></tr>
        """

    def render_papers(self):
        start = (self.current_page - 1) * self.papers_per_page
        end = start + self.papers_per_page
        current_papers = self.papers[start:end]

        if not current_papers:
            return "<div class='no-papers'>No papers available for this page.</div>"

        papers_html = "".join([self.format_paper(paper, idx + start + 1) for idx, paper in enumerate(current_papers)])
        return f"""
        <table border="0" cellpadding="0" cellspacing="0" class="itemlist">
            {papers_html}
        </table>
        """

    def next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
        return self.render_papers()

    def prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
        return self.render_papers()


paper_manager = PaperManager()


def get_hf_results(query: str, max_results: int, backdays: int):
    paper_manager.papers_per_page = max_results
    paper_manager.query = query
    if backdays <= 3:
        paper_manager.sort_method = 'new'
    fetch = paper_manager.fetch_papers()
    return paper_manager.papers


def get_hf_message(result):
    try:
        # Safely handle summary, authors, and other attributes
        summary = getattr(result, 'summary', 'No summary available').replace('\n', ' ')
        authors = ', '.join([author.name for author in getattr(result.paper, 'authors', [])]) 
        title = getattr(result, 'title', 'No title available')
        entry_id = getattr(result, 'entry_id', 'No URL available')

        message = (
            f"**Upvote:** {result.paper.upvotes}\n"
            f"**Source:** Daily Papers\n"
            f"**Title:** {title}\n"
            f"**Authors:** {authors}\n"
            f"**Summary:** {summary}\n"
            f"**URL:** {entry_id}"
        )
        return message
    except Exception as e:
        logging.error(f"Error creating message: {e}")
        return "Unable to retrieve the message for this result."


if __name__ == '__main__':
    results = get_hf_results('test', 20)
    print(results)