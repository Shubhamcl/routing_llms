#!/usr/bin/env python3
"""
Rate responses in an enhanced dataset using OpenRouter API.
Reads prompts and responses, then gets GPT-4 to rate them on a 1-5 scale.
Saves the dataset with ratings included.
"""

import json
import csv
import argparse
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import aiohttp
from dotenv import load_dotenv
import os
from datetime import datetime
import re

# Load environment variables
load_dotenv()

"""
- **Accuracy**: Is the information correct and factual?
- kept aside for now
"""

class AsyncRatingClient:
    """Async client for rating responses using OpenRouter API."""
    
    def __init__(self, api_key: str, max_concurrent: int = 15):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Shubhamcl/smart_routing",
            "X-Title": "Smart Routing Response Rating"
        }
        self.semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests
    
    def create_rating_prompt(self, original_prompt: str, model_response: str) -> str:
        """Create a prompt for rating the model response."""
        return f"""You are an expert evaluator tasked with rating the quality of AI model responses. 

Please rate the following response on a scale from 1 to 5 based on these criteria:
- **Accuracy**: Is the information correct?
- **Relevance**: Does it directly address the user's question/request?
- **Completeness**: Does it provide a thorough answer?
- **Clarity**: Is it well-written and easy to understand?
- **Helpfulness**: Would this response be useful to the user?

Rating Scale:
- **1**: Very Poor - Incorrect, irrelevant, or unhelpful
- **2**: Poor - Mostly incorrect or not very helpful
- **3**: Average - Somewhat helpful but has issues
- **4**: Good - Helpful and mostly accurate
- **5**: Excellent - Highly accurate, relevant, and helpful

**Original User Prompt:**
{original_prompt}

**Model Response to Rate:**
{model_response}

Please provide only a single number (1, 2, 3, 4, or 5) as your rating, followed by a brief explanation in parentheses.

Rating:"""
    
    async def rate_response(self, session: aiohttp.ClientSession, 
                          original_prompt: str, model_response: str,
                          rating_model: str = "openai/gpt-4o-mini", 
                          max_retries: int = 3) -> tuple[Optional[int], Optional[str]]:
        """Rate a model response using the specified rating model with retry logic."""
        
        rating_prompt = self.create_rating_prompt(original_prompt, model_response)
        
        data = {
            "model": rating_model,
            "messages": [
                {"role": "user", "content": rating_prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.1  # Low temperature for consistent ratings
        }
        
        for attempt in range(max_retries):
            async with self.semaphore:  # Limit concurrent requests
                try:
                    async with session.post(
                        self.base_url,
                        headers=self.headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            rating_text = result["choices"][0]["message"]["content"].strip()
                            
                            # Extract rating number from response
                            rating_num = self.extract_rating(rating_text)
                            if rating_num is not None:
                                return rating_num, rating_text
                            else:
                                # If rating extraction fails, try again with a more explicit prompt
                                if attempt < max_retries - 1:
                                    print(f"  Failed to extract rating (attempt {attempt + 1}), retrying...")
                                    # Modify prompt to be more explicit
                                    data["messages"][0]["content"] = rating_prompt + "\n\nIMPORTANT: Please start your response with just the number (1, 2, 3, 4, or 5)."
                                    await asyncio.sleep(1)  # Brief delay before retry
                                    continue
                                else:
                                    return None, f"Failed to extract rating after {max_retries} attempts: {rating_text}"
                        else:
                            error_text = await response.text()
                            if attempt < max_retries - 1:
                                print(f"  API Error {response.status} (attempt {attempt + 1}), retrying...")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            else:
                                print(f"API Error {response.status} after {max_retries} attempts: {error_text}")
                                return None, f"API Error after retries: {error_text}"
                            
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        print(f"  Request timeout (attempt {attempt + 1}), retrying...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        print(f"Request timeout after {max_retries} attempts")
                        return None, "Request timeout after retries"
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Request failed (attempt {attempt + 1}): {e}, retrying...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        print(f"Request failed after {max_retries} attempts: {e}")
                        return None, f"Request failed after retries: {e}"
        
        return None, f"All {max_retries} attempts failed"
    
    def extract_rating(self, rating_text: str) -> Optional[int]:
        """Extract the numeric rating from the response text."""
        # Look for patterns like "Rating: 4" or just "4" at the start
        patterns = [
            r'(?:Rating:?\s*)?([1-5])(?:\s*[\(\-]|$)',  # "Rating: 4" or "4 (" or "4"
            r'^([1-5])',  # Number at start of response
            r'([1-5])/5',  # "4/5" format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, rating_text)
            if match:
                try:
                    rating = int(match.group(1))
                    if 1 <= rating <= 5:
                        return rating
                except ValueError:
                    continue
        
        # If no clear rating found, return None
        return None
    
    async def rate_batch_responses(self, dataset: List[Dict], rating_model: str,
                                 max_retries: int = 3, progress_callback=None) -> List[Dict]:
        """Rate responses for a batch of prompts concurrently."""
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i, item in enumerate(dataset):
                task = self._process_single_rating(
                    session, item, rating_model, max_retries, i, progress_callback
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return successful results
            successful_results = []
            for result in results:
                if isinstance(result, Exception):
                    print(f"Rating task failed with exception: {result}")
                else:
                    successful_results.append(result)
            
            return successful_results
    
    async def _process_single_rating(self, session: aiohttp.ClientSession, item: Dict,
                                   rating_model: str, max_retries: int, index: int, 
                                   progress_callback=None) -> Dict:
        """Process a single item (rate the response)."""
        
        original_prompt = item.get('prompt', '')
        model_response = item.get('model_response', '')
        item_id = item.get('id', f'item_{index}')
        
        # Skip if no response to rate
        if not model_response or model_response.startswith('[ERROR:'):
            item['rating'] = None
            item['rating_explanation'] = "No valid response to rate"
            item['rating_model'] = rating_model
            item['rating_timestamp'] = datetime.now().isoformat()
            if progress_callback:
                progress_callback(index, item_id, False, None)
            return item
        
        # Get rating with retry logic
        rating_num, rating_text = await self.rate_response(
            session=session,
            original_prompt=original_prompt,
            model_response=model_response,
            rating_model=rating_model,
            max_retries=max_retries
        )
        
        # Update item with rating
        if rating_num is not None:
            item['rating'] = rating_num
            item['rating_explanation'] = rating_text
            item['rating_model'] = rating_model
            item['rating_timestamp'] = datetime.now().isoformat()
            if progress_callback:
                progress_callback(index, item_id, True, rating_num)
        else:
            item['rating'] = None
            item['rating_explanation'] = rating_text or "Failed to get rating"
            item['rating_model'] = rating_model
            item['rating_timestamp'] = datetime.now().isoformat()
            item['rating_error'] = True
            if progress_callback:
                progress_callback(index, item_id, False, None)
        
        return item

def load_enhanced_dataset(file_path: str) -> List[Dict]:
    """Load the enhanced dataset from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_rated_dataset(data: List[Dict], output_file: str, format_type: str = "jsonl"):
    """Save the dataset with ratings."""
    
    if format_type == "jsonl":
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    elif format_type == "csv":
        if data:
            # Collect all possible fieldnames from all items
            all_fieldnames = set()
            for item in data:
                all_fieldnames.update(item.keys())
            fieldnames = sorted(list(all_fieldnames))
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
    
    print(f"Rated dataset saved to: {output_file}")

async def rate_dataset_responses(
    input_file: str,
    rating_model: str = "openai/gpt-4o-mini",
    output_prefix: str = None,
    max_concurrent: int = 15,
    max_retries: int = 3,
    limit: Optional[int] = None
):
    """Rate all responses in the dataset using async requests with retry logic."""
    
    # Load API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        print("Error: Please set your OPENROUTER_API_KEY in the .env file")
        print("Get your API key from: https://openrouter.ai/keys")
        return
    
    # Load dataset
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found")
        return
    
    print(f"Loading enhanced dataset from {input_file}...")
    data = load_enhanced_dataset(input_file)
    
    if limit:
        data = data[:limit]
        print(f"Processing limited to first {limit} items")
    
    print(f"Loaded {len(data)} items to rate")
    
    # Check if responses exist
    valid_responses = sum(1 for item in data if item.get('model_response') and not item.get('model_response', '').startswith('[ERROR:'))
    print(f"Found {valid_responses} valid responses to rate")
    
    # Initialize async rating client
    client = AsyncRatingClient(api_key, max_concurrent=max_concurrent)
    
    # Generate output filename
    if output_prefix is None:
        # Extract base name from input file
        input_stem = Path(input_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"rated_{input_stem}_{timestamp}"
    
    # Progress tracking
    print(f"Rating responses using model: {rating_model}")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"Max retries per request: {max_retries}")
    print("-" * 60)
    
    successful_ratings = 0
    failed_ratings = 0
    processed_count = 0
    start_time = time.time()
    rating_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    def progress_callback(index: int, item_id: str, success: bool, rating: Optional[int]):
        nonlocal successful_ratings, failed_ratings, processed_count
        processed_count += 1
        
        if success and rating is not None:
            successful_ratings += 1
            rating_distribution[rating] += 1
            print(f"✓ {processed_count}/{len(data)}: {item_id} (Rating: {rating}/5)")
        else:
            failed_ratings += 1
            print(f"✗ {processed_count}/{len(data)}: {item_id} (Rating failed)")
        
        # Show progress every 50 items
        if processed_count % 50 == 0 or processed_count in [10, 25, 100]:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = (len(data) - processed_count) / rate if rate > 0 else 0
            avg_rating = sum(r * count for r, count in rating_distribution.items()) / max(successful_ratings, 1)
            print(f"Progress: {processed_count}/{len(data)} ({processed_count/len(data)*100:.1f}%) "
                  f"Rate: {rate:.1f}/s ETA: {eta:.0f}s Avg Rating: {avg_rating:.2f}")
    
    # Process all ratings concurrently
    try:
        results = await client.rate_batch_responses(
            dataset=data,
            rating_model=rating_model,
            max_retries=max_retries,
            progress_callback=progress_callback
        )
        
        # Wait for all progress callbacks to complete
        while processed_count < len(data):
            await asyncio.sleep(0.1)
        
    except Exception as e:
        print(f"Error during batch rating: {e}")
        results = data  # Use original data with any partial results
    
    # Calculate final stats
    elapsed_time = time.time() - start_time
    average_rating = sum(r * count for r, count in rating_distribution.items()) / max(successful_ratings, 1)
    
    print(f"\nRating complete!")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Average rate: {len(data)/elapsed_time:.1f} ratings/second")
    print(f"Successful: {successful_ratings}/{len(data)}")
    print(f"Failed: {failed_ratings}/{len(data)}")
    print(f"Average rating: {average_rating:.2f}/5")
    
    print(f"\nRating Distribution:")
    for rating, count in rating_distribution.items():
        percentage = (count / successful_ratings * 100) if successful_ratings > 0 else 0
        print(f"  {rating}/5: {count} responses ({percentage:.1f}%)")
    
    # Save in both formats
    jsonl_output = f"{output_prefix}.jsonl"
    csv_output = f"{output_prefix}.csv"
    
    save_rated_dataset(results, jsonl_output, "jsonl")
    save_rated_dataset(results, csv_output, "csv")
    
    # Generate summary
    summary = {
        "input_file": input_file,
        "rating_model": rating_model,
        "total_items": len(data),
        "successful_ratings": successful_ratings,
        "failed_ratings": failed_ratings,
        "average_rating": average_rating,
        "rating_distribution": rating_distribution,
        "total_time_seconds": elapsed_time,
        "average_rate_per_second": len(data) / elapsed_time,
        "rating_params": {
            "max_concurrent": max_concurrent
        },
        "output_files": [jsonl_output, csv_output],
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = f"{output_prefix}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Rate responses in enhanced dataset using OpenRouter API (Async)")
    
    parser.add_argument(
        "--input", "-i",
        default="enhanced_dataset_meta_llama_llama_3.1_8b_instruct_20250811_140130.jsonl",
        help="Input enhanced dataset file"
    )
    
    parser.add_argument(
        "--rating-model", "-r",
        default="openai/gpt-oss-120b",
        help="Model to use for rating (default: openai/gpt-oss-120b)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file prefix (default: auto-generated based on input and timestamp)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=15,
        help="Maximum concurrent requests (default: 15)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per failed request (default: 3)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit processing to first N items (for testing)"
    )
    
    args = parser.parse_args()
    
    print("OpenRouter Response Rating Tool (Async)")
    print("=" * 50)
    print(f"Input file: {args.input}")
    print(f"Rating model: {args.rating_model}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Max retries: {args.max_retries}")
    if args.limit:
        print(f"Limit: {args.limit} items")
    print("=" * 50)
    
    # Run the async function
    asyncio.run(rate_dataset_responses(
        input_file=args.input,
        rating_model=args.rating_model,
        output_prefix=args.output,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        limit=args.limit
    ))

if __name__ == "__main__":
    main()
