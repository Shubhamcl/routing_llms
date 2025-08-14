#!/usr/bin/env python3
"""
Generate responses for the dataset using OpenRouter API.
Reads prompts from prompts_2000.jsonl and generates responses using specified model.
Saves the enhanced dataset with model responses.
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

# Load environment variables
load_dotenv()

class AsyncOpenRouterClient:
    """Async client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: str, max_concurrent: int = 10):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Shubhamcl/smart_routing",
            "X-Title": "Smart Routing Evaluation"
        }
        self.semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests
    
    async def generate_response(self, session: aiohttp.ClientSession, prompt: str, 
                              model: str, max_tokens: int = 1000, 
                              temperature: float = 0.7) -> Optional[str]:
        """Generate a response using the specified model."""
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
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
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        print(f"API Error {response.status}: {error_text}")
                        return None
                        
            except asyncio.TimeoutError:
                print(f"Request timeout for prompt")
                return None
            except Exception as e:
                print(f"Request failed: {e}")
                return None
    
    async def generate_batch_responses(self, prompts_data: List[Dict], model: str,
                                     max_tokens: int = 1000, temperature: float = 0.7,
                                     progress_callback=None) -> List[Dict]:
        """Generate responses for a batch of prompts concurrently."""
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i, item in enumerate(prompts_data):
                task = self._process_single_item(
                    session, item, model, max_tokens, temperature, i, progress_callback
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return successful results
            successful_results = []
            for result in results:
                if isinstance(result, Exception):
                    print(f"Task failed with exception: {result}")
                else:
                    successful_results.append(result)
            
            return successful_results
    
    async def _process_single_item(self, session: aiohttp.ClientSession, item: Dict,
                                 model: str, max_tokens: int, temperature: float,
                                 index: int, progress_callback=None) -> Dict:
        """Process a single item (prompt + response generation)."""
        
        prompt = item['prompt']
        item_id = item.get('id', f'item_{index}')
        
        # Generate response
        response = await self.generate_response(
            session=session,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Update item with response
        if response:
            item['model_response'] = response
            item['model_name'] = model
            item['generation_timestamp'] = datetime.now().isoformat()
            item['generation_params'] = {
                'max_tokens': max_tokens,
                'temperature': temperature
            }
            if progress_callback:
                progress_callback(index, item_id, True, len(response))
        else:
            item['model_response'] = "[ERROR: Failed to generate response]"
            item['model_name'] = model
            item['generation_timestamp'] = datetime.now().isoformat()
            item['generation_error'] = True
            if progress_callback:
                progress_callback(index, item_id, False, 0)
        
        return item

def load_dataset(file_path: str) -> List[Dict]:
    """Load the dataset from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_enhanced_dataset(data: List[Dict], output_file: str, format_type: str = "jsonl"):
    """Save the enhanced dataset with model responses."""
    
    if format_type == "jsonl":
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    elif format_type == "csv":
        if data:
            fieldnames = list(data[0].keys())
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
    
    print(f"Enhanced dataset saved to: {output_file}")

async def generate_responses_for_dataset(
    input_file: str = "prompts_2000.jsonl",
    model: str = "meta-llama/llama-3.1-8b-instruct",
    output_prefix: str = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    max_concurrent: int = 10,
    limit: Optional[int] = None
):
    """Generate responses for all prompts in the dataset using async requests."""
    
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
    
    print(f"Loading dataset from {input_file}...")
    data = load_dataset(input_file)
    
    if limit:
        data = data[:limit]
        print(f"Processing limited to first {limit} prompts")
    
    print(f"Loaded {len(data)} prompts")
    
    # Initialize async client
    client = AsyncOpenRouterClient(api_key, max_concurrent=max_concurrent)
    
    # Generate output filename
    if output_prefix is None:
        model_name = model.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"enhanced_dataset_{model_name}_{timestamp}"
    
    # Progress tracking
    print(f"Generating responses using model: {model}")
    print(f"Max concurrent requests: {max_concurrent}")
    print("-" * 60)
    
    successful_responses = 0
    failed_responses = 0
    processed_count = 0
    start_time = time.time()
    
    def progress_callback(index: int, item_id: str, success: bool, response_length: int):
        nonlocal successful_responses, failed_responses, processed_count
        processed_count += 1
        
        if success:
            successful_responses += 1
            print(f"✓ {processed_count}/{len(data)}: {item_id} ({response_length} chars)")
        else:
            failed_responses += 1
            print(f"✗ {processed_count}/{len(data)}: {item_id} (failed)")
        
        # Show progress every 50 items or at milestones
        if processed_count % 50 == 0 or processed_count in [10, 25, 100, 500]:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = (len(data) - processed_count) / rate if rate > 0 else 0
            print(f"Progress: {processed_count}/{len(data)} ({processed_count/len(data)*100:.1f}%) "
                  f"Rate: {rate:.1f}/s ETA: {eta:.0f}s")
    
    # Process all prompts concurrently
    try:
        results = await client.generate_batch_responses(
            prompts_data=data,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            progress_callback=progress_callback
        )
        
        # Wait for all progress callbacks to complete
        while processed_count < len(data):
            await asyncio.sleep(0.1)
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        results = data  # Use original data with any partial results
    
    # Calculate final stats
    elapsed_time = time.time() - start_time
    
    print(f"\nGeneration complete!")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Average rate: {len(data)/elapsed_time:.1f} prompts/second")
    print(f"Successful: {successful_responses}/{len(data)}")
    print(f"Failed: {failed_responses}/{len(data)}")
    
    # Save in both formats
    jsonl_output = f"{output_prefix}.jsonl"
    csv_output = f"{output_prefix}.csv"
    
    save_enhanced_dataset(results, jsonl_output, "jsonl")
    save_enhanced_dataset(results, csv_output, "csv")
    
    # Generate summary
    summary = {
        "input_file": input_file,
        "model": model,
        "total_prompts": len(data),
        "successful_responses": successful_responses,
        "failed_responses": failed_responses,
        "total_time_seconds": elapsed_time,
        "average_rate_per_second": len(data) / elapsed_time,
        "generation_params": {
            "max_tokens": max_tokens,
            "temperature": temperature,
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
    parser = argparse.ArgumentParser(description="Generate responses for dataset using OpenRouter API (Async)")
    
    parser.add_argument(
        "--input", "-i",
        default="prompts_2000.jsonl",
        help="Input dataset file (default: prompts_2000.jsonl)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="meta-llama/llama-3.1-8b-instruct",
        help="Model to use for generation (default: meta-llama/llama-3.1-8b-instruct)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file prefix (default: auto-generated based on model and timestamp)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate (default: 1000)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10, increase for faster processing)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit processing to first N prompts (for testing)"
    )
    
    args = parser.parse_args()
    
    print("OpenRouter Dataset Response Generator (Async)")
    print("=" * 50)
    print(f"Input file: {args.input}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Max concurrent: {args.max_concurrent}")
    if args.limit:
        print(f"Limit: {args.limit} prompts")
    print("=" * 50)
    
    # Run the async function
    asyncio.run(generate_responses_for_dataset(
        input_file=args.input,
        model=args.model,
        output_prefix=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        limit=args.limit
    ))

if __name__ == "__main__":
    main()
