import os
from datetime import datetime
import time
import json
import random
import string
import logging
import sys
import argparse
from dataclasses import dataclass
import asyncio
from typing import List, Dict, Optional
import google.generativeai as genai


@dataclass
class EmailConfig:
    """Configuration for test email generation"""
    min_word_count: int = 100
    max_word_count: int = 500
    min_emails: int = 1
    max_emails: int = 100
    base_delay: float = 1.0


class RateLimiter:
    """Smart rate limiter with adaptive backoff strategy"""
    
    def __init__(self, base_delay: float = 1.0):
        self.base_delay = base_delay
        self.consecutive_failures = 0
        self.failure_timestamps = []
        self.last_request_time = 0
        self.logger = logging.getLogger(__name__)
    
    async def wait(self):
        """Wait according to smart rate limiting strategy"""
        # Calculate dynamic delay based on recent failures
        now = time.time()
        
        # Basic delay between all requests
        delay = self.base_delay
        
        # Clear old failure timestamps (older than 60 seconds)
        self.failure_timestamps = [t for t in self.failure_timestamps if now - t < 60]
        
        # Exponential backoff for consecutive failures
        if self.consecutive_failures > 0:
            delay *= (2 ** min(self.consecutive_failures, 6))  # Cap at 64x base delay
        
        # Additional delay based on failure density in the last minute
        failure_count = len(self.failure_timestamps)
        if failure_count > 3:
            delay += min(failure_count, 10)  # Add up to 10 seconds for heavy failure rates
        
        # Ensure minimum time between requests
        time_since_last = now - self.last_request_time if self.last_request_time else delay
        if time_since_last < delay:
            additional_wait = delay - time_since_last
            self.logger.debug(f"Rate limiting: waiting {additional_wait:.2f}s")
            await asyncio.sleep(additional_wait)
        
        self.last_request_time = time.time()
    
    def record_success(self):
        """Record a successful API call"""
        self.consecutive_failures = 0
    
    def record_failure(self):
        """Record a failed API call"""
        self.consecutive_failures += 1
        self.failure_timestamps.append(time.time())
        self.logger.warning(f"API call failed. Consecutive failures: {self.consecutive_failures}")


class TestEmailMetadata:
    """Generates test email metadata"""
    
    def __init__(self):
        self.domains = ['testcompany.com', 'testing.org', 'qamail.net', 'unittest.io']
        self.departments = ['engineering', 'qa', 'devops', 'support', 'performance']
        self.used_subjects = set()
    
    def generate_email(self) -> str:
        """Generate a random test email address"""
        name = ''.join(random.choices(string.ascii_lowercase, k=8))
        department = random.choice(self.departments)
        domain = random.choice(self.domains)
        return f"{name}.{department}@{domain}"
    
    def generate_unique_subject(self, test_id: str) -> str:
        """Generate a unique subject line for test emails"""
        templates = [
            "[TEST] Automated Test Email - {test_id}",
            "[TEST] QA Email Generation - {test_id}",
            "[TEST] System Notification Test - {test_id}",
            "[TEST] Automated Message - {test_id}",
            "[TEST] Test Email System - {test_id}"
        ]
        
        subject = random.choice(templates).format(test_id=test_id)
        
        # Ensure uniqueness by adding a random suffix if needed
        if subject in self.used_subjects:
            subject = f"{subject} ({random.randint(1000, 9999)})"
        
        self.used_subjects.add(subject)
        return subject


class EmailGenerator:
    """Generates test emails using Google Generative AI"""
    
    def __init__(self, api_key: str):
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.metadata = TestEmailMetadata()
        self.rate_limiter = RateLimiter()
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
            self.logger.info("Successfully initialized Gemini 2.0 Flash model")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini AI: {str(e)}")
            raise
        
        self.test_disclaimer = """
+==================================================+
|            AUTOMATED TEST EMAIL                  |
|            DO NOT REPLY OR TAKE ACTION           |
+==================================================+
Test ID: {test_id}
Generated: {timestamp}
This is an automated test email. The content is generated 
for testing purposes only.
+==================================================+
"""

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('test_emails.log')
            ]
        )
    
    def generate_test_id(self) -> str:
        """Generate a unique test ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        return f"TEST-{timestamp}-{random_suffix}"
    
    def generate_prompt(self, word_count: int, test_id: str) -> str:
        """Generate a prompt for email content creation"""
        from_email = self.metadata.generate_email()
        to_email = self.metadata.generate_email()
        timestamp = datetime.now()
        subject = self.metadata.generate_unique_subject(test_id)
        
        prompt = f"""Generate a test email with this structure:
                    From: {from_email}
                    To: {to_email}
                    Date: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}
                    Subject: {subject}

                    {self.test_disclaimer.format(
                        test_id=test_id,
                        timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    )}

                    Requirements:
                    - Approximately {word_count} words
                    - Generate generic business email content
                    - Include a professional signature
                    - The email should appear realistic but clearly labeled as a test
                    
                    End with a professional signature including:
                    - Name
                    - Title
                    - Department
                    - Test environment details"""
        
        return prompt
    
    async def call_api(self, prompt: str) -> Optional[str]:
        """Call the Gemini AI API to generate content"""
        await self.rate_limiter.wait()
        
        try:
            self.logger.debug(f"Calling Gemini API with prompt of length {len(prompt)}")
            response = await self.model.generate_content_async(prompt)
            self.rate_limiter.record_success()
            return response.text
        except Exception as e:
            self.rate_limiter.record_failure()
            self.logger.error(f"Exception during API call: {str(e)}")
            
            # Extract error message to check for rate limiting
            error_message = str(e).lower()
            if "rate" in error_message and "limit" in error_message:
                self.logger.warning("Rate limit detected, adding extra delay")
                await asyncio.sleep(5)
            
            return None
    
    async def generate_single_email(self, word_count: int, email_number: int) -> Dict:
        """Generate a single test email"""
        test_id = self.generate_test_id()
        
        try:
            prompt = self.generate_prompt(word_count, test_id)
            content = await self.call_api(prompt)
            
            if content:
                return {
                    "email_number": email_number,
                    "test_id": test_id,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "word_count": len(content.split())
                }
            else:
                return {
                    "email_number": email_number,
                    "test_id": test_id,
                    "content": None,
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "error": "API returned no content"
                }
        except Exception as e:
            self.logger.error(f"Failed to generate test email #{email_number}: {str(e)}")
            return {
                "email_number": email_number,
                "test_id": test_id,
                "content": None,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            }
    
    async def generate_bulk_emails(self, word_count: int, num_emails: int,
                                  output_dir: str = "test_emails") -> List[Dict]:
        """Generate multiple test emails in bulk"""
        self.logger.info(f"Starting bulk generation of {num_emails} test emails")
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        batch_id = f"BATCH-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        try:
            for i in range(1, num_emails + 1):
                self.logger.info(f"Generating email {i}/{num_emails}")
                result = await self.generate_single_email(word_count, i)
                results.append(result)
                
                if result["status"] == "success":
                    filename = f"email_{result['test_id']}.txt"
                    file_path = os.path.join(output_dir, filename)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(result["content"])
                    
                    self.logger.info(f"Saved email to {file_path}")
                else:
                    self.logger.warning(f"Failed to generate email: {result.get('error', 'Unknown error')}")
            
            # Generate batch report
            report = {
                "batch_id": batch_id,
                "generation_time": datetime.now().isoformat(),
                "configuration": {
                    "word_count": word_count,
                    "num_emails": num_emails,
                    "output_directory": output_dir
                },
                "statistics": {
                    "successful_generations": sum(1 for r in results if r["status"] == "success"),
                    "failed_generations": sum(1 for r in results if r["status"] == "failed"),
                    "average_word_count": sum(r.get("word_count", 0) for r in results if r["status"] == "success") / 
                                        max(1, sum(1 for r in results if r["status"] == "success"))
                },
                "results": results
            }
            
            report_path = os.path.join(output_dir, f"generation_report_{batch_id}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Generation complete. Report saved to {report_path}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error during bulk generation: {str(e)}")
            raise


async def main_async(args):
    """Async main function"""
    # Validate word count
    if not (EmailConfig.min_word_count <= args.word_count <= EmailConfig.max_word_count):
        print(f"Error: Word count must be between {EmailConfig.min_word_count} and "
              f"{EmailConfig.max_word_count}")
        return
    
    # Validate number of emails
    if not (EmailConfig.min_emails <= args.num_emails <= EmailConfig.max_emails):
        print(f"Error: Number of emails must be between {EmailConfig.min_emails} and "
              f"{EmailConfig.max_emails}")
        return
    
    print("\n=== Test Email Generator ===")
    print(f"Configuration:")
    print(f"- Output Directory: {args.output_dir}")
    print(f"- Emails to Generate: {args.num_emails}")
    print(f"- Words per Email: {args.word_count}")
    print("===================================")
    
    try:
        generator = EmailGenerator(args.api_key)
        
        results = await generator.generate_bulk_emails(
            args.word_count,
            args.num_emails,
            args.output_dir
        )
        
        # Generate summary
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        
        print("\n=== Generation Summary ===")
        print(f"Successfully generated: {successful}/{args.num_emails} emails")
        print(f"Failed generations: {failed}")
        print(f"Output directory: {args.output_dir}")
        
        if successful > 0:
            avg_words = sum(r.get("word_count", 0) for r in results if r["status"] == "success") / successful
            print(f"Average word count: {avg_words:.1f}")
        
        print("\nDetailed reports have been saved to the output directory.")
        print("=======================")
    
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during email generation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description="Generate test emails using Gemini AI")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--word-count", type=int, default=150,
                      help="Word count for each email")
    parser.add_argument("--num-emails", type=int, required=True,
                      help="Number of emails to generate")
    parser.add_argument("--output-dir", default="test_emails",
                      help="Output directory for generated emails")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    # Run the async main function
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

# python main.py --api-key AIzaSyDV4c47DjkrLYw7aCGSpao3L3Wkykw2JDk --num-emails 5 --word-count 150