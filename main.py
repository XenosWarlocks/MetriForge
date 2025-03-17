import google.generativeai as genai
import os
from datetime import datetime
import time
import json
import random
import string
import logging
import sys
from dataclasses import dataclass
import hashlib
import asyncio


@dataclass
class EmailConfig:
    """Configuration for authentic email warmup"""
    min_word_count: int = 0  # No minimum
    max_word_count: int = float('inf')  # No maximum
    word_count_tolerance: int = 0  # No tolerance
    min_emails: int = 1
    max_emails: int = 100
    rate_limit_delay: int = 60  # Increased delay for more natural timing


class EnhancedTestEmailMetadata:
    """Enhanced metadata generator for authentic emails"""

    def __init__(self):
        self.first_names = ['Alex', 'Sam', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Jamie']
        self.last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']
        self.roles = ['Platform Engineer', 'SRE', 'DevOps Engineer', 'Performance Analyst', 'Infrastructure Lead']
        self.departments = ['Platform', 'Infrastructure', 'SRE', 'DevOps', 'Performance']
        self.team_contexts = [
            "Weekly Performance Review",
            "System Health Check",
            "Infrastructure Update",
            "Platform Stability Report",
            "Service Reliability Metrics"
        ]
        self.used_subjects = set()

    def generate_authentic_email(self):
        """Generate authentic-looking email and name"""
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        role = random.choice(self.roles)
        department = random.choice(self.departments)

        email = f"{first_name.lower()}.{last_name.lower()}@testcompany.com"
        display_name = f"{first_name} {last_name} | {department} {role}"
        return email, display_name

    def generate_natural_subject(self, content_hint=None, max_attempts=10):
        """Generate more natural-sounding subject lines with retry limit
        
        Args:
            content_hint: Optional hint about the content to make subject more relevant
            max_attempts: Maximum number of attempts before using fallback method
        """
        attempts = 0
        
        # Base subject templates
        base_templates = [
            "Quick question about our team",
            "Thoughts on recent trends?",
            "Can we discuss performance metrics?",
            "Team input needed",
            "Looking for feedback on recent changes"
        ]
        
        while attempts < max_attempts:
            context = random.choice(self.team_contexts)
            
            # Create more natural subject lines
            subject_templates = [
                f"Quick question about {context}",
                f"Thoughts on recent trends?",
                f"Can we discuss {context}?",
                f"Team input needed: {context}",
                f"Looking for feedback on recent changes"
            ]
            
            # Add content-specific templates if hint is provided
            if content_hint:
                subject_templates.extend([
                    f"Follow-up on {content_hint}",
                    f"Updates regarding {content_hint}",
                    f"Review needed: {content_hint}"
                ])
            
            subject = random.choice(subject_templates)
            subject_hash = hashlib.md5(subject.encode()).hexdigest()
            
            if subject_hash not in self.used_subjects:
                self.used_subjects.add(subject_hash)
                return subject
            
            attempts += 1
        
        # Fallback: Add uniqueness with timestamp and random suffix
        timestamp = datetime.now().strftime("%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        
        # Ensure we have a valid template to use as fallback
        if not subject_templates:
            subject_templates = base_templates
        
        base_subject = random.choice(subject_templates)
        subject = f"{base_subject} ({timestamp}-{random_suffix})"
        
        subject_hash = hashlib.md5(subject.encode()).hexdigest()
        self.used_subjects.add(subject_hash)
        return subject


class SmartTestEmailGenerator:
    def __init__(self, api_key: str):
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.metadata = EnhancedTestEmailMetadata()
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini AI: {str(e)}")
            raise

        self.test_disclaimer = """
+==================================================+
|            AUTOMATED TEST EMAIL                   |
|            THIS IS A TEST EMAIL                   |
|            DO NOT REPLY OR TAKE ACTION            |
+==================================================+
Report ID: {test_id}
Generated: {timestamp}
This is an automated test email for testing purposes.
+==================================================+
"""

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('test_emails.log')
            ]
        )

    def generate_test_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        return f"TEST-{timestamp}-{random_suffix}"

    def generate_prompt(self, word_count: int, test_id: str) -> str:
        """Generate more conversational and authentic email prompts

        Args:
            word_count (int): Target word count for the email
            test_id (str): Test identifier
        """
        try:
            from_email, from_name = self.metadata.generate_authentic_email()
            to_email, to_name = self.metadata.generate_authentic_email()
            
            # Generate a content hint from team contexts and roles
            content_hint = random.choice(self.metadata.departments + self.metadata.roles)
            subject = self.metadata.generate_natural_subject(content_hint=content_hint)

            prompt_template = f"""Generate a natural, conversational email between team members with this context:
                        From: {from_name} <{from_email}>
                        To: {to_name} <{to_email}>
                        Subject: {subject}

                        Requirements:
                        - Write in a casual, professional tone
                        - Ask for specific input or feedback
                        - Include 1-2 specific questions
                        - Reference recent team context
                        - Focus on collaboration and team input
                        - End with a natural signature
                        - The email should be approximately {word_count} words in length

                        Note: Write as a regular email without any test disclaimers or headers"""

            return prompt_template.strip()
        except Exception as e:
            self.logger.error(f"Error generating prompt: {str(e)}")
            # Fallback to a simple prompt if subject generation fails
            fallback_subject = f"Test Email {test_id}"
            return f"""Generate a professional test email with subject: {fallback_subject}
                    The email should be approximately {word_count} words in length."""

    def verify_word_count(self, content: str, word_count: int) -> bool:
        """Verify the content meets word count requirements with tolerance"""
        return True  # Always return True

    async def generate_single_email(self, word_count: int, email_number: int) -> dict:
        test_id = self.generate_test_id()

        try:
            # Generate the full prompt including subject line
            prompt = self.generate_prompt(word_count, test_id)
            
            # Extract the subject line from the prompt for later use
            subject_line = None
            for line in prompt.split('\n'):
                if line.strip().startswith('Subject:'):
                    subject_line = line.strip()[len('Subject:'):].strip()
                    break
            
            response = await self.model.generate_content_async(prompt)
            body_content = response.text
            words = len(body_content.split())

            if not self.verify_word_count(body_content, word_count):
                lower_bound = word_count - EmailConfig.word_count_tolerance
                upper_bound = word_count + EmailConfig.word_count_tolerance

                raise ValueError(
                    f"Generated content does not meet word count requirements ({lower_bound}-{upper_bound} words). Current word count: {words}")
            else:
                self.logger.info(f"Successfully generated email {email_number} with {words} words.")

            # Construct the full email with headers and body
            full_email = f"Subject: {subject_line}\n\n{body_content}"

            return {
                "email_number": email_number,
                "test_id": test_id,
                "subject": subject_line,  # Store subject separately for reference
                "content": full_email,    # Store the complete email with subject
                "body": body_content,     # Store just the body for reference
                "timestamp": datetime.now().isoformat(),
                "status": "success"
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
                                     output_dir: str = "test_emails") -> list:
        self.logger.info(f"Starting bulk generation of {num_emails} test emails")
        os.makedirs(output_dir, exist_ok=True)
        results = []

        batch_id = f"EMAIL-BATCH-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        for i in range(1, num_emails + 1):
            self.logger.info(f"Generating test email {i}/{num_emails}")

            result = await self.generate_single_email(word_count, i)
            results.append(result)

            if result["status"] == "success":
                filename = f"test_email_{result['test_id']}.txt"
                file_path = os.path.join(output_dir, filename)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(result["content"])

                self.logger.info(f"Saved email to {file_path}")

            if i < num_emails:
                time.sleep(EmailConfig.rate_limit_delay)

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
                "average_word_count": sum(len(r["content"].split()) if r["content"] else 0
                                          for r in results) / max(len(results), 1)
            },
            "results": results
        }

        report_path = os.path.join(output_dir, f"generation_report_{batch_id}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Generation complete. Report saved to {report_path}")
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate smart test emails using Gemini AI")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--num-emails", type=int, required=True,
                      help="Number of test emails to generate")
    parser.add_argument("--word-count", type=int, default=200,
                      help="Word count limit for each email")
    parser.add_argument("--output-dir", default="test_emails",
                      help="Output directory for test emails")
    parser.add_argument("--verbose", action='store_true',
                      help="Enable detailed logging output")
    parser.add_argument("--word-count-tolerance", type=int, default=20,
                      help="Tolerance for word count (allows for +/- this many words).")

    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    # Update EmailConfig with word count tolerance
    EmailConfig.word_count_tolerance = 0  # Set to 0
    EmailConfig.min_word_count = 0  # Set to 0
    EmailConfig.max_word_count = float('inf')  # Set to infinity

    # Validate word count
    lower_bound = args.word_count - EmailConfig.word_count_tolerance
    upper_bound = args.word_count + EmailConfig.word_count_tolerance
    if not (EmailConfig.min_word_count <= args.word_count <= EmailConfig.max_word_count):
        print(
            f"Warning: Target word count ({args.word_count}) is outside the recommended range ({EmailConfig.min_word_count}-{EmailConfig.max_word_count}). Effective range with tolerance: {lower_bound}-{upper_bound}")

    # Validate number of emails
    if not (EmailConfig.min_emails <= args.num_emails <= EmailConfig.max_emails):
        print(f"Error: Number of emails must be between {EmailConfig.min_emails} and "
              f"{EmailConfig.max_emails}")
        return

    print("\n=== Smart Test Email Generator ===")
    print(f"Configuration:")
    print(f"- Output Directory: {args.output_dir}")
    print(f"- Emails to Generate: {args.num_emails}")
    print(f"- Target Words per Email: {args.word_count}")
    print(f"- Word Count Tolerance: +/- {args.word_count_tolerance}")
    print("===================================")

    try:
        # Initialize generator with configurationss
        generator = SmartTestEmailGenerator(args.api_key)

        # Run the generation
        asyncio.run(generator.generate_bulk_emails(
            args.word_count,
            args.num_emails,
            args.output_dir
        ))

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during email generation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
