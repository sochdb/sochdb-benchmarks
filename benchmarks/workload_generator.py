"""
Workload Generator for Memory Benchmark
========================================

Generates synthetic and real-world datasets for benchmarking:

1. Synthetic Production Workload
   - 10k users × 10 sessions × 200 messages
   - 100k-1M knowledge snippets
   - Mix of short/long facts + updates

2. LoCoMo-style Conversational Dataset
   - Long-term conversational memory
   - QA + event summarization tasks
"""

import random
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from memory_benchmark_harness import Message, Document


# =============================================================================
# Synthetic Conversation Templates
# =============================================================================

class ConversationTemplates:
    """Templates for realistic agent conversations"""

    CUSTOMER_SUPPORT = [
        ("user", "I need help with {issue}"),
        ("assistant", "I can help with that. Can you provide more details about {issue}?"),
        ("user", "{details}"),
        ("assistant", "Thank you. Let me {action} for you."),
        ("user", "{followup}"),
        ("assistant", "{resolution}"),
    ]

    PRODUCT_INQUIRY = [
        ("user", "Tell me about {product}"),
        ("assistant", "{product} is {description}. Would you like to know more?"),
        ("user", "What about {feature}?"),
        ("assistant", "{feature} allows you to {capability}."),
        ("user", "How much does it cost?"),
        ("assistant", "Pricing starts at {price}. {discount_info}"),
    ]

    TECHNICAL_SUPPORT = [
        ("user", "I'm getting error {error_code}"),
        ("assistant", "Let's troubleshoot {error_code}. What were you doing when it occurred?"),
        ("user", "{action}"),
        ("assistant", "Try {fix_step_1}."),
        ("user", "{result_1}"),
        ("assistant", "Good. Now {fix_step_2}."),
        ("user", "It worked!"),
        ("assistant", "Excellent! Is there anything else?"),
    ]

    PREFERENCES = [
        ("user", "I prefer {preference_type} to be {value}"),
        ("assistant", "I've noted your preference for {preference_type}: {value}"),
        ("user", "Also, I don't like {dislike}"),
        ("assistant", "Got it. I'll remember you don't like {dislike}."),
    ]


class FactGenerator:
    """Generate knowledge facts and documents"""

    FACT_TEMPLATES = [
        "The {company} {product} is designed for {use_case}.",
        "{product} supports {feature_list} out of the box.",
        "Pricing for {tier} tier starts at ${price}/month.",
        "{product} integrates with {platform_list}.",
        "Our {compliance} certification was updated on {date}.",
        "{feature} was deprecated in version {version}.",
        "The maximum {limit_type} is {limit_value}.",
        "{api_endpoint} accepts {http_method} requests with {params}.",
    ]

    VARIABLES = {
        "company": ["Acme Corp", "TechStart", "GlobalSoft", "CloudBase"],
        "product": ["Dashboard", "API Gateway", "Analytics Platform", "Data Pipeline"],
        "use_case": ["enterprise analytics", "real-time monitoring", "data integration"],
        "feature_list": ["SSO, SCIM, webhooks", "batch processing, streaming", "SQL, NoSQL, vector search"],
        "tier": ["Starter", "Pro", "Enterprise", "Custom"],
        "price": ["29", "99", "299", "999"],
        "platform_list": ["Salesforce, HubSpot", "Slack, Teams", "AWS, Azure, GCP"],
        "compliance": ["SOC 2 Type II", "GDPR", "HIPAA", "ISO 27001"],
        "date": ["2025-01-15", "2025-06-30", "2024-12-01"],
        "feature": ["Legacy API v1", "XML export", "FTP integration"],
        "version": ["3.0", "4.5", "5.0"],
        "limit_type": ["API rate limit", "file size", "user count"],
        "limit_value": ["1000 req/min", "100MB", "unlimited"],
        "api_endpoint": ["/api/v1/search", "/api/v2/ingest", "/api/v1/query"],
        "http_method": ["POST", "GET", "PUT"],
        "params": ["JSON body", "query params", "multipart form"],
    }

    @classmethod
    def generate_fact(cls) -> str:
        """Generate a random fact"""
        template = random.choice(cls.FACT_TEMPLATES)

        # Fill in variables
        fact = template
        for var, options in cls.VARIABLES.items():
            if "{" + var + "}" in fact:
                fact = fact.replace("{" + var + "}", random.choice(options))

        return fact


# =============================================================================
# Workload Generators
# =============================================================================

class SyntheticWorkloadGenerator:
    """Generate synthetic production workload"""

    @staticmethod
    def generate_conversation(
        user_id: str,
        session_id: str,
        num_messages: int = 20
    ) -> List[Message]:
        """Generate a synthetic conversation"""
        messages = []
        timestamp = time.time() - (num_messages * 60)  # Start N minutes ago

        # Pick conversation type
        template = random.choice([
            ConversationTemplates.CUSTOMER_SUPPORT,
            ConversationTemplates.PRODUCT_INQUIRY,
            ConversationTemplates.TECHNICAL_SUPPORT,
            ConversationTemplates.PREFERENCES,
        ])

        # Generate messages from template (repeat if needed)
        for i in range(num_messages):
            role, content_template = template[i % len(template)]

            # Simple variable filling
            content = content_template.format(
                issue="login failure",
                details="I can't access my account since yesterday",
                action="trying to upload a large file",
                followup="Where can I find the reset link?",
                resolution="I've sent it to your email",
                product="Enterprise Dashboard",
                description="a comprehensive analytics solution",
                feature="custom reports",
                capability="create scheduled reports with custom filters",
                price="$299",
                discount_info="Volume discounts available",
                error_code="ERR_TIMEOUT",
                fix_step_1="increase the timeout setting",
                result_1="still seeing the error",
                fix_step_2="try splitting the file into smaller chunks",
                preference_type="email notifications",
                value="daily summaries only",
                dislike="marketing emails",
            )

            messages.append(Message(
                role=role,
                content=content,
                timestamp=timestamp,
                metadata={"user_id": user_id, "session_id": session_id}
            ))

            timestamp += random.randint(30, 300)  # 30s - 5min between messages

        return messages

    @staticmethod
    def generate_users_and_sessions(
        num_users: int = 100,
        sessions_per_user: int = 10,
        messages_per_session: int = 20
    ) -> List[Tuple[str, str, List[Message]]]:
        """
        Generate full user dataset.

        Returns:
            List of (user_id, session_id, messages)
        """
        dataset = []

        for user_idx in range(num_users):
            user_id = f"user_{user_idx:05d}"

            for session_idx in range(sessions_per_user):
                session_id = f"session_{user_idx:05d}_{session_idx:03d}"

                messages = SyntheticWorkloadGenerator.generate_conversation(
                    user_id, session_id, messages_per_session
                )

                dataset.append((user_id, session_id, messages))

        return dataset

    @staticmethod
    def generate_knowledge_docs(
        tenant_id: str,
        num_docs: int = 10000
    ) -> List[Document]:
        """Generate knowledge documents"""
        docs = []

        for i in range(num_docs):
            content = FactGenerator.generate_fact()

            docs.append(Document(
                content=content,
                metadata={
                    "tenant_id": tenant_id,
                    "doc_id": f"doc_{i:06d}",
                    "category": random.choice(["product", "pricing", "technical", "compliance"]),
                    "created": time.time() - random.randint(0, 365 * 24 * 3600)
                }
            ))

        return docs


class LoCoMoDatasetLoader:
    """
    Loader for LoCoMo-style conversational memory benchmark.

    Note: This is a placeholder. Real LoCoMo dataset would need to be
    downloaded and parsed from the actual benchmark.
    """

    @staticmethod
    def load_dataset() -> List[Dict[str, Any]]:
        """
        Load LoCoMo dataset.

        Returns:
            List of conversation dicts with QA tasks
        """
        # Placeholder - would load real LoCoMo data
        # For now, return synthetic conversations with QA
        conversations = []

        for i in range(100):  # 100 test conversations
            messages = SyntheticWorkloadGenerator.generate_conversation(
                f"locomo_user_{i}",
                f"locomo_session_{i}",
                num_messages=50  # Longer conversations
            )

            # Generate QA pairs
            qa_pairs = [
                {
                    "question": "What issue did the user report?",
                    "answer": "login failure",
                    "turn": 0
                },
                {
                    "question": "What was the resolution?",
                    "answer": "password reset link sent to email",
                    "turn": 5
                },
            ]

            conversations.append({
                "conversation_id": f"locomo_{i}",
                "messages": messages,
                "qa_pairs": qa_pairs
            })

        return conversations


# =============================================================================
# Query Generator
# =============================================================================

class QueryGenerator:
    """Generate test queries for context retrieval"""

    QUERY_TEMPLATES = [
        "What did the user say about {topic}?",
        "When did {event} happen?",
        "What is the {product} pricing?",
        "How do I {action}?",
        "What are the {feature} details?",
        "Show me information about {subject}",
        "Summarize the conversation about {topic}",
    ]

    TOPICS = [
        "login issues", "password reset", "pricing", "enterprise features",
        "API integration", "SSO setup", "billing", "support",
    ]

    @classmethod
    def generate_query(cls) -> str:
        """Generate a random query"""
        template = random.choice(cls.QUERY_TEMPLATES)
        return template.format(
            topic=random.choice(cls.TOPICS),
            event="the last login",
            product="Enterprise",
            action="reset my password",
            feature="SSO",
            subject="API limits"
        )

    @classmethod
    def generate_queries(cls, num_queries: int = 100) -> List[str]:
        """Generate multiple queries"""
        return [cls.generate_query() for _ in range(num_queries)]
