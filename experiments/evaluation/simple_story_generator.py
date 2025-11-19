"""
Simple Story Generator - Tractable Task for 410M Models

Generates SHORT stories (500-600 tokens) with SIMPLE factual questions.
Designed to actually be answerable by small models.

Key constraints:
- Story length: 500-600 tokens (leaves room for instructions + generation)
- Simple facts: One clear answer per question
- Short answers: Single words or short phrases ("Berlin", "Alice", "red")
- No ambiguity: Facts are explicitly stated
- Minimal distractors: Only if model can handle base task

Author: Halcyon AI Research
Date: 2025-11-18
"""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class SimpleFact:
    """A simple factual statement with question and answer."""
    fact_type: str  # "person", "place", "object", "date", "action"
    subject: str    # "Alice"
    predicate: str  # "traveled to"
    value: str      # "Berlin"
    question: str   # "Where did Alice travel?"
    answer: str     # "Berlin"


@dataclass
class SimpleStory:
    """A short story with planted facts."""
    text: str
    facts: List[SimpleFact]
    questions: List[Dict[str, str]]
    length_tokens: int


class SimpleStoryGenerator:
    """Generate simple, answerable stories for small models."""

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)

        # Simple name pool
        self.names = [
            "Alice", "Bob", "Carol", "David", "Emma",
            "Frank", "Grace", "Henry", "Isabel", "Jack"
        ]

        # Simple locations
        self.places = [
            "Paris", "London", "Berlin", "Rome", "Tokyo",
            "Madrid", "Athens", "Vienna", "Prague", "Oslo"
        ]

        # Simple objects
        self.objects = [
            "book", "key", "map", "letter", "photo",
            "coin", "watch", "ring", "bag", "phone"
        ]

        # Simple colors
        self.colors = [
            "red", "blue", "green", "yellow", "black",
            "white", "silver", "golden", "purple", "orange"
        ]

        # Simple dates
        self.dates = [
            "Monday", "Tuesday", "Wednesday", "March 15th", "April 3rd",
            "May 21st", "June 8th", "morning", "afternoon", "evening"
        ]

    def generate_story(
        self,
        num_facts: int = 3,
        target_length: int = 500
    ) -> SimpleStory:
        """
        Generate a simple story with planted facts.

        Args:
            num_facts: Number of facts to plant (3-5)
            target_length: Target token count (~500)

        Returns:
            SimpleStory with text and questions
        """
        # Choose main character
        main_char = random.choice(self.names)

        # Generate facts
        facts = self._generate_facts(main_char, num_facts)

        # Build story
        story_text = self._build_story(main_char, facts, target_length)

        # Generate questions
        questions = self._generate_questions(facts)

        # Estimate tokens (rough: words * 1.3)
        length_tokens = int(len(story_text.split()) * 1.3)

        return SimpleStory(
            text=story_text,
            facts=facts,
            questions=questions,
            length_tokens=length_tokens
        )

    def _generate_facts(self, main_char: str, num_facts: int) -> List[SimpleFact]:
        """Generate simple, unambiguous facts."""
        facts = []
        used_types = set()

        # Ensure variety
        fact_generators = [
            self._fact_travel,
            self._fact_found,
            self._fact_met,
            self._fact_when,
            self._fact_color
        ]

        random.shuffle(fact_generators)

        for generator in fact_generators[:num_facts]:
            fact = generator(main_char)
            facts.append(fact)
            used_types.add(fact.fact_type)

        return facts

    def _fact_travel(self, subject: str) -> SimpleFact:
        """Generate travel fact: 'Alice traveled to Berlin'"""
        place = random.choice(self.places)
        return SimpleFact(
            fact_type="travel",
            subject=subject,
            predicate="traveled to",
            value=place,
            question=f"Where did {subject} travel?",
            answer=place
        )

    def _fact_found(self, subject: str) -> SimpleFact:
        """Generate found object fact: 'Alice found a key'"""
        obj = random.choice(self.objects)
        return SimpleFact(
            fact_type="found",
            subject=subject,
            predicate="found",
            value=obj,
            question=f"What did {subject} find?",
            answer=obj
        )

    def _fact_met(self, subject: str) -> SimpleFact:
        """Generate meeting fact: 'Alice met Bob'"""
        other = random.choice([n for n in self.names if n != subject])
        return SimpleFact(
            fact_type="met",
            subject=subject,
            predicate="met",
            value=other,
            question=f"Who did {subject} meet?",
            answer=other
        )

    def _fact_when(self, subject: str) -> SimpleFact:
        """Generate time fact: 'This happened on Monday'"""
        when = random.choice(self.dates)
        return SimpleFact(
            fact_type="when",
            subject="event",
            predicate="happened",
            value=when,
            question="When did this happen?",
            answer=when
        )

    def _fact_color(self, subject: str) -> SimpleFact:
        """Generate color fact: 'The bag was red'"""
        obj = random.choice(self.objects)
        color = random.choice(self.colors)
        return SimpleFact(
            fact_type="color",
            subject=obj,
            predicate="was",
            value=color,
            question=f"What color was the {obj}?",
            answer=color
        )

    def _build_story(
        self,
        main_char: str,
        facts: List[SimpleFact],
        target_length: int
    ) -> str:
        """Build a coherent story embedding the facts."""
        parts = []

        # Opening (50-80 words)
        parts.append(f"This is a story about {main_char}. ")
        parts.append(f"{main_char} was an ordinary person living an ordinary life. ")
        parts.append("One day, something interesting happened. ")

        # Embed each fact with context (80-100 words per fact)
        for i, fact in enumerate(facts):
            # Add transition
            transitions = [
                "Then, ",
                "After that, ",
                "Next, ",
                "Later, ",
                "Subsequently, "
            ]
            if i > 0:
                parts.append(random.choice(transitions))

            # State the fact CLEARLY
            if fact.fact_type == "travel":
                parts.append(f"{fact.subject} traveled to {fact.value}. ")
                parts.append(f"The journey to {fact.value} was memorable. ")
                parts.append(f"{fact.value} was a beautiful place. ")

            elif fact.fact_type == "found":
                parts.append(f"{fact.subject} found a {fact.value}. ")
                parts.append(f"The {fact.value} was unexpected. ")
                parts.append(f"{fact.subject} picked up the {fact.value}. ")

            elif fact.fact_type == "met":
                parts.append(f"{fact.subject} met {fact.value}. ")
                parts.append(f"{fact.value} was friendly. ")
                parts.append(f"{fact.subject} and {fact.value} talked for a while. ")

            elif fact.fact_type == "when":
                parts.append(f"This all happened on {fact.value}. ")
                parts.append(f"{fact.value} was a significant time. ")
                parts.append(f"Nobody would forget {fact.value}. ")

            elif fact.fact_type == "color":
                parts.append(f"There was a {fact.subject}. ")
                parts.append(f"The {fact.subject} was {fact.value}. ")
                parts.append(f"Its {fact.value} color stood out. ")

            # Add filler to reach target length
            if len(' '.join(parts).split()) < target_length - 100:
                fillers = [
                    f"{main_char} thought about what was happening. ",
                    "The situation was interesting. ",
                    "Things were progressing steadily. ",
                    f"{main_char} remained focused. ",
                    "The experience was memorable. "
                ]
                parts.append(random.choice(fillers))

        # Closing (30-50 words)
        parts.append(f"In the end, {main_char} was satisfied. ")
        parts.append("Everything had worked out well. ")
        parts.append("It was a good day. ")

        return ''.join(parts)

    def _generate_questions(self, facts: List[SimpleFact]) -> List[Dict[str, str]]:
        """Generate questions from facts."""
        questions = []

        for fact in facts:
            questions.append({
                "question": fact.question,
                "answer": fact.answer,
                "type": fact.fact_type
            })

        return questions

    def generate_dataset(
        self,
        num_stories: int = 10,
        num_facts_per_story: int = 3
    ) -> List[SimpleStory]:
        """Generate multiple simple stories."""
        stories = []

        for i in range(num_stories):
            story = self.generate_story(
                num_facts=num_facts_per_story,
                target_length=500
            )
            stories.append(story)

        return stories


if __name__ == "__main__":
    # Test the generator
    gen = SimpleStoryGenerator(seed=42)
    story = gen.generate_story(num_facts=3)

    print("="*70)
    print("SAMPLE SIMPLE STORY")
    print("="*70)
    print(f"Length: {story.length_tokens} tokens")
    print()
    print("STORY TEXT:")
    print(story.text)
    print()
    print("="*70)
    print("QUESTIONS:")
    print("="*70)
    for q in story.questions:
        print(f"Q: {q['question']}")
        print(f"A: {q['answer']}")
        print()
