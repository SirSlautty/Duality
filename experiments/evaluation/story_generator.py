"""
Synthetic Story Generator for Long-Horizon Memory Evaluation

Generates stories with:
- Planted facts (codes, dates, names)
- Characters with attributes and relationships
- Distractor subplots (conflicting or irrelevant information)
- Questions testing recall, consistency, and resolution

Used for Phase 5 evaluation of DRAI's long-horizon memory capabilities.

Author: Halcyon AI Research
Date: 2025-11-18
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class PlantedFact:
    """A fact planted in the story that should be recalled."""
    fact_type: str  # "code", "date", "name", "location", etc.
    value: str
    context: str  # Where/how it appears in story
    question: str  # Question to test recall
    correct_answer: str
    position: str  # "early", "middle", "late"


@dataclass
class Character:
    """A character in the story with attributes."""
    name: str
    attributes: Dict[str, str]  # {"occupation": "doctor", "age": "45", etc.}
    relationships: Dict[str, str]  # {"Alice": "sister", "Bob": "colleague"}
    alive: bool = True
    importance: str = "main"  # "main", "secondary", "minor"


@dataclass
class Distractor:
    """A distractor subplot or conflicting information."""
    distractor_type: str  # "conflict", "irrelevant", "misleading"
    content: str  # The distractor text
    conflicts_with: Optional[str] = None  # Which planted fact it conflicts with
    position: str = "middle"  # Where to insert


@dataclass
class Story:
    """A complete story with planted facts, characters, and distractors."""
    text: str
    planted_facts: List[PlantedFact]
    characters: List[Character]
    distractors: List[Distractor]
    questions: List[Dict[str, str]]  # [{"question": "...", "answer": "...", "type": "..."}]
    metadata: Dict[str, any]


class StoryGenerator:
    """
    Generate synthetic stories for long-horizon memory evaluation.

    Example:
        >>> gen = StoryGenerator(seed=42)
        >>> story = gen.generate_story(
        ...     num_facts=5,
        ...     num_distractors=2,
        ...     target_length=1500
        ... )
        >>> print(story.text)
        >>> for q in story.questions:
        ...     print(f"Q: {q['question']}")
        ...     print(f"A: {q['answer']}")
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize story generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # Templates for different story elements
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """Load story templates and building blocks."""
        return {
            "names": {
                "male": ["James", "Robert", "John", "Michael", "William", "David", "Richard", "Joseph"],
                "female": ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica"],
                "neutral": ["Alex", "Taylor", "Jordan", "Casey", "Morgan", "Riley", "Avery", "Quinn"]
            },
            "occupations": [
                "doctor", "engineer", "teacher", "lawyer", "artist", "journalist",
                "scientist", "detective", "pilot", "architect", "chef", "musician"
            ],
            "locations": [
                "New York", "London", "Paris", "Tokyo", "Berlin", "Sydney",
                "Toronto", "Rome", "Madrid", "Vienna", "Prague", "Amsterdam"
            ],
            "objects": [
                "key", "letter", "photograph", "book", "map", "document",
                "briefcase", "phone", "computer", "notebook", "watch", "ring"
            ],
            "codes": [
                "4932", "7851", "2648", "9173", "5429", "8361", "1754", "6209"
            ],
            "dates": [
                "March 15th", "June 23rd", "November 7th", "April 30th",
                "September 12th", "February 28th", "August 19th", "December 5th"
            ]
        }

    def generate_story(
        self,
        num_facts: int = 5,
        num_characters: int = 3,
        num_distractors: int = 2,
        target_length: int = 1500,
        theme: str = "mystery"
    ) -> Story:
        """
        Generate a complete story with planted facts and distractors.

        Args:
            num_facts: Number of facts to plant
            num_characters: Number of characters
            num_distractors: Number of distractor subplots
            target_length: Target length in tokens (approximate)
            theme: Story theme ("mystery", "adventure", "drama")

        Returns:
            Story object with text, facts, questions, etc.
        """
        # 1. Generate characters
        characters = self._generate_characters(num_characters)

        # 2. Generate planted facts
        planted_facts = self._generate_planted_facts(num_facts, characters)

        # 3. Generate distractors
        distractors = self._generate_distractors(num_distractors, planted_facts, characters)

        # 4. Construct story narrative
        story_text = self._construct_story(
            characters, planted_facts, distractors, theme, target_length
        )

        # 5. Generate questions
        questions = self._generate_questions(planted_facts, characters, distractors)

        # 6. Metadata
        metadata = {
            "num_facts": num_facts,
            "num_characters": num_characters,
            "num_distractors": num_distractors,
            "theme": theme,
            "length_tokens": len(story_text.split()),
            "length_chars": len(story_text)
        }

        return Story(
            text=story_text,
            planted_facts=planted_facts,
            characters=characters,
            distractors=distractors,
            questions=questions,
            metadata=metadata
        )

    def _generate_characters(self, num_characters: int) -> List[Character]:
        """Generate character list with attributes and relationships."""
        characters = []

        # Ensure diversity
        genders = ["male", "female", "neutral"]
        used_names = set()

        for i in range(num_characters):
            # Pick name
            gender = genders[i % len(genders)]
            available_names = [n for n in self.templates["names"][gender] if n not in used_names]
            name = random.choice(available_names)
            used_names.add(name)

            # Attributes
            attributes = {
                "occupation": random.choice(self.templates["occupations"]),
                "age": str(random.randint(25, 65)),
                "hometown": random.choice(self.templates["locations"])
            }

            # Importance
            if i == 0:
                importance = "main"
            elif i < num_characters // 2:
                importance = "secondary"
            else:
                importance = "minor"

            characters.append(Character(
                name=name,
                attributes=attributes,
                relationships={},
                alive=True,
                importance=importance
            ))

        # Add relationships
        if len(characters) >= 2:
            relationships = ["colleague", "friend", "rival", "partner", "mentor"]
            for i in range(len(characters) - 1):
                rel = random.choice(relationships)
                characters[i].relationships[characters[i+1].name] = rel
                # Reciprocal
                inverse = {"colleague": "colleague", "friend": "friend", "rival": "rival",
                          "partner": "partner", "mentor": "student"}
                characters[i+1].relationships[characters[i].name] = inverse.get(rel, rel)

        return characters

    def _generate_planted_facts(
        self,
        num_facts: int,
        characters: List[Character]
    ) -> List[PlantedFact]:
        """Generate facts to plant in the story."""
        facts = []
        positions = ["early", "middle", "late"]

        # Ensure variety of fact types
        fact_types = ["code", "date", "object", "location", "attribute"] * (num_facts // 5 + 1)
        random.shuffle(fact_types)

        for i in range(num_facts):
            fact_type = fact_types[i]
            position = positions[i % len(positions)]
            char = random.choice(characters)

            if fact_type == "code":
                value = random.choice(self.templates["codes"])
                context = f"{char.name} mentioned the access code: {value}"
                question = "What was the access code mentioned in the story?"
                answer = value

            elif fact_type == "date":
                value = random.choice(self.templates["dates"])
                context = f"The event was scheduled for {value}"
                question = "When was the event scheduled?"
                answer = value

            elif fact_type == "object":
                value = random.choice(self.templates["objects"])
                context = f"{char.name} was searching for a {value}"
                question = f"What was {char.name} searching for?"
                answer = f"a {value}" if value[0] not in 'aeiou' else f"an {value}"

            elif fact_type == "location":
                value = random.choice(self.templates["locations"])
                context = f"{char.name} traveled to {value}"
                question = f"Where did {char.name} travel to?"
                answer = value

            else:  # attribute
                attr_key = random.choice(list(char.attributes.keys()))
                value = char.attributes[attr_key]
                context = f"{char.name} worked as a {value}" if attr_key == "occupation" else f"{char.name} was from {value}"
                question = f"What was {char.name}'s {attr_key}?"
                answer = value

            facts.append(PlantedFact(
                fact_type=fact_type,
                value=value,
                context=context,
                question=question,
                correct_answer=answer,
                position=position
            ))

        return facts

    def _generate_distractors(
        self,
        num_distractors: int,
        planted_facts: List[PlantedFact],
        characters: List[Character]
    ) -> List[Distractor]:
        """Generate distractor subplots."""
        distractors = []

        for i in range(num_distractors):
            dist_type = random.choice(["conflict", "irrelevant", "misleading"])

            if dist_type == "conflict" and planted_facts:
                # Conflicting information
                fact = random.choice(planted_facts)
                if fact.fact_type == "code":
                    wrong_code = random.choice([c for c in self.templates["codes"] if c != fact.value])
                    content = f"Later, someone mentioned the code might be {wrong_code}, but this was incorrect."
                    conflicts_with = fact.value
                elif fact.fact_type == "date":
                    wrong_date = random.choice([d for d in self.templates["dates"] if d != fact.value])
                    content = f"There was confusion about the date - some thought it was {wrong_date}."
                    conflicts_with = fact.value
                else:
                    content = f"There were conflicting reports about what actually happened."
                    conflicts_with = fact.value

            elif dist_type == "irrelevant":
                # Completely irrelevant subplot
                irrelevant_topics = [
                    "Meanwhile, a discussion about the weather occupied several minutes.",
                    "A lengthy debate about the best coffee in town ensued.",
                    "Someone shared a detailed story about their recent vacation.",
                    "There was an extended conversation about local sports teams.",
                    "A detailed discussion about architecture filled the next hour."
                ]
                content = random.choice(irrelevant_topics)
                conflicts_with = None

            else:  # misleading
                # Misleading but not directly conflicting
                char = random.choice(characters)
                misleading_texts = [
                    f"{char.name} seemed distracted and mentioned several unrelated things.",
                    f"The conversation drifted to other topics, making it hard to focus.",
                    f"Several interruptions made it difficult to track the main thread.",
                    f"Background noise and side conversations created confusion."
                ]
                content = random.choice(misleading_texts)
                conflicts_with = None

            distractors.append(Distractor(
                distractor_type=dist_type,
                content=content,
                conflicts_with=conflicts_with,
                position="middle"  # Most distractors in middle
            ))

        return distractors

    def _construct_story(
        self,
        characters: List[Character],
        planted_facts: List[PlantedFact],
        distractors: List[Distractor],
        theme: str,
        target_length: int
    ) -> str:
        """Construct the full story narrative."""

        # Story sections
        sections = {
            "early": [],
            "middle": [],
            "late": []
        }

        # Opening (introduces characters)
        main_char = characters[0]
        opening = f"The story begins with {main_char.name}, a {main_char.attributes['occupation']} from {main_char.attributes['hometown']}. "

        # Introduce other characters
        for char in characters[1:]:
            rel = main_char.relationships.get(char.name, "acquaintance")
            opening += f"{char.name}, {main_char.name}'s {rel}, was also involved. "

        sections["early"].append(opening)

        # Add planted facts to appropriate sections
        for fact in planted_facts:
            sections[fact.position].append(fact.context + ". ")

        # Add distractors
        for dist in distractors:
            sections[dist.position].append(dist.content + " ")

        # Add narrative connective tissue based on theme
        if theme == "mystery":
            sections["early"].append("Strange events had been unfolding over the past few days. ")
            sections["middle"].append("The investigation led to unexpected discoveries. ")
            sections["late"].append("Eventually, the pieces began to fall into place. ")
        elif theme == "adventure":
            sections["early"].append("The journey was about to begin. ")
            sections["middle"].append("Challenges arose that tested everyone's resolve. ")
            sections["late"].append("The destination was finally in sight. ")
        else:  # drama
            sections["early"].append("Tensions had been building for some time. ")
            sections["middle"].append("Conflicts emerged that changed relationships. ")
            sections["late"].append("Resolutions were found, though not without cost. ")

        # Shuffle within sections for naturalistic flow
        for section in sections.values():
            random.shuffle(section)

        # Construct final text
        story_parts = []
        story_parts.extend(sections["early"])
        story_parts.extend(sections["middle"])
        story_parts.extend(sections["late"])

        # Add conclusion
        story_parts.append(f"In the end, {main_char.name} reflected on everything that had happened. ")

        story_text = " ".join(story_parts)

        # Pad to target length if needed
        current_tokens = len(story_text.split())
        if current_tokens < target_length * 0.8:
            # Add more narrative filler
            filler = self._generate_filler(target_length - current_tokens, characters, theme)
            # Insert filler in middle
            middle_point = len(story_text) // 2
            story_text = story_text[:middle_point] + " " + filler + " " + story_text[middle_point:]

        return story_text

    def _generate_filler(
        self,
        num_tokens: int,
        characters: List[Character],
        theme: str
    ) -> str:
        """Generate narrative filler to reach target length."""
        filler_templates = {
            "mystery": [
                "The investigation continued methodically. ",
                "Each clue was carefully examined. ",
                "Questions outnumbered answers. ",
                "The mystery deepened with each revelation. "
            ],
            "adventure": [
                "The journey tested their limits. ",
                "New landscapes unfolded before them. ",
                "Each day brought fresh challenges. ",
                "The adventure was far from over. "
            ],
            "drama": [
                "Emotions ran high throughout. ",
                "Conversations revealed hidden depths. ",
                "Relationships evolved and changed. ",
                "The human element remained central. "
            ]
        }

        templates = filler_templates.get(theme, filler_templates["drama"])
        filler = ""

        while len(filler.split()) < num_tokens:
            filler += random.choice(templates)
            if random.random() < 0.3 and characters:
                char = random.choice(characters)
                filler += f"{char.name} considered the situation carefully. "

        return filler

    def _generate_questions(
        self,
        planted_facts: List[PlantedFact],
        characters: List[Character],
        distractors: List[Distractor]
    ) -> List[Dict[str, str]]:
        """Generate questions testing recall, consistency, and resolution."""
        questions = []

        # Factual recall questions (from planted facts)
        for fact in planted_facts:
            questions.append({
                "question": fact.question,
                "answer": fact.correct_answer,
                "type": "factual_recall",
                "difficulty": "medium"
            })

        # Consistency questions (affected by distractors)
        for dist in distractors:
            if dist.distractor_type == "conflict" and dist.conflicts_with:
                questions.append({
                    "question": f"Was there any conflicting information about {dist.conflicts_with}?",
                    "answer": "Yes, there were conflicting reports.",
                    "type": "consistency_check",
                    "difficulty": "hard"
                })

        # Character questions
        if characters:
            main_char = characters[0]
            questions.append({
                "question": f"Who was the main character in this story?",
                "answer": main_char.name,
                "type": "character_identification",
                "difficulty": "easy"
            })

            if len(characters) > 1:
                rel = main_char.relationships.get(characters[1].name, "unknown")
                questions.append({
                    "question": f"What was the relationship between {main_char.name} and {characters[1].name}?",
                    "answer": rel,
                    "type": "relationship",
                    "difficulty": "medium"
                })

        return questions

    def generate_dataset(
        self,
        num_stories: int = 50,
        varied_complexity: bool = True,
        save_path: Optional[str] = None
    ) -> List[Story]:
        """
        Generate a dataset of multiple stories.

        Args:
            num_stories: Number of stories to generate
            varied_complexity: Vary num_facts/distractors across stories
            save_path: Optional path to save dataset as JSON

        Returns:
            List of Story objects
        """
        stories = []

        for i in range(num_stories):
            if varied_complexity:
                num_facts = random.randint(3, 7)
                num_distractors = random.randint(0, 5)
                target_length = random.randint(1000, 2000)
            else:
                num_facts = 5
                num_distractors = 2
                target_length = 1500

            theme = random.choice(["mystery", "adventure", "drama"])

            story = self.generate_story(
                num_facts=num_facts,
                num_distractors=num_distractors,
                target_length=target_length,
                theme=theme
            )

            stories.append(story)

        if save_path:
            self._save_dataset(stories, save_path)

        return stories

    def _save_dataset(self, stories: List[Story], path: str):
        """Save dataset to JSON file."""
        dataset = []

        for story in stories:
            dataset.append({
                "text": story.text,
                "planted_facts": [
                    {
                        "type": f.fact_type,
                        "value": f.value,
                        "question": f.question,
                        "answer": f.correct_answer,
                        "position": f.position
                    }
                    for f in story.planted_facts
                ],
                "characters": [
                    {
                        "name": c.name,
                        "attributes": c.attributes,
                        "importance": c.importance
                    }
                    for c in story.characters
                ],
                "distractors": [
                    {
                        "type": d.distractor_type,
                        "content": d.content,
                        "conflicts_with": d.conflicts_with
                    }
                    for d in story.distractors
                ],
                "questions": story.questions,
                "metadata": story.metadata
            })

        with open(path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"Dataset saved to {path}")
        print(f"Total stories: {len(stories)}")
        print(f"Avg length: {sum(s.metadata['length_tokens'] for s in stories) / len(stories):.0f} tokens")


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("SYNTHETIC STORY GENERATOR - Example Output")
    print("="*70)

    gen = StoryGenerator(seed=42)

    # Generate single story
    story = gen.generate_story(
        num_facts=5,
        num_characters=3,
        num_distractors=2,
        target_length=800,  # Shorter for example
        theme="mystery"
    )

    print("\n" + "="*70)
    print("STORY TEXT")
    print("="*70)
    print(story.text)

    print("\n" + "="*70)
    print("PLANTED FACTS")
    print("="*70)
    for i, fact in enumerate(story.planted_facts, 1):
        print(f"{i}. [{fact.fact_type}] {fact.value} ({fact.position})")
        print(f"   Context: {fact.context}")

    print("\n" + "="*70)
    print("QUESTIONS")
    print("="*70)
    for i, q in enumerate(story.questions, 1):
        print(f"{i}. Q: {q['question']}")
        print(f"   A: {q['answer']}")
        print(f"   Type: {q['type']}, Difficulty: {q['difficulty']}")
        print()

    print("="*70)
    print("METADATA")
    print("="*70)
    for key, value in story.metadata.items():
        print(f"{key}: {value}")

    print("\n" + "="*70)
    print("Generating small dataset...")
    print("="*70)
    stories = gen.generate_dataset(num_stories=10, varied_complexity=True)
    print(f"Generated {len(stories)} stories")
    print(f"Length range: {min(s.metadata['length_tokens'] for s in stories)} - {max(s.metadata['length_tokens'] for s in stories)} tokens")
